"""Frozen-checkpoint original photos, VAL robustness and probability export.

No training or threshold selection occurs here. The geometry control is a known
integer crop translation, not an estimate of robustness to real camera motion.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..augmentations import apply_nuisance
from ..io import finish_record, new_directory, read_json, run_record, sha256, write_json
from ..methods.dinov2 import _source_provenance
from .data import PreparedData
from .metrics import write_reports
from .model import load_model
from .prepared import PreparedAugmentations
from .train import evaluate, gpu_preflight, ownership_masks


CONDITIONS = (
    {"id": "original", "kind": "identity"},
    {"id": "exposure_low", "kind": "exposure", "gain": .65},
    {"id": "exposure_high", "kind": "exposure", "gain": 1.4},
    {"id": "white_balance_cool", "kind": "white_balance", "gains": [.85, 1., 1.15]},
    {"id": "shadow", "kind": "shadow", "template": "diagonal", "strength": .45, "edge_width": .055},
    {"id": "translation_reference", "kind": "translation_reference"},
    {"id": "translation", "kind": "translation"},
)


def _shift(array, dx, dy):
    """Exact integer nearest-neighbour translation on first two axes; zero fill."""
    output = np.zeros_like(array)
    height, width = array.shape[:2]
    x0, x1 = max(0, -dx), min(width, width - dx)
    y0, y1 = max(0, -dy), min(height, height - dy)
    if x1 > x0 and y1 > y0:
        output[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = array[y0:y1, x0:x1]
    return output


class ConditionCrops:
    """Transform cached originals only; labels never select a photometric region."""
    def __init__(self, prepared, data, condition):
        self.prepared, self.data, self.condition = prepared, data, condition
        self.dx = max(1, round(data.crop_size / 64))
        self.dy = -self.dx
        self._ownership = {}

    def original_crop(self, image_id, row):
        if self.data.images[image_id]["split"] != "val":
            raise ValueError("Robustness evaluation accepts VAL originals only")
        rgb, target, valid = self.prepared.original_crop(image_id, row)
        kind = self.condition["kind"]
        if kind == "identity":
            return rgb, target, valid
        if kind not in {"translation", "translation_reference"}:
            # Transform the whole real crop rectangle, not just annotated pixels.
            support = np.zeros_like(valid)
            support[:row["height"], :row["width"]] = True
            changed = apply_nuisance(rgb, support, self.condition)["rgb"]
            return changed, target, valid
        if image_id not in self._ownership:
            image = self.data.images[image_id]
            self._ownership[image_id] = ownership_masks(
                self.data.crops_by_image[image_id], image["width"], image["height"], self.data.crop_size)
        owner = self._ownership[image_id][row["crop_id"]]
        # Keep exactly those original owned pixels that land in the same crop's
        # owned region. Their translated counterparts form the paired support.
        common = valid & owner & _shift(owner, -self.dx, -self.dy)
        if kind == "translation_reference":
            return rgb, target, common
        return (_shift(rgb, self.dx, self.dy),
                _shift(target.transpose(1, 2, 0), self.dx, self.dy).transpose(2, 0, 1),
                _shift(common, self.dx, self.dy))


def bernoulli_entropy(probability):
    """Mean independent class entropy in bits; not an image compression cost."""
    values = np.asarray(probability, dtype=np.float64)
    if (values.ndim != 3 or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 1)):
        raise ValueError("Expected finite C,H,W independent probabilities in [0,1]")
    p = np.clip(values, 1e-12, 1 - 1e-12)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    entropy[(values == 0) | (values == 1)] = 0
    return entropy.mean(axis=0).astype(np.float32)


def _checked_run(training_run):
    root = Path(training_run).expanduser().resolve()
    record = read_json(root / "run.json")
    if (record.get("kind") != "dino_semantic_training"
            or record.get("status") not in {"completed_exploratory", "completed_smoke"}):
        raise ValueError("Evaluation requires a completed semantic training run; active runs are not loaded")
    for name in ("config.json", "best.pth", "summary.json"):
        if record.get("artifact_sha256", {}).get(name) != sha256(root / name):
            raise ValueError(f"Training artifact hash mismatch: {name}")
    config = read_json(root / "config.json")
    if config != record.get("config"):
        raise ValueError("Training run/config mismatch")
    saved = torch.load(root / "best.pth", map_location="cpu", weights_only=True)
    if (saved.get("kind") != "frozen_dinov2_semantic_head" or saved.get("schema_version") != 1
            or saved.get("config") != config or saved.get("backbone_provenance") != record.get("model")
            or saved.get("epoch") != read_json(root / "summary.json").get("best_epoch")):
        raise ValueError("Best checkpoint contract differs from completed training record")
    provenance = saved["backbone_provenance"]
    sources, weights = provenance.get("sources", {}), provenance.get("checkpoint", {})
    if (not sources.get("python_tree_sha256") or not weights.get("sha256")
            or sha256(config["checkpoint"]) != weights["sha256"]
            or _source_provenance(config["source_root"])["python_tree_sha256"] != sources["python_tree_sha256"]):
        raise ValueError("Frozen DINO source/checkpoint fingerprint differs from training")
    return root, record, config, saved


def _delta(current, baseline):
    metrics = ("miou", "iou", "f1", "precision", "recall", "loss")
    return {key: (current.get(key) - baseline.get(key)
                  if current.get(key) is not None and baseline.get(key) is not None else None)
            for key in metrics}


def _comparisons(reports):
    result = {}
    for condition, report in reports.items():
        baseline_name = "translation_reference" if condition.startswith("translation") else "original"
        baseline = reports[baseline_name]
        item = {"baseline": baseline_name, "direction": "condition minus paired reference; metric fractions, not percentages",
                "primary": _delta(report["primary"], baseline["primary"])}
        for field, identity in (("per_class", "category_id"), ("per_group", "name"),
                                ("per_building", "building_id")):
            old = {row[identity]: row for row in baseline[field]}
            item[field] = [{identity: row[identity], **_delta(row, old[row[identity]])} for row in report[field]]
        result[condition] = item
    return result


def evaluate_checkpoint(training_run, out, augmentations=None, device="cuda:0", batch_size=4,
                        export_probabilities=False, smoke=False, no_amp=False,
                        evaluation_mode="robustness", splits=None):
    """Evaluate a verified completed best.pth; robustness remains VAL-only."""
    if evaluation_mode == "originals":
        return evaluate_original_checkpoint(training_run, out, augmentations=augmentations,
                                            device=device, batch_size=batch_size,
                                            export_probabilities=export_probabilities,
                                            smoke=smoke, no_amp=no_amp, splits=splits)
    if evaluation_mode != "robustness" or splits is not None:
        raise ValueError("--splits requires --evaluation-mode originals; robustness remains VAL-only")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    root, training_record, config, saved = _checked_run(training_run)
    data = PreparedData(config["dataset"])
    if (data.metadata["sha256"] != config["dataset_provenance"]["sha256"]
            or data.categories != config["categories"]
            or data.metadata["category_to_channel"] != config["category_to_channel"]):
        raise ValueError("Dataset/ontology fingerprint differs from checkpoint training data")
    prepared = PreparedAugmentations(augmentations or config["augmentations"]["path"], data)
    if prepared.metadata["run_sha256"] != config["augmentations"]["run_sha256"]:
        raise ValueError("Prepared augmentation snapshot differs from training")
    ids = config["validation_image_ids"]
    if not ids or len(set(ids)) != len(ids) or any(key not in data.image_ids("val") for key in ids):
        raise ValueError("Checkpoint validation IDs are not a distinct nonempty VAL subset")
    partial = bool(smoke or config.get("smoke"))
    if partial:
        ids = ids[:2]
    groups = config["class_groups"]
    # All contracts and CPU-restricted loading above precede GPU initialization.
    gpu = gpu_preflight(device)
    destination = new_directory(out)
    settings = {"training_run": str(root), "training_run_sha256": sha256(root / "run.json"),
                "head_checkpoint": str(root / "best.pth"), "head_sha256": sha256(root / "best.pth"),
                "dataset_provenance": data.metadata, "augmentations": prepared.metadata,
                "categories": data.categories, "category_to_channel": data.metadata["category_to_channel"],
                "class_groups": groups, "conditions": list(CONDITIONS), "validation_image_ids": ids,
                "device": str(device), "batch_size": batch_size, "smoke": partial,
                "amp": bool(config["amp"] and str(device).startswith("cuda") and not no_amp),
                "export_probabilities": bool(export_probabilities),
                "translation": {"dx": max(1, round(data.crop_size / 64)), "dy": -max(1, round(data.crop_size / 64)),
                                "units": "native crop pixels", "scope": "known integer crop displacement; not real viewpoint geometry",
                                "support": "same originally owned source pixels in translation_reference and translation"},
                "test_evaluated": False, "selection": "no training, threshold tuning or condition selection",
                "uncertainty": "mean independent Bernoulli class entropy; annotation-support diagnostics, not temporal change or image bits"}
    record = run_record("semantic_robustness", settings)
    record["gpu_preflight"] = gpu
    write_json(destination / "run.json", record)
    write_json(destination / "config.json", settings)
    try:
        model, provenance = load_model(config["source_root"], config["checkpoint"], len(data.categories),
                                       device, config.get("trust_checkpoint", False))
        for key, fingerprint in (("sources", "python_tree_sha256"), ("checkpoint", "sha256")):
            if provenance[key][fingerprint] != saved["backbone_provenance"][key][fingerprint]:
                raise ValueError("Loaded backbone fingerprint differs from saved checkpoint")
        model.head.load_state_dict(saved["head_state_dict"], strict=True)
        reports, exports = {}, []
        for condition in CONDITIONS:
            name = condition["id"]
            folder = destination / name
            folder.mkdir()
            probability_folder = folder / "probabilities"
            if export_probabilities:
                probability_folder.mkdir()

            def export(*, image_id, row, crop, valid, probability, logits):
                key = hashlib.sha256(f"{image_id}:{row['crop_id']}".encode()).hexdigest()[:24]
                path = probability_folder / (key + ".npz")
                np.savez_compressed(path, probabilities=probability.astype(np.float32),
                                    targets=crop[1], valid=valid,
                                    mean_bernoulli_entropy=bernoulli_entropy(probability))
                exports.append({"condition": name, "image_id": image_id, "crop_id": row["crop_id"],
                                "building_id": data.images[image_id]["building_id"],
                                "crop": {key: row[key] for key in ("x", "y", "width", "height")},
                                "path": path.relative_to(destination).as_posix(), "sha256": sha256(path),
                                "valid_pixels": int(valid.sum())})

            report = evaluate(model, data, ids, device, batch_size, amp=settings["amp"], smoke=partial,
                              groups=groups, prepared=ConditionCrops(prepared, data, condition),
                              prediction_callback=export if export_probabilities else None)
            report.update(condition=condition, prediction_source="frozen best checkpoint on deterministic VAL condition",
                          comparison_reference="translation_reference" if name.startswith("translation") else "original")
            reports[name] = report
            write_json(folder / "metrics.json", report)
            write_reports(folder, report)
            values = report["primary"]
            print(f"VAL {name}: mIoU={values['miou']}; Recall={values['recall']}; loss={values['loss']}", flush=True)
        write_json(destination / "comparisons.json", _comparisons(reports))
        if export_probabilities:
            (destination / "probabilities.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in exports), encoding="utf-8")
        summary = {"status": "completed_smoke" if partial else "completed_exploratory", "out": str(destination),
                   "training_run": str(root), "checkpoint_epoch": saved["epoch"],
                   "conditions": {name: report["primary"] for name, report in reports.items()},
                   "exported_crops": len(exports), "test_evaluated": False,
                   "physical_disjointness_verified": data.metadata["leakage_check"]["physical_disjointness_verified"],
                   "scope": "VAL diagnostic, annotated support only; no new independent generalization claim"}
        write_json(destination / "summary.json", summary)
        finish_record(destination, record, summary["status"])
        return summary
    except BaseException as error:
        finish_record(destination, record, "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", str(error))
        raise


def evaluate_original_checkpoint(training_run, out, augmentations=None, device="cuda:0", batch_size=4,
                                 export_probabilities=False, smoke=False, no_amp=False, splits=None):
    """Frozen best-VAL head, unchanged threshold and full native-image galleries.

    TEST is opt-in through the requested split list. TRAIN/VAL RGB and targets
    come from verified cached originals; TEST is decoded from the same immutable
    dataset snapshot. All image regions receive predictions for visual review,
    but quantitative metrics remain restricted to the original annotated union.
    """
    from PIL import Image
    from .metrics import write_full_image_gallery, write_full_image_panels, semantic_legend
    from ..visualization import write_static_gallery as write_project_gallery

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    splits = ("val",) if splits is None else tuple(splits)
    if not splits or len(splits) != len(set(splits)) or not set(splits) <= {"train", "val", "test"}:
        raise ValueError("Request distinct nonempty train/val/test splits")
    root, training_record, config, saved = _checked_run(training_run)
    training_data = PreparedData(config["dataset"])
    if (training_data.metadata["sha256"] != config["dataset_provenance"]["sha256"]
            or training_data.categories != config["categories"]
            or training_data.metadata["category_to_channel"] != config["category_to_channel"]):
        raise ValueError("Dataset/ontology fingerprint differs from checkpoint training data")
    prepared = PreparedAugmentations(augmentations or config["augmentations"]["path"], training_data)
    if prepared.metadata["run_sha256"] != config["augmentations"]["run_sha256"]:
        raise ValueError("Prepared augmentation snapshot differs from training")
    # Constructing the explicit TEST reader validates labels/identities only.
    # No TEST RGB is opened unless the inference loop below requests it.
    test_data = PreparedData(config["dataset"], splits=("test",), allow_test=True) if "test" in splits else None
    readers = {split: test_data if split == "test" else training_data for split in splits}
    ids = {}
    partial = bool(smoke or config.get("smoke"))
    for split, data in readers.items():
        selected = data.image_ids(split) if split == "test" else config[f"{'training' if split == 'train' else 'validation'}_image_ids"]
        if not selected or len(selected) != len(set(selected)) or any(key not in data.image_ids(split) for key in selected):
            raise ValueError(f"Checkpoint {split.upper()} IDs must be a distinct nonempty matching subset")
        if not config.get("smoke") and set(selected) != set(data.image_ids(split)):
            raise ValueError(f"Completed training record omits original {split.upper()} images")
        ids[split] = list(selected[:2] if partial else selected)
    gpu = gpu_preflight(device)
    destination = new_directory(out)
    settings = {"training_run": str(root), "training_run_sha256": sha256(root / "run.json"),
                "head_checkpoint": str(root / "best.pth"), "head_sha256": sha256(root / "best.pth"),
                "checkpoint_epoch": saved["epoch"], "checkpoint_selection": "saved best VAL mIoU; unchanged",
                "dataset_provenance": training_data.metadata, "augmentations": prepared.metadata,
                "categories": training_data.categories, "category_to_channel": training_data.metadata["category_to_channel"],
                "class_groups": config["class_groups"], "splits": list(splits), "image_ids": ids,
                "device": str(device), "batch_size": batch_size, "smoke": partial,
                "amp": bool(config["amp"] and str(device).startswith("cuda") and not no_amp),
                "export_probabilities": bool(export_probabilities), "threshold": .5,
                "evaluation_mode": "originals", "test_requested": "test" in splits,
                "selection": "no training, checkpoint reselection, threshold fitting or TEST tuning",
                "prediction_grid": "native RGB; full grid, fixed midpoint ownership; all classes retained as independent bits",
                "evaluation_scope": "annotated opaque non-crowd union only; unannotated pixels remain unknown",
                "gallery_scope": "whole original photos; predictions outside annotation support are unverified",
                "unknown_train": "separate image-level diagnostics; facade disjointness unconfirmed"}
    record = run_record("semantic_original_evaluation", settings)
    record["gpu_preflight"] = gpu
    write_json(destination / "run.json", record)
    write_json(destination / "config.json", settings)
    try:
        model, provenance = load_model(config["source_root"], config["checkpoint"], len(training_data.categories),
                                       device, config.get("trust_checkpoint", False))
        for key, fingerprint in (("sources", "python_tree_sha256"), ("checkpoint", "sha256")):
            if provenance[key][fingerprint] != saved["backbone_provenance"][key][fingerprint]:
                raise ValueError("Loaded backbone fingerprint differs from saved checkpoint")
        model.head.load_state_dict(saved["head_state_dict"], strict=True)
        reports, exports, all_sections = {}, [], []
        for split, data in readers.items():
            folder = destination / split
            folder.mkdir()
            sections = []
            if export_probabilities:
                (folder / "probabilities").mkdir()

            def export_crop(*, image_id, row, crop, valid, probability, logits):
                key = hashlib.sha256(f"{image_id}:{row['crop_id']}".encode()).hexdigest()[:24]
                path = folder / "probabilities" / (key + ".npz")
                np.savez_compressed(path, probabilities=probability.astype(np.float32),
                                    targets=crop[1], valid=valid,
                                    mean_bernoulli_entropy=bernoulli_entropy(probability))
                exports.append({"split": split, "image_id": image_id, "crop_id": row["crop_id"],
                                "building_id": data.images[image_id]["building_id"],
                                "crop": {key: row[key] for key in ("x", "y", "width", "height")},
                                "path": path.relative_to(destination).as_posix(), "sha256": sha256(path),
                                "valid_pixels": int(valid.sum())})

            def export_image(*, image_id, image, rgb, target, prediction, valid):
                # Cached training originals encode annotation support, not the
                # complete alpha plane. Read only source headers for RGB; RGBA
                # needs its verified alpha channel to avoid painting transparent
                # pixels as reliable predictions. TEST already cached this plane.
                if data._cache is not None and data._cache[0] == image_id:
                    opaque = data._cache[3]
                else:
                    source_path = Path(image["image_path"])
                    if sha256(source_path) != image["sha256"]:
                        raise ValueError(f"Original image hash mismatch: {source_path}")
                    with Image.open(source_path) as original:
                        if original.size != (image["width"], image["height"]) or original.mode not in {"RGB", "RGBA"}:
                            raise ValueError(f"Original image grid/mode differs from prepared snapshot: {source_path}")
                        opaque = np.array(original.getchannel("A")) == 255 if original.mode == "RGBA" else None
                sections.append(write_full_image_panels(
                    folder, rgb=rgb, target=target, prediction=prediction, valid=valid,
                    categories=data.categories, image_id=image_id, building_id=image.get("building_id"),
                    split=split, source_name=image.get("file_name", Path(image["image_path"]).name),
                    prediction_valid=opaque))

            report = evaluate(model, data, ids[split], device, batch_size, amp=settings["amp"],
                              smoke=False, split=split, groups=config["class_groups"],
                              prepared=None if split == "test" else prepared,
                              prediction_callback=export_crop if export_probabilities else None,
                              image_prediction_callback=export_image)
            report.update(smoke=partial, checkpoint_epoch=saved["epoch"], checkpoint_sha256=settings["head_sha256"],
                          prediction_source="frozen best-VAL checkpoint; original native-grid RGB",
                          full_image_predictions=True, images_visualized=len(sections))
            reports[split] = report
            write_json(folder / "metrics.json", report)
            write_reports(folder, report)
            write_full_image_gallery(folder, sections, data.categories, split=split,
                                     checkpoint_name=f"best.pth — epoch {saved['epoch']}")
            for section in sections:
                combined = dict(section, panels=[dict(panel, **{
                    key: f"{split}/{panel[key]}" for key in ("src", "href") if key in panel})
                    for panel in section["panels"]])
                combined["caption"] = section.get("caption", "").replace("photos/", f"{split}/photos/")
                all_sections.append(combined)
            values = report["primary"]
            print(f"{split.upper()} originals: mIoU={values['miou']}; Recall={values['recall']}; loss={values['loss']}", flush=True)
        write_project_gallery(destination, title="DINOv2 — семантическая сегментация целых фотографий",
                              sections=all_sections, legend=semantic_legend(training_data.categories),
                              notes=[f"Сохранённый best.pth, эпоха {saved['epoch']}; sigmoid ≥ 0.5; без повторного обучения и подбора порогов.",
                                     "GT — разметка на фото; GT MASKS — все размеченные классы. Серая клетка означает неизвестную разметку, не здоровый фон.",
                                     "PREDICTED показывает прогноз на всём фото. Метрики охватывают только размеченные непрозрачные пиксели без crowd; прогноз вне этой поддержки не оценён.",
                                     "Чёрный в PREDICTED MASKS означает отсутствие классов выше порога, не подтверждённое отсутствие повреждений. Непрозрачная поддержка исходного RGB учитывается отдельно.",
                                     "Пересечения классов показаны диагональными полосами всех активных каналов. Точные пересекающиеся маски сохранены в masks.npz; argmax не применяется.",
                                     "На странице — облегчённые превью целых фото; по клику открывается исходное разрешение выбранной панели.",
                                     "TRAIN и VAL — диагностика разработки; TEST — exploratory-проверка зафиксированного checkpoint. Независимость неизвестных TRAIN-адресов от holdout пока не подтверждена."])
        if export_probabilities:
            (destination / "probabilities.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in exports), encoding="utf-8")
        summary = {"status": "completed_smoke" if partial else "completed_exploratory", "out": str(destination),
                   "training_run": str(root), "checkpoint_epoch": saved["epoch"], "checkpoint_sha256": settings["head_sha256"],
                   "splits": {split: report["primary"] for split, report in reports.items()},
                   "images": {split: len(keys) for split, keys in ids.items()},
                   "images_visualized": len(all_sections), "exported_crops": len(exports),
                   "test_evaluated": "test" in splits,
                   "test_rgb_loaded": bool(test_data and test_data.metadata["test_rgb_loaded"]),
                   "test_annotations_rasterized": bool(test_data and test_data.metadata["test_annotations_rasterized"]),
                   "physical_disjointness_verified": training_data.metadata["leakage_check"]["physical_disjointness_verified"],
                   "scope": "frozen-checkpoint semantic multilabel metrics on annotated union; no background or temporal-change claim"}
        write_json(destination / "summary.json", summary)
        finish_record(destination, record, summary["status"])
        return summary
    except BaseException as error:
        finish_record(destination, record, "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", str(error))
        raise
