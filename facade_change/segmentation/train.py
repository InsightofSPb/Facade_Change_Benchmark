"""Small frozen-feature training loop with native-grid, facade-level validation."""
from __future__ import annotations

import contextlib
import importlib.metadata
import math
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ..io import finish_record, new_directory, read_json, run_record, sha256, write_json
from .data import PreparedData
from .metrics import (aggregate_metrics, confusion_counts, group_confusion_counts,
                      loss_statistics, merge_loss_statistics, resolve_groups, write_reports,
                      _class_metrics, _group_metrics, _summary)
from .model import load_model, masked_multilabel_loss
from .prepared import PreparedAugmentations


def dino_paths(methods_config=None, source_root=None, checkpoint=None):
    """Read only DINO paths from an existing methods/benchmark config."""
    seen, hashes = set(), {}

    def options(path):
        path = Path(path).expanduser().resolve()
        if path in seen:
            raise ValueError("Recursive methods_config reference")
        seen.add(path)
        hashes[str(path)] = sha256(path)
        value = read_json(path)
        result = {}
        if value.get("methods_config"):
            nested = Path(value["methods_config"]).expanduser()
            result.update(options(nested if nested.is_absolute() else path.parent / nested))
        for key in ("methods", "method_options"):
            mapping = value.get(key, {})
            if isinstance(mapping, dict):
                for name, item in mapping.get("dinov2", {}).items():
                    if name in {"source_root", "checkpoint_path", "dino_root", "dino_checkpoint"} and item:
                        candidate = Path(item).expanduser()
                        result[name] = str((candidate if candidate.is_absolute() else path.parent / candidate).resolve())
        return result

    configured = options(methods_config) if methods_config else {}
    root = source_root or configured.get("source_root") or configured.get("dino_root")
    weights = checkpoint or configured.get("checkpoint_path") or configured.get("dino_checkpoint")
    if not root or not weights:
        raise ValueError("Provide DINOv2 paths using --methods-config or both --source-root and --checkpoint")
    root, weights = Path(root).expanduser().resolve(), Path(weights).expanduser().resolve()
    if not (root / "hubconf.py").is_file() or not weights.is_file():
        raise FileNotFoundError(f"Existing DINOv2 source/weights not found: {root}, {weights}")
    return str(root), str(weights), hashes


def gpu_preflight(device):
    target = torch.device(device)
    if target.type == "cpu":
        return {"device": "cpu"}
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"Requested device is unavailable: {device}")
    torch.cuda.set_device(target)
    free, total = torch.cuda.mem_get_info(target)
    report = {"device": str(target), "name": torch.cuda.get_device_name(target),
              "free_bytes": free, "total_bytes": total}
    try:
        report["compute_processes"] = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as error:
        report["process_query_error"] = str(error)
    print(f"GPU: {report['name']}; свободно {free / 2**30:.2f}/{total / 2**30:.2f} GiB", flush=True)
    if free < 2 * 2**30:
        raise RuntimeError("Less than 2 GiB GPU memory available; free memory or choose another device. Other processes were not changed.")
    return report


def ownership_masks(rows, image_width, image_height, crop_size):
    """Partition overlapping grid crops at overlap midpoints, scoring each pixel once.

    No full-resolution C-channel prediction array is needed. This is a fixed
    centre-ownership protocol, not logit averaging or a prediction-dependent mask.
    """
    axes = []
    for position, extent, maximum in (("x", "width", image_width), ("y", "height", image_height)):
        values = {}
        for row in rows:
            key, size = row[position], row[extent]
            if key in values and values[key] != size:
                raise ValueError("Inconsistent crop grid extent")
            values[key] = size
        starts = sorted(values)
        if not starts or starts[0] != 0 or starts[-1] + values[starts[-1]] != maximum:
            raise ValueError("Prepared grid does not cover image boundaries")
        if any(a + values[a] < b for a, b in zip(starts, starts[1:])):
            raise ValueError("Prepared grid has uncovered gaps")
        edges = [0] + [(a + values[a] + b) // 2 for a, b in zip(starts, starts[1:])] + [maximum]
        axes.append({start: (edges[i], edges[i + 1]) for i, start in enumerate(starts)})
    if len(rows) != len(axes[0]) * len(axes[1]) or len({(r['x'], r['y']) for r in rows}) != len(rows):
        raise ValueError("Prepared crops must be a complete rectangular grid")
    result = {}
    for row in rows:
        x0, x1 = axes[0][row["x"]]
        y0, y1 = axes[1][row["y"]]
        mask = np.zeros((crop_size, crop_size), dtype=bool)
        mask[y0-row["y"]:y1-row["y"], x0-row["x"]:x1-row["x"]] = True
        result[row["crop_id"]] = mask
    return result


def _tensor_batch(crops, device):
    rgb = torch.from_numpy(np.stack([c[0] for c in crops])).permute(0, 3, 1, 2)
    target = torch.from_numpy(np.stack([c[1] for c in crops]))
    valid = torch.from_numpy(np.stack([c[2] for c in crops]))
    return (rgb.to(device=device, dtype=torch.float32) / 255.,
            target.to(device=device, dtype=torch.float32), valid.to(device))


def _autocast(device, enabled):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled else contextlib.nullcontext()


def fixed_batches(prepared, records, batch_size, rng):
    """Visit every supervised prepared sample once; shuffle order only."""
    order = rng.permutation(len(records))
    for offset in range(0, len(order), batch_size):
        yield [prepared.load(records[int(index)]) for index in order[offset:offset + batch_size]]


def evaluate(model, data, ids, device, batch_size, amp=False, smoke=False, gallery_out=None,
             split="val", groups=None, prepared=None, prediction_callback=None,
             preview_buffer=None, image_prediction_callback=None):
    model.eval()
    groups = groups if groups is not None else resolve_groups(data.categories)
    if split not in {"train", "val", "test"} or any(data.images[key]["split"] != split for key in ids):
        raise ValueError("Evaluation IDs must belong to the named split")
    records, previews, shown = [], [], set()
    collect_previews = gallery_out is not None or preview_buffer is not None
    for image_id in tqdm(ids, desc=f"DINO {split.upper()} оценка", unit="фото"):
        image = data.images[image_id]
        all_rows = data.crops_by_image[image_id]
        ownership = ownership_masks(all_rows, image["width"], image["height"], data.crop_size)
        rows = all_rows
        counts = np.zeros((len(data.categories), 3), dtype=np.int64)
        group_counts = np.zeros((len(groups), 3), dtype=np.int64)
        losses = []
        candidates = []
        # Visualisation uses native-grid bitsets rather than C-channel full-image
        # tensors. This preserves multilabel overlaps and bounds memory for large
        # photographs. Every crop runs once, including unannotated image regions.
        full = None
        if image_prediction_callback is not None:
            if not 1 <= len(data.categories) <= 16:
                raise ValueError("Native full-image bit masks require 1–16 category channels")
            shape = (image["height"], image["width"])
            full = {"rgb": np.zeros((*shape, 3), dtype=np.uint8),
                    "target": np.zeros(shape, dtype=np.uint16),
                    "prediction": np.zeros(shape, dtype=np.uint16),
                    "valid": np.zeros(shape, dtype=bool),
                    "coverage": np.zeros(shape, dtype=bool)}

        def process_batch(items):
            rgb, _, _ = _tensor_batch([item[1] for item in items], device)
            with torch.no_grad(), _autocast(device, amp):
                logits = model(rgb)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite validation logits")
            probabilities = logits.float().sigmoid().cpu().numpy()
            raw_logits = logits.float().cpu().numpy()
            predictions = raw_logits >= 0
            for (row, crop, valid), prediction, probability, values in zip(items, predictions, probabilities, raw_logits):
                counts[:] += confusion_counts(prediction, crop[1], valid)
                group_counts[:] += group_confusion_counts(prediction, crop[1], valid, groups)
                losses.append(loss_statistics(probability, crop[1], valid, logits=values))
                if full is not None:
                    x, y, width, height = (row[key] for key in ("x", "y", "width", "height"))
                    owner = ownership[row["crop_id"]][:height, :width]
                    bits = (np.uint16(1) << np.arange(len(data.categories), dtype=np.uint16))[:, None, None]
                    arrays = {"rgb": crop[0], "valid": crop[2],
                              "target": np.sum(crop[1].astype(np.uint16) * bits, axis=0, dtype=np.uint16),
                              "prediction": np.sum(prediction.astype(np.uint16) * bits, axis=0, dtype=np.uint16)}
                    for name, array in arrays.items():
                        full[name][y:y + height, x:x + width][owner] = array[:height, :width][owner]
                    full["coverage"][y:y + height, x:x + width][owner] = True
                if prediction_callback is not None:
                    prediction_callback(image_id=image_id, row=row, crop=crop, valid=valid,
                                        probability=probability, logits=values)
                building = image["building_id"]
                identity = ("building", building) if building else ("unknown-image", image_id)
                if collect_previews and identity not in shown and len(previews) < 6:
                    previews.append({"image_id": image_id, "building_id": building,
                                     "crop_id": row["crop_id"], "rgb": crop[0].copy(),
                                     "target": crop[1].copy(), "valid": crop[2].copy(),
                                     "prediction": prediction.copy()})
                    shown.add(identity)

        supervised_crops = 0
        for row in rows:
            crop = prepared.original_crop(image_id, row) if prepared is not None else data.crop(image_id, row)
            valid = crop[2] & ownership[row["crop_id"]]
            if not valid.any() and full is None:
                continue
            supervised_crops += 1
            candidates.append((row, crop, valid))
            if len(candidates) == batch_size:
                process_batch(candidates)
                candidates = []
            if smoke and supervised_crops >= 4:
                break
        if candidates:
            process_batch(candidates)
        if full is not None:
            if not full["coverage"].all():
                raise ValueError("Full-image gallery requires all native grid crops; disable crop-limited smoke")
            full.pop("coverage")
            image_prediction_callback(image_id=image_id, image=image, **full)
        if not losses:
            empty = np.zeros((len(data.categories), 0, 0), dtype=np.float32)
            losses.append(loss_statistics(empty, empty.astype(bool), np.zeros((0, 0), dtype=bool)))
        records.append({"image_id": image_id, "building_id": image["building_id"], "counts": counts.tolist(),
                        "group_counts": group_counts.tolist(),
                        "loss_stats": merge_loss_statistics(losses)})
    report = aggregate_metrics(records, data.categories, groups=groups, allow_unknown_buildings=split == "train")
    report.update(smoke=smoke, threshold=.5, split=split, test_evaluated=split == "test",
                  prediction_source="original unaugmented crops, model.eval() after epoch",
                  crop_overlap_accounting="each native pixel once; fixed midpoint ownership",
                  evaluation_scope="annotated opaque non-crowd union only; not full-image background evaluation",
                  image_records=records)
    if preview_buffer is not None:
        preview_buffer.extend(previews)
    if gallery_out is not None:
        from .metrics import write_gallery
        write_gallery(gallery_out, previews, data.categories, split=split)
    return report


def train_evaluation_epochs(epochs, fraction):
    """Full original TRAIN evaluations; VAL remains an every-epoch operation."""
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("train_eval_fraction must be finite and in (0, 1]")
    interval = max(1, math.ceil(epochs * fraction))
    return list(range(interval, epochs + 1, interval)) + ([] if epochs % interval == 0 else [epochs])


class OnlineTrainingMetrics:
    """Accumulate existing optimizer-batch predictions without another forward.

    These diagnostics pool repeated crop/augmentation pixels from changing model
    states. They intentionally do not claim native-pixel or facade aggregation.
    """
    def __init__(self, categories, groups, device):
        self.categories, self.groups = categories, groups
        self.counts = torch.zeros((len(categories), 3), dtype=torch.int64, device=device)
        self.group_counts = torch.zeros((len(groups), 3), dtype=torch.int64, device=device)
        self.stats = torch.zeros((4, len(categories)), dtype=torch.float64, device=device)
        self.valid_pixels = torch.zeros((), dtype=torch.int64, device=device)
        self.samples = 0

    @torch.no_grad()
    def update(self, logits, target, valid):
        values, truth, support = logits.detach().float(), target.bool(), valid[:, None]
        prediction = values >= 0
        axes = (0, 2, 3)
        self.counts += torch.stack(((prediction & truth & support).sum(axes),
                                   (prediction & ~truth & support).sum(axes),
                                   (~prediction & truth & support).sum(axes)), dim=1)
        for index, group in enumerate(self.groups):
            pred_group = prediction[:, group["channels"]].any(1) & valid
            true_group = truth[:, group["channels"]].any(1) & valid
            self.group_counts[index] += torch.stack(((pred_group & true_group).sum(),
                                                     (pred_group & ~true_group).sum(),
                                                     (~pred_group & true_group).sum()))
        probability = values.sigmoid()
        bce = torch.nn.functional.softplus(values) - values * target
        # Accumulate compact C-channel statistics on device, transfer once/epoch.
        self.stats += torch.stack(((bce * support).sum(axes),
                                   (probability * support).sum(axes),
                                   (target * support).sum(axes),
                                   (probability * target * support).sum(axes))).double()
        self.valid_pixels += valid.sum()
        self.samples += len(logits)

    def report(self, optimizer_loss):
        arrays = self.stats.cpu().tolist()
        stats = dict(zip(("bce_sum", "probability_sum", "target_sum", "intersection"), arrays))
        stats["valid_pixels"] = int(self.valid_pixels.item())
        counts, groups = self.counts.cpu().numpy(), self.group_counts.cpu().numpy()
        classes = _class_metrics(counts, self.categories, stats)
        return {"split": "train", "diagnostic_only": True, "test_evaluated": False,
                "aggregation": "pixel-pooled optimizer samples; original and fixed augmented crops, repeated overlaps",
                "prediction_source": "existing train-mode batch logits during changing model weights; no extra forward",
                "checkpoint_selection": False, "samples": self.samples,
                "valid_pixel_observations": stats["valid_pixels"],
                "optimizer_loss": optimizer_loss,
                "optimizer_loss_aggregation": "mean mini-batch objectives during weight updates",
                "primary": _summary(classes, stats), "per_class": classes,
                "per_group": _group_metrics(groups, self.groups, stats)}


def _save_checkpoint(path, model, optimizer, epoch, config, provenance, score, train_score=None):
    value = {"schema_version": 1, "kind": "frozen_dinov2_semantic_head", "epoch": epoch,
             "head_state_dict": {key: tensor.detach().cpu() for key, tensor in model.head.state_dict().items()},
             "optimizer_state_dict": optimizer.state_dict(), "config": config,
             "backbone_provenance": provenance, "val_miou": score,
             "train_miou": train_score,
             "selection_metric": ("unselected_last_epoch" if path.name == "last.pth" else
                                  "original_train_facade_miou" if train_score is not None else
                                  "val_facade_miou")}
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _format_metrics(values):
    parts = []
    for key, label in (("loss", "loss"), ("f1", "F1"), ("miou", "mIoU"),
                       ("precision", "Precision"), ("recall", "Recall")):
        value = values.get(key)
        text = "n/a" if value is None else (f"{value:.4f}" if key == "loss" else f"{value:.2%}")
        parts.append(f"{label}={text}")
    return "; ".join(parts)


def _print_epoch_report(split, report):
    print(f"{split.upper()} | {_format_metrics(report['primary'])}", flush=True)
    for group in report.get("per_group", []):
        print(f"  {group['name']}: {_format_metrics(group)}", flush=True)
    auxiliary = report.get("auxiliary_image_average", {})
    if auxiliary.get("image_count"):
        print(f"  TRAIN без установленного фасада, {auxiliary['image_count']} фото "
              f"(диагностика по изображениям): {_format_metrics(auxiliary)}", flush=True)


def train(dataset, out, augmentations=None, methods_config=None, source_root=None,
          checkpoint=None, device="cuda:0", epochs=20, batch_size=4,
          learning_rate=1e-3, seed=42, no_amp=False, smoke=False,
          trust_checkpoint=False, class_groups=None, train_eval_fraction=.2):
    if min(epochs, batch_size) < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Epochs, batch size and learning rate must be positive")
    if not augmentations:
        raise ValueError("Training requires --augmentations from a completed --prepare-only run")
    train_evaluation_epochs(epochs, train_eval_fraction)
    root, weights, config_hashes = dino_paths(methods_config, source_root, checkpoint)
    data = PreparedData(dataset)
    prepared = PreparedAugmentations(augmentations, data)
    groups = resolve_groups(data.categories, class_groups)
    train_ids, val_ids = data.image_ids("train"), data.image_ids("val")
    if not train_ids or not val_ids:
        raise ValueError("Both TRAIN and VAL are required")
    if smoke:
        train_ids, val_ids = train_ids[:2], val_ids[:2]
        epochs = 1
    train_eval_epochs = train_evaluation_epochs(epochs, train_eval_fraction)
    selected_train = set(train_ids)
    all_samples = [item for item in prepared.records
                   if item["split"] == "train" and item["image_id"] in selected_train]
    samples = [item for item in all_samples if item["valid_pixels"] > 0]
    if smoke:
        # Only the run is partial; the reusable preparation always remains complete.
        samples = samples[:min(12, len(samples))]
    if not samples:
        raise ValueError("Prepared TRAIN has no supervised samples")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    gpu = gpu_preflight(device)
    amp = str(device).startswith("cuda") and not no_amp
    config = {"dataset": str(data.dataset_path), "dataset_provenance": data.metadata,
              "source_root": root, "checkpoint": weights, "methods_config_sha256": config_hashes,
              "device": str(device), "epochs": epochs, "batch_size": batch_size,
              "learning_rate": learning_rate, "seed": seed,
              "amp": amp, "smoke": smoke, "trust_checkpoint": trust_checkpoint,
              "categories": data.categories, "category_to_channel": data.metadata["category_to_channel"],
              "class_groups": groups,
              "class_groups_file_sha256": sha256(class_groups) if class_groups else None,
              "training_image_ids": train_ids, "validation_image_ids": val_ids,
              "sampler": "all fixed original+two-augmentation TRAIN samples once per epoch; shuffled order only",
              "training_samples": len(samples), "prepared_train_samples": len(all_samples),
              "skipped_unannotated_samples": sum(item["valid_pixels"] == 0 for item in all_samples),
              "augmentations": prepared.metadata,
              "train_eval_fraction": train_eval_fraction, "train_eval_epochs": train_eval_epochs,
              "epoch_evaluation": "VAL originals every epoch; TRAIN originals on scheduled epochs; fixed native-pixel ownership",
              "online_train_evaluation": "pixel-pooled existing optimizer logits every epoch; augmented and repeated crop pixels; changing weights",
              "unknown_train_aggregation": "separate equal-image diagnostic, never counted as known facades",
              "decision": "independent sigmoid >=0.5; primary best.pth by equal-facade VAL mIoU; diagnostic best-train.pth by scheduled original TRAIN known-facade mIoU",
              "supervision_assumption": "inside annotated union, unmarked categories are negative; outside union all categories unknown"}
    destination = new_directory(out)
    record = run_record("dino_semantic_training", config)
    record["gpu_preflight"] = gpu
    record["environment"]["packages"]["pycocotools"] = importlib.metadata.version("pycocotools")
    write_json(destination / "run.json", record)
    write_json(destination / "config.json", config)
    try:
        print(f"TRAIN: {len(train_ids)} фото, {len(samples)} сохранённых примеров/эпоха; "
              f"VAL: {len(val_ids)} фото; классов: {len(data.categories)}; TEST не оценивается.", flush=True)
        print(f"Результаты: {destination}", flush=True)
        print("DINOv2 заморожен. Аугментации читаются с диска. VAL — на оригиналах каждую эпоху.", flush=True)
        print(f"Полная TRAIN оценка: эпохи {train_eval_epochs}; TRAIN online — каждую эпоху на учебных batch.", flush=True)
        model, provenance = load_model(root, weights, len(data.categories), device, trust_checkpoint)
        record["model"] = provenance
        write_json(destination / "run.json", record)
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=learning_rate, weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = (torch.amp.GradScaler("cuda", enabled=amp)
                  if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")
                  else torch.cuda.amp.GradScaler(enabled=amp))
        best, best_epoch, history = -1., 0, []
        best_train_score, best_train_epoch = -1., None
        from .metrics import write_gallery
        for epoch in range(1, epochs + 1):
            start = time.monotonic()
            model.train()
            losses = []
            online = OnlineTrainingMetrics(data.categories, groups, device)
            batches = fixed_batches(prepared, samples, batch_size, np.random.default_rng(seed + epoch))
            progress = tqdm(batches, total=math.ceil(len(samples) / batch_size),
                            desc=f"DINO TRAIN {epoch}/{epochs}", unit="batch")
            for crops in progress:
                rgb, target, valid = _tensor_batch(crops, device)
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device, amp):
                    logits = model(rgb)
                    loss = masked_multilabel_loss(logits, target, valid)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss; reduce LR or try --no-amp")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise RuntimeError("Non-finite training gradients; try --no-amp")
                scaler.step(optimizer)
                scaler.update()
                online.update(logits, target, valid)
                losses.append(float(loss.detach()))
                progress.set_postfix(loss=f"{np.mean(losses):.4f}")
            online_report = online.report(float(np.mean(losses)))
            write_json(destination / f"train-online-epoch-{epoch:03d}.json", online_report)
            _print_epoch_report("train online (augmented, pooled)", online_report)
            reports, previews = {}, {}
            splits = [("train", train_ids)] if epoch in train_eval_epochs else []
            splits.append(("val", val_ids))
            for split, ids in splits:
                previews[split] = []
                reports[split] = evaluate(model, data, ids, device, batch_size, amp, smoke,
                                          split=split, groups=groups, prepared=prepared,
                                          preview_buffer=previews[split])
                write_json(destination / f"{split}-epoch-{epoch:03d}.json", reports[split])
                write_reports(destination / f"epoch-{epoch:03d}" / split, reports[split])
                _print_epoch_report(split, reports[split])
            score = reports["val"]["primary"]["miou"]
            if score is None:
                raise RuntimeError("VAL has no defined annotated-support IoU")
            entry = {"epoch": epoch, "optimizer_loss": float(np.mean(losses)),
                     "optimizer_loss_aggregation": "mean augmented mini-batch objectives during weight updates",
                     "learning_rate": optimizer.param_groups[0]["lr"],
                     "seconds": time.monotonic() - start,
                     "train_evaluated": "train" in reports,
                     "train_online": online_report["primary"],
                     "train": reports.get("train", {}).get("primary"), "val": reports["val"]["primary"]}
            for split in ("train", "val"):
                for metric in ("miou", "f1", "precision", "recall", "loss"):
                    entry[f"{split}_{metric}"] = reports.get(split, {}).get("primary", {}).get(metric)
            history.append(entry)
            write_json(destination / "history.json", history)
            _save_checkpoint(destination / "last.pth", model, optimizer, epoch, config, provenance, score)
            if score > best:
                best, best_epoch = score, epoch
                _save_checkpoint(destination / "best.pth", model, optimizer, epoch, config, provenance, score)
                write_json(destination / "best-val.json", reports["val"])
                write_json(destination / "train-at-best-val.json", {
                    "epoch": epoch, "evaluated": "train" in reports,
                    "report": reports.get("train"), "online_diagnostic": online_report})
                write_gallery(destination / "best-val", previews["val"], data.categories,
                              split="val", epoch=epoch, checkpoint_name="best.pth")
            train_score = reports.get("train", {}).get("primary", {}).get("miou")
            if train_score is not None and train_score > best_train_score:
                best_train_score, best_train_epoch = train_score, epoch
                _save_checkpoint(destination / "best-train.pth", model, optimizer, epoch,
                                 config, provenance, score, train_score=train_score)
                write_json(destination / "best-train.json", reports["train"])
                write_gallery(destination / "best-train", previews["train"], data.categories,
                              split="train", epoch=epoch, checkpoint_name="best-train.pth")
                print(f"Сохранён best-train.pth: TRAIN mIoU={train_score:.2%}, эпоха {epoch}", flush=True)
            scheduler.step()
            print(f"Эпоха {epoch}: лучший VAL mIoU={best:.2%} (эпоха {best_epoch})", flush=True)
        saved = torch.load(destination / "best.pth", map_location="cpu", weights_only=True)
        model.head.load_state_dict(saved["head_state_dict"], strict=True)
        # The saved report and gallery already correspond to the exact best VAL
        # head. Reuse them instead of running another full inference pass.
        final = read_json(destination / "best-val.json")
        write_json(destination / "metrics.json", final)
        write_reports(destination, final)
        shutil.copy2(destination / "best-val" / "gallery.html", destination / "gallery.html")
        shutil.copytree(destination / "best-val" / "previews", destination / "previews")
        train_best = read_json(destination / "best-train.json") if best_train_epoch is not None else None
        if train_best is not None:
            write_json(destination / "train-metrics.json", train_best)
            write_reports(destination / "train", train_best)
        summary = {"status": "completed_smoke" if smoke else "completed_exploratory", "out": str(destination),
                   "best_epoch": best_epoch, "best_val_miou": best, "metrics": final["primary"],
                   "best_train_epoch": best_train_epoch,
                   "best_train_miou": best_train_score if best_train_epoch is not None else None,
                   "train_metrics": train_best["primary"] if train_best is not None else None,
                   "train_metrics_checkpoint": "best-train.pth" if train_best is not None else None,
                   "train_checkpoint_selection": "known-facade original TRAIN mIoU among scheduled evaluations only; diagnostic, not model selection",
                   "train_eval_epochs": train_eval_epochs,
                   "training_samples_per_epoch": len(samples),
                   "augmentations": str(prepared.path), "test_evaluated": False,
                   "scope": "semantic multilabel metrics on annotated union; no background or temporal-change claim",
                   "physical_disjointness_verified": data.metadata["leakage_check"]["physical_disjointness_verified"]}
        write_json(destination / "summary.json", summary)
        finish_record(destination, record, summary["status"])
        return summary
    except BaseException as error:
        finish_record(destination, record, "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", str(error))
        raise
