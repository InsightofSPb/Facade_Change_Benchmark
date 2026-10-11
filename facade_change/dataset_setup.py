"""Two-path preparation using the existing inventory and frozen facade split."""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .data import DEFAULT_METADATA_RULES, build_manifest, normalized_name, preprocess_coco, read_rename_maps
from .io import finish_record, new_directory, read_json, run_record, sha256, write_json
from .preparation import prepare_dataset, semantic_crop_boxes, semantic_crop_count


def _names(images):
    return {normalized_name(Path(str(row.get(key) or "")).name)
            for row in images for key in ("file_name", "resolved_file_name")
            if row.get(key)}


def _cohort(split, key):
    return {tuple(row.get(field) for field in
                  ("sha256", "building_id", "view_id", "year", "split"))
            for row in split.get(key, [])}


def _validate_previous(path, split=None, record=None):
    split = split if split is not None else read_json(path)
    record = record if record is not None else read_json(path.with_name("run.json"))
    if (split.get("mode") != "reviewed" or split.get("development_only") is not False or
            record.get("kind") != "dataset_preparation" or
            not record.get("status", "").startswith("completed")):
        raise ValueError(f"Previous split must come from a completed reviewed preparation: {path}")
    if not all(split.get(key) for key in
               ("gold_cohort", "historical_cohort", "building_assignments", "provenance", "fractions")):
        raise ValueError(f"Reviewed split lacks cohort provenance: {path}; review its migration")
    provenance, config = split["provenance"], record.get("config", {})
    if (not provenance.get("gold_manifest_sha256") or
            provenance.get("manifest_sha256") != config.get("manifest_sha256") or
            split.get("seed") != config.get("seed") or
            split["fractions"].get("val") != config.get("val_fraction") or
            split["fractions"].get("test") != config.get("test_fraction")):
        raise ValueError(f"Split provenance disagrees with its preparation run: {path}")
    for artifact in (path, path.with_name("manifest.json")):
        digest = record.get("artifact_sha256", {}).get(artifact.name)
        if not digest or sha256(artifact) != digest:
            raise ValueError(f"Previous preparation artifact is unverified or changed: {artifact}")
    return split


def discover_previous_split(runs_root, image_names):
    """Select a completed, compatible extension; never choose by directory mtime.

    Only actual preparation runs qualify, not benchmark copies of split.json.
    Incomparable histories require an explicit choice instead of losing owners.
    """
    candidates = []
    for path in sorted(Path(runs_root).rglob("split.json")):
        manifest_path, record_path = path.with_name("manifest.json"), path.with_name("run.json")
        if not manifest_path.is_file() or not record_path.is_file():
            continue
        try:
            record = read_json(record_path)
            if (record.get("kind") != "dataset_preparation" or
                    not record.get("status", "").startswith("completed")):
                continue
            split = read_json(path)
            if split.get("mode") != "reviewed" or split.get("development_only") is not False:
                continue
            manifest = read_json(manifest_path)
            historical_images = manifest.get("images", []) + split.get("historical_cohort", []) + split.get("gold_cohort", [])
            if not _names(historical_images) & image_names:
                continue
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"Cannot inspect previous preparation {path}: {exc}") from exc
        _validate_previous(path, split, record)
        candidates.append((path.resolve(), split, record.get("finished_utc", "")))
    if not candidates:
        return None
    first = candidates[0][1]
    for path, split, _ in candidates:
        if (split.get("seed") != first.get("seed") or split.get("fractions") != first.get("fractions") or
                split["provenance"].get("gold_manifest_sha256") != first["provenance"].get("gold_manifest_sha256") or
                _cohort(split, "gold_cohort") != _cohort(first, "gold_cohort")):
            raise ValueError("Several original reviewed splits match these images; choose --previous-split explicitly: " +
                             ", ".join(str(item[0]) for item in candidates))
    all_owners, all_history = {}, set()
    for path, split, _ in candidates:
        for building, owner in split["building_assignments"].items():
            if building in all_owners and all_owners[building] != owner:
                raise ValueError(f"Conflicting historical split for {building}: {path}; choose --previous-split explicitly")
            all_owners[building] = owner
        all_history.update(_cohort(split, "historical_cohort"))
    complete = [(path, split, stamp) for path, split, stamp in candidates
                if split["building_assignments"] == all_owners and
                _cohort(split, "historical_cohort") == all_history]
    if not complete:
        raise ValueError("Reviewed split histories branched; choose --previous-split explicitly: " +
                         ", ".join(str(item[0]) for item in candidates))
    return max(complete, key=lambda item: (item[2], str(item[0])))[0]


def _csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _image_reports(out, images):
    fields = ["image_id", "file_name", "image_path", "image_status", "resolution_rule",
              "view_id", "building_id", "year", "reviewed", "notes", "split", "exclusion_reasons"]
    rows = [{**row, "reviewed": "true" if row.get("metadata_status") == "reviewed" else "false",
             "notes": row.get("metadata_notes", ""),
             "exclusion_reasons": ";".join(row.get("preparation_exclusion_reasons", []))}
            for row in images]
    _csv(out / "metadata_review_with_names.csv", rows, fields)
    _csv(out / "missing_images.csv", [r for r in rows if r["image_status"] == "missing"], fields)
    _csv(out / "image_issues.csv", [r for r in rows if r["image_status"] != "ready"], fields)


def _crop_index(out, images, crop_size, crop_stride):
    counts = Counter()
    with (out / "semantic_crops.jsonl").open("w", encoding="utf-8") as stream:
        for row in images:
            if row["split"] not in {"train", "val", "test"}:
                continue
            for x, y, width, height in semantic_crop_boxes(row["width"], row["height"], crop_size, crop_stride):
                crop = {"crop_id": f"{row['image_id']}:{x}:{y}:{crop_size}",
                        "image_id": row["image_id"], "file_name": row["file_name"],
                        "image_path": row["image_path"], "image_sha256": row["sha256"],
                        "building_id": row["building_id"], "view_id": row["view_id"], "split": row["split"],
                        "x": x, "y": y, "width": width, "height": height,
                        "pad_right": crop_size - width, "pad_bottom": crop_size - height}
                stream.write(json.dumps(crop, ensure_ascii=False, allow_nan=False) + "\n")
                counts[row["split"]] += 1
    return {name: counts[name] for name in ("train", "val", "test")}


def prepare_from_paths(coco_path, image_root, out=None, runs_root=None, previous_split=None,
                       metadata_rules=DEFAULT_METADATA_RULES, rename_maps=None,
                       crop_size=512, crop_stride=448, metadata_csv=None):
    """Audit current COCO, extend reviewed owners and index native semantic crops.

    Does not align temporal pairs, rasterize masks, train models or copy images.
    Extra files outside COCO are never added to the dataset.
    """
    coco_path, image_root = Path(coco_path).expanduser().resolve(), Path(image_root).expanduser().resolve()
    if not coco_path.is_file() or not image_root.is_dir():
        raise ValueError("COCO must be an existing file and images must be an existing directory")
    semantic_crop_count(1, 1, crop_size, crop_stride)  # Validate geometry before creating a run.
    runs_root = Path(runs_root or Path(__file__).resolve().parents[1] / "runs").expanduser().resolve()
    rename_maps = sorted({Path(p).expanduser().resolve() for p in
                          (rename_maps if rename_maps is not None else image_root.glob("*image_rename_map.csv"))})
    metadata_rules = Path(metadata_rules).expanduser().resolve() if metadata_rules else None
    metadata_csv = Path(metadata_csv).expanduser().resolve() if metadata_csv else None
    input_hashes = {str(path): sha256(path) for path in
                    [coco_path, *rename_maps, *([metadata_rules] if metadata_rules else []),
                     *([metadata_csv] if metadata_csv else [])]}
    coco = read_json(coco_path)
    if not isinstance(coco.get("images"), list) or not coco["images"]:
        raise ValueError("COCO images must be a nonempty list")
    image_names = _names(coco["images"])
    # Rename aliases help discover the old preparation even after a full rename.
    rename_groups, _, _ = read_rename_maps(rename_maps)
    for group in set(rename_groups.values()):
        names = {normalized_name(Path(name).name) for name in group}
        if names & image_names:
            image_names.update(names)
    split_error = None
    previous = None
    try:
        previous_path = (Path(previous_split).expanduser().resolve() if previous_split else
                         discover_previous_split(runs_root, image_names))
        previous = _validate_previous(previous_path) if previous_path else None
    except (ValueError, OSError) as exc:
        previous_path, split_error = None, str(exc)
    previous_manifest = previous_path.with_name("manifest.json") if previous_path else None
    if previous_manifest and not previous_manifest.is_file():
        raise ValueError(f"Previous split requires its prepared manifest: {previous_manifest}")
    if previous_path:
        input_hashes[str(previous_path)] = sha256(previous_path)
        input_hashes[str(previous_manifest)] = sha256(previous_manifest)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = new_directory(out or runs_root / f"{stamp}-dataset-preparation")
    config = {"coco_json": str(coco_path), "image_roots": [str(image_root)],
              "rename_maps": [str(p) for p in rename_maps],
              "metadata_rules": str(metadata_rules) if metadata_rules else None,
              "metadata_csv": str(metadata_csv) if metadata_csv else None,
              "previous_split": str(previous_path) if previous_path else None,
              "balance_unit": "crops", "crop_size": crop_size, "crop_stride": crop_stride,
              "input_sha256": input_hashes, "image_selection": "current_coco_images_only"}
    record = run_record("dataset_setup", config)
    write_json(out / "run.json", record)
    print(f"Проверка COCO: {len(coco['images'])} изображений. Выход: {out}", flush=True)
    selection = str(previous_path) if previous_path else ('не выбрано' if split_error else 'не найдено; первое reviewed-разбиение')
    print(f"Предыдущее разбиение: {selection}", flush=True)
    if split_error:
        print(f"Требуется выбор прежнего split: {split_error}", flush=True)
    print(f"Rename-map: {', '.join(str(p) for p in rename_maps) or 'не найден; используются точные имена и хеш-префиксы'}", flush=True)
    try:
        preprocessing = preprocess_coco(coco_path, out / "preprocessed")
        inventory_config = {"coco_json": preprocessing["coco_path"], "image_roots": [str(image_root)],
                            "rename_maps": config["rename_maps"], "metadata_rules": config["metadata_rules"]}
        write_json(out / "inventory_config.json", inventory_config)
        inventory = build_manifest(out / "inventory_config.json", out / "inventory",
                                   overrides=metadata_csv, previous_manifest=previous_manifest)
        images = inventory["images"]
        issues = [row for row in images if row["image_status"] != "ready"]
        for row in issues:
            print(f"{row['image_status']}: {row['file_name']} (COCO image_id={row['image_id']})", flush=True)
            for candidate in row.get("path_candidates", []):
                print(f"  кандидат: {candidate}", flush=True)
        _image_reports(out, images)
        summary = {"out": str(out), "image_count": len(images), "ready_image_count": len(images) - len(issues),
                   "all_images_ready": not issues, "all_files_present": all(row.get("image_path") for row in images),
                   "missing_image_names": [row["file_name"] for row in images if row["image_status"] == "missing"],
                   "image_status": inventory["summary"]["image_status"],
                   "inventory": inventory["summary"], "annotation_preprocessing": preprocessing,
                   "previous_split": config["previous_split"], "split_selection_error": split_error,
                   "crop_size": crop_size, "crop_stride": crop_stride, "crop_splits": {},
                   "crop_count": 0, "status": "needs_split_review" if split_error else "running"}
        annotation_issues = {key: inventory["summary"][key] for key in
                             ("orphan_annotation_ids", "unknown_category_annotation_ids")}
        summary["annotation_issues"] = annotation_issues
        for key, values in annotation_issues.items():
            if values:
                print(f"{key}: {values}", flush=True)
        if not split_error:
            fractions = previous["fractions"] if previous else {"train": .70, "val": .10, "test": .20}
            reviewed = [row for row in images if row["image_status"] == "ready" and
                        row.get("metadata_status") == "reviewed" and row.get("building_id") and row.get("view_id")
                        and type(row.get("year")) is int and 1800 <= row["year"] <= 2100]
            if not reviewed or (previous is None and len({row["building_id"] for row in reviewed}) <
                                sum(fraction > 0 for fraction in fractions.values())):
                summary["status"] = "needs_metadata_review"
                print("Недостаточно подтверждённых фасадов для split; см. metadata_review_with_names.csv", flush=True)
            else:
                prepared = prepare_dataset(out / "inventory/manifest.json", out / "prepared", split_mode="reviewed",
                                           previous_split=previous_path, seed=previous["seed"] if previous else 42,
                                           val_fraction=fractions["val"], test_fraction=fractions["test"],
                                           balance_unit="crops", crop_size=crop_size, crop_stride=crop_stride)
                images = prepared["images"]
                _image_reports(out, images)
                counts = _crop_index(out, images, crop_size, crop_stride)
                balance = prepared["summary"]["split_balance"]
                if sum(counts.values()) != balance["eligible_crop_count"]:
                    raise ValueError("Crop index count differs from the split balancing count")
                summary.update(preparation=prepared["summary"], balance=balance,
                               crop_count=sum(counts.values()), crop_splits=counts)
                unreviewed = [row for row in images if row["image_status"] == "ready" and row["split"] == "excluded"]
                summary["metadata_review_names"] = [row["file_name"] for row in unreviewed]
                for row in unreviewed:
                    print(f"Нужны метаданные: {row['file_name']} — {', '.join(row['preparation_exclusion_reasons'])}", flush=True)
                summary["status"] = ("needs_metadata_review" if unreviewed else "completed_with_issues"
                                     if issues or any(annotation_issues.values()) or
                                     prepared["summary"]["missing_gold_sha256"] else "completed")
                write_json(out / "dataset.json", {
                    "schema_version": 1, "kind": "semantic_crop_index", "source_coco": preprocessing["coco_path"],
                    "source_coco_sha256": sha256(preprocessing["coco_path"]), "original_coco": str(coco_path),
                    "manifest": str(out / "prepared/manifest.json"), "split": str(out / "prepared/split.json"),
                    "crops": str(out / "semantic_crops.jsonl"), "crop_size": crop_size, "stride": crop_stride,
                    "crop_splits": counts, "categories": prepared["categories"],
                    "coordinate_frame": "native_stored_RGB_no_EXIF_rotation",
                    "edge_policy": "anchor_last_crop_to_edge; small_images_pad_right_bottom",
                    "labels": "original COCO IDs and polygons; overlaps preserved; masks not rasterized",
                    "unannotated_pixels": "unknown, not automatically background",
                    "padding": "invalid support; ignore in training and evaluation",
                    "temporal_pairs": str(out / "prepared/pairs.csv"),
                    "temporal_alignment": "not_run", "temporal_change_ground_truth": "not_created"})
        for path, digest in input_hashes.items():
            if sha256(path) != digest:
                raise ValueError(f"Input changed during preparation: {path}")
        lines = [f"Статус: {summary['status']}",
                 f"COCO: {len(images)}; проверены и пригодны: {summary['ready_image_count']}; проблемные: {len(issues)}",
                 f"Предыдущий split: {previous_path}", f"Кропы: {crop_size}×{crop_size}, шаг {crop_stride}"]
        for name, stats in summary.get("balance", {}).get("partitions", {}).items():
            lines.append(f"{name}: {stats['building_count']} фасадов; {stats['image_count']} фото "
                         f"({stats['image_fraction']:.2%}); {stats['crop_count']} кропов "
                         f"({stats['crop_fraction']:.2%}); цель {stats['target_crop_fraction']:.2%}")
        lines.extend(["Старые фасады сохраняют split. Баланс приблизительный; кропы одного фасада не разделяются.",
                      "Кропы записаны координатами; исходные фото и COCO не изменены.",
                      "Число кропов геометрическое: фильтрация по поддержке/разметке может изменить доли.",
                      f"Отчёт: {out / 'summary.json'}", f"Проверка имён: {out / 'metadata_review_with_names.csv'}"])
        if split_error:
            lines.append(split_error)
        text = "\n".join(lines) + "\n"
        write_json(out / "summary.json", summary)
        (out / "summary.txt").write_text(text, encoding="utf-8")
        record["summary"] = summary
        finish_record(out, record, summary["status"])
        print(text, end="", flush=True)
        return summary
    except (Exception, KeyboardInterrupt) as exc:
        finish_record(out, record, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", str(exc))
        raise
