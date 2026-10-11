"""Two-path inventory and versioned physical-building split preparation."""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .data import DEFAULT_METADATA_RULES, build_manifest, normalized_name, preprocess_coco, read_rename_maps
from .io import finish_record, new_directory, read_json, run_record, sha256, write_json
from .preparation import prepare_dataset, semantic_crop_boxes, semantic_crop_count
from .semantic_training import include_auxiliary_training


def _names(images):
    return {normalized_name(Path(str(row.get(key) or "")).name)
            for row in images for key in ("file_name", "resolved_file_name")
            if row.get(key)}


def _cohort(split, key):
    return {tuple(row.get(field) for field in
                  ("sha256", "building_id", "view_id", "year", "split"))
            for row in split.get(key, [])}


def _merged_semantic_history(splits):
    """Permit null chronology to be filled while keeping byte/group ownership fixed."""
    history = {}
    for split in splits:
        for row in split["historical_cohort"]:
            digest = row["sha256"]
            current = {key: row.get(key) for key in ("building_id", "view_id", "year", "split")}
            old = history.get(digest)
            if old:
                for key, value in current.items():
                    if key in {"view_id", "year"} and (not value or not old[key]):
                        current[key] = value or old[key]
                    elif value != old[key]:
                        raise ValueError("Conflicting reviewed image history; choose --previous-split explicitly")
            history[digest] = current
    return history


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
            split.get("split_policy", "legacy") != config.get("split_policy", "legacy") or
            split["fractions"].get("val") != config.get("val_fraction") or
            split["fractions"].get("test") != config.get("test_fraction")):
        raise ValueError(f"Split provenance disagrees with its preparation run: {path}")
    for artifact in (path, path.with_name("manifest.json")):
        digest = record.get("artifact_sha256", {}).get(artifact.name)
        if not digest or sha256(artifact) != digest:
            raise ValueError(f"Previous preparation artifact is unverified or changed: {artifact}")
    return split


def discover_previous_split(runs_root, image_names, split_policy=None):
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
            if split_policy is not None and split.get("split_policy", "legacy") != split_policy:
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
                (_merged_semantic_history([split]) == _merged_semantic_history([item[1] for item in candidates])
                 if first.get("split_policy", "legacy") != "legacy" else
                 _cohort(split, "historical_cohort") == all_history)]
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
              "view_id", "building_id", "year", "reviewed", "notes", "split", "exclusion_reasons",
              "semantic_eligible", "temporal_eligible", "temporal_exclusion_reasons", "training_role"]
    rows = [{**row, "reviewed": "true" if row.get("metadata_status") == "reviewed" else "false",
             "notes": row.get("metadata_notes", ""),
             "temporal_exclusion_reasons": ";".join(row.get("temporal_exclusion_reasons", [])),
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
                        "building_id": row["building_id"], "view_id": row.get("view_id"), "split": row["split"],
                        "training_role": row.get("training_role", "reviewed"),
                        "x": x, "y": y, "width": width, "height": height,
                        "pad_right": crop_size - width, "pad_bottom": crop_size - height}
                stream.write(json.dumps(crop, ensure_ascii=False, allow_nan=False) + "\n")
                counts[row["split"]] += 1
    return {name: counts[name] for name in ("train", "val", "test")}


def _class_coverage(out, prepared, coco):
    """Report annotation presence, not pixel coverage or segmentation quality."""
    images = {row["image_id"]: row for row in prepared["images"]
              if row.get("split") in {"train", "val", "test"}}
    counts, image_ids, buildings = Counter(), {}, {}
    for annotation in coco["annotations"]:
        row = images.get(annotation["image_id"])
        if row is None:
            continue
        key = (row["split"], annotation["category_id"])
        counts[key] += 1
        image_ids.setdefault(key, set()).add(row["image_id"])
        if row.get("building_id") and row.get("metadata_status") == "reviewed":
            buildings.setdefault(key, set()).add(row["building_id"])
    rows = [{"split": split, "category_id": category["id"], "category_name": category["name"],
             "annotation_count": counts[(split, category["id"])],
             "image_count": len(image_ids.get((split, category["id"]), set())),
             "building_count": len(buildings.get((split, category["id"]), set()))}
            for split in ("train", "val", "test") for category in coco["categories"]]
    _csv(out / "class_coverage.csv", rows,
         ["split", "category_id", "category_name", "annotation_count", "image_count", "building_count"])
    missing = {split: [row["category_name"] for row in rows
                      if row["split"] == split and not row["annotation_count"]]
               for split in ("train", "val", "test")}
    for split, names in missing.items():
        if names:
            print(f"Нет аннотаций классов в {split}: {', '.join(names)}", flush=True)
    return {"path": str(out / "class_coverage.csv"), "missing_classes": missing,
            "scope": "COCO annotation presence; no pixel rasterization; not used to optimize split"}


def _split_transition(out, previous_path, previous, prepared):
    current = read_json(out / "prepared/split.json")
    before, after = previous["building_assignments"], current["building_assignments"]
    changes = [{"building_id": building, "previous": before.get(building), "current": after.get(building)}
               for building in sorted(before.keys() | after.keys()) if before.get(building) != after.get(building)]
    report = {"previous_split": str(previous_path), "previous_split_sha256": sha256(previous_path),
              "current_split": str(out / "prepared/split.json"),
              "current_split_sha256": sha256(out / "prepared/split.json"),
              "previous_policy": previous.get("split_policy", "legacy"),
              "current_policy": current["split_policy"], "building_changes": changes,
              "old_train_now_holdout": [row["building_id"] for row in changes
                                        if row["previous"] == "train" and row["current"] in {"val", "test"}],
              "absent_previous_gold_sha256": sorted({row["sha256"] for row in previous["gold_cohort"]} -
                                                   {row.get("sha256") for row in prepared["images"]}),
              "note": "New split version; previous outputs unchanged. Retrain any model exposed to new held-out buildings."}
    write_json(out / "split_transition.json", report)
    print(f"Новая версия split: изменено назначений/состава {len(changes)} зданий; "
          f"из прежнего TRAIN в VAL/TEST: {len(report['old_train_now_holdout'])}. "
          "См. split_transition.json; старые обученные модели не являются baseline нового split.", flush=True)
    return report


def prepare_from_paths(coco_path, image_root, out=None, runs_root=None, previous_split=None,
                       metadata_rules=DEFAULT_METADATA_RULES, rename_maps=None,
                       crop_size=512, crop_stride=448, metadata_csv=None, split_policy="facades-v3",
                       include_unreviewed_train=None):
    """Audit current COCO, extend reviewed owners and index native semantic crops.

    Does not align temporal pairs, rasterize masks, train models or copy images.
    Extra files outside COCO are never added to the dataset.
    """
    coco_path, image_root = Path(coco_path).expanduser().resolve(), Path(image_root).expanduser().resolve()
    if split_policy not in {"legacy", "series-v2", "facades-v3"}:
        raise ValueError("split_policy must be legacy, series-v2 or facades-v3")
    if include_unreviewed_train is None:
        include_unreviewed_train = split_policy == "facades-v3"
    balance_unit = "buildings" if split_policy == "facades-v3" else "crops"
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
                         discover_previous_split(runs_root, image_names, split_policy))
        if previous_path is None and split_policy == "facades-v3":
            previous_path = discover_previous_split(runs_root, image_names, "series-v2")
        if previous_path is None and split_policy != "legacy":
            previous_path = discover_previous_split(runs_root, image_names, "legacy")
        previous = _validate_previous(previous_path) if previous_path else None
        if previous and split_policy == "legacy" and previous.get("split_policy", "legacy") != "legacy":
            raise ValueError("Use --split-policy series-v2 to extend a series-v2 split")
    except (ValueError, OSError) as exc:
        previous_path, split_error = None, str(exc)
    previous_manifest = previous_path.with_name("manifest.json") if previous_path else None
    migration = bool(previous and previous.get("split_policy", "legacy") != split_policy)
    inherited = previous if not migration else None
    inherited_path = previous_path if not migration else None
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
              "split_policy": split_policy, "new_split_version": migration,
              "include_unreviewed_train": include_unreviewed_train,
              "previous_split_use": "metadata_cache_and_transition_only" if migration else "frozen_extension",
              "balance_unit": balance_unit, "crop_size": crop_size, "crop_stride": crop_stride,
              "input_sha256": input_hashes, "image_selection": "current_coco_images_only"}
    record = run_record("dataset_setup", config)
    write_json(out / "run.json", record)
    print(f"Проверка COCO: {len(coco['images'])} изображений. Выход: {out}", flush=True)
    selection = str(previous_path) if previous_path else ('не выбрано' if split_error else 'не найдено; первое reviewed-разбиение')
    print(f"Предыдущее разбиение: {selection}", flush=True)
    print(f"Политика split: {split_policy}" + ("; новая версия, старые назначения не переносятся" if migration else ""), flush=True)
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
                   "split_policy": split_policy, "new_split_version": migration,
                   "crop_size": crop_size, "crop_stride": crop_stride, "crop_splits": {},
                   "crop_count": 0, "status": "needs_split_review" if split_error else "running"}
        annotation_issues = {key: inventory["summary"][key] for key in
                             ("orphan_annotation_ids", "unknown_category_annotation_ids")}
        summary["annotation_issues"] = annotation_issues
        for key, values in annotation_issues.items():
            if values:
                print(f"{key}: {values}", flush=True)
        if not split_error:
            fractions = inherited["fractions"] if inherited else (
                {"train": .80, "val": .08, "test": .12} if split_policy != "legacy" else
                {"train": .70, "val": .10, "test": .20})
            reviewed = [row for row in images if row["image_status"] == "ready" and
                        row.get("metadata_status") == "reviewed" and row.get("building_id") and row.get("sha256")
                        and (split_policy != "legacy" or (row.get("view_id") and
                        type(row.get("year")) is int and 1800 <= row["year"] <= 2100))]
            if not reviewed or (split_policy == "legacy" and inherited is None and len({row["building_id"] for row in reviewed}) <
                                sum(fraction > 0 for fraction in fractions.values())):
                summary["status"] = "needs_metadata_review"
                print("Недостаточно подтверждённых фасадов для split; см. metadata_review_with_names.csv", flush=True)
            else:
                prepared = prepare_dataset(out / "inventory/manifest.json", out / "prepared", split_mode="reviewed",
                                           previous_split=inherited_path, seed=inherited["seed"] if inherited else 42,
                                           val_fraction=fractions["val"], test_fraction=fractions["test"],
                                           balance_unit=balance_unit, crop_size=crop_size, crop_stride=crop_stride,
                                           split_policy=split_policy)
                semantic, auxiliary = include_auxiliary_training(
                    prepared, read_json(out / "prepared/split.json"), include_unreviewed_train)
                images = semantic["images"]
                _image_reports(out, images)
                counts = _crop_index(out, images, crop_size, crop_stride)
                balance = prepared["summary"]["split_balance"]
                extra_crops = sum(semantic_crop_count(row["width"], row["height"], crop_size, crop_stride)
                                  for row in images if row.get("training_role") == "auxiliary_semantic_train")
                if sum(counts.values()) != balance["eligible_crop_count"] + extra_crops:
                    raise ValueError("Crop index count differs from the split balancing count")
                summary.update(preparation=prepared["summary"], balance=balance,
                               crop_count=sum(counts.values()), crop_splits=counts,
                               auxiliary_train_image_count=len(auxiliary["included_images"]),
                               auxiliary_train_crop_count=extra_crops,
                               physical_disjointness_verified=auxiliary["physical_disjointness_verified"],
                               semantic_image_splits=dict(Counter(row["split"] for row in images)),
                               auxiliary_training=auxiliary)
                semantic_path = out / "semantic_manifest.json"
                semantic["summary"]["reviewed_eligible_crop_count"] = balance["eligible_crop_count"]
                semantic["summary"]["eligible_crop_count"] = sum(counts.values())
                write_json(semantic_path, semantic)
                write_json(out / "auxiliary_training.json", auxiliary)
                for row in auxiliary["included_images"]:
                    print(f"Семантический TRAIN: {row['file_name']} — адрес не установлен; временные пары исключены", flush=True)
                for row in auxiliary["blocked_images"] + auxiliary["duplicate_images"]:
                    print(f"Не добавлено в TRAIN: {row['file_name']} — {row['reason']}", flush=True)
                series = balance.get("series_constraints", {}).get("buildings", {})
                if series:
                    _csv(out / "building_series.csv", [
                        {"building_id": building, **stats,
                         "view_years": json.dumps(stats["view_years"], ensure_ascii=False)}
                        for building, stats in sorted(series.items())],
                        ["building_id", "split", "series_class", "max_distinct_years_per_view",
                         "image_count", "crop_count", "observations_without_temporal_metadata",
                         "historical_assignment", "view_years"])
                summary["class_coverage"] = _class_coverage(out, semantic, read_json(preprocessing["coco_path"]))
                if migration:
                    summary["split_transition"] = _split_transition(out, previous_path, previous, prepared)
                unreviewed = [row for row in images if row["image_status"] == "ready" and row["split"] == "excluded"]
                summary["metadata_review_names"] = [row["file_name"] for row in unreviewed]
                for row in unreviewed:
                    print(f"Нужны метаданные: {row['file_name']} — {', '.join(row['preparation_exclusion_reasons'])}", flush=True)
                if unreviewed and split_policy == "series-v2":
                    print("Доли кропов относятся к подтверждённой части. Добавление новых зданий может "
                          "изменить их: существующие назначения этой версии останутся фиксированными. "
                          "До обучения завершите проверку metadata_review_with_names.csv.", flush=True)
                summary["status"] = ("needs_metadata_review" if unreviewed else "completed_with_issues"
                                     if issues or any(annotation_issues.values()) or
                                     prepared["summary"]["missing_gold_sha256"] else "completed")
                if auxiliary["included_images"] and not unreviewed and not any(annotation_issues.values()):
                    summary["status"] = "completed_exploratory_train_only"
                constraints = balance.get("series_constraints", {})
                if constraints.get("status") == "insufficient":
                    summary["status"] = "needs_split_review"
                    for issue in constraints.get("issues", []):
                        print(f"Ограничение состава split: {issue}", flush=True)
                write_json(out / "dataset.json", {
                    "schema_version": 1, "kind": "semantic_crop_index", "source_coco": preprocessing["coco_path"],
                    "source_coco_sha256": sha256(preprocessing["coco_path"]), "original_coco": str(coco_path),
                    "manifest": str(semantic_path), "split": str(out / "prepared/split.json"),
                    "reviewed_manifest": str(out / "prepared/manifest.json"),
                    "physical_disjointness_verified": auxiliary["physical_disjointness_verified"],
                    "auxiliary_training": str(out / "auxiliary_training.json"),
                    "evaluation_scope": ("exploratory_until_training_building_membership_is_resolved"
                                         if auxiliary["included_images"] else "reviewed_building_split"),
                    "crops": str(out / "semantic_crops.jsonl"), "crop_size": crop_size, "stride": crop_stride,
                    "crop_splits": counts, "categories": prepared["categories"],
                    "split_policy": split_policy,
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
        lines.append("Подтверждённые здания (доли кропов справочные):")
        for name, stats in summary.get("balance", {}).get("partitions", {}).items():
            lines.append(f"{name}: {stats['building_count']} фасадов; {stats['image_count']} фото "
                         f"({stats['image_fraction']:.2%}); {stats['crop_count']} кропов "
                         f"({stats['crop_fraction']:.2%})")
        if summary.get("crop_count"):
            lines.append(f"Дополнительно в семантическом TRAIN без установленного адреса: "
                         f"{summary['auxiliary_train_image_count']} фото; {summary['auxiliary_train_crop_count']} кропов.")
            for name, count in summary["crop_splits"].items():
                lines.append(f"Весь семантический {name}: {summary['semantic_image_splits'].get(name, 0)} фото; "
                             f"{count} кропов ({count / summary['crop_count']:.2%}).")
        lines.extend([("Создана новая версия split; старые результаты сохранены отдельно." if migration else
                       "Внутри выбранной версии прежние назначения сохраняются."),
                      "Все подтверждённые виды и кропы одного здания остаются вместе.",
                      "Кропы записаны координатами; исходные фото и COCO не изменены.",
                      "Число кропов геометрическое: фильтрация по поддержке/разметке может изменить доли.",
                      f"Отчёт: {out / 'summary.json'}", f"Проверка имён: {out / 'metadata_review_with_names.csv'}"])
        if summary.get("auxiliary_train_image_count"):
            lines.append("Все добавленные снимки доступны для семантического обучения. "
                         "Независимость TRAIN от VAL/TEST по физическим зданиям пока не подтверждена "
                         "для снимков без адреса; результаты обучения с ними — exploratory.")
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
