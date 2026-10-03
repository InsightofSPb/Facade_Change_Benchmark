"""Prepare one reproducible dataset run: inventory, reviewed split, alignment, crops."""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facade_change.batch import run_batch
from facade_change.data import build_manifest, read_overrides
from facade_change.io import finish_record, new_directory, read_json, run_record, sha256, write_json
from facade_change.preparation import prepare_dataset


def load_config(path):
    """Resolve config-relative paths and reject unsupported or invalid options early."""
    config = read_json(path)
    defaults = {
        "split": {"train": .70, "val": .10, "test": .20, "seed": 42, "mode": "reviewed"},
        "alignment": {"methods": ["cascade"], "device": "cuda", "checkpoint": "auto",
                      "trust_checkpoint": True, "download_weights": False, "max_side": 1024,
                      "ransac_threshold": 3., "limit": 0},
        "crops": {"tile_size": 256, "stride": 128, "min_valid_fraction": .8,
                  "method": "cascade", "controls": 1},
    }
    allowed = {"coco_json", "image_roots", "manifest_path", "metadata_csv", "pair_policy", *defaults}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Config must be an object with only supported dataset options")
    for name, values in defaults.items():
        supplied = config.get(name, {})
        if not isinstance(supplied, dict) or set(supplied) - set(values):
            raise ValueError(f"Unsupported {name} options; expected {sorted(values)}")
        config[name] = {**values, **supplied}
    config.setdefault("pair_policy", "adjacent")
    if not isinstance(config["pair_policy"], str) or config["pair_policy"] not in {"adjacent", "first-anchor", "all"}:
        raise ValueError("pair_policy must be adjacent, first-anchor or all")
    split, alignment, crops = (config[name] for name in defaults)
    for key in ("train", "val", "test"):
        value = split[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"split.{key} must be a finite number")
    if split["train"] <= 0 or min(split["val"], split["test"]) < 0 or not math.isclose(
            sum(split[key] for key in ("train", "val", "test")), 1., abs_tol=1e-9, rel_tol=0):
        raise ValueError("Split fractions must sum to 1, with positive train and nonnegative val/test")
    if not isinstance(split["mode"], str) or split["mode"] not in {"reviewed", "dev"}:
        raise ValueError("split.mode must be reviewed or dev")
    integers = {"split.seed": split["seed"], "alignment.limit": alignment["limit"],
                "alignment.max_side": alignment["max_side"], "crops.tile_size": crops["tile_size"],
                "crops.stride": crops["stride"], "crops.controls": crops["controls"]}
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integers.values()):
        raise ValueError(f"Integer values required for {', '.join(integers)}")
    if alignment["limit"] < 0 or alignment["max_side"] < 8 or crops["controls"] < 0:
        raise ValueError("limit/controls must be nonnegative and max_side >= 8")
    if crops["tile_size"] < 8 or not 0 < crops["stride"] <= crops["tile_size"]:
        raise ValueError("Crops require tile_size >= 8 and 0 < stride <= tile_size")
    for name, value in {"ransac_threshold": alignment["ransac_threshold"],
                        "min_valid_fraction": crops["min_valid_fraction"]}.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")
    if crops["min_valid_fraction"] > 1:
        raise ValueError("min_valid_fraction must be <= 1")
    methods = alignment["methods"]
    if not isinstance(methods, list) or not methods or any(
            not isinstance(method, str) or method not in {"sift", "loftr", "cascade"}
            for method in methods) or len(set(methods)) != len(methods):
        raise ValueError("alignment.methods must be a unique nonempty list of sift, loftr, cascade")
    if crops["method"] not in methods:
        raise ValueError("crops.method must occur in alignment.methods")
    if not isinstance(alignment["device"], str) or (alignment["device"] != "cpu"
            and not alignment["device"].startswith("cuda")):
        raise ValueError("alignment.device must be cpu or a CUDA device")
    if any(not isinstance(alignment[key], bool) for key in ("trust_checkpoint", "download_weights")):
        raise ValueError("trust_checkpoint/download_weights must be booleans")

    def absolute(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Input paths must be nonempty strings")
        item = Path(value).expanduser()
        return str((item if item.is_absolute() else path.parent / item).resolve())

    config["coco_json"] = absolute(config.get("coco_json"))
    roots = config.get("image_roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError("image_roots must be a nonempty list")
    config["image_roots"] = sorted({absolute(root) for root in roots})
    for key in ("manifest_path", "metadata_csv"):
        config[key] = absolute(config[key]) if config.get(key) is not None else None
    if alignment["checkpoint"] != "auto":
        alignment["checkpoint"] = absolute(alignment["checkpoint"])
    return config


def run_dataset(config_path, out, prepare_only=False):
    """Run all requested stages in a new directory; preserve partial results on failure."""
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    annotation_hash = sha256(config["coco_json"])
    manifest_path = Path(config["manifest_path"]) if config["manifest_path"] else None
    manifest = read_json(manifest_path) if manifest_path else None
    if manifest is not None:
        source = manifest.get("source", {})
        if source.get("coco_json") != config["coco_json"] or source.get("coco_sha256") != annotation_hash:
            raise ValueError("Reused manifest COCO path/hash disagrees with coco_json; build a fresh inventory")
        if sorted({str(Path(root).expanduser().resolve()) for root in source.get("image_roots", [])}) != config["image_roots"]:
            raise ValueError("Reused manifest image_roots disagree with config; build a fresh inventory")
    metadata_hash = sha256(config["metadata_csv"]) if config["metadata_csv"] else None
    out = new_directory(out)
    record = run_record("dataset", {**config, "config_path": str(config_path),
                                   "input_config_sha256": sha256(config_path), "prepare_only": prepare_only,
                                   "runner_sha256": sha256(Path(__file__)),
                                   "coco_sha256": annotation_hash,
                                   "manifest_sha256": sha256(manifest_path) if manifest_path else None,
                                   "metadata_sha256": metadata_hash})
    write_json(out / "run.json", record)
    normalized_config = out / (record["started_utc"][:10] + "_config.json")
    write_json(normalized_config, config)
    summary = {"status": "running", "inventory_reused": manifest_path is not None,
               "crop_count": 0, "crop_splits": {}, "index_path": None, "alignment": None}
    try:
        if manifest is None:
            manifest = build_manifest(normalized_config, out / "inventory")
            manifest_path = out / "inventory/manifest.json"
        summary["input_issues"] = sum(row.get("image_status") != "ready" for row in manifest["images"])
        overrides = read_overrides(config["metadata_csv"], {str(row["image_id"]) for row in manifest["images"]})
        incomplete = []
        for original in manifest["images"]:
            row = {**original, **overrides.get(str(original["image_id"]), {})}
            year = row.get("year")
            if row.get("image_status") == "ready" and (row.get("metadata_status") != "reviewed"
                    or not row.get("building_id") or not row.get("view_id") or isinstance(year, bool)
                    or not isinstance(year, int) or not 1800 <= year <= 2100):
                incomplete.append(row["image_id"])
        split = config["split"]
        needs_review = split["mode"] == "reviewed" and bool(incomplete)
        prepared = prepare_dataset(manifest_path, out / "prepared", overrides=config["metadata_csv"],
                                   split_mode="dev" if needs_review else split["mode"],
                                   pair_policy=config["pair_policy"], seed=split["seed"],
                                   val_fraction=split["val"], test_fraction=split["test"])
        summary.update(preparation=prepared["summary"], metadata_review="prepared/metadata_review.csv",
                       group_review="prepared/group_review.csv")
        if needs_review:
            summary.update(status="needs_metadata_review", unreviewed_ready_image_ids=incomplete)
        else:
            if not prepare_only:
                alignment = dict(config["alignment"])
                crops = dict(config["crops"])
                summary["alignment"] = run_batch(out / "prepared/manifest.json", out / "batch",
                                                  seed=split["seed"], allow_inferred_metadata=split["mode"] == "dev",
                                                  crops=True, crop_method=crops.pop("method"), **alignment, **crops)
            crop_rows = []
            images = {str(row["image_id"]): row for row in prepared["images"]}
            derivatives = read_json(out / "batch/derivatives.json") if not prepare_only else {}
            for pair_id, derivative in derivatives.items():
                if not derivative.get("crops"):
                    continue
                crop_root = Path("batch") / derivative["crops"]
                document = read_json(out / crop_root / "crops.json")
                for crop in document["crops"]:
                    path = crop_root / crop["path"]
                    crop_rows.append({**crop, "pair_id": pair_id,
                                      "dataset_crop_id": pair_id + "/" + crop["crop_id"], "path": path.as_posix(),
                                      "reference_rgb": (path / "reference_rgb.png").as_posix(),
                                      "source_rgb": (path / "source_rgb.png").as_posix(),
                                      "geometric_overlap": (path / "geometric_overlap.png").as_posix(),
                                      "original_reference_path": images[str(crop["reference_id"])]["image_path"],
                                      "original_source_path": images[str(crop["source_id"])]["image_path"]})
            index = {"schema_version": 1, "source_annotations": {"path": config["coco_json"],
                     "sha256": annotation_hash, "categories": prepared["categories"]},
                     "semantic_masks": {"rasterized": False}, "temporal_ground_truth": "not_created",
                     "split": read_json(out / "prepared/split.json"), "pairs": prepared["pairs"], "crops": crop_rows}
            index_name = record["started_utc"][:10] + "_dataset_index.json"
            if sha256(config["coco_json"]) != annotation_hash:
                raise ValueError("Source COCO changed during dataset preparation")
            write_json(out / index_name, index)
            issues = summary["input_issues"] or (summary["alignment"] and any(
                summary["alignment"][key] for key in ("failed_runs", "rejected_runs", "derivative_failures")))
            summary.update(crop_count=len(crop_rows), crop_splits=dict(Counter(row["split"] for row in crop_rows)),
                           index_path=index_name, status="completed_with_issues" if issues else
                           "completed" if prepare_only else "completed_needs_review")
        write_json(out / "summary.json", summary)
        (out / "summary.txt").write_text(
            f"Status: {summary['status']}\nInput issues: {summary['input_issues']}\n"
            f"Crops: {summary['crop_count']}; splits: {summary['crop_splits']}\n"
            f"Metadata review: {summary['metadata_review']}\nGroup review: {summary['group_review']}\n"
            f"Dataset index: {summary['index_path']}\n"
            "COCO semantic masks were not rasterized; temporal ground truth was not created.\n",
            encoding="utf-8")
        record["summary"] = summary
        finish_record(out, record, summary["status"])
        return summary
    except (Exception, KeyboardInterrupt) as exc:
        finish_record(out, record, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    result = run_dataset(args.config, args.out, args.prepare_only)
    print(f"{result['status']}: {args.out / 'summary.txt'}")
    sys.exit(2 if result["status"] in {"needs_metadata_review", "completed_with_issues"} else 0)
