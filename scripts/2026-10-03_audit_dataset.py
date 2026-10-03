"""Audit a new COCO export, rechecking bytes and carrying reviewed metadata by identity."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facade_change.data import REVIEW_FIELDS, build_manifest, normalized_name, preprocess_coco, read_filename_rules
from facade_change.io import finish_record, new_directory, read_json, run_record, sha256, write_json

spec = importlib.util.spec_from_file_location(
    "dataset_runner", Path(__file__).with_name("2026-10-03_prepare_dataset.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def confirmed_rows(images, fixes_path=None):
    """Resolve human-confirmed fixes by unique filename, never by an old COCO ID."""
    by_name = defaultdict(list)
    for row in images:
        by_name[normalized_name(Path(row["file_name"]).name)].append(row)
    fixes = read_json(fixes_path) if fixes_path else []
    if not isinstance(fixes, list):
        raise ValueError("Metadata fixes must be a JSON list")
    updates = {}
    for fix in fixes:
        if not isinstance(fix, dict) or set(fix) - {
                "file_name", "view_id", "building_id", "year", "notes"}:
            raise ValueError("Unsupported metadata fix fields")
        if any(not isinstance(fix.get(key), str) or not fix[key].strip()
               for key in ("file_name", "view_id", "building_id")):
            raise ValueError("Each fix needs file_name, view_id and building_id")
        year = fix.get("year")
        if isinstance(year, bool) or not isinstance(year, int) or not 1800 <= year <= 2100:
            raise ValueError("Each fix needs a valid integer year")
        name = normalized_name(Path(fix["file_name"]).name)
        matches = by_name.get(name, [])
        if len(matches) != 1 or name in updates:
            raise ValueError(f"Missing, ambiguous or duplicate metadata fix: {name}")
        updates[name] = fix
    rows = []
    for image in images:
        row = {key: image.get(key) for key in REVIEW_FIELDS}
        row.update(reviewed="true" if image["metadata_status"] == "reviewed" else "false",
                   notes=image.get("metadata_notes", ""))
        name = normalized_name(Path(image["file_name"]).name)
        if name in updates:
            fix = updates[name]
            row.update({key: fix[key] for key in ("view_id", "building_id", "year")})
            row.update(reviewed="true", notes=fix.get("notes", "Human-confirmed metadata"))
        rows.append(row)
    return rows, sorted(updates)


def run_audit(config_path, previous_manifest, out, metadata_fixes=None):
    config_path = Path(config_path).expanduser().resolve()
    config = runner.load_config(config_path)
    if config["manifest_path"] is not None:
        raise ValueError("Audit builds a current inventory; set manifest_path=null and use --previous-manifest")
    out = new_directory(out)
    record = run_record("dataset_audit", {
        "config_path": str(config_path), "input_config_sha256": sha256(config_path),
        "previous_manifest": str(Path(previous_manifest).resolve()) if previous_manifest else None,
        "previous_manifest_sha256": sha256(previous_manifest) if previous_manifest else None,
        "metadata_fixes": str(Path(metadata_fixes).resolve()) if metadata_fixes else None,
        "metadata_fixes_sha256": sha256(metadata_fixes) if metadata_fixes else None,
        "image_selection": "current_coco_images_only", "config": config,
        "runner_sha256": sha256(Path(__file__)),
        "coco_sha256": sha256(config["coco_json"]),
    })
    write_json(out / "run.json", record)
    try:
        preprocessing = preprocess_coco(config["coco_json"], out / "preprocessed")
        inventory_config = {**config}
        if preprocessing["summary"]["removed_annotation_count"]:
            inventory_config["coco_json"] = preprocessing["coco_path"]
        write_json(out / "inventory_config.json", inventory_config)
        inventory = build_manifest(out / "inventory_config.json", out / "inventory", overrides=config["metadata_csv"],
                                   previous_manifest=previous_manifest)
        rows, fixed_names = confirmed_rows(inventory["images"], metadata_fixes)
        rules = read_filename_rules(config["metadata_rules"], inventory["images"])
        fixed_names = sorted(set(fixed_names) | {normalized_name(Path(r["file_name"]).name)
                             for r in inventory["images"] if str(r["image_id"]) in rules})
        metadata_path = out / "metadata_confirmed.csv"
        with metadata_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        effective = {**inventory_config, "manifest_path": str(out / "inventory/manifest.json"),
                     "metadata_csv": str(metadata_path)}
        # Audit is exploratory: leave freezing or extending gold to the dataset command.
        effective["split"] = {**config["split"], "mode": "dev"}
        effective["previous_split"] = None
        write_json(out / "dataset_config.json", effective)
        result = runner.run_dataset(out / "dataset_config.json", out / "dataset", prepare_only=True)
        prepared = read_json(out / "dataset/prepared/manifest.json")
        coco = read_json(inventory["source"]["coco_json"])
        if (sha256(config["coco_json"]) != record["config"]["coco_sha256"] or
                sha256(inventory["source"]["coco_json"]) != inventory["source"]["coco_sha256"]):
            raise ValueError("Source COCO changed during audit")
        annotation_counts = Counter(str(a["image_id"]) for a in coco.get("annotations", []))
        pair_counts = Counter(str(p[key]) for p in prepared["pairs"]
                              for key in ("reference_id", "source_id"))
        years = defaultdict(set)
        for row in prepared["images"]:
            if not row["preparation_exclusion_reasons"]:
                years[row["view_id"]].add(row["year"])
        review_rows = []
        for row in prepared["images"]:
            key = str(row["image_id"])
            reasons = row["preparation_exclusion_reasons"]
            pairing = ("metadata_or_image_unresolved" if reasons else "pair_candidate" if pair_counts[key]
                       else "single_year_view" if len(years[row["view_id"]]) == 1
                       else "eligible_without_pair")
            review_rows.append({**{field: row.get(field) for field in REVIEW_FIELDS},
                               "reviewed": "true" if row["metadata_status"] == "reviewed" else "false",
                               "notes": row.get("metadata_notes", ""),
                               "file_name": normalized_name(Path(row["file_name"]).name),
                               "image_path": row.get("image_path"), "image_status": row["image_status"],
                               "annotation_count": annotation_counts[key], "pairing_status": pairing,
                               "pair_count": pair_counts[key], "exclusion_reasons": ";".join(reasons)})
        fields = REVIEW_FIELDS + ["file_name", "image_path", "image_status", "annotation_count",
                                  "pairing_status", "pair_count", "exclusion_reasons"]
        with (out / "metadata_review_with_names.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(review_rows)
        names_by_id = {str(r["image_id"]): r["file_name"] for r in review_rows}
        confirmed_pairs = [{**p, "reference_file": names_by_id[str(p["reference_id"])],
                            "source_file": names_by_id[str(p["source_id"])]}
                           for p in prepared["pairs"]
                           if names_by_id[str(p["reference_id"])] in fixed_names
                           or names_by_id[str(p["source_id"])] in fixed_names]
        pair_fields = ["reference_id", "source_id", "reference_file", "source_file",
                       "reference_year", "source_year", "view_id"]
        with (out / "confirmed_pairs.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=pair_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(confirmed_pairs)
        annotation_issues = {key: inventory["summary"][key] for key in (
            "orphan_annotation_ids", "unknown_category_annotation_ids", "annotation_nonpositive_area_ids")}
        audit = {"schema_version": 1, "inventory": inventory["summary"],
                 "preparation": prepared["summary"], "metadata_fixes": fixed_names,
                 "confirmed_pairs": confirmed_pairs, "annotation_issues": annotation_issues,
                 "annotation_preprocessing": preprocessing,
                 "pairing_status": dict(Counter(r["pairing_status"] for r in review_rows)),
                 "unreviewed_ready_image_count": sum(r["image_status"] == "ready" and
                                                      r["reviewed"] != "true" for r in review_rows),
                 "images_without_annotations": [r["image_id"] for r in review_rows if not r["annotation_count"]],
                 "scope": "COCO metadata and image bytes; semantic masks not rasterized; "
                          "temporal pairs are candidates, geometry not checked; exploratory dev only"}
        write_json(out / "audit.json", audit)
        (out / "summary.txt").write_text(
            f"COCO: {config['coco_json']}\nImages: {inventory['summary']['image_count']}; "
            f"annotations: {inventory['summary']['annotation_count']}\n"
            f"Annotations removed (stored area <= 0): {preprocessing['summary']['removed_annotation_count']}\n"
            f"Annotation removal log: {preprocessing['report_path']}\n"
            f"Image status: {inventory['summary']['image_status']}\n"
            f"Reused with SHA-256 check: {inventory['summary']['reused_image_count']}; "
            f"decoded: {inventory['summary']['decoded_image_count']}\n"
            f"Confirmed filename fixes: {len(fixed_names)}\n"
            f"Unreviewed ready images: {audit['unreviewed_ready_image_count']}\n"
            f"Images without annotations: {len(audit['images_without_annotations'])}\n"
            f"Annotation metadata issues: { {key: len(value) for key, value in annotation_issues.items()} }\n"
            f"Pairing status: {audit['pairing_status']}\n"
            + (out / "dataset/prepared/summary.txt").read_text(encoding="utf-8")
            + "Review table with filenames: metadata_review_with_names.csv\n",
            encoding="utf-8")
        if sha256(config_path) != record["config"]["input_config_sha256"] or (metadata_fixes and
                sha256(metadata_fixes) != record["config"]["metadata_fixes_sha256"]):
            raise ValueError("Audit config or metadata fixes changed during audit")
        status = "completed_with_issues" if result["input_issues"] or any(annotation_issues.values()) else "completed"
        record["summary"] = audit
        finish_record(out, record, status)
        return audit, status
    except (Exception, KeyboardInterrupt) as exc:
        finish_record(out, record, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", str(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--previous-manifest", type=Path)
    parser.add_argument("--metadata-fixes", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    _, status = run_audit(args.config, args.previous_manifest, args.out, args.metadata_fixes)
    print(f"{status}: {args.out / 'summary.txt'}")
    sys.exit(2 if status == "completed_with_issues" else 0)
