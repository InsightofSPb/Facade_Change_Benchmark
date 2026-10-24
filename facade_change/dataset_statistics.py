"""Count COCO facade histories without loading photographs or changing splits.

Created 2026-10-19. Pass a COCO path or use data/result_coco_402.json in the checkout.
Facades are grouped by the project's building_id, pooling all their views.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

from facade_change.data import (
    DEFAULT_METADATA_RULES, filename_metadata, normalized_name,
    read_filename_rules, read_overrides, read_rename_maps,
)
from facade_change.io import new_directory, read_json, run_record, sha256, write_json

VIEW_SUFFIX = re.compile(
    r"(?:(?:side|balcony|ornament|ornamentbalcony|vase|closeup|left|right|top|"
    r"bottom|roof|wall|facade|front|back|rear|detail|repairs|church|chuch)"
    r"(?:v?\d+)?|v\d+)", re.IGNORECASE,
)


class MetadataNames:
    """Resolve supplied rename CSVs without scanning or decoding image files."""

    def __init__(self, paths):
        _, self.canonical, _ = read_rename_maps(paths)
        self.by_basename = defaultdict(set)
        for name, target in self.canonical.items():
            self.by_basename[normalized_name(Path(name).name)].add(target)

    def canonical_name(self, name):
        name = name.replace("\\", "/")
        if name in self.canonical:
            return Path(self.canonical[name]).name
        targets = self.by_basename[normalized_name(Path(name).name)]
        if len(targets) > 1:
            raise ValueError(f"Ambiguous rename mapping for {name}; use a full relative name")
        return Path(next(iter(targets))).name if targets else Path(name).name


def known_group(name, aliases):
    """Pool owner-approved view suffixes only under an already known identity.

    Do not strip numeric address components or invent identities for camera names.
    This descriptive fallback never confirms metadata for training or alignment.
    """
    stem = Path(normalized_name(name)).stem.casefold()
    for alias in sorted(aliases, key=lambda value: (-len(value), value)):
        if stem != alias and not stem.startswith(alias + "_"):
            continue
        tokens = stem[len(alias):].strip("_").split("_") if stem != alias else []
        years = [int(token) for token in tokens if re.fullmatch(r"(?:19|20)\d{2}", token)]
        if len(years) > 1 or any(
            not re.fullmatch(r"(?:19|20)\d{2}", token) and not VIEW_SUFFIX.fullmatch(token)
            for token in tokens
        ):
            continue
        return {"building_id": aliases[alias], "year": years[0] if years else None,
                "metadata_status": "inferred", "metadata_source": "known_group_view_suffix"}
    return {}


def image_metadata(coco, rules_path, metadata_csv=None, rename_maps=None):
    images = coco.get("images") if isinstance(coco, dict) else None
    if not isinstance(images, list):
        raise ValueError("COCO must contain an images list")
    ids = set()
    for row in images:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), (int, str))
                or isinstance(row.get("id"), bool) or not str(row["id"]).strip()
                or not isinstance(row.get("file_name"), str) or not row["file_name"].strip()):
            raise ValueError("Each COCO image needs a nonempty id and file_name")
        key = str(row["id"])
        if key in ids:
            raise ValueError(f"Duplicate COCO image id: {key}")
        ids.add(key)
    names = MetadataNames(rename_maps)
    rows = [{"image_id": row["id"], "file_name": row["file_name"],
             "resolved_file_name": names.canonical_name(row["file_name"])} for row in images]
    rules = read_filename_rules(rules_path, rows, resolver=names)
    overrides = read_overrides(metadata_csv, ids)
    aliases = {}
    for rule in read_json(rules_path):
        for alias in (rule["view_id"], rule["building_id"]):
            alias = alias.strip().casefold()
            building = rule["building_id"].strip()
            if alias in aliases and aliases[alias] != building:
                raise ValueError(f"Ambiguous building alias in metadata rules: {alias}")
            aliases[alias] = building
    result = []
    for row in rows:
        key = str(row["image_id"])
        name = row["resolved_file_name"]
        metadata = filename_metadata(name)
        if key in overrides:
            metadata.update(overrides[key])
        elif key in rules:
            metadata.update(rules[key])
        else:
            metadata.update(known_group(name, aliases))
        result.append({**row, **metadata})
    return result


def summarize(images):
    grouped = defaultdict(list)
    for image in images:
        if image.get("building_id"):
            grouped[image["building_id"]].append(image)
    facades = []
    for building, members in sorted(grouped.items()):
        year_counts = Counter(row["year"] for row in members if row["year"] is not None)
        years = sorted(year_counts)
        facades.append({
            "building_id": building, "image_count": len(members),
            "observation_years": len(years), "years": years,
            "first_year": min(years) if years else None,
            "last_year": max(years) if years else None,
            "span_years": max(years) - min(years) if years else None,
            "has_history": len(years) >= 2,
            "undated_images": len(members) - sum(year_counts.values()),
            "inferred_group_images": sum(row["metadata_status"] != "reviewed" for row in members),
            "images_per_year": dict(sorted(year_counts.items())),
        })
    histories = [row for row in facades if row["has_history"]]
    distributions = {}
    for field, selection in (("observation_years", facades), ("span_years", histories)):
        bins = {}
        for facade in selection:
            key = facade[field]
            bucket = bins.setdefault(key, {field: key, "facades": 0, "images": 0, "undated_images": 0})
            bucket["facades"] += 1
            bucket["images"] += facade["image_count"]
            bucket["undated_images"] += facade["undated_images"]
        distributions[field] = [bins[key] for key in sorted(bins)]
    spans = [row["span_years"] for row in histories]
    history_images = sum(row["image_count"] for row in histories)
    unidentified = sum(not row.get("building_id") for row in images)
    duplicate_names = Counter(row["file_name"] for row in images)
    summary = {
        "images": len(images), "identified_facades": len(facades),
        "histories": len(histories), "history_images": history_images,
        "facades_without_history": len(facades) - len(histories),
        "single_image_facades": sum(row["image_count"] == 1 for row in facades),
        "other_images": len(images) - history_images,
        "unidentified_images": unidentified,
        "images_without_year": sum(row["year"] is None for row in images),
        "inferred_group_images": sum(row["inferred_group_images"] for row in facades),
        "history_facades_with_undated_images": sum(row["undated_images"] > 0 for row in histories),
        "mean_history_span_years": mean(spans) if spans else None,
        "median_history_span_years": median(spans) if spans else None,
        "min_history_span_years": min(spans) if spans else None,
        "max_history_span_years": max(spans) if spans else None,
        "mean_history_observation_years": mean(row["observation_years"] for row in histories) if histories else None,
        "mean_history_images": mean(row["image_count"] for row in histories) if histories else None,
        "duplicate_file_names": {name: count for name, count in sorted(duplicate_names.items()) if count > 1},
    }
    assert sum(row["image_count"] for row in facades) + unidentified == len(images)
    return {"summary": summary, "distributions": distributions, "facades": facades}


def format_report(report):
    s = report["summary"]
    lines = [
        "COCO DATASET STATISTICS", "",
        f"Images (all COCO records): {s['images']}",
        f"Identified facade/building groups: {s['identified_facades']}",
        f"Histories (at least 2 distinct years): {s['histories']} ({s['history_images']} images)",
        f"Facades without an established multi-year history: {s['facades_without_history']}",
        f"  Of these, exactly one image: {s['single_image_facades']}",
        f"Other images (single-year/undated facades and unidentified elements): {s['other_images']}",
        f"  Unidentified images: {s['unidentified_images']}",
        f"Images without a known year: {s['images_without_year']}",
        f"Images grouped by known identity + view suffix: {s['inferred_group_images']}", "",
    ]
    for label, key in (
        ("Mean history span (last year - first year)", "mean_history_span_years"),
        ("Median history span", "median_history_span_years"),
        ("Minimum history span", "min_history_span_years"),
        ("Maximum history span", "max_history_span_years"),
        ("Mean distinct years per history", "mean_history_observation_years"),
        ("Mean images per history", "mean_history_images"),
    ):
        value = s[key]
        lines.append(f"{label}: {value:.2f}" if value is not None else f"{label}: N/A")
    for title, field in (("DISTINCT OBSERVATION YEARS (all identified facades)", "observation_years"),
                         ("ELAPSED SPAN IN YEARS (histories only)", "span_years")):
        lines += ["", title, "Years | Facades | Images | Undated images"]
        for row in report["distributions"][field]:
            lines.append(f"{row[field]:>5} | {row['facades']:>7} | {row['images']:>6} | {row['undated_images']:>14}")
    lines += ["", "Years are pooled across all views of one building_id; every image is counted.",
              "A three-year history means 3 observed years, not necessarily a 3-year span.",
              "The 0-year row means no known dates; it does not mean an empty facade.",
              "Means give each facade equal weight. Undated members count as images only.",
              "Unidentified images are not asserted to be distinct physical facades.",
              "Counts describe metadata, not usable aligned pairs or verified state changes."]
    if s["history_facades_with_undated_images"]:
        lines.append("Some histories contain undated images: their observed spans may be incomplete.")
    if s["duplicate_file_names"]:
        lines.append("WARNING: repeated file_name values exist; records were retained. See the JSON report.")
    return "\n".join(lines) + "\n"


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(coco_path, out=None, metadata_rules=DEFAULT_METADATA_RULES,
        metadata_csv=None, rename_maps=None):
    coco_path = Path(coco_path).expanduser().resolve()
    metadata_rules = Path(metadata_rules).expanduser().resolve()
    metadata_csv = Path(metadata_csv).expanduser().resolve() if metadata_csv else None
    rename_maps = [Path(path).expanduser().resolve() for path in (rename_maps or [])]
    sources = [coco_path, metadata_rules, *rename_maps, *([metadata_csv] if metadata_csv else [])]
    hashes = {str(path): sha256(path) for path in sources}
    images = image_metadata(read_json(coco_path), metadata_rules, metadata_csv, rename_maps)
    report = summarize(images)
    stamp = datetime.now(timezone.utc)
    prefix = stamp.date().isoformat()
    out = new_directory(out or coco_path.parent / "dataset_statistics" / stamp.strftime("%Y-%m-%dT%H%M%S%fZ"))
    config = {"coco_path": str(coco_path), "metadata_rules": str(metadata_rules),
              "metadata_csv": str(metadata_csv) if metadata_csv else None,
              "rename_maps": [str(path) for path in rename_maps], "input_sha256": hashes,
              "script_sha256": sha256(Path(__file__)), "group_unit": "building_id",
              "history_rule": "at_least_two_distinct_known_years_across_all_views",
              "span_rule": "max(years)-min(years)", "out": str(out)}
    record = run_record("dataset_statistics", config)
    record["run_id"] = out.name
    run_path = out / f"{prefix}_run.json"
    write_json(run_path, record)
    try:
        if any(sha256(path) != hashes[str(path)] for path in sources):
            raise ValueError("Input changed during statistics calculation; rerun with stable inputs")
        report["source"] = config
        write_json(out / f"{prefix}_dataset_statistics.json", report)
        text = format_report(report)
        (out / f"{prefix}_dataset_statistics.txt").write_text(text, encoding="utf-8")
        facade_rows = [{**row, "years": ";".join(map(str, row["years"])),
                        "images_per_year": ";".join(f"{year}:{count}" for year, count in row["images_per_year"].items())}
                       for row in report["facades"]]
        write_csv(out / f"{prefix}_facades.csv", facade_rows,
                  ["building_id", "image_count", "observation_years", "years", "first_year", "last_year",
                   "span_years", "has_history", "undated_images", "inferred_group_images", "images_per_year"])
        write_csv(out / f"{prefix}_images.csv", images,
                  ["image_id", "file_name", "resolved_file_name", "building_id", "view_id", "year",
                   "metadata_status", "metadata_source"])
        record.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat(),
                      artifact_sha256={path.name: sha256(path) for path in sorted(out.iterdir()) if path != run_path})
        write_json(run_path, record)
    except Exception as exc:
        record.update(status="failed", error=str(exc), finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(run_path, record)
        raise
    print(text, end="")
    print(f"\nReports: {out}")
    return report, out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coco_path", type=Path, nargs="?",
                        default=Path(__file__).resolve().parents[1] / "data" / "result_coco_402.json")
    parser.add_argument("--out", type=Path, help="New output directory; existing directories are never overwritten")
    parser.add_argument("--metadata-rules", type=Path, default=DEFAULT_METADATA_RULES)
    parser.add_argument("--metadata-csv", type=Path, help="Reviewed metadata using current COCO image IDs")
    parser.add_argument("--rename-map", type=Path, action="append", dest="rename_maps",
                        help="Optional image_rename_map.csv; repeat for several maps")
    args = parser.parse_args(argv)
    try:
        run(**vars(args))
    except (OSError, ValueError) as exc:
        print(f"Statistics failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
