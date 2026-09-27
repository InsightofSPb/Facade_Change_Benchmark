"""COCO image inventory; unresolved rows stay visible and are never discarded."""
from __future__ import annotations

import csv
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from .io import load_rgb, new_directory, read_json, run_record, sha256, finish_record, write_json

PREFIX = re.compile(r"^[0-9a-fA-F]{8}-")
YEAR = re.compile(r"^(?P<view>.+)_(?P<year>(?:19|20)\d{2})$")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
REVIEW_FIELDS = ["image_id", "view_id", "building_id", "year", "reviewed", "notes"]


def normalized_name(name: str) -> str:
    return PREFIX.sub("", name, count=1)


def filename_metadata(name: str) -> dict:
    match = YEAR.fullmatch(Path(normalized_name(name)).stem)
    return {"view_id": match["view"] if match else None,
            "building_id": None, "year": int(match["year"]) if match else None,
            "metadata_status": "inferred" if match else "unknown",
            "metadata_source": "filename_suffix" if match else None}


class ImageResolver:
    """Index only explicitly supplied roots. Ambiguous matches are errors."""

    def __init__(self, roots):
        self.roots = sorted({Path(root).expanduser().resolve() for root in roots})
        if not self.roots or any(not root.is_dir() for root in self.roots):
            raise ValueError("Every image_root must be an existing directory")
        self.exact = defaultdict(set)
        self.normalized = defaultdict(set)
        self.folded = defaultdict(set)
        for root in self.roots:
            for folder, dirs, files in os.walk(root, followlinks=False):
                dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                for name in sorted(files):
                    path = (Path(folder) / name).resolve()
                    if path.suffix.lower() not in EXTENSIONS or not self.contains(path):
                        continue
                    self.exact[name].add(path)
                    self.normalized[normalized_name(name)].add(path)
                    self.folded[normalized_name(name).casefold()].add(path)

    def contains(self, path: Path) -> bool:
        return any(path == root or root in path.parents for root in self.roots)

    def resolve(self, row: dict) -> tuple[Path | None, str, list[str]]:
        direct = set()
        names = {Path(str(row.get(key) or "")).name for key in ("file_name", "path")}
        names.discard("")
        for key in ("file_name", "path"):
            raw = row.get(key)
            if not raw:
                continue
            path = Path(raw).expanduser()
            candidates = [path] if path.is_absolute() else [root / path for root in self.roots]
            for item in candidates:
                resolved = item.resolve()
                if self.contains(resolved) and resolved.is_file():
                    direct.add(resolved)
        stages = [("explicit_path", direct),
                  ("exact_basename", set().union(*(self.exact[n] for n in names))),
                  ("stripped_prefix", set().union(*(self.normalized[normalized_name(n)] for n in names))),
                  ("casefold", set().union(*(self.folded[normalized_name(n).casefold()] for n in names)))]
        for rule, candidates in stages:
            if len(candidates) > 1:
                return None, "ambiguous_" + rule, sorted(str(p) for p in candidates)
            if len(candidates) == 1:
                return next(iter(candidates)), rule, []
        return None, "not_found", []


def read_overrides(path: str | Path | None, ids: set[str]) -> dict:
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    seen = set()
    for row in rows:
        key = str(row.get("image_id", ""))
        if key not in ids or key in seen:
            raise ValueError(f"Unknown/duplicate override image_id: {key}")
        seen.add(key)
        reviewed = row.get("reviewed", "").strip().lower()
        if reviewed not in {"true", "false", ""}:
            raise ValueError(f"reviewed must be true/false for {key}")
        if reviewed != "true":
            continue
        if not all(row.get(field, "").strip() for field in ("view_id", "building_id", "year")):
            raise ValueError(f"Reviewed row {key} needs view_id, building_id and year")
        year = int(row["year"])
        if not 1800 <= year <= 2100:
            raise ValueError(f"Invalid year for {key}: {year}")
        result[key] = {"view_id": row["view_id"].strip(), "building_id": row["building_id"].strip(),
                       "year": year, "metadata_status": "reviewed", "metadata_source": "override_csv",
                       "metadata_notes": row.get("notes", "")}
    return result


def build_manifest(config_path: str | Path, out: str | Path, overrides=None) -> dict:
    config_path = Path(config_path).expanduser().resolve()
    config = read_json(config_path)
    # Paths in the config are relative to the config's directory, never the shell cwd.
    def absolute(value):
        p = Path(value).expanduser()
        return str(p.resolve() if p.is_absolute() else (config_path.parent / p).resolve())
    coco_path = Path(absolute(config["coco_json"]))
    roots = [absolute(root) for root in config["image_roots"]]
    coco = read_json(coco_path)
    images = coco["images"]
    ids = {str(row["id"]) for row in images}
    if len(ids) != len(images):
        raise ValueError("Duplicate COCO image ids")
    metadata = read_overrides(overrides, ids)
    resolver = ImageResolver(roots)
    config = {"coco_json": str(coco_path), "image_roots": roots,
              "coco_sha256": sha256(coco_path),
              "overrides_sha256": sha256(overrides) if overrides else None}
    out = new_directory(out)
    record = run_record("manifest", config)
    write_json(out / "run.json", record)
    try:
        rows = []
        for image in images:
            row = {"image_id": image["id"], "file_name": image["file_name"],
                   "coco_path": image.get("path"), "coco_width": image["width"],
                   "coco_height": image["height"], "split": "unassigned"}
            row.update(filename_metadata(image["file_name"]))
            row.update(metadata.get(str(image["id"]), {}))
            path, rule, candidates = resolver.resolve(image)
            row.update(image_path=str(path) if path else None, resolution_rule=rule,
                       image_status="missing" if rule == "not_found" else "ambiguous")
            if candidates:
                row["path_candidates"] = candidates
            if path:
                try:
                    before = sha256(path)
                    rgb, support = load_rgb(path)
                    if sha256(path) != before:
                        raise ValueError("Image changed during inspection")
                    h, w = rgb.shape[:2]
                    row.update(width=w, height=h, sha256=before, opaque_fraction=float(support.mean()))
                    row["image_status"] = "ready" if (w, h) == (image["width"], image["height"]) else "dimension_mismatch"
                except (OSError, ValueError) as exc:
                    row.update(image_status="decode_error", error=str(exc))
            rows.append(row)
        paths = Counter(row["image_path"] for row in rows if row["image_path"])
        for row in rows:
            if row["image_path"] and paths[row["image_path"]] > 1:
                row["image_status"] = "duplicate_resolved_path"
        anns = coco.get("annotations", [])
        cats = coco.get("categories", [])
        category_ids = {cat["id"] for cat in cats}
        summary = {"image_count": len(rows), "annotation_count": len(anns),
                   "category_count": len(cats), "image_status": dict(Counter(r["image_status"] for r in rows)),
                   "metadata_status": dict(Counter(r["metadata_status"] for r in rows)),
                   "annotation_nonpositive_area_ids": [a["id"] for a in anns if a.get("area", 1) <= 0],
                   "orphan_annotation_ids": [a["id"] for a in anns if str(a["image_id"]) not in ids],
                   "unknown_category_annotation_ids": [a["id"] for a in anns if a["category_id"] not in category_ids],
                   "annotation_check_scope": "Metadata only; polygons not rasterized, nothing removed"}
        result = {"schema_version": 1, "source": config, "categories": cats, "summary": summary, "images": rows}
        write_json(out / "manifest.json", result)
        with (out / "metadata_review.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            writer.writerows({**{k: r.get(k) for k in REVIEW_FIELDS},
                              "reviewed": "true" if r["metadata_status"] == "reviewed" else "false",
                              "notes": r.get("metadata_notes", "")} for r in rows)
        (out / "summary.txt").write_text(
            f"COCO images: {len(rows)}\nAnnotations: {len(anns)}\n"
            f"Image status: {summary['image_status']}\nMetadata status: {summary['metadata_status']}\n"
            "Filename candidates require review; no split or temporal ground truth has been created.\n",
            encoding="utf-8")
        status = "completed" if all(r["image_status"] == "ready" for r in rows) else "completed_with_issues"
        finish_record(out, record, status)
        return result
    except Exception as exc:
        finish_record(out, record, "failed", str(exc))
        raise
