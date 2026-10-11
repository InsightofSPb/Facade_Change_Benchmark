"""COCO image inventory; unresolved rows stay visible and are never discarded."""
from __future__ import annotations

import csv
import os
import re
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath

from .io import load_rgb, new_directory, read_json, run_record, sha256, finish_record, write_json

PREFIX = re.compile(r"^[0-9a-fA-F]{8}-")
YEAR = re.compile(r"^(?P<view>.+)_(?P<year>(?:19|20)\d{2})$")
VARIANT_YEAR = re.compile(r"^.+_(?P<year>(?:19|20)\d{2})_v\d+$", re.IGNORECASE)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
REVIEW_FIELDS = ["image_id", "view_id", "building_id", "year", "reviewed", "notes"]
DEFAULT_METADATA_RULES = Path(__file__).resolve().parents[1] / "configs/2026-10-03_metadata_rules.json"


def preprocess_coco(coco_path, out) -> dict:
    """Copy COCO unchanged except annotations with explicitly stored area <= 0."""
    coco_path = Path(coco_path).expanduser().resolve()
    source_hash = sha256(coco_path)
    rule = "remove_only_explicit_area_le_zero"
    out = new_directory(out)
    record = run_record("coco_preprocessing", {"coco_path": str(coco_path),
                        "coco_sha256": source_hash, "rule": rule})
    write_json(out / "run.json", record)
    try:
        coco = read_json(coco_path)
        annotations = coco.get("annotations", [])
        retained, removed = [], []
        for annotation in annotations:
            (removed if "area" in annotation and annotation["area"] <= 0 else retained).append(annotation)
        images = coco.get("images", [])
        categories = coco.get("categories", [])
        image_lookup = {str(row["id"]): row for row in images}
        category_lookup = {str(row["id"]): row for row in categories}
        annotated_ids = {str(row["image_id"]) for row in retained}
        summary = {"image_count": len(images), "category_count": len(categories),
                   "original_annotation_count": len(annotations), "retained_annotation_count": len(retained),
                   "removed_annotation_count": len(removed),
                   "images_without_annotations_count": sum(str(row["id"]) not in annotated_ids for row in images)}
        cleaned_path = out / "annotations.json"
        write_json(cleaned_path, {**coco, "annotations": retained})
        report_path = out / "removed_annotations.json"
        report = {"rule": rule, "summary": summary,
                  "source": {"coco_path": str(coco_path), "sha256": source_hash},
                  "output": {"coco_path": str(cleaned_path), "sha256": sha256(cleaned_path)},
                  "removed_annotations": [{
                      "annotation_id": row["id"], "image_id": row["image_id"],
                      "file_name": image_lookup.get(str(row["image_id"]), {}).get("file_name"),
                      "category_id": row["category_id"],
                      "category_name": category_lookup.get(str(row["category_id"]), {}).get("name"),
                      "area": row["area"], "bbox": row.get("bbox"),
                  } for row in removed]}
        write_json(report_path, report)
        if sha256(coco_path) != source_hash:
            raise ValueError("Source COCO changed during preprocessing")
        finish_record(out, record, "completed")
        return {"coco_path": str(cleaned_path), "report_path": str(report_path), "summary": summary}
    except Exception as exc:
        finish_record(out, record, "failed", str(exc))
        raise


def normalized_name(name: str) -> str:
    return PREFIX.sub("", name, count=1)


def filename_metadata(name: str) -> dict:
    stem = Path(normalized_name(name)).stem
    match = YEAR.fullmatch(stem)
    variant = VARIANT_YEAR.fullmatch(stem) if match is None else None
    # A variant suffix may denote another view or another photo of the same view.
    # Recover its year, but never merge v1/v2 or apply confirmed view rules to it.
    year_match = match or variant
    return {"view_id": match["view"] if match else None,
            "building_id": None, "year": int(year_match["year"]) if year_match else None,
            "metadata_status": "inferred" if year_match else "unknown",
            "metadata_source": "filename_suffix" if year_match else None}


def _rename_name(value, path, line, field):
    """CSV names are relative image names, never destinations outside roots."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Rename map {path}:{line} needs a nonempty {field}")
    name = value.strip().replace("\\", "/")
    if ("\x00" in name or Path(name).is_absolute() or PureWindowsPath(name).drive
            or any(part in {"", ".", ".."} for part in name.split("/"))
            or Path(name).suffix.lower() not in EXTENSIONS):
        raise ValueError(f"Rename map {path}:{line} has an unsafe image name: {value!r}")
    return name


def read_rename_maps(paths):
    """Validate explicit CSVs; return bidirectional chains, final names, hashes."""
    if paths is None:
        paths = []
    if not isinstance(paths, (list, tuple)):
        raise ValueError("rename_maps must be a list of CSV paths")
    forward, reverse, sources = {}, {}, []
    for supplied in paths:
        path = Path(supplied).expanduser().resolve()
        source_hash = sha256(path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if (not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames)
                    or not {"old_name", "new_name"}.issubset(reader.fieldnames)):
                raise ValueError(f"Rename map {path} needs old_name,new_name columns")
            for line, row in enumerate(reader, 2):
                old = _rename_name(row.get("old_name"), path, line, "old_name")
                new = _rename_name(row.get("new_name"), path, line, "new_name")
                if old in forward and forward[old] != new:
                    raise ValueError(f"Conflicting rename map row for {old}")
                if new in reverse and reverse[new] != old:
                    raise ValueError(f"Conflicting rename map destination: {new}")
                forward[old], reverse[new] = new, old
        if sha256(path) != source_hash:
            raise ValueError(f"Rename map changed while reading: {path}")
        sources.append({"path": str(path), "sha256": source_hash})
    identities = {old for old, new in forward.items() if old == new}
    forward = {old: new for old, new in forward.items() if old != new}
    reverse = {new: old for old, new in forward.items()}
    checked = set()
    for start in sorted(forward):
        visited, node = set(), start
        while node in forward and node not in checked:
            if node in visited:
                raise ValueError(f"Rename map cycle containing {node}")
            visited.add(node)
            node = forward[node]
        checked.update(visited)
    groups = {name: (name,) for name in identities}
    canonical = {name: name for name in identities}
    for start in sorted(set(forward) - set(reverse)):
        chain, node = [start], start
        while node in forward:
            node = forward[node]
            chain.append(node)
        group = tuple(chain)
        for name in chain:
            groups[name], canonical[name] = group, node
    return groups, canonical, sources


def _normalized_relative_name(name):
    path = Path(name)
    return path.with_name(normalized_name(path.name)).as_posix()


class ImageResolver:
    """Index only explicitly supplied roots. Ambiguous matches are errors."""

    def __init__(self, roots, rename_maps=None):
        self.roots = sorted({Path(root).expanduser().resolve() for root in roots})
        if not self.roots or any(not root.is_dir() for root in self.roots):
            raise ValueError("Every image_root must be an existing directory")
        self.exact = defaultdict(set)
        self.normalized = defaultdict(set)
        self.folded = defaultdict(set)
        self.relative = defaultdict(set)
        self.relative_normalized = defaultdict(set)
        self.relative_folded = defaultdict(set)
        self.rename_groups, self.rename_canonical, self.rename_map_sources = read_rename_maps(rename_maps)
        self.rename_basenames = defaultdict(set)
        self.rename_normalized = defaultdict(set)
        for name, group in self.rename_groups.items():
            self.rename_basenames[Path(name).name].add(group)
            self.rename_normalized[normalized_name(Path(name).name)].add(group)
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
                    if root in path.parents:
                        relative = path.relative_to(root).as_posix()
                        self.relative[relative].add(path)
                        self.relative_normalized[_normalized_relative_name(relative)].add(path)
                        self.relative_folded[_normalized_relative_name(relative).casefold()].add(path)

    def contains(self, path: Path) -> bool:
        return any(path == root or root in path.parents for root in self.roots)

    def _rename_matches(self, names):
        names = {str(raw).replace("\\", "/") for raw in names}
        stages = [{self.rename_groups[name] for name in names if name in self.rename_groups},
                  set().union(*(self.rename_basenames[Path(name).name] for name in names)),
                  set().union(*(self.rename_normalized[normalized_name(Path(name).name)] for name in names))]
        for groups in stages:
            if groups:
                return set().union(*groups), len(groups) > 1
        return set(), False

    def canonical_name(self, name, resolved_path=None):
        """Mapped final basename for metadata; COCO identity is kept separately."""
        raw = str(name).replace("\\", "/")
        if raw in self.rename_canonical:
            return Path(self.rename_canonical[raw]).name
        if resolved_path is not None:
            candidates = {resolved_path.relative_to(root).as_posix() for root in self.roots
                          if root in resolved_path.parents}
            mapped = {self.rename_canonical[item] for item in candidates if item in self.rename_canonical}
            if len(mapped) == 1:
                return Path(next(iter(mapped))).name
        aliases, ambiguous = self._rename_matches([raw])
        if aliases and not ambiguous:
            return Path(self.rename_canonical[next(iter(aliases))]).name
        return Path(raw).name

    def _alias_candidates(self, names, mode):
        candidates = set()
        for name in names:
            qualified = len(Path(name).parts) > 1
            if mode == "exact":
                index, key = (self.relative, name) if qualified else (self.exact, name)
            elif mode == "normalized":
                index, key = ((self.relative_normalized, _normalized_relative_name(name)) if qualified
                              else (self.normalized, normalized_name(name)))
            else:
                index, key = ((self.relative_folded, _normalized_relative_name(name).casefold()) if qualified
                              else (self.folded, normalized_name(name).casefold()))
            candidates.update(index[key])
        return candidates

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
        for rule, candidates in [("explicit_path", direct),
                                 ("exact_basename", set().union(*(self.exact[n] for n in names)))]:
            if len(candidates) > 1:
                return None, "ambiguous_" + rule, sorted(str(p) for p in candidates)
            if len(candidates) == 1:
                return next(iter(candidates)), rule, []
        aliases, ambiguous = self._rename_matches(row.get(key) for key in ("file_name", "path") if row.get(key))
        mapped = self._alias_candidates(aliases, "exact")
        if ambiguous:
            return None, "ambiguous_rename_map", sorted(str(p) for p in mapped)
        for rule, candidates in [("rename_map", mapped),
                  ("stripped_prefix", self._alias_candidates(names | aliases, "normalized")),
                  ("casefold", self._alias_candidates(names | aliases, "folded"))]:
            if len(candidates) > 1:
                return None, "ambiguous_" + rule, sorted(str(p) for p in candidates)
            if len(candidates) == 1:
                return next(iter(candidates)), rule, []
        return None, "not_found", []


def read_overrides(path: str | Path | None, ids: set[str]) -> dict:
    """Read confirmed building identity, with optional reviewed temporal metadata.

    Blank view/year values explicitly remain unknown. Semantic preparation can
    use building-only rows; temporal preparation must still require both fields.
    """
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
        building = (row.get("building_id") or "").strip()
        if not building:
            raise ValueError(f"Reviewed row {key} needs building_id")
        year_text = (row.get("year") or "").strip()
        try:
            year = int(year_text) if year_text else None
        except ValueError:
            raise ValueError(f"Invalid year for {key}: {year_text}") from None
        if year is not None and not 1800 <= year <= 2100:
            raise ValueError(f"Invalid year for {key}: {year}")
        result[key] = {"view_id": (row.get("view_id") or "").strip() or None, "building_id": building,
                       "year": year, "metadata_status": "reviewed", "metadata_source": "override_csv",
                       "metadata_notes": row.get("notes", "")}
    return result


def read_filename_rules(path, images, resolver=None) -> dict:
    """Apply confirmed views, then filename exceptions, to current image IDs."""
    if path is None:
        return {}
    rules = read_json(path)
    if not isinstance(rules, list):
        raise ValueError("Metadata rules must be a JSON list")
    by_name, by_view = defaultdict(list), defaultdict(list)
    for row in images:
        name = Path(row.get("resolved_file_name") or row["file_name"]).name
        by_name[normalized_name(name)].append(row)
        candidate = filename_metadata(name)
        if candidate["view_id"]:
            by_view[candidate["view_id"]].append((row, candidate["year"]))
    view_rules, filename_rules, seen = {}, [], set()
    for rule in rules:
        if not isinstance(rule, dict) or any(not isinstance(rule.get(k), str) or not rule[k].strip()
                for k in ("view_id", "building_id")):
            raise ValueError("Each metadata rule needs view_id and building_id")
        if "file_name" not in rule:
            if "year" in rule:
                raise ValueError("View metadata rules omit year; it comes from the filename")
            view = rule["view_id"].strip()
            old = view_rules.get(view)
            if old and old["building_id"].strip() != rule["building_id"].strip():
                raise ValueError(f"Conflicting building assignments for view rule: {view}")
            view_rules.setdefault(view, rule)
            continue
        if not isinstance(rule["file_name"], str) or not rule["file_name"].strip():
            raise ValueError("Filename metadata rules need a nonempty file_name")
        year = rule.get("year")
        if isinstance(year, bool) or not isinstance(year, int) or not 1800 <= year <= 2100:
            raise ValueError("Each metadata rule needs a valid integer year")
        name = normalized_name(resolver.canonical_name(rule["file_name"]) if resolver
                               else Path(rule["file_name"]).name)
        if name in seen or len(by_name[name]) > 1:
            raise ValueError(f"Duplicate or ambiguous metadata rule: {name}")
        seen.add(name)
        filename_rules.append((name, rule))
    result = {}
    for view, rule in view_rules.items():
        for image, year in by_view[view]:
            key = str(image.get("image_id", image.get("id")))
            result[key] = {"view_id": view, "building_id": rule["building_id"].strip(),
                           "year": year, "metadata_status": "reviewed", "metadata_source": "view_rules",
                           "metadata_notes": rule.get("notes", "Human-confirmed view mapping")}
    for name, rule in filename_rules:
        for image in by_name[name]:
            key = str(image.get("image_id", image.get("id")))
            result[key] = {"view_id": rule["view_id"].strip(), "building_id": rule["building_id"].strip(),
                           "year": rule["year"], "metadata_status": "reviewed", "metadata_source": "filename_rules",
                           "metadata_notes": rule.get("notes", "Human-confirmed filename mapping")}
    return result


def build_manifest(config_path: str | Path, out: str | Path, overrides=None,
                   previous_manifest=None) -> dict:
    """Inventory the current COCO; reuse verified RGB metadata by image identity."""
    config_path = Path(config_path).expanduser().resolve()
    config = read_json(config_path)
    # Paths in the config are relative to the config's directory, never the shell cwd.
    def absolute(value):
        p = Path(value).expanduser()
        return str(p.resolve() if p.is_absolute() else (config_path.parent / p).resolve())
    coco_path = Path(absolute(config["coco_json"]))
    roots = [absolute(root) for root in config["image_roots"]]
    rename_values = config.get("rename_maps", [])
    if rename_values is None:
        rename_values = []
    if not isinstance(rename_values, list):
        raise ValueError("rename_maps must be a list of CSV paths")
    resolver = ImageResolver(roots, [absolute(path) for path in rename_values])
    canonical_basenames = defaultdict(set)
    for name in set(resolver.rename_canonical.values()):
        canonical_basenames[normalized_name(Path(name).name)].add(name)
    for name, identities in canonical_basenames.items():
        if len(identities) > 1:
            raise ValueError(f"Ambiguous canonical rename-map basename: {name}: {sorted(identities)}")
    coco_hash = sha256(coco_path)
    coco = read_json(coco_path)
    images = coco["images"]
    ids = {str(row["id"]) for row in images}
    if len(ids) != len(images):
        raise ValueError("Duplicate COCO image ids")
    resolved = [resolver.resolve(image) for image in images]
    metadata_images = [{**image, "resolved_file_name": resolver.canonical_name(image["file_name"], path)}
                       for image, (path, _, _) in zip(images, resolved)]
    previous_path = Path(previous_manifest).expanduser().resolve() if previous_manifest else None
    previous_hash = sha256(previous_path) if previous_path else None
    previous_names = defaultdict(list)
    if previous_path:
        previous = read_json(previous_path)
        for row in previous["images"]:
            name = normalized_name(resolver.canonical_name(row.get("resolved_file_name") or row.get("file_name") or ""))
            if name:
                previous_names[name].append(row)
    current_names = Counter(normalized_name(image["resolved_file_name"]) for image in metadata_images)
    rules_value = config.get("metadata_rules", str(DEFAULT_METADATA_RULES))
    rules_path = Path(absolute(rules_value)) if rules_value else None
    rules_hash = sha256(rules_path) if rules_path else None
    metadata = {**read_filename_rules(rules_path, metadata_images, resolver), **read_overrides(overrides, ids)}
    config = {"coco_json": str(coco_path), "image_roots": roots,
              "coco_sha256": coco_hash,
              "previous_manifest_path": str(previous_path) if previous_path else None,
              "previous_manifest_sha256": previous_hash,
              "metadata_rules_path": str(rules_path) if rules_path else None,
              "metadata_rules_sha256": rules_hash,
              "rename_maps": resolver.rename_map_sources,
              "overrides_sha256": sha256(overrides) if overrides else None}
    out = new_directory(out)
    record = run_record("manifest", config)
    write_json(out / "run.json", record)
    try:
        rows = []
        reused_count = decoded_count = inherited_count = 0
        for image, (path, rule, candidates) in zip(metadata_images, resolved):
            row = {"image_id": image["id"], "file_name": image["file_name"],
                   "resolved_file_name": image["resolved_file_name"],
                   "coco_path": image.get("path"), "coco_width": image["width"],
                   "coco_height": image["height"], "split": "unassigned",
                   "inventory_validation": "not_resolved"}
            row.update(filename_metadata(image["resolved_file_name"]))
            name = normalized_name(image["resolved_file_name"])
            old_rows = previous_names.get(name, [])
            old = old_rows[0] if len(old_rows) == current_names[name] == 1 else None
            row.update(image_path=str(path) if path else None, resolution_rule=rule,
                       image_status="missing" if rule == "not_found" else "ambiguous")
            if candidates:
                row["path_candidates"] = candidates
            if path:
                try:
                    before = sha256(path)
                    same_identity = bool(old and old.get("image_status") == "ready"
                                         and old.get("sha256") == before
                                         and type(old.get("width")) is int and type(old.get("height")) is int
                                         and (old.get("width"), old.get("height")) == (image["width"], image["height"]))
                    opaque = old.get("opaque_fraction") if old else None
                    cache_complete = (same_identity and isinstance(opaque, (int, float))
                                      and not isinstance(opaque, bool) and 0 <= opaque <= 1)
                    if cache_complete:
                        w, h = old["width"], old["height"]
                        opaque_fraction = old["opaque_fraction"]
                        row.update(inventory_validation="reused_sha256_checked", reused_from_image_id=old["image_id"])
                    else:
                        row["inventory_validation"] = "decoded"
                        decoded_count += 1
                        rgb, support = load_rgb(path)
                        h, w = rgb.shape[:2]
                        opaque_fraction = float(support.mean())
                    if sha256(path) != before:
                        raise ValueError("Image changed during inspection")
                    if cache_complete:
                        reused_count += 1
                    row.update(width=w, height=h, sha256=before, opaque_fraction=opaque_fraction)
                    row["image_status"] = "ready" if (w, h) == (image["width"], image["height"]) else "dimension_mismatch"
                    if same_identity and row["image_status"] == "ready" and old.get("metadata_status") == "reviewed":
                        row.update({key: old.get(key) for key in ("view_id", "building_id", "year")})
                        if "metadata_notes" in old:
                            row["metadata_notes"] = old["metadata_notes"]
                        row.update(metadata_status="reviewed", metadata_source="previous_manifest",
                                   metadata_inherited_from_image_id=old["image_id"])
                        inherited_count += 1
                except (OSError, ValueError) as exc:
                    row.update(image_status="decode_error", error=str(exc))
            row.update(metadata.get(str(image["id"]), {}))
            rows.append(row)
        paths = Counter(row["image_path"] for row in rows if row["image_path"])
        for row in rows:
            if row["image_path"] and paths[row["image_path"]] > 1:
                row["image_status"] = "duplicate_resolved_path"
        anns = coco.get("annotations", [])
        cats = coco.get("categories", [])
        category_ids = {cat["id"] for cat in cats}
        summary = {"image_count": len(rows), "annotation_count": len(anns),
                   "reused_image_count": reused_count, "decoded_image_count": decoded_count,
                   "inherited_reviewed_metadata_count": inherited_count,
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
            f"Reused verified RGB metadata: {reused_count}; decode attempts: {decoded_count}; "
            f"inherited reviewed metadata: {inherited_count}.\n"
            "Filename candidates require review; no split or temporal ground truth has been created.\n",
            encoding="utf-8")
        if sha256(coco_path) != coco_hash:
            raise ValueError("Source COCO changed during inspection")
        if previous_path and sha256(previous_path) != previous_hash:
            raise ValueError("Previous manifest changed during inspection")
        if rules_path and sha256(rules_path) != rules_hash:
            raise ValueError("Metadata rules changed during inspection")
        for source in resolver.rename_map_sources:
            if sha256(source["path"]) != source["sha256"]:
                raise ValueError(f"Rename map changed during inspection: {source['path']}")
        status = "completed" if all(r["image_status"] == "ready" for r in rows) else "completed_with_issues"
        finish_record(out, record, status)
        return result
    except Exception as exc:
        finish_record(out, record, "failed", str(exc))
        raise
