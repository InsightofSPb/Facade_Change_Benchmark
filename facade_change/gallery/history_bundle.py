#!/usr/bin/env python3
"""Collect existing façade histories; requires Python 3.9+, no dependencies.

Input contract: scripts/real_eval/night_prepare.py, real_temporal_prepared v1.
The original manifest is preserved. Its frame paths remain relative to the
archive root. Original-image paths, when requested, are indexed in
2026-10-18-originals.json. Collection never runs inference.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read JSON {path}: {error}") from error


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_hash(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label}: expected a lowercase SHA-256 checksum")
    return value


def required(row, fields, label):
    if not isinstance(row, dict):
        raise ValueError(f"{label}: expected an object")
    missing = [key for key in fields if key not in row or row[key] is None]
    if missing:
        raise ValueError(f"{label}: missing required fields: {', '.join(missing)}")


def collect(prepared, output, include_originals=False):
    root = prepared.expanduser().resolve()
    manifest_path = root if root.is_file() else root / "manifest.json"
    if manifest_path.name != "manifest.json" or not manifest_path.is_file():
        raise ValueError(f"Expected an existing prepared directory or manifest.json: {prepared}")
    root = manifest_path.parent
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"Output already exists; choose a new --out path: {output}")
    if not output.name.endswith(".tar.gz"):
        raise ValueError("--out must be a new .tar.gz archive path")
    manifest = read_json(manifest_path)
    if manifest.get("kind") != "real_temporal_prepared" or manifest.get("schema_version") != 1:
        raise ValueError("Expected kind=real_temporal_prepared and schema_version=1")
    required(manifest, ["categories", "category_to_channel", "sequences"], "manifest")
    categories = manifest["categories"]
    if not isinstance(categories, list) or not 1 <= len(categories) <= 16:
        raise ValueError("manifest.categories must contain 1–16 categories")
    for category in categories:
        required(category, ["id", "name"], "category")
    ids = [str(category["id"]) for category in categories]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate category IDs")
    mapping = manifest["category_to_channel"]
    if isinstance(mapping, list):
        for row in mapping:
            required(row, ["category_id", "channel"], "category_to_channel")
        pairs = [(str(row["category_id"]), row["channel"]) for row in mapping]
    elif isinstance(mapping, dict):
        pairs = [(str(key), value) for key, value in mapping.items()]
    else:
        raise ValueError("category_to_channel must be a list or dictionary")
    if len(pairs) != len(ids) or dict(pairs) != {key: i for i, key in enumerate(ids)}:
        raise ValueError("category_to_channel disagrees with the category/bit order")

    assets, warnings = {}, []

    def add(path, archive_path, expected=None):
        name = PurePosixPath(archive_path)
        if name.is_absolute() or ".." in name.parts or "\\" in archive_path:
            raise ValueError(f"Unsafe archive path: {archive_path}")
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"Required input file is missing: {path}")
        actual = checksum(path)
        if expected is not None and actual != valid_hash(expected, str(path)):
            raise ValueError(f"SHA-256 mismatch: {path}. Use the unchanged source run.")
        entry = {"path": str(name), "sha256": actual, "bytes": path.stat().st_size}
        if str(name) in assets and assets[str(name)][1] != entry:
            raise ValueError(f"Conflicting destination in archive: {name}")
        assets[str(name)] = (path, entry)

    add(manifest_path, "manifest.json")
    sequences = manifest["sequences"]
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("manifest.sequences must be a nonempty list")
    seen, ownership, frame_count = set(), {}, 0
    for sequence in sequences:
        required(sequence, ["sequence_id", "building_id", "view_id", "split", "crop_xywh", "shape", "frames"], "sequence")
        label = str(sequence["sequence_id"])
        if label in seen:
            raise ValueError(f"Duplicate sequence_id: {label}")
        seen.add(label)
        building, split = sequence["building_id"], sequence["split"]
        if split not in ("train", "val", "test") or not building:
            raise ValueError(f"{label}: invalid building_id or split")
        if building in ownership and ownership[building] != split:
            raise ValueError(f"Building appears in multiple splits: {building}")
        ownership[building] = split
        frames = sequence["frames"]
        if not isinstance(frames, list) or len(frames) < 2:
            raise ValueError(f"{label}: expected at least two prepared frames")
        years = []
        for index, frame in enumerate(frames):
            required(frame, ["frame", "year", "image_id", "image_sha256", "artifact_path", "artifact_sha256"], f"{label} frame {index}")
            if frame["frame"] != index or type(frame["year"]) is not int:
                raise ValueError(f"{label}: invalid frame index or noninteger year")
            years.append(frame["year"])
            valid_hash(frame["image_sha256"], f"{label} original image")
            relative = frame["artifact_path"]
            if not isinstance(relative, str) or not relative:
                raise ValueError(f"{label}: invalid artifact_path")
            name = PurePosixPath(relative)
            if name.is_absolute() or ".." in name.parts or "\\" in relative or name.parts[0] != "frames":
                raise ValueError(f"{label}: expected safe relative frames/... path, got {relative!r}")
            source = (root / relative).resolve()
            if not source.is_relative_to(root):
                raise ValueError(f"Frame resolves outside the prepared directory: {source}")
            add(source, relative, frame["artifact_sha256"])
            frame_count += 1
        if years != sorted(set(years)):
            raise ValueError(f"{label}: this prepared format requires strictly increasing years")

    metadata, missing_metadata = [], []
    for name in ("run.json", "summary.json", "facades.csv", "views.csv", "candidate_crops.csv"):
        path = root / name
        if path.is_file():
            add(path, name)
            metadata.append(name)
        else:
            missing_metadata.append(name)
    audit = None
    audit_ref = manifest.get("audit_path")
    if audit_ref:
        audit_path = Path(audit_ref).expanduser()
        if not audit_path.is_absolute():
            raise ValueError("manifest.audit_path must be absolute; refusing to guess its base")
        if audit_path.is_file():
            add(audit_path, "source/audit.json", manifest.get("audit_file_sha256"))
            audit = read_json(audit_path)
        else:
            warnings.append(f"Optional audit file is unavailable: {audit_path}")
    else:
        warnings.append("No audit_path in manifest; original-image paths and transforms are unavailable")

    originals = []
    if include_originals:
        if audit is None:
            raise ValueError("--include-originals requires the audit.json referenced by manifest.audit_path; restore it or omit this flag")
        if audit.get("kind") != "real_temporal_audit" or not isinstance(audit.get("sequences"), list):
            raise ValueError("Referenced audit is not a real_temporal_audit with sequences")
        parents = {row["sequence_id"]: row for row in audit["sequences"]}
        for sequence in sequences:
            parent_id = sequence.get("parent_sequence_id")
            parent = parents.get(parent_id)
            if parent is None:
                raise ValueError(f"No audit parent for prepared sequence {sequence['sequence_id']}: {parent_id}")
            for field in ("building_id", "view_id", "split"):
                if parent.get(field) != sequence[field]:
                    raise ValueError(f"Audit/prepared {field} mismatch: {sequence['sequence_id']}")
            by_id = {str(row["image_id"]): row for row in parent["frames"]}
            for frame in sequence["frames"]:
                original = by_id.get(str(frame["image_id"]))
                required(original, ["image_path", "sha256", "year", "source_to_reference", "width", "height"], f"audit image {frame['image_id']}")
                if original["sha256"] != frame["image_sha256"] or original["year"] != frame["year"]:
                    raise ValueError(f"Audit/prepared image identity mismatch: {frame['image_id']}")
                matrix = original["source_to_reference"]
                if (not isinstance(matrix, list) or len(matrix) != 3 or any(
                        not isinstance(row, list) or len(row) != 3 or any(
                            type(value) not in (int, float) or not math.isfinite(value) for value in row) for row in matrix)):
                    raise ValueError(f"Invalid 3×3 source_to_reference: image {frame['image_id']}")
                source = Path(original["image_path"]).expanduser()
                if not source.is_absolute():
                    raise ValueError(f"Original path is not absolute: {source}")
                suffix = source.suffix.lower()
                if not suffix or len(suffix) > 12 or not suffix[1:].isalnum():
                    suffix = ".image"
                relative = f"originals/{original['sha256']}{suffix}"
                add(source, relative, original["sha256"])
                originals.append({"sequence_id": sequence["sequence_id"], "parent_sequence_id": parent_id,
                                  "frame": frame["frame"], "image_id": frame["image_id"], "year": frame["year"],
                                  "artifact_path": relative, "sha256": original["sha256"],
                                  "width": original["width"], "height": original["height"],
                                  "source_to_reference": matrix, "crop_xywh": sequence["crop_xywh"]})
    else:
        warnings.append("Original full photographs are not included; use --include-originals to collect the selected histories' originals")

    summary = {"schema_version": 1, "kind": "facade_prototype_input_bundle", "prepared_manifest": "manifest.json",
               "prepared_manifest_sha256": assets["manifest.json"][1]["sha256"],
               "scope": "selected prepared crop histories, not the full dataset",
               "sequences": len(sequences), "buildings": len(ownership), "frames": frame_count,
               "unique_frame_files": sum(name.startswith("frames/") for name in assets),
               "categories": len(categories), "audit_included": audit is not None,
               "original_files": sum(name.startswith("originals/") for name in assets),
               "optional_metadata_included": metadata, "optional_metadata_missing": missing_metadata,
               "warnings": warnings,
               "validation_scope": "manifest identities, safe paths and SHA-256; numeric arrays are copied without decoding",
               "frame_array_contract": {"rgb": "uint8 H,W,3", "support": "bool H,W, geometric support",
                                        "packed": "uint16 H,W, multilabel bits; zero means unknown"},
               "files": [entry for _, entry in assets.values()]}
    generated = {"2026-10-18-bundle.json": summary}
    if originals:
        generated["2026-10-18-originals.json"] = {
            "coordinates": "native earliest-reference pixel centres, x=column, y=row",
            "frames": originals}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="facade-collect-", dir=output.parent) as temporary:
        archive_path = Path(temporary) / "bundle.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            for name, (path, entry) in assets.items():
                info = archive.gettarinfo(str(path), arcname=name)
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
                if checksum(path) != entry["sha256"]:
                    raise ValueError(f"Input changed during collection: {path}")
            for name, document in generated.items():
                content = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
                info = tarfile.TarInfo(name)
                info.size = len(content)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(content))
        # Atomic create without replacing an existing destination.
        os.link(archive_path, output)
    print(json.dumps({key: summary[key] for key in ("sequences", "buildings", "frames", "unique_frame_files",
                                                   "categories", "audit_included", "original_files", "warnings")},
                     ensure_ascii=False, indent=2))
    print(f"Archive: {output}")
    print(f"SHA-256: {checksum(output)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path, help="Existing prepared directory or manifest.json")
    parser.add_argument("--out", required=True, type=Path, help="New .tar.gz output; never overwritten")
    parser.add_argument("--include-originals", action="store_true", help="Also copy verified originals from the referenced audit")
    args = parser.parse_args(argv)
    try:
        collect(args.prepared, args.out, args.include_originals)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
