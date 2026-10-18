#!/usr/bin/env python3
"""Export a validated, visualization-only facade browser dataset.

The input is an existing collector bundle; no model, alignment estimation or
scientific evaluation is run. Saved raw-RGB homographies are applied to full
photos for display. Numeric packed GT and geometric support are preserved in a
lossless RGB PNG (R=low byte, G=high byte, B=255 for supported pixels).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from pathlib import Path

import numpy as np
import PIL
from PIL import Image
import scipy
from scipy import ndimage

DATE = "2026-10-18"
LABELS = {
    "WATER_STAIN": "Следы увлажнения",
    "MISSING_ELEMENT": "Утраченный элемент",
    "DELAMINATION": "Отслоение",
    "SPALLING": "Выкрашивание",
    "ORNAMENT_INTACT": "Сохранный декор",
    "CRACK": "Трещина",
    "REPAIRS": "Следы ремонта",
    "EFFLORESCENCE": "Высолы",
    "TEXT_OR_IMAGES": "Надписи и изображения",
    "CORROSION": "Коррозия",
    "ADVERTISEMENTS": "Реклама",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    require(not Path(relative).is_absolute() and path.is_relative_to(root.resolve()),
            f"Unsafe bundle path: {relative}")
    require(path.is_file(), f"Missing bundle file: {relative}")
    return path


def display_image(image: Image.Image, path: Path, max_size=1400):
    image = image.copy()
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    image.save(path, format="WEBP", quality=88, method=5)
    return image.size


def saved_warp(image: Image.Image, matrix, size):
    """Map output pixel centres back into the source using the saved matrix.

    Dataset centres are at integer coordinates. Pillow uses half-integer pixel
    centres, so its inverse transform is T(+0.5) H^-1 T(-0.5).
    """
    inverse = np.linalg.inv(matrix)
    plus = np.array([[1, 0, .5], [0, 1, .5], [0, 0, 1]], dtype=np.float64)
    minus = np.array([[1, 0, -.5], [0, 1, -.5], [0, 0, 1]], dtype=np.float64)
    inverse = plus @ inverse @ minus
    inverse /= inverse[2, 2]
    return image.transform(size, Image.Transform.PERSPECTIVE,
                           inverse.ravel()[:8], Image.Resampling.BILINEAR,
                           fillcolor=(23, 34, 44))


def pair_candidates(previous, current, categories, pair_id, min_pixels):
    before, before_support = previous
    after, after_support = current
    valid = before_support & after_support & (before != 0) & (after != 0)
    changed = valid & (before != after)
    events = []
    all_components = 0
    rejected_components = 0
    # Eight-neighbour connectivity; no morphology or smoothing.
    structure = np.ones((3, 3), dtype=np.uint8)
    for category in categories:
        bit = 1 << category["channel"]
        old = (before & bit) != 0
        new = (after & bit) != 0
        for direction, mask in (("added", valid & ~old & new),
                                ("removed", valid & old & ~new)):
            labels, count = ndimage.label(mask, structure=structure)
            all_components += int(count)
            areas = np.bincount(labels.ravel())
            slices = ndimage.find_objects(labels)
            for component, region in enumerate(slices, 1):
                area = int(areas[component])
                if area < min_pixels:
                    rejected_components += 1
                    continue
                ys, xs = region
                phrase = "Добавлена метка" if direction == "added" else "Удалена метка"
                events.append({
                    "id": f"{pair_id}:{category['id']}:{direction}:{component}",
                    "category_id": category["id"], "direction": direction,
                    "area_pixels": area,
                    "bbox": [xs.start, ys.start, xs.stop - xs.start, ys.stop - ys.start],
                    "label": f"{phrase}: {category['label'].lower()}",
                })
    events.sort(key=lambda event: (-event["area_pixels"], event["category_id"], event["id"]))
    valid_pixels = int(valid.sum())
    changed_pixels = int(changed.sum())
    return {
        "valid_pixels": valid_pixels, "changed_pixels": changed_pixels,
        "changed_fraction": changed_pixels / valid_pixels if valid_pixels else None,
        "events": events, "candidate_components_before_filter": all_components,
        "candidate_components_below_minimum": rejected_components,
    }


def export(source: Path, dist: Path, min_pixels=64):
    source = source.resolve()
    dist = dist.resolve()
    data_dir, assets_dir = dist / "data", dist / "assets"
    data_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    bundle = read_json(source / f"{DATE}-bundle.json")
    manifest = read_json(source / "manifest.json")
    originals = read_json(source / f"{DATE}-originals.json")
    audit = read_json(source / "source/audit.json")
    require(bundle["kind"] == "facade_prototype_input_bundle", "Unexpected bundle kind")
    require(manifest["kind"] == "real_temporal_prepared" and manifest["schema_version"] == 1,
            "Unexpected prepared manifest")
    require(audit["kind"] == "real_temporal_audit", "Unexpected audit kind")
    checked = {}
    for entry in bundle["files"]:
        path = inside(source, entry["path"])
        require(path.stat().st_size == entry["bytes"], f"Size mismatch: {entry['path']}")
        digest = sha256(path)
        require(digest == entry["sha256"], f"SHA-256 mismatch: {entry['path']}")
        checked[entry["path"]] = digest
    require(checked["manifest.json"] == bundle["prepared_manifest_sha256"], "Manifest SHA mismatch")
    require(checked["source/audit.json"] == manifest["audit_file_sha256"], "Audit SHA mismatch")
    channels = {int(row["category_id"]): int(row["channel"]) for row in manifest["category_to_channel"]}
    category_ids = [int(category["id"]) for category in manifest["categories"]]
    require(len(category_ids) == len(set(category_ids)) == len(channels), "Duplicate category mapping")
    require(set(category_ids) == set(channels), "Category IDs differ from bit mapping")
    require(sorted(channels.values()) == list(range(len(channels))) and len(channels) <= 16,
            "Invalid packed category channels")
    categories = [{"id": category["id"], "name": category["name"],
                   "label": LABELS.get(category["name"], category["name"]),
                   "color": category["color"], "channel": channels[category["id"]]}
                  for category in manifest["categories"]]
    valid_bits = sum(1 << channel for channel in channels.values())
    source_originals = {(frame["sequence_id"], frame["frame"]): frame for frame in originals["frames"]}
    require(len(source_originals) == len(originals["frames"]), "Duplicate original frame mapping")
    audit_sequences = {sequence["sequence_id"]: sequence for sequence in audit["sequences"]}
    facade_splits = {}
    facades, checks, output_assets = [], [], []
    all_events = 0
    for sequence in manifest["sequences"]:
        sid, fid = sequence["sequence_id"], sequence["building_id"]
        require(isinstance(sid, str) and re.fullmatch(r"[A-Za-z0-9_-]+", sid),
                "Sequence ID must be a safe directory name")
        require(fid not in facade_splits,
                f"Multiple selected views/crops for building {fid}: this prototype supports one history per building")
        require(len(sequence["frames"]) >= 2, f"At least two observations required: {sid}")
        facade_splits[fid] = sequence["split"]
        audit_sequence = audit_sequences[sequence["parent_sequence_id"]]
        for key in ("building_id", "view_id", "split"):
            require(sequence[key] == audit_sequence[key], f"Audit identity mismatch: {sid}:{key}")
        audit_frames = {frame["image_id"]: frame for frame in audit_sequence["frames"]}
        first = source_originals[sid, sequence["frames"][0]["frame"]]
        reference_size = (int(first["width"]), int(first["height"]))
        require(np.allclose(first["source_to_reference"], np.eye(3), atol=1e-9), f"First reference not identity: {sid}")
        require(audit_sequence["shape"] == list(reference_size[::-1]), f"Reference shape mismatch: {sid}")
        x, y, width, height = sequence["crop_xywh"]
        require(x >= 0 and y >= 0 and x + width <= reference_size[0] and y + height <= reference_size[1],
                f"Crop outside reference: {sid}")
        require(sequence["shape"] == [height, width], f"Crop shape mismatch: {sid}")
        subdir = assets_dir / f"{DATE}-{sid}"
        subdir.mkdir(exist_ok=True)
        output_frames, arrays = [], []
        previous_year = None
        for index, frame in enumerate(sequence["frames"]):
            year = int(frame["year"])
            require(previous_year is None or year > previous_year, f"Non-unique/unordered year: {sid}")
            previous_year = year
            original = source_originals[sid, frame["frame"]]
            af = audit_frames[frame["image_id"]]
            for key in ("image_id", "year"):
                require(frame[key] == original[key] == af[key], f"Original identity mismatch: {sid}:{key}")
            require(original["crop_xywh"] == sequence["crop_xywh"], f"Original crop mismatch: {sid}")
            require(original["sha256"] == frame["image_sha256"] == af["sha256"], "Original image hashes differ")
            require(checked[original["artifact_path"]] == original["sha256"], "Original SHA missing")
            require(checked[frame["artifact_path"]] == frame["artifact_sha256"], "Frame artifact SHA differs")
            matrix = np.asarray(original["source_to_reference"], dtype=np.float64)
            require(matrix.shape == (3, 3) and np.isfinite(matrix).all() and abs(np.linalg.det(matrix)) > 1e-12,
                    f"Invalid homography: {sid}:{index}")
            require(np.array_equal(matrix, np.asarray(af["source_to_reference"])), "Audit homography mismatch")
            with np.load(inside(source, frame["artifact_path"]), allow_pickle=False) as loaded:
                rgb, support, packed = loaded["rgb"].copy(), loaded["support"].copy(), loaded["packed"].copy()
            require(rgb.dtype == np.uint8 and rgb.shape == (height, width, 3), "RGB dtype/shape mismatch")
            require(support.dtype == np.bool_ and support.shape == (height, width), "Support dtype/shape mismatch")
            require(packed.dtype == np.uint16 and packed.shape == (height, width), "Packed GT dtype/shape mismatch")
            require(not np.any(packed.astype(np.uint32) & (0xffff ^ valid_bits)), "Undeclared label bit")
            image = Image.open(inside(source, original["artifact_path"]))
            require(image.size == (original["width"], original["height"]) == (af["width"], af["height"]),
                    f"Original dimensions mismatch: {sid}:{index}")
            image = image.convert("RGB")
            aligned = saved_warp(image, matrix, reference_size)
            crop_pixels = np.asarray(aligned)[y:y+height, x:x+width]
            error = np.abs(crop_pixels.astype(np.float32) - rgb.astype(np.float32))[support]
            checks.append({"sequence_id": sid, "index": index, "year": year,
                           "comparison_scope": "native RGB warp crop on saved geometric support, before WebP encoding",
                           "mean_absolute_channel_difference_0_255": float(error.mean()) if len(error) else None,
                           "p95_absolute_channel_difference_0_255": float(np.percentile(error, 95)) if len(error) else None})
            basename = f"{DATE}-{year}-{index:03d}"
            urls = {}
            dimensions = {}
            for role, visual in (("original", image), ("aligned", aligned), ("crop", Image.fromarray(rgb))):
                path = subdir / f"{basename}-{role}.webp"
                dimensions[role] = display_image(visual, path)
                urls[f"{role}_url"] = path.relative_to(dist).as_posix()
                output_assets.append(path)
            encoded = np.stack([(packed & 255).astype(np.uint8),
                                (packed >> 8).astype(np.uint8), support.astype(np.uint8) * 255], axis=-1)
            mask_path = subdir / f"{basename}-mask.png"
            Image.fromarray(encoded).save(mask_path, format="PNG", optimize=True)
            reloaded = np.asarray(Image.open(mask_path))
            require(np.array_equal(reloaded, encoded), "Mask PNG failed exact byte roundtrip")
            require(np.array_equal(reloaded[:, :, 0].astype(np.uint16) | (reloaded[:, :, 1].astype(np.uint16) << 8), packed),
                    "Mask bit packing failed exact roundtrip")
            output_assets.append(mask_path)
            known = support & (packed != 0)
            output_frames.append({
                "id": f"{sid}:{index}", "index": index, "year": year, **urls,
                "mask_url": mask_path.relative_to(dist).as_posix(),
                "source_width": image.width, "source_height": image.height,
                "aligned_width": dimensions["aligned"][0], "aligned_height": dimensions["aligned"][1],
                "source_to_reference": original["source_to_reference"],
                "known_pixels": int(known.sum()), "support_pixels": int(support.sum()),
                "class_pixels": {str(category["id"]): int((support & ((packed & (1 << category["channel"])) != 0)).sum())
                                 for category in categories},
            })
            arrays.append((packed, support))
        pairs = []
        for index in range(1, len(output_frames)):
            pair_id = f"{sid}:{index-1}:{index}"
            details = pair_candidates(arrays[index - 1], arrays[index], categories, pair_id, min_pixels)
            all_events += len(details["events"])
            pairs.append({"id": pair_id, "from_index": index - 1, "to_index": index,
                          "from_year": output_frames[index - 1]["year"], "to_year": output_frames[index]["year"], **details})
        facades.append({"id": fid, "sequence_id": sid, "view_id": sequence["view_id"],
                        "split": sequence["split"], "crop_xywh": sequence["crop_xywh"],
                        "reference_width": reference_size[0], "reference_height": reference_size[1],
                        "missing_years": sequence["missing_years"], "frames": output_frames, "pairs": pairs})
        print(f"Exported {fid}: {len(output_frames)} observations, {sum(len(pair['events']) for pair in pairs)} candidates", flush=True)
    frame_count = sum(len(facade["frames"]) for facade in facades)
    require(len(facades) == bundle["sequences"] and frame_count == bundle["frames"], "Export coverage mismatch")
    payload = {
        "schema_version": 1, "kind": "facade_atlas_browser_dataset", "created_on": DATE,
        "scope": "selected prepared crop histories with associated full source views",
        "categories": categories, "facades": facades,
        "contract": {
            "ground_truth": "existing spatial class-set annotations; not verified physical event ground truth",
            "unknown": "packed=0 is unlabelled, not intact or healthy",
            "support": "geometric image coverage, not an occlusion/visibility mask",
            "mask_encoding": "RGB PNG: packed=R+256*G, geometric_support=B==255",
            "crop_coordinates": "native earliest-reference pixel centres; event bbox is relative to crop",
            "event_candidates": {"comparison": "adjacent available years only", "valid_pixels": "support_before & support_after & packed_before!=0 & packed_after!=0",
                                 "connectivity": 8, "minimum_component_pixels": min_pixels, "morphology": "none", "cap": None,
                                 "meaning": "per-class additions/removals of annotation bits; candidates requiring review"},
            "changed_fraction": "different complete packed class sets divided by jointly supported and labelled pixels; not facade risk or full-facade damage percentage",
            "alignment": "saved source_to_reference homography applied for visualization; no new estimation or dense alignment quality claim",
            "full_image_coverage": "whole source photograph of the selected view, not proof of complete facade coverage",
        },
    }
    data_path = data_dir / f"{DATE}-facades.json"
    write_json(data_path, payload)
    for path in output_assets:
        with Image.open(path) as image:
            image.load()
    output_fingerprints = [{"path": path.relative_to(dist).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
                           for path in [data_path, *output_assets]]
    validation = {
        "schema_version": 1, "status": "passed", "date": DATE,
        "source_bundle_sha256": sha256(source / f"{DATE}-bundle.json"),
        "prepared_manifest_sha256": checked["manifest.json"],
        "input_file_hashes_checked": len(checked), "numeric_frames_checked": frame_count,
        "mask_exact_roundtrips": frame_count, "original_dimensions_and_audit_transforms_checked": frame_count,
        "facades": len(facade_splits), "sequences": len(facades), "frames": frame_count,
        "adjacent_pairs": sum(len(facade["pairs"]) for facade in facades),
        "event_candidates_after_minimum_filter": all_events,
        "output_assets_decoded": len(output_assets),
        "validation_scope": "All files listed by the input bundle; historical external hashes in the manifest are not locally available and are not revalidated.",
        "warp_comparison_note": "Interpolation implementation differences are diagnostic only; this compares rendering of the saved transform, not physical alignment accuracy.",
        "warp_crop_comparisons": checks,
        "output_files": output_fingerprints,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pillow": PIL.__version__},
    }
    write_json(data_dir / f"{DATE}-validation.json", validation)
    print(json.dumps({key: validation[key] for key in ("status", "input_file_hashes_checked", "facades", "frames", "adjacent_pairs", "event_candidates_after_minimum_filter", "output_assets_decoded")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Extracted collector bundle directory")
    parser.add_argument("--dist", required=True, type=Path, help="Browser dist directory")
    parser.add_argument("--minimum-component-pixels", type=int, default=64)
    arguments = parser.parse_args()
    require(arguments.minimum_component_pixels > 0, "Minimum component size must be positive")
    export(arguments.input, arguments.dist, arguments.minimum_component_pixels)
