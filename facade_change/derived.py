"""Native-grid crop export and small procedural controls, never real damage GT."""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import Canvas, absolute_residual, warp_pair
from .io import finish_record, load_rgb, new_directory, read_json, run_record, sha256, write_json


def _save(path, array):
    Image.fromarray(array).save(path)


def _identity(observations, split, group_id):
    if split not in {"dev", "exploratory", "train", "val", "test"}:
        raise ValueError("Crops require an explicit dev/exploratory or reviewed train/val/test split")
    rows = [observations[key] for key in ("reference", "source")]
    buildings = {row.get("building_id") for row in rows}
    reviewed = all(row.get("metadata_status") == "reviewed" for row in rows)
    if split in {"train", "val", "test"} and (not reviewed or len(buildings) != 1 or not rows[0].get("building_id")):
        raise ValueError("Scientific splits require reviewed observations of one known building")
    if split in {"train", "val", "test"} and any(row.get("split") != split for row in rows):
        raise ValueError("Requested split must match both stored observation splits; crops cannot assign or relabel splits")
    if split in {"dev", "exploratory"} and any(row.get("split") in {"train", "val", "test", "excluded"} for row in rows):
        raise ValueError("Crops cannot relabel an assigned or excluded observation as exploratory")
    building = rows[0].get("building_id") if len(buildings) == 1 else None
    expected = building or "unreviewed:" + str(rows[0].get("view_id"))
    if group_id is not None and group_id != expected:
        raise ValueError(f"group_id must preserve the pair's building/view identity: {expected}")
    return {"group_id": expected, "building_id": building, "view_id": rows[0].get("view_id"),
            "split": split, "metadata_status": "reviewed" if reviewed else "inferred",
            "physical_visibility": "unknown; geometric support does not exclude occluders"}


def _starts(length, size, stride):
    last = max(0, length - size)
    return sorted(set(range(0, last + 1, stride)) | {last})


def build_crops(pair_run, out, tile_size=256, stride=128, min_valid_fraction=.8,
                split="dev", group_id=None):
    """Warp original RGB directly into each fixed crop; decode each original once.

    Saved pair RGB PNGs are deliberately never read. Crop translation is composed
    with native transforms before interpolation, preserving reference pixel scale.
    """
    if not isinstance(tile_size, int) or tile_size < 8 or not isinstance(stride, int) or not 0 < stride <= tile_size:
        raise ValueError("tile_size >= 8 and 0 < stride <= tile_size are required")
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in (0, 1]")
    pair_run = Path(pair_run).resolve()
    pair_record_hash = sha256(pair_run / "run.json")
    pair_record = read_json(pair_run / "run.json")
    if not pair_record.get("status", "").startswith("completed"):
        raise ValueError("Crop input must be a completed pair run")
    if pair_record["status"] == "completed_rejected" or ("quality_gate" in pair_record and not pair_record["quality_gate"]["passed"]):
        raise ValueError("Rejected alignment cannot produce dataset crops")
    geometry_path = pair_run / "geometry.json"
    geometry_hash = sha256(geometry_path)
    if geometry_hash != pair_record.get("artifact_sha256", {}).get("geometry.json"):
        raise ValueError("Pair geometry changed or its provenance hash is absent")
    geometry = read_json(geometry_path)
    observations = pair_record["observations"]
    identity = _identity(observations, split, group_id)
    matrices = [np.asarray(geometry[key], dtype=float) for key in ("reference_to_canvas", "source_to_reference")]
    if any(matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3 for matrix in matrices):
        raise ValueError("Pair geometry requires finite nonsingular 3x3 transforms")
    # The saved canvas translation cannot change reference pixel scale.
    expected = np.eye(3)
    expected[:2, 2] = matrices[0][:2, 2]
    if not np.allclose(matrices[0], expected, atol=1e-10):
        raise ValueError("reference_to_canvas must be a translation at native pixel scale")
    out = new_directory(out)
    config = {"pair_run": str(pair_run), "pair_run_sha256": pair_record_hash,
              "geometry_sha256": geometry_hash, "tile_size": tile_size, "stride": stride,
              "min_valid_fraction": min_valid_fraction, "alignment_status": pair_record["status"],
              "alignment_quality_gate": pair_record.get("quality_gate", "legacy_run_without_gate"), **identity}
    record = run_record("native_rgb_crops", config)
    write_json(out / "run.json", record)
    try:
        arrays = []
        for role in ("reference", "source"):
            row = observations[role]
            if sha256(row["image_path"]) != row["sha256"]:
                raise ValueError(f"Original image changed: {row['image_id']}")
            rgb, support = load_rgb(row["image_path"])
            if rgb.shape[:2] != (row["height"], row["width"]) or sha256(row["image_path"]) != row["sha256"]:
                raise ValueError(f"Original image dimensions/content changed while reading: {row['image_id']}")
            arrays.extend([rgb, support])
        crops, rejected = [], 0
        gallery = []
        for y in _starts(int(geometry["height"]), tile_size, stride):
            for x in _starts(int(geometry["width"]), tile_size, stride):
                offset = np.array([[1., 0, -x], [0, 1, -y], [0, 0, 1]])
                canvas = Canvas(tile_size, tile_size, offset @ matrices[0], matrices[1])
                warped = warp_pair(*arrays, canvas)
                fraction = float(warped["overlap"].mean())
                if fraction < min_valid_fraction:
                    rejected += 1
                    continue
                name = f"crop-x{x:05d}-y{y:05d}"
                crop_dir = new_directory(out / name)
                for role in ("reference", "source"):
                    _save(crop_dir / f"{role}_rgb.png", warped[role])
                    _save(crop_dir / f"{role}_support.png", warped[role + "_support"].astype(np.uint8) * 255)
                _save(crop_dir / "geometric_overlap.png", warped["overlap"].astype(np.uint8) * 255)
                _save(crop_dir / "not_comparable.png", (~warped["overlap"]).astype(np.uint8) * 255)
                residual = absolute_residual(warped["source"], warped["reference"])
                residual[~warped["overlap"]] = 0
                _save(crop_dir / "absolute_residual_rgb.png", residual)
                files = {path.name: sha256(path) for path in sorted(crop_dir.glob("*.png"))}
                crop = {"crop_id": name, "path": name, "bbox_canvas_xyxy": [x, y, x + tile_size, y + tile_size],
                        "tile_shape": [tile_size, tile_size], "valid_fraction": fraction,
                        "reference_to_crop": canvas.reference_to_canvas.tolist(),
                        "source_to_crop": (canvas.reference_to_canvas @ canvas.source_to_reference).tolist(),
                        "reference_id": observations["reference"]["image_id"], "source_id": observations["source"]["image_id"],
                        "source_image_sha256": {key: observations[key]["sha256"] for key in observations},
                        "alignment_status": pair_record["status"],
                        "sampling": "one direct interpolation from each original RGB per crop; no resize",
                        "artifact_sha256": files, **identity}
                write_json(crop_dir / "crop.json", crop)
                crops.append(crop)
                gallery.append(f'<tr><td>{html.escape(name)}<br>{fraction:.1%} overlap</td>'
                               f'<td><img src="{name}/reference_rgb.png"></td>'
                               f'<td><img src="{name}/source_rgb.png"></td>'
                               f'<td><img src="{name}/absolute_residual_rgb.png"></td></tr>')
        if sha256(pair_run / "run.json") != pair_record_hash or sha256(geometry_path) != geometry_hash:
            raise ValueError("Pair provenance or geometry changed while generating crops")
        result = {"schema_version": 1, "config": config, "crops": crops,
                  "summary": {"crop_count": len(crops), "rejected_low_overlap": rejected, **identity}}
        write_json(out / "crops.json", result)
        _gallery(out, "Native RGB crops", "Reference / later image / RGB difference. Physical occlusion unknown.", gallery)
        finish_record(out, record, "completed" if crops else "completed_empty")
        return result["summary"]
    except Exception as exc:
        finish_record(out, record, "failed", f"{type(exc).__name__}: {exc}")
        raise


def _gallery(out, title, note, rows):
    (out / "gallery.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>' + html.escape(title) + '</title>'
        '<style>body{font:16px system-ui}td{padding:8px;vertical-align:top}img{max-width:256px}</style>'
        '<h1>' + html.escape(title) + '</h1><p>' + html.escape(note) + '</p><table>'
        + "\n".join(rows) + '</table>', encoding="utf-8")


def controlled_examples(crops_path, out, seed=42, max_crops=3):
    """Four procedural state×nuisance cases plus matched sham controls per crop.

    State: a localized color edit. Nuisance: a known integer translation and
    global photometric change. Masks label the procedural edit, not facade damage.
    """
    import cv2

    if not isinstance(max_crops, int) or max_crops < 1:
        raise ValueError("max_crops must be a positive integer")
    crops_path = Path(crops_path).resolve()
    if crops_path.is_dir():
        crops_path = crops_path / "crops.json"
    document = read_json(crops_path)
    crops_record = read_json(crops_path.parent / "run.json")
    if not crops_record.get("status", "").startswith("completed") or crops_record.get("artifact_sha256", {}).get("crops.json") != sha256(crops_path):
        raise ValueError("Controls require an unchanged completed crops.json")
    if not document["crops"]:
        raise ValueError("No valid crops available for controls")
    out = new_directory(out)
    record = run_record("procedural_state_nuisance_controls", {
        "crops_path": str(crops_path), "crops_sha256": sha256(crops_path),
        "seed": seed, "max_crops": max_crops, "label_scope": "procedural edit only; not real damage ground truth"})
    write_json(out / "run.json", record)
    try:
        rng = np.random.default_rng(seed)
        examples, gallery = [], []
        for crop in document["crops"][:max_crops]:
            directory = crops_path.parent / crop["path"]
            for name in ("reference_rgb.png", "reference_support.png"):
                if sha256(directory / name) != crop["artifact_sha256"][name]:
                    raise ValueError(f"Crop image changed: {directory / name}")
            rgb, _ = load_rgb(directory / "reference_rgb.png")
            with Image.open(directory / "reference_support.png") as image:
                support = np.asarray(image) == 255
            height, width = support.shape
            candidates = np.argwhere(support)
            if len(candidates) == 0:
                raise ValueError("Control crop has no reference support")
            center = candidates[np.argmin(np.sum((candidates - np.array([height / 2, width / 2])) ** 2, axis=1))]
            cy, cx = map(int, center)
            radius = max(2, min(height, width) // 8)
            yy, xx = np.ogrid[:height, :width]
            edit_region = (((xx - cx) / radius) ** 2 + ((yy - cy) / radius) ** 2 <= 1) & support
            inner = cv2.erode(edit_region.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            seam = edit_region & ~inner
            # This is explicitly a color-state proxy. It makes no realism claim.
            patch = rgb.copy()
            patch[edit_region] = 255 - patch[edit_region]
            dx, dy = int(rng.choice([-3, 3])), int(rng.choice([-2, 2]))
            for state, nuisance, sham in ((False, False, False), (False, True, False),
                                           (True, False, False), (True, True, False),
                                           (False, False, True), (False, True, True)):
                name = f"{crop['crop_id']}-state{int(state)}-nuisance{int(nuisance)}" + ("-sham" if sham else "")
                target = new_directory(out / name)
                current = rgb.copy()
                if state or sham:
                    current[edit_region] = (rgb if sham else patch)[edit_region]
                edit = edit_region if state else np.zeros_like(edit_region)
                transform = np.array([[1., 0, dx if nuisance else 0], [0, 1, dy if nuisance else 0], [0, 0, 1]])
                if nuisance:
                    current = np.clip(current.astype(np.float32) * .85 + 12, 0, 255).round().astype(np.uint8)
                current = cv2.warpPerspective(current, transform, (width, height), flags=cv2.INTER_LINEAR)
                source_support = cv2.warpPerspective(support.astype(np.uint8), transform, (width, height), flags=cv2.INTER_NEAREST).astype(bool)
                source_edit = cv2.warpPerspective(edit.astype(np.uint8), transform, (width, height), flags=cv2.INTER_NEAREST).astype(bool)
                to_reference = np.linalg.inv(transform)
                restored_support = cv2.warpPerspective(source_support.astype(np.uint8), to_reference, (width, height), flags=cv2.INTER_NEAREST).astype(bool)
                comparable = support & restored_support
                labels = edit.astype(np.uint8)
                labels[~comparable] = 255
                for filename, array in (("reference_rgb.png", rgb), ("source_rgb.png", current),
                                         ("reference_support.png", support.astype(np.uint8) * 255),
                                         ("source_support.png", source_support.astype(np.uint8) * 255),
                                         ("comparable_reference.png", comparable.astype(np.uint8) * 255),
                                         ("not_comparable_reference.png", (~comparable).astype(np.uint8) * 255),
                                         ("true_edit_mask_reference.png", edit.astype(np.uint8) * 255),
                                         ("true_edit_mask_source.png", source_edit.astype(np.uint8) * 255),
                                         ("insertion_boundary_reference.png", seam.astype(np.uint8) * 255),
                                         ("labels_reference.png", labels)):
                    _save(target / filename, array)
                example = {"example_id": name, "path": name, "parent_crop_id": crop["crop_id"],
                           "base_observation_role": "reference",
                           "state_change": state, "nuisance": nuisance, "sham_self_paste": sham,
                           "reference_to_source": transform.tolist(), "source_to_reference": to_reference.tolist(),
                           "nuisance_parameters": {"gain": .85 if nuisance else 1., "offset": 12 if nuisance else 0,
                                                   "translation_xy": [dx, dy] if nuisance else [0, 0]},
                           "state_edit": "pixelwise RGB complement inside ellipse" if state else "none",
                           "labels": {"0": "procedural state unchanged", "1": "procedural state edit", "255": "not geometrically comparable"},
                           "label_scope": "procedural edit only; no real damage GT", "seed": seed,
                           "parent_crop_provenance_sha256": sha256(directory / "crop.json"),
                           **{key: crop[key] for key in ("group_id", "building_id", "view_id", "split", "metadata_status", "physical_visibility", "source_image_sha256")}}
                write_json(target / "example.json", example)
                examples.append(example)
                gallery.append(f'<tr><td>{html.escape(name)}</td><td><img src="{name}/reference_rgb.png"></td>'
                               f'<td><img src="{name}/source_rgb.png"></td><td><img src="{name}/true_edit_mask_reference.png"></td></tr>')
        result = {"schema_version": 1, "examples": examples,
                  "summary": {"source_crop_count": min(max_crops, len(document["crops"])), "example_count": len(examples),
                              "factorial_cases_per_crop": 4, "sham_controls_per_crop": 2, "label_scope": "procedural only"}}
        write_json(out / "examples.json", result)
        _gallery(out, "Procedural state × nuisance controls", "Reference / current / true procedural edit. Known geometry is in example.json; all variants keep their parent group and split.", gallery)
        finish_record(out, record, "completed")
        return result["summary"]
    except Exception as exc:
        finish_record(out, record, "failed", f"{type(exc).__name__}: {exc}")
        raise
