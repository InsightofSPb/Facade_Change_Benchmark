"""One pair per run; all output paths are new and inputs are read-only."""
from __future__ import annotations

import csv
import html
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .alignment import LoFTRMatcher, SIFTMatcher, align
from .geometry import absolute_residual, expanded_canvas, warp_pair
from .io import finish_record, load_rgb, new_directory, read_json, run_record, sha256, write_json


def select_pair(manifest, reference_id, source_id, allow_inferred=False):
    by_id = {str(row["image_id"]): row for row in manifest["images"]}
    if len(by_id) != len(manifest["images"]):
        raise ValueError("Duplicate manifest image ids")
    if str(reference_id) == str(source_id):
        raise ValueError("A temporal pair requires two distinct image ids")
    try:
        reference, source = by_id[str(reference_id)], by_id[str(source_id)]
    except KeyError as exc:
        raise ValueError(f"image_id absent from manifest: {exc}") from exc
    for row in (reference, source):
        if row["image_status"] != "ready":
            raise ValueError(f"Image {row['image_id']} is {row['image_status']}")
        if row["metadata_status"] != "reviewed" and not allow_inferred:
            raise ValueError("Review this pair in metadata_review.csv, rebuild with --overrides; or explicitly allow inferred metadata")
        if row.get("year") is None or not row.get("view_id"):
            raise ValueError("Pair requires known year and view_id, even with --allow-inferred-metadata")
    if reference["view_id"] != source["view_id"]:
        raise ValueError("Pair must belong to the same view_id")
    if reference.get("building_id") != source.get("building_id"):
        raise ValueError("Pair building_id values differ")
    if int(reference["year"]) >= int(source["year"]):
        raise ValueError("Reference must be earlier than source for this temporal diagnostic")
    return reference, source


def save_png(path, array):
    Image.fromarray(array).save(path)


def save_diagnostics(out: Path, warped: dict, alignment, canvas) -> dict:
    overlap = warped["overlap"]
    count = int(overlap.sum())
    if count == 0:
        raise ValueError("No fully supported geometric overlap")
    for name in ("reference", "source"):
        save_png(out / f"{name}_rgb.png", warped[name])
    for name in ("reference_support", "source_support", "overlap", "reference_only", "source_only"):
        save_png(out / f"{name}.png", warped[name].astype(np.uint8) * 255)
    residual = absolute_residual(warped["source"], warped["reference"])
    residual[~overlap] = 0
    save_png(out / "absolute_residual_rgb.png", residual)
    score = residual.mean(axis=2, dtype=np.float32)
    score[~overlap] = np.nan
    np.save(out / "rgb_mean_absolute_difference.npy", score, allow_pickle=False)
    mean_difference = float(score[overlap].mean())
    del score

    # Preview downscaling is separate from native scientific arrays.
    yy, xx = np.ogrid[:canvas.height, :canvas.width]
    checker = np.where(((xx // 128 % 2) == (yy // 128 % 2))[..., None], warped["reference"], warped["source"])
    preview = Image.fromarray(checker)
    preview.thumbnail((1400, 1000))
    preview.save(out / "checkerboard_preview.jpg", quality=92)
    del checker
    overlay = ((warped["reference"].astype(np.uint16) + warped["source"]) // 2).astype(np.uint8)
    overlay[~overlap] = [40, 40, 40]
    for name, array in (("overlay_preview", overlay), ("residual_preview", residual)):
        preview = Image.fromarray(array)
        preview.thumbnail((1400, 1000))
        preview.save(out / f"{name}.jpg", quality=92)

    # Three unresized detail strips (reference/source/residual) from overlap extent.
    ys = np.flatnonzero(overlap.any(axis=1))
    detail_files = []
    crop_records = []
    for i, fraction in enumerate((.25, .5, .75)):
        cy = int(ys[min(len(ys) - 1, int(fraction * len(ys)))])
        xs = np.flatnonzero(overlap[cy])
        cx = int(xs[len(xs) // 2])
        x0, y0 = max(0, cx - 128), max(0, cy - 128)
        x1, y1 = min(canvas.width, x0 + 256), min(canvas.height, y0 + 256)
        strip = np.concatenate([warped["reference"][y0:y1, x0:x1], warped["source"][y0:y1, x0:x1],
                                residual[y0:y1, x0:x1]], axis=1)
        filename = f"detail_{i}.png"
        save_png(out / filename, strip)
        detail_files.append(filename)
        crop_records.append({"file": filename, "bbox_canvas_xyxy": [x0, y0, x1, y1], "resampled": False})
    write_json(out / "detail_crops.json", crop_records)
    with (out / "matches.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_x", "source_y", "reference_x", "reference_y", "inlier"])
        for source, reference, inlier in zip(alignment.matches.source, alignment.matches.reference, alignment.inliers):
            writer.writerow([*source, *reference, int(inlier)])
    metrics = {**alignment.diagnostics, "overlap_pixels": count,
               "overlap_fraction_of_reference_support": count / int(warped["reference_support"].sum()),
               "source_only_pixels": int(warped["source_only"].sum()),
               "reference_only_pixels": int(warped["reference_only"].sum()),
               "mean_rgb_difference_on_overlap": mean_difference,
               "interpretation": "Alignment diagnostics only; no damage labels, F1 or AP"}
    write_json(out / "diagnostics.json", metrics)
    figures = [("Наложение, только пересечение", "overlay_preview.jpg"),
               ("Шахматное совмещение", "checkerboard_preview.jpg"),
               ("Абсолютная RGB-разность", "residual_preview.jpg")]
    figures += [("Деталь: опора / поздний кадр / разность; исходный масштаб", name) for name in detail_files]
    markup = "".join(f'<figure><figcaption>{html.escape(title)}</figcaption><img src="{name}"></figure>' for title, name in figures)
    (out / "gallery.html").write_text(
        '<!doctype html><html lang="ru"><meta charset="utf-8"><title>Проверка совмещения</title>'
        '<style>body{font:16px system-ui;max-width:1500px;margin:32px auto;padding:16px;background:#f4f4f4}'
        'figure{margin:24px 0}img{max-width:100%}pre{white-space:pre-wrap}</style>'
        '<h1>Проверка совмещения одной пары</h1>'
        '<p>Разность не равна повреждению. Маска overlap учитывает геометрию и alpha, '
        'но не устраняет деревья, автомобили, леса, блики или тени. Нужна визуальная проверка.</p>'
        + markup + '</html>', encoding="utf-8")
    (out / "summary.txt").write_text(
        f"Method: {alignment.matches.metadata['id']}\nCanvas: {canvas.width}x{canvas.height}\n"
        f"Matches/inliers: {metrics['matches']}/{metrics['inliers']}\n"
        f"Median inlier reprojection error (native ref pixels): {metrics['inlier_reprojection_median_px']:.4f}\n"
        f"Overlap pixels: {count}\nNew geometric support: {metrics['source_only_pixels']} pixels\n"
        "Status: needs_visual_review. RGB difference is not a damage mask.\n"
        "Gallery: gallery.html; numerical diagnostics: diagnostics.json; provenance: run.json\n", encoding="utf-8")
    return metrics


def run_pair(manifest_path, reference_id, source_id, out, method="sift", checkpoint=None,
             device="cpu", max_side=1024, ransac_threshold=3., confidence=.4,
             seed=42, max_canvas_pixels=50_000_000, max_canvas_side=16000,
             allow_inferred_metadata=False) -> dict:
    config = {"manifest_path": str(Path(manifest_path).resolve()), "manifest_sha256": sha256(manifest_path),
              "reference_id": str(reference_id), "source_id": str(source_id), "method": method,
              "checkpoint": str(Path(checkpoint).resolve()) if checkpoint else None, "device": device,
              "max_side": max_side, "ransac_threshold": ransac_threshold, "confidence": confidence,
              "seed": seed, "max_canvas_pixels": max_canvas_pixels, "max_canvas_side": max_canvas_side,
              "allow_inferred_metadata": allow_inferred_metadata}
    out = new_directory(out)
    record = run_record("pair_alignment", config)
    write_json(out / "run.json", record)
    start = time.monotonic()
    try:
        manifest = read_json(manifest_path)
        reference, source = select_pair(manifest, reference_id, source_id, allow_inferred_metadata)
        record["observations"] = {"reference": reference, "source": source}
        arrays = []
        for row in (reference, source):
            if sha256(row["image_path"]) != row["sha256"]:
                raise ValueError(f"Image changed since manifest: {row['image_id']}")
            rgb, opaque = load_rgb(row["image_path"])
            if sha256(row["image_path"]) != row["sha256"]:
                raise ValueError(f"Image changed while decoding: {row['image_id']}")
            if rgb.shape[:2] != (row["height"], row["width"]):
                raise ValueError("Decoded dimensions differ from manifest")
            arrays.extend([rgb, opaque])
        if method == "sift":
            matcher = SIFTMatcher()
        elif method == "loftr":
            matcher = LoFTRMatcher(checkpoint, device, confidence)
        else:
            raise ValueError(f"Unsupported method {method}")
        result = align(*arrays, matcher, max_side, ransac_threshold, seed)
        record["matcher"] = result.matches.metadata
        canvas = expanded_canvas(arrays[0].shape, arrays[2].shape, result.source_to_reference,
                                 max_canvas_pixels, max_canvas_side)
        write_json(out / "geometry.json", canvas.as_dict())
        warped = warp_pair(*arrays, canvas)
        metrics = save_diagnostics(out, warped, result, canvas)
        record["elapsed_seconds"] = time.monotonic() - start
        finish_record(out, record, "completed_needs_review")
        return metrics
    except Exception as exc:
        record["elapsed_seconds"] = time.monotonic() - start
        finish_record(out, record, "failed", f"{type(exc).__name__}: {exc}")
        raise
