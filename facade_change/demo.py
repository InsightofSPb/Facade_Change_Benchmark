"""Procedural fixture with known geometry; no real facade or research evidence."""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from .data import build_manifest, REVIEW_FIELDS
from .geometry import transform_points
from .io import new_directory, read_json, write_json
from .pipeline import run_pair


def make_fixture(out: str | Path) -> Path:
    out = new_directory(out)
    (out / "images").mkdir()
    rng = np.random.default_rng(2718)
    world = Image.fromarray(rng.integers(30, 225, (384, 576, 3), dtype=np.uint8))
    draw = ImageDraw.Draw(world)
    for _ in range(130):
        x, y = rng.integers(0, 550), rng.integers(0, 360)
        radius = int(rng.integers(3, 17))
        draw.ellipse((int(x), int(y), int(x + radius), int(y + radius)),
                     fill=tuple(int(c) for c in rng.integers(0, 256, 3)))
    for y in range(10, 380, 48):
        draw.text((30, y), f"SYNTHETIC geometry {y}", fill=(255, 255, 255))
    # Two translated crops of one world, later source has new area to the right.
    reference = world.crop((0, 0, 512, 384))
    source = world.crop((64, 0, 576, 384))
    reference.save(out / "images" / "aaaa0001-synthetic_facade_2010.png")
    source.save(out / "images" / "aaaa0002-synthetic_facade_2020.png")
    images = [{"id": i, "file_name": name, "path": "/data/upload/1/" + name, "width": 512, "height": 384}
              for i, name in enumerate(("aaaa0001-synthetic_facade_2010.png", "aaaa0002-synthetic_facade_2020.png"))]
    write_json(out / "coco.json", {"images": images, "annotations": [], "categories": []})
    write_json(out / "paths.json", {"coco_json": "coco.json", "image_roots": ["images"]})
    with (out / "reviewed.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows({"image_id": i, "view_id": "synthetic_view", "building_id": "synthetic_building",
                          "year": year, "reviewed": "true", "notes": "Procedurally generated, not a real facade"}
                         for i, year in enumerate((2010, 2020)))
    write_json(out / "ground_truth_geometry.json", {"source_to_reference": [[1, 0, 64], [0, 1, 0], [0, 0, 1]],
                                                  "purpose": "Code verification only"})
    return out


def run_demo(out):
    out = new_directory(out)
    fixture = make_fixture(out / "fixture")
    build_manifest(fixture / "paths.json", out / "manifest", fixture / "reviewed.csv")
    metrics = run_pair(out / "manifest" / "manifest.json", 0, 1, out / "sift", max_side=512)
    estimated = np.array(read_json(out / "sift" / "geometry.json")["source_to_reference"])
    expected = np.array(read_json(fixture / "ground_truth_geometry.json")["source_to_reference"])
    probes = np.array([[80., 40.], [400., 40.], [400., 340.], [80., 340.]])
    max_error = float(np.linalg.norm(transform_points(probes, estimated) - transform_points(probes, expected), axis=1).max())
    result = {"fixture": "synthetic_only", "max_probe_error_native_px": max_error,
              "passed": max_error < .5, "diagnostics": metrics}
    write_json(out / "verification.json", result)
    if not result["passed"]:
        raise ValueError(f"Synthetic transform recovery failed: {max_error:.4f}px")
    return result
