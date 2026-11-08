"""Shared COCO projection and annotation-derived targets for pairs and histories.

RGB and geometric support remain independent from semantic annotation coverage.
Unannotated pixels never become healthy-background or no-change labels.
"""
from __future__ import annotations

import base64
from collections import defaultdict
import io
from pathlib import Path

import numpy as np
from PIL import Image

from .io import read_json
from scripts.real_eval.real_gt_metrics import derive_labels


SCOPE = "selected target-class differences on common annotated support; outside unknown"


PALETTE = np.array([[240, 96, 84], [67, 169, 145], [98, 135, 220], [233, 177, 69],
                    [166, 110, 204], [80, 187, 204], [228, 119, 178], [169, 171, 68],
                    [210, 136, 71], [107, 155, 109], [142, 147, 163]], dtype=np.uint8)


def resolve_path(value, parent):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else parent / path).resolve()


def project_masks(packed, matrix, bbox, support):
    """Use the RGB crop's source-to-anchor H and rectangle; discrete nearest sampling."""
    import cv2
    x, y, width, height = bbox
    matrix = np.asarray(matrix, dtype=np.float64)
    if (packed.dtype != np.uint16 or packed.ndim != 2 or matrix.shape != (3, 3)
            or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12
            or support.dtype != bool or support.shape != (height, width)):
        raise ValueError("Invalid annotation raster, geometry, rectangle or crop support")
    shift = np.array([[1., 0, -x], [0, 1., -y], [0, 0, 1.]])
    result = cv2.warpPerspective(packed, shift @ matrix, (width, height), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    result[~support] = 0
    return result


def year_consensus(observations):
    """All alternatives must annotate and agree; disagreement is unknown, never a new date."""
    first = observations[0]
    support = np.logical_and.reduce([row["support"] for row in observations])
    known = support.copy()
    for row in observations:
        known &= (row["packed"] != 0) & (row["packed"] == first["packed"])
    packed = np.where(known, first["packed"], 0).astype(np.uint16)
    return {"packed": packed, "support": support,
            "possible_targets": np.bitwise_or.reduce([row["packed"] for row in observations])}


def target_difference(before, after, channels, radius):
    """Adapt the existing annotation-change evaluator to selected target channels."""
    support = before["support"] & after["support"]
    old, new = before["packed"], after["packed"]
    layers = []
    for channel in channels:
        # Bit 0 retains known annotation support even when this target is absent.
        # Bit 1 represents the selected target. Original bitsets are never collapsed.
        a = (old != 0).astype(np.uint16) | (((old & (1 << channel)) != 0).astype(np.uint16) << 1)
        b = (new != 0).astype(np.uint16) | (((new & (1 << channel)) != 0).astype(np.uint16) << 1)
        layers.append(derive_labels(a, b, support, 2, radius))
    stack = np.stack(layers)
    target = np.full(support.shape, 255, dtype=np.uint8)
    target[np.all(stack == 0, axis=0)] = 0
    target[np.any(stack == 1, axis=0)] = 1
    known = support & (old != 0) & (new != 0)
    selected_bits = sum(1 << channel for channel in channels)
    raw_changed = ((old & selected_bits) != (new & selected_bits)) & known
    ambiguous = raw_changed & (target == 255)
    return target, stack, support, ambiguous


def event_target(before, after, channels, radius=1, min_known_pixels=64,
                 min_known_fraction=.001, min_changed_pixels=8):
    target, layers, support, ambiguous = target_difference(before, after, channels, radius)
    known = int((target != 255).sum())
    changed = int((target == 1).sum())
    geometric = int(support.sum())
    fraction = known / geometric if geometric else 0.0
    selected_bits = sum(1 << channel for channel in channels)
    possible = before.get("possible_targets", before["packed"]) | after.get("possible_targets", after["packed"])
    uncomparable = int(((target == 255) & ((possible & selected_bits) != 0)).sum())
    result = {"label": "", "review_status": "unreviewed", "comparable": "unknown",
              "known_pixels": known, "changed_pixels": changed,
              "stable_pixels": int((target == 0).sum()), "unknown_pixels": int((target == 255).sum()),
              "known_fraction": fraction, "boundary_ambiguous_pixels": int(ambiguous.sum()),
              "uncomparable_target_pixels": uncomparable,
              "target_scope": SCOPE, "label_kind": "annotation_derived",
              "event_type": "", "reason": ""}
    if not known:
        result["reason"] = "no_common_annotation_support"
    elif known < min_known_pixels or fraction < min_known_fraction:
        result["reason"] = "insufficient_common_annotation_support"
    elif changed >= min_changed_pixels:
        result.update(label=1, review_status="derived", comparable="annotation_support_only",
                      event_type="annotation_change", reason="target_difference_on_annotated_support")
    elif changed or ambiguous.any():
        result["reason"] = "tiny_or_boundary_ambiguous_difference"
    elif uncomparable:
        result["reason"] = "target_annotations_without_comparable_support"
    else:
        result.update(label=0, review_status="derived", comparable="annotation_support_only",
                      event_type="annotation_stable", reason="stable_on_annotated_support_only")
    return result, target, layers


def _thumbnail(array):
    image = Image.fromarray(array)
    image.thumbnail((128, 128))
    stream = io.BytesIO()
    image.save(stream, "PNG")
    return '<img loading="lazy" src="data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode() + '">'


def mask_rgb(packed, categories):
    colors = np.full((*packed.shape, 3), 115, dtype=np.uint8)
    # Blend overlaps for display only; stored bitsets preserve every label.
    total, count = np.zeros_like(colors, dtype=np.float32), np.zeros(packed.shape, dtype=np.float32)
    for i in range(len(categories)):
        mask = (packed & (1 << i)) != 0
        total[mask] += PALETTE[i % len(PALETTE)]
        count[mask] += 1
    colors[count > 0] = (total[count > 0] / count[count > 0, None]).astype(np.uint8)
    return colors


def load_annotation_source(manifest_path, fingerprints):
    """Read exact reviewed COCO bytes and retain all multilabel channels."""
    fp = fingerprints
    manifest_path = Path(manifest_path).expanduser().resolve()
    fp.add(manifest_path)
    manifest = read_json(manifest_path)
    source = manifest.get("source", {})
    if not source.get("coco_json") or not source.get("coco_sha256"):
        raise ValueError("Annotation manifest needs source.coco_json and source.coco_sha256; "
                         "use the reviewed temporal/semantic manifest, not a bare COCO file")
    coco_path = resolve_path(source["coco_json"], manifest_path.parent)
    fp.add(coco_path, source["coco_sha256"])
    coco = read_json(coco_path)
    categories = sorted(coco["categories"], key=lambda c: c["id"])
    if (not 1 <= len(categories) <= 16 or len({c["id"] for c in categories}) != len(categories)
            or len({c["name"] for c in categories}) != len(categories)
            or any(type(c["id"]) is not int or c["id"] < 0 for c in categories)):
        raise ValueError("Expected 1–16 uniquely named categories with nonnegative integer IDs")
    if {(c["id"], c["name"]) for c in manifest["categories"]} != {(c["id"], c["name"]) for c in categories}:
        raise ValueError("COCO categories differ from the annotation manifest")
    channel_map = {c["id"]: i for i, c in enumerate(categories)}
    coco_images = {str(row["id"]): row for row in coco["images"]}
    if len(coco_images) != len(coco["images"]):
        raise ValueError("Duplicate COCO image ID")
    by_hash, annotations, ids = {}, defaultdict(list), set()
    for row in manifest["images"]:
        if not row.get("sha256"):
            continue
        if row["sha256"] in by_hash:
            raise ValueError("Ambiguous duplicate image SHA in annotation manifest")
        by_hash[row["sha256"]] = row
    for annotation in coco.get("annotations", []):
        if annotation["id"] in ids:
            raise ValueError("Duplicate COCO annotation ID")
        ids.add(annotation["id"])
        if (str(annotation["image_id"]) not in coco_images
                or annotation["category_id"] not in channel_map):
            raise ValueError("Annotation references an unknown image or category")
        annotations[str(annotation["image_id"])].append(annotation)
    return {"manifest_path": manifest_path, "coco_path": coco_path, "categories": categories,
            "channel_map": channel_map, "images": by_hash, "coco_images": coco_images,
            "annotations": annotations}
