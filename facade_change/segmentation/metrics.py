"""Multi-label semantic metrics on annotated pixels, with equal facade weights.

The caller must evaluate each native image pixel once. Overlapping training or
inference crops are not statistical units and must be merged before this module.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def confusion_counts(
    prediction: np.ndarray, target: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """Return C x 3 int64 counts (TP, FP, FN); overlaps remain multi-label.

    ``valid`` is the intersection of annotated support, opaque image pixels,
    and the native image extent. Values outside it never enter a denominator.
    """
    prediction, target, valid = map(np.asarray, (prediction, target, valid))
    if prediction.ndim != 3 or target.shape != prediction.shape:
        raise ValueError("prediction and target must have identical C,H,W shapes")
    if valid.shape != prediction.shape[1:]:
        raise ValueError("valid must have shape H,W")
    if any(array.dtype != np.bool_ for array in (prediction, target, valid)):
        raise ValueError("prediction, target and valid must be boolean arrays")
    support = valid[None, :, :]
    axes = (1, 2)
    return np.stack(
        (
            np.sum(prediction & target & support, axis=axes, dtype=np.int64),
            np.sum(prediction & ~target & support, axis=axes, dtype=np.int64),
            np.sum(~prediction & target & support, axis=axes, dtype=np.int64),
        ),
        axis=1,
    )


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return float(sum(finite) / len(finite)) if finite else None


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _class_metrics(counts: np.ndarray, categories: list[dict]) -> list[dict]:
    result = []
    for channel, (category, class_counts) in enumerate(zip(categories, counts)):
        tp, fp, fn = map(int, class_counts)
        result.append(
            {
                "channel": channel,
                "category_id": category["id"],
                "name": category["name"],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "iou": _ratio(tp, tp + fp + fn),
                "recall": _ratio(tp, tp + fn),
                "precision": _ratio(tp, tp + fp),
            }
        )
    return result


def _summary(classes: list[dict]) -> dict:
    return {
        "miou": _mean(row["iou"] for row in classes),
        "recall": _mean(row["recall"] for row in classes),
        "precision": _mean(row["precision"] for row in classes),
        "eligible_classes": {
            metric: sum(row[metric] is not None for row in classes)
            for metric in ("iou", "recall", "precision")
        },
    }


def aggregate_metrics(
    image_records: Iterable[dict[str, Any]], categories: list[dict]
) -> dict:
    """Aggregate image counts within facades, then give facades equal weight.

    Category order is the channel order; no sorting or implicit remapping occurs.
    Each record is ``{image_id, building_id, counts: [[tp, fp, fn], ...]}``.
    A class with empty union has undefined IoU. A false positive for a class
    absent from ground truth gives IoU zero, rather than excluding that class.
    Unknown building membership is rejected instead of inventing facade IDs.
    """
    if not categories or any("id" not in item or "name" not in item for item in categories):
        raise ValueError("categories must contain id and name in model channel order")
    if len({item["id"] for item in categories}) != len(categories):
        raise ValueError("category IDs must be unique")
    grouped: dict[str, dict] = {}
    seen_images = set()
    for record in image_records:
        image_id = record["image_id"]
        if image_id in seen_images:
            raise ValueError(f"duplicate native image record: {image_id}")
        seen_images.add(image_id)
        building_id = record.get("building_id")
        if not isinstance(building_id, str) or not building_id.strip():
            raise ValueError(f"confirmed building_id required for image {image_id}")
        counts = np.asarray(record["counts"])
        if counts.shape != (len(categories), 3):
            raise ValueError("image counts must have one [TP, FP, FN] row per channel")
        if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
            raise ValueError("confusion counts must be nonnegative integers")
        if np.any(counts > np.iinfo(np.int64).max):
            raise ValueError("confusion counts exceed int64 capacity")
        group = grouped.setdefault(
            building_id,
            {"counts": np.zeros((len(categories), 3), dtype=np.int64), "images": 0},
        )
        group["counts"] += counts.astype(np.int64)
        group["images"] += 1

    per_building = []
    pooled = np.zeros((len(categories), 3), dtype=np.int64)
    for building_id, group in sorted(grouped.items()):
        pooled += group["counts"]
        classes = _class_metrics(group["counts"], categories)
        per_building.append(
            {
                "building_id": building_id,
                "image_count": group["images"],
                **_summary(classes),
                "classes": classes,
            }
        )
    per_class = []
    for channel, category in enumerate(categories):
        class_rows = [building["classes"][channel] for building in per_building]
        per_class.append(
            {
                "channel": channel,
                "category_id": category["id"],
                "name": category["name"],
                **{
                    metric: _mean(row[metric] for row in class_rows)
                    for metric in ("iou", "recall", "precision")
                },
                "eligible_buildings": {
                    metric: sum(row[metric] is not None for row in class_rows)
                    for metric in ("iou", "recall", "precision")
                },
            }
        )
    pooled_classes = _class_metrics(pooled, categories)
    return {
        "aggregation": "sum native-image counts within each facade; equal mean of facade metrics",
        "evaluation_support": "annotated opaque native pixels only; unannotated pixels ignored",
        "undefined_metric_policy": "null when denominator is zero; excluded from that metric's mean",
        "image_count": len(seen_images),
        "primary": {
            **{
                metric: _mean(building[metric] for building in per_building)
                for metric in ("miou", "recall", "precision")
            },
            "building_count": len(per_building),
            "eligible_buildings": {
                metric: sum(building[metric] is not None for building in per_building)
                for metric in ("miou", "recall", "precision")
            },
        },
        "per_class": per_class,
        "per_building": per_building,
        "pixel_pooled_secondary": {**_summary(pooled_classes), "classes": pooled_classes},
    }


def write_reports(out: str | Path, metrics: dict) -> dict[str, str]:
    """Write class/facade CSVs; the caller owns the full JSON run report."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    classes_path, buildings_path = out / "per_class.csv", out / "per_building.csv"
    class_fields = [
        "channel", "category_id", "name", "iou", "recall", "precision",
        "iou_buildings", "recall_buildings", "precision_buildings",
    ]
    with classes_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=class_fields)
        writer.writeheader()
        for row in metrics["per_class"]:
            writer.writerow({
                **{key: row[key] for key in class_fields[:6]},
                **{f"{key}_buildings": value for key, value in row["eligible_buildings"].items()},
            })
    building_fields = [
        "building_id", "image_count", "miou", "recall", "precision",
        "iou_classes", "recall_classes", "precision_classes",
    ]
    with buildings_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=building_fields)
        writer.writeheader()
        for row in metrics["per_building"]:
            writer.writerow({
                **{key: row[key] for key in building_fields[:5]},
                **{f"{key}_classes": value for key, value in row["eligible_classes"].items()},
            })
    return {"per_class": str(classes_path), "per_building": str(buildings_path)}
