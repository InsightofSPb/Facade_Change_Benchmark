"""Diagnose frozen compression/visual decisions without fitting a new rule."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .core import COMPRESSION, COUNT_KEYS, case_identity, ratio


RATE_KEYS = (
    "baseline_recall", "confirmed_recall", "lost_recall",
    "rejected_baseline_tp_fraction", "retained_baseline_tp_fraction",
    "added_recall", "rescued_baseline_misses", "added_precision_h1",
    "extra_h1_fpr", "extra_h0_fpr", "removed_h1_fpr", "removed_h0_fpr",
    "and_lost_recall", "and_removed_h0_fpr",
)
STAT_KEYS = ("mean", "p10", "p50", "p90", "min", "max")


def diagnostic_row(case, baseline, counts):
    """Separate confirmed/lost baseline detections and compression-only errors."""
    values = {}
    for key in COUNT_KEYS:
        value = counts[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError("Diagnostic counts must be nonnegative integers")
        values[key] = int(value)
    hypothesis = case["hypothesis"]
    if hypothesis not in {"H0", "H1"}:
        raise ValueError("Diagnostic cases must identify H0 or H1")
    positives = sum(values[key] for key in COUNT_KEYS[:4])
    negatives = sum(values[key] for key in COUNT_KEYS[4:])
    baseline_tp = values["tp_both"] + values["tp_baseline_only"]
    h1, h0 = hypothesis == "H1", hypothesis == "H0"
    recovered, added_fp = values["tp_compression_only"], values["fp_compression_only"]
    lost_recall = ratio(values["tp_baseline_only"], positives) if h1 else None
    rejected_fraction = ratio(values["tp_baseline_only"], baseline_tp) if h1 else None
    removed_h0_fpr = ratio(values["fp_baseline_only"], negatives) if h0 else None
    return {
        **case_identity(case), "baseline": baseline, "compression": COMPRESSION,
        **values, "positive_pixels": positives, "negative_pixels": negatives,
        "baseline_recall": ratio(baseline_tp, positives) if h1 else None,
        "confirmed_recall": ratio(values["tp_both"], positives) if h1 else None,
        "lost_recall": lost_recall,
        "rejected_baseline_tp_fraction": rejected_fraction,
        "retained_baseline_tp_fraction": 1. - rejected_fraction if rejected_fraction is not None else None,
        "added_recall": ratio(recovered, positives) if h1 else None,
        "rescued_baseline_misses": ratio(recovered, recovered + values["fn_both"]) if h1 else None,
        "added_precision_h1": ratio(recovered, recovered + added_fp) if h1 else None,
        "extra_h1_fpr": ratio(added_fp, negatives) if h1 else None,
        "extra_h0_fpr": ratio(added_fp, negatives) if h0 else None,
        "removed_h1_fpr": ratio(values["fp_baseline_only"], negatives) if h1 else None,
        "removed_h0_fpr": removed_h0_fpr,
        "and_lost_recall": lost_recall,
        "and_removed_h0_fpr": removed_h0_fpr,
    }


def diagnostic_macro(rows):
    """Average defined case rates within buildings, then weight buildings equally."""
    rows = list(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["building_id"]].append(row)
    buildings = []
    for building, cases in sorted(groups.items()):
        means = {}
        for key in RATE_KEYS:
            values = [row[key] for row in cases if row[key] is not None]
            means[key] = float(np.mean(values)) if values else None
        buildings.append({
            "building_id": building, "case_count": len(cases), "means": means,
            "pixel_counts_diagnostic_only": {key: sum(row[key] for row in cases) for key in COUNT_KEYS},
        })
    means, metric_building_counts = {}, {}
    for key in RATE_KEYS:
        values = [building["means"][key] for building in buildings if building["means"][key] is not None]
        means[key] = float(np.mean(values)) if values else None
        metric_building_counts[key] = len(values)
    return {
        "case_count": len(rows), "building_count": len(buildings), "means": means,
        "metric_building_counts": metric_building_counts, "buildings": buildings,
        "pixel_counts_diagnostic_only": {key: sum(row[key] for row in rows) for key in COUNT_KEYS},
    }


def _validate_labels(labels):
    if labels.ndim != 2 or labels.dtype != np.uint8 or not np.isin(labels, (0, 1, 255)).all():
        raise ValueError("Expected 2-D uint8 labels containing only 0, 1 and 255")


def positive_boundary(labels):
    """GT-positive pixels touching a known GT-negative pixel in eight neighbors."""
    _validate_labels(labels)
    height, width = labels.shape
    negative = np.pad(labels == 0, 1, mode="constant", constant_values=False)
    touching = np.zeros(labels.shape, dtype=bool)
    for row_offset in (-1, 0, 1):
        for col_offset in (-1, 0, 1):
            if row_offset or col_offset:
                touching |= negative[1 + row_offset:1 + row_offset + height,
                                     1 + col_offset:1 + col_offset + width]
    return (labels == 1) & touching


def _statistics(values):
    if values is None or not values.size:
        return dict.fromkeys(STAT_KEYS)
    values = np.asarray(values, dtype=np.float64)
    percentiles = np.percentile(values, (10, 50, 90))
    return dict(zip(STAT_KEYS, (float(values.mean()), *(float(v) for v in percentiles),
                                     float(values.min()), float(values.max()))))


def feature_rows(case, geo_pred, compression_pred, labels, scores, raw_scores,
                 reference, source, threshold):
    """Describe saved pixel groups; associations do not identify failure causes."""
    _validate_labels(labels)
    for prediction in (geo_pred, compression_pred):
        if prediction.shape != labels.shape or prediction.dtype != bool:
            raise ValueError("Feature predictions must be same-grid boolean masks")
    valid = labels != 255
    for name, values in (("scores", scores), ("raw_scores", raw_scores)):
        if values is not None and (values.shape != labels.shape or not np.isfinite(values[valid]).all()):
            raise ValueError(f"Feature {name} must be finite on evaluated pixels and match the label grid")
    if scores is None:
        raise ValueError("Feature rows require saved MSDZip scores")
    if not np.isfinite(threshold):
        raise ValueError("Feature threshold must be finite")
    for rgb in (reference, source):
        if rgb.shape != (*labels.shape, 3) or rgb.dtype != np.uint8:
            raise ValueError("Feature RGB inputs must be native uint8 RGB on the label grid")
    positive, negative = labels == 1, labels == 0
    boundary = positive_boundary(labels)
    kept_tp = positive & geo_pred & compression_pred
    rejected_tp = positive & geo_pred & ~compression_pred
    groups = {
        "kept_tp": kept_tp,
        "rejected_tp": rejected_tp,
        "removed_fp": negative & geo_pred & ~compression_pred,
        "retained_fp": negative & geo_pred & compression_pred,
        "msd_only_tp": positive & ~geo_pred & compression_pred,
        "missed_by_both": positive & ~geo_pred & ~compression_pred,
        "rejected_tp_boundary": rejected_tp & boundary,
        "rejected_tp_other": rejected_tp & ~boundary,
        "kept_tp_boundary": kept_tp & boundary,
        "kept_tp_other": kept_tp & ~boundary,
    }
    rgb_abs = np.abs(source.astype(np.int16) - reference.astype(np.int16)).mean(axis=2)
    threshold_gap = float(threshold) - scores.astype(np.float64)
    features = {"msd_score": scores, "raw_bpb": raw_scores,
                "rgb_abs_meanchannel": rgb_abs, "threshold_gap": threshold_gap}
    rows = []
    for group, mask in groups.items():
        row = {
            **case_identity(case), "group": group, "pixel_count": int(mask.sum()),
            "positive_boundary_pixel_count": int(np.count_nonzero(mask & boundary)),
            "positive_other_pixel_count": int(np.count_nonzero(mask & positive & ~boundary)),
        }
        for name, values in features.items():
            row.update({name + "_" + key: value for key, value in
                        _statistics(values[mask] if values is not None else None).items()})
        rows.append(row)
    return rows
