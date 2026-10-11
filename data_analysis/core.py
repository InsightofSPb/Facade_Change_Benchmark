"""Shared calculations from the original offline complementarity analysis.

The benchmark remains responsible for input validation and evaluation policy.
No scorer, checkpoint, normalization or threshold selection is implemented here.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from facade_change.hypothesis_benchmark import _metrics_from_counts

COMPRESSION = "msdzip_mod256"
BASELINES = ("anychange", "dinov2", "geoscd", "rgb_diff", "lpips")
METHODS = (COMPRESSION, *BASELINES)
FUSION = "geoscd+msdzip_and"
SPLITS = ("val", "test")
COUNT_KEYS = ("tp_both", "tp_compression_only", "tp_baseline_only", "fn_both",
              "fp_both", "fp_compression_only", "fp_baseline_only", "tn_both")


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def pair_counts(compression_pred, baseline_pred, labels):
    if (compression_pred.dtype != bool or baseline_pred.dtype != bool
            or compression_pred.shape != labels.shape or baseline_pred.shape != labels.shape
            or labels.ndim != 2 or labels.dtype != np.uint8
            or not np.isin(labels, [0, 1, 255]).all()):
        raise ValueError("Expected same-grid boolean predictions and uint8 labels 0/1/255")
    positive, negative = labels == 1, labels == 0
    masks = (compression_pred & baseline_pred, compression_pred & ~baseline_pred,
             baseline_pred & ~compression_pred, ~compression_pred & ~baseline_pred)
    return dict(zip(COUNT_KEYS, (int(np.count_nonzero(mask & target))
                                for target in (positive, negative) for mask in masks)))


def case_identity(case):
    keys = ("case_id", "base_id", "building_id", "view_id", "split", "state",
            "scenario_id", "nuisance_kind", "hypothesis", "sham_self_paste",
            "comparable_fraction", "retained_visible_edit_fraction", "exclude_from_visible_recall")
    return {key: case.get(key) for key in keys}


def prediction_metrics(case, method, prediction, labels):
    positive, negative = labels == 1, labels == 0
    counts = (np.count_nonzero(prediction & positive), np.count_nonzero(prediction & negative),
              np.count_nonzero(~prediction & positive), np.count_nonzero(~prediction & negative))
    return {**case_identity(case), "method": method,
            **_metrics_from_counts(case, counts, int(np.count_nonzero(labels == 255)))}


def slices(rows, aggregate, splits=SPLITS):
    result = {"by_split": {part: aggregate([r for r in rows if r["split"] == part]) for part in splits}}
    for dimension in ("state", "nuisance_kind", "scenario_id", "hypothesis"):
        groups = defaultdict(list)
        for row in rows:
            groups[row["split"], row[dimension]].append(row)
        result["by_" + dimension] = [{"split": part, dimension: value, **aggregate(values)}
                                   for (part, value), values in sorted(groups.items())]
    return result
