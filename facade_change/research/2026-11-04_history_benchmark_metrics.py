"""Shared spatial support, overlap merging and class-conditioned building metrics."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


BINS = 2049
METRICS = ("f1", "iou", "precision", "recall", "fpr", "ap_binned")


def slices(data):
    return [("overall", "ANY_CHANGE", list(range(len(data.categories))))] + [
        ("class", row["name"], [i]) for i, row in enumerate(data.categories)] + [
        ("group", group["name"], [i for i, row in enumerate(data.categories)
                                  if row["name"] in group["categories"]]) for group in data.groups]


def slice_targets(data, arrays):
    """Class/group positives are membership changes; negatives are stable annotated ROI."""
    target, layers = arrays["target"], arrays["class_targets"]
    result = [target]
    for _, _, channels in slices(data)[1:]:
        positive = np.any(layers[channels] == 1, axis=0)
        bits = sum(1 << i for i in channels)
        roi = ((arrays["before"] | arrays["after"]) & bits) != 0
        labels = np.full(target.shape, 255, np.uint8)
        labels[(target == 0) & roi] = 0
        labels[positive] = 1
        result.append(labels)
    return np.stack(result)


def collect(data, samples, prediction, native=False, block_size=256, fusion=None):
    """Merge each view/year-transition pixel once, with bounded canvas memory.

    The histogram permits exact strict-threshold counts on a fixed 1/2048 grid.
    Only AP is approximated by score binning. Raw float maps remain available.
    """
    grouped = defaultdict(list)
    for sample in samples:
        grouped[tuple(sample[k] for k in ("building_id", "view_id", "previous_year", "current_year"))].append(sample)
    count_slices = len(slices(data))
    flags = [native] if fusion is None else [row["native"] for row in fusion]
    histograms = {}
    coverage = dict(crop_candidates=len(samples), crop_candidates_with_known_pixels=0,
                    spatial_transitions=len(grouped), geometric_pixels=0, known_pixels=0,
                    conflicting_overlap_pixels=0)
    coverage["crop_candidates_with_known_pixels"] = sum(s["known_pixels"] > 0 for s in samples)
    for key, rows in sorted(grouped.items()):
        histogram = histograms.setdefault(key[0], np.zeros((count_slices, 2, BINS), np.int64))
        blocks = defaultdict(list)
        for row in rows:
            x, y, width, height = row["bbox_xywh"]
            for by in range(y // block_size, (y + height - 1) // block_size + 1):
                for bx in range(x // block_size, (x + width - 1) // block_size + 1):
                    blocks[bx, by].append(row)
        for (bx, by), selected in sorted(blocks.items()):
            scores = np.zeros((len(flags), block_size, block_size), np.float64)
            counts = np.zeros(scores.shape[1:], np.uint32)
            truth = np.full((count_slices, *counts.shape), 255, np.uint8)
            conflict = np.zeros(truth.shape, bool)
            for row in selected:
                values, available = prediction(row)
                if fusion is None:
                    values = values[None]
                arrays = data.target(row["event_id"])
                common = arrays["support"]
                if (values.shape != (len(flags), *common.shape)
                        or available.dtype != bool or available.shape != common.shape
                        or np.any(common & ~available) or not np.isfinite(values[:, common]).all()
                        or np.any(values[:, common] < 0) or np.any(values[:, common] > 1)):
                    raise ValueError("A method/mode lacks scores on the fixed common evaluation support")
                for channel, is_native in enumerate(flags):
                    if is_native and not np.isin(values[channel, common], (0, 1)).all():
                        raise ValueError("Native mask was converted to continuous scores")
                x, y, width, height = row["bbox_xywh"]
                left, top = max(x, bx * block_size), max(y, by * block_size)
                right, bottom = min(x + width, (bx + 1) * block_size), min(y + height, (by + 1) * block_size)
                src = (slice(top - y, bottom - y), slice(left - x, right - x))
                dst = (slice(top - by * block_size, bottom - by * block_size),
                       slice(left - bx * block_size, right - bx * block_size))
                valid = common[src]
                for channel, is_native in enumerate(flags):
                    value = np.where(valid, values[(channel, *src)], 0)
                    destination = scores[(channel, *dst)]
                    if is_native:
                        np.maximum(destination, value, out=destination)
                    else:
                        destination += value
                counts[dst] += valid
                labels = slice_targets(data, arrays)
                target = np.where(valid[None], labels[(slice(None), *src)], 255)
                existing = truth[(slice(None), *dst)]
                ambiguous = conflict[(slice(None), *dst)]
                ambiguous |= (existing != 255) & (target != 255) & (existing != target)
                np.copyto(existing, target, where=(target != 255) & ~ambiguous)
                existing[ambiguous] = 255
            geometric = counts > 0
            for channel, is_native in enumerate(flags):
                if not is_native:
                    scores[channel] /= np.maximum(counts, 1)
            merged = (np.all(scores > np.array([r["threshold"] for r in fusion])[:, None, None], axis=0)
                      if fusion is not None else scores[0])
            coverage["geometric_pixels"] += int(geometric.sum())
            coverage["known_pixels"] += int(((truth[0] != 255) & geometric).sum())
            coverage["conflicting_overlap_pixels"] += int((conflict[0] & geometric).sum())
            for channel, target in enumerate(truth):
                known = (target != 255) & geometric
                quantized = np.ceil(merged[known] * (BINS - 1)).astype(np.int64)
                for label in (0, 1):
                    histogram[channel, label] += np.bincount(quantized[target[known] == label], minlength=BINS)
    coverage["known_fraction"] = (coverage["known_pixels"] / coverage["geometric_pixels"]
                                  if coverage["geometric_pixels"] else None)
    return histograms, coverage


def curves(histogram):
    cumulative = histogram[..., ::-1].cumsum(axis=-1)[..., ::-1]
    strict = np.concatenate((cumulative[..., 1:], np.zeros((*cumulative.shape[:-1], 1), np.int64)), axis=-1)
    fp, tp = strict[..., 0, :].astype(float), strict[..., 1, :].astype(float)
    negative, positive = histogram[..., 0, :].sum(axis=-1), histogram[..., 1, :].sum(axis=-1)
    fn = positive[..., None] - tp
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.full_like(tp, np.nan), where=denominator > 0)
    fpr = np.divide(fp, negative[..., None], out=np.full_like(fp, np.nan), where=negative[..., None] > 0)
    return tp, fp, fn, f1, fpr


def calibrate(histograms):
    """One threshold for all classes/groups; equal-building overall F1 on full VAL."""
    if not histograms:
        raise ValueError("No calibration buildings")
    pooled = sum(histograms.values())[0]
    if not pooled[0].sum() or not pooled[1].sum():
        raise ValueError("Full VAL needs known changed and stable pixels; no arbitrary threshold fallback")
    values = [curves(hist[0:1]) for hist in histograms.values()]
    means = []
    for index in (3, 4):
        rows = np.stack([v[index][0] for v in values])
        count = np.isfinite(rows).sum(axis=0)
        means.append(np.divide(np.nansum(rows, axis=0), count, out=np.full(BINS, np.nan), where=count > 0))
    f1, fpr = means
    best = np.flatnonzero(np.isclose(f1, np.nanmax(f1), atol=1e-12, rtol=0))
    best = best[np.isclose(fpr[best], np.nanmin(fpr[best]), atol=1e-12, rtol=0)]
    index = int(best[-1])
    return dict(threshold=index / (BINS - 1), selection_split="val",
        criterion="maximum building-macro overall F1; minimum FPR; highest threshold",
        prediction_rule="score > threshold", grid_intervals=BINS - 1,
        buildings=len(histograms), known_pixels=int(pooled.sum()),
        calibration_f1=float(f1[index]), calibration_fpr=float(fpr[index]))


def metrics(histogram, threshold, native=False):
    scaled = threshold * (BINS - 1)
    if not np.isclose(scaled, round(scaled), atol=1e-10, rtol=0):
        raise ValueError("Reporting threshold must belong to the recorded calibration grid")
    index = int(round(scaled))
    negative, positive = (int(histogram[i].sum()) for i in (0, 1))
    fp, tp = (int(histogram[i, index + 1:].sum()) for i in (0, 1))
    fn, tn = positive - tp, negative - fp
    def ratio(a, b):
        return a / b if b else None
    ap = None
    if positive and not native:
        cumulative = histogram[:, ::-1].cumsum(axis=1)
        precision = np.divide(cumulative[1], cumulative.sum(axis=0),
                              out=np.zeros(BINS, float), where=cumulative.sum(axis=0) > 0)
        ap = float(np.sum(precision * histogram[1, ::-1]) / positive)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, positive_pixels=positive, negative_pixels=negative,
        known_pixels=positive + negative, f1=ratio(2 * tp, 2 * tp + fp + fn),
        iou=ratio(tp, tp + fp + fn), precision=ratio(tp, tp + fp), recall=ratio(tp, positive),
        fpr=ratio(fp, negative), ap_binned=ap)


def report(data, histograms, threshold, native=False):
    per_building, macro = [], []
    for building, histogram in sorted(histograms.items()):
        for i, (kind, name, _) in enumerate(slices(data)):
            per_building.append(dict(building_id=building, slice_type=kind, slice=name,
                                     **metrics(histogram[i], threshold, native)))
    for kind, name, _ in slices(data):
        rows = [r for r in per_building if r["slice_type"] == kind and r["slice"] == name]
        row = dict(slice_type=kind, slice=name, buildings=sum(r["known_pixels"] > 0 for r in rows),
                   positive_buildings=sum(r["positive_pixels"] > 0 for r in rows))
        for key in ("tp", "fp", "fn", "tn", "known_pixels", "positive_pixels", "negative_pixels"):
            row[key] = sum(r[key] for r in rows)
        for key in METRICS:
            values = [r[key] for r in rows if r[key] is not None]
            row[key] = float(np.mean(values)) if values else None
            row[key + "_buildings"] = len(values)
        macro.append(row)
    return dict(per_building=per_building, macro=macro, threshold=threshold,
        aggregation="merge view/year-transition overlaps; sum counts within building; equal building metric means",
        class_interpretation="localisation conditioned on annotated class; not predicted class recognition",
        ap_definition=f"average precision after {BINS}-bin score quantization; unavailable for native masks")
