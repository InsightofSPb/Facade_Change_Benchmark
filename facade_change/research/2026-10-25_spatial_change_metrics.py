"""Bounded-memory spatial merging and facade-weighted, binned PR evaluation."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..segmentation.metrics import aggregate_metrics


BINS = 1024


def binary_target(target):
    positive = np.any(target == 1, axis=0)
    known = positive | np.all(target != 255, axis=0)
    return np.where(known, positive.astype(np.uint8), 255)


def collect_histograms(data, samples, prediction, binary_only=False, native_mask=False, block_size=256):
    """Average overlapping scores before counting each view/transition pixel once.

    Native binary masks retain their decision rule and merge by union. Coverage is
    geometric, never label-dependent. Conflicting known targets fail explicitly.
    No full facade canvas or all-crop probability collection is held in RAM.
    """
    groups = defaultdict(list)
    for sample in samples:
        key = tuple(sample[k] for k in ("building_id", "view_id", "previous_year", "current_year"))
        groups[key].append(sample)
    channels = len(data.categories)
    histogram = {}
    coverage = {"transitions": len(groups), "crop_events": len(samples), "unique_geometric_pixels": 0}
    for key, rows in sorted(groups.items()):
        bins = histogram.setdefault(key[0], np.zeros((channels + 1, 2, BINS), np.int64))
        blocks = defaultdict(list)
        for row in rows:
            x, y, width, height = row["bbox_xywh"]
            for by in range(y // block_size, (y + height - 1) // block_size + 1):
                for bx in range(x // block_size, (x + width - 1) // block_size + 1):
                    blocks[bx, by].append(row)
        for (bx, by), selected in sorted(blocks.items()):
            shape = (block_size, block_size)
            score_channels = 1 if binary_only else channels
            scores = np.zeros((score_channels, *shape), np.float64)
            counts = np.zeros(shape, np.uint32)
            truth = np.full((channels, *shape), 255, np.uint8)
            for row in selected:
                values, geometric, labels = prediction(row)
                x, y, width, height = row["bbox_xywh"]
                if (values.shape != (score_channels, height, width) or geometric.shape != (height, width)
                        or labels.shape != (channels, height, width) or geometric.dtype != bool
                        or not np.isfinite(values[:, geometric]).all()
                        or np.any(values[:, geometric] < 0) or np.any(values[:, geometric] > 1)):
                    raise ValueError("Prediction shape, finite range or geometry differs from the shared grid")
                if native_mask and not np.isin(values[:, geometric], (0, 1)).all():
                    raise ValueError("Native masks must retain binary decisions")
                left, top = max(x, bx * block_size), max(y, by * block_size)
                right, bottom = min(x + width, (bx + 1) * block_size), min(y + height, (by + 1) * block_size)
                src = (slice(top - y, bottom - y), slice(left - x, right - x))
                dst = (slice(top - by * block_size, bottom - by * block_size),
                       slice(left - bx * block_size, right - bx * block_size))
                valid = geometric[src]
                value = np.where(valid[None], values[(slice(None), *src)], 0)
                region = scores[(slice(None), *dst)]
                if native_mask:
                    np.maximum(region, value, out=region)
                else:
                    region += value
                counts[dst] += valid
                target = np.where(valid[None], labels[(slice(None), *src)], 255)
                existing = truth[(slice(None), *dst)]
                conflict = (existing != 255) & (target != 255) & (existing != target)
                if conflict.any():
                    raise ValueError("Overlapping crops have contradictory known targets on the same pixel")
                np.copyto(existing, target, where=(target != 255))
            supported = counts > 0
            coverage["unique_geometric_pixels"] += int(supported.sum())
            if not native_mask:
                scores /= np.maximum(counts, 1)[None]
            targets = np.concatenate((truth, binary_target(truth)[None]))
            score_rows = [(channels, scores[0])] if binary_only else [
                *enumerate(scores), (channels, np.max(scores, axis=0))]
            for channel, value in score_rows:
                valid = (targets[channel] != 255) & supported
                quantized = np.minimum((value[valid] * (BINS - 1)).astype(np.int64), BINS - 1)
                label = targets[channel][valid]
                for positive in (0, 1):
                    bins[channel, positive] += np.bincount(quantized[label == positive], minlength=BINS)
    return histogram, coverage


def _curves(histogram):
    tp = histogram[:, 1, ::-1].cumsum(axis=1)[:, ::-1].astype(np.float64)
    fp = histogram[:, 0, ::-1].cumsum(axis=1)[:, ::-1].astype(np.float64)
    positives = histogram[:, 1].sum(axis=1)
    fn = positives[:, None] - tp
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.full_like(tp, np.nan), where=denominator != 0)
    return tp, fp, fn, f1


def calibrate(histograms, channels):
    curves = [_curves(hist)[3] for hist in histograms.values()]
    pooled = sum(histograms.values())
    thresholds, reasons = [], []
    for channel in range(channels):
        if not pooled[channel, 1].sum() or not pooled[channel, 0].sum():
            thresholds.append(0.5)
            reasons.append("VAL lacks known positive or negative pixels; fixed 0.5")
            continue
        values = np.stack([curve[channel] for curve in curves])
        count = np.isfinite(values).sum(axis=0)
        means = np.divide(np.nansum(values, axis=0), count,
                          out=np.full(BINS, -1., dtype=np.float64), where=count > 0)
        best = np.flatnonzero(np.isclose(means, means.max(), rtol=0, atol=1e-12))[-1]
        thresholds.append(float(best / (BINS - 1)))
        reasons.append("maximum equal-building F1 on VAL; highest threshold wins ties")
    return thresholds, reasons


def average_precision(histogram):
    positives = histogram[1].sum()
    if not positives:
        return None
    tp = histogram[1, ::-1].cumsum()
    fp = histogram[0, ::-1].cumsum()
    precision = np.divide(tp, tp + fp, out=np.zeros(BINS, float), where=tp + fp > 0)
    return float(np.sum(precision * histogram[1, ::-1]) / positives)


def _report(histograms, categories, thresholds, channel_indices, native_mask):
    records = []
    for building, hist in sorted(histograms.items()):
        tp, fp, fn, _ = _curves(hist)
        counts = []
        for channel, threshold in zip(channel_indices, thresholds):
            index = int(np.ceil(threshold * (BINS - 1) - 1e-9))
            index = min(max(index, 0), BINS - 1)
            counts.append([int(tp[channel, index]), int(fp[channel, index]), int(fn[channel, index])])
        records.append({"image_id": building, "building_id": building, "counts": counts})
    report = aggregate_metrics(records, categories)
    report["evaluation_support"] = "derived per-class annotation support intersected with selected pair geometry"
    report["aggregation"] = "merge crop overlaps per view/transition; sum counts within building; equal building weights"
    report["score_bins"] = BINS
    report["auprc_definition"] = ("unavailable for native binary masks" if native_mask else
                                "step average precision after uniform 1024-bin score quantization; null without positives")
    for row in report["per_building"]:
        hist = histograms[row["building_id"]]
        for values, channel in zip(row["classes"], channel_indices):
            values["known_pixels"] = int(hist[channel].sum())
            values["positive_pixels"] = int(hist[channel, 1].sum())
            values["auprc_binned"] = None if native_mask else average_precision(hist[channel])
    for index, channel in enumerate(channel_indices):
        available = [r["classes"][index]["auprc_binned"] for r in report["per_building"]
                     if r["classes"][index]["auprc_binned"] is not None]
        report["per_class"][index]["auprc_binned"] = float(np.mean(available)) if available else None
        report["per_class"][index]["auprc_buildings"] = len(available)
    ap = [r["auprc_binned"] for r in report["per_class"] if r["auprc_binned"] is not None]
    report["primary"]["auprc_binned"] = float(np.mean(ap)) if ap else None
    return report


def report(histograms, categories, thresholds, binary_only=False, native_mask=False):
    channels = len(categories)
    result = {"binary": _report(histograms, [{"id": "any_target_change", "name": "ANY_TARGET_CHANGE"}],
                               [thresholds[-1]], [channels], native_mask)}
    if not binary_only:
        result["classes"] = _report(histograms, categories, thresholds[:-1], list(range(channels)), native_mask)
    result["thresholds"] = thresholds
    return result


def paired_bootstrap(reports, replicates=500, seed=42):
    if "pair" not in reports or replicates < 1:
        return {}
    base = {r["building_id"]: r["f1"] for r in reports["pair"]["classes"]["per_building"]}
    result = {}
    for name, value in reports.items():
        if name == "pair" or "classes" not in value:
            continue
        other = {r["building_id"]: r["f1"] for r in value["classes"]["per_building"]}
        differences = [other[k] - base[k] for k in sorted(base.keys() & other.keys())
                       if other[k] is not None and base[k] is not None]
        if not differences:
            continue
        rng = np.random.default_rng(seed)
        means = [float(np.mean(rng.choice(differences, len(differences), replace=True))) for _ in range(replicates)]
        result[name] = {"delta_macro_building_f1": float(np.mean(differences)),
                        "ci95": np.quantile(means, [.025, .975]).tolist(),
                        "buildings": len(differences), "replicates": replicates,
                        "unit": "building; conditional on fitted checkpoints, not training-seed uncertainty"}
    return result
