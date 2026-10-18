"""Deterministic prediction-only ROIs and explicit semantic fusion on native grids."""
from __future__ import annotations

import numpy as np


def regions(mask, *, min_area=16, max_regions=8, padding=16, min_size=56):
    """4-connected components, largest first; padded RGB boxes, no GT inputs.

    A run-length union avoids a Python object per pixel and needs only NumPy.
    The returned x0,y0,x1,y1 boxes remain on the original image grid.
    """
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.ndim != 2:
        raise ValueError('ROI mask must be boolean H,W')
    if (any(type(v) is not int for v in (min_area, max_regions, padding, min_size))
            or min_area < 1 or max_regions < 1 or padding < 0 or min_size < 1):
        raise ValueError('Invalid ROI bounds')
    parents, runs, previous = [], [], []

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for y, row in enumerate(mask):
        edges = np.diff(np.r_[False, row, False].astype(np.int8))
        current = []
        left = 0
        for x0, x1 in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            i = len(parents)
            parents.append(i)
            runs.append((int(x0), y, int(x1), y + 1))
            while left < len(previous) and previous[left][1] <= x0:
                left += 1
            j = left
            while j < len(previous) and previous[j][0] < x1:
                a, b = find(i), find(previous[j][2])
                parents[a] = b
                j += 1
            current.append((x0, x1, i))
        previous = current
    components = {}
    for i, (x0, y0, x1, y1) in enumerate(runs):
        key = find(i)
        item = components.setdefault(key, [x0, y0, x1, y1, 0])
        item[:4] = min(item[0], x0), min(item[1], y0), max(item[2], x1), max(item[3], y1)
        item[4] += x1 - x0
    ordered = sorted((v for v in components.values() if v[4] >= min_area),
                     key=lambda v: (-v[4], v[1], v[0]))[:max_regions]
    h, w = mask.shape
    boxes = []
    for x0, y0, x1, y1, _ in ordered:
        x0, y0, x1, y1 = max(0, x0-padding), max(0, y0-padding), min(w, x1+padding), min(h, y1+padding)
        for axis, limit in ((0, w), (1, h)):
            lo, hi = (x0, x1) if axis == 0 else (y0, y1)
            size = min(limit, max(min_size, hi-lo))
            lo = max(0, min(limit-size, (lo+hi-size)//2))
            if axis == 0:
                x0, x1 = lo, lo+size
            else:
                y0, y1 = lo, lo+size
        box = (x0, y0, x1, y1)
        if box not in boxes:
            boxes.append(box)
    return boxes


def softmax(scores, scale=1.0):
    value = np.asarray(scores, dtype=np.float32) * float(scale)
    value = np.exp(value - value.max(axis=0, keepdims=True))
    return value / value.sum(axis=0, keepdims=True)


def semantic_prediction(native, vocabulary, variant, confidence=.25):
    """Keep the actual query channels, including non-target classes for diagnosis.

    SAM queries are independent; flat fusion is winner-take-all postprocessing.
    Layered fusion allows overlap between groups and keeps SAM target instances
    independent. Stock CLIP logits are renormalized within each declared group.
    Fixed checkpoints retain their native class set (including BACKGROUND).
    """
    if variant not in ('fixed', 'domain', 'expanded_flat', 'expanded_layered'):
        raise ValueError(f'Unknown semantic variant: {variant}')
    scores = np.asarray(native['scores'], np.float32)
    names = list(native['class_names'])
    if scores.ndim != 3 or len(names) != len(scores) or len(set(names)) != len(names):
        raise ValueError('Invalid semantic channels')
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Semantic scores must be finite probabilities')
    if not 0 <= confidence <= 1:
        raise ValueError('Semantic confidence must lie in [0,1]')
    groups = {row['name']: row.get('group', 'target') for row in vocabulary}
    independent = native.get('score_kind') in {
        'independent_sigmoid', 'independent_probabilities',
        'max_instance_score_times_mask_probability',
    }
    predicted = np.zeros(scores.shape, bool)
    result_scores = scores.copy()
    if variant == 'expanded_layered':
        for group in sorted({groups.get(name, 'target') for name in names}):
            indices = [i for i, name in enumerate(names) if groups.get(name, 'target') == group]
            local = scores[indices]
            raw = native.get('extra_arrays', {}).get('raw_scores')
            if raw is not None and not independent:
                scale = native.get('metadata', {}).get('softmax_scale', 100.0)
                local = softmax(raw[indices], scale)
                result_scores[indices] = local
            if independent:
                predicted[indices] = local >= confidence
            else:
                winner = local.argmax(axis=0)
                for j, i in enumerate(indices):
                    predicted[i] = (winner == j) & (local[j] >= confidence)
    elif independent and variant in ('domain', 'fixed'):
        predicted = scores >= confidence
    else:
        winner = scores.argmax(axis=0)
        for i in range(len(names)):
            predicted[i] = (winner == i) & (scores[i] >= confidence)
    return result_scores, predicted, names


def target_union(predicted, names, categories):
    target = {c['name'] for c in categories}
    selected = [i for i, name in enumerate(names) if name in target]
    return np.any(predicted[selected], axis=0) if selected else np.zeros(predicted.shape[1:], bool)


def stitch(parts, shape, channels=None):
    """Max probabilities / union predictions in overlaps; zero outside all ROIs."""
    dimensions = tuple(shape) if channels is None else (channels, *shape)
    scores, predicted = np.zeros(dimensions, np.float32), np.zeros(dimensions, bool)
    covered = np.zeros(shape, bool)
    for box, local_scores, local_prediction in parts:
        x0, y0, x1, y1 = box
        expected = ((y1-y0, x1-x0) if channels is None else (channels, y1-y0, x1-x0))
        if (not 0 <= x0 < x1 <= shape[1] or not 0 <= y0 < y1 <= shape[0]
                or local_scores.shape != expected or local_prediction.shape != expected
                or local_prediction.dtype != bool or np.isinf(local_scores).any()):
            raise ValueError('Malformed ROI projection')
        target = (..., slice(y0, y1), slice(x0, x1))
        scores[target] = np.maximum(scores[target], np.nan_to_num(local_scores, nan=0))
        predicted[target] |= local_prediction
        covered[y0:y1, x0:x1] = True
    return scores, predicted, covered
