"""GT area statistics for reviewing cached real pairs; no inference or file writes.

A class percentage describes changes in that class's membership within its
known union. It is not an accuracy metric or all-class change conditioned on a
class region. Unknown pixels and ignored annotation boundaries never enter its
numerator or known denominator.
"""
from __future__ import annotations

import numpy as np

THRESHOLDS = (0, 1, 5, 10, 15, 20, 30, 50)


def _categories(categories):
    if not isinstance(categories, (list, tuple)) or not 1 <= len(categories) <= 16:
        raise ValueError("Expected 1–16 COCO categories in channel order")
    ids = []
    for category in categories:
        if (not isinstance(category, dict) or type(category.get("id")) is not int
                or category["id"] < 0 or not isinstance(category.get("name"), str)):
            raise ValueError("Invalid COCO category ID/name")
        ids.append(category["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate COCO category ID")


def _percent(numerator, denominator):
    return 100. * numerator / denominator if denominator else None


def pair_statistics(arrays, categories):
    """Return global and per-class area counts on the prepared annotation support."""
    _categories(categories)
    try:
        support, labels, old, new = (arrays[name] for name in
                                    ("support", "labels", "reference_bits", "source_bits"))
    except KeyError as exc:
        raise ValueError("Prepared pair is missing annotation arrays") from exc
    if (not all(isinstance(array, np.ndarray) for array in (support, labels, old, new))
            or support.dtype != np.bool_ or support.ndim != 2 or min(support.shape) < 1
            or labels.dtype != np.uint8 or labels.shape != support.shape
            or old.dtype != np.uint16 or new.dtype != np.uint16
            or old.shape != support.shape or new.shape != support.shape
            or not np.isin(labels, [0, 1, 255]).all()):
        raise ValueError("Invalid support, change labels or uint16 class bitsets")
    limit = 1 << len(categories)
    if np.any(old.astype(np.uint32) >= limit) or np.any(new.astype(np.uint32) >= limit):
        raise ValueError("Class bitset contains a channel outside the category mapping")
    if np.any(labels[~support] != 255):
        raise ValueError("Change labels outside geometric support must be ignored")
    known = support & (labels != 255)
    if np.any(known & ((old == 0) | (new == 0))):
        raise ValueError("Unannotated pixels cannot have known change labels")
    if np.any((labels[known] == 1) != (old[known] != new[known])):
        raise ValueError("Global change labels disagree with annotation bitsets")
    total = int(labels.size)
    support_count, known_count = int(support.sum()), int(known.sum())
    changed = int((known & (labels == 1)).sum())
    classes = []
    for channel, category in enumerate(categories):
        before, after = (old & (1 << channel)) != 0, (new & (1 << channel)) != 0
        roi = known & (before | after)
        roi_count = int(roi.sum())
        gain, loss = int((known & ~before & after).sum()), int((known & before & ~after).sum())
        changed_count = gain + loss
        classes.append({"channel": channel, "class_id": category["id"], "class_name": category["name"],
                        "roi_pixels": roi_count, "changed_pixels": changed_count,
                        "gain_pixels": gain, "loss_pixels": loss, "stable_pixels": roi_count - changed_count,
                        "change_known_percent": _percent(changed_count, roi_count),
                        "change_crop_percent": _percent(changed_count, total),
                        "coverage_percent": _percent(roi_count, total)})
    return {"total_pixels": total, "support_pixels": support_count, "known_pixels": known_count,
            "changed_pixels": changed, "unchanged_pixels": known_count - changed,
            "change_known_percent": _percent(changed, known_count),
            "change_crop_percent": _percent(changed, total),
            "coverage_percent": _percent(known_count, total),
            "coverage_support_percent": _percent(known_count, support_count), "classes": classes}


def summarize(rows, categories):
    """Count ready pairs/facades passing inclusive GT-percentage cutoffs.

    For the class-specific ``known`` denominator, use that class's known union
    ROI. Both denominator modes exclude pairs without known global/class ROI,
    including the 0% cutoff. Unknown-ready pairs remain in overall coverage.
    """
    _categories(categories)
    ready = [row for row in rows if row.get("status") == "ready"]
    for row in ready:
        if not row.get("building_id") or not isinstance(row.get("stats"), dict):
            raise ValueError("Ready review rows require a facade and pair statistics")
    thresholds = []
    for channel in [None, *range(len(categories))]:
        counts = []
        for row in ready:
            stats = row["stats"]
            if channel is None:
                changed, known = stats["changed_pixels"], stats["known_pixels"]
            else:
                by_channel = {entry["channel"]: entry for entry in stats["classes"]}
                entry = by_channel[channel]
                changed, known = entry["changed_pixels"], entry["roi_pixels"]
            counts.append((row["building_id"], changed, known, stats["total_pixels"]))
        identity = {"channel": channel, "class_id": None if channel is None else categories[channel]["id"],
                    "class_name": "Все классы" if channel is None else categories[channel]["name"]}
        for denominator in ("known", "crop"):
            for threshold in THRESHOLDS:
                # Integer comparison preserves inclusive cutoffs without float rounding.
                selected = [building for building, changed, known, total in counts
                            if known > 0 and total > 0
                            and 100 * changed >= threshold * (known if denominator == "known" else total)]
                thresholds.append({**identity, "denominator": denominator, "threshold_percent": threshold,
                                   "pairs": len(selected), "facades": len(set(selected))})
    return {"ready_pairs": len(ready), "facades": len({row["building_id"] for row in ready}),
            "unknown_pairs": sum(row["stats"]["known_pixels"] == 0 for row in ready), "thresholds": thresholds}
