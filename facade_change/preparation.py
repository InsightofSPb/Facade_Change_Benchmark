"""Reuse a completed inventory to prepare temporal candidates and group splits."""
from __future__ import annotations

import copy
import csv
import random
import re
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path

from .data import REVIEW_FIELDS, normalized_name, read_overrides
from .io import finish_record, new_directory, read_json, run_record, sha256, write_json


def validate_partitions(images: list[dict], allow_missing_views=False) -> None:
    """Reject physical-group, view, and byte-duplicate leakage across partitions."""
    owners = {key: {} for key in ("building_id", "view_id", "sha256")}
    view_buildings = defaultdict(set)
    for row in images:
        split = row.get("split")
        if split not in {"train", "val", "test"}:
            continue
        if row.get("metadata_status") != "reviewed" or not row.get("building_id"):
            raise ValueError(f"Unreviewed physical group in {split}: {row['image_id']}")
        for key, values in owners.items():
            value = row.get(key)
            if not value:
                if key == "view_id" and allow_missing_views:
                    continue
                raise ValueError(f"Image {row['image_id']} lacks {key} for a reviewed split")
            if value in values and values[value] != split:
                raise ValueError(f"Split leakage: {key}={value} belongs to {values[value]} and {split}")
            values[value] = split
        if row.get("view_id"):
            view_buildings[row["view_id"]].add(row["building_id"])
    if any(len(buildings) > 1 for buildings in view_buildings.values()):
        raise ValueError("A reviewed view_id belongs to multiple building_id values")


def possible_group_overlaps(views: list[str]) -> dict[str, list[str]]:
    """Flag overlapping terminal house-number ranges; this is not a grouping rule."""
    parsed = {}
    for view in views:
        match = re.fullmatch(r"(.+)_([0-9]+)(?:-([0-9]+))?", view)
        if match:
            low, high = int(match[2]), int(match[3] or match[2])
            parsed[view] = (match[1].casefold(), min(low, high), max(low, high))
    result = defaultdict(list)
    for first, second in combinations(sorted(parsed), 2):
        a, b = parsed[first], parsed[second]
        if a[0] == b[0] and max(a[1], b[1]) <= min(a[2], b[2]):
            result[first].append(second)
            result[second].append(first)
    return dict(result)


def _write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _crop_geometry(crop_size, crop_stride):
    if type(crop_size) is not int or crop_size <= 0:
        raise ValueError("crop_size must be a positive integer")
    if type(crop_stride) is not int or not 0 < crop_stride <= crop_size:
        raise ValueError("crop_stride must be an integer in [1, crop_size]")
    return {"crop_size": crop_size, "crop_stride": crop_stride,
            "overlap": crop_size - crop_stride,
            "origin_policy": "regular stride plus final edge-anchored origin",
            "small_image_policy": "partial crop padded to crop_size; padding is ignored",
            "scope": "single-observation semantic crops, not aligned temporal pairs"}


def _crop_dimensions(width, height):
    if any(type(value) is not int or value <= 0 for value in (width, height)):
        raise ValueError("Semantic crop dimensions must be positive integers")


def _crop_origins(length, crop_size, crop_stride):
    # Same edge-anchored grid as derived._starts, without importing RGB tools.
    last = max(0, length - crop_size)
    yield from range(0, last + 1, crop_stride)
    if last % crop_stride:
        yield last


def semantic_crop_boxes(width, height, crop_size=512, crop_stride=448):
    """Yield native-grid (x, y, width, height); small axes need ignored padding.

    No photograph is opened or resized. The final crop reaches each image edge;
    therefore its overlap can exceed the nominal ``crop_size - crop_stride``.
    """
    _crop_geometry(crop_size, crop_stride)
    _crop_dimensions(width, height)
    for y in _crop_origins(height, crop_size, crop_stride):
        for x in _crop_origins(width, crop_size, crop_stride):
            yield x, y, min(crop_size, width - x), min(crop_size, height - y)


def semantic_crop_count(width, height, crop_size=512, crop_stride=448):
    """Count the same semantic grid without materializing crops or their origins."""
    _crop_geometry(crop_size, crop_stride)
    _crop_dimensions(width, height)
    return ((max(0, width - crop_size) + crop_stride - 1) // crop_stride + 1) * (
        (max(0, height - crop_size) + crop_stride - 1) // crop_stride + 1)


def _valid_year(year):
    return type(year) is int and 1800 <= year <= 2100


def _building_series(images):
    """Count distinct reviewed years per view, never photos across different views."""
    grouped = defaultdict(list)
    for row in images:
        grouped[row["building_id"]].append(row)
    result = {}
    for building, rows in sorted(grouped.items()):
        years = defaultdict(set)
        incomplete = 0
        for row in rows:
            if row.get("view_id") and _valid_year(row.get("year")):
                years[row["view_id"]].add(row["year"])
            else:
                incomplete += 1
        length = max((len(values) for values in years.values()), default=0)
        kind = "unknown" if incomplete else "short" if length <= 2 else "medium" if length <= 7 else "long"
        result[building] = {"max_distinct_years_per_view": length, "series_class": kind,
                            "view_years": {view: sorted(values) for view, values in sorted(years.items())},
                            "observations_without_temporal_metadata": incomplete,
                            "image_count": len(rows)}
    return result


def _split_series_buildings(images, weights, fractions, fixed, seed, split_policy="series-v2",
                            crop_weights=None):
    """Constrained deterministic building moves/swaps; holdouts use medium series."""
    if any(row.get("metadata_status") != "reviewed" or not row.get("building_id")
           or row.get("image_status") != "ready" or not row.get("sha256") for row in images):
        raise ValueError(f"{split_policy} requires reviewed, ready images with building_id and SHA-256")
    series = _building_series(images)
    holdouts = [name for name in ("val", "test") if fractions[name] > 0]
    medium = {building for building, stats in series.items() if stats["series_class"] == "medium"}
    history = {building for building in medium if series[building]["max_distinct_years_per_view"] >= 5}
    movable = sorted(medium - fixed.keys())
    random.Random(seed).shuffle(movable)
    movable.sort(key=lambda group: -weights[group])
    assignments = {**{building: "train" for building in weights}, **fixed}
    active = [name for name, fraction in fractions.items() if fraction > 0]
    targets = {name: fraction * sum(weights.values()) for name, fraction in fractions.items()}
    available_history = sum(fixed.get(group) != "train" for group in history)
    history_goal = 2 if holdouts and available_history >= 2 * len(holdouts) else 1
    desired_buildings = None
    minimum_buildings = {name: 2 for name in holdouts}
    if split_policy == "facades-v3":
        minimum_buildings = {name: 7 if name == "test" else 2 for name in holdouts}
        desired_buildings = {name: (max(7, min(10, round(fractions[name] * len(weights))))
                                   if name == "test" else max(2, round(fractions[name] * len(weights))))
                             for name in holdouts}

    def score(candidate):
        medium_counts = Counter(candidate[group] for group in medium)
        history_counts = Counter(candidate[group] for group in history)
        counts = Counter()
        for group, weight in weights.items():
            counts[candidate[group]] += weight
        # Coverage precedes crop fractions. Under shortage, squared deficits spread
        # scarce buildings across holdouts; no non-medium building is substituted.
        coverage = sum(max(0, minimum_buildings[name] - medium_counts[name]) ** 2 for name in holdouts)
        result = (coverage, sum(max(0, history_goal - history_counts[name]) ** 2 for name in holdouts),
                  max(0, 1 - medium_counts["train"]))
        if desired_buildings is not None:
            excess = max(0, counts["test"] - 10) if "test" in holdouts else 0
            result = (excess,) + result + (sum((counts[name] - desired_buildings[name]) ** 2 for name in holdouts),)
        return result + (sum((counts[name] - targets[name]) ** 2 for name in active),)

    for _ in range(max(1, 6 * len(movable))):
        best_score, best = score(assignments), None
        for group in movable:
            for target in active:
                if target == assignments[group]:
                    continue
                candidate = {**assignments, group: target}
                candidate_score = score(candidate)
                if candidate_score < best_score:
                    best_score, best = candidate_score, candidate
        for first, second in combinations(movable, 2):
            if assignments[first] == assignments[second]:
                continue
            candidate = {**assignments, first: assignments[second], second: assignments[first]}
            candidate_score = score(candidate)
            if candidate_score < best_score:
                best_score, best = candidate_score, candidate
        if best is None:
            break
        assignments = best

    medium_counts = Counter(assignments[group] for group in medium)
    history_counts = Counter(assignments[group] for group in history)
    issues = []
    for name in holdouts:
        if medium_counts[name] < minimum_buildings[name]:
            issues.append(f"{name}: only {medium_counts[name]} medium-series buildings; "
                          f"requested at least {minimum_buildings[name]}")
        if history_counts[name] < history_goal:
            issues.append(f"{name}: only {history_counts[name]} buildings with 5–7 reviewed years; requested {history_goal}")
    if not medium_counts["train"]:
        issues.append("train: no medium-series building remains")
    exceptions = sorted(group for group in weights.keys() & fixed.keys()
                        if fixed[group] in holdouts and group not in medium)
    if exceptions:
        issues.append("Historical holdout owners no longer satisfy the medium-series rule; ownership stays frozen")
    if desired_buildings is not None:
        counts = Counter(assignments[group] for group in weights)
        for name, target in desired_buildings.items():
            if counts[name] != target:
                issues.append(f"{name}: {counts[name]} buildings; desired {target}; frozen owners and series rules take precedence")
    for building, stats in series.items():
        stats.update(split=assignments[building], crop_count=(crop_weights if crop_weights is not None else weights).get(building),
                     historical_assignment=building in fixed)
    return assignments, {
        "status": "insufficient" if issues else "satisfied", "issues": issues,
        "minimum_medium_buildings_per_holdout": 2,
        "minimum_medium_buildings_by_holdout": minimum_buildings,
        "desired_building_counts": desired_buildings,
        "test_building_range": [7, 10] if split_policy == "facades-v3" else None,
        "preferred_5_to_7_year_buildings_per_holdout": history_goal,
        "medium_buildings_by_split": {name: medium_counts[name] for name in fractions},
        "history_buildings_by_split": {name: history_counts[name] for name in fractions},
        "historical_policy_exceptions": exceptions,
        "length_unit": "distinct reviewed years within one view; building uses maximum view length",
        "fresh_train_only": ["short: 1–2 years", "long: 8+ years", "incomplete temporal metadata"],
        "buildings": series,
    }


def split_reviewed_buildings(images, val_fraction=.10, test_fraction=.20, seed=42,
                             fixed_assignments=None, balance_unit="images",
                             crop_size=512, crop_stride=448, split_policy="legacy"):
    """Target whole-building fractions; historical owners never move.

    ``legacy`` retains the original unconstrained allocation. ``series-v2``
    requires crop weights and reviewed physical groups: fresh short, long, or
    incompletely dated buildings stay in TRAIN; only medium series enter holdouts.
    Fractions are explicit arguments (the two-path v2 entrypoint uses .08/.12).
    ``facades-v3`` instead balances physical-building counts, aiming for 7–10 TEST
    buildings and at least two VAL buildings before the soft requested fractions.
    """
    if balance_unit not in {"images", "crops", "buildings"}:
        raise ValueError("balance_unit must be images, crops or buildings")
    if split_policy not in {"legacy", "series-v2", "facades-v3"}:
        raise ValueError("split_policy must be legacy, series-v2 or facades-v3")
    if split_policy == "series-v2" and balance_unit != "crops":
        raise ValueError("series-v2 requires balance_unit=crops")
    if split_policy == "facades-v3" and balance_unit != "buildings":
        raise ValueError("facades-v3 requires balance_unit=buildings")
    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("Split fractions must be nonnegative and leave a positive training fraction")
    crop_geometry = _crop_geometry(crop_size, crop_stride)
    image_weights = Counter(row["building_id"] for row in images)
    crop_weights = Counter()
    crop_metadata_complete = True
    for row in images:
        try:
            count = semantic_crop_count(row.get("width"), row.get("height"), crop_size, crop_stride)
        except ValueError as exc:
            if balance_unit == "crops":
                raise ValueError(f"Image {row.get('image_id')} lacks valid dimensions for crop balancing") from exc
            crop_metadata_complete = False
            continue
        crop_weights[row["building_id"]] += count
    crop_total = sum(crop_weights.values()) if crop_metadata_complete else None
    weights = crop_weights if balance_unit == "crops" else image_weights
    if balance_unit == "buildings":
        weights = Counter({group: 1 for group in image_weights})
    fractions = {"train": 1 - val_fraction - test_fraction, "val": val_fraction, "test": test_fraction}
    active = [name for name, fraction in fractions.items() if fraction > 0]
    fixed_assignments = dict(fixed_assignments or {})
    if any(not group or name not in active for group, name in fixed_assignments.items()):
        raise ValueError("Fixed building assignments must use a nonempty building_id and an active split")
    if not weights or (split_policy == "legacy" and not fixed_assignments and len(weights) < len(active)):
        raise ValueError("Too few reviewed buildings for nonempty requested splits; use --split-mode dev until reviewed")
    groups = sorted(group for group in weights if group not in fixed_assignments)
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda group: -weights[group])
    target_total = sum(weights.values())
    targets = {name: fraction * target_total for name, fraction in fractions.items()}
    counts = {name: sum(size for group, size in weights.items() if fixed_assignments.get(group) == name)
              for name in fractions}
    group_counts = {name: sum(fixed_assignments.get(group) == name for group in weights) for name in fractions}
    assignments = dict(fixed_assignments)

    def objective(candidate):
        return sum((candidate[name] - targets[name]) ** 2 for name in active)

    legacy_groups = groups if split_policy == "legacy" else []
    for i, group in enumerate(legacy_groups):
        empty = [name for name in active if not group_counts[name]]
        choices = empty if len(groups) - i == len(empty) else active
        split = min(choices, key=lambda name: objective({**counts, name: counts[name] + weights[group]}))
        assignments[group] = split
        counts[split] += weights[group]
        group_counts[split] += 1

    # Improve single-building moves and pair swaps. This bounded local search is
    # deterministic and does not claim the globally closest possible partition.
    for _ in range(max(1, 4 * len(legacy_groups))):
        best_score, best = objective(counts), None
        for group in legacy_groups:
            source = assignments[group]
            if group_counts[source] <= 1:
                continue
            for target in active:
                if target == source:
                    continue
                candidate = {**counts, source: counts[source] - weights[group], target: counts[target] + weights[group]}
                score = objective(candidate)
                if score < best_score - 1e-9:
                    best_score, best = score, (group, None, source, target, candidate)
        for first, second in combinations(legacy_groups, 2):
            source, target = assignments[first], assignments[second]
            if source == target or weights[first] == weights[second]:
                continue
            delta = weights[first] - weights[second]
            candidate = {**counts, source: counts[source] - delta, target: counts[target] + delta}
            score = objective(candidate)
            if score < best_score - 1e-9:
                best_score, best = score, (first, second, source, target, candidate)
        if best is None:
            break
        first, second, source, target, counts = best
        assignments[first] = target
        if second is None:
            group_counts[source] -= 1
            group_counts[target] += 1
        else:
            assignments[second] = source

    series_constraints = None
    if split_policy != "legacy":
        assignments, series_constraints = _split_series_buildings(
            images, weights, fractions, fixed_assignments, seed, split_policy, crop_weights)
        group_counts = {name: sum(assignments[group] == name for group in weights) for name in fractions}
    image_counts = {name: sum(size for group, size in image_weights.items() if assignments[group] == name)
                    for name in fractions}
    crop_counts = {name: sum(size for group, size in crop_weights.items() if assignments[group] == name)
                   for name in fractions} if crop_metadata_complete else {name: None for name in fractions}
    balance = {
        "split_policy": split_policy,
        "target_unit": balance_unit, "method": "seeded_largest_first_then_building_moves_and_swaps",
        "eligible_image_count": len(images), "reviewed_building_count": len(weights),
        "eligible_crop_count": crop_total, "crop_metadata_complete": crop_metadata_complete,
        "crop_geometry": crop_geometry,
        "fixed_building_count": len(weights.keys() & fixed_assignments.keys()),
        "new_building_count": len(groups),
        "absent_historical_building_count": len(fixed_assignments.keys() - weights.keys()),
        "note": (f"Whole buildings and historical assignments stay intact. {balance_unit.capitalize()} fractions "
                 "are approximate; global optimality is not guaranteed. Crop counts describe single observations, "
                 "not temporal-pair crops or annotation-filtered samples."),
        "partitions": {name: {
            "target_image_fraction": fractions[name], "target_image_count": fractions[name] * len(images),
            "image_count": image_counts[name], "image_fraction": image_counts[name] / len(images),
            "image_count_deviation": image_counts[name] - fractions[name] * len(images),
            "image_fraction_deviation": image_counts[name] / len(images) - fractions[name],
            "target_crop_fraction": fractions[name],
            "target_crop_count": fractions[name] * crop_total if crop_total is not None else None,
            "crop_count": crop_counts[name],
            "crop_fraction": crop_counts[name] / crop_total if crop_total else None,
            "crop_count_deviation": crop_counts[name] - fractions[name] * crop_total if crop_total is not None else None,
            "crop_fraction_deviation": crop_counts[name] / crop_total - fractions[name] if crop_total else None,
            "building_count": group_counts[name], "building_fraction": group_counts[name] / len(weights),
            "target_building_fraction": fractions[name], "target_building_count": fractions[name] * len(weights),
            "building_fraction_deviation": group_counts[name] / len(weights) - fractions[name],
        } for name in fractions},
    }
    if series_constraints is not None:
        balance["method"] = "series_constraints_then_seeded_building_moves_and_swaps"
        balance["series_constraints"] = series_constraints
    if split_policy == "facades-v3":
        balance["note"] += (" Physical-building counts determine this split. The 7–10 TEST-building aim and "
                            "series coverage precede requested fractions; image and crop fractions are diagnostics only.")
    return assignments, balance


def prepare_dataset(manifest_path, out, overrides=None, split_mode="dev",
                    pair_policy="first-anchor", seed=42, val_fraction=.10, test_fraction=.20,
                    assets_config=None, previous_split=None, balance_unit="images",
                    crop_size=512, crop_stride=448, split_policy="legacy") -> dict:
    """Prepare metadata without opening source photographs or rasterizing COCO.

    ``dev`` puts all usable observations in one exploratory partition. ``reviewed``
    partitions confirmed buildings; unresolved observations remain in the manifest
    with explicit exclusion reasons. ``previous_split`` freezes historical building
    ownership and the initial gold cohort while new buildings target the same
    fractions, weighted by images or recorded single-observation semantic crops.
    Image continuity uses source bytes rather than COCO numeric ids.
    ``series-v2`` additionally permits semantic-only observations without a view
    or year, requires crop balancing, and reports medium-series holdout coverage.
    Such observations cannot form temporal pairs. Legacy eligibility is unchanged.
    ``facades-v3`` uses the same series eligibility but targets building counts.
    ``first-anchor`` uses one earliest-year photo per view, breaking ties by
    normalized basename and SHA-256 without dropping other original observations.
    No crop or temporal ground truth is implied.
    """
    if split_mode not in {"dev", "reviewed"}:
        raise ValueError("split_mode must be dev or reviewed")
    if balance_unit not in {"images", "crops", "buildings"}:
        raise ValueError("balance_unit must be images, crops or buildings")
    if split_policy not in {"legacy", "series-v2", "facades-v3"}:
        raise ValueError("split_policy must be legacy, series-v2 or facades-v3")
    if split_policy == "series-v2" and (split_mode != "reviewed" or balance_unit != "crops"):
        raise ValueError("series-v2 requires split_mode=reviewed and balance_unit=crops")
    if split_policy == "facades-v3" and (split_mode != "reviewed" or balance_unit != "buildings"):
        raise ValueError("facades-v3 requires split_mode=reviewed and balance_unit=buildings")
    crop_geometry = _crop_geometry(crop_size, crop_stride)
    if previous_split and split_mode != "reviewed":
        raise ValueError("previous_split requires split_mode=reviewed; dev cannot extend a gold split")
    if pair_policy not in {"adjacent", "first-anchor", "all"}:
        raise ValueError("pair_policy must be adjacent, first-anchor or all")
    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("Split fractions must be nonnegative and leave a positive training fraction")
    manifest_path = Path(manifest_path).expanduser().resolve()
    original_hash = sha256(manifest_path)
    manifest = copy.deepcopy(read_json(manifest_path))
    images = manifest["images"]
    ids = {str(row["image_id"]) for row in images}
    if len(ids) != len(images):
        raise ValueError("Duplicate manifest image ids")
    changes = read_overrides(overrides, ids)
    previous = None
    previous_path = Path(previous_split).expanduser().resolve() if previous_split else None
    previous_hash = sha256(previous_path) if previous_path else None
    fractions = {"train": 1 - val_fraction - test_fraction, "val": val_fraction, "test": test_fraction}
    if previous_path:
        previous = read_json(previous_path)
        if previous.get("mode") != "reviewed" or previous.get("development_only") is not False:
            raise ValueError("Previous split must be a reviewed train/val/test split")
        if previous.get("split_policy", "legacy") != split_policy:
            raise ValueError("Previous split policy must match; create a new split version instead of extending")
        if not all(key in previous for key in ("gold_cohort", "historical_cohort", "fractions", "provenance")):
            raise ValueError("Previous split lacks gold cohort provenance; recreate its initial split from the original reviewed manifest without previous_split")
        if previous.get("seed") != seed or previous["fractions"] != fractions:
            raise ValueError("Previous split seed and fractions must match; historical gold targets are frozen")
        if not isinstance(previous.get("building_assignments"), dict):
            raise ValueError("Previous split lacks valid building_assignments")
        if not isinstance(previous["gold_cohort"], list) or not previous["gold_cohort"] or not isinstance(previous["historical_cohort"], list):
            raise ValueError("Previous split lacks valid gold/historical image cohorts")
        if not isinstance(previous["provenance"], dict) or not previous["provenance"].get("gold_manifest_sha256"):
            raise ValueError("Previous split lacks initial gold manifest provenance")
    config = {"manifest_path": str(manifest_path), "manifest_sha256": original_hash,
              "overrides_path": str(Path(overrides).resolve()) if overrides else None,
              "overrides_sha256": sha256(overrides) if overrides else None,
              "assets_config_path": str(Path(assets_config).resolve()) if assets_config else None,
              "assets_config_sha256": sha256(assets_config) if assets_config else None,
              "split_mode": split_mode, "split_policy": split_policy, "pair_policy": pair_policy, "seed": seed,
              "val_fraction": val_fraction, "test_fraction": test_fraction,
              "balance_unit": balance_unit, "crop_geometry": crop_geometry,
              "previous_split_path": str(previous_path) if previous_path else None,
              "previous_split_sha256": previous_hash,
              "source_images_revalidated": False}
    out = new_directory(out)
    record = run_record("dataset_preparation", config)
    write_json(out / "run.json", record)
    try:
        eligible = []
        for row in images:
            row.update(changes.get(str(row["image_id"]), {}))
            reasons = []
            if row.get("image_status") != "ready":
                reasons.append("image_" + str(row.get("image_status", "unknown")))
            if not row.get("sha256"):
                reasons.append("missing_image_hash")
            if split_mode == "reviewed":
                if row.get("metadata_status") != "reviewed":
                    reasons.append("metadata_not_reviewed")
                if not row.get("building_id"):
                    reasons.append("unknown_building")
            temporal_reasons = list(reasons)
            if not row.get("view_id"):
                temporal_reasons.append("unknown_view")
            if not _valid_year(row.get("year")):
                temporal_reasons.append("unknown_or_invalid_year")
            row.update(semantic_eligible=not reasons and row.get("metadata_status") == "reviewed"
                       and bool(row.get("building_id")), temporal_eligible=not temporal_reasons,
                       temporal_exclusion_reasons=temporal_reasons)
            if split_policy == "legacy":
                reasons = temporal_reasons
            row.update(split="excluded" if reasons else "dev", preparation_exclusion_reasons=reasons,
                       gold_member=False)
            if not reasons:
                eligible.append(row)

        assignments, split_balance = {}, None
        gold_cohort, historical_cohort, extended_cohort = [], [], []
        if split_mode == "reviewed":
            fixed = previous["building_assignments"] if previous else {}
            assignments, split_balance = split_reviewed_buildings(
                eligible, val_fraction, test_fraction, seed, fixed, balance_unit, crop_size, crop_stride, split_policy)
            for row in eligible:
                row["split"] = assignments[row["building_id"]]
            validate_partitions(images, allow_missing_views=split_policy != "legacy")
            identity_fields = ("image_id", "file_name", "sha256", "view_id", "building_id", "year", "split")
            gold_cohort = copy.deepcopy(previous["gold_cohort"]) if previous else [
                {key: row.get(key) for key in identity_fields} for row in eligible]
            historical_cohort = copy.deepcopy(previous["historical_cohort"]) if previous else copy.deepcopy(gold_cohort)
            for row in historical_cohort + gold_cohort:
                required = ("sha256", "file_name", "building_id", "split")
                if split_policy == "legacy":
                    required += ("view_id", "year")
                if any(not row.get(key) for key in required):
                    raise ValueError("Previous split has incomplete image ownership records")
                if assignments.get(row["building_id"]) != row["split"]:
                    raise ValueError("Previous split image ownership contradicts building_assignments")
            known_hashes = {}
            known_names = defaultdict(set)
            name_variants = defaultdict(set)

            def ownership_changed(old, current, allow_incomplete_record=False):
                for key in ("building_id", "view_id", "year", "split"):
                    if split_policy != "legacy" and key in {"view_id", "year"}:
                        if not old.get(key) or (allow_incomplete_record and not current.get(key)):
                            continue  # Missing chronology can be completed; confirmed values stay frozen.
                    if old.get(key) != current.get(key):
                        return True
                return False

            for row in historical_cohort + gold_cohort:
                old = known_hashes.get(row["sha256"])
                if old and ownership_changed(old, row, allow_incomplete_record=True):
                    raise ValueError("Split cohort has conflicting SHA-256 image ownership")
                known_hashes[row["sha256"]] = {**row, **{
                    key: old[key] for key in ("view_id", "year") if old and old.get(key)}}
                name = normalized_name(row["file_name"])
                known_names[name].add(row["sha256"])
                name_variants[name].add((row["file_name"], row["sha256"]))
            gold_hashes = {row["sha256"] for row in gold_cohort}
            for row in eligible:
                old = known_hashes.get(row["sha256"])
                if old and ownership_changed(old, row):
                    raise ValueError(f"Historical image ownership changed for SHA-256 {row['sha256']}; review building/view/year metadata before extending")
                name = normalized_name(row["file_name"])
                old_hashes = known_names.get(name)
                if old_hashes and row["sha256"] not in old_hashes:
                    raise ValueError(f"Original image bytes changed for {row['file_name']}; review the replacement explicitly before extending")
                name_variants[name].add((row["file_name"], row["sha256"]))
                row["gold_member"] = row["sha256"] in gold_hashes
                snapshot = {key: row.get(key) for key in identity_fields}
                extended_cohort.append({**snapshot, "gold_member": row["gold_member"]})
                if old is None:
                    historical_cohort.append(snapshot)
                    known_hashes[row["sha256"]] = snapshot
                elif split_policy != "legacy":
                    known_hashes[row["sha256"]] = snapshot
            if split_policy != "legacy":
                # Preserve the initial gold snapshot. Later historical snapshots
                # retain newly confirmed chronology so it cannot change unnoticed.
                for row in historical_cohort:
                    latest = known_hashes[row["sha256"]]
                    for key in ("view_id", "year"):
                        if not row.get(key) and latest.get(key):
                            row[key] = latest[key]
            for name, variants in name_variants.items():
                if len({raw for raw, _ in variants}) > 1 and len({digest for _, digest in variants}) > 1:
                    raise ValueError(f"Ambiguous normalized filename {name}; distinct originals must be reviewed explicitly")
            validate_partitions([{**row, "metadata_status": "reviewed"} for row in historical_cohort] + eligible,
                                allow_missing_views=split_policy != "legacy")

        by_view = defaultdict(list)
        for row in eligible:
            if row["temporal_eligible"]:
                by_view[row["view_id"]].append(row)
        pairs, excluded_pairs = [], []
        for view, observations in sorted(by_view.items()):
            by_year = defaultdict(list)
            for row in sorted(observations, key=lambda r: (r["year"], str(r["image_id"]))):
                by_year[row["year"]].append(row)
            years = sorted(by_year)
            if pair_policy == "adjacent":
                year_pairs = zip(years, years[1:])
            elif pair_policy == "first-anchor":
                year_pairs = ((years[0], year) for year in years[1:])
                anchor = min(by_year[years[0]], key=lambda row: (
                    normalized_name(Path(row["file_name"]).name), row["sha256"]))
            else:
                year_pairs = combinations(years, 2)
            for earlier, later in year_pairs:
                references = [anchor] if pair_policy == "first-anchor" else by_year[earlier]
                for reference, source in product(references, by_year[later]):
                    pair = {"pair_id": f"{reference['image_id']}-{source['image_id']}",
                            "reference_id": reference["image_id"], "source_id": source["image_id"],
                            "reference_year": earlier, "source_year": later, "view_id": view,
                            "building_id": reference.get("building_id"), "split": reference["split"],
                            "pair_policy": pair_policy,
                            "metadata_status": "reviewed" if all(r.get("metadata_status") == "reviewed"
                                                                  for r in (reference, source)) else "inferred",
                            "temporal_ground_truth": False}
                    reason = None
                    if reference.get("building_id") != source.get("building_id"):
                        reason = "conflicting_or_partial_building_metadata"
                    elif reference["split"] != source["split"]:
                        reason = "different_partitions"
                    elif reference["sha256"] == source["sha256"]:
                        reason = "identical_image_bytes"
                    if reason:
                        excluded_pairs.append({**pair, "reason": reason})
                    else:
                        pairs.append(pair)

        all_views = defaultdict(list)
        for row in images:
            if row.get("view_id"):
                all_views[row["view_id"]].append(row)
        overlaps = possible_group_overlaps(list(all_views))
        group_rows = []
        for view, rows in sorted(all_views.items()):
            buildings = sorted({r["building_id"] for r in rows if r.get("building_id")})
            group_rows.append({"view_id": view, "image_count": len(rows),
                               "image_ids": ";".join(str(r["image_id"]) for r in rows),
                               "building_ids": ";".join(buildings),
                               "all_metadata_reviewed": all(r.get("metadata_status") == "reviewed" for r in rows),
                               "possible_overlapping_views": ";".join(overlaps.get(view, [])),
                               "review_note": "Confirm physical building across all views; filename overlap is only a warning"})
        summary = {"image_count": len(images), "eligible_image_count": len(eligible),
                   "semantic_eligible_image_count": sum(row["semantic_eligible"] for row in images),
                   "temporal_eligible_image_count": sum(row["temporal_eligible"] for row in images),
                   "temporal_exclusion_reasons": dict(Counter(reason for row in images
                                                               for reason in row["temporal_exclusion_reasons"])),
                   "eligible_crop_count": split_balance["eligible_crop_count"] if split_balance else None,
                   "pair_count": len(pairs), "excluded_pair_count": len(excluded_pairs),
                   "image_splits": dict(Counter(r["split"] for r in images)),
                   "pair_splits": dict(Counter(r["split"] for r in pairs)),
                   "exclusion_reasons": dict(Counter(reason for r in images for reason in r["preparation_exclusion_reasons"])),
                   "reviewed_building_count": len({r["building_id"] for r in eligible}) if split_mode == "reviewed" else 0,
                   "historical_building_count": len(assignments),
                   "gold_image_count": len(gold_cohort),
                   "current_gold_image_count": sum(r.get("gold_member", False) for r in eligible),
                   "missing_gold_sha256": sorted({r["sha256"] for r in gold_cohort} - {r["sha256"] for r in eligible}),
                   "split_mode": split_mode, "split_policy": split_policy,
                   "split_balance": split_balance,
                   "evaluation_ready": False, "source_images_revalidated": False,
                   "temporal_ground_truth": False}
        if assets_config:
            from .assets import inventory_assets
            assets = inventory_assets(assets_config, out / "assets")
            summary["assets"] = assets["summary"]
        manifest["inventory_summary"] = manifest.pop("summary", {})
        manifest.update(schema_version=2, preparation=config, summary=summary, pairs=pairs,
                        excluded_pairs=excluded_pairs,
                        ground_truth={"coco": "single_observation_semantic_annotations",
                                      "temporal_change": "not_created",
                                      "alignment_review": "pending"})
        write_json(out / "manifest.json", manifest)
        pair_fields = ["pair_id", "reference_id", "source_id", "reference_year", "source_year",
                       "view_id", "building_id", "split", "pair_policy", "metadata_status", "temporal_ground_truth"]
        _write_csv(out / "pairs.csv", pairs, pair_fields)
        _write_csv(out / "metadata_review.csv", [{**r, "reviewed": "true" if r.get("metadata_status") == "reviewed" else "false",
                                                 "notes": r.get("metadata_notes", "")} for r in images], REVIEW_FIELDS)
        group_fields = ["view_id", "image_count", "image_ids", "building_ids", "all_metadata_reviewed",
                        "possible_overlapping_views", "review_note"]
        _write_csv(out / "group_review.csv", group_rows, group_fields)
        write_json(out / "split.json", {"schema_version": 2, "mode": split_mode, "seed": seed,
                                        "split_policy": split_policy,
                                        "fractions": fractions, "building_assignments": assignments,
                                        "balance_unit": balance_unit, "crop_geometry": crop_geometry,
                                        "gold_cohort": gold_cohort, "extended_cohort": extended_cohort,
                                        "historical_cohort": historical_cohort,
                                        "provenance": {
                                            "manifest_path": str(manifest_path), "manifest_sha256": original_hash,
                                            "overrides_path": config["overrides_path"], "overrides_sha256": config["overrides_sha256"],
                                            "previous_split_path": config["previous_split_path"], "previous_split_sha256": previous_hash,
                                            "gold_manifest_sha256": previous["provenance"]["gold_manifest_sha256"] if previous else original_hash,
                                        },
                                        "reviewed_metadata_required_for_training": True,
                                        "development_only": split_mode == "dev", "balance": split_balance,
                                        "summary": summary})
        balance_text = ""
        if split_balance:
            for name, stats in split_balance["partitions"].items():
                balance_text += (f"{name}: {stats['image_count']} images ({stats['image_fraction']:.1%}); "
                                 f"target {stats['target_image_fraction']:.1%}; "
                                 f"deviation {100 * stats['image_fraction_deviation']:+.2f} percentage points; "
                                 f"{stats['building_count']} buildings ({stats['building_fraction']:.1%}).\n")
                if stats["crop_count"] is not None:
                    balance_text += (f"  Single-observation semantic crops: {stats['crop_count']} "
                                     f"({stats['crop_fraction']:.1%}); target {stats['target_crop_fraction']:.1%}; "
                                     f"deviation {100 * stats['crop_fraction_deviation']:+.2f} percentage points.\n")
            balance_text += split_balance["note"] + "\n"
            if split_policy != "legacy":
                constraints = split_balance["series_constraints"]
                balance_text += (f"Series constraints: {constraints['status']}; "
                                 f"medium buildings: {constraints['medium_buildings_by_split']}; "
                                 f"5–7-year buildings: {constraints['history_buildings_by_split']}.\n")
                for issue in constraints["issues"]:
                    balance_text += f"  {issue}\n"
            balance_text += (f"Frozen gold: {len(gold_cohort)} images; present: {summary['current_gold_image_count']}; "
                             f"missing source hashes: {len(summary['missing_gold_sha256'])}.\n")
        (out / "summary.txt").write_text(
            f"Images: {len(images)}; eligible: {len(eligible)}; temporal candidates: {len(pairs)}\n"
            f"Split mode: {split_mode}; policy: {split_policy}; image splits: {summary['image_splits']}\n"
            f"Semantic eligible: {summary['semantic_eligible_image_count']}; "
            f"temporal eligible: {summary['temporal_eligible_image_count']}\n"
            f"Excluded observations: {summary['exclusion_reasons']}\n"
            f"Excluded pairs: {len(excluded_pairs)}; policy: {pair_policy}\n"
            "Reused inventory hashes; no source images opened or revalidated.\n"
            "COCO labels describe individual observations, not temporal change ground truth.\n"
            + balance_text
            + ("All usable data is exploratory dev-only. No train/val/test split was made.\n" if split_mode == "dev"
               else "Reviewed physical buildings are disjoint; temporal labels and alignment review remain pending.\n"),
            encoding="utf-8")
        if sha256(manifest_path) != original_hash:
            raise ValueError("Source manifest changed during preparation")
        if previous_path and sha256(previous_path) != previous_hash:
            raise ValueError("Previous split changed during preparation")
        finish_record(out, record, "completed_dev_only" if split_mode == "dev" else "completed_needs_alignment_and_label_review")
        return manifest
    except Exception as exc:
        finish_record(out, record, "failed", str(exc))
        raise
