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


def validate_partitions(images: list[dict]) -> None:
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
                raise ValueError(f"Image {row['image_id']} lacks {key} for a reviewed split")
            if value in values and values[value] != split:
                raise ValueError(f"Split leakage: {key}={value} belongs to {values[value]} and {split}")
            values[value] = split
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


def split_reviewed_buildings(images, val_fraction=.10, test_fraction=.20, seed=42,
                             fixed_assignments=None):
    """Target image fractions using whole buildings; historical owners never move."""
    weights = Counter(row["building_id"] for row in images)
    fractions = {"train": 1 - val_fraction - test_fraction, "val": val_fraction, "test": test_fraction}
    active = [name for name, fraction in fractions.items() if fraction > 0]
    fixed_assignments = dict(fixed_assignments or {})
    if any(not group or name not in active for group, name in fixed_assignments.items()):
        raise ValueError("Fixed building assignments must use a nonempty building_id and an active split")
    if not weights or (not fixed_assignments and len(weights) < len(active)):
        raise ValueError("Too few reviewed buildings for nonempty requested splits; use --split-mode dev until reviewed")
    groups = sorted(group for group in weights if group not in fixed_assignments)
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda group: -weights[group])
    targets = {name: fraction * len(images) for name, fraction in fractions.items()}
    counts = {name: sum(size for group, size in weights.items() if fixed_assignments.get(group) == name)
              for name in fractions}
    group_counts = {name: sum(fixed_assignments.get(group) == name for group in weights) for name in fractions}
    assignments = dict(fixed_assignments)

    def objective(candidate):
        return sum((candidate[name] - targets[name]) ** 2 for name in active)

    for i, group in enumerate(groups):
        empty = [name for name in active if not group_counts[name]]
        choices = empty if len(groups) - i == len(empty) else active
        split = min(choices, key=lambda name: objective({**counts, name: counts[name] + weights[group]}))
        assignments[group] = split
        counts[split] += weights[group]
        group_counts[split] += 1

    # Improve single-building moves and pair swaps. This bounded local search is
    # deterministic and does not claim the globally closest possible partition.
    for _ in range(max(1, 4 * len(groups))):
        best_score, best = objective(counts), None
        for group in groups:
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
        for first, second in combinations(groups, 2):
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

    balance = {
        "target_unit": "images", "method": "seeded_largest_first_then_building_moves_and_swaps",
        "eligible_image_count": len(images), "reviewed_building_count": len(weights),
        "fixed_building_count": len(weights.keys() & fixed_assignments.keys()),
        "new_building_count": len(groups),
        "absent_historical_building_count": len(fixed_assignments.keys() - weights.keys()),
        "note": "Whole buildings and historical assignments stay intact. Image fractions are approximate; global optimality is not guaranteed.",
        "partitions": {name: {
            "target_image_fraction": fractions[name], "target_image_count": targets[name],
            "image_count": counts[name], "image_fraction": counts[name] / len(images),
            "image_count_deviation": counts[name] - targets[name],
            "image_fraction_deviation": counts[name] / len(images) - fractions[name],
            "building_count": group_counts[name], "building_fraction": group_counts[name] / len(weights),
        } for name in fractions},
    }
    return assignments, balance


def prepare_dataset(manifest_path, out, overrides=None, split_mode="dev",
                    pair_policy="first-anchor", seed=42, val_fraction=.10, test_fraction=.20,
                    assets_config=None, previous_split=None) -> dict:
    """Prepare metadata without opening source photographs or rasterizing COCO.

    ``dev`` puts all usable observations in one exploratory partition. ``reviewed``
    partitions confirmed buildings; unresolved observations remain in the manifest
    with explicit exclusion reasons. ``previous_split`` freezes historical building
    ownership and the initial gold cohort while new buildings target the same image
    fractions. Image continuity uses source bytes rather than COCO numeric ids.
    ``first-anchor`` uses one earliest-year photo per view, breaking ties by
    normalized basename and SHA-256 without dropping other original observations.
    No crop or temporal ground truth is implied.
    """
    if split_mode not in {"dev", "reviewed"}:
        raise ValueError("split_mode must be dev or reviewed")
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
              "split_mode": split_mode, "pair_policy": pair_policy, "seed": seed,
              "val_fraction": val_fraction, "test_fraction": test_fraction,
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
            if not row.get("view_id"):
                reasons.append("unknown_view")
            year = row.get("year")
            if isinstance(year, bool) or not isinstance(year, int) or not 1800 <= year <= 2100:
                reasons.append("unknown_or_invalid_year")
            if not row.get("sha256"):
                reasons.append("missing_image_hash")
            if split_mode == "reviewed":
                if row.get("metadata_status") != "reviewed":
                    reasons.append("metadata_not_reviewed")
                if not row.get("building_id"):
                    reasons.append("unknown_building")
            row.update(split="excluded" if reasons else "dev", preparation_exclusion_reasons=reasons,
                       gold_member=False)
            if not reasons:
                eligible.append(row)

        assignments, split_balance = {}, None
        gold_cohort, historical_cohort, extended_cohort = [], [], []
        if split_mode == "reviewed":
            fixed = previous["building_assignments"] if previous else {}
            assignments, split_balance = split_reviewed_buildings(eligible, val_fraction, test_fraction, seed, fixed)
            for row in eligible:
                row["split"] = assignments[row["building_id"]]
            validate_partitions(images)
            identity_fields = ("image_id", "file_name", "sha256", "view_id", "building_id", "year", "split")
            gold_cohort = copy.deepcopy(previous["gold_cohort"]) if previous else [
                {key: row[key] for key in identity_fields} for row in eligible]
            historical_cohort = copy.deepcopy(previous["historical_cohort"]) if previous else copy.deepcopy(gold_cohort)
            for row in historical_cohort + gold_cohort:
                if any(not row.get(key) for key in ("sha256", "file_name", "view_id", "building_id", "year", "split")):
                    raise ValueError("Previous split has incomplete image ownership records")
                if assignments.get(row["building_id"]) != row["split"]:
                    raise ValueError("Previous split image ownership contradicts building_assignments")
            known_hashes = {}
            known_names = defaultdict(set)
            name_variants = defaultdict(set)
            for row in historical_cohort + gold_cohort:
                old = known_hashes.get(row["sha256"])
                if old and any(old[key] != row[key] for key in ("building_id", "view_id", "year", "split")):
                    raise ValueError("Split cohort has conflicting SHA-256 image ownership")
                known_hashes[row["sha256"]] = row
                name = normalized_name(row["file_name"])
                known_names[name].add(row["sha256"])
                name_variants[name].add((row["file_name"], row["sha256"]))
            gold_hashes = {row["sha256"] for row in gold_cohort}
            for row in eligible:
                old = known_hashes.get(row["sha256"])
                if old and any(old[key] != row[key] for key in ("building_id", "view_id", "year", "split")):
                    raise ValueError(f"Historical image ownership changed for SHA-256 {row['sha256']}; review building/view/year metadata before extending")
                name = normalized_name(row["file_name"])
                old_hashes = known_names.get(name)
                if old_hashes and row["sha256"] not in old_hashes:
                    raise ValueError(f"Original image bytes changed for {row['file_name']}; review the replacement explicitly before extending")
                name_variants[name].add((row["file_name"], row["sha256"]))
                row["gold_member"] = row["sha256"] in gold_hashes
                snapshot = {key: row[key] for key in identity_fields}
                extended_cohort.append({**snapshot, "gold_member": row["gold_member"]})
                if old is None:
                    historical_cohort.append(snapshot)
                    known_hashes[row["sha256"]] = snapshot
            for name, variants in name_variants.items():
                if len({raw for raw, _ in variants}) > 1 and len({digest for _, digest in variants}) > 1:
                    raise ValueError(f"Ambiguous normalized filename {name}; distinct originals must be reviewed explicitly")
            validate_partitions([{**row, "metadata_status": "reviewed"} for row in historical_cohort] + eligible)

        by_view = defaultdict(list)
        for row in eligible:
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
                   "pair_count": len(pairs), "excluded_pair_count": len(excluded_pairs),
                   "image_splits": dict(Counter(r["split"] for r in images)),
                   "pair_splits": dict(Counter(r["split"] for r in pairs)),
                   "exclusion_reasons": dict(Counter(reason for r in images for reason in r["preparation_exclusion_reasons"])),
                   "reviewed_building_count": len({r["building_id"] for r in eligible}) if split_mode == "reviewed" else 0,
                   "historical_building_count": len(assignments),
                   "gold_image_count": len(gold_cohort),
                   "current_gold_image_count": sum(r.get("gold_member", False) for r in eligible),
                   "missing_gold_sha256": sorted({r["sha256"] for r in gold_cohort} - {r["sha256"] for r in eligible}),
                   "split_mode": split_mode,
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
                                        "fractions": fractions, "building_assignments": assignments,
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
            balance_text += split_balance["note"] + "\n"
            balance_text += (f"Frozen gold: {len(gold_cohort)} images; present: {summary['current_gold_image_count']}; "
                             f"missing source hashes: {len(summary['missing_gold_sha256'])}.\n")
        (out / "summary.txt").write_text(
            f"Images: {len(images)}; eligible: {len(eligible)}; temporal candidates: {len(pairs)}\n"
            f"Split mode: {split_mode}; image splits: {summary['image_splits']}\n"
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
