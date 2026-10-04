"""Frozen, score-independent case selection for a small H0/H1 comparison."""
from __future__ import annotations

import hashlib
import json
import copy
import re
from collections import defaultdict
from pathlib import Path


QUICK_STATES = ("unchanged", "crack", "paint_patch")
QUICK_FAMILIES = ("shadow", "exposure", "contrast", "white_balance", "blur", "occlusion")

INPUT_HASH_KEYS = frozenset(("parent_run", "parent_summary", "parent_index", "split", "config"))


def validate_input_hashes(actual, expected):
    """Require the entire inherited dataset fingerprint, not just RGB identity."""
    for hashes in (actual, expected):
        if not isinstance(hashes, dict) or set(hashes) != INPUT_HASH_KEYS:
            raise ValueError("Selection requires all five parent dataset input hashes")
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
               for value in hashes.values()):
            raise ValueError("Selection input hashes must be SHA256 values")
    if actual != expected:
        raise ValueError("Selection belongs to a different parent dataset fingerprint")


def replay_selection(bases, cases, input_sha256, selection_path):
    """Replay a recorded selection exactly, with no new sampling or score access.

    Returned rows always come from the currently verified parent index. The
    selection's base records are identity assertions, never replacement rows.
    The five hashes bind scenario, support and GT artifacts to the same parent.
    """
    selection_path = Path(selection_path).expanduser().resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(selection, dict) or selection.get("schema_version") != 1:
        raise ValueError("Unsupported saved selection schema")
    validate_input_hashes(input_sha256, selection.get("input_sha256"))
    bases, cases = list(bases), list(cases)
    # Also validate unselected bases: replay must not hide facade-level leakage.
    _base_candidates(bases, selection.get("seed", 42))
    by_base = {row["base_id"]: row for row in bases}
    by_case = {}
    for case in cases:
        identifier = case["case_id"]
        if identifier in by_case or case["base_id"] not in by_base:
            raise ValueError("Saved selection requires unique cases with known bases")
        by_case[identifier] = case
    saved_bases, identifiers = selection.get("bases"), selection.get("case_ids")
    if not isinstance(saved_bases, list) or not saved_bases or not isinstance(identifiers, list) or not identifiers:
        raise ValueError("Saved selection must contain bases and case IDs")
    if any(not isinstance(value, str) for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Saved selection duplicates or invalidates case IDs")
    selected_bases, seen = [], set()
    for recorded in saved_bases:
        identifier = recorded.get("base_id")
        if identifier in seen or identifier not in by_base:
            raise ValueError("Saved selection duplicates or refers to an unknown base")
        base = by_base[identifier]
        if recorded != base:
            raise ValueError("Saved base identity disagrees with its verified parent index")
        seen.add(identifier)
        selected_bases.append(base)
    splits = [row["split"] for row in selected_bases]
    priority = {"val": 0, "test": 1, "train": 2}
    if not {"val", "test"}.issubset(splits) or splits != sorted(splits, key=priority.get):
        raise ValueError("Saved selection must finish validation before test")
    selected_cases, grouped = [], defaultdict(list)
    for identifier in identifiers:
        case = by_case.get(identifier)
        if case is None or case["base_id"] not in seen:
            raise ValueError("Saved selection refers to an unknown or unselected case")
        base = by_base[case["base_id"]]
        if any(case.get(key) != base.get(key) for key in ("building_id", "split", "view_id")):
            raise ValueError("Saved case identity disagrees with its verified base")
        if "parent_dataset_crop_id" in case and case["parent_dataset_crop_id"] != base.get("dataset_crop_id"):
            raise ValueError("Saved case belongs to a different parent crop")
        expected_hypothesis = "H1" if case["state"] in {"crack", "paint_patch"} else "H0"
        if case.get("hypothesis") != expected_hypothesis or bool(case.get("sham_self_paste")) != (case["state"] == "self_paste"):
            raise ValueError("Saved case state/hypothesis/control metadata disagrees")
        spec = case.get("scenario")
        if not isinstance(spec, dict) or spec.get("id") != case["scenario_id"] or spec.get("kind") != case["nuisance_kind"]:
            raise ValueError("Saved case scenario metadata disagrees")
        if "reference_rgb" in case and case["reference_rgb"] != base["path"] + "/reference_rgb.png":
            raise ValueError("Saved case reference does not belong to its base")
        if "reference_support" in case and case["reference_support"] != base["path"] + "/reference_support.png":
            raise ValueError("Saved case support does not belong to its base")
        selected_cases.append(case)
        grouped[case["base_id"]].append(case)
    if [row["case_id"] for base in selected_bases for row in grouped[base["base_id"]]] != identifiers:
        raise ValueError("Saved case ordering disagrees with saved base ordering")
    if any(not grouped[base["base_id"]] for base in selected_bases):
        raise ValueError("Saved selection contains a base without cases")
    if selection.get("selected_base_count", len(selected_bases)) != len(selected_bases) or selection.get("selected_case_count", len(selected_cases)) != len(selected_cases):
        raise ValueError("Saved selection counts disagree")
    if selection.get("mode") == "quick":
        quick = selection.get("quick_selection")
        if not isinstance(quick, dict):
            raise ValueError("Saved quick selection requires its frozen per-base scenario metadata")
        frozen = quick.get("bases", [])
        if [row.get("base_id") for row in frozen] != [row["base_id"] for row in selected_bases]:
            raise ValueError("Saved quick-selection base order disagrees")
        for base, spec in zip(selected_bases, frozen):
            rows = grouped[base["base_id"]]
            if spec.get("case_ids") != [row["case_id"] for row in rows]:
                raise ValueError("Saved quick-selection case IDs disagree")
            families, variants = _eligible_scenarios(base, rows)
            scenarios = spec.get("selected_scenario_ids", [])
            if len(rows) != 6 or len(scenarios) != 2 or set(variants) != set(scenarios) or len(families) != 2:
                raise ValueError("Saved quick selection requires two shared conditions and six cases per base")
            if set(spec.get("selected_families", [])) != set(families):
                raise ValueError("Saved quick-selection families disagree")
        if len({row["building_id"] for row in selected_bases}) != len(selected_bases) or "train" in splits:
            raise ValueError("Saved quick selection requires distinct validation/test buildings")
    subset = copy.deepcopy(selection)
    subset.update(replayed_selection_path=str(selection_path),
                  replayed_selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest())
    return selected_bases, selected_cases, subset


def _rank(seed, *parts):
    value = json.dumps([seed, *parts], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _base_candidates(bases, seed):
    by_building, identifiers, ownership = defaultdict(list), set(), {}
    for base in bases:
        identifier, building, split = base["base_id"], base["building_id"], base["split"]
        if identifier in identifiers:
            raise ValueError("Quick selection requires unique base IDs")
        identifiers.add(identifier)
        if not building or split not in {"train", "val", "test"}:
            raise ValueError("Quick selection requires a building and a known split")
        if building in ownership and ownership[building] != split:
            raise ValueError("Quick selection cannot move a building across splits")
        ownership[building] = split
        if split in {"val", "test"}:
            by_building[split, building].append(base)
    candidates = {"val": [], "test": []}
    for (split, building), rows in by_building.items():
        chosen = min(rows, key=lambda row: (_rank(seed, split, building, row["base_id"]), row["base_id"]))
        candidates[split].append(chosen)
    for split, rows in candidates.items():
        rows.sort(key=lambda row: (_rank(seed, split, row["building_id"]), row["building_id"]))
    return candidates


def _eligible_scenarios(base, rows):
    variants = defaultdict(dict)
    for row in rows:
        if any(row.get(key) != base.get(key) for key in ("building_id", "split", "view_id")):
            raise ValueError("Quick-selection case identity disagrees with its base")
        if row["state"] not in QUICK_STATES or row["nuisance_kind"] not in QUICK_FAMILIES:
            continue
        scenario, state = row["scenario_id"], row["state"]
        if state in variants[scenario]:
            raise ValueError("Quick selection found duplicate state/scenario cases")
        variants[scenario][state] = row
    families = defaultdict(list)
    for identifier, states in variants.items():
        if set(states) != set(QUICK_STATES):
            raise ValueError("Quick selection requires all three states for each eligible scenario")
        kinds = {row["nuisance_kind"] for row in states.values()}
        specs = {json.dumps(row.get("scenario"), sort_keys=True, ensure_ascii=False) for row in states.values()}
        if len(kinds) != 1 or len(specs) != 1:
            raise ValueError("Quick-selection scenario metadata differs between states")
        family = next(iter(kinds))
        families[family].append(identifier)
    for identifiers in families.values():
        identifiers.sort()
    if len(families) < 2:
        raise ValueError("Each quick-selection base requires two distinct eligible nuisance families")
    return families, variants


def _scenario_groups(family, identifiers, variants):
    groups = defaultdict(list)
    for identifier in identifiers:
        spec = variants[identifier][QUICK_STATES[0]].get("scenario") or {}
        # Strength and templates come from frozen augmentation metadata, not GT.
        # Old small fixtures without strength metadata use scenario IDs instead.
        value = ["strength", spec["strength"]] if family == "shadow" and "strength" in spec else ["scenario_id", identifier]
        key = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        groups[key].append(identifier)
    return groups


def select_quick_subset(bases, cases, max_bases=10, seed=42):
    """Choose at most ``max_bases`` buildings and six frozen cases per base.

    This operates on validated source-index rows, never opens RGB, labels or
    masks, and never reads scores. Validation comes first so its threshold can
    be frozen before any test previews. A shortage of buildings is recorded,
    rather than filled with correlated crops from the same facade.
    """
    if type(max_bases) is not int or max_bases < 2:
        raise ValueError("quick_bases must be an integer of at least 2")
    if type(seed) is not int:
        raise ValueError("Quick-selection seed must be an integer")
    bases, cases = list(bases), list(cases)
    candidates = _base_candidates(bases, seed)
    available = {split: len(rows) for split, rows in candidates.items()}
    if not all(available.values()):
        raise ValueError("Quick selection requires both validation and test buildings")

    target_val = max(1, round(max_bases * .3))
    allocation = {"val": min(target_val, available["val"])}
    allocation["test"] = min(max_bases - allocation["val"], available["test"])
    remainder = max_bases - sum(allocation.values())
    additional_val = min(remainder, available["val"] - allocation["val"])
    allocation["val"] += additional_val
    remainder -= additional_val
    allocation["test"] += min(remainder, available["test"] - allocation["test"])
    selected_bases = [base for split in ("val", "test") for base in candidates[split][:allocation[split]]]

    cases_by_base, case_ids = defaultdict(list), set()
    known_bases = {row["base_id"] for row in bases}
    for row in cases:
        if row["case_id"] in case_ids or row["base_id"] not in known_bases:
            raise ValueError("Quick selection requires unique case IDs with known bases")
        case_ids.add(row["case_id"])
        cases_by_base[row["base_id"]].append(row)

    family_order = sorted(QUICK_FAMILIES, key=lambda family: _rank(seed, "family_order", family))
    selected_cases, by_base = [], []
    family_occurrences, variant_occurrences = defaultdict(int), defaultdict(lambda: defaultdict(int))
    for position, base in enumerate(selected_bases):
        families, variants = _eligible_scenarios(base, cases_by_base[base["base_id"]])
        offset = 2 * position % len(family_order)
        rotated = family_order[offset:] + family_order[:offset]
        chosen_families = [family for family in rotated if family in families][:2]
        chosen_scenarios, chosen_groups = [], []
        for family in chosen_families:
            groups = _scenario_groups(family, families[family], variants)
            order = sorted(groups, key=lambda group: _rank(seed, "variant_order", family, group))
            group = order[family_occurrences[family] % len(order)]
            # Cycle templates within a strength; other families cycle scenarios.
            options = sorted(groups[group], key=lambda identifier: _rank(seed, "template_order", family, group, identifier))
            identifier = options[variant_occurrences[family][group] % len(options)]
            chosen_scenarios.append(identifier)
            chosen_groups.append(json.loads(group))
            family_occurrences[family] += 1
            variant_occurrences[family][group] += 1
        rows = [variants[scenario][state] for state in QUICK_STATES for scenario in chosen_scenarios]
        selected_cases.extend(rows)
        by_base.append({"base_id": base["base_id"], "building_id": base["building_id"], "split": base["split"],
                        "eligible_families": sorted(families),
                        "eligible_scenario_ids_by_family": dict(sorted(families.items())),
                        "selected_families": chosen_families, "selected_scenario_ids": chosen_scenarios,
                        "selected_variant_groups": chosen_groups,
                        "case_ids": [row["case_id"] for row in rows]})

    metadata = {"schema_version": 1, "mode": "quick", "seed": seed,
                "algorithm": "sha256 building/crop ranking; seeded cyclic nuisance-family pairs and intensity variants",
                "variant_allocation_strategy": "Cycle seed-shuffled shadow strengths, then templates within each strength; cycle seed-shuffled scenario IDs for other families",
                "selection_inputs": "base/case identity and state/scenario metadata only; no labels, masks or scores",
                "requested_max_bases": max_bases, "selected_base_count": len(selected_bases),
                "available_buildings_by_split": available, "target_validation_bases": target_val,
                "selected_buildings_by_split": allocation, "distinct_building_count": len(selected_bases),
                "states": list(QUICK_STATES), "conditions_per_state": 2, "cases_per_base": 6,
                "selected_case_count": len(selected_cases), "eligible_families": list(QUICK_FAMILIES),
                "family_order": family_order, "excluded_states": ["self_paste"],
                "selected_variant_occurrences_by_family": {
                    family: dict(sorted(counts.items())) for family, counts in sorted(variant_occurrences.items())},
                "excluded_nuisance_kinds": ["identity", "jpeg"], "bases": by_base,
                "scope": "Small exploratory subset; no selection or stopping based on test metrics"}
    return selected_bases, selected_cases, metadata
