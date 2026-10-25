"""Frozen annotation-stratified selection of existing aligned RGB crop pairs.

CSV diagnostics rank a development cohort; they never replace pixel targets,
change the reviewed split, or turn unknown context into a negative label.
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .io import read_json, sha256, write_json


VERSION = "annotation-balanced-v1"
DEFAULTS = dict(splits=["train", "val"], min_valid_fraction=.8,
                min_known_pixels=64, min_known_fraction=.01, min_changed_pixels=8,
                min_stable_class_pixels=8, area_direction_threshold=.01,
                change_fraction_edges=[.01, .05, .1, .25], max_pairs=500,
                max_pairs_per_building=40, max_crops_per_photo_pair=4,
                max_same_pair_iou=.25)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def policy(value):
    unknown = set(value) - set(DEFAULTS) - {"strategy"}
    if unknown or value.get("strategy") != VERSION:
        raise ValueError(f"Unknown annotation selection policy/keys: {sorted(unknown)}")
    result = {**DEFAULTS, **value}
    parts = result["splits"]
    if (not isinstance(parts, list) or not parts or len(set(parts)) != len(parts)
            or set(parts) - {"train", "val", "test"}):
        raise ValueError("selection.splits must contain unique explicit partitions")
    for name in ("min_known_pixels", "min_changed_pixels", "min_stable_class_pixels",
                 "max_pairs", "max_pairs_per_building", "max_crops_per_photo_pair"):
        if type(result[name]) is not int or result[name] < 1:
            raise ValueError(f"selection.{name} must be a positive integer")
    for name in ("min_valid_fraction", "min_known_fraction", "area_direction_threshold", "max_same_pair_iou"):
        if (isinstance(result[name], bool) or not isinstance(result[name], (int, float))
                or not math.isfinite(result[name]) or not 0 < result[name] <= 1):
            raise ValueError(f"selection.{name} must be finite and in (0, 1]")
    edges = result["change_fraction_edges"]
    if (not isinstance(edges, list) or not edges or any(type(x) not in (int, float)
            or not math.isfinite(x) or not 0 < x < 1 for x in edges)
            or sorted(set(edges)) != edges):
        raise ValueError("change_fraction_edges must be strictly increasing fractions in (0,1)")
    return result


def _csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Missing or duplicate CSV columns: {path}")
        return list(reader)


def _number(row, name, integer=False, optional=False):
    raw = row.get(name)
    if optional and raw in (None, ""):
        return None
    try:
        value = int(raw) if integer else float(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {name} in annotation statistics") from exc
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {name} in annotation statistics")
    return value


def _identity(diagnostic, indexed):
    for key in ("dataset_crop_id", "pair_id", "building_id", "view_id", "split", "alignment_method"):
        if diagnostic.get(key) != indexed.get(key):
            raise ValueError(f"Statistics/index identity mismatch: {key}")
    for key in ("reference_year", "source_year"):
        if type(indexed.get(key)) is not int or _number(diagnostic, key, True) != indexed[key]:
            raise ValueError(f"Statistics/index year mismatch: {key}")
    if indexed["source_year"] <= indexed["reference_year"]:
        raise ValueError("Selected pairs must be chronological")


def _artifact(root, record, name):
    path = root / name
    actual = sha256(path)
    if record.get("artifact_sha256", {}).get(name) != actual:
        raise ValueError(f"Statistics/plan artifact changed or lacks its hash: {name}")
    return path, actual


def load_statistics(root, crop_root, index, parent_summary, parent_inputs):
    """Bind the console calculation to this exact crop snapshot, not CSV names."""
    root = Path(root).expanduser().resolve()
    run = read_json(root / "run.json")
    if run.get("kind") != "crop_statistics" or run.get("status") != "completed":
        raise ValueError("--statistics-run must be a completed crop_statistics run")
    files = {str(root / "run.json"): sha256(root / "run.json")}
    loaded = {}
    for name in ("summary.json", "experiment_contract.json", "crops.csv", "class_changes.csv"):
        path, actual = _artifact(root, run, name)
        files[str(path)] = actual
        loaded[name] = read_json(path) if name.endswith(".json") else _csv(path)
    summary, contract = loaded["summary.json"], loaded["experiment_contract.json"]
    original_root = Path(run.get("config", {}).get("crops", ""))
    if not original_root.is_absolute():
        raise ValueError("Statistics must record their original absolute crop root")
    for name, key in (("run.json", "parent_run"), ("summary.json", "parent_summary"),
                      (parent_summary["index_path"], "parent_index"), ("split.json", "split")):
        if contract.get("inputs", {}).get(str(original_root / name)) != parent_inputs[key]:
            raise ValueError(f"Statistics belong to another crop snapshot: {name}")
    if contract.get("annotation_core_sha256") != sha256(Path(__file__).with_name("annotation_targets.py")):
        raise ValueError("Annotation statistics were produced by a different target core")
    if (summary.get("all_crop_pairs") != len(index["crops"])
            or summary.get("processed_crop_pairs") != len(loaded["crops.csv"])):
        raise ValueError("Statistics summary and CSV counts disagree")
    categories = {c["id"]: c["name"] for c in index["categories"]}
    if {c["category_id"]: c["category"] for c in summary["classes"]} != categories:
        raise ValueError("Statistics and crop ontology disagree")
    return loaded["crops.csv"], loaded["class_changes.csv"], files, summary


def _iou(a, b):
    x0, y0, x1, y1 = a
    u0, v0, u1, v1 = b
    common = max(0, min(x1, u1) - max(x0, u0)) * max(0, min(y1, v1) - max(y0, v0))
    return common / ((x1-x0)*(y1-y0) + (u1-u0)*(v1-v0) - common)


def select_pairs(rows, split, categories, crop_rows, class_rows, selection):
    """Pure deterministic selection; no image decode, masks, models or writes."""
    cfg = policy(selection)
    indexed, ownership, pair_owners, evidence, reasons = {}, {}, {}, defaultdict(list), {}
    for row in rows:
        identifier, building, part = row["dataset_crop_id"], row["building_id"], row["split"]
        if identifier in indexed or not building or split["building_assignments"].get(building) != part:
            raise ValueError("Duplicate crop or reviewed building/split mismatch")
        indexed[identifier] = row
        for image_hash in row["source_image_sha256"].values():
            if ownership.setdefault(image_hash, part) != part:
                raise ValueError("Original image hash leaks between partitions")
        owner = (building, row["view_id"], tuple(sorted(row["source_image_sha256"].items())))
        if pair_owners.setdefault(row["pair_id"], owner) != owner:
            raise ValueError("One photo-pair ID has inconsistent owners/images")
        bbox, shape = row["bbox_canvas_xyxy"], row["tile_shape"]
        if (len(bbox) != 4 or any(type(x) is not int for x in bbox) or len(shape) != 2
                or any(type(x) is not int or x < 8 for x in shape)
                or [bbox[3]-bbox[1], bbox[2]-bbox[0]] != shape):
            raise ValueError("Invalid native crop rectangle/grid")
        if part not in cfg["splits"]:
            reasons[identifier] = "outside_requested_split"
        elif not 0 <= row["valid_fraction"] <= 1:
            raise ValueError("Invalid geometric valid_fraction")
        elif row["valid_fraction"] < cfg["min_valid_fraction"]:
            reasons[identifier] = "below_min_valid_fraction"
    diagnostics = {}
    for row in crop_rows:
        identifier = row["dataset_crop_id"]
        if identifier in diagnostics or identifier not in indexed:
            raise ValueError("Duplicate/unknown crop in statistics")
        _identity(row, indexed[identifier])
        n, changed = _number(row, "known_pixels", True), _number(row, "changed_pixels", True)
        geometric = round(indexed[identifier]["valid_fraction"]*math.prod(indexed[identifier]["tile_shape"]))
        fraction = _number(row, "known_fraction")
        if (not 0 <= changed <= n <= geometric
                or not math.isclose(fraction, n/geometric if geometric else 0., abs_tol=1e-9)):
            raise ValueError("Crop statistics disagree with geometric support")
        diagnostics[identifier] = row
    if any(r["split"] in cfg["splits"] and r["dataset_crop_id"] not in diagnostics for r in rows):
        raise ValueError("Statistics do not cover every crop in the requested splits")
    ontology, seen = {c["id"]: c["name"] for c in categories}, set()
    for raw in class_rows:
        identifier, cid = raw["dataset_crop_id"], _number(raw, "category_id", True)
        if (identifier not in diagnostics or (identifier, cid) in seen
                or ontology.get(cid) != raw["category"]):
            raise ValueError("Unknown/duplicate class diagnostic or ontology mismatch")
        seen.add((identifier, cid))
        row = indexed[identifier]
        _identity(raw, row)
        values = {k: _number(raw, k, k in {"known_pixels", "changed_pixels", "uncomparable_target_pixels"},
                              optional=k in {"change_fraction", "before_fraction", "after_fraction", "area_delta"})
                  for k in ("known_pixels", "known_fraction", "changed_pixels", "change_fraction",
                            "before_fraction", "after_fraction", "area_delta", "uncomparable_target_pixels")}
        n, changed = values["known_pixels"], values["changed_pixels"]
        geometric = round(row["valid_fraction"]*math.prod(row["tile_shape"]))
        if (not 0 <= changed <= n <= geometric or not 0 <= values["known_fraction"] <= 1
                or values["uncomparable_target_pixels"] < 0
                or not math.isclose(values["known_fraction"], n/geometric if geometric else 0., abs_tol=1e-9)):
            raise ValueError("Invalid annotation pixel counts/fractions")
        if n:
            if (any(values[k] is None for k in ("change_fraction", "before_fraction", "after_fraction", "area_delta"))
                    or any(not 0 <= values[k] <= 1 for k in ("change_fraction", "before_fraction", "after_fraction"))
                    or not math.isclose(values["change_fraction"], changed/n, abs_tol=1e-9)
                    or not math.isclose(values["area_delta"], values["after_fraction"]-values["before_fraction"], abs_tol=1e-9)
                    or abs(values["area_delta"]) > values["change_fraction"]+1e-9):
                raise ValueError("Class fractions disagree with their common annotation support")
        if identifier in reasons or n < cfg["min_known_pixels"] or values["known_fraction"] < cfg["min_known_fraction"]:
            continue
        if changed >= cfg["min_changed_pixels"]:
            role = "changed"
            band = sum(values["change_fraction"] >= edge for edge in cfg["change_fraction_edges"])
            threshold = cfg["area_direction_threshold"]
            direction = "increase" if values["area_delta"] >= threshold else "decrease" if values["area_delta"] <= -threshold else "small_net_area"
        elif changed == 0 and min(values["before_fraction"], values["after_fraction"])*n >= cfg["min_stable_class_pixels"]-1e-7:
            role, band, direction = "stable_present_on_common_support", "stable", "stable"
        else:
            continue
        evidence[identifier].append(dict(category_id=cid, category=raw["category"], role=role,
                                        magnitude_band=band, area_direction=direction, **values))
    # Exact paired RGB/support duplicates are redundant. Different late photos remain eligible.
    unique = {}
    for identifier in sorted(evidence, key=lambda x: (_digest(x), x)):
        row = indexed[identifier]
        signature = (row["building_id"], row["view_id"], *(row["artifact_sha256"][name]
                     for name in ("reference_rgb.png", "source_rgb.png", "reference_support.png", "source_support.png")))
        if signature in unique:
            reasons[identifier] = "duplicate_rgb_pair"
        else:
            unique[signature] = identifier
    eligible = {key: indexed[key] for key in evidence if key not in reasons}
    if not eligible:
        raise ValueError("No crop pair has sufficient changed or stable-present class evidence")
    buckets, views = defaultdict(list), defaultdict(list)
    for identifier, row in eligible.items():
        views[row["split"], row["building_id"], row["view_id"]].append(identifier)
        for item in evidence[identifier]:
            buckets[row["split"], item["category_id"], item["role"], item["magnitude_band"], item["area_direction"]].append(identifier)
    if cfg["max_pairs"] < len(views):
        raise ValueError(f"max_pairs must be >= {len(views)} to retain every eligible building/view")
    rarity = {key: (len({eligible[x]["building_id"] for x in ids}), len(ids), str(key)) for key, ids in buckets.items()}
    selected, chosen, chosen_for = [], set(), {}
    building_count, view_count, pair_count, date_count, gap_count = (Counter() for _ in range(5))
    pair_rectangles = defaultdict(list)

    def blocked(identifier):
        row = eligible[identifier]
        if building_count[row["building_id"]] >= cfg["max_pairs_per_building"]:
            return "building_budget"
        if pair_count[row["pair_id"]] >= cfg["max_crops_per_photo_pair"]:
            return "photo_pair_budget"
        if any(_iou(row["bbox_canvas_xyxy"], box) > cfg["max_same_pair_iou"] for box in pair_rectangles[row["pair_id"]]):
            return "same_pair_spatial_overlap"
        return None

    def best(ids, bucket=None):
        candidates = [x for x in ids if x not in chosen and not blocked(x)]
        def rank(identifier):
            row, items = eligible[identifier], evidence[identifier]
            if bucket is not None:
                items = [i for i in items if (row["split"], i["category_id"], i["role"],
                         i["magnitude_band"], i["area_direction"]) == bucket]
            view = (row["building_id"], row["view_id"])
            date = (*view, row["reference_year"], row["source_year"])
            gap = row["source_year"]-row["reference_year"]
            gap_bin = 0 if gap <= 3 else 1 if gap <= 7 else 2
            return (building_count[row["building_id"]], view_count[view], pair_count[row["pair_id"]],
                    date_count[date], gap_count[(*view, gap_bin)],
                    -max(i["known_fraction"] for i in items),
                    -max(i["change_fraction"] for i in items), _digest(identifier), identifier)
        return min(candidates, key=rank) if candidates else None

    def take(identifier, why):
        row = eligible[identifier]
        selected.append(row)
        chosen.add(identifier)
        chosen_for[identifier] = why
        view = (row["building_id"], row["view_id"])
        building_count[row["building_id"]] += 1
        view_count[view] += 1
        pair_count[row["pair_id"]] += 1
        date_count[(*view, row["reference_year"], row["source_year"])] += 1
        gap = row["source_year"]-row["reference_year"]
        gap_count[(*view, 0 if gap <= 3 else 1 if gap <= 7 else 2)] += 1
        pair_rectangles[row["pair_id"]].append(row["bbox_canvas_xyxy"])

    for view in sorted(views):
        choice = best(views[view])
        if choice is None:
            raise ValueError("Selection budgets prevent retaining every eligible view; increase the explicit budgets")
        take(choice, dict(phase="view_coverage", view=list(view)))
    ordered_buckets = sorted(buckets, key=lambda x: rarity[x])
    while len(selected) < cfg["max_pairs"]:
        progress = False
        for bucket in ordered_buckets:
            choice = best(buckets[bucket], bucket)
            if choice is not None:
                take(choice, dict(phase="class_round_robin", bucket=list(bucket)))
                progress = True
            if len(selected) >= cfg["max_pairs"]:
                break
        if not progress:
            break
    for row in rows:
        identifier = row["dataset_crop_id"]
        if identifier not in chosen and identifier not in reasons:
            reasons[identifier] = (blocked(identifier) or "global_pair_budget") if identifier in eligible else "no_eligible_class_evidence"
    # The synthetic branch needs each reference RGB/support only once per building/view.
    synthetic, base_keys, base_members = [], {}, defaultdict(list)
    for row in selected:
        key = _digest([row["building_id"], row["view_id"], row["artifact_sha256"]["reference_rgb.png"],
                       row["artifact_sha256"]["reference_support.png"]])
        if key not in base_keys:
            base_keys[key] = row["dataset_crop_id"]
            synthetic.append(row)
        base_members[base_keys[key]].append(row["dataset_crop_id"])
    details = {identifier: dict(chosen_for=chosen_for[identifier], evidence=sorted(evidence[identifier], key=lambda x: x["category_id"]),
                               original_pair_status=diagnostics[identifier]["status"],
                               original_pair_reason=diagnostics[identifier]["reason"])
               for identifier in sorted(chosen)}
    selected_classes = []
    for category in categories:
        cid = category["id"]
        for role in ("changed", "stable_present_on_common_support"):
            available = [key for key in eligible if any(i["category_id"] == cid and i["role"] == role for i in evidence[key])]
            picked = [key for key in chosen if any(i["category_id"] == cid and i["role"] == role for i in evidence[key])]
            selected_classes.append(dict(category_id=cid, category=category["name"], role=role,
                eligible_pairs=len(available), selected_pairs=len(picked),
                eligible_buildings=len({eligible[k]["building_id"] for k in available}),
                selected_buildings=len({eligible[k]["building_id"] for k in picked})))
    report = dict(eligible_pair_count=len(eligible), selected_pair_count=len(selected),
                  synthetic_base_count=len(synthetic), selected_building_count=len(building_count),
                  eligible_view_count=len(views), selected_view_count=len(view_count),
                  selected_by_building=dict(sorted(building_count.items())),
                  selected_by_split=dict(Counter(r["split"] for r in selected)), classes=selected_classes,
                  selected_photo_pair_count=len(pair_count),
                  excluded_reasons=dict(Counter(reasons.values())),
                  unrepresented_buckets=[list(k) for k in ordered_buckets if not chosen.intersection(buckets[k])],
                  statistical_unit="building; overlapping windows and temporal pairs are correlated",
                  label_scope="class differences/stability on common annotated support; outside unknown")
    return dict(selection=cfg, selected=selected, synthetic_selected=synthetic,
                synthetic_members=dict(base_members), selected_details=details,
                excluded_crops=[dict(dataset_crop_id=k, reason=v) for k, v in sorted(reasons.items())], report=report)


def build_plan(crop_root, index, split, parent_summary, config, statistics_run, parent_inputs):
    crops, classes, files, statistics = load_statistics(statistics_run, crop_root, index, parent_summary, parent_inputs)
    result = select_pairs(index["crops"], split, index["categories"], crops, classes, config["selection"])
    return dict(kind="crop_selection_plan", schema_version=1, algorithm=VERSION,
                input_sha256=parent_inputs, external_inputs=files, config=config,
                selection_code_sha256=sha256(__file__), renderer_code_sha256=sha256(Path(__file__).with_name("augmentations.py")),
                generation_code_sha256=sha256(Path(__file__).with_name("hypothesis_dataset.py")),
                source_statistics_summary=statistics, **result)


def verify_plan(plan, config, parent_inputs):
    if (plan.get("kind") != "crop_selection_plan" or plan.get("algorithm") != VERSION
            or plan.get("input_sha256") != parent_inputs or plan.get("config") != config
            or plan.get("selection_code_sha256") != sha256(__file__)
            or plan.get("renderer_code_sha256") != sha256(Path(__file__).with_name("augmentations.py"))
            or plan.get("generation_code_sha256") != sha256(Path(__file__).with_name("hypothesis_dataset.py"))):
        raise ValueError("Frozen selection plan differs from the current data/config/selector")
    for path, expected in plan["external_inputs"].items():
        if sha256(path) != expected:
            raise ValueError(f"Selection statistics changed: {path}")


def load_plan(path, config, parent_inputs):
    root = Path(path).expanduser().resolve()
    record = read_json(root / "run.json")
    if record.get("kind") != "crop_selection" or record.get("status") != "completed":
        raise ValueError("--selection-plan must point to a completed crop selection run")
    selection_path, selection_hash = _artifact(root, record, "selection.json")
    plan = read_json(selection_path)
    verify_plan(plan, config, parent_inputs)
    return plan, {str(root / "run.json"): sha256(root / "run.json"), str(selection_path): selection_hash}


def write_plan(out, plan, budget):
    """Reuse the experiment RunStore ledger; save no RGB, masks or features."""
    RunStore = importlib.import_module("facade_change.2026-10-14_temporal_experiment").RunStore
    store = RunStore(out, dict(kind="crop_selection", input_sha256=plan["input_sha256"],
                               selection=plan["selection"], selection_code_sha256=plan["selection_code_sha256"]))
    store.record["kind"] = "crop_selection"
    try:
        store.contract(dict(input_sha256=plan["input_sha256"], external_inputs=plan["external_inputs"],
                            selection_code_sha256=plan["selection_code_sha256"], scope=plan["report"]["label_scope"]))
        write_json(store.out / "selection.json", plan)
        write_json(store.out / "config.json", plan["config"])
        summary = dict(status="completed", **plan["report"], generation_budget=budget, rgb_generated=False)
        write_json(store.out / "summary.json", summary)
        verify_plan(plan, plan["config"], plan["input_sha256"])
        store.finish("completed", summary=summary, artifact_sha256={p.name: sha256(p) for p in store.out.iterdir()
                     if p.is_file() and p.name not in {"run.json", "events.jsonl", ".run.lock"}})
        return summary
    except BaseException as exc:
        store.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        store.close()
