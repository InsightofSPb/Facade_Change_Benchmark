"""Existing RGB/provenance boundaries and TRAIN/VAL-only experiment selection."""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..hypothesis_benchmark import _building_macro, _case_inputs, _metrics_from_counts
from ..io import finish_record, new_directory, read_json, run_record, sha256, write_json


STATES = ("unchanged", "crack", "paint_patch")
SPLITS = ("train", "val")
SCOPE = "EXPLORATORY procedural TRAIN/VAL feasibility; VAL calibration, no TEST RGB/labels, no confidence intervals"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def positive_integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def parent_selection(dataset_run, max_buildings_per_split=2, max_bases_per_building=1,
                     scenarios_per_base=2, seed=42):
    """Freeze balanced matched triples before RGB/GT access; TEST metadata only.

    Budgets of zero use all available buildings/bases/scenarios. Every selected
    base has exactly the same scenario identities for all three states.
    """
    for name, value in (("max_buildings_per_split", max_buildings_per_split),
                        ("max_bases_per_building", max_bases_per_building),
                        ("scenarios_per_base", scenarios_per_base)):
        positive_integer(value, name, 0)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    loader = importlib.import_module("facade_change.2026-10-04_msdzip_h0")
    root, parent, index, split, fingerprints = loader._parent(dataset_run)
    if digest(parent["config"]) != parent.get("config_sha256"):
        raise ValueError("Parent configuration hash disagrees")
    bases, by_base, ownership, ids = {}, defaultdict(list), {}, set()
    for base in index["bases"]:
        identifier, building, part = base["base_id"], base["building_id"], base["split"]
        if (not isinstance(identifier, str) or identifier in bases or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier)
                or not building or part not in {"train", "val", "test"}):
            raise ValueError("Duplicate base or invalid facade/split identity")
        if split["building_assignments"].get(building) != part:
            raise ValueError("Facade split leakage or reviewed assignment disagreement")
        if building in ownership and ownership[building] != part:
            raise ValueError("Facade split leakage")
        ownership[building] = part
        bases[identifier] = base
    for case in index["cases"]:
        identifier = case["case_id"]
        base = bases.get(case["base_id"])
        if not isinstance(identifier, str) or identifier in ids or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) or not base or any(case.get(k) != base.get(k)
                                                for k in ("building_id", "view_id", "split")):
            raise ValueError("Duplicate case or case/base identity disagreement")
        ids.add(identifier)
        if case["parent_dataset_crop_id"] != base["dataset_crop_id"]:
            raise ValueError("Case crop identity disagrees")
        expected = "H1" if case["state"] in {"crack", "paint_patch"} else "H0"
        if (case["hypothesis"] != expected
                or bool(case.get("sham_self_paste")) != (case["state"] == "self_paste")):
            raise ValueError("Case state/hypothesis identity disagrees")
        declared = {s["id"]: s for s in parent["config"]["scenarios"]}
        scenario = declared.get(case["scenario_id"])
        if not scenario or scenario["kind"] != case["nuisance_kind"]:
            raise ValueError("Case scenario/nuisance identity disagrees with the parent protocol")
        if "scenario" in case and case["scenario"] != scenario:
            raise ValueError("Case scenario parameters disagree with the parent protocol")
        by_base[case["base_id"]].append(case)
    rank = lambda value: digest([seed, value])
    selected_bases, selected_cases = [], []
    for part in SPLITS:
        buildings = sorted({b["building_id"] for b in bases.values() if b["split"] == part}, key=rank)
        if not buildings:
            raise ValueError(f"Missing {part.upper()} facades; TEST cannot replace TRAIN/VAL")
        if max_buildings_per_split:
            buildings = buildings[:max_buildings_per_split]
        for building in buildings:
            candidates = sorted((b for b in bases.values() if b["building_id"] == building),
                                key=lambda b: rank(b["base_id"]))
            if max_bases_per_building:
                candidates = candidates[:max_bases_per_building]
            for base in candidates:
                rows = [c for c in by_base[base["base_id"]] if c["state"] in STATES and not c["sham_self_paste"]]
                keys = {(c["state"], c["scenario_id"]) for c in rows}
                scenarios = sorted({c["scenario_id"] for c in rows}, key=rank)
                if (len(keys) != len(rows) or not scenarios
                        or keys != {(state, scenario) for state in STATES for scenario in scenarios}):
                    raise ValueError("Selected base lacks matched unchanged/crack/paint scenario triples")
                if scenarios_per_base:
                    scenarios = scenarios[:scenarios_per_base]
                selected_bases.append(base)
                selected_cases.extend(sorted((c for c in rows if c["scenario_id"] in scenarios),
                                             key=lambda c: (c["scenario_id"], STATES.index(c["state"]))))
    return {"root": root, "parent": parent, "index": index, "split": split,
            "fingerprints": fingerprints, "bases": selected_bases, "cases": selected_cases,
            "selection_options": {"max_buildings_per_split": max_buildings_per_split,
                                  "max_bases_per_building": max_bases_per_building,
                                  "scenarios_per_base": scenarios_per_base, "seed": seed}}


def rgb_inputs(selection, case, checked):
    loader = importlib.import_module("facade_change.2026-10-04_msdzip_h0")
    base = next(b for b in selection["bases"] if b["base_id"] == case["base_id"])
    return loader._rgb_inputs(selection["root"], selection["parent"], base, case, checked)


def evaluation_inputs(selection, case, checked):
    base = next(b for b in selection["bases"] if b["base_id"] == case["base_id"])
    return _case_inputs(selection["root"], selection["parent"], base, case, checked)


def assert_inputs_unchanged(selection, checked):
    for relative, value in checked.items():
        if sha256(selection["root"] / relative) != value:
            raise ValueError(f"Input changed during experiment: {relative}")
    loader = importlib.import_module("facade_change.2026-10-04_msdzip_h0")
    if loader._parent(selection["root"])[-1] != selection["fingerprints"]:
        raise ValueError("Parent provenance changed during experiment")


class ExperimentRun:
    """One fresh run plus a small append-only event log; uses existing run API."""
    def __init__(self, out, kind, selection, options):
        self.out = new_directory(out)
        self.config = {"dataset_run": str(selection["root"]),
                       "input_sha256": selection["fingerprints"],
                       "selection_options": selection["selection_options"],
                       "case_ids": [c["case_id"] for c in selection["cases"]],
                       "requested_splits": list(SPLITS), "scope": SCOPE, **options}
        self.record = run_record(kind, self.config)
        self.sequence = 0
        write_json(self.out / "run.json", self.record)
        write_json(self.out / "experiment_contract.json", self.config)
        write_json(self.out / "selected_cases.json", {"bases": selection["bases"], "cases": selection["cases"]})
        self.event("started", config_sha256=self.record["config_sha256"])

    def event(self, kind, **values):
        self.sequence += 1
        record = {"schema_version": 1, "run_id": self.out.name,
                  "event_id": f"{self.out.name}:{self.sequence}", "sequence": self.sequence,
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "event_type": kind, "config_sha256": self.record["config_sha256"], **values}
        with (self.out / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        return record["event_id"]

    def finish(self, summary, checked):
        write_json(self.out / "summary.json", summary)
        write_json(self.out / "input_artifacts.json", checked)
        self.event("completed", summary_sha256=sha256(self.out / "summary.json"))
        finish_record(self.out, self.record, "completed_exploratory")
        return summary

    def fail(self, exc):
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        self.event(status, error=str(exc))
        finish_record(self.out, self.record, status, str(exc))


def csv_rows(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
                          if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()}
                         for row in rows)


def metric_row(case, scores, labels, threshold, method):
    if scores.shape != labels.shape or scores.dtype != np.float32:
        raise ValueError("Scores must be native-grid float32")
    valid = labels != 255
    if not np.isfinite(scores[valid]).all():
        raise ValueError("Nonfinite score on evaluation support")
    predicted, positive = scores > np.float32(threshold), labels == 1
    counts = [int((predicted & positive).sum()), int((predicted & (labels == 0)).sum()),
              int((~predicted & positive).sum()), int((~predicted & (labels == 0)).sum())]
    identity = {k: case.get(k) for k in ("case_id", "base_id", "building_id", "split", "state",
                                       "scenario_id", "nuisance_kind", "hypothesis",
                                       "comparable_fraction", "retained_visible_edit_fraction")}
    return {**identity, "method": method, "threshold": float(np.float32(threshold)),
            **_metrics_from_counts(case, counts, int((~valid).sum()))}


def calibrate_recall(items, fpr_targets=(.01, .05), minimum_threshold=0., max_grid=512):
    """VAL-only Recall under facade-macro H0 FPR; strict float32 decisions."""
    if not fpr_targets or any(isinstance(value, bool) or not 0 <= value <= 1 or not np.isfinite(value) for value in fpr_targets):
        raise ValueError("FPR targets must be finite probabilities")
    val = [item for item in items if item["case"]["split"] == "val"]
    if not {"H0", "H1"}.issubset({item["case"]["hypothesis"] for item in val}):
        raise ValueError("Calibration requires VAL H0 and H1 cases")
    values = np.concatenate([item["scores"][item["labels"] != 255] for item in val])
    values = values[np.isfinite(values) & (values >= minimum_threshold)]
    grid = np.unique(np.concatenate([np.array([minimum_threshold], np.float32), values]))
    if len(grid) > max_grid:
        grid = grid[np.unique(np.linspace(0, len(grid)-1, max_grid, dtype=np.int64))]
    curve = []
    # Sort each case once: no correlated pixel pooling and no repeated map scans.
    prepared = []
    for item in val:
        labels, scores = item["labels"], item["scores"]
        positives = np.sort(scores[labels == 1])
        negatives = np.sort(scores[labels == 0])
        prepared.append((item["case"], positives, negatives))
    for threshold in grid:
        recalls, false_alarms = defaultdict(list), defaultdict(list)
        for case, positives, negatives in prepared:
            if case["hypothesis"] == "H1" and len(positives):
                recalls[case["building_id"]].append(1 - np.searchsorted(positives, threshold, side="right") / len(positives))
            if case["hypothesis"] == "H0" and len(negatives):
                false_alarms[case["building_id"]].append(1 - np.searchsorted(negatives, threshold, side="right") / len(negatives))
        if not recalls or not false_alarms:
            raise ValueError("Calibration lacks visible H1 positives or evaluated H0 negatives")
        curve.append({"threshold": float(threshold),
                      "recall": float(np.mean([np.mean(x) for x in recalls.values()])),
                      "h0_pixel_fpr": float(np.mean([np.mean(x) for x in false_alarms.values()]))})
    choices = []
    for target in fpr_targets:
        eligible = [row for row in curve if row["h0_pixel_fpr"] <= target]
        if not eligible:
            raise ValueError("Calibration grid has no feasible H0 threshold")
        best = max(eligible, key=lambda row: (row["recall"], -row["h0_pixel_fpr"], row["threshold"]))
        choices.append({"fpr_target": float(target), **best, "selection_split": "val",
                        "criterion": "maximum facade-macro visible H1 Recall subject to H0 FPR; ties lower FPR then higher threshold"})
    return {"choices": choices, "curve": curve, "grid_policy": f"at most {max_grid} ranked unique VAL values; includes minimum and maximum; no TEST"}


def reports(run, items, methods, fpr_targets=(.01, .05), minimum_threshold=0.):
    rows, calibration, summaries, facades, classes = [], {}, {}, [], []
    for method in methods:
        selected = [item for item in items if item["method"] == method]
        choice = calibrate_recall(selected, fpr_targets, minimum_threshold)
        calibration[method] = choice
        summaries[method] = []
        for setting in choice["choices"]:
            current = []
            for item in selected:
                row = metric_row(item["case"], item["scores"], item["labels"], setting["threshold"], method)
                row["fpr_target"] = setting["fpr_target"]
                rows.append(row)
                current.append(row)
            summaries[method].append({"fpr_target": setting["fpr_target"], "threshold": setting["threshold"],
                 "by_split": {part: _building_macro([r for r in current if r["split"] == part]) for part in SPLITS},
                 "by_state": [{"split": part, "state": state, **_building_macro([r for r in current if r["split"] == part and r["state"] == state])}
                              for part in SPLITS for state in STATES]})
            for part in SPLITS:
                aggregated = _building_macro([r for r in current if r["split"] == part])
                facades.extend({"method": method, "split": part, "fpr_target": setting["fpr_target"],
                                "threshold": setting["threshold"], "building_id": b["building_id"],
                                "case_count": b["case_count"], **b["means"]} for b in aggregated["buildings"])
                for state in STATES:
                    grouped = _building_macro([r for r in current if r["split"] == part and r["state"] == state])
                    classes.append({"method": method, "split": part, "state": state,
                                    "fpr_target": setting["fpr_target"], "threshold": setting["threshold"],
                                    "case_count": grouped["case_count"], "building_count": grouped["building_count"],
                                    **grouped["means"]})
    write_json(run.out / "threshold_selection.json", calibration)
    csv_rows(run.out / "case_metrics.csv", rows)
    csv_rows(run.out / "facades.csv", facades)
    csv_rows(run.out / "classes.csv", classes)
    lines = [SCOPE, "Recall: visible positive pixels; aggregation: cases within facade, then equal facades."]
    for method, settings in summaries.items():
        for setting in settings:
            for part in SPLITS:
                means = setting["by_split"][part]["means"]
                percent = lambda value: "n/a" if value is None else f"{value:.2%}"
                lines.append(f"{part.upper()} {method} @VAL H0FPR<={setting['fpr_target']:.1%}: Recall={percent(means['recall'])}, H0FPR={percent(means['h0_pixel_fpr'])}, threshold={setting['threshold']:.6g}")
    (run.out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return summaries


def inspection_coverage(run, items, tile_size=32, fractions=(.05, .1, .25)):
    """GT-free coarse tile ranking; visible-H1 coverage is an evaluation only.

    A tile's maximum score protects narrow high-score regions. Ties use grid
    order. Budget is a fraction of nonempty reference-support tiles; actual
    pixel fraction is reported because partial tiles and rounding vary.
    This diagnoses inspection routing and is not a pixel change prediction.
    """
    rows = []
    for item in items:
        scores, labels, case = item["scores"], item["labels"], item["case"]
        support = np.isfinite(scores)
        candidates = []
        for y in range(0, scores.shape[0], tile_size):
            for x in range(0, scores.shape[1], tile_size):
                ys, xs = slice(y, y+tile_size), slice(x, x+tile_size)
                region = scores[ys, xs][support[ys, xs]]
                if len(region):
                    candidates.append((float(region.max()), y, x))
        candidates.sort(key=lambda value: (-value[0], value[1], value[2]))
        for fraction in fractions:
            count = max(1, int(np.ceil(len(candidates)*fraction)))
            selected = np.zeros(support.shape, bool)
            for _, y, x in candidates[:count]:
                selected[y:y+tile_size, x:x+tile_size] = True
            selected &= support
            positive = labels == 1
            rows.append({"method": item["method"], "case_id": case["case_id"], "building_id": case["building_id"],
                         "split": case["split"], "state": case["state"], "tile_size": tile_size,
                         "nominal_tile_fraction": fraction, "eligible_tiles": len(candidates), "selected_tiles": count,
                         "selected_reference_pixel_fraction": float(selected.sum()/support.sum()),
                         "recall": float((selected & positive).sum()/positive.sum()) if positive.any() else None})
    csv_rows(run.out / "inspection_budget.csv", rows)
    return {"policy": "max score per nonoverlapping reference-support tile; deterministic grid-order ties; no GT for selection",
            "scope": "inspection coverage, not prediction FPR; rounded tile budgets, actual pixel coverage recorded",
            "tile_size": tile_size,
            "settings": [{"method": method, "split": part, "nominal_tile_fraction": fraction,
                          **_building_macro([r for r in rows if r["method"] == method and r["split"] == part
                                             and r["nominal_tile_fraction"] == fraction])}
                         for method in sorted({r["method"] for r in rows}) for part in SPLITS for fraction in fractions]}
