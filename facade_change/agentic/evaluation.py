"""Offline, fixed-cohort evaluation of saved agentic runs (2026-11-08).

Only this module reads evaluation labels. It never calls models, changes a
threshold, or turns a textual answer into a pixel mask. Missing predictions
remain in the cohort and in end-to-end recall and coverage denominators.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ..io import finish_record, new_directory, read_json, run_record, sha256, write_json


MODES = ("baseline", "pair", "fixed_context", "agent")
DECISIONS = {"change": "change", "changed": "change", "no_change": "no_change",
             "insufficient_data": "insufficient_data", "incomparable": "incomparable",
             "not_comparable": "incomparable"}
COST_FIELDS = ("seconds", "input_tokens", "output_tokens", "images", "model_calls",
               "model_requests", "qwen_calls", "tool_calls", "cache_hits", "model_loads",
               "model_load_attempts", "retries", "worker_starts", "worker_restarts")
MEMORY_FIELDS = ("peak_allocated_bytes", "peak_reserved_bytes")
ALLOCATION_FIELDS = ("incremental_seconds", "shared_preparation_seconds",
                     "allocated_with_shared_seconds", "pair_geometry_plus_increment_seconds")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _ratio(a, b):
    return a / b if b else None


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _usage_nodes(value):
    """Read only documented nested usage records, including OOM attempts."""
    if isinstance(value, dict):
        yield value
        for key in ("memory", "usage", "attempts", "calls", "per_call"):
            if key in value:
                yield from _usage_nodes(value[key])
    elif isinstance(value, list):
        for item in value:
            yield from _usage_nodes(item)


def _measured_cost(costs, key):
    aliases = {"seconds": ("seconds", "elapsed_seconds", "wall_seconds"),
               "peak_allocated_bytes": ("peak_allocated_bytes", "max_memory_allocated_bytes"),
               "peak_reserved_bytes": ("peak_reserved_bytes", "max_memory_reserved_bytes")}
    names = aliases.get(key, (key,))
    nodes = _usage_nodes(costs) if key in MEMORY_FIELDS else [costs]
    values = [node.get(name) for node in nodes for name in names]
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)
              and np.isfinite(v) and v >= 0]
    return (max(values) if key in MEMORY_FIELDS else values[0]) if values else None


def cost_allocation(case, mode, costs):
    """Describe observed cost and explicit allocation without charging setup twice.

    Allocation is arithmetic on the recorded cache state, not a cold-start
    benchmark. Pair-only inference needs geometry, not the other method maps.
    """
    preparation = case.get("preparation_costs", case.get("baseline_costs", {})) or {}
    shared = _measured_cost(preparation, "seconds")
    incremental = _measured_cost(costs, "seconds") if mode != "baseline" else None
    geometry = _measured_cost(preparation, "geometry_seconds")
    return {
        "cost_scope": costs.get("scope") or ("shared case preparation including configured methods"
            if mode == "baseline" else "incremental mode execution after shared preparation"),
        "preparation_cost_scope": preparation.get("scope"),
        "cost_regime": costs.get("regime", preparation.get("regime", "observed cache state; cold start not assumed")),
        "incremental_seconds": incremental,
        "shared_preparation_seconds": shared,
        "allocated_with_shared_seconds": (shared + incremental
            if mode in {"fixed_context", "agent"} and shared is not None and incremental is not None else None),
        "pair_geometry_plus_increment_seconds": (geometry + incremental
            if mode == "pair" and geometry is not None and incremental is not None else None),
        "cached_source_usage": [node["source_usage"] for node in _usage_nodes(costs) if "source_usage" in node],
    }


def _manifest(path):
    document = read_json(path)
    if not isinstance(document, dict) or not isinstance(document.get("cases"), list):
        raise ValueError("Fixed case manifest must contain a cases list")
    if not isinstance(document.get("availability_policy"), dict):
        raise ValueError("Fixed manifest must specify the availability_policy")
    cases, seen, buildings = document["cases"], set(), {}
    if not cases:
        raise ValueError("The fixed case cohort is empty")
    for case in cases:
        identifier = case.get("case_id")
        if (not isinstance(identifier, str) or not identifier or identifier in seen
                or Path(identifier).name != identifier or identifier in {".", ".."}):
            raise ValueError("Case IDs must be distinct safe path components")
        seen.add(identifier)
        if not case.get("building_id") or not case.get("split"):
            raise ValueError("Every fixed case needs building_id and split")
        building, split = case["building_id"], case["split"]
        if buildings.setdefault(building, split) != split:
            raise ValueError("The same building occurs in different splits")
    # Run provenance may differ; the cases and availability policy may not.
    return document, _digest({"cases": cases, "availability_policy": document["availability_policy"]})


def _resolve(root, value):
    if not isinstance(value, str) or not value:
        raise ValueError("Expected an explicit artifact path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def _verify_recorded_artifacts(root, inputs):
    path = root / "run.json"
    if not path.is_file():
        return
    inputs[str(path)] = sha256(path)
    record = read_json(path)
    for relative, expected in record.get("artifact_sha256", {}).items():
        artifact = (root / relative).resolve()
        if Path(relative).is_absolute() or root not in artifact.parents:
            raise ValueError("Saved run artifact escapes its root")
        if not artifact.is_file() or sha256(artifact) != expected:
            raise ValueError(f"Saved run artifact changed: {relative}")
        inputs[str(artifact)] = expected


def _binary(root, value):
    path = _resolve(root, value)
    if path.suffix == ".npy":
        result = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            result = np.asarray(image)
    if (result.ndim != 2 or result.dtype not in (np.dtype(bool), np.dtype("uint8"))
            or not (np.isin(result, [0, 1]).all() or np.isin(result, [0, 255]).all())):
        raise ValueError("Pixel evaluation requires an actual binary 2-D mask")
    return result != 0


def _mask_metrics(prediction, target, support):
    if prediction.shape != target.shape or target.shape != support.shape:
        raise ValueError("Prediction, target and support must share the same grid")
    tp = int(np.count_nonzero(prediction & target & support))
    fp = int(np.count_nonzero(prediction & ~target & support))
    fn = int(np.count_nonzero(~prediction & target & support))
    tn = int(np.count_nonzero(~prediction & ~target & support))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "valid_pixels": int(support.sum()),
            "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn), "iou": _ratio(tp, tp + fp + fn),
            "fpr": _ratio(fp, fp + tn)}


def box_iou(left, right):
    """Continuous xyxy box IoU; does not rasterize language into a mask."""
    arrays = [np.asarray(v, dtype=float) for v in (left, right)]
    if any(v.shape != (4,) or not np.isfinite(v).all() or v[2] <= v[0] or v[3] <= v[1]
           for v in arrays):
        raise ValueError("Localization boxes must have finite, positive xyxy extents")
    a, b = arrays
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return float(intersection / union)


def _pixel_result(prediction, target, prediction_root, target_root):
    """Require explicit temporal-mask semantics and the exact coordinate frame."""
    if not target or not target.get("mask_path") or not prediction.get("mask_path"):
        return {"status": "not_available"}
    if prediction.get("mask_task") != "temporal_change" or target.get("mask_task") != "temporal_change":
        return {"status": "not_evaluated", "reason": "mask_is_not_explicit_temporal_change"}
    frame = prediction.get("coordinate_system")
    if not frame or frame != target.get("coordinate_system"):
        return {"status": "not_evaluated", "reason": "coordinate_system_mismatch"}
    if not prediction.get("support_path") or not target.get("support_path"):
        return {"status": "not_evaluated", "reason": "explicit_support_required"}
    try:
        pred = _binary(prediction_root, prediction["mask_path"])
        truth = _binary(target_root, target["mask_path"])
        support = _binary(prediction_root, prediction["support_path"])
        known = _binary(target_root, target["support_path"])
        if support.shape != known.shape:
            raise ValueError("Prediction and evaluation support grids differ")
        values = _mask_metrics(pred, truth, support & known)
        return {"status": "evaluated", "coordinate_system": frame, **values}
    except (OSError, ValueError) as error:
        return {"status": "error", "reason": str(error)}


def _labels(path, identifiers):
    if path is None:
        return {}, None
    document = read_json(path)
    if document.get("label_scope") != "region_temporal_change":
        raise ValueError("Labels must explicitly describe region_temporal_change, not single-image damage")
    values = {}
    for row in document.get("cases", []):
        identifier = row.get("case_id")
        if identifier not in identifiers or identifier in values:
            raise ValueError("Evaluation labels contain duplicate or unknown case IDs")
        if row.get("change") is not None and type(row["change"]) is not bool:
            raise ValueError("Region change labels must be boolean or null")
        values[identifier] = row
    return values, {"path": str(Path(path).resolve()), "sha256": sha256(path),
                    "label_scope": document["label_scope"], "provenance": document.get("provenance")}


def _result(run, case, mode):
    case_root = run / "cases" / case["case_id"]
    case_file = case_root / "case.json"
    if not case_file.is_file():
        return {"status": "missing", "decision": "insufficient_data"}, {}, case_root, {}
    actual = read_json(case_file)
    for key in ("case_id", "building_id", "view_id", "split"):
        if key in case and actual.get(key) != case[key]:
            raise ValueError(f"Case identity changed: {case['case_id']} {key}")
    if mode == "baseline":
        prediction = dict(actual.get("baseline") or {})
        costs = actual.get("baseline_costs", actual.get("costs", {}))
        root = case_root
    else:
        root = case_root / "modes" / mode
        path = root / "result.json"
        prediction = read_json(path) if path.is_file() else {"status": "missing", "decision": "insufficient_data"}
        costs = read_json(root / "costs.json") if (root / "costs.json").is_file() else {}
    if not prediction:
        prediction = {"status": "missing", "decision": "insufficient_data"}
    return prediction, costs, root, actual


def _row(run, case, mode, target, target_root):
    prediction, costs, prediction_root, actual = _result(run, case, mode)
    decision = DECISIONS.get(prediction.get("decision"), "insufficient_data")
    status = prediction.get("status", "completed" if decision in {"change", "no_change"} else "abstained")
    decided = status == "completed" and decision in {"change", "no_change"}
    row = {"run_id": run.name, "run_path": str(run), "case_id": case["case_id"],
           "building_id": case["building_id"], "split": case["split"], "mode": mode,
           "status": status, "decision": decision, "decided": decided,
           "change": target.get("change") if target else None,
           "error": prediction.get("error"), "reason": prediction.get("reason"),
           "evidence_count": len(prediction.get("evidence_ids", [])),
           "limitations": prediction.get("limitations", [])}
    evidence_path = prediction_root / "evidence.json"
    evidence = read_json(evidence_path).get("evidence", []) if evidence_path.is_file() else []
    evidence_ids = {item.get("evidence_id") for item in evidence}
    used_ids = prediction.get("evidence_ids", [])
    row["invalid_evidence_ids"] = sorted(set(used_ids) - evidence_ids)
    row["evidence_reference_valid_fraction"] = _ratio(sum(item in evidence_ids for item in used_ids), len(used_ids))
    trace_path = prediction_root / "trace.jsonl"
    trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()] if trace_path.is_file() else []
    tools = [event for event in trace if event.get("event_type") == "tool_returned"]
    row["tool_results"] = len(tools)
    row["tool_errors"] = sum(event.get("status") in {"error", "failed", "unavailable"} for event in tools)
    row["tool_success_fraction"] = _ratio(sum(event.get("status") not in {"error", "failed", "unavailable"} for event in tools), len(tools))
    for key in (*COST_FIELDS, *MEMORY_FIELDS):
        row[key] = _measured_cost(costs, key)
    row["costs_incomplete"] = any(bool(node.get("incomplete") or node.get("unobserved_worker_cost"))
                                  for node in _usage_nodes(costs))
    row.update(cost_allocation(actual, mode, costs))
    row["pixels"] = _pixel_result(prediction, target, prediction_root, target_root)
    row["classes"] = {}
    for name, expected in (target or {}).get("class_masks", {}).items():
        candidate = prediction.get("class_masks", {}).get(name, {})
        row["classes"][str(name)] = _pixel_result(candidate, expected, prediction_root, target_root)
    location, boxes = prediction.get("localization"), (target or {}).get("boxes_xyxy")
    row["localization"] = {"status": "not_available"}
    if location and boxes:
        if not location.get("coordinate_system") or location["coordinate_system"] != target.get("coordinate_system"):
            row["localization"] = {"status": "not_evaluated", "reason": "coordinate_system_mismatch"}
        elif location.get("bbox_xyxy"):
            row["localization"] = {"status": "evaluated", "best_box_iou": max(box_iou(location["bbox_xyxy"], box) for box in boxes)}
    return row


def _summary(rows):
    known = [row for row in rows if row["change"] is not None]
    positive = [row for row in known if row["change"]]
    negative = [row for row in known if not row["change"]]
    decided = [row for row in rows if row["decided"]]
    tp = sum(row["decided"] and row["decision"] == "change" for row in positive)
    fp = sum(row["decided"] and row["decision"] == "change" for row in negative)
    fn = sum(row["decided"] and row["decision"] == "no_change" for row in positive)
    tn = sum(row["decided"] and row["decision"] == "no_change" for row in negative)
    result = {"cases": len(rows), "labeled_cases": len(known), "decided_cases": len(decided),
              "status_counts": dict(Counter(row["status"] for row in rows)),
              "decision_counts": dict(Counter(row["decision"] for row in rows)),
              "tp": tp, "fp": fp, "fn_decided": fn, "tn": tn,
              "unresolved_positive": len(positive) - tp - fn,
              "unresolved_negative": len(negative) - fp - tn,
              "coverage": _ratio(len(decided), len(rows)), "recall": _ratio(tp, len(positive)),
              "conditional_recall": _ratio(tp, tp + fn), "precision": _ratio(tp, tp + fp),
              "fpr": _ratio(fp, len(negative)), "conditional_fpr": _ratio(fp, fp + tn),
              "abstention_rate": _ratio(sum(row["status"] == "abstained" for row in rows), len(rows)),
              "error_rate": _ratio(sum(row["status"] in {"error", "failed", "missing"} for row in rows), len(rows))}
    result["invalid_evidence_references"] = sum(len(row.get("invalid_evidence_ids", [])) for row in rows)
    result["incomplete_cost_cases"] = sum(row.get("costs_incomplete", False) for row in rows)
    result["evidence_reference_valid_fraction"] = _mean([row.get("evidence_reference_valid_fraction") for row in rows])
    result["tool_results"] = sum(row.get("tool_results", 0) for row in rows)
    result["tool_errors"] = sum(row.get("tool_errors", 0) for row in rows)
    result["tool_success_fraction"] = _ratio(result["tool_results"] - result["tool_errors"], result["tool_results"])
    for key in (*COST_FIELDS, *ALLOCATION_FIELDS):
        present = [row[key] for row in rows if row[key] is not None]
        result[key] = sum(present) if present else None
        result[key + "_reported_cases"] = len(present)
    for key in MEMORY_FIELDS:
        present = [row[key] for row in rows if row[key] is not None]
        result[key] = max(present) if present else None
    pixels = [row["pixels"] for row in rows if row["pixels"]["status"] == "evaluated"]
    result["pixel_cases"] = len(pixels)
    result["pixel_iou"] = _mean([item["iou"] for item in pixels])
    result["pixel_f1"] = _mean([item["f1"] for item in pixels])
    locations = [row["localization"]["best_box_iou"] for row in rows if row["localization"]["status"] == "evaluated"]
    result["localized_cases"] = len(locations)
    result["region_box_iou"] = _mean(locations)
    return result


def aggregate(rows):
    """Case counts per building, then equal-building macro metrics, per split."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["run_path"], row["mode"], row["split"])].append(row)
    summaries, facades = [], []
    for (run_path, mode, split), members in sorted(groups.items()):
        buildings = defaultdict(list)
        for row in members:
            buildings[row["building_id"]].append(row)
        header = {"run_path": run_path, "run_id": Path(run_path).name, "mode": mode, "split": split}
        local = [{**header, "building_id": building, **_summary(values)} for building, values in sorted(buildings.items())]
        facades.extend(local)
        current = {**header, **_summary(members), "facades": len(local)}
        for key in ("recall", "precision", "fpr", "conditional_recall", "conditional_fpr", "coverage",
                    "abstention_rate", "error_rate", "pixel_iou", "pixel_f1", "region_box_iou",
                    "evidence_reference_valid_fraction", "tool_success_fraction"):
            current[key + "_macro"] = _mean([item[key] for item in local])
        summaries.append(current)
    return summaries, facades


def aggregate_classes(rows):
    groups = defaultdict(list)
    for row in rows:
        for name, values in row["classes"].items():
            if values["status"] == "evaluated":
                groups[(row["run_path"], row["mode"], row["split"], name)].append((row["building_id"], values))
    output = []
    for (run_path, mode, split, name), values in sorted(groups.items()):
        buildings = defaultdict(list)
        for building, value in values:
            buildings[building].append(value)
        row = {"run_path": run_path, "run_id": Path(run_path).name, "mode": mode, "split": split,
               "class_id": name, "cases": len(values), "facades": len(buildings)}
        for metric in ("precision", "recall", "f1", "iou", "fpr"):
            row[metric + "_macro"] = _mean([_mean([value[metric] for value in group]) for group in buildings.values()])
        output.append(row)
    return output


def _artifact_fingerprints(document, root, inputs):
    """Hash explicit mask/support inputs, not arbitrary filesystem references."""
    if isinstance(document, dict):
        for key, value in document.items():
            if key in {"mask_path", "support_path"} and value:
                path = _resolve(root, value)
                if path.is_file():
                    inputs[str(path)] = sha256(path)
            elif isinstance(value, (dict, list)):
                _artifact_fingerprints(value, root, inputs)
    elif isinstance(document, list):
        for value in document:
            _artifact_fingerprints(value, root, inputs)


def _csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row if not isinstance(row[key], (dict, list))))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def compare_runs(run_dirs, out, cases_path, labels_path=None):
    """Compare saved runs on an explicit identical cohort; labels are offline only."""
    manifest, fingerprint = _manifest(cases_path)
    roots = [Path(path).expanduser().resolve() for path in run_dirs]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("Provide distinct saved run directories")
    inputs = {str(Path(cases_path).resolve()): sha256(cases_path)}
    source_runs = []
    for root in roots:
        _verify_recorded_artifacts(root, inputs)
        _, actual = _manifest(root / "input_manifest.json")
        if actual != fingerprint:
            raise ValueError("Runs must share the exact fixed cases and availability policy")
        inputs[str(root / "input_manifest.json")] = sha256(root / "input_manifest.json")
        summary_path = root / "summary.json"
        source_summary = read_json(summary_path) if summary_path.is_file() else {}
        if summary_path.is_file():
            inputs[str(summary_path)] = sha256(summary_path)
        source_runs.append({"run_id": root.name, "run_path": str(root),
                            "actual_run_elapsed_seconds": _measured_cost(source_summary, "seconds"),
                            "cost_scope": source_summary.get("cost_scope", source_summary.get("scope")),
                            "cost_regime": source_summary.get("cost_regime")})
    labels, label_source = _labels(labels_path, {case["case_id"] for case in manifest["cases"]})
    if labels_path:
        inputs[str(Path(labels_path).resolve())] = sha256(labels_path)
    target_root = Path(labels_path).resolve().parent if labels_path else Path(cases_path).resolve().parent
    _artifact_fingerprints(list(labels.values()), target_root, inputs)
    for root in roots:
        for case in manifest["cases"]:
            base = root / "cases" / case["case_id"]
            paths = [base / "case.json", *[base / "modes" / mode / name for mode in MODES[1:]
                                          for name in ("result.json", "costs.json", "evidence.json", "trace.jsonl")]]
            for path in paths:
                if path.is_file():
                    inputs[str(path)] = sha256(path)
                    if path.suffix == ".json":
                        _artifact_fingerprints(read_json(path), path.parent, inputs)
    output = new_directory(out)
    config = {"runs": [str(root) for root in roots], "cohort_sha256": fingerprint,
              "availability_policy": manifest["availability_policy"], "input_sha256": inputs,
              "labels": label_source, "modes": list(MODES), "threshold_tuning": False}
    record = run_record("agentic_offline_comparison", config)
    write_json(output / "run.json", record)
    try:
        rows = [_row(root, case, mode, labels.get(case["case_id"]), target_root)
                for root in roots for case in manifest["cases"] for mode in MODES]
        summary, facades = aggregate(rows)
        classes = aggregate_classes(rows)
        result = {"schema_version": 1, "status": "completed", "cohort_sha256": fingerprint,
                  "availability_policy": manifest["availability_policy"], "labels": label_source,
                  "source_runs": source_runs,
                  "summary": summary, "facades": facades, "classes": classes, "cases": rows,
                  "interpretation": [
                      "Recall includes unresolved/missing positive cases in its denominator; conditional metrics use decided cases.",
                      "FPR is accompanied by coverage and conditional FPR: abstention is not a correct negative decision.",
                      "Macro metrics weight buildings equally; case-level counts are diagnostics.",
                      "Pixel masks require explicit temporal_change semantics, matching coordinates and known support.",
                      "Coverage describes the supplied cohort; proposal completeness is not proven unless the cohort includes missed candidates.",
                      "seconds measures baseline shared preparation or VLM incremental execution; these scopes differ and must not be compared as end-to-end latency.",
                      "shared_preparation_seconds is counted once per case in actual run cost. Context/agent allocation adds that preparation to the incremental mode; pair allocation adds only separately measured geometry. Allocations are alternatives, not summable run totals.",
                      "Costs use the recorded cache state. Original cached source usage is retained separately and is never charged again. A cold-start comparison requires a separately controlled run.",
                      "No thresholds or fusion were fitted; unknown costs are null, not zero."]}
        write_json(output / "comparison.json", result)
        write_json(output / "input_manifest.json", manifest)
        _csv(output / "comparison.csv", summary)
        _csv(output / "facades.csv", facades)
        _csv(output / "cases.csv", rows)
        _csv(output / "classes.csv", classes)
        from .report import render_comparison
        render_comparison(output, result)
        for path, expected in inputs.items():
            if sha256(path) != expected:
                raise ValueError(f"Comparison input changed during evaluation: {path}")
        finish_record(output, record, "completed")
        return result
    except BaseException as error:
        finish_record(output, record, "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", str(error))
        raise
