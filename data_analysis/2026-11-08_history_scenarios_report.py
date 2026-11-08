"""Rebuild exploratory history reports from float maps, using the canonical evaluator.

No model, encoder or optimizer is imported by this module. Count/overlap/AP semantics
come from the existing canonical-history evaluator. Selection deliberately follows
recall, then *lower FPR*, then higher grid threshold, as preregistered here.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import csv
from functools import lru_cache
import html
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np

from facade_change.io import read_json, sha256, write_json

metrics = importlib.import_module("facade_change.research.2026-11-04_history_benchmark_metrics")
PREFIX = "2026-11-08_"
DEFAULT_PRACTICAL = dict(f1_gain=.02, iou_gain=.015, maximum_fpr_increase=.005,
                         maximum_recall_loss=.01, relative_fpr_reduction=.20,
                         alternative_maximum_f1_loss=.005)


class InsufficientData(ValueError):
    """Known coverage does not permit the stated metric or VAL selection."""


def _canonical(dataset):
    return getattr(dataset, "evaluation_data", getattr(dataset, "canonical", getattr(getattr(dataset, "source", None), "data", dataset)))


def _mean(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def fixed_macro(rows):
    """Match the memory evaluator's fixed positive-building F1/IoU cohort."""
    result = {}
    for key in metrics.METRICS:
        eligible = [r for r in rows if r.get(key) is not None and
                    (key not in ("f1", "iou") or r.get("positive_pixels", 0) > 0)]
        result[key] = _mean(r[key] for r in eligible)
        result[key + "_buildings"] = len(eligible)
    for key in ("tp", "fp", "fn", "tn", "known_pixels", "positive_pixels", "negative_pixels"):
        result[key] = sum(int(r.get(key, 0)) for r in rows)
    result["buildings"] = sum(r.get("known_pixels", 0) > 0 for r in rows)
    return result


def select_threshold(buildings, budget=.05):
    """Select from canonical strict-score grid, with no undefined-metric fallback."""
    if not 0 <= budget < 1:
        raise ValueError("FPR budget must be a fraction in [0,1)")
    if not buildings:
        raise InsufficientData("No VAL buildings")
    recall, fpr = [], []
    for value in buildings.values():
        histogram = value[0] if value.ndim == 3 else value
        tp, _, _, _, fp = metrics.curves(histogram[None])
        positives = histogram[1].sum()
        recall.append(tp[0] / positives if positives else np.full(metrics.BINS, np.nan))
        fpr.append(fp[0])
    def column_mean(values):
        values = np.asarray(values)
        n = np.isfinite(values).sum(axis=0)
        return np.divide(np.nansum(values, axis=0), n, out=np.full(metrics.BINS, np.nan), where=n > 0)
    recall, fpr = column_mean(recall), column_mean(fpr)
    eligible = np.flatnonzero(np.isfinite(recall) & np.isfinite(fpr) & (fpr <= budget + 1e-12))
    if not len(eligible):
        raise InsufficientData("VAL needs known positive and negative pixels for FPR-constrained selection")
    index = max(eligible, key=lambda i: (recall[i], -fpr[i], i))
    return dict(threshold=int(index) / (metrics.BINS - 1), selection_split="val",
                criterion="maximum building-macro recall; minimum FPR; highest threshold",
                prediction_rule="score > threshold", val_fpr_budget=float(budget),
                calibration_recall=float(recall[index]), calibration_fpr=float(fpr[index]),
                val_budget_met=True, buildings=len(buildings))


def select_operating_points(histograms, budgets=(.05, .10, .15), frozen_thresholds=None):
    """Each budget changes only a VAL threshold; interventions may pin their parent's."""
    frozen_thresholds = frozen_thresholds or {}
    result = {}
    for budget in budgets:
        result[float(budget)] = {}
        for name, buildings in histograms.items():
            frozen = frozen_thresholds.get(str(budget), frozen_thresholds.get(budget, {})).get(name)
            if frozen is not None:
                value = dict(frozen) if isinstance(frozen, dict) else dict(threshold=float(frozen))
                rows = [metrics.metrics(h[0], value["threshold"]) for h in buildings.values()]
                macro = fixed_macro(rows)
                value.update(val_fpr_budget=budget, calibration_recall=macro["recall"],
                             calibration_fpr=macro["fpr"], val_budget_met=(macro["fpr"] is not None and
                             macro["fpr"] <= budget + 1e-12), threshold_origin="frozen parent checkpoint")
            else:
                value = select_threshold(buildings, budget)
            result[float(budget)][name] = value
    return result


def summarize_histograms(data, histograms, threshold, method="model", split="val", budget=.05):
    report = metrics.report(_canonical(data), histograms, threshold)
    rows = [dict(method=method, split=split, val_fpr_budget=budget, threshold=threshold, **r)
            for r in report["per_building"]]
    aggregates = []
    for kind, name, _ in metrics.slices(_canonical(data)):
        subset = [r for r in rows if r["slice_type"] == kind and r["slice"] == name]
        aggregates.append(dict(method=method, split=split, val_fpr_budget=budget, threshold=threshold,
                               slice_type=kind, slice=name, **fixed_macro(subset)))
    return aggregates, rows


def evaluate_predictor(dataset, predict, split="val", spec=None, budget=.05, threshold=None):
    """Streaming training callback: inference occurs before evaluator accesses labels."""
    data = (getattr(dataset, "canonical", _canonical(dataset)) if (spec or {}).get("original_full_support")
            or (spec or {}).get("id") == "stack-original" else _canonical(dataset))
    samples = [row for row in data.samples if row["split"] == split]
    next_row = {row["event_id"]: row for row in samples}
    @lru_cache(maxsize=24)
    def prediction(event_id):
        result = predict(next_row[event_id])
        if isinstance(result, tuple):
            result = dict(score=result[0], support=result[1])
        return result
    if (spec or {}).get("scenario") == "C" or (spec or {}).get("task") == "progression":
        data = _ProgressionData(data, lambda row: prediction(row["event_id"]))
    histograms, coverage = collect_results(data, samples, lambda row: prediction(row["event_id"]))
    selected = select_threshold(histograms, budget) if threshold is None else dict(threshold=float(threshold))
    aggregate, rows = summarize_histograms(data, histograms, selected["threshold"],
        method=(spec or {}).get("id", "model"), split=split, budget=budget)
    overall = next(r for r in aggregate if r["slice_type"] == "overall")
    return dict(**overall, prediction_score=None, per_building=rows, metrics=aggregate,
                coverage=coverage, operating_point=selected)


def paired_bootstrap(rows, method, control, metric="f1", repetitions=2000, seed=13):
    """Resample paired whole buildings, retaining undefined outcomes as missing."""
    if repetitions < 1:
        raise ValueError("Bootstrap repetitions must be positive")
    by_method = {name: {r["building_id"]: r for r in rows if r["method"] == name and
                       r.get("slice_type", "overall") == "overall"} for name in (method, control)}
    a, b = by_method[method], by_method[control]
    if set(a) != set(b):
        raise ValueError("Paired bootstrap requires identical building/event support cohorts")
    buildings = sorted(a)
    differences = []
    for building in buildings:
        left, right = a[building], b[building]
        if left.get("positive_pixels") != right.get("positive_pixels") or left.get("negative_pixels") != right.get("negative_pixels"):
            raise ValueError("Paired comparison targets/support differ")
        defined = (left.get(metric) is not None and right.get(metric) is not None and
                   (metric not in ("f1", "iou") or left.get("positive_pixels", 0) > 0))
        differences.append(left[metric] - right[metric] if defined else np.nan)
    values = np.asarray(differences, float)
    finite = np.isfinite(values)
    result = dict(method=method, control=control, metric=metric, buildings=len(buildings),
                  eligible_buildings=int(finite.sum()), delta=None, ci_low=None, ci_high=None,
                  bootstrap_supports_direction=None, repetitions=repetitions, training_seed=13,
                  interpretation="exploratory paired building bootstrap; inspected TEST; not training-seed variability",
                  per_building=[dict(building_id=name, delta=float(v) if np.isfinite(v) else None)
                                for name, v in zip(buildings, values)])
    if not finite.any():
        return result
    rng = np.random.default_rng(seed)
    draws = []
    # Bounded memory even when a dataset contains many buildings.
    for start in range(0, repetitions, 256):
        sample = values[rng.integers(0, len(values), (min(256, repetitions - start), len(values)))]
        counts = np.isfinite(sample).sum(axis=1)
        valid = counts > 0
        draws.extend((np.nansum(sample[valid], axis=1) / counts[valid]).tolist())
    low, high = np.quantile(draws, [.025, .975])
    delta = float(np.nanmean(values))
    result.update(delta=delta, ci_low=float(low), ci_high=float(high),
                  bootstrap_supports_direction=bool(low > 0) if delta >= 0 else bool(high < 0))
    return result


def practical_status(model, control, thresholds=None, predictive=False, prediction_improved=None,
                     execution_status="completed", reason=None):
    """Practical usefulness and CI direction are deliberately separate decisions."""
    if execution_status in ("error", "failed", "ERROR"):
        return "ERROR"
    if execution_status in ("insufficient_data", "INSUFFICIENT_DATA"):
        return "INSUFFICIENT_DATA"
    if execution_status not in ("completed", "COMPLETE", "complete"):
        return "INCOMPLETE"
    if not control or any(model.get(k) is None or control.get(k) is None for k in ("f1", "iou", "recall", "fpr")):
        return "INSUFFICIENT_DATA"
    if predictive and prediction_improved is None:
        return "INSUFFICIENT_DATA"
    limits = {**DEFAULT_PRACTICAL, **(thresholds or {})}
    delta = {key: model[key] - control[key] for key in ("f1", "iou", "recall", "fpr")}
    tolerance = 1e-12
    gain = ((delta["f1"] >= limits["f1_gain"] - tolerance or delta["iou"] >= limits["iou_gain"] - tolerance) and
            delta["fpr"] <= limits["maximum_fpr_increase"] + tolerance and delta["recall"] >= -limits["maximum_recall_loss"] - tolerance)
    reduce_fp = (control["fpr"] > 0 and model["fpr"] <= control["fpr"] * (1 - limits["relative_fpr_reduction"]) + tolerance and
                 delta["recall"] >= -limits["maximum_recall_loss"] - tolerance and delta["f1"] >= -limits["alternative_maximum_f1_loss"] - tolerance)
    if (gain or reduce_fp) and (not predictive or prediction_improved is True):
        return "PRACTICALLY_USEFUL"
    if delta["f1"] > 0 or delta["iou"] > 0 or reduce_fp:
        return "BELOW_PRACTICAL_THRESHOLD"
    return "NO_GAIN"


def _write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys or ["status"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "NA" if value is None else json.dumps(value, ensure_ascii=False)
                             if isinstance(value, (list, dict)) else value for key, value in row.items()})


def _number(value):
    return "NA" if value is None else f"{float(value):.5f}"


def _metadata(row):
    spec = row.get("spec", {})
    merged = {**spec, **row, "method": row.get("method", spec.get("id", row.get("id", "unknown")))}
    if "threshold_reference" in merged:
        merged["threshold_parent"] = merged["threshold_reference"]
    return merged


class SavedMapDataset:
    """Minimal read-only canonical adapter for report-only, never loading a model."""
    def __init__(self, root, manifest, categories, groups=()):
        self.root = Path(root)
        self.categories, self.groups = categories, list(groups)
        unique = {}
        for record in manifest:
            unique.setdefault(record["event_id"], record)
        self.samples = []
        self.records = unique
        for row in unique.values():
            required = ("building_id", "view_id", "previous_year", "current_year", "bbox_xywh", "split")
            if any(key not in row for key in required):
                raise ValueError("Saved maps lack original coordinates/year/split metadata")
            self.samples.append({**row, "known_pixels": row.get("known_pixels", 1)})
        self.canonical = self

    def target(self, event_id):
        record = self.records[event_id]
        with np.load(_map_path(self.root, record), allow_pickle=False) as values:
            fields = ("target", "support", "class_targets", "before", "after")
            if any(key not in values for key in fields):
                raise ValueError("Saved maps lack target/slice metadata for canonical report-only")
            return {key: values[key].copy() for key in fields}


def _map_path(root, record):
    path = Path(record["path"])
    path = path if path.is_absolute() else Path(root) / path
    expected = record.get("sha256", record.get("artifact_sha256"))
    if expected and sha256(path) != expected:
        raise ValueError(f"Saved map hash mismatch: {path}")
    return path


def _map_predictions(root, records):
    by_event = {row["event_id"]: row for row in records}
    if len(by_event) != len(records):
        raise ValueError("Duplicate method/split/event map")
    @lru_cache(maxsize=24)
    def load(event_id):
        with np.load(_map_path(root, by_event[event_id]), allow_pickle=False) as value:
            key = "score" if "score" in value else "scores"
            record = by_event[event_id]
            extras = {k: value[k].copy() for k in ("stack_raw", "stack_raw7", "fusion_signal_logit") if k in value}
            return dict(score=np.asarray(value[key], np.float32), support=np.asarray(value["support"], bool),
                        extras=extras, stack_contract=record.get("stack_contract"),
                        fusion_parameters=record.get("fusion_parameters"))
    return by_event, lambda row: load(row["event_id"])


def _prediction_diagnostics(root, records):
    grouped = defaultdict(lambda: defaultdict(lambda: [0., 0]))
    for record in records:
        with np.load(_map_path(root, record), allow_pickle=False) as arrays:
            target, support = arrays["target"], arrays["support"].astype(bool)
            if "forecast_support" in arrays:
                support = support & arrays["forecast_support"].astype(bool)
            for key in ("l1", "cosine", "U", "nll", "coverage90"):
                if key not in arrays:
                    continue
                value = np.asarray(arrays[key], float)
                if value.ndim == 3:
                    value = value.mean(axis=0)
                if value.shape != target.shape:
                    continue
                for label, mask in (("all", support), ("stable", support & (target == 0)),
                                    ("changed", support & (target == 1))):
                    valid = mask & np.isfinite(value)
                    stat = grouped[record["building_id"]][key + "_" + label]
                    stat[0] += float(value[valid].sum())
                    stat[1] += int(valid.sum())
    details = []
    for building, values in grouped.items():
        for metric, (total, count) in values.items():
            details.append(dict(building_id=building, metric=metric, value=total / count if count else None,
                                valid_pixel_crops=count,
                                aggregation="pixel-crop diagnostic within building; equal buildings for summary"))
    summary = {key: _mean(r["value"] for r in details if r["metric"] == key)
               for key in sorted({r["metric"] for r in details})}
    return summary, details


def _evaluation_data(dataset, root, records):
    """Use saved task targets if supplied, otherwise the existing canonical dataset."""
    data = _canonical(dataset)
    first = records[0]
    with np.load(_map_path(root, first), allow_pickle=False) as arrays:
        has_targets = all(key in arrays for key in ("target", "class_targets", "before", "after", "support"))
    if has_targets and all(key in first for key in ("bbox_xywh", "view_id", "previous_year", "current_year")):
        return SavedMapDataset(root, records, data.categories, data.groups)
    return data


def _same_cohort(root, left, right):
    """Fail closed when a claimed paired comparison removes different hard cases."""
    a, b = {r["event_id"]: r for r in left}, {r["event_id"]: r for r in right}
    if set(a) != set(b):
        raise ValueError("Compared methods have different event IDs")
    for event in a:
        with np.load(_map_path(root, a[event]), allow_pickle=False) as x, np.load(_map_path(root, b[event]), allow_pickle=False) as y:
            if any(not np.array_equal(x[key], y[key]) for key in ("target", "support")):
                raise ValueError(f"Compared methods have different target/support: {event}")
        for key in ("building_id", "view_id", "bbox_xywh", "previous_year", "current_year"):
            if key in a[event] and a[event].get(key) != b[event].get(key):
                raise ValueError(f"Compared map coordinates differ: {event}/{key}")


class _MappedBuildingHistograms(Mapping):
    """Read-only views of one method/split; no complete array is materialized."""
    def __init__(self, path, buildings):
        self._array = np.load(path, mmap_mode="r", allow_pickle=False)
        self._buildings = tuple(buildings)
        self._positions = {name: i for i, name in enumerate(buildings)}

    def __getitem__(self, building):
        return self._array[self._positions[building]]

    def __iter__(self):
        return iter(self._buildings)

    def __len__(self):
        return len(self._buildings)


class _HistogramSpool(Mapping):
    """Ephemeral, rebuildable histogram index; scientific inputs remain saved maps.

    Only filenames and building IDs survive between methods. Each collection
    uses the canonical evaluator's one-method memory, then writes building
    slices directly to a .npy mmap without allocating a stacked copy.
    """
    def __init__(self, root):
        self.root = Path(root)
        self.index = {}

    def __setitem__(self, key, histograms):
        buildings = tuple(sorted(histograms))
        if not buildings:
            self.index[key] = (None, buildings)
            return
        first = histograms[buildings[0]]
        name = hashlib.sha256(json.dumps(key).encode("utf-8")).hexdigest()
        path = self.root / (PREFIX + name + ".npy")
        mapped = np.lib.format.open_memmap(path, mode="w+", dtype=first.dtype,
                                          shape=(len(buildings), *first.shape))
        try:
            for i, building in enumerate(buildings):
                value = histograms[building]
                if value.shape != first.shape or value.dtype != first.dtype:
                    raise ValueError("Histogram shape/dtype differs between buildings")
                mapped[i] = value
            mapped.flush()
        finally:
            del mapped
        self.index[key] = (path, buildings)

    def __getitem__(self, key):
        path, buildings = self.index[key]
        return _MappedBuildingHistograms(path, buildings) if path is not None else {}

    def __contains__(self, key):
        return key in self.index

    def __iter__(self):
        return iter(self.index)

    def __len__(self):
        return len(self.index)


def build_report(dataset, manifest, out, config, records):
    """Bound report histogram RAM to one method/split; clean scratch on any exit."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=PREFIX + "histograms-", dir=out) as scratch:
        return _build_report(dataset, manifest, out, config, records, _HistogramSpool(scratch))


def _build_report(dataset, manifest, out, config, records, histograms):
    """Rebuild one main table and details from existing float maps, without inference.

    ``manifest`` is a list of per-method/per-event NPZ references. Each record
    retains canonical event geometry and hashes. ``records`` are training/stage
    records with id/method, status and resource measurements, including failures.
    Missing models remain visible as INCOMPLETE; no number is imputed.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if "practical" in config and "practical_thresholds" in config and config["practical"] != config["practical_thresholds"]:
        raise ValueError("Conflicting practical-threshold configuration aliases")
    if "bootstrap_repeats" in config and "bootstrap_repetitions" in config and config["bootstrap_repeats"] != config["bootstrap_repetitions"]:
        raise ValueError("Conflicting bootstrap repetition configuration aliases")
    manifest = [_metadata(row) for row in manifest]
    records = [_metadata(row) for row in (records or [])]
    metadata = {row["method"]: row for row in records}
    for row in manifest:
        metadata.setdefault(row["method"], row)
    matrix = config.get("matrix", config.get("resolved_matrix", []))
    if isinstance(matrix, dict):
        matrix = matrix.get("variants", [])
    for row in matrix:
        item = _metadata(row)
        metadata.setdefault(item["method"], {**item, "status": "incomplete"})
    if dataset is None:
        ontology = config.get("ontology", {})
        categories = config.get("categories", ontology.get("categories", []))
        groups = config.get("groups", ontology.get("groups", []))
        if not categories and manifest:
            raise ValueError("report-only requires the saved ontology categories and channel order")
        dataset = SavedMapDataset(out, manifest, categories, groups)
    data = _canonical(dataset)
    grouped = defaultdict(list)
    for row in manifest:
        grouped[row["method"], row["split"]].append(row)
    budgets = tuple(config.get("fpr_budgets", (.05, .10, .15)))
    if .05 not in budgets:
        raise ValueError("Main table requires the preregistered 5% VAL FPR budget")
    coverage, predictions, prediction_rows = {}, {}, []
    errors = {}
    frozen = config.get("frozen_thresholds", {})
    frozen = {str(k): dict(v) for k, v in frozen.items()}
    existing_stack = getattr(dataset, "frozen", None)
    if existing_stack is not None:
        frozen.setdefault("0.05", {})["stack-original"] = dict(existing_stack.fpr_threshold)
    decisions_path = out / (PREFIX + "frozen_val_decisions.json")
    if decisions_path.exists():
        saved_decisions = read_json(decisions_path)
        for budget in budgets:
            for method, decision in saved_decisions.get(str(budget), {}).items():
                if decision.get("threshold") is not None:
                    frozen.setdefault(str(budget), {})[method] = dict(decision,
                        calibration_recall=decision.get("recall"), calibration_fpr=decision.get("fpr"),
                        threshold_origin="frozen before TEST")
    for (method, split), maps in sorted(grouped.items()):
        current = _evaluation_data(dataset, out, maps)
        by_event, prediction = _map_predictions(out, maps)
        samples = [s for s in current.samples if s["split"] == split]
        if set(by_event) != {row["event_id"] for row in samples}:
            raise ValueError(f"Saved maps incomplete for fixed cohort: {method}/{split}")
        hist, covered = collect_results(current, samples, prediction)
        histograms[method, split], coverage[method, split] = hist, covered
        if method == "stack-original" and existing_stack is not None:
            existing_stack.parity(hist, split)
            covered["baseline_original_counts_parity"] = "passed"
        summary, details = _prediction_diagnostics(out, maps)
        predictions[method, split] = summary
        prediction_rows.extend(dict(method=method, split=split, **row) for row in details)
        del hist
    operating = {budget: {} for budget in budgets}
    # Interventions use their original checkpoint's threshold, even on VAL.
    ordering = sorted(metadata, key=lambda method: bool(metadata[method].get("threshold_parent")))
    for method in ordering:
        if (method, "val") not in histograms:
            continue
        for budget in budgets:
            try:
                parent = metadata[method].get("threshold_parent")
                if parent:
                    if parent not in operating[budget]:
                        raise ValueError("Intervention threshold parent is missing")
                    selected = dict(operating[budget][parent], threshold_origin=parent,
                                    intervention_threshold_frozen=True)
                elif method in frozen.get(str(budget), {}):
                    selected = dict(frozen[str(budget)][method])
                    selected.update(threshold_origin="saved frozen baseline", val_fpr_budget=budget)
                else:
                    selected = select_threshold(histograms[method, "val"], budget)
                operating[budget][method] = selected
            except InsufficientData as exc:
                errors[method] = str(exc)
    aggregates, per_building = [], []
    # Sort descriptor keys only: sorting .items() would retain every mmap/view.
    for method, split in sorted(histograms):
        hist = histograms[method, split]
        for budget in budgets:
            if method not in operating[budget]:
                continue
            current = _evaluation_data(dataset, out, grouped[method, split])
            selected = operating[budget][method]
            macro, rows = summarize_histograms(current, hist, selected["threshold"], method, split, budget)
            target_type = ("annotation-derived latest change (semantic memory downstream)"
                if metadata.get(method, {}).get("scenario") == "A" else
                metadata.get(method, {}).get("target_type", "annotation-derived latest change"))
            for row in (*macro, *rows):
                row.update(target_type=target_type, overall_val_recall=selected.get("calibration_recall"),
                           overall_val_fpr=selected.get("calibration_fpr"))
            aggregates.extend(macro)
            per_building.extend(rows)
    overall = {(r["method"], r["split"], r["val_fpr_budget"]): r for r in aggregates if r["slice_type"] == "overall"}
    def val_key(method):
        val = overall.get((method, "val", .05), {})
        if val.get("recall") is None or val.get("fpr") is None:
            return (-float("inf"), -float("inf"), method)
        return (val["recall"], -val["fpr"], method)
    def task(method):
        return metadata[method].get("task", metadata[method].get("mode", "direct"))
    def no_history(method):
        return metadata[method].get("control", "history") in (
            "pair-only", "last-only", "repeat-history", "repeat", "copy-last", "pair", "independent",
            "regularization", "regularized", "last-pair", "first-current", "latest", "first", "none") or method == "stack"
    def output_kind(method):
        if method.endswith("-residual+stack"):
            return "residual-fusion"
        if method.endswith("+stack"):
            return "fusion"
        if method.endswith("-residual"):
            return "residual"
        if method.endswith("-prediction-best"):
            return "profile"
        return "standalone"
    selected_main = set()
    families = defaultdict(list)
    for method in metadata:
        meta = metadata[method]
        if meta.get("selected_main"):
            selected_main.add(method)
        elif (not no_history(method) and not meta.get("threshold_parent") and not meta.get("ablation")
              and meta.get("checkpoint_selection") != "prediction" and "-residual" not in method
              and meta.get("control", "history") in ("history", "learned")):
            families[task(method), meta.get("family", "scenario"), output_kind(method)].append(method)
    for methods in families.values():
        available = [m for m in methods if (m, "val", .05) in overall]
        selected_main.add(max(available, key=val_key) if available else sorted(methods)[0])
    # Keep an intelligible baseline even before the first training stage completes.
    if "stack" in metadata:
        selected_main.add("stack")
    if not selected_main:
        selected_main.update(metadata)
    comparisons, details = [], []
    for method in sorted(metadata):
        meta = metadata[method]
        explicit_control = meta.get("comparison_control", meta.get("reference"))
        candidates = [m for m in metadata if m != method and m != "stack-original" and task(m) == task(method) and no_history(m)
                      and output_kind(m) == output_kind(method) and metadata[m].get("scenario") == meta.get("scenario")
                      and not metadata[m].get("threshold_parent") and (m, "val", .05) in overall]
        # Primary control is the strongest corresponding no-history candidate chosen on VAL.
        family_candidates = [m for m in candidates if metadata[m].get("family") == meta.get("family")]
        candidates = family_candidates or candidates
        same_size = [m for m in candidates if metadata[m].get("size") == meta.get("size")]
        candidates = same_size or candidates
        control = (None if method in ("stack", "stack-original") else
                   explicit_control or (max(candidates, key=val_key) if candidates else None))
        current = overall.get((method, "test", .05), {})
        reference = overall.get((control, "test", .05), {})
        execution = meta.get("status", "completed" if current else "incomplete")
        reason = meta.get("reason", errors.get(method))
        if not current and execution == "completed" and method not in errors:
            execution, reason = "incomplete", "Training completed; frozen TEST export/evaluation not completed"
        if method in errors:
            execution = "insufficient_data"
        paired = None
        if current and reference and control != method:
            _same_cohort(out, grouped[method, "test"], grouped[control, "test"])
            rows = [r for r in per_building if r["split"] == "test" and r["val_fpr_budget"] == .05]
            for metric in ("f1", "iou", "recall", "fpr"):
                value = paired_bootstrap(rows, method, control, metric,
                                         repetitions=config.get("bootstrap_repeats", config.get("bootstrap_repetitions", 2000)), seed=13)
                comparisons.append(value)
                if metric == "f1":
                    paired = value
        predictive = task(method) in ("actual-next", "h0", "actual_next", "stable-state")
        profile_key = "U_stable" if task(method) in ("h0", "stable-state") else "U_all"
        profile_method = method[:-6] if method.endswith("+stack") else method
        profile_control = str(control)[:-6] if str(control).endswith("+stack") else str(control)
        prediction_model = predictions.get((profile_method + "-prediction-best", "test"),
            predictions.get((method, "test"), {})).get(profile_key)
        learned_last = f"{task(method)}-{meta.get('family')}-{meta.get('size')}-last-only"
        copy_last = task(method) + "-copy-last"
        prediction_reference = predictions.get((learned_last + "-prediction-best", "test"),
            predictions.get((learned_last, "test"), {})).get(profile_key)
        copy_prediction = predictions.get((copy_last, "test"), {}).get(profile_key)
        prediction_improved = (prediction_model < prediction_reference and prediction_model < copy_prediction
            if prediction_model is not None and prediction_reference is not None and copy_prediction is not None else None)
        status = practical_status(current, reference, config.get("practical", config.get("practical_thresholds")), predictive,
                                  prediction_improved, execution, reason)
        if method in ("stack", "stack-original") and current:
            status = "BASELINE"
        prediction_text = "N/A" if not predictive else (
            f"{_number(prediction_model)}; {learned_last}={_number(prediction_reference)}; "
            f"{copy_last}={_number(copy_prediction)} bits/component")
        details.append(dict(task=task(method), model=method, family=meta.get("family"), size=meta.get("size"),
            target_type=meta.get("target_type", "annotation-derived latest change"), control=control,
            f1=current.get("f1"), iou=current.get("iou"), precision=current.get("precision"),
            recall=current.get("recall"), fpr=current.get("fpr"), ap_binned=current.get("ap_binned"),
            buildings=current.get("buildings"), known_pixels=current.get("known_pixels"),
            delta_f1=paired.get("delta") if paired else None,
            ci_low=paired.get("ci_low") if paired else None, ci_high=paired.get("ci_high") if paired else None,
            bootstrap_supports_direction=paired.get("bootstrap_supports_direction") if paired else None,
            prediction_vs_control=prediction_text, prediction_improved=prediction_improved if predictive else None,
            prediction_nll=prediction_model if predictive else None,
            learned_last_only_nll=prediction_reference if predictive else None,
            copy_last_nll=copy_prediction if predictive else None,
            forecasting_reference=learned_last if predictive else None,
            seconds=meta.get("seconds", meta.get("training_seconds", meta.get("train_seconds"))), parameters=meta.get("parameters"),
            optimizer_steps=meta.get("optimizer_steps"), peak_vram_bytes=meta.get("peak_vram_bytes"),
            status=status, reason=reason, selected_on="VAL", exploratory=True,
            prediction_checkpoint=meta.get("prediction_checkpoint"), detection_checkpoint=meta.get("detection_checkpoint")))
    named_comparisons = []
    selected_rows = [r for r in per_building if r["split"] == "test" and r["val_fpr_budget"] == .05]
    for method, control, factor in planned_comparisons(metadata, overall):
        _same_cohort(out, grouped[method, "test"], grouped[control, "test"])
        p, q = metadata[method].get("parameters"), metadata[control].get("parameters")
        for metric in ("f1", "iou", "recall", "fpr"):
            comparison = paired_bootstrap(selected_rows, method, control, metric,
                repetitions=config.get("bootstrap_repeats", config.get("bootstrap_repetitions", 2000)), seed=13)
            comparison.update(changed_factor=factor, parameters=p, control_parameters=q,
                relative_parameter_gap=abs(p-q)/max(p,q) if p and q else None,
                equal_compute_claim=False)
            named_comparisons.append(comparison)
    semantic, semantic_buildings = semantic_state_metrics(dataset, out, manifest)
    for row in details:
        if row["task"] == "semantic-state":
            row["task"] = "A-downstream-change"
            row["target_type"] = "annotation-derived latest change (semantic memory downstream)"
    main = [row for row in details if row["model"] in selected_main]
    main.extend(_semantic_main_rows(semantic, semantic_buildings, metadata,
        config.get("bootstrap_repeats", config.get("bootstrap_repetitions", 2000)),
        config.get("practical", config.get("practical_thresholds"))))
    if not main:
        main = [dict(task="matrix", model="not-started", control=None, f1=None, iou=None, recall=None,
                     fpr=None, delta_f1=None, ci_low=None, ci_high=None, prediction_vs_control="N/A",
                     seconds=None, status="INCOMPLETE", reason="No completed model/maps")]
    cohorts = {}
    for method in metadata:
        current = grouped.get((method, "test"), [])
        lengths = [int(r.get("observations", r.get("history_length", r.get("frame", 2) + 1))) for r in current]
        cohorts[method] = dict(all=len(current), at_least_three_observations=sum(n >= 3 for n in lengths),
                               cohorts_identical=bool(current) and all(n >= 3 for n in lengths))
    summary = dict(schema_version=1, training_seeds=[13], exploratory=True, main_table=main,
                   detailed_table=details, thresholds={str(k): v for k, v in operating.items()},
                   coverage={"/".join(k): v for k, v in coverage.items()}, cohorts=cohorts,
                   prediction_diagnostics={"/".join(k): v for k, v in predictions.items()},
                   interpretation="Synthetic checks validate implementation, not real-facade history benefit. "
                   "Previously inspected TEST is exploratory. Bootstrap is across buildings, not training seeds.",
                   class_interpretation="Change localization within annotated classes; not semantic recognition",
                   model_bits="Quantized-latent model bits/component; no measured compressed stream or RGB bpp")
    if manifest and config.get("gallery_limit", 4) > 0:
        gallery_methods = sorted(selected_main | {"stack"})
        summary["gallery"] = build_gallery(out, manifest, operating[.05], limit=config.get("gallery_limit", 4), methods=gallery_methods)
    _write_csv(out / (PREFIX + "main_table.csv"), main)
    _write_csv(out / (PREFIX + "detailed_table.csv"), details)
    _write_csv(out / (PREFIX + "metrics.csv"), aggregates)
    _write_csv(out / (PREFIX + "per_building.csv"), per_building)
    _write_csv(out / (PREFIX + "comparisons.csv"), comparisons)
    _write_csv(out / (PREFIX + "ablation_comparisons.csv"), named_comparisons)
    _write_csv(out / (PREFIX + "prediction_metrics.csv"), prediction_rows)
    _write_csv(out / (PREFIX + "semantic_metrics.csv"), semantic)
    _write_csv(out / (PREFIX + "semantic_per_building.csv"), semantic_buildings)
    write_json(out / (PREFIX + "main_table.json"), main)
    write_json(out / (PREFIX + "report.json"), summary)
    write_json(out / (PREFIX + "thresholds.json"), summary["thresholds"])
    write_json(out / (PREFIX + "report_inputs.json"), dict(categories=data.categories, groups=data.groups,
               config=config, records=records, manifest=manifest))
    columns = ("task", "target_type", "model", "control", "f1", "iou", "recall", "fpr", "delta_f1_ci",
               "prediction_vs_control", "seconds", "status")
    lines = ["ГЛАВНАЯ ТАБЛИЦА — exploratory TEST, пороги по VAL FPR ≤ 5%, seed=13", "",
             " | ".join(columns)]
    for row in main:
        display = dict(row)
        display["delta_f1_ci"] = f"{_number(row.get('delta_f1'))} [{_number(row.get('ci_low'))}, {_number(row.get('ci_high'))}]"
        for key in ("f1", "iou", "recall", "fpr", "seconds"):
            display[key] = _number(row.get(key))
        lines.append(" | ".join(str(display.get(key) if display.get(key) is not None else "NA") for key in columns))
    lines += ["", summary["interpretation"], summary["class_interpretation"], summary["model_bits"],
              "Semantic state and progression proxy targets form separate tasks; their F1 values are not a shared ranking.",
              "F1/IoU: fixed GT-positive building cohort. FPR: all buildings with known negatives. Undefined metrics: NA.",
              "Status and bootstrap direction are separate fields. Main choices use VAL only; full details preserve all variants."]
    text = "\n".join(lines) + "\n"
    (out / (PREFIX + "report.txt")).write_text(text, encoding="utf-8")
    print(text)
    return summary


def report_only(out):
    """Public no-inference reconstruction entrypoint."""
    out = Path(out)
    path = out / (PREFIX + "report_inputs.json")
    if not path.exists():
        config_path = out / (PREFIX + "resolved_config.json")
        config = read_json(config_path) if config_path.exists() else {}
        matrix_path = out / (PREFIX + "resolved_matrix.json")
        if matrix_path.exists():
            config["matrix"] = read_json(matrix_path)
        records_path = out / (PREFIX + "training_records.json")
        records = read_json(records_path) if records_path.exists() else []
        if isinstance(records, dict):
            records = records.get("records", list(records.values()))
        maps_path = out / (PREFIX + "maps_manifest.json")
        manifest = []
        if maps_path.exists():
            saved = read_json(maps_path)
            manifest = saved.get("maps", [])
            config.update(categories=saved.get("categories", []), groups=saved.get("groups", []))
        return build_report(None, manifest, out, config, records)
    inputs = read_json(path)
    config = {**inputs["config"], "categories": inputs["categories"], "groups": inputs["groups"]}
    return build_report(None, inputs["manifest"], out, config, inputs["records"])


def _display_map(value, low, high, support=None):
    from PIL import Image
    value = np.asarray(value, float)
    if value.ndim == 3:
        value = value.mean(axis=0)
    normalized = np.clip((value - low) / max(high - low, 1e-12), 0, 1)
    rgb = np.stack((normalized, 1 - np.abs(2 * normalized - 1), 1 - normalized), axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.)
    if support is not None and np.asarray(support).shape == normalized.shape:
        rgb[~np.asarray(support, bool)] = (.25, .25, .25)
    return Image.fromarray(np.uint8(rgb * 255)).resize((224, 224), Image.Resampling.NEAREST)


def build_gallery(out, manifest, thresholds, limit=4, methods=None):
    """Fixed common examples plus explicit failure gallery; TRAIN/VAL-only scales.

    RGB is shown only when it was actually saved/referenced. Missing RGB or
    optional signals are listed, never synthesized or independently stretched.
    """
    from PIL import Image, ImageDraw
    out = Path(out)
    manifest = [_metadata(row) for row in manifest]
    if methods is not None:
        manifest = [row for row in manifest if row["method"] in set(methods)]
    keys = ("score", "l1", "cosine", "U", "G", "semantic_before", "semantic_after", "semantic_probability", "progression")
    samples = defaultdict(list)
    for row in sorted(manifest, key=lambda r: (r["method"], r["event_id"])):
        if row["split"] not in ("train", "val"):
            continue
        with np.load(_map_path(out, row), allow_pickle=False) as arrays:
            for key in keys:
                if key not in arrays or key == "score":
                    continue
                value = np.asarray(arrays[key], float).ravel()
                value = value[np.isfinite(value)]
                # Fixed deterministic bounded sampling per artifact, common across all methods.
                if len(value):
                    samples[key].extend(value[np.linspace(0, len(value) - 1, min(128, len(value)), dtype=int)].tolist())
                if len(samples[key]) > 32768:
                    samples[key] = samples[key][::2]
    scales = dict(score=[0., 1.])
    for key, value in samples.items():
        low, high = np.quantile(value, [.01, .99])
        if key == "G":
            radius = max(abs(low), abs(high), 1e-8)
            low, high = -radius, radius
        scales[key] = [float(low), float(max(high, low + 1e-8))]
    test = [r for r in manifest if r["split"] == "test"]
    by_event = defaultdict(list)
    for row in test:
        by_event[row["event_id"]].append(row)
    # Hash order is deterministic and independent of model scores/targets.
    import hashlib
    fixed = sorted(by_event, key=lambda event: hashlib.sha256(("13:" + event).encode()).hexdigest())[:limit]
    errors = []
    for event, rows in by_event.items():
        score = 0.
        for row in rows:
            selected = thresholds.get(row["method"])
            if selected is None:
                continue
            with np.load(_map_path(out, row), allow_pickle=False) as arrays:
                if "score" not in arrays:
                    continue
                valid = arrays["support"].astype(bool) & (arrays["target"] != 255)
                wrong = (arrays["score"] > selected["threshold"]) != (arrays["target"] == 1)
                score = max(score, float(np.mean(wrong[valid])) if valid.any() else 0.)
        errors.append((score, event))
    failures = [event for _, event in sorted(errors, key=lambda item: (-item[0], item[1]))[:limit]]
    gallery = out / (PREFIX + "gallery")
    gallery.mkdir(exist_ok=True)
    entries = []
    for role, events in (("fixed", fixed), ("failures", failures)):
        for number, event in enumerate(events):
            panels, missing = [], []
            rows = sorted(by_event[event], key=lambda r: (r["method"] not in ("stack", "stack-original"), r["method"]))
            for row in rows:
                with np.load(_map_path(out, row), allow_pickle=False) as arrays:
                    support = arrays["support"].astype(bool)
                    if row is rows[0]:
                        if "rgb_history" in arrays or "history_rgb" in arrays:
                            history = arrays["rgb_history"] if "rgb_history" in arrays else arrays["history_rgb"]
                            dates = arrays["years"] if "years" in arrays else row.get("dates", row.get("years", []))
                            for i, rgb in enumerate(history):
                                rgb = np.asarray(rgb, np.uint8)
                                if rgb.shape[0] == 3 and rgb.shape[-1] != 3:
                                    rgb = np.moveaxis(rgb, 0, -1)
                                label = str(dates[i]) if i < len(dates) else "date unavailable"
                                panels.append(("RGB " + label, Image.fromarray(rgb).resize((224, 224))))
                        else:
                            missing.append("RGB history not saved")
                        target = arrays["target"]
                        truth = np.full((*target.shape, 3), (120, 120, 120), np.uint8)
                        truth[target == 0], truth[target == 1] = (0, 0, 0), (255, 255, 255)
                        panels.append(("annotation-derived GT; unknown=gray", Image.fromarray(truth).resize((224, 224))))
                    for key in keys:
                        if key not in arrays:
                            continue
                        if key not in scales:
                            missing.append(key + " lacks TRAIN/VAL scale")
                            continue
                        low, high = scales[key]
                        panels.append((f"{row['method']} {key} [{low:.3g},{high:.3g}]",
                                       _display_map(arrays[key], low, high, support)))
                    selected = thresholds.get(row["method"])
                    if selected is not None and "score" in arrays:
                        predicted = arrays["score"] > selected["threshold"]
                        target = arrays["target"]
                        valid = support & (target != 255)
                        overlay = np.full((*target.shape, 3), (100, 100, 100), np.uint8)
                        overlay[valid] = (0, 0, 0)
                        overlay[valid & predicted & (target == 1)] = (0, 200, 0)
                        overlay[valid & predicted & (target == 0)] = (255, 40, 40)
                        overlay[valid & ~predicted & (target == 1)] = (40, 80, 255)
                        panels.append((row["method"] + " TP green FP red FN blue",
                                       Image.fromarray(overlay).resize((224, 224))))
            if not panels:
                continue
            columns, width, height = 4, 280, 256
            canvas = Image.new("RGB", (columns * width, ((len(panels) + columns - 1) // columns) * height), "white")
            draw = ImageDraw.Draw(canvas)
            for i, (label, picture) in enumerate(panels):
                x, y = (i % columns) * width, (i // columns) * height
                draw.text((x + 2, y + 2), label[:45], fill="black")
                canvas.paste(picture, (x, y + 24))
            name = PREFIX + f"{role}-{number:03d}.png"
            canvas.save(gallery / name)
            entries.append(dict(role=role, event_id=event, path=name, missing=sorted(set(missing))))
    write_json(gallery / (PREFIX + "manifest.json"), dict(scales=scales, scale_source="TRAIN/VAL only; common per signal",
        fixed_policy="SHA256(13:event_id), independent of scores and targets",
        failure_policy="largest per-event maximum method error fraction at frozen VAL threshold; diagnostic only",
        entries=entries))
    page = ["<!doctype html><meta charset='utf-8'><title>History experiment gallery</title>",
            "<h1>Fixed examples and failure diagnostics</h1>",
            "<p>Common scales per signal are fitted on TRAIN/VAL. Unknown is gray; no per-image scaling.</p>"]
    for entry in entries:
        page.append(f"<h2>{html.escape(entry['role'])}: {html.escape(entry['event_id'])}</h2>"
                    f"<p>{html.escape('; '.join(entry['missing']))}</p><img style='max-width:100%' src='{entry['path']}'>")
    (gallery / (PREFIX + "index.html")).write_text("\n".join(page), encoding="utf-8")
    return dict(path=str(gallery / (PREFIX + "index.html")), examples=len(entries), scales=scales)


class _ProgressionData:
    """Task-specific evaluation labels; accessed strictly after prediction."""
    def __init__(self, data, predict):
        self.data, self.predict = data, predict
        self.samples, self.categories, self.groups = data.samples, data.categories, data.groups
        self.rows = {row["event_id"]: row for row in self.samples}

    def target(self, event_id):
        result = self.predict(self.rows[event_id])
        extras = result.get("extras", {})
        if "progression_target" not in extras:
            raise ValueError("Progression evaluation requires its own annotation-derived proxy target")
        arrays = dict(self.data.target(event_id))
        target = np.asarray(extras["progression_target"])
        arrays["target"] = np.where(target < 0, 255, target).astype(np.uint8)
        arrays["support"] = np.asarray(arrays["support"], bool) & (target >= 0) & (target != 255)
        arrays["class_targets"] = np.full_like(arrays["class_targets"], 255)
        return arrays


def _stack_logits(raw, contract):
    raw = np.asarray(raw, np.float64)
    if contract.get("kind") == "synthetic_identity":
        probability = np.clip(raw[0], 1e-8, 1 - 1e-8)
        return np.log(probability) - np.log1p(-probability)
    model = contract.get("model", contract)
    normalizer = contract["normalizer"]
    low, high = np.asarray(normalizer["low"]), np.asarray(normalizer["high"])
    shape = raw.shape[1:]
    values = raw.reshape(raw.shape[0], -1).T
    width = high - low
    normalized = np.clip((values - low) / np.maximum(width, 1e-8), 0, 1)
    normalized[:, width < 1e-8] = 0
    parameters = np.asarray(model["parameters"])
    logits = parameters[0] + ((normalized - np.asarray(model["center"])) / np.asarray(model["scale"])) @ parameters[1:]
    return logits.reshape(shape)


def collect_results(data, samples, predict):
    """Canonical overlap merge, including frozen stack's original raw-map order."""
    if not samples:
        return {}, dict(crop_candidates=0, known_pixels=0, geometric_pixels=0)
    first = predict(samples[0])
    if isinstance(first, tuple):
        return metrics.collect(data, samples, predict)
    extras = first.get("extras", {})
    raw_key = "stack_raw" if "stack_raw" in extras else "stack_raw7" if "stack_raw7" in extras else None
    contract = first.get("stack_contract", extras.get("stack_contract"))
    if raw_key is None or contract is None:
        return metrics.collect(data, samples, lambda row: (
            np.asarray(predict(row).get("score", predict(row).get("scores")), np.float32),
            np.asarray(predict(row)["support"], bool)))
    n = len(extras[raw_key])
    fusion = first.get("fusion_parameters", extras.get("fusion_parameters"))
    flags = [name == "anychange" for name in contract.get("method_names", [""] * n)]
    if len(flags) != n:
        raise ValueError("Frozen stack channel contract mismatch")
    if fusion is not None:
        flags.append(False)
    def channels(row):
        result = predict(row)
        current = result.get("extras", {})
        raw = np.asarray(current[raw_key], np.float32)
        if fusion is not None:
            signal = np.asarray(current["fusion_signal_logit"], np.float32)
            raw = np.concatenate((raw, signal[None]))
        return raw, np.asarray(result["support"], bool)
    histograms = {}
    coverage = dict(crop_candidates=len(samples), known_pixels=0, geometric_pixels=0,
                    conflicting_overlap_pixels=0, stack_merge="raw detector maps before nonlinear stack transform")
    for key, _, values, truth, geometric, conflict in metrics.merged_blocks(
            data, samples, channels, flags, logit_channels=[n] if fusion is not None else []):
        logits = _stack_logits(values[:n], contract)
        if fusion is not None:
            weights = fusion.get("weight", fusion.get("weights"))
            logits = float(fusion["bias"]) + float(weights[0]) * logits + float(weights[1]) * values[n]
        score = np.exp(-np.logaddexp(0., -logits))
        histogram = histograms.setdefault(key[0], np.zeros((len(metrics.slices(data)), 2, metrics.BINS), np.int64))
        coverage["geometric_pixels"] += int(geometric.sum())
        coverage["known_pixels"] += int(((truth[0] != 255) & geometric).sum())
        coverage["conflicting_overlap_pixels"] += int((conflict[0] & geometric).sum())
        for channel, target in enumerate(truth):
            known = (target != 255) & geometric
            quantized = np.ceil(score[known] * (metrics.BINS - 1)).astype(np.int64)
            for label in (0, 1):
                histogram[channel, label] += np.bincount(quantized[target[known] == label], minlength=metrics.BINS)
    coverage["known_fraction"] = coverage["known_pixels"] / coverage["geometric_pixels"] if coverage["geometric_pixels"] else None
    return histograms, coverage


class _SemanticClassData:
    def __init__(self, base, root, records, channel):
        self.categories = [base.categories[channel]]
        self.groups = []
        # Current state is one observation, regardless of which earlier frame
        # supplies a temporal pair in a track with missing observations.
        self.samples = [dict(row, previous_year=row["current_year"]) for row in base.samples]
        self.channel, self.root = channel, root
        self.records = {r["event_id"]: r for r in records}

    @lru_cache(maxsize=24)
    def values(self, event):
        with np.load(_map_path(self.root, self.records[event]), allow_pickle=False) as arrays:
            return {key: arrays[key].copy() for key in ("semantic_probability", "semantic_target", "semantic_known", "support")}

    def target(self, event):
        arrays = self.values(event)
        known = arrays["semantic_known"].astype(bool)
        if known.ndim == 3:
            known = known[self.channel] if known.shape[0] > 1 else known[0]
        target = arrays["semantic_target"][self.channel].astype(np.uint8)
        target = np.where(known, target, 255).astype(np.uint8)
        return dict(target=target, support=arrays["support"].astype(bool) & known,
                    class_targets=target[None], before=np.ones(target.shape, np.uint32),
                    after=np.ones(target.shape, np.uint32))


def semantic_state_metrics(dataset, root, manifest):
    """Per-class multilabel state metrics at fixed .5, after spatial overlap merge."""
    grouped = defaultdict(list)
    for row in manifest:
        if row.get("scenario", row.get("spec", {}).get("scenario")) == "A" and not row["method"].endswith("+stack"):
            grouped[row["method"], row["split"]].append(row)
    per_building, macro = [], []
    for (method, split), records in sorted(grouped.items()):
        base = _evaluation_data(dataset, root, records)
        samples = [r for r in base.samples if r["split"] == split]
        for channel, category in enumerate(base.categories):
            current = _SemanticClassData(base, root, records, channel)
            def prediction(row):
                value = current.values(row["event_id"])
                return value["semantic_probability"][channel], value["support"].astype(bool)
            histograms, coverage = metrics.collect(current, [r for r in current.samples if r["split"] == split], prediction)
            rows = [dict(method=method, split=split, building_id=building, category_id=category["id"],
                         category=category["name"], slice_type="class", slice=category["name"],
                         target_type="semantic state", threshold=.5, **metrics.metrics(hist[0], .5))
                    for building, hist in histograms.items()]
            per_building.extend(rows)
            # Semantic F1 is defined wherever class/predicted positives permit it;
            # unlike latest-change F1, false positives on absent class count here.
            values = dict(method=method, split=split, category_id=category["id"], category=category["name"],
                          target_type="semantic state", threshold=.5, known_pixels=coverage["known_pixels"])
            for key in metrics.METRICS:
                values[key] = _mean(r[key] for r in rows)
                values[key + "_buildings"] = sum(r[key] is not None for r in rows)
            macro.append(values)
    return macro, per_building


def _semantic_main_rows(semantic, per_building, metadata, bootstrap_repetitions, practical=None):
    methods = sorted({r["method"] for r in semantic})
    scores = {}
    for method in methods:
        for split in ("val", "test"):
            subset = [r for r in per_building if r["method"] == method and r["split"] == split]
            buildings = sorted({r["building_id"] for r in subset})
            scores[method, split] = {metric: _mean(
                _mean(r[metric] for r in subset if r["building_id"] == building) for building in buildings)
                for metric in metrics.METRICS}
    controls = [m for m in methods if metadata[m].get("control") in ("independent", "repeat")]
    control = max(controls, key=lambda method: (scores[method, "val"]["f1"]
        if scores[method, "val"]["f1"] is not None else -1., method)) if controls else None
    result = []
    histories = [m for m in methods if metadata[m].get("control") == "learned"]
    for method in histories:
        current, reference = scores[method, "test"], scores.get((control, "test"), {})
        difference = current.get("f1") - reference.get("f1") if current.get("f1") is not None and reference.get("f1") is not None else None
        # Collapse class measurements within each building, then bootstrap buildings.
        paired_rows = []
        for name in (method, control):
            buildings = sorted({r["building_id"] for r in per_building if r["method"] == name and r["split"] == "test"})
            for building in buildings:
                subset = [r for r in per_building if r["method"] == name and r["split"] == "test" and r["building_id"] == building]
                paired_rows.append(dict(method=name, building_id=building, slice_type="overall",
                    positive_pixels=sum(r["positive_pixels"] for r in subset), negative_pixels=sum(r["negative_pixels"] for r in subset),
                    f1=_mean(r["f1"] for r in subset), semantic_f1=_mean(r["f1"] for r in subset)))
        ci = paired_bootstrap(paired_rows, method, control, metric="semantic_f1", repetitions=bootstrap_repetitions) if control else {}
        result.append(dict(task="semantic-state", model=method + "-semantic-state", control=control,
            target_type="semantic state (multilabel; fixed threshold 0.5)", f1=current.get("f1"), iou=current.get("iou"),
            precision=current.get("precision"), recall=current.get("recall"), fpr=current.get("fpr"),
            delta_f1=difference, ci_low=ci.get("ci_low"), ci_high=ci.get("ci_high"),
            bootstrap_supports_direction=ci.get("bootstrap_supports_direction"), prediction_vs_control="N/A",
            status=practical_status(current, reference, practical), seconds=metadata[method].get("train_seconds"),
            selected_on="VAL", exploratory=True,
            reason="Semantic state metrics are not latest-change metrics and do not establish downstream gain"))
    return result


def planned_comparisons(metadata, overall):
    """Named, one-factor controls plus explicitly multi-factor best-family comparison."""
    pairs = set()
    def add(method, control, factor):
        if method != control and (method, "test", .05) in overall and (control, "test", .05) in overall:
            pairs.add((method, control, factor))
    base = {}
    for method, row in metadata.items():
        if (row.get("scenario") == "main" and row.get("control") == "history" and not row.get("ablation") and
                not row.get("threshold_parent") and row.get("checkpoint_selection") != "prediction" and
                method.endswith("-history")):
            base[row.get("task"), row.get("family"), row.get("size")] = method
    for (task, family, size), method in base.items():
        prefix = f"{task}-{family}-{size}-"
        for control, factor in (("pair-only" if task == "direct" else "last-only", "old_observation_content"),
                               ("repeat-history", "old_content_equal_steps_time_support"),
                               ("mean", "temporal_model_vs_order_free_history"),
                               ("absolute-delta", "signed_vs_absolute_delta"),
                               ("ordinal-time", "real_intervals_vs_ordinal_time")):
            add(method, prefix + control, factor)
        for intervention in ("repeat", "shuffle-common", "shuffle"):
            add(method + "-intervention-" + intervention, method,
                "fixed_checkpoint_history_intervention:" + intervention)
        if size == "large" and (task, family, "medium") in base:
            add(method, base[task, family, "medium"], "model_capacity")
        if family == "transformer" and (task, "convgru", size) in base:
            add(method, base[task, "convgru", size], "architecture_at_declared_size")
    def key(method):
        row = overall.get((method, "val", .05), {})
        return (row.get("recall") if row.get("recall") is not None else -1.,
                -row["fpr"] if row.get("fpr") is not None else -1., method)
    for task in ("direct", "actual-next", "h0"):
        choices = {family: [m for (t, f, _), m in base.items() if t == task and f == family]
                   for family in ("convgru", "transformer")}
        if all(choices.values()):
            add(max(choices["transformer"], key=key), max(choices["convgru"], key=key),
                "best_VAL_family:architecture_and_potentially_capacity")
    for method, control, factor in (("A-learned", "A-repeat", "semantic_memory_old_content"),
        ("A-confidence", "A-independent", "confidence_filter"), ("A-mean", "A-independent", "temporal_averaging"),
        ("B-history", "B-pair", "history_training_auxiliary"),
        ("B-history", "B-regularized", "history_auxiliary_vs_feature_regularization"),
        ("C-history", "C-latest", "progression_history_vs_latest_pair"),
        ("C-history", "C-first", "progression_history_vs_first_current")):
        add(method, control, factor)
    return sorted(pairs)
