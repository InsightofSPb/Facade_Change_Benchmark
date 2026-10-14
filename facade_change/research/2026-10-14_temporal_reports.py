"""Rebuild temporal metrics and galleries from authenticated saved maps only.

No model modules are imported. Thresholds use synthetic VAL no-event controls,
including every nuisance condition, with equal facade weights. The first-frame
state target and the adjacent-event target are never pooled.
"""
from __future__ import annotations

import base64
import hashlib
import html
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ..hypothesis_benchmark import _building_macro, _metrics_from_counts
from ..io import sha256
from .common import csv_rows


DATE = "2026-10-14"
GROUP_FIELDS = ("method", "variant", "mode", "target")
IDENTITY_FIELDS = ("unit_id", *GROUP_FIELDS, "domain", "split", "building_id",
                   "sequence_id", "frame", "arm", "event", "nuisance")
SUCCESS = {"completed", "complete", "success", "succeeded"}
NATIVE = {"native", "native_binary", "native_mask", "binary"}


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    csv_rows(temporary, rows)
    temporary.replace(path)


def _group(record):
    return tuple(str(record.get(name, "")) for name in GROUP_FIELDS)


def _is_native(record):
    return record.get("output_kind") in NATIVE


def _is_initialization(record):
    return record.get("status") == "initialized" and record.get("frame") == 0


def _numeric_labels(record):
    return record.get("domain") == "synthetic" or record.get("label_verified") is True


def _calibration_control(record):
    return (record.get("domain") == "synthetic" and record.get("split") == "val"
            and record.get("arm") == "clean" and record.get("hypothesis") == "H0")


def load_artifact(out, record):
    """Authenticate one completed unit and enforce the saved-array contract."""
    root = Path(out).resolve()
    path = (root / record["artifact_path"]).resolve()
    if root not in path.parents:
        raise ValueError(f"Artifact must be inside run directory: {path}")
    if not path.is_file() or sha256(path) != record.get("artifact_sha256"):
        raise ValueError(f"Missing or corrupt unit artifact: {record['unit_id']}")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    required = {"scores", "labels", "support", "reference", "source"}
    if not required.issubset(arrays):
        raise ValueError(f"Missing saved arrays for {record['unit_id']}: {required - arrays.keys()}")
    scores, labels, support = (arrays[name] for name in ("scores", "labels", "support"))
    if scores.ndim != 2 or scores.dtype != np.float32:
        raise ValueError("Temporal scores must be native-grid float32 HxW")
    if labels.shape != scores.shape or labels.dtype != np.uint8 or not np.isin(labels, [0, 1, 255]).all():
        raise ValueError("Temporal labels must be uint8 0/1/255 on the score grid")
    if support.shape != scores.shape or support.dtype != bool:
        raise ValueError("Temporal support must be boolean on the score grid")
    if not np.isfinite(scores[support]).all() or np.any((labels != 255) & ~support):
        raise ValueError("Invalid score or evaluated label outside geometric support")
    for name in ("reference", "source"):
        if arrays[name].shape != (*scores.shape, 3) or arrays[name].dtype != np.uint8:
            raise ValueError(f"Saved {name} must be uint8 RGB on the native score grid")
    if _is_native(record):
        prediction = arrays.get("native_prediction")
        if prediction is None or prediction.shape != scores.shape or prediction.dtype != bool:
            raise ValueError("Native methods require their actual boolean prediction")
    if _numeric_labels(record) and record.get("hypothesis") not in {"H0", "H1"}:
        raise ValueError("Quantitative units require a target-specific H0/H1 designation")
    if record.get("hypothesis") == "H0" and np.any(labels == 1):
        raise ValueError("H0 unit contains positive temporal labels")
    if not record.get("building_id"):
        raise ValueError("Temporal units require a physical facade/building identity")
    return arrays


def calibrate_thresholds(items, fpr_targets=(.01, .05), max_grid=4096):
    """Choose the lowest feasible score threshold using only clean-arm VAL H0.

    Each item is ``{'record': record, 'arrays': {'scores', 'labels', 'support'}}``.
    A callable producing fresh iterators enables two bounded-memory file passes.
    Threshold candidates are sorted unique H0 scores (a deterministic subset for
    large maps); the maximum is always included, yielding no H0 detections. FPR
    is measured exactly at every candidate, averaging units within facade then
    facades equally. H1, TRAIN, real and TEST values never enter selection.
    """
    if max_grid < 2 or any(not 0 <= value < 1 for value in fpr_targets):
        raise ValueError("Need max_grid >= 2 and FPR budgets in [0, 1)")
    if callable(items):
        iterate = items
    else:
        materialized = list(items)
        iterate = lambda: iter(materialized)
    identities, buildings, source_ids, candidates = set(), set(), [], []
    valid_units = 0
    for item in iterate():
        record, arrays = item["record"], item["arrays"]
        if not _calibration_control(record):
            continue
        source_ids.append(record["unit_id"])
        identities.add(_group(record))
        if len(identities) > 1:
            raise ValueError("Calibrate each method/variant/mode/target separately")
        if _is_native(record):
            raise ValueError("Native binary decisions cannot be calibrated")
        scores, labels, support = (arrays[name] for name in ("scores", "labels", "support"))
        if np.any(labels == 1):
            raise ValueError("H0 calibration contains temporal positives")
        values = np.sort(scores[(labels == 0) & support])
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite H0 calibration values")
        if not len(values):
            continue
        valid_units += 1
        buildings.add(record["building_id"])
        indices = np.linspace(0, len(values) - 1, min(max_grid, len(values)), dtype=int)
        candidates.append(values[indices])
    provenance = {"selection_split": "val", "selection_domain": "synthetic",
                  "selection_arm": "clean", "nuisances": "all",
                  "prediction_rule": "score > threshold",
                  "aggregation": "mean units within facade, then mean facades",
                  "criterion": "lowest candidate threshold with facade-macro H0 FPR <= budget",
                  "h0_building_count": len(buildings),
                  "h0_unit_count": valid_units,
                  "source_unit_ids": source_ids}
    if not candidates:
        return {"status": "missing_calibration", "choices": [], "curve": [], **provenance}
    grid = np.unique(np.concatenate(candidates))
    if len(grid) > max_grid:
        grid = grid[np.linspace(0, len(grid) - 1, max_grid, dtype=int)]
    by_building, unit_counts = {}, Counter()
    second_ids = []
    for item in iterate():
        record, arrays = item["record"], item["arrays"]
        if not _calibration_control(record):
            continue
        second_ids.append(record["unit_id"])
        values = np.sort(arrays["scores"][(arrays["labels"] == 0) & arrays["support"]])
        if not len(values):
            continue
        building = record["building_id"]
        rates = (len(values) - np.searchsorted(values, grid, side="right")) / len(values)
        by_building[building] = by_building.get(building, np.zeros(len(grid))) + rates
        unit_counts[building] += 1
    if source_ids != second_ids:
        raise ValueError("Calibration units changed between saved-map passes")
    fpr = np.mean([values / unit_counts[building] for building, values in by_building.items()], axis=0)
    curve = [{"threshold": float(value), "h0_pixel_fpr": float(rate)}
             for value, rate in zip(grid, fpr)]
    choices = []
    for target in fpr_targets:
        feasible = np.flatnonzero(fpr <= target)
        if not len(feasible):
            raise ValueError("Calibration failed to include the no-detections threshold")
        choices.append({"fpr_target": float(target), **curve[int(feasible[0])]})
    return {"status": "calibrated", "choices": choices, "curve": curve,
            "grid_policy": f"at most {max_grid} ranked H0 candidates; includes maximum",
            **provenance}


def _metric(record, arrays, setting):
    labels, support = arrays["labels"], arrays["support"]
    prediction = (arrays["native_prediction"] if _is_native(record)
                  else arrays["scores"] > np.float32(setting["threshold"])) & support
    positive, negative = labels == 1, labels == 0
    counts = [int((prediction & positive).sum()), int((prediction & negative).sum()),
              int((~prediction & positive).sum()), int((~prediction & negative).sum())]
    identity = {name: record.get(name) for name in IDENTITY_FIELDS}
    return {**identity, "hypothesis": record["hypothesis"],
            "decision": "author_native" if _is_native(record) else "val_calibrated",
            "threshold": setting.get("threshold"), "fpr_target": setting.get("fpr_target"),
            "artifact_path": record["artifact_path"], "artifact_sha256": record["artifact_sha256"],
            "elapsed_seconds": record.get("elapsed_seconds"),
            **_metrics_from_counts(record, counts, int((labels == 255).sum()))}


def _array_digest(arrays, names=("labels", "support")):
    digest = hashlib.sha256()
    for name in names:
        value = np.ascontiguousarray(arrays[name])
        digest.update(str((name, value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _aggregates(rows, extra=()):
    fields = (*GROUP_FIELDS, "domain", "split", "decision", "fpr_target", *extra)
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in fields)].append(row)
    result = []
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        aggregate = _building_macro(values)
        result.append({**dict(zip(fields, key)), "unit_count": len(values),
                       "building_count": aggregate["building_count"],
                       "threshold": values[0]["threshold"],
                       **aggregate["means"],
                       "metric_building_counts": aggregate["metric_building_counts"]})
    return result


def _png(array):
    picture = Image.fromarray(array)
    picture.thumbnail((240, 240))
    stream = io.BytesIO()
    picture.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def _gallery_row(record, arrays, settings):
    support, scores, labels = (arrays[name] for name in ("support", "scores", "labels"))
    values = scores[support]
    low, high = (float(values.min()), float(values.max())) if len(values) else (0., 0.)
    scale = np.zeros(scores.shape, dtype=np.float32)
    if high > low:
        scale[support] = (scores[support] - low) / (high - low)
    heat = np.stack((255 * scale, 90 * (1 - np.abs(2 * scale - 1)), 255 * (1 - scale)), axis=-1)
    heat = heat.astype(np.uint8)
    heat[~support] = (80, 80, 80)
    label_rgb = np.full((*labels.shape, 3), 80, dtype=np.uint8)
    if _numeric_labels(record):
        label_rgb[labels == 0] = (0, 0, 0)
        label_rgb[labels == 1] = (245, 195, 25)
    reference_caption = record.get("reference_display") or "Reference RGB"
    cells = [(reference_caption, arrays["reference"]), ("Current RGB", arrays["source"]),
             (f"Score [{low:.4g}, {high:.4g}]", heat),
             ("Temporal GT" if _numeric_labels(record) else "Unreviewed: no temporal GT", label_rgb)]
    for setting in settings:
        prediction = (arrays["native_prediction"] if _is_native(record)
                      else scores > np.float32(setting["threshold"])) & support
        display = np.full((*scores.shape, 3), 80, dtype=np.uint8)
        display[support] = (0, 0, 0)
        display[prediction] = (230, 50, 170)
        caption = ("Author native prediction" if _is_native(record)
                   else f"VAL FPR budget {setting['fpr_target']:.0%}; t={setting['threshold']:.5g}")
        cells.append((caption, display))
    caption = " | ".join(str(record.get(field, "")) for field in IDENTITY_FIELDS if field != "unit_id")
    figures = "".join(f'<figure><img src="{_png(array)}" alt="{html.escape(name)}">'
                      f'<figcaption>{html.escape(name)}</figcaption></figure>' for name, array in cells)
    return f'<section><h3>{html.escape(caption)}</h3><div class="panels">{figures}</div></section>\n'


def _paired_history(rows):
    """Compare only identical completed units; independent mode thresholds remain explicit."""
    fields = ("method", "variant", "target", "domain", "split", "sequence_id", "frame", "arm", "fpr_target")
    grouped = defaultdict(dict)
    for row in rows:
        mode = row["mode"]
        if mode == "previous" or str(mode).startswith("history"):
            grouped[tuple(row.get(field) for field in fields)][mode] = row
    paired = []
    for key, modes in grouped.items():
        previous = modes.get("previous")
        if previous is None:
            continue
        for mode, history in modes.items():
            if mode == "previous":
                continue
            paired.append({**dict(zip(fields, key)), "building_id": previous["building_id"],
                           "mode": mode, "previous_unit_id": previous["unit_id"],
                           "history_unit_id": history["unit_id"],
                           "previous_threshold": previous["threshold"],
                           "history_threshold": history["threshold"],
                           "previous_recall": previous["recall"], "history_recall": history["recall"],
                           "recall_delta": history["recall"] - previous["recall"]
                           if history["recall"] is not None and previous["recall"] is not None else None,
                           "previous_h0_pixel_fpr": previous["h0_pixel_fpr"],
                           "history_h0_pixel_fpr": history["h0_pixel_fpr"]})
    return paired


def _history_summary(pairs):
    fields = ("method", "variant", "mode", "target", "domain", "split", "fpr_target")
    grouped = defaultdict(list)
    for row in pairs:
        grouped[tuple(row[field] for field in fields)].append(row)
    result = []
    metrics = ("previous_recall", "history_recall", "recall_delta",
               "previous_h0_pixel_fpr", "history_h0_pixel_fpr")
    for key, rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        facades = defaultdict(list)
        for row in rows:
            facades[row["building_id"]].append(row)
        means = {}
        counts = {}
        for metric in metrics:
            values = [float(np.mean([row[metric] for row in units if row[metric] is not None]))
                      for units in facades.values() if any(row[metric] is not None for row in units)]
            means[metric] = float(np.mean(values)) if values else None
            counts[metric] = len(values)
        result.append({**dict(zip(fields, key)), "matched_unit_count": len(rows),
                       "building_count": len(facades), "metric_building_counts": counts, **means})
    return result


def build_reports(out, records, fpr_targets=(.01, .05)):
    """Rebuild dated report artifacts; failed/unfinished units remain in coverage.

    ``records`` is the latest record per unit, rather than an append-only attempt
    log. Corrupt completed results fail closed. Missing calibration leaves a
    continuous method qualitative while retaining its score gallery.
    """
    out = Path(out).resolve()
    records = list(records)
    identifiers = [record["unit_id"] for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Report input must contain one latest record per unit")
    successful = [record for record in records if record.get("status") in SUCCESS]
    grouped = defaultdict(list)
    for record in successful:
        if record.get("split") == "test":
            raise ValueError("This exploratory temporal protocol excludes TEST")
        grouped[_group(record)].append(record)
    calibration = []
    settings_by_group = {}
    for key, values in sorted(grouped.items()):
        native = {_is_native(record) for record in values}
        if len(native) != 1:
            raise ValueError("Output kind changes within a method variant")
        if native == {True}:
            selected = {"status": "author_native", "choices": [{"threshold": None, "fpr_target": None}],
                        "curve": [], "prediction_rule": "saved author-native boolean mask"}
        else:
            def controls():
                for record in values:
                    if _calibration_control(record):
                        yield {"record": record, "arrays": load_artifact(out, record)}
            selected = calibrate_thresholds(controls, fpr_targets)
        settings_by_group[key] = selected["choices"]
        calibration.append({**dict(zip(GROUP_FIELDS, key)), **selected})
    rows, coverage, label_signatures = [], [], {}
    gallery = out / f"{DATE}_temporal_gallery.html"
    temporary_gallery = gallery.with_suffix(".html.tmp")
    with temporary_gallery.open("w", encoding="utf-8") as stream:
        stream.write('<!doctype html><html lang="en"><meta charset="utf-8"><title>Temporal change results</title>'
                     '<style>body{font:15px system-ui;max-width:1550px;margin:24px auto;background:#f5f6f8;color:#17202b}'
                     'section{background:white;padding:14px;margin:18px 0;border:1px solid #d9dfe7}'
                     '.panels{display:flex;flex-wrap:wrap}figure{margin:6px}img{max-width:240px}'
                     'figcaption{max-width:240px;font-size:12px}h3{font-size:14px}</style>'
                     '<h1>Temporal change: saved results</h1><p>Exploratory TRAIN/VAL only. '
                     'First-frame state and adjacent events are separate targets. '
                     'Real candidates have no numerical temporal accuracy without verified labels. '
                     'No confidence intervals are estimated in this exploratory report.</p>'
                     '<p>Legend: score blue → red (per-map display range only); prediction magenta; '
                     'temporal change label yellow; no change black; ignored/outside support grey. '
                     'Display normalization never affects calibrated decisions. '
                     'Thresholds: synthetic VAL clean arm (no structural event), all nuisance conditions.</p>')
        for record in successful:
            arrays = load_artifact(out, record)
            key = _group(record)
            settings = settings_by_group[key]
            mode = str(record.get("mode"))
            if mode == "previous" or mode.startswith("history"):
                match = tuple(record.get(name) for name in ("method", "variant", "target", "domain", "split",
                                                            "sequence_id", "frame", "arm"))
                signature = _array_digest(arrays)
                if match in label_signatures and label_signatures[match] != signature:
                    raise ValueError("Previous/history labels or support disagree on matched units")
                label_signatures[match] = signature
            quantitative = _numeric_labels(record)
            if quantitative:
                rows.extend(_metric(record, arrays, setting) for setting in settings)
            coverage.append({**{name: record.get(name) for name in IDENTITY_FIELDS},
                             "status": record["status"], "elapsed_seconds": record.get("elapsed_seconds"),
                             "numeric_temporal_labels": quantitative,
                             "calibrated_or_native": bool(settings),
                             "support_pixel_count": int(arrays["support"].sum()),
                             "evaluated_pixel_count": int((arrays["labels"] != 255).sum()) if quantitative else 0,
                             "support_fraction": float(arrays["support"].mean())})
            stream.write(_gallery_row(record, arrays, settings))
        failed = [record for record in records
                  if record.get("status") not in SUCCESS and not _is_initialization(record)]
        if failed:
            stream.write('<h2>Unfinished or failed units</h2><ul>')
            for record in failed:
                stream.write(f'<li>{html.escape(str(record["unit_id"]))}: '
                             f'{html.escape(str(record.get("status")))} — '
                             f'{html.escape(str(record.get("error", "")))}</li>')
            stream.write('</ul>')
        seen_metadata = set()
        for record in records:
            metadata = record.get("method_metadata", {})
            encoded = json.dumps(metadata, sort_keys=True, ensure_ascii=False, indent=2)
            identity = record.get("method"), record.get("variant"), encoded
            if not metadata or identity in seen_metadata:
                continue
            seen_metadata.add(identity)
            title = f"{record.get('method')} / {record.get('variant')}: weights, provenance and limitations"
            stream.write(f'<details><summary>{html.escape(title)}</summary>'
                         f'<pre>{html.escape(encoded)}</pre></details>')
        stream.write('</html>')
    temporary_gallery.replace(gallery)
    coverage.extend({**{name: record.get(name) for name in IDENTITY_FIELDS},
                     "status": record.get("status"), "error": record.get("error"),
                     "initialization_only": _is_initialization(record),
                     "numeric_temporal_labels": False,
                     "elapsed_seconds": record.get("elapsed_seconds")}
                    for record in records if record.get("status") not in SUCCESS)
    aggregates = _aggregates(rows)
    pairs = _paired_history(rows)
    method_coverage = []
    all_groups = defaultdict(list)
    for record in records:
        all_groups[_group(record)].append(record)
    for key, values in sorted(all_groups.items()):
        metadata = {json.dumps(record.get("method_metadata", {}), sort_keys=True):
                    record.get("method_metadata", {}) for record in values}
        method_coverage.append({**dict(zip(GROUP_FIELDS, key)), "unit_count": len(values),
                                "initialized_unit_count": sum(_is_initialization(record) for record in values),
                                "status_counts": dict(Counter(record.get("status", "unknown") for record in values)),
                                "elapsed_seconds": float(sum(record.get("elapsed_seconds") or 0 for record in values)),
                                "method_metadata": list(metadata.values())})
    summary = {"scope": "exploratory TRAIN/VAL; real unreviewed candidates are qualitative only",
               "reconstruction": "saved unit records and SHA-256-verified NPZ maps; no model inference",
               "aggregation": "units within physical facade, then equal facades; no confidence intervals",
               "completed_unit_count": len(successful), "record_count": len(records),
               "initialized_unit_count": sum(_is_initialization(record) for record in records),
               "failed_or_unfinished_unit_count": len(failed),
               "status_counts": dict(Counter(record.get("status", "unknown") for record in records)),
               "elapsed_seconds": float(sum(record.get("elapsed_seconds") or 0 for record in records)),
               "calibration": calibration, "metrics": aggregates, "method_coverage": method_coverage,
               "history_comparisons": _history_summary(pairs),
               "source_records_sha256": hashlib.sha256(json.dumps(records, sort_keys=True,
                                                                   allow_nan=False).encode()).hexdigest(),
               "unreviewed_real_unit_count": sum(record.get("domain") == "real" and not _numeric_labels(record)
                                                 for record in successful),
               "gallery_path": gallery.name,
               "coverage_path": f"{DATE}_temporal_coverage.csv"}
    _atomic_json(out / f"{DATE}_temporal_thresholds.json", calibration)
    _atomic_csv(out / f"{DATE}_temporal_units.csv", rows)
    _atomic_csv(out / f"{DATE}_temporal_facades.csv", _aggregates(rows, ("building_id",)))
    _atomic_csv(out / f"{DATE}_temporal_events.csv", _aggregates(rows, ("event",)))
    _atomic_csv(out / f"{DATE}_temporal_nuisances.csv", _aggregates(rows, ("nuisance",)))
    _atomic_csv(out / f"{DATE}_temporal_coverage.csv", coverage)
    _atomic_csv(out / f"{DATE}_temporal_history_pairs.csv", pairs)
    _atomic_csv(out / f"{DATE}_temporal_history_comparison.csv", summary["history_comparisons"])
    _atomic_json(out / f"{DATE}_temporal_summary.json", summary)
    text_path = out / f"{DATE}_temporal_summary.txt"
    lines = [summary["scope"], summary["aggregation"],
             f"Completed scoring units: {len(successful)}/{len(records) - summary['initialized_unit_count']}; "
             f"initializations: {summary['initialized_unit_count']}; saved scoring seconds: {summary['elapsed_seconds']:.3f}",
             "H0 thresholds: synthetic VAL clean arm (no structural event), all nuisance conditions."]
    for row in aggregates:
        identity = " / ".join(str(row[field]) for field in (*GROUP_FIELDS, "domain", "split"))
        percent = lambda value: "n/a" if value is None else f"{value:.2%}"
        lines.append(f"{identity}; budget={row['fpr_target']}; Recall={percent(row['recall'])}; "
                     f"H0 FPR={percent(row['h0_pixel_fpr'])}; facades={row['building_count']}")
    temporary = text_path.with_suffix(".txt.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(text_path)
    return summary
