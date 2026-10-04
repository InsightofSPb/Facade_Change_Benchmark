"""Exploratory CPU change scores with validation-only threshold calibration."""
from __future__ import annotations

import csv
import hashlib
import html
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .io import finish_record, load_rgb, new_directory, read_json, run_record, sha256, write_json


THRESHOLDS = np.linspace(0, 1, 101, dtype=np.float32)
METRICS = ("f1", "iou", "precision", "recall", "h0_pixel_fpr", "comparable_fraction", "retained_visible_edit_fraction")
SCOPE = "EXPLORATORY: procedural synthetic controls only; no real-damage benchmark or confidence intervals"


def _inside(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Dataset artifacts must use relative paths")
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("Dataset artifact path escapes its root")
    return path


def _checked(root, record, relative, expected=None):
    path = _inside(root, relative)
    digest = sha256(path)
    if record.get("artifact_sha256", {}).get(relative) != digest or (expected is not None and expected != digest):
        raise ValueError(f"Dataset artifact changed or its hash disagrees: {relative}")
    return path, digest


def _select(index, split, limit, expected_signatures):
    bases, cases_by_base, ownership, case_ids = {}, defaultdict(list), {}, set()
    for base in index["bases"]:
        identifier, building, part = base["base_id"], base["building_id"], base["split"]
        if identifier in bases or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
            raise ValueError("Duplicate or unsafe base ID")
        if not building or part not in {"train", "val", "test"} or split["building_assignments"].get(building) != part:
            raise ValueError("Base disagrees with the reviewed building split")
        if building in ownership and ownership[building] != part:
            raise ValueError("Building leakage across partitions")
        ownership[building] = part
        bases[identifier] = base
    for case in index["cases"]:
        identifier = case["case_id"]
        if identifier in case_ids or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
            raise ValueError("Duplicate or unsafe case ID")
        case_ids.add(identifier)
        base = bases.get(case["base_id"])
        if not base or any(case.get(key) != base.get(key) for key in ("building_id", "split", "view_id")):
            raise ValueError("Case identity disagrees with its frozen base")
        if case["parent_dataset_crop_id"] != base["dataset_crop_id"]:
            raise ValueError("Case parent crop identity disagrees with its base")
        hypothesis = "H1" if case["state"] in {"crack", "paint_patch"} else "H0"
        if case["hypothesis"] != hypothesis or bool(case.get("sham_self_paste")) != (case["state"] == "self_paste"):
            raise ValueError("Case state/hypothesis/control metadata disagrees")
        cases_by_base[case["base_id"]].append(case)
    selected = []
    for part in ("train", "val", "test"):
        candidates = sorted((base for base in bases.values() if base["split"] == part),
                            key=lambda base: hashlib.sha256(base["base_id"].encode()).hexdigest())
        selected.extend(candidates[:limit] if limit else candidates)
    if not {"val", "test"}.issubset({base["split"] for base in selected}):
        raise ValueError("The frozen trial selection requires both validation and test bases")
    cases = []
    for base in selected:
        rows = sorted(cases_by_base[base["base_id"]], key=lambda row: row["case_id"])
        signatures = [(row["state"], row["scenario_id"]) for row in rows]
        if len(signatures) != len(set(signatures)) or set(signatures) != expected_signatures:
            raise ValueError("Selected base is missing or duplicates frozen state/scenario cases")
        cases.extend(rows)
    return selected, cases


def _mask(path, shape):
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.shape != shape or array.dtype != np.uint8 or not np.isin(array, [0, 255]).all():
        raise ValueError(f"Expected native-grid binary uint8 mask: {path.name}")
    return array == 255


def _case_inputs(root, parent, base, case, checked_hashes):
    paths = {}
    required = ("reference_rgb", "source_rgb", "reference_support", "source_support", "labels_reference", "comparable",
                "true_visibility", "source_occlusion", "true_edit_mask_reference", "visible_edit_mask_reference", "geometry")
    for key in required:
        path, digest = _checked(root, parent, case[key], case["artifact_sha256"][key])
        checked_hashes[case[key]] = digest
        paths[key] = path
    base_path = _inside(root, base["path"])
    if paths["reference_rgb"] != base_path / "reference_rgb.png" or paths["reference_support"] != base_path / "reference_support.png":
        raise ValueError("Case reference/support do not belong to its selected base")
    geometry = read_json(paths["geometry"])
    identity = np.eye(3).tolist()
    if any(geometry.get(key) != identity or case.get(key) != identity for key in ("source_to_reference", "reference_to_source")):
        raise ValueError("Synthetic benchmark requires identity geometry on the native crop grid")
    reference, opaque = load_rgb(paths["reference_rgb"])
    source, source_opaque = load_rgb(paths["source_rgb"])
    shape = reference.shape[:2]
    if source.shape != reference.shape:
        raise ValueError("Case RGB images have different native grids")
    support = _mask(paths["reference_support"], shape)
    source_support = _mask(paths["source_support"], shape)
    visibility = _mask(paths["true_visibility"], shape)
    comparable = _mask(paths["comparable"], shape)
    full_edit = _mask(paths["true_edit_mask_reference"], shape)
    visible_edit = _mask(paths["visible_edit_mask_reference"], shape)
    occlusion = _mask(paths["source_occlusion"], shape)
    with Image.open(paths["labels_reference"]) as image:
        labels = np.asarray(image)
    if labels.shape != shape or labels.dtype != np.uint8 or not np.isin(labels, [0, 1, 255]).all():
        raise ValueError("Expected native-grid labels 0/1/255")
    if (np.any(support & ~opaque) or np.any(source_support & ~source_opaque) or np.any(full_edit & ~support)
            or not np.array_equal(comparable, support & source_support & visibility)
            or not np.array_equal(labels != 255, comparable)
            or not np.array_equal(visible_edit, full_edit & comparable)
            or not np.array_equal(labels[comparable], full_edit[comparable].astype(np.uint8))
            or not np.array_equal(occlusion, support & ~visibility)):
        raise ValueError("Case known evaluation masks/labels disagree")
    full, visible = int(full_edit.sum()), int(visible_edit.sum())
    if case["full_edit_pixel_count"] != full or case["visible_edit_pixel_count"] != visible:
        raise ValueError("Case edit coverage metadata disagrees with its masks")
    if (case["hypothesis"] == "H0" and full) or (case["hypothesis"] == "H1" and not full):
        raise ValueError("Case hypothesis disagrees with planted state GT")
    if not np.isclose(case["comparable_fraction"], comparable.mean()) or bool(case["exclude_from_visible_recall"]) != bool(full and not visible):
        raise ValueError("Case comparable/hidden-state metadata disagrees")
    return reference, source, support, labels


def _counts(scores, labels, thresholds):
    positive = np.sort(scores[labels == 1])
    negative = np.sort(scores[labels == 0])
    thresholds = np.asarray(thresholds, dtype=np.float32)
    tp = len(positive) - np.searchsorted(positive, thresholds, side="right")
    fp = len(negative) - np.searchsorted(negative, thresholds, side="right")
    return tp, fp, len(positive) - tp, len(negative) - fp


def _calibrate(validation):
    h1, h0 = defaultdict(list), defaultdict(list)
    for case, counts in validation:
        if case["sham_self_paste"]:
            continue
        tp, fp, fn, tn = counts
        if case["hypothesis"] == "H1" and np.any(tp + fn > 0):
            h1[case["building_id"]].append(2 * tp / (2 * tp + fp + fn))
        elif case["hypothesis"] == "H0" and np.any(fp + tn > 0):
            h0[case["building_id"]].append(fp / (fp + tn))
    if not h1 or not h0:
        raise ValueError("Validation needs visible H1 positives and comparable non-sham H0 negatives")
    f1 = np.mean([np.mean(values, axis=0) for values in h1.values()], axis=0)
    fpr = np.mean([np.mean(values, axis=0) for values in h0.values()], axis=0)
    tied = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))
    tied = tied[np.isclose(fpr[tied], fpr[tied].min(), rtol=0, atol=1e-12)]
    chosen = int(tied[-1])
    return {"threshold": float(THRESHOLDS[chosen]), "grid_index": chosen,
            "selection_split": "val", "prediction_rule": "score > threshold",
            "comparison_dtype": "float32; JSON records the exact float32 grid values",
            "criterion": "max building-macro visible-H1 case F1; then min non-sham H0 pixel FPR; then highest threshold",
            "h1_building_count": len(h1), "h0_building_count": len(h0),
            "curve": [{"threshold": float(value), "h1_building_macro_f1": float(f1[i]),
                       "h0_building_macro_pixel_fpr": float(fpr[i])} for i, value in enumerate(THRESHOLDS)]}


def _case_metrics(case, scores, labels, threshold):
    tp, fp, fn, tn = (int(value) for value in _counts(scores, labels, threshold))
    visible_h1 = case["hypothesis"] == "H1" and tp + fn > 0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "evaluated_pixel_count": tp + fp + fn + tn,
            "ignored_pixel_count": int((labels == 255).sum()),
            "f1": 2 * tp / (2 * tp + fp + fn) if visible_h1 else None,
            "iou": tp / (tp + fp + fn) if visible_h1 else None,
            "precision": tp / (tp + fp) if visible_h1 and tp + fp else (0. if visible_h1 else None),
            "recall": tp / (tp + fn) if visible_h1 else None,
            "h0_pixel_fpr": fp / (fp + tn) if case["hypothesis"] == "H0" and fp + tn else None}


def _building_macro(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["building_id"]].append(row)
    means, counts, buildings = {}, {}, []
    for building, values in sorted(groups.items()):
        metrics = {metric: float(np.mean([row[metric] for row in values if row.get(metric) is not None]))
                   if any(row.get(metric) is not None for row in values) else None for metric in METRICS}
        buildings.append({"building_id": building, "case_count": len(values), "means": metrics})
    for metric in METRICS:
        values = [building["means"][metric] for building in buildings if building["means"][metric] is not None]
        means[metric] = float(np.mean(values)) if values else None
        counts[metric] = len(values)
    return {"case_count": len(rows), "building_count": len(buildings), "means": means,
            "metric_building_counts": counts, "buildings": buildings}


def _aggregates(rows):
    result = {"by_split": {part: _building_macro([row for row in rows if row["split"] == part])
                           for part in ("train", "val", "test")}}
    for dimension in ("state", "nuisance_kind", "scenario_id", "hypothesis"):
        groups = defaultdict(list)
        for row in rows:
            groups[row["split"], row[dimension]].append(row)
        result["by_" + dimension] = [{"split": part, dimension: value, **_building_macro(values)}
                                    for (part, value), values in sorted(groups.items())]
    return result


def run_hypothesis_benchmark(dataset_run, out, methods=None, max_bases_per_split=1):
    """Score frozen procedural cases; calibrate on validation and evaluate unchanged test."""
    from .scorers import score_change, scorer_metadata

    methods = list(methods) if methods is not None else ["rgb_diff", "ssim"]
    if not methods or len(set(methods)) != len(methods) or any(method not in {"rgb_diff", "ssim"} for method in methods):
        raise ValueError("methods must contain unique rgb_diff/ssim names")
    if type(max_bases_per_split) is not int or max_bases_per_split < 0:
        raise ValueError("max_bases_per_split must be nonnegative; zero selects all bases")
    root = Path(dataset_run).expanduser().resolve()
    parent_hash = sha256(root / "run.json")
    parent = read_json(root / "run.json")
    if parent.get("kind") != "hypothesis_dataset" or parent.get("status") != "completed_needs_review":
        raise ValueError("Input must be a completed procedural hypothesis dataset")
    summary_path, summary_hash = _checked(root, parent, "summary.json")
    parent_summary = read_json(summary_path)
    index_path, index_hash = _checked(root, parent, parent_summary["index_path"])
    split_path, split_hash = _checked(root, parent, "split.json")
    config_path, config_hash = _checked(root, parent, "config.json")
    index, split = read_json(index_path), read_json(split_path)
    if split.get("mode") != "reviewed" or split.get("development_only") is not False or index["split"] != split:
        raise ValueError("Benchmark requires the existing reviewed split snapshot")
    if parent["config"]["input_sha256"]["config"] != config_hash or parent_summary["case_count"] != len(index["cases"]):
        raise ValueError("Parent configuration/case-count provenance disagrees")
    expected = {(state, scenario["id"]) for state in parent["config"]["states"] for scenario in parent["config"]["scenarios"]}
    bases, cases = _select(index, split, max_bases_per_split, expected)
    inputs = {"parent_run": parent_hash, "parent_summary": summary_hash, "parent_index": index_hash,
              "split": split_hash, "config": config_hash}
    out = new_directory(out)
    record = run_record("hypothesis_benchmark", {"dataset_run": str(root), "input_sha256": inputs, "methods": methods,
                        "max_bases_per_split": max_bases_per_split, "threshold_grid": THRESHOLDS.tolist(),
                        "scorers": {method: scorer_metadata(method) for method in methods}, "scope": SCOPE,
                        "scorer_inputs": "reference RGB, synthetic source RGB, base reference support only; no oracle masks"})
    write_json(out / "run.json", record)
    try:
        shutil.copyfile(split_path, out / "split.json")
        shutil.copyfile(summary_path, out / "parent_summary.json")
        write_json(out / "selection.json", {"algorithm": "SHA256(base_id) ranking within inherited split, before scores or GT",
                                           "max_bases_per_split": max_bases_per_split, "bases": bases,
                                           "case_ids": [case["case_id"] for case in cases], "input_sha256": inputs})
        selected_hashes, calibration, score_paths = {}, {method: [] for method in methods}, {}
        base_cases = defaultdict(list)
        for case in cases:
            base_cases[case["base_id"]].append(case)
        for method in methods:
            (out / "scores" / method).mkdir(parents=True)
            (out / "predictions" / method).mkdir(parents=True)
            (out / "heatmaps" / method).mkdir(parents=True)
        for i, base in enumerate(bases, 1):
            for method in methods:
                print(f"Base {i}/{len(bases)}: {base['split']} {base['building_id']} / {base['base_id']}; {method}", flush=True)
                for case in base_cases[base["base_id"]]:
                    reference, source, support, labels = _case_inputs(root, parent, base, case, selected_hashes)
                    scores = score_change(reference, source, support.copy(), method=method)
                    if (scores.shape != support.shape or scores.dtype != np.float32
                            or not np.isfinite(scores[support]).all() or not np.isnan(scores[~support]).all()
                            or np.any(scores[support] < 0) or np.any(scores[support] > 1)):
                        raise ValueError("Scorer must return native float32 [0,1] scores and NaN outside base support")
                    relative = Path("scores") / method / (case["case_id"] + ".npy")
                    np.save(out / relative, scores, allow_pickle=False)
                    score_paths[method, case["case_id"]] = relative.as_posix()
                    if case["split"] == "val":
                        calibration[method].append((case, _counts(scores, labels, THRESHOLDS)))
        thresholds = {method: _calibrate(calibration[method]) for method in methods}
        write_json(out / "threshold_selection.json", {"scope": "validation only; test labels never enter calibration",
                                                       "exclude_self_paste": True, "methods": thresholds})
        rows, galleries = [], {method: [] for method in methods}
        bases_by_id = {base["base_id"]: base for base in bases}
        gallery_ids = _gallery_selection(cases)
        for method in methods:
            threshold = thresholds[method]["threshold"]
            for case in cases:
                _, _, support, labels = _case_inputs(root, parent, bases_by_id[case["base_id"]], case, selected_hashes)
                scores = np.load(out / score_paths[method, case["case_id"]], allow_pickle=False)
                prediction = scores > np.float32(threshold)  # Known occlusion is deliberately not applied to predictions.
                prediction_path = (Path("predictions") / method / (case["case_id"] + ".png")).as_posix()
                Image.fromarray(prediction.astype(np.uint8) * 255).save(out / prediction_path)
                row = {**{key: case.get(key) for key in ("case_id", "base_id", "building_id", "view_id", "split", "state", "scenario_id",
                                                         "nuisance_kind", "hypothesis", "sham_self_paste", "full_edit_pixel_count",
                                                         "visible_edit_pixel_count", "comparable_fraction", "retained_visible_edit_fraction",
                                                         "exclude_from_visible_recall", "reference_file_name", "source_file_name",
                                                         "reference_year", "source_year", "source_image_sha256")},
                       "method": method, "threshold": threshold, **_case_metrics(case, scores, labels, threshold),
                       "score_path": score_paths[method, case["case_id"]], "prediction_path": prediction_path,
                       "score_sha256": sha256(out / score_paths[method, case["case_id"]]),
                       "prediction_sha256": sha256(out / prediction_path)}
                rows.append(row)
                if case["case_id"] in gallery_ids:
                    heatmap_path = (Path("heatmaps") / method / (case["case_id"] + ".png")).as_posix()
                    values = np.nan_to_num(scores, nan=0)
                    heatmap = np.stack([values * 255, values * 80, (1 - values) * 255], axis=-1).astype(np.uint8)
                    heatmap[~support] = 64
                    Image.fromarray(heatmap).save(out / heatmap_path)
                    galleries[method].append((case, row, heatmap_path))
        _gallery(out, root, galleries)
        for relative, digest in selected_hashes.items():
            if sha256(root / relative) != digest:
                raise ValueError(f"Selected dataset artifact changed during benchmark: {relative}")
        for path, digest in ((root / "run.json", parent_hash), (summary_path, summary_hash), (index_path, index_hash),
                             (split_path, split_hash), (config_path, config_hash)):
            if sha256(path) != digest:
                raise ValueError(f"Parent provenance changed during benchmark: {path.name}")
        primary, sham = {}, {}
        for method in methods:
            primary[method] = _aggregates([row for row in rows if row["method"] == method and not row["sham_self_paste"]])
            sham[method] = _aggregates([row for row in rows if row["method"] == method and row["sham_self_paste"]])
        write_json(out / "metrics.json", {"scope": SCOPE, "aggregation": "case means within building, then equal building means; no pooled-pixel metric",
                                         "primary": primary, "sham_controls": sham, "cases": rows})
        fields = ["method", "case_id", "base_id", "building_id", "view_id", "split", "hypothesis", "state", "scenario_id",
                  "nuisance_kind", "sham_self_paste", "threshold", "tp", "fp", "fn", "tn", "evaluated_pixel_count", "ignored_pixel_count",
                  "f1", "iou", "precision", "recall", "h0_pixel_fpr", "full_edit_pixel_count", "visible_edit_pixel_count",
                  "retained_visible_edit_fraction", "comparable_fraction", "exclude_from_visible_recall", "score_path", "prediction_path"]
        with (out / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        summary = {"status": "completed_exploratory", "scope": SCOPE, "selected_base_count": len(bases),
                   "selected_case_count": len(cases), "scored_case_method_count": len(rows),
                   "bases_by_split": dict(Counter(base["split"] for base in bases)),
                   "cases_by_split": dict(Counter(case["split"] for case in cases)),
                   "buildings_by_split": {part: len({base["building_id"] for base in bases if base["split"] == part}) for part in ("train", "val", "test")},
                   "thresholds": {method: thresholds[method]["threshold"] for method in methods},
                   "primary": {method: {part: {key: value for key, value in aggregate.items() if key != "buildings"}
                                        for part, aggregate in primary[method]["by_split"].items()} for method in methods},
                   "sham_controls": {method: {part: {key: value for key, value in aggregate.items() if key != "buildings"}
                                              for part, aggregate in sham[method]["by_split"].items()} for method in methods},
                   "source_dataset_counts": {"base_count": parent_summary["selected_crop_count"],
                                             "case_count": parent_summary["case_count"],
                                             "building_count": parent_summary["selected_building_count"]},
                   "parent_summary_path": "parent_summary.json", "metrics_path": "metrics.json", "csv_metrics_path": "metrics.csv",
                   "threshold_selection_path": "threshold_selection.json", "selection_path": "selection.json", "gallery_path": "gallery.html"}
        write_json(out / "summary.json", summary)
        test_lines = []
        for method in methods:
            aggregate = summary["primary"][method]["test"]
            metric_text = "; ".join(name + "=" + (f"{aggregate['means'][name]:.6f}" if aggregate["means"][name] is not None else "undefined")
                                     for name in ("f1", "iou", "precision", "recall", "h0_pixel_fpr", "comparable_fraction", "retained_visible_edit_fraction"))
            test_lines.append(f"TEST {method}: {metric_text}; buildings={aggregate['building_count']}; metric building counts={aggregate['metric_building_counts']}.\n")
        (out / "summary.txt").write_text(
            f"{SCOPE}.\nBases: {len(bases)}; cases: {len(cases)}; buildings by split: {summary['buildings_by_split']}.\n"
            f"Validation-only frozen thresholds: {summary['thresholds']}.\n"
            + ''.join(test_lines) +
            "Scores use RGB and base reference support only. Labels/visibility are evaluation-only; 255 is ignored.\n"
            "Primary excludes self_paste; matched sham controls are reported separately.\n"
            "F1/IoU/precision/recall require visible H1 positives; H0 is evaluated by pixel false-positive rate.\n"
            "Means weight cases within buildings, then buildings equally. No confidence intervals are claimed.\n",
            encoding="utf-8")
        record["summary"] = summary
        finish_record(out, record, summary["status"])
        return summary
    except (Exception, KeyboardInterrupt) as exc:
        finish_record(out, record, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", str(exc))
        raise


def _gallery_selection(cases):
    groups = defaultdict(list)
    for case in cases:
        groups[case["state"], case["nuisance_kind"]].append(case)
    def representative(case):
        spec = case.get("scenario", {})
        return (-float(spec.get("strength", 0)), {"test": 0, "val": 1, "train": 2}[case["split"]],
                -float(spec.get("sigma", 0)), float(spec.get("quality", 100)), case["case_id"])
    selected = set()
    for _, values in sorted(groups.items()):
        selected.add(min(values, key=representative)["case_id"])
    # A second strong shadow template per state uses the remaining four slots.
    for state in sorted({case["state"] for case in cases}):
        shadows = groups.get((state, "shadow"), [])
        if shadows:
            first = min(shadows, key=representative)
            other = [case for case in shadows if case.get("scenario", {}).get("template") != first.get("scenario", {}).get("template")]
            if other:
                selected.add(min(other, key=representative)["case_id"])
    return selected  # Four states × eight nuisance families plus four other shadow templates.


def _gallery(out, root, methods):
    rows = ['<!doctype html><meta charset="utf-8"><title>Exploratory synthetic change scores</title>',
            '<style>body{font:16px system-ui;margin:24px}td{padding:8px;vertical-align:top}img{max-width:150px}</style>',
            '<h1>Exploratory synthetic change scores</h1><p>' + html.escape(SCOPE) + '</p>',
            '<p>Reference / synthetic source / planted edit / score (blue 0 → red 1) / frozen-threshold prediction. '
            'Prediction keeps known occlusions; evaluation alone ignores label 255. At most 36 cases per method.</p>']
    for method, cases in methods.items():
        rows.append('<h2>' + method + '</h2><table>')
        for case, metric, heatmap in cases:
            label = html.escape(f"{case['split']} {case['building_id']} {case['state']} {case['scenario_id']}; t={metric['threshold']:.2f}")
            paths = []
            for key in ("reference_rgb", "source_rgb", "true_edit_mask_reference"):
                source = _inside(root, case[key])
                relative = Path("gallery_inputs") / source.relative_to(root)
                destination = out / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copyfile(source, destination)
                if sha256(destination) != case["artifact_sha256"][key]:
                    raise ValueError("Copied gallery input disagrees with its selected case hash")
                paths.append(relative.as_posix())
            paths += [heatmap, metric["prediction_path"]]
            rows.append('<tr><td>' + label + '</td>' + ''.join(
                '<td><img loading="lazy" src="' + html.escape(path, quote=True) + '"></td>' for path in paths) + '</tr>')
        rows.append('</table>')
    (out / "gallery.html").write_text('\n'.join(rows), encoding="utf-8")
