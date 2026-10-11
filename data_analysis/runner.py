"""Reuse verified benchmark maps once for comparison, screening and recall audit."""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

from facade_change.benchmark_results import ReuseResults
from facade_change.benchmark_subset import replay_selection
from facade_change.hypothesis_benchmark import _building_macro, _case_inputs, _select
from facade_change.io import finish_record, new_directory, read_json, run_record, sha256, write_json

from .core import BASELINES, COMPRESSION, FUSION, METHODS, SPLITS, case_identity, pair_counts, prediction_metrics, slices
from .recall import diagnostic_macro, diagnostic_row, feature_rows
from .report import write_reports

SCOPE = ("EXPLORATORY cached VAL/TEST analysis with frozen original thresholds; "
         "GeoSCD AND MSDZip fixed before this calculation; TEST previously inspected; "
         "recall losses and score distributions are diagnostics, not causal proof; "
         "procedural controls, no training, threshold tuning or confidence intervals")


def verified_fusion(path, reuse, fingerprint, selected):
    """Verify an optional completed result without depending on archive scripts."""
    root = Path(path).expanduser().resolve()
    digest = sha256(root / "run.json")
    record = read_json(root / "run.json")
    if record.get("kind") != "frozen_geoscd_fusion" or record.get("status") != "completed_exploratory":
        raise ValueError("Fusion reference must be a completed frozen_geoscd_fusion run")
    config = record.get("config", {})
    serialized = json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    if hashlib.sha256(serialized).hexdigest() != record.get("config_sha256"):
        raise ValueError("Fusion configuration SHA256 disagrees")
    source = config.get("source", {})
    if any(source.get(key) != reuse.provenance[key] for key in ("run_sha256", "config_sha256", "selection_sha256")):
        raise ValueError("Fusion reference comes from another benchmark")
    methods = config.get("methods", [])
    if (not isinstance(methods, list) or len(set(methods)) != len(methods)
            or not {COMPRESSION, "geoscd", FUSION}.issubset(methods)):
        raise ValueError("Fusion reference must include MSDZip, GeoSCD and their fixed AND result")
    if (config.get("fusion_method") != FUSION or config.get("rule") != "AND"
            or config.get("requested_splits") != list(SPLITS)
            or config.get("input_sha256") != fingerprint
            or config.get("case_ids") != [c["case_id"] for c in selected]
            or not {COMPRESSION, "geoscd"}.issubset(config.get("thresholds", {}))
            or any(method not in reuse.thresholds or threshold != reuse.thresholds[method]["threshold"]
                   for method, threshold in config.get("thresholds", {}).items())):
        raise ValueError("Fusion rule, selection or thresholds differ from the frozen benchmark")
    hashes = {}
    for relative in ("frozen_contract.json", "summary.json", "metrics.csv", "selected_cases.json"):
        value = sha256(root / relative)
        if record.get("artifact_sha256", {}).get(relative) != value:
            raise ValueError(f"Fusion artifact changed or SHA256 disagrees: {relative}")
        hashes[relative] = value
    summary = read_json(root / "summary.json")
    selection = read_json(root / "selected_cases.json")
    if (read_json(root / "frozen_contract.json") != config
            or summary.get("status") != "completed_exploratory"
            or summary.get("thresholds") != config.get("thresholds")
            or summary.get("source") != source
            or summary.get("analyzed_case_count") != len(selected)
            or [case_identity(c) for c in selection.get("cases", [])] != [case_identity(c) for c in selected]):
        raise ValueError("Fusion summary/contract/selection disagree")
    if not {COMPRESSION, "geoscd", FUSION}.issubset(summary.get("methods", {})):
        raise ValueError("Fusion summary is missing required original/AND methods")
    with (root / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {}
    expected = {(method, c["case_id"]) for method in config.get("methods", []) for c in selected}
    for row in rows:
        key = row.get("method"), row.get("case_id")
        if key in indexed or key not in expected:
            raise ValueError("Fusion metrics duplicate or change method/case identity")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("Fusion reference is missing per-case metrics")
    return {"root": root, "run_sha256": digest, "hashes": hashes, "summary": summary,
            "rows": indexed, "config_sha256": record["config_sha256"]}


def run_analysis(run_path, out, fusion_run=None, gallery_limit=60):
    if type(gallery_limit) is not int or gallery_limit < 0:
        raise ValueError("Gallery limit must be a nonnegative integer")
    print("Проверка сохранённых карт, исходных порогов и выборки...", flush=True)
    reuse = ReuseResults(run_path)
    if set(METHODS) - set(reuse.methods):
        raise ValueError("Benchmark is missing required methods: " + ", ".join(sorted(set(METHODS) - set(reuse.methods))))
    loader = importlib.import_module("facade_change.2026-10-04_msdzip_h0")._parent
    root, parent, index, split, fingerprint = loader(reuse.record["config"]["dataset_run"])
    signatures = {(state, scenario["id"]) for state in parent["config"]["states"] for scenario in parent["config"]["scenarios"]}
    bases, cases = _select(index, split, 0, signatures)
    bases, cases, _ = replay_selection(bases, cases, fingerprint, reuse._checked("selection.json"))
    reuse.validate_selection(fingerprint, bases, cases)
    selected = [c for c in cases if c["split"] in SPLITS and not c["sham_self_paste"]]
    if set(c["split"] for c in selected) != set(SPLITS):
        raise ValueError("The original selection must include both VAL and TEST; no substitute subset is selected")
    reference = verified_fusion(fusion_run, reuse, fingerprint, selected) if fusion_run else None
    reference_source = ({"run_path": str(reference["root"]), "run_sha256": reference["run_sha256"],
                         "config_sha256": reference["config_sha256"]} if reference else None)
    thresholds = {m: reuse.thresholds[m]["threshold"] for m in METHODS}
    config = {"source": reuse.provenance, "fusion_reference": reference_source, "dataset_run": str(root),
              "input_sha256": fingerprint, "methods": list(METHODS), "fusion_method": FUSION,
              "requested_splits": list(SPLITS), "case_ids": [c["case_id"] for c in selected],
              "thresholds": thresholds, "prediction_rule": "score > np.float32(original threshold); author masks preserved",
              "diagnostic_focus": "Recall losses; not a change to historical threshold/fusion selection",
              "val_screening_rules": ["OR", "AND"], "test_fusion_rules": ["GeoSCD AND MSDZip only"],
              "gallery_limit": gallery_limit, "scope": SCOPE,
              "analysis_source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    out = new_directory(out)
    record = run_record("cached_method_analysis", config)
    write_json(out / "run.json", record)
    write_json(out / "analysis_contract.json", config)
    metric_rows, pair_rows, features, entries, val_fusions = [], [], [], [], []
    checked_hashes, score_hashes = {}, {}
    by_base = {b["base_id"]: b for b in bases}
    try:
        for i, case in enumerate(selected, 1):
            rgb_a, rgb_b, support, labels = _case_inputs(root, parent, by_base[case["base_id"]], case, checked_hashes)
            predictions, scores, raw_scores = {}, {}, {}
            for method in METHODS:
                cached = reuse.load(method, case["case_id"])
                values = cached["scores"]
                if values.shape != labels.shape or not np.array_equal(np.isnan(values), ~support):
                    raise ValueError("Cached score grid/support disagrees with the verified dataset")
                prediction = values > np.float32(thresholds[method])
                if "native_prediction" in cached and reuse.metadata(method).get("output_kind") == "native_mask":
                    if not np.array_equal(prediction, cached["native_prediction"] & support):
                        raise ValueError("Frozen decision changes the author binary mask")
                row = prediction_metrics(case, method, prediction, labels)
                if any(row[key] != cached["row"][key] for key in ("tp", "fp", "fn", "tn", "ignored_pixel_count")):
                    raise ValueError("Recomputed frozen counts disagree with the original benchmark")
                metric_rows.append(row)
                predictions[method], scores[method], raw_scores[method] = prediction, values, cached["raw_scores"]
                score_hashes[cached["row"]["score_path"]] = cached["row"]["score_sha256"]
                raw_path = cached["row"].get("raw_score_path") or cached["row"].get("native_bpb_path")
                if raw_path:
                    score_hashes[raw_path] = sha256(reuse._checked(raw_path))
            compression = predictions[COMPRESSION]
            diagnostics = {}
            for baseline in BASELINES:
                diagnostic = diagnostic_row(case, baseline, pair_counts(compression, predictions[baseline], labels))
                pair_rows.append(diagnostic)
                diagnostics[baseline] = diagnostic
                # Retain the old OR/AND screening capability on VAL only.
                if case["split"] == "val":
                    for rule, mask in (("or", compression | predictions[baseline]), ("and", compression & predictions[baseline])):
                        val_fusions.append(prediction_metrics(case, baseline + "+msdzip_" + rule, mask, labels))
            predictions[FUSION] = predictions["geoscd"] & compression
            metric_rows.append(prediction_metrics(case, FUSION, predictions[FUSION], labels))
            if reference:
                for row in metric_rows[-len(METHODS)-1:]:
                    saved = reference["rows"].get((row["method"], case["case_id"]))
                    if saved and any(row[key] != int(saved[key]) for key in ("tp", "fp", "fn", "tn", "ignored_pixel_count")):
                        raise ValueError("Per-case metrics differ from the previous frozen fusion run")
            features.extend(feature_rows(case, predictions["geoscd"], compression, labels,
                                         scores[COMPRESSION], raw_scores[COMPRESSION], rgb_a, rgb_b, thresholds[COMPRESSION]))
            if gallery_limit:
                entries.append({"case": case, "reference": rgb_a, "source": rgb_b, "labels": labels,
                                "predictions": predictions, "scores": scores, "raw_scores": raw_scores,
                                "recall": diagnostics["geoscd"]})
            print(f"{i}/{len(selected)}: {case['split']} {case['building_id']} {case['state']}/{case['scenario_id']}", flush=True)
        output_methods = (*METHODS, FUSION)
        summary = {"status": "completed_exploratory", "scope": SCOPE, "requested_splits": list(SPLITS),
                   "analyzed_case_count": len(selected), "excluded_sham_case_count": sum(c["sham_self_paste"] for c in cases),
                   "cases_by_split": {p: sum(c["split"] == p for c in selected) for p in SPLITS},
                   "buildings_by_split": {p: len({c["building_id"] for c in selected if c["split"] == p}) for p in SPLITS},
                   "thresholds": thresholds, "fusion_method": FUSION,
                   "aggregation": "case means within facade, then equal facade means; pixel counts diagnostic only",
                   "methods": {m: slices([r for r in metric_rows if r["method"] == m], _building_macro) for m in output_methods},
                   "pairs": {m: slices([r for r in pair_rows if r["baseline"] == m], diagnostic_macro) for m in BASELINES},
                   "val_fusion_diagnostics": {m: slices([r for r in val_fusions if r["method"] == m], _building_macro, ("val",))
                                              for m in sorted({r["method"] for r in val_fusions})},
                   "score_diagnostics": features, "source": reuse.provenance, "fusion_reference": reference_source,
                   "limitations": ["TEST already inspected; diagnostic analysis, not an untouched final test",
                                   "Procedural controls, not real-damage ground truth", "No threshold/rule search on TEST",
                                   "Score/colour/boundary associations do not prove cause", "No confidence intervals"]}
        if reference:
            for method in output_methods:
                if method in reference["summary"].get("methods", {}):
                    if summary["methods"][method]["by_split"] != reference["summary"]["methods"][method]["by_split"]:
                        raise ValueError("Facade-level VAL/TEST parity failed against the previous frozen run")
        write_reports(out, summary, metric_rows, pair_rows, features, entries, gallery_limit)
        write_json(out / "selected_cases.json", {"cases": selected, "thresholds": {m: reuse.thresholds[m] for m in METHODS}})
        write_json(out / "input_artifacts.json", {"dataset_sha256": checked_hashes, "scores_sha256": score_hashes,
                                                 "fusion_artifact_sha256": reference["hashes"] if reference else {}})
        reuse.validate_selection(fingerprint, bases, cases)
        if loader(root)[-1] != fingerprint:
            raise ValueError("Parent provenance changed during analysis")
        for relative, digest in checked_hashes.items():
            if sha256(root / relative) != digest:
                raise ValueError("Dataset artifact changed during analysis")
        for relative, digest in score_hashes.items():
            reuse._checked(relative, digest)
        if reference:
            if sha256(reference["root"] / "run.json") != reference["run_sha256"]:
                raise ValueError("Fusion reference changed during analysis")
            for relative, digest in reference["hashes"].items():
                if sha256(reference["root"] / relative) != digest:
                    raise ValueError("Fusion artifact changed during analysis")
        finish_record(out, record, "completed_exploratory")
        print((out / "summary.txt").read_text(encoding="utf-8"), flush=True)
        print("Отчёт:", out / "summary.json", "\nГалерея:", out / "gallery.html", flush=True)
        return summary
    except (Exception, KeyboardInterrupt) as exc:
        finish_record(out, record, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", str(exc))
        raise
