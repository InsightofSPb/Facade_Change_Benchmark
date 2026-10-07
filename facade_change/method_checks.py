"""Small local checks before the same frozen ten-crop comparison."""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
from PIL import Image

from .benchmark_config import benchmark_arguments
from .benchmark_subset import replay_selection
from .hypothesis_benchmark import _select
from .io import finish_record, new_directory, run_record, sha256, write_json
from .methods.remote import RemotePool, RemoteScorer


def score_diagnostics(scores, support, prediction=None):
    values = scores[support]
    quantiles = np.quantile(values.astype(np.float64), [0, .01, .5, .99, 1])
    return {"min": float(quantiles[0]), "p01": float(quantiles[1]), "median": float(quantiles[2]),
            "p99": float(quantiles[3]), "max": float(quantiles[4]),
            "distinct_float32_scores": int(len(np.unique(values))),
            "positive_at_zero_fraction": float((values > 0).mean()),
            "native_changed_fraction": float(prediction[support].mean()) if prediction is not None else None}


def check_methods(benchmark_config, out, methods, max_val_bases=1, bitstream_check=False):
    args = benchmark_arguments({"benchmark_config": benchmark_config})
    options = args.get("method_options", {})
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Select distinct methods")
    from .methods.registry import ALL_METHODS
    allowed = {name for name in ALL_METHODS if name.startswith(("rscd_", "jpegls_", "arib_bps_")) or name in {"h264_rgb", "bcm_net_rgb"}}
    if set(methods) - allowed:
        raise ValueError("methods-check accepts RSCD and lossless image/video codec adapters")
    controls, fingerprints = [], None
    rng = np.random.default_rng(42)
    for name in ("unchanged", "paint_patch", "wraparound", "wraparound_reverse"):
        a = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        b = a.copy()
        if name == "paint_patch":
            b[10:18, 10:18] = 180
        if name == "wraparound":
            a.fill(255)
            b.fill(0)
        if name == "wraparound_reverse":
            a.fill(0)
            b.fill(255)
        controls.append((name, a, b, np.ones((32, 32), dtype=bool)))
    validation = []
    if any(method.startswith("rscd_") for method in methods):
        if type(max_val_bases) is not int or max_val_bases < 1:
            raise ValueError("max_val_bases must be positive")
        h0 = importlib.import_module(".2026-10-04_msdzip_h0", __package__)
        root, parent, index, split, fingerprints = h0._parent(args["dataset_run"])
        expected = {(state, scenario["id"]) for state in parent["config"]["states"] for scenario in parent["config"]["scenarios"]}
        bases, cases = _select(index, split, 0, expected)
        selection = args.get("selection_path") or (Path(args["reuse_run"]) / "selection.json" if args.get("reuse_run") else None)
        if not selection:
            raise ValueError("RSCD checks require the original saved selection")
        bases, cases, _ = replay_selection(bases, cases, fingerprints, selection)
        selected = [base for base in bases if base["split"] == "val"][:max_val_bases]
        checked = {}
        for base in selected:
            selected_cases = [case for case in cases if case["base_id"] == base["base_id"]]
            for case in selected_cases:
                a, b, support = h0._rgb_inputs(root, parent, base, case, checked)
                validation.append((case["case_id"], a, b, support))
            a, _, support = h0._rgb_inputs(root, parent, base, selected_cases[0], checked)
            validation.append((base["base_id"]+"-identity-AA", a, a.copy(), support))
    if any(method.startswith("arib_bps_") or method == "bcm_net_rgb" for method in methods) and fingerprints is None:
        h0 = importlib.import_module(".2026-10-04_msdzip_h0", __package__)
        _, _, _, _, fingerprints = h0._parent(args["dataset_run"])
    out = new_directory(out)
    record = run_record("method_checks", {"benchmark_config": str(Path(benchmark_config).resolve()),
        "benchmark_config_sha256": sha256(benchmark_config),
        "method_options": {method: options.get(method, {}) for method in methods},
        "methods": list(methods), "max_val_bases": max_val_bases, "input_sha256": fingerprints,
        "bitstream_check": bool(bitstream_check),
        "scope": "small codec roundtrips and validation-only RSCD/identity controls; no training, tuning or TEST inference"})
    write_json(out / "run.json", record)
    reports = {}
    try:
        with RemotePool(log_dir=out / "worker_logs") as pool:
            for method in methods:
                scorer_options = {"device": args.get("device", "cpu"),
                                  "trust_checkpoint": args.get("trust_checkpoint", False),
                                  **options.get(method, {})}
                if not method.startswith("rscd_"):
                    scorer_options["tile_size"], scorer_options["stride"] = 32, 32
                if method.startswith("arib_bps_") or method == "bcm_net_rgb":
                    scorer_options["dataset_fingerprint"] = fingerprints
                    if bitstream_check:
                        scorer_options["cost_mode"] = "bitstream"
                scorer = RemoteScorer(method, scorer_options, pool)
                scorer.activate()
                rows = []
                for name, a, b, support in (validation if method.startswith("rscd_") else controls):
                    scores = scorer(a, b, support)
                    rows.append({"control": name, **score_diagnostics(scores, support, scorer.native_prediction),
                                 "codec_stats": scorer.metadata.get("last_codec_stats")})
                    folder = out / method
                    folder.mkdir(exist_ok=True)
                    np.save(folder / (name+".npy"), scores, allow_pickle=False)
                    if scorer.native_prediction is not None:
                        Image.fromarray(scorer.native_prediction.astype(np.uint8)*255).save(folder / (name+"-native.png"))
                    print(f"{method} / {name}: {rows[-1]}", flush=True)
                reports[method] = {"metadata": scorer.metadata, "controls": rows}
                write_json(out / "report.json", reports)
                scorer.close()
        finish_record(out, record, "completed_needs_review")
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        finish_record(out, record, "failed")
        raise
    return {"out": str(out), "status": "completed_needs_review", "methods": list(reports)}
