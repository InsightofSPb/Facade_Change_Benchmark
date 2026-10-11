"""Paid colour-update feasibility, independently decodable supported RGB tiles."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from ..io import sha256, write_json
from ..methods.rgb import RGBScorer
from .common import (ExperimentRun, SCOPE, assert_inputs_unchanged, csv_rows,
                     evaluation_inputs, parent_selection, positive_integer,
                     reports, rgb_inputs, metric_row, inspection_coverage)
from ..hypothesis_benchmark import _building_macro
from . import paid_codec


def tiles(shape, size, stride):
    for row in range(0, shape[0], stride):
        for col in range(0, shape[1], stride):
            yield row, col, slice(row, min(row + size, shape[0])), slice(col, min(col + size, shape[1]))


def run_paid(dataset_run, out, max_buildings_per_split=2, max_bases_per_building=1,
             scenarios_per_base=2, seed=42, tile_size=32, stride=16,
             proposal_threshold=.08, fpr_targets=(.01, .05), source_lineage=None):
    positive_integer(tile_size, "tile_size")
    positive_integer(stride, "stride")
    if stride > tile_size:
        raise ValueError("stride must not leave gaps between tiles")
    if isinstance(proposal_threshold, bool) or not np.isfinite(proposal_threshold) or not 0 <= proposal_threshold <= 1:
        raise ValueError("proposal_threshold must be a finite RGB-DIFF probability")
    selection = parent_selection(dataset_run, max_buildings_per_split, max_bases_per_building,
                                 scenarios_per_base, seed)
    run = ExperimentRun(out, "paid_hypothesis_feasibility", selection,
                        {"tile_size": tile_size, "stride": stride,
                         "source_benchmark": source_lineage,
                         "proposal_threshold": float(np.float32(proposal_threshold)),
                         "fpr_targets": list(fpr_targets), "codec": "stdlib zlib; fixed level 9",
                         "update": "paid support bitmap + 1 mode and 3 colour bytes: fill or modular shift",
                         "nuisance": "paid identity or quantized RGB affine; both arms minimize full cost independently",
                         "lossless_scope": "reference-support RGB bytes; unsupported decoder pixels retain reference",
                         "reference_cost": "reference RGB and geometric support shared in both arms; sha256-bound in each stream",
                         "decision": "score > VAL threshold >= 0; max tile delta bits/support-pixel inside transmitted C",
                         "oracle": "oracle_visible_GT_paid: labels==1 paid bitmap; filename oracle_paid; diagnostic only, no calibration/ranking/operational claim",
                         "cost_scope": "independent overlapping tile streams; totals not whole-image compression rate",
                         "source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}})
    items, checked, ledger, oracle_rows, proposal_rows = [], {}, [], [], []
    (run.out / "streams").mkdir()
    for method in ("rgb_paid", "rgb_diff"):
        (run.out / "scores" / method).mkdir(parents=True)
    try:
        scorer = RGBScorer("rgb_diff")
        for number, case in enumerate(selection["cases"], 1):
            reference, target, support = rgb_inputs(selection, case, checked)
            raw_diff = scorer(reference, target, support)
            candidate = (raw_diff > np.float32(proposal_threshold)) & support
            score = np.full(support.shape, np.float32(-1e30))
            score[~support] = np.nan
            # Evaluation truth is read separately; it enters only the named oracle.
            _, _, eval_support, labels = evaluation_inputs(selection, case, checked)
            if not np.array_equal(eval_support, support):
                raise ValueError("Evaluation changed model support")
            oracle = (labels == 1) & support
            proposal_rows.append(metric_row(case, raw_diff, labels, proposal_threshold, "fixed_rgb_proposal"))
            costs = {"h0": 0, "rgb_paid": 0, "oracle_paid": 0}
            bundles = {method: bytearray() for method in costs}
            started = time.perf_counter()
            for row, col, ys, xs in tiles(support.shape, tile_size, stride):
                s = support[ys, xs]
                if not s.any():
                    continue
                a, b = reference[ys, xs], target[ys, xs]
                c, truth = candidate[ys, xs], oracle[ys, xs]
                h0, account0 = paid_codec.encode(a, b, s, None)
                h1, account1 = paid_codec.encode(a, b, s, c)
                upper, account_oracle = paid_codec.encode(a, b, s, truth)
                for method, stream, accounting in (("h0", h0, account0), ("rgb_paid", h1, account1),
                                                    ("oracle_paid", upper, account_oracle)):
                    decoded = paid_codec.decode(a, s, stream)
                    if not np.array_equal(decoded[s], b[s]):
                        raise ValueError("Paid codec failed exact supported RGB round trip")
                    if accounting["total_bits"] != len(stream) * 8:
                        raise ValueError("Paid codec accounting disagrees with actual stream size")
                    offset = len(bundles[method])
                    bundles[method].extend(stream)
                    costs[method] += len(stream) * 8
                    ledger.append({"case_id": case["case_id"], "building_id": case["building_id"],
                                   "split": case["split"], "row": row, "col": col, "method": method,
                                   "supported_pixels": int(s.sum()), "stream_offset": offset,
                                   "stream_bytes": len(stream), "stream_sha256": hashlib.sha256(stream).hexdigest(),
                                   "roundtrip_verified": True, **accounting, **accounting["partitions"]})
                delta = np.float32((len(h0) - len(h1)) * 8 / int(s.sum()))
                destination = score[ys, xs]
                destination[c] = np.maximum(destination[c], delta)
                oracle_rows.append({"case_id": case["case_id"], "building_id": case["building_id"],
                                    "split": case["split"], "state": case["state"],
                                    "scenario_id": case["scenario_id"], "row": row, "col": col,
                                    "oracle_mask_positive_pixels": int(truth.sum()),
                                    "delta_bits": (len(h0)-len(upper))*8,
                                    "delta_bits_per_supported_pixel": (len(h0)-len(upper))*8/int(s.sum())})
            for method, bundle in bundles.items():
                (run.out / "streams" / f"{case['case_id']}.{method}.bin").write_bytes(bundle)
            score_path = run.out / "scores" / "rgb_paid" / f"{case['case_id']}.npy"
            np.save(score_path, score, allow_pickle=False)
            np.save(run.out / "scores" / "rgb_diff" / f"{case['case_id']}.npy", raw_diff, allow_pickle=False)
            items.extend({"case": case, "scores": values, "labels": labels, "method": method}
                         for method, values in (("rgb_paid", score), ("rgb_diff", raw_diff)))
            run.event("case_scored", case_id=case["case_id"], score_sha256=sha256(score_path),
                      actual_bits=costs, full_diagnostic_seconds=time.perf_counter()-started,
                      all_streams_roundtrip_verified=True)
            print(f"Paid H0/H1 {number}/{len(selection['cases'])}: {case['split']} {case['building_id']} {case['state']}/{case['scenario_id']}; delta={costs['h0']-costs['rgb_paid']} bits", flush=True)
        csv_rows(run.out / "tile_costs.csv", ledger)
        csv_rows(run.out / "oracle_diagnostics.csv", oracle_rows)
        csv_rows(run.out / "proposal_coverage.csv", proposal_rows)
        methods = reports(run, items, ("rgb_paid", "rgb_diff"), fpr_targets)
        inspection = inspection_coverage(run, items)
        assert_inputs_unchanged(selection, checked)
        summary = {"status": "completed_exploratory", "scope": SCOPE, "methods": methods,
                   "case_count": len(selection["cases"]), "actual_stream_count": len(ledger),
                   "all_streams_roundtrip_verified": True,
                   "proposal_stage": {part: _building_macro([r for r in proposal_rows if r["split"] == part]) for part in ("train", "val")},
                   "inspection_coverage": inspection,
                   "limitations": ["Simple colour update, not semantic state C,V or event ontology",
                                   "Identity/affine nuisance cannot model arbitrary shadow, blur, or occlusion",
                                   "GT oracle is diagnostic, not a trained or usable detector and not a mathematical upper bound",
                                   "Lossless on shared reference support only; overlap stream costs are not whole-image bitrate",
                                   "VAL used for calibration; independent future held-out facades needed"]}
        return run.finish(summary, checked)
    except (Exception, KeyboardInterrupt) as exc:
        run.fail(exc)
        raise
