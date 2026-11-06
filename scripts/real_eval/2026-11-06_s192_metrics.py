#!/usr/bin/env python3
"""Recalculate and report the completed canonical-history benchmark.

F1, IoU, precision, recall and FPR are recalculated from saved per-building
TP/FP/FN/TN, then averaged equally over buildings with defined denominators.
The original evaluator already merged overlapping crops. Its binned AP is
retained, not reconstructed from confusion counts. Existing fusion results
are included; this script does not create new fusion rules or tune thresholds.

Python 3.9+, standard library only. The source run is read-only.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path


METRICS = ("f1", "iou", "precision", "recall", "fpr", "ap_binned")
COUNTS = ("tp", "fp", "fn", "tn")
DEFAULT_RUN = (
    "/home/sasha/Facade_Change_Benchmark/runs/2026-11-04-s192-full-001"
)


def scores(counts):
    tp, fp, fn, tn = (counts[key] for key in COUNTS)
    fractions = {
        "f1": (2 * tp, 2 * tp + fp + fn),
        "iou": (tp, tp + fp + fn),
        "precision": (tp, tp + fp),
        "recall": (tp, tp + fn),
        "fpr": (fp, fp + tn),
    }
    return {key: a / b if b else None for key, (a, b) in fractions.items()}


def check_equal(actual, expected, label):
    if actual is None or expected is None:
        matches = actual is expected
    else:
        matches = math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)
    if not matches:
        raise ValueError(f"Saved/recalculated mismatch at {label}: {expected} vs {actual}")


def aggregate(summary):
    output, seen, split_buildings, supports = [], set(), {}, {}
    collections = (("single", summary["results"]),
                   ("fusion", summary.get("fusion_results", [])))
    for result_type, results in collections:
        for result in results:
            identity = tuple(result[key] for key in ("method", "mode", "split"))
            if identity in seen:
                raise ValueError(f"Duplicate result: {identity}")
            seen.add(identity)
            grouped, sample_keys, support = defaultdict(list), set(), {}
            for saved in result["metrics"]["per_building"]:
                building = saved["building_id"]
                split = result["split"]
                if split_buildings.setdefault(building, split) != split:
                    raise ValueError(f"Building appears in multiple splits: {building}")
                key = (saved["slice_type"], saved["slice"])
                sample_key = (building, *key)
                if sample_key in sample_keys:
                    raise ValueError(f"Duplicate building/slice: {identity}, {sample_key}")
                sample_keys.add(sample_key)
                counts = {name: saved[name] for name in COUNTS}
                if any(type(v) is not int or v < 0 for v in counts.values()):
                    raise ValueError(f"Invalid confusion counts: {identity}, {sample_key}")
                values = scores(counts)
                for name, value in values.items():
                    check_equal(value, saved[name], f"{identity}/{sample_key}/{name}")
                values["ap_binned"] = saved["ap_binned"]
                if values["ap_binned"] is not None and not (
                        math.isfinite(values["ap_binned"])
                        and 0 <= values["ap_binned"] <= 1):
                    raise ValueError(f"Invalid saved AP: {identity}, {sample_key}")
                positive = counts["tp"] + counts["fn"]
                negative = counts["fp"] + counts["tn"]
                support[sample_key] = (positive, negative)
                for name, value in (("positive_pixels", positive),
                                    ("negative_pixels", negative),
                                    ("known_pixels", positive + negative)):
                    check_equal(value, saved[name], f"{identity}/{sample_key}/{name}")
                grouped[key].append(dict(counts, **values))
            if not grouped:
                raise ValueError(f"No per-building confusion counts for {identity}")
            if supports.setdefault(result["split"], support) != support:
                raise ValueError(f"Methods/modes have different evaluation targets: {identity}")
            recorded = {
                (r["slice_type"], r["slice"]): r for r in result["metrics"]["macro"]
            }
            if set(recorded) != set(grouped):
                raise ValueError(f"Macro/per-building slice mismatch: {identity}")
            for (kind, name), buildings in grouped.items():
                row = dict(zip(("method", "mode", "split"), identity))
                row.update(result_type=result_type, slice_type=kind, slice=name,
                           threshold=result["metrics"]["threshold"],
                           buildings=sum(sum(b[k] for k in COUNTS) > 0 for b in buildings),
                           positive_buildings=sum(b["tp"] + b["fn"] > 0 for b in buildings))
                for metric in METRICS:
                    values = [b[metric] for b in buildings if b[metric] is not None]
                    row[metric] = math.fsum(values) / len(values) if values else None
                    row[metric + "_buildings"] = len(values)
                    check_equal(row[metric], recorded[kind, name][metric],
                                f"{identity}/{kind}/{name}/macro-{metric}")
                row.update({key: sum(b[key] for b in buildings) for key in COUNTS})
                row.update(positive_pixels=row["tp"] + row["fn"],
                           negative_pixels=row["fp"] + row["tn"],
                           known_pixels=sum(row[key] for key in COUNTS))
                output.append(row)
    if not output:
        raise ValueError("No results to report")
    return output


def number(value):
    return "NA" if value is None else f"{value:.6f}"


def build_report(root, summary, rows, source_hash):
    lines = [
        "CANONICAL-HISTORY METRICS", f"Source: {root}",
        f"Summary SHA-256: {source_hash}",
        f"Protocol: {summary['protocol']}",
        f"Calibration: {summary['calibration']}",
        "Recalculated F1/IoU/precision/recall/FPR from saved per-building counts.",
        "Equal-building means; undefined denominators remain NA (not zero).",
        "Overlaps were merged by the original evaluator, before these counts were saved.",
        "AP is the original binned average precision, not an exact raw-score AUPRC.",
        "Class/group rows measure change localisation within annotated classes;",
        "they are not semantic class prediction or semantic mIoU.",
        f"Targets: {summary['target_scope']}",
        f"Independent event labels: {summary['independent_event_labels']}",
        "VAL was used for calibration; TEST independence also depends on checkpoint exposure.",
        "Fusion rows below are existing frozen combinations; no new rules were fitted.",
        "Counts: " + json.dumps(summary.get("counts", {}), ensure_ascii=False),
        "Scores are fractions in [0, 1]; deltas below are also in fraction units.",
        "Per-metric effective building counts and confusion totals are in the CSV.",
    ]
    tables = defaultdict(list)
    for row in rows:
        tables[row["split"], row["slice_type"], row["slice"]].append(row)
    split_order = {"train": 0, "val": 1, "test": 2}
    kind_order = {"overall": 0, "class": 1, "group": 2}
    mode_order = {"previous": 0, "earliest": 1, "history": 2}
    ordered = sorted(tables, key=lambda k: (split_order.get(k[0], 9),
                                          kind_order.get(k[1], 9), k[2]))
    width = max(20, max(len(row["method"]) for row in rows))
    for split, kind, name in ordered:
        lines.extend(["", f"{split.upper()} | {kind.upper()} | {name}"])
        lines.append(f"{'Method':<{width}}  {'Mode':<8} {'Bld':>4} {'PosB':>4} "
                     + " ".join(f"{label:>10}" for label in
                                ("F1", "IoU", "Precision", "Recall", "FPR", "AP(binned)")))
        for row in sorted(tables[split, kind, name],
                          key=lambda r: (mode_order.get(r["mode"], 9), r["method"])):
            lines.append(f"{row['method']:<{width}}  {row['mode']:<8} "
                         f"{row['buildings']:>4} {row['positive_buildings']:>4} "
                         + " ".join(f"{number(row[m]):>10}" for m in METRICS))

    overall = {(r["split"], r["method"], r["mode"]): r
               for r in rows if r["slice_type"] == "overall"}
    for split in sorted({k[0] for k in overall}, key=lambda s: split_order.get(s, 9)):
        lines.extend(["", f"{split.upper()} | HISTORY minus PREVIOUS | ANY_CHANGE",
                      f"{'Method':<{width}} " + " ".join(
                          f"{m:>10}" for m in ("dF1", "dIoU", "dRecall", "dFPR"))])
        for method in sorted({k[1] for k in overall if k[0] == split}):
            before = overall.get((split, method, "previous"))
            after = overall.get((split, method, "history"))
            if before is None or after is None:
                continue
            delta = [after[m] - before[m]
                     if after[m] is not None and before[m] is not None else None
                     for m in ("f1", "iou", "recall", "fpr")]
            lines.append(f"{method:<{width}} " + " ".join(f"{number(v):>10}" for v in delta))
    lines.extend(["", "CHECKPOINT EXPOSURE (recorded by the original run)",
                  json.dumps(summary.get("checkpoint_exposure", {}),
                             ensure_ascii=False, indent=2), ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path(DEFAULT_RUN))
    parser.add_argument("--out", type=Path, help="New output directory; created automatically if omitted")
    args = parser.parse_args()
    root = args.run.expanduser().resolve()
    state = json.loads((root / "run.json").read_text(encoding="utf-8-sig"))
    raw = (root / "summary.json").read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    summary = json.loads(raw.decode("utf-8-sig"))
    if state.get("status") != "completed" or summary.get("status") != "completed":
        raise ValueError("The source run is not completed")
    if summary.get("kind") != "canonical_history_benchmark":
        raise ValueError("Expected a canonical_history_benchmark run")
    if state.get("summary_sha256") != source_hash:
        raise ValueError("summary.json does not match the SHA-256 committed in run.json")
    rows = aggregate(summary)
    report = build_report(root, summary, rows, source_hash)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    out = (args.out or root.parent / f"{stamp}_s192_metrics").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    report_path = out / f"{stamp}_metrics.txt"
    csv_path = out / f"{stamp}_metrics.csv"
    report_path.write_text(report, encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(report)
    print(f"Saved text report: {report_path}")
    print(f"Saved metric table: {csv_path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"Metrics report failed: {exc}") from exc
