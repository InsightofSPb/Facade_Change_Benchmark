#!/usr/bin/env python3
"""CPU-only VAL metrics from a fixed snapshot of an active temporal run."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from facade_change.io import sha256, write_json
from facade_change.research.common import csv_rows, digest

reports = importlib.import_module("facade_change.research.2026-10-14_temporal_reports")
DATE = "2026-10-17"


def _snapshot(root):
    """Freeze names once; a committed unit marker and its artifact are immutable."""
    paths = sorted((root / "units").glob("*.json"))
    hashes, documents, records = {}, {}, []
    for name in ("run.json", "experiment_contract.json", "preflight.json",
                 "resolved_config.json", "request.json", "method_status.json"):
        path = root / name
        if path.exists():
            raw = path.read_bytes()
            documents[name] = json.loads(raw)
            hashes[name] = hashlib.sha256(raw).hexdigest()
    for path in paths:
        raw = path.read_bytes()
        record = json.loads(raw)
        if path.stem != record["unit_id"]:
            raise ValueError(f"Unit identity differs from marker filename: {path}")
        hashes[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        records.append({key: value for key, value in record.items() if key != "method_metadata"})
    run = documents["run.json"]
    if digest(run["config"]) != run["config_sha256"]:
        raise ValueError("Source run configuration hash disagrees")
    return records, documents, hashes


def _keys(rows):
    result = {(row["sequence_id"], row["frame"]) for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate sequence/frame inside a method/variant/mode/target group")
    return result


def _expected(groups, documents):
    count = documents["preflight.json"]["coverage"]["synthetic"]["by_split"]["val"]
    allowed = set(documents["experiment_contract.json"]["sequence_ids"])
    candidates = []
    for group, rows in groups.items():
        if group[2:] != ("previous", "step_event"):
            continue
        keys = _keys(rows)
        by_sequence = defaultdict(set)
        metadata = {}
        for row in rows:
            sequence = row["sequence_id"]
            by_sequence[sequence].add(row["frame"])
            identity = row["building_id"], row["arm"]
            if sequence in metadata and metadata[sequence] != identity:
                raise ValueError("Sequence identity changes within committed units")
            metadata[sequence] = identity
        if (len(by_sequence) == count and all(frames == {1, 2, 3, 4} for frames in by_sequence.values())
                and set(by_sequence) <= allowed):
            if {arm for _, arm in metadata.values()} != {"clean", "event"}:
                raise ValueError("Complete synthetic VAL reference lacks clean/event arms")
            candidates.append((group, keys, metadata))
    if not candidates:
        raise ValueError("No complete five-frame synthetic VAL previous group; cannot certify report coverage")
    candidates.sort(key=lambda item: (item[0][0] != "msdzip_mod256", item[0]))
    group, expected, metadata = candidates[0]
    if any(keys != expected or identities != metadata for _, keys, identities in candidates):
        raise ValueError("Complete previous groups disagree on synthetic VAL identities")
    return expected, metadata, group


def _load(root, record, verified):
    """Read only metric arrays; never decompress RGB, auxiliary heads or memory."""
    path = (root / record["artifact_path"]).resolve()
    if root not in path.parents:
        raise ValueError("Artifact escapes the source run")
    stat = path.stat()
    stamp = stat.st_size, stat.st_mtime_ns
    if path in verified:
        if verified[path] != (stamp, record["artifact_sha256"]):
            raise ValueError(f"Committed artifact changed during report: {path}")
    else:
        if sha256(path) != record["artifact_sha256"]:
            raise ValueError(f"Corrupt committed artifact: {path}")
        verified[path] = stamp, record["artifact_sha256"]
    names = ["scores", "labels", "support"]
    if reports._is_native(record):
        names.append("native_prediction")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in names}
    scores, labels, support = (arrays[name] for name in names[:3])
    if scores.ndim != 2 or scores.dtype != np.float32 or labels.dtype != np.uint8 or support.dtype != bool:
        raise ValueError("Invalid saved score/label/support dtypes")
    if labels.shape != scores.shape or support.shape != scores.shape or not np.isin(labels, [0, 1, 255]).all():
        raise ValueError("Saved label/support grid or label values disagree")
    if not np.isfinite(scores[support]).all() or np.any((labels != 255) & ~support):
        raise ValueError("Nonfinite supported score or label outside support")
    if record.get("hypothesis") not in {"H0", "H1"} or record["hypothesis"] == "H0" and np.any(labels == 1):
        raise ValueError("Invalid target-specific H0/H1 labels")
    if "native_prediction" in arrays:
        prediction = arrays["native_prediction"]
        if prediction.dtype != bool or prediction.shape != scores.shape:
            raise ValueError("Native method requires its saved boolean prediction")
    return arrays


def run(run_path, out_path):
    root, out = Path(run_path).expanduser().resolve(), Path(out_path).expanduser().resolve()
    if out == root or root in out.parents:
        raise ValueError("Report output must be outside the read-only source run")
    if out.exists():
        raise FileExistsError(f"Use a NEW report output directory: {out}")
    records, documents, source_hashes = _snapshot(root)
    groups = defaultdict(list)
    for record in records:
        if record.get("domain") == "synthetic" and record.get("split") == "val":
            group = reports._group(record)
            groups[group]  # Preserve initialized or incomplete groups in coverage.
            if record.get("status") in reports.SUCCESS and record.get("frame", 0) > 0:
                groups[group].append(record)
    expected, identities, reference_group = _expected(groups, documents)
    config = documents.get("resolved_config.json", documents["experiment_contract.json"]["config"])
    for method in config.get("methods", []):
        variants = ["current_train" if method.startswith("msdzip_") else "frozen"]
        if method.removeprefix("msdzip_") in config.get("legacy_checkpoints", {}):
            variants.append("legacy_provenance_qualified")
        for variant in variants:
            for mode in ("previous", "earliest", *(["history"] if method in config.get("history_methods", []) else [])):
                groups[(method, variant, mode, "earliest_state" if mode == "earliest" else "step_event")]
    out.mkdir(parents=True, exist_ok=False)
    print(f"Snapshot: {len(records)} committed markers; expected VAL units/group: {len(expected)}", flush=True)
    rows, coverage, calibration, signatures, verified = [], [], [], {}, {}
    for group, selected in sorted(groups.items()):
        keys = _keys(selected)
        if keys - expected:
            raise ValueError(f"Unexpected synthetic VAL identities: {group}")
        for record in selected:
            if (record["building_id"], record["arm"]) != identities[record["sequence_id"]]:
                raise ValueError("Group facade or clean/event identity differs from reference")
        complete = keys == expected
        identity = dict(zip(reports.GROUP_FIELDS, group))
        coverage.append({**identity, "completed_units": len(keys), "expected_units": len(expected),
                         "status": "complete" if complete else "incomplete_skipped",
                         "missing_keys": sorted(expected - keys)})
        print(f"{' / '.join(group)}: {len(keys)}/{len(expected)} "
              f"{'computing CPU metrics' if complete else 'incomplete; metrics skipped'}", flush=True)
        if not complete:
            continue
        kinds = {reports._is_native(record) for record in selected}
        if len(kinds) != 1:
            raise ValueError("Mixed native and continuous outputs inside one method group")
        if True in kinds:
            choice = {"status": "author_native", "choices": [{"threshold": None, "fpr_target": None}], "curve": []}
        else:
            def controls():
                for record in selected:
                    if reports._calibration_control(record):
                        yield {"record": record, "arrays": _load(root, record, verified)}
            choice = reports.calibrate_thresholds(controls, (.01, .05))
            if choice["status"] != "calibrated":
                raise ValueError(f"Complete group has no evaluated VAL H0 calibration: {group}")
        calibration.append({**identity, **choice})
        for record in selected:
            arrays = _load(root, record, verified)
            if record["mode"] in {"previous", "history"}:
                match = record["method"], record["variant"], record["sequence_id"], record["frame"]
                signature = reports._array_digest(arrays)
                if match in signatures and signatures[match] != signature:
                    raise ValueError("Previous/history labels or support disagree; comparison is invalid")
                signatures[match] = signature
            rows.extend(reports._metric(record, arrays, setting) for setting in choice["choices"])
    metrics = reports._aggregates(rows)
    history = reports._history_summary(reports._paired_history(rows))
    summary = {"scope": "Exploratory synthetic VAL snapshot; no TEST, real metrics, inference or confidence intervals",
               "source_run": str(root), "snapshot_utc": datetime.now(timezone.utc).isoformat(),
               "source_run_config_sha256": documents["run.json"]["config_sha256"],
               "reference_group": list(reference_group), "expected_keys": sorted(expected),
               "source_json_sha256": source_hashes, "verified_archive_count": len(verified),
               "method_status_snapshot": {name: {key: value for key, value in status.items() if key != "method_metadata"}
                                          for name, status in documents.get("method_status.json", {}).items()},
               "coverage": coverage, "calibration": calibration, "metrics": metrics, "history_comparisons": history}
    write_json(out / f"{DATE}_quick_summary.json", summary)
    write_json(out / f"{DATE}_source_units.json", records)
    for name, values in (("metrics", metrics), ("units", rows), ("coverage", coverage), ("history", history)):
        csv_rows(out / f"{DATE}_quick_{name}.csv", values)
    lines = [summary["scope"], "Method / variant / mode / target | operating point | Recall | H0 FPR | facades"]
    for row in metrics:
        percent = lambda value: "n/a" if value is None else f"{value:.2%}"
        operating = "native" if row["fpr_target"] is None else f"VAL H0 <= {row['fpr_target']:.0%}"
        lines.append(f"{' / '.join(str(row[key]) for key in reports.GROUP_FIELDS)} | {operating} | "
                     f"{percent(row['recall'])} | {percent(row['h0_pixel_fpr'])} | {row['building_count']}")
    for row in coverage:
        if row["status"] != "complete":
            lines.append(f"INCOMPLETE: {' / '.join(row[key] for key in reports.GROUP_FIELDS)} "
                         f"{row['completed_units']}/{row['expected_units']}; no metrics")
    (out / f"{DATE}_quick_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Saved compact report: {out}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Existing running or completed temporal experiment")
    parser.add_argument("--out", required=True, help="New report directory outside the source run")
    args = parser.parse_args(argv)
    run(args.run, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
