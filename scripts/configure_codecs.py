#!/usr/bin/env python
"""Extend the same saved comparison; retain good maps, replace RSCD explicitly."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from facade_change.benchmark_config import benchmark_arguments
from facade_change.benchmark_results import ReuseResults
from facade_change.io import read_json, write_json


def configure(reuse_run=None, out=None, worker_python=None, destination=None, repo_root=ROOT):
    repo = Path(repo_root).resolve()
    original = benchmark_arguments({"benchmark_config": str(repo / "configs/benchmark.local.json")})
    if reuse_run:
        candidate = Path(reuse_run).expanduser().resolve()
    else:
        # Prefer the most complete comparison, then the latest completed run.
        candidates = []
        for file in (repo / "runs").glob("*/run.json"):
            record = read_json(file)
            if (record.get("kind") == "hypothesis_benchmark" and record.get("status") == "completed_exploratory"
                    and all(name in record.get("config", {}).get("methods", []) for name in ("rscd_cmu", "rscd_diff_cmu", "rscd_pscd"))):
                candidates.append((len(record["config"]["methods"]), file.stat().st_mtime_ns, file.parent))
        if not candidates:
            raise FileNotFoundError("No completed all-method comparison; specify --reuse-run with its complete directory")
        candidate = max(candidates, key=lambda row: row[:2])[2]
    reuse = ReuseResults(candidate)
    if reuse.summary.get("selected_base_count") != 10 or reuse.summary.get("selected_case_count") != 60:
        raise ValueError("This stage requires the existing ten-crop, sixty-case comparison")
    rscd = [name for name in ("rscd_cmu", "rscd_diff_cmu", "rscd_pscd") if name in reuse.methods]
    if len(rscd) != 3:
        raise ValueError("Expected the completed comparison with all three RSCD checkpoints")
    for name in rscd:
        if name not in original.get("method_options", {}):
            raise ValueError(f"Missing local RSCD source/weight options: {name}")
    python = str(Path(worker_python or sys.executable).expanduser().resolve())
    if not Path(python).is_file():
        raise FileNotFoundError(f"Worker Python missing: {python}")
    added = ["jpegls_mod256", "h264_rgb"]
    method_options = {name: original["method_options"][name] for name in rscd}
    method_options.update({name: {"worker_python": python, "tile_size": 32, "stride": 16} for name in added})
    destination = Path(destination).expanduser().resolve() if destination else repo / "configs/codecs.local.json"
    if destination.exists():
        raise FileExistsError(f"Configuration already exists: {destination}; use --config-out for a new file")
    recompute = rscd + [name for name in added if name in reuse.methods]
    config = {"dataset_run": original["dataset_run"],
              "out": str(Path(out).expanduser().resolve() if out else repo / "runs/2026-10-07-codecs-001"),
              "reuse_run": str(candidate), "selection_path": str(candidate / "selection.json"),
              "methods": rscd + added, "recompute_methods": recompute,
              "device": original.get("device", "cuda:0"),
              "trust_checkpoint": original.get("trust_checkpoint", False),
              "max_bases_per_split": 0, "method_options": method_options}
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, config)
    return {"config": str(destination), "reuse_run": str(candidate),
            "cached_methods": [name for name in reuse.methods if name not in recompute],
            "recompute_methods": recompute, "new_methods": [name for name in added if name not in reuse.methods]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reuse-run")
    p.add_argument("--out")
    p.add_argument("--worker-python")
    p.add_argument("--config-out", dest="destination")
    args = vars(p.parse_args())
    print(json.dumps(configure(**args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
