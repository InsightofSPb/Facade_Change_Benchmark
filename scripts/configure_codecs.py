#!/usr/bin/env python
"""Extend the same saved comparison; retain good maps, replace RSCD explicitly."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from facade_change.benchmark_config import benchmark_arguments
from facade_change.benchmark_results import ReuseResults
from facade_change.benchmark_subset import validate_input_hashes
from facade_change.io import read_json, sha256


def _dataset_fingerprint(dataset_run):
    # Verify metadata and its hashes without decoding RGB or touching TEST data.
    h0 = importlib.import_module("facade_change.2026-10-04_msdzip_h0")
    return h0._parent(dataset_run)[-1]


def _training_options(training_run, dataset_run, reuse):
    run = Path(training_run).expanduser().resolve()
    record = read_json(run / "run.json")
    if (not isinstance(record, dict) or record.get("schema_version") != 1
            or record.get("kind") != "arib_h0_training"
            or record.get("status") != "completed_exploratory"):
        raise ValueError("ArIB requires a completed H0 training run with schema version 1")
    config = record.get("config")
    if not isinstance(config, dict):
        raise ValueError("ArIB training configuration is missing")
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False,
                                      allow_nan=False).encode()).hexdigest()
    if record.get("config_sha256") != digest:
        raise ValueError("ArIB training configuration hash changed")
    representations = config.get("representations")
    if (not isinstance(representations, list) or not representations
            or any(value not in ("abs", "mod256") for value in representations)
            or len(set(representations)) != len(representations)):
        raise ValueError("ArIB training representations must be unique abs/mod256 values")
    if config.get("tile_size") != 32:
        raise ValueError("This comparison requires ArIB training on 32-pixel tiles")
    if config.get("author_config") not in {"imagenet32_config", "imagenet64_config",
                                         "imagenet64_small_config", "cifar_config"}:
        raise ValueError("Unsupported original ArIB configuration")
    validate_input_hashes(config.get("input_sha256"), reuse.input_sha256)
    validate_input_hashes(config["input_sha256"], _dataset_fingerprint(dataset_run))
    artifacts = record.get("artifact_sha256")
    if not isinstance(artifacts, dict):
        raise ValueError("ArIB training artifact hashes are missing")
    for relative in ["summary.json"] + [f"{rep}/{component}.pth"
            for rep in representations for component in ("sig", "ins")]:
        path = (run / relative).resolve()
        if run not in path.parents:
            raise ValueError(f"ArIB training artifact escapes its run: {relative}")
        if not path.is_file() or artifacts.get(relative) != sha256(path):
            raise ValueError(f"ArIB training artifact missing or hash changed: {relative}")
    summary = read_json(run / "summary.json")
    if (not isinstance(summary, dict) or summary.get("status") != "completed_exploratory"
            or not isinstance(summary.get("representations"), dict)
            or set(summary["representations"]) != set(representations)):
        raise ValueError("ArIB training summary disagrees with completed representations")
    return run, config


def configure(reuse_run=None, out=None, worker_python=None, destination=None, repo_root=ROOT,
              training_run=None, source_root=None, cost_mode="bitstream", neural_device="cuda:0"):
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
    protected = [candidate, Path(original["dataset_run"]).expanduser().resolve()]
    if training_run is not None:
        if cost_mode not in {"bitstream", "theoretical"}:
            raise ValueError("ArIB cost_mode must be bitstream or theoretical")
        run, training = _training_options(training_run, original["dataset_run"], reuse)
        protected.append(run)
        source = Path(source_root).expanduser().resolve() if source_root else repo / "third_party/arib_bps"
        for representation in ("abs", "mod256"):
            if representation not in training["representations"]:
                continue
            name = "arib_bps_" + representation
            added.append(name)
            method_options[name] = {"worker_python": python, "source_root": str(source),
                "training_run": str(run), "device": neural_device, "tile_size": 32, "stride": 16,
                "seed": training.get("seed", 42), "worker_timeout": 3600, "cost_mode": cost_mode}
    destination = Path(destination).expanduser().resolve() if destination else repo / "configs/codecs.local.json"
    if destination.exists():
        raise FileExistsError(f"Configuration already exists: {destination}; use --config-out for a new file")
    output = Path(out).expanduser().resolve() if out else repo / "runs/2026-10-07-codecs-001"
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; use --out for a new run")
    for root in protected:
        if any(path == root or root in path.parents for path in (destination, output)):
            raise ValueError(f"Configuration and output must be outside the immutable input run: {root}")
    if destination == output or output in destination.parents or destination in output.parents:
        raise ValueError("Configuration must be outside the new benchmark output directory")
    recompute = rscd + [name for name in added if name in reuse.methods]
    config = {"dataset_run": original["dataset_run"],
              "out": str(output),
              "reuse_run": str(candidate), "selection_path": str(candidate / "selection.json"),
              "methods": rscd + added, "recompute_methods": recompute,
              "device": original.get("device", "cuda:0"),
              "trust_checkpoint": original.get("trust_checkpoint", False),
              "max_bases_per_split": 0, "method_options": method_options}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return {"config": str(destination), "reuse_run": str(candidate),
            "cached_methods": [name for name in reuse.methods if name not in recompute],
            "recompute_methods": recompute, "new_methods": [name for name in added if name not in reuse.methods]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reuse-run")
    p.add_argument("--out")
    p.add_argument("--worker-python")
    p.add_argument("--training-run", help="Completed ArIB H0 run; add only its fitted residual representations")
    p.add_argument("--source-root", help="Original ArIB sources (default: third_party/arib_bps)")
    p.add_argument("--cost-mode", choices=("bitstream", "theoretical"), default="bitstream")
    p.add_argument("--neural-device", default="cuda:0")
    p.add_argument("--config-out", dest="destination")
    args = vars(p.parse_args())
    print(json.dumps(configure(**args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
