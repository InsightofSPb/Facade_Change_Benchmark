#!/usr/bin/env python3
"""Compare two committed checkouts on shared synthetic CPU inputs.

Restores the 2026-11-17 scratch comparison's fixture generators and scripted
pool, with stronger artifact coverage. This is not model-quality evidence.
No fetch, weights, real data or API calls. Keep --out for inspection; use a new
directory for every run. --verify-only rechecks saved artifacts without writing.
"""
from __future__ import annotations

import argparse
from fnmatch import fnmatchcase
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch


BASELINE = "bf4f4c47e321259a8023812776c21252c006fd27"
REPORT = "2026-11-17-parity-report.json"
JSON_ARTIFACTS = (
    "rgb/metrics.json", "rgb/threshold_selection.json",
    "history/manifest.json", "agent/results.json", "score/metrics.private.json",
    "agent/questions/*/*/exposure.json", "agent/questions/*/*/messages.json",
    "agent/questions/*/*/calls/*.json", "agent/questions/*/*/retrieval-*.json",
    "agent/questions/*/*/reply-*.json",
)
BINARY_ARTIFACTS = (
    "rgb/scores/**/*.npy", "rgb/predictions/**/*.png", "rgb/heatmaps/**/*.png",
    "history/**/*.png", "agent/**/*.png", "agent/**/*.npy",
)
# These are the entire exclusion list, scoped to artifact AND JSON field path.
# '*' in a field path matches exactly one component, including a list index.
# Replace values, not keys: absent/extra fields still fail comparison.
EXCLUSIONS = (
    ("rgb/metrics.json", "cases/*/scoring_seconds", "wall_clock"),
    ("agent/results.json", "contract_sha256", "source_contract"),
    ("agent/results.json", "results/*/costs/elapsed_seconds", "wall_clock"),
    ("agent/results.json", "results/*/costs/calls/*/sha256", "raw_call_hash"),
    ("score/metrics.private.json", "input_sha256", "run_provenance"),
    ("score/metrics.private.json", "cases/*/costs/elapsed_seconds", "wall_clock"),
    ("score/metrics.private.json", "cases/*/costs/calls/*/sha256", "raw_call_hash"),
    ("score/metrics.private.json", "summaries/*/*/cost_totals/elapsed_seconds",
     "wall_clock"),
    ("score/metrics.private.json",
     "summaries/*/*/families/*/cost_totals/elapsed_seconds", "wall_clock"),
    ("agent/questions/*/*/exposure.json",
     "tools/*/result/calls/*/request/timeout_seconds", "remaining_wall_clock"),
    ("agent/questions/*/*/calls/*.json", "request/timeout_seconds",
     "remaining_wall_clock"),
    ("agent/questions/*/*/calls/*.json", "request/geometry/www_contract_sha256",
     "source_contract"),
    ("agent/questions/*/*/retrieval-*.json",
     "result/calls/*/request/timeout_seconds", "remaining_wall_clock"),
    ("agent/questions/*/*/reply-*.json", "source_call_sha256", "raw_call_hash"),
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args],
                                   text=True).strip()


def first_difference(left, right, path="$"):
    if type(left) is not type(right):
        return f"{path}: types {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return (f"{path}: missing={sorted(left.keys() - right.keys())[:5]}, "
                    f"extra={sorted(right.keys() - left.keys())[:5]}")
        for key in sorted(left):
            found = first_difference(left[key], right[key], f"{path}/{key}")
            if found:
                return found
    elif isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: lengths {len(left)} != {len(right)}"
        for index, (a, b) in enumerate(zip(left, right)):
            found = first_difference(a, b, f"{path}/{index}")
            if found:
                return found
    elif left != right:
        return f"{path}: {repr(left)[:160]} != {repr(right)[:160]}"
    return None


def require_equal(left, right, label):
    problem = first_difference(left, right, label)
    if problem:
        raise ValueError(problem)


def file_snapshot(root):
    return {p.relative_to(root).as_posix(): sha256(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def verify_fixtures(out):
    original = read(out / "fixture-sha256.json")
    require_equal(original, file_snapshot(out / "fixtures"), "fixture mutation")
    return original


def normalize(value, artifact, roots, path=()):
    for pattern, field, reason in EXCLUSIONS:
        parts = field.split("/")
        if (fnmatchcase(artifact, pattern) and len(parts) == len(path)
                and all(a == "*" or a == b for a, b in zip(parts, path))):
            return "<excluded:" + reason + ">"
    if isinstance(value, dict):
        return {k: normalize(v, artifact, roots, path + (k,))
                for k, v in value.items()}
    if isinstance(value, list):
        return [normalize(v, artifact, roots, path + (str(i),))
                for i, v in enumerate(value)]
    if isinstance(value, str):
        # Only absolute path prefixes, never substring replacement in prose.
        for root, replacement in roots:
            if value == str(root) or value.startswith(str(root) + "/"):
                return replacement + value[len(str(root)):]
    return value


def capture_artifacts(root, fixtures):
    payload = {"json": {}, "binary_sha256": {}}
    roots = ((root, "<output>"), (fixtures, "<fixtures>"))
    for patterns, target, convert in (
            (JSON_ARTIFACTS, payload["json"], read),
            (BINARY_ARTIFACTS, payload["binary_sha256"], sha256)):
        for pattern in patterns:
            paths = sorted(root.glob(pattern))
            if not paths:
                raise ValueError(f"Missing required artifact group: {root}/{pattern}")
            for path in paths:
                relative = path.relative_to(root).as_posix()
                target[relative] = normalize(convert(path), relative, roots)
    for required in ("rgb/gallery.html", "agent/gallery.html", "history/gallery.html"):
        if not (root / required).is_file():
            raise ValueError(f"Missing report: {root / required}")
    return payload


def verify(out, commits):
    original = verify_fixtures(out)
    if set(commits) != {"baseline", "candidate"}:
        raise ValueError("Report must identify baseline and candidate commits")
    payloads = []
    for name, commit in commits.items():
        checkout = out / "checkouts" / name
        require_equal(commit, git(checkout, "rev-parse", "HEAD"), "checkout SHA")
        if git(checkout, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError(f"Checkout changed: {checkout}")
        payloads.append(capture_artifacts(out / name, out / "fixtures"))
    require_equal(payloads[0], payloads[1], "artifact drift")
    documents = payloads[0]["json"]
    binaries = payloads[0]["binary_sha256"]
    return {
        "status": "passed", "numerical_tolerance": 0,
        "rgb_rows": len(documents["rgb/metrics.json"]["cases"]),
        "history_sequences": len(documents["history/manifest.json"]["sequences"]),
        "history_png_files": sum(p.startswith("history/") for p in binaries),
        "agent_outcomes": len(documents["agent/results.json"]["results"]),
        "exposure_records": sum(p.endswith("/exposure.json") for p in documents),
        "compared_json_artifacts": len(documents),
        "compared_binary_artifacts": len(binaries),
        "immutable_input_files": len(original), "inputs_unchanged": True,
        "fixture_manifest_sha256": digest(original),
        "normalized_artifacts_sha256": digest(payloads[0]),
    }


def prepare_inputs(out):
    """Use the recovered generator from BASELINE; retain all original files."""
    from test_hypothesis_dataset import HypothesisDatasetTests, HistoryDatasetTests
    from test_2026_11_15_www_curation import ReviewAndMetricsTest

    root = out / "fixtures"
    root.mkdir()
    names = iter(("crops", "history", "www"))

    def retained_directory(*args, **kwargs):
        directory = root / next(names)
        directory.mkdir()
        return SimpleNamespace(name=str(directory), cleanup=lambda: None)

    crops, history, agent = (HypothesisDatasetTests(), HistoryDatasetTests(),
                             ReviewAndMetricsTest())
    # Only redirect the three test fixture roots. Algorithms and hash guards
    # run unchanged; no old prepared/frozen run is copied into either checkout.
    for fixture in (crops, history, agent):
        with patch("tempfile.TemporaryDirectory", retained_directory):
            fixture.setUp()
    questions, answers, index = agent.approve()
    write(out / "inputs.json", {
        "crops": str(crops.parent), "crop_config": str(crops.config),
        "history_dataset": str(history.dataset), "questions": str(questions),
        "answers": str(answers), "index": str(index),
        "inventory": str(agent.inventory), "agent_config": agent.config,
    })
    write(out / "fixture-sha256.json", file_snapshot(root))


def run_backend(inputs, out):
    from facade_change.hypothesis_dataset import (
        prepare_hypothesis_dataset, prepare_history_dataset)
    from facade_change.hypothesis_benchmark import run_hypothesis_benchmark

    demo = importlib.import_module("facade_change.agentic.2026-11-15_www_demo")
    metrics = importlib.import_module("facade_change.agentic.2026-11-15_www_evaluation")
    out.mkdir()
    prepare_hypothesis_dataset(inputs["crops"], inputs["crop_config"], out / "prepared")
    run_hypothesis_benchmark(out / "prepared", out / "rgb", methods=["rgb_diff", "ssim"])
    prepare_history_dataset(inputs["history_dataset"], out / "history")
    config = dict(inputs["agent_config"], cache_root=str(out / "agent-cache"))

    class ScriptedPool(demo.SyntheticPool):
        """Same explicit fake responses as the recovered scratch comparison."""

        def call(self, role, payload):
            if role == "qwen" and payload["geometry"]["frame"]["region_id"] == "error":
                return {"status": "error", "error": "explicit fixture worker failure",
                        "usage": {"model_calls": 1}}
            result = super().call(role, payload)
            if role == "qwen":
                value = json.loads(result["text"])
                region = payload["geometry"]["frame"]["region_id"]
                if value["type"] == "final" and region in {"stable", "occluded"}:
                    value.update(
                        decision="no_change" if region == "stable" else "insufficient_data",
                        boxes=[], limitations=(
                            ["Fixture occlusion"] if region == "occluded" else []))
                    result["text"] = json.dumps(value)
            return result

    metrics.plan(inputs["questions"], inputs["answers"], inputs["inventory"], config,
                 out / "plan", archive_index=inputs["index"], synthetic=True)
    demo.runner.run(inputs["questions"], config, out / "agent",
                    archive_index=inputs["index"], pool_factory=ScriptedPool, synthetic=True)
    metrics.score(inputs["questions"], inputs["answers"], out / "agent",
                  out / "plan/plan.private.json", out / "score")


def worker(stage, out):
    checkout = Path.cwd().resolve()
    sys.path[:0] = [str(checkout), str(checkout / "tests")]
    import facade_change
    import numpy as np
    import random

    if not Path(facade_change.__file__).resolve().is_relative_to(checkout):
        raise ValueError("Imported facade_change from a different checkout")
    np.random.seed(0)
    random.seed(0)
    # Fail closed on accidental network use, even with synthetic=True.
    def offline(*args, **kwargs):
        raise RuntimeError("Network is forbidden in synthetic parity workers")

    with patch.object(socket.socket, "connect", offline), \
            patch.object(socket.socket, "connect_ex", offline), \
            patch.object(socket, "create_connection", offline):
        if stage == "prepare":
            prepare_inputs(out)
        else:
            run_backend(read(out / "inputs.json"), out / stage)


def execute_worker(stage, out, checkout):
    environment = dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH="",
                       CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", PYTHONNOUSERSITE="1")
    log = out / (stage + ".log")
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            [sys.executable, "-s", "-B", str(Path(__file__).resolve()),
             "--worker", stage, str(out)], cwd=checkout, env=environment,
            stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{stage} worker exited {result.returncode}; see {log}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline", default=BASELINE, help="Full existing commit SHA")
    parser.add_argument("--candidate",
                        help="Required full dev_late commit SHA; never implicit HEAD")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    out = args.out.expanduser().resolve()
    if args.verify_only:
        try:
            result = verify(out, read(out / REPORT)["commits"])
            print(json.dumps(result, indent=2))
            return 0
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            print(f"Parity verification failed: {error}", file=sys.stderr)
            return 1
    commits = {"baseline": args.baseline, "candidate": args.candidate}
    if args.baseline == args.candidate:
        parser.error("Baseline and candidate must be different commits")
    for name, value in commits.items():
        if not value or not re.fullmatch(r"[0-9a-f]{40}", value):
            parser.error(f"{name} must be an explicit full commit SHA")
        if git(args.repo, "rev-parse", value + "^{commit}") != value:
            parser.error(f"Unknown commit: {value}")
    if out.exists():
        parser.error(f"Refusing to overwrite {out}; choose a fresh --out")
    out.mkdir(parents=True)
    report = {
        "schema_version": 1, "kind": "synthetic_cross_checkout_parity_v2",
        "status": "running", "commits": commits,
        "driver_sha256": sha256(__file__),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {name: importlib.metadata.version(name) for name in
                                     ("numpy", "Pillow", "opencv-python-headless",
                                      "scipy", "torch")}},
        "parameters": {"fixture_source": args.baseline, "python_hash_seed": 0,
                       "random_seed": 0, "numpy_seed": 0, "numerical_tolerance": 0,
                       "methods": ["rgb_diff", "ssim"], "model_pool": "ScriptedPool (fake)",
                       "network": "blocked in workers", "gpu": "disabled"},
        "comparison_scope": {"json": JSON_ARTIFACTS, "binary_sha256": BINARY_ARTIFACTS},
        "exclusions": EXCLUSIONS,
        "path_normalization": (
            "Only absolute output/fixture prefixes become <output>/<fixtures>; "
            "list order stays exact."),
        "limits": (
            "Synthetic contract check, not real-data or model quality. "
            "HTML existence only; lifecycle journals and unrelated artifacts "
            "are outside comparison_scope."),
    }
    write(out / REPORT, report)
    try:
        for name, commit in commits.items():
            git(args.repo, "worktree", "add", "--detach", str(out / "checkouts" / name), commit)
        execute_worker("prepare", out, out / "checkouts/baseline")
        for name in commits:
            try:
                execute_worker(name, out, out / "checkouts" / name)
            finally:
                verify_fixtures(out)
        report["comparison"] = verify(out, commits)
        inputs = read(out / "inputs.json")
        history = read(out / "baseline/history/manifest.json")["config"]
        contract = read(out / "baseline/agent/contract.json")
        report["parameters"].update(
            crop_config=read(inputs["crop_config"]),
            history={k: v for k, v in history.items() if k not in {"dataset", "input_sha256"}},
            www_conditions=contract["conditions"], www_protocol=contract["protocol"],
            www_budgets=contract["budgets"])
        report["status"] = "passed"
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        report.update(status="failed", error=str(error))
    write(out / REPORT, report)
    print(json.dumps({"status": report["status"], "report": str(out / REPORT),
                      **({"error": report["error"]} if "error" in report
                         else report["comparison"])},
                     indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        worker(sys.argv[2], Path(sys.argv[3]))
    else:
        raise SystemExit(main())
