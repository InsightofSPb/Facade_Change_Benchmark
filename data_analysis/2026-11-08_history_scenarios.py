"""One sequential, resumable runner for the three history hypotheses (seed 13)."""
from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager, redirect_stdout
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import shlex
import subprocess
import sys
import time

from facade_change.history_features import digest
from facade_change.io import read_json, sha256, write_json

PREFIX = "2026-11-08_"
ROOT = Path(__file__).resolve().parents[1]
LEGACY_CONFIG = ROOT / "configs/2026-11-08_history_three_scenarios.json"
DEFAULT_CONFIG = ROOT / "configs/2026-11-09_history_three_scenarios_reduced.json"
Store = importlib.import_module("facade_change.2026-10-14_temporal_experiment").RunStore


def modules():
    return tuple(importlib.import_module(name) for name in (
        "facade_change.research.2026-11-08_history_scenarios_data",
        "facade_change.research.2026-11-08_history_scenarios_models",
        "facade_change.research.2026-11-08_history_scenarios_training",
        "data_analysis.2026-11-08_history_scenarios_report"))


def resolve_config(path=None, *, device=None, memory_run=None, smoke=False):
    supplied = read_json(path) if path else {}
    legacy = supplied.get("protocol") == "history-three-scenarios-v1"
    defaults = read_json(LEGACY_CONFIG if legacy else DEFAULT_CONFIG)
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    config = {**defaults, **supplied}
    if device is not None:
        config["device"] = device
    if memory_run is not None:
        config["memory_run"] = str(Path(memory_run).expanduser().resolve())
    if type(config["seed"]) is not int or config["seed"] != 13:
        raise ValueError("This experiment permits exactly one training seed: 13")
    for key in ("epochs", "patience", "effective_batch", "microbatch", "bootstrap_repeats",
                "cpu_threads", "profile_steps", "gallery_limit"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["microbatch"] != 1:
        raise ValueError("Streaming uses microbatch=1; effective_batch is gradient accumulation")
    for key in ("lr", "quantizer_step", "min_scale", "grad_clip"):
        if not isinstance(config[key], (int, float)) or not 0 < config[key] < float("inf"):
            raise ValueError(f"Invalid {key}")
    if config["fpr_budgets"] != [.05, .10, .15]:
        raise ValueError("Freeze the primary .05 and secondary .10/.15 FPR budgets")
    if set(config["practical"]) != set(defaults["practical"]):
        raise ValueError("Practical thresholds must specify all documented keys, with no unknown keys")
    config["smoke"] = bool(smoke)
    if config.get("matrix_protocol") == "reduced-v2":
        if not 1 <= config["epochs"] <= 60:
            raise ValueError("Primary epochs must be explicitly between 1 and 60")
        if type(config["fusion_epochs"]) is not int or not 1 <= config["fusion_epochs"] <= 10:
            raise ValueError("Fusion epochs must be explicitly between 1 and 10 (default 5)")
        if config["effective_batch"] != 8:
            raise ValueError("The reduced protocol uses effective_batch=8")
        if config["patience"] != 12:
            raise ValueError("The reduced protocol uses primary F1 patience=12")
    if smoke:
        config.update(epochs=2, patience=2, effective_batch=2, amp=False)
        if not legacy:
            config["fusion_epochs"] = 1
    return config


def source_snapshot():
    files = [Path(__file__), ROOT / "data_analysis/__main__.py", DEFAULT_CONFIG, LEGACY_CONFIG]
    files += list((ROOT / "facade_change/research").glob("2026-11-08_history_scenarios_*.py"))
    files += list((ROOT / "data_analysis").glob("2026-11-08_history_scenarios_*.py"))
    # Existing adapters/evaluator are part of the scientific implementation too.
    files += [ROOT / name for name in (
        "data_analysis/2026-11-06_history_fusion.py",
        "data_analysis/2026-11-08_variability_memory.py",
        "data_analysis/2026-11-08_memory_evaluate.py",
        "facade_change/history_features.py",
        "facade_change/annotation_targets.py",
        "facade_change/segmentation/data.py",
        "facade_change/io.py",
        "facade_change/research/2026-10-25_spatial_change_model.py",
        "facade_change/research/2026-11-04_history_benchmark.py",
        "facade_change/research/2026-11-04_history_benchmark_data.py",
        "facade_change/research/2026-11-04_history_benchmark_metrics.py",
        "facade_change/2026-10-14_temporal_experiment.py")]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(files))}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "diff", "HEAD", "--", *hashes], cwd=ROOT)
        # Untracked new source is covered by content hashes, even before a commit.
        status = subprocess.check_output(["git", "status", "--porcelain", "--", *hashes], cwd=ROOT)
    except (OSError, subprocess.SubprocessError):
        commit, dirty, status = None, b"", b"unavailable"
    return dict(git_commit=commit, code_sha256=hashes,
                dirty_sha256=hashlib.sha256(dirty + status).hexdigest(), dirty=bool(status))


def hardware(config, *, require_available=False):
    import torch
    memory = {}
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable"):
                memory[key] = int(value.split()[0]) * 1024
    result = dict(python=platform.python_version(), torch=str(torch.__version__),
                  platform=platform.platform(), cpu_count=os.cpu_count(), ram=memory,
                  device=config["device"], cuda_available=torch.cuda.is_available())
    if require_available and memory.get("MemAvailable", float("inf")) < config["minimum_free_ram_gib"] * 2**30:
        raise RuntimeError("Insufficient free RAM; close other tasks or choose a later launch. Budget unchanged.")
    if str(config["device"]).startswith("cuda"):
        if not torch.cuda.is_available():
            result["cuda_status"] = "NOT_MEASURED: CUDA unavailable in this interpreter"
            if require_available:
                raise RuntimeError("CUDA unavailable in this interpreter; real CUDA preflight must run locally")
            return result
        free, total = torch.cuda.mem_get_info(config["device"])
        result.update(free_vram_bytes=free, total_vram_bytes=total,
                      gpu_name=torch.cuda.get_device_name(config["device"]))
        try:
            probe = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                    "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
            result["gpu_processes"] = probe.stdout.strip().splitlines()
            result["gpu_process_query_status"] = probe.returncode
        except (OSError, subprocess.SubprocessError) as exc:
            result["gpu_process_query_error"] = str(exc)
        if require_available and free < config["minimum_free_vram_gib"] * 2**30:
            raise RuntimeError(f"Only {free / 2**30:.2f} GiB VRAM free; no process is stopped and no model budget changed")
    return result


@contextmanager
def device_lock(config):
    """One runner GPU task per account/device, without touching foreign processes."""
    if not str(config["device"]).startswith("cuda"):
        yield
        return
    import fcntl
    name = hashlib.sha256(str(config["device"]).encode()).hexdigest()[:12]
    lock = Path("/tmp") / f"2026-11-08-facade-history-gpu-{os.getuid()}-{name}.lock"
    with lock.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another history-scenarios runner holds this GPU lock") from exc
        yield


def profile_models(model_module, dataset, config):
    """Measured synthetic-shape forward/backward cost, not a promised wall-time."""
    import torch
    torch.set_num_threads(config["cpu_threads"])
    candidates = dataset.split_samples("train")
    if not candidates:
        raise ValueError("No TRAIN transitions")
    # Profile the longest TRAIN prefix, without consulting VAL/TEST content.
    # The encoder contract supplies shape even when a few feature units still
    # need preparation. Profiling itself never opens annotations or target RGB.
    channels, (height, width) = dataset.channels, dataset.grid
    observations = max(2, max(int(row["frame"]) + 1 for row in candidates))
    rows = []
    sizes = ["smoke"] if config["smoke"] else ["medium", "large"]
    for family in ("convgru", "transformer"):
        for size in sizes:
            for task in ("direct", "actual-next", "h0"):
                torch.manual_seed(13)
                model = model_module.TemporalSpatialModel(channels, family=family, size=size, task=task,
                                                          min_scale=config["min_scale"]).to(config["device"])
                optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                               weight_decay=config["weight_decay"])
                length = observations if task == "direct" else observations - 1
                x = torch.randn(1, length, channels, height, width, device=config["device"])
                support = torch.ones(1, length, 1, height, width, dtype=torch.bool, device=config["device"])
                years = torch.arange(length, device=config["device"], dtype=torch.float32)[None] * 3 + 2000
                target = torch.tensor([2000 + 3 * (observations - 1)], device=config["device"])
                padding = torch.ones(1, length, dtype=torch.bool, device=config["device"])
                if str(config["device"]).startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(config["device"])
                    torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(config["profile_steps"]):
                    model.zero_grad(set_to_none=True)
                    prediction = model(x, support, years, target, padding)
                    loss = (prediction["logits"].square().mean() if task == "direct" else
                            prediction["mean"].square().mean() + prediction["scale"].mean())
                    loss.backward()
                    optimizer.step()
                if str(config["device"]).startswith("cuda"):
                    torch.cuda.synchronize()
                seconds = (time.perf_counter() - started) / config["profile_steps"]
                rows.append(dict(family=family, size=size, task=task,
                    parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    seconds_per_forward_backward=seconds, profile_steps=config["profile_steps"],
                    shape=[1, length, channels, height, width], amp_used=False,
                    prefix_policy="longest TRAIN history; includes AdamW state allocation",
                    peak_vram_bytes=(torch.cuda.max_memory_allocated(config["device"])
                                     if str(config["device"]).startswith("cuda") else None),
                    full_budget_train_seconds_lower_estimate=seconds * len(candidates) * config["epochs"],
                    caveat="Measured shape only; excludes IO, VAL, fusion, exports; lengths vary; no runtime guarantee"))
                del optimizer, model, x, prediction, loss
    return rows


def old_evidence(config):
    """Only inventory real saved evidence; never populate results from prompt numbers."""
    records = {}
    for key in ("prior_evaluation_run", "prior_followups_run"):
        root = Path(config[key])
        if not root.exists():
            records[key] = dict(status="unavailable", path=str(root))
            continue
        artifacts = {str(p): sha256(p) for p in sorted(root.glob("*.csv"))}
        reconstructed = []
        for path in sorted(root.glob("*per_building.csv")):
            grouped = {}
            with path.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    if (row.get("split") != "test" or row.get("slice_type", "overall") != "overall"
                            or row.get("method") not in ("stack", "convgru", "convgru-last")
                            or row.get("seed", "13") != "13"):
                        continue
                    budget = row.get("val_fpr_budget")
                    if budget not in (None, "", "0.05"):
                        continue
                    if not all(row.get(k) not in (None, "") for k in ("tp", "fp", "fn", "tn")):
                        continue
                    tp, fp, fn, tn = [int(row[k]) for k in ("tp", "fp", "fn", "tn")]
                    divide = lambda a, b: a / b if b else None
                    item = dict(building_id=row["building_id"], tp=tp, fp=fp, fn=fn, tn=tn,
                                positive_pixels=tp+fn, negative_pixels=fp+tn,
                                f1=divide(2*tp,2*tp+fp+fn), iou=divide(tp,tp+fp+fn),
                                precision=divide(tp,tp+fp), recall=divide(tp,tp+fn), fpr=divide(fp,fp+tn))
                    grouped.setdefault(row["method"], []).append(item)
            for method, rows in sorted(grouped.items()):
                if len({r["building_id"] for r in rows}) != len(rows):
                    reconstructed.append(dict(method=method, status="ambiguous_duplicate_building_rows",
                                              source=str(path)))
                    continue
                # Explicit undefined values survive; do not copy historical prompt numbers.
                macro = {}
                for metric in ("f1", "iou", "precision", "recall", "fpr"):
                    values = [r[metric] for r in rows if r[metric] is not None and
                              (metric not in ("f1", "iou") or r["positive_pixels"] > 0)]
                    macro[metric] = sum(values)/len(values) if values else None
                reconstructed.append(dict(method=method, status="reconstructed_from_saved_counts",
                                          source=str(path), source_sha256=artifacts[str(path)],
                                          buildings=len(rows), metrics=macro,
                                          threshold_scope="saved original threshold; do not infer missing FPR budget"))
        records[key] = dict(status="available", path=str(root), artifacts=artifacts, reconstructed=reconstructed,
                            run=read_json(root / "run.json") if (root / "run.json").exists() else None)
    return records


def storage_budget(out, dataset, config, variants, profiles):
    """Make the cost of preserving immutable epoch checkpoints reviewable."""
    sizes = {(p["family"], p["size"], p["task"]): p["parameters"] for p in profiles}
    parameter_epochs = sum(sizes.get((s.get("family"), s.get("size"), s["task"]), 0)
                           * config["epochs"] for s in variants)
    checkpoint_bytes = parameter_epochs * 12  # fp32 parameters + two Adam moments
    predictive = sum(s["task"] in ("actual-next", "h0") for s in variants)
    token_components = dataset.channels * dataset.grid[0] * dataset.grid[1]
    latent_bytes = len(dataset.samples) * predictive * token_components * 4 * 3
    free = shutil.disk_usage(out).free
    return dict(free_bytes=free, planned_checkpoint_payload_bytes=checkpoint_bytes,
                predictive_map_payload_bytes_one_selection=latent_bytes,
                free_exceeds_checkpoint_payload=free > checkpoint_bytes,
                caveat="Payload estimate only: excludes ABC checkpoints, maps, two selections, galleries and filesystem overhead; early stopping may reduce epochs",
                retention="Every completed epoch is immutable; no silent cleanup or reduced training budget")


def safe_output(out, config):
    out = Path(out).expanduser().resolve()
    for key in ("memory_run", "benchmark_run", "fusion_run", "prior_evaluation_run", "prior_followups_run"):
        source = Path(config[key]).expanduser().resolve()
        if out == source or source in out.parents or out in source.parents:
            raise ValueError(f"Output overlaps a protected source: {source}")
    return out


def report_only(out, *, event_id=None, methods=None):
    """No dataset loading, encoder, model, optimizer or source-run dependency."""
    import fcntl
    out = Path(out).expanduser().resolve()
    if not (out / "run.json").exists():
        raise ValueError("Missing run.json; report-only requires an existing run")
    with (out / ".run.lock").open("rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Cannot rebuild report while the run is active") from exc
        reporter = importlib.import_module("data_analysis.2026-11-08_history_scenarios_report")
        if event_id is not None:
            return reporter.render_event(out, event_id, methods=methods)
        return reporter.report_only(out)


def local_smoke(out, config, ledger):
    """Run the full tiny synthetic pipeline before real training; preserve its evidence."""
    destination = out / (PREFIX + "local-smoke")
    record = destination / "run.json"
    if record.exists() and read_json(record).get("status") == "completed":
        ledger.event("local_smoke_reused", run=str(destination), run_sha256=sha256(record))
        return
    mode = "resume" if record.exists() else "smoke"
    smoke_args = argparse.Namespace(mode=mode, config=str(DEFAULT_CONFIG), out=str(destination),
                                    device="cpu", memory_run=None, synthetic=True, worker=True)
    print("Local synthetic end-to-end smoke (CPU); details in the dated smoke log", flush=True)
    import torch
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        with (out / (PREFIX + "local-smoke.log")).open("a", encoding="utf-8") as stream, redirect_stdout(stream):
            run(smoke_args)
    finally:
        # CPU smoke uses strict determinism. Restore the enclosing CUDA policy
        # before real training (CUDA interpolation backward may be warn-only).
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
    ledger.event("local_smoke_completed", run=str(destination), run_sha256=sha256(record),
                 interpretation="implementation fixture, not facade-history evidence")


def _continue_command(args, report_path):
    mode = "resume" if args.mode == "resume" else "full"
    command = [sys.executable, "-s", "-u", "-m", "data_analysis", "history-scenarios", mode,
               "--config", str(Path(args.config or DEFAULT_CONFIG).resolve()),
               "--out", str(Path(args.out).resolve()), "--preflight-report", str(report_path)]
    if args.device:
        command += ["--device", args.device]
    if args.memory_run:
        command += ["--memory-run", args.memory_run]
    if args.synthetic:
        command += ["--synthetic"]
    return shlex.join(command)


def _reduced_matrix(training, config):
    return training.matrix_contract(config)


def _frozen_json(path, value):
    """Verify an existing scientific contract without rewriting its bytes."""
    if Path(path).exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen scientific contract differs: {path}")
        return
    write_json(path, value)


def run_reduced(args, config):
    """Schema-2 orchestration: a measured resource gate precedes real training."""
    import numpy as np
    import torch
    preflight = importlib.import_module("facade_change.research.2026-11-08_history_scenarios_preflight")
    progress_module = importlib.import_module("facade_change.research.2026-11-08_history_scenarios_progress")
    data_module, model_module, training, reporter = modules()
    out = safe_output(args.out, config)
    previous = read_json(out / "request.json") if (out / "request.json").exists() else None
    if args.mode == "resume" and previous is None:
        raise ValueError("Resume requires an existing run")
    if previous and previous.get("smoke") and not config["smoke"]:
        config = resolve_config(args.config, device=args.device, memory_run=args.memory_run, smoke=True)
    old_status = read_json(out / "run.json").get("status") if (out / "run.json").exists() else None
    continue_preflight = args.mode == "full" and old_status in ("preflight_completed", "preflight_not_measured")
    store = Store(out, config, resume=args.mode == "resume" or continue_preflight)
    dataset = None
    try:
        torch.set_num_threads(config["cpu_threads"])
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=str(config["device"]).startswith("cuda"))
        snapshot = source_snapshot()
        matrix = _reduced_matrix(training, config)
        store.record["kind"] = "history_three_scenarios"
        write_json(out / "run.json", store.record)
        contract = dict(protocol=config["protocol"], config=config, source=snapshot,
                        torch=str(torch.__version__), numpy=np.__version__, matrix=matrix,
                        fit_split="train", selection_split="val", schema_version=2,
                        training_seeds=[13], synthetic=config["smoke"], test_previously_inspected=True)
        store.contract(contract)
        for suffix, value in (("resolved_matrix.json", matrix), ("resolved_config.json", config)):
            _frozen_json(out / (PREFIX + suffix), value)
        _frozen_json(out / (PREFIX + "training_contract.json"), dict(
            primary_optimizer="AdamW", primary_options=training.options(config),
            fusion_optimizer="Adam", fusion_lr=config["fusion_lr"],
            fusion_weight_decay=config["fusion_weight_decay"], fusion_epochs=config["fusion_epochs"],
            fusion_loss="known-pixel masked BCE; equal physical-building contribution",
            fusion_selection=config["fusion_selection"], detection_selection=config["detection_selection"],
            retention=config["checkpoint_retention"], normalizer="TRAIN fitted; shared frozen transform",
            forecast_inputs="strictly past before current observation; signed deltas; real year intervals",
            model_bits="quantized logistic model bits/component; no actual bitstream size claim",
            thresholds="VAL-only .05/.10/.15; no extra fit per budget",
            seed=13, test_scope=config["test_scope"], target_scope=config["target_scope"]))
        store.event("matrix_frozen_before_training", planned_fit_counts=matrix["planned_fit_counts"],
                    matrix_sha256=sha256(out / (PREFIX + "resolved_matrix.json")))
        resources = hardware(config)
        write_json(out / (PREFIX + "hardware.json"), resources)
        report_path = out / (PREFIX + "resource_report.json")
        # Unavailable owner data/CUDA is a truthful preflight result in Cloud.
        try:
            dataset = (data_module.SyntheticData() if config["smoke"] else data_module.ScenarioData(
                config["memory_run"], benchmark_run=config["benchmark_run"], fusion_run=config["fusion_run"]))
            dataset.__enter__()
            if str(config["device"]).startswith("cuda") and not resources["cuda_available"]:
                raise RuntimeError("CUDA unavailable; owner hardware profile is NOT_MEASURED")
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
            if args.mode not in ("preflight", "full"):
                raise
            report = preflight.unavailable_report(config, out, source_snapshot=snapshot,
                                                  reason=str(exc), hardware=resources)
            store.finish("preflight_not_measured", resource_report=str(report_path), reason=str(exc))
            print(f"Preflight NOT_MEASURED: {exc}\nResource report: {report_path}", flush=True)
            print("After completing a compatible local preflight, continue with:\n" +
                  _continue_command(args, report_path), flush=True)
            return report
        with device_lock(config):
            hardware(config, require_available=True)
            if hasattr(dataset, "prepare_missing"):
                with progress_module.stage("prepare-cache", ledger=store):
                    dataset.prepare_missing(store, config["device"], splits=("train", "val"))
            dataset.verify()
            summary = dataset.summary(inspect_targets=False)
            availability = {key: summary.pop(key) for key in ("effective_cache_index_sha256",
                "missing_feature_observations", "prepared_feature_observations") if key in summary}
            write_json(out / (PREFIX + "cache_availability.json"), availability)
            data_path = out / (PREFIX + "data_contract.json")
            if data_path.exists() and read_json(data_path) != summary:
                raise ValueError("Data/source snapshot changed on resume; use a new run")
            _frozen_json(data_path, summary)
            store.event("data_verified", data_sha256=sha256(data_path))
            supplied_path = getattr(args, "preflight_report", None)
            if not supplied_path and args.mode == "resume" and report_path.exists():
                supplied_path = str(report_path)
            resource_report = read_json(supplied_path) if supplied_path and Path(supplied_path).exists() else None
            compatible, reasons = (preflight.compatible(resource_report, dataset, config, snapshot, resources)
                                   if resource_report else (False, ["No measured preflight report supplied"]))
            needs_profile = args.mode == "preflight" or (not config["smoke"] and not compatible)
            if needs_profile:
                with progress_module.stage("preflight", ledger=store):
                    resource_report = preflight.run_preflight(dataset, config, out,
                        source_snapshot=snapshot, hardware=resources)
                ready, report_reasons = preflight.compatible(resource_report, dataset, config, snapshot, resources)
                store.event("preflight_completed_without_training", previous_report_rejection=reasons,
                            report_compatible=ready, reasons=report_reasons,
                            report_sha256=sha256(report_path))
                store.finish("preflight_completed" if ready else "preflight_not_measured",
                             resource_report=str(report_path), planned_fit_counts=matrix["planned_fit_counts"])
                print(f"Only preflight completed; no primary or fusion training started.\nResource report: {report_path}", flush=True)
                print("Continue explicitly with:\n" + _continue_command(args, report_path), flush=True)
                return resource_report
            if not config["smoke"]:
                observed = hardware(config, require_available=True)
                checks = preflight.check_resources(resource_report, config, out, hardware=observed)
                store.event("preflight_accepted", source_path=str(supplied_path), checks=checks,
                            source_sha256=sha256(supplied_path))
                if Path(supplied_path).resolve() != report_path.resolve():
                    write_json(report_path, resource_report)
                # Keep a local audit trail, while the original profile remains immutable.
                write_json(out / (PREFIX + "preflight_acceptance.json"), checks)
            else:
                store.event("synthetic_resource_gate_exempt", reason="CPU implementation fixture; not real resource evidence")
            if hasattr(dataset, "prepare_missing"):
                with progress_module.stage("prepare-export-cache", ledger=store):
                    dataset.prepare_missing(store, config["device"])
            evidence_path = out / (PREFIX + "previous_evidence.json")
            if not evidence_path.exists():
                write_json(evidence_path, old_evidence(config))
            records = training.train_all(dataset, out, config, ledger=store, resume=args.mode == "resume")
            if source_snapshot() != snapshot:
                raise ValueError("Source changed during training; freeze the checkout before resuming")
            manifest = training.export_all(dataset, records, out, config, ledger=store, resume=args.mode == "resume")
            dataset.verify()
            if source_snapshot() != snapshot:
                raise ValueError("Source changed during evaluation; results cannot be marked complete")
            with progress_module.stage("report", ledger=store):
                result = reporter.build_report(dataset, manifest, out, {**config, "matrix": matrix}, records)
            artifacts = {str(p.relative_to(out)): sha256(p) for p in out.glob(PREFIX + "*") if p.is_file()}
            partial = any(r.get("status", "").upper() in ("ERROR", "INCOMPLETE") for r in records)
            store.finish("completed_partial" if partial else "completed", report_artifacts=artifacts,
                         scientific_claim="synthetic fixture only" if config["smoke"] else "exploratory")
            return result
    except BaseException as exc:
        if args.mode in ("preflight", "full") and not (out / (PREFIX + "resource_report.json")).exists():
            preflight.unavailable_report(config, out, source_snapshot=source_snapshot(),
                reason=f"{type(exc).__name__}: {exc}", hardware=hardware(config))
        store.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            if dataset is not None:
                dataset.__exit__(None, None, None)
        finally:
            store.close()


def run(args):
    import numpy as np
    import torch
    if args.mode == "report-only":
        return report_only(args.out, event_id=getattr(args, "event_id", None),
                           methods=getattr(args, "method", None))
    config = resolve_config(args.config, device=args.device, memory_run=args.memory_run,
                            smoke=args.mode == "smoke" or args.synthetic)
    if config.get("matrix_protocol") == "reduced-v2":
        return run_reduced(args, config)
    out = safe_output(args.out, config)
    if args.mode == "resume":
        if not (out / "request.json").exists():
            raise ValueError("Resume requires an existing run")
        previous = read_json(out / "request.json")
        if previous.get("smoke") and not config["smoke"]:
            config = resolve_config(args.config, device=args.device, memory_run=args.memory_run, smoke=True)
    data_module, model_module, training, reporter = modules()
    torch.set_num_threads(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=str(config["device"]).startswith("cuda"))
    # A synthetic smoke is always explicitly identified; it is never real evidence.
    dataset = (data_module.SyntheticData() if config["smoke"] else data_module.ScenarioData(
        config["memory_run"], benchmark_run=config["benchmark_run"], fusion_run=config["fusion_run"]))
    with dataset:
        variants = training.resolve_matrix(config)
        matrix = dict(protocol=training.PROTOCOL, options=training.options(config), variants=variants,
                      seed=13, test_previously_inspected=True,
                      progression_rule=importlib.import_module(
                          "facade_change.research.2026-11-08_history_scenarios_abc").PROGRESSION_RULE)
        store = Store(out, config, resume=args.mode == "resume")
        store.record["kind"] = "history_three_scenarios"
        write_json(out / "run.json", store.record)
        try:
            snapshot = source_snapshot()
            contract = dict(protocol=config["protocol"], config=config, source=snapshot,
                torch=str(torch.__version__), numpy=np.__version__, matrix=matrix,
                fit_split="train", selection_split="val", test_previously_inspected=True,
                training_seeds=[13], uncertainty="paired building bootstrap; no training-seed variability",
                target_scope=config["target_scope"], synthetic=config["smoke"])
            store.contract(contract)
            write_json(out / (PREFIX + "resolved_matrix.json"), matrix)
            write_json(out / (PREFIX + "resolved_config.json"), config)
            write_json(out / (PREFIX + "training_contract.json"), dict(
                optimizer="AdamW", optimizer_options=training.options(config),
                architectures=model_module.SIZES, pretrained="frozen DINOv2 ViT-S/14 only",
                trainable="temporal module, shared dense decoder and probabilistic head; scratch initialization",
                losses=dict(direct="known-pixel binary cross-entropy",
                    actual_next="quantized logistic NLL on all geometrically valid TRAIN transitions",
                    h0="same NLL restricted to fully known stable TRAIN patches",
                    A="masked multilabel semantic BCE",
                    B="pair BCE; optional stable temporal auxiliary plus reconstruction anti-collapse",
                    C="binary BCE on declared annotation-derived progression proxy"),
                weighting="equal buildings through inverse event-count weights; crop overlap merged for evaluation",
                fusion="three TRAIN-fitted parameters; identical capacity across signals and controls",
                model_bits="bits/component, TRAIN feature normalization, one shared quantization step",
                selection="VAL only; predictive and downstream checkpoint choices recorded separately"))
            store.event("matrix_frozen_before_training", matrix_sha256=sha256(out / (PREFIX + "resolved_matrix.json")))
            with device_lock(config):
                resources = hardware(config, require_available=True)
                hardware_path = out / (PREFIX + "hardware.json")
                if not hardware_path.exists():
                    write_json(hardware_path, resources)
                store.event("resources_observed", resources=resources,
                            initial_hardware_sha256=sha256(hardware_path))
                if hasattr(dataset, "prepare_missing") and args.mode != "preflight":
                    dataset.prepare_missing(store, config["device"])
                dataset.verify()
                summary = dataset.summary()
                data_path = out / (PREFIX + "data_contract.json")
                if data_path.exists() and read_json(data_path) != summary:
                    raise ValueError("Data/source snapshot changed on resume; use a new run")
                write_json(data_path, summary)
                store.event("data_verified", data_sha256=sha256(out / (PREFIX + "data_contract.json")))
                evidence_path = out / (PREFIX + "previous_evidence.json")
                evidence = old_evidence(config)
                if evidence_path.exists() and read_json(evidence_path) != evidence:
                    raise ValueError("Historical evidence changed during resume; original snapshot is retained")
                if not evidence_path.exists():
                    write_json(evidence_path, evidence)
                profile_path = out / (PREFIX + "profile.json")
                if not profile_path.exists():
                    write_json(profile_path, profile_models(model_module, dataset, config))
                    store.event("resource_profile", artifact_sha256=sha256(profile_path))
                disk = storage_budget(out, dataset, config, variants, read_json(profile_path))
                disk_path = out / (PREFIX + "storage_budget.json")
                if not disk_path.exists():
                    write_json(disk_path, disk)
                store.event("storage_observed", storage=disk, initial_budget_sha256=sha256(disk_path))
                print(f"Preserved checkpoint payload estimate: {disk['planned_checkpoint_payload_bytes']/2**30:.1f} GiB; "
                      f"free disk: {disk['free_bytes']/2**30:.1f} GiB; maps require additional space", flush=True)
                if args.mode == "preflight":
                    store.finish("preflight_completed", matrix_variants=len(variants), data=summary)
                    print(f"Preflight completed: {profile_path}\nResolved matrix: {out / (PREFIX + 'resolved_matrix.json')}", flush=True)
                    return summary
                if not config["smoke"]:
                    local_smoke(out, config, store)
                records = training.train_all(dataset, out, config, ledger=store, resume=args.mode == "resume")
                if source_snapshot() != snapshot:
                    raise ValueError("Source changed during training; freeze the checkout before resuming")
                store.event("all_val_choices_frozen_before_test", records=records)
                manifest = training.export_all(dataset, records, out, config, ledger=store, resume=args.mode == "resume")
                dataset.verify()
                if source_snapshot() != snapshot:
                    raise ValueError("Source changed during evaluation; results cannot be marked complete")
                result = reporter.build_report(dataset, manifest, out, {**config, "matrix": matrix}, records)
                artifacts = {str(p.relative_to(out)): sha256(p) for p in out.glob(PREFIX + "*") if p.is_file()}
                state = ("completed_partial" if any(r.get("status", "").upper() in ("ERROR", "INCOMPLETE")
                                                     for r in records) else "completed")
                store.finish(state, report_artifacts=artifacts, scientific_claim="exploratory; see practical statuses")
                return result
        except BaseException as exc:
            store.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                         error=f"{type(exc).__name__}: {exc}")
            try:
                reporter.report_only(out)
            except (OSError, ValueError, KeyError) as report_error:
                store.event("partial_report_error", error=str(report_error))
                print(f"Partial report unavailable: {report_error}", file=sys.stderr)
            raise
        finally:
            store.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", choices=("preflight", "smoke", "full", "resume", "report-only"))
    result.add_argument("--config", default=str(DEFAULT_CONFIG), help="Frozen JSON experiment configuration")
    result.add_argument("--out", required=True, help="New run directory; same path only for resume/report-only")
    result.add_argument("--memory-run", help="Override saved memory cache path")
    result.add_argument("--device", help="cpu or cuda:0; recorded and must match on resume")
    result.add_argument("--synthetic", action="store_true", help="Explicit synthetic fixture for preflight/resume testing")
    result.add_argument("--preflight-report", help="Compatible measured resource JSON required before real full training")
    result.add_argument("--event-id", help="In report-only, render any saved event without loading source models")
    result.add_argument("--method", action="append", help="Optional method ID for --event-id; repeat to compare methods")
    result.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        if (args.event_id or args.method) and args.mode != "report-only":
            raise ValueError("--event-id and --method apply only to report-only")
        if args.method and not args.event_id:
            raise ValueError("--method requires --event-id")
        # Preserve the owner's working encoder environment, as existing memory CLI does.
        if not args.worker and args.mode not in ("smoke", "report-only") and not args.synthetic:
            config = resolve_config(args.config, device=args.device, memory_run=args.memory_run)
            record_path = Path(config["benchmark_run"]) / "run.json"
            if record_path.exists():
                record = read_json(record_path)
                python = record.get("config", {}).get("options", {}).get("dinov2", {}).get("worker_python")
                if python and os.path.abspath(python) != os.path.abspath(sys.executable):
                    command = [python, "-s", "-u", "-m", "data_analysis", "history-scenarios",
                               *(sys.argv[1:] if argv is None else argv), "--worker"]
                    print(f"History experiment worker: {python}", flush=True)
                    return subprocess.run(command, cwd=ROOT, check=True).returncode
        run(args)
        return 0
    except (ValueError, OSError, KeyError, RuntimeError, ImportError, subprocess.SubprocessError) as exc:
        cli.exit(1, f"Ошибка history-scenarios: {exc}\n")


if __name__ == "__main__":
    main()
