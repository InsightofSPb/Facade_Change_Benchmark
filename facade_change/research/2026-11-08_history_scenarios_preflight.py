"""Bounded, TRAIN-only resource profiling for the reduced history experiment.

Timing outputs are engineering observations, never scientific TEST evidence.
All fitted objects are temporary and the caller's RNG state is restored. No
system page caches are flushed and no existing run/cache is modified.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, nullcontext, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import platform
import random
import resource
import shutil
import subprocess
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

PREFIX = "2026-11-08_"
PROTOCOL = "history-resource-preflight-v2"
COUNTS = dict(primary=18, score_fusion=48, feature_fusion=8, short_fusion=56, total=74)


def _module(suffix):
    return importlib.import_module("facade_change.research.2026-11-08_history_scenarios_" + suffix)


def _report():
    return importlib.import_module("data_analysis.2026-11-08_history_scenarios_report")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def hardware_snapshot(device="cpu"):
    memory = {}
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, value = line.split(":", 1)
            if name in ("MemTotal", "MemAvailable"):
                memory[name] = int(value.split()[0]) * 1024
    info = dict(device=str(device), python=platform.python_version(), torch=str(torch.__version__),
                numpy=np.__version__, platform=platform.platform(), cpu_count=os.cpu_count(), ram=memory,
                cuda_available=torch.cuda.is_available(), cuda_runtime=torch.version.cuda)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(device)
        info.update(free_vram_bytes=free, total_vram_bytes=total,
                    gpu_name=torch.cuda.get_device_name(device),
                    gpu_capability=list(torch.cuda.get_device_capability(device)),
                    own_allocated_vram_bytes=torch.cuda.memory_allocated(device),
                    own_reserved_vram_bytes=torch.cuda.memory_reserved(device))
        try:
            result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                     "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
            info.update(gpu_processes=result.stdout.strip().splitlines(), gpu_process_query_status=result.returncode)
        except (OSError, subprocess.SubprocessError) as exc:
            info["gpu_process_query_error"] = str(exc)
    return info


def _hardware_identity(hardware):
    """Free resources/process lists vary; hardware and software identity do not."""
    keys = ("device", "python", "torch", "numpy", "platform", "cpu_count", "cuda_available",
            "cuda_runtime", "total_vram_bytes", "gpu_name", "gpu_capability")
    result = {key: hardware.get(key) for key in keys}
    result["total_ram_bytes"] = hardware.get("ram", {}).get("MemTotal")
    return result


def fingerprint(dataset, config, source_snapshot, hardware=None):
    hardware = hardware or hardware_snapshot(config.get("device", "cpu"))
    # Metadata only: inspect_targets=False must not open TEST labels/predictions.
    data = dataset.summary(inspect_targets=False)
    needed = {observation["observation_id"] for track in dataset.tracks.values()
              if track["split"] in ("train", "val") for observation in track["observations"]}
    for key in ("effective_cache_index_sha256", "prepared_feature_observations"):
        data.pop(key, None)
    data["missing_feature_observations"] = sorted(set(data.get("missing_feature_observations", [])) & needed)
    if hasattr(dataset, "cache_index"):
        restricted = {key: ({k:v for k,v in value.items() if k not in ("artifact_root", "artifact_path") and not (k == "unit_sha256" and value.get("origin") == "run_overlay")}
                            if value is not None else None)
                      for key,value in dataset.cache_index.items() if key.split(":",1)[-1] in needed}
        data["measured_train_val_cache_sha256"] = _digest(restricted)
    configuration = {k: v for k, v in config.items() if k not in ("out", "preflight_report", "resume", "mode")}
    identity = dict(protocol=PROTOCOL, code=source_snapshot, config=configuration,
                    data=data, hardware=_hardware_identity(hardware))
    return dict(sha256=_digest(identity), components={key: _digest(value) for key, value in identity.items()},
                identity=identity)


def compatible(report, dataset, config, source_snapshot, hardware=None):
    current = fingerprint(dataset, config, source_snapshot, hardware)
    reasons = []
    if report.get("protocol") != PROTOCOL:
        reasons.append("resource report protocol differs")
    if report.get("fingerprint", {}).get("sha256") != current["sha256"]:
        old = report.get("fingerprint", {}).get("components", {})
        reasons += [f"fingerprint differs: {key}" for key, value in current["components"].items() if old.get(key) != value]
    max_age = config.get("preflight_max_age_hours")
    if max_age is not None:
        age = time.time() - report.get("created_unix", 0)
        if age < -60 or age > float(max_age) * 3600:
            reasons.append("resource report is expired or has an invalid timestamp")
    if report.get("status") != "MEASURED" or not report.get("ready_for_full", False):
        reasons.append("resource report is incomplete or was measured only on synthetic data")
    return not reasons, reasons


def deterministic_examples(dataset, predictive=False):
    rows = sorted(dataset.split_samples("train", predictive=predictive),
                  key=lambda row: (int(row["frame"]), row["event_id"]))
    if not rows:
        return []
    # Preserve all three named quantiles even if a tiny fixture repeats a row.
    return [dict(role=role, row=rows[min(len(rows)-1, int((len(rows)-1) * quantile))])
            for role, quantile in (("typical", .5), ("long", .9), ("maximum", 1.))]


def _rss_bytes():
    # Linux ru_maxrss is KiB and is a process high-water mark, not an incremental allocation.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if platform.system() == "Darwin" else 1024)


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _timed(call, device="cpu"):
    _sync(device)
    start = time.perf_counter()
    value = call()
    _sync(device)
    return value, time.perf_counter() - start


@contextmanager
def isolated_rng(threads=None):
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
             torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    old_threads = torch.get_num_threads()
    try:
        if threads is not None:
            torch.set_num_threads(threads)
        random.seed(13)
        np.random.seed(13)
        torch.manual_seed(13)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(13)
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        if state[3] is not None:
            torch.cuda.set_rng_state_all(state[3])
        if torch.get_num_threads() != old_threads:
            torch.set_num_threads(old_threads)


class _CachedExample:
    """Bounded one-example host cache, separating repeated input IO from compute."""
    METHODS = {"inputs", "pair_inputs", "feature_target", "targets", "semantic", "annotation_history",
               "evaluation_support", "scores", "stack_components"}

    def __init__(self, dataset):
        self.dataset, self.cache = dataset, {}

    def __getattr__(self, key):
        value = getattr(self.dataset, key)
        if key not in self.METHODS:
            return value
        def read(row, *args, **kwargs):
            if row["split"] == "test":
                raise AssertionError("Preflight cannot read TEST scientific data")
            cache_key = (key, row["event_id"], repr(args), repr(sorted(kwargs.items())))
            if cache_key not in self.cache:
                self.cache[cache_key] = value(row, *args, **kwargs)
            return self.cache[cache_key]
        return read


def _input_bundle(dataset, row, spec, normalizer, training, device):
    # Cache exactly the inputs/loss targets this task needs; no other examples.
    training._inputs(dataset, row, spec, normalizer, device)
    dataset.targets(row)
    if spec["task"] in ("actual-next", "h0"):
        dataset.feature_target(row)
    if spec.get("scenario") in ("A", "B", "C"):
        dataset.annotation_history(row)
    if spec.get("scenario") == "A":
        dataset.semantic(row)


def _scaler(enabled):
    if hasattr(getattr(torch, "amp", None), "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _profile_training(model, dataset, row, spec, normalizer, config, training):
    opts = training.options(config)
    device, accumulation = opts["device"], opts["effective_batch"]
    amp = str(device).startswith("cuda") and opts["amp"]
    steps, warmup = int(config.get("profile_steps", 3)), int(config.get("profile_warmup", config.get("profile_warmup_steps", 1)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts["lr"], weight_decay=opts["weight_decay"])
    scaler = _scaler(amp)
    timings, losses = [], []
    counts = Counter(r["building_id"] for r in dataset.split_samples("train", spec["task"] in ("actual-next", "h0")))
    weight = sum(counts.values()) / (len(counts) * counts[row["building_id"]])
    def cycle():
        optimizer.zero_grad(set_to_none=True)
        used, total = 0, 0.
        for _ in range(accumulation):
            context = torch.autocast(device_type="cuda", dtype=torch.float16) if amp else nullcontext()
            with context:
                loss = training._loss(model, dataset, row, spec, normalizer, opts)
                if loss is None:
                    continue
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite preflight loss; no automatic budget reduction")
                weighted = loss * weight / accumulation
            scaler.scale(weighted).backward()
            total += float(loss.detach())
            used += 1
        if used:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        return (total / used if used else None), used
    model.train()
    for _ in range(warmup):
        cycle()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(steps):
        value, duration = _timed(cycle, device)
        timings.append(duration)
        losses.append(value[0])
    result = dict(status="MEASURED" if any(v is not None for v in losses) else "NO_ELIGIBLE_PIXELS",
                  seconds_per_optimizer_cycle=timings, seconds_per_microbatch=float(np.median(timings)) / accumulation,
                  warmup_optimizer_cycles=warmup, measured_optimizer_cycles=steps,
                  microbatch=1, effective_batch=accumulation, amp_used=amp,
                  loss=losses, loss_weighting="equal buildings; repeated selected event in bounded profile only",
                  peak_host_rss_bytes=_rss_bytes(), host_ram_semantics="process high-water RSS, not incremental model RAM",
                  peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else None,
                  peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(device) if str(device).startswith("cuda") else None)
    return result, optimizer, scaler


def _write_probe(path, value, *, torch_payload=False):
    path = Path(path)
    start = time.perf_counter()
    with path.open("wb") as stream:
        if torch_payload:
            torch.save(value, stream)
        else:
            np.save(stream, np.asarray(value), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(seconds=time.perf_counter()-start, bytes=path.stat().st_size)


def validation_examples(dataset):
    """Select at most two complete spatial transitions, using VAL metadata only.

    Positive-bearing scenes come first, with distinct buildings preferred. All
    their overlapping crops are retained; no crop cap silently changes geometry.
    """
    groups={}
    for row in dataset.split_samples("val"):
        key=tuple(row[name] for name in ("building_id","view_id","previous_year","current_year"))
        groups.setdefault(key,[]).append(row)
    ordered=sorted(groups,key=lambda key:(not any(row.get("positive_pixels",0)>0 for row in groups[key]),key))
    if not ordered:
        return []
    selected=[ordered[0]]
    different=[key for key in ordered[1:] if key[0]!=selected[0][0]]
    if different or len(ordered)>1:
        selected.append(different[0] if different else ordered[1])
    return sorted([row for key in selected for row in groups[key]],key=lambda row:row["event_id"])


def _profile_validation(model, dataset, spec, normalizer, config, training, directory):
    rows = validation_examples(dataset)
    if not rows:
        return dict(status="NOT_MEASURED", reason="No VAL events"), None
    saved, seconds, writing, first = {}, [], [], None
    for row in rows:
        prediction, duration = _timed(lambda: training.prediction(model, dataset, row, spec, normalizer, config), config["device"])
        if first is None:
            first = prediction
        retained_extras={key:value for key,value in prediction.get("extras",{}).items() if key=="progression_target"}
        saved[row["event_id"]] = dict(score=prediction["score"],support=prediction["support"],extras=retained_extras)
        seconds.append(duration)
        event_dir = directory / row["event_id"]
        event_dir.mkdir()
        for key in ("score", "support"):
            writing.append(_write_probe(event_dir / (key + ".npy"), prediction[key]))
    reporter = _report()
    class ValidationSubset:
        def __init__(self):
            self.samples = rows
            self.events = {row["event_id"]: row for row in rows}
        def __getattr__(self, key):
            return getattr(dataset, key)
    canonical = _module("data").EvaluationData(ValidationSubset())
    if spec.get("scenario")=="C":
        canonical=reporter._ProgressionData(canonical,lambda row:saved[row["event_id"]])
    (histograms, coverage), merge_seconds = _timed(lambda: reporter.collect_results(canonical, rows, lambda row: saved[row["event_id"]]))
    stage = dict(status="MEASURED", event_ids=[row["event_id"] for row in rows], split="val",
                 inference_seconds=seconds, seconds_per_event=float(np.mean(seconds)),
                 overlap_merge_histograms_seconds=merge_seconds,
                 scientific_array_write_seconds=sum(item["seconds"] for item in writing),
                 scientific_array_write_bytes=sum(item["bytes"] for item in writing),
                 sampling="at most two positive-prioritized VAL spatial transitions; distinct buildings preferred; all overlapping crops retained",
                 profile_scene_count=len({tuple(row[k] for k in ("building_id","view_id","previous_year","current_year")) for row in rows}),
                 profile_building_count=len({row["building_id"] for row in rows}))
    try:
        point, stage["threshold_selection_seconds"] = _timed(lambda: reporter.select_threshold(histograms, .05))
        summary, stage["metrics_seconds"] = _timed(lambda: reporter.summarize_histograms(canonical, histograms, point["threshold"], method="profile", split="val"))
        # Same saved data path, without model calls, for report-only component timing.
        def replay(row):
            event_dir = directory / row["event_id"]
            return dict(score=np.load(event_dir / "score.npy", mmap_mode="r", allow_pickle=False),
                        support=np.load(event_dir / "support.npy", mmap_mode="r", allow_pickle=False))
        _, stage["report_only_replay_seconds"] = _timed(lambda: reporter.collect_results(canonical, rows, replay))
        paired = [{**row, "method": method} for row in summary[1] for method in ("profile", "same-profile-reference")]
        _, stage["bootstrap_seconds"] = _timed(lambda: reporter.paired_bootstrap(paired, "profile", "same-profile-reference", repetitions=config.get("bootstrap_repeats", 2000)))
        stage["bootstrap_note"] = "timing identical saved VAL predictions; not an effect estimate"
    except reporter.InsufficientData as exc:
        stage["threshold_metrics_status"] = "NOT_MEASURED"
        stage["threshold_metrics_reason"] = str(exc)
    def render():
        panel = reporter._display_map(first["score"], 0., 1., first["support"])
        panel.save(directory / "profile-panel.png")
    _, stage["gallery_panel_seconds"] = _timed(render)
    stage["report_scope"] = "saved-map merge/histograms/threshold/metrics/bootstrap and scientific PNG panel on bounded VAL sample"
    if spec["task"] == "direct" and spec["family"] == "convgru" and spec["control"] == "history" and stage.get("threshold_metrics_status") != "NOT_MEASURED":
        # Exercise the actual schema-v2 report-only entrypoint, no dataset/model
        # supplied on replay. All temporary maps retain their VAL split.
        report_root = directory / "saved-report-probe"
        store = _module("storage").SharedMapStore(report_root)
        manifest = []
        for row in rows:
            prediction = saved[row["event_id"]]
            arrays = canonical.target(row["event_id"])
            manifest.append(store.save(row,spec["id"],spec,
                dict(score=prediction["score"],support=prediction["support"],extras=arrays)))
        report_config = {**config,"categories":dataset.categories,"groups":dataset.groups,
                         "matrix":[],"gallery_limit":1}
        records = [dict(spec=spec,id=spec["id"],status="completed",checkpoint=None)]
        def rebuild():
            with redirect_stdout(io.StringIO()):
                return reporter.build_report(None,manifest,report_root,report_config,records)
        _,stage["report_initial_build_seconds"] = _timed(rebuild)
        def replay_report():
            with redirect_stdout(io.StringIO()):
                return reporter.report_only(report_root)
        _,stage["report_only_full_entrypoint_seconds"] = _timed(replay_report)
        _,stage["gallery_full_entrypoint_seconds"] = _timed(lambda: reporter.build_gallery(
            report_root,manifest,{spec["id"]:point},limit=1,event_ids=[rows[0]["event_id"]],
            directory=report_root / "profile-val-gallery"))
        stage["report_probe_artifact_bytes"] = sum(path.stat().st_size for path in report_root.rglob("*") if path.is_file())
        stage["report_only_scope"] = "actual report-only and any-event gallery entrypoints on schema-v2 maps from two complete VAL spatial transitions; fixed overhead/sample cost, not full-run timing"
        _module("storage").clear_reader_cache(report_root)
    return stage, first


def _cache_bytes(dataset):
    roots = []
    for name, attribute in (("memory", "saved"), ("benchmark", "source"), ("fusion", "frozen")):
        value = getattr(dataset, attribute, None)
        root = getattr(value, "root", getattr(value, "out", None))
        if root is not None:
            roots.append((name, Path(root).resolve()))
    seen, records = set(), []
    for name, root in roots:
        total, files = 0, 0
        for directory, _, names in os.walk(root):
            for name_ in names:
                path = Path(directory) / name_
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                identity = (stat.st_dev, stat.st_ino)
                if identity in seen:
                    continue
                seen.add(identity)
                total += stat.st_size
                files += 1
        records.append(dict(kind=name, path=str(root), bytes=total, unique_files=files))
    return dict(status="MEASURED" if roots else "NOT_APPLICABLE_SYNTHETIC", roots=records,
                bytes=sum(row["bytes"] for row in records), policy="existing read-only inputs, separate from output budget; inode-deduplicated")


def storage_estimate(dataset, config, model_profiles=None, fusion_profiles=None):
    """Uncompressed schema-v2 upper estimate; no TEST numeric data is opened."""
    rows = dataset.samples
    shapes = Counter((int(row["bbox_xywh"][3]), int(row["bbox_xywh"][2])) for row in rows)
    pixels = sum(height * width * count for (height, width), count in shapes.items())
    events = len(rows)
    observations = {observation["observation_id"]: track["bbox_xywh"] for track in dataset.tracks.values()
                    for observation in track["observations"]}
    observation_pixels = sum(int(box[2]) * int(box[3]) for box in observations.values())
    channels, grid_pixels = int(dataset.channels), int(np.prod(dataset.grid))
    classes = len(dataset.categories)
    # Target/class targets uint8, before/after uint16, support bool. Packed dtype
    # is checked on one TRAIN example; no need to decode TEST to infer schema.
    train = deterministic_examples(dataset)
    packed_itemsize = 2
    if train:
        packed_itemsize = int(np.asarray(dataset.targets(train[0]["row"])["before"]).dtype.itemsize)
    components = {
        "shared_targets_and_geometry": pixels * (1 + classes + 2 * packed_itemsize + 1),
        "shared_raw_stack_components":pixels * len(dataset.stack_contract.get("method_names",[])) * 4,
        "C_proxy_target_and_support_upper":pixels * 4,
        "reference_support_upper":pixels * 2,
        "shared_rgb_and_frame_support": observation_pixels * 4,
        "shared_normalized_dino_targets": len(observations) * channels * grid_pixels * 4,
        "trained_variant_float_scores": pixels * 74 * 4,
        "predictive_mean_and_scale": events * 8 * channels * grid_pixels * 4 * 2,
        "predictive_diagnostic_rasters": pixels * 8 * (6 * 4 + 1),
        "copy_last_scale_and_diagnostics_upper": events * 2 * channels * grid_pixels * 4 + pixels * 2 * 6 * 4,
        "fixed_reference_scores_upper": pixels * 5 * 4,
        "A_state_unique_prefix_upper": observation_pixels * 3 * classes * 4,
        "shared_semantic_targets_known": observation_pixels * (classes + 1),
    }
    parameters = {row["model_id"]: row["parameters"] for row in (model_profiles or []) if "parameters" in row}
    # At most top3 plus IoU/recall/NLL-or-semantic: six unique winners,
    # plus latest weights/Adam moments. Exact dedup makes actual storage smaller.
    checkpoint_upper = sum(parameters.values()) * 4 * (6 + 3)
    fusion_parameters = sum(row["parameters"] * row["maps_to_fit_count"] for row in (fusion_profiles or []))
    components["checkpoint_winners_and_latest_upper"] = checkpoint_upper
    components["fusion_checkpoint_selected_and_resume_upper"] = fusion_parameters * 4 * 4
    permanent = sum(components.values())
    # Atomic next-latest and output staging coexist temporarily. Histograms are
    # small by comparison but estimated explicitly from metadata, not GT.
    buildings = len({row["building_id"] for row in rows})
    histogram_spool = buildings * (1 + classes + len(dataset.groups)) * 2 * 2049 * 8
    temporary = (max(parameters.values(), default=0) * 4 * 3 + histogram_spool +
                 max((h*w for h, w in shapes), default=0) * (channels + 100) * 4)
    return dict(status="ESTIMATED", schema_version=2, dtype_policy="float32/original exact dtype; no lossy quantization",
                events=events, unique_observations=len(observations), event_shapes=[dict(shape=list(shape), events=count) for shape,count in sorted(shapes.items())],
                components_uncompressed_bytes=components, permanent_uncompressed_upper_bytes=permanent,
                temporary_upper_bytes=temporary, recommended_free_output_bytes=int((permanent+temporary)*float(config.get("resource_safety_factor",1.25))),
                policy="upper budget before content dedup/compression; exact schema fields, counts and sampled packed dtype",
                exclusions=["filesystem metadata and small JSON/CSV/HTML", "optional missing DINO cache preparation", "full-gallery PNG size depends on content"],
                uncertainty="1.25 workspace headroom is an explicit estimate, not a bound or guarantee; no storage cap",
                previous_orientation="58 GiB payload / 70–80 GiB workspace for 2089 events was preliminary, not a limit")


def check_resources(report, config, out, hardware=None, *, raise_on_failure=True):
    hardware = hardware or hardware_snapshot(config.get("device", "cpu"))
    directory = Path(out).resolve()
    while not directory.exists():
        directory = directory.parent
    disk = shutil.disk_usage(directory)
    disk_required = report.get("storage", {}).get("recommended_free_output_bytes", 0)
    profiles = report.get("training_profiles", []) + report.get("fusion_profiles", [])
    peak = max((max(row.get("peak_allocated_vram_bytes") or 0, row.get("peak_reserved_vram_bytes") or 0)
                for row in profiles), default=0)
    safety = float(config.get("resource_safety_factor",1.25))
    required_vram = max(float(config.get("minimum_free_vram_gib", 0))*2**30, peak*safety)
    peak_ram = max([report.get("peak_host_rss_bytes",0)] + [row.get("peak_host_rss_bytes",0) for row in profiles])
    required_ram = max(float(config.get("minimum_free_ram_gib", 0))*2**30,peak_ram*safety)
    own_reserved=hardware.get("own_reserved_vram_bytes",0)
    own_allocated=hardware.get("own_allocated_vram_bytes",0)
    if str(config.get("device","cpu")).startswith("cuda") and torch.cuda.is_available() and "own_reserved_vram_bytes" not in hardware:
        own_reserved=torch.cuda.memory_reserved(config["device"])
        own_allocated=torch.cuda.memory_allocated(config["device"])
    own_reusable=max(0,own_reserved-own_allocated)
    available_vram=hardware.get("free_vram_bytes",0)+own_reusable
    failures = []
    if disk.free < disk_required:
        failures.append(f"free disk {disk.free} bytes < estimated required {disk_required}")
    available_ram = hardware.get("ram", {}).get("MemAvailable")
    if available_ram is not None and available_ram < required_ram:
        failures.append(f"free RAM {available_ram} bytes < configured required {int(required_ram)}")
    if str(config.get("device", "cpu")).startswith("cuda"):
        if not hardware.get("cuda_available"):
            failures.append("CUDA unavailable")
        elif available_vram < required_vram:
            failures.append(f"free VRAM < measured/configured required {int(required_vram)} bytes")
    if failures and raise_on_failure:
        raise RuntimeError("Resource preflight blocked: " + "; ".join(failures) + ". No data or budget was changed.")
    return dict(passed=not failures, failures=failures, free_disk_bytes=disk.free,
                required_disk_bytes=disk_required, free_ram_bytes=available_ram,
                required_ram_bytes=int(required_ram), required_vram_bytes=int(required_vram),
                free_vram_bytes=hardware.get("free_vram_bytes"), available_vram_to_this_process_bytes=available_vram,
                own_reusable_vram_bytes=own_reusable, gpu_processes=hardware.get("gpu_processes"),
                action="No deletion, process termination or automatic model/data/budget reduction")


def _text_report(report):
    lines = ["History scenarios resource preflight", "", f"Status: {report['status']}",
             "Planned: 18 primary/control fits + 48 score-fusion + 8 feature-fusion (56 short, 74 total)",
             f"Ready for real full: {report['ready_for_full']}",
             f"Fingerprint: {report['fingerprint']['sha256']}",
             "All timing inputs are TRAIN/VAL; no scientific TEST predictions or metrics.",
             "Synthetic CPU timings cannot establish real-data/CUDA cost or scientific quality.", ""]
    for key in ("eligible_counts", "resource_checks", "storage", "time_estimates", "not_measured", "limitations"):
        lines.extend([key + ":", json.dumps(report.get(key), ensure_ascii=False, indent=2), ""])
    return "\n".join(lines)


def _run_preflight(dataset, config, out, *, source_snapshot, hardware=None, progress=None):
    """Measure a short real/synthetic profile and write an immutable report result.

    ``progress(stage, **fields)`` is optional. Calling this never starts full
    training, exports scientific TEST, or writes a fitted normalizer to a run.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    hardware = hardware or hardware_snapshot(config.get("device", "cpu"))
    def emit(stage, **fields):
        if progress is not None:
            progress(stage, **fields)
        else:
            print("preflight " + stage + " " + " ".join(f"{key}={value}" for key,value in fields.items()), flush=True)
    if dataset is None:
        return unavailable_report(config,out,source_snapshot=source_snapshot,reason="Real dataset unavailable",hardware=hardware)
    summary = dataset.summary(inspect_targets=False)
    result = dict(protocol=PROTOCOL, status="NOT_MEASURED", ready_for_full=False,
                  created_unix=time.time(), created_utc=datetime.now(timezone.utc).isoformat(),
                  synthetic=bool(getattr(dataset, "synthetic", False)), hardware=hardware,
                  fingerprint=fingerprint(dataset,config,source_snapshot,hardware), counts=COUNTS,
                  eligible_counts=summary["counts"], data_contract=summary,
                  training_profiles=[], fusion_profiles=[], validation_profiles=[],
                  not_measured=[], limitations=["Early stopping is unknown; upper-epoch estimates are not promised completion times.",
                  "Observed first IO may already benefit from the OS cache; no cache was flushed.",
                  "Random initialized/profile-updated outputs do not predict trained artifact compressibility.",
                  "Host RSS is process high-water mark; report-only is measured on a bounded VAL saved-map sample."])
    cuda_missing = str(config.get("device", "cpu")).startswith("cuda") and not hardware.get("cuda_available")
    needed_observations = {observation["observation_id"] for track in dataset.tracks.values()
                           if track["split"] in ("train", "val") for observation in track["observations"]}
    missing_features = sorted(set(summary.get("missing_feature_observations", [])) & needed_observations)
    if hasattr(dataset, "missing_features"):
        missing_features = dataset.missing_features(splits=("train", "val"))
    initial_checks = check_resources({},config,out,hardware,raise_on_failure=False)
    if cuda_missing or missing_features or not initial_checks["passed"]:
        result["not_measured"].append("CUDA unavailable" if cuda_missing else "Required frozen feature cache entries are missing" if missing_features else "; ".join(initial_checks["failures"]))
        result["storage"] = storage_estimate(dataset, config)
        result["input_cache_inventory"] = _cache_bytes(dataset)
        result["resource_checks"] = check_resources(result, config, out, hardware, raise_on_failure=False)
        _atomic_json(out / (PREFIX + "resource_report.json"), result)
        (out / (PREFIX + "resource_report.txt")).write_text(_text_report(result), encoding="utf-8")
        return result
    training, data_module = _module("training"), _module("data")
    matrix = training.resolve_matrix(config)
    if len(matrix) != 18:
        raise ValueError("Resource preflight requires the reduced 18-primary matrix")
    result["storage"] = storage_estimate(dataset,config)
    result["resource_checks"] = check_resources(result,config,out,hardware,raise_on_failure=False)
    _atomic_json(out/(PREFIX+"resource_report.json"),result)
    if not result["resource_checks"]["passed"]:
        result["not_measured"].extend(result["resource_checks"]["failures"])
        _atomic_json(out/(PREFIX+"resource_report.json"),result)
        (out/(PREFIX+"resource_report.txt")).write_text(_text_report(result),encoding="utf-8")
        return result
    with isolated_rng(config.get("cpu_threads", 4)), tempfile.TemporaryDirectory(prefix=PREFIX+"profile-temporary-", dir=out) as scratch:
        scratch = Path(scratch)
        emit("normalization", split="train")
        normalizer, normalizer_seconds = _timed(lambda: data_module.fit_normalizer(dataset))
        result["normalizer"] = dict(status="MEASURED", fit_split="train", seconds=normalizer_seconds,
                                    observation_count=len(normalizer.observation_ids), persisted_to_main_run=False)
        emit("progression-eligibility",split="train,val")
        progression_rows=[]
        coverage_started=time.perf_counter()
        for row in dataset.samples:
            if row["split"] not in ("train","val"):
                continue
            coverage=training._progression(dataset,row,"cpu")["coverage"]
            progression_rows.append(dict(event_id=row["event_id"],building_id=row["building_id"],split=row["split"],coverage=coverage))
        result["progression_eligibility"]={split:_module("abc").progression_review_manifest(
            [row for row in progression_rows if row["split"]==split]) for split in ("train","val")}
        result["progression_eligibility_seconds"]=time.perf_counter()-coverage_started
        progression_eligible=all(row["status"]=="ELIGIBLE" for row in result["progression_eligibility"].values())
        result["eligible_fit_plan"]={**COUNTS,"primary_eligible_by_progression_review":18 if progression_eligible else 16,
            "actual_primary_fits":0,"actual_fusion_fits":0,"profile_optimizer_updates_are_temporary":True}
        for spec in matrix:
            emit("primary", model=spec["id"])
            if spec.get("scenario")=="C" and not progression_eligible:
                result["training_profiles"].append(dict(model_id=spec["id"],task=spec["task"],family=spec["family"],
                    status="INSUFFICIENT_DATA",reason="TRAIN/VAL progression coverage gate failed",history_role="not applicable"))
                continue
            examples = deterministic_examples(dataset, spec["task"] in ("actual-next", "h0"))
            if not examples:
                result["not_measured"].append(spec["id"]+": no TRAIN examples")
                continue
            model = training.build_model(spec,dataset,config)
            parameters = sum(p.numel() for p in model.parameters())
            seen = {}
            for example in examples:
                role, row = example["role"], example["row"]
                if row["event_id"] in seen:
                    result["training_profiles"].append({**seen[row["event_id"]], "history_role":role,
                        "measurement_reused_from":seen[row["event_id"]]["history_role"]})
                    continue
                emit("optimizer", model=spec["id"], history=role, frame=row["frame"])
                cached = _CachedExample(dataset)
                _, first_io = _timed(lambda: _input_bundle(cached,row,spec,normalizer,training,config["device"]), config["device"])
                _, repeated_io = _timed(lambda: _input_bundle(dataset,row,spec,normalizer,training,config["device"]), config["device"])
                profile, optimizer, scaler = _profile_training(model,cached,row,spec,normalizer,config,training)
                profile.update(model_id=spec["id"], task=spec["task"], family=spec["family"], parameters=parameters,
                    event_id=row["event_id"], split="train", history_role=role, observation_count=int(row["frame"])+1,
                    input_shape=list(cached.inputs(row,predictive=spec["task"] in ("actual-next","h0"))["features"].shape),
                    first_observed_input_io_seconds=first_io, repeated_input_io_seconds=repeated_io,
                    grouping="exact model ID and actual TRAIN event; repeated quantile row reused explicitly")
                seen[row["event_id"]] = profile
                result["training_profiles"].append(profile)
            directory = scratch / spec["id"]
            directory.mkdir()
            validation, prediction = _profile_validation(model,dataset,spec,normalizer,config,training,directory)
            result["validation_profiles"].append(dict(model_id=spec["id"], **validation))
            # Resume payload includes actual allocated optimizer moments/scaler/RNG.
            payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                           rng=training._rng_state(), progress=dict(profile_only=True))
            profile["checkpoint_write"] = _write_probe(directory / "resume-probe.pt",payload,torch_payload=True)
            if prediction is not None:
                writes = []
                for key,value in prediction.get("extras",{}).items():
                    if isinstance(value,np.ndarray):
                        writes.append(_write_probe(directory / ("diagnostic-"+key+".npy"),value))
                profile["diagnostic_array_write"] = dict(bytes=sum(item["bytes"] for item in writes),
                    seconds=sum(item["seconds"] for item in writes), format="uncompressed exact NPY")
            profile["temporary_materialized_bytes"] = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
            del model, optimizer, scaler, cached, payload, prediction
            shutil.rmtree(directory)  # Only this newly created, private profile scratch.
            _atomic_json(out/(PREFIX+"resource_report.json"),result)
        emit("fusion")
        result["fusion_profiles"] = _profile_fusion(dataset,normalizer,config,training,emit)
    result["status"] = "SYNTHETIC_MEASURED" if result["synthetic"] else "MEASURED"
    result["input_cache_inventory"] = _cache_bytes(dataset)
    result["storage"] = storage_estimate(dataset,config,result["training_profiles"],result["fusion_profiles"])
    canonical_counts={}
    for split in ("train","val","test"):
        canonical_blocks=set()
        for row in dataset.split_samples(split):
            x,y,width,height=row["bbox_xywh"]
            scene=tuple(row[key] for key in ("building_id","view_id","previous_year","current_year"))
            canonical_blocks.update((scene,bx,by) for bx in range(x//256,(x+width-1)//256+1) for by in range(y//256,(y+height-1)//256+1))
        canonical_counts[split]=len(canonical_blocks)
    result["canonical_blocks_by_split"]=canonical_counts
    result["canonical_train_blocks"]=canonical_counts["train"]
    result["time_estimates"] = _time_estimates(result,config)
    result["resource_checks"] = check_resources(result,config,out,hardware_snapshot(config["device"]), raise_on_failure=False)
    measured_models = {row["model_id"] for row in result["training_profiles"] if row["status"] == "MEASURED"}
    expected_models = {spec["id"] for spec in matrix if spec.get("scenario") != "C"}
    measured_validation = {row["model_id"] for row in result["validation_profiles"] if row["status"] == "MEASURED" and row.get("threshold_metrics_status") != "NOT_MEASURED"}
    missing_required = sorted(expected_models - (measured_models & measured_validation))
    result["not_measured"].extend("Required model profile incomplete: " + name for name in missing_required)
    result["ready_for_full"] = (not result["synthetic"] and result["resource_checks"]["passed"]
                                 and not missing_required and len(result["fusion_profiles"]) == 5
                                 and all(row["status"] == "MEASURED" for row in result["fusion_profiles"]))
    result["peak_host_rss_bytes"] = _rss_bytes()
    writes={row["model_id"]:row.get("temporary_materialized_bytes",0) for row in result["training_profiles"] if "temporary_materialized_bytes" in row}
    result["measured_artifact_writes"] = dict(status="MEASURED",materialized_bytes=sum(writes.values()),
        peak_temporary_probe_bytes=max(writes.values(),default=0),per_model=writes,
        policy="stat sizes of actually materialized private profile artifacts before cleanup; repeated overwrite IO not counted twice; not trained-output compressibility")
    result["test_access"] = "metadata counts only; no TEST targets, inference, score maps or metrics"
    if result["synthetic"]:
        result["not_measured"].append("Real-data and CUDA resource costs: NOT_MEASURED")
    _atomic_json(out / (PREFIX+"resource_report.json"),result)
    (out / (PREFIX+"resource_report.txt")).write_text(_text_report(result),encoding="utf-8")
    emit("complete", status=result["status"], ready_for_full=result["ready_for_full"])
    return result


def _profile_fusion(dataset,normalizer,config,training,emit):
    fusion = _module("fusion")
    row = deterministic_examples(dataset)[0]["row"]
    device, opts = config["device"], training.options(config)
    amp = str(device).startswith("cuda") and opts["amp"]
    accumulation = opts["effective_batch"]
    # The current observation is read only after the causal H0 forward.
    spec = next(item for item in training.resolve_matrix(config) if item["task"] == "h0" and item["control"] == "history")
    model = training.build_model(spec,dataset,config).eval()
    with torch.inference_mode():
        predicted = training.prediction(model,dataset,row,spec,normalizer,config)
    target = dataset.targets(row)
    support = torch.as_tensor(np.asarray(predicted["support"]), device=device)[None,None]
    labels = torch.as_tensor(np.asarray(target["target"]),device=device)[None,None].float()
    valid = support & (labels != 255)
    stack = torch.as_tensor(np.asarray(dataset.scores(row)[0]),device=device)[None,None].float()
    current = training._feature_target(dataset,row,normalizer,device)[0]
    forecast = torch.as_tensor(predicted["extras"]["forecast"],device=device)[None].float().detach()
    native_support = F.interpolate(support.float(),size=dataset.grid,mode="nearest").bool()
    native_stack = F.interpolate(stack,size=dataset.grid,mode="bilinear",align_corners=False)
    scene_keys=("building_id","view_id","previous_year","current_year")
    scene=tuple(row[key] for key in scene_keys)
    scene_rows=[item for item in dataset.split_samples("train") if tuple(item[key] for key in scene_keys)==scene]
    class SceneSubset:
        samples=scene_rows
        events={item["event_id"]:item for item in scene_rows}
        def __getattr__(self,key):
            return getattr(dataset,key)
    canonical=_module("data").EvaluationData(SceneSubset())
    frozen_cache={}
    def frozen_prediction(item):
        if item["event_id"] not in frozen_cache:
            prediction=(predicted if item["event_id"]==row["event_id"] else
                        training.prediction(model,dataset,item,spec,normalizer,config))
            raw,_=dataset.stack_components(item)
            frozen_cache[item["event_id"]]=dict(score=prediction["score"],support=prediction["support"],
                stack_raw=raw,stack_contract=dataset.stack_contract)
        return frozen_cache[item["event_id"]]
    # First pass includes preparation; second pass measures only frozen compact
    # fields and canonical merging. Forecast tensors are never recopied here.
    (block_key,block),prepare_seconds=_timed(lambda: next(fusion.iter_score_blocks(canonical,scene_rows,frozen_prediction,halo=2)),device)
    frozen_count=len(frozen_cache)
    (block_key,block),merge_seconds=_timed(lambda: next(fusion.iter_score_blocks(canonical,scene_rows,frozen_prediction,halo=2)),device)
    if len(frozen_cache)!=frozen_count:
        raise RuntimeError("Frozen fusion profiling unexpectedly recomputed a base output")
    del model
    merged={key:torch.as_tensor(block[key],device=device)[None] for key in ("stack","method","support","target")}
    merged["stack"],merged["method"]=merged["stack"].float(),merged["method"].float()
    core=block["core_slices"]
    core_mask=torch.zeros_like(merged["support"],dtype=torch.bool)
    core_mask[...,core[0],core[1]]=True
    merged_valid=merged["support"].bool() & (merged["target"]!=255) & core_mask
    results=[]
    for kind in ("logreg","score-mlp","score-cnn","feature-mlp","cross-attention"):
        emit("fusion-head",kind=kind)
        head=fusion.build_head(kind,channels=dataset.channels).to(device)
        optimizer=torch.optim.Adam(head.parameters(),lr=opts["fusion_lr"],weight_decay=opts.get("fusion_weight_decay",0.))
        scaler=_scaler(amp)
        feature=kind in ("feature-mlp","cross-attention")
        def cycle():
            optimizer.zero_grad(set_to_none=True)
            total=0.
            for _ in range(accumulation):
                context=torch.autocast(device_type="cuda",dtype=torch.float16) if amp else nullcontext()
                with context:
                    if feature:
                        probability,_=fusion.feature_prediction(head,current,forecast,native_support,native_stack,output_size=labels.shape[-2:])
                    else:
                        logits=fusion.score_logits(head,merged["stack"],merged["method"],merged["support"])
                        loss=F.binary_cross_entropy_with_logits(logits[merged_valid],merged["target"][merged_valid].float()) if merged_valid.any() else logits.sum()*0
                if feature:
                    loss=F.binary_cross_entropy(probability.float()[valid],labels[valid]) if valid.any() else probability.sum()*0
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite fusion profile loss")
                scaler.scale(loss/accumulation).backward()
                total+=float(loss.detach())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head.parameters(),opts["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            return total/accumulation
        warmup=config.get("profile_warmup",config.get("profile_warmup_steps",1))
        for _ in range(warmup):
            cycle()
        timings=[]
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(config.get("profile_steps",3)):
            _,seconds=_timed(cycle,device)
            timings.append(seconds)
        head.eval()
        def inference():
            context=torch.autocast(device_type="cuda",dtype=torch.float16) if amp else nullcontext()
            with torch.inference_mode(), context:
                if feature:
                    return fusion.feature_prediction(head,current,forecast,native_support,native_stack,output_size=labels.shape[-2:])[0]
                return fusion.score_prediction(head,merged["stack"],merged["method"],merged["support"])
        for _ in range(warmup):
            inference()
        inference_times=[_timed(inference,device)[1] for _ in range(config.get("profile_steps",3))]
        results.append(dict(kind=kind,status="MEASURED",parameters=sum(p.numel() for p in head.parameters()),
            seconds_per_optimizer_cycle=timings,seconds_per_event=[value/accumulation for value in timings],
            inference_seconds_per_unit=inference_times,inference_split="train frozen inputs; no TEST predictions",
            base_cache_preparation_seconds=prepare_seconds if not feature else None,
            event_id=row["event_id"],split="train",frozen_inputs=True,effective_batch=accumulation,
            microbatch=1,amp_used=amp,optimizer="Adam",lr=opts["fusion_lr"],weight_decay=opts.get("fusion_weight_decay",0.),
            warmup_optimizer_cycles=warmup, measured_shape=list(current.shape) if feature else list(merged["stack"].shape),
            measured_unit="crop event" if feature else "canonical merged block with halo2 and core256 loss",
            maps_to_fit_count=4 if feature else 16,canonical_input_preparation_seconds=merge_seconds if not feature else None,
            grouping="same head/input shapes across methods; score profiles merge actual overlapping TRAIN crops before head",
            peak_host_rss_bytes=_rss_bytes(),
            peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else None,
            peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(device) if str(device).startswith("cuda") else None,
            flops=dict(status="NOT_MEASURED",reason="No FLOPs counter installed; parameters are not compute equivalence")))
        del head,optimizer,scaler
    return results


def _time_estimates(report,config):
    estimates=[]
    counts=report["eligible_counts"]
    train,val=counts["train"]["transitions"],counts["val"]["transitions"]
    all_events=sum(row["transitions"] for row in counts.values())
    for identifier in sorted({row["model_id"] for row in report["training_profiles"]}):
        profiles=[row for row in report["training_profiles"] if row["model_id"]==identifier and row["status"]=="MEASURED"]
        if not profiles:
            continue
        n=counts["train"].get("predictive_transitions",train) if profiles[0]["task"] in ("actual-next","h0") else train
        times=[row["seconds_per_microbatch"]+row["repeated_input_io_seconds"] for row in profiles]
        validation=next((row for row in report["validation_profiles"] if row["model_id"]==identifier),{})
        sampled=max(1,len(validation.get("event_ids",[])))
        ratio=val/sampled
        val_inference=validation.get("seconds_per_event",0)*val
        val_merge=validation.get("overlap_merge_histograms_seconds",0)*ratio
        val_metrics=(validation.get("threshold_selection_seconds",0)+validation.get("metrics_seconds",0))*max(1,counts["val"].get("buildings",1)/2)
        checkpoint=max((row.get("checkpoint_write",{}).get("seconds",0) for row in profiles),default=0)
        stages=dict(train_input_compute=float(np.median(times))*n,VAL_inference=val_inference,
                    VAL_overlap_merge=val_merge,VAL_calibration_metrics=val_metrics,checkpoint_write=checkpoint)
        fixed=val_inference+val_merge+val_metrics+checkpoint
        export_write=(validation.get("scientific_array_write_seconds",0)/sampled+
            max((row.get("diagnostic_array_write",{}).get("seconds",0) for row in profiles),default=0))*all_events
        final=dict(inference=validation.get("seconds_per_event",0)*all_events,
                   scientific_arrays_write=export_write,
                   report_only_saved_map_replay=validation.get("report_only_replay_seconds",0)/sampled*all_events,
                   bootstrap=validation.get("bootstrap_seconds",0)*max(1,counts.get("test",{}).get("buildings",1)/2),
                   gallery_panels=validation.get("gallery_panel_seconds",0)*min(all_events,2*config.get("gallery_limit",12)))
        estimates.append(dict(model_id=identifier,status="ESTIMATED",eligible_train_events=n,
            epoch_stages_seconds=stages,epoch_seconds=sum(stages.values()),
            epoch_seconds_range=[min(times)*n+fixed,max(times)*n+fixed],
            maximum_epochs=config.get("epochs",60),maximum_budget_seconds=sum(stages.values())*config.get("epochs",60),
            final_stages_seconds=final,final_inference_source="VAL rates extrapolated to event metadata counts; no TEST inference",
            uncertainty="observed typical/long/max range is not a confidence interval; IO/content/thermal/overlap load may exceed range"))
    report_probe=next((row for row in report["validation_profiles"] if "report_only_full_entrypoint_seconds" in row),{})
    report_probe_stages=dict(status="MEASURED" if report_probe else "NOT_MEASURED",
        saved_schema2_events=len(report_probe.get("event_ids",[])),
        report_only_seconds=report_probe.get("report_only_full_entrypoint_seconds"),
        any_event_gallery_seconds=report_probe.get("gallery_full_entrypoint_seconds"),
        scope="bounded VAL-only saved scientific data; full-run scalability remains an estimate")
    fusion=[]
    block_count=report.get("canonical_train_blocks",train)
    for row in report["fusion_profiles"]:
        units=block_count if row["kind"] in ("logreg","score-mlp","score-cnn") else train
        score_head=row["kind"] in ("logreg","score-mlp","score-cnn")
        val_units=report.get("canonical_blocks_by_split",{}).get("val",val) if score_head else val
        final_units=sum(report.get("canonical_blocks_by_split",{}).values()) if score_head else all_events
        inference=float(np.median(row["inference_seconds_per_unit"]))
        calibration=float(np.mean([item.get("threshold_selection_seconds",0)+item.get("metrics_seconds",0)
                                   for item in report["validation_profiles"]])) if report["validation_profiles"] else 0.
        fusion.append(dict(kind=row["kind"],fits=row["maps_to_fit_count"],units_per_epoch=units,
            maximum_epochs=config.get("fusion_epochs",5),
            train_seconds=float(np.median(row["seconds_per_event"]))*row["maps_to_fit_count"]*units*config.get("fusion_epochs",5),
            VAL_inference_seconds=inference*val_units*row["maps_to_fit_count"]*config.get("fusion_epochs",5),
            VAL_calibration_metrics_seconds=calibration*row["maps_to_fit_count"]*config.get("fusion_epochs",5)*max(1,counts["val"].get("buildings",1)/2),
            final_inference_seconds=inference*final_units*row["maps_to_fit_count"],
            frozen_input_preparation_seconds=(row.get("canonical_input_preparation_seconds") or 0)*units*row["maps_to_fit_count"],
            source="measured frozen TRAIN head inference and VAL canonical calibration; extrapolated by metadata, no TEST predictions",
            uncertainty="one actual shape group, same head dimensions; content affects masks and IO"))
    return dict(status="ESTIMATED",primary=estimates,primary_upper_epoch_budget_seconds=sum(row["maximum_budget_seconds"] for row in estimates),
                fusion_default_epochs=config.get("fusion_epochs",5),fusion=fusion,report_only_probe=report_probe_stages,
                fusion_train_seconds=sum(row["train_seconds"] for row in fusion),
                final_primary_stages_seconds={name:sum(row["final_stages_seconds"][name] for row in estimates) for name in estimates[0]["final_stages_seconds"]} if estimates else {},
                unknown_overhead=["full-dataset report/gallery scaling beyond measured sample", "checkpoint selection metadata and dedup hashing", "missing cache extraction", "filesystem load changes"],
                not_a_runtime_guarantee=True)


def unavailable_report(config,out,*,source_snapshot,reason,hardware=None):
    """Persist honest readiness information when local owner inputs are absent."""
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    hardware=hardware or hardware_snapshot(config.get("device","cpu"))
    identity=dict(protocol=PROTOCOL,code=source_snapshot,config=config,data=None,hardware=_hardware_identity(hardware))
    result=dict(protocol=PROTOCOL,status="NOT_MEASURED",ready_for_full=False,counts=COUNTS,
        created_unix=time.time(),created_utc=datetime.now(timezone.utc).isoformat(),
        fingerprint=dict(sha256=_digest(identity),identity=identity,components={k:_digest(v) for k,v in identity.items()}),
        hardware=hardware,eligible_counts=None,training_profiles=[],fusion_profiles=[],validation_profiles=[],
        storage=dict(status="NOT_MEASURED",reason="Actual data shapes/counts unavailable; no fixed58GiB assumption"),
        time_estimates=dict(status="NOT_MEASURED"),not_measured=[reason,"Real-data/CUDA cost is not inferred from a CPU fixture"],
        input_cache_inventory=dict(status="NOT_MEASURED"),limitations=["Run preflight on the owner's prepared dataset before full."],
        resource_checks=check_resources({},config,out,hardware,raise_on_failure=False),
        test_access="none")
    _atomic_json(out/(PREFIX+"resource_report.json"),result)
    (out/(PREFIX+"resource_report.txt")).write_text(_text_report(result),encoding="utf-8")
    return result


def run_preflight(dataset, config, out, *, source_snapshot, hardware=None, progress=None):
    """Public wrapper persists a partial diagnostic on interruption/error, then re-raises."""
    tracker = _module("progress").Progress(stage="preflight",total=18+5,
        interval=config.get("progress_interval_seconds",30))
    def emit(stage, **fields):
        tracker.update(1 if stage in ("primary","fusion-head") else 0,current_stage=stage,**fields)
        if progress is not None:
            progress(stage,**fields)
    try:
        with tracker:
            return _run_preflight(dataset,config,out,source_snapshot=source_snapshot,hardware=hardware,progress=emit)
    except BaseException as exc:
        path=Path(out)/(PREFIX+"resource_report.json")
        if path.exists():
            result=json.loads(path.read_text())
            result.update(status="NOT_MEASURED",ready_for_full=False,error=dict(type=type(exc).__name__,message=str(exc)))
            result.setdefault("not_measured",[]).append("Profile interrupted before all measurements completed")
            _atomic_json(path,result)
            (Path(out)/(PREFIX+"resource_report.txt")).write_text(_text_report(result),encoding="utf-8")
        else:
            unavailable_report(config,out,source_snapshot=source_snapshot,reason=str(exc),hardware=hardware)
        raise
