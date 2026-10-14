"""Resumable TRAIN/VAL temporal comparisons on saved raw-RGB histories.

This is an exploratory state-memory experiment, not a paid H0/H1 codec.
The existing method registry, workers and append-only research event schema
remain the boundaries for computation and provenance.
"""
from __future__ import annotations

import importlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .io import environment, read_json, run_record, sha256, sync_directory, write_json
from .research.common import ExperimentRun, digest

DEFAULT_METHODS = ("rgb_diff", "ssim", "lpips", "dinov2", "semantic_dino",
                   "anychange", "geoscd", "msdzip_abs", "msdzip_mod256")
HISTORY_METHODS = ("dinov2", "semantic_dino", "msdzip_abs", "msdzip_mod256")
PATH_KEYS = {"worker_python", "source_root", "checkpoint_path", "dino_root", "dino_checkpoint",
             "py_utils_root", "sam_checkpoint", "backbone_checkpoint", "calibration_checkpoint",
             "training_run", "model_config", "current_manifest"}


def _module(name):
    return importlib.import_module(f".research.2026-10-14_temporal_{name}", __package__)


def _progress(message):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}] {message}", flush=True)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def _atomic_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


class RunStore:
    """Atomic unit markers are authoritative; JSONL remains an append-only audit."""

    def __init__(self, out, config, resume=False):
        self.out = Path(out).expanduser().resolve()
        existed = self.out.exists()
        if existed and not resume:
            raise FileExistsError(f"Output already exists; use --resume: {self.out}")
        self.out.mkdir(parents=True, exist_ok=True)
        self._lock = (self.out / ".run.lock").open("a+")
        import fcntl
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock.close()
            raise RuntimeError(f"Another runner is using {self.out}") from exc
        self.sequence = 0
        request = self.out / "request.json"
        if request.exists() and read_json(request) != config:
            self.close()
            raise ValueError("Resume configuration differs; use a new output directory")
        _atomic_json(request, config)
        record_path = self.out / "run.json"
        self.record = read_json(record_path) if record_path.exists() else run_record("temporal_experiment", config)
        events = self.out / "events.jsonl"
        if events.exists():
            raw = events.read_bytes()
            lines = raw.splitlines()
            broken_tail = False
            for i, line in enumerate(lines):
                try:
                    self.sequence = max(self.sequence, int(json.loads(line)["sequence"]))
                except (ValueError, KeyError):
                    if i != len(lines) - 1 or raw.endswith(b"\n"):
                        self.close()
                        raise ValueError("Corrupt event ledger before its trailing interrupted write")
                    broken_tail = True
            if raw and not raw.endswith(b"\n"):
                if broken_tail:
                    boundary = raw.rfind(b"\n") + 1
                    fragment = self.out / f"2026-10-14-interrupted-ledger-{time.time_ns()}.bin"
                    fragment.write_bytes(raw[boundary:])
                    with events.open("r+b") as stream:
                        stream.truncate(boundary)
                        stream.flush()
                        os.fsync(stream.fileno())
                else:
                    with events.open("ab") as stream:
                        stream.write(b"\n")
        self.record.update(status="running", run_id=self.out.name)
        _atomic_json(record_path, self.record)
        self.event("resumed" if existed else "started")

    def event(self, kind, **values):
        event = ExperimentRun.event(self, kind, **values)
        with (self.out / "events.jsonl").open("rb") as stream:
            os.fsync(stream.fileno())
        return event

    def contract(self, value):
        path = self.out / "experiment_contract.json"
        if path.exists() and read_json(path) != value:
            raise ValueError("Resume data, weights, source code or environment changed; use a new output directory")
        _atomic_json(path, value)
        self.event("preflight_verified", contract_sha256=sha256(path))

    def read_unit(self, unit_id):
        path = self.out / "units" / f"{unit_id}.json"
        if not path.exists():
            return None
        record = read_json(path)
        artifact = self.out / record["artifact_path"]
        if record["unit_id"] != unit_id or sha256(artifact) != record["artifact_sha256"]:
            raise ValueError(f"Committed artifact is corrupt: {artifact}")
        with np.load(artifact, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        return record, arrays

    def save_unit(self, record, arrays):
        unit_id = record["unit_id"]
        if (self.out / "units" / f"{unit_id}.json").exists():
            raise FileExistsError(f"Refusing to overwrite completed unit {unit_id}")
        relative = f"maps/{unit_id}.npz"
        _atomic_npz(self.out / relative, arrays)
        record = {**record, "artifact_path": relative, "artifact_sha256": sha256(self.out / relative)}
        record["source_event_id"] = self.event("unit_artifact_saved", unit_id=unit_id,
                                               artifact_sha256=record["artifact_sha256"])
        _atomic_json(self.out / "units" / f"{unit_id}.json", record)
        self.event("unit_completed", unit_id=unit_id)
        return record

    def records(self):
        records = []
        for path in sorted((self.out / "units").glob("*.json")):
            record = read_json(path)
            if sha256(self.out / record["artifact_path"]) != record["artifact_sha256"]:
                raise ValueError(f"Committed artifact is corrupt: {path}")
            records.append(record)
        return records

    def finish(self, status, **values):
        self.record.update(status=status, finished_utc=datetime.now(timezone.utc).isoformat(), **values)
        _atomic_json(self.out / "run.json", self.record)
        self.event(status, **values)

    def close(self):
        if getattr(self, "_lock", None):
            self._lock.close()
            self._lock = None


def _path(value, parent):
    value = Path(value).expanduser()
    # Resolving a venv interpreter symlink silently selects the base Python.
    return os.path.abspath(parent / value if not value.is_absolute() else value)


def resolve_config(config_path=None, **overrides):
    parent = Path(config_path).expanduser().resolve().parent if config_path else Path.cwd()
    supplied = read_json(config_path) if config_path else {}
    supplied = {**supplied, **{k: v for k, v in overrides.items() if v is not None}}
    defaults = {"methods": list(DEFAULT_METHODS), "device": "cuda:0", "seed": 42,
                "history_methods": list(HISTORY_METHODS), "memory_exemplars": 3,
                "memory_update_fpr": .01, "crops_per_building": 1,
                "real_tile_size": 512, "real_stride": 256, "real_min_support_fraction": .8,
                "training": {}, "legacy_checkpoints": {}, "method_options": {},
                "alignment_runs": [], "semantic_run": "auto", "methods_config": "auto"}
    allowed = set(defaults) | {"dataset", "history_run", "training_python", "max_buildings_per_split"}
    unknown = set(supplied) - allowed
    if unknown:
        raise ValueError(f"Unknown temporal settings: {sorted(unknown)}")
    config = {**defaults, **supplied}
    if not config.get("history_run") and not config.get("dataset"):
        raise ValueError("Provide --dataset or --history-run")
    if (not config["methods"] or len(set(config["methods"])) != len(config["methods"])
            or set(config["methods"]) - set(DEFAULT_METHODS)):
        raise ValueError("Temporal methods must be distinct members of the documented nine-method comparison")
    if set(config["history_methods"]) - set(HISTORY_METHODS):
        raise ValueError("History is supported for DINOv2, semantic DINO and the two MSDZip representations")
    for key in ("dataset", "history_run", "training_python"):
        if config.get(key):
            config[key] = _path(config[key], parent)
    config["alignment_runs"] = [_path(p, parent) for p in config["alignment_runs"]]
    repository = Path(__file__).resolve().parents[1]
    methods_file = config["methods_config"]
    if methods_file == "auto":
        methods_file = next((p for p in (repository / "configs/methods.local.json",
                                         repository / "configs/methods.example.json") if p.is_file()), None)
    elif methods_file:
        methods_file = Path(_path(methods_file, parent))
    options = {}
    if methods_file:
        options = read_json(methods_file).get("methods", {})
        options = {name: {key: _path(value, Path(methods_file).parent) if key in PATH_KEYS and value else value
                          for key, value in values.items()} for name, values in options.items()}
    for name, values in config["method_options"].items():
        options[name] = {**options.get(name, {}), **{
            key: _path(value, parent) if key in PATH_KEYS and value else value for key, value in values.items()}}
    config["methods_config"] = str(methods_file) if methods_file else None
    semantic = config["semantic_run"]
    if semantic == "auto":
        candidates = []
        for path in (repository / "runs").glob("*/run.json"):
            try:
                record = read_json(path)
                if record.get("kind") == "dino_semantic_training" and record.get("status") == "completed_exploratory":
                    candidates.append((record.get("finished_utc", record.get("started_utc", "")), str(path.parent)))
            except (OSError, ValueError):
                continue
        semantic = max(candidates)[1] if candidates else None
    elif semantic:
        semantic = _path(semantic, parent)
    config["semantic_run"] = semantic
    if semantic:
        options["semantic_dino"] = {**options.get("semantic_dino", {}), "training_run": semantic}
        if "worker_python" not in options["semantic_dino"]:
            options["semantic_dino"]["worker_python"] = options.get("dinov2", {}).get("worker_python", sys.executable)
        if config.get("dataset"):
            ds = Path(config["dataset"])
            dataset_json = ds / "dataset.json" if ds.is_dir() else ds
            metadata = read_json(dataset_json) if dataset_json.is_file() else {}
            manifest = (dataset_json.parent / metadata["manifest"] if metadata.get("manifest")
                        else ds / "prepared/manifest.json" if ds.is_dir() else ds)
            if manifest.is_file():
                options["semantic_dino"]["current_manifest"] = str(manifest)
    config["method_options"] = {name: options.get(name, {}) for name in config["methods"]}
    if not config.get("training_python"):
        candidate = options.get("dinov2", {}).get("worker_python")
        config["training_python"] = candidate if candidate and Path(candidate).is_file() else sys.executable
    config["legacy_checkpoints"] = {key: _path(value, parent) for key, value in config["legacy_checkpoints"].items()}
    if set(config["legacy_checkpoints"]) - {"abs", "mod256"}:
        raise ValueError("legacy_checkpoints keys must be abs and/or mod256")
    return config


def _fingerprints(config):
    files, missing = {}, {}
    for method, options in config["method_options"].items():
        errors = []
        for key, value in options.items():
            if key not in PATH_KEYS or not value:
                continue
            path = Path(value)
            if not path.exists():
                errors.append(f"Missing {key}: {path}")
            elif path.is_file():
                files[str(path)] = sha256(path)
            elif key == "training_run":
                for name in ("run.json", "config.json", "best.pth", "summary.json"):
                    item = path / name
                    if item.is_file():
                        files[str(item)] = sha256(item)
                    else:
                        errors.append(f"Missing training artifact: {item}")
            else:
                for item in sorted(path.rglob("*.py")):
                    files[str(item)] = sha256(item)
        if method == "semantic_dino" and not options.get("training_run"):
            errors.append("No completed semantic training run found; provide --semantic-run")
        if errors:
            missing[method] = errors
    for path in config["legacy_checkpoints"].values():
        if not Path(path).is_file():
            missing[path] = [f"Missing legacy checkpoint: {path}"]
        else:
            files[path] = sha256(path)
    return files, missing


def _prepare_history(config, store):
    if config.get("history_run"):
        _progress(f"Reusing prepared synthetic histories: {config['history_run']}")
        return config
    target = store.out / "prepared_history"
    complete = target / "run.json"
    if not complete.exists() or not read_json(complete).get("status", "").startswith("completed"):
        _progress(f"Preparing synthetic histories: {target}")
        if target.exists():
            abandoned = target.with_name(f"prepared_history_interrupted_{time.time_ns()}")
            target.rename(abandoned)
            store.event("preparation_retry", previous=str(abandoned))
        from .hypothesis_dataset import prepare_history_dataset
        prepare_history_dataset(config["dataset"], target, seed=config["seed"],
                                crops_per_building=config["crops_per_building"],
                                max_buildings_per_split=config.get("max_buildings_per_split", 0),
                                splits=("train", "val"))
    else:
        _progress(f"Reusing prepared synthetic histories: {target}")
    return {**config, "history_run": str(target)}


def _train(config, out, smoke=False):
    forbidden = {"dataset_run", "out", "resume", "representations", "max_steps_per_call"} & set(config["training"])
    if forbidden:
        raise ValueError(f"Training settings cannot override orchestration fields: {sorted(forbidden)}")
    options = {"dataset_run": config["history_run"], "out": str(out), "device": config["device"],
               "seed": config["seed"], **config["training"]}
    representations = [name.removeprefix("msdzip_") for name in config["methods"] if name.startswith("msdzip_")]
    if not representations:
        return {}
    _progress(f"MSDZip {', '.join(representations)}: {'smoke' if smoke else 'full'} training; output {out}")
    options["representations"] = representations
    if smoke:
        options.update(epochs=1, max_train_bytes=128, max_val_bytes=64, model_batch_size=2,
                       window_groups=1, timesteps=2, hidden_dim=8, ffn_dim=16, vocab_dim=4,
                       checkpoint_every_steps=1)
    executable = config.get("training_python")
    if not executable:
        candidate = config["method_options"].get("dinov2", {}).get("worker_python")
        executable = candidate if candidate and Path(candidate).is_file() else sys.executable
    log_path = out.parent / (out.name + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def invoke(paused):
        args = [str(executable), "-u", "-s", "-m", "facade_change", "msdzip-train"]
        values = {**options, "resume": out.exists()}
        if paused:
            values["max_steps_per_call"] = 1
        for key, value in values.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    args.append(flag)
            elif isinstance(value, (list, tuple)):
                args.extend([flag, *map(str, value)])
            else:
                args.extend([flag, str(value)])
        with log_path.open("a", encoding="utf-8") as log:
            env = os.environ.copy()
            if smoke:
                env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            process = subprocess.Popen(args, cwd=Path(__file__).resolve().parents[1],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                code = process.wait()
            except BaseException:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.terminate()
                raise
            finally:
                process.stdout.close()
        if code:
            raise RuntimeError(f"MSDZip training exited {code}; inspect {log_path}")
    if smoke and not out.exists():
        invoke(True)
    invoke(False)
    if read_json(out / "run.json").get("status") != "completed_exploratory":
        raise RuntimeError(f"MSDZip training did not complete: {out}")
    return {representation: str(out / f"2026-10-04_msdzip_h0_{representation}.pt") for representation in representations}


def _variants(config, trained):
    variants = []
    for method in config["methods"]:
        options = dict(config["method_options"].get(method, {}))
        if method not in {"rgb_diff", "ssim"}:
            options.setdefault("device", config["device"])
        if method.startswith("msdzip_"):
            representation = method.removeprefix("msdzip_")
            options["checkpoint_path"] = trained.get(representation, "")
            options.setdefault("worker_python", config.get("training_python") or sys.executable)
            msdzip = importlib.import_module(".2026-10-04_msdzip_h0", __package__)
            options["dataset_fingerprint"] = msdzip._parent(config["history_run"])[-1]
        variants.append({"method": method, "variant": "current_train" if method.startswith("msdzip_") else "frozen",
                         "id": method, "options": options})
        if method.startswith("msdzip_") and representation in config["legacy_checkpoints"]:
            old = {**options, "checkpoint_path": config["legacy_checkpoints"][representation]}
            old.pop("dataset_fingerprint", None)
            variants.append({"method": method, "variant": "legacy_provenance_qualified",
                             "id": method + "_legacy", "options": old})
    return variants


def _scorer(variant, pool):
    from .methods.remote import RemoteScorer
    return RemoteScorer(variant["method"], variant["options"], pool)


def _activate_verified(store, variant, scorer, sequences, stage):
    metadata = dict(scorer.activate())
    metadata.pop("worker_stderr_log", None)
    snapshot = metadata.pop("gpu_preflight", None)
    if snapshot:
        store.event("method_gpu_snapshot", method=variant["id"], resources=snapshot)
    worker = dict(metadata.get("worker_environment", {}))
    worker.pop("git_commit", None)
    worker.pop("git_dirty", None)
    metadata["worker_environment"] = worker
    if variant["method"].startswith("msdzip_"):
        train = set(metadata.get("buildings", {}).get("train", []))
        overlap = {part: sorted(train & {row["building_id"] for row in sequences if row["split"] == part})
                   for part in ("train", "val")}
        audit = {"training_buildings_available": bool(train), "current_split_overlap": overlap,
                 "independent_current_val": bool(train) and not overlap["val"],
                 "comparison": "legacy versus current training changes data/adaptation; not an architecture ablation"}
        scorer.temporal_origin_audit = audit
        metadata["temporal_origin_audit"] = audit
    target = store.out / "method_provenance" / f"{stage}_{variant['id']}.json"
    if target.exists() and read_json(target) != metadata:
        raise ValueError(f"Actual method weights, sources or worker environment changed: {variant['id']}")
    _atomic_json(target, metadata)


def _outputs(scorer, reference, source, support):
    if not support.any():
        return {"scores": np.full(support.shape, np.nan, np.float32)}
    scores = np.asarray(scorer(reference, source, support))
    if scores.dtype != np.float32 or scores.shape != support.shape or not np.isfinite(scores[support]).all():
        raise ValueError("Method returned invalid scores on geometric support")
    arrays = {"scores": scores.copy()}
    for key in ("raw_scores", "native_prediction"):
        value = getattr(scorer, key, None)
        if value is not None:
            arrays[key] = np.asarray(value).copy()
    arrays.update({key: value.copy() for key, value in getattr(scorer, "auxiliary_arrays", {}).items()})
    return arrays


def _labels(sequence, first, previous, current, mode, support):
    labels = np.full(support.shape, 255, np.uint8)
    if sequence["domain"] != "synthetic":
        return labels
    if mode == "earliest":
        visible = first["visibility"] & current["visibility"] & support
        changed = np.any(first["latent_rgb"] != current["latent_rgb"], axis=2)
        labels[visible] = changed[visible].astype(np.uint8)
    else:
        labels = current["event_labels"].copy()
        labels[~support] = 255
    return labels


def _identity(sequence, variant, mode, index):
    return digest([sequence["sequence_id"], variant["id"], mode, index])[:32]


def _unit_record(sequence, variant, mode, index, current, elapsed, scorer):
    return {"unit_id": _identity(sequence, variant, mode, index), "method": variant["method"],
            "variant": variant["variant"], "variant_id": variant["id"], "mode": mode,
            "target": "earliest_state" if mode == "earliest" else "step_event",
            "domain": sequence["domain"], "split": sequence["split"], "building_id": sequence["building_id"],
            "sequence_id": sequence["sequence_id"], "frame": index,
            "arm": sequence.get("arm"), "event": current.get("event", "unknown"),
            "nuisance": current.get("nuisance", "unknown"),
            "hypothesis": "H0" if sequence.get("arm") == "clean" else "H1",
            "output_kind": scorer.metadata.get("output_kind", "score"),
            "elapsed_seconds": elapsed, "status": "completed", "method_metadata": {
                **scorer.metadata, "temporal_origin_audit": getattr(scorer, "temporal_origin_audit", None)},
            "label_verified": sequence["domain"] == "synthetic"}


def _pairs(store, sequences, variant, scorer):
    inputs = _module("inputs")
    for sequence in sequences:
        first = inputs.load_frame(sequence, 0)
        previous = first
        for index in range(1, len(sequence["frames"])):
            current = inputs.load_frame(sequence, index)
            for mode, reference in (("previous", previous), ("earliest", first)):
                identifier = _identity(sequence, variant, mode, index)
                if store.read_unit(identifier):
                    continue
                support = reference["support"] & current["support"]
                started = time.monotonic()
                arrays = _outputs(scorer, reference["rgb"], current["rgb"], support)
                arrays.update(reference=reference["rgb"], source=current["rgb"], support=support,
                              labels=_labels(sequence, first, previous, current, mode, support))
                record = _unit_record(sequence, variant, mode, index, current, time.monotonic()-started, scorer)
                store.save_unit(record, arrays)
                print(f"{variant['id']} {sequence['sequence_id']} {mode} frame {index + 1}: saved", flush=True)
            previous = current


def _history(store, sequences, variant, scorer, config):
    records = [record for record in store.records() if
               record["variant_id"] == variant["id"] and record["mode"] == "previous"
               and record["domain"] == "synthetic" and record["split"] == "val" and record["arm"] == "clean"]
    def items():
        for record in records:
            yield {"record": record, "arrays": store.read_unit(record["unit_id"])[1]}
    calibration = _module("reports").calibrate_thresholds(items, (config["memory_update_fpr"],))
    if calibration["status"] != "calibrated":
        raise ValueError("History update rule requires completed previous-frame VAL H0 calibration")
    threshold = calibration["choices"][0]["threshold"]
    _atomic_json(store.out / f"memory_rule_{variant['id']}.json", calibration)
    memory_class = _module("history").TemporalMemory
    inputs = _module("inputs")
    for sequence in sequences:
        memory = memory_class(config["memory_exemplars"])
        previous = first = inputs.load_frame(sequence, 0)
        preceding_hash = None
        for index in range(len(sequence["frames"])):
            current = first if index == 0 else inputs.load_frame(sequence, index)
            identifier = _identity(sequence, variant, "history", index)
            ready = store.read_unit(identifier)
            if ready:
                record, arrays = ready
                if record.get("previous_history_artifact") != preceding_hash:
                    raise ValueError("History resume chain differs from committed frame order")
                memory = memory_class.from_state_dict({key[8:]: value for key, value in arrays.items()
                                                       if key.startswith("memory__")})
                preceding_hash = record["artifact_sha256"]
                previous = current
                continue
            auxiliary = []
            def compare(reference, source, support):
                result = _outputs(scorer, reference, source, support)
                auxiliary.append({key: value for key, value in result.items() if key != "scores"})
                return result["scores"]
            started = time.monotonic()
            _progress(f"{variant['id']} history {sequence['sequence_id']}: frame {index + 1}/{len(sequence['frames'])}")
            result = memory.score_then_update(current["rgb"], current["support"], compare, threshold)
            support = previous["support"] & current["support"] & result["support"]
            scores = result["scores"].astype(np.float32)
            scores[~support] = np.nan
            arrays = {"scores": scores, "reference": previous["rgb"], "source": current["rgb"],
                      "support": support, "labels": _labels(sequence, first, previous, current, "history", support),
                      "accepted_change": result["accepted_change"], "reference_frame": result["reference_frame"]}
            for number, values in enumerate(auxiliary):
                selected = (result["winning_comparison"] == number) & support
                for key, value in values.items():
                    if key not in arrays:
                        arrays[key] = np.full(value.shape, False if value.dtype == bool else np.nan, value.dtype)
                    arrays[key][..., selected] = value[..., selected]
            arrays.update({"memory__" + key: value for key, value in memory.state_dict().items()})
            record = _unit_record(sequence, variant, "history", index, current, time.monotonic()-started, scorer)
            record.update(previous_history_artifact=preceding_hash, memory_update_threshold=threshold)
            record["reference_display"] = "Previous observation for context; history compares current-episode exemplars selected by reference_frame"
            if index == 0:
                record["status"] = "initialized"
            record = store.save_unit(record, arrays)
            preceding_hash = record["artifact_sha256"]
            previous = current


def _smoke_window(frames, max_side=16):
    """Bound a machinery check on native pixels; never select by temporal labels.

    The same deterministic window is used at every date. This is only for tiny
    MSDZip smoke checkpoints, whose two-byte batches are unsuitable for dense
    full-frame inference. Experiment inputs and checkpoint parameters stay intact.
    """
    if not frames or type(max_side) is not int or max_side < 1:
        raise ValueError("Smoke window requires frames and a positive integer side")
    shape = frames[0]["support"].shape
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("Smoke frames require nonempty native support")
    for frame in frames:
        if (frame["support"].shape != shape or frame["support"].dtype != bool
                or frame["rgb"].shape != (*shape, 3) or frame["rgb"].dtype != np.uint8):
            raise ValueError("Smoke frames must share a native uint8 RGB/boolean support grid")
    common = np.logical_and.reduce([frame["support"] for frame in frames])
    if not common.any():
        raise ValueError("MSDZip smoke history has no common geometric support")
    height, width = shape
    h, w = min(max_side, height), min(max_side, width)
    integral = np.pad(common.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    counts = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
    y, x = map(int, np.unravel_index(int(counts.argmax()), counts.shape))
    selected = [{"rgb": frame["rgb"][y:y+h, x:x+w].copy(),
                 "support": frame["support"][y:y+h, x:x+w].copy()} for frame in frames]
    return selected, {"crop_xywh": [x, y, w, h], "native_shape": list(shape),
                      "scored_shape": [h, w], "common_supported_pixels": int(counts[y, x]),
                      "selection_policy": "maximum common geometric support; native pixels, no resize; smoke only"}


def _smoke(store, sequences, variant, scorer, config):
    marker = store.out / "smoke" / f"{variant['id']}.json"
    if marker.exists():
        old = read_json(marker)
        if old.get("status") == "passed" and sha256(store.out / old["artifact"]) == old["sha256"]:
            _progress(f"Smoke {variant['id']}: reusing verified saved result")
            return
    inputs = _module("inputs")
    chosen = [next(sequence for sequence in sequences if sequence["domain"] == "synthetic")]
    real = next((sequence for sequence in sequences if sequence["domain"] == "real"), None)
    if real:
        chosen.append(real)
    arrays, smoke_inputs, prepared = {}, {}, []
    for number, sequence in enumerate(chosen):
        with_history = sequence["domain"] == "synthetic" and variant["method"] in config["history_methods"]
        indices = range(len(sequence["frames"])) if with_history else (0, 1)
        frames = [inputs.load_frame(sequence, index) for index in indices]
        if variant["method"].startswith("msdzip_"):
            frames, selection = _smoke_window(frames)
        else:
            h, w = frames[0]["support"].shape
            selection = {"crop_xywh": [0, 0, w, h], "native_shape": [h, w],
                         "scored_shape": [h, w], "selection_policy": "full native smoke input"}
        smoke_inputs[sequence["domain"]] = {**selection, "sequence_id": sequence["sequence_id"]}
        prepared.append(frames)
        reference, current = frames[:2]
        support = reference["support"] & current["support"]
        _progress(f"Smoke {variant['id']}: {sequence['domain']} pair, grid {selection['scored_shape']}, "
                  f"supported pixels {int(support.sum())}")
        started = time.monotonic()
        result = _outputs(scorer, reference["rgb"], current["rgb"], support)
        arrays[f"scores_{number}"] = result["scores"]
        _progress(f"Smoke {variant['id']}: {sequence['domain']} pair completed in {time.monotonic() - started:.1f}s")
    if variant["method"] in config["history_methods"]:
        memory_class = _module("history").TemporalMemory
        memory = memory_class(config["memory_exemplars"])
        for index, frame in enumerate(prepared[0]):
            _progress(f"Smoke {variant['id']}: synthetic history frame {index + 1}/{len(prepared[0])}")
            memory.score_then_update(frame["rgb"], frame["support"], scorer, 1.)
            memory = memory_class.from_state_dict(memory.state_dict())
        arrays.update({"memory__" + key: value for key, value in memory.state_dict().items()})
    artifact = Path("smoke") / f"{variant['id']}.npz"
    _atomic_npz(store.out / artifact, arrays)
    with np.load(store.out / artifact, allow_pickle=False) as reloaded:
        for key, value in arrays.items():
            if not np.array_equal(reloaded[key], value, equal_nan=True):
                raise ValueError("Smoke save/reload parity failed")
    _atomic_json(marker, {"status": "passed", "artifact": str(artifact), "sha256": sha256(store.out / artifact),
                          "real_pair_tested": real is not None, "method_metadata": scorer.metadata,
                          "smoke_inputs": smoke_inputs})
    store.event("smoke_passed", method=variant["id"], real_pair_tested=real is not None)
    _progress(f"Smoke {variant['id']}: passed (including saved-artifact reload)")


def _reports(store, statuses=None):
    records = store.records()
    for name, status in (statuses or {}).items():
        if status["status"] == "failed":
            records.append({"unit_id": "method-" + name, "method": name, "variant": "unavailable",
                            "mode": "all", "domain": "all", "status": "failed", "error": status["error"]})
    return _module("reports").build_reports(store.out, records)


def run_temporal_experiment(out, config_path=None, resume=False, report_only=False,
                            preflight_only=False, smoke_only=False, **overrides):
    output = Path(out).expanduser().resolve()
    if resume or report_only:
        saved = read_json(output / "request.json") if (output / "request.json").exists() else None
        config = saved if saved is not None and not config_path and not any(v is not None for v in overrides.values()) else resolve_config(config_path, **overrides)
    else:
        config = resolve_config(config_path, **overrides)
    store = RunStore(output, config, resume=resume or report_only)
    _progress(f"Temporal experiment {'resuming' if resume else 'starting'}: {output}")
    statuses = read_json(output / "method_status.json") if (output / "method_status.json").exists() else {}
    previous_signal = signal.getsignal(signal.SIGTERM)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        if report_only:
            summary = _reports(store, statuses)
            store.finish("reports_rebuilt", report=summary)
            return summary
        resolved = _prepare_history(config, store)
        _progress("Loading histories, checking source hashes, and selecting real crops from saved geometry")
        sequences, inputs, coverage = _module("inputs").load_sequences(resolved)
        if not sequences:
            raise ValueError("No eligible TRAIN/VAL sequences")
        _progress(f"Inputs ready: {sum(s['domain'] == 'synthetic' for s in sequences)} synthetic histories; "
                  f"{sum(s['domain'] == 'real' for s in sequences)} real crop histories")
        _progress("Verifying method sources, checkpoints, and execution environment")
        weights, missing = _fingerprints(resolved)
        env = environment()
        contract = {"config": resolved, "inputs": inputs, "weights_and_method_sources": weights,
                    "source_sha256": env["source_sha256"], "packages": env["packages"],
                    "sequence_ids": [s["sequence_id"] for s in sequences],
                    "scope": "TRAIN/VAL exploratory temporal scores, not paid H0/H1; no TEST"}
        store.contract(contract)
        resources = {"free_disk_bytes": __import__("shutil").disk_usage(output).free}
        if __import__("shutil").which("nvidia-smi"):
            device_info = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
            resources["nvidia_smi"] = device_info.stdout or device_info.stderr
        _atomic_json(output / "preflight.json", {"missing_method_resources": missing, "coverage": coverage,
                                                 "resources": resources, "test_rgb_loaded": False})
        _atomic_json(output / "resolved_config.json", resolved)
        _progress(f"Preflight complete; missing method resources: {len(missing)}")
        if preflight_only:
            store.finish("preflight_complete", missing_methods=missing)
            return {"status": "preflight_complete", "missing_methods": missing, "coverage": coverage}
        from .methods.remote import RemotePool
        smoke_weights, trained, training_errors = {}, {}, {}
        for name in (name for name in config["methods"] if name.startswith("msdzip_")):
            representation = name.removeprefix("msdzip_")
            try:
                smoke_weights.update(_train({**resolved, "methods": [name]},
                                            output / "smoke_training" / representation, smoke=True))
            except Exception as exc:
                training_errors[name] = str(exc)
        with RemotePool(output / "logs") as pool:
            for variant in _variants(resolved, smoke_weights):
                name = variant["id"]
                try:
                    _progress(f"Smoke {name}: initializing method")
                    if variant["method"] in missing:
                        raise ValueError("; ".join(missing[variant["method"]]))
                    if variant["variant"] == "current_train" and variant["method"] in training_errors:
                        raise ValueError(training_errors[variant["method"]])
                    scorer = _scorer(variant, pool)
                    _activate_verified(store, variant, scorer, sequences, "smoke")
                    _smoke(store, sequences, variant, scorer, resolved)
                    statuses[name] = {"status": "smoke_passed"}
                except Exception as exc:
                    statuses[name] = {"status": "failed", "stage": "smoke", "error": f"{type(exc).__name__}: {exc}"}
                    store.event("method_failed", method=name, **statuses[name])
                    _progress(f"Smoke {name}: failed: {statuses[name]['error']}")
                finally:
                    pool.release()
                    _atomic_json(output / "method_status.json", statuses)
        if smoke_only:
            result = {"status": "smoke_complete", "methods": statuses}
            store.finish("smoke_complete", methods=statuses)
            return result
        _progress("Smoke stage complete; starting full training for methods that passed")
        for name in (name for name in config["methods"] if name.startswith("msdzip_")):
            if statuses.get(name, {}).get("status") != "smoke_passed":
                continue
            representation = name.removeprefix("msdzip_")
            try:
                trained.update(_train({**resolved, "methods": [name]}, output / "training" / representation))
            except Exception as exc:
                statuses[name] = {"status": "failed", "stage": "training", "error": str(exc)}
        with RemotePool(output / "logs") as pool:
            for variant in _variants(resolved, trained):
                name = variant["id"]
                if statuses.get(name, {}).get("status") == "failed":
                    continue
                try:
                    _progress(f"Full inference {name}: initializing method")
                    scorer = _scorer(variant, pool)
                    _activate_verified(store, variant, scorer, sequences, "inference")
                    _pairs(store, sequences, variant, scorer)
                    if variant["method"] in config["history_methods"]:
                        _history(store, sequences, variant, scorer, resolved)
                    statuses[name] = {"status": "completed", "method_metadata": scorer.metadata}
                    store.event("method_completed", method=name)
                except Exception as exc:
                    statuses[name] = {"status": "failed", "stage": "inference", "error": f"{type(exc).__name__}: {exc}"}
                    store.event("method_failed", method=name, **statuses[name])
                    _progress(f"Full inference {name}: failed: {statuses[name]['error']}")
                finally:
                    pool.release()
                    _atomic_json(output / "method_status.json", statuses)
        _progress("Rebuilding tables and gallery from saved result maps")
        summary = _reports(store, statuses)
        status = "completed_with_failures" if any(s["status"] == "failed" for s in statuses.values()) else "completed_exploratory"
        store.finish(status, methods=statuses)
        return {"status": status, "out": str(output), "methods": statuses, "report": summary}
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        store.finish(status, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_signal)
        store.close()


def launch_detached(argv=None):
    """Detach only the requested runner; never kill another process or auto-reboot."""
    from .cli import parser
    args = list(sys.argv[1:] if argv is None else argv)
    parsed = vars(parser().parse_args(args))
    out = Path(parsed["out"]).expanduser().resolve()
    if out.exists() and not (parsed["resume"] or parsed["report_only"]):
        raise FileExistsError(f"Output exists; add --resume: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_name(out.name + ".log")
    pid_path = out.with_name(out.name + ".pid")
    args = [value for value in args if value != "--detach"]
    with log_path.open("ab") as log, open(os.devnull, "rb") as null:
        process = subprocess.Popen([sys.executable, "-u", "-s", "-m", "facade_change", *args],
                                   cwd=Path(__file__).resolve().parents[1], stdin=null,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pid_path.write_text(str(process.pid) + "\n", encoding="ascii")
    return {"status": "launched", "pid": process.pid, "log": str(log_path), "out": str(out),
            "note": "No time limit. After reboot rerun with --resume; keep the computer awake."}
