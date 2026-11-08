"""Evaluate completed memory heads from an interrupted run without training."""
from __future__ import annotations

import argparse
import fcntl
import importlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from facade_change.io import read_json, sha256, write_json

memory = importlib.import_module("data_analysis.2026-11-08_variability_memory")
fusion, metrics, PREFIX = memory.fusion, memory.metrics, memory.PREFIX


class SavedMemory:
    """Read-only, locked snapshot of committed features and selected heads."""

    def __init__(self, path):
        self.out = Path(path).expanduser().resolve()
        self.hashes = {}
        self.lock = None

    def __enter__(self):
        self.lock = (self.out / ".run.lock").open("rb")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            for name in ("run.json", "request.json", "experiment_contract.json", PREFIX + "encoder.json"):
                self.pin(self.out / name)
            self.record = read_json(self.out / "run.json")
            self.config = read_json(self.out / "request.json")
            self.contract = read_json(self.out / "experiment_contract.json")
            if (self.record.get("kind") != "canonical_variability_memory"
                    or self.record.get("config") != self.config
                    or self.contract.get("config") != self.config
                    or self.contract.get("protocol") != "canonical-variability-memory-v1"):
                raise ValueError("Expected a variability-memory run with its original configuration")
            self.encoder_sha256 = self.hashes[str(self.out / (PREFIX + "encoder.json"))]
            return self
        except BaseException as exc:
            self.lock.close()
            if isinstance(exc, BlockingIOError):
                raise ValueError("Memory training is still running; stop it with Ctrl+C first") from exc
            raise

    def __exit__(self, *exception):
        self.lock.close()

    def pin(self, path):
        value = sha256(path)
        if self.hashes.setdefault(str(path), value) != value:
            raise ValueError(f"Saved memory input changed: {path}")

    def read_unit(self, identifier):
        path = self.out / "units" / (identifier + ".json")
        if not path.exists():
            raise ValueError(f"Missing completed unit {identifier}; evaluation never trains or extracts features")
        self.pin(path)
        record, arrays = memory.benchmark.read_unit(self.out, identifier)
        artifact = fusion.data_module._inside(self.out, record["artifact_path"])
        expected = self.hashes.setdefault(str(artifact), record["artifact_sha256"])
        if expected != record["artifact_sha256"]:
            raise ValueError(f"Saved memory artifact changed: {artifact}")
        if identifier.startswith(PREFIX + "features-") and (
                record.get("kind") != "frozen_observation_features"
                or record.get("encoder_sha256") != self.encoder_sha256
                or record.get("target_labels_used") is not False):
            raise ValueError(f"Feature encoder contract differs: {identifier}")
        return record, arrays

    def verify(self):
        for path, expected in self.hashes.items():
            if sha256(path) != expected:
                raise ValueError(f"Saved memory input changed: {path}")


def validate_code(saved):
    """Only the previously audited progress-only runner change is compatible."""
    import torch

    model = importlib.import_module("facade_change.research.2026-11-08_variability_memory_model")
    recurrent = importlib.import_module("facade_change.research.2026-10-25_spatial_change_model")
    code = {Path(module.__file__).name: sha256(module.__file__) for module in
            (memory, model, metrics, fusion, memory.history_features, fusion.data_module, recurrent)}
    original = saved.contract["code"]
    if original != code:
        migration_path = Path(memory.__file__).with_name(PREFIX + "memory_progress_compatibility.json")
        migration = read_json(migration_path)
        runner = Path(memory.__file__).name
        legacy = {**code, runner: migration["before_sha256"]}
        if original != legacy or code[runner] != migration["after_sha256"]:
            raise ValueError("Training implementation changed; evaluate with the original compatible code")
    if saved.contract["torch"] != str(torch.__version__) or saved.contract["numpy"] != np.__version__:
        raise ValueError("Torch/NumPy versions differ; use the original memory worker environment")
    return code


def evaluate_saved(saved, source, frozen, inputs, args, device, code, checkpoints):
    out = Path(args.out).expanduser().resolve()
    for original in (saved.out, source.root, source.data.root, source.data.history_root, frozen.root):
        if out == original or out in original.parents or original in out.parents:
            raise ValueError("Output must be separate from all inputs")
    config = dict(memory_run=str(saved.out), input_sha256=dict(saved.hashes), seeds=args.seeds,
                  device=device, history_shift_tokens=saved.config["history_shift_tokens"])
    contract = dict(protocol="canonical-variability-memory-evaluation-v1", config=config,
                    code={**code, Path(__file__).name: sha256(__file__)},
                    training_performed=False, feature_extraction_performed=False,
                    fit_split="train", selection_split="val", test_previously_inspected=True)
    if (out / "run.json").exists() and read_json(out / "run.json").get("status") == "completed":
        if read_json(out / "request.json") != config or read_json(out / "experiment_contract.json") != contract:
            raise ValueError("Completed evaluation has different inputs, settings or code")
        for name, expected in read_json(out / "run.json")["output_sha256"].items():
            if sha256(out / name) != expected:
                raise ValueError(f"Completed evaluation output is corrupt: {name}")
        print(f"Already completed: {out}", flush=True)
        return read_json(out / (PREFIX + "summary.json"))
    store = fusion.data_module.Store(out, config, resume=args.resume)
    store.record["kind"] = "canonical_variability_memory_evaluation"
    write_json(out / "run.json", store.record)
    try:
        store.contract(contract)
        store.event("selected_memory_checkpoints_frozen_before_test", checkpoints=checkpoints)
        aggregate, rows, robustness, returns = [], [], [], []
        shifts = config["history_shift_tokens"]
        shape = argparse.Namespace(hidden=saved.config["hidden"], device=device)
        with memory.progress(total=len(args.seeds) * (3 + len(shifts)),
                             desc="Saved memory evaluation", unit="split", position=0, leave=True) as progress:
            for seed in args.seeds:
                models = {mode: memory.load_model(saved, PREFIX + f"model-{mode}-{seed}", shape, mode)[0]
                          for mode in memory.MODES}
                val = memory.evaluate(inputs, models, "val", label=f"VAL seed={seed}", position=1)
                thresholds = memory.calibrate(val, frozen)
                write_json(out / (PREFIX + f"thresholds-{seed}.json"), thresholds)
                for split in ("val", "train", "test"):
                    hist = val if split == "val" else memory.evaluate(
                        inputs, models, split, label=f"{split.upper()} seed={seed}", position=1)
                    frozen.parity(hist["stack"], split)
                    a, r = memory.reporting(source.data, hist, thresholds, split, seed)
                    aggregate.extend(a)
                    rows.extend(r)
                    for name, buildings in hist.items():
                        for building, histogram in buildings.items():
                            value = metrics.metrics(histogram[-1], thresholds[name]["threshold"], name == "anychange")
                            returns.append(dict(method=name, seed=seed, split=split, building_id=building,
                                threshold=thresholds[name]["threshold"], candidate_pixels=value["positive_pixels"],
                                detected_pixels=value["tp"], recall=value["recall"],
                                scope="annotation-matched return to a known older state; not reviewed events"))
                    progress.update(1)
                for shift in shifts:
                    hist = memory.evaluate(inputs, models, "test", history_shift=shift,
                                           label=f"TEST shift={shift} seed={seed}", position=1)
                    a, _ = memory.reporting(source.data, hist, thresholds, "test", seed)
                    robustness.extend(dict(history_shift_tokens=shift, approximate_pixels=14 * shift, **r) for r in a)
                    progress.update(1)
                del models
        comparisons = memory.paired_comparisons(rows, args.seeds)
        for name, values in (("metrics", aggregate), ("per_building", rows), ("comparisons", comparisons),
                             ("history_shift", robustness), ("return_to_past", returns)):
            memory.write_csv(out / (PREFIX + name + ".csv"), values)
        source.verify()
        frozen.verify()
        saved.verify()
        summary = dict(kind="canonical_variability_memory_evaluation", status="completed",
            memory_run=str(saved.out), source_status=saved.record["status"],
            evaluated_seeds=args.seeds, originally_requested_seeds=saved.config["seeds"],
            features=dict(observations=len(source.data.observations), newly_computed=0),
            training_performed=False, checkpoints=checkpoints,
            baseline_parity="exact stack/previous TP/FP/FN/TN on every building/split",
            fit_split="train", selection_split="val", test_previously_inspected=True,
            independent_event_labels=False, results=aggregate, comparisons=comparisons)
        write_json(out / (PREFIX + "summary.json"), summary)
        lines = ["SAVED VARIABILITY MEMORY — exploratory evaluation, no training or DINO extraction",
                 f"Evaluated seeds: {args.seeds}; original training plan: {saved.config['seeds']}",
                 "FPR budget selected on VAL only. TEST actual FPR may exceed 5%.",
                 "F1/IoU: fixed GT-positive facade cohort. FPR includes negative-only facades.",
                 "Annotation-derived changes; checkpoint exposure and inspected TEST remain limitations.",
                 "Facade bootstrap does not measure variability across training seeds."]
        lines.extend(f"Checkpoint seed={c['seed']} {c['mode']}: VAL-selected epoch {c['selected_epoch']}"
                     for c in checkpoints)
        for row in aggregate:
            if row["slice_type"] == "overall":
                values = " ".join(f"{key}={row[key]:.5f}" if row[key] is not None else f"{key}=NA"
                                  for key in ("f1", "iou", "precision", "recall", "fpr", "ap_binned"))
                lines.append(f"{row['split']:5s} seed={row['seed']} {row['method']:15s} {values}")
        report = "\n".join(lines) + "\n"
        (out / (PREFIX + "report.txt")).write_text(report, encoding="utf-8")
        memory.tqdm.write(report)
        outputs = {path.name: sha256(path) for path in sorted(out.glob(PREFIX + "*")) if path.is_file()}
        store.finish("completed", summary_sha256=outputs[PREFIX + "summary.json"], output_sha256=outputs)
        return summary
    except BaseException as exc:
        store.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        store.close()


def run(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Specify distinct completed seeds with --seeds")
    with SavedMemory(args.run) as saved:
        if set(args.seeds) - set(saved.config["seeds"]):
            raise ValueError("Requested seeds are absent from the original training plan")
        code = validate_code(saved)
        device = args.device or saved.config["device"]
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable in the original worker environment")
            free, total = torch.cuda.mem_get_info(device)
            print(f"Evaluation GPU: {free / 2**30:.2f}/{total / 2**30:.2f} GiB free", flush=True)
            if free < 512 * 2**20:
                raise ValueError("Less than 512 MiB GPU memory is free")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=device.startswith("cuda"))
        checkpoints = []
        for seed in args.seeds:
            for mode in memory.MODES:
                record, _ = saved.read_unit(PREFIX + f"model-{mode}-{seed}")
                if (record.get("kind") != "causal_variability_head" or record.get("mode") != mode
                        or record.get("seed") != seed or record.get("fit_split") != "train"
                        or record.get("selection_split") != "val" or record.get("test_used_for_fit") is not False
                        or not 1 <= record.get("selected_epoch", 0) <= saved.config["epochs"]):
                    raise ValueError(f"Invalid saved checkpoint: {mode} seed={seed}")
                checkpoints.append({k: record[k] for k in
                                    ("unit_id", "mode", "seed", "selected_epoch", "parameters")})
        with memory.progress(total=2, desc="Validate original benchmark and fusion", unit="input") as progress:
            source = fusion.Source(saved.config["source"])
            progress.update(1)
            frozen = memory.FrozenStack(saved.config["fusion"], source)
            progress.update(1)
        if source.hashes != saved.config["source_sha256"] or frozen.hashes != saved.config["fusion_sha256"]:
            raise ValueError("Benchmark or frozen fusion differs from the training inputs")
        inputs = memory.Inputs(source, saved, frozen, device)
        with memory.progress(sorted(source.data.observations), desc="Verify saved features", unit="observation") as progress:
            for identifier in progress:
                inputs.feature(identifier)
        return evaluate_saved(saved, source, frozen, inputs, args, device, code, checkpoints)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Stopped variability-memory run with completed heads")
    parser.add_argument("--out", required=True, help="Separate evaluation output directory")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="Completed seeds to evaluate, e.g. 13")
    parser.add_argument("--device", help="Defaults to the training device")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if not args.worker:
            request = read_json(Path(args.run).expanduser().resolve() / "request.json")
            record = read_json(Path(request["source"]) / "run.json")
            python = record["config"]["options"]["dinov2"].get("worker_python", sys.executable)
            command = [python, "-s", "-u", "-m", "data_analysis", "memory-evaluate", "--worker"]
            for key, value in vars(args).items():
                if key == "worker" or value is False or value is None:
                    continue
                command.append("--" + key.replace("_", "-"))
                if value is True:
                    continue
                if key in ("run", "out"):
                    value = str(Path(value).expanduser().resolve())
                command.extend(map(str, value if isinstance(value, list) else [value]))
            print(f"Evaluation worker: {python}", flush=True)
            raise SystemExit(subprocess.run(command, cwd=Path(__file__).resolve().parents[1]).returncode)
        run(args)
    except (ValueError, OSError, KeyError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Ошибка memory-evaluate: {exc}\n")


if __name__ == "__main__":
    main()
