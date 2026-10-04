"""Original MSDZip predictor: train on H0 residuals, then freeze byte NLL.

The upstream class is untouched. This is an offline adapter, not its online
compression loop: every prediction recomputes the full preceding-byte window.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

from .hypothesis_benchmark import _checked, _inside, _mask, _select
from .io import finish_record, load_rgb, new_directory, read_json, run_record, sha256, write_json


SOURCE_SHA256 = "159a46f2d0736b784c0b078b88039f2e1477c09817c5889b7cf48788d3b32f72"
SOURCE_COMMIT = "cbbc03797aebe3e820bd11b4f995ae3f0b6824d0"
SOURCE_PATH = Path(__file__).resolve().parents[1] / "third_party/2026-10-04_msdzip_compress_model.py"
PROTOCOL = {
    "byte_order": "native-grid row-major, interleaved RGB; uint8",
    "context": "previous timesteps bytes; excludes target; zero left padding at each contiguous supported run",
    "support": "base reference geometric support only; no edit, visibility, occlusion, or evaluation labels",
    "lane": "native flat RGB byte offset modulo fixed model batchsize; same in train and evaluation",
    "inactive_lanes": "zero contexts; padding targets excluded from loss and maps",
    "cache": "model.last=[] before every original forward; no cross-window or cross-case state",
    "evaluation": "frozen weights, eval mode, no optimizer; logits[:, -1, :] as upstream",
    "window_groups": "extra independent group dimension, fixed checkpoint group count; original fixed native-offset parameter lanes are preserved",
    "score": "mean channel negative log2 probability per pixel; normalized as 1-exp(-bpb/8)",
    "model_bits": "ideal probability NLL only; not an arithmetic-coded file size; weights, headers, padding, and H1 explanation costs excluded",
    "vocabulary": "all 256 byte values, fixed for both representations",
    "cuda_numerics": "TF32 disabled; cuDNN benchmark disabled and deterministic convolutions requested; fixed seeds; cross-device bitwise identity is not claimed",
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Original MSDZip requires PyTorch in the active environment") from exc
    return torch


def _device(name):
    torch = _torch()
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("MSDZip device must be cpu or cuda[:index]")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but PyTorch reports no available CUDA device")
        number = torch.cuda.current_device() if device.index is None else device.index
        if number >= torch.cuda.device_count():
            raise ValueError("Requested CUDA device does not exist")
        free, total = torch.cuda.mem_get_info(number)
        print(f"MSDZip CUDA {number}: free {free / 2**30:.2f}/{total / 2**30:.2f} GiB; existing processes are not modified", flush=True)
    return device


def _model_args(model_batch_size, timesteps, hidden_dim, ffn_dim, vocab_dim):
    values = (model_batch_size, timesteps, hidden_dim, ffn_dim, vocab_dim)
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("MSDZip architecture dimensions must be positive integers")
    if timesteps & (timesteps - 1):
        raise ValueError("Original MSDZip requires power-of-two timesteps")
    if timesteps * vocab_dim % hidden_dim:
        raise ValueError("Original MSDZip requires timesteps*vocab_dim divisible by hidden_dim")
    return {"batchsize": model_batch_size, "layers": int(math.log2(timesteps)) + 1,
            "hidden_dim": hidden_dim, "ffn_dim": ffn_dim, "vocab_dim": vocab_dim,
            "timesteps": timesteps, "vocab_size": 256}


def _new_model(args, device):
    if sha256(SOURCE_PATH) != SOURCE_SHA256:
        raise ValueError("Vendored original MSDZip source hash disagrees with pinned upstream")
    checked = _model_args(args["batchsize"], args["timesteps"], args["hidden_dim"], args["ffn_dim"], args["vocab_dim"])
    if args != checked:
        raise ValueError("MSDZip checkpoint architecture disagrees with the original byte predictor")
    spec = importlib.util.spec_from_file_location("facade_original_msdzip", SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MixedModel(**args).to(device)


def residual_bytes(reference, source, representation):
    if (reference.dtype != np.uint8 or source.dtype != np.uint8
            or reference.ndim != 3 or reference.shape[-1] != 3 or source.shape != reference.shape):
        raise ValueError("MSDZip requires same-grid native uint8 RGB inputs")
    difference = source.astype(np.int16) - reference.astype(np.int16)
    if representation == "abs":
        return np.abs(difference).astype(np.uint8)
    if representation == "mod256":
        return np.remainder(difference, 256).astype(np.uint8)
    raise ValueError("MSDZip residual representation must be abs or mod256")


def _windows(residual, support, positions, timesteps):
    """Only preceding supported bytes enter each window; target is never read."""
    if support.dtype != bool or support.shape != residual.shape[:2]:
        raise ValueError("MSDZip support must be a same-grid boolean mask")
    flat = residual.reshape(-1)
    valid = np.repeat(support.reshape(-1), 3)
    positions = np.asarray(positions, dtype=np.int64)
    if np.any(positions < 0) or np.any(positions >= len(flat)) or not valid[positions].all():
        raise ValueError("Sampled target byte lies outside native geometric support")
    # Each unsupported byte starts a new run; padded history cannot see across it.
    starts = np.maximum.accumulate(np.where(valid, 0, np.arange(len(flat)) + 1))
    offsets = positions[:, None] + np.arange(-timesteps, 0)[None, :]
    included = offsets >= starts[positions, None]
    contexts = np.zeros(offsets.shape, dtype=np.uint8)
    contexts[included] = flat[offsets[included]]
    return contexts, flat[positions].copy()


def _batches(contexts, targets, positions, batchsize, rng=None):
    """Put every sampled target in its permanent native-offset parameter lane."""
    queues = [np.flatnonzero(positions % batchsize == lane) for lane in range(batchsize)]
    if rng is not None:
        for queue in queues:
            rng.shuffle(queue)
    for offset in range(max(map(len, queues), default=0)):
        indices = np.full(batchsize, -1, dtype=np.int64)
        batch = np.zeros((batchsize, contexts.shape[1]), dtype=np.int64)
        labels = np.zeros(batchsize, dtype=np.int64)
        for lane, queue in enumerate(queues):
            if offset < len(queue):
                index = int(queue[offset])
                indices[lane] = index
                batch[lane] = contexts[index]
                labels[lane] = targets[index]
        yield batch, labels, indices


@contextmanager
def _numerics(device):
    torch = _torch()
    if device.type != "cuda":
        yield
        return
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _forward_original(model, contexts):
    model.last = []
    logits = model(contexts)[:, -1, :]
    model.last = []
    return logits


def _predict(model, contexts):
    with _numerics(contexts.device):
        logits = _forward_original(model, contexts)
    if logits.shape != (model.batchsize, 256) or not _torch().isfinite(logits).all():
        raise ValueError("Original MSDZip produced non-finite or unexpected byte logits")
    return logits


def _batch_groups(contexts, targets, positions, batchsize, window_groups, rng=None):
    """Bounded independent windows, with fixed-size padding in the final group."""
    pending = []
    for item in _batches(contexts, targets, positions, batchsize, rng):
        pending.append(item)
        if len(pending) == window_groups:
            yield tuple(np.stack([item[column] for item in pending]) for column in range(3))
            pending = []
    if pending:
        padding = (np.zeros((batchsize, contexts.shape[1]), np.int64),
                   np.zeros(batchsize, np.int64), np.full(batchsize, -1, np.int64))
        pending.extend([padding] * (window_groups - len(pending)))
        yield tuple(np.stack([item[column] for item in pending]) for column in range(3))


def _predict_groups(model, contexts):
    torch = _torch()
    with _numerics(contexts.device):
        if hasattr(torch, "vmap"):
            logits = torch.vmap(lambda x: _forward_original(model, x))(contexts)
        else:
            logits = torch.stack([_forward_original(model, x) for x in contexts])
    if logits.shape != (contexts.shape[0], model.batchsize, 256) or not torch.isfinite(logits).all():
        raise ValueError("Original MSDZip produced non-finite or unexpected grouped byte logits")
    return logits


def _parent(dataset_run):
    root = Path(dataset_run).expanduser().resolve()
    record = read_json(root / "run.json")
    if record.get("kind") != "hypothesis_dataset" or record.get("status") != "completed_needs_review":
        raise ValueError("MSDZip H0 training requires a completed procedural hypothesis dataset")
    summary_path, summary_hash = _checked(root, record, "summary.json")
    summary = read_json(summary_path)
    index_path, index_hash = _checked(root, record, summary["index_path"])
    split_path, split_hash = _checked(root, record, "split.json")
    _, config_hash = _checked(root, record, "config.json")
    index, split = read_json(index_path), read_json(split_path)
    if split.get("mode") != "reviewed" or split.get("development_only") is not False or index.get("split") != split:
        raise ValueError("MSDZip H0 requires the existing reviewed building split")
    if record["config"]["input_sha256"]["config"] != config_hash or summary["case_count"] != len(index["cases"]):
        raise ValueError("MSDZip parent config/case-count provenance disagrees")
    fingerprints = {"parent_run": sha256(root / "run.json"), "parent_summary": summary_hash,
                    "parent_index": index_hash, "split": split_hash, "config": config_hash}
    return root, record, index, split, fingerprints


def _rgb_inputs(root, parent, base, case, selected_hashes):
    paths = {}
    for key in ("reference_rgb", "source_rgb", "reference_support", "geometry"):
        paths[key], digest = _checked(root, parent, case[key], case["artifact_sha256"][key])
        selected_hashes[case[key]] = digest
    directory = _inside(root, base["path"])
    if paths["reference_rgb"] != directory / "reference_rgb.png" or paths["reference_support"] != directory / "reference_support.png":
        raise ValueError("H0 RGB/support identity disagrees with its frozen base")
    identity = np.eye(3).tolist()
    geometry = read_json(paths["geometry"])
    if any(case.get(key) != identity or geometry.get(key) != identity for key in ("source_to_reference", "reference_to_source")):
        raise ValueError("H0 residual training requires identity procedural geometry")
    reference, opaque = load_rgb(paths["reference_rgb"])
    source, source_opaque = load_rgb(paths["source_rgb"])
    if source.shape != reference.shape:
        raise ValueError("H0 RGB images have different native grids")
    support = _mask(paths["reference_support"], reference.shape[:2])
    if np.any(support & (~opaque | ~source_opaque)):
        raise ValueError("H0 support includes nonopaque RGB pixels")
    return reference, source, support


def _allocate(groups, budget):
    """Equal building/scenario cells, then equal cases, with capacity redistribution."""
    allocation = {case["case_id"]: 0 for rows in groups.values() for case in rows}
    capacities = {case["case_id"]: case["capacity"] for rows in groups.values() for case in rows}
    budget = min(budget, sum(capacities.values()))
    active = sorted(groups)
    while budget and active:
        share = max(1, budget // len(active))
        next_active = []
        for group in active:
            available = [case["case_id"] for case in groups[group]
                         if allocation[case["case_id"]] < capacities[case["case_id"]]]
            quota = min(share, budget, sum(capacities[key] - allocation[key] for key in available))
            while quota and available:
                case_share = max(1, quota // len(available))
                for key in available:
                    count = min(case_share, quota, capacities[key] - allocation[key])
                    allocation[key] += count
                    quota -= count
                    budget -= count
                available = [key for key in available if allocation[key] < capacities[key]]
            if available:
                next_active.append(group)
            if not budget:
                break
        active = next_active
    return allocation


def _sampling_plan(root, parent, bases, cases, part, budget, seed, checked):
    groups = defaultdict(list)
    selected = sorted((case for case in cases if case["split"] == part and case["state"] == "unchanged"
                       and case["hypothesis"] == "H0" and not case["sham_self_paste"]), key=lambda case: case["case_id"])
    if not selected:
        raise ValueError(f"MSDZip needs non-sham unchanged H0 cases in {part}")
    # Read support only for planning. Test and H1 RGB are never opened.
    for case in selected:
        path, digest = _checked(root, parent, case["reference_support"], case["artifact_sha256"]["reference_support"])
        checked[case["reference_support"]] = digest
        with Image.open(path) as image:
            shape = np.asarray(image).shape
        support = _mask(path, shape)
        groups[case["building_id"], case["scenario_id"]].append({**case, "capacity": int(support.sum()) * 3})
    allocation = _allocate(groups, budget)
    manifest = []
    for case in selected:
        count = allocation[case["case_id"]]
        if not count:
            continue
        support_path = _inside(root, case["reference_support"])
        with Image.open(support_path) as image:
            support = np.asarray(image) == 255
        candidates = np.flatnonzero(np.repeat(support.reshape(-1), 3))
        rng = np.random.default_rng(int(_digest([seed, part, case["case_id"]])[:16], 16))
        positions = np.sort(rng.choice(candidates, size=count, replace=False)).tolist()
        manifest.append({"case_id": case["case_id"], "base_id": case["base_id"], "building_id": case["building_id"],
                         "scenario_id": case["scenario_id"], "split": part, "target_byte_offsets": positions})
    return manifest


def _samples(root, parent, bases, cases, manifest, representation, timesteps, checked):
    by_base = {base["base_id"]: base for base in bases}
    by_case = {case["case_id"]: case for case in cases}
    windows, labels, indices = [], [], []
    for sample in manifest:
        case = by_case[sample["case_id"]]
        reference, source, support = _rgb_inputs(root, parent, by_base[case["base_id"]], case, checked)
        positions = np.asarray(sample["target_byte_offsets"], np.int64)
        contexts, targets = _windows(residual_bytes(reference, source, representation), support, positions, timesteps)
        windows.append(contexts)
        labels.append(targets)
        indices.append(positions)
    if not windows:
        raise ValueError("Byte budget selected no H0 training or validation targets")
    return np.concatenate(windows), np.concatenate(labels), np.concatenate(indices)


def _epoch(model, samples, device, window_groups, optimizer=None, rng=None):
    torch = _torch()
    total, count = 0., 0
    model.train(optimizer is not None)
    contexts, labels, positions = samples
    started = time.perf_counter()
    for step, (batch, targets, indices) in enumerate(_batch_groups(contexts, labels, positions, model.batchsize, window_groups, rng), 1):
        active = torch.as_tensor(indices >= 0, device=device)
        x = torch.as_tensor(batch, device=device)
        y = torch.as_tensor(targets, device=device)
        with _numerics(device), torch.set_grad_enabled(optimizer is not None):
            logits = _predict_groups(model, x)
            losses = torch.nn.functional.cross_entropy(logits[active], y[active], reduction="none")
            loss = losses.mean()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total += float(losses.detach().sum().cpu()) / math.log(2)
        count += int(active.sum())
        if step % 100 == 0:
            elapsed = time.perf_counter() - started
            print(f"  {'train' if optimizer else 'val'} grouped batches={step}; bytes={count}; bpb={total / count:.5f}; bytes/s={count / elapsed:.1f}", flush=True)
    return total / count


def train_msdzip_h0(dataset_run, out, representations=("abs", "mod256"), device="cpu", epochs=5,
                    max_train_bytes=2000000, max_val_bytes=200000, model_batch_size=32,
                    timesteps=16, hidden_dim=256, ffn_dim=4096, vocab_dim=16, lr=.001,
                    seed=42, max_bases_per_split=0, trust_checkpoint=False, window_groups=16):
    """Fresh original predictors; H0 train fitting and H0 val checkpoint selection."""
    representations = tuple(representations)
    if not representations or len(set(representations)) != len(representations) or any(rep not in {"abs", "mod256"} for rep in representations):
        raise ValueError("representations must contain unique abs/mod256 names")
    for name, value in (("epochs", epochs), ("max_train_bytes", max_train_bytes), ("max_val_bytes", max_val_bytes), ("window_groups", window_groups)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(max_bases_per_split) is not int or max_bases_per_split < 0 or type(seed) is not int:
        raise ValueError("Base budget must be nonnegative and seed an integer")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be finite and positive")
    args = _model_args(model_batch_size, timesteps, hidden_dim, ffn_dim, vocab_dim)
    torch = _torch()
    target_device = _device(device)
    root, parent, index, split, fingerprints = _parent(dataset_run)
    signatures = {(state, scenario["id"]) for state in parent["config"]["states"] for scenario in parent["config"]["scenarios"]}
    bases, cases = _select(index, split, max_bases_per_split, signatures)
    checked = {}
    plans = {part: _sampling_plan(root, parent, bases, cases, part, budget, seed, checked)
             for part, budget in (("train", max_train_bytes), ("val", max_val_bytes))}
    sampling = {"algorithm": "equal building/scenario cells, then cases; redistribute exhausted capacities; sample without replacement",
                "seed": seed, "requested_bytes": {"train": max_train_bytes, "val": max_val_bytes},
                "selected_bytes": {part: sum(len(item["target_byte_offsets"]) for item in plan) for part, plan in plans.items()},
                "same_target_identities_across_representations": True, "partitions": plans}
    sampling_digest = _digest(sampling)
    out = new_directory(out)
    config = {"dataset_run": str(root), "input_sha256": fingerprints, "representations": list(representations),
              "model_args": args, "epochs": epochs, "lr": lr, "seed": seed, "device": str(target_device),
              "max_bases_per_split": max_bases_per_split, "protocol": PROTOCOL,
              "sampling_sha256": sampling_digest, "source_sha256": SOURCE_SHA256,
              "trust_checkpoint": bool(trust_checkpoint), "window_groups": window_groups,
              "execution_backend": "torch.vmap" if hasattr(torch, "vmap") else "serial original forwards"}
    record = run_record("msdzip_h0_training", config)
    write_json(out / "run.json", record)
    try:
        shutil.copyfile(root / "split.json", out / "2026-10-04_reviewed_split.json")
        write_json(out / "2026-10-04_sampling.json", sampling)
        results = {}
        for representation in representations:
            samples = {part: _samples(root, parent, bases, cases, plan, representation, timesteps, checked)
                       for part, plan in plans.items()}
            torch.manual_seed(seed)
            if target_device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            model = _new_model(args, target_device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            best, history = math.inf, []
            checkpoint = out / f"2026-10-04_msdzip_h0_{representation}.pt"
            metadata = {"schema_version": 1, "kind": "original_msdzip_h0_frozen", "representation": representation,
                        "source_sha256": SOURCE_SHA256, "source_commit": SOURCE_COMMIT, "model_args": args,
                        "input_sha256": fingerprints, "reviewed_split": split, "protocol": PROTOCOL,
                        "sampling_sha256": sampling_digest, "sampling": {key: value for key, value in sampling.items() if key != "partitions"},
                        "buildings": {part: sorted({item["building_id"] for item in plan}) for part, plan in plans.items()},
                        "training_case_ids": [item["case_id"] for item in plans["train"]],
                        "validation_case_ids": [item["case_id"] for item in plans["val"]],
                        "selection": "minimum frozen-weight non-sham H0 validation NLL; no H1/test RGB or labels",
                        "lr": lr, "seed": seed, "epochs_requested": epochs}
            metadata.update(window_groups=window_groups, execution_backend=config["execution_backend"],
                            maximum_optimizer_targets=model_batch_size * window_groups)
            for epoch in range(1, epochs + 1):
                print(f"MSDZip {representation}: epoch {epoch}/{epochs}; H0 bytes train={len(samples['train'][1])}, val={len(samples['val'][1])}", flush=True)
                started = time.perf_counter()
                train_bpb = _epoch(model, samples["train"], target_device, window_groups, optimizer, np.random.default_rng(seed + epoch))
                train_seconds = time.perf_counter() - started
                val_started = time.perf_counter()
                val_bpb = _epoch(model, samples["val"], target_device, window_groups)
                val_seconds = time.perf_counter() - val_started
                history.append({"epoch": epoch, "train_bpb": train_bpb, "val_bpb": val_bpb,
                                "train_seconds": train_seconds, "val_seconds": val_seconds,
                                "train_bytes_per_second": len(samples["train"][1]) / train_seconds,
                                "val_bytes_per_second": len(samples["val"][1]) / val_seconds})
                write_json(out / f"2026-10-04_training_history_{representation}.json", history)
                print(f"MSDZip {representation}: train_bpb={train_bpb:.6f}; val_bpb={val_bpb:.6f}; train bytes/s={history[-1]['train_bytes_per_second']:.1f}; val bytes/s={history[-1]['val_bytes_per_second']:.1f}", flush=True)
                if not math.isfinite(train_bpb) or not math.isfinite(val_bpb):
                    raise ValueError("MSDZip training produced non-finite byte loss")
                if val_bpb < best:
                    best = val_bpb
                    best_metadata = {**metadata, "best_epoch": epoch, "best_val_bpb": val_bpb}
                    state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
                    temporary = checkpoint.with_suffix(".tmp")
                    torch.save({"metadata": best_metadata, "state_dict": state}, temporary)
                    temporary.replace(checkpoint)
            results[representation] = {"checkpoint_path": checkpoint.name, "checkpoint_sha256": sha256(checkpoint),
                                       "best_val_bpb": best, "best_epoch": best_metadata["best_epoch"], "history": history,
                                       "metadata": best_metadata}
            del samples, optimizer, model
            if target_device.type == "cuda":
                torch.cuda.empty_cache()
        for relative, digest in checked.items():
            if sha256(_inside(root, relative)) != digest:
                raise ValueError(f"Selected H0 artifact changed during training: {relative}")
        if _parent(root)[-1] != fingerprints:
            raise ValueError("Parent provenance changed during MSDZip training")
        summary = {"status": "completed_exploratory", "scope": "H0 residual surprise; original MSDZip predictor; no full H0/H1 code comparison",
                   "input_sha256": fingerprints, "model_args": args, "protocol": PROTOCOL, "results": results,
                   "sampling_sha256": sampling_digest, "sampled_bytes": sampling["selected_bytes"],
                   "selected_artifact_sha256": checked, "buildings": results[representations[0]]["metadata"]["buildings"]}
        write_json(out / "2026-10-04_msdzip_h0_summary.json", summary)
        record["summary"] = summary
        finish_record(out, record, summary["status"])
        return summary
    except (Exception, KeyboardInterrupt) as exc:
        finish_record(out, record, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", str(exc))
        raise


class MSDZipScorer:
    """Load once; frozen original byte probabilities on each independent case."""

    def __init__(self, checkpoint_path, device="cpu", dataset_fingerprint=None, trust_checkpoint=False, representation=None):
        torch = _torch()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.device = _device(device)
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError as exc:
            if not trust_checkpoint:
                raise RuntimeError("This PyTorch lacks weights_only loading; use explicit trust_checkpoint for a trusted checkpoint") from exc
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or set(checkpoint) != {"metadata", "state_dict"}:
            raise ValueError("MSDZip requires its H0 training checkpoint with metadata")
        metadata = checkpoint["metadata"]
        if (metadata.get("schema_version") != 1 or metadata.get("kind") != "original_msdzip_h0_frozen"
                or metadata.get("representation") not in {"abs", "mod256"}
                or metadata.get("source_sha256") != SOURCE_SHA256 or metadata.get("source_commit") != SOURCE_COMMIT
                or metadata.get("protocol") != PROTOCOL):
            raise ValueError("MSDZip checkpoint representation, source, or scoring protocol disagrees")
        if representation is not None and representation != metadata["representation"]:
            raise ValueError("MSDZip checkpoint representation differs from requested method")
        split = metadata.get("reviewed_split", {})
        fingerprint = metadata.get("input_sha256", {})
        if (split.get("mode") != "reviewed" or split.get("development_only") is not False
                or set(fingerprint) != {"parent_run", "parent_summary", "parent_index", "split", "config"}):
            raise ValueError("MSDZip checkpoint lacks reviewed split/dataset provenance")
        buildings = metadata.get("buildings", {})
        if (set(buildings) != {"train", "val"} or not buildings["train"] or not buildings["val"]
                or set(buildings["train"]) & set(buildings["val"])
                or any(split["building_assignments"].get(building) != part for part, values in buildings.items() for building in values)):
            raise ValueError("MSDZip checkpoint training buildings disagree with reviewed split")
        if type(metadata.get("window_groups")) is not int or metadata["window_groups"] < 1:
            raise ValueError("MSDZip checkpoint lacks its fixed window group count")
        if dataset_fingerprint is not None and dataset_fingerprint != fingerprint:
            raise ValueError("MSDZip checkpoint dataset/split fingerprint differs from benchmark")
        self.model = _new_model(metadata["model_args"], self.device)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.metadata = {**metadata, "checkpoint_path": str(self.checkpoint_path), "checkpoint_sha256": sha256(self.checkpoint_path),
                         "checkpoint_dataset_verified": dataset_fingerprint is not None, "device": str(self.device)}
        self.raw_scores = None

    def __call__(self, reference, source, support):
        torch = _torch()
        residual = residual_bytes(reference, source, self.metadata["representation"])
        if support.dtype != bool or support.shape != reference.shape[:2] or not support.any():
            raise ValueError("MSDZip requires same-grid boolean geometric support")
        positions = np.flatnonzero(np.repeat(support.reshape(-1), 3))
        contexts, targets = _windows(residual, support, positions, self.model.timesteps)
        byte_bits = np.full(residual.size, np.nan, dtype=np.float32)
        with torch.inference_mode():
            for batch, labels, indices in _batch_groups(contexts, targets, positions, self.model.batchsize, self.metadata["window_groups"]):
                active = indices >= 0
                active_tensor = torch.as_tensor(active, device=self.device)
                logits = _predict_groups(self.model, torch.as_tensor(batch, device=self.device))
                bits = torch.nn.functional.cross_entropy(logits[active_tensor], torch.as_tensor(labels[active], device=self.device), reduction="none") / math.log(2)
                byte_bits[positions[indices[active]]] = bits.cpu().numpy().astype(np.float32)
        self.model.last = []
        self.raw_scores = byte_bits.reshape(residual.shape).mean(axis=-1).astype(np.float32)
        scores = (-np.expm1(-self.raw_scores / np.float32(8))).astype(np.float32)
        if (not np.isfinite(scores[support]).all() or not np.isnan(scores[~support]).all()
                or np.any(scores[support] < 0) or np.any(scores[support] > 1)):
            raise ValueError("MSDZip produced an invalid native-grid score map")
        return scores
