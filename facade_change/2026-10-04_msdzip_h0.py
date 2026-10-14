"""Original MSDZip predictor: train on H0 residuals, then freeze byte NLL.

The upstream class is untouched. This is an offline adapter, not its online
compression loop: every prediction recomputes the full preceding-byte window.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
from PIL import Image

from .hypothesis_benchmark import _checked, _inside, _mask, _select
from .io import finish_record, load_rgb, new_directory, read_json, run_record, sha256, sync_directory, write_json


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
    if record.get("kind") == "history_dataset":
        return _history_parent(root, record)
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


def _history_parent(root, record):
    """Read/audit history metadata without opening event or TEST artifacts."""
    if record.get("status") != "completed_exploratory":
        raise ValueError("MSDZip history training requires a completed history dataset")
    summary_path, summary_hash = _checked(root, record, "summary.json")
    index_path, index_hash = _checked(root, record, "manifest.json")
    split_path, split_hash = _checked(root, record, "split.json")
    config_path, config_hash = _checked(root, record, "config.json")
    index, split, config = read_json(index_path), read_json(split_path), read_json(config_path)
    if (index.get("kind") != "persistent_procedural_histories" or index.get("config") != config
            or record.get("config") != config or split.get("development_only", False)
            or split.get("mode", "reviewed") != "reviewed"):
        raise ValueError("MSDZip history metadata/config/reviewed split disagrees")
    bases, owners = {}, {key: {} for key in ("building_id", "view_id", "image_sha256")}
    for base in index["bases"]:
        identifier, building, part = base["base_id"], base["building_id"], base["split"]
        if (identifier in bases or not building or part not in {"train", "val", "test"}
                or split["building_assignments"].get(building) != part
                or base.get("source_to_reference") != np.eye(3).tolist()):
            raise ValueError("History base identity/geometry disagrees with reviewed split")
        for key, seen in owners.items():
            value = base.get(key)
            if value is not None and value in seen and seen[value] != part:
                raise ValueError(f"History {key} leakage across partitions")
            if value is not None:
                seen[value] = part
        bases[identifier] = base
    cases, sequence_ids, clean_bases = [], set(), set()
    for sequence in index["sequences"]:
        base = bases.get(sequence["base_id"])
        if (not base or sequence["sequence_id"] in sequence_ids
                or any(sequence.get(key) != base.get(key) for key in ("building_id", "split"))
                or sequence.get("arm") not in {"clean", "event"}):
            raise ValueError("History sequence identity disagrees with its base")
        sequence_ids.add(sequence["sequence_id"])
        if sequence["arm"] != "clean" or sequence["split"] == "test":
            continue
        if base["base_id"] in clean_bases:
            raise ValueError("History has duplicate clean arms")
        clean_bases.add(base["base_id"])
        frames = sequence["frames"]
        if len(frames) != 5 or [frame["frame"] for frame in frames] != list(range(1, 6)):
            raise ValueError("History H0 requires five chronological clean frames")
        for previous, current in zip(frames, frames[1:]):
            if current.get("event") != "none" or current.get("event_pixels") != 0:
                raise ValueError("Clean history contains a procedural event")
            # Per-frame support also excludes known procedural occlusion. It is
            # evaluation information, not geometry available to the byte model.
            support_path = base["support"]
            support_hash = record["artifact_sha256"][support_path]
            paths = {"reference_rgb": previous["paths"]["rgb"], "source_rgb": current["paths"]["rgb"],
                     "reference_support": support_path, "source_support": support_path}
            hashes = {"reference_rgb": previous["artifact_sha256"]["rgb"], "source_rgb": current["artifact_sha256"]["rgb"],
                      "reference_support": support_hash, "source_support": support_hash}
            cases.append({"case_id": f"{sequence['sequence_id']}-{previous['frame']}-{current['frame']}",
                          "base_id": base["base_id"], "building_id": base["building_id"], "split": base["split"],
                          "scenario_id": f"{previous['nuisance']['kind']}-{current['nuisance']['kind']}",
                          "state": "unchanged", "hypothesis": "H0", "sham_self_paste": False,
                          "history_clean_pair": True, "artifact_sha256": hashes, **paths})
    summary = read_json(summary_path)
    if summary.get("sequences") != len(index["sequences"]) or summary.get("selected_crops") != len(bases):
        raise ValueError("History summary counts disagree with manifest")
    # Histories are generated exclusively from metadata-reviewed buildings. Older
    # prepared split files contain assignments without these legacy H0 flags.
    split = {**split, "mode": "reviewed", "development_only": False}
    fingerprints = {"parent_run": sha256(root / "run.json"), "parent_summary": summary_hash,
                    "parent_index": index_hash, "split": split_hash, "config": config_hash}
    return root, record, {**index, "cases": cases}, split, fingerprints


def _rgb_inputs(root, parent, base, case, selected_hashes):
    paths = {}
    if case.get("history_clean_pair"):
        for key in ("reference_rgb", "source_rgb", "reference_support", "source_support"):
            paths[key], digest = _checked(root, parent, case[key], case["artifact_sha256"][key])
            selected_hashes[case[key]] = digest
        reference, opaque = load_rgb(paths["reference_rgb"])
        source, source_opaque = load_rgb(paths["source_rgb"])
        if reference.shape != source.shape:
            raise ValueError("History H0 RGB grids differ")
        support = _mask(paths["reference_support"], reference.shape[:2])
        source_support = _mask(paths["source_support"], reference.shape[:2])
        if not np.array_equal(support, source_support) or np.any(support & (~opaque | ~source_opaque)):
            raise ValueError("History H0 geometric support differs or includes nonopaque RGB")
        return reference, source, support
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


def _epoch(model, samples, device, window_groups, optimizer=None, rng=None, progress_description=None,
           progress_state=None, step_callback=None):
    torch = _torch()
    progress_state = progress_state if progress_state is not None else {}
    total, count = progress_state.get("total_bits", 0.), progress_state.get("count", 0)
    start_step = progress_state.get("step", 0)
    model.train(optimizer is not None)
    contexts, labels, positions = samples
    started = time.perf_counter()
    phase = "train" if optimizer is not None else "val"
    try:
        from tqdm import tqdm
    except ImportError:
        progress = nullcontext(None)
        print("tqdm unavailable; using text progress. Install with: python -m pip install 'tqdm>=4.66,<5'", flush=True)
    else:
        progress = tqdm(total=len(labels), desc=progress_description or phase,
                        unit="B", unit_scale=True, dynamic_ncols=True, mininterval=1)
    with progress as bar:
        if bar is not None:
            bar.update(count)
        for step, (batch, targets, indices) in enumerate(_batch_groups(contexts, labels, positions, model.batchsize, window_groups, rng), 1):
            if step <= start_step:
                continue
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
            processed = int(active.sum())
            count += processed
            progress_state.update(step=step, total_bits=total, count=count)
            if step_callback is not None:
                step_callback()
            if bar is not None:
                bar.set_postfix(bpb=f"{total / count:.5f}", refresh=False)
                bar.update(processed)
            elif step % 100 == 0:
                elapsed = time.perf_counter() - started
                print(f"  {phase} grouped batches={step}; bytes={count}; bpb={total / count:.5f}; bytes/s={count / elapsed:.1f}", flush=True)
    return total / count


def _atomic_torch_save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        _torch().save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def _load_checkpoint(path, trust_checkpoint):
    try:
        return _torch().load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        if not trust_checkpoint:
            raise RuntimeError("This PyTorch lacks weights_only loading; explicitly trust a known checkpoint") from exc
        return _torch().load(path, map_location="cpu")


def _rng_state(device):
    numpy_state = np.random.get_state()
    return {"torch": _torch().get_rng_state(), "python": random.getstate(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "cuda": _torch().cuda.get_rng_state_all() if device.type == "cuda" else []}


def _restore_rng(state, device):
    _torch().set_rng_state(state["torch"])
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    if device.type == "cuda":
        _torch().cuda.set_rng_state_all(state["cuda"])


def _training_events(out, config_sha256):
    """Recover only an incomplete final append, preserving its original bytes."""
    path = out / "events.jsonl"
    if not path.exists():
        return [], None
    data = path.read_bytes()
    lines, events, offset = data.splitlines(keepends=True), [], 0
    recovery = None
    for number, line in enumerate(lines):
        try:
            event = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            if number != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError("MSDZip event log corruption before an incomplete final append") from exc
            digest = hashlib.sha256(line).hexdigest()
            orphan = out / f"2026-10-14_msdzip_event_tail_{digest[:16]}.bin"
            if orphan.exists():
                if orphan.read_bytes() != line:
                    raise ValueError("MSDZip event recovery artifact hash collision")
            else:
                with orphan.open("xb") as stream:
                    stream.write(line)
                    stream.flush()
                    os.fsync(stream.fileno())
            with path.open("r+b") as stream:
                stream.truncate(offset)
                stream.flush()
                os.fsync(stream.fileno())
            recovery = {"orphan_path": orphan.name, "orphan_sha256": digest, "bytes": len(line)}
            break
        if (not isinstance(event, dict) or event.get("sequence") != len(events) + 1
                or event.get("run_id") != out.name or event.get("config_sha256") != config_sha256):
            raise ValueError("MSDZip event sequence/run/config identity is corrupt")
        events.append(event)
        offset += len(line)
        if not line.endswith(b"\n"):
            # A complete JSON object may have lost only its trailing newline.
            with path.open("ab") as stream:
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            recovery = {"newline_restored": True, "bytes": 0}
    return events, recovery


def _recorded_event_prefix(path, expected):
    """A kill after appending retry/reuse may precede the run.json update."""
    digest = hashlib.sha256()
    for line in path.read_bytes().splitlines(keepends=True):
        digest.update(line)
        if digest.hexdigest() == expected:
            return True
    return False


class _PauseTraining(Exception):
    """An explicitly requested smoke/recovery stop after durable optimizer steps."""


def train_msdzip_h0(dataset_run, out, representations=("abs", "mod256"), device="cpu", epochs=5,
                    max_train_bytes=2000000, max_val_bytes=200000, model_batch_size=32,
                    timesteps=16, hidden_dim=256, ffn_dim=4096, vocab_dim=16, lr=.001,
                    seed=42, max_bases_per_split=0, trust_checkpoint=False, window_groups=16,
                    resume=False, checkpoint_every_steps=100, max_steps_per_call=0):
    """Fit H0 predictors, with exact compatible resume from committed steps.

    History datasets use clean-arm adjacent observed RGB pairs. Existing damage
    in the base is preserved; clean does not mean a physically undamaged facade.
    max_steps_per_call is only a safe interruption budget, not a change of fit.
    """
    from .research.common import ExperimentRun

    representations = tuple(representations)
    if not representations or len(set(representations)) != len(representations) or any(rep not in {"abs", "mod256"} for rep in representations):
        raise ValueError("representations must contain unique abs/mod256 names")
    for name, value in (("epochs", epochs), ("max_train_bytes", max_train_bytes), ("max_val_bytes", max_val_bytes),
                        ("window_groups", window_groups), ("checkpoint_every_steps", checkpoint_every_steps)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (type(max_bases_per_split) is not int or max_bases_per_split < 0 or type(seed) is not int or seed < 0
            or type(max_steps_per_call) is not int or max_steps_per_call < 0):
        raise ValueError("Base/step budgets and seed must be nonnegative integers")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be finite and positive")
    args = _model_args(model_batch_size, timesteps, hidden_dim, ffn_dim, vocab_dim)
    if sha256(SOURCE_PATH) != SOURCE_SHA256:
        raise ValueError("Vendored original MSDZip source hash disagrees with pinned upstream")
    torch = _torch()
    root, parent, index, split, fingerprints = _parent(dataset_run)
    if parent["kind"] == "history_dataset":
        bases = []
        for part in ("train", "val"):
            candidates = sorted((base for base in index["bases"] if base["split"] == part),
                                key=lambda base: _digest(base["base_id"]))
            bases.extend(candidates[:max_bases_per_split] if max_bases_per_split else candidates)
        selected_bases = {base["base_id"] for base in bases}
        cases = [case for case in index["cases"] if case["base_id"] in selected_bases]
    else:
        signatures = {(state, scenario["id"]) for state in parent["config"]["states"] for scenario in parent["config"]["scenarios"]}
        bases, cases = _select(index, split, max_bases_per_split, signatures)
    checked = {}
    plans = {part: _sampling_plan(root, parent, bases, cases, part, budget, seed, checked)
             for part, budget in (("train", max_train_bytes), ("val", max_val_bytes))}
    # Fingerprint all selected model inputs before accepting a resume or a GPU.
    selected_ids = {item["case_id"] for plan in plans.values() for item in plan}
    for case in cases:
        if case["case_id"] not in selected_ids:
            continue
        for key in ("reference_rgb", "source_rgb", "reference_support", "source_support", "geometry"):
            if key in case:
                _, digest = _checked(root, parent, case[key], case["artifact_sha256"][key])
                checked[case[key]] = digest
    sampling = {"algorithm": "equal building/scenario cells, then cases; redistribute exhausted capacities; sample without replacement",
                "seed": seed, "requested_bytes": {"train": max_train_bytes, "val": max_val_bytes},
                "selected_bytes": {part: sum(len(item["target_byte_offsets"]) for item in plan) for part, plan in plans.items()},
                "same_target_identities_across_representations": True, "partitions": plans}
    sampling_digest = _digest(sampling)
    config = {"dataset_run": str(root), "input_sha256": fingerprints, "representations": list(representations),
              "model_args": args, "epochs": epochs, "lr": lr, "seed": seed, "device": str(torch.device(device)),
              "max_bases_per_split": max_bases_per_split, "protocol": PROTOCOL,
              "sampling_sha256": sampling_digest, "source_sha256": SOURCE_SHA256,
              "window_groups": window_groups, "checkpoint_every_steps": checkpoint_every_steps,
              "selected_artifact_sha256": checked,
              "adapter_source_sha256": {name: sha256(Path(__file__).parent / name) for name in
                                        (Path(__file__).name, "hypothesis_benchmark.py", "io.py")},
              "runtime": {"torch": str(torch.__version__), "numpy": str(np.__version__),
                          "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()},
              "training_dataset_kind": parent["kind"],
              "pair_policy": "adjacent observed clean-arm frames" if parent["kind"] == "history_dataset" else "unchanged non-sham H0 cases",
              "execution_backend": "torch.vmap" if hasattr(torch, "vmap") else "serial original forwards"}
    out = Path(out).expanduser().resolve()
    existing = out.exists()
    if existing:
        if not resume:
            raise FileExistsError(f"Training output exists; use resume=True with exactly compatible inputs: {out}")
        record = read_json(out / "run.json")
        if record.get("kind") != "msdzip_h0_training" or record.get("config") != config:
            raise ValueError("MSDZip resume config/data/source fingerprint is incompatible; use a new output directory")
        if read_json(out / "2026-10-04_sampling.json") != sampling:
            raise ValueError("MSDZip resume sampling changed")
        events, recovery = _training_events(out, record["config_sha256"])
        if record.get("status") != "running":
            for relative, expected in record.get("artifact_sha256", {}).items():
                if sha256(out / relative) != expected:
                    if relative == "events.jsonl" and _recorded_event_prefix(out / relative, expected):
                        continue
                    raise ValueError(f"Recorded MSDZip training artifact changed: {relative}")
    else:
        new_directory(out)
        record = run_record("msdzip_h0_training", config)
        write_json(out / "run.json", record)
        shutil.copyfile(root / "split.json", out / "2026-10-04_reviewed_split.json")
        write_json(out / "2026-10-04_sampling.json", sampling)
        events, recovery = [], None
    # Reuse the research ledger's schema/method rather than a second event API.
    run = ExperimentRun.__new__(ExperimentRun)
    run.out, run.record = out, record
    run.sequence = max((event["sequence"] for event in events), default=0)
    if recovery is not None:
        run.event("event_tail_recovered", **recovery)
    if existing and record.get("status") == "completed_exploratory":
        summary = read_json(out / "2026-10-04_msdzip_h0_summary.json")
        run.event("completed_reuse", summary_sha256=sha256(out / "2026-10-04_msdzip_h0_summary.json"))
        finish_record(out, record, "completed_exploratory")
        return summary
    run.event("retry" if existing else "started", previous_status=record["status"], max_steps_per_call=max_steps_per_call)
    record.pop("error", None)
    record["status"] = "running"
    write_json(out / "run.json", record)
    results, steps_this_call = {}, 0
    try:
        target_device = _device(device)
        for representation in representations:
            checkpoint = out / f"2026-10-04_msdzip_h0_{representation}.pt"
            resume_path = out / f"2026-10-14_msdzip_resume_{representation}.pt"
            saved = _load_checkpoint(resume_path, trust_checkpoint) if resume_path.exists() else None
            if saved is not None and (saved.get("kind") != "msdzip_training_resume" or saved.get("config_sha256") != _digest(config)
                                      or saved.get("representation") != representation):
                raise ValueError("MSDZip optimizer checkpoint config/representation differs")
            if saved is not None and saved["status"] == "completed":
                if sha256(checkpoint) != saved["result"]["checkpoint_sha256"]:
                    raise ValueError("Completed MSDZip representation checkpoint changed")
                results[representation] = saved["result"]
                run.event("representation_reuse", representation=representation)
                continue
            samples = {part: _samples(root, parent, bases, cases, plan, representation, timesteps, checked)
                       for part, plan in plans.items()}
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % 2**32)
            if target_device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            model = _new_model(args, target_device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            metadata = {"schema_version": 1, "kind": "original_msdzip_h0_frozen", "representation": representation,
                        "source_sha256": SOURCE_SHA256, "source_commit": SOURCE_COMMIT, "model_args": args,
                        "input_sha256": fingerprints, "reviewed_split": split, "protocol": PROTOCOL,
                        "sampling_sha256": sampling_digest, "sampling": {key: value for key, value in sampling.items() if key != "partitions"},
                        "buildings": {part: sorted({item["building_id"] for item in plan}) for part, plan in plans.items()},
                        "training_case_ids": [item["case_id"] for item in plans["train"]],
                        "validation_case_ids": [item["case_id"] for item in plans["val"]],
                        "selection": "minimum frozen-weight non-sham H0 validation NLL; no H1/test RGB or labels",
                        "training_dataset_kind": parent["kind"], "pair_policy": config["pair_policy"],
                        "lr": lr, "seed": seed, "epochs_requested": epochs,
                        "window_groups": window_groups, "execution_backend": config["execution_backend"],
                        "maximum_optimizer_targets": model_batch_size * window_groups}
            state = {"epoch": 1, "global_step": 0, "progress": {}, "history": [], "best": None,
                     "train_seconds": 0., "status": "running"}
            if saved is not None:
                model.load_state_dict(saved["state_dict"], strict=True)
                optimizer.load_state_dict(saved["optimizer"])
                state = saved["training"]
                _restore_rng(saved["rng"], target_device)
                if state["best"] is not None:
                    _atomic_torch_save(checkpoint, state["best"])
                run.event("checkpoint_resume", representation=representation, epoch=state["epoch"],
                          step=state["progress"].get("step", 0), global_step=state["global_step"])

            def save_progress(status="running", result=None):
                payload = {"kind": "msdzip_training_resume", "schema_version": 1,
                           "config_sha256": _digest(config), "representation": representation, "status": status,
                           "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "training": state, "rng": _rng_state(target_device), "result": result}
                _atomic_torch_save(resume_path, payload)

            def after_step():
                nonlocal steps_this_call
                steps_this_call += 1
                state["global_step"] += 1
                state["train_seconds"] = prior_train_seconds + time.perf_counter() - started
                pause = bool(max_steps_per_call and steps_this_call >= max_steps_per_call)
                if state["global_step"] % checkpoint_every_steps == 0 or pause:
                    save_progress()
                if pause:
                    raise _PauseTraining("Requested optimizer-step budget reached")

            if saved is None:
                save_progress()
            for epoch in range(state["epoch"], epochs + 1):
                print(f"MSDZip {representation}: epoch {epoch}/{epochs}; resume batch={state['progress'].get('step', 0)}", flush=True)
                prior_train_seconds = state["train_seconds"]
                started = time.perf_counter()
                train_bpb = _epoch(model, samples["train"], target_device, window_groups, optimizer,
                                   np.random.default_rng(seed + epoch), f"MSDZip {representation} train {epoch}/{epochs}",
                                   progress_state=state["progress"], step_callback=after_step)
                train_seconds = prior_train_seconds + time.perf_counter() - started
                # Persist a completed train pass before validation, which is safe
                # to replay after interruption because it has no optimizer steps.
                state["train_seconds"] = train_seconds
                save_progress()
                val_started = time.perf_counter()
                val_bpb = _epoch(model, samples["val"], target_device, window_groups,
                                 progress_description=f"MSDZip {representation} val {epoch}/{epochs}")
                val_seconds = time.perf_counter() - val_started
                if not math.isfinite(train_bpb) or not math.isfinite(val_bpb):
                    raise ValueError("MSDZip training produced non-finite byte loss")
                state["history"].append({"epoch": epoch, "train_bpb": train_bpb, "val_bpb": val_bpb,
                                         "train_seconds": train_seconds, "val_seconds": val_seconds,
                                         "train_bytes_per_second": len(samples["train"][1]) / max(train_seconds, 1e-9),
                                         "val_bytes_per_second": len(samples["val"][1]) / max(val_seconds, 1e-9)})
                if state["best"] is None or val_bpb < state["best"]["metadata"]["best_val_bpb"]:
                    state["best"] = {"metadata": {**metadata, "best_epoch": epoch, "best_val_bpb": val_bpb},
                                     "state_dict": {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}}
                    _atomic_torch_save(checkpoint, state["best"])
                state.update(epoch=epoch + 1, progress={}, train_seconds=0.)
                save_progress()
                write_json(out / f"2026-10-04_training_history_{representation}.json", state["history"])
                run.event("epoch_completed", representation=representation, epoch=epoch, train_bpb=train_bpb,
                          val_bpb=val_bpb, global_step=state["global_step"], resume_sha256=sha256(resume_path))
            write_json(out / f"2026-10-04_training_history_{representation}.json", state["history"])
            best_metadata = state["best"]["metadata"]
            results[representation] = {"checkpoint_path": checkpoint.name, "checkpoint_sha256": sha256(checkpoint),
                                       "best_val_bpb": best_metadata["best_val_bpb"], "best_epoch": best_metadata["best_epoch"],
                                       "history": state["history"], "metadata": best_metadata}
            save_progress("completed", results[representation])
            run.event("representation_completed", representation=representation, checkpoint_sha256=sha256(checkpoint))
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
                   "selected_artifact_sha256": checked, "buildings": results[representations[0]]["metadata"]["buildings"],
                   "training_dataset_kind": parent["kind"], "pair_policy": config["pair_policy"], "test_rgb_loaded": False}
        write_json(out / "2026-10-04_msdzip_h0_summary.json", summary)
        record["summary"] = summary
        run.event("completed", summary_sha256=sha256(out / "2026-10-04_msdzip_h0_summary.json"))
        finish_record(out, record, summary["status"])
        return summary
    except _PauseTraining as exc:
        run.event("interrupted", reason=str(exc), optimizer_steps_this_call=steps_this_call)
        finish_record(out, record, "interrupted", str(exc))
        return {"status": "interrupted", "out": str(out), "optimizer_steps_this_call": steps_this_call,
                "resume": True, "results": results}
    except (Exception, KeyboardInterrupt) as exc:
        # Never snapshot a potentially half-updated optimizer on a signal/error.
        # The last atomically committed step is authoritative on the next call.
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        run.event(status, error=str(exc))
        finish_record(out, record, status, str(exc))
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
