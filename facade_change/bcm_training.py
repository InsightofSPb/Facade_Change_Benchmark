"""Bounded H0-only adaptation of the original monochrome BCM-Net network."""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from .codec_training import sample_h0_tiles
from .benchmark_subset import validate_input_hashes
from .hypothesis_benchmark import _select
from .io import finish_record, new_directory, read_json, run_record, sha256, write_json
from .methods.base import integer_option


def validate_bcm_training_run(training_run, dataset_fingerprint=None):
    """Verify a completed local pilot without importing Torch or reading RGB."""
    run = Path(training_run).expanduser().resolve()
    record = read_json(run / "run.json")
    if (not isinstance(record, dict) or record.get("schema_version") != 1
            or record.get("kind") != "bcm_h0_training"
            or record.get("status") != "completed_exploratory"):
        raise ValueError("BCM requires a completed H0 training run with schema version 1")
    config = record.get("config")
    if not isinstance(config, dict):
        raise ValueError("BCM training configuration is missing")
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False,
                                      allow_nan=False).encode()).hexdigest()
    if digest != record.get("config_sha256"):
        raise ValueError("BCM training configuration hash changed")
    if (config.get("tile_size") != 32 or config.get("qp") != 37
            or config.get("channels") != ["R", "G", "B"]):
        raise ValueError("BCM comparison requires 32-pixel RGB tiles and fixed VTM QP 37")
    validate_input_hashes(config.get("input_sha256"), config.get("input_sha256"))
    if dataset_fingerprint is not None and config.get("input_sha256") != dataset_fingerprint:
        raise ValueError("BCM H0 checkpoint belongs to a different parent dataset fingerprint")
    artifacts = record.get("artifact_sha256")
    if not isinstance(artifacts, dict):
        raise ValueError("BCM training artifact hashes are missing")
    for relative in ("summary.json", "sampling.json", "history.json", "base_reconstruction.json", "model.pth"):
        path = (run / relative).resolve()
        if run not in path.parents:
            raise ValueError(f"BCM training artifact escapes its run: {relative}")
        if not path.is_file() or artifacts.get(relative) != sha256(path):
            raise ValueError(f"BCM training artifact missing or hash changed: {relative}")
    summary = read_json(run / "summary.json")
    if (not isinstance(summary, dict) or summary.get("status") != "completed_exploratory"
            or summary.get("checkpoint") != "model.pth"
            or summary.get("channels") != config["channels"]):
        raise ValueError("BCM training summary disagrees with the completed RGB checkpoint")
    return run, record, summary


def _rgb_tile_pairs(root, parent, bases, cases, plan, tile_size, checked):
    """Preserve integer RGB A/B; only geometric support affects the padding."""
    h0 = importlib.import_module(".2026-10-04_msdzip_h0", __package__)
    by_base = {row["base_id"]: row for row in bases}
    by_case = {row["case_id"]: row for row in cases}
    pairs, cached_id = [], None
    for item in plan:
        case = by_case[item["case_id"]]
        if case["case_id"] != cached_id:
            a, b, support = h0._rgb_inputs(root, parent, by_base[case["base_id"]], case, checked)
            a, b = a.copy(), b.copy()
            a[~support], b[~support] = 0, 0
            cached_id = case["case_id"]
        pair = []
        for image in (a, b):
            tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            crop = image[item["row"]:item["row"]+tile_size, item["col"]:item["col"]+tile_size]
            tile[:crop.shape[0], :crop.shape[1]] = crop
            pair.append(tile)
        pairs.append(pair)
    return np.asarray(pairs, dtype=np.uint8)


def _tensor_channels(array, target):
    import torch
    # The author's model is monochrome: channels are independent samples,
    # never an invented three-channel convolution or color conversion.
    return torch.from_numpy(np.ascontiguousarray(array)).permute(0, 3, 1, 2).reshape(
        -1, 1, array.shape[1], array.shape[2]).to(device=target, dtype=torch.float32)


def train_bcm_h0(dataset_run, out, source_root, vtm_encoder, vtm_decoder, vtm_config,
                 init_checkpoint=None, vtm_scc_config=None, device="cpu", epochs=3,
                 max_train_patches=160, max_val_patches=16, batch_size=1,
                 tile_size=32, lr=1e-4, seed=42, trust_checkpoint=False):
    import torch
    from tqdm import tqdm
    from .methods.bcm_net import BCMCodec
    from .methods.bcm_training_core import bcm_nll

    epochs = integer_option(epochs, "epochs", 1)
    seed = integer_option(seed, "seed", 0)
    batch_size = integer_option(batch_size, "batch_size", 1)
    tile_size = integer_option(tile_size, "tile_size", 32)
    if tile_size != 32:
        raise ValueError("The bounded BCM comparison uses exactly 32-pixel tiles")
    budgets = {"train": integer_option(max_train_patches, "max_train_patches", 1),
               "val": integer_option(max_val_patches, "max_val_patches", 1)}
    if isinstance(lr, bool) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("BCM device must be cpu or cuda")
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA is unavailable")
        free, total = torch.cuda.mem_get_info(target)
        print(f"BCM-Net: free {free/2**30:.2f}/{total/2**30:.2f} GiB; other processes unchanged", flush=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.set_num_threads(min(4, torch.get_num_threads()))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    h0 = importlib.import_module(".2026-10-04_msdzip_h0", __package__)
    root, parent, index, split, fingerprints = h0._parent(dataset_run)
    output = Path(out).expanduser().resolve()
    source_directory = Path(source_root).expanduser().resolve()
    if output == root or root in output.parents or output == source_directory or source_directory in output.parents:
        raise ValueError("BCM training output must be outside the immutable dataset and author source directories")
    signatures = {(state, scenario["id"]) for state in parent["config"]["states"]
                  for scenario in parent["config"]["scenarios"]}
    bases, cases = _select(index, split, 0, signatures)
    checked = {}
    plans = sample_h0_tiles(root, parent, bases, cases, tile_size, budgets, seed, checked)
    codec = BCMCodec(source_root, checkpoint=init_checkpoint, device=str(target),
                     vtm_encoder=vtm_encoder, vtm_decoder=vtm_decoder,
                     vtm_config=vtm_config, vtm_scc_config=vtm_scc_config,
                     qp=37, cost_mode="bitstream", seed=seed,
                     trust_checkpoint=trust_checkpoint)
    network = codec.network.to(target).requires_grad_(True)
    config = {"dataset_run": str(root), "input_sha256": fingerprints,
        "source_root": str(Path(source_root).expanduser().resolve()), "author": codec.metadata,
        "init_checkpoint": str(Path(init_checkpoint).expanduser().resolve()) if init_checkpoint else None,
        "vtm_encoder": str(Path(vtm_encoder).expanduser().resolve()),
        "vtm_decoder": str(Path(vtm_decoder).expanduser().resolve()),
        "vtm_config": str(Path(vtm_config).expanduser().resolve()),
        "vtm_scc_config": str(Path(vtm_scc_config).expanduser().resolve()) if vtm_scc_config else None,
        "qp": 37, "channels": ["R", "G", "B"], "device": str(target), "epochs": epochs,
        "batch_size": batch_size, "tile_size": tile_size, "lr": lr, "seed": seed,
        "trust_checkpoint": bool(trust_checkpoint), "patch_budget": budgets,
        "protocol": {
            "architecture": "pinned original 8-bit monochrome BCM-Net; shared weights over separate RGB channels",
            "training": "teacher-forced exact signed-residual logistic NLL; Adam betas 0.9/0.999; lr times 0.75 each 20 epochs",
            "adaptation": "bounded facade-disjoint H0 pilot; explicit initial checkpoint or fresh weights; not medical training reproduction",
            "base": "author VTM two-frame random-access RGB-channel base, fixed QP 37; full scorer charges base and lossless residual",
            "reference": "exact previous RGB A channel available at the lossless decoder; VTM base A is not substituted",
            "input": "native uint8 RGB, signed B minus VTM base B; original A reference; unsupported pixels and padding zero",
            "loss_domain": "all zero-padded tile pixels; geometric support fractions recorded; no labels",
            "checkpoint_selection": "residual theoretical bits per RGB byte, mean within building then equal buildings; fixed VTM bases",
            "augmentation": "none in this bounded pilot",
            "test": "no TEST bytes or labels loaded for fitting or checkpoint selection"}}
    out = new_directory(out)
    record = run_record("bcm_h0_training", config)
    write_json(out / "run.json", record)
    try:
        arrays, bases_b, base_report = {}, {}, {}
        for part, plan in plans.items():
            arrays[part] = _rgb_tile_pairs(root, parent, bases, cases, plan, tile_size, checked)
            reconstructed, rows = [], []
            for pair, item in tqdm(list(zip(arrays[part], plan)), desc=f"BCM VTM {part}",
                                   unit="tile", dynamic_ncols=True):
                base_a, base_b, stats = codec.base_reconstruct(pair[0], pair[1])
                if any(value.dtype != np.uint8 or value.shape != pair[0].shape
                       for value in (base_a, base_b)):
                    raise ValueError("BCM VTM base reconstruction must preserve the uint8 RGB tile grid")
                reconstructed.append(base_b)
                rows.append({**item, "base_stats": stats,
                    "reference_rgb_sha256": hashlib.sha256(pair[0].tobytes()).hexdigest(),
                    "source_rgb_sha256": hashlib.sha256(pair[1].tobytes()).hexdigest(),
                    "base_a_rgb_sha256": hashlib.sha256(base_a.tobytes()).hexdigest(),
                    "base_b_rgb_sha256": hashlib.sha256(base_b.tobytes()).hexdigest()})
            bases_b[part] = np.stack(reconstructed)
            base_report[part] = rows
        write_json(out / "sampling.json", {"partitions": plans, "selected_rgb_support_sha256": checked,
            "algorithm": "equal building/scenario cells; SHA256-ranked nonoverlapping H0 tile candidates"})
        write_json(out / "base_reconstruction.json", base_report)
        optimizer = torch.optim.Adam(network.parameters(), lr=lr, betas=(.9, .999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=.75)
        generator = np.random.default_rng(seed)
        best, history = float("inf"), []
        for epoch in range(1, epochs+1):
            network.train()
            total, count = 0., 0
            order = generator.permutation(len(arrays["train"]))
            with tqdm(total=len(order), desc=f"BCM train {epoch}/{epochs}", unit="tile", dynamic_ncols=True) as progress:
                for start in range(0, len(order), batch_size):
                    ids = order[start:start+batch_size]
                    base = _tensor_channels(bases_b["train"][ids], target)
                    reference = _tensor_channels(arrays["train"][ids, 0], target)
                    source = _tensor_channels(arrays["train"][ids, 1], target)
                    optimizer.zero_grad(set_to_none=True)
                    bits = bcm_nll(network, source-base, base, reference=reference)
                    loss = bits.mean() / (tile_size * tile_size)
                    if not torch.isfinite(loss):
                        raise RuntimeError("BCM training objective became nonfinite")
                    loss.backward()
                    optimizer.step()
                    total += float(loss.detach().cpu()) * len(ids)
                    count += len(ids)
                    progress.update(len(ids))
                    progress.set_postfix(bpb=f"{total/count:.5f}", refresh=False)
            network.eval()
            groups = defaultdict(list)
            with torch.no_grad():
                for pair, base, item in tqdm(list(zip(arrays["val"], bases_b["val"], plans["val"])),
                                             desc="BCM val", unit="tile", dynamic_ncols=True):
                    base_t = _tensor_channels(base[None], target)
                    reference = _tensor_channels(pair[0][None], target)
                    source = _tensor_channels(pair[1][None], target)
                    value = float((bcm_nll(network, source-base_t, base_t,
                                           reference=reference).mean() / (tile_size*tile_size)).cpu())
                    if not math.isfinite(value):
                        raise RuntimeError("BCM validation objective became nonfinite")
                    groups[item["building_id"]].append(value)
            val = float(np.mean([np.mean(values) for values in groups.values()]))
            history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                            "train_residual_theoretical_bpb": total/count,
                            "val_building_macro_residual_theoretical_bpb": val})
            if val < best:
                best = val
                torch.save({key: value.detach().cpu() for key, value in network.state_dict().items()}, out / "model.pth")
            write_json(out / "history.json", history)
            print(f"BCM-Net: val residual bpb={val:.6f}, best={best:.6f}", flush=True)
            scheduler.step()
        write_json(out / "summary.json", {"status": "completed_exploratory", "checkpoint": "model.pth",
            "channels": config["channels"], "selected_patches": {part: len(plan) for part, plan in plans.items()},
            "best_val_building_macro_residual_theoretical_bpb": best, "history": history,
            "scope": "H0 domain adaptation pilot; no bitstream roundtrip or change-detection accuracy claimed by training"})
        finish_record(out, record, "completed_exploratory")
    except BaseException as exc:
        finish_record(out, record, "failed", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        codec.close()
    return {"out": str(out), "selected_patches": {part: len(plan) for part, plan in plans.items()},
            "status": "completed_exploratory"}
