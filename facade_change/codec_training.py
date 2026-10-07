"""Bounded H0-only adaptation of original ArIB SIG/INS networks."""
from __future__ import annotations

import hashlib
import importlib
import math
import random
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from .hypothesis_benchmark import _select
from .io import finish_record, new_directory, run_record, write_json
from .methods.base import integer_option
from .methods.compression import rgb_residual


def sample_h0_tiles(root, parent, bases, cases, tile_size, budgets, seed, checked):
    """Round-robin building/scenario cells; deterministic candidates; no labels."""
    h0 = importlib.import_module(".2026-10-04_msdzip_h0", __package__)
    by_id = {base["base_id"]: base for base in bases}
    cells = {part: defaultdict(list) for part in budgets}
    for case in sorted(cases, key=lambda row: row["case_id"]):
        part = case["split"]
        if (part not in budgets or case["state"] != "unchanged" or case["hypothesis"] != "H0"
                or case["sham_self_paste"]):
            continue
        _, _, support = h0._rgb_inputs(root, parent, by_id[case["base_id"]], case, checked)
        for row in range(0, support.shape[0], tile_size):
            for col in range(0, support.shape[1], tile_size):
                if not support[row:row+tile_size, col:col+tile_size].any():
                    continue
                identity = f"{seed}/{case['case_id']}/{row}/{col}"
                cells[part][case["building_id"], case["scenario_id"]].append((
                    hashlib.sha256(identity.encode()).hexdigest(),
                    {"case_id": case["case_id"], "building_id": case["building_id"],
                     "base_id": case["base_id"], "split": part, "row": row, "col": col,
                     "support_fraction": float(support[row:row+tile_size, col:col+tile_size].sum()
                                               / (tile_size * tile_size))}))
    plans = {}
    for part, groups in cells.items():
        def cell_rank(cell):
            building, scenario = cell
            return (hashlib.sha256(f"{seed}/scenario/{scenario}".encode()).hexdigest(),
                    hashlib.sha256(f"{seed}/building/{building}".encode()).hexdigest())
        queues = [deque(item[1] for item in sorted(groups[cell], key=lambda item: item[0]))
                  for cell in sorted(groups, key=cell_rank)]
        selected = []
        while queues and len(selected) < budgets[part]:
            active = []
            for queue in queues:
                if queue and len(selected) < budgets[part]:
                    selected.append(queue.popleft())
                if queue:
                    active.append(queue)
            queues = active
        if not selected:
            raise ValueError(f"No eligible non-sham H0 {part} tiles")
        plans[part] = selected
    return plans


def _tiles(root, parent, bases, cases, plan, representation, tile_size, checked):
    h0 = importlib.import_module(".2026-10-04_msdzip_h0", __package__)
    by_base, by_case = {row["base_id"]: row for row in bases}, {row["case_id"]: row for row in cases}
    result, cached_id, residual = [], None, None
    for item in plan:
        case = by_case[item["case_id"]]
        if case["case_id"] != cached_id:
            a, b, support = h0._rgb_inputs(root, parent, by_base[case["base_id"]], case, checked)
            residual = rgb_residual(a, b, representation)
            residual[~support] = 0
            cached_id = case["case_id"]
        tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        crop = residual[item["row"]:item["row"]+tile_size, item["col"]:item["col"]+tile_size]
        tile[:crop.shape[0], :crop.shape[1]] = crop
        result.append(tile)
    return np.stack(result)


def train_arib_h0(dataset_run, out, source_root, representations=("mod256",), device="cpu",
                  author_config="imagenet32_config", epochs=3, max_train_patches=160,
                  max_val_patches=16, batch_size=1, tile_size=32, lr=2e-4, seed=42):
    import torch
    from tqdm import tqdm
    from .methods.arib_bps import configure_torch, load_author_model

    epochs = integer_option(epochs, "epochs", 1)
    batch_size = integer_option(batch_size, "batch_size", 1)
    tile_size = integer_option(tile_size, "tile_size", 32)
    if tile_size % 32:
        raise ValueError("Use tiles divisible by 32 with the original ArIB model")
    budgets = {"train": integer_option(max_train_patches, "max_train_patches", 1),
               "val": integer_option(max_val_patches, "max_val_patches", 1)}
    representations = tuple(representations)
    if not representations or len(set(representations)) != len(representations) or set(representations) - {"abs", "mod256"}:
        raise ValueError("representations must be unique abs/mod256 values")
    if isinstance(lr, bool) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    target = configure_torch(device, seed)
    if target.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    h0 = importlib.import_module(".2026-10-04_msdzip_h0", __package__)
    root, parent, index, split, fingerprints = h0._parent(dataset_run)
    signatures = {(state, scenario["id"]) for state in parent["config"]["states"] for scenario in parent["config"]["scenarios"]}
    bases, cases = _select(index, split, 0, signatures)
    checked = {}
    plans = sample_h0_tiles(root, parent, bases, cases, tile_size, budgets, seed, checked)
    model, author = load_author_model(source_root, author_config, dropout=0.2 if author_config == "imagenet32_config" else 0.0)
    out = new_directory(out)
    config = {"dataset_run": str(root), "input_sha256": fingerprints, "source_root": str(Path(source_root).resolve()),
              "author_config": author_config, "author": author, "representations": list(representations),
              "device": str(target),
              "epochs": epochs, "batch_size": batch_size, "tile_size": tile_size, "lr": lr, "seed": seed,
              "patch_budget": budgets, "protocol": {
                  "architecture": "unmodified pinned original ArIB-BPS SIG and INS",
                  "training": "separate original SIG/INS forward().sum() objectives and Adam; fresh weights",
                  "adaptation": "bounded epochs on facade-disjoint synthetic H0 residual-image tiles; not author ImageNet training reproduction",
                  "checkpoint_selection": "H0 validation theoretical bpd averaged within building, then equal buildings",
                  "val_randomness": "same fixed latent-sampling seed at every epoch",
                  "input": "native RGB int16 subtraction, uint8 abs or modulo 256; base geometric support only",
                  "loss_domain": "complete zero-padded tiles, including unsupported zeros; support fractions recorded per tile",
                  "test": "no TEST bytes or labels loaded for training or checkpoint selection"}}
    record = run_record("arib_h0_training", config)
    write_json(out / "run.json", record)
    try:
        write_json(out / "sampling.json", {"partitions": plans, "selected_rgb_support_sha256": checked,
            "algorithm": "equal building/scenario cells; SHA256-ranked nonoverlapping tile candidates; fixed across representations"})
        summaries = {}
        for representation in representations:
            random.seed(seed)
            torch.manual_seed(seed)
            if representation != representations[0]:
                model, _ = load_author_model(source_root, author_config, dropout=author["dropout"])
            torch.manual_seed(seed)
            arrays = {part: _tiles(root, parent, bases, cases, plan, representation, tile_size, checked)
                      for part, plan in plans.items()}
            folder = out / representation
            folder.mkdir()
            history = {}
            for component in ("sig", "ins"):
                net = getattr(model, component).to(target)
                optimizer = torch.optim.Adam(net.parameters(), lr=lr)
                best, epochs_log = float("inf"), []
                generator = np.random.default_rng(seed)
                for epoch in range(1, epochs+1):
                    net.train()
                    total, count = 0.0, 0
                    order = generator.permutation(len(arrays["train"]))
                    with tqdm(total=len(order), desc=f"ArIB {representation}/{component} train {epoch}/{epochs}",
                              unit="tile", dynamic_ncols=True) as progress:
                        for start in range(0, len(order), batch_size):
                            batch = arrays["train"][order[start:start+batch_size]]
                            x = torch.from_numpy(batch).permute(0, 3, 1, 2).to(target)
                            optimizer.zero_grad(set_to_none=True)
                            loss = net(x).sum()
                            if not torch.isfinite(loss):
                                raise RuntimeError("Original ArIB training objective became nonfinite")
                            loss.backward()
                            optimizer.step()
                            total += float(loss.detach().cpu()) * len(batch)
                            count += len(batch)
                            progress.update(len(batch))
                            progress.set_postfix(bpd=f"{total/count:.5f}", refresh=False)
                    net.eval()
                    groups = defaultdict(list)
                    devices = [target.index or 0] if target.type == "cuda" else []
                    with torch.random.fork_rng(devices=devices), torch.no_grad():
                        torch.manual_seed(seed+1000)
                        for tile, item in tqdm(list(zip(arrays["val"], plans["val"])),
                                               desc=f"ArIB {representation}/{component} val", unit="tile", dynamic_ncols=True):
                            x = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(target)
                            value = float(net(x).sum().cpu())
                            if not math.isfinite(value):
                                raise RuntimeError("Original ArIB validation objective became nonfinite")
                            groups[item["building_id"]].append(value)
                    val = float(np.mean([np.mean(values) for values in groups.values()]))
                    epochs_log.append({"epoch": epoch, "train_theoretical_bpd": total/count,
                                       "val_building_macro_theoretical_bpd": val})
                    if val < best:
                        best = val
                        torch.save({key: value.detach().cpu() for key, value in net.state_dict().items()}, folder / (component + ".pth"))
                    write_json(folder / (component + "_history.json"), epochs_log)
                    print(f"ArIB {representation}/{component}: val bpd={val:.6f}, best={best:.6f}", flush=True)
                history[component] = {"best_val_theoretical_bpd": best, "epochs": epochs_log}
                net.to("cpu")
                del optimizer
                if target.type == "cuda":
                    torch.cuda.empty_cache()
            summaries[representation] = history
        write_json(out / "summary.json", {"status": "completed_exploratory", "representations": summaries,
                   "selected_patches": {part: len(plan) for part, plan in plans.items()},
                   "scope": "H0 domain adaptation pilot; no change-detection performance or published SOTA replication claimed"})
        finish_record(out, record, "completed_exploratory")
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        finish_record(out, record, "failed")
        raise
    return {"out": str(out), "selected_patches": {part: len(plan) for part, plan in plans.items()}, "status": "completed_exploratory"}
