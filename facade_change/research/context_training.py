"""H0 TRAIN adaptation of reference-only heads; frozen original byte predictor."""
from __future__ import annotations

import importlib
import math
import time
from pathlib import Path

import numpy as np

from ..io import sha256, write_json
from .common import (ExperimentRun, SCOPE, assert_inputs_unchanged, csv_rows,
                     evaluation_inputs, parent_selection, positive_integer,
                     reports, rgb_inputs, inspection_coverage)


def _frozen_logits(module, scorer, contexts, targets, positions):
    torch = module._torch()
    values = np.empty((len(targets), 256), np.float32)
    with torch.no_grad():
        for batch, _, indices in module._batch_groups(contexts, targets, positions,
                                                      scorer.model.batchsize, scorer.metadata["window_groups"]):
            active = indices >= 0
            logits = module._predict_groups(scorer.model, torch.as_tensor(batch, device=scorer.device))
            values[indices[active]] = logits[torch.as_tensor(active, device=scorer.device)].cpu().numpy()
    return values


def _samples(module, selection, plans, scorer, encoder, context, checked, feature_cache):
    by_case = {case["case_id"]: case for case in selection["cases"]}
    result = {}
    for part, plan in plans.items():
        windows, targets, positions, base_indices, shapes = [], [], [], [], {}
        for sample in plan:
            case = by_case[sample["case_id"]]
            if case["split"] != part or case["hypothesis"] != "H0" or case["state"] != "unchanged":
                raise ValueError("Context training samples must be H0 in their original partition")
            reference, source, support = rgb_inputs(selection, case, checked)
            base = case["base_id"]
            if base not in feature_cache:
                with module._numerics(scorer.device):
                    features, preparation = context.extract_reference(reference, support, encoder, scorer.device)
                feature_cache[base] = (features, preparation)
            offsets = np.asarray(sample["target_byte_offsets"], np.int64)
            x, y = module._windows(module.residual_bytes(reference, source, "mod256"), support,
                                   offsets, scorer.model.timesteps)
            windows.append(x)
            targets.append(y)
            positions.append(offsets)
            base_indices.extend([base] * len(offsets))
            shapes[base] = reference.shape[:2]
        samples = {"contexts": np.concatenate(windows), "targets": np.concatenate(targets),
                   "positions": np.concatenate(positions), "bases": np.asarray(base_indices), "shapes": shapes}
        print(f"Reference context: caching frozen {part} probabilities for {len(samples['targets'])} H0 bytes", flush=True)
        samples["logits"] = _frozen_logits(module, scorer, samples["contexts"], samples["targets"], samples["positions"])
        result[part] = samples
    return result


def _bias(context, head, samples, rows, features):
    import torch
    device = next(head.parameters()).device
    result = torch.zeros((len(rows), 256), device=device)
    identities = samples["bases"][rows]
    for base in np.unique(identities):
        selected = np.flatnonzero(identities == base)
        corrections = context.bias_for_positions(head, features[base][0], samples["shapes"][base],
                                                   samples["positions"][rows[selected]])
        result = result.index_copy(0, torch.as_tensor(selected, device=device), corrections)
    return result


def _epoch(context, head, samples, features, batch_size, optimizer=None, seed=42):
    import torch
    indices = np.arange(len(samples["targets"]))
    if optimizer is not None:
        np.random.default_rng(seed).shuffle(indices)
    total = 0.
    head.train(optimizer is not None)
    device = next(head.parameters()).device
    for start in range(0, len(indices), batch_size):
        rows = indices[start:start + batch_size]
        with torch.set_grad_enabled(optimizer is not None):
            logits = torch.as_tensor(samples["logits"][rows], device=device)
            logits = logits + _bias(context, head, samples, rows, features)
            labels = torch.as_tensor(samples["targets"][rows].astype(np.int64), device=device)
            losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            if not torch.isfinite(losses).all():
                raise ValueError("Reference head produced nonfinite H0 NLL")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                losses.mean().backward()
                optimizer.step()
        total += float(losses.detach().sum().cpu()) / math.log(2)
    return total / len(indices)


def score_case(module, scorer, context, heads, features, reference, source, support):
    """Same frozen causal byte windows for all arms; no label inputs."""
    torch = module._torch()
    residual = module.residual_bytes(reference, source, "mod256")
    positions = np.flatnonzero(np.repeat(support.reshape(-1), 3))
    windows, targets = module._windows(residual, support, positions, scorer.model.timesteps)
    values = {mode: np.full(residual.size, np.nan, np.float32) for mode in ("baseline", *heads)}
    with torch.no_grad():
        grids = {mode: head.grid_logits(features) for mode, head in heads.items()}
        for batch, labels, indices in module._batch_groups(windows, targets, positions,
                                                          scorer.model.batchsize, scorer.metadata["window_groups"]):
            active = indices >= 0
            selected = indices[active]
            logits = module._predict_groups(scorer.model, torch.as_tensor(batch, device=scorer.device))
            logits = logits[torch.as_tensor(active, device=scorer.device)]
            y = torch.as_tensor(labels[active], device=scorer.device)
            for mode in values:
                conditioned = logits
                if mode != "baseline":
                    conditioned = logits + context.bias_for_positions(heads[mode], features, support.shape,
                                                 positions[selected], grid_logits=grids[mode])
                bits = torch.nn.functional.cross_entropy(conditioned, y, reduction="none") / math.log(2)
                values[mode][positions[selected]] = bits.cpu().numpy()
    return {mode: values.reshape(residual.shape).mean(axis=-1).astype(np.float32)
            for mode, values in values.items()}


def run_context(dataset_run, out, msdzip_checkpoint, dino_root, dino_checkpoint,
                device="cpu", epochs=3, max_train_bytes=32768, max_val_bytes=8192,
                batch_size=512, lr=.001, width=32, tokens=4, seed=42,
                max_buildings_per_split=2, max_bases_per_building=1, scenarios_per_base=2,
                fpr_targets=(.01, .05), trust_checkpoint=False,
                expected_msdzip_sha256=None, expected_dino_sha256=None, source_lineage=None):
    for name, value in (("epochs", epochs), ("max_train_bytes", max_train_bytes),
                        ("max_val_bytes", max_val_bytes), ("batch_size", batch_size),
                        ("width", width), ("tokens", tokens)):
        positive_integer(value, name)
    if isinstance(lr, bool) or not np.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    selection = parent_selection(dataset_run, max_buildings_per_split, max_bases_per_building,
                                 scenarios_per_base, seed)
    checkpoint_hashes = {"msdzip": sha256(msdzip_checkpoint), "dinov2": sha256(dino_checkpoint)}
    if ((expected_msdzip_sha256 and checkpoint_hashes["msdzip"] != expected_msdzip_sha256)
            or (expected_dino_sha256 and checkpoint_hashes["dinov2"] != expected_dino_sha256)):
        raise ValueError("Checkpoint changed relative to the source benchmark")
    run = ExperimentRun(out, "reference_context_feasibility", selection,
                        {"msdzip_checkpoint": str(Path(msdzip_checkpoint).resolve()),
                         "source_benchmark": source_lineage,
                         "dino_root": str(Path(dino_root).resolve()), "dino_checkpoint": str(Path(dino_checkpoint).resolve()),
                         "checkpoint_sha256": checkpoint_hashes, "device": device,
                         "epochs": epochs, "max_train_bytes": max_train_bytes, "max_val_bytes": max_val_bytes,
                         "batch_size": batch_size, "lr": lr, "width": width, "tokens": tokens,
                         "fpr_targets": list(fpr_targets), "trust_checkpoint": bool(trust_checkpoint),
                         "arms": ["baseline", "spatial", "spectral"],
                         "primary_contrast": "spectral vs spatial: same trainable capacity, initialization, samples, optimizer and schedule",
                         "probability": "frozen original MSDZip logits + reference-only token head; complete 256-byte vocabulary",
                         "cost_mode": "model_bits/NLL only; no neural bitstream or entropy decode claim",
                         "selection": "H0 TRAIN only; each head checkpoint chosen by minimum H0 VAL NLL",
                         "spectral": "SET-inspired adaptation of frozen reference DINO features, full rFFT amplitude+phase; not author SET reproduction",
                         "source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}})
    checked, feature_cache, history, heads, items = {}, {}, [], {}, []
    module = importlib.import_module("facade_change.2026-10-04_msdzip_h0")
    torch = module._torch()
    old_threads = torch.get_num_threads()
    try:
        if str(device).startswith("cpu"):
            torch.set_num_threads(min(4, old_threads))
        from . import context
        scorer = module.MSDZipScorer(msdzip_checkpoint, device=device,
                                     dataset_fingerprint=selection["fingerprints"],
                                     trust_checkpoint=trust_checkpoint, representation="mod256")
        encoder, encoder_provenance = context.load_reference_encoder(dino_root, dino_checkpoint,
                                                                      scorer.device, trust_checkpoint)
        write_json(run.out / "model_provenance.json", {"msdzip": scorer.metadata, "reference_encoder": encoder_provenance})
        frozen_hash = {key: value.detach().cpu().clone() for key, value in scorer.model.state_dict().items()}
        plans = {part: module._sampling_plan(selection["root"], selection["parent"], selection["bases"],
                                             selection["cases"], part, budget, seed, checked)
                 for part, budget in (("train", max_train_bytes), ("val", max_val_bytes))}
        write_json(run.out / "sampling.json", plans)
        samples = _samples(module, selection, plans, scorer, encoder, context, checked, feature_cache)
        channel_counts = {int(pair[0].shape[0]) for pair in feature_cache.values()}
        if len(channel_counts) != 1:
            raise ValueError("Reference feature channels changed between bases")
        channels = channel_counts.pop()
        parameter_counts = {}
        baseline_nll = {}
        for part, data in samples.items():
            logits = torch.as_tensor(data["logits"])
            labels = torch.as_tensor(data["targets"].astype(np.int64))
            baseline_nll[part] = float(torch.nn.functional.cross_entropy(logits, labels).item()) / math.log(2)
        for mode in ("spatial", "spectral"):
            torch.manual_seed(seed)
            if scorer.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            head = context.ReferenceHead(mode, channels, width, tokens).to(scorer.device)
            parameter_counts[mode] = sum(p.numel() for p in head.parameters())
            optimizer = torch.optim.Adam(head.parameters(), lr=lr)
            best, best_epoch, best_state = math.inf, 0, None
            checkpoint = run.out / f"{mode}.pt"
            for epoch in range(1, epochs + 1):
                started = time.perf_counter()
                with module._numerics(scorer.device):
                    train_nll = _epoch(context, head, samples["train"], feature_cache, batch_size, optimizer, seed + epoch)
                    val_nll = _epoch(context, head, samples["val"], feature_cache, batch_size)
                row = {"mode": mode, "epoch": epoch, "train_bpb": train_nll, "val_bpb": val_nll,
                       "seconds": time.perf_counter()-started}
                history.append(row)
                csv_rows(run.out / "training_history.csv", history)
                run.event("epoch_completed", **row)
                print(f"Context {mode} {epoch}/{epochs}: train={train_nll:.6f} val={val_nll:.6f} bpb", flush=True)
                if val_nll < best:
                    best, best_epoch = val_nll, epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                    torch.save({"metadata": {"mode": mode, "channels": channels, "width": width, "tokens": tokens,
                                             "best_epoch": epoch, "val_bpb": val_nll,
                                             "input_sha256": selection["fingerprints"], "sampling_sha256": sha256(run.out / "sampling.json"),
                                             "checkpoint_sha256": checkpoint_hashes,
                                             "experiment_config_sha256": run.record["config_sha256"]},
                                "state_dict": best_state}, checkpoint)
            head.load_state_dict(best_state, strict=True)
            head.eval()
            for parameter in head.parameters():
                parameter.requires_grad_(False)
            heads[mode] = head
            run.event("head_frozen", mode=mode, best_epoch=best_epoch, val_bpb=best,
                      checkpoint_sha256=sha256(checkpoint), trainable_parameters=parameter_counts[mode])
        if parameter_counts["spatial"] != parameter_counts["spectral"]:
            raise ValueError("Spatial and spectral capacity differs")
        for mode in ("baseline", *heads):
            (run.out / "scores" / mode).mkdir(parents=True)
            (run.out / "raw_bpb" / mode).mkdir(parents=True)
        for number, case in enumerate(selection["cases"], 1):
            reference, source, support = rgb_inputs(selection, case, checked)
            base = case["base_id"]
            if base not in feature_cache:
                with module._numerics(scorer.device):
                    feature_cache[base] = context.extract_reference(reference, support, encoder, scorer.device)
            features = feature_cache[base][0]
            started = time.perf_counter()
            with module._numerics(scorer.device):
                raw = score_case(module, scorer, context, heads, features, reference, source, support)
            _, _, _, labels = evaluation_inputs(selection, case, checked)
            for mode, bpb in raw.items():
                scores = (-np.expm1(-bpb / np.float32(8))).astype(np.float32)
                if not np.isfinite(scores[support]).all() or not np.isnan(scores[~support]).all():
                    raise ValueError("Invalid support in context score map")
                path = run.out / "scores" / mode / f"{case['case_id']}.npy"
                np.save(path, scores, allow_pickle=False)
                np.save(run.out / "raw_bpb" / mode / f"{case['case_id']}.npy", bpb, allow_pickle=False)
                items.append({"case": case, "labels": labels, "scores": scores, "method": mode})
                run.event("case_scored", case_id=case["case_id"], method=mode, score_sha256=sha256(path),
                          cost_mode="model_bits", total_scoring_seconds=time.perf_counter()-started)
            print(f"Context score {number}/{len(selection['cases'])}: {case['split']} {case['building_id']} {case['state']}/{case['scenario_id']}", flush=True)
        if any(not torch.equal(value, scorer.model.state_dict()[key].cpu()) for key, value in frozen_hash.items()):
            raise ValueError("Original MSDZip weights changed during head adaptation")
        if sha256(msdzip_checkpoint) != checkpoint_hashes["msdzip"] or sha256(dino_checkpoint) != checkpoint_hashes["dinov2"]:
            raise ValueError("Source checkpoint changed during experiment")
        methods = reports(run, items, ("baseline", "spatial", "spectral"), fpr_targets)
        inspection = inspection_coverage(run, items)
        assert_inputs_unchanged(selection, checked)
        write_json(run.out / "reference_preparation.json", {base: pair[1] for base, pair in feature_cache.items()})
        summary = {"status": "completed_exploratory", "scope": SCOPE, "methods": methods,
                   "baseline_h0_sample_bpb": baseline_nll, "history": history,
                   "matched_trainable_parameters": parameter_counts,
                   "selected_bytes": {part: len(data["targets"]) for part, data in samples.items()},
                   "case_count": len(selection["cases"]), "original_weights_unchanged": True,
                   "inspection_coverage": inspection,
                   "limitations": ["Only theoretical byte NLL; no decoder or neural bitstream validation",
                                   "Frozen baseline vs trained heads confounds extra adaptation and reference context; primary contrast spatial vs spectral",
                                   "SET-inspired external adapter, not original SET inside VFM layers or its attention optimization",
                                   "VAL checkpoint selection and calibration reuse the same facades",
                                   "Synthetic matched controls; not real event ground truth"]}
        return run.finish(summary, checked)
    except (Exception, KeyboardInterrupt) as exc:
        run.fail(exc)
        raise
    finally:
        torch.set_num_threads(old_threads)
