"""Canonical last-pair residual experiment with causal observation memory."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from functools import lru_cache
import importlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from tqdm import tqdm

from facade_change import history_features
from facade_change.io import read_json, sha256, write_json

fusion = importlib.import_module("data_analysis.2026-11-06_history_fusion")
benchmark, metrics = fusion.benchmark, fusion.metrics
PREFIX = "2026-11-08_"
MODES = ("pair", "mean", "convgru", "convgru-last")


def progress(items=None, **options):
    """Nested terminal bars, with fewer updates and no cursor-up in log files."""
    terminal = sys.stderr.isatty()
    if not terminal:
        options["position"] = 0
    options.setdefault("leave", False)
    return tqdm(items, dynamic_ncols=True, mininterval=.5 if terminal else 5., **options)


def pin_contract(store, contract):
    path = store.out / "experiment_contract.json"
    if path.exists():
        previous = read_json(path)
        migration_path = Path(__file__).with_name("2026-11-08_memory_progress_compatibility.json")
        migration = read_json(migration_path)
        runner = Path(__file__).name
        legacy = {**contract, "code": {**contract["code"], runner: migration["before_sha256"]}}
        # Pin both endpoints: a future scientific edit cannot inherit this exception.
        if previous == legacy and contract["code"][runner] == migration["after_sha256"]:
            store.contract(previous)
            store.event("progress_only_code_update", previous_code=previous["code"],
                        executing_code=contract["code"], compatibility_sha256=sha256(migration_path),
                        computation_unchanged=True)
            return
    store.contract(contract)


class FrozenStack:
    def __init__(self, path, source):
        self.root = Path(path).expanduser().resolve()
        record = read_json(self.root / "run.json")
        request = read_json(self.root / "request.json")
        if (record.get("kind") != "canonical_history_fusion" or record.get("status") != "completed"
                or request.get("source_sha256") != source.hashes):
            raise ValueError("--fusion must be a completed fusion of exactly --run")
        identifier = fusion.PREFIX + "model-stack--previous"
        saved = benchmark.read_unit(self.root, identifier)
        if saved is None or saved[0]["variant"]["columns"] != list(range(len(fusion.METHODS))):
            raise ValueError("Frozen reference must be plain stack/previous with all seven methods")
        self.model = saved[1]
        self.normalizer = {k: np.asarray(v[:len(fusion.METHODS)]) for k, v in
                           read_json(self.root / (fusion.PREFIX + "normalizer.json")).items()}
        self.fpr_threshold = read_json(self.root / (fusion.PREFIX + "fpr05_thresholds.json"))["stack--previous"]
        self.f1_threshold = read_json(self.root / (fusion.PREFIX + "thresholds.json"))["stack--previous"]["threshold"]
        names = ["run.json", "request.json", fusion.PREFIX + "normalizer.json",
                 fusion.PREFIX + "fpr05_thresholds.json", fusion.PREFIX + "thresholds.json",
                 fusion.PREFIX + "per_building.csv", saved[0]["artifact_path"]]
        self.hashes = {str(self.root / name): sha256(self.root / name) for name in names}

    def logits(self, raw):
        x = fusion.normalize(raw, self.normalizer)
        return self.model["parameters"][0] + (
            (x - self.model["center"]) / self.model["scale"]) @ self.model["parameters"][1:]

    def verify(self):
        for path, expected in self.hashes.items():
            if sha256(path) != expected:
                raise ValueError(f"Frozen fusion changed: {path}")

    def parity(self, histograms, split):
        with (self.root / (fusion.PREFIX + "per_building.csv")).open(encoding="utf-8-sig") as stream:
            expected = {r["building_id"]: r for r in csv.DictReader(stream)
                        if r["split"] == split and r["method"] == "stack" and r["mode"] == "previous"
                        and r["slice_type"] == "overall"}
        if set(expected) != set(histograms):
            raise ValueError("Frozen baseline building cohort differs")
        for building, hist in histograms.items():
            actual = metrics.metrics(hist[0], self.f1_threshold)
            for key in ("tp", "fp", "fn", "tn"):
                if actual[key] != int(expected[building][key]):
                    raise ValueError(f"Frozen stack parity failed: {split}/{building}/{key}")


def prepare_features(source, store, device):
    """One frozen DINO feature pass, independent of annotation targets."""
    settings = {**source.config["options"]["dinov2"],
                "config_path": str(source.root / "run.json"),
                "config_sha256": sha256(source.root / "run.json"),
                "trust_checkpoint": source.config.get("trust_checkpoint", False)}
    contract = history_features.encoder_contract(settings, device, True, source.data.fp)
    original = read_json(source.root / "method_contracts/dinov2.json")
    source.data.fp.add(source.root / "method_contracts/dinov2.json")
    if (contract["checkpoint_sha256"] != original["checkpoint"]["sha256"]
            or contract["python_tree_sha256"] != original["sources"]["python_tree_sha256"]):
        raise ValueError("Memory encoder differs from the source DINO weights/source")
    path = store.out / (PREFIX + "encoder.json")
    if path.exists() and read_json(path) != contract:
        raise ValueError("Feature encoder contract changed; choose a new output")
    write_json(path, contract)
    encoder, computed = None, 0
    try:
        observations = progress(sorted(source.data.observations), desc="Frozen DINO features",
                                unit="observation", position=1)
        with observations:
            for identifier in observations:
                unit = PREFIX + "features-" + identifier
                cached = store.read_unit(unit)
                if cached:
                    if cached[0]["encoder_sha256"] != sha256(path):
                        raise ValueError("Cached features have a different encoder contract")
                    observations.set_postfix(new=computed, refresh=False)
                    continue
                rgb, support = source.data.frame(identifier)
                valid = history_features.supported_patches(support)
                if not valid.any():
                    arrays = dict(patch_tokens=np.zeros((256, 384), np.float16), patch_support=valid)
                else:
                    if encoder is None:
                        encoder = history_features.DinoEncoder(settings, device, contract)
                    arrays = encoder.encode([rgb], [support], save_patch_tokens=True)[0]
                store.save_unit(dict(unit_id=unit, kind="frozen_observation_features",
                    observation_id=identifier, encoder_sha256=sha256(path), target_labels_used=False), arrays)
                computed += 1
                observations.set_postfix(new=computed, refresh=False)
    finally:
        if encoder:
            encoder.close()
    return dict(observations=len(source.data.observations), newly_computed=computed, contract=contract)


class Inputs:
    def __init__(self, source, store, frozen, device):
        self.source, self.store, self.frozen, self.device = source, store, frozen, device
        self.readers = source.readers[:len(fusion.METHODS)]

    @lru_cache(maxsize=128)
    def feature(self, identifier):
        value = self.store.read_unit(PREFIX + "features-" + identifier)
        if value is None:
            raise ValueError(f"Missing feature cache for {identifier}")
        arrays = value[1]
        if (arrays["patch_tokens"].shape != (256, 384)
                or arrays["patch_support"].shape != (256,)
                or not np.isfinite(arrays["patch_tokens"]).all()):
            raise ValueError("Invalid frozen feature cache")
        return arrays["patch_tokens"].T.reshape(384, 16, 16), arrays["patch_support"].reshape(1, 16, 16)

    @lru_cache(maxsize=32)
    def scores(self, event_id):
        row = self.source.data.events[event_id]
        values = [reader(row) for reader in self.readers]
        support = np.logical_and.reduce([v[1] for v in values])
        scores = np.stack([v[0] for v in values]).astype(np.float32)
        scores[:, ~support] = 0
        return scores, support

    def tensors(self, row, history_shift=0):
        import torch
        observations = self.source.data.tracks[row["track_id"]]["observations"][:row["frame"] + 1]
        features, support = zip(*(self.feature(r["observation_id"]) for r in observations))
        features, support = np.stack(features), np.stack(support)
        if history_shift:
            # Perturb older history only, without wraparound or changing the last pair.
            features[:-2, :, :, history_shift:] = features[:-2, :, :, :-history_shift].copy()
            support[:-2, :, :, history_shift:] = support[:-2, :, :, :-history_shift].copy()
            features[:-2, :, :, :history_shift] = 0
            support[:-2, :, :, :history_shift] = False
        def tensor(value, dtype=torch.float32):
            return torch.as_tensor(np.asarray(value), device=self.device, dtype=dtype)
        return (tensor(features), tensor(support, torch.bool), tensor([r["year"] for r in observations]),
                tensor(self.scores(row["event_id"])[0]))

    @lru_cache(maxsize=48)
    def annotation(self, identifier):
        # Used only for post-prediction return-to-past diagnostics, never tensors().
        result = benchmark.read_unit(self.source.data.root, "observation-" + identifier)
        expected = self.source.data.index["observations"][identifier]
        if result is None or result[0]["artifact_sha256"] != expected["artifact_sha256"]:
            raise ValueError("Projected observation labels differ from preparation")
        self.source.data.fp.add(self.source.data.root / result[0]["artifact_path"], expected["artifact_sha256"])
        return result[1]

    def return_to_past(self, row):
        arrays = self.source.data.target(row["event_id"])
        after = arrays["after"]
        result = np.zeros(after.shape, bool)
        observations = self.source.data.tracks[row["track_id"]]["observations"][:row["frame"] - 1]
        for observation in observations:
            old = self.annotation(observation["observation_id"])
            result |= old["support"] & (after != 0) & (old["packed"] == after)
        return result & arrays["support"] & (arrays["target"] == 1)


def evaluate(inputs, models, split, history_shift=0, label=None, position=2):
    """Merge raw scores and correction logits before applying the frozen fusion."""
    import torch
    names = [*fusion.METHODS, "stack", *models]
    for model in models.values():
        model.eval()

    @lru_cache(maxsize=24)
    def prediction(event_id):
        raw, support = inputs.scores(event_id)
        corrections = []
        if models:
            tensors = inputs.tensors(inputs.source.data.events[event_id], history_shift)
            with torch.inference_mode():
                corrections = [model(*tensors).cpu().numpy() for model in models.values()]
        # Annotation-derived diagnostic channel is appended after model inference.
        returning = inputs.return_to_past(inputs.source.data.events[event_id])
        return np.concatenate((raw, np.stack([*corrections, returning]))), support

    samples = [s for s in inputs.source.data.samples if s["split"] == split]
    histograms = {name: {} for name in names}
    flags = [m == "anychange" for m in fusion.METHODS] + [False] * len(models) + [True]
    blocks = metrics.merged_blocks(inputs.source.data, samples, lambda row: prediction(row["event_id"]),
                                   flags, logit_channels=range(7, 7 + len(models)))
    block_keys = set()
    for row in samples:
        x, y, width, height = row["bbox_xywh"]
        key = tuple(row[k] for k in ("building_id", "view_id", "previous_year", "current_year"))
        block_keys.update((*key, bx, by)
                          for bx in range(x // 256, (x + width - 1) // 256 + 1)
                          for by in range(y // 256, (y + height - 1) // 256 + 1))
    description = label or f"Evaluate {split.upper()} shift={history_shift}"
    for key, _, scores, truth, geometric, _ in progress(
            blocks, total=len(block_keys), desc=description, unit="block", position=position):
        valid = geometric & np.any(truth != 255, axis=0)
        if not valid.any():
            continue
        raw, labels = scores[:, valid].T, truth[:, valid]
        base = inputs.frozen.logits(raw[:, :7])
        logits = np.column_stack((base, *(base + raw[:, i] for i in range(7, 7 + len(models)))))
        # Stable sigmoid; the logits remain unclipped for training and merging.
        probability = np.exp(-np.logaddexp(0., -logits))
        values = np.column_stack((raw[:, :7], probability))
        for column, name in enumerate(names):
            hist = histograms[name].setdefault(key[0], np.zeros((len(labels) + 1, 2, metrics.BINS), np.int64))
            quantized = np.ceil(values[:, column] * (metrics.BINS - 1)).astype(np.int64)
            for channel in range(len(labels)):
                for label in (0, 1):
                    hist[channel, label] += np.bincount(quantized[labels[channel] == label], minlength=metrics.BINS)
            returning = (labels[0] == 1) & (raw[:, -1] > .5)
            hist[-1, 1] += np.bincount(quantized[returning], minlength=metrics.BINS)
    return histograms


def calibrate(histograms, frozen):
    specs = [dict(id=name, native=name == "anychange") for name in histograms]
    buildings = sorted(next(iter(histograms.values())))
    overall = {b: np.stack([histograms[v["id"]][b][0] for v in specs]) for b in buildings}
    selected = {v["id"]: dict(threshold=.5) for v in specs}
    thresholds = fusion.select_fpr_thresholds(specs, overall, selected)
    thresholds["stack"] = frozen.fpr_threshold
    return thresholds


def fixed_macro(rows):
    """Positive-facade F1/IoU are a fixed cohort, including zero-recall methods."""
    result = {}
    for metric in metrics.METRICS:
        eligible = [r for r in rows if r[metric] is not None]
        if metric in ("f1", "iou"):
            eligible = [r for r in eligible if r["positive_pixels"] > 0]
        result[metric] = float(np.mean([r[metric] for r in eligible])) if eligible else None
        result[metric + "_buildings"] = len(eligible)
    return result


def reporting(data, histograms, thresholds, split, seed):
    rows, aggregate = [], []
    for name, buildings in histograms.items():
        threshold = thresholds[name]["threshold"]
        report = metrics.report(data, buildings, threshold, name == "anychange")
        current = [dict(method=name, seed=seed, split=split, threshold=threshold, **r)
                   for r in report["per_building"]]
        rows.extend(current)
        for kind, label, _ in metrics.slices(data):
            subset = [r for r in current if r["slice_type"] == kind and r["slice"] == label]
            aggregate.append(dict(method=name, seed=seed, split=split, slice_type=kind, slice=label,
                threshold=threshold, val_fpr_budget=.05,
                known_pixels=sum(r["known_pixels"] for r in subset),
                f1_cohort="facades with positive GT pixels", **fixed_macro(subset)))
    return aggregate, rows


def load_model(store, identifier, args, mode):
    import torch
    module = importlib.import_module("facade_change.research.2026-11-08_variability_memory_model")
    model = module.VariabilityMemory(hidden=args.hidden, mode=mode).to(args.device)
    cached = store.read_unit(identifier)
    if cached:
        model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in cached[1].items()})
    return model, cached


def train(inputs, store, args, mode, seed):
    import torch
    from torch.nn import functional as F

    torch.manual_seed(seed)
    identifier = PREFIX + f"model-{mode}-{seed}"
    model, cached = load_model(store, identifier, args, mode)
    if cached:
        return model, cached[0]
    samples = [s for s in inputs.source.data.samples if s["split"] == "train" and s["known_pixels"] > 0]
    counts = Counter(s["building_id"] for s in samples)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    best, state, best_epoch = (-1., -1.), None, None
    epochs = progress(range(1, args.epochs + 1), desc=f"{mode} seed={seed}", unit="epoch", position=2)
    for epoch in epochs:
        order = np.random.default_rng(seed + epoch).permutation(len(samples))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.
        batches = progress(order, desc=f"TRAIN {mode} seed={seed} epoch={epoch}/{args.epochs}",
                           unit="crop", position=3)
        for number, index in enumerate(batches, 1):
            row = samples[index]
            features = inputs.tensors(row)
            arrays = inputs.source.data.target(row["event_id"])
            valid = arrays["support"] & (arrays["target"] != 255)
            raw = features[-1].permute(1, 2, 0).detach().cpu().numpy()
            base = torch.as_tensor(inputs.frozen.logits(raw.reshape(-1, 7)).reshape(valid.shape),
                                   dtype=torch.float32, device=args.device)
            target = torch.as_tensor(arrays["target"].astype(np.float32), device=args.device)
            mask = torch.as_tensor(valid, device=args.device)
            logits = base + model(*features)
            weight = len(samples) / (len(counts) * counts[row["building_id"]])
            loss = F.binary_cross_entropy_with_logits(logits[mask], target[mask]) * weight
            group_size = min(args.batch_size, len(samples) - ((number - 1) // args.batch_size) * args.batch_size)
            (loss / group_size).backward()
            loss_total += float(loss.detach())
            batches.set_postfix(BCE=f"{loss_total / number:.5f}", refresh=False)
            if number % args.batch_size == 0 or number == len(samples):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        val = evaluate(inputs, {mode: model}, "val",
                       label=f"VAL {mode} seed={seed} epoch={epoch}/{args.epochs}", position=3)
        thresholds = calibrate(val, inputs.frozen)
        rows = [metrics.metrics(h[0], thresholds[mode]["threshold"]) for h in val[mode].values()]
        score = fixed_macro(rows)
        criterion = (score["recall"], -score["fpr"])
        if criterion > best:
            best, best_epoch = criterion, epoch
            state = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
        store.event("memory_epoch", mode=mode, seed=seed, epoch=epoch,
                    train_bce=loss_total / len(samples), val=score)
        epochs.set_postfix(recall=f"{score['recall']:.5f}", FPR=f"{score['fpr']:.5f}", refresh=False)
        tqdm.write(f"{mode} seed={seed} epoch={epoch}/{args.epochs}: BCE={loss_total / len(samples):.6f}, "
                   f"VAL recall={score['recall']:.5f}, FPR={score['fpr']:.5f}")
    if state is None:
        raise ValueError("No valid memory checkpoint")
    record = store.save_unit(dict(unit_id=identifier, kind="causal_variability_head", mode=mode, seed=seed,
        selected_epoch=best_epoch, selection_split="val", fit_split="train", test_used_for_fit=False,
        criterion="maximum VAL recall at VAL FPR <= .05; minimum FPR; earliest epoch",
        parameters=sum(p.numel() for p in model.parameters()),
        train_weighting="equal total building weight; mean BCE within each known crop"), state)
    model.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
    return model, record


def paired_comparisons(rows, seeds):
    output = []
    for seed in seeds:
        subset = [r for r in rows if r["split"] == "test" and r["seed"] == seed and r["slice_type"] == "overall"]
        by_method = {name: {r["building_id"]: r for r in subset if r["method"] == name}
                     for name in ("stack", *MODES)}
        for reference in ("stack", "pair", "mean", "convgru-last"):
            for metric in ("f1", "recall", "fpr"):
                differences = []
                for building, a in by_method["convgru"].items():
                    b = by_method[reference][building]
                    if a[metric] is None or b[metric] is None or (metric == "f1" and not a["positive_pixels"]):
                        continue
                    differences.append(a[metric] - b[metric])
                if not differences:
                    continue
                values = np.asarray(differences)
                rng = np.random.default_rng(seed)
                draws = values[rng.integers(0, len(values), (2000, len(values)))].mean(axis=1)
                low, high = np.quantile(draws, [.025, .975])
                output.append(dict(seed=seed, method="convgru", reference=reference, metric=metric,
                    buildings=len(values), delta=float(values.mean()), ci_low=float(low), ci_high=float(high),
                    interpretation="exploratory paired facade bootstrap; TEST previously inspected"))
    return output


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if (min(args.epochs, args.hidden, args.batch_size) < 1 or args.lr <= 0
            or not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or any(s < 1 or s > 15 for s in args.history_shift_tokens)):
        raise ValueError("Need positive dimensions/epochs/lr and distinct seeds")
    with progress(total=2, desc="Validate source and frozen fusion", unit="input", position=0) as validation:
        source = fusion.Source(args.run)
        validation.update(1)
        frozen = FrozenStack(args.fusion, source)
        validation.update(1)
    out = Path(args.out).expanduser().resolve()
    for original in (source.root, source.data.root, source.data.history_root, frozen.root):
        if out == original or out in original.parents or original in out.parents:
            raise ValueError("Output must be separate from all inputs")
    config = dict(source=str(source.root), source_sha256=source.hashes,
        fusion=str(frozen.root), fusion_sha256=frozen.hashes, device=args.device,
        epochs=args.epochs, hidden=args.hidden, batch_size=args.batch_size, lr=args.lr, seeds=args.seeds,
        history_shift_tokens=args.history_shift_tokens)
    if (out / "run.json").exists() and read_json(out / "run.json").get("status") == "completed":
        if read_json(out / "request.json") != config:
            raise ValueError("Completed run has different settings")
        if sha256(out / (PREFIX + "summary.json")) != read_json(out / "run.json")["summary_sha256"]:
            raise ValueError("Completed summary is corrupt")
        print(f"Already completed: {out}", flush=True)
        return read_json(out / (PREFIX + "summary.json"))
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable in the DINO worker environment")
        free, total = torch.cuda.mem_get_info(args.device)
        print(f"Memory experiment GPU: {free / 2**30:.2f}/{total / 2**30:.2f} GiB free", flush=True)
        if free < 512 * 2**20:
            raise ValueError("Less than 512 MiB GPU memory is free")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # CUDA bilinear interpolation backward is not deterministic on every supported
    # Torch version. Warn rather than abort; repeated seeds quantify this setting.
    torch.use_deterministic_algorithms(True, warn_only=args.device.startswith("cuda"))
    store = fusion.data_module.Store(out, config, resume=args.resume)
    store.record["kind"] = "canonical_variability_memory"
    write_json(out / "run.json", store.record)
    stages = progress(total=4, desc="Memory 1/4: frozen features", unit="stage", position=0, leave=True)
    try:
        model_module = importlib.import_module("facade_change.research.2026-11-08_variability_memory_model")
        pin_contract(store, dict(protocol="canonical-variability-memory-v1", config=config,
            code={Path(path).name: sha256(path) for path in
                  (__file__, model_module.__file__, metrics.__file__, fusion.__file__, history_features.__file__,
                   fusion.data_module.__file__, importlib.import_module(
                       "facade_change.research.2026-10-25_spatial_change_model").__file__)},
            torch=str(torch.__version__), numpy=np.__version__, fit_split="train", selection_split="val",
            determinism="strict CPU; CUDA warn-only for unsupported deterministic kernels",
            target_scope=source.summary["target_scope"], test_previously_inspected=True,
            checkpoint_exposure=source.summary.get("checkpoint_exposure", {}),
            history="strictly before current; current enters later prefixes only",
            overlap="raw scores and residual logits merged before frozen stack sigmoid",
            f1_cohort="fixed GT-positive facades; negative-only facades retained for FPR",
            training_caveat="crop BCE may repeat overlap pixels; evaluation merges them exactly"))
        features = prepare_features(source, store, args.device)
        stages.update(1)
        stages.set_description("Memory 2/4: train heads")
        inputs = Inputs(source, store, frozen, args.device)
        checkpoints = []
        with progress(total=len(args.seeds) * len(MODES), desc="Train heads", unit="model", position=1) as heads:
            for seed in args.seeds:
                for mode in MODES:
                    heads.set_description(f"Train heads: {mode} seed={seed}")
                    model, record = train(inputs, store, args, mode, seed)
                    checkpoints.append({k: record[k] for k in
                                        ("unit_id", "mode", "seed", "selected_epoch", "parameters")})
                    del model
                    heads.update(1)
        store.event("all_memory_checkpoints_frozen_before_test", checkpoints=checkpoints)
        stages.update(1)
        stages.set_description("Memory 3/4: final evaluation")
        aggregate, rows, robustness, returns = [], [], [], []
        with progress(total=len(args.seeds) * (3 + len(args.history_shift_tokens)),
                      desc="Final evaluation", unit="split", position=1) as evaluations:
            for seed in args.seeds:
                models = {mode: load_model(store, PREFIX + f"model-{mode}-{seed}", args, mode)[0] for mode in MODES}
                evaluations.set_description(f"Final evaluation seed={seed} VAL")
                val = evaluate(inputs, models, "val", label=f"Final VAL seed={seed}")
                thresholds = calibrate(val, frozen)
                write_json(out / (PREFIX + f"thresholds-{seed}.json"), thresholds)
                for split in ("val", "train", "test"):
                    evaluations.set_description(f"Final evaluation seed={seed} {split.upper()}")
                    hist = val if split == "val" else evaluate(
                        inputs, models, split, label=f"Final {split.upper()} seed={seed}")
                    frozen.parity(hist["stack"], split)
                    a, r = reporting(source.data, hist, thresholds, split, seed)
                    aggregate.extend(a)
                    rows.extend(r)
                    for name, buildings in hist.items():
                        for building, histogram in buildings.items():
                            value = metrics.metrics(histogram[-1], thresholds[name]["threshold"], name == "anychange")
                            returns.append(dict(method=name, seed=seed, split=split, building_id=building,
                                threshold=thresholds[name]["threshold"], candidate_pixels=value["positive_pixels"],
                                detected_pixels=value["tp"], recall=value["recall"],
                                scope="annotation-matched return to a known older state; not reviewed events"))
                    evaluations.update(1)
                for shift in args.history_shift_tokens:
                    evaluations.set_description(f"History shift={shift} seed={seed}")
                    hist = evaluate(inputs, models, "test", history_shift=shift,
                                    label=f"TEST shift={shift} seed={seed}")
                    a, _ = reporting(source.data, hist, thresholds, "test", seed)
                    robustness.extend(dict(history_shift_tokens=shift, approximate_pixels=14 * shift, **r) for r in a)
                    evaluations.update(1)
                del models
        stages.update(1)
        stages.set_description("Memory 4/4: reports and verification")
        comparisons = paired_comparisons(rows, args.seeds)
        write_csv(out / (PREFIX + "metrics.csv"), aggregate)
        write_csv(out / (PREFIX + "per_building.csv"), rows)
        write_csv(out / (PREFIX + "comparisons.csv"), comparisons)
        write_csv(out / (PREFIX + "history_shift.csv"), robustness)
        write_csv(out / (PREFIX + "return_to_past.csv"), returns)
        source.verify()
        frozen.verify()
        summary = dict(kind="canonical_variability_memory", status="completed", features=features,
            checkpoints=checkpoints, baseline_parity="exact stack/previous TP/FP/FN/TN on every building/split",
            fit_split="train", selection_split="val", test_previously_inspected=True,
            independent_event_labels=False, results=aggregate, comparisons=comparisons)
        write_json(out / (PREFIX + "summary.json"), summary)
        lines = ["CAUSAL VARIABILITY MEMORY — exploratory canonical evaluation",
                 "FPR budget selected on VAL only. TEST actual FPR may exceed 5%.",
                 "F1/IoU: fixed GT-positive facade cohort. FPR includes negative-only facades.",
                 "Annotation-derived changes; source checkpoint exposure and inspected TEST remain limitations."]
        for row in aggregate:
            if row["slice_type"] == "overall":
                lines.append(f"{row['split']:5s} seed={row['seed']} {row['method']:15s} "
                             f"F1={row['f1']:.5f} recall={row['recall']:.5f} FPR={row['fpr']:.5f}")
        report = "\n".join(lines) + "\n"
        (out / (PREFIX + "report.txt")).write_text(report, encoding="utf-8")
        tqdm.write(report)
        store.finish("completed", summary_sha256=sha256(out / (PREFIX + "summary.json")))
        stages.update(1)
        return summary
    except BaseException as exc:
        store.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        stages.close()
        store.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("run", "fusion", "out"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8, help="Gradient accumulation over chronological crops")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 29, 47])
    parser.add_argument("--history-shift-tokens", type=int, nargs="+", default=[],
                        help="Optional fixed-threshold TEST sensitivity to older-history shifts, e.g. 1 7")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if not args.worker:
            record = read_json(Path(args.run).expanduser().resolve() / "run.json")
            python = record["config"]["options"]["dinov2"].get("worker_python", sys.executable)
            command = [python, "-s", "-u", "-m", "data_analysis", "variability-memory", "--worker"]
            for key, value in vars(args).items():
                if key == "worker" or value is False or value == []:
                    continue
                command.append("--" + key.replace("_", "-"))
                if value is True:
                    continue
                if key in ("run", "fusion", "out"):
                    value = str(Path(value).expanduser().resolve())
                command.extend(map(str, value if isinstance(value, list) else [value]))
            print(f"Memory worker: {python}", flush=True)
            raise SystemExit(subprocess.run(command, cwd=Path(__file__).resolve().parents[1]).returncode)
        run(args)
    except (ValueError, OSError, KeyError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Ошибка memory: {exc}\n")


if __name__ == "__main__":
    main()
