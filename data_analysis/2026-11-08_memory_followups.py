"""Cached-feature history ablation, seven-score MLP and causal transition banks."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from functools import lru_cache
import importlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch import nn

from facade_change.io import read_json, sha256, write_json

evaluation = importlib.import_module("data_analysis.2026-11-08_memory_evaluate")
bank_module = importlib.import_module("facade_change.research.2026-11-08_transition_bank")
memory, fusion, metrics = evaluation.memory, evaluation.fusion, evaluation.metrics
PREFIX = "2026-11-08_"


class HistoryInput(nn.Module):
    """One checkpoint, matched old-frame support, unchanged final pair and dates."""

    def __init__(self, model, repeat=False):
        super().__init__()
        self.model, self.repeat = model, repeat

    def forward(self, features, support, years, scores):
        common = support.clone()
        common[:-2] &= support[-2:-1]
        if self.repeat:
            features = features.clone()
            features[:-2] = features[-2:-1]
        return self.model(features, common, years, scores)


def cached_histograms(store, identifier, calculate):
    cached = store.read_unit(identifier)
    if cached:
        record, arrays = cached
        return {name: {b: arrays[key] for b, key in buildings.items()}
                for name, buildings in record["histogram_index"].items()}
    histograms = calculate()
    arrays, index = {}, {}
    for name, buildings in histograms.items():
        index[name] = {}
        for building, hist in buildings.items():
            key = "h" + str(len(arrays))
            arrays[key] = hist
            index[name][building] = key
    store.save_unit(dict(unit_id=identifier, kind="followup_histograms", histogram_index=index), arrays)
    return histograms


def paired_differences(rows, comparisons, seed):
    output = []
    groups = sorted({(r["cohort"], r["val_fpr_budget"]) for r in rows if r["split"] == "test"})
    for cohort, budget in groups:
        selected = [r for r in rows if r["split"] == "test" and r["slice_type"] == "overall"
                    and r["cohort"] == cohort and r["val_fpr_budget"] == budget]
        by_method = {name: {r["building_id"]: r for r in selected if r["method"] == name}
                     for name in {r["method"] for r in selected}}
        for method, reference in comparisons:
            for metric in ("f1", "recall", "fpr"):
                values = []
                for b in sorted(set(by_method.get(method, {})) & set(by_method.get(reference, {}))):
                    a, other = by_method[method][b], by_method[reference][b]
                    if a[metric] is None or other[metric] is None or (metric == "f1" and not a["positive_pixels"]):
                        continue
                    values.append(a[metric] - other[metric])
                if not values:
                    continue
                values = np.array(values)
                rng = np.random.default_rng(seed)
                draws = values[rng.integers(len(values), size=(2000, len(values)))].mean(1)
                low, high = np.quantile(draws, [.025, .975])
                output.append(dict(cohort=cohort, val_fpr_budget=budget, method=method, reference=reference,
                    metric=metric, buildings=len(values), delta=float(values.mean()), ci_low=float(low),
                    ci_high=float(high), seed=seed,
                    interpretation="exploratory paired facade bootstrap; previously inspected TEST; one seed"))
    return output


def append_reports(data, histograms, operating, split, seed, cohort, aggregate, rows, returns):
    for budget, selected in operating.items():
        a, r = evaluation.fpr_reporting(data, histograms, selected, split, seed, budget)
        aggregate.extend(dict(cohort=cohort, **v) for v in a)
        rows.extend(dict(cohort=cohort, **v) for v in r)
        returns.extend(dict(cohort=cohort, val_fpr_budget=budget, **v)
                       for v in evaluation.return_rows(histograms, selected, split, seed))


def history_ablation(inputs, saved, store, args, checkpoint):
    shape = argparse.Namespace(hidden=saved.config["hidden"], device=inputs.device)
    original = memory.load_model(saved, checkpoint["unit_id"], shape, "convgru")[0]
    models = {"convgru-original": original, "convgru-common": HistoryInput(original),
              "convgru-repeat": HistoryInput(original, repeat=True)}

    def collect(current, split, cohort):
        def calculate():
            values = memory.evaluate(current, models, split, label=f"Ablation {cohort} {split.upper()}", position=1)
            return {k: v for k, v in values.items() if k == "stack" or k in models}
        return cached_histograms(store, PREFIX + f"ablation-{cohort}-{split}", calculate)

    val = collect(inputs, "val", "all")
    operating = evaluation.fpr_thresholds(val, inputs.frozen, args.fpr_budgets)
    for selected in operating.values():
        for name in ("convgru-common", "convgru-repeat"):
            selected[name] = dict(selected["convgru-original"],
                                 threshold_origin="unmodified checkpoint on full VAL; no recalibration")
    write_json(store.out / (PREFIX + "ablation_thresholds.json"), {str(k): v for k, v in operating.items()})
    aggregate, rows, returns = [], [], []
    eligible = copy.copy(inputs)
    eligible.source = copy.copy(inputs.source)
    eligible.source.data = copy.copy(inputs.source.data)
    eligible.source.data.samples = [r for r in inputs.source.data.samples if r["frame"] >= 2]
    eligible_is_full = len(eligible.source.data.samples) == len(inputs.source.data.samples)
    for cohort, current in (("all", inputs), ("at_least_three_observations", eligible)):
        for split in ("val", "train", "test"):
            if not any(r["split"] == split for r in current.source.data.samples):
                continue
            cache_cohort = "all" if cohort == "all" or eligible_is_full else cohort
            hist = val if cache_cohort == "all" and split == "val" else collect(current, split, cache_cohort)
            if cohort == "all":
                inputs.frozen.parity(hist["stack"], split)
            append_reports(inputs.source.data, hist, operating, split, args.seed, cohort, aggregate, rows, returns)
    differences = paired_differences(rows, [("convgru-common", "convgru-original"),
                                          ("convgru-repeat", "convgru-common")], args.seed)
    for name, values in (("metrics", aggregate), ("per_building", rows), ("comparisons", differences),
                         ("return_to_past", returns)):
        memory.write_csv(store.out / (PREFIX + "ablation_" + name + ".csv"), values)
    return dict(checkpoint=checkpoint, results=aggregate, comparisons=differences,
                three_observation_cohort_equals_full=eligible_is_full,
                threshold_policy="original full-VAL thresholds frozen across interventions and cohorts",
                support_policy="common/repeat intersect older support with previous-frame support; final pair unchanged")


def build_blocks(inputs, bank, store):
    identifier = PREFIX + "followup-block-index"
    cached = store.read_unit(identifier)
    if cached:
        return cached[0]["blocks"], cached[0]["coverage"]

    @lru_cache(maxsize=48)
    def prediction(event_id):
        raw, support = inputs.scores(event_id)
        # Predictions are complete before annotation-only return diagnostics are read.
        extra = bank.maps(event_id)
        returning = inputs.return_to_past(inputs.source.data.events[event_id])
        return np.concatenate((raw, extra, returning[None])), support

    flags = [m == "anychange" for m in fusion.METHODS] + [False] * 4 + [True]
    index = []
    for split in ("train", "val", "test"):
        samples = [r for r in inputs.source.data.samples if r["split"] == split]
        blocks = metrics.merged_blocks(inputs.source.data, samples,
                                       lambda r: prediction(r["event_id"]), flags)
        for key, position, scores, truth, support, _ in memory.progress(
                blocks, desc=f"Cache merged scores and transitions: {split.upper()}", unit="block", position=1):
            valid = support & np.any(truth != 255, axis=0)
            if not valid.any():
                continue
            unit = PREFIX + "followup-block-" + fusion.digest([key, position])[:24]
            record = dict(unit_id=unit, kind="merged_followup_scores", split=split, building_id=key[0],
                          transition=list(key), block=list(position), pixels=int(valid.sum()))
            if store.read_unit(unit) is None:
                store.save_unit(record, dict(x=scores[:, valid].T, truth=truth[:, valid]))
            index.append(record)
    coverage = [bank.coverage[k] for k in sorted(bank.coverage)]
    store.save_unit(dict(unit_id=identifier, kind="merged_followup_index", blocks=index, coverage=coverage), {})
    return index, coverage


def training_pixels(store, blocks):
    """All known TRAIN pixels after overlap merge, stored without keeping maps in RAM."""
    identifier = PREFIX + "followup-training-pixels"
    cached = store.read_unit(identifier)
    if cached:
        record = cached[0]
    else:
        counts = Counter()
        for item in blocks:
            if item["split"] == "train":
                truth = store.read_unit(item["unit_id"])[1]["truth"]
                counts[item["building_id"]] += int((truth[0] != 255).sum())
        buildings = sorted(b for b, count in counts.items() if count)
        if len(buildings) < 2:
            raise ValueError("Need at least two TRAIN buildings with known target pixels")
        path = store.out / (identifier + ".f32")
        temporary = path.with_suffix(".partial")
        n = 0
        with temporary.open("wb") as stream:
            for item in memory.progress([b for b in blocks if b["split"] == "train"],
                                        desc="Write all TRAIN pixels", unit="block", position=1):
                arrays = store.read_unit(item["unit_id"])[1]
                valid = arrays["truth"][0] != 255
                x = arrays["x"][valid]
                rows = np.column_stack((x[:, :11], arrays["truth"][0, valid],
                                        np.full(len(x), buildings.index(item["building_id"])))).astype(np.float32)
                rows.tofile(stream)
                n += len(rows)
        temporary.replace(path)
        record = store.save_unit(dict(unit_id=identifier, kind="merged_training_pixels", split="train",
            path=path.name, sha256=sha256(path), shape=[n, 13], buildings=buildings,
            counts=[counts[b] for b in buildings], weighting="equal total weight per building"), {})
    path = store.out / record["path"]
    if sha256(path) != record["sha256"]:
        raise ValueError("Follow-up training pixel cache is corrupt")
    matrix = np.memmap(path, mode="r", dtype=np.float32, shape=tuple(record["shape"]))
    if set(np.unique(matrix[:, 11])) != {0, 1}:
        raise ValueError("TRAIN needs both stable and changed annotation-derived targets")
    weights = 1 / (len(record["buildings"]) * np.asarray(record["counts"], np.float64))
    return matrix, weights[matrix[:, 12].astype(np.int64)], record


class ScoreMLP(nn.Module):
    """Seven scores, 16 ReLU units, one logit; initialized to the refitted linear model."""

    def __init__(self, parameters):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(7, 16), nn.ReLU(), nn.Linear(16, 1))
        with torch.no_grad():
            self.layers[0].weight[:14] = torch.cat((torch.eye(7), -torch.eye(7)))
            self.layers[0].bias.zero_()
            self.layers[2].weight.zero_()
            weights = torch.as_tensor(parameters[1:], dtype=torch.float32)
            self.layers[2].weight[0, :14] = torch.cat((weights, -weights))
            self.layers[2].bias.fill_(float(parameters[0]))

    def forward(self, values):
        return self.layers(values).squeeze(-1)


def standardized(raw, linear):
    return ((fusion.normalize(raw, {"low": linear["low"], "high": linear["high"]})
             - linear["center"]) / linear["scale"]).astype(np.float32)


def fit_linear(matrix, weights, store, args):
    identifier = PREFIX + "followup-linear"
    cached = store.read_unit(identifier)
    if cached:
        return cached[1]
    raw = matrix[:, :7]
    low, high = np.quantile(raw, [.01, .99], axis=0)
    normalizer = dict(low=low, high=high)
    values = fusion.normalize(raw, normalizer)
    memory.tqdm.write(f"Fit 8-parameter linear fusion on {len(matrix):,} merged TRAIN pixels")
    model = fusion.fit_logistic(values, matrix[:, 11], weights, args.l2)
    model.update(normalizer)
    store.save_unit(dict(unit_id=identifier, kind="seven_score_linear", parameters=8,
                         fit_split="train", all_known_training_pixels=True), model)
    return model


def fit_correction(matrix, weights, frozen, store, args):
    """One monotone, two-parameter correction shared by all three bank conditions."""
    from scipy.optimize import minimize
    from scipy.special import expit

    identifier = PREFIX + "transition-correction"
    cached = store.read_unit(identifier)
    if cached:
        return cached[1]
    available = matrix[:, 10].astype(np.float64)
    distance = matrix[:, 7] / np.maximum(available, 1e-8)
    weighted_valid = weights * available
    if weighted_valid.sum() == 0:
        raise ValueError("No supported frozen-feature patch on TRAIN")
    center = np.dot(weighted_valid, distance) / weighted_valid.sum()
    scale = max(float(np.sqrt(np.dot(weighted_valid, (distance - center) ** 2) / weighted_valid.sum())), .01)
    x = np.column_stack((available, available * (distance - center) / scale))
    base, y = frozen.logits(matrix[:, :7]), matrix[:, 11]

    def objective(parameters):
        logits = base + x @ parameters
        loss = np.dot(weights, np.logaddexp(0, logits) - y * logits) + .5 * args.l2 * parameters[1] ** 2
        gradient = x.T @ (weights * (expit(logits) - y)) + np.array([0, args.l2 * parameters[1]])
        return float(loss), gradient

    fitted = minimize(objective, np.zeros(2), jac=True, method="L-BFGS-B",
                      bounds=[(None, None), (0, None)], options=dict(maxiter=300, ftol=1e-10))
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise ValueError(f"Transition correction optimization failed: {fitted.message}")
    model = dict(parameters=fitted.x, center=np.array(center), scale=np.array(scale))
    store.save_unit(dict(unit_id=identifier, kind="shared_transition_logit_correction", parameters=2,
        fit_split="train", fit_memory="leave-building-out global bank", slope_nonnegative=True,
        personal_and_foreign_weights_identical=True), model)
    return model


def add_histogram(histograms, name, building, probability, truth, returning):
    histogram = histograms.setdefault(name, {}).setdefault(
        building, np.zeros((len(truth) + 1, 2, metrics.BINS), np.int64))
    bins = np.ceil(np.clip(probability, 0, 1) * (metrics.BINS - 1)).astype(np.int64)
    for i, labels in enumerate(truth):
        for label in (0, 1):
            histogram[i, label] += np.bincount(bins[labels == label], minlength=metrics.BINS)
    histogram[-1, 1] += np.bincount(bins[(truth[0] == 1) & returning], minlength=metrics.BINS)


def evaluate_blocks(store, blocks, split, frozen, linear, model=None, correction=None, only_mlp=False):
    result = {}
    if model is not None:
        model.eval()
        device = next(model.parameters()).device
    selected = [r for r in blocks if r["split"] == split]
    for row in memory.progress(selected, desc=f"Score fusion {split.upper()}", unit="block", position=2):
        arrays = store.read_unit(row["unit_id"])[1]
        x, truth = arrays["x"], arrays["truth"]
        returning = x[:, -1] > .5
        values = {}
        if not only_mlp:
            values.update({name: x[:, i] for i, name in enumerate(fusion.METHODS)})
            base = frozen.logits(x[:, :7])
            logits = {"stack": base,
                      "linear-all": linear["parameters"][0] + standardized(x[:, :7], linear) @ linear["parameters"][1:]}
            if correction is not None:
                a = x[:, 10]
                for column, name in enumerate(("transition-global", "transition-own", "transition-foreign"), 7):
                    distance = x[:, column] / np.maximum(a, 1e-8)
                    p = correction["parameters"]
                    logits[name] = base + a * (p[0] + p[1] * (distance - correction["center"]) / correction["scale"])
                values["transition-distance"] = x[:, 7]
            values.update({name: np.exp(-np.logaddexp(0., -value)) for name, value in logits.items()})
        if model is not None:
            with torch.inference_mode():
                z = torch.as_tensor(standardized(x[:, :7], linear), device=device)
                values["mlp-16"] = torch.sigmoid(model(z)).cpu().numpy()
        for name, probability in values.items():
            add_histogram(result, name, row["building_id"], probability, truth, returning)
    return result


def fit_mlp(matrix, weights, linear, inputs, store, blocks, args):
    identifier = PREFIX + f"followup-mlp-{args.seed}"
    torch.manual_seed(args.seed)
    model = ScoreMLP(linear["parameters"]).to(inputs.device)
    cached = store.read_unit(identifier)
    if cached:
        model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in cached[1].items()})
        return model, cached[0]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
    rng = np.random.default_rng(args.seed)
    best, state, records = None, None, []
    for epoch in memory.progress(range(args.epochs + 1), desc="MLP epochs (0 = linear initialization)",
                                 unit="epoch", position=1):
        loss_total = 0.
        if epoch:
            model.train()
            indices = rng.permutation(len(matrix))
            batches = range(0, len(indices), args.batch_size)
            for start in memory.progress(batches, desc=f"MLP TRAIN epoch={epoch}/{args.epochs}",
                                         unit="batch", position=2):
                selected = indices[start:start + args.batch_size]
                z = torch.as_tensor(standardized(matrix[selected, :7], linear), device=inputs.device)
                y = torch.as_tensor(np.array(matrix[selected, 11]), device=inputs.device)
                w = torch.as_tensor((weights[selected] * len(matrix)).astype(np.float32), device=inputs.device)
                loss = (nn.functional.binary_cross_entropy_with_logits(model(z), y, reduction="none") * w).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                loss_total += float(loss.detach()) * len(selected) / len(matrix)
        hist = evaluate_blocks(store, blocks, "val", inputs.frozen, linear, model, only_mlp=True)
        overall = {b: value[0][None] for b, value in hist["mlp-16"].items()}
        threshold = fusion.select_fpr_thresholds([dict(id="mlp-16", native=False)], overall, {}, .05)["mlp-16"]
        score = (threshold["calibration_recall"], -threshold["calibration_fpr"], -epoch)
        if best is None or score > best:
            best, selected_epoch = score, epoch
            state = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
        record = dict(epoch=epoch, train_bce=loss_total if epoch else None,
                      val_recall=score[0], val_fpr=-score[1], threshold=threshold["threshold"])
        records.append(record)
        memory.write_csv(store.out / (PREFIX + "mlp_epochs.csv"), records)
        memory.tqdm.write(f"mlp seed={args.seed} epoch={epoch}/{args.epochs}: "
                          f"BCE={record['train_bce']}, VAL recall={score[0]:.5f}, FPR={-score[1]:.5f}")
    record = store.save_unit(dict(unit_id=identifier, kind="seven_score_mlp", seed=args.seed, parameters=145,
        selected_epoch=selected_epoch, selection_split="val", fit_split="train", test_used_for_fit=False,
        epoch_zero="exact refitted linear function", all_known_training_pixels_each_epoch=True), state)
    model.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
    return model, record


def followups(inputs, saved, store, args, checkpoint):
    with memory.progress(total=5, desc="Memory follow-ups", unit="stage", position=0, leave=True) as stages:
        ablation = history_ablation(inputs, saved, store, args, checkpoint)
        stages.update(1)
        bank = bank_module.TransitionBank(inputs, store, args, memory.progress)
        bank_record = bank.prepare()
        blocks, coverage = build_blocks(inputs, bank, store)
        memory.write_csv(store.out / (PREFIX + "transition_coverage.csv"), coverage)
        stages.update(1)
        matrix, weights, pixel_record = training_pixels(store, blocks)
        linear = fit_linear(matrix, weights, store, args)
        correction = fit_correction(matrix, weights, inputs.frozen, store, args)
        stages.update(1)
        model, mlp_record = fit_mlp(matrix, weights, linear, inputs, store, blocks, args)
        stages.update(1)
        store.event("followup_models_frozen_before_test", mlp=mlp_record, correction=correction["parameters"].tolist())

        def collect(split):
            return cached_histograms(store, PREFIX + "followup-result-" + split,
                lambda: evaluate_blocks(store, blocks, split, inputs.frozen, linear, model, correction))

        val = collect("val")
        operating = evaluation.fpr_thresholds(val, inputs.frozen, args.fpr_budgets)
        write_json(store.out / (PREFIX + "followup_thresholds.json"), {str(k): v for k, v in operating.items()})
        aggregate, rows, returns = [], [], []
        for split in ("val", "train", "test"):
            hist = val if split == "val" else collect(split)
            inputs.frozen.parity(hist["stack"], split)
            append_reports(inputs.source.data, hist, operating, split, args.seed, "all", aggregate, rows, returns)
        comparisons = paired_differences(rows, [("linear-all", "stack"), ("mlp-16", "linear-all"),
            ("transition-global", "stack"), ("transition-own", "transition-global"),
            ("transition-own", "transition-foreign")], args.seed)
        for name, values in (("metrics", aggregate), ("per_building", rows), ("comparisons", comparisons),
                             ("return_to_past", returns)):
            memory.write_csv(store.out / (PREFIX + "followup_" + name + ".csv"), values)
        stages.update(1)
    summary = dict(status="completed", seed=args.seed, fpr_budgets=args.fpr_budgets,
        ablation=ablation, bank=bank_record, mlp=mlp_record, training_pixels=pixel_record,
        results=aggregate, comparisons=comparisons, feature_extraction_performed=False,
        detector_inference_performed=False, test_previously_inspected=True,
        correction_parameters=correction["parameters"].tolist(),
        transition_memory="PatchCore-inspired signed-difference memory; not the published PatchCore algorithm",
        label_provenance="manual single-image annotations; derived aligned transition targets",
        personal_memory="strict past of same crop; prediction gate; size-matched with one other TRAIN building")
    write_json(store.out / (PREFIX + "followup_summary.json"), summary)
    lines = ["MEMORY FOLLOW-UPS — exploratory, cached DINO and detector maps",
             "TRAIN/VAL/TEST keep the original facade split. No feature/detector extraction.",
             "FPR budgets refer to full-VAL building macro FPR; actual TEST FPR is reported.",
             "One ConvGRU checkpoint: original/common-support/repeated-history; thresholds stay fixed.",
             "Linear/MLP use all known merged TRAIN pixels with equal total building weights.",
             "Transition banks: TRAIN-only global, causal predicted-stable own prefix, matched TRAIN donor.",
             "TRAIN queries exclude their whole building from the global and donor banks.",
             "Global/own/foreign share the same two correction parameters; thresholds chosen separately on VAL.",
             f"MLP selected epoch: {mlp_record['selected_epoch']} (0 = refitted linear initialization).",
             f"Stable-memory admission threshold: {bank.stable_threshold:.6f} (TRAIN only).",
             f"Shared transition correction: offset={correction['parameters'][0]:.6f}, "
             f"slope={correction['parameters'][1]:.6f}; a zero slope ignores bank distances.",
             f"Matched personal memory: {sum(r['matched'] > 0 for r in coverage)}/{len(coverage)} transitions.",
             "Derived change labels; inspected TEST; facade bootstrap is not seed variability."]
    for title, results in (("SAME-CHECKPOINT ABLATION", ablation["results"]), ("FUSION AND TRANSITION BANKS", aggregate)):
        lines.extend(["", title])
        for row in results:
            if row["slice_type"] != "overall":
                continue
            values = " ".join(f"{key}={row[key]:.5f}" if row[key] is not None else f"{key}=NA"
                              for key in ("f1", "iou", "precision", "recall", "fpr", "ap_binned"))
            lines.append(f"VAL-budget={100 * row['val_fpr_budget']:g}% {row['split']:5s} "
                         f"{row['cohort']} {row['method']:20s} {values}")
    report = "\n".join(lines) + "\n"
    (store.out / (PREFIX + "followup_report.txt")).write_text(report, encoding="utf-8")
    memory.tqdm.write(report)
    return summary


def validate_arguments(args):
    if any(value < 1 for value in (args.epochs, args.batch_size, args.bank_per_building,
                                   args.foreign_per_building, args.personal_size)):
        raise ValueError("Epochs, batch size and bank sizes must be positive")
    if not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.l2) or args.l2 <= 0:
        raise ValueError("--lr and --l2 must be finite positive numbers")
    if not 0 < args.min_known_fraction <= 1 or not 0 < args.stable_quantile < 1:
        raise ValueError("Require 0 < --min-known-fraction <= 1 and 0 < --stable-quantile < 1")
    if (not args.fpr_budgets or len(set(args.fpr_budgets)) != len(args.fpr_budgets)
            or any(not np.isfinite(b) or not 0 < b < 1 for b in args.fpr_budgets)):
        raise ValueError("Use distinct fractional FPR budgets, e.g. 0.05 0.10 0.15")


def run(args):
    validate_arguments(args)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import scipy
    with evaluation.SavedMemory(args.run) as saved:
        code = evaluation.validate_code(saved)
        device = args.device or saved.config["device"]
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable in the original worker environment")
            free, total = torch.cuda.mem_get_info(device)
            print(f"Follow-up GPU: {free / 2**30:.2f}/{total / 2**30:.2f} GiB free", flush=True)
            if free < 1024 * 2**20:
                raise ValueError("Less than 1 GiB GPU memory is free; finish other GPU work first")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=device.startswith("cuda"))
        if args.seed not in saved.config["seeds"]:
            raise ValueError("Seed is absent from the original memory training plan")
        checkpoint, _ = saved.read_unit(PREFIX + f"model-convgru-{args.seed}")
        if (checkpoint.get("kind") != "causal_variability_head" or checkpoint.get("mode") != "convgru"
                or checkpoint.get("seed") != args.seed or checkpoint.get("fit_split") != "train"
                or checkpoint.get("selection_split") != "val" or checkpoint.get("test_used_for_fit") is not False
                or not 1 <= checkpoint.get("selected_epoch", 0) <= saved.config["epochs"]):
            raise ValueError("Invalid completed ConvGRU checkpoint")
        source = fusion.Source(saved.config["source"])
        frozen = memory.FrozenStack(saved.config["fusion"], source)
        if source.hashes != saved.config["source_sha256"] or frozen.hashes != saved.config["fusion_sha256"]:
            raise ValueError("Benchmark/fusion differs from the original training inputs")
        inputs = memory.Inputs(source, saved, frozen, device)
        for identifier in memory.progress(sorted(source.data.observations), desc="Verify cached DINO", unit="observation"):
            inputs.feature(identifier)
        out = Path(args.out).expanduser().resolve()
        for original in (saved.out, source.root, source.data.root, source.data.history_root, frozen.root):
            if out == original or out in original.parents or original in out.parents:
                raise ValueError("Output must be separate from every input directory")
        config = {k: v for k, v in vars(args).items() if k not in ("resume", "worker", "out", "run")}
        config.update(memory_run=str(saved.out), input_sha256=dict(saved.hashes), device=device)
        contract = dict(protocol="canonical-memory-followups-v1", config=config,
            code={**code, Path(__file__).name: sha256(__file__), Path(bank_module.__file__).name: sha256(bank_module.__file__),
                  Path(evaluation.__file__).name: sha256(evaluation.__file__)},
            numpy=np.__version__, scipy=scipy.__version__, torch=str(torch.__version__),
            fit_split="train", selection_split="val",
            test_previously_inspected=True)
        if (out / "run.json").exists() and read_json(out / "run.json").get("status") == "completed":
            if read_json(out / "request.json") != config or read_json(out / "experiment_contract.json") != contract:
                raise ValueError("Completed follow-up run has different settings, inputs or code")
            for name, expected in read_json(out / "run.json")["output_sha256"].items():
                if sha256(out / name) != expected:
                    raise ValueError(f"Corrupt follow-up output: {name}")
            print(f"Already completed: {out}", flush=True)
            return read_json(out / (PREFIX + "followup_summary.json"))
        store = fusion.data_module.Store(out, config, resume=args.resume)
        store.record["kind"] = "canonical_memory_followups"
        write_json(out / "run.json", store.record)
        try:
            store.contract(contract)
            summary = followups(inputs, saved, store, args, checkpoint)
            source.verify()
            frozen.verify()
            saved.verify()
            outputs = {str(path.relative_to(out)): sha256(path) for path in sorted(out.rglob("*"))
                       if path.is_file() and (path.name.startswith(PREFIX) or path.parent.name == "units")}
            store.finish("completed", output_sha256=outputs)
            return summary
        except BaseException as exc:
            store.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
            raise
        finally:
            store.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Stopped variability-memory run; existing features and ConvGRU")
    parser.add_argument("--out", required=True, help="New, separate follow-up run")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=10, help="MLP only; original ConvGRU is never retrained")
    parser.add_argument("--device", help="Defaults to original memory device")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--l2", type=float, default=.01)
    parser.add_argument("--fpr-budgets", type=float, nargs="+", default=[.05, .10, .15])
    parser.add_argument("--bank-per-building", type=int, default=256)
    parser.add_argument("--foreign-per-building", type=int, default=512)
    parser.add_argument("--personal-size", type=int, default=128)
    parser.add_argument("--min-known-fraction", type=float, default=.5)
    parser.add_argument("--stable-quantile", type=float, default=.5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        validate_arguments(args)
        if not args.worker:
            request = read_json(Path(args.run).expanduser().resolve() / "request.json")
            record = read_json(Path(request["source"]) / "run.json")
            python = record["config"]["options"]["dinov2"].get("worker_python", sys.executable)
            command = [python, "-s", "-u", "-m", "data_analysis", "memory-followups", "--worker"]
            for key, value in vars(args).items():
                if key == "worker" or value is False or value is None:
                    continue
                command.append("--" + key.replace("_", "-"))
                if value is True:
                    continue
                if key in ("run", "out"):
                    value = str(Path(value).expanduser().resolve())
                command.extend(map(str, value if isinstance(value, list) else [value]))
            print(f"Follow-up worker: {python}", flush=True)
            raise SystemExit(subprocess.run(command, cwd=Path(__file__).resolve().parents[1]).returncode)
        run(args)
    except (ValueError, OSError, KeyError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Ошибка memory-followups: {exc}\n")


if __name__ == "__main__":
    main()
