"""Offline fusion of canonical history maps; only semantic context needs inference."""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import importlib
import itertools
import json
from pathlib import Path

import numpy as np

from facade_change.io import read_json, sha256, write_json
from facade_change.history_features import digest
from facade_change.methods.remote import RemotePool, RemoteScorer

benchmark = importlib.import_module("facade_change.research.2026-11-04_history_benchmark")
metrics, data_module = benchmark.metrics, benchmark.data_module
METHODS, MODES = benchmark.METHODS, benchmark.MODES
DATE = "2026-11-06"
PREFIX = DATE + "_"
SPLITS = ("train", "val", "test")
training = importlib.import_module("data_analysis.2026-11-08_fusion_training")


class Source:
    """Read-only source maps, labels and RGB with their original fingerprints."""

    def __init__(self, path):
        self.root = Path(path).expanduser().resolve()
        self.record = read_json(self.root / "run.json")
        if (self.record.get("kind") != "canonical_history_benchmark"
                or self.record.get("status") != "completed" or self.record.get("pilot")):
            raise ValueError("--run must be a completed full canonical-history benchmark")
        self.hashes = {}
        for name in ("summary", "maps_index", "thresholds"):
            path = self.root / (name + ".json")
            self.hashes[str(path)] = sha256(path)
            if self.record.get(name + "_sha256") != self.hashes[str(path)]:
                raise ValueError(f"Source {name} hash differs from completed run")
        self.hashes[str(self.root / "run.json")] = sha256(self.root / "run.json")
        self.summary = read_json(self.root / "summary.json")
        self.config = self.record["config"]
        self.data = data_module.Dataset(self.config["prepared"])
        self.maps = read_json(self.root / "maps_index.json")
        self.thresholds = read_json(self.root / "thresholds.json")
        self.columns = [(mode, method) for mode in MODES for method in METHODS]
        self.native = [method == "anychange" for mode, method in self.columns]
        expected = set(self.data.events)
        if set(self.maps) != set(METHODS):
            raise ValueError("This comparison needs all seven saved methods")
        assignments = {}
        for sample in self.data.samples:
            building, split = sample["building_id"], sample["split"]
            if split not in SPLITS or assignments.setdefault(building, split) != split:
                raise ValueError("A building occurs in multiple splits")
        self.assignments = assignments
        self.readers = []
        for mode, method in self.columns:
            if set(self.maps[method].get(mode, {})) != expected:
                raise ValueError(f"Missing or mismatched events: {method}/{mode}")
            self.readers.append(benchmark.prediction_reader(self.root, self.maps[method][mode]))
        self.contract = read_json(self.root / "method_contracts/semantic_dino.json")
        self.hashes[str(self.root / "method_contracts/semantic_dino.json")] = sha256(
            self.root / "method_contracts/semantic_dino.json")

    def verify(self):
        self.data.fp.verify()
        for path, expected in self.hashes.items():
            if sha256(path) != expected:
                raise ValueError(f"Source changed: {path}")

    def blocks(self, samples, context):
        @lru_cache(maxsize=24)
        def semantic(observation_id):
            result = context.read_unit(PREFIX + "semantic-" + observation_id)
            if result is None:
                raise ValueError(f"Missing predicted semantic context for {observation_id}")
            return result[1]

        def prediction(row):
            arrays, available = [], None
            for reader in self.readers:
                values, support = reader(row)
                arrays.append(values)
                available = support.copy() if available is None else available & support
            observations = self.data.tracks[row["track_id"]]["observations"]
            first, previous, current = [semantic(observations[i]["observation_id"])
                                        for i in (0, row["frame"] - 1, row["frame"])]
            for reference in (previous, first):
                arrays.extend(np.maximum(reference["probabilities"], current["probabilities"]).astype(np.float32))
                available &= reference["support"] & current["support"]
            return np.stack(arrays), available

        flags = self.native + [False] * (2 * len(self.data.categories))
        yield from metrics.merged_blocks(self.data, samples, prediction, flags)


def verify_semantic_identity(expected, actual):
    """The new side branch must use the same frozen head, backbone and geometry."""
    for key in ("head_checkpoint", "backbone_provenance", "category_to_channel"):
        if key not in expected or key not in actual:
            raise ValueError(f"Semantic provenance lacks {key}")
    if (expected["head_checkpoint"]["sha256"] != actual["head_checkpoint"]["sha256"]
            or expected["backbone_provenance"]["checkpoint"]["sha256"]
            != actual["backbone_provenance"]["checkpoint"]["sha256"]
            or expected["backbone_provenance"]["sources"]["python_tree_sha256"]
            != actual["backbone_provenance"]["sources"]["python_tree_sha256"]
            or expected["category_to_channel"] != actual["category_to_channel"]):
        raise ValueError("Semantic head/backbone/ontology differs from the source run")


def semantic_context(source, store, device=None):
    """Cache each observation once; never send targets to the semantic worker."""
    data = source.data
    pool, scorer = RemotePool(log_dir=store.out / "worker_logs"), None
    done, computed = 0, 0
    try:
        for observation_id in sorted(data.observations):
            unit_id = PREFIX + "semantic-" + observation_id
            cached = store.read_unit(unit_id)
            if cached:
                done += 1
                continue
            if scorer is None:
                args = argparse.Namespace(device=device or source.config.get("device"),
                                          trust_checkpoint=source.config.get("trust_checkpoint", False))
                options, _ = benchmark.scorer_options(
                    "semantic_dino", source.config.get("options", {}), args, data)
                options["export_probabilities"] = True
                scorer = RemoteScorer("semantic_dino", options, pool)
                scorer.activate()
                verify_semantic_identity(source.contract, scorer.metadata)
                benchmark._pin_model_files(scorer.metadata, data.fp)
                write_json(store.out / (PREFIX + "semantic_model.json"), scorer.metadata)
            rgb, support = data.frame(observation_id)
            # The adapter evaluates an identical RGB endpoint only once.
            scorer(rgb, rgb, support)
            auxiliary = scorer.auxiliary_arrays
            a, b = (auxiliary.get(name) for name in
                    ("probabilities_reference", "probabilities_source"))
            shape = (len(data.categories), *support.shape)
            if (a is None or b is None or a.shape != shape or b.shape != shape
                    or not np.isfinite(a[:, support]).all() or not np.isfinite(b[:, support]).all()
                    or np.any(a[:, support] < 0) or np.any(a[:, support] > 1)
                    or np.any(b[:, support] < 0) or np.any(b[:, support] > 1)):
                raise ValueError("Semantic worker did not export valid class probabilities")
            actual_categories = scorer.metadata["categories"]
            channels = {c["id"]: i for i, c in enumerate(actual_categories)}
            order = [channels[c["id"]] for c in data.categories]
            probabilities = a[order].astype(np.float16)
            store.save_unit(dict(unit_id=unit_id, kind="predicted_semantic_context",
                observation_id=observation_id, inputs="one RGB observation only",
                representation="independent sigmoid per class; float16 cache; not GT"),
                dict(probabilities=probabilities, support=support))
            done += 1
            computed += 1
            if done % 25 == 0 or done == len(data.observations):
                print(f"Semantic context: {done}/{len(data.observations)}, new={computed}", flush=True)
    finally:
        if scorer is not None:
            scorer.close()
        pool.release()
    return dict(observations=done, events=len(data.samples), newly_computed=computed, target_labels_used=False)


def training_sample(source, store, limit, seed, context=None):
    context = store if context is None else context
    if limit == 0:
        return training.all_pixels(source, store, context)
    unit_id = PREFIX + "train-sample"
    cached = store.read_unit(unit_id)
    if cached:
        return cached[1]
    rows, labels, buildings = [], [], []
    for building in sorted(b for b, part in source.assignments.items() if part == "train"):
        rng = np.random.default_rng(int(digest([seed, building])[:16], 16))
        chosen_x, chosen_y, keys = None, None, np.empty(0)
        samples = [s for s in source.data.samples if s["building_id"] == building]
        for _, _, scores, truth, geometric, _ in source.blocks(samples, context):
            valid = geometric & (truth[0] != 255)
            x = scores[:, valid].T.astype(np.float32)
            y = truth[0, valid]
            if not len(y):
                continue
            candidate_keys = rng.random(len(y))
            chosen_x = x if chosen_x is None else np.concatenate((chosen_x, x))
            chosen_y = y if chosen_y is None else np.concatenate((chosen_y, y))
            keys = np.concatenate((keys, candidate_keys))
            if len(keys) > limit:
                keep = np.argpartition(keys, limit - 1)[:limit]
                chosen_x, chosen_y, keys = chosen_x[keep], chosen_y[keep], keys[keep]
        if chosen_x is not None:
            order = np.argsort(keys)
            rows.append(chosen_x[order])
            labels.append(chosen_y[order])
            buildings.extend([building] * len(order))
            print(f"TRAIN sample: {building}, {len(order)} unique pixels", flush=True)
    if not labels or len(np.unique(np.concatenate(labels))) != 2:
        raise ValueError("TRAIN sampling needs both changed and stable annotated pixels")
    arrays = dict(x=np.concatenate(rows), y=np.concatenate(labels), buildings=np.array(buildings))
    store.save_unit(dict(unit_id=unit_id, kind="training_sample", split="train",
        sampling="uniform priority reservoir per building after spatial overlap merge",
        per_building_limit=limit, seed=seed), arrays)
    return arrays


def normalize(values, normalizer):
    width = normalizer["high"] - normalizer["low"]
    result = np.clip((values - normalizer["low"]) / np.maximum(width, 1e-8), 0, 1)
    result[:, width < 1e-8] = 0
    return result


def semantic_features(context, kind, groups):
    if kind == "stack":
        return np.empty((len(context), 0), dtype=context.dtype)
    if kind == "stack_classes":
        return context
    return np.column_stack([context[:, channels].max(axis=1) for channels in groups])


def design(scores, context, variant, groups):
    x = scores[:, variant["columns"]]
    if "context_size" in variant:
        start = variant["context_offset"]
        context = context[:, start:start + variant["context_size"]]
    semantic = semantic_features(context, variant["kind"], groups)
    if not semantic.shape[1]:
        return x
    interactions = (x[:, :, None] * semantic[:, None, :]).reshape(len(x), -1)
    return np.concatenate((x, semantic, interactions), axis=1)


def fit_logistic(x, y, sample_weight, l2):
    from scipy.optimize import minimize
    from scipy.special import expit

    x = np.asarray(x, dtype=np.float64)
    weight = sample_weight / sample_weight.sum()
    center = np.sum(x * weight[:, None], axis=0)
    scale = np.sqrt(np.sum((x - center) ** 2 * weight[:, None], axis=0))
    scale = np.maximum(scale, .05)
    z = (x - center) / scale
    prevalence = float(np.dot(weight, y))
    start = np.zeros(z.shape[1] + 1)
    start[0] = np.log(prevalence / (1 - prevalence))

    def objective(parameters):
        logits = parameters[0] + z @ parameters[1:]
        error = weight * (expit(logits) - y)
        loss = np.dot(weight, np.logaddexp(0, logits) - y * logits)
        loss += .5 * l2 * np.dot(parameters[1:], parameters[1:])
        gradient = np.r_[error.sum(), z.T @ error + l2 * parameters[1:]]
        return float(loss), gradient

    fitted = minimize(objective, start, method="L-BFGS-B", jac=True,
                      options=dict(maxiter=300, ftol=1e-10, gtol=1e-6))
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise ValueError(f"Fusion optimization failed: {fitted.message}")
    return dict(center=center, scale=scale, parameters=fitted.x,
                iterations=np.array(fitted.nit), objective=np.array(fitted.fun))


def variants(source, all_pairs=False):
    """Small default matrix; --all-pairs adds all 21 unordered method pairs."""
    sets = [("anychange", "semantic_dino"), ("anychange", "msdzip_abs"),
            ("anychange", "semantic_dino", "msdzip_abs"),
            ("lpips", "dinov2"), ("lpips", "dinov2", "semantic_dino"), METHODS]
    if all_pairs:
        sets.extend(itertools.combinations(METHODS, 2))
    sets = sorted({tuple(sorted(s)) for s in sets})
    result = []
    for mode in MODES:
        for method in METHODS:
            result.append(dict(method=method, mode=mode, kind="single",
                columns=[source.columns.index((mode, method))], native=method == "anychange"))
        for subset in sets:
            columns = [source.columns.index((mode, method)) for method in subset]
            for rule in ("and", "or", "mean", "mean_scaled"):
                result.append(dict(method=rule + "__" + "__".join(subset), mode=mode,
                    kind=rule, columns=columns, native=rule in ("and", "or")))
    for modes in [(m,) for m in MODES] + [("previous", "history")]:
        for kind in ("stack", "stack_groups", "stack_classes"):
            result.append(dict(method=kind, mode="+".join(modes), kind=kind,
                columns=[i for i, (mode, _) in enumerate(source.columns) if mode in modes], native=False,
                context_size=len(source.data.categories),
                context_offset=len(source.data.categories) if modes == ("earliest",) else 0))
    for item in result:
        item["id"] = item["method"] + "--" + item["mode"]
    return result


def fit_models(source, store, specifications, sample, l2):
    class_index = {c["name"]: i for i, c in enumerate(source.data.categories)}
    groups = [[class_index[name] for name in group["categories"]] for group in source.data.groups]
    raw = sample["x"][:, :len(source.columns)]
    # A single column at a time also bounds the quantile workspace for memmaps.
    low, high = np.array([np.quantile(raw[:, i], [.01, .99])
                          for i in range(raw.shape[1])]).T
    for i, native in enumerate(source.native):
        if native:
            low[i], high[i] = 0, 1
    normalizer = dict(low=low, high=high)
    streaming = "ranges" in sample
    if not streaming:
        normalized = normalize(raw, normalizer)
        context = sample["x"][:, len(source.columns):]
        unique, counts = np.unique(sample["buildings"], return_counts=True)
        weights = {b: 1 / n for b, n in zip(unique, counts)}
        sample_weight = np.array([weights[b] for b in sample["buildings"]])
    sample_hash = (sample["pixels_sha256"] if streaming
                   else digest(sample["buildings"].tolist()))
    models = {}
    for item in specifications:
        if not item["kind"].startswith("stack"):
            continue
        unit_id = PREFIX + "model-" + item["id"]
        cached = store.read_unit(unit_id)
        if cached:
            models[item["id"]] = cached[1]
            continue
        print(f"Fit TRAIN: {item['id']}", flush=True)
        if streaming:
            import sys
            model = training.fit_streaming(sample, normalizer, item, groups, sys.modules[__name__], l2)
        else:
            x = design(normalized, context, item, groups)
            model = fit_logistic(x, sample["y"], sample_weight, l2)
        store.save_unit(dict(unit_id=unit_id, kind="logistic_fusion", split="train",
            variant=item, l2=l2, sample_sha256=sample_hash,
            weighting="equal total weight per building; no class rebalancing"), model)
        models[item["id"]] = model
    return normalizer, groups, models


def predict(raw, context, item, normalizer, groups, models, component_thresholds):
    columns = item["columns"]
    kind = item["kind"]
    if kind == "single":
        return raw[:, columns[0]]
    if kind in ("and", "or"):
        binary = raw[:, columns] > component_thresholds[columns]
        return np.all(binary, axis=1) if kind == "and" else np.any(binary, axis=1)
    if kind == "mean":
        return raw[:, columns].mean(axis=1)
    scaled = normalize(raw, normalizer)
    if kind == "mean_scaled":
        return scaled[:, columns].mean(axis=1)
    from scipy.special import expit
    model = models[item["id"]]
    features = design(scaled, context, item, groups)
    logits = model["parameters"][0] + (
        (features - model["center"]) / model["scale"]) @ model["parameters"][1:]
    return expit(logits)


def building_histograms(source, store, specifications, normalizer, groups, models, split, context=None):
    """One building in memory; completed histogram units survive interruption."""
    result, overall = [], {}
    context_reader = store if context is None else context
    thresholds = np.array([source.thresholds[m][mode]["threshold"] for mode, m in source.columns])
    for building in sorted(b for b, part in source.assignments.items() if part == split):
        unit_id = PREFIX + "hist-" + split + "-" + digest(building)[:20]
        cached = store.read_unit(unit_id)
        if cached:
            hist = cached[1]["histograms"]
            if cached[0]["variant_ids"] != [v["id"] for v in specifications]:
                raise ValueError("Cached histogram variant order differs")
        else:
            hist = np.zeros((len(specifications), len(metrics.slices(source.data)), 2, metrics.BINS), np.int64)
            samples = [s for s in source.data.samples if s["building_id"] == building]
            for block_number, (_, _, scores, truth, geometric, _) in enumerate(source.blocks(samples, context_reader), 1):
                valid = geometric & np.any(truth != 255, axis=0)
                if not valid.any():
                    continue
                values, truth = scores[:, valid].T, truth[:, valid]
                raw, context = values[:, :len(source.columns)], values[:, len(source.columns):]
                masks = [(truth == label) for label in (0, 1)]
                for j, item in enumerate(specifications):
                    score = predict(raw, context, item, normalizer, groups, models, thresholds)
                    if not np.isfinite(score).all() or np.any(score < 0) or np.any(score > 1):
                        raise ValueError(f"Invalid fused scores: {item['id']}")
                    quantized = np.ceil(score * (metrics.BINS - 1)).astype(np.int64)
                    for channel in range(len(truth)):
                        for label in (0, 1):
                            hist[j, channel, label] += np.bincount(
                                quantized[masks[label][channel]], minlength=metrics.BINS)
                if block_number % 25 == 0:
                    print(f"Evaluate {split.upper()}: {building}, blocks={block_number}", flush=True)
            store.save_unit(dict(unit_id=unit_id, kind="fusion_histograms", split=split,
                building_id=building, variant_ids=[v["id"] for v in specifications]),
                dict(histograms=hist))
        overall[building] = hist[:, 0].copy()
        result.append((building, unit_id))
        print(f"Evaluate {split.upper()}: {building} ({len(result)} buildings)", flush=True)
    return result, overall


def select_thresholds(source, specifications, overall):
    selected = {}
    for j, item in enumerate(specifications):
        if item["kind"] == "single":
            threshold = source.thresholds[item["method"]][item["mode"]]
        elif item["native"]:
            threshold = dict(threshold=.5, selection_split=None, criterion="frozen component thresholds")
        else:
            histogram = {b: h[j][None] for b, h in overall.items()}
            threshold = metrics.calibrate(histogram)
        selected[item["id"]] = threshold
    return selected


def reports(source, store, specifications, units, selected, split):
    rows = []
    for building, unit_id in units:
        values = store.read_unit(unit_id)[1]["histograms"]
        for i, item in enumerate(specifications):
            report = metrics.report(source.data, {building: values[i]},
                                    selected[item["id"]]["threshold"], item["native"])
            rows.extend(dict(method=item["method"], mode=item["mode"], split=split,
                             **r) for r in report["per_building"])
    aggregates = []
    for item in specifications:
        for kind, name, _ in metrics.slices(source.data):
            subset = [r for r in rows if r["method"] == item["method"] and r["mode"] == item["mode"]
                      and r["slice_type"] == kind and r["slice"] == name]
            total_keys = ("tp", "fp", "fn", "tn", "known_pixels", "positive_pixels", "negative_pixels")
            result = dict(method=item["method"], mode=item["mode"], split=split,
                slice_type=kind, slice=name, threshold=selected[item["id"]]["threshold"],
                buildings=sum(r["known_pixels"] > 0 for r in subset),
                positive_buildings=sum(r["positive_pixels"] > 0 for r in subset))
            for key in total_keys:
                result[key] = sum(r[key] for r in subset)
            for key in metrics.METRICS:
                values = [r[key] for r in subset if r[key] is not None]
                result[key] = float(np.mean(values)) if values else None
                result[key + "_buildings"] = len(values)
            aggregates.append(result)
    return aggregates, rows


def select_fpr_thresholds(specifications, overall, selected, budget=.05):
    """Maximize VAL macro recall subject to a VAL macro FPR budget."""
    output = {}
    for j, item in enumerate(specifications):
        if item["native"]:
            output[item["id"]] = dict(threshold=selected[item["id"]]["threshold"],
                selection_split=None, criterion="fixed binary decision; budget may be exceeded")
            continue
        recall, f1, fpr = [], [], []
        for histogram in overall.values():
            tp, _, _, f, fp = metrics.curves(histogram[j][None])
            positive = histogram[j, 1].sum()
            recall.append(tp[0] / positive if positive else np.full(metrics.BINS, np.nan))
            f1.append(f[0])
            fpr.append(fp[0])
        recall, f1, fpr = (np.nanmean(rows, axis=0) for rows in (recall, f1, fpr))
        eligible = np.flatnonzero(np.isfinite(recall) & np.isfinite(fpr) & (fpr <= budget + 1e-12))
        if not len(eligible):
            raise ValueError("No VAL operating point satisfies the requested FPR budget")
        index = max(eligible, key=lambda i: (recall[i], f1[i], -fpr[i], i))
        output[item["id"]] = dict(threshold=int(index) / (metrics.BINS - 1), selection_split="val",
            criterion="maximum macro recall at VAL macro FPR <= 0.05; F1/FPR/threshold tie breaks",
            calibration_recall=float(recall[index]), calibration_fpr=float(fpr[index]))
    return output


def operating_points(store, specifications, units, selected, split, budget=.05):
    import csv

    values = defaultdict(list)
    details = []
    for building, unit_id in units:
        histograms = store.read_unit(unit_id)[1]["histograms"]
        for i, item in enumerate(specifications):
            result = metrics.metrics(histograms[i, 0], selected[item["id"]]["threshold"], item["native"])
            values[item["id"]].append(result)
            details.append(dict(method=item["method"], mode=item["mode"], split=split,
                building_id=building, threshold=selected[item["id"]]["threshold"], **result))
    with (store.out / ("2026-11-08_fpr05_per_building-" + split + ".csv")).open(
            "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    rows = []
    for item in specifications:
        row = dict(method=item["method"], mode=item["mode"], split=split,
                   threshold=selected[item["id"]]["threshold"], val_fpr_budget=budget,
                   calibrated=not item["native"], buildings=len(units))
        for name in metrics.METRICS:
            numbers = [r[name] for r in values[item["id"]] if r[name] is not None]
            row[name] = float(np.mean(numbers)) if numbers else None
            row[name + "_buildings"] = len(numbers)
        for name in ("f1", "iou"):
            positive = [r[name] for r in values[item["id"]] if r["positive_pixels"] > 0]
            row[name + "_positive"] = float(np.mean(positive)) if positive else None
            row[name + "_positive_buildings"] = len(positive)
        row["actual_fpr_within_budget"] = row["fpr"] is not None and row["fpr"] <= budget + 1e-12
        rows.append(row)
    return rows


def check_baseline_parity(source, aggregates, building_rows):
    lookup = {(r["split"], r["method"], r["mode"], r["slice_type"], r["slice"]): r for r in aggregates}
    buildings = {(r["split"], r["method"], r["mode"], r["building_id"], r["slice_type"], r["slice"]): r
                 for r in building_rows}
    checked = 0
    aliases = {"anychange_and_" + "_and_".join(extra):
               "and__" + "__".join(sorted(("anychange", *extra)))
               for extra in (("semantic_dino",), ("msdzip_abs",), ("semantic_dino", "msdzip_abs"))}
    for result in source.summary["results"] + source.summary.get("fusion_results", []):
        method = aliases.get(result["method"], result["method"])
        prefix = (result["split"], method, result["mode"])
        for original in result["metrics"]["macro"]:
            current = lookup[(*prefix, original["slice_type"], original["slice"])]
            for key in metrics.METRICS:
                a, b = original[key], current[key]
                if (a is None) != (b is None) or (a is not None and not np.isclose(a, b, atol=1e-10, rtol=0)):
                    raise ValueError(f"Single-method parity failed: {prefix}/{original['slice']}/{key}")
            checked += 1
        for original in result["metrics"]["per_building"]:
            current = buildings[(*prefix, original["building_id"], original["slice_type"], original["slice"])]
            if any(current[k] != original[k] for k in ("tp", "fp", "fn", "tn")):
                raise ValueError(f"Per-building baseline counts changed: {prefix}")
    return checked


def comparisons(aggregates, rows, seed):
    """Paired building bootstrap; thresholds and comparison identities stay frozen."""
    overall = [r for r in aggregates if r["slice_type"] == "overall"]
    val = [r for r in overall if r["split"] == "val" and r["method"] in METHODS]
    best = max(val, key=lambda r: (r["f1"], -r["fpr"], r["method"], r["mode"]))
    baseline = (best["method"], best["mode"])
    pairs = []
    identities = sorted({(r["method"], r["mode"]) for r in overall})
    pairs.extend((identity, baseline, "versus_VAL_selected_single") for identity in identities if identity != baseline)
    pairs.extend(((kind, mode), ("stack", mode), "semantic_ablation")
                 for kind in ("stack_groups", "stack_classes") for mode in (*MODES, "previous+history"))
    pairs.extend(((kind, "previous+history"), (kind, "previous"), "history_ablation")
                 for kind in ("stack", "stack_groups", "stack_classes"))
    output = []
    for candidate, reference, reason in pairs:
        for split in SPLITS:
            selected = [r for r in rows if r["split"] == split and r["slice_type"] == "overall"]
            left = {r["building_id"]: r["f1"] for r in selected
                    if (r["method"], r["mode"]) == candidate and r["f1"] is not None}
            right = {r["building_id"]: r["f1"] for r in selected
                     if (r["method"], r["mode"]) == reference and r["f1"] is not None}
            buildings = sorted(left.keys() & right.keys())
            if not buildings:
                continue
            delta = np.array([left[b] - right[b] for b in buildings])
            rng = np.random.default_rng(int(digest([seed, candidate, reference, split])[:16], 16))
            boot = delta[rng.integers(len(delta), size=(2000, len(delta)))].mean(axis=1)
            interval = np.quantile(boot, [.025, .975])
            output.append(dict(method=candidate[0], mode=candidate[1], split=split,
                reference_method=reference[0], reference_mode=reference[1], reason=reason,
                buildings=len(buildings), f1_delta=float(delta.mean()),
                ci_low=float(interval[0]), ci_high=float(interval[1]),
                improved_buildings=int((delta > 1e-12).sum()), worsened_buildings=int((delta < -1e-12).sum()),
                uncertainty="paired building bootstrap; descriptive, no multiple-comparison correction"))
    return output, dict(method=baseline[0], mode=baseline[1], selected_on="VAL")


def write_outputs(store, aggregates, rows, deltas, baseline, selection, operating):
    import csv

    for name, data in (("metrics", aggregates), ("per_building", rows), ("comparisons", deltas),
                       ("fpr05", operating)):
        path = store.out / (PREFIX + name + ".csv")
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    def value(v):
        return "NA" if v is None else f"{100 * v:.2f}"
    lines = ["S192 FUSION: TRAIN / VAL / TEST",
             "Fit: TRAIN; output thresholds: VAL; all TEST results are reported.",
             "Scores below are percentages, equal building weights; AP is binned.",
             "Classes/groups: annotation-conditioned change localisation, not semantic mIoU.",
             "Semantic routing: predicted sigmoid context; no GT class enters prediction.",
             f"VAL-selected single baseline: {baseline['method']} / {baseline['mode']}",
             f"VAL-selected fusion: {selection['method']} / {selection['mode']}",
             "Bootstrap intervals are exploratory and not corrected for multiple comparisons.", ""]
    for split in SPLITS:
        lines.append(split.upper() + " | OVERALL")
        for row in aggregates:
            if row["split"] == split and row["slice_type"] == "overall":
                lines.append(f"{row['method']:78s} {row['mode']:17s} "
                             + " ".join(f"{value(row[k]):>7s}" for k in metrics.METRICS))
        lines.append("")
    lines.insert(9, "Columns: method, mode, F1, IoU, precision, recall, FPR, AP")
    for item in deltas:
        if item["reason"] in ("semantic_ablation", "history_ablation"):
            lines.append(f"{item['split']} {item['method']}/{item['mode']} vs "
                         f"{item['reference_method']}/{item['reference_mode']}: "
                         f"delta F1={100 * item['f1_delta']:+.2f} pp; "
                         f"95% CI [{100 * item['ci_low']:+.2f}, {100 * item['ci_high']:+.2f}]")
    lines.extend(["", "Operating points selected at VAL FPR <= 5% are in " + PREFIX + "fpr05.csv.",
                  "Their actual TRAIN/TEST FPR is reported; the budget is not imposed using TEST labels."])
    text = "\n".join(lines) + "\n"
    (store.out / (PREFIX + "report.txt")).write_text(text, encoding="utf-8")
    print(text, flush=True)


def run(args):
    try:
        import scipy
        from scipy.optimize import minimize  # Validate dependency before GPU work.
    except ImportError as exc:
        raise ValueError("SciPy is required; run in the existing research Python environment") from exc
    source = Source(args.run)
    out = Path(args.out).expanduser().resolve()
    if out == source.root or out in source.root.parents or source.root in out.parents:
        raise ValueError("Output must be separate from the original run")
    if out == source.data.root or out in source.data.root.parents or source.data.root in out.parents:
        raise ValueError("Output must be separate from preparation")
    if args.sample_per_building < 0 or args.l2 <= 0:
        raise ValueError("--sample-per-building must be >= 0 (0 means all); --l2 must be positive")
    config = dict(source=str(source.root), source_sha256=source.hashes,
        seed=args.seed, sample_per_building=args.sample_per_building, l2=args.l2,
        semantic_device=args.semantic_device, all_pairs=args.all_pairs)
    cached_context = None
    if getattr(args, "semantic_cache", None):
        cached_context = training.SemanticCache(args.semantic_cache, source)
        if out == cached_context.out or out in cached_context.out.parents or cached_context.out in out.parents:
            raise ValueError("Output must be separate from the read-only semantic cache")
        config.update(semantic_cache=str(cached_context.out), semantic_cache_sha256=cached_context.hashes)
    if (out / "run.json").exists() and read_json(out / "run.json").get("status") == "completed":
        if read_json(out / "request.json") != config:
            raise ValueError("Completed output has different inputs/options; choose a new --out")
        result = read_json(out / "run.json")
        if result.get("summary_sha256") != sha256(out / (PREFIX + "summary.json")):
            raise ValueError("Completed summary is corrupt")
        print(f"Already completed: {out / (PREFIX + 'report.txt')}", flush=True)
        return read_json(out / (PREFIX + "summary.json"))
    store = data_module.Store(out, config, resume=args.resume)
    store.record["kind"] = "canonical_history_fusion"
    write_json(out / "run.json", store.record)
    try:
        specifications = variants(source, args.all_pairs)
        print(f"Fusion variants: {len(specifications)}; semantic observations: "
              f"{len(source.data.observations)}; TRAIN/VAL/TEST buildings: "
              f"{[sum(p == split for p in source.assignments.values()) for split in SPLITS]}", flush=True)
        semantic_file = Path(benchmark.__file__).parents[1] / "methods/2026-10-14_semantic_dino.py"
        store.contract(dict(protocol="canonical-history-fusion-v1", config=config,
            prepared_sha256=source.data.fp.files[str(source.data.root / "index.json")],
            code={Path(__file__).name: sha256(__file__), Path(metrics.__file__).name: sha256(metrics.__file__),
                  semantic_file.name: sha256(semantic_file), Path(training.__file__).name: sha256(training.__file__)},
            numeric_environment=dict(numpy=np.__version__, scipy=scipy.__version__),
            variants=specifications, fit_split="train", threshold_split="val",
            test_previously_inspected=True, checkpoint_exposure=source.summary.get("checkpoint_exposure", {}),
            semantic_input="max endpoint sigmoid; EARLIEST uses first/current, others previous/current",
            test_used_for_fit=False, target_scope=source.summary["target_scope"]))
        if cached_context:
            context = dict(observations=len(source.data.observations), events=len(source.data.samples),
                           newly_computed=0, target_labels_used=False, reused_from=str(cached_context.out))
        else:
            context = semantic_context(source, store, args.semantic_device)
        context_reader = cached_context or store
        sample = training_sample(source, store, args.sample_per_building, args.seed, context_reader)
        normalizer, groups, models = fit_models(source, store, specifications, sample, args.l2)
        write_json(out / (PREFIX + "normalizer.json"),
                   {k: v.tolist() for k, v in normalizer.items()})
        val_units, overall = building_histograms(
            source, store, specifications, normalizer, groups, models, "val", context_reader)
        selected = select_thresholds(source, specifications, overall)
        write_json(out / (PREFIX + "thresholds.json"), selected)
        fpr_selected = select_fpr_thresholds(specifications, overall, selected)
        write_json(out / (PREFIX + "fpr05_thresholds.json"), fpr_selected)
        operating = operating_points(store, specifications, val_units, fpr_selected, "val")
        aggregates, rows = reports(source, store, specifications, val_units, selected, "val")
        eligible = [r for r in aggregates if r["slice_type"] == "overall" and r["method"] not in METHODS]
        winner = max(eligible, key=lambda r: (r["f1"], -r["fpr"], r["method"], r["mode"]))
        selection = dict(method=winner["method"], mode=winner["mode"], selected_on="VAL",
                         criterion="building macro F1; minimum FPR; deterministic name tie break")
        write_json(out / (PREFIX + "selection.json"), selection)
        store.event("selection_frozen_before_test", selection=selection)
        for split in ("train", "test"):
            units, _ = building_histograms(source, store, specifications, normalizer, groups, models, split,
                                          context_reader)
            a, r = reports(source, store, specifications, units, selected, split)
            aggregates.extend(a)
            rows.extend(r)
            operating.extend(operating_points(store, specifications, units, fpr_selected, split))
        parity = check_baseline_parity(source, aggregates, rows)
        deltas, baseline = comparisons(aggregates, rows, args.seed)
        source.verify()
        if cached_context:
            cached_context.verify()
        summary = dict(kind="canonical_history_fusion", status="completed",
            source=str(source.root), source_sha256=source.hashes, variants=len(specifications),
            rows=len(aggregates), context=context, selection=selection, baseline=baseline,
            baseline_parity_rows=parity, fit_split="train", threshold_split="val",
            checkpoint_exposure=source.summary.get("checkpoint_exposure", {}),
            independent_event_labels=False, test_previously_inspected=True,
            results=aggregates, comparisons=deltas, fpr05=operating)
        write_json(out / (PREFIX + "summary.json"), summary)
        write_outputs(store, aggregates, rows, deltas, baseline, selection, operating)
        store.finish("completed", summary_sha256=sha256(out / (PREFIX + "summary.json")),
                     variants=len(specifications), rows=len(aggregates), baseline_parity_rows=parity)
        return summary
    except BaseException as exc:
        store.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        store.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Completed canonical history benchmark")
    parser.add_argument("--out", required=True, help="New analysis directory, separate from source")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--semantic-device", help="Default: same device as the source run")
    parser.add_argument("--sample-per-building", type=int, default=4000, help="0: all known TRAIN pixels")
    parser.add_argument("--semantic-cache", help="Read-only completed fusion run; prohibit semantic inference")
    parser.add_argument("--l2", type=float, default=.01, help="Fixed TRAIN ridge penalty, not TEST tuned")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--all-pairs", action="store_true", help="Add every unordered pair of seven methods")
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"Ошибка fusion: {exc}\n")


if __name__ == "__main__":
    main()
