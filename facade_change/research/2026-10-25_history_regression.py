"""Masked class-fraction supervision from completed, projected annotation targets.

Fractions describe annotation differences on comparable annotated pixels only.
They are neither whole-crop change probabilities nor independently reviewed events.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import sys

import numpy as np

from ..history_features import _inside, _npz, progress
from ..io import read_json, sha256, write_json


history = importlib.import_module("facade_change.research.2026-10-25_history_models")
SCOPE = "per-class annotation difference fraction on comparable annotated support; outside unknown"
IDENTITY = ("event_id", "prefix_sha256", "track_id", "building_id", "view_id", "split",
            "previous_year", "current_year")


def completed_index(path, kind, fingerprints):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "index.json"
    record_path = path.parent / "run.json"
    fingerprints.add(record_path)
    record = read_json(record_path)
    expected = record.get("artifact_sha256", {}).get(path.name)
    if record.get("status") != "completed" or not expected:
        raise ValueError("Expected a completed, hash-bound target export")
    fingerprints.add(path, expected)
    document = read_json(path)
    if document.get("kind") != kind:
        raise ValueError(f"Expected {kind}, got {document.get('kind')}")
    return path.parent, document


def fractions_from_maps(layers, min_known_pixels=64, min_known_fraction=.001):
    if (layers.dtype != np.uint8 or layers.ndim != 3 or min(layers.shape) < 1
            or not np.isin(layers, [0, 1, 255]).all()):
        raise ValueError("Class targets must be uint8 C,H,W with values 0/1/255")
    known = (layers != 255).sum(axis=(1, 2))
    changed = (layers == 1).sum(axis=(1, 2))
    valid = (known >= min_known_pixels) & (known / np.prod(layers.shape[1:]) >= min_known_fraction)
    values = np.full(len(known), np.nan, dtype=np.float32)
    values[valid] = changed[valid] / known[valid]
    return {"fractions": values, "valid": valid, "known_pixels": known,
            "changed_pixels": changed, "crop_pixels": np.full(len(known), np.prod(layers.shape[1:]))}


def coverage(samples, categories):
    result = {}
    for split in sorted({s["split"] for s in samples}):
        rows = [s for s in samples if s["split"] == split]
        classes = []
        for c, category in enumerate(categories):
            usable = [s for s in rows if s["valid"][c]]
            values = np.array([s["fractions"][c] for s in usable])
            classes.append({**category, "valid": len(usable), "unknown": len(rows)-len(usable),
                            "zero": int((values == 0).sum()), "nonzero": int((values > 0).sum()),
                            "buildings": len({s["building_id"] for s in usable}),
                            "min": float(values.min()) if len(values) else None,
                            "max": float(values.max()) if len(values) else None,
                            "constant": bool(np.ptp(values) == 0) if len(values) else None})
        result[split] = {"events": len(rows), "usable_events": sum(bool(s["valid"].any()) for s in rows),
                         "classes": classes}
    return result


def prepare(args):
    data = history.load_histories(args.features, args.splits, args.min_years, args.history_root)
    root, source = completed_index(args.annotation_run, "history_annotation_targets", data.fingerprints)
    channels = source["contract"]["target_channels"]
    categories = [{k: source["contract"]["categories"][c][k] for k in ("id", "name")} for c in channels]
    if not categories or len({c["id"] for c in categories}) != len(categories):
        raise ValueError("Invalid annotation category mapping")
    lookup = {r["event_id"]: r for r in source["events"]}
    if len(lookup) != len(source["events"]):
        raise ValueError("Duplicate source event ID")
    store = history.HistoryRun(Path(args.out), vars(args).copy())
    try:
        store.contract({"inputs": dict(data.fingerprints.files), "module_sha256": sha256(__file__),
                        "scope": SCOPE, "categories": categories,
                        "denominator": "per-class nonignored pixels; minimum fraction is relative to crop area",
                        "rgb_recomputed": False, "dino_recomputed": False,
                        "source_binary_labels_used": False, "reviewed_event_gt_created": False})
        samples, missing = [], 0
        for sample in progress(data.samples, "Summarize cached class targets"):
            old = lookup.get(sample["event_id"])
            if old is None:
                missing += 1
                continue
            if any(old[k] != sample[k] for k in IDENTITY):
                raise ValueError(f"Stale or mismatched annotation event: {sample['event_id']}")
            path = _inside(root, old["target_path"])
            data.fingerprints.add(path, old["target_sha256"])
            with np.load(path, allow_pickle=False) as archive:
                if archive["category_ids"].tolist() != [c["id"] for c in categories]:
                    raise ValueError("Target channel/category order mismatch")
                targets = fractions_from_maps(archive["class_targets"], args.min_known_pixels, args.min_known_fraction)
            if len(targets["fractions"]) != len(categories):
                raise ValueError("Target layer count differs from ontology")
            samples.append({**{k: sample[k] for k in IDENTITY}, **targets})
        if not samples:
            raise ValueError("No matching source events; export annotation targets for the selected splits first")
        splits = {}
        for split in args.splits:
            rows = [s for s in samples if s["split"] == split]
            if not rows:
                continue
            path = store.out / f"{split}.npz"
            _npz(path, {"event_ids": np.array([s["event_id"] for s in rows]),
                        "category_ids": np.array([c["id"] for c in categories]),
                        **{k: np.stack([s[k] for s in rows]) for k in
                           ("fractions", "valid", "known_pixels", "changed_pixels", "crop_pixels")}})
            splits[split] = {"path": path.name, "sha256": sha256(path),
                              "events": [{k: s[k] for k in IDENTITY} for s in rows]}
        summary = {"scope": SCOPE, "missing_source_events": missing, "by_split": coverage(samples, categories),
                   "reviewed_event_gt_created": False, "unknown_is_zero": False}
        write_json(store.out / "index.json", {"schema_version": 1, "kind": "history_class_fractions",
                   "categories": categories, "splits": splits, "scope": SCOPE,
                   "min_years": args.min_years, "min_known_pixels": args.min_known_pixels,
                   "min_known_fraction": args.min_known_fraction, "annotation_run": str(root),
                   "annotation_index_sha256": sha256(root / "index.json"),
                   "input_sha256": data.fingerprints.files})
        history.finish_run(store, data, summary)
        return {"out": str(store.out), **summary}
    except BaseException as exc:
        store.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        store.close()


def load_targets(data, path):
    root, index = completed_index(path, "history_class_fractions", data.fingerprints)
    categories = index["categories"]
    if (index.get("schema_version") != 1 or index.get("scope") != SCOPE or not categories
            or len({c["id"] for c in categories}) != len(categories)
            or len({c["name"] for c in categories}) != len(categories)):
        raise ValueError("Invalid class-fraction schema or ontology")
    lookup = {r["event_id"]: r for r in data.samples}
    result, seen = [], set()
    # Files are split-specific: training never opens the TEST target array.
    for split in data.coverage["splits"]:
        info = index["splits"].get(split)
        if info is None:
            continue
        path = _inside(root, info["path"])
        data.fingerprints.add(path, info["sha256"])
        with np.load(path, allow_pickle=False) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        rows = info["events"]
        shape = (len(rows), len(categories))
        if (arrays["event_ids"].tolist() != [r["event_id"] for r in rows]
                or arrays["category_ids"].tolist() != [c["id"] for c in categories]
                or any(arrays[k].shape != shape for k in
                       ("fractions", "valid", "known_pixels", "changed_pixels", "crop_pixels"))
                or arrays["valid"].dtype != bool or arrays["fractions"].dtype != np.float32):
            raise ValueError("Invalid fraction array shape, dtype or identity")
        y, valid, known, changed, pixels = (arrays[k] for k in
                                            ("fractions", "valid", "known_pixels", "changed_pixels", "crop_pixels"))
        if (any(a.dtype.kind not in "iu" for a in (known, changed, pixels))
                or np.any(pixels <= 0) or np.any(known < 0) or np.any(known > pixels)
                or np.any(changed < 0) or np.any(changed > known)
                or not np.array_equal(valid, (known >= index["min_known_pixels"]) &
                                      (known / pixels >= index["min_known_fraction"]))
                or not np.isnan(y[~valid]).all() or not np.isfinite(y[valid]).all()
                or not np.allclose(y[valid], changed[valid] / known[valid], atol=1e-7)):
            raise ValueError("Invalid fractions, denominator counts or availability mask")
        for i, row in enumerate(rows):
            if row["event_id"] in seen or row["split"] != split:
                raise ValueError("Duplicate event or target split mismatch")
            seen.add(row["event_id"])
            sample = lookup.get(row["event_id"])
            if sample is None:
                continue
            if any(sample[k] != row[k] for k in IDENTITY):
                raise ValueError("Stale class target prefix or metadata mismatch")
            result.append({**sample, **{k: arrays[k][i] for k in
                                       ("fractions", "valid", "known_pixels", "changed_pixels", "crop_pixels")}})
    return result, categories, {"missing_events": len(set(lookup)-seen), "by_split": coverage(result, categories)}


def loss_weights(samples):
    """Equal class weight; within each class equal building weight, then event weight."""
    valid = np.stack([s["valid"] for s in samples])
    weights = np.zeros(valid.shape, dtype=np.float64)
    active = valid.any(axis=0).sum()
    if not active:
        raise ValueError("No comparable class targets")
    for c in range(valid.shape[1]):
        counts = Counter(s["building_id"] for s, ok in zip(samples, valid[:, c]) if ok)
        for i, s in enumerate(samples):
            if valid[i, c]:
                weights[i, c] = len(samples) / (active * len(counts) * counts[s["building_id"]])
    return weights


def loss(logits, rows, weights, torch):
    values = np.stack([np.where(s["valid"], s["fractions"], 0) for s in rows])
    target = torch.as_tensor(values, dtype=torch.float32, device=logits.device)
    weight = torch.as_tensor(weights, dtype=torch.float32, device=logits.device)
    return ((logits.sigmoid() - target).square() * weight).sum(1).mean()


def build_model(spec):
    import torch
    networks = importlib.import_module(history.NETWORK_MODULE)
    class FractionHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = networks.HistoryClassifier(**{k: v for k, v in spec.items() if k != "outputs"})
            self.backbone.head[-1] = torch.nn.Linear(spec["width"], spec["outputs"])

        def forward(self, **batch):
            output, embedding = self.backbone(**batch)
            return output.reshape(-1, spec["outputs"]), embedding
    return FractionHead()


def report_metrics(samples, predictions):
    y, valid = (np.stack([s[k] for s in samples]) for k in ("fractions", "valid"))
    predictions = np.asarray(predictions)
    if predictions.shape != y.shape or not np.isfinite(predictions[valid]).all():
        raise ValueError("Nonfinite or incorrectly shaped fraction predictions")
    weights = loss_weights(samples)
    error = np.where(valid, predictions - y, 0)
    columns = []
    for c in range(y.shape[1]):
        w = weights[:, c]
        total = w.sum()
        columns.append({"mae": float((w * np.abs(error[:, c])).sum()/total) if total else None,
                        "rmse": float(np.sqrt((w * error[:, c]**2).sum()/total)) if total else None,
                        "events": int(valid[:, c].sum()),
                        "buildings": len({s["building_id"] for s, ok in zip(samples, valid[:, c]) if ok})})
    return {"macro_building_mae": float(np.mean([r["mae"] for r in columns if r["mae"] is not None])),
            "macro_building_rmse": float(np.mean([r["rmse"] for r in columns if r["rmse"] is not None])),
            "per_class": columns, "reference_scope": SCOPE,
            "reference_kind": "annotation_derived", "independent_temporal_event_test": False}


def model_result(samples, scores):
    return {"val": report_metrics(samples, scores), "objective": "class_fractions"}, {
        "predicted_fractions": scores, "target_fractions": np.stack([s["fractions"] for s in samples]),
        "valid": np.stack([s["valid"] for s in samples]),
        "known_pixels": np.stack([s["known_pixels"] for s in samples])}


def fit_baselines(samples):
    y, weights = np.stack([s["fractions"] for s in samples]), loss_weights(samples)
    result = {"zero": {"constant": [0.] * y.shape[1]}, "train_mean": {"constant": []},
              "train_median": {"constant": []}, **{name: {"coefficients": []} for name in history.SCORES}}
    cosine = [history.cosine_scores(s) for s in samples]
    for c in range(y.shape[1]):
        take = weights[:, c] > 0
        v, w = y[take, c], weights[take, c]
        result["train_mean"]["constant"].append(float(np.average(v, weights=w)) if len(v) else 0.)
        order = np.argsort(v)
        median = float(v[order][np.searchsorted(np.cumsum(w[order]), w.sum()/2)]) if len(v) else 0.
        result["train_median"]["constant"].append(median)
        for name in history.SCORES:
            x = np.array([[1., row[name]] for row in cosine])[take]
            coefficients = np.linalg.lstsq(x * np.sqrt(w[:, None]), v * np.sqrt(w), rcond=None)[0] if len(v) else np.zeros(2)
            result[name]["coefficients"].append(coefficients.tolist())
    return result


def baseline_predictions(info, samples):
    if "constant" in info:
        return np.tile(info["constant"], (len(samples), 1))
    values = np.array([[1., history.cosine_scores(s)[info["score"]]] for s in samples])
    return np.clip(values @ np.asarray(info["coefficients"]).T, 0, 1)


def usable_rows(samples, trained):
    rows = [{**s, "valid": s["valid"] & trained} for s in samples]
    return [s for s in rows if s["valid"].any()]


def train_models(args):
    data = history.load_histories(args.features, ("train", "val"), args.min_years, args.history_root)
    samples, categories, cover = load_targets(data, args.class_targets)
    train = [s for s in samples if s["split"] == "train" and s["valid"].any()]
    if not train:
        raise ValueError("TRAIN has no usable class fractions; inspect the fraction summary")
    trained = np.stack([s["valid"] for s in train]).any(axis=0)
    val = usable_rows([s for s in samples if s["split"] == "val"], trained)
    if not val:
        raise ValueError("VAL has no usable targets for trained classes; inspect the fraction summary")
    out = Path(args.out)
    if args.resume and (out / "run.json").exists() and read_json(out / "run.json").get("status") == "completed":
        raise ValueError("Training already completed; use a new output directory")
    store = history.HistoryRun(out, {k: v for k, v in vars(args).items() if k != "resume"}, resume=args.resume)
    try:
        import torch
        networks = importlib.import_module(history.NETWORK_MODULE)
        contract = {"inputs": data.fingerprints.files, "module_sha256": sha256(__file__),
                    "trainer_sha256": sha256(history.__file__), "networks_sha256": sha256(networks.__file__),
                    "torch": str(torch.__version__), "numpy": np.__version__, "objective": "class_fractions",
                    "loss": "masked MSE; equal classes, then buildings, then valid events",
                    "primary_metric": "macro_building_mae", "categories": categories, "scope": SCOPE,
                    "test_used": False, "train_event_ids": [s["event_id"] for s in train],
                    "val_event_ids": [s["event_id"] for s in val]}
        store.contract(contract)
        history.torch_runtime(args.device, args.seed)
        write_json(store.out / "coverage.json", {"features": data.coverage, "targets": cover,
                                                 "trained_classes": trained.tolist()})
        baselines = fit_baselines(train)
        for name, info in baselines.items():
            if name in history.SCORES:
                info["score"] = name
            info["val"] = report_metrics(val, baseline_predictions(info, val))
        models = {}
        for model in args.models:
            for time_mode in args.time_modes:
                torch.manual_seed(args.seed)
                spec = {"dimension": data.index["encoder"]["dimensions"], "model": model,
                        "width": args.width, "dropout": args.dropout, "time_mode": time_mode,
                        "outputs": len(categories)}
                models[f"{model}_{time_mode}"] = history.train_one(
                    store, train, val, spec, args, torch, objective=sys.modules[__name__])
        bundle = {"schema_version": 1, "objective": "class_fractions", "models": models,
                  "baselines": baselines, "categories": categories, "trained_classes": trained.tolist(),
                  "networks_sha256": contract["networks_sha256"], "regression_sha256": sha256(__file__),
                  "encoder": data.index["encoder"], "min_years": args.min_years, "seed": args.seed,
                  "building_splits": {t["building_id"]: t["split"] for t in data.tracks.values()},
                  "scope": SCOPE, "test_used": False}
        write_json(store.out / "models.json", bundle)
        write_json(store.out / "metrics.json", {"scope": SCOPE, "categories": categories,
                   "primary_metric": "macro_building_mae", "used_for_checkpoint_selection": True,
                   "splits": {"val": {name: info["val"] for name, info in {**baselines, **models}.items()}},
                   "independent_temporal_event_test": False})
        history.finish_run(store, data, {"models": list(models), "targets": cover,
                           "selection": "minimum VAL macro_building_mae", "scope": SCOPE,
                           "test_used": False, "trained_classes": trained.tolist()})
        return {"out": str(store.out), "models": list(models), "scope": SCOPE}
    except BaseException as exc:
        store.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        store.close()


def paired_intervals(samples, predictions, repeats, seed):
    """Paired building bootstrap of the same class-macro MAE; models stay fixed."""
    groups = sorted({s["building_id"] for s in samples})
    if len(groups) < 2 or repeats == 0:
        return {"unit": "building_id", "reason": "fewer than two buildings or bootstrap disabled"}
    per_building = {}
    for name, values in predictions.items():
        columns = []
        for building in groups:
            ids = [i for i, s in enumerate(samples) if s["building_id"] == building]
            rows = [samples[i] for i in ids]
            metrics = report_metrics(rows, values[ids])
            columns.append([r["mae"] if r["mae"] is not None else np.nan for r in metrics["per_class"]])
        per_building[name] = np.array(columns)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), (repeats, len(groups)))
    estimates, replicates = {}, {}
    for name, errors in per_building.items():
        active = np.isfinite(errors).any(axis=0)
        errors = errors[:, active]
        estimates[name] = float(np.nanmean(errors, axis=0).mean())
        bootstrap = []
        for draw in draws:
            selected = errors[draw]
            bootstrap.append(float(np.nanmean(selected, axis=0).mean())
                             if np.isfinite(selected).any(axis=0).all() else np.nan)
        replicates[name] = np.array(bootstrap)
    result = {"unit": "building_id", "metric": "macro_building_mae", "comparisons": {}}
    for name in predictions:
        refs = []
        if "_" in name and name not in {"train_mean", "train_median"}:
            reference = "pair_" + name.rsplit("_", 1)[1]
            if reference in predictions and reference != name:
                refs.append(reference)
        if name.endswith("_gaps") and name[:-5] + "_none" in predictions:
            refs.append(name[:-5] + "_none")
        for ref in refs:
            delta = replicates[name] - replicates[ref]
            delta = delta[np.isfinite(delta)]
            result["comparisons"][f"{name}_minus_{ref}"] = {
                "difference": estimates[name]-estimates[ref],
                "percentile_95": np.quantile(delta, [.025, .975]).tolist() if len(delta) else None,
                "valid_replicates": len(delta), "negative_favors_first": True}
    return result


def evaluate_models(args, run, bundle):
    if args.labels:
        raise ValueError("Fraction models use --class-targets; event CSV labels define a different task")
    data = history.load_histories(args.features, args.splits, bundle["min_years"], args.history_root)
    if data.index["encoder"] != bundle["encoder"]:
        raise ValueError("Feature encoder differs from training")
    for t in data.tracks.values():
        if bundle["building_splits"].get(t["building_id"], t["split"]) != t["split"]:
            raise ValueError("A training building changed splits")
    trained = np.asarray(bundle["trained_classes"], dtype=bool)
    categories = bundle["categories"]
    cover = {"status": "unlabelled inference; no metrics"}
    if args.class_targets:
        samples, current_categories, cover = load_targets(data, args.class_targets)
        if current_categories != categories:
            raise ValueError("Evaluation ontology differs from training")
        samples = usable_rows(samples, trained)
    else:
        samples = data.samples
    if not samples:
        raise ValueError("No usable evaluation events")
    networks = importlib.import_module(history.NETWORK_MODULE)
    if sha256(networks.__file__) != bundle["networks_sha256"] or sha256(__file__) != bundle["regression_sha256"]:
        raise ValueError("Regression implementation changed since training")
    data.fingerprints.add(run / "run.json")
    data.fingerprints.add(run / "models.json")
    for info in bundle["models"].values():
        data.fingerprints.add(_inside(run, info["checkpoint"]), info["checkpoint_sha256"])
    store = history.HistoryRun(Path(args.out), vars(args).copy())
    try:
        import torch
        store.contract({"inputs": data.fingerprints.files, "module_sha256": sha256(__file__),
                        "scope": SCOPE, "categories": categories, "training_parameters_frozen": True})
        history.torch_runtime(args.device, bundle["seed"])
        predictions = {name: baseline_predictions(info, samples) for name, info in bundle["baselines"].items()}
        shuffled = {}
        for name, info in bundle["models"].items():
            model = build_model(info["spec"]).to(args.device)
            payload = torch.load(_inside(run, info["checkpoint"]), map_location="cpu", weights_only=True)
            if payload["spec"] != info["spec"]:
                raise ValueError("Checkpoint specification mismatch")
            model.load_state_dict(payload["state_dict"], strict=True)
            values, embeddings = history.predict(model, samples, args.device, args.batch_size)
            values[:, ~trained] = np.nan
            predictions[name] = values
            _npz(store.out / f"{name}_embeddings.npz", {"event_ids": np.array([s["event_id"] for s in samples]),
                 "history_through_year": np.array([s["previous_year"] for s in samples]),
                 "category_ids": np.array([c["id"] for c in categories]), "trained_classes": trained,
                 "predicted_fractions": values, "history_embeddings": embeddings})
            if info["spec"]["model"] in {"gru", "transformer"} and args.class_targets:
                ids = [i for i, s in enumerate(samples) if len(s["past"]) >= 3]
                if ids:
                    values, _ = history.predict(model, [samples[i] for i in ids], args.device, args.batch_size, True, bundle["seed"])
                    shuffled[name] = (ids, values)
            del model
        metrics, intervals, shuffle_metrics = {}, {}, {}
        for split in args.splits if args.class_targets else []:
            ids = [i for i, s in enumerate(samples) if s["split"] == split]
            if not ids:
                continue
            rows = [samples[i] for i in ids]
            values = {name: p[ids] for name, p in predictions.items()}
            metrics[split] = {name: report_metrics(rows, p) for name, p in values.items()}
            intervals[split] = paired_intervals(rows, values, args.bootstrap, bundle["seed"])
            shuffle_metrics[split] = {}
            for name, (original_ids, p) in shuffled.items():
                take = [j for j, i in enumerate(original_ids) if samples[i]["split"] == split]
                if take:
                    indices = [original_ids[j] for j in take]
                    matched = [samples[i] for i in indices]
                    shuffle_metrics[split][name] = {"normal": report_metrics(matched, predictions[name][indices]),
                                                  "shuffled": report_metrics(matched, p[take])}
        rows = []
        for i, sample in enumerate(samples):
            for c, category in enumerate(categories):
                valid = bool(sample["valid"][c]) if args.class_targets else False
                rows.append({**{k: sample[k] for k in IDENTITY}, "category_id": category["id"],
                    "category": category["name"], "target_valid": valid,
                    "target_fraction": float(sample["fractions"][c]) if valid else "",
                    "known_pixels": int(sample["known_pixels"][c]) if args.class_targets else "",
                    "trained": bool(trained[c]),
                    **{name: float(p[i, c]) if trained[c] else "" for name, p in predictions.items()}})
        history.write_csv(store.out / "predictions.csv", rows, (*IDENTITY, "category_id", "category", "target_valid",
                          "target_fraction", "known_pixels", "trained", *predictions))
        write_json(store.out / "metrics.json", {"scope": SCOPE, "categories": categories, "splits": metrics,
                   "paired_mae_differences": intervals, "past_shuffle": shuffle_metrics,
                   "primary_metric": "macro_building_mae", "independent_temporal_event_test": False})
        history.finish_run(store, data, {"scope": SCOPE, "events": len(samples), "coverage": cover,
                                         "metrics": metrics, "models": list(bundle["models"])})
        return {"out": str(store.out), "events": len(samples), "scope": SCOPE}
    except BaseException as exc:
        store.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        store.close()


def parser():
    cli = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    cli.add_argument("--features", required=True)
    cli.add_argument("--history-root")
    cli.add_argument("--annotation-run", required=True, help="Completed history-targets run; masks are reused")
    cli.add_argument("--out", required=True)
    cli.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val"])
    cli.add_argument("--min-years", type=int, default=3)
    cli.add_argument("--min-known-pixels", type=int, default=64)
    cli.add_argument("--min-known-fraction", type=float, default=.001)
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.min_years < 2 or args.min_known_pixels < 1 or not 0 <= args.min_known_fraction <= 1:
        cli.error("Invalid prefix length or known-support thresholds")
    try:
        print(json.dumps(prepare(args), ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"history fractions failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
