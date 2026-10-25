"""Train, evaluate and inspect spatial target-class changes from cached histories."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from functools import lru_cache
import html
import importlib
import io
from pathlib import Path

import numpy as np
from PIL import Image

from ..history_features import _inside, digest
from ..io import read_json, sha256, write_json


history = importlib.import_module("facade_change.research.2026-10-25_history_models")
data_module = importlib.import_module("facade_change.research.2026-10-25_spatial_change_data")
metrics = importlib.import_module("facade_change.research.2026-10-25_spatial_change_metrics")
NETWORK = "facade_change.research.2026-10-25_spatial_change_model"
MODES = ("pair", "mean", "convgru", "convgru-last")


def load_data(args, splits):
    return data_module.SpatialData(args.features, args.targets, splits, args.history_root,
                                   args.target_features, args.categories, args.auxiliary, args.smoke_buildings)


def data_request(args):
    return {key: getattr(args, key) for key in
            ("features", "targets", "history_root", "target_features", "categories", "auxiliary", "smoke_buildings")}


def run_store(args, data):
    out = Path(args.out).expanduser().resolve()
    if (out / "run.json").exists() and read_json(out / "run.json").get("status") == "completed":
        raise ValueError("Completed runs are immutable; use a new --out")
    store = history.HistoryRun(out, {k: v for k, v in vars(args).items() if k != "resume"}, args.resume)
    try:
        source_paths = [Path(__file__), Path(data_module.__file__), Path(metrics.__file__),
                        Path(__file__).with_name("2026-10-25_spatial_change_model.py"),
                        Path(__file__).parents[1] / "segmentation/model.py",
                        Path(__file__).parents[1] / "segmentation/metrics.py"]
        store.contract({"task": "spatial target-class annotation changes", "identity": data.identity,
                        "inputs": dict(data.fp.files), "sources": {str(p): sha256(p) for p in source_paths},
                        "statistical_unit": "building", "future_images": False,
                        "target_validity_is_model_input": False, "geometry_is_visibility": False,
                        "feature_adapter": "identity; SET-inspired adaptation is a separate future experiment"})
        write_json(store.out / "coverage.json", {
            **data.coverage, "class_crop_counts": {k: v.tolist() for k, v in data.class_coverage.items()},
            "columns": ["positive", "negative", "unknown"], "categories": data.categories,
            "count_scope": "overlapping target crops on selected pair geometry; not independent or final pixel counts"})
        return store
    except BaseException:
        store.close()
        raise


def tensors(item, device):
    import torch
    result = {k: torch.from_numpy(item[k]).to(device) for k in ("features", "support", "years")}
    result["output_size"] = item["target"].shape[-2:]
    if item["extra"].shape[1]:
        result["extra"] = torch.from_numpy(item["extra"]).to(device)
        result["available"] = torch.from_numpy(item["available"]).to(device)
    return result


def model_prediction(model, data, device):
    import torch
    model.eval()

    @lru_cache(maxsize=96)
    def cached(event_id):
        sample = lookup[event_id]
        item = data.load(sample)
        with torch.inference_mode():
            probability = model(**tensors(item, device)).sigmoid()[0].cpu().numpy()
        return probability, item["geometry"], item["target"]

    lookup = {s["event_id"]: s for s in data.samples}
    return lambda sample: cached(sample["event_id"])


def finish(store, data, summary):
    data.fp.verify()
    write_json(store.out / "inputs.json", dict(data.fp.files))
    write_json(store.out / "summary.json", summary)
    store.finish("completed", summary=summary, inputs_sha256=sha256(store.out / "inputs.json"),
                 summary_sha256=sha256(store.out / "summary.json"))


def train(args):
    data = load_data(args, ("train", "val"))
    train_rows = [s for s in data.samples if s["split"] == "train"]
    val_rows = [s for s in data.samples if s["split"] == "val"]
    if not train_rows or not val_rows:
        raise ValueError("Training and checkpoint selection require both TRAIN and VAL")
    coverage = data.class_coverage["train"]
    if np.any(coverage[:, :2].sum(1) == 0):
        raise ValueError("A selected class has no known TRAIN pixels; select supported --categories")
    store = run_store(args, data)
    try:
        torch = history.torch_runtime(args.device, args.seed)
        if args.device == "cpu":
            torch.set_num_threads(min(4, torch.get_num_threads()))
        networks = importlib.import_module(NETWORK)
        from ..segmentation.model import masked_multilabel_loss
        building_counts = Counter(s["building_id"] for s in train_rows)
        records, final_reports = [], {}
        for mode in args.models:
            torch.manual_seed(args.seed)
            spec = dict(feature_channels=data.dimension, target_channels=len(data.categories),
                        mode=mode, hidden=args.hidden, extra_channels=len(data.extra_spec["channels"]))
            model = networks.SpatialChangeModel(**spec).to(args.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            folder = store.out / mode
            folder.mkdir(exist_ok=True)
            last, best = folder / "last.pth", folder / "best.pth"
            epoch_start, best_score, best_epoch, stale = 0, -1., 0, 0
            if last.exists():
                payload = torch.load(last, map_location="cpu", weights_only=True)
                if payload["identity"] != data.identity or payload["spec"] != spec:
                    raise ValueError("Resume checkpoint channels, features or context differ")
                model.load_state_dict(payload["model"])
                optimizer.load_state_dict(payload["optimizer"])
                epoch_start, best_score = payload["epoch"], payload["best_score"]
                best_epoch, stale = payload["best_epoch"], payload["stale"]
            for epoch in range(epoch_start, args.epochs):
                if stale >= args.patience:
                    break
                model.train()
                order = np.random.default_rng(args.seed + epoch).permutation(len(train_rows))
                total_loss = 0.
                for offset in range(0, len(order), args.batch_size):
                    optimizer.zero_grad(set_to_none=True)
                    batch = order[offset:offset + args.batch_size]
                    for index in batch:
                        sample = train_rows[int(index)]
                        item = data.load(sample)
                        logits = model(**tensors(item, args.device))
                        labels = torch.from_numpy(item["target"]).to(args.device)
                        weight = len(train_rows) / (len(building_counts) * building_counts[sample["building_id"]])
                        loss = masked_multilabel_loss(logits, (labels == 1)[None], (labels != 255)[None])
                        (loss * weight / len(batch)).backward()
                        total_loss += float(loss.detach().cpu()) * weight
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    optimizer.step()
                hist, _ = metrics.collect_histograms(data, val_rows, model_prediction(model, data, args.device))
                report = metrics.report(hist, data.categories, [0.5] * (len(data.categories) + 1))
                score = report["classes"]["primary"]["f1"]
                score = -1. if score is None else float(score)
                improved = best_epoch == 0 or score > best_score + 1e-12
                if improved:
                    best_score, best_epoch, stale = score, epoch + 1, 0
                else:
                    stale += 1
                payload = {"schema_version": 1, "spec": spec, "identity": data.identity,
                           "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "epoch": epoch + 1, "best_score": best_score, "best_epoch": best_epoch, "stale": stale}
                if improved:
                    history.atomic_checkpoint(torch, best, payload)
                history.atomic_checkpoint(torch, last, payload)
                store.event("spatial_epoch", model=mode, epoch=epoch + 1,
                            train_loss=total_loss / len(train_rows), val_macro_building_f1=score,
                            checkpoint_sha256=sha256(last), best_epoch=best_epoch)
                print(f"{mode}: epoch {epoch + 1}/{args.epochs}, loss={total_loss / len(train_rows):.5f}, "
                      f"VAL building F1={score:.5f}", flush=True)
            payload = torch.load(best, map_location="cpu", weights_only=True)
            model.load_state_dict(payload["model"])
            hist, merged = metrics.collect_histograms(data, val_rows, model_prediction(model, data, args.device))
            thresholds, reasons = metrics.calibrate(hist, len(data.categories) + 1)
            result = metrics.report(hist, data.categories, thresholds)
            result["coverage"] = merged
            final_reports[mode] = result
            write_json(folder / "val_metrics.json", result)
            record = {"name": mode, "spec": spec, "checkpoint": best.relative_to(store.out).as_posix(),
                      "checkpoint_sha256": sha256(best), "thresholds": thresholds, "threshold_reasons": reasons,
                      "best_epoch": best_epoch, "parameters": sum(p.numel() for p in model.parameters())}
            records.append(record)
            store.event("spatial_model_completed", **record)
            del model, optimizer, payload
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        write_json(store.out / "models.json", {"schema_version": 1, "kind": "spatial_change_models",
                    "identity": data.identity, "models": records, "data_request": data_request(args),
                    "checkpoint_selection": "VAL equal-building macro-class F1 at fixed 0.5",
                    "calibration": "class thresholds and separate maximum-class binary score calibrated on VAL"})
        write_json(store.out / "metrics.json", final_reports)
        finish(store, data, {"models": [r["name"] for r in records], "train_events": len(train_rows),
                            "val_events": len(val_rows), "scientific_result": "not established by CPU fixtures"})
        return {"out": str(store.out)}
    except BaseException as exc:
        store.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        store.close()


def baseline_options(args, name):
    config_path = Path(args.methods_config).expanduser().resolve() if args.methods_config else None
    if config_path is None:
        for candidate in ("configs/methods.local.json", "configs/methods.example.json"):
            if Path(candidate).is_file():
                config_path = Path(candidate).resolve()
                break
    config = read_json(config_path) if config_path else {}
    options = dict(config.get("methods", {}).get(name, {}))
    paths = importlib.import_module("facade_change.2026-10-14_temporal_experiment").PATH_KEYS
    for key in paths & options.keys():
        path = Path(options[key]).expanduser()
        options[key] = str(path if path.is_absolute() else config_path.parent / path)
    options["device"] = args.device
    if args.trust_checkpoint:
        options["trust_checkpoint"] = True
    # Parameter-free RGB scorers do not accept runtime device/checkpoint options.
    if name in ("rgb_diff", "ssim"):
        options.pop("device", None)
        options.pop("trust_checkpoint", None)
    return options


def baseline_prediction(args, name, data):
    if name == "dino-difference":
        import torch
        from torch.nn import functional as F

        def predict(sample):
            item = data.load(sample)
            value = torch.from_numpy(item["features"][-2:])
            score = ((1 - F.cosine_similarity(value[0], value[1], dim=0)) / 2).clamp(0, 1)
            valid = torch.from_numpy(item["support"][-2:, 0].all(0))
            score = score.masked_fill(~valid, 0)
            score = F.interpolate(score[None, None], size=item["target"].shape[-2:],
                                  mode="bilinear", align_corners=False)[0].numpy()
            return score, item["geometry"], item["target"]

        return predict, {"method": name, "output_kind": "score", "encoder": data.history.index["encoder"],
                         "score": "(1-cosine)/2 on frozen patches, bilinear native-grid reconstruction"}, None
    from ..methods.remote import RemotePool, RemoteScorer
    scorer = RemoteScorer(name, baseline_options(args, name), RemotePool(Path(args.out) / "worker_logs"))
    try:
        metadata = scorer.activate()
    except BaseException:
        scorer.close()
        raise

    def predict(sample):
        item = data.load(sample)
        reference, source = data.rgb_pair(sample)
        if not item["geometry"].any():
            return np.zeros((1, *item["geometry"].shape), np.float32), item["geometry"], item["target"]
        scores = scorer(reference, source, item["geometry"])
        if metadata["output_kind"] == "native_mask":
            if scorer.native_prediction is None:
                raise ValueError("Native-mask adapter did not preserve its native prediction")
            scores = scorer.native_prediction.astype(np.float32)
        return np.where(item["geometry"], scores, 0)[None], item["geometry"], item["target"]

    return predict, metadata, scorer


def evaluation(args):
    data = load_data(args, tuple(args.splits))
    if len(set(args.splits)) != len(args.splits):
        raise ValueError("Duplicate evaluation splits")
    if "test" in args.splits and args.baselines and not args.calibration_run:
        if any(name != "anychange" for name in args.baselines):
            raise ValueError("TEST baselines require a completed VAL --calibration-run; no TEST threshold fitting")
    training, model_records = None, []
    if args.training_run:
        root = Path(args.training_run).expanduser().resolve()
        data.fp.add(root / "run.json")
        data.fp.add(root / "models.json")
        if read_json(root / "run.json").get("status") != "completed":
            raise ValueError("Training run is not completed")
        training = read_json(root / "models.json")
        if training.get("kind") != "spatial_change_models" or training["identity"] != data.identity:
            raise ValueError("Training/evaluation features, target channels or auxiliary vocabulary differ")
        available = {r["name"]: r for r in training["models"]}
        requested = args.models or list(available)
        if set(requested) - available.keys():
            raise ValueError("Requested spatial model is absent from training run")
        model_records = [available[name] for name in requested]
        for row in model_records:
            data.fp.add(_inside(root, row["checkpoint"]), row["checkpoint_sha256"])
    elif args.models:
        raise ValueError("--models requires --training-run")
    if not model_records and not args.baselines:
        raise ValueError("Select a training run and/or baselines")
    calibration = {}
    if args.calibration_run:
        path = Path(args.calibration_run).expanduser().resolve()
        data.fp.add(path / "run.json")
        data.fp.add(path / "calibration.json")
        saved = read_json(path / "calibration.json")
        if (read_json(path / "run.json").get("status") != "completed"
                or saved["identity"] != data.identity or saved["split"] != "val"):
            raise ValueError("Calibration must come from a compatible completed VAL run")
        calibration = saved["methods"]
    store = run_store(args, data)
    closable = None
    try:
        torch = history.torch_runtime(args.device, args.seed)
        if args.device == "cpu":
            torch.set_num_threads(min(4, torch.get_num_threads()))
        reports, units, calibrations = {}, [], {}
        jobs = [(r["name"], r) for r in model_records] + [(name, None) for name in args.baselines]
        for name, model_record in jobs:
            binary_only = model_record is None
            model = None
            if model_record:
                networks = importlib.import_module(NETWORK)
                payload = torch.load(_inside(root, model_record["checkpoint"]), map_location="cpu", weights_only=True)
                if payload["identity"] != data.identity or payload["spec"] != model_record["spec"]:
                    raise ValueError("Checkpoint payload differs from recorded model identity")
                model = networks.SpatialChangeModel(**model_record["spec"]).to(args.device)
                model.load_state_dict(payload["model"])
                predictor = model_prediction(model, data, args.device)
                metadata = {"method": name, "output_kind": "score", "checkpoint_sha256": model_record["checkpoint_sha256"]}
                thresholds = model_record["thresholds"]
            else:
                predictor, metadata, closable = baseline_prediction(args, name, data)
                thresholds = [0.5] * (len(data.categories) + 1)
            native = metadata["output_kind"] == "native_mask"
            # Worker log locations are run-specific, not part of method identity.
            provenance = {k: v for k, v in metadata.items() if k != "worker_stderr_log"}
            if name in calibration:
                if calibration[name]["provenance"] != provenance:
                    raise ValueError("Baseline calibration model/checkpoint/implementation differs")
                thresholds = calibration[name]["thresholds"]
            saved_units = {}
            for index, sample in enumerate(data.samples):
                unit_id = "spatial-" + digest([name, sample["event_id"]])
                cached = store.read_unit(unit_id)
                if cached is None:
                    probability, geometric, target = predictor(sample)
                    record = {"unit_id": unit_id, "method": name, "event_id": sample["event_id"],
                              "split": sample["split"], "building_id": sample["building_id"],
                              "view_id": sample["view_id"], "bbox_xywh": sample["bbox_xywh"],
                              "previous_year": sample["previous_year"], "current_year": sample["current_year"],
                              "output_kind": metadata["output_kind"], "binary_only": binary_only,
                              "provenance_sha256": digest(provenance)}
                    record = store.save_unit(record, {"probabilities": probability.astype(np.float32),
                                                       "geometry": geometric, "target": target})
                else:
                    record = cached[0]
                    if record["provenance_sha256"] != digest(provenance):
                        raise ValueError("Resume inference implementation/checkpoint changed")
                saved_units[sample["event_id"]] = record
                if index % 100 == 0 or index + 1 == len(data.samples):
                    print(f"{name}: {index + 1}/{len(data.samples)} spatial maps", flush=True)

            @lru_cache(maxsize=96)
            def read_prediction(event_id):
                record = saved_units[event_id]
                with np.load(store.out / record["artifact_path"], allow_pickle=False) as archive:
                    return tuple(archive[k].copy() for k in ("probabilities", "geometry", "target"))

            prediction = lambda sample: read_prediction(sample["event_id"])
            split_histograms = {}
            for split in args.splits:
                selected = [s for s in data.samples if s["split"] == split]
                if not selected:
                    raise ValueError(f"No evaluable spatial events for split {split}")
                split_histograms[split] = metrics.collect_histograms(data, selected, prediction, binary_only, native)
            if binary_only and not native and name not in calibration:
                if "val" not in split_histograms or "test" in args.splits:
                    raise ValueError("Continuous baseline needs a separate VAL calibration before TEST")
                thresholds, _ = metrics.calibrate(split_histograms["val"][0], len(data.categories) + 1)
            calibrations[name] = {"thresholds": thresholds, "provenance": provenance}
            for split, (hist, coverage) in split_histograms.items():
                result = metrics.report(hist, data.categories, thresholds, binary_only, native)
                result.update(coverage=coverage, provenance=provenance)
                reports.setdefault(split, {})[name] = result
            for record in saved_units.values():
                units.append({**record, "thresholds": thresholds})
            store.event("spatial_method_completed", method=name, provenance=provenance, thresholds=thresholds)
            if closable:
                closable.close()
                closable = None
            del model, predictor
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        for split, methods in reports.items():
            write_json(store.out / f"metrics-{split}.json", {
                "methods": methods, "paired_vs_pair": metrics.paired_bootstrap(methods, args.bootstrap, args.seed),
                "status": "VAL is used for model selection; TEST uses frozen checkpoints and thresholds"})
        write_json(store.out / "calibration.json", {"identity": data.identity, "split": "val" if args.splits == ["val"] else "mixed",
                                                  "methods": calibrations})
        write_json(store.out / "index.json", {"kind": "spatial_change_evaluation", "schema_version": 1,
                   "data_request": data_request(args), "splits": args.splits, "units": units,
                   "identity": data.identity})
        render_gallery(data, units, store.out, store.out / "gallery.html", args.gallery_limit)
        finish(store, data, {"methods": [name for name, _ in jobs], "splits": args.splits,
                            "maps": len(units), "gallery": "gallery.html"})
        return {"out": str(store.out)}
    except BaseException as exc:
        store.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        if closable:
            closable.close()
        store.close()


def image_uri(array):
    image = Image.fromarray(array)
    image.thumbnail((180, 180))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def overlay(rgb, mask, known):
    result = rgb.astype(np.float32).copy()
    result[mask & known] = .45 * result[mask & known] + .55 * np.array([255, 45, 30])
    result[~known] = .25 * result[~known] + .75 * np.array([130, 130, 130])
    return result.astype(np.uint8)


def render_gallery(data, units, root, path, limit):
    by_event, methods = {}, []
    for unit in units:
        by_event.setdefault(unit["event_id"], {})[unit["method"]] = unit
        if unit["method"] not in methods:
            methods.append(unit["method"])
    # Round-robin buildings and transitions; same examples for every method.
    groups = {}
    for sample in sorted(data.samples, key=lambda r: (r["current_year"], r["event_id"])):
        groups.setdefault(sample["building_id"], []).append(sample)
    chosen = []
    while len(chosen) < limit and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(chosen) < limit:
                chosen.append(groups[key].pop(0))
    parts = ['<!doctype html><meta charset="utf-8"><title>Spatial change comparison</title>',
             '<style>body{font:15px sans-serif;margin:24px;background:#f4f4f4}table{border-collapse:collapse}',
             'td,th{padding:6px;border:1px solid #bbb}img{max-width:180px}section{margin:24px 0}</style>',
             '<h1>Изменения целевых классов</h1><p>Красное — изменение, серое — неизвестная область.',
             ' Цели получены сравнением аннотаций. Одинаковые примеры для всех методов;',
             ' выбор по зданиям и переходам, без отбора по качеству предсказания.</p>']
    for sample in chosen:
        rgb = data.rgb_pair(sample)[1]
        target = data.load(sample)["target"]
        parts.append("<section><h2>" + html.escape(f"{sample['building_id']} / {sample['view_id']} "
                     f"{sample['previous_year']} → {sample['current_year']}") + "</h2><div>")
        for observation in sample["observations"]:
            source = _inside(data.history.history_root, observation["rgb"])
            data.fp.add(source, observation["rgb_sha256"])
            with Image.open(source) as image:
                uri = image_uri(np.array(image))
            parts.append(f'<span>{observation["year"]}<img src="{uri}"></span>')
        parts.append("</div><table><tr><th>Класс</th><th>Целевая маска</th>" +
                     "".join("<th>" + html.escape(m) + "</th>" for m in methods) + "</tr>")
        all_targets = [*target, metrics.binary_target(target)]
        for channel, (category, labels) in enumerate(zip([*data.categories, {"name": "ANY_TARGET_CHANGE"}], all_targets)):
            known = labels != 255
            parts.append("<tr><th>" + html.escape(category["name"]) + '</th><td><img src="' +
                         image_uri(overlay(rgb, labels == 1, known)) + '"></td>')
            for name in methods:
                unit = by_event[sample["event_id"]].get(name)
                if unit is None or (unit["binary_only"] and channel < len(data.categories)):
                    parts.append("<td>—</td>")
                    continue
                artifact = root / unit["artifact_path"]
                if sha256(artifact) != unit["artifact_sha256"]:
                    raise ValueError("Gallery map hash differs from evaluation")
                with np.load(artifact, allow_pickle=False) as archive:
                    scores = archive["probabilities"]
                value = scores[0] if unit["binary_only"] else (
                    scores.max(0) if channel == len(data.categories) else scores[channel])
                parts.append('<td><img src="' + image_uri(overlay(rgb, value >= unit["thresholds"][channel], known)) + '"></td>')
            parts.append("</tr>")
        parts.append("</table></section>")
    path.write_text("\n".join(parts), encoding="utf-8")


def gallery(args):
    root = Path(args.evaluation_run).expanduser().resolve()
    if read_json(root / "run.json").get("status") != "completed":
        raise ValueError("Gallery requires a completed evaluation")
    index = read_json(root / "index.json")
    if index.get("kind") != "spatial_change_evaluation":
        raise ValueError("Unexpected evaluation artifact")
    data = data_module.SpatialData(**index["data_request"], splits=index["splits"])
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    render_gallery(data, index["units"], root, out / "gallery.html", args.gallery_limit)
    data.fp.verify()
    write_json(out / "source.json", {"evaluation_index": str(root / "index.json"),
                                    "sha256": sha256(root / "index.json")})
    return {"gallery": str(out / "gallery.html")}


def parser():
    cli = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = cli.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        sub = commands.add_parser(name, allow_abbrev=False)
        sub.add_argument("--features", required=True)
        sub.add_argument("--targets", required=True)
        sub.add_argument("--history-root")
        sub.add_argument("--target-features")
        sub.add_argument("--categories", nargs="+")
        sub.add_argument("--auxiliary")
        sub.add_argument("--smoke-buildings", type=int, default=0)
        sub.add_argument("--out", required=True)
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--resume", action="store_true")
        sub.add_argument("--models", nargs="+", choices=MODES,
                         default=list(MODES) if name == "train" else None)
        if name == "train":
            sub.add_argument("--epochs", type=int, default=40)
            sub.add_argument("--patience", type=int, default=8)
            sub.add_argument("--batch-size", type=int, default=8)
            sub.add_argument("--hidden", type=int, default=32)
            sub.add_argument("--learning-rate", type=float, default=1e-3)
            sub.add_argument("--weight-decay", type=float, default=1e-4)
        else:
            from ..methods.registry import ALL_METHODS
            sub.add_argument("--training-run")
            sub.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["val"])
            sub.add_argument("--baselines", nargs="*", choices=("dino-difference", *ALL_METHODS),
                             default=["dino-difference", "anychange"])
            sub.add_argument("--methods-config")
            sub.add_argument("--trust-checkpoint", action="store_true")
            sub.add_argument("--calibration-run")
            sub.add_argument("--bootstrap", type=int, default=500)
            sub.add_argument("--gallery-limit", type=int, default=24)
    sub = commands.add_parser("gallery", allow_abbrev=False)
    sub.add_argument("--evaluation-run", required=True)
    sub.add_argument("--out", required=True)
    sub.add_argument("--gallery-limit", type=int, default=24)
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "train" and (min(args.epochs, args.patience, args.batch_size, args.hidden) < 1
                                    or not np.isfinite(args.learning_rate) or args.learning_rate <= 0
                                    or not np.isfinite(args.weight_decay) or args.weight_decay < 0):
        raise ValueError("Training sizes/rate must be positive and weight decay nonnegative")
    if getattr(args, "models", None) and len(set(args.models)) != len(args.models):
        raise ValueError("Duplicate model modes")
    if args.command == "evaluate" and (len(set(args.baselines)) != len(args.baselines) or args.bootstrap < 0):
        raise ValueError("Duplicate baselines or negative bootstrap count")
    result = {"train": train, "evaluate": evaluation, "gallery": gallery}[args.command](args)
    print(result, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
