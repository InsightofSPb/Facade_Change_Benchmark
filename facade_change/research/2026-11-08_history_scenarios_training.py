"""Seed-13 streamed training, immutable epoch checkpoints and frozen map export.

The runner owns the existing RunStore lock/ledger. This module never opens a
second provenance store, never evaluates TEST during fitting, and never invokes
an image encoder or a base detector. Prediction calls read inputs before targets.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import nullcontext, contextmanager
from functools import lru_cache
from types import SimpleNamespace
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import time
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from facade_change.io import read_json, sha256

PREFIX = "2026-11-08_"
PROTOCOL = "history-three-scenarios-training-v1"


def _module(suffix):
    return importlib.import_module("facade_change.research.2026-11-08_history_scenarios_" + suffix)


def _report():
    return importlib.import_module("data_analysis.2026-11-08_history_scenarios_report")


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _event(ledger, kind, **values):
    return ledger.event(kind, **values) if ledger is not None else None


def options(config):
    result = dict(epochs=60, patience=12, lr=3e-4, weight_decay=.01,
                  effective_batch=8, microbatch=1, amp=True, grad_clip=5., seed=13,
                  quantizer_step=.1, min_scale=.05, device="cpu", smoke=False,
                  fusion_epochs=30, fusion_lr=.01, auxiliary_weight=.1,
                  fusion_weight_decay=0., matrix_protocol="legacy-v1")
    if config.get("matrix_protocol") == "reduced-v2":
        result.update(fusion_epochs=5,fusion_lr=.001)
    result.update(config.get("training", {}))
    result.update({k: v for k, v in config.items() if k in result})
    if result["seed"] != 13:
        raise ValueError("This experiment uses exactly training seed 13")
    if result["microbatch"] != 1:
        raise ValueError("Streamed implementation supports microbatch=1; effective_batch is accumulated")
    if min(result["epochs"], result["patience"], result["effective_batch"]) < 1:
        raise ValueError("Positive epochs, patience and effective_batch required")
    if result["lr"] <= 0 or result["quantizer_step"] <= 0 or result["min_scale"] <= 0:
        raise ValueError("Learning rate, quantizer step and scale floor must be positive")
    if result["matrix_protocol"] == "reduced-v2":
        if not isinstance(result["epochs"],int) or result["epochs"]>60:
            raise ValueError("Reduced primary budget is an integer maximum of 60 epochs")
        if not isinstance(result["fusion_epochs"],int) or not 1 <= result["fusion_epochs"] <= 10:
            raise ValueError("Fusion budget must explicitly be an integer between 1 and 10 epochs")
    return result


def resolve_matrix(config):
    """Resolve all scientific comparisons before any TEST access."""
    opts = options(config)
    if opts["matrix_protocol"] == "reduced-v2":
        return _reduced_matrix(opts)
    sizes = ["smoke"] if opts["smoke"] else ["medium", "large"]
    result = []
    for task in ("direct", "actual-next", "h0"):
        for family in ("convgru", "transformer"):
            for size in sizes:
                for control in ("history", "pair-only" if task == "direct" else "last-only",
                                "mean", "repeat-history"):
                    result.append(dict(id=f"{task}-{family}-{size}-{control}", task=task,
                        family=family, size=size, control=control, delta="signed", time_mode="real",
                        scenario="main", target_type="annotation-derived latest change"))
                # One changed factor per comparison, at a predeclared size.
                if size == sizes[0]:
                    for factor, changes in (("absolute-delta", {"delta": "absolute"}),
                                            ("ordinal-time", {"time_mode": "index"})):
                        result.append(dict(id=f"{task}-{family}-{size}-{factor}", task=task,
                            family=family, size=size, control="history", delta="signed",
                            time_mode="real", scenario="main", ablation=factor,
                            target_type="annotation-derived latest change", **{}))
                        result[-1].update(changes)
    for mode in ("independent", "mean", "confidence", "learned", "repeat"):
        result.append(dict(id="A-" + mode, task="semantic-state", scenario="A", mode=mode,
            family="semantic-memory", size="shared", control=mode, target_type="semantic state"))
    for mode in ("pair", "regularized", "history"):
        result.append(dict(id="B-" + mode, task="direct", scenario="B", mode=mode,
            family="pair-aux", size="shared", control=mode, target_type="annotation-derived latest change"))
    for mode in ("latest", "first", "history"):
        result.append(dict(id="C-" + mode, task="progression", scenario="C", mode=mode,
            family="progression", size="shared", control=mode, target_type="annotation-derived progression proxy"))
    return result


def _reduced_matrix(opts):
    size = "smoke" if opts["smoke"] else "large"
    result=[]
    for task in ("direct","actual-next","h0"):
        for family in ("convgru","transformer"):
            for control in ("history","pair-only" if task=="direct" else "last-only"):
                result.append(dict(id=f"{task}-{family}-{size}-{control}",task=task,family=family,size=size,
                    control=control,delta="signed",time_mode="real",scenario="main",
                    matrix_protocol="reduced-v2",target_type="annotation-derived latest change"))
    for scenario,modes,task,family,target in (
        ("A",("learned","repeat"),"semantic-state","semantic-memory","semantic state"),
        ("B",("history","regularized"),"direct","pair-aux","annotation-derived latest change"),
        ("C",("history","latest"),"progression","progression","annotation-derived progression proxy")):
        for mode in modes:
            result.append(dict(id=scenario+"-"+mode,scenario=scenario,mode=mode,task=task,family=family,
                size="shared",hidden=64,control=mode,matrix_protocol="reduced-v2",target_type=target))
    return result


def matched_control(spec):
    if spec.get("scenario")=="A": return "A-repeat"
    if spec.get("scenario")=="B": return "B-regularized"
    if spec.get("scenario")=="C": return "C-latest"
    control="pair-only" if spec["task"]=="direct" else "last-only"
    return f"{spec['task']}-{spec['family']}-{spec['size']}-{control}"


def resolve_fusion_matrix(config):
    if options(config)["matrix_protocol"] != "reduced-v2":
        return []
    result=[]
    for parent in resolve_matrix(config):
        if parent["scenario"]=="C": continue
        kinds=["logreg","score-mlp","score-cnn"]
        if parent["task"]=="h0": kinds.extend(["feature-mlp","cross-attention"])
        for kind in kinds:
            result.append(dict(**{k:v for k,v in parent.items() if k!="id"},id=parent["id"]+"+"+kind,
                parent_method=parent["id"],base_method=parent["id"],fusion_kind=kind,
                fit_kind="feature-fusion" if kind in ("feature-mlp","cross-attention") else "score-fusion",
                matched_control_method=matched_control(parent)+"+"+kind))
    return result


def matrix_contract(config):
    result=dict(protocol=PROTOCOL,options=options(config),variants=resolve_matrix(config),seed=13,
        test_previously_inspected=True,progression_rule=_module("abc").PROGRESSION_RULE)
    if options(config)["matrix_protocol"]=="reduced-v2":
        result.update(fusion_variants=resolve_fusion_matrix(config),planned_fit_counts=dict(primary=18,score_fusion=48,feature_fusion=8,total=74),
            checkpoint_selection="top3 VAL building-macro F1 at recall-selected FPR<=.05 operating point; earlier epoch ties",
            early_stopping="patience12 on strictly improving primary VAL F1 only; secondary improvements do not reset")
    return result


def _tensor(value, device, dtype=torch.float32):
    return torch.as_tensor(np.asarray(value).copy(), device=device, dtype=dtype)


def _normalizer(dataset, out, ledger=None):
    path = Path(out) / (PREFIX + "feature_normalization.json")
    marker = Path(out) / (PREFIX + "feature_normalization_integrity.json")
    data = _module("data")
    if path.exists():
        state = read_json(path)
        if marker.exists():
            if read_json(marker)["artifact_sha256"] != sha256(path):
                raise ValueError("Feature normalization artifact hash mismatch")
        else:
            # Recover a crash after the atomic payload but before its marker.
            rebuilt = data.fit_normalizer(dataset)
            if rebuilt.state_dict() != state:
                raise ValueError("Uncommitted feature normalization differs from TRAIN")
            event = _event(ledger,"normalization_recovered",split="train",artifact_sha256=sha256(path))
            _json(marker,dict(artifact_sha256=sha256(path),source_event_id=event))
        return data.FrozenNormalizer.from_state_dict(state)
    fitted = data.fit_normalizer(dataset)
    _json(path, fitted.state_dict())
    event = _event(ledger, "normalization_fitted", split="train", artifact_path=str(path),
                   artifact_sha256=sha256(path))
    _json(marker,dict(artifact_sha256=sha256(path),source_event_id=event))
    return fitted


def _inputs(dataset, row, spec, normalizer, device, intervention=None, pair_only=False):
    predictive = spec["task"] in ("actual-next", "h0")
    arrays = dataset.pair_inputs(row) if pair_only else dataset.inputs(row, predictive=predictive)
    # Normalization is a frozen TRAIN transform, applied to past alone for forecasting.
    x = normalizer.apply(arrays["features"], arrays["support"])
    values = dict(features=_tensor(x, device)[None],
        support=_tensor(arrays["support"], device, torch.bool)[None],
        years=_tensor(arrays["years"], device)[None],
        target_year=_tensor(arrays["target_year"], device).reshape(1),
        padding=_tensor(arrays.get("padding", np.ones(len(x), bool)), device, torch.bool)[None])
    keep = 1 if predictive else 2
    old = values["features"].shape[1] - keep
    if old > 0:
        # Same content-independent support transform for every learned control.
        values["support"][:, :old] &= values["support"][:, old:old + 1]
        if intervention in ("shuffle", "shuffle-common"):
            common = values["support"][:, :old].all(1, keepdim=True)
            values["support"][:, :old] &= common
    if intervention and intervention != "shuffle-common":
        changed = _module("models").intervene_history(values["features"], values["support"],
            values["padding"], mode=intervention, preserve=keep, seed=13)
        values["features"], values["support"] = changed[:2]
    return values


def _resize(value, shape, mode="bilinear"):
    return F.interpolate(value, size=shape, mode=mode,
                         **({"align_corners": False} if mode == "bilinear" else {}))


def _feature_target(dataset, row, normalizer, device):
    item = dataset.feature_target(row)
    return (_tensor(normalizer.apply(item["features"], item["support"]), device)[None],
            _tensor(item["support"], device, torch.bool)[None])


def _labels(dataset, row, device, shape=None):
    item = dataset.targets(row)
    target = _tensor(item["target"], device)[None, None]
    valid = _tensor(item["support"], device, torch.bool)[None, None] & (target != 255)
    if shape is not None and tuple(target.shape[-2:]) != tuple(shape):
        # Conservative stable/changed supervision: every covered RGB pixel known;
        # any positive makes a feature patch changed, never stable.
        all_known = F.adaptive_avg_pool2d(valid.float(), shape) >= 1 - 1e-6
        positive = F.adaptive_max_pool2d(((target == 1) & valid).float(), shape)
        target, valid = positive, all_known
    return target, valid


def _semantic_inputs(dataset, row, device):
    item = dataset.semantic(row)
    if isinstance(item, tuple):
        probs, support = item
    else:
        probs, support = item["probabilities"], item["support"]
    return _tensor(probs, device)[None], _tensor(support, device, torch.bool)[None]


def _annotations(dataset, row, device):
    item = dataset.annotation_history(row)
    if "probabilities" in item:
        gt = item["probabilities"]
    elif "labels" in item:
        gt = item["labels"]
    elif "semantic_gt" in item:
        gt = item["semantic_gt"]
    elif "packed" in item:
        packed = np.asarray(item["packed"])
        count = len(_class_names(dataset))
        gt = np.stack([(packed & (1 << k)) != 0 for k in range(count)], axis=1)
    else:
        raise ValueError("Annotation history requires explicit semantic channels or packed masks")
    known = item.get("known", np.asarray(item.get("packed", 0)) != 0)
    support = item["support"]
    return _tensor(gt, device)[None], _tensor(known, device, torch.bool)[None], _tensor(support, device, torch.bool)[None]


def _class_names(dataset):
    if hasattr(dataset, "class_names"):
        return list(dataset.class_names)
    canonical = getattr(dataset, "canonical", None)
    categories = getattr(canonical, "categories", None) or getattr(dataset, "categories", [])
    return [x["name"] for x in categories]


def build_model(spec, dataset, config):
    opts = options(config)
    channels = getattr(dataset, "channels", None)
    if channels is None:
        row = dataset.samples[0]
        channels = dataset.inputs(row, predictive=False)["features"].shape[1]
    if spec.get("scenario", "main") == "main":
        model = _module("models").TemporalSpatialModel(channels=channels, family=spec["family"],
            size=spec["size"], task=spec["task"], control=spec["control"], delta=spec["delta"],
            time_mode=spec["time_mode"], min_scale=opts["min_scale"])
    else:
        model = _module("abc").build_scenario_model(spec, channels, len(_class_names(dataset)))
    return model.to(opts["device"])


def _forward(model, dataset, row, spec, normalizer, device, intervention=None, pair_only=False):
    inputs = _inputs(dataset, row, spec, normalizer, device, intervention, pair_only=pair_only)
    if spec.get("scenario", "main") == "main":
        output = model(**inputs)
    else:
        semantic, semantic_support = _semantic_inputs(dataset, row, device) if spec["scenario"] == "A" else (None, None)
        if semantic is not None:
            # Semantic memory works at semantic pixel resolution, independent of DINO grid.
            support = semantic_support
        else:
            support = inputs["support"]
        output = _module("abc").scenario_forward(model, spec, inputs["features"], support,
            inputs["years"], semantic_probs=semantic, padding=inputs["padding"])
        output["_semantic_probs"] = semantic
    return output, inputs


def _progression(dataset, row, device):
    gt, known, support = _annotations(dataset, row, device)
    years = dataset.inputs(row)["years"]
    result = _module("abc").progression_target(gt[0].cpu().numpy(), known[0].cpu().numpy(), support[0].cpu().numpy(),
                                              _class_names(dataset), years)
    return result


def _loss(model, dataset, row, spec, normalizer, opts):
    output, inputs = _forward(model, dataset, row, spec, normalizer, opts["device"])
    scenario = spec.get("scenario", "main")
    if scenario == "main" and spec["task"] in ("actual-next", "h0"):
        target, support = _feature_target(dataset, row, normalizer, opts["device"])
        support = support & inputs["support"][:, -1] & inputs["support"][:, 0]
        if spec["task"] == "h0":
            labels, known = _labels(dataset, row, opts["device"], target.shape[-2:])
            support = support & known & (labels == 0)
        if not support.any():
            return None
        nll = _module("models").quantized_logistic_nll(target, output["mean"], output["scale"],
                                                     step=opts["quantizer_step"], min_scale=opts["min_scale"])
        return nll.masked_select(support.expand_as(nll)).mean()
    if scenario == "main":
        target, valid = _labels(dataset, row, opts["device"])
        logits = _resize(output["logits"], target.shape[-2:])
        valid &= _resize(output["support"].float(), target.shape[-2:], "nearest").bool()
        return F.binary_cross_entropy_with_logits(logits[valid], target[valid]) if valid.any() else None
    target, known = _labels(dataset, row, opts["device"], inputs["features"].shape[-2:])
    semantic_gt = semantic_known = stable = None
    if scenario == "A":
        semantic_gt, semantic_known, annotation_support = _annotations(dataset, row, opts["device"])
        semantic_known = semantic_known & annotation_support
    elif scenario == "B" and spec["mode"] == "history":
        gt, gt_known, gt_support = _annotations(dataset, row, opts["device"])
        if gt_known.ndim == 5:
            gt_known = gt_known.squeeze(2)
        if gt_support.ndim == 5:
            gt_support = gt_support.squeeze(2)
        stable = _module("abc").stable_correspondences(gt, gt_known, gt_support)
        stable = F.adaptive_avg_pool2d(stable.float().flatten(0, 1)[:, None], inputs["features"].shape[-2:]) >= 1 - 1e-6
        stable = stable.reshape(1, -1, *inputs["features"].shape[-2:])
    elif scenario == "C":
        progression = _progression(dataset, row, opts["device"])
        target = _tensor(progression["target"], opts["device"])[None, None]
        known = target >= 0
        target = target.clamp_min(0)
    if not known.any() and scenario != "A":
        return None
    return _module("abc").scenario_loss(model, spec, output, inputs["features"], inputs["support"],
        inputs["years"], target[:, 0], known[:, 0], semantic_probs=output.get("_semantic_probs"),
        semantic_gt=semantic_gt, semantic_known=semantic_known, stable_history=stable)


def prediction(model, dataset, row, spec, normalizer, config, intervention=None):
    """Return raw float maps. GT is accessed only after the model forward."""
    opts = options(config)
    model.eval()
    with torch.inference_mode():
        output, inputs = _forward(model, dataset, row, spec, normalizer, opts["device"], intervention,
                                  pair_only=spec.get("scenario") == "B")
        target_item = dataset.targets(row)  # This is evaluation only, after causal forecast.
        shape = target_item["target"].shape
        extras = {}
        if spec["task"] in ("actual-next", "h0"):
            target, target_support = _feature_target(dataset, row, normalizer, opts["device"])
            nll = _module("models").quantized_logistic_nll(target, output["mean"], output["scale"],
                                                        step=opts["quantizer_step"], min_scale=opts["min_scale"])
            l1 = (target - output["mean"]).abs().mean(1, keepdim=True)
            cosine = 1 - F.cosine_similarity(target, output["mean"], dim=1)[:, None]
            surprise = nll.mean(1, keepdim=True)
            score = -torch.expm1(-surprise / 8.)
            support = target_support & inputs["support"][:, -1] & inputs["support"][:, 0]
            extras = dict(forecast=output["mean"][0].cpu().numpy(), scale=output["scale"][0].cpu().numpy(),
                target_features=target[0].cpu().numpy(), U=_resize(surprise, shape)[0, 0].cpu().numpy(),
                l1=_resize(l1, shape)[0, 0].cpu().numpy(), cosine=_resize(cosine, shape)[0, 0].cpu().numpy(),
                residual_score=_resize(-torch.expm1(-l1), shape)[0, 0].cpu().numpy(),
                coverage90=_resize(((target - output["mean"]).abs() <= math.log(19.) * output["scale"]).float().mean(1,keepdim=True), shape)[0,0].cpu().numpy())
        elif spec.get("scenario") == "A":
            semantic_sequence = output["semantic_sequence"]
            semantic = semantic_sequence[:,-1] if opts["matrix_protocol"]=="reduced-v2" else torch.sigmoid(output["semantic_logits"])
            # Soft probability of state disagreement; target remains latest change.
            score = (semantic - semantic_sequence[:, -2]).abs().amax(1, keepdim=True)
            _, sem_support = _semantic_inputs(dataset, row, opts["device"])
            support = sem_support[:, -1] & sem_support[:, -2]
            extras = dict(semantic_probability=semantic[0].cpu().numpy(),
                          semantic_before=semantic_sequence[0, -2].cpu().numpy())
        else:
            score = torch.sigmoid(output["logits"])
            support = output.get("support", inputs["support"][:, -1] & inputs["support"][:, -2])
        score = _resize(score, shape)[0, 0].cpu().numpy().astype(np.float32)
        supported = _resize(support.float(), shape, "nearest")[0, 0].bool().cpu().numpy()
        if spec["task"] in ("actual-next", "h0"):
            extras["forecast_support"] = supported
            score[~supported] = 0.
        support = dataset.evaluation_support(row) if hasattr(dataset, "evaluation_support") else np.asarray(target_item["support"], bool).copy()
        if spec.get("scenario") == "C":
            p = _progression(dataset, row, opts["device"])
            extras["progression_target"] = np.asarray(p["target"], np.int16)
        return dict(score=score, support=support, extras=extras)


def _rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _load_torch(path):
    # Only locally-created, hash-verified checkpoints from this run are loaded.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _snapshot(root, epoch, state, contract, ledger):
    path = Path(state.get("snapshot_path", root / (PREFIX + f"epoch-{epoch:04d}.pt")))
    if path.exists():
        raise FileExistsError(f"Epoch checkpoint already exists: {path}")
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    record = dict(path=str(path), sha256=sha256(path), epoch=epoch, contract=contract)
    record["source_event_id"] = _event(ledger, "epoch_checkpoint_saved", **record)
    _json(root / (PREFIX + "latest.json"), record)
    return record


def _profile_prediction(model, dataset, spec, normalizer, config):
    if spec["task"] not in ("actual-next", "h0"):
        return None
    per_building = defaultdict(list)
    for row in dataset.split_samples("val", predictive=True):
        predicted = prediction(model, dataset, row, spec, normalizer, config)
        item = dataset.targets(row)
        valid = predicted["extras"].get("forecast_support", predicted["support"]) & item["support"]
        if spec["task"] == "h0":
            valid &= item["target"] == 0
        if valid.any():
            per_building[row["building_id"]].append(float(predicted["extras"]["U"][valid].mean()))
    return float(np.mean([np.mean(v) for v in per_building.values()])) if per_building else None


def _validation(model, dataset, spec, normalizer, config, evaluator):
    @lru_cache(maxsize=24)
    def cached(identifier):
        return prediction(model,dataset,by_id[identifier],spec,normalizer,config)
    by_id={row["event_id"]:row for row in dataset.split_samples("val")}
    callback=lambda row:cached(row["event_id"])
    result = evaluator(dataset, callback, split="val", spec=spec, budget=.05)
    result["prediction_score"] = _profile_prediction(model, dataset, spec, normalizer, config)
    if options(config)["matrix_protocol"]=="reduced-v2" and spec.get("scenario")=="A":
        result["semantic_miou"]=_semantic_val_miou(dataset,callback)
    return result


def _criterion(result):
    recall, fpr = result.get("recall"), result.get("fpr")
    if recall is None or fpr is None or fpr > .05 + 1e-12:
        return (-1., -1.)
    return (float(recall), -float(fpr))


def train_variant(spec, dataset, out_dir, config, evaluator=None, ledger=None, resume=False,
                  normalizer=None):
    opts = options(config)
    if opts["matrix_protocol"] == "reduced-v2":
        return _train_reduced_variant(spec,dataset,out_dir,config,evaluator,ledger,resume,normalizer)
    root = Path(out_dir) / "models" / spec["id"]
    root.mkdir(parents=True, exist_ok=True)
    final = root / (PREFIX + "selection.json")
    normalizer = normalizer or _normalizer(dataset, out_dir, ledger)
    contract = _digest(dict(protocol=PROTOCOL, spec=spec, config=config,
                            source_sha256=sha256(__file__),normalizer_sha256=_digest(normalizer.state_dict())))
    if final.exists():
        record = read_json(final)
        if not resume or record["contract"] != contract:
            raise ValueError("Completed model/config mismatch; use a new run")
        for name in ("checkpoint", "prediction_checkpoint"):
            pointer = record.get(name)
            if pointer and sha256(pointer["path"]) != pointer["sha256"]:
                raise ValueError("Completed model checkpoint hash mismatch")
        return record
    normalizer = normalizer or _normalizer(dataset, out_dir, ledger)
    evaluator = evaluator or _report().evaluate_predictor
    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(13)
    model = build_model(spec, dataset, opts)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rows = dataset.split_samples("train", predictive=spec["task"] in ("actual-next", "h0"))
    counts = Counter(r["building_id"] for r in rows)
    if not rows:
        record = dict(spec=spec, id=spec["id"], status="INSUFFICIENT_DATA", reason="No TRAIN transitions", contract=contract)
        _json(final, record)
        return record
    start = time.perf_counter()
    device = opts["device"]
    cuda = device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts["lr"], weight_decay=opts["weight_decay"]) if parameters else None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opts["epochs"]) if optimizer else None
    scaler = torch.amp.GradScaler("cuda", enabled=cuda and opts["amp"]) if hasattr(getattr(torch, "amp", None), "GradScaler") else torch.cuda.amp.GradScaler(enabled=cuda and opts["amp"])
    first, steps, best_epoch, stale = 1, 0, 0, 0
    previous_peak_vram = 0
    best, best_prediction = (-1., -1.), math.inf
    selected = prediction_selected = None
    history = []
    latest = root / (PREFIX + "latest.json")
    if latest.exists():
        if not resume:
            raise FileExistsError("Incomplete model already exists; use resume")
        pointer = read_json(latest)
        if pointer["contract"] != contract or sha256(pointer["path"]) != pointer["sha256"]:
            raise ValueError("Resume checkpoint/config hash mismatch")
        state = _load_torch(pointer["path"])
        model.load_state_dict(state["model"])
        if optimizer:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
        _restore_rng(state["rng"])
        first, steps = state["epoch"] + 1, state["steps"]
        previous_peak_vram = state.get("peak_vram_bytes", 0)
        best, best_prediction = tuple(state["best"]), state["best_prediction"]
        best_epoch, stale, selected = state["best_epoch"], state["stale"], state["selected"]
        prediction_selected, history = state["prediction_selected"], state["history"]
        _event(ledger, "model_resumed", model=spec["id"], completed_epoch=first - 1,
               checkpoint_sha256=pointer["sha256"])
    stop_reason = "early_stopping" if stale >= opts["patience"] else "budget"
    elapsed_before = history[-1]["seconds"] if history else 0.
    end = first if stale >= opts["patience"] else (opts["epochs"] + 1 if parameters else 2)
    for epoch in range(first, end):
        model.train()
        order = np.random.permutation(len(rows))
        loss_sum, used = 0., 0
        if optimizer:
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(order), opts["effective_batch"]):
                group = order[offset:offset + opts["effective_batch"]]
                group_used = 0
                for index in group:
                    row = rows[int(index)]
                    context = torch.autocast(device_type="cuda", dtype=torch.float16) if cuda and opts["amp"] else nullcontext()
                    with context:
                        loss = _loss(model, dataset, row, spec, normalizer, opts)
                        if loss is None:
                            continue
                        if not torch.isfinite(loss):
                            raise RuntimeError("Nonfinite model loss; no automatic scientific config changes")
                        weight = len(rows) / (len(counts) * counts[row["building_id"]])
                        weighted = loss * weight / len(group)
                    scaler.scale(weighted).backward()
                    loss_sum += float(loss.detach()) * weight
                    used += 1
                    group_used += 1
                if group_used:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opts["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                    steps += 1
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if not used:
                record = dict(spec=spec, id=spec["id"], status="INSUFFICIENT_DATA", parameters=parameters,
                    reason="No valid supervised TRAIN pixels for this task", contract=contract, optimizer_steps=steps)
                _json(final, record)
                return record
        try:
            val = _validation(model, dataset, spec, normalizer, opts, evaluator)
        except _report().InsufficientData as exc:
            record = dict(spec=spec, id=spec["id"], status="INSUFFICIENT_DATA", parameters=parameters,
                reason=str(exc), contract=contract, optimizer_steps=steps)
            _json(final, record)
            return record
        criterion = _criterion(val)
        improved = criterion > best
        profile = val.get("prediction_score")
        profile_improved = profile is not None and profile < best_prediction - 1e-12
        if improved:
            best, best_epoch, stale = criterion, epoch, 0
        else:
            stale += 1
        if profile_improved:
            best_prediction = profile
            stale = 0
        epoch_record = dict(epoch=epoch, train_loss=loss_sum / used if used else None,
            val={k: val.get(k) for k in ("recall", "fpr", "threshold", "prediction_score")},
            optimizer_steps=steps, seconds=elapsed_before + time.perf_counter() - start)
        history.append(epoch_record)
        # Selection points at immutable snapshots. Prospective pointer has no own hash
        # inside the checkpoint, avoiding a circular hash; JSON pointer owns verification.
        path = root / (PREFIX + f"epoch-{epoch:04d}.pt")
        attempt = 1
        while path.exists():
            # A crash between atomic snapshot and latest-pointer commit may
            # leave an orphan. Preserve it and append a new deterministic retry.
            attempt += 1
            path = root / (PREFIX + f"epoch-{epoch:04d}-attempt-{attempt:03d}.pt")
        if improved:
            selected = dict(path=str(path), epoch=epoch)
        if profile_improved:
            prediction_selected = dict(path=str(path), epoch=epoch)
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict() if optimizer else None,
            scheduler=scheduler.state_dict() if scheduler else None, scaler=scaler.state_dict(), rng=_rng_state(),
            epoch=epoch, steps=steps, best=best, best_prediction=best_prediction,
            best_epoch=best_epoch, stale=stale, selected=selected, prediction_selected=prediction_selected,
            history=history, spec=spec, contract=contract, snapshot_path=str(path),
            peak_vram_bytes=max(previous_peak_vram,torch.cuda.max_memory_allocated(device) if cuda else 0))
        _snapshot(root, epoch, state, contract, ledger)
        _json(root / (PREFIX + "curves.json"), history)
        _event(ledger, "history_scenarios_epoch", model=spec["id"], **epoch_record)
        print(f"{spec['id']} epoch {epoch}/{opts['epochs']} loss={epoch_record['train_loss']} "
              f"VAL recall={val.get('recall')} FPR={val.get('fpr')} NLL={profile}", flush=True)
        # Predictors get opportunity to improve either predeclared selection objective.
        if stale >= opts["patience"]:
            stop_reason = "early_stopping"
            break
    status = "completed" if selected else "INSUFFICIENT_DATA"
    for pointer in (selected, prediction_selected):
        if pointer:
            pointer["sha256"] = sha256(pointer["path"])
    record = dict(spec=spec, id=spec["id"], status=status, contract=contract,
        checkpoint=selected, prediction_checkpoint=prediction_selected, parameters=parameters,
        train_seconds=elapsed_before + time.perf_counter() - start, optimizer_steps=steps,
        peak_vram_bytes=max(previous_peak_vram,torch.cuda.max_memory_allocated(device) if cuda else 0),
        selected_epoch=best_epoch, best_val_recall=best[0] if selected else None,
        best_val_fpr=-best[1] if selected else None,
        best_prediction_score=best_prediction if math.isfinite(best_prediction) else None,
        checkpoint_selection="VAL recall at macro FPR<=.05; lower FPR; earlier epoch",
        prediction_selection="minimum building-macro quantized latent NLL on all/stable VAL respectively",
        stop_reason=stop_reason, completed_epochs=len(history), seed=13,
        reason=None if selected else "VAL lacks defined recall/FPR for selection")
    _json(final, record)
    _event(ledger, "model_selection_frozen", **record)
    return record


def train_all(dataset, out, config, ledger=None, resume=False):
    out = Path(out)
    specs = resolve_matrix(config)
    normalizer = _normalizer(dataset, out, ledger)
    matrix_path = out / (PREFIX + "resolved_matrix.json")
    matrix = matrix_contract(config)
    if matrix_path.exists() and read_json(matrix_path) != matrix:
        raise ValueError("Resolved scientific matrix changed; use a new run")
    completed_path=out/(PREFIX+"maps_manifest.json")
    reduced=options(config)["matrix_protocol"]=="reduced-v2"
    if reduced and completed_path.exists() and read_json(completed_path).get("completed"):
        if not resume:
            raise FileExistsError("Completed scientific export exists; use resume or a new output")
        records=_verify_completed_reduced(out,config,normalizer,read_json(completed_path))
        _event(ledger,"primary_stages_skipped",reason="completed compatible scientific export",records=len(records))
        print(f"history-progress stage=primary-models status=skipped-completed records={len(records)}",flush=True)
        return records
    if not matrix_path.exists():
        _json(matrix_path, matrix)
    records = []
    # Partial export may already contain completed short fits. Preserve their
    # scientific records while primary stages are checked, rather than truncating
    # the shared record index back to its eighteen primary entries.
    head_path=out/(PREFIX+"fusion_records.json")
    preserved_heads=read_json(head_path) if reduced and resume and head_path.exists() else []
    def save_records(primary_records):
        _json(out/(PREFIX+"training_records.json"),[*primary_records,*preserved_heads])
    progression_coverage = []
    for row in dataset.samples:
        if options(config)["matrix_protocol"]=="reduced-v2" and row["split"]=="test":
            continue
        result = _progression(dataset, row, "cpu")
        progression_coverage.append(dict(event_id=row["event_id"], building_id=row["building_id"],
            split=row["split"], coverage=result["coverage"]))
    review = {split: _module("abc").progression_review_manifest(
        [r for r in progression_coverage if r["split"] == split]) for split in ("train", "val", "test")}
    if reduced:
        review_path=out/(PREFIX+"progression_review_manifest.json")
        previous_review=read_json(review_path) if resume and review_path.exists() else {}
        review["test"]=previous_review.get("test",dict(status="NOT_INSPECTED",reason="TEST coverage waits for frozen selections"))
    _json(out / (PREFIX + "progression_review_manifest.json"), review)
    try:
        for spec in specs:
            if spec.get("scenario") == "C" and any(review[x]["status"] != "ELIGIBLE" for x in ("train", "val")):
                records.append(dict(id=spec["id"],spec=spec,status="INSUFFICIENT_DATA",
                    reason="Progression requires at least two buildings and a positive building in TRAIN and VAL",
                    review_manifest=PREFIX + "progression_review_manifest.json"))
                save_records(records)
                continue
            try:
                record = train_variant(spec, dataset, out, config, ledger=ledger,
                                       resume=resume, normalizer=normalizer)
            except _report().InsufficientData as exc:
                record = dict(id=spec["id"], spec=spec, status="INSUFFICIENT_DATA", reason=str(exc))
                _event(ledger, "model_insufficient_data", **record)
            except (RuntimeError, MemoryError) as exc:
                record = dict(id=spec["id"], spec=spec, status="ERROR", reason=str(exc))
                _event(ledger, "model_failed", **record)
            records.append(record)
            save_records(records)
    except BaseException:
        remaining = specs[len(records):]
        save_records(records + [dict(id=s["id"], spec=s,
              status="INCOMPLETE", reason="Execution interrupted before selection") for s in remaining])
        raise
    _event(ledger, "all_model_selections_frozen_before_test", models=records,
           resolved_matrix_sha256=sha256(matrix_path))
    return records


def load_selected(dataset, record, config, prediction_best=False):
    pointer = record.get("prediction_checkpoint") if prediction_best else record.get("checkpoint")
    if pointer is None or sha256(pointer["path"]) != pointer["sha256"]:
        raise ValueError("Selected checkpoint missing or changed")
    model = build_model(record["spec"], dataset, config)
    model.load_state_dict(_load_torch(pointer["path"])["model"])
    model.eval()
    return model


class StackFusion(nn.Module):
    """Same three trainable parameters for every method and no-history control."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1., 0.]))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, stack, signal):
        stack, signal = stack.clamp(1e-5, 1 - 1e-5), signal.clamp(1e-5, 1 - 1e-5)
        return self.bias + self.weight[0] * torch.logit(stack) + self.weight[1] * torch.logit(signal)


def _save_map(out, dataset, row, method, spec, result, ledger, manifest, extra_metadata=None):
    targets = dataset.targets(row)
    arrays = {**targets, "score": np.asarray(result["score"], np.float32),
              "support": np.asarray(result["support"], bool), **result.get("extras", {})}
    arrays["target"] = np.where(arrays["support"], arrays["target"], 255).astype(np.uint8)
    arrays["class_targets"] = np.where(arrays["support"][None], arrays["class_targets"], 255).astype(np.uint8)
    if spec.get("scenario") == "A":
        annotation = dataset.annotation_history(row)
        arrays.update(semantic_target=annotation["labels"][-1].astype(np.uint8),
                      semantic_known=annotation["known"][-1, 0])
    if spec.get("scenario") == "C":
        target = arrays["progression_target"]
        arrays["target"] = np.where(target < 0, 255, target).astype(np.uint8)
        arrays["support"] &= target >= 0
        arrays["class_targets"] = np.full_like(arrays["class_targets"], 255)
    # Keep RGB and dates for a small fixed/common gallery, not every model's tensor RAM.
    if method in ("stack", "stack-original"):
        history = dataset.tracks[row["track_id"]]["observations"][:row["frame"] + 1]
        arrays["history_rgb"] = np.stack([dataset.frame(x["observation_id"])[0] for x in history])
        arrays["years"] = np.asarray([x["year"] for x in history])
    identifier = PREFIX + "scenarios-map-" + _digest([method, row["event_id"]])[:24]
    metadata = dict(unit_id=identifier, kind="history_scenario_float_maps", method=method,
        task=spec["task"], family=spec.get("family"), size=spec.get("size"), control=spec.get("control"),
        scenario=spec.get("scenario", "main"), target_type=spec.get("target_type"), spec=spec,
        **{k: row[k] for k in ("event_id", "building_id", "split", "track_id", "frame",
                               "bbox_xywh", "view_id", "previous_year", "current_year", "known_pixels") if k in row})
    metadata.update(observations=int(row["frame"]) + 1, **(extra_metadata or {}))
    if ledger is not None:
        existing = ledger.read_unit(identifier)
        saved = existing[0] if existing else ledger.save_unit(metadata, arrays)
        metadata.update(path=saved["artifact_path"], artifact_sha256=saved["artifact_sha256"],
                        source_event_id=saved.get("source_event_id"))
    else:
        path = Path(out) / "maps" / (identifier + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            os.replace(temporary, path)
        metadata.update(path=str(path.relative_to(out)), artifact_sha256=sha256(path))
    manifest.append(metadata)
    return metadata


def _map_arrays(out, row):
    path = Path(out) / row["path"]
    if sha256(path) != row["artifact_sha256"]:
        raise ValueError("Saved float-map hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def _fit_fusion(dataset, rows, method, out, config, ledger):
    """Only saved TRAIN maps; equal building weight, bounded one-event memory."""
    opts = options(config)
    root = Path(out) / "fusion" / method
    root.mkdir(parents=True, exist_ok=True)
    final = root / (PREFIX + "fusion.json")
    source_hash = _digest([(r["event_id"], r["artifact_sha256"]) for r in rows])
    if final.exists():
        result = read_json(final)
        if result["source_hash"] != source_hash:
            raise ValueError("Fusion TRAIN map hashes changed")
        pointer = read_json(root / (PREFIX + "latest.json"))
        if sha256(pointer["path"]) != pointer["sha256"]:
            raise ValueError("Fusion checkpoint hash mismatch")
        state = _load_torch(pointer["path"])
        expected_weight = state["model"]["weight"].tolist()
        expected_bias = float(state["model"]["bias"])
        if result["weight"] != expected_weight or result["bias"] != expected_bias:
            raise ValueError("Fusion parameters differ from committed checkpoint")
        return result
    model = StackFusion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts["fusion_lr"], weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.)
    counts = Counter(r["building_id"] for r in rows)
    by_id = {r["event_id"]: r for r in dataset.samples}
    start_epoch = 0
    latest = root / (PREFIX + "latest.json")
    contract = _digest(dict(source_hash=source_hash,epochs=opts["fusion_epochs"],lr=opts["fusion_lr"],source=sha256(__file__)))
    if latest.exists():
        pointer = read_json(latest)
        if pointer["contract"] != contract or sha256(pointer["path"]) != pointer["sha256"]:
            raise ValueError("Fusion resume contract/checkpoint mismatch")
        state = _load_torch(pointer["path"])
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        _restore_rng(state["rng"])
        start_epoch = state["epoch"]
    for epoch in range(start_epoch, opts["fusion_epochs"]):
        optimizer.zero_grad(set_to_none=True)
        for saved in rows:
            if saved["split"] != "train":
                raise ValueError("Fusion fit accepts TRAIN maps only")
            item = _map_arrays(out, saved)
            stack, support = dataset.scores(by_id[saved["event_id"]])
            valid = item["support"] & support & (item["target"] != 255)
            if "forecast_support" in item:
                valid &= item["forecast_support"]
            if not valid.any():
                continue
            a, b = torch.from_numpy(stack[valid]), torch.from_numpy(item["score"][valid])
            target = torch.from_numpy(item["target"][valid].astype(np.float32))
            loss = F.binary_cross_entropy_with_logits(model(a, b), target)
            (loss / (len(counts) * counts[saved["building_id"]])).backward()
        if any(p.grad is not None for p in model.parameters()):
            optimizer.step()
        scheduler.step()
        path = root / (PREFIX + f"epoch-{epoch+1:04d}.pt")
        attempt = 1
        while path.exists():
            attempt += 1
            path = root / (PREFIX + f"epoch-{epoch+1:04d}-attempt-{attempt:03d}.pt")
        _snapshot(root,epoch+1,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),rng=_rng_state(),epoch=epoch+1,snapshot_path=str(path)),contract,ledger)
    result = dict(weight=model.weight.detach().tolist(), bias=float(model.bias.detach()),
        parameters=3, epochs=opts["fusion_epochs"], fit_split="train", source_hash=source_hash,
        scheme="sigmoid(b + w_stack logit(stack) + w_signal logit(signal)); missing forecast falls back to stack")
    _json(final, result)
    _event(ledger, "fusion_fitted", method=method, artifact_path=str(final), artifact_sha256=sha256(final))
    return result


def _fused(result, baseline, fitted):
    signal = np.asarray(result["score"])
    logit = lambda v: np.log(np.clip(v, 1e-5, 1 - 1e-5) / np.clip(1 - v, 1e-5, 1 - 1e-5))
    score = 1 / (1 + np.exp(-np.clip(fitted["bias"] + fitted["weight"][0] * logit(baseline)
                                       + fitted["weight"][1] * logit(signal), -80, 80)))
    available = result.get("extras", {}).get("forecast_support")
    if available is not None:
        score = np.where(available, score, baseline)
    return dict(score=score.astype(np.float32), support=result["support"], extras=result.get("extras", {}))


def _copy_distribution(dataset, normalizer, task, opts):
    sums, count = None, None
    for row in dataset.split_samples("train", predictive=True):
        inputs = _inputs(dataset, row, dict(task=task), normalizer, "cpu")
        target, support = _feature_target(dataset, row, normalizer, "cpu")
        support &= inputs["support"][:, -1] & inputs["support"][:, 0]
        if task == "h0":
            label, known = _labels(dataset, row, "cpu", target.shape[-2:])
            support &= known & (label == 0)
        residual = (target - inputs["features"][:, -1]).square()
        mask = support.expand_as(residual)
        a, n = residual.masked_fill(~mask, 0).sum((0, 2, 3)), mask.sum((0, 2, 3))
        sums, count = (a, n) if sums is None else (sums + a, count + n)
    if sums is None or not bool((count > 0).all()):
        return None
    # Logistic variance = pi^2 scale^2/3, zero-mean error around copy-last.
    return ((sums / count).sqrt() * math.sqrt(3) / math.pi).clamp_min(opts["min_scale"]).numpy()


def _copy_predict(dataset, row, normalizer, task, scale, opts):
    inputs = _inputs(dataset, row, dict(task=task), normalizer, "cpu")
    forecast = inputs["features"][:, -1]  # Completed before target accessor.
    target, target_support = _feature_target(dataset, row, normalizer, "cpu")
    support = inputs["support"][:, -1] & inputs["support"][:, 0] & target_support
    scale = torch.as_tensor(scale)[None, :, None, None].expand_as(forecast)
    nll = _module("models").quantized_logistic_nll(target, forecast, scale,
        step=opts["quantizer_step"], min_scale=opts["min_scale"]).mean(1, keepdim=True)
    residual = (target - forecast).abs().mean(1, keepdim=True)
    shape = dataset.targets(row)["target"].shape
    u, l1 = [_resize(x, shape)[0, 0].numpy() for x in (nll, residual)]
    available = _resize(support.float(), shape, "nearest")[0, 0].bool().numpy()
    score = np.where(available, -np.expm1(-u / 8.), 0)
    return dict(score=score, support=dataset.targets(row)["support"], extras=dict(
        U=u, l1=l1, forecast=forecast[0].numpy(), scale=scale[0].numpy(),
        residual_score=-np.expm1(-l1), forecast_support=available,
        cosine=_resize(1 - F.cosine_similarity(target, forecast,dim=1)[:,None],shape)[0,0].numpy()))



def _stack_contract(dataset):
    # Reuse the data adapter's exact channel order and native-mask policies.
    # In particular AnyChange must use OR on overlaps, never probability mean.
    return dataset.stack_contract if hasattr(dataset, "stack_contract") else None


def _cached_result(ledger, method, event_id):
    if ledger is None:
        return None
    identifier=PREFIX+"scenarios-map-"+_digest([method,event_id])[:24]
    saved=ledger.read_unit(identifier)
    if saved is None:
        return None
    item=saved[1]
    excluded=("score","support","target","class_targets","before","after")
    return dict(score=item["score"],support=item["support"],extras={k:v for k,v in item.items() if k not in excluded})


def export_all(dataset, records, out, config, ledger=None, resume=False):
    if options(config)["matrix_protocol"] == "reduced-v2":
        return _export_reduced(dataset,records,out,config,ledger=ledger,resume=resume)
    """Export TRAIN/VAL, freeze all heads/thresholds, then access TEST.

    Completed RunStore maps skip model forward on partial resume. Report-only
    reads the manifest; it never calls this function. Model fitting and export
    are sequential and bounded by one event and at most three small heads.
    """
    out,opts=Path(out),options(config)
    normalizer=_normalizer(dataset,out,ledger)
    manifest_path=out/(PREFIX+"maps_manifest.json")
    if manifest_path.exists():
        old=read_json(manifest_path)
        if old.get("completed"):
            if old["config_hash"]!=_digest(config) or old.get("normalizer_sha256")!=_digest(normalizer.state_dict()):
                raise ValueError("Completed export scientific configuration/normalization changed")
            for item in old["maps"]:
                _map_arrays(out,item)
            return old["maps"]
    selected=[r for r in records if r["status"]=="completed"]
    by_id={r["id"]:r for r in selected}
    manifest=[]
    fusions={}
    stack_contract=_stack_contract(dataset)
    copy_scales={task:_copy_distribution(dataset,normalizer,task,opts) for task in ("actual-next","h0")}
    for task,scale in copy_scales.items():
        if scale is not None:
            _json(out/(PREFIX+task+"-copy-last.json"),dict(scale=scale.tolist(),fit_split="train",
                distribution="logistic",mean="previous feature",quantizer_step=opts["quantizer_step"]))

    def save(row,method,spec,result,metadata=None):
        return _save_map(out,dataset,row,method,spec,result,ledger,manifest,metadata)

    def baseline(split):
        for row in dataset.split_samples(split):
            for method in (("stack","stack-original") if stack_contract else ("stack",)):
                spec=dict(id=method,task="direct",scenario="baseline",family="stack",size="frozen",
                          control="stack",target_type="annotation-derived latest change")
                result=_cached_result(ledger,method,row["event_id"])
                if result is None:
                    score,support=dataset.scores(row)
                    if method=="stack" and hasattr(dataset,"evaluation_support"):
                        support=dataset.evaluation_support(row)
                    extras={}
                    if stack_contract:
                        extras["stack_raw"]=dataset.stack_components(row)[0]
                    result=dict(score=score,support=support,extras=extras)
                save(row,method,spec,result,dict(stack_contract=stack_contract,original_full_support=method=="stack-original"))

    def models(split):
        for record in selected:
            spec,method=record["spec"],record["id"]
            model=None
            last_model=None
            last_record=by_id.get(f"{spec['task']}-{spec.get('family')}-{spec.get('size')}-last-only")
            profile_model=None
            for row in dataset.split_samples(split):
                result=_cached_result(ledger,method,row["event_id"])
                if result is None:
                    if model is None:
                        model=load_selected(dataset,record,opts)
                    result=prediction(model,dataset,row,spec,normalizer,opts)
                    if last_record and spec.get("control")=="history":
                        if last_model is None:
                            last_model=load_selected(dataset,last_record,opts)
                        last=prediction(last_model,dataset,row,last_record["spec"],normalizer,opts)
                        result["extras"]["G"]=last["extras"]["U"]-result["extras"]["U"]
                save(row,method,spec,result,dict(checkpoint_selection="downstream",
                    checkpoint_sha256=record["checkpoint"]["sha256"]))
                if spec["task"] in ("actual-next","h0"):
                    residual_method=method+"-residual"
                    residual=_cached_result(ledger,residual_method,row["event_id"])
                    if residual is None:
                        residual={**result,"score":np.where(result["extras"]["forecast_support"],result["extras"]["residual_score"],0)}
                    save(row,residual_method,{**spec,"id":residual_method},residual,dict(checkpoint_selection="downstream"))
                if spec.get("control")=="history" and spec.get("scenario")=="main":
                    for intervention in ("repeat","shuffle-common","shuffle"):
                        name=method+"-intervention-"+intervention
                        perturbed=_cached_result(ledger,name,row["event_id"])
                        if perturbed is None:
                            if model is None:
                                model=load_selected(dataset,record,opts)
                            perturbed=prediction(model,dataset,row,spec,normalizer,opts,intervention)
                        save(row,name,spec,perturbed,dict(inference_variant=intervention,
                            threshold_parent=method,threshold_reference=method,
                            checkpoint_sha256=record["checkpoint"]["sha256"],checkpoint_selection="same-frozen-checkpoint"))
                if spec["task"] in ("actual-next","h0") and record.get("prediction_checkpoint"):
                    name=method+"-prediction-best"
                    result_profile=_cached_result(ledger,name,row["event_id"])
                    if result_profile is None:
                        if profile_model is None:
                            profile_model=load_selected(dataset,record,opts,prediction_best=True)
                        result_profile=prediction(profile_model,dataset,row,spec,normalizer,opts)
                    save(row,name,spec,result_profile,dict(checkpoint_selection="prediction",
                        checkpoint_sha256=record["prediction_checkpoint"]["sha256"]))
            del model,last_model,profile_model
        for task,scale in copy_scales.items():
            if scale is None:
                continue
            spec=dict(id=task+"-copy-last",task=task,scenario="main",family="copy-last",size="fixed",
                      control="copy-last",target_type="annotation-derived latest change")
            for row in dataset.split_samples(split):
                result=_cached_result(ledger,spec["id"],row["event_id"])
                if result is None:
                    result=_copy_predict(dataset,row,normalizer,task,scale,opts)
                    result["support"]=dataset.evaluation_support(row) if hasattr(dataset,"evaluation_support") else result["support"]
                save(row,spec["id"],spec,result)

    def fusion(split):
        by_event={r["event_id"]:r for r in dataset.split_samples(split)}
        raw=[r for r in manifest if r["split"]==split and r["method"] in fusions]
        for saved in raw:
            method=saved["method"]
            row=by_event[saved["event_id"]]
            fused=_cached_result(ledger,method+"+stack",row["event_id"])
            if fused is None:
                item=_map_arrays(out,saved)
                result=dict(score=item["score"],support=item["support"],extras={k:v for k,v in item.items() if k not in ("score","support")})
                fused=_fused(result,dataset.scores(row)[0],fusions[method])
                signal=np.clip(result["score"],1e-5,1-1e-5)
                fused["extras"]["fusion_signal_logit"]=np.log(signal/(1-signal)).astype(np.float32)
                if stack_contract:
                    fused["extras"]["stack_raw"]=dataset.stack_components(row)[0]
            save(row,method+"+stack",{**saved["spec"],"id":method+"+stack"},fused,
                dict(stack_contract=stack_contract,fusion_parameters=fusions[method]))

    baseline("train")
    models("train")
    # Same fixed-capacity fusion for every raw/residual predictor and all controls.
    fusion_methods=sorted({r["method"] for r in manifest if r["split"]=="train" and
        r.get("scenario") not in ("C","baseline") and not r.get("inference_variant") and
        r.get("checkpoint_selection")!="prediction"})
    for method in fusion_methods:
        fusions[method]=_fit_fusion(dataset,[r for r in manifest if r["method"]==method and r["split"]=="train"],method,out,opts,ledger)
    fusion("train")
    baseline("val")
    models("val")
    fusion("val")
    decisions={}
    val_by_method=defaultdict(list)
    for saved in manifest:
        if saved["split"]=="val":
            val_by_method[saved["method"]].append(saved)
    for budget in config.get("fpr_budgets",(.05,.10,.15)):
        decisions[str(budget)]={}
        # Parents first; intervention methods copy thresholds afterwards.
        for method,maps in sorted(val_by_method.items(),key=lambda pair:bool(pair[1][0].get("threshold_parent"))):
            metadata=maps[0]
            parent=metadata.get("threshold_parent")
            if parent:
                decisions[str(budget)][method]={**decisions[str(budget)][parent],"threshold_parent":parent}
                continue
            lookup={r["event_id"]:r for r in maps}
            def cached(row):
                item=_map_arrays(out,lookup[row["event_id"]])
                return dict(score=item["score"],support=item["support"],
                    extras={k:v for k,v in item.items() if k not in ("score","support")},
                    stack_contract=metadata.get("stack_contract"),fusion_parameters=metadata.get("fusion_parameters"))
            try:
                val=_report().evaluate_predictor(dataset,cached,split="val",spec={**metadata["spec"],
                    "original_full_support":metadata.get("original_full_support",False),
                    "stack_contract":metadata.get("stack_contract"),"fusion_parameters":metadata.get("fusion_parameters")},budget=budget)
                decisions[str(budget)][method]={k:val.get(k) for k in ("threshold","recall","fpr")}
            except _report().InsufficientData as exc:
                decisions[str(budget)][method]=dict(status="INSUFFICIENT_DATA",reason=str(exc))
    _json(out/(PREFIX+"frozen_val_decisions.json"),decisions)
    _event(ledger,"all_val_decisions_frozen_before_test",methods=len(val_by_method),budgets=list(decisions),
           artifact_sha256=sha256(out/(PREFIX+"frozen_val_decisions.json")))
    baseline("test")
    models("test")
    fusion("test")
    canonical=getattr(dataset,"canonical",dataset)
    categories=canonical.categories
    groups=getattr(canonical,"groups",[])
    _json(manifest_path,dict(completed=True,config_hash=_digest(config),normalizer_sha256=_digest(normalizer.state_dict()),
        maps=manifest,categories=categories,groups=groups,
        interpretation="exploratory; single seed13; model bits are not coded-stream sizes",
        forecast_support="all compared methods share first/previous/current token geometry; original stack full-support diagnostic separate",
        old_history_support="old support intersect last-past support identically for all learned controls"))
    return manifest


def _semantic_val_miou(dataset,predict):
    """Canonical overlap, then per-building semantic-class IoU (not change IoU)."""
    base=dataset.evaluation_for(split="val") if hasattr(dataset,"evaluation_for") else getattr(dataset,"evaluation_data",getattr(dataset,"canonical",dataset))
    rows=dataset.split_samples("val")
    per_building=defaultdict(list)
    by_id={r["event_id"]:r for r in rows}
    for channel,category in enumerate(base.categories):
        def target(event):
            row=by_id[event]
            annotation=dataset.annotation_history(row)
            labels=annotation["labels"][-1,channel].astype(np.uint8)
            support=dataset.evaluation_support(row)
            valid=annotation["known"][-1,0]&support
            labels=np.where(valid,labels,255).astype(np.uint8)
            return dict(target=labels,class_targets=labels[None],support=support,
                before=np.ones(labels.shape,np.uint16),after=np.ones(labels.shape,np.uint16))
        adapter=SimpleNamespace(samples=rows,categories=[category],groups=[],target=target)
        histogram,_=_report().metrics.collect(adapter,rows,
            lambda row:(predict(row)["extras"]["semantic_probability"][channel],predict(row)["support"]))
        for building,hist in histogram.items():
            value=_report().metrics.metrics(hist[0],.5)["iou"]
            if value is not None: per_building[building].append(value)
    return float(np.mean([np.mean(v) for v in per_building.values()])) if per_building else None


def _primary_f1(result):
    value=result.get("f1")
    fpr=result.get("fpr")
    if value is None or fpr is None or not np.isfinite(value) or fpr>.05+1e-12:
        return -math.inf
    return float(value)


def _train_reduced_variant(spec,dataset,out_dir,config,evaluator=None,ledger=None,resume=False,normalizer=None):
    opts=options(config)
    normalizer=normalizer or _normalizer(dataset,out_dir,ledger)
    root=Path(out_dir)/"models"/spec["id"]
    root.mkdir(parents=True,exist_ok=True)
    final=root/(PREFIX+"selection.json")
    contract=_digest(dict(protocol="history-three-scenarios-reduced-v2",spec=spec,config=config,
        source_sha256=sha256(__file__),normalizer_sha256=_digest(normalizer.state_dict())))
    manager=_module("checkpoints").CheckpointManager(root,contract,ledger=ledger,restore_contract=dict(spec=spec,options=opts))
    if final.exists():
        record=read_json(final)
        if not resume or record["contract"]!=contract: raise ValueError("Completed model/config mismatch")
        for pointer in record.get("retained",[]):
            if sha256(pointer["path"])!=pointer["sha256"]: raise ValueError("Selected weights hash mismatch")
        _event(ledger,"model_stage_skipped",model=spec["id"],reason="completed compatible primary fit")
        print(f"history-progress model={spec['id']} status=skipped-completed",flush=True)
        return record
    random.seed(13); np.random.seed(13); torch.manual_seed(13)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(13)
    model=build_model(spec,dataset,opts)
    device=opts["device"]; cuda=device.startswith("cuda")
    if cuda: torch.cuda.reset_peak_memory_stats(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=opts["lr"],weight_decay=opts["weight_decay"])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=opts["epochs"])
    scaler=torch.amp.GradScaler("cuda",enabled=cuda and opts["amp"]) if hasattr(getattr(torch,"amp",None),"GradScaler") else torch.cuda.amp.GradScaler(enabled=cuda and opts["amp"])
    rows=dataset.split_samples("train",predictive=spec["task"] in ("actual-next","h0"))
    counts=Counter(r["building_id"] for r in rows)
    evaluator=evaluator or _report().evaluate_predictor
    first,steps,stale,best_f1=1,0,0,-math.inf
    history=[]; peak=0; elapsed_before=0.
    state=manager.resume()
    if state is not None:
        if not resume: raise FileExistsError("Incomplete reduced fit exists; use resume")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state["scaler"])
        _restore_rng(state["rng"])
        first=state["epoch"]+1;steps=state["steps"];stale=state["stale"];best_f1=state["best_f1"]
        history=state["history"];peak=state.get("peak_vram_bytes",0);elapsed_before=state.get("active_seconds",0.)
        _event(ledger,"model_resumed",model=spec["id"],completed_epoch=first-1)
    started=time.perf_counter();reason="early_stopping" if stale>=opts["patience"] else "budget"
    end=first if stale>=opts["patience"] else opts["epochs"]+1
    try:
        for epoch in range(first,end):
            model.train();optimizer.zero_grad(set_to_none=True)
            order=np.random.permutation(len(rows));loss_sum=0.;used=0
            with _module("progress").Progress(stage="TRAIN",model=spec["id"],epoch=epoch,total=len(rows),ledger=ledger) as progress:
                for offset in range(0,len(order),opts["effective_batch"]):
                    group=order[offset:offset+opts["effective_batch"]];group_used=0
                    for index in group:
                        row=rows[int(index)]
                        context=torch.autocast(device_type="cuda",dtype=torch.float16) if cuda and opts["amp"] else nullcontext()
                        with context:
                            loss=_loss(model,dataset,row,spec,normalizer,opts)
                            if loss is None:
                                progress.update(skipped_unknown=True);continue
                            if not torch.isfinite(loss): raise RuntimeError("Nonfinite loss; scientific budget unchanged")
                            weight=len(rows)/(len(counts)*counts[row["building_id"]])
                            objective=loss*weight/len(group)
                        scaler.scale(objective).backward();loss_sum+=float(loss.detach())*weight;used+=1;group_used+=1
                        progress.update(loss=loss_sum/used,last_val_epoch=history[-1]["epoch"] if history else None,
                            last_val_f1=history[-1]["val"]["f1"] if history else None,best_val_f1=best_f1 if math.isfinite(best_f1) else None)
                    if group_used:
                        scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),opts["grad_clip"])
                        scaler.step(optimizer);scaler.update();steps+=1
                    optimizer.zero_grad(set_to_none=True)
            if not used: raise _report().InsufficientData("No supervised TRAIN pixels for this task")
            scheduler.step()
            with _module("progress").Progress(stage="VAL",model=spec["id"],epoch=epoch,total=1,ledger=ledger) as progress:
                val=_validation(model,dataset,spec,normalizer,opts,evaluator)
                progress.update(val_f1=val.get("f1"),val_recall=val.get("recall"),val_fpr=val.get("fpr"))
            f1=_primary_f1(val)
            if not math.isfinite(f1): raise _report().InsufficientData("No defined VAL F1 at the requested FPR budget")
            if f1>best_f1+1e-12: best_f1=f1;stale=0
            else: stale+=1
            epoch_record=dict(epoch=epoch,train_loss=loss_sum/used,val={k:val.get(k) for k in
                ("f1","iou","recall","fpr","threshold","prediction_score","semantic_miou")},optimizer_steps=steps,
                seconds=elapsed_before+time.perf_counter()-started)
            history.append(epoch_record)
            peak=max(peak,torch.cuda.max_memory_allocated(device) if cuda else 0)
            state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),rng=_rng_state(),epoch=epoch,steps=steps,stale=stale,best_f1=best_f1,
                history=history,peak_vram_bytes=peak,active_seconds=epoch_record["seconds"],spec=spec,contract=contract)
            manager.save_epoch(model.state_dict(),state,epoch_record["val"],epoch)
            _json(root/(PREFIX+"curves.json"),history)
            _event(ledger,"history_scenarios_epoch",model=spec["id"],**epoch_record)
            if stale>=opts["patience"]: reason="early_stopping";break
    except _report().InsufficientData as exc:
        record=dict(id=spec["id"],spec=spec,status="INSUFFICIENT_DATA",reason=str(exc),contract=contract,
                    optimizer_steps=steps,completed_epochs=len(history))
        _json(final,record);return record
    selection=manager.finalize();primary=selection.get("checkpoint")
    record=dict(id=spec["id"],spec=spec,status="completed" if primary else "INSUFFICIENT_DATA",contract=contract,
        **{k:v for k,v in selection.items() if k not in ("latest","contract")},parameters=sum(p.numel() for p in model.parameters()),
        train_seconds=elapsed_before+time.perf_counter()-started,optimizer_steps=steps,peak_vram_bytes=peak,
        selected_epoch=primary.get("epoch") if primary else None,completed_epochs=len(history),stop_reason=reason,seed=13,
        primary_checkpoint_id=primary.get("sha256") if primary else None,
        checkpoint_selection="primary VAL building-macro F1 at recall-selected FPR<=.05; earlier epoch ties",
        early_stopping="strict primary F1 improvement only; secondary metrics never reset patience",fit_kind="primary")
    _json(final,record);_event(ledger,"model_selection_frozen",**record)
    return record


def _frozen_predict(store,lookup,row):
    arrays=store.read(lookup[row["event_id"]],fields=["score","support","stack_raw"])
    return dict(score=arrays["score"],support=arrays["support"],stack_raw=arrays["stack_raw"],
        stack_contract=lookup[row["event_id"]].get("source_stack_contract",lookup[row["event_id"]].get("stack_contract")))


def _score_blocks(dataset,store,lookup,split,halo=2):
    data=dataset.evaluation_for(split=split) if hasattr(dataset,"evaluation_for") else getattr(dataset,"evaluation_data",getattr(dataset,"canonical",dataset))
    return _module("fusion").iter_score_blocks(data,dataset.split_samples(split),
        lambda row:_frozen_predict(store,lookup,row),halo=halo)


def _block_tensors(block,device):
    return {key:_tensor(block[key],device,torch.bool if key=="support" else torch.float32)[None]
            for key in ("stack","method","support","target")}


@contextmanager
def _score_outputs(head,dataset,store,lookup,split,device,scratch):
    """Merge frozen inputs then head with halo. Temporary VAL blocks are retired.

    Cropped outputs agree exactly on overlaps; the canonical evaluator's final
    averaging therefore preserves the nonlinear-head result. Memory is bounded
    by one halo block and one event; no whole facade canvas is allocated.
    """
    scratch=Path(scratch);scratch.mkdir(parents=True,exist_ok=True)
    head.eval()
    with tempfile.TemporaryDirectory(prefix="2026-11-08-fusion-blocks-",dir=scratch) as directory:
        paths={}
        with torch.inference_mode():
            for key,block in _score_blocks(dataset,store,lookup,split):
                tensors=_block_tensors(block,device)
                probability=_module("fusion").apply_score_tile(head,tensors["stack"],tensors["method"],
                    tensors["support"],block["core_slices"])[0,0].cpu().numpy().astype(np.float32)
                path=Path(directory)/(_digest(key)+".npy")
                np.save(path,probability,allow_pickle=False);paths[key]=path
        def predicted(row):
            x,y,width,height=row["bbox_xywh"]
            scene=tuple(row[k] for k in ("building_id","view_id","previous_year","current_year"))
            score=np.zeros((height,width),np.float32)
            for by in range(y//256,(y+height-1)//256+1):
                for bx in range(x//256,(x+width-1)//256+1):
                    block=np.load(paths[(scene,(bx,by))],mmap_mode="r",allow_pickle=False)
                    left,top=max(x,bx*256),max(y,by*256)
                    right,bottom=min(x+width,(bx+1)*256),min(y+height,(by+1)*256)
                    score[top-y:bottom-y,left-x:right-x]=block[top-by*256:bottom-by*256,left-bx*256:right-bx*256]
            return dict(score=score,support=dataset.evaluation_support(row),extras={})
        yield predicted


def _feature_head_inputs(store,parent,device):
    arrays=store.read(parent,fields=["forecast","target_features","forecast_support","stack_score","support","target"])
    current=_tensor(arrays["target_features"],device)[None]
    forecast=_tensor(arrays["forecast"],device)[None]
    shape=current.shape[-2:]
    support=F.adaptive_avg_pool2d(_tensor(np.asarray(arrays["forecast_support"]) & np.asarray(arrays["support"]),device)[None,None],shape)>=1-1e-6
    stack=F.adaptive_avg_pool2d(_tensor(arrays["stack_score"],device)[None,None],shape)
    target=_tensor(arrays["target"],device)[None,None]
    known=(_tensor(arrays["support"],device,torch.bool)[None,None])&(target!=255)
    return current,forecast,support,stack,target,known


def _feature_head_predict(head,store,parent,device):
    current,forecast,support,stack,target,known=_feature_head_inputs(store,parent,device)
    with torch.inference_mode():
        probability,_=_module("fusion").feature_prediction(head,current,forecast,support,stack,output_size=target.shape[-2:])
    # The fixed common target support is shared with every comparator.
    arrays=store.read(parent,fields=["support"])
    return dict(score=probability[0,0].cpu().numpy(),support=arrays["support"],extras={})


def _train_reduced_fusion(spec,dataset,store,parent_maps,out,config,ledger=None,resume=False):
    opts=options(config);kind=spec["fusion_kind"];device=opts["device"]
    feature=kind in ("feature-mlp","cross-attention")
    lookup={r["event_id"]:r for r in parent_maps}
    train_rows=dataset.split_samples("train")
    channels=int(store.read(lookup[train_rows[0]["event_id"]],fields=["forecast"])["forecast"].shape[0]) if feature else None
    root=Path(out)/"fusion"/spec["id"];root.mkdir(parents=True,exist_ok=True)
    final=root/(PREFIX+"selection.json")
    contract=_digest(dict(spec=spec,options=opts,source=sha256(__file__),parent_maps=[(r["event_id"],r["artifact_sha256"]) for r in parent_maps]))
    manager=_module("checkpoints").CheckpointManager(root,contract,ledger=ledger,restore_contract=dict(spec=spec,channels=channels))
    if final.exists():
        record=read_json(final)
        if not resume or record["contract"]!=contract: raise ValueError("Fusion completed contract mismatch")
        manager.finalize()
        _event(ledger,"fusion_stage_skipped",model=spec["id"])
        return record
    torch.manual_seed(13);np.random.seed(13);random.seed(13)
    head=_module("fusion").build_head(kind,channels=channels).to(device)
    optimizer=torch.optim.Adam(head.parameters(),lr=opts["fusion_lr"],weight_decay=opts["fusion_weight_decay"])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda epoch:1.)
    cuda=device.startswith("cuda")
    scaler=torch.amp.GradScaler("cuda",enabled=cuda and opts["amp"]) if hasattr(getattr(torch,"amp",None),"GradScaler") else torch.cuda.amp.GradScaler(enabled=cuda and opts["amp"])
    totals=Counter()
    block_count=0
    if feature:
        for row in train_rows:
            arrays=store.read(lookup[row["event_id"]],fields=["target","support"])
            totals[row["building_id"]]+=int(np.count_nonzero((arrays["target"]!=255)&arrays["support"]))
        block_count=len(train_rows)
    else:
        for key,block in _score_blocks(dataset,store,lookup,"train"):
            core=block["core_slices"]
            totals[key[0][0]]+=int(np.count_nonzero((block["target"][(0,*core)]!=255)&block["support"][(0,*core)]))
            block_count+=1
    totals={k:v for k,v in totals.items() if v>0}
    if not totals:
        record=dict(id=spec["id"],spec=spec,status="INSUFFICIENT_DATA",reason="No known TRAIN pixels",contract=contract)
        _json(final,record);return record
    first,steps=1,0;curves=[];elapsed=0.;peak=0
    state=manager.resume()
    if state is not None:
        if not resume: raise FileExistsError("Fusion incomplete; use resume")
        head.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]);scaler.load_state_dict(state["scaler"]);_restore_rng(state["rng"])
        first=state["epoch"]+1;steps=state["steps"];curves=state["history"];elapsed=state["active_seconds"];peak=state.get("peak_vram_bytes",0)
    if cuda: torch.cuda.reset_peak_memory_stats(device)
    started=time.perf_counter()
    for epoch in range(first,opts["fusion_epochs"]+1):
        head.train();optimizer.zero_grad(set_to_none=True);loss_total=0.;used=0;pending=0
        iterable=((row,None) for row in train_rows) if feature else _score_blocks(dataset,store,lookup,"train")
        with _module("progress").Progress(stage="fusion-TRAIN",model=spec["id"],epoch=epoch,total=block_count,ledger=ledger) as progress:
            for number,(key,block) in enumerate(iterable,1):
                context=torch.autocast(device_type="cuda",dtype=torch.float16) if cuda and opts["amp"] else nullcontext()
                with context:
                    if feature:
                        row=key;building=row["building_id"]
                        current,forecast,support,stack,target,known=_feature_head_inputs(store,lookup[row["event_id"]],device)
                        probability,_=_module("fusion").feature_prediction(head,current,forecast,support,stack,output_size=target.shape[-2:])
                        with torch.autocast(device_type="cuda",enabled=False) if cuda else nullcontext():
                            losses=F.binary_cross_entropy(probability.float().clamp(1e-6,1-1e-6),target.float().masked_fill(~known,0),reduction="none")
                    else:
                        building=key[0][0];tensors=_block_tensors(block,device);core=block["core_slices"]
                        logits=_module("fusion").score_logits(head,tensors["stack"],tensors["method"],tensors["support"])[...,core[0],core[1]]
                        target=tensors["target"][...,core[0],core[1]]
                        known=tensors["support"][...,core[0],core[1]]&(target!=255)
                        losses=F.binary_cross_entropy_with_logits(logits,target.masked_fill(~known,0),reduction="none")
                    objective=None
                    if building in totals and known.any():
                        loss=losses.masked_select(known).sum()/totals[building]/len(totals)
                        # Constant scaling: a partial/unknown final group must not
                        # increase the relative contribution of its buildings.
                        objective=loss*block_count/opts["effective_batch"]
                if objective is not None:
                    if not torch.isfinite(objective): raise RuntimeError("Nonfinite fusion loss")
                    scaler.scale(objective).backward();loss_total+=float(loss.detach());used+=1;pending+=1
                if (number%opts["effective_batch"]==0 or number==block_count) and pending:
                    scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(head.parameters(),opts["grad_clip"])
                    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);steps+=1;pending=0
                progress.update(loss=loss_total,last_val_epoch=curves[-1]["epoch"] if curves else None,
                    last_val_f1=curves[-1]["val"]["f1"] if curves else None)
        if not used: raise _report().InsufficientData("Fusion TRAIN has no usable supervision")
        scheduler.step();head.eval()
        with _module("progress").Progress(stage="fusion-VAL",model=spec["id"],epoch=epoch,total=1,ledger=ledger) as progress:
            if feature:
                val=_report().evaluate_predictor(dataset,lambda row:_feature_head_predict(head,store,lookup[row["event_id"]],device),split="val",spec=spec,budget=.05)
            else:
                with _score_outputs(head,dataset,store,lookup,"val",device,Path(out)/"work") as predict:
                    val=_report().evaluate_predictor(dataset,predict,split="val",spec=spec,budget=.05)
            progress.update(val_f1=val.get("f1"),val_fpr=val.get("fpr"))
        if not math.isfinite(_primary_f1(val)): raise _report().InsufficientData("Fusion VAL F1 is undefined")
        metrics={k:val.get(k) for k in ("f1","iou","recall","fpr","threshold")}
        curves.append(dict(epoch=epoch,train_loss=loss_total,val=metrics,optimizer_steps=steps,seconds=elapsed+time.perf_counter()-started))
        peak=max(peak,torch.cuda.max_memory_allocated(device) if cuda else 0)
        state=dict(model=head.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
            rng=_rng_state(),epoch=epoch,steps=steps,history=curves,active_seconds=curves[-1]["seconds"],peak_vram_bytes=peak)
        manager.save_epoch(head.state_dict(),state,metrics,epoch)
        _json(root/(PREFIX+"curves.json"),curves)
    selection=manager.finalize();primary=selection.get("checkpoint")
    record=dict(id=spec["id"],spec=spec,status="completed" if primary else "INSUFFICIENT_DATA",contract=contract,
        **{k:v for k,v in selection.items() if k not in ("contract","latest")},parent_method=spec["parent_method"],
        base_method=spec["parent_method"],fusion_kind=kind,fit_kind=spec["fit_kind"],channels=channels,
        parameters=sum(p.numel() for p in head.parameters()),optimizer_steps=steps,train_seconds=elapsed+time.perf_counter()-started,
        peak_vram_bytes=peak,completed_epochs=len(curves),selected_epoch=primary.get("epoch") if primary else None,
        fusion_checkpoint_id=primary.get("sha256") if primary else None,checkpoint_selection="VAL building-macro F1; earlier epoch ties",
        matched_control_method=spec["matched_control_method"],fit_split="train",seed=13,
        objective="BCE; equal total known-pixel mean weight per building; fixed five/default maximum ten epochs",
        optimizer="Adam",lr=opts["fusion_lr"],weight_decay=opts["fusion_weight_decay"])
    _json(final,record);_event(ledger,"fusion_selection_frozen",**record)
    return record


def _load_head(record,device):
    pointer=record["checkpoint"]
    if sha256(pointer["path"])!=pointer["sha256"]: raise ValueError("Fusion winner hash mismatch")
    head=_module("fusion").build_head(record["fusion_kind"],channels=record.get("channels")).to(device)
    head.load_state_dict(_load_torch(pointer["path"])["model"])
    return head.eval()


def _export_reduced(dataset,records,out,config,ledger=None,resume=False):
    """18 fixed base fits, 56 short frozen-input heads, one primary TEST export."""
    out=Path(out);opts=options(config);device=opts["device"]
    store=_module("storage").SharedMapStore(out,ledger=ledger,dataset=dataset)
    normalizer=_normalizer(dataset,out,ledger)
    path=out/(PREFIX+"maps_manifest.json")
    head_path=out/(PREFIX+"fusion_records.json")
    if path.exists() and read_json(path).get("completed"):
        document=read_json(path)
        if document["config_hash"]!=_digest(config) or document["normalizer_sha256"]!=_digest(normalizer.state_dict()):
            raise ValueError("Completed scientific export configuration changed")
        saved_records=_verify_completed_reduced(out,config,normalizer,document)
        store.verify(document["maps"])
        records[:]=saved_records
        return document["maps"]
    primary=[r for r in records if r.get("fit_kind","primary")=="primary" and r["status"]=="completed"]
    primary.sort(key=lambda r:(r["spec"].get("control") in ("history","learned"),r["id"]))
    by_id={r["id"]:r for r in primary}
    contract=dataset.stack_contract
    copy_scales={task:_copy_distribution(dataset,normalizer,task,opts) for task in ("actual-next","h0")}
    for task,scale in copy_scales.items():
        if scale is not None:
            _json(out/(PREFIX+task+"-copy-last.json"),dict(scale=scale.tolist(),fit_split="train",distribution="logistic",
                quantizer_step=opts["quantizer_step"],normalizer_sha256=_digest(normalizer.state_dict())))

    def metadata(spec,record=None,variant="standalone"):
        parent=spec.get("parent_method",spec["id"])
        base=by_id.get(parent)
        return dict(base_method=parent,parent_method=spec.get("parent_method"),score_variant=variant,
            fusion_kind=spec.get("fusion_kind"),matched_control_method=spec.get("matched_control_method",matched_control(spec)) if spec.get("scenario")!="baseline" else None,
            primary_checkpoint_id=base.get("primary_checkpoint_id") if base else None,
            checkpoint_sha256=base["checkpoint"]["sha256"] if base else None,
            fusion_checkpoint_id=record.get("fusion_checkpoint_id") if record and spec.get("fusion_kind") else None,
            checkpoint_selection="primary-val-f1",canonical_score=bool(spec.get("fusion_kind")))

    def references(split):
        for row in _module("progress").progress(dataset.split_samples(split),stage="export-references-"+split,ledger=ledger):
            for name in ("stack","stack-original"):
                if store.get(name,row["event_id"]): continue
                score,support=dataset.scores(row)
                if name=="stack": support=dataset.evaluation_support(row)
                spec=dict(id=name,task="direct",scenario="baseline",family="stack",size="frozen",control="stack",target_type="annotation-derived latest change")
                store.save(row,name,spec,dict(score=score,support=support,extras=dict(stack_raw=dataset.stack_components(row)[0],stack_score=score)),
                    dict(**metadata(spec),stack_contract=contract,original_full_support=name=="stack-original"))
            for task,scale in copy_scales.items():
                name=task+"-copy-last"
                if scale is None or store.get(name,row["event_id"]): continue
                spec=dict(id=name,task=task,scenario="main",family="copy-last",size="fixed",control="copy-last",target_type="annotation-derived latest change")
                result=_copy_predict(dataset,row,normalizer,task,scale,opts)
                result["support"]=dataset.evaluation_support(row)
                store.save(row,name,spec,result,metadata(spec,variant="U"))
        # Independent semantic processing is a fixed reference, never a neural fit.
        spec=dict(id="A-independent",task="semantic-state",scenario="A",mode="independent",family="semantic-memory",size="shared",hidden=64,control="independent",target_type="semantic state")
        model=None
        for row in dataset.split_samples(split):
            if store.get(spec["id"],row["event_id"]): continue
            if model is None: model=build_model(spec,dataset,opts)
            result=prediction(model,dataset,row,spec,normalizer,opts)
            store.save(row,spec["id"],spec,result,dict(**metadata(spec),fixed_reference=True))
        del model

    def base_outputs(split):
        for record in primary:
            spec,name=record["spec"],record["id"]
            model=None
            for row in _module("progress").progress(dataset.split_samples(split),stage="export-base-"+split,model=name,ledger=ledger):
                saved=store.get(name,row["event_id"])
                if saved is None:
                    if model is None: model=load_selected(dataset,record,opts)
                    result=prediction(model,dataset,row,spec,normalizer,opts)
                    result["extras"]["stack_raw"]=dataset.stack_components(row)[0]
                    result["extras"]["stack_score"]=dataset.scores(row)[0]
                    if spec["task"] in ("actual-next","h0") and spec["control"]=="history":
                        last=store.get(matched_control(spec),row["event_id"])
                        if last is not None:
                            result["extras"]["G"]=np.asarray(store.read(last,fields=["U"])["U"])-result["extras"]["U"]
                    saved=store.save(row,name,spec,result,dict(**metadata(spec,record,"U" if spec["task"] in ("actual-next","h0") else "standalone"),source_stack_contract=contract))
                if spec["task"] in ("actual-next","h0"):
                    residual=name+"-residual"
                    if store.get(residual,row["event_id"]) is None:
                        arrays=store.read(saved,fields=["residual_score","support"])
                        residual_spec={**spec,"id":residual}
                        meta=metadata(spec,record,"residual")
                        meta["matched_control_method"]=matched_control(spec)+"-residual"
                        store.save(row,residual,residual_spec,dict(score=arrays["residual_score"],support=arrays["support"]),meta,
                            refs=store.references(saved))
            del model

    for split in ("train","val"):
        references(split);base_outputs(split)
    _event(ledger,"primary_train_val_scientific_cache_frozen",models=len(primary),primary_checkpoint_export_only=True)
    heads=[]
    for spec in resolve_fusion_matrix(config):
        if spec["parent_method"] not in by_id:
            heads.append(dict(id=spec["id"],spec=spec,status="INCOMPLETE",reason="Primary model unavailable",fit_kind=spec["fit_kind"],parent_method=spec["parent_method"]))
            continue
        parent_maps=sorted([r for r in store.records if r["method"]==spec["parent_method"] and r["split"] in ("train","val")],key=lambda r:(r["split"],r["event_id"]))
        try:
            record=_train_reduced_fusion(spec,dataset,store,parent_maps,out,config,ledger=ledger,resume=resume)
        except _report().InsufficientData as exc:
            record=dict(id=spec["id"],spec=spec,status="INSUFFICIENT_DATA",reason=str(exc),fit_kind=spec["fit_kind"],parent_method=spec["parent_method"])
        except (RuntimeError,MemoryError) as exc:
            record=dict(id=spec["id"],spec=spec,status="ERROR",reason=str(exc),fit_kind=spec["fit_kind"],parent_method=spec["parent_method"])
        heads.append(record);_json(head_path,heads)
    completed_heads=[r for r in heads if r["status"]=="completed"]

    def head_outputs(split):
        rows=dataset.split_samples(split)
        for record in completed_heads:
            name,spec=record["id"],record["spec"]
            pending=[row for row in rows if store.get(name,row["event_id"]) is None]
            if not pending:
                print(f"history-progress stage=export-fusion-{split} model={name} status=skipped-completed",flush=True)
                continue
            lookup={r["event_id"]:r for r in store.records if r["method"]==spec["parent_method"] and r["split"]==split}
            head=_load_head(record,device)
            context=_score_outputs(head,dataset,store,lookup,split,device,out/"work") if spec["fit_kind"]=="score-fusion" else nullcontext(None)
            with context as predicted:
                for row in _module("progress").progress(pending,stage="export-fusion-"+split,model=name,ledger=ledger):
                    parent=lookup[row["event_id"]]
                    result=predicted(row) if predicted else _feature_head_predict(head,store,parent,device)
                    meta=metadata(spec,record,spec["fusion_kind"])
                    meta["overlap_order"]="canonical inputs merge -> head with halo2 -> unique core -> crop" if spec["fit_kind"]=="score-fusion" else "native feature head -> sigmoid -> bilinear pixel resize -> canonical overlap mean"
                    store.save(row,name,spec,result,meta,refs=store.references(parent))
            del head

    head_outputs("train");head_outputs("val")
    # Primary scores, residuals, fixed references and all heads get VAL-only decisions.
    decisions={};val_methods=defaultdict(list)
    for record in store.records:
        if record["split"]=="val": val_methods[record["method"]].append(record)
    for budget in config.get("fpr_budgets",(.05,.10,.15)):
        decisions[str(budget)]={}
        for method,maps in _module("progress").progress(sorted(val_methods.items()),stage="freeze-VAL-thresholds",model=str(budget),ledger=ledger):
            meta=maps[0];lookup={r["event_id"]:r for r in maps}
            def predict(row):
                arrays=store.read(lookup[row["event_id"]])
                extras={}
                for key in ("progression_target",):
                    if key in arrays:extras[key]=arrays[key]
                if meta.get("scenario")=="baseline":extras["stack_raw"]=arrays["stack_raw"]
                return dict(score=arrays["score"],support=arrays["support"],extras=extras,
                    stack_contract=contract if meta.get("scenario")=="baseline" else None,
                    canonical_score=meta.get("canonical_score",False))
            try:
                val=_report().evaluate_predictor(dataset,predict,split="val",spec={**meta["spec"],
                    "original_full_support":meta.get("original_full_support",False)},budget=budget)
                decisions[str(budget)][method]={k:val.get(k) for k in ("threshold","f1","iou","recall","fpr")}
            except _report().InsufficientData as exc:
                decisions[str(budget)][method]=dict(status="INSUFFICIENT_DATA",reason=str(exc))
    _json(out/(PREFIX+"frozen_val_decisions.json"),decisions)
    selection_metadata={r["id"]:{**r.get("spec",{}),**r} for r in [*records,*heads]}
    for saved in store.records:
        selection_metadata.setdefault(saved["method"],{**saved.get("spec",{}),**saved})
    overall={(method,"val",.05):value for method,value in decisions["0.05"].items()}
    main=_report().choose_main_variants(selection_metadata,overall)
    _json(out/(PREFIX+"main_selection.json"),dict(selected=main,selection_split="val",criterion="building-macro F1 at recall-selected FPR<=.05"))
    _event(ledger,"all_val_decisions_frozen_before_test",methods=len(val_methods),budgets=list(decisions),
        thresholds_sha256=sha256(out/(PREFIX+"frozen_val_decisions.json")),main_selection_sha256=sha256(out/(PREFIX+"main_selection.json")))
    references("test");base_outputs("test");head_outputs("test")
    review_path=out/(PREFIX+"progression_review_manifest.json")
    review=read_json(review_path) if review_path.exists() else {}
    test_coverage=[]
    for row in dataset.split_samples("test"):
        target=_progression(dataset,row,"cpu")
        test_coverage.append(dict(event_id=row["event_id"],building_id=row["building_id"],split="test",coverage=target["coverage"]))
    review["test"]=_module("abc").progression_review_manifest(test_coverage)
    _json(review_path,review)
    existing={r["id"] for r in records}
    records.extend(r for r in heads if r["id"] not in existing)
    _json(out/(PREFIX+"training_records.json"),records)
    counts=dict(planned=dict(primary=18,score_fusion=48,feature_fusion=8,total=74),
        completed={kind:sum(r["status"]=="completed" and r.get("fit_kind","primary")==kind for r in records)
                   for kind in ("primary","score-fusion","feature-fusion")})
    _json(out/(PREFIX+"fit_counts.json"),counts)
    canonical=getattr(dataset,"canonical",dataset)
    manifest=store.records
    _json(path,dict(completed=True,storage_schema=2,config_hash=_digest(config),normalizer_sha256=_digest(normalizer.state_dict()),
        maps=manifest,categories=canonical.categories,groups=getattr(canonical,"groups",[]),fit_counts=counts,
        storage=store.storage_summary(),interpretation="exploratory TEST; primary checkpoint only; no coded-stream compression claim"))
    return manifest



def _verify_completed_reduced(out,config,normalizer,document):
    """Verify a completed run's complete fit index and retained weights read-only."""
    out=Path(out)
    if document.get("config_hash")!=_digest(config) or document.get("normalizer_sha256")!=_digest(normalizer.state_dict()):
        raise ValueError("Completed scientific export configuration/normalization changed")
    records=read_json(out/(PREFIX+"training_records.json"))
    expected={spec["id"]:spec for spec in [*resolve_matrix(config),*resolve_fusion_matrix(config)]}
    if len(records)!=len(expected) or {r["id"] for r in records}!=set(expected):
        raise ValueError("Completed training_records does not contain the full planned primary/fusion matrix")
    heads=read_json(out/(PREFIX+"fusion_records.json"))
    by_id={r["id"]:r for r in records}
    if any(by_id.get(row["id"])!=row for row in heads):
        raise ValueError("Completed fusion records disagree with the full training index")
    if {row["id"] for row in heads}!={spec["id"] for spec in resolve_fusion_matrix(config)}:
        raise ValueError("Completed fusion record coverage is incomplete")
    options_hash=options(config)
    verified={}
    for record in records:
        spec=expected[record["id"]]
        if record.get("spec")!=spec:
            raise ValueError("Completed fit specification changed")
        if record.get("status")!="completed":
            continue
        fusion=record.get("fit_kind","primary")!="primary"
        if fusion:
            parent_maps=sorted([row for row in document["maps"] if row["method"]==spec["parent_method"] and row["split"] in ("train","val")],key=lambda row:(row["split"],row["event_id"]))
            contract=_digest(dict(spec=spec,options=options_hash,source=sha256(__file__),
                parent_maps=[(row["event_id"],row["artifact_sha256"]) for row in parent_maps]))
        else:
            contract=_digest(dict(protocol="history-three-scenarios-reduced-v2",spec=spec,config=config,
                source_sha256=sha256(__file__),normalizer_sha256=_digest(normalizer.state_dict())))
        if record.get("contract")!=contract:
            raise ValueError("Completed fit source/config/cache contract changed")
        selection_path=out/("fusion" if fusion else "models")/record["id"]/(PREFIX+"selection.json")
        if read_json(selection_path)!=record:
            raise ValueError("Completed fit selection differs from the retained scientific index")
        pointers=[*record.get("retained",[]),*[record.get(key) for key in ("checkpoint","prediction_checkpoint","semantic_checkpoint")]]
        if not record.get("checkpoint"):
            raise ValueError("Completed fit lacks its primary checkpoint")
        for pointer in pointers:
            if pointer is None:continue
            path=Path(pointer["path"])
            key=str(path)
            if key not in verified:verified[key]=sha256(path)
            if verified[key]!=pointer["sha256"]:
                raise ValueError("Completed retained primary/fusion checkpoint hash mismatch")
    return records
