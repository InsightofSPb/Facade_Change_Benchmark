"""Fixed downstream fusion heads for the reduced history experiment.

These heads consume frozen predictions. The score-head contract is *merge raw
stack components and the principal method score on the canonical grid, compute
the stack score, clip/logit the two signals, then apply the head*. In particular,
averaging crop-level nonlinear head outputs is not this contract. A score CNN
has radius two and block processing must include its two-pixel input halo.

Feature heads instead operate on the complete native feature grid of each crop
(16 x 16 tokens for a real 224 x 224 crop), resize the resulting probability map
with bilinear align_corners=False, and then merge overlaps arithmetically. Both
feature heads use exactly this order. Annotation/known masks only enter loss;
neither kind of head accepts them as an input feature.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import importlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SCORE_KINDS = ("logreg", "score-mlp", "score-cnn")
FEATURE_KINDS = ("feature-mlp", "cross-attention")
KINDS = SCORE_KINDS + FEATURE_KINDS
SCORE_CLIP = 1e-6
SCHEMA = "history-scenarios-fusion-v1"
SCORE_ORDER = "canonical_raw_stack_OR_and_mean_method_then_stack_then_clip_logit_then_head"
FEATURE_ORDER = "native_full_crop_head_then_sigmoid_then_bilinear_align_corners_false_then_overlap_mean"


def _support(value, shape, device):
    value = torch.as_tensor(value, device=device, dtype=torch.bool)
    if value.ndim == 3:
        value = value[:, None]
    if tuple(value.shape) != (shape[0], 1, shape[-2], shape[-1]):
        raise ValueError("Geometric support must have shape B,1,H,W")
    return value


def _one_channel(value, name):
    if value.ndim == 3:
        value = value[:, None]
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(name + " must have shape B,1,H,W")
    return value


def _masked_frozen(value, support, name):
    # where, not multiplication: an invalid NaN must not propagate through a
    # convolution/attention neighbourhood. Geometry, never GT, defines this.
    value = torch.where(support, value.detach(), torch.zeros_like(value))
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name + " contains non-finite values on geometric support")
    return value


def score_inputs(stack, method, support, clip=SCORE_CLIP):
    """Return the identical two frozen logit channels used by all score heads."""
    if not 0 < clip < .5:
        raise ValueError("Score clipping must be strictly between zero and one half")
    stack, method = _one_channel(stack, "stack"), _one_channel(method, "method")
    if stack.shape != method.shape or stack.device != method.device:
        raise ValueError("Stack and method must share shape and device")
    mask = _support(support, stack.shape, stack.device)
    values = torch.cat((_masked_frozen(stack, mask, "stack"),
                        _masked_frozen(method, mask, "method")), 1)
    values = torch.logit(values.clamp(clip, 1 - clip))
    return torch.where(mask, values, torch.zeros_like(values))


class ScoreHead(nn.Module):
    """Fixed score-only architecture; forward receives prepared logit channels."""

    def __init__(self, kind):
        super().__init__()
        self.kind, self.channels = kind, None
        self.halo = 2 if kind == "score-cnn" else 0
        if kind == "logreg":
            self.net = nn.Conv2d(2, 1, 1)
            nn.init.ones_(self.net.weight)
            nn.init.zeros_(self.net.bias)
        elif kind == "score-mlp":
            self.net = nn.Sequential(nn.Conv2d(2, 32, 1), nn.GELU(), nn.Conv2d(32, 1, 1))
        elif kind == "score-cnn":
            self.net = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1), nn.GELU(),
                nn.Conv2d(16, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 1, 1))
        else:
            raise ValueError("Unknown score head: " + str(kind))

    def forward(self, values):
        if values.ndim != 4 or values.shape[1] != 2:
            raise ValueError("Score head requires B,2,H,W prepared logit channels")
        return self.net(values)


def coordinates(height, width, device, dtype):
    """Explicit x,y crop coordinates; no positions inferred from annotation."""
    y = torch.linspace(-1, 1, height, device=device, dtype=dtype)
    x = torch.linspace(-1, 1, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), 0)[None]


def positional_encoding(height, width, device, dtype, dimension=64):
    """Fixed two-dimensional sine/cosine positions shared by feature heads."""
    if dimension % 4:
        raise ValueError("2D positional encoding dimension must be divisible by four")
    coords = coordinates(height, width, device, dtype)
    frequency = torch.exp(torch.arange(dimension // 4, device=device, dtype=dtype)
                          * (-math.log(10000.) / max(dimension // 4 - 1, 1)))
    x = coords[:, :1] * math.pi * frequency[None, :, None, None]
    y = coords[:, 1:] * math.pi * frequency[None, :, None, None]
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), 1)


class FeatureHead(nn.Module):
    """Shared inputs, projections, coordinates and final stack combination."""

    def __init__(self, kind, channels):
        super().__init__()
        if kind not in FEATURE_KINDS or channels < 1:
            raise ValueError("A feature head requires its kind and positive channel count")
        self.kind, self.channels, self.halo = kind, int(channels), None
        self.current_projection = nn.Conv2d(channels, 64, 1)
        self.forecast_projection = nn.Conv2d(channels, 64, 1)
        if kind == "cross-attention":
            self.attention = nn.MultiheadAttention(64, 4, dropout=0., batch_first=True)
            self.norm_attention = nn.LayerNorm(64)
            self.ffn = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
            self.norm_ffn = nn.LayerNorm(64)
        # Raw coordinates and geometric support are available to both models.
        width = (128 if kind == "feature-mlp" else 64) + 3
        self.pointwise = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, 1, 1))
        # Both get precisely the same final learned combination with stack.
        self.stack_combination = nn.Conv2d(2, 1, 1)
        nn.init.ones_(self.stack_combination.weight)
        nn.init.zeros_(self.stack_combination.bias)

    def forward(self, current, forecast, support, stack):
        if current.ndim != 4 or current.shape[1] != self.channels or current.shape != forecast.shape:
            raise ValueError("Current and forecast must share B,C,H,W and the configured channels")
        mask = _support(support, current.shape, current.device)
        current = _masked_frozen(current, mask, "current features")
        forecast = _masked_frozen(forecast, mask, "forecast mean")
        stack = _one_channel(stack, "stack")
        if tuple(stack.shape) != tuple(mask.shape):
            raise ValueError("Feature stack score must be resized to the same native grid")
        stack = _masked_frozen(stack, mask, "stack")
        height, width = current.shape[-2:]
        position = positional_encoding(height, width, current.device, current.dtype)
        q = torch.where(mask, self.current_projection(current) + position, 0.)
        kv = torch.where(mask, self.forecast_projection(forecast) + position, 0.)
        if self.kind == "feature-mlp":
            representation = torch.cat((q, kv), 1)
        else:
            query = q.flatten(2).transpose(1, 2)
            keys = kv.flatten(2).transpose(1, 2)
            invalid = ~mask.flatten(2).squeeze(1)
            # MultiheadAttention is undefined when every key is masked. A zero
            # dummy key is unmasked solely for this case, and every query/output
            # remains explicitly masked; no attention matrices are retained.
            safe_invalid = invalid.clone()
            empty = invalid.all(1)
            safe_invalid[empty, 0] = False
            attended, _ = self.attention(query, keys, keys,
                key_padding_mask=safe_invalid, need_weights=False)
            attended = torch.where(invalid[..., None], 0., attended)
            value = self.norm_attention(query + attended)
            value = self.norm_ffn(value + self.ffn(value))
            value = torch.where(invalid[..., None], 0., value)
            representation = value.transpose(1, 2).reshape(-1, 64, height, width)
        coords = coordinates(height, width, current.device, current.dtype).expand(current.shape[0], -1, -1, -1)
        auxiliary = torch.cat((mask.to(current.dtype), coords), 1)
        latent = self.pointwise(torch.cat((representation, auxiliary), 1))
        stack_logit = torch.where(mask, torch.logit(stack.clamp(SCORE_CLIP, 1 - SCORE_CLIP)), 0.)
        logits = self.stack_combination(torch.cat((stack_logit, latent), 1))
        return torch.where(mask, logits, torch.zeros_like(logits))


def build_head(kind, channels=None):
    if kind in SCORE_KINDS:
        return ScoreHead(kind)
    if kind in FEATURE_KINDS:
        if channels is None:
            raise ValueError("Feature channel count is required")
        return FeatureHead(kind, int(channels))
    raise ValueError("Unknown fusion kind: " + str(kind))


def parameter_count(head):
    return sum(parameter.numel() for parameter in head.parameters())


def head_contract(head):
    return dict(schema=SCHEMA, kind=head.kind, channels=head.channels,
        parameters=parameter_count(head), frozen_base=True, fit_split="train", selection_split="val",
        annotation_masks_are_features=False, score_clip=SCORE_CLIP, unsupported_input="zero after logit",
        unsupported_output="zero probability; excluded by geometric support", halo=head.halo,
        order=SCORE_ORDER if head.kind in SCORE_KINDS else FEATURE_ORDER,
        feature_inputs=(None if head.kind in SCORE_KINDS else
            ["TRAIN-normalized current spatial features", "causal H0 mean", "geometric support",
             "same stack score", "fixed 2D positions"]),
        feature_native_grid=(None if head.kind in SCORE_KINDS else [16, 16]),
        feature_grid_smoke_exception=True,
        flops="NOT_MEASURED", measured_time="NOT_MEASURED")


def serialize_head(head):
    """Small reproducible head payload; large base weights are never included."""
    return dict(kind=head.kind, channels=head.channels, contract=head_contract(head),
        state_dict={name: value.detach().cpu().tolist() for name, value in head.state_dict().items()})


def restore_head(payload, device="cpu"):
    # Construction consumes RNG; restoring an exported head must not perturb
    # either a primary model's initialization or a resumed training trajectory.
    with torch.random.fork_rng(devices=[]):
        result = build_head(payload["kind"], payload.get("channels"))
    reference = result.state_dict()
    state = {name: torch.as_tensor(value, dtype=reference[name].dtype)
             for name, value in payload["state_dict"].items()}
    result.load_state_dict(state, strict=True)
    if payload.get("contract", {}).get("order", head_contract(result)["order"]) != head_contract(result)["order"]:
        raise ValueError("Fusion reconstruction order differs from the saved contract")
    return result.to(device).eval()


def score_logits(head, stack, method, support):
    if head.kind not in SCORE_KINDS:
        raise ValueError("score_logits requires a score head")
    values = score_inputs(stack, method, support)
    logits = head(values)
    return torch.where(_support(support, logits.shape, logits.device), logits, torch.zeros_like(logits))


def score_prediction(head, stack, method, support):
    logits = score_logits(head, stack, method, support)
    return torch.where(_support(support, logits.shape, logits.device), logits.sigmoid(), 0.)


def feature_logits(head, current, forecast, support, stack):
    if head.kind not in FEATURE_KINDS:
        raise ValueError("feature_logits requires a feature head")
    return head(current, forecast, support, stack)


def feature_prediction(head, current, forecast, support, stack, output_size=None):
    """Identical native-head/sigmoid/resize/mask policy for both feature heads."""
    logits = feature_logits(head, current, forecast, support, stack)
    mask = _support(support, logits.shape, logits.device)
    probability = torch.where(mask, logits.sigmoid(), 0.)
    if output_size is not None and tuple(output_size) != tuple(probability.shape[-2:]):
        probability = F.interpolate(probability, size=output_size, mode="bilinear", align_corners=False)
        mask = F.interpolate(mask.float(), size=output_size, mode="nearest").bool()
        probability = torch.where(mask, probability, 0.)
    return probability, mask


def apply_score_tile(head, stack, method, support, core=None):
    """Evaluate a supplied merged-input halo tile; return its core probability.

    The caller must supply at least head.halo real neighbouring pixels outside
    the core, except at a true raster edge. Padding a block edge with zeros is
    not equivalent to padding the complete image edge.
    """
    probability = score_prediction(head, stack, method, support)
    return probability if core is None else probability[..., core[0], core[1]]


def apply_score_blocks(head, stack, method, support, block_size=256):
    """Exact differentiable tiled forward on already merged raster inputs."""
    if block_size < 1 or head.kind not in SCORE_KINDS:
        raise ValueError("Positive block size and score head required")
    stack, method = _one_channel(stack, "stack"), _one_channel(method, "method")
    mask = _support(support, stack.shape, stack.device)
    height, width = stack.shape[-2:]
    radius = head.halo
    result_rows = []
    for top in range(0, height, block_size):
        result_columns = []
        bottom = min(top + block_size, height)
        for left in range(0, width, block_size):
            right = min(left + block_size, width)
            y0, y1 = max(0, top - radius), min(height, bottom + radius)
            x0, x1 = max(0, left - radius), min(width, right + radius)
            core = slice(top - y0, bottom - y0), slice(left - x0, right - x0)
            result_columns.append(apply_score_tile(head, stack[..., y0:y1, x0:x1],
                method[..., y0:y1, x0:x1], mask[..., y0:y1, x0:x1], core))
        result_rows.append(torch.cat(result_columns, -1))
    return torch.cat(result_rows, -2)


def iter_score_blocks(data, samples, predict, halo=2, block_size=256, cache_events=16):
    """Yield canonical merged frozen inputs with a real neighbouring halo.

    ``data`` is the existing evaluator adapter (``categories``, ``groups``,
    ``target(event_id)``); ``predict(row)`` returns ``stack_raw``,
    ``stack_contract``, ``score`` and geometric ``support``. Raw stack may also
    live in ``extras``. The result is ``((scene_key, (bx, by)), block)`` where
    ``scene_key`` is (building, view, previous_year, current_year). Block arrays
    stack/method/support/target have shape 1,H,W. ``core_slices`` excludes the
    halo, so fitting/counting does not count neighbouring pixels twice.

    The existing canonical merger supplies mean continuous signals, native
    AnyChange OR, and conflict-to-unknown labels. Translation changes only the
    grid origin. Prediction callbacks receive the original row. At most
    ``cache_events`` prediction records are retained; no full facade raster is
    allocated. Annotation labels are returned separately, never score inputs.
    """
    if block_size < 1 or halo < 0 or cache_events < 1:
        raise ValueError("Positive block/cache sizes and nonnegative halo required")
    metrics = importlib.import_module("facade_change.research.2026-11-04_history_benchmark_metrics")
    report = importlib.import_module("data_analysis.2026-11-08_history_scenarios_report")
    originals = {row["event_id"]: row for row in samples}
    if len(originals) != len(samples):
        raise ValueError("Score-fusion samples contain duplicate event IDs")

    @lru_cache(maxsize=cache_events)
    def prediction(event_id):
        return predict(originals[event_id])

    def fields(result):
        extra = result.get("extras", {})
        raw = result.get("stack_raw", extra.get("stack_raw", extra.get("stack_raw7")))
        contract = result.get("stack_contract", extra.get("stack_contract"))
        if raw is None or contract is None:
            raise ValueError("Canonical fusion needs raw stack channels and their frozen transform contract")
        score = result.get("score", result.get("scores"))
        if score is None:
            raise ValueError("Canonical fusion needs the principal method score")
        return np.asarray(raw), contract, np.asarray(score), np.asarray(result["support"], bool)

    groups = defaultdict(list)
    for row in samples:
        groups[tuple(row[k] for k in ("building_id", "view_id", "previous_year", "current_year"))].append(row)
    for scene, rows in sorted(groups.items()):
        prediction.cache_clear()
        first_raw, contract, _, _ = fields(prediction(rows[0]["event_id"]))
        names = contract.get("method_names", [""] * len(first_raw))
        if len(names) != len(first_raw):
            raise ValueError("Frozen stack channel names and raw maps disagree")
        native = [name == "anychange" for name in names] + [False]
        keys = set()
        for row in rows:
            x, y, width, height = row["bbox_xywh"]
            keys.update((bx, by) for bx in range(x // block_size, (x + width - 1) // block_size + 1)
                        for by in range(y // block_size, (y + height - 1) // block_size + 1))
        for bx, by in sorted(keys):
            origin_x, origin_y = bx * block_size - halo, by * block_size - halo
            extent = block_size + 2 * halo
            translated = []
            for row in rows:
                x, y, width, height = row["bbox_xywh"]
                if x < origin_x + extent and y < origin_y + extent and x + width > origin_x and y + height > origin_y:
                    translated.append(dict(row, bbox_xywh=[x - origin_x, y - origin_y, width, height]))

            def channels(row):
                raw, current_contract, score, support = fields(prediction(row["event_id"]))
                if current_contract != contract:
                    raise ValueError("Frozen stack contract changes within a spatial transition")
                if score.ndim == 3 and score.shape[0] == 1:
                    score = score[0]
                return np.concatenate((raw, score[None]), axis=0), support

            found = False
            for _, position, values, truth, geometric, conflict in metrics.merged_blocks(
                    data, translated, channels, native, block_size=extent):
                if position != (0, 0):
                    continue
                found = True
                logits = report._stack_logits(values[:-1], contract)
                stack = np.exp(-np.logaddexp(0., -logits))
                stack = np.where(geometric, stack, 0.).astype(np.float32)
                method = np.where(geometric, values[-1], 0.).astype(np.float32)
                yield (scene, (bx, by)), dict(stack=stack[None], method=method[None],
                    support=geometric[None], target=truth[:1], truth=truth, conflict=conflict,
                    core_slices=(slice(halo, halo + block_size), slice(halo, halo + block_size)),
                    origin_xy=(origin_x, origin_y), core_origin_xy=(bx * block_size, by * block_size),
                    block_size=block_size, halo=halo, order=SCORE_ORDER)
                break
            if not found:
                raise RuntimeError("Canonical score-fusion block unexpectedly absent")


def masked_building_bce(logits, target, known, support, building_ids=None):
    """BCE with equal building contribution; GT/known enter only this loss.

    With streamed batches, the trainer additionally uses inverse event counts
    per building across an epoch. This batch helper groups repeated building
    IDs instead of weighting buildings by their pixel or crop counts.
    """
    target = _one_channel(target, "target")
    mask = _support(support, logits.shape, logits.device)
    known = _support(known, logits.shape, logits.device)
    valid = mask & known & ((target == 0) | (target == 1))
    if target.shape != logits.shape:
        raise ValueError("BCE target and logits must share shape")
    cleaned_target = torch.where(valid, target, 0.).to(logits.dtype)
    losses = F.binary_cross_entropy_with_logits(logits, cleaned_target, reduction="none")
    ids = list(range(logits.shape[0])) if building_ids is None else list(building_ids)
    if len(ids) != logits.shape[0]:
        raise ValueError("One building ID per batch item is required")
    grouped = defaultdict(list)
    for index, building in enumerate(ids):
        grouped[building].append(index)
    contributions = []
    for indices in grouped.values():
        observed = valid[indices]
        count = observed.sum()
        if bool(count):
            contributions.append(torch.where(observed, losses[indices], 0.).sum() / count)
    return torch.stack(contributions).mean() if contributions else logits.sum() * 0.


def fusion_training_contract(epochs=5, lr=.001, weight_decay=0.):
    if not isinstance(epochs, int) or not 1 <= epochs <= 10:
        raise ValueError("Fusion epochs must be explicitly set between one and ten; default is five")
    if lr <= 0 or weight_decay < 0:
        raise ValueError("Fusion learning rate must be positive and weight decay nonnegative")
    return dict(epochs=epochs, optimizer="Adam", lr=float(lr), weight_decay=float(weight_decay),
        loss="BCEWithLogits on known annotation AND geometry", building_weight="equal buildings",
        fit_split="train", selection_split="val", base_networks="frozen cached outputs",
        epoch_selection="maximum building-macro F1 at VAL max-recall FPR<=0.05 threshold; earlier epoch on tie",
        hyperparameter_search=False, automatic_epoch_increase=False)
