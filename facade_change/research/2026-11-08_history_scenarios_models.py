"""Spatial temporal heads and parallel quantized-latent probability models.

DINO is an external frozen encoder. Every parameter in this module is trained
from scratch. Factorized attention adapts the temporal/spatial separation in
TSViT (CVPR 2023) and TimeSformer (ICML 2021), using PyTorch attention primitives;
it does not import their training framework or claim video pretraining.

Inputs use one fixed TRAIN-fitted feature space, and a past-defined grid. No
normalization, registration, image selection or labels occur inside forward.
"""
from __future__ import annotations

import importlib
import math
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

ConvGRUCell = importlib.import_module(
    "facade_change.research.2026-10-25_spatial_change_model").ConvGRUCell

SIZES = {
    "smoke": dict(width=16, gru_depth=1, transformer_depth=1, heads=4),
    "medium": dict(width=128, gru_depth=2, transformer_depth=4, heads=8),
    "large": dict(width=192, gru_depth=3, transformer_depth=7, heads=8),
}
TASKS = ("direct", "actual-next", "h0")
CONTROLS = ("history", "pair-only", "last-only", "mean", "repeat-history")


def _check_inputs(features, support, years, target_year, padding, task):
    if features.ndim != 5:
        raise ValueError("Features must be B,T,C,H,W spatial maps")
    b, t, _, h, w = features.shape
    if support.shape != (b, t, 1, h, w) or years.shape != (b, t):
        raise ValueError("Support or observation-time shape does not match features")
    if target_year.shape != (b,) or padding.shape != (b, t):
        raise ValueError("Target times B and padding B,T are required")
    padding = padding.bool()
    lengths = padding.sum(1)
    if bool((lengths < (2 if task == "direct" else 1)).any()):
        raise ValueError("Too few observations for this task")
    if not torch.equal(padding, torch.arange(t, device=features.device)[None] < lengths[:, None]):
        raise ValueError("Padding must be a right-padded valid prefix (True means valid)")
    for row, n in enumerate(lengths.tolist()):
        date = years[row, :n]
        if not bool(torch.isfinite(date).all()) or not bool(torch.isfinite(target_year[row])):
            raise ValueError("Observed dates must be finite, with recorded date precision")
        if bool((date[1:] <= date[:-1]).any()):
            raise ValueError("Observations must be strictly chronological")
        if task == "direct" and not bool(target_year[row] == date[-1]):
            raise ValueError("Direct target time must equal the current observation time")
        if task != "direct" and not bool(target_year[row] > date[-1]):
            raise ValueError("Forecast accepts strictly past observations only")
    valid = support.bool() & padding[:, :, None, None, None]
    if not bool(torch.isfinite(features.masked_select(valid.expand_as(features))).all()):
        raise ValueError("Supported feature values must be finite")
    return valid, lengths


def recompute_deltas(features, support, kind="signed"):
    """Adjacent differences are invalid unless both endpoint tokens exist."""
    if kind not in ("signed", "absolute"):
        raise ValueError("Delta must be signed or absolute")
    valid = support.bool()
    clean = features.masked_fill(~valid, 0)
    delta = torch.zeros_like(clean)
    delta_valid = torch.zeros_like(valid)
    delta_valid[:, 1:] = valid[:, 1:] & valid[:, :-1]
    delta[:, 1:] = (clean[:, 1:] - clean[:, :-1]).masked_fill(~delta_valid[:, 1:], 0)
    if kind == "absolute":
        delta = delta.abs()
    return delta, delta_valid


def intervene_history(features, support, padding=None, mode="repeat", preserve=2, seed=13):
    """Replace/reorder older content while preserving the last ``preserve`` frames.

    Timestamp slots, padding and support are unchanged. Repeat uses the first
    preserved frame (previous for a direct pair, last past for a predictor).
    It fails if that frame is unsupported where an old frame is supported: the
    caller must first construct a common-support cohort for BOTH variants.
    Derived features are deliberately not accepted; recompute them afterwards.
    """
    if mode not in ("repeat", "shuffle") or preserve < 1:
        raise ValueError("Unknown intervention or invalid preserved suffix")
    valid = support.bool()
    if padding is None:
        padding = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
    valid = valid & padding[:, :, None, None, None]
    result = features.masked_fill(~valid, 0).clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for row, n in enumerate(padding.sum(1).tolist()):
        old = n - preserve
        if old <= 0:
            continue
        if mode == "repeat":
            if bool((valid[row, :old] & ~valid[row, old:old + 1]).any()):
                raise ValueError("repeat intervention requires common source support")
            result[row, :old] = result[row, old:old + 1].expand_as(result[row, :old])
        else:
            order = torch.randperm(old, generator=generator).to(features.device)
            # Fixed support is essential: do not move a missing token into data.
            source_valid = valid[row, :old][order]
            if bool((valid[row, :old] & ~source_valid).any()):
                raise ValueError("shuffle intervention requires common old-frame support")
            result[row, :old] = result[row, :old][order]
    return result.masked_fill(~valid, 0), support.clone()


def _last(values, lengths, offset=1):
    return values[torch.arange(len(values), device=values.device), lengths - offset]


def _control_inputs(features, support, years, padding, lengths, control, task):
    """Apply one information intervention before deriving delta or time inputs."""
    b, _, c, h, w = features.shape
    if control == "history":
        return features, support, years, padding
    if control == "repeat-history":
        value, valid = intervene_history(features, support, padding, "repeat", 2 if task == "direct" else 1)
        return value, valid, years, padding
    if control in ("pair-only", "last-only"):
        take = 2 if task == "direct" else 1
        indices = lengths[:, None] - take + torch.arange(take, device=features.device)[None]
        rows = torch.arange(b, device=features.device)[:, None]
        return features[rows, indices], support[rows, indices], years[rows, indices], padding.new_ones((b, take))
    # Orderless aggregation. One pooled past map is presented at the last known
    # past time. No averaged/fabricated date or adjacent historical delta exists.
    past = support.clone()
    if task == "direct":
        past[torch.arange(b, device=features.device), lengths - 1] = False
    value = (features.masked_fill(~past, 0)).sum(1) / past.sum(1).clamp_min(1)
    valid = past.any(1)
    if task == "direct":
        values = torch.stack((value, _last(features, lengths)), 1)
        masks = torch.stack((valid, _last(support, lengths)), 1)
        dates = torch.stack((_last(years, lengths, 2), _last(years, lengths)), 1)
    else:
        values, masks, dates = value[:, None], valid[:, None], _last(years, lengths)[:, None]
    return values, masks, dates, padding.new_ones(dates.shape)


def _time_channels(years, target_year, padding, time_mode):
    dates = years.masked_fill(~padding, 0)
    b, t = dates.shape
    order = torch.arange(t, device=dates.device, dtype=dates.dtype)[None].expand(b, t)
    gap = torch.zeros_like(dates)
    gap[:, 1:] = (dates[:, 1:] - dates[:, :-1]).clamp_min(0)
    age = (target_year[:, None] - dates).clamp_min(0)
    if time_mode == "index":
        gap = (order > 0).to(dates.dtype)
        # Preserve the order and one nominal forecast step, removing real gaps.
        is_forecast = (target_year > _last(dates, padding.sum(1))).to(dates.dtype)
        age = padding.sum(1)[:, None].to(dates.dtype) - 1 + is_forecast[:, None] - order
    elif time_mode == "none":
        order, gap, age = torch.zeros_like(dates), torch.zeros_like(dates), torch.zeros_like(dates)
    values = torch.stack((torch.log1p(order), torch.log1p(gap), torch.log1p(age)), 2)
    return values.masked_fill(~padding[:, :, None], 0)


def _position_2d(height, width, channels, device, dtype):
    """Fixed 2D sinusoidal positions, not a global pooled embedding."""
    quarter = channels // 4
    frequency = torch.exp(-math.log(10000.0) * torch.arange(quarter, device=device) / max(quarter - 1, 1))
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    x, y = x.reshape(-1, 1) * frequency, y.reshape(-1, 1) * frequency
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), 1).to(dtype)


class _AttentionAxis(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, value, valid):
        # PyTorch attention returns NaN for all-masked rows. A zero dummy key
        # avoids it; every absent query and its residual are zeroed afterwards.
        safe = valid.clone()
        absent = ~safe.any(1)
        safe[absent, 0] = True
        value = value.masked_fill(~valid[..., None], 0)
        norm = self.norm1(value)
        attended = self.attention(norm, norm, norm, key_padding_mask=~safe, need_weights=False)[0]
        value = (value + attended).masked_fill(~valid[..., None], 0)
        return (value + self.ffn(self.norm2(value))).masked_fill(~valid[..., None], 0)


class _FactorizedBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.temporal = _AttentionAxis(width, heads)
        self.spatial = _AttentionAxis(width, heads)

    def forward(self, value, valid):
        b, t, n, d = value.shape
        value = self.temporal(value.permute(0, 2, 1, 3).reshape(b * n, t, d),
                              valid.permute(0, 2, 1).reshape(b * n, t))
        value = value.reshape(b, n, t, d).permute(0, 2, 1, 3)
        return self.spatial(value.reshape(b * t, n, d), valid.reshape(b * t, n)).reshape(b, t, n, d)


class TemporalSpatialModel(nn.Module):
    """Same input information and dense decoder for both architecture families.

    ``padding=True`` denotes an observed timestep. Forecast calls accept ONLY
    past frames and a known future ``target_year``. The fixed target encoder is
    not part of this model; stop-gradient targets belong exclusively to loss.
    Both next-prediction tasks use the same head; H0 differs only in TRAIN-loss
    supervision, never through GT-controlled forward support.
    """
    def __init__(self, channels=384, family="convgru", size="medium", task="direct",
                 control="history", delta="signed", min_scale=0.05, time_mode="real"):
        super().__init__()
        if (family not in ("convgru", "transformer") or size not in SIZES or task not in TASKS
                or control not in CONTROLS or delta not in ("signed", "absolute")
                or time_mode not in ("real", "index", "none") or channels < 1 or min_scale <= 0):
            raise ValueError("Invalid temporal model configuration")
        if (task == "direct" and control == "last-only") or (task != "direct" and control == "pair-only"):
            raise ValueError("Direct uses pair-only; forecast uses last-only control")
        self.spec = dict(channels=channels, family=family, size=size, task=task, control=control,
                         delta=delta, min_scale=min_scale, time_mode=time_mode)
        self.family, self.task, self.control = family, task, control
        self.delta, self.time_mode, self.min_scale = delta, time_mode, min_scale
        cfg, width = SIZES[size], SIZES[size]["width"]
        self.project = nn.Sequential(nn.Conv2d(2 * channels + 5, width, 1), nn.GELU())
        if family == "convgru":
            self.cells = nn.ModuleList([ConvGRUCell(width) for _ in range(cfg["gru_depth"])])
        else:
            self.blocks = nn.ModuleList([_FactorizedBlock(width, cfg["heads"]) for _ in range(cfg["transformer_depth"])])
        # Identical decoder/head for a given task and scale across families.
        self.decoder = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(width, 1 if task == "direct" else 2 * channels, 1))

    def forward(self, features, support, years, target_year, padding=None) -> Dict[str, torch.Tensor]:
        if padding is None:
            padding = torch.ones(features.shape[:2], device=features.device, dtype=torch.bool)
        target_year = torch.as_tensor(target_year, device=features.device)
        if target_year.ndim == 0 and features.shape[0] == 1:
            target_year = target_year.reshape(1)
        support, lengths = _check_inputs(features, support, years, target_year, padding, self.task)
        if features.shape[2] != self.spec["channels"]:
            raise ValueError("Feature channel count differs from the checkpoint contract")
        output_support = (_last(support, lengths) & _last(support, lengths, 2)
                          if self.task == "direct" else support.any(1))
        features, support, years, padding = _control_inputs(
            features.masked_fill(~support, 0), support, years, padding, lengths, self.control, self.task)
        lengths = padding.sum(1)
        delta, delta_valid = recompute_deltas(features, support, self.delta)
        time = _time_channels(years, target_year, padding, self.time_mode).to(features.dtype)
        b, t, _, h, w = features.shape
        encoded = torch.cat((features, delta, time[..., None, None].expand(b, t, 3, h, w),
                             support.to(features.dtype), delta_valid.to(features.dtype)), 2)
        value = self.project(encoded.reshape(b * t, -1, h, w)).reshape(b, t, -1, h, w)
        value = value.masked_fill(~support, 0)
        if self.family == "convgru":
            for cell in self.cells:
                state = value.new_zeros((b, value.shape[2], h, w))
                sequence = []
                for index in range(t):
                    temporal = time[:, index, 1:, None, None].expand(b, 2, h, w)
                    proposal = cell(value[:, index], state, temporal)
                    state = torch.where(support[:, index], proposal, state)
                    sequence.append(state)
                value = torch.stack(sequence, 1)
            context = value[:, -1]
        else:
            width = value.shape[2]
            value = value.permute(0, 1, 3, 4, 2).reshape(b, t, h * w, width)
            valid = support[:, :, 0].reshape(b, t, h * w)
            value = (value + _position_2d(h, w, width, value.device, value.dtype)).masked_fill(~valid[..., None], 0)
            for block in self.blocks:
                value = block(value, valid)
            # Last supported token at each spatial position, not a global pool.
            positions = torch.arange(t, device=value.device)[None, :, None].expand(b, t, h * w)
            index = positions.masked_fill(~valid, -1).max(1).values.clamp_min(0)
            context = value.gather(1, index[:, None, :, None].expand(b, 1, h * w, width))[:, 0]
            context = context.reshape(b, h, w, width).permute(0, 3, 1, 2)
        available = support.any(1)
        context = context.masked_fill(~available, 0)
        raw = self.decoder(context)
        if self.task == "direct":
            return dict(logits=raw, support=output_support)
        mean, raw_scale = raw.chunk(2, 1)
        return dict(mean=mean, scale=F.softplus(raw_scale.float()) + self.min_scale, support=output_support)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def quantized_logistic_nll(target, mean, scale, step=0.1, min_scale=0.05):
    """Exact logistic bin mass in bits/component, with an unbounded quantizer.

    Q(z)=step*round(z/step); its bin is [Q(z)-step/2,Q(z)+step/2].
    log(sigmoid(upper)-sigmoid(lower)) is evaluated without subtracting CDFs.
    No clipping or density proxy silently caps large surprise. Summing channels
    gives bits/token; neither measure is an achieved coded-stream size.
    """
    if step <= 0 or min_scale <= 0:
        raise ValueError("Quantizer step and minimum scale must be positive")
    if target.shape != mean.shape or scale.shape != mean.shape:
        raise ValueError("Target, mean and scale must have identical shapes")
    if not bool(torch.isfinite(target).all() & torch.isfinite(mean).all() & torch.isfinite(scale).all()):
        raise ValueError("Probability inputs must be finite")
    if bool((scale <= 0).any()):
        raise ValueError("Scale must be positive")
    # float32 minimum for proper NLL even under autocast; doubles stay doubles.
    dtype = torch.float64 if target.dtype == torch.float64 else torch.float32
    target, mean, scale = target.detach().to(dtype), mean.to(dtype), scale.to(dtype).clamp_min(min_scale)
    center = torch.round(target / step) * step
    upper, lower = (center + step / 2 - mean) / scale, (center - step / 2 - mean) / scale
    width = step / scale
    logmass = F.logsigmoid(upper) + F.logsigmoid(-lower) + torch.log(-torch.expm1(-width))
    return -logmass / math.log(2.0)


def history_gain(target, history_mean, history_scale, last_mean, last_scale, support,
                 step=0.1, min_scale=0.05):
    """Same target/quantizer/support: positive means history explains more bits."""
    history = quantized_logistic_nll(target, history_mean, history_scale, step, min_scale)
    last = quantized_logistic_nll(target, last_mean, last_scale, step, min_scale)
    return (last - history).mean(dim=-3, keepdim=True).masked_fill(~support.bool(), float("nan"))


def probability_diagnostics(target, mean, scale, support, step=0.1, min_scale=0.05):
    bits = quantized_logistic_nll(target, mean, scale, step, min_scale)
    valid = support.bool().expand_as(bits)
    count = int(valid.sum())
    if count == 0:
        return dict(count=0, nll_bits_per_component=None, interval_95_coverage=None,
                    scale_floor_fraction=None, numerical_saturation_fraction=None)
    # The central 95% logistic interval is mean +/- log(39)*scale.
    return dict(count=count, nll_bits_per_component=float(bits[valid].mean().detach()),
                interval_95_coverage=float(((target - mean).abs() <= math.log(39) * scale)[valid].float().mean()),
                scale_floor_fraction=float((scale <= min_scale * 1.01)[valid].float().mean()),
                numerical_saturation_fraction=float((bits > -math.log2(torch.finfo(bits.dtype).tiny))[valid].float().mean()))
