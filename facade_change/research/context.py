"""Reference-only logit conditioning for a frozen original MSDZip predictor.

This is a SET-inspired research adaptation, not a reproduction of SET or an
actual entropy codec. Both learned-token branches use the same architecture;
only their spatial/spectral inputs differ. Current RGB and residual targets are
never inputs to the reference feature provider or the correction head.
"""
from __future__ import annotations

import math
from numbers import Integral

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from ..methods.base import pad_rgb_pair, validate_rgb_inputs
from ..methods.dinov2 import _freeze, _load_dino, _rgb_tensor


def load_reference_encoder(source_root, checkpoint_path, device="cpu", trust_checkpoint=False):
    """Reuse the existing local-only, source-checked DINOv2 loader."""
    model, provenance = _load_dino(source_root, checkpoint_path, trust_checkpoint)
    return _freeze(model, device), provenance


def extract_reference(reference, support, model, device="cpu"):
    """Return frozen DINO patch features and explicit native/padded geometry.

    Unsupported reference RGB is replaced by deterministic black before the
    existing reflect padding. Only base geometric support is permitted here.
    The returned tensor is detached but is not an inference-mode tensor: head
    training may safely save it for backward without updating the backbone.
    """
    reference, _, support = validate_rgb_inputs(reference, reference, support)
    if getattr(model, "patch_size", None) != 14:
        raise ValueError("Reference conditioning requires the existing DINOv2 patch size 14")
    known = reference.copy()
    known[~support] = 0
    padded, _, padding = pad_rgb_pair(known, known, multiple=14)
    model = _freeze(model, device)
    with torch.no_grad():
        mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1)
        tokens = model.forward_features((_rgb_tensor(padded, device) - mean) / std)["x_norm_patchtokens"]
    height, width = padding.padded_shape
    grid = (height // 14, width // 14)
    if (not isinstance(tokens, torch.Tensor) or tokens.ndim != 3 or tokens.shape[0] != 1
            or tokens.shape[1] != grid[0] * grid[1] or tokens.shape[2] < 1
            or not tokens.is_floating_point() or not torch.isfinite(tokens).all()):
        raise RuntimeError("Unexpected or nonfinite reference DINO patch tokens")
    features = tokens[0].transpose(0, 1).reshape(tokens.shape[2], *grid).detach().clone()
    return features, {
        "native_shape": list(reference.shape[:2]), "padded_shape": list(padding.padded_shape),
        "patch_grid": list(grid), "padding": padding.as_dict(), "features": "x_norm_patchtokens",
        "normalization": {"rgb_scale": 255., "mean": [.485, .456, .406], "std": [.229, .224, .225]},
        "context": "frozen reference RGB only; unsupported base geometric support set to black",
        "target_features": False, "weights_frozen": True,
    }


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_features(features, channels):
    if (not isinstance(features, torch.Tensor) or features.ndim != 3
            or features.shape[0] != channels or min(features.shape[1:]) < 1
            or not features.is_floating_point() or not torch.isfinite(features).all()
            or features.requires_grad):
        raise ValueError("Reference features must be finite detached D x H x W floats")


def _geometry(features, shape, channels):
    _validate_features(features, channels)
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise ValueError("shape must be the native (height, width)")
    height, width = (_positive_integer(value, "native dimension") for value in shape)
    padded = (((height + 13) // 14) * 14, ((width + 13) // 14) * 14)
    if tuple(features.shape[1:]) != (padded[0] // 14, padded[1] // 14):
        raise ValueError("Reference feature grid disagrees with native DINO padding")
    top, left = (padded[0] - height) // 2, (padded[1] - width) // 2
    return height, width, padded, top, left


class _TokenResidual(nn.Module):
    """Shared lightweight learned-token attention, without cross-case state."""

    def __init__(self, channels, width, tokens):
        super().__init__()
        self.width = width
        self.input = nn.Conv2d(channels, width, 1)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.tokens = nn.Parameter(torch.empty(tokens, width))
        nn.init.normal_(self.tokens, std=.02)
        self.output = nn.Conv2d(width, channels, 1)

    def forward(self, values):
        projected = self.input(values).flatten(2).transpose(1, 2)
        attention = functional.softmax(self.query(projected) @ self.key(self.tokens).T
                                       / math.sqrt(self.width), dim=-1)
        update = (attention @ self.value(self.tokens)).transpose(1, 2)
        update = update.reshape(values.shape[0], self.width, *values.shape[2:])
        return values + self.output(update)


class ReferenceHead(nn.Module):
    """Two capacity-matched token branches and a zero-initialized RGB byte head.

    Spectral mode retains the complete rFFT coefficient grid and both amplitude
    and phase. It never filters frequency bands or Fourier-transforms target RGB.
    Spatial mode uses the same two token modules and averages their updates.
    Learned spectral updates are projected to a real feature map by irFFT,
    including its DC/Nyquist constraints; the token transform is not claimed to
    be invertible. Neither branch changes or compresses the raw RGB signal.
    """

    def __init__(self, mode, in_channels, width=32, tokens=4):
        super().__init__()
        if mode not in {"spatial", "spectral"}:
            raise ValueError("ReferenceHead mode must be spatial or spectral")
        self.mode = mode
        self.in_channels = _positive_integer(in_channels, "in_channels")
        width, tokens = _positive_integer(width, "width"), _positive_integer(tokens, "tokens")
        self.branches = nn.ModuleList([_TokenResidual(self.in_channels, width, tokens) for _ in range(2)])
        self.logit_head = nn.Conv2d(self.in_channels, 3 * 256, 1)
        nn.init.zeros_(self.logit_head.weight)
        nn.init.zeros_(self.logit_head.bias)

    def grid_logits(self, features):
        """Return 768 x patch-H x patch-W logits without native interpolation."""
        _validate_features(features, self.in_channels)
        values = features.to(dtype=self.logit_head.weight.dtype).unsqueeze(0)
        if self.mode == "spatial":
            updated = (self.branches[0](values) + self.branches[1](values)) / 2
        else:
            spectrum = torch.fft.rfft2(values, norm="ortho")
            amplitude = self.branches[0](torch.log1p(spectrum.abs()))
            phase = self.branches[1](torch.angle(spectrum) / math.pi) * math.pi
            # Amplitude is nonnegative; an unbounded positive learned log value
            # is rejected if it overflows rather than silently rescaled.
            spectrum = torch.polar(torch.expm1(amplitude.clamp_min(0)), phase)
            updated = torch.fft.irfft2(spectrum, s=tuple(values.shape[-2:]), norm="ortho")
        result = self.logit_head(updated)
        if not torch.isfinite(result).all():
            raise RuntimeError("Reference conditioning produced nonfinite byte-logit corrections")
        return result[0]

    def forward(self, features, shape):
        """Return native H x W x RGB x 256 corrections; zero means parity."""
        height, width, padded, top, left = _geometry(features, shape, self.in_channels)
        logits = functional.interpolate(self.grid_logits(features).unsqueeze(0), size=padded,
                                         mode="bilinear", align_corners=False)
        logits = logits[0, :, top:top + height, left:left + width]
        return logits.permute(1, 2, 0).reshape(height, width, 3, 256)


def bias_for_positions(head, features, shape, positions, *, grid_logits=None):
    """Gather N x 256 corrections at permanent native RGB byte offsets.

    -1 is an inactive padding lane and receives zero. Sampling the patch logits
    directly avoids materializing an H x W x 768 training tensor. This matches
    the head's bilinear padded-grid interpolation and native crop.
    A precomputed grid is accepted only detached in eval mode; it is a per-case
    inference optimization, never a way to bypass head gradients in training.
    """
    height, width, padded, top, left = _geometry(features, shape, head.in_channels)
    positions = torch.as_tensor(positions, device=features.device)
    if positions.ndim != 1 or positions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("positions must be a one-dimensional integer array of native byte offsets")
    positions = positions.to(torch.int64)
    if ((positions < -1) | (positions >= height * width * 3)).any():
        raise ValueError("Byte offset is outside the native reference grid")
    if grid_logits is not None and (head.training or not isinstance(grid_logits, torch.Tensor)
          or grid_logits.requires_grad or grid_logits.shape != (3 * 256, *features.shape[-2:])
          or grid_logits.device != features.device or grid_logits.dtype != head.logit_head.weight.dtype
          or not torch.isfinite(grid_logits).all()):
        raise ValueError("Cached logits require eval mode and a finite detached same-device 768 x H x W grid")
    result = features.new_zeros((len(positions), 256), dtype=head.logit_head.weight.dtype)
    if not (positions >= 0).any():
        return result
    if grid_logits is None:
        grid_logits = head.grid_logits(features)
    logits = grid_logits.reshape(3, 256, *features.shape[-2:])
    rows, values = [], []
    for channel in range(3):
        selected = torch.nonzero((positions >= 0) & (positions % 3 == channel), as_tuple=False).flatten()
        if not len(selected):
            continue
        pixels = positions[selected] // 3
        y, x = pixels // width + top, pixels % width + left
        grid = torch.stack((2 * (x.to(logits.dtype) + .5) / padded[1] - 1,
                            2 * (y.to(logits.dtype) + .5) / padded[0] - 1), dim=-1).reshape(1, -1, 1, 2)
        sampled = functional.grid_sample(logits[channel:channel + 1], grid, mode="bilinear",
                                         padding_mode="border", align_corners=False)[0, :, :, 0].T
        rows.append(selected)
        values.append(sampled)
    return result.index_copy(0, torch.cat(rows), torch.cat(values))
