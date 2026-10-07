"""Differentiable residual likelihood for the original BCM-Net modules.

The author network exposes inference-only compression APIs. This wrapper uses
its existing feature/context/parameter modules and the published four-phase
logistic-mixture NLL (US20240428927A1, equations 9 and 17). It does not modify the
network, arithmetic coder, or VTM base layer. These are theoretical residual
bits, excluding base-layer bits, stream headers and coder quantization.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor


def _raw_plane(value: Tensor, name: str, low: int, high: int) -> None:
    import torch

    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch tensor")
    if value.ndim != 4 or value.shape[1] != 1 or any(size <= 0 for size in value.shape):
        raise ValueError(f"{name} must have nonempty shape (N, 1, H, W)")
    if value.dtype == torch.bool or value.is_complex():
        raise ValueError(f"{name} must contain integer-valued samples")
    checked = value.detach()
    if not torch.isfinite(checked).all() or (checked < low).any() or (checked > high).any():
        raise ValueError(f"{name} must contain finite samples in [{low}, {high}]")
    if checked.is_floating_point() and (checked != checked.round()).any():
        raise ValueError(f"{name} must contain integer-valued samples")


def _logistic_mixture_bits(params: Tensor, target: Tensor, mixtures: int) -> Tensor:
    """Target-bin NLL in bits, with the author's raw signed unit-width bins."""
    import torch
    import torch.nn.functional as functional

    n, channels, height, width = target.shape
    if params.shape != (n, 3 * channels * mixtures, height, width):
        raise ValueError("BCM parameter shape does not match residual phase")
    params = params.reshape(n, 3, channels, mixtures, height, width)
    log_weights = functional.log_softmax(params[:, 0], dim=2)
    means = params[:, 1]
    log_scales = params[:, 2].clamp_min(-7.0)  # original EntropyModel constant
    inverse_scale = torch.exp(-log_scales)
    lower = (target.unsqueeze(2) - 0.5 - means) * inverse_scale
    upper = lower + inverse_scale

    # log(sigmoid(upper) - sigmoid(lower)), without saturated-CDF subtraction.
    # For extremely large scales, log(1-exp(-exp(-log_scale))) needs a small-x
    # expansion to preserve gradients even when exp(-log_scale) underflows.
    small = inverse_scale < 1e-4
    small_correction = torch.log1p(-inverse_scale * 0.5 + inverse_scale.square() / 6.0)
    log_width = torch.where(
        small,
        -log_scales + small_correction,
        torch.log(-torch.expm1(-inverse_scale.clamp_min(1e-4))),
    )
    log_bin = -functional.softplus(-upper) - functional.softplus(lower) + log_width
    return -torch.logsumexp(log_weights + log_bin, dim=2) / math.log(2.0)


def bcm_bits_map(network: Any, residues: Tensor, base: Tensor,
                 reference: Tensor | None = None) -> Tensor:
    """Return differentiable theoretical bits with shape ``(N, 1, H, W)``.

    Inputs are native 8-bit monochrome planes, processed per RGB channel by the
    caller: ``residues = B - base`` in [-255, 255], ``base = VTM(B)`` and optional
    ``reference = A`` already restored losslessly at the decoder. No automatic
    predictor substitution, RGB conversion, padding, or unsigned wrap occurs.
    H/W must be positive and even. Raw samples may be signed integer tensors or
    integer-valued floating tensors; feature inputs are converted to float32.
    """
    import torch

    if getattr(network, "bit_depth", None) != 8:
        raise ValueError("BCM training likelihood currently supports bit_depth=8 only")
    _raw_plane(residues, "residues", -255, 255)
    _raw_plane(base, "base", 0, 255)
    if residues.shape != base.shape or residues.device != base.device:
        raise ValueError("residues and base must have identical shapes and devices")
    if any(size % 2 for size in residues.shape[-2:]):
        raise ValueError("Original BCM four-phase splitting requires even H and W")
    target = base.to(torch.float32) + residues.to(torch.float32)
    if (target.detach() < 0).any() or (target.detach() > 255).any():
        raise ValueError("base + residues must reconstruct native samples in [0, 255]")
    if reference is not None:
        _raw_plane(reference, "reference", 0, 255)
        if reference.shape != base.shape or reference.device != base.device:
            raise ValueError("reference and base must have identical shapes and devices")

    lossy_feats = network.feats_extract_lossy_rec(base.to(torch.float32) / 255.0)
    forward = None if reference is None else reference.to(torch.float32) / 255.0
    backward = None if forward is None else forward.clone().detach()
    forward_feats = network.feats_extract_forward_ref(forward)
    backward_feats = network.feats_extract_backward_ref(backward)
    context = network.i2cg(lossy_feats=lossy_feats, forward_ref_feats=forward_feats,
                           backward_ref_feats=backward_feats)

    entropy = network.entropy_models
    phases = entropy.spatial_split(residues.to(torch.float32))
    mixtures = entropy.discrete_logistic_mixture_model.K
    phase_bits = []
    for index, phase in enumerate(phases):
        if index == 0:
            params = entropy.params_estimator[index](context)
        else:
            # Complete earlier phases are decoder-known; the current phase and
            # future phases must never enter their own parameter estimation.
            auto_context = entropy.auto_ctx_extraction[index - 1](
                torch.cat(phases[:index], dim=1))
            params = entropy.params_estimator[index](torch.cat([context, auto_context], dim=1))
        phase_bits.append(_logistic_mixture_bits(params, phase, mixtures))
    return entropy.spatial_merge(phase_bits)


def bcm_nll(network: Any, residues: Tensor, base: Tensor,
            reference: Tensor | None = None) -> Tensor:
    """Return ``(N,)`` total theoretical residual bits for each input plane.

    For mean bits per channel sample, divide ``bcm_nll(...).mean()`` by H*W.
    The caller adds the actual VTM base-layer cost when reporting codec rates.
    """
    return bcm_bits_map(network, residues, base, reference).sum(dim=(1, 2, 3))
