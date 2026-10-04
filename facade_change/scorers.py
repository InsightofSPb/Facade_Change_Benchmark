"""Compatibility imports for the common method registry.

The original RGB/SSIM, residual codec, and frozen MSDZip calculations now live
in ``facade_change.methods``. Existing callers and private test helpers keep
working; importing this module does not load learned models or their weights.
"""
from __future__ import annotations

from copy import deepcopy

from .methods.base import integer_option as _integer_option
from .methods.base import rgb_images as _rgb_images
from .methods.base import validate_rgb_inputs
from .methods.compression import _COMPRESSION_METHODS, ResidualCompressionScorer, rgb_residual
from .methods.registry import LEGACY_OPTIONS as _OPTIONS
from .methods.registry import make_method, validate_legacy_options as _validate_options
from .methods.rgb import _METADATA, RGBScorer as _MethodRGBScorer, _inputs, _score_rgb, _ssim_change


_NEURAL_METHODS = {"msdzip_abs": "abs", "msdzip_mod256": "mod256"}


class _RGBScorer(_MethodRGBScorer):
    """Keep the original audit hook at this compatibility-module boundary."""

    def __call__(self, reference_rgb, source_rgb, geometric_support):
        return score_change(reference_rgb, source_rgb, geometric_support, self.method)


def make_scorer(method, **options):
    """Compatibility alias: construct the selected adapter once per worker."""
    if isinstance(method, str) and method in _METADATA:
        _validate_options(options)
        return _RGBScorer(method)
    return make_method(method, **options)


def scorer_metadata(method, **options):
    """Return a fresh serializable description for a run's provenance."""
    if isinstance(method, str) and method in _METADATA:
        _validate_options(options)
        metadata = deepcopy(_METADATA[method])
        metadata["output_kind"] = "score"
        return metadata
    scorer = make_scorer(method, **options)
    try:
        return deepcopy(scorer.metadata)
    finally:
        close = getattr(scorer, "close", None)
        if close is not None:
            close()


def score_change(reference_rgb, source_rgb, geometric_support, method, **options):
    """Legacy one-call entry point; reuse make_scorer for repeated inference."""
    if isinstance(method, str) and method in _METADATA:
        _validate_options(options)
        return _score_rgb(reference_rgb, source_rgb, geometric_support, method)
    scorer = make_scorer(method, **options)
    try:
        return scorer(reference_rgb, source_rgb, geometric_support)
    finally:
        close = getattr(scorer, "close", None)
        if close is not None:
            close()
