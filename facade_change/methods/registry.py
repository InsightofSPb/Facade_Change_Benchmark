"""Lazy method registry for the common H0/H1 runner.

Only the requested adapter is imported. Native masks retain their own decision
rule; a registry entry never substitutes a continuous proxy for that mask.
"""
from __future__ import annotations

import importlib


LEGACY_METHODS = (
    "rgb_diff", "ssim", "zstd_abs", "zstd_mod256", "lzma_abs", "lzma_mod256",
    "msdzip_abs", "msdzip_mod256",
)
EXTERNAL_METHODS = (
    "lpips", "dinov2", "rscd_cmu", "rscd_diff_cmu", "rscd_pscd", "anychange", "geoscd",
    "jpegls_abs", "jpegls_mod256", "h264_rgb", "arib_bps_abs", "arib_bps_mod256",
)
ALL_METHODS = LEGACY_METHODS + EXTERNAL_METHODS

LEGACY_OPTIONS = {
    "compression_tile_size", "compression_stride", "zstd_level", "lzma_preset",
    "checkpoint_path", "device", "dataset_fingerprint", "trust_checkpoint",
}

_EXTERNAL_ADAPTERS = {
    "lpips": ("lpips", "LPIPSScorer"),
    "dinov2": ("dinov2", "DINOv2Scorer"),
    "rscd_cmu": ("rscd", "RSCDScorer"),
    "rscd_diff_cmu": ("rscd", "RSCDScorer"),
    "rscd_pscd": ("rscd", "RSCDScorer"),
    "anychange": ("anychange", "AnyChangeScorer"),
    "geoscd": ("geoscd", "GeoSCDScorer"),
    "jpegls_abs": ("lossless", "ClassicalCodecScorer"),
    "jpegls_mod256": ("lossless", "ClassicalCodecScorer"),
    "h264_rgb": ("lossless", "ClassicalCodecScorer"),
    "arib_bps_abs": ("arib_bps", "ArIBScorer"),
    "arib_bps_mod256": ("arib_bps", "ArIBScorer"),
}


def validate_legacy_options(options):
    unknown = set(options) - LEGACY_OPTIONS
    if unknown:
        raise ValueError(f"Unknown scorer options: {', '.join(sorted(unknown))}")


def make_method(method, **options):
    """Build one adapter; labels and evaluation visibility are never options."""
    if not isinstance(method, str) or method not in ALL_METHODS:
        raise ValueError(f"Unknown change scorer: {method}")
    if method in LEGACY_METHODS:
        validate_legacy_options(options)
        if method in {"rgb_diff", "ssim"}:
            from .rgb import RGBScorer
            return RGBScorer(method)
        if method.startswith("msdzip_"):
            from .msdzip import MSDZipScorer
            return MSDZipScorer(method.removeprefix("msdzip_"), **{
                key: value for key, value in options.items()
                if key in {"checkpoint_path", "device", "dataset_fingerprint", "trust_checkpoint"}
            })
        from .compression import ResidualCompressionScorer
        return ResidualCompressionScorer(method, **{
            key: value for key, value in options.items()
            if key in {"compression_tile_size", "compression_stride", "zstd_level", "lzma_preset"}
        })
    name, class_name = _EXTERNAL_ADAPTERS[method]
    module = importlib.import_module(f".{name}", __package__)
    adapter = getattr(module, class_name)
    if method.startswith(("rscd_", "jpegls_", "arib_bps_")) or method == "h264_rgb":
        return adapter(method=method, **options)
    return adapter(**options)
