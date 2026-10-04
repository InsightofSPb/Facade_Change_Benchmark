"""Native RGB, padding, and output contracts shared by method adapters.

Padding is model-internal context only. The native image and evaluator grid are
never resized, and no labels or nuisance/visibility masks enter these helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np

from ..io import sha256


def rgb_images(reference, source):
    reference, source = np.asarray(reference), np.asarray(source)
    for image in (reference, source):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
            raise ValueError("Scorers require native uint8 RGB without conversion or resizing")
    if source.shape != reference.shape:
        raise ValueError("Reference and source must share the same native RGB grid")
    return reference, source


def validate_rgb_inputs(reference, source, support):
    """Validate native RGB and base support without reading evaluation labels."""
    reference, source = rgb_images(reference, source)
    support = np.asarray(support)
    if support.dtype != np.bool_ or support.shape != reference.shape[:2]:
        raise ValueError("Scorers require a same-grid boolean geometric support mask")
    if not support.any():
        raise ValueError("Scorers require at least one geometrically supported pixel")
    return reference, source, support


def integer_option(value, name, minimum, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be in [{minimum}, {maximum or 'infinity'}]")
    return value


@dataclass(frozen=True)
class Padding:
    """Native-grid location within a symmetrically reflection-padded image."""

    top: int
    bottom: int
    left: int
    right: int
    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]

    def as_dict(self):
        return {"mode": "reflect", "placement": "symmetric; odd surplus at trailing edge",
                "top": self.top, "bottom": self.bottom, "left": self.left, "right": self.right,
                "original_shape": list(self.original_shape), "padded_shape": list(self.padded_shape)}


def pad_rgb_pair(reference, source, multiple=14):
    """Pad RGB to the next patch multiple without changing any native pixels.

    For 256 x 256 and multiple 14 this yields 266 x 266 with five pixels on
    every side. NumPy reflection also defines the degenerate one-pixel axis.
    """
    reference, source = rgb_images(reference, source)
    multiple = integer_option(multiple, "multiple", 1)
    height, width = reference.shape[:2]
    extra_height, extra_width = (-height) % multiple, (-width) % multiple
    top, left = extra_height // 2, extra_width // 2
    bottom, right = extra_height - top, extra_width - left
    padding = Padding(top, bottom, left, right, (height, width),
                      (height + extra_height, width + extra_width))
    widths = ((top, bottom), (left, right), (0, 0))
    return (np.pad(reference, widths, mode="reflect"),
            np.pad(source, widths, mode="reflect"), padding)


def restore_map(values, padding):
    """Crop an already reconstructed padded HxW map back to its native grid."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape != padding.padded_shape:
        raise ValueError("Restored map must be two-dimensional on the full padded grid")
    height, width = padding.original_shape
    return values[padding.top:padding.top + height,
                  padding.left:padding.left + width].copy()


def mask_scores(values, support):
    """Enforce the native continuous-score contract without per-map scaling."""
    values, support = np.asarray(values), np.asarray(support)
    if support.dtype != np.bool_ or values.shape != support.shape or values.ndim != 2:
        raise ValueError("Score map and boolean support must share the native HxW grid")
    scores = values.astype(np.float32, copy=True)
    if (not np.isfinite(scores[support]).all()
            or np.any(scores[support] < 0) or np.any(scores[support] > 1)):
        raise RuntimeError("Change scorer produced an invalid supported score")
    scores[~support] = np.nan
    return scores


def file_provenance(path):
    """Record the exact on-disk source/weight artifact, without loading it."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Method artifact not found: {path}")
    return {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
