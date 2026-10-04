"""Existing native RGB-difference and weighted Gaussian SSIM adapters.

The legacy calculations and constants are preserved exactly.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from .base import validate_rgb_inputs


_METADATA = {
    "rgb_diff": {
        "method": "rgb_diff", "implementation_version": 1,
        "signal": "native uint8 encoded-sRGB; no resize, gamma conversion, or learned weights",
        "formula": "mean_RGB(abs(reference_uint8 - source_uint8)) / 255",
        "output": "native float32 [0,1]; higher means more change; NaN outside geometric support",
        "support": "supplied base geometric support only; no true edit, nuisance, or occlusion labels",
    },
    "ssim": {
        "method": "ssim", "implementation_version": 1,
        "signal": "native uint8 encoded-sRGB / 255; no resize or gamma conversion",
        "formula": "clip((1 - mean_RGB(SSIM_channel)) / 2, 0, 1)",
        "ssim_formula": "((2*mu_x*mu_y+C1)*(2*cov_xy+C2))/((mu_x^2+mu_y^2+C1)*(var_x+var_y+C2))",
        "gaussian": {"window_size": [11, 11], "sigma_pixels": 1.5, "normalized_weight_sum": 1},
        "constants": {"data_range": 1, "K1": .01, "K2": .03, "C1": .01 ** 2, "C2": .03 ** 2},
        "moments": "population weighted moments; no sample covariance correction",
        "support": "Gaussian weights renormalized over supplied base geometric support only; no label inputs",
        "boundary": "outside image is unsupported; constant-zero filtering with support-weight renormalization",
        "numerics": "float64 moments; negative variance roundoff clamped to zero, covariance to Cauchy bounds; locally identical windows score exactly zero",
        "output": "native float32 [0,1]; higher means more change; NaN outside geometric support",
        "primary_source": "https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf",
        "adaptations": ["average three RGB channel SSIM maps", "normalize Gaussian moments over geometric support",
                        "retain native image borders with truncated normalized windows", "convert similarity to (1-SSIM)/2"],
    },
}


class RGBScorer:
    def __init__(self, method):
        self.method = method
        self.metadata = deepcopy(_METADATA[method])
        self.metadata["output_kind"] = "score"
        self.raw_scores = None

    def __call__(self, reference_rgb, source_rgb, geometric_support):
        return _score_rgb(reference_rgb, source_rgb, geometric_support, self.method)



def _inputs(reference, source, support):
    return validate_rgb_inputs(reference, source, support)


def _ssim_change(reference, source, support):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("SSIM needs the existing align extra: python -m pip install -e '.[align]'") from exc
    # A finite 11-pixel Gaussian has exactly five pixels of influence per side.
    positions = np.arange(-5, 6, dtype=np.float64)
    kernel = np.exp(-.5 * (positions / 1.5) ** 2)
    kernel /= kernel.sum()
    def gaussian(values):
        return cv2.sepFilter2D(values, cv2.CV_64F, kernel, kernel, borderType=cv2.BORDER_CONSTANT)

    weights = support.astype(np.float64)
    mass = gaussian(weights)
    safe_mass = np.where(mass > 0, mass, 1)[..., None]
    x, y = reference.astype(np.float64) / 255, source.astype(np.float64) / 255
    channel_weights = weights[..., None]
    mu_x, mu_y = gaussian(x * channel_weights) / safe_mass, gaussian(y * channel_weights) / safe_mass
    var_x = np.maximum(0, gaussian(x * x * channel_weights) / safe_mass - mu_x * mu_x)
    var_y = np.maximum(0, gaussian(y * y * channel_weights) / safe_mass - mu_y * mu_y)
    covariance = gaussian(x * y * channel_weights) / safe_mass - mu_x * mu_y
    bound = np.sqrt(var_x * var_y)
    covariance = np.clip(covariance, -bound, bound)
    c1, c2 = .01 ** 2, .03 ** 2
    similarity = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)
                  / ((mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)))
    change = np.clip((1 - similarity.mean(axis=2)) / 2, 0, 1)
    # Eliminate floating-point residue in windows whose supported RGB is equal.
    different = np.any(reference != source, axis=2) & support
    change[gaussian(different.astype(np.float64)) == 0] = 0
    return change


def _score_rgb(reference_rgb, source_rgb, geometric_support, method):
    """Compute a native float32 score map without reading any evaluation labels."""
    reference, source, support = _inputs(reference_rgb, source_rgb, geometric_support)
    if method == "rgb_diff":
        difference = np.abs(reference.astype(np.int16) - source.astype(np.int16))
        values = difference.mean(axis=2, dtype=np.float64) / 255
    elif np.array_equal(reference[support], source[support]):
        values = np.zeros(support.shape, dtype=np.float64)
    else:
        values = _ssim_change(reference, source, support)
    scores = values.astype(np.float32)
    if not np.isfinite(scores[support]).all() or np.any(scores[support] < 0) or np.any(scores[support] > 1):
        raise RuntimeError("Change scorer produced an invalid supported score")
    scores[~support] = np.nan
    return scores

