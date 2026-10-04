"""Small native-RGB change scorers; no labels or visibility oracle are inputs.

SSIM follows Wang et al. (2004), Eq. 13 and weighted moments Eq. 14–16:
https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf
RGB-channel averaging, geometric-support weight normalization, and a change-map
conversion are explicit adaptations. No image resize or gamma conversion occurs.
"""
from __future__ import annotations

from copy import deepcopy
import importlib
import lzma
from numbers import Integral

import numpy as np


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


_COMPRESSION_METHODS = {
    "zstd_abs": ("zstd", "abs"),
    "zstd_mod256": ("zstd", "mod256"),
    "lzma_abs": ("lzma", "abs"),
    "lzma_mod256": ("lzma", "mod256"),
}
_NEURAL_METHODS = {"msdzip_abs": "abs", "msdzip_mod256": "mod256"}
_OPTIONS = {"compression_tile_size", "compression_stride", "zstd_level", "lzma_preset",
            "checkpoint_path", "device", "dataset_fingerprint", "trust_checkpoint"}


def scorer_metadata(method, **options):
    """Return a fresh serializable description for a run's provenance."""
    if isinstance(method, str) and method in _METADATA:
        _validate_options(options)
        return deepcopy(_METADATA[method])
    return deepcopy(make_scorer(method, **options).metadata)


def _validate_options(options):
    unknown = set(options) - _OPTIONS
    if unknown:
        raise ValueError(f"Unknown scorer options: {', '.join(sorted(unknown))}")


def _rgb_images(reference, source):
    reference, source = np.asarray(reference), np.asarray(source)
    for image in (reference, source):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
            raise ValueError("Scorers require native uint8 RGB without conversion or resizing")
    if source.shape != reference.shape:
        raise ValueError("Reference and source must share the same native RGB grid")
    return reference, source


def rgb_residual(reference, source, representation):
    """Transform source minus reference after signed subtraction; return RGB uint8."""
    reference, source = _rgb_images(reference, source)
    difference = source.astype(np.int16) - reference.astype(np.int16)
    if representation == "abs":
        return np.abs(difference).astype(np.uint8)
    if representation == "mod256":
        return np.remainder(difference, 256).astype(np.uint8)
    raise ValueError(f"Unknown RGB residual representation: {representation}")


def validate_rgb_inputs(reference, source, support):
    """Validate native RGB and base geometric support without reading any labels."""
    reference, source = _rgb_images(reference, source)
    support = np.asarray(support)
    if support.dtype != np.bool_ or support.shape != reference.shape[:2]:
        raise ValueError("Scorers require a same-grid boolean geometric support mask")
    if not support.any():
        raise ValueError("Scorers require at least one geometrically supported pixel")
    return reference, source, support


def _integer_option(value, name, minimum, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be in [{minimum}, {maximum or 'infinity'}]")
    return value


class ResidualCompressionScorer:
    """Actual tile stream lengths, with fixed byte layout, grid, and score scale."""

    def __init__(self, method, compression_tile_size=32, compression_stride=16,
                 zstd_level=3, lzma_preset=3):
        codec, representation = _COMPRESSION_METHODS[method]
        self.tile_size = _integer_option(compression_tile_size, "compression_tile_size", 1)
        self.stride = _integer_option(compression_stride, "compression_stride", 1, self.tile_size)
        zstd_level = _integer_option(zstd_level, "zstd_level", -131072, 22)
        lzma_preset = _integer_option(lzma_preset, "lzma_preset", 0, 9)
        self.representation = representation
        self.raw_scores = None
        if codec == "zstd":
            try:
                import zstandard
            except ImportError as exc:
                raise RuntimeError("zstd scorers need zstandard: python -m pip install zstandard") from exc
            self._compressor = zstandard.ZstdCompressor(level=zstd_level, write_checksum=False,
                                                       write_content_size=True, write_dict_id=False)
            self.compress = self._compressor.compress
            codec_options = {"level": zstd_level, "checksum": False, "content_size": True,
                             "dictionary": None, "zstandard_version": zstandard.__version__}
        else:
            self.compress = lambda data: lzma.compress(data, format=lzma.FORMAT_XZ,
                                                     check=lzma.CHECK_CRC64, preset=lzma_preset)
            codec_options = {"preset": lzma_preset, "format": "XZ", "check": "CRC64"}
        self.metadata = {
            "method": method, "implementation_version": 1, "codec": codec,
            "codec_options": codec_options, "representation": representation,
            "signal": "native uint8 RGB residual; int16 source-reference before abs or modulo 256",
            "byte_layout": "C-order interleaved RGB; no resize, color conversion, or oracle labels",
            "tile_size": self.tile_size, "stride": self.stride,
            "tile_grid": "top-left origins range(0,H,stride), range(0,W,stride); fixed square tiles",
            "support": "base geometric support only; unsupported RGB residual bytes replaced by zero",
            "boundary": "incomplete image-border tiles padded with neutral zero RGB bytes",
            "native_formula": "8 * len(compressed_tile_stream) / (tile_size * tile_size * 3)",
            "native_units": "bits per residual byte; includes codec headers and full neutral-padded tile denominator",
            "aggregation": "arithmetic mean of native tile bits-per-byte over every covering tile",
            "score_formula": "1-exp(-native_bits_per_byte/8); fixed monotone scale; no per-map normalization",
            "output": "float32 [0,1]; higher means less compressible residual; NaN outside geometric support",
            "raw_output": "float32 native bits per byte; NaN outside geometric support",
        }

    def __call__(self, reference_rgb, source_rgb, geometric_support):
        reference, source, support = validate_rgb_inputs(reference_rgb, source_rgb, geometric_support)
        residual = rgb_residual(reference, source, self.representation)
        residual[~support] = 0
        height, width = support.shape
        totals = np.zeros((height, width), dtype=np.float64)
        counts = np.zeros((height, width), dtype=np.uint32)
        tile = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
        denominator = tile.size
        for row in range(0, height, self.stride):
            end_row = min(row + self.tile_size, height)
            for col in range(0, width, self.stride):
                end_col = min(col + self.tile_size, width)
                tile.fill(0)
                tile[:end_row - row, :end_col - col] = residual[row:end_row, col:end_col]
                bpb = 8 * len(self.compress(tile.tobytes(order="C"))) / denominator
                totals[row:end_row, col:end_col] += bpb
                counts[row:end_row, col:end_col] += 1
        values = totals / counts
        self.raw_scores = values.astype(np.float32)
        self.raw_scores[~support] = np.nan
        scores = (-np.expm1(-values / 8)).astype(np.float32)
        scores[~support] = np.nan
        return scores


class _RGBScorer:
    def __init__(self, method):
        self.method = method
        self.metadata = deepcopy(_METADATA[method])
        self.raw_scores = None

    def __call__(self, reference_rgb, source_rgb, geometric_support):
        return score_change(reference_rgb, source_rgb, geometric_support, self.method)


def make_scorer(method, **options):
    """Construct once per run; every call exposes its most recent native raw map."""
    if not isinstance(method, str) or method not in _METADATA.keys() | _COMPRESSION_METHODS.keys() | _NEURAL_METHODS.keys():
        raise ValueError(f"Unknown change scorer: {method}")
    _validate_options(options)
    if method in _METADATA:
        return _RGBScorer(method)
    if method in _NEURAL_METHODS:
        module = importlib.import_module(".2026-10-04_msdzip_h0", __package__)
        scorer = module.MSDZipScorer(representation=_NEURAL_METHODS[method], **{
            key: value for key, value in options.items()
            if key in {"checkpoint_path", "device", "dataset_fingerprint", "trust_checkpoint"}
        })
        if scorer.metadata.get("representation") != _NEURAL_METHODS[method]:
            raise ValueError(f"Checkpoint representation does not match method {method}")
        return scorer
    return ResidualCompressionScorer(method, **{
        key: value for key, value in options.items()
        if key in {"compression_tile_size", "compression_stride", "zstd_level", "lzma_preset"}
    })


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


def score_change(reference_rgb, source_rgb, geometric_support, method, **options):
    """Compatibility entry point; use make_scorer once for weights and raw maps."""
    if isinstance(method, str) and method in _METADATA:
        _validate_options(options)
        return _score_rgb(reference_rgb, source_rgb, geometric_support, method)
    return make_scorer(method, **options)(reference_rgb, source_rgb, geometric_support)
