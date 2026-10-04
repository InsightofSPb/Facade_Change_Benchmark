"""Existing actual zstd/LZMA residual-byte compression maps.

Tile grid, codec options, stream overhead, and score scale are unchanged.
"""
from __future__ import annotations

import lzma

import numpy as np

from .base import integer_option, rgb_images, validate_rgb_inputs


_COMPRESSION_METHODS = {
    "zstd_abs": ("zstd", "abs"),
    "zstd_mod256": ("zstd", "mod256"),
    "lzma_abs": ("lzma", "abs"),
    "lzma_mod256": ("lzma", "mod256"),
}


def rgb_residual(reference, source, representation):
    """Transform source minus reference after signed subtraction; return RGB uint8."""
    reference, source = rgb_images(reference, source)
    difference = source.astype(np.int16) - reference.astype(np.int16)
    if representation == "abs":
        return np.abs(difference).astype(np.uint8)
    if representation == "mod256":
        return np.remainder(difference, 256).astype(np.uint8)
    raise ValueError(f"Unknown RGB residual representation: {representation}")



class ResidualCompressionScorer:
    """Actual tile stream lengths, with fixed byte layout, grid, and score scale."""

    def __init__(self, method, compression_tile_size=32, compression_stride=16,
                 zstd_level=3, lzma_preset=3):
        codec, representation = _COMPRESSION_METHODS[method]
        self.tile_size = integer_option(compression_tile_size, "compression_tile_size", 1)
        self.stride = integer_option(compression_stride, "compression_stride", 1, self.tile_size)
        zstd_level = integer_option(zstd_level, "zstd_level", -131072, 22)
        lzma_preset = integer_option(lzma_preset, "lzma_preset", 0, 9)
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
            "method": method, "implementation_version": 1, "codec": codec, "output_kind": "score",
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

