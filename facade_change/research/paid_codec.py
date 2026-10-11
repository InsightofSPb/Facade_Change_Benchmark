"""Small paid predictive RGB codec for an H0/H1 feasibility experiment.

The shared reference and its geometric support are decoder context. Target
RGB, C and V are never free decoder context: C is a transmitted bitmap, V a
transmitted colour approximation, and the remaining RGB is losslessly coded.
This is not a semantic codec or a complete nuisance model. Both arms choose
among identity and the same RGB-only fitted affine predictor family by total
actual stream size. Zlib settings and residual representation are identical.

Losslessness applies to reference support. Unsupported output retains the
reference RGB; target bytes outside this support are deliberately not coded.
"""
from __future__ import annotations

import hashlib
import struct
import zlib

import numpy as np


MAGIC = b"FCP1"
VERSION = 1
# magic, version, H1 flag, reserved, H/W and four section lengths,
# shared reference hash, shared support hash and section payload hash.
HEADER = struct.Struct(">4sBBHIIIIII32s32s32s")
AFFINE = struct.Struct(">3H3h")
ZLIB_LEVEL = 9
GAIN_SCALE = 256
OFFSET_SCALE = 16
SUPPORT_CONTRACT = "lossless_on_reference_support; unsupported output retains reference RGB"
FIT_CONTRACT = "RGB-only per-channel trimmed affine fit; 3 rounds; retain residuals <=75th percentile; gain Q8, offset Q4"
PARTITIONS = ("header_bits", "nuisance_bits", "C_bits", "V_bits", "residual_bits")


def _inputs(reference, support, target=None, candidate=None):
    if (not isinstance(reference, np.ndarray) or reference.ndim != 3
            or reference.shape[2] != 3 or reference.dtype != np.uint8
            or not reference.shape[0] or not reference.shape[1]):
        raise ValueError("Reference must be a nonempty HWC3 uint8 RGB array")
    if (not isinstance(support, np.ndarray) or support.dtype != bool
            or support.shape != reference.shape[:2] or not support.any()):
        raise ValueError("Support must be a same-grid nonempty boolean mask")
    if target is not None and (not isinstance(target, np.ndarray)
            or target.shape != reference.shape or target.dtype != np.uint8):
        raise ValueError("Target must be same-grid HWC3 uint8 RGB")
    if candidate is not None and (not isinstance(candidate, np.ndarray)
            or candidate.shape != support.shape or candidate.dtype != bool
            or np.any(candidate & ~support)):
        raise ValueError("Candidate must be a same-grid boolean subset of reference support")


def _shared_hashes(reference, support):
    return (hashlib.sha256(reference.tobytes(order="C")).digest(),
            hashlib.sha256(np.packbits(support.reshape(-1), bitorder="big").tobytes()).digest())


def _affine_parameters(reference, target, support):
    """Robust target-fitted parameters are subsequently transmitted and paid."""
    gains, offsets = [], []
    for channel in range(3):
        x = reference[..., channel][support].astype(np.float64)
        y = target[..., channel][support].astype(np.float64)
        gain, offset = 1., float(np.median(y - x))
        for _ in range(3):
            residual = np.abs(y - (gain * x + offset))
            retained = residual <= np.quantile(residual, .75)
            a, b = x[retained], y[retained]
            mean_a, mean_b = float(a.mean()), float(b.mean())
            variance = float(np.mean((a - mean_a) ** 2))
            if variance > 1e-12:
                gain = float(np.clip(np.mean((a - mean_a) * (b - mean_b)) / variance, 0., 4.))
            offset = float(np.clip(mean_b - gain * mean_a, -255., 255.))
        gains.append(int(np.rint(np.clip(gain, 0., 4.) * GAIN_SCALE)))
        offsets.append(int(np.rint(np.clip(offset, -255., 255.) * OFFSET_SCALE)))
    return AFFINE.pack(*gains, *offsets)


def _predict(reference, nuisance):
    if nuisance == b"\x00":
        return reference.copy()
    if len(nuisance) != 1 + AFFINE.size or nuisance[0] != 1:
        raise ValueError("Invalid nuisance predictor section")
    values = AFFINE.unpack(nuisance[1:])
    gains, offsets = np.asarray(values[:3], dtype=np.int64), np.asarray(values[3:], dtype=np.int64)
    if np.any(gains > 4 * GAIN_SCALE) or np.any(np.abs(offsets) > 255 * OFFSET_SCALE):
        raise ValueError("Affine predictor parameters exceed the declared range")
    # Defined integer rounding, shared by encoder and decoder; no target access.
    transformed = reference.astype(np.int64) * gains + offsets * (GAIN_SCALE // OFFSET_SCALE)
    return np.clip((transformed + GAIN_SCALE // 2) // GAIN_SCALE, 0, 255).astype(np.uint8)


def _update(predictor, candidate, values):
    if len(values) != 4 or values[0] not in (0, 1):
        raise ValueError("Invalid paid colour-update section")
    result = predictor.copy()
    color = np.frombuffer(values[1:], dtype=np.uint8)
    if values[0] == 0:
        result[candidate] = color
    else:
        result[candidate] = ((result[candidate].astype(np.uint16) + color) % 256).astype(np.uint8)
    return result


def _color_options(predictor, target, candidate):
    if not candidate.any():
        return (b"\x00\x00\x00\x00",)
    fill = np.rint(np.median(target[candidate], axis=0)).astype(np.uint8)
    differences = target[candidate].astype(np.int16) - predictor[candidate].astype(np.int16)
    shift = (np.rint(np.median(differences, axis=0)).astype(np.int16) % 256).astype(np.uint8)
    return (b"\x00" + fill.tobytes(), b"\x01" + shift.tobytes())


def _compressed(data):
    return zlib.compress(data, level=ZLIB_LEVEL)


def _decompressed(data, expected):
    """Reject truncated/trailing streams and bound decompression to the grid."""
    decoder = zlib.decompressobj()
    try:
        result = decoder.decompress(data, expected + 1)
    except zlib.error as exc:
        raise ValueError("Malformed compressed section") from exc
    if (len(result) != expected or not decoder.eof or decoder.unused_data
            or decoder.unconsumed_tail):
        raise ValueError("Compressed section has invalid size, trailing data or truncation")
    return result


def _stream(reference, support, nuisance, bitmap, color, residual, h1):
    reference_hash, support_hash = _shared_hashes(reference, support)
    payload = nuisance + bitmap + color + residual
    height, width = support.shape
    header = HEADER.pack(MAGIC, VERSION, int(h1), 0, height, width,
                         len(nuisance), len(bitmap), len(color), len(residual),
                         reference_hash, support_hash, hashlib.sha256(payload).digest())
    stream = header + payload
    partitions = dict(zip(PARTITIONS, (8 * len(header), 8 * len(nuisance),
                                      8 * len(bitmap), 8 * len(color), 8 * len(residual))))
    accounting = {"partitions": partitions, "total_bits": 8 * len(stream),
                  "total_bytes": len(stream), "arm": "H1" if h1 else "H0",
                  "predictor_mode": "affine" if nuisance[0] else "identity",
                  "update_mode": ("fill" if color[0] == 0 else "modular_shift") if h1 else None,
                  "supported_pixel_count": int(support.sum()),
                  "candidate_pixel_count": None,
                  "support_contract": SUPPORT_CONTRACT,
                  "theta_fit": FIT_CONTRACT if nuisance[0] else "identity; no fitted parameters",
                  "predictor_selection": "minimum total actual stream bytes; stable identity/fill ties",
                  "residual_representation": "(target - predictor) mod256; row-major supported pixels, interleaved RGB",
                  "compressor": {"name": "stdlib_zlib", "level": ZLIB_LEVEL,
                                 "runtime_version": zlib.ZLIB_RUNTIME_VERSION}}
    return stream, accounting


def encode(reference, target, support, candidate=None):
    """Return the minimum complete paid stream for one fixed hypothesis arm.

    ``candidate=None`` is H0. A boolean mask, including an empty one, is H1;
    H1 always pays the compressed full bitmap and its four-byte update flag/V.
    Both arms search the same identity/affine nuisance predictor family. Colour
    parameters and nuisance parameters use RGB only, never GT or case metadata.
    """
    _inputs(reference, support, target, candidate)
    h1 = candidate is not None
    bitmap = (_compressed(np.packbits(candidate.reshape(-1), bitorder="big").tobytes()) if h1 else b"")
    modes = (b"\x00", b"\x01" + _affine_parameters(reference, target, support))
    best = None
    for nuisance in modes:
        predictor = _predict(reference, nuisance)
        colors = _color_options(predictor, target, candidate) if h1 else (b"",)
        for color in colors:
            updated = _update(predictor, candidate, color) if h1 else predictor
            residual = ((target[support].astype(np.int16) - updated[support].astype(np.int16)) % 256).astype(np.uint8)
            stream, accounting = _stream(reference, support, nuisance, bitmap, color,
                                         _compressed(residual.tobytes(order="C")), h1)
            accounting["candidate_pixel_count"] = int(candidate.sum()) if h1 else 0
            if best is None or len(stream) < len(best[0]):
                best = stream, accounting
    return best


def decode(reference, support, stream):
    """Decode supported target RGB, strictly requiring the shared reference."""
    _inputs(reference, support)
    if not isinstance(stream, bytes) or len(stream) < HEADER.size:
        raise ValueError("Codec stream must be complete bytes")
    (magic, version, h1, reserved, height, width, nuisance_size, bitmap_size,
     color_size, residual_size, reference_hash, support_hash, payload_hash) = HEADER.unpack(stream[:HEADER.size])
    if (magic != MAGIC or version != VERSION or h1 not in (0, 1) or reserved
            or (height, width) != support.shape):
        raise ValueError("Codec header/version/grid is invalid")
    if (nuisance_size not in (1, 1 + AFFINE.size) or residual_size < 1
            or (h1 and (bitmap_size < 1 or color_size != 4))
            or (not h1 and (bitmap_size or color_size))):
        raise ValueError("Codec section framing disagrees with its hypothesis")
    if HEADER.size + nuisance_size + bitmap_size + color_size + residual_size != len(stream):
        raise ValueError("Codec section lengths disagree with complete stream size")
    expected_reference_hash, expected_support_hash = _shared_hashes(reference, support)
    if reference_hash != expected_reference_hash or support_hash != expected_support_hash:
        raise ValueError("Shared reference/support hash disagrees")
    payload = stream[HEADER.size:]
    if hashlib.sha256(payload).digest() != payload_hash:
        raise ValueError("Codec payload hash disagrees")
    position = 0
    nuisance = payload[position:position + nuisance_size]
    position += nuisance_size
    bitmap = payload[position:position + bitmap_size]
    position += bitmap_size
    color = payload[position:position + color_size]
    position += color_size
    compressed_residual = payload[position:]
    predictor = _predict(reference, nuisance)
    if h1:
        pixel_count = height * width
        packed = _decompressed(bitmap, (pixel_count + 7) // 8)
        bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="big")
        if bits[pixel_count:].any():
            raise ValueError("Paid C bitmap contains noncanonical padding bits")
        candidate = bits[:pixel_count].reshape(support.shape).astype(bool)
        if np.any(candidate & ~support):
            raise ValueError("Paid C bitmap exceeds reference support")
        predictor = _update(predictor, candidate, color)
    raw = _decompressed(compressed_residual, int(support.sum()) * 3)
    residual = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
    result = reference.copy()
    result[support] = ((predictor[support].astype(np.uint16) + residual) % 256).astype(np.uint8)
    return result


def compare_hypotheses(reference, target, support, candidate):
    """Encode both hypotheses and return their actual paid length difference."""
    if candidate is None:
        raise ValueError("H0/H1 comparison requires an explicit H1 candidate mask")
    h0_stream, h0_accounting = encode(reference, target, support)
    h1_stream, h1_accounting = encode(reference, target, support, candidate)
    delta = 8 * (len(h0_stream) - len(h1_stream))
    return {"h0_stream": h0_stream, "h1_stream": h1_stream,
            "h0_accounting": h0_accounting, "h1_accounting": h1_accounting,
            "delta_bits": delta, "delta_bits_per_supported_pixel": delta / int(support.sum())}
