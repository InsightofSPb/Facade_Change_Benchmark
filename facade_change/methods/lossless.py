"""Actual JPEG-LS residual streams and RGB H.264 I/P streams, with exact decoding."""
from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np

from .base import file_provenance, integer_option, validate_rgb_inputs
from .compression import rgb_residual


def _binary(name):
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} is required; install FFmpeg before running these methods")
    return path


def _run(arguments, data=None, timeout=120):
    try:
        result = subprocess.run(arguments, input=data, capture_output=True, timeout=timeout, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.decode(errors="replace")[-4000:]) from exc
    return result.stdout


class FFmpegCodec:
    def __init__(self, kind):
        if kind not in {"jpegls", "h264"}:
            raise ValueError("Expected JPEG-LS or lossless RGB H.264")
        self.kind, self.ffmpeg, self.ffprobe = kind, _binary("ffmpeg"), _binary("ffprobe")
        encoders = _run([self.ffmpeg, "-hide_banner", "-encoders"]).decode()
        required = "jpegls" if kind == "jpegls" else "libx264rgb"
        if required not in encoders:
            raise RuntimeError(f"FFmpeg lacks required encoder {required}")
        self.metadata = {"ffmpeg": _run([self.ffmpeg, "-version"]).decode().splitlines()[0],
                         "binaries": {"ffmpeg": file_provenance(self.ffmpeg),
                                      "ffprobe": file_provenance(self.ffprobe)},
                         "encoder": required, "pixel_domain": "RGB24, no YUV conversion",
                         "frame_layout": "contiguous HxWx3 uint8 RGB; rawvideo input at 1 fps",
                         "encoder_options": ({"threads": 1, "pixel_format": "rgb24",
                                              "format": "image2pipe"} if kind == "jpegls" else
                                             {"qp": 0, "preset": "medium", "threads": 1,
                                              "pixel_format": "rgb24", "gop": 250,
                                              "b_frames": 0, "references": 1, "format": "h264",
                                              "x264_params": "scenecut=0:rc-lookahead=0:sync-lookahead=0"}),
                         "roundtrip": "every encoded tile decoded and compared byte-for-byte"}

    def encode(self, reference, source):
        height, width = source.shape[:2]
        frames = source[None] if self.kind == "jpegls" else np.stack((reference, source))
        command = [self.ffmpeg, "-v", "error", "-threads", "1", "-f", "rawvideo",
                   "-pix_fmt", "rgb24", "-video_size", f"{width}x{height}", "-framerate", "1",
                   "-i", "pipe:0", "-frames:v", str(len(frames)), "-an"]
        if self.kind == "jpegls":
            command += ["-c:v", "jpegls", "-threads", "1", "-pix_fmt", "rgb24", "-f", "image2pipe"]
        else:
            command += ["-c:v", "libx264rgb", "-qp", "0", "-preset", "medium", "-pix_fmt", "rgb24",
                        "-threads", "1", "-g", "250", "-bf", "0", "-refs", "1",
                        "-x264-params", "scenecut=0:rc-lookahead=0:sync-lookahead=0", "-f", "h264"]
        stream = _run(command + ["pipe:1"], frames.tobytes())
        decoded = _run([self.ffmpeg, "-v", "error", "-threads", "1", "-i", "pipe:0",
                        "-frames:v", str(len(frames)), "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-threads", "1", "pipe:1"], stream)
        if decoded != frames.tobytes():
            raise RuntimeError(f"{self.kind} failed exact RGB encode/decode verification")
        if self.kind == "jpegls":
            return len(stream), {"charged_bytes": len(stream), "stream_bytes": len(stream),
                                 "roundtrip_verified": True}
        packets = json.loads(_run([self.ffprobe, "-v", "error", "-f", "h264", "-i", "pipe:0",
                                   "-show_packets", "-show_entries", "packet=size,flags",
                                   "-of", "json"], stream))["packets"]
        if (len(packets) != 2 or "K" not in packets[0]["flags"] or "K" in packets[1]["flags"]
                or sum(int(packet["size"]) for packet in packets) != len(stream)):
            raise RuntimeError("Expected exactly one complete I packet followed by one P packet")
        charged = int(packets[1]["size"])
        return charged, {"charged_bytes": charged, "stream_bytes": len(stream),
                         "reference_i_bytes": int(packets[0]["size"]), "p_bytes": charged,
                         "roundtrip_verified": True}


class TileCodecScorer:
    """Common local map; actual file/packet costs are distinct from model NLL."""
    def _configure_tiles(self, method, codec, tile_size=32, stride=16, representation="mod256"):
        self.tile_size = integer_option(tile_size, "tile_size", 16)
        self.stride = integer_option(stride, "stride", 1, self.tile_size)
        if representation not in {"abs", "mod256", "rgb_pair"}:
            raise ValueError("Unknown codec input representation")
        self.codec, self.representation = codec, representation
        self.raw_scores, self.native_prediction = None, None
        self.metadata = {
            "method": method, "output_kind": "score", "implementation_version": 1,
            "codec": codec.metadata, "representation": representation,
            "tile_size": self.tile_size, "stride": self.stride,
            "tile_grid": "top-left origins range(0,H,stride), range(0,W,stride)",
            "boundary": "zero-padded complete tiles; full tile RGB-byte denominator",
            "support": "base geometric support only; unsupported input pixels zeroed",
            "raw_units": "actual encoded bits per RGB byte; headers included",
            "score_formula": "1-exp(-bits_per_byte/8); average covering tiles; no map normalization",
            "cost_scope": ("complete residual image stream" if representation != "rgb_pair" else
                           "P-frame packet including its headers, conditioned on separately decoded I frame; I cost recorded separately"),
        }

    def __call__(self, reference, source, support):
        reference, source, support = validate_rgb_inputs(reference, source, support)
        left, right = reference.copy(), source.copy()
        if self.representation != "rgb_pair":
            right = rgb_residual(left, right, self.representation)
        left[~support], right[~support] = 0, 0
        height, width = support.shape
        totals = np.zeros(support.shape, dtype=np.float64)
        counts = np.zeros(support.shape, dtype=np.uint32)
        tiles, charged_bytes, stream_bytes, i_bytes = 0, 0, 0, 0
        for row in range(0, height, self.stride):
            end_row = min(row + self.tile_size, height)
            for col in range(0, width, self.stride):
                end_col = min(col + self.tile_size, width)
                region = np.s_[row:end_row, col:end_col]
                if not support[region].any():
                    continue
                a = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
                b = np.zeros_like(a)
                a[:end_row-row, :end_col-col] = left[region]
                b[:end_row-row, :end_col-col] = right[region]
                length, stats = self.codec.encode(a, b)
                bpb = 8 * length / b.size
                charged_bytes += length
                stream_bytes += stats["stream_bytes"]
                i_bytes += stats.get("reference_i_bytes", 0)
                totals[region] += bpb
                counts[region] += 1
                tiles += 1
        values = np.divide(totals, counts, out=np.zeros_like(totals), where=counts != 0)
        self.raw_scores = values.astype(np.float32)
        self.raw_scores[~support] = np.nan
        self.metadata["last_codec_stats"] = {"tiles": tiles, "cost_mode": "bitstream", "charged_bytes": charged_bytes,
              "all_stream_bytes": stream_bytes, "reference_i_bytes": i_bytes,
              "all_tiles_roundtrip_verified": True,
              "note": "overlapping independent tile streams; totals are not a whole-image compression rate"}
        result = (-np.expm1(-values / 8)).astype(np.float32)
        result[~support] = np.nan
        return result

    def close(self):
        if hasattr(self.codec, "close"):
            self.codec.close()


class ClassicalCodecScorer(TileCodecScorer):
    def __init__(self, method, tile_size=32, stride=16, **options):
        if method not in {"jpegls_abs", "jpegls_mod256", "h264_rgb"}:
            raise ValueError("Unknown classical image/video codec")
        unknown = set(options) - {"device", "trust_checkpoint"}
        if unknown:
            raise ValueError(f"Unknown classical codec options: {', '.join(sorted(unknown))}")
        representation = "rgb_pair" if method == "h264_rgb" else method.split("_")[-1]
        self._configure_tiles(method, FFmpegCodec("h264" if method == "h264_rgb" else "jpegls"),
                              tile_size, stride, representation)
