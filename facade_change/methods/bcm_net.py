"""Pinned BCM-Net/VTM lossless codec, adapted to two temporal RGB frames.

Each native RGB channel uses the author's monochrome network and container.
No network layer, entropy coder, or upstream source is replaced here.
"""
from __future__ import annotations

import importlib.util
import hashlib
import io
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from ..io import read_json, sha256
from .arib_bps import configure_torch
from .base import file_provenance, integer_option, rgb_images
from .dinov2 import _assert_package_source, _checkpoint
from .lossless import TileCodecScorer

PROVENANCE = Path(__file__).resolve().parents[2] / "third_party/bcm_net_provenance.json"


def load_author_model(source_root):
    root = Path(source_root).expanduser().resolve()
    manifest = read_json(PROVENANCE)
    for relative, digest in manifest["sha256"].items():
        if not (root / relative).is_file() or sha256(root / relative) != digest:
            raise ValueError(f"BCM-Net source differs from pinned original: {relative}")
    _assert_package_source("Modules", root / "Modules")
    sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location("facade_bcm_author_network", root / "Network.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise RuntimeError("BCM requires its isolated environment; run scripts/setup_bcm_codec.sh") from exc
    finally:
        sys.path.remove(str(root))
    return module.Network(**manifest["architecture"]), {
        "repository": manifest["repository"], "commit": manifest["commit"],
        "source_manifest_sha256": sha256(PROVENANCE), "architecture": manifest["architecture"],
    }


def split_vvc_access_units(stream):
    """Partition an actual Annex-B two-frame VVC stream at its second AUD.

    VTM15 emits an AUD at the front of each access unit. The complete first
    prefix (including parameter sets) is reference cost; every remaining byte
    belongs to B. Reject ambiguous framing instead of estimating packet costs.
    """
    import re
    starts = list(re.finditer(b"\x00\x00(?:\x00)?\x01", stream))
    if not starts or starts[0].start() != 0:
        raise RuntimeError("VTM stream is not complete Annex-B")
    auds, pictures = [], []
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(stream)
        nal = stream[start.end():end]
        if len(nal) < 2 or nal[0] & 0xc0 or not (nal[1] & 7):
            raise RuntimeError("Invalid VVC NAL header")
        kind = (nal[1] >> 3) & 31
        if kind == 20:
            auds.append(start.start())
        if kind <= 11:
            pictures.append(start.start())
    if len(auds) != 2 or not (any(x < auds[1] for x in pictures) and any(x > auds[1] for x in pictures)):
        raise RuntimeError("Expected exactly two VVC access units with picture data and AUD framing")
    return stream[:auds[1]], stream[auds[1]:]


def pack_channel(stream, corrections, height, width):
    """Byte-compatible with the original MRNet merge_bitstreams format."""
    if len(corrections) != 2 or any(len(frame) != 4 for frame in corrections):
        raise ValueError("BCM container requires two frames with four residual streams each")
    data = bytearray(struct.pack(">HHHI", 2, width, height, len(stream)))
    data.extend(stream)
    for frame in corrections:
        for correction in frame:
            data.extend(struct.pack(">I", len(correction)))
            data.extend(correction)
    return bytes(data)


def unpack_channel(data):
    f = io.BytesIO(data)
    def read(n):
        result = f.read(n)
        if len(result) != n:
            raise ValueError("Truncated BCM container")
        return result
    count, width, height, size = struct.unpack(">HHHI", read(10))
    if count != 2 or width != 32 or height != 32:
        raise ValueError("Expected BCM two-frame 32x32 container")
    stream = read(size)
    corrections = [[read(struct.unpack(">I", read(4))[0]) for _ in range(4)] for _ in range(2)]
    if f.read(1):
        raise ValueError("Trailing bytes in BCM container")
    return stream, corrections


class BCMCodec:
    def __init__(self, source_root, checkpoint=None, device="cpu", vtm_encoder=None,
                 vtm_decoder=None, vtm_config=None, vtm_scc_config=None, qp=37,
                 cost_mode="bitstream", training_run=None, seed=42,
                 trust_checkpoint=False, dataset_fingerprint=None, timeout=120):
        import torch
        if cost_mode not in {"bitstream", "theoretical"}:
            raise ValueError("BCM cost_mode must be bitstream or theoretical")
        if qp != 37:
            raise ValueError("This BCM adaptation fixes the author's MRNet QP to37")
        if any(value is None for value in (vtm_encoder, vtm_decoder, vtm_config)):
            raise ValueError("BCM requires VTM15 encoder, decoder and RA_gop16 config paths")
        self.timeout = integer_option(timeout, "timeout", 1, 600)
        self.qp = integer_option(qp, "qp", 0, 63)
        self.cost_mode, self.tile_size = cost_mode, 32
        self.vtm_encoder, self.vtm_decoder = [str(Path(p).expanduser().resolve()) for p in (vtm_encoder, vtm_decoder)]
        ra = Path(vtm_config).expanduser().resolve()
        scc = (Path(vtm_scc_config).expanduser().resolve() if vtm_scc_config else ra.parent / "per-class/classSCC.cfg")
        self.configs = [str(ra), str(scc)]
        manifest = read_json(PROVENANCE)
        for path, relative in zip((ra, scc), manifest["vtm"]["encoder_configs"]):
            if not path.is_file() or sha256(path) != manifest["vtm"]["sha256"][relative]:
                raise ValueError(f"BCM VTM configuration differs from pinned original: {relative}")
        vtm = {"encoder": file_provenance(self.vtm_encoder), "decoder": file_provenance(self.vtm_decoder),
               "configs": [file_provenance(p) for p in self.configs], "version": manifest["vtm"]["version"],
               "source_commit": manifest["vtm"]["commit"], "options": {
                   "InputChromaFormat": 400, "InputBitDepth": 8, "OutputBitDepth": 8,
                   "FrameRate": 30, "FramesToBeEncoded": 2, "Level": "6.1", "QP": self.qp,
                   "AccessUnitDelimiter": 1, "TemporalFilterFutureReference": 0}}
        record, run = None, None
        if training_run is not None:
            if checkpoint is not None:
                raise ValueError("Choose BCM training_run or checkpoint, not both")
            from ..bcm_training import validate_bcm_training_run
            run, record, _ = validate_bcm_training_run(training_run, dataset_fingerprint)
            checkpoint = run / "model.pth"
        self.device = configure_torch(device, seed, name="BCM-Net")
        if self.device.type == "cpu":
            torch.set_num_threads(1)
        self.network, source = load_author_model(source_root)
        weights = None
        if checkpoint is not None:
            state, loading = _checkpoint(checkpoint, trust_checkpoint)
            if isinstance(state, dict) and "network" in state:
                state = state["network"]
            self.network.load_state_dict(state, strict=True)
            weights = {**file_provenance(checkpoint), "loading": loading}
        self.network.to(self.device).eval().requires_grad_(False)
        self.metadata = {"name": "BCM-Net", "source": source, "checkpoints": weights,
            "vtm": vtm, "cost_mode": cost_mode, "seed": seed, "tile_size": 32,
            "adaptation": "MRNet 8-bit architecture reused per native RGB channel; temporal two-frame A/B; shared H0-fitted weights",
            "reference": "B correction conditioned on byte-exact decoded A; unavailable backward reference duplicated by author network",
            "causal_base_adaptation": "author RA temporal filter retained with future reads disabled; reference access unit cannot use B samples",
            "cost": "B VTM access unit plus four author residual streams, their16byte lengths and4byte base-stream length, summed overRGB",
            "container": "Three independent original MRNet two-frame monochrome containers, in fixed R/G/B order; each6byte geometry header assigned toA;4byte full-base length charged toB",
            "roundtrip": ("every tile reconstructed from serialized containers and compared byte-for-byte" if cost_mode == "bitstream" else
                          "theoretical residual likelihood only; use bitstream preflight for exact restoration"),
            "theoretical_cost": "actual B VTM bytes +20byte lengths/channel +published discrete-bin NLL; distinct from coder integer CDF rate",
            "numerics": "float32 original source; CUDA TF32 disabled; no torch/framework source patches"}
        if record:
            expected = record["config"]["author"]
            if expected["source"] != source:
                raise ValueError("BCM evaluation source differs from H0 training source")
            for name in ("encoder", "decoder"):
                if expected["vtm"][name]["sha256"] != vtm[name]["sha256"]:
                    raise ValueError("BCM evaluation VTM binary differs from H0 training binary")
            if [p["sha256"] for p in expected["vtm"]["configs"]] != [p["sha256"] for p in vtm["configs"]]:
                raise ValueError("BCM evaluation base configs differ from H0 training")
            if expected["vtm"]["options"] != vtm["options"]:
                raise ValueError("BCM evaluation base options differ from H0 training")
            self.metadata["training_run"] = file_provenance(run / "run.json")

    def _run(self, command):
        try:
            result = subprocess.run(command, capture_output=True, check=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"BCM VTM command exceeded {self.timeout}s") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("BCM VTM failed: " + (exc.stdout + exc.stderr).decode(errors="replace")[-4000:]) from exc
        # VTM15 identifies itself on startup, including successful commands.
        if b"VVCSoftware: VTM Encoder Version 15.0 " not in result.stdout and b"VVCSoftware: VTM Decoder Version 15.0 " not in result.stdout:
            raise RuntimeError("BCM requires actual VTM15.0 binaries")

    def _decode_base(self, stream, directory, channel):
        stream_path, decoded = directory / f"{channel}.vvc", directory / f"{channel}.decoded.raw"
        stream_path.write_bytes(stream)
        self._run([self.vtm_decoder, "-b", str(stream_path), "-o", str(decoded), "-d", "8"])
        data = decoded.read_bytes()
        if len(data) != 2 * 32 * 32:
            raise RuntimeError("BCM VTM decoder must return exactly two32x32monochrome8bit frames")
        return np.frombuffer(data, dtype=np.uint8).reshape(2, 32, 32).copy()

    def _base_channels(self, reference, source, directory):
        rgb_images(reference, source)
        if reference.shape != (32, 32, 3):
            raise ValueError("BCM RGB adaptation requires complete32x32 tiles")
        bases, streams, partitions = [], [], []
        for channel in range(3):
            raw, stream_path, recon = [directory / f"{channel}.{suffix}" for suffix in ("input.raw", "vvc", "encoder.raw")]
            raw.write_bytes(np.stack((reference[:, :, channel], source[:, :, channel])).tobytes())
            command = [self.vtm_encoder]
            for config in self.configs:
                command += ["-c", config]
            command += [f"--InputFile={raw}", f"--BitstreamFile={stream_path}", f"--ReconFile={recon}",
                "--SourceWidth=32", "--SourceHeight=32", "--InputBitDepth=8", "--OutputBitDepth=8",
                "--InputChromaFormat=400", "--FrameRate=30", "--FramesToBeEncoded=2", "--Level=6.1",
                f"--QP={self.qp}", "--AccessUnitDelimiter=1", "--TemporalFilterFutureReference=0"]
            self._run(command)
            stream = stream_path.read_bytes()
            first, second = split_vvc_access_units(stream)
            decoded = self._decode_base(stream, directory, channel)
            # Compare actual decoder reconstruction with encoder's declared base.
            if recon.read_bytes() != decoded.tobytes():
                raise RuntimeError("VTM encoder and decoder base reconstructions disagree")
            bases.append(decoded)
            streams.append(stream)
            partitions.append({"reference_base_bytes": len(first), "b_base_bytes": len(second), "stream_bytes": len(stream)})
        pair = np.stack(bases, axis=-1)
        return pair[0], pair[1], streams, partitions

    def base_reconstruct(self, reference, source):
        with tempfile.TemporaryDirectory(prefix="facade-bcm-base-") as tmp:
            a, b, _, parts = self._base_channels(reference, source, Path(tmp))
        return a, b, {"channels": parts, "b_base_bytes": sum(p["b_base_bytes"] for p in parts),
                      "reference_base_bytes": sum(p["reference_base_bytes"] for p in parts),
                      "base_stream_bytes": sum(p["stream_bytes"] for p in parts), "decoder_base_verified": True}

    def _tensor(self, data):
        import torch
        return torch.from_numpy(data.copy()).unsqueeze(0).unsqueeze(0).to(self.device).float()

    def encode(self, reference, source):
        import torch
        with tempfile.TemporaryDirectory(prefix="facade-bcm-") as tmp, torch.inference_mode():
            directory = Path(tmp)
            base_a, base_b, streams, parts = self._base_channels(reference, source, directory)
            containers, charged, reference_cost, reference_hashes = [], 0, 0, []
            restored = np.empty((2, 32, 32, 3), dtype=np.uint8)
            for channel in range(3):
                a, b = self._tensor(reference[:, :, channel]), self._tensor(source[:, :, channel])
                ba, bb = self._tensor(base_a[:, :, channel]), self._tensor(base_b[:, :, channel])
                ca = self.network.compress(a - ba, ba, -255, 255)
                cb = self.network.compress(b - bb, bb, -255, 255, ref_forward=a)
                data = pack_channel(streams[channel], [ca, cb], 32, 32)
                stored_base, corrections = unpack_channel(data)
                decoded_base = self._decode_base(stored_base, directory, channel)
                da, db = [self._tensor(frame) for frame in decoded_base]
                decoded_a = self.network.decompress(corrections[0], da, -255, 255) + da
                decoded_b = self.network.decompress(corrections[1], db, -255, 255, ref_forward=decoded_a) + db
                for frame, tensor in enumerate((decoded_a, decoded_b)):
                    values = tensor.squeeze().cpu().numpy()
                    expected = (reference, source)[frame][:, :, channel]
                    if not np.array_equal(values, expected):
                        raise RuntimeError("Original BCM-Net failed exact RGB roundtrip; no score accepted")
                    restored[frame, :, :, channel] = values.astype(np.uint8)
                # Full base length contains B's size; charge its4byte field toB.
                b_cost = parts[channel]["b_base_bytes"] + 20 + sum(map(len, corrections[1]))
                charged += b_cost
                reference_cost += len(data) - b_cost
                first_base, _ = split_vvc_access_units(stored_base)
                reference_data = data[:6] + first_base + b"".join(
                    struct.pack(">I", len(c)) + c for c in corrections[0])
                reference_hashes.append(hashlib.sha256(reference_data).hexdigest())
                containers.append(data)
            if not np.array_equal(restored, np.stack((reference, source))):
                raise RuntimeError("BCM failed exact RGB roundtrip")
            size = sum(map(len, containers))
        return charged, {"charged_bytes": charged, "stream_bytes": size,
            "reference_i_bytes": reference_cost, "reference_bytes": reference_cost,
            "reference_channel_sha256": reference_hashes,
            "b_base_bytes": sum(p["b_base_bytes"] for p in parts), "roundtrip_verified": True}

    def theoretical_bpb(self, reference, source):
        import torch
        from .bcm_training_core import bcm_nll
        _, base, stats = self.base_reconstruct(reference, source)
        bits = 8 * (stats["b_base_bytes"] + 3 * 20)
        with torch.inference_mode():
            for channel in range(3):
                original, decoded_base, ref = [self._tensor(x[:, :, channel]) for x in (source, base, reference)]
                bits += float(bcm_nll(self.network, original - decoded_base, decoded_base, ref).sum().cpu())
        return bits / source.size

    def close(self):
        self.network = None


class BCMScorer(TileCodecScorer):
    def __init__(self, method="bcm_net_rgb", source_root=None, training_run=None, device="cpu",
                 tile_size=32, stride=16, **options):
        if method != "bcm_net_rgb" or tile_size != 32:
            raise ValueError("BCM uses bcm_net_rgb with fixed32x32tiles")
        if training_run is None:
            raise ValueError("BCM benchmark scoring requires an H0 training_run")
        codec = BCMCodec(source_root, training_run=training_run, device=device, **options)
        self._configure_tiles(method, codec, tile_size, stride, "rgb_pair")
        self.metadata.update(source=codec.metadata["source"], checkpoints=codec.metadata["checkpoints"], device=str(device))
        self.metadata["cost_scope"] = codec.metadata["cost"]
        if self.theoretical:
            self.metadata["raw_units"] = "actual VTM B base plus theoretical residual bits per RGB byte; not an actual complete bitstream rate"
            self.metadata["cost_scope"] = codec.metadata["theoretical_cost"]
