"""Original ArIB-BPS network/coder; explicit H0 residual-image adaptation."""
from __future__ import annotations

import importlib
import importlib.util
import hashlib
import json
import random
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from ..io import read_json, sha256
from .base import file_provenance
from .dinov2 import _assert_package_source, _checkpoint
from .lossless import TileCodecScorer
from .arib_compat import install_posterior_rgb_compatibility


PROVENANCE = Path(__file__).resolve().parents[2] / "third_party/arib_bps_provenance.json"


def load_author_model(source_root, config_name="imagenet32_config", dropout=0.0):
    """Verify unmodified author sources, then build its exact configuration."""
    import sys
    root = Path(source_root).expanduser().resolve()
    expected = read_json(PROVENANCE)
    for relative, digest in expected["sha256"].items():
        if not (root / relative).is_file() or sha256(root / relative) != digest:
            raise ValueError(f"ArIB-BPS source differs from pinned original: {relative}")
    if config_name not in {"imagenet32_config", "imagenet64_config", "imagenet64_small_config", "cifar_config"}:
        raise ValueError("Select an existing original ArIB-BPS configuration")
    for name in ("modules", "utils", "config"):
        _assert_package_source(name, root / "src" / name)
    sys.path.insert(0, str(root / "src"))
    try:
        module = importlib.import_module("modules.arib_bps")
    except ImportError as exc:
        raise RuntimeError("Compile the original ArIB mixcoder first: bash scripts/setup_neural_codecs.sh") from exc
    spec = importlib.util.spec_from_file_location("facade_arib_config", root / "src/config" / (config_name + ".py"))
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    arguments = asdict(config_module.CFG)
    return module.ARIB_BPS(**arguments, dropout=dropout), {
        "repository": expected["repository"], "commit": expected["commit"],
        "source_manifest_sha256": sha256(PROVENANCE), "config_name": config_name,
        "model_arguments": arguments, "dropout": dropout,
        "coder_binary": file_provenance(root / "src/utils/coder/mixcoder.so"),
    }


def configure_torch(device, seed, name="ArIB-BPS"):
    import torch
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("ArIB-BPS device must be cpu or cuda")
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA is unavailable")
        free, total = torch.cuda.mem_get_info(target)
        print(f"{name}: free {free/2**30:.2f}/{total/2**30:.2f} GiB; other processes unchanged", flush=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return target


class ArIBCodec:
    def __init__(self, source_root, training_run, representation, device="cpu", seed=42,
                 trust_checkpoint=False, dataset_fingerprint=None, cost_mode="bitstream"):
        import torch
        if cost_mode not in {"theoretical", "bitstream"}:
            raise ValueError("ArIB cost_mode must be theoretical or bitstream")
        run = Path(training_run).expanduser().resolve()
        record = read_json(run / "run.json")
        if (record.get("schema_version") != 1 or record.get("kind") != "arib_h0_training"
                or record.get("status") != "completed_exploratory"):
            raise ValueError("ArIB requires a completed H0 training run")
        serialized = json.dumps(record["config"], sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        if hashlib.sha256(serialized).hexdigest() != record.get("config_sha256"):
            raise ValueError("ArIB training configuration hash changed")
        for relative in ("sampling.json", "summary.json"):
            if record["artifact_sha256"].get(relative) != sha256(run / relative):
                raise ValueError(f"ArIB training artifact hash changed: {relative}")
        if dataset_fingerprint is not None and record["config"]["input_sha256"] != dataset_fingerprint:
            raise ValueError("ArIB H0 checkpoint belongs to a different dataset snapshot")
        if representation not in record["config"]["representations"]:
            raise ValueError("Representation was not fitted in this H0 training run")
        self.device = configure_torch(device, seed)
        if self.device.type == "cpu":
            torch.set_num_threads(1)
        if self.device.type == "cpu" and cost_mode == "bitstream":
            # Avoid one source of shape-dependent numeric differences. This
            # setting does not replace the mandatory exact roundtrip check.
            torch.backends.mkldnn.enabled = False
        self.model, source = load_author_model(source_root, record["config"]["author_config"])
        weights = {}
        for component in ("sig", "ins"):
            relative = f"{representation}/{component}.pth"
            path = run / relative
            if record["artifact_sha256"].get(relative) != sha256(path):
                raise ValueError(f"ArIB checkpoint hash changed: {relative}")
            state, loading = _checkpoint(path, trust_checkpoint)
            getattr(self.model, component).load_state_dict(state, strict=True)
            weights[component] = {**file_provenance(path), "loading": loading}
        self.model.to(self.device).eval().requires_grad_(False)
        self.compatibility = install_posterior_rgb_compatibility(self.model)
        self.seed = seed
        self.metadata = {"name": "ArIB-BPS", "source": source, "checkpoints": weights,
            "cost_mode": cost_mode,
            "training_run": file_provenance(run / "run.json"), "seed": seed,
            "numeric_compatibility": self.compatibility.metadata,
            "cost": ("original inference().sum() theoretical variational bpd; not actual coded bits" if cost_mode == "theoretical" else
                     "actual author compress_to_file stream, including header and bits-back initialization"),
            "roundtrip": ("not performed in theoretical scoring; separate bitstream check required" if cost_mode == "theoretical" else
                          "every tile restored by author decompress_from_file and verified exact RGB; mismatch rejects score"),
            "numerics": "float32; CUDA TF32 disabled; CPU scoring uses one thread; CPU bitstreams disable oneDNN; exact roundtrip still required",
            "adaptation": "original architecture; fresh H0 residual-image training; not published pretrained image results"}
        self.tile_size = record["config"]["tile_size"]

    def theoretical_bpb(self, source):
        import torch
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        tensor = torch.from_numpy(source.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            return float(self.model.inference(tensor).sum().cpu())

    def encode(self, reference, source):
        import torch
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        tensor = torch.from_numpy(source.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with tempfile.TemporaryDirectory(prefix="facade-arib-") as temporary, torch.inference_mode():
            stream, image = Path(temporary) / "tile.arib", Path(temporary) / "restored.png"
            self.model.compress_to_file(tensor, str(stream))
            self.model.decompress_from_file(str(stream), str(image), str(self.device))
            with Image.open(image) as decoded:
                restored = np.asarray(decoded.convert("RGB"))
            if not np.array_equal(restored, source):
                raise RuntimeError("Original ArIB-BPS failed exact RGB roundtrip; no score accepted")
            length = stream.stat().st_size
        return length, {"charged_bytes": length, "stream_bytes": length, "roundtrip_verified": True}

    def close(self):
        self.compatibility.close()
        self.model = None


class ArIBScorer(TileCodecScorer):
    def __init__(self, method, source_root, training_run, device="cpu", tile_size=32, stride=16,
                 seed=42, trust_checkpoint=False, dataset_fingerprint=None, cost_mode="bitstream", **options):
        if options:
            raise ValueError(f"Unknown ArIB options: {', '.join(sorted(options))}")
        representation = method.removeprefix("arib_bps_")
        codec = ArIBCodec(source_root, training_run, representation, device, seed,
                          trust_checkpoint, dataset_fingerprint, cost_mode)
        if tile_size != codec.tile_size:
            raise ValueError("ArIB evaluation tile size must equal the H0 training tile size")
        self._configure_tiles(method, codec, tile_size, stride, representation)
        self.metadata.update(source=codec.metadata["source"], checkpoints=codec.metadata["checkpoints"],
                             device=str(device), settings={"seed": seed, "tile_size": tile_size, "stride": stride,
                                                          "cost_mode": cost_mode})
