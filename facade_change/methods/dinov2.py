"""Frozen local DINOv2 ViT-S/14 features on the native evaluation grid.

The cosine feature baseline is reused from the owner's public change benchmark.
Symmetric reflection padding is explicit preparation for this new dataset; it is
not an architecture change or a claim to reproduce an author's image protocol.
"""
from __future__ import annotations

import hashlib
import importlib
import pickle
import sys
from pathlib import Path

import numpy as np

from .base import (file_provenance, mask_scores, pad_rgb_pair, restore_map,
                   validate_rgb_inputs)


def _source_provenance(root):
    """Fingerprint local Python sources, including relative names, without Git."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Method source directory not found: {root}")
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        digest.update(relative.as_posix().encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        count += 1
    if not count:
        raise ValueError(f"No Python source files in {root}")
    return {"path": str(root), "python_tree_sha256": digest.hexdigest(),
            "python_file_count": count}


def _assert_package_source(name, root):
    """Refuse to reuse a same-named package from an unrelated installation."""
    module = sys.modules.get(name)
    if module is None:
        return
    paths = list(getattr(module, "__path__", []))
    filename = getattr(module, "__file__", None)
    if filename:
        paths.append(filename)
    expected = Path(root).resolve()
    if not paths or any(expected != Path(path).resolve()
                        and expected not in Path(path).resolve().parents for path in paths):
        raise RuntimeError(f"Conflicting imported {name}; use an isolated method worker")


def _checkpoint(path, trust_checkpoint=False, numpy_metadata=False):
    """Try restricted loading first; legacy pickle needs explicit local trust."""
    import torch

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Method checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True), "weights_only"
    except (TypeError, pickle.UnpicklingError) as safe_error:
        if numpy_metadata and hasattr(torch.serialization, "safe_globals"):
            # NumPy 1.26 exposes a compatibility numpy._core package without
            # its multiarray attribute. Import the actual version's submodule.
            namespace = "numpy._core" if int(np.__version__.split(".")[0]) >= 2 else "numpy.core"
            scalar = importlib.import_module(namespace + ".multiarray").scalar
            allowed = [np.dtype, scalar]
            if tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 6):
                # Torch 2.6 supports exact-name aliases. Both pickle spellings
                # refer to this same fixed NumPy scalar constructor.
                allowed += [(scalar, name + ".multiarray.scalar")
                            for name in ("numpy.core", "numpy._core")]
            allowed += [type(np.dtype(name)) for name in
                        ("float32", "float64", "int32", "int64")]
            try:
                with torch.serialization.safe_globals(allowed):
                    record = torch.load(path, map_location="cpu", weights_only=True)
                return record, "weights_only_numpy_metadata"
            except (TypeError, pickle.UnpicklingError) as metadata_error:
                safe_error = metadata_error
        if not trust_checkpoint:
            raise RuntimeError(
                "Restricted checkpoint loading failed. Only for an explicitly trusted "
                "local checkpoint set trust_checkpoint=True."
            ) from safe_error
        try:
            record = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            # Old PyTorch has no weights_only parameter at all.
            record = torch.load(path, map_location="cpu")
        return record, "trusted_pickle"


def _load_dino(source_root, checkpoint_path, trust_checkpoint=False):
    import torch

    source_root = Path(source_root).expanduser().resolve()
    if not (source_root / "hubconf.py").is_file():
        raise FileNotFoundError(f"Local DINOv2 hubconf.py not found in {source_root}")
    _assert_package_source("dinov2", source_root / "dinov2")
    sources = _source_provenance(source_root)
    weights = file_provenance(checkpoint_path)
    state, loading = _checkpoint(checkpoint_path, trust_checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError("DINOv2 ViT-S/14 checkpoint must be a nonempty state dictionary")
    # pretrained=False forbids the upstream URL-backed checkpoint loader.
    model = torch.hub.load(str(source_root), "dinov2_vits14",
                           source="local", pretrained=False)
    model.load_state_dict(state, strict=True)
    if model.patch_size != 14:
        raise ValueError("Expected the original ViT-S/14 backbone with patch size 14")
    _assert_package_source("dinov2", source_root / "dinov2")
    return model, {"sources": sources, "checkpoint": weights,
                   "checkpoint_loading": loading, "checkpoint_keys": len(state)}


def _rgb_tensor(image, device):
    import torch

    return torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).to(
        device=device, dtype=torch.float32) / 255.0


def _freeze(model, device):
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class DINOv2Scorer:
    def __init__(self, **options):
        root = options.get("source_root") or options.get("dino_root")
        checkpoint = options.get("checkpoint_path") or options.get("dino_checkpoint")
        if not root or not checkpoint:
            raise ValueError("DINOv2 requires source_root and checkpoint_path")
        self.device = options.get("device", "cpu")
        model, provenance = _load_dino(root, checkpoint,
                                       bool(options.get("trust_checkpoint", False)))
        self.model = _freeze(model, self.device)
        self.raw_scores = None
        self.native_prediction = None
        self.metadata = {
            "method": "dinov2", "output_kind": "score", "backbone": "dinov2_vits14",
            "features": "x_norm_patchtokens", "device": str(self.device),
            "normalization": {"rgb_scale": 255.0, "mean": [0.485, 0.456, 0.406],
                              "std": [0.229, 0.224, 0.225]},
            "score": "(1 - cosine_similarity) / 2; bilinear interpolation; no per-pair scaling",
            "input_preparation": "symmetric reflect padding to next multiple of 14, then crop",
            "preparation_scope": "new-dataset adapter; native RGB and evaluator grid unchanged",
            "weights_frozen": True, "downloads": False, **provenance,
        }

    def __call__(self, reference, source, support):
        import torch
        import torch.nn.functional as functional

        reference, source, support = validate_rgb_inputs(reference, source, support)
        image0, image1, padding = pad_rgb_pair(reference, source, multiple=14)
        self.metadata["last_padding"] = padding.as_dict()
        height, width = padding.padded_shape
        with torch.inference_mode():
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            features = [self.model.forward_features((_rgb_tensor(image, self.device) - mean) / std)
                        ["x_norm_patchtokens"] for image in (image0, image1)]
            expected_tokens = height // 14 * (width // 14)
            if (features[0].ndim != 3 or features[0].shape[0] != 1
                    or features[0].shape[1] != expected_tokens
                    or features[0].shape != features[1].shape):
                raise RuntimeError("Unexpected DINOv2 patch-token shape on the padded grid")
            scores = (1 - functional.cosine_similarity(features[0], features[1], dim=-1)) / 2
            scores = scores.reshape(1, 1, height // 14, width // 14)
            scores = functional.interpolate(scores, size=(height, width), mode="bilinear",
                                            align_corners=False)[0, 0].clamp(0, 1)
            values = scores.cpu().numpy().astype(np.float32)
        self.raw_scores = None
        self.native_prediction = None
        return mask_scores(restore_map(values, padding), support)

    def close(self):
        self.model = None
        self.raw_scores = None
        self.native_prediction = None
