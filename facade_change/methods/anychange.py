"""Original automatic AnyChange inference, with explicit device and cache lifetime.

The supplied author source is imported as an isolated namespace. Only constructor
placement and checkpoint loading differ from the upstream wrapper; SAM, proposals,
matching, thresholds and mask-union inference are the existing public benchmark.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np

from .base import file_provenance, integer_option, mask_scores, validate_rgb_inputs


MASK_SETTINGS = {
    "points_per_side": 32, "points_per_batch": 32, "pred_iou_thresh": 0.5,
    "stability_score_thresh": 0.95, "stability_score_offset": 1.0,
    "box_nms_thresh": 0.7, "min_mask_region_area": 0,
}
CHANGE_SETTINGS = {
    "change_confidence_threshold": 145, "auto_threshold": False,
    "use_normalized_feature": True, "area_thresh": 0.8, "match_hist": False,
    "object_sim_thresh": 60, "bitemporal_match": True,
}


def _namespace(name, path):
    path = Path(path).resolve()
    existing = sys.modules.get(name)
    if existing is not None:
        locations = [Path(p).resolve() for p in getattr(existing, "__path__", [])]
        if path not in locations:
            raise RuntimeError(f"Conflicting imported source package: {name}")
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module


def _author_modules(source_root):
    package = source_root / "torchange"
    subtree = package / "models" / "segment_any_change"
    if not (subtree / "anychange.py").is_file():
        raise FileNotFoundError(f"AnyChange author source not found: {subtree}")
    for name, path in (("torchange", package), ("torchange.models", package / "models"),
                       ("torchange.models.segment_any_change", subtree)):
        _namespace(name, path)
    module = importlib.import_module("torchange.models.segment_any_change.anychange")
    sam = importlib.import_module("torchange.models.segment_any_change.segment_anything")
    generator = importlib.import_module("torchange.models.segment_any_change.simple_maskgen")
    decoder = importlib.import_module("torchange.models.segment_any_change.segment_anything.utils.amg")
    return module, sam, generator, decoder


def _safe_state(path, trust_checkpoint=False):
    import torch

    try:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError as exc:
        if not trust_checkpoint:
            raise RuntimeError("This PyTorch lacks weights_only loading; use explicit trust_checkpoint for a trusted checkpoint") from exc
        state = torch.load(str(path), map_location="cpu")
    if not isinstance(state, dict) or not state or not all(
            isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise RuntimeError("SAM checkpoint must contain only a nonempty tensor state_dict")
    return state


def _construct_model(source_root, checkpoint, device, trust_checkpoint, points_per_batch):
    import torch

    module, sam_module, generator_module, decoder_module = _author_modules(source_root)
    sam = sam_module.sam_model_registry["vit_h"](checkpoint=None)
    sam.load_state_dict(_safe_state(checkpoint, trust_checkpoint), strict=True)
    sam = sam.to(device).eval()
    for parameter in sam.parameters():
        parameter.requires_grad_(False)

    # Upstream auto-selects CUDA and captures neck tensors in a closure. Build
    # its same short initialization on the requested device, so those captured
    # tensors and mask-generator tensors cannot disagree on cpu/cuda:N.
    model = module.AnyChange.__new__(module.AnyChange)
    model.device = torch.device(device)
    model.sam = sam
    mask_settings = dict(MASK_SETTINGS, points_per_batch=points_per_batch)
    model.maskgen = generator_module.SimpleMaskGenerator(sam, **mask_settings)
    model.set_hyperparameters(**CHANGE_SETTINGS)
    model.embed_data1 = model.embed_data2 = None
    layernorm = sam.image_encoder.neck[3]
    weight = layernorm.weight.data.reshape(-1, 1, 1)
    bias = layernorm.bias.data.reshape(-1, 1, 1)
    model.inv_transform = lambda embedding: (embedding - bias) / weight
    return model, decoder_module.rle_to_mask


def _source_provenance(source_root):
    subtree = source_root / "torchange" / "models" / "segment_any_change"
    paths = sorted(subtree.rglob("*.py"))
    if not paths:
        raise FileNotFoundError(f"AnyChange author source not found: {subtree}")
    records = []
    for path in paths:
        entry = file_provenance(path)
        entry["relative_path"] = path.relative_to(source_root).as_posix()
        records.append(entry)
    for filename in ("LICENSE", "NOTICE"):
        if (source_root / filename).is_file():
            entry = file_provenance(source_root / filename)
            entry["relative_path"] = filename
            records.append(entry)
    manifest = [{"path": entry["relative_path"], "sha256": entry["sha256"]} for entry in records]
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"source_root": str(source_root), "manifest_sha256": digest, "files": records}


def _union_masks(mask_data, decoder, shape):
    prediction = np.zeros(shape, dtype=bool)
    for rle in mask_data["rles"]:
        mask = np.asarray(decoder(rle))
        if mask.shape != shape or mask.dtype != np.bool_:
            raise RuntimeError("AnyChange mask must be boolean on the original RGB grid")
        prediction |= mask
    return prediction


class AnyChangeScorer:
    """Frozen SAM ViT-H + author's automatic AnyChange native binary mask."""

    def __init__(self, **options):
        allowed = {"source_root", "checkpoint_path", "device", "trust_checkpoint", "points_per_batch"}
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"Unknown AnyChange options: {', '.join(sorted(unknown))}")
        for key in ("source_root", "checkpoint_path"):
            if not options.get(key):
                raise ValueError(f"AnyChange requires {key}")
        source_root = Path(options["source_root"]).expanduser().resolve()
        checkpoint = Path(options["checkpoint_path"]).expanduser().resolve()
        device = str(options.get("device", "cpu"))
        points_per_batch = integer_option(options.get("points_per_batch", 32), "points_per_batch", 1)
        source_record = _source_provenance(source_root)
        checkpoint_record = file_provenance(checkpoint)
        self.model, self.decode_rle = _construct_model(
            source_root, checkpoint, device, bool(options.get("trust_checkpoint", False)), points_per_batch)
        self.raw_scores = None
        self.native_prediction = None
        self.metadata = {
            "method": "anychange", "implementation_version": 1, "output_kind": "native_mask",
            "architecture": "SAM ViT-H; original AnyChange automatic bitemporal matching",
            "source_root": str(source_root), "checkpoint_path": str(checkpoint), "device": device,
            "source": source_record, "checkpoint": checkpoint_record,
            "author_settings": {**MASK_SETTINGS, "points_per_batch": points_per_batch, **CHANGE_SETTINGS},
            "input": "native uint8 RGB; author's SAM resize/preprocessing; no extra resizing",
            "output": "union of retained author masks on native grid; binary 0/1; NaN outside geometric support",
            "threshold_policy": "author native mask only; no validation calibration",
            "cache_policy": "clear_cached_embedding before and after every pair, including failures",
            "prompts": "none; no labels, nuisance masks, or edit visibility provided",
            "adaptations": ["explicit requested device in author initialization",
                            "weights_only tensor checkpoint loading; strict state_dict",
                            "union of retained masks, as existing public benchmark"],
            "primary_source": "https://github.com/Z-Zheng/pytorch-change-models",
        }

    def __call__(self, reference, source, support):
        import torch

        reference, source, support = validate_rgb_inputs(reference, source, support)
        self.native_prediction = None
        self.raw_scores = None
        self.model.clear_cached_embedding()
        try:
            with torch.inference_mode():
                masks, _, _ = self.model.forward(reference, source)
                prediction = _union_masks(masks, self.decode_rle, reference.shape[:2])
                scores = mask_scores(prediction.astype(np.float32), support)
                self.native_prediction = prediction
                return scores
        finally:
            self.model.clear_cached_embedding()
