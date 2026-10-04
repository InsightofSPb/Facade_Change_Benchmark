"""Original frozen RSCD heads with local DINOv2 and author state loading."""
from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path

import numpy as np

from .base import (file_provenance, mask_scores, pad_rgb_pair, restore_map,
                   validate_rgb_inputs)
from .dinov2 import (_assert_package_source, _checkpoint, _freeze, _load_dino,
                     _rgb_tensor, _source_provenance)


def _local_imports(source_root, py_utils_root):
    source_root = Path(source_root).expanduser().resolve()
    py_utils_root = Path(py_utils_root).expanduser().resolve()
    package_roots = {"robust_scene_change_detect": source_root / "src" / "robust_scene_change_detect",
                     "py_utils": py_utils_root / "src" / "py_utils"}
    for name, root in package_roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"Local {name} source package not found: {root}")
        _assert_package_source(name, root)
    for root in (py_utils_root / "src", source_root / "src"):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    models = importlib.import_module("robust_scene_change_detect.models")
    backbone = importlib.import_module("robust_scene_change_detect.models.backbone_dinov2")
    utils = importlib.import_module("py_utils.utils_torch")
    for name, root in package_roots.items():
        _assert_package_source(name, root)
    return models, backbone, utils


class RSCDScorer:
    def __init__(self, **options):
        import torch

        method = str(options.get("method", "rscd-cmu")).replace("_", "-")
        if method not in ("rscd-cmu", "rscd-diff-cmu", "rscd-pscd"):
            raise ValueError("RSCD method must select CMU, Diff-CMU, or PSCD")
        required = ("source_root", "dino_root", "py_utils_root", "checkpoint_path", "dino_checkpoint")
        if any(not options.get(name) for name in required):
            raise ValueError("RSCD requires " + ", ".join(required))
        self.device = options.get("device", "cpu")
        trust = bool(options.get("trust_checkpoint", False))
        weight_info = file_provenance(options["checkpoint_path"])
        record, loading = _checkpoint(options["checkpoint_path"], trust, numpy_metadata=True)
        if not isinstance(record, dict) or not isinstance(record.get("model"), dict):
            raise ValueError("RSCD checkpoint must contain the author model state dictionary")
        args = record.get("args")
        if not isinstance(args, dict) or not isinstance(args.get("model"), dict):
            raise ValueError("RSCD checkpoint must contain args['model'] author settings")
        model_options = copy.deepcopy(args["model"])
        if model_options.get("name") != "dino2 + cross_attention":
            raise ValueError("Expected an original two-cross-attention DINO RSCD checkpoint")
        if model_options.get("dino-model", "dinov2_vits14") != "dinov2_vits14":
            raise ValueError("RSCD checkpoint expects a different DINO backbone")
        if (not model_options.get("freeze-dino", True)
                or model_options.get("unfreeze-dino-last-n-layer", 0) != 0):
            raise ValueError("Expected the original frozen RSCD DINO backbone")
        dino, dino_info = _load_dino(options["dino_root"], options["dino_checkpoint"], trust)
        models, backbone, utils = _local_imports(options["source_root"], options["py_utils_root"])
        original_loader = backbone._get_dino
        try:
            # get_model freezes its backbone; its trained head remains trainable
            # until the author's requires_grad-directed state loader has run.
            backbone._get_dino = lambda *args, **kwargs: dino
            model = torch.nn.DataParallel(models.get_model(**copy.deepcopy(model_options)))
        finally:
            backbone._get_dino = original_loader
        model, unused = utils.load_grad_required_state(
            model, record["model"], verbose=False, return_details=True)
        if unused:
            raise ValueError(f"Unused RSCD checkpoint keys: {list(unused)[:8]}")
        if not record["model"]:
            raise ValueError("RSCD checkpoint contains no trained head keys")
        self.model = _freeze(model.module, self.device)
        if not hasattr(self.model, "upsample"):
            raise ValueError("Expected the original RSCD bilinear output decoder")
        self.raw_scores = None
        self.native_prediction = None
        self.metadata = {
            "method": method.replace("-", "_"), "output_kind": "score", "device": str(self.device),
            "normalization": {"rgb_scale": 255.0, "mean": None, "std": None},
            "score": "author logits softmax change channel 1",
            "native_prediction": "author logits argmax (binary); evaluated separately",
            "input_preparation": "symmetric reflect padding to next multiple of 14, then crop",
            "preparation_scope": "new-dataset adapter; native RGB and evaluator grid unchanged",
            "decoder_preparation": "author upsample.size set to padded grid, then symmetric crop",
            "checkpoint": weight_info, "checkpoint_loading": loading,
            "checkpoint_model_args": model_options,
            "checkpoint_key_audit": {"head_keys": len(record["model"]), "unused_keys": []},
            "sources": {"rscd": _source_provenance(options["source_root"]),
                        "py_utils": _source_provenance(options["py_utils_root"])},
            "backbone": dino_info, "weights_frozen": True, "downloads": False,
        }

    def __call__(self, reference, source, support):
        import torch

        reference, source, support = validate_rgb_inputs(reference, source, support)
        image0, image1, padding = pad_rgb_pair(reference, source, multiple=14)
        self.metadata["last_padding"] = padding.as_dict()
        height, width = padding.padded_shape
        self.model.upsample.size = (height, width)
        with torch.inference_mode():
            logits = self.model(_rgb_tensor(image0, self.device), _rgb_tensor(image1, self.device))
            if tuple(logits.shape) != (1, height, width, 2):
                raise RuntimeError("Unexpected RSCD logits shape on the padded grid")
            if not torch.isfinite(logits).all():
                raise RuntimeError("RSCD produced nonfinite logits")
            values = logits.softmax(dim=-1)[0, ..., 1].cpu().numpy().astype(np.float32)
            prediction = logits.argmax(dim=-1)[0].cpu().numpy().astype(bool)
        self.raw_scores = None
        self.native_prediction = restore_map(prediction, padding).astype(bool)
        self.native_prediction[~support] = False
        return mask_scores(restore_map(values, padding), support)

    def close(self):
        self.model = None
        self.raw_scores = None
        self.native_prediction = None
