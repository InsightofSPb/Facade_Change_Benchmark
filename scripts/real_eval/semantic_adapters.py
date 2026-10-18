"""Lazy frozen segmentation adapters for the 2026-10-18 semantic routes.

No pretrained downloads, training, dataset access, or GPU work at module import.
Inputs are native-grid RGB uint8. Output scores are C,H,W, finite [0,1].
Stock CLIP/LPOSS probabilities are vocabulary-dependent *uncalibrated* softmax
scores; raw cosine/propagated scores are retained for explicit layer projection.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from functools import lru_cache
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

VERSION = "2026-10-18-semantic-adapters-v1"
LEGACY_CLASSES = (
    "BACKGROUND", "CRACK", "SPALLING", "DELAMINATION", "MISSING_ELEMENT",
    "WATER_STAIN", "EFFLORESCENCE", "CORROSION", "ORNAMENT_INTACT",
    "REPAIRS", "TEXT_OR_IMAGES",
)
KINDS = ("dino_semantic", "semantic_dino", "sam3", "stock_maskclip", "stock_lposs", "finetuned_maskclip")


@lru_cache(maxsize=128)
def _hash_cached(path, size, mtime_ns):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_info(path):
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Expected a file: {path}")
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size,
            "sha256": _hash_cached(str(path), stat.st_size, stat.st_mtime_ns)}


def _source_info(root, names):
    root = Path(root).expanduser().resolve(strict=True)
    rows = []
    for name in names:
        path = root / name
        candidates = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for file in candidates:
            if file.is_file() and "__pycache__" not in file.parts:
                info = _file_info(file)
                rows.append({"path": str(file.relative_to(root)), "sha256": info["sha256"]})
    if not rows:
        raise FileNotFoundError(f"No adapter source files found in {root}")
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return {"root": str(root), "files": rows, "sha256": digest}


def _one_local(value, label, candidates):
    if value and value != "auto":
        return str(Path(value).expanduser().resolve(strict=True))
    existing = sorted({str(p.expanduser().resolve()) for p in candidates if p.expanduser().exists()})
    if len(existing) != 1:
        detail = ", ".join(existing) if existing else "none"
        raise FileNotFoundError(f"Set {label} to an existing local path; auto candidates: {detail}. Downloads are disabled.")
    return existing[0]


def _cache_roots(config):
    home = Path(config.get("cache_home", Path.home())).expanduser()
    torch_home = Path(os.environ.get("TORCH_HOME", home / ".cache/torch"))
    hf_hub = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", home / ".cache/huggingface")) / "hub"))
    return home, torch_home, hf_hub


def preflight(config):
    """CPU-only resolve/hash local dependencies. Does not import torch or models.

    Returns {ok, errors, config, files, sources, implementation_version}. The
    resolved config is suitable for make_segmenter. Missing assets are explicit;
    auto searches only exact known cache locations, never downloads anything.
    """
    cfg = dict(config)
    result = {"ok": False, "errors": [], "config": cfg, "files": {}, "sources": {},
              "implementation_version": VERSION, "downloads": False}
    kind = cfg.get("kind", cfg.get("adapter"))
    cfg["kind"] = kind
    try:
        if kind not in KINDS:
            raise ValueError(f"Unknown segmentation adapter {kind!r}; available: {KINDS}")
        result["files"]["adapter"] = _file_info(__file__)
        if kind in {"dino_semantic", "semantic_dino"}:
            root = Path(cfg["training_run"]).expanduser().resolve(strict=True)
            cfg["training_run"] = str(root)
            for name in ("best.pth", "config.json", "run.json", "summary.json"):
                result["files"][name] = _file_info(root / name)
            training = json.loads((root / "config.json").read_text())
            result["files"]["backbone"] = _file_info(training["checkpoint"])
            result["sources"]["backbone"] = _source_info(training["source_root"], ["dinov2", "hubconf.py"])
            if cfg.get("current_manifest"):
                result["files"]["current_manifest"] = _file_info(cfg["current_manifest"])
            package = Path(__file__).resolve().parents[2] / "facade_change"
            result["sources"]["semantic_dino"] = _source_info(package, ["segmentation", "methods/2026-10-14_semantic_dino.py"])
        elif kind == "sam3":
            cfg["checkpoint"] = _file_info(cfg["checkpoint"])["path"]
            result["files"]["checkpoint"] = _file_info(cfg["checkpoint"])
            if cfg.get("bpe_path"):
                result["files"]["bpe"] = _file_info(cfg["bpe_path"])
            if cfg.get("source_root"):
                result["sources"]["sam3"] = _source_info(cfg["source_root"], ["sam3"])
            else:
                # Top-level find_spec does not import SAM3/CUDA.
                spec = importlib.util.find_spec("sam3")
                if spec and spec.submodule_search_locations:
                    package = Path(next(iter(spec.submodule_search_locations)))
                    result["sources"]["sam3"] = _source_info(package, ["model", "model_builder.py", "sam"])
                    bpe = package / "assets/bpe_simple_vocab_16e6.txt.gz"
                    if not cfg.get("bpe_path") and bpe.is_file():
                        cfg["bpe_path"] = str(bpe)
                        result["files"]["bpe"] = _file_info(bpe)
                else:
                    result["package_check"] = "SAM3 package must be checked in its worker environment"
        else:
            root = Path(cfg["source_root"]).expanduser().resolve(strict=True)
            cfg["source_root"] = str(root)
            result["sources"]["lposs"] = _source_info(root, ["ovs_heritage", "models", "segmentation", "tools/finetune.py", "configs/maskclip.yaml"])
            if kind == "finetuned_maskclip":
                cfg["checkpoint"] = _file_info(cfg["checkpoint"])["path"]
                result["files"]["checkpoint"] = _file_info(cfg["checkpoint"])
                for name in ("models/maskclip/maskclip.py", "tools/finetune.py", "configs/maskclip.yaml"):
                    result["files"][name] = _file_info(root / name)
            else:
                _, torch_home, hf_hub = _cache_roots(cfg)
                clip_cache = hf_hub / "models--laion--CLIP-ViT-B-16-laion2B-s34B-b88K/snapshots"
                clip_candidates = list(clip_cache.glob("*/open_clip_pytorch_model.bin"))
                clip_candidates += list((torch_home.parent / "clip").glob("*ViT-B-16*laion2b_s34b_b88k*"))
                cfg["clip_checkpoint"] = _one_local(cfg.get("clip_checkpoint"), "clip_checkpoint", clip_candidates)
                result["files"]["clip_checkpoint"] = _file_info(cfg["clip_checkpoint"])
                model_config = Path(cfg.get("model_config", root / "configs/stock_lposs.yaml"))
                if not model_config.is_absolute():
                    model_config = root / model_config
                cfg["model_config"] = str(model_config.resolve(strict=True))
                result["files"]["model_config"] = _file_info(model_config)
                if kind == "stock_lposs":
                    candidates = list((torch_home / "hub").glob("facebookresearch_dino_*"))
                    candidates += [root / "third_party/dino"]
                    cfg["dino_source_root"] = _one_local(cfg.get("dino_source_root"), "dino_source_root", candidates)
                    cfg["dino_checkpoint"] = _one_local(cfg.get("dino_checkpoint"), "dino_checkpoint", [torch_home / "hub/checkpoints/dino_vitbase16_pretrain.pth"])
                    result["files"]["dino_checkpoint"] = _file_info(cfg["dino_checkpoint"])
                    result["sources"]["dino"] = _source_info(cfg["dino_source_root"], ["hubconf.py", "vision_transformer.py", "utils.py"])
        result["ok"] = True
    except (OSError, ValueError, KeyError, ImportError) as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result


def _vocabulary(vocabulary):
    if not isinstance(vocabulary, (list, tuple)) or not vocabulary:
        raise ValueError("Vocabulary must be a nonempty list")
    normalized = []
    for item in vocabulary:
        name, prompts = item.get("name"), item.get("prompts")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Vocabulary class name must be nonempty")
        if not isinstance(prompts, (list, tuple)) or not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise ValueError(f"Class {name!r} needs nonempty text prompts")
        normalized.append({"name": name, "prompts": list(prompts), "group": item.get("group", "target")})
    if len({row["name"] for row in normalized}) != len(normalized):
        raise ValueError("Duplicate vocabulary class names")
    return normalized


def _rgb(value):
    value = np.asarray(value)
    if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8 or min(value.shape[:2]) == 0:
        raise ValueError("Segmenter expects nonempty RGB uint8 H,W,3")
    return np.ascontiguousarray(value)


def validate_prediction(result, shape):
    """Check shared contract without silently clipping or renormalizing scores."""
    values = np.asarray(result["scores"], dtype=np.float32)
    names = result["class_names"]
    if values.shape != (len(names), *tuple(shape)) or len(set(names)) != len(names):
        raise ValueError("Segmentation score shape or class ordering is invalid")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Segmentation scores must be finite and within [0,1]")
    result["scores"] = values
    result.setdefault("extra_arrays", {})
    return result


def softmax_scores(raw_scores, scale=100.0):
    """Explicit uncalibrated class competition; never apply to existing probabilities."""
    raw = np.asarray(raw_scores, np.float32)
    if raw.ndim != 3 or not np.isfinite(raw).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Expected finite C,H,W native scores and positive softmax scale")
    shifted = (raw - raw.max(axis=0, keepdims=True)) * float(scale)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=0, keepdims=True)).astype(np.float32)


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
        # NumPy cannot represent torch.bfloat16 (SAM3 CUDA autocast default).
        if value.is_floating_point():
            value = value.float()
        value = value.cpu().numpy()
    return np.array(value, copy=True)


def _local_clip_architecture(name):
    if not isinstance(name, str) or ":" in name:
        raise ValueError("Use a built-in local OpenCLIP architecture name; remote model configs are disabled")
    return name


@contextmanager
def _source_path(root):
    root = str(Path(root).resolve())
    sys.path.insert(0, root)
    try:
        yield
    finally:
        sys.path.remove(root)


@contextmanager
def _cwd(root):
    previous = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_tensors(torch, path, trust_checkpoint=False):
    try:
        return torch.load(path, map_location="cpu", weights_only=not trust_checkpoint)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        if not trust_checkpoint:
            raise RuntimeError("This PyTorch needs trust_checkpoint:true for legacy torch.load") from exc
        return torch.load(path, map_location="cpu")


class _BaseSegmenter:
    def close(self):
        model = getattr(self, "model", None)
        if model is not None and hasattr(model, "cpu"):
            model.cpu()
        self.model = None

    def predict(self, rgb):
        rgb = _rgb(rgb)
        return validate_prediction(self._predict(rgb), rgb.shape[:2])


class _DinoSegmenter(_BaseSegmenter):
    def __init__(self, config, vocabulary):
        module = importlib.import_module("facade_change.methods.2026-10-14_semantic_dino")
        keys = ("training_run", "device", "batch_size", "current_manifest", "expected_categories", "crop_size", "crop_stride")
        self.scorer = module.SemanticDINOScorer(**{k: config[k] for k in keys if k in config})
        self.class_names = [c["name"] for c in self.scorer.metadata["categories"]]
        self.metadata = dict(self.scorer.metadata, runtime_vocabulary_ignored=True)

    def _predict(self, rgb):
        return {"scores": self.scorer._probabilities(rgb), "class_names": self.class_names,
                "score_kind": "independent_sigmoid", "extra_arrays": {}}

    def close(self):
        self.scorer.close()


def sam3_image_state(state):
    """Project an official merged image/video checkpoint onto its image detector."""
    if not isinstance(state, dict) or not state:
        raise ValueError("SAM3 checkpoint must contain a nonempty model state mapping")
    if any(key.startswith("detector.") for key in state):
        return {key.removeprefix("detector."): value for key, value in state.items()
                if key.startswith("detector.")}
    return state  # A direct image-model state is checked strictly by load_state_dict.


class _SAM3Segmenter(_BaseSegmenter):
    def __init__(self, config, vocabulary):
        import torch
        if config.get("source_root"):
            sys.path.insert(0, config["source_root"])
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        self.config, self.vocabulary = config, _vocabulary(vocabulary)
        self.class_names = [row["name"] for row in self.vocabulary]
        self.device = config.get("device", "cuda:0")
        if self.device.startswith("cuda"):
            torch.cuda.set_device(torch.device(self.device))
        # Upstream's loader uses strict=False and only prints missing keys.
        # Build locally, then require complete image-model state ourselves.
        self.model = build_sam3_image_model(
            checkpoint_path=None, load_from_HF=False,
            bpe_path=config.get("bpe_path"), device="cpu",
            eval_mode=True, enable_segmentation=True, enable_inst_interactivity=False,
            compile=False,
        )
        saved = _load_tensors(torch, config["checkpoint"], config.get("trust_checkpoint", False))
        state = sam3_image_state(saved.get("model", saved))
        self.model.load_state_dict(state, strict=True)
        del state, saved
        # Some upstream versions move only for device == "cuda", not cuda:0.
        self.model.to(self.device).eval().requires_grad_(False)
        self.processor = Sam3Processor(self.model, device=self.device,
                                       confidence_threshold=float(config.get("confidence_threshold", .25)))
        self.metadata = {"adapter": "sam3", "class_names": self.class_names,
                         "score": "max_instance_and_prompt(instance_score * mask_probability)",
                         "mask_probability_field": "masks_logits (official processor already applies sigmoid)",
                         "prompt_reset": "reset_all_prompts before every independent query",
                         "mask_threshold": float(config.get("mask_threshold", .5)),
                         "confidence_threshold": float(config.get("confidence_threshold", .25)),
                         "precision": config.get("precision", "bfloat16"),
                         "prompt_texts": [prompt for row in self.vocabulary for prompt in row["prompts"]],
                         "checkpoint_loading": "strict=True; detector subtree for image model", "downloads": False}

    def _predict(self, rgb):
        import torch
        from PIL import Image
        autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if self.device.startswith("cuda") and self.config.get("precision", "bfloat16") == "bfloat16"
                    else nullcontext())
        with torch.inference_mode(), autocast:
            state = self.processor.set_image(Image.fromarray(rgb))
            return sam3_queries(self.processor, state, self.vocabulary, rgb.shape[:2],
                                mask_threshold=float(self.config.get("mask_threshold", .5)))


def sam3_queries(processor, state, vocabulary, shape, mask_threshold=.5):
    """Run independent prompts and copy tensors before mutable processor reuse."""
    vocabulary = _vocabulary(vocabulary)
    if not 0 <= mask_threshold <= 1:
        raise ValueError("mask_threshold must be in [0,1]")
    height, width = shape
    scores = np.zeros((len(vocabulary), height, width), np.float32)
    all_masks, all_probabilities, all_confidences, all_classes, all_prompts, all_boxes = [], [], [], [], [], []
    prompts = []
    for channel, entry in enumerate(vocabulary):
        for prompt in entry["prompts"]:
            prompt_index = len(prompts)
            prompts.append(prompt)
            processor.reset_all_prompts(state)
            output = processor.set_text_prompt(prompt=prompt, state=state)
            confidence = _numpy(output["scores"]).astype(np.float32).reshape(-1)
            # Upstream calls this masks_logits but it is post-sigmoid probability.
            probability = _numpy(output["masks_logits"]).astype(np.float32)
            if probability.ndim == 4 and probability.shape[1] == 1:
                probability = probability[:, 0]
            if probability.shape != (len(confidence), height, width):
                raise ValueError("SAM3 mask probability shape differs from native RGB grid")
            if (not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1)
                    or not np.isfinite(confidence).all() or np.any(confidence < 0) or np.any(confidence > 1)):
                raise ValueError("SAM3 processor scores/masks_logits are not probabilities in [0,1]")
            if len(confidence):
                scores[channel] = np.maximum(scores[channel], (confidence[:, None, None] * probability).max(0))
                boxes = _numpy(output["boxes"]).astype(np.float32)
                if boxes.shape != (len(confidence), 4):
                    raise ValueError("SAM3 boxes shape mismatch")
                all_masks.append(probability > mask_threshold)
                all_probabilities.append(probability)
                all_confidences.append(confidence)
                all_classes.extend([channel] * len(confidence))
                all_prompts.extend([prompt_index] * len(confidence))
                all_boxes.append(boxes)
    processor.reset_all_prompts(state)
    extra = {
        "instance_masks": np.concatenate(all_masks) if all_masks else np.empty((0, height, width), bool),
        "instance_probabilities": np.concatenate(all_probabilities) if all_probabilities else np.empty((0, height, width), np.float32),
        "instance_scores": np.concatenate(all_confidences) if all_confidences else np.empty(0, np.float32),
        "instance_class_indices": np.asarray(all_classes, np.int32),
        "instance_prompt_indices": np.asarray(all_prompts, np.int32),
        "instance_boxes": np.concatenate(all_boxes) if all_boxes else np.empty((0, 4), np.float32),
    }
    return {"scores": scores, "class_names": [row["name"] for row in vocabulary],
            "score_kind": "max_instance_score_times_mask_probability", "extra_arrays": extra,
            "metadata": {"prompt_texts": prompts}}


class _StockSegmenter(_BaseSegmenter):
    def __init__(self, config, vocabulary):
        import torch
        import yaml
        self.root, self.config = config["source_root"], config
        with _source_path(self.root):
            features = importlib.import_module("ovs_heritage.stock_features")
            native = importlib.import_module("ovs_heritage.stock_lposs")
            vocab = importlib.import_module("ovs_heritage.vocabulary")
        settings = yaml.safe_load(Path(config["model_config"]).read_text())
        clip = settings["clip"]
        self.mode = "maskclip_raw" if config["kind"] == "stock_maskclip" else "lposs"
        device = native.resolve_device(config.get("device", "cuda:0"))
        native.graph_preflight(self.mode, device)
        self.model = features.StockFeatureModel(
            clip_model=_local_clip_architecture(clip["model"]), clip_pretrained=config["clip_checkpoint"],
            patch_size=clip["patch_size"], image_size=clip["image_size"],
        )
        if self.mode == "lposs":
            dino = settings["dino"]
            encoder = torch.hub.load(config["dino_source_root"], dino["model"], source="local", pretrained=False)
            saved = _load_tensors(torch, config["dino_checkpoint"], config.get("trust_checkpoint", False))
            state = saved.get("teacher", saved)
            state = {key.removeprefix("module.").removeprefix("backbone."): value for key, value in state.items()}
            encoder.load_state_dict(state, strict=True)
            self.model.dino_encoder = encoder.eval().requires_grad_(False)
            self.model.dino_patch_size, self.model.dino_feature_type = dino["patch_size"], dino["feature_type"]
            encoder.blocks[-1].attn.qkv.register_forward_hook(
                lambda _module, _inputs, output: self.model._dino_hook.__setitem__("qkv", output))
        self.model.to(device.resolved_device).eval().requires_grad_(False)
        entries = _vocabulary(vocabulary)
        self.class_names = [item["name"] for item in entries]
        classes = [vocab.RuntimeClass(name=item["name"], prompts=tuple(item["prompts"])) for item in entries]
        self.prototypes = vocab.build_prototypes(classes, self.model.encode_text, device=device.resolved_device)
        graph = dict(settings["graph"])
        graph.update(config.get("graph", {}))
        graph["vit_patch_size"] = settings["dino"]["patch_size"]
        self.engine = native.StockOVSEngine(self.model, native.UpstreamGraphPropagator() if self.mode == "lposs" else None,
                                            device=device, cosine_scale=1.0, graph_parameters=graph)
        self.scale = float(config.get("softmax_scale", 100.0))
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("softmax_scale must be finite and positive")
        self.metadata = {"adapter": config["kind"], "native_implementation": native.IMPLEMENTATION_ID,
                         "class_names": self.class_names, "raw_scores": "unscaled cosine / graph-propagated cosine",
                         "raw_score_scale": 1.0, "softmax_scale": self.scale,
                         "probability": "vocabulary softmax, uncalibrated; not independent class probability",
                         "input_color_order": "rgb", "downloads": False}

    def _predict(self, rgb):
        import torch
        image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.
        with _source_path(self.root), torch.inference_mode():
            output = self.engine.run(image, self.prototypes, mode=self.mode, inference="whole")
        raw = _numpy(output.propagated_scores[0]).astype(np.float32)
        return {"scores": softmax_scores(raw, self.scale), "class_names": self.class_names,
                "score_kind": "vocabulary_softmax_uncalibrated",
                "extra_arrays": {"raw_scores": raw, "seed_scores": _numpy(output.seed_scores[0]).astype(np.float32)},
                "softmax_scale": self.scale}


def semantic_checkpoint_state(state):
    """Retain the exact semantic wrapper keys; drop only unused LPOSS DINO state."""
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint model_state must be a nonempty mapping")
    omitted = sorted(key for key in state if key.startswith("base_model.dino_encoder."))
    excluded = set(omitted)
    semantic = {key: value for key, value in state.items() if key not in excluded}
    if not any(key.startswith("base_model.clip_backbone.") for key in semantic):
        # Legacy direct MaskCLIP wrapper has no LPOSS shell.
        if not any(key.startswith("base_model.backbone.") for key in semantic):
            raise ValueError("No known native MaskCLIP semantic subtree found in checkpoint")
    return semantic, omitted


class _FineTunedSegmenter(_BaseSegmenter):
    def __init__(self, config, vocabulary):
        import torch
        from omegaconf import OmegaConf
        import open_clip
        self.config, self.root = config, config["source_root"]
        self.device = config.get("device", "cuda:0")
        if not self.device.startswith("cuda"):
            raise ValueError("Legacy MaskCLIP constructor requires CUDA; use the existing lposs environment")
        torch.cuda.set_device(torch.device(self.device))
        saved = _load_tensors(torch, config["checkpoint"], config.get("trust_checkpoint", True))
        state, omitted = semantic_checkpoint_state(saved.get("model_state", saved))
        self.class_names = list(LEGACY_CLASSES)
        with _source_path(self.root), _cwd(self.root):
            native = importlib.import_module("models.maskclip.maskclip")
            common_spec = importlib.util.spec_from_file_location("_semantic_routes_native_finetune", Path(self.root) / "tools/finetune.py")
            common = importlib.util.module_from_spec(common_spec)
            common_spec.loader.exec_module(common)
            cfg = OmegaConf.load(Path(self.root) / "configs/maskclip.yaml")["model"]
            # The full checkpoint contains every CLIP parameter and class embedding.
            # Avoid pretrained downloads while preserving the native architecture.
            factory = native.create_model_from_pretrained
            native.create_model_from_pretrained = lambda name, **kwargs: (open_clip.create_model(_local_clip_architecture(name), pretrained=None), None)
            try:
                clip = native.MaskClip(backbone=cfg.backbone, decode_head=cfg.decode_head,
                                       clip_model=cfg.clip_model, class_names=self.class_names)
            finally:
                native.create_model_from_pretrained = factory
            if any(key.startswith("base_model.clip_backbone.") for key in state):
                base = torch.nn.Module()
                base.add_module("clip_backbone", clip)
            else:
                base = clip
            mixers = [common.EmbeddingMixer(clip.decode_head.text_channels)] if config.get("use_embedding_mixer", False) else []
            self.model = common.FineTuneWrapper(base, mixers=mixers, mix_strategy=config.get("mix_strategy", "add"))
            self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        self.color_order = config.get("input_color_order", "bgr")
        if self.color_order not in {"bgr", "rgb"}:
            raise ValueError("input_color_order must be rgb or bgr")
        self.patch_size = int(clip.patch_size)
        self.metadata = {"adapter": "finetuned_maskclip", "class_names": self.class_names,
                         "runtime_vocabulary_ignored": True, "semantic_state_loading": "strict=True",
                         "omitted_unused_dino_state_keys": omitted,
                         "checkpoint_epoch": saved.get("epoch"),
                         "selection_score_name": saved.get("selection_score_name"),
                         "input_color_order": self.color_order,
                         "input_color_reason": "legacy mmcv LoadImageFromFile pipeline has no RGB conversion",
                         "score": "native FineTuneWrapper probability average; no additional softmax",
                         "spatial": "right/bottom edge-pad to patch multiple, native bilinear upsample, crop padding",
                         "downloads": False}

    def _predict(self, rgb):
        import torch
        import torch.nn.functional as functional
        height, width = rgb.shape[:2]
        value = rgb[..., ::-1].copy() if self.color_order == "bgr" else rgb
        value = np.pad(value, ((0, (-height) % self.patch_size), (0, (-width) % self.patch_size), (0, 0)), mode="edge")
        # Native MaskClip normalizes uint8 internally. Float255 would be wrong.
        inputs = torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            scores, _ = self.model(inputs)
            scores = functional.interpolate(scores, size=value.shape[:2], mode="bilinear", align_corners=False)
        return {"scores": _numpy(scores[0, :, :height, :width]).astype(np.float32),
                "class_names": self.class_names, "score_kind": "native_exclusive_probability", "extra_arrays": {}}


def make_segmenter(config, vocabulary):
    checked = preflight(config)
    if not checked["ok"]:
        raise RuntimeError("Segmentation preflight failed: " + "; ".join(checked["errors"]))
    cfg = checked["config"]
    classes = {"dino_semantic": _DinoSegmenter, "semantic_dino": _DinoSegmenter,
               "sam3": _SAM3Segmenter, "stock_maskclip": _StockSegmenter,
               "stock_lposs": _StockSegmenter, "finetuned_maskclip": _FineTunedSegmenter}
    adapter = classes[cfg["kind"]](cfg, vocabulary)
    adapter.metadata["provenance"] = checked
    return adapter
