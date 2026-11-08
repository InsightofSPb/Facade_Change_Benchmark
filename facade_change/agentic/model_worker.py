"""Real offline Qwen/SAM3/Jina/DINO adapters, imported only in owned workers.

API sources checked for the 2026-11-08 integration:
* Qwen pinned config + Transformers qwen3_5 multimodal processor documentation;
* facebookresearch/sam3 at SAM3_REVISION, model_builder.py and
  model/sam3_image_processor.py (masks_logits is post-sigmoid probability);
* jinaai/jina-clip-v2 and its separately pinned jina-clip-implementation code.

No fake model is selectable from the production CLI. CPU tests inject adapters
at the protocol boundary and explicitly identify themselves as contract tests.
"""
from __future__ import annotations

import copy
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
from PIL import Image

from ..io import load_rgb, sha256
from .models import (DEFAULTS, QWEN_REVISION, SAM3_REVISION, JINA_REVISION,
                     JINA_CODE_REVISION, digest, positive_number)


SAM3_SOURCE_HASHES = {
    "model_builder.py": "d71d6d3e485ec3eae48bbc2ba676f401b5853d65c4195a91d077b04da38121c2",
    "model/sam3_image_processor.py": "d8738a0efb6138b01c0dc5deceffd29de9e675860a9a1ed3822b08766373333b",
}


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _bounded(request, config, name):
    ceiling = _integer(config[name], name)
    value = _integer(request.get(name, ceiling), name)
    if value > ceiling:
        raise ValueError(f"{name} exceeds configured ceiling {ceiling}")
    return value


def _resource(path):
    path = Path(path).expanduser().resolve()
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def _versions():
    values = {"python": sys.version.split()[0], "executable": sys.executable}
    for name in ("torch", "torchvision", "transformers", "huggingface-hub", "sam3", "einops", "timm", "Pillow"):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def _package_resources():
    """Bind cached results to installed dependency versions without importing GPU libraries."""
    resources = []
    for name in ("torch", "torchvision", "transformers", "huggingface-hub", "sam3", "einops", "timm", "Pillow", "numpy"):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        metadata = next((p for p in distribution.files or [] if p.name == "METADATA" and ".dist-info" in str(p)), None)
        if metadata is not None:
            resources.append(_resource(distribution.locate_file(metadata)))
    return resources


def _snapshot(model_id, revision):
    from huggingface_hub import snapshot_download
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("A full immutable model revision is required")
    root = Path(snapshot_download(model_id, revision=revision, local_files_only=True)).resolve()
    weights = sorted(root.glob("*.safetensors")) + sorted(root.glob("pytorch_model*.bin"))
    if not weights:
        raise FileNotFoundError(f"No local model weights in pinned snapshot: {root}")
    index = root / "model.safetensors.index.json"
    if index.exists():
        expected = set(json.loads(index.read_text())["weight_map"].values())
        if any(not (root / name).is_file() for name in expected):
            raise FileNotFoundError(f"Missing weight shards in pinned snapshot: {root}")
    # Hashes are collected once per worker, before loading. Actual checksum work
    # is charged to check/load time; no metadata-only success for absent weights.
    files = sorted({*weights, *root.glob("*.json"), *root.glob("*.jinja"), *root.glob("*.py"),
                    *root.glob("*.txt"), *root.glob("*.model"), *root.glob("*.tiktoken")})
    return str(root), [_resource(path) for path in files]


def _image(path, maximum):
    with Image.open(path) as header:
        w, h = header.size
    if h * w > _integer(maximum, "max_image_pixels"):
        raise ValueError(f"Image {w}x{h} exceeds configured max_image_pixels; no implicit downscale")
    rgb, opacity = load_rgb(path)
    return Image.fromarray(rgb), {"path": str(Path(path).resolve()), "sha256": sha256(path),
                                  "shape_hw": [h, w], "fully_opaque": bool(opacity.all())}


def _cuda_state(device):
    import torch
    if not str(device).startswith("cuda"):
        return {"device": str(device)}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the configured worker environment")
    free, total = torch.cuda.mem_get_info(device)
    return {"device": str(device), "device_name": torch.cuda.get_device_name(device),
            "free_bytes": free, "total_bytes": total,
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "scope": "this worker torch allocator; free_bytes is device-wide snapshot"}


def _cuda_preflight(device):
    state = _cuda_state(device)
    if str(device).startswith("cuda"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=False)
            state["compute_processes"] = result.stdout.strip() if result.returncode == 0 else "unavailable: " + result.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            state["compute_processes"] = f"unavailable: {exc}"
    return state


class Adapter:
    def __init__(self, role, options):
        allowed = set(DEFAULTS[role]) | {"python", "local_files_only", "checkpoint_path", "source_root"}
        if set(options) - allowed:
            raise ValueError(f"Unexpected {role} options: {sorted(set(options) - allowed)}")
        if options.get("local_files_only", True) is not True:
            raise ValueError("Workers require local_files_only=True; downloads are never implicit")
        self.role, self.options = role, {**DEFAULTS[role], **options}
        self.model = self.processor = None
        self.checked = False
        self.environment_resources = _package_resources()
        self.provenance = {"role": role, "options": self.options, "versions": _versions(),
                           "model": self.options.get("model_id", role), "revision": self.options.get("revision"),
                           "config_sha256": digest(self.options), "local_files_only": True,
                           "downloads": False, "resources": self.environment_resources.copy()}
        self.usage = {}

    def _begin(self):
        self.usage = {"model_loads": 0, "model_calls": 0, "images": 0,
                      "input_tokens": 0, "output_tokens": 0}

    def call(self, request):
        self._begin()
        started = time.monotonic()
        try:
            result = self.handle(request)
            if self.model is not None:
                self.usage["memory"] = _cuda_state(self.options["device"])
            return {"status": "ok", **result, "usage": {**self.usage, "seconds": time.monotonic() - started},
                    "provenance": self.provenance}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            # Do not require torch to import just to classify an optional model
            # dependency failure. CUDA OOM is returned explicitly to supervisor.
            name, message = type(exc).__name__, str(exc)
            oom = name in {"OutOfMemoryError", "CUDAOutOfMemoryError"} or "cuda out of memory" in message.lower()
            try:
                memory = _cuda_state(self.options.get("device", "cpu")) if "torch" in sys.modules else {}
            except Exception:
                memory = {}
            return {"status": "error", "error": {"code": "out_of_memory" if oom else "model_error",
                                                     "message": f"{name}: {message}"},
                    "usage": {**self.usage, "seconds": time.monotonic() - started, "memory": memory},
                    "provenance": self.provenance}

    def handle(self, request):
        raise NotImplementedError

    def close(self):
        self.model = self.processor = None


class QwenAdapter(Adapter):
    def _check(self):
        if self.checked:
            return
        options = self.options
        if options["model_id"] != "Qwen/Qwen3.5-4B" or options["revision"] != QWEN_REVISION:
            raise ValueError("This integration requires the specified Qwen3.5-4B revision")
        if options["precision"] != "bfloat16" or options["attention"] != "sdpa" or options["device"] != "cuda":
            raise ValueError("Qwen's verified mode is BF16, SDPA, cuda; no silent fallback")
        if options["enable_thinking"] is not False:
            raise ValueError("Qwen's configured mode requires enable_thinking=False")
        for name in ("max_images", "max_image_pixels", "image_min_pixels", "image_max_pixels", "max_input_tokens", "max_new_tokens"):
            _integer(options[name], name)
        _integer(options["seed"], "seed", 0)
        if type(options["do_sample"]) is not bool or not 0 < float(options["top_p"]) <= 1:
            raise ValueError("do_sample must be boolean and top_p must be in (0,1]")
        positive_number(options["temperature"], "temperature")
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        self.provenance["hardware"] = _cuda_preflight("cuda")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Configured CUDA device does not support BF16")
        self.snapshot, resources = _snapshot(options["model_id"], options["revision"])
        resources.append(_resource(inspect.getfile(Qwen3_5ForConditionalGeneration)))
        self.provenance.update(snapshot=self.snapshot, resources=resources + self.environment_resources,
                               enable_thinking=False, batch_size=1,
                               processing={"image_min_pixels": options["image_min_pixels"],
                                           "image_max_pixels": options["image_max_pixels"],
                                           "original_pixels_limit": options["max_image_pixels"]})
        self.processor = AutoProcessor.from_pretrained(
            options["model_id"], revision=options["revision"], local_files_only=True,
            min_pixels=_integer(options["image_min_pixels"], "image_min_pixels"),
            max_pixels=_integer(options["image_max_pixels"], "image_max_pixels"),
            size={"shortest_edge": options["image_min_pixels"], "longest_edge": options["image_max_pixels"]})
        if options["image_min_pixels"] > options["image_max_pixels"]:
            raise ValueError("image_min_pixels exceeds image_max_pixels")
        self.provenance["resources"].append(_resource(inspect.getfile(type(self.processor))))
        self.provenance["image_processor"] = self.processor.image_processor.to_dict()
        self.checked = True

    def _load(self):
        self._check()
        if self.model is not None:
            return
        import torch
        from transformers import Qwen3_5ForConditionalGeneration
        self.usage["model_loads"] += 1
        started = time.monotonic()
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.options["model_id"], revision=self.options["revision"], local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        self.usage["load_seconds"] = time.monotonic() - started
        self.provenance["model_dtype"] = str(next(self.model.parameters()).dtype)

    def _inputs(self, request):
        self._check()
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Qwen requires a nonempty messages list")
        maximum_images = _bounded(request, self.options, "max_images")
        maximum_tokens = _bounded(request, self.options, "max_input_tokens")
        rendered, images = [], []
        for message in messages:
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] not in {"system", "user", "assistant"}
                    or not isinstance(message["content"], list)):
                raise ValueError("Messages require only role and typed content; tools are serialized as observations")
            content = []
            for block in message["content"]:
                if not isinstance(block, dict):
                    raise ValueError("Content blocks must be objects")
                if block.get("type") == "text" and set(block) <= {"type", "text"} and isinstance(block.get("text"), str):
                    content.append({"type": "text", "text": block["text"]})
                elif block.get("type") == "image" and set(block) <= {"type", "path", "evidence_id"}:
                    image, info = _image(block["path"], self.options["max_image_pixels"])
                    images.append(info)
                    if len(images) > maximum_images:
                        raise ValueError("Message images exceed configured/request max_images")
                    content.append({"type": "image", "image": image})
                else:
                    raise ValueError("Only text and local RGB image blocks are accepted")
            rendered.append({"role": message["role"], "content": content})
        if not images:
            raise ValueError("Facade Qwen requests must include actual images")
        inputs = self.processor.apply_chat_template(
            rendered, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", enable_thinking=False)
        input_tokens = int(inputs["input_ids"].shape[-1])
        if input_tokens > maximum_tokens:
            raise ValueError(f"Input has {input_tokens} tokens, exceeds max_input_tokens={maximum_tokens}; no truncation")
        if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
            raise RuntimeError("Multimodal processor did not materialize image tensors")
        self.usage.update(input_tokens=input_tokens, images=len(images))
        processing = {"images": images, "image_grid_thw": inputs["image_grid_thw"].tolist(),
                      "input_tokens": input_tokens, "max_input_tokens": maximum_tokens}
        return inputs, processing

    def handle(self, request):
        allowed = {"op", "messages", "max_new_tokens", "max_input_tokens", "max_images", "timeout_seconds", "geometry"}
        if request.get("op") == "check":
            if set(request) - {"op", "load_model", "timeout_seconds"}:
                raise ValueError("Unexpected model check arguments")
            self._check()
            if request.get("load_model", False):
                self._load()
            return {"check": "imports_and_pinned_local_weights", "model_loaded": self.model is not None}
        if set(request) - allowed or request.get("op") not in {"generate", "measure"}:
            raise ValueError("Unsupported Qwen request")
        inputs, processing = self._inputs(request)
        maximum_new = _bounded(request, self.options, "max_new_tokens")
        if request["op"] == "measure":
            return {"processing": processing, "input_tokens": processing["input_tokens"],
                    "images": len(processing["images"]), "model_loaded": self.model is not None}
        self._load()
        import torch
        torch.manual_seed(_integer(self.options["seed"], "seed", 0))
        torch.cuda.manual_seed_all(self.options["seed"])
        self.usage["model_calls"] += 1
        generation = {"max_new_tokens": maximum_new, "do_sample": bool(self.options["do_sample"])}
        if generation["do_sample"]:
            generation.update(temperature=positive_number(self.options["temperature"], "temperature"),
                              top_p=positive_number(self.options["top_p"], "top_p"))
        if request.get("timeout_seconds"):
            generation["max_time"] = positive_number(request["timeout_seconds"], "timeout_seconds")
        inputs = inputs.to("cuda")
        with torch.inference_mode():
            output = self.model.generate(**inputs, **generation)
        generated = output[:, inputs["input_ids"].shape[1]:]
        self.usage["output_tokens"] = int(generated.shape[1])
        text = self.processor.batch_decode(generated, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]
        return {"text": text, "processing": processing,
                "generation": {**generation, "enable_thinking": False, "seed": self.options["seed"],
                               "actual_output_tokens": self.usage["output_tokens"],
                               "reached_max_new_tokens": self.usage["output_tokens"] == maximum_new}}


class SAM3Adapter(Adapter):
    def _check(self):
        if self.checked:
            return
        if self.options["revision"] != SAM3_REVISION or self.options["device"] != "cuda":
            raise ValueError("SAM3 requires the specified revision and builder device='cuda'")
        if self.options["resolution"] != 1008:
            raise ValueError("The verified SAM3 processor uses resolution 1008")
        _integer(self.options["checkpoint_bytes"], "checkpoint_bytes")
        threshold = self.options["confidence_threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("SAM3 confidence_threshold must be numeric in [0,1]")
        import torch
        import sam3
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        self.provenance["hardware"] = _cuda_preflight("cuda")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("SAM3 verified autocast mode requires BF16")
        checkpoint = Path(self.options["checkpoint_path"]).expanduser().resolve()
        if not checkpoint.is_file() or checkpoint.stat().st_size != self.options["checkpoint_bytes"]:
            raise ValueError(f"SAM3 checkpoint missing or size differs from {self.options['checkpoint_bytes']} bytes: {checkpoint}")
        sources = [Path(inspect.getfile(build_sam3_image_model)), Path(inspect.getfile(Sam3Processor))]
        resources = [_resource(checkpoint), *[_resource(path) for path in sources]]
        expected = list(SAM3_SOURCE_HASHES.values())
        source_matches = [row["sha256"] for row in resources[1:]] == expected
        if self.options.get("strict_source", True) and not source_matches:
            raise ValueError("Installed SAM3 builder/processor differs from the specified revision")
        package_root = Path(sam3.__file__).resolve().parent
        already = {row["path"] for row in resources}
        resources.extend(_resource(p) for p in sorted(package_root.rglob("*.py")) if str(p.resolve()) not in already)
        resources.append(_resource(package_root / "assets" / "bpe_simple_vocab_16e6.txt.gz"))
        direct = None
        try:
            distribution = importlib.metadata.distribution("sam3")
            text = distribution.read_text("direct_url.json")
            direct = json.loads(text) if text else None
        except importlib.metadata.PackageNotFoundError:
            pass
        observed = (direct or {}).get("vcs_info", {}).get("commit_id")
        if observed and observed != SAM3_REVISION:
            raise ValueError("Installed SAM3 direct_url commit differs from requested revision")
        self.provenance.update(resources=resources + self.environment_resources, package_path=str(Path(sam3.__file__).resolve()),
                               direct_url=direct, revision_verification=("unverified_critical_sources_differ" if not source_matches else
                                    "installed_vcs_commit_and_critical_hashes" if observed else "critical_builder_processor_hashes"),
                               masks_logits_semantics="post_sigmoid_probabilities_0_1",
                               mask_threshold=0.5, model_precision="float32", autocast="bfloat16",
                               processor_resolution=1008)
        self.checked = True

    def _load(self):
        self._check()
        if self.model is not None:
            return
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        self.usage["model_loads"] += 1
        started = time.monotonic()
        self.model = build_sam3_image_model(checkpoint_path=self.options["checkpoint_path"],
                                          load_from_HF=False, device="cuda", eval_mode=True,
                                          enable_segmentation=True, enable_inst_interactivity=False,
                                          compile=False)
        import torch
        if next(self.model.parameters()).dtype != torch.float32:
            raise ValueError("SAM3 model must retain the verified FP32 parameter dtype")
        threshold = float(self.options["confidence_threshold"])
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("SAM3 confidence_threshold must be in [0,1]")
        self.processor = Sam3Processor(self.model, resolution=1008, device="cuda", confidence_threshold=threshold)
        self.usage["load_seconds"] = time.monotonic() - started

    def handle(self, request):
        if request.get("op") == "check":
            if set(request) - {"op", "load_model", "timeout_seconds"}:
                raise ValueError("Unexpected SAM3 check arguments")
            self._check()
            if request.get("load_model", False):
                self._load()
            return {"check": "imports_checkpoint_and_critical_source_hashes", "model_loaded": self.model is not None}
        if request.get("op") != "segment" or set(request) - {"op", "image_path", "text", "out", "geometry", "timeout_seconds"}:
            raise ValueError("SAM3 accepts a local crop and one text concept only")
        if not isinstance(request.get("text"), str) or not request["text"].strip() or len(request["text"]) > 512:
            raise ValueError("SAM3 text must be a nonempty concept of at most 512 characters")
        image, info = _image(request["image_path"], self.options["max_image_pixels"])
        output = Path(request["out"]).expanduser().resolve()
        # Do not create output until inference succeeds, allowing one idempotent
        # retry of an OOM without overwriting any completed result.
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"SAM3 output directory is not empty: {output}")
        self._load()
        import torch
        self.usage.update(model_calls=1, images=1)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(image)
            state = self.processor.set_text_prompt(prompt=request["text"], state=state)
        probabilities = state["masks_logits"].detach().float().cpu().numpy()
        masks = state["masks"].detach().cpu().numpy()
        scores = state["scores"].detach().float().cpu().numpy().reshape(-1)
        boxes = state["boxes"].detach().float().cpu().numpy().reshape(-1, 4)
        if probabilities.ndim == 4 and probabilities.shape[1] == 1:
            probabilities, masks = probabilities[:, 0], masks[:, 0]
        h, w = info["shape_hw"]
        if (probabilities.shape != (len(scores), h, w) or masks.shape != probabilities.shape
                or masks.dtype != np.bool_ or len(boxes) != len(scores)
                or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1)
                or not np.array_equal(masks, probabilities > .5)
                or not np.isfinite(boxes).all() or not np.isfinite(scores).all()
                or np.any(scores < 0) or np.any(scores > 1)):
            raise ValueError("SAM3 returned incompatible native-grid masks/probabilities/boxes")
        output.mkdir(parents=True, exist_ok=True)
        mask_path, probability_path = output / "masks.npy", output / "probabilities.npy"
        np.save(mask_path, masks, allow_pickle=False)
        np.save(probability_path, probabilities, allow_pickle=False)
        return {"status": "ok" if len(masks) else "empty", "masks_path": str(mask_path),
                "probabilities_path": str(probability_path), "boxes": boxes.tolist(), "scores": scores.tolist(),
                "native_shape": [h, w], "coordinate_system": "crop_native_xy",
                "mask_grid": "integer pixel centres; x=column,y=row", "boxes_format": "xyxy_native_pixels",
                "prompt": request["text"], "image": info, "geometry": request.get("geometry"),
                "artifacts": [_resource(mask_path), _resource(probability_path)],
                "interpretation": "text-conditioned segmentation, not a temporal-change verdict"}


class JinaAdapter(Adapter):
    def _check(self):
        if self.checked:
            return
        options = self.options
        if options["model_id"] != "jinaai/jina-clip-v2":
            raise ValueError("The joint encoder adapter requires jinaai/jina-clip-v2")
        if not options.get("trust_remote_code"):
            raise ValueError("Jina adapter requires explicit trust_remote_code for pinned implementation")
        if options["truncate_dim"] not in {32, 64, 128, 256, 512, 768, 1024}:
            raise ValueError("Unsupported Jina matryoshka dimension")
        if options["precision"] not in {"float32", "bfloat16"} or options["device"] not in {"cpu", "cuda"}:
            raise ValueError("Jina needs explicit cpu/cuda and float32/bfloat16 configuration")
        from transformers import AutoModel
        from huggingface_hub import snapshot_download
        import einops, timm  # noqa: F401 -- preflight dependencies, no install side effects
        self.snapshot, resources = _snapshot(options["model_id"], options["revision"])
        code_revision = options["code_revision"]
        if not isinstance(code_revision, str) or len(code_revision) != 40:
            raise ValueError("Jina remote implementation needs a full pinned code_revision")
        code = Path(snapshot_download("jinaai/jina-clip-implementation", revision=code_revision,
                                      local_files_only=True)).resolve()
        source_files = sorted(code.glob("*.py"))
        if not source_files:
            raise FileNotFoundError(f"Jina remote implementation is absent from local cache: {code}")
        resources.extend(_resource(p) for p in source_files)
        self.text_snapshot = Path(snapshot_download("jinaai/jina-embeddings-v3",
                                      revision=options["text_config_revision"], local_files_only=True)).resolve()
        self.text_code = Path(snapshot_download("jinaai/xlm-roberta-flash-implementation",
                                      revision=options["text_code_revision"], local_files_only=True)).resolve()
        resources.append(_resource(self.text_snapshot / "config.json"))
        text_sources = sorted(self.text_code.glob("*.py"))
        if not text_sources:
            raise FileNotFoundError(f"Pinned Jina text implementation is absent: {self.text_code}")
        resources.extend(_resource(p) for p in text_sources)
        self.provenance.update(snapshot=self.snapshot, implementation_snapshot=str(code), resources=resources + self.environment_resources,
                               space={"model": options["model_id"], "representation": "joint_image_text",
                                      "family": "jina-clip-v2", "model_id": options["model_id"],
                                      "revision": options["revision"], "code_revision": code_revision,
                                      "text_config_revision": options["text_config_revision"],
                                      "text_code_revision": options["text_code_revision"],
                                      "dimensions": options["truncate_dim"], "joint_image_text": True,
                                      "preprocessing": {"image": "pinned Jina image processor",
                                                        "text": "pinned Jina tokenizer; optional retrieval.query task",
                                                        "normalize_embeddings": True,
                                                        "truncate_dim": options["truncate_dim"]}},
                               preprocessing="pinned model tokenizer and image processor, native Jina encode APIs",
                               use_text_flash_attn=False, use_vision_xformers=False)
        self.checked = True

    def _load(self):
        self._check()
        if self.model is not None:
            return
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoImageProcessor
        options = self.options
        precision = {"float32": torch.float32, "bfloat16": torch.bfloat16}.get(options["precision"])
        if precision is None:
            raise ValueError("Jina precision must explicitly be float32 or bfloat16")
        self.provenance["hardware"] = _cuda_preflight(options["device"])
        self.usage["model_loads"] += 1
        started = time.monotonic()
        # Upstream _build_text_tower hardcodes revision=None. Its configuration
        # can nevertheless point to a local directory. Bind that directory to
        # the explicitly pinned config/code, without changing model weights or
        # patching an installed package / shared HuggingFace cache.
        self.text_bundle = tempfile.TemporaryDirectory(prefix="facade-jina-text-")
        local_text = Path(self.text_bundle.name)
        for source in self.text_code.glob("*.py"):
            shutil.copyfile(source, local_text / source.name)
        text_config = json.loads((self.text_snapshot / "config.json").read_text())
        text_config["auto_map"] = {key: value.split("--")[-1] for key, value in text_config["auto_map"].items()}
        (local_text / "config.json").write_text(json.dumps(text_config), encoding="utf-8")
        config = AutoConfig.from_pretrained(options["model_id"], revision=options["revision"],
                                           code_revision=options["code_revision"],
                                           trust_remote_code=True, local_files_only=True)
        config.text_config.hf_model_name_or_path = str(local_text)
        config.use_text_flash_attn = False
        config.use_vision_xformers = False
        self.model = AutoModel.from_pretrained(
            options["model_id"], revision=options["revision"], code_revision=options["code_revision"],
            config=config, local_files_only=True, trust_remote_code=True,
            torch_dtype=precision).to(options["device"]).eval()
        # Upstream lazy getters otherwise resolve _name_or_path without revision.
        # Inject pinned local tokenizer/processor explicitly, retaining real APIs.
        self.model.tokenizer = AutoTokenizer.from_pretrained(self.snapshot, code_revision=options["code_revision"],
                                                             local_files_only=True, trust_remote_code=True)
        self.model.preprocess = AutoImageProcessor.from_pretrained(self.snapshot, code_revision=options["code_revision"],
                                                                   local_files_only=True, trust_remote_code=True)
        if not callable(getattr(self.model, "encode_image", None)) or not callable(getattr(self.model, "encode_text", None)):
            raise ValueError("Pinned Jina model lacks encode_image/encode_text API")
        self.usage["load_seconds"] = time.monotonic() - started

    def handle(self, request):
        if request.get("op") == "check":
            if set(request) - {"op", "load_model", "timeout_seconds"}:
                raise ValueError("Unexpected Jina check arguments")
            self._check()
            if request.get("load_model", False):
                self._load()
            return {"check": "imports_and_pinned_model_and_code_cache", "model_loaded": self.model is not None}
        if request.get("op") != "embed" or set(request) - {"op", "texts", "image_paths", "task", "geometry", "timeout_seconds"}:
            raise ValueError("Unsupported Jina embedding request")
        if ("texts" in request) == ("image_paths" in request):
            raise ValueError("Provide exactly one of texts or image_paths")
        text_mode = "texts" in request
        values = request["texts"] if text_mode else request["image_paths"]
        if not isinstance(values, list) or not values or any(not isinstance(v, str) for v in values):
            raise ValueError("Embeddings require a nonempty string list")
        if len(values) > self.options["max_texts" if text_mode else "max_images"]:
            raise ValueError("Embedding batch exceeds configured limit")
        if text_mode and any(len(value) > self.options["max_text_chars"] for value in values):
            raise ValueError("Text exceeds max_text_chars; no silent truncation")
        inputs = values if text_mode else [_image(path, self.options["max_image_pixels"])[0] for path in values]
        task = request.get("task")
        if task not in {None, "retrieval.query"}:
            raise ValueError("Jina supports no task or retrieval.query in this adapter")
        self._load()
        options = {"batch_size": 1, "show_progress_bar": False, "convert_to_numpy": True,
                   "normalize_embeddings": True, "truncate_dim": self.options["truncate_dim"]}
        self.usage.update(model_calls=1, images=0 if text_mode else len(values))
        if text_mode:
            tokens = self.model.tokenizer(values, padding=False, truncation=False)["input_ids"]
            limit = getattr(self.model.tokenizer, "model_max_length", 8192)
            if any(len(row) > min(limit, 8192) for row in tokens):
                raise ValueError("Jina text exceeds tokenizer context; no silent truncation")
            self.usage["input_tokens"] = sum(len(row) for row in tokens)
            embeddings = self.model.encode_text(inputs, task=task, **options)
        else:
            embeddings = self.model.encode_image(inputs, **options)
        embeddings = np.asarray(embeddings, np.float32)
        if embeddings.shape != (len(values), self.options["truncate_dim"]) or not np.isfinite(embeddings).all():
            raise ValueError("Invalid Jina embedding dimensions/values")
        return {"vectors": embeddings.tolist(), "space": self.provenance["space"],
                "modality": "text" if text_mode else "image", "task": task}

    def close(self):
        super().close()
        if getattr(self, "text_bundle", None):
            self.text_bundle.cleanup()


class DinoAdapter(Adapter):
    def _check(self):
        if self.checked:
            return
        import torch  # noqa: F401 -- verify configured environment without loading weights
        from ..methods.dinov2 import _source_provenance
        from .retrieval import dino_embedding_space
        checkpoint = _resource(self.options["checkpoint_path"])
        source = _source_provenance(self.options["source_root"])
        self.contract = {"checkpoint_sha256": checkpoint["sha256"], "python_tree_sha256": source["python_tree_sha256"]}
        resources = [checkpoint, *[_resource(p) for p in sorted(Path(self.options["source_root"]).rglob("*.py"))]]
        resources += [_resource(p) for p in (Path(__file__).parents[1] / "history_features.py",
                                             Path(__file__).parents[1] / "methods" / "dinov2.py",
                                             Path(__file__).with_name("retrieval.py"))]
        self.provenance.update(resources=resources + self.environment_resources, sources=source,
                               model="dinov2_vits14", revision=checkpoint["sha256"],
                               space=dino_embedding_space(checkpoint["sha256"], source["python_tree_sha256"]),
                               preprocessing="existing history_features.DinoEncoder; bottom/right unsupported padding to 14; no resize")
        self.checked = True

    def _load(self):
        self._check()
        if self.model is not None:
            return
        from ..history_features import DinoEncoder
        self.usage["model_loads"] += 1
        started = time.monotonic()
        self.model = DinoEncoder(self.options, self.options["device"], self.contract)
        self.usage["load_seconds"] = time.monotonic() - started
        self.provenance["encoder_load"] = self.model.provenance

    def handle(self, request):
        if request.get("op") == "check":
            if set(request) - {"op", "load_model", "timeout_seconds"}:
                raise ValueError("Unexpected DINO check arguments")
            self._check()
            if request.get("load_model", False):
                self._load()
            return {"check": "local_dino_source_and_checkpoint", "model_loaded": self.model is not None}
        if request.get("op") != "embed" or set(request) - {"op", "image_paths", "support_paths", "geometry", "timeout_seconds"}:
            raise ValueError("DINO accepts local image_paths and optional support_paths")
        paths = request.get("image_paths")
        supports = request.get("support_paths", [None] * len(paths or []))
        if not isinstance(paths, list) or not paths or len(paths) > self.options["max_images"] or len(supports) != len(paths):
            raise ValueError("Invalid image/support list")
        self._load()
        vectors = []
        for path, support_path in zip(paths, supports):
            image, _ = _image(path, self.options["max_image_pixels"])
            rgb = np.asarray(image)
            if support_path and Path(support_path).suffix == ".npy":
                support = np.load(support_path, allow_pickle=False)
            elif support_path:
                with Image.open(support_path) as source_mask:
                    support = np.asarray(source_mask)
                if support.dtype != np.uint8 or support.ndim != 2 or not np.isin(support, [0, 255]).all():
                    raise ValueError("DINO support image must be binary uint8 0/255")
                support = support == 255
            else:
                support = np.ones(rgb.shape[:2], bool)
            if support.dtype != bool or support.shape != rgb.shape[:2]:
                raise ValueError("DINO support must be a native-grid boolean array")
            h, w = support.shape
            dh, dw = (-h) % 14, (-w) % 14
            padded = np.pad(rgb, ((0, dh), (0, dw), (0, 0)), mode="constant")
            valid = np.pad(support, ((0, dh), (0, dw)), mode="constant")
            vectors.append(self.model.encode([padded], [valid])[0]["z"].tolist())
            self.usage["images"] += 1
            self.usage["model_calls"] += 1
        return {"vectors": vectors, "space": self.provenance["space"], "modality": "image"}


def make_adapter(role, options):
    return {"qwen": QwenAdapter, "sam3": SAM3Adapter, "jina": JinaAdapter, "dino": DinoAdapter}[role](role, options)


def serve(requests, responses, factory=make_adapter):
    adapter = None
    try:
        for line in requests:
            identifier = None
            request = {}
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("Worker request must be an object")
                identifier = request.pop("id")
                if request.get("op") == "init":
                    if adapter is not None or set(request) != {"op", "role", "options"} or request["role"] not in DEFAULTS:
                        raise ValueError("Invalid or repeated worker initialization")
                    adapter = factory(request["role"], request["options"])
                    result = {"status": "ok", "provenance": adapter.provenance}
                elif request.get("op") == "close":
                    result = {"status": "ok"}
                elif adapter is None:
                    raise ValueError("Worker must be initialized first")
                else:
                    result = adapter.call(request)
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                result = {"status": "error", "error": {"code": "protocol_error", "message": str(exc)}}
            responses.write(json.dumps({"id": identifier, **result}, allow_nan=False) + "\n")
            responses.flush()
            if isinstance(request, dict) and request.get("op") == "close":
                break
    finally:
        if adapter:
            adapter.close()


def main():
    # Preserve a dedicated protocol FD: native/Python model logs go to stderr.
    control = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    try:
        serve(sys.stdin, control)
    finally:
        control.close()


if __name__ == "__main__":
    main()
