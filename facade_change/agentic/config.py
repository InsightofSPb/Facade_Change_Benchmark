"""Explicit local configuration; no discovery of latest runs or model downloads."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from ..io import read_json
from ..methods.registry import ALL_METHODS


ROOT = Path(__file__).resolve().parents[2]
QWEN_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SAM3_REVISION = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"
JINA_REVISION = "e10d47f5691d0454a0fb5d13f46f2199b74cb436"
EXCLUDED_METHODS = {"geoscd", "rscd_cmu", "rscd_diff_cmu", "rscd_pscd"}
MODES = ("pair", "fixed_context", "agent")
AVAILABILITY_POLICY = {
    "version": 1, "history": "dated available prefix; current observations excluded",
    "similar_cases": "train only; exclude target building, views and image hashes",
    "gt": "evaluator only; never available to methods, retrieval or VLM",
    "unknown_visibility": "unknown; geometric support is not physical visibility",
}


def defaults():
    return {
        "schema_version": 1,
        "store_root": str(ROOT / "data/agentic/regions"),
        "knowledge_root": str(ROOT / "data/agentic/knowledge"),
        "cache_root": str(ROOT / "data/agentic/cache"),
        "gpu_mode": "sequential", "worker_timeout": 300, "retrieval_encoder": None,
        "modes": list(MODES), "methods": ["ssim", "lzma_abs"],
        "method_options": {}, "thresholds": {"ssim": .15},
        "diagnostic_baseline": "ssim", "fusion": None,
        "max_region_pixels": 4000000,
        "alignment": {"method": "sift", "device": "cpu"},
        "budgets": {
            "max_actions": 4, "max_model_calls": 5, "max_images": 8,
            "max_total_images": 32, "max_input_tokens": 4096,
            "max_total_input_tokens": 16384, "max_output_tokens": 1024,
            "max_total_output_tokens": 4096, "max_seconds": 300,
            "max_invalid_replies": 1, "max_same_action": 1,
            "history_limit": 3, "retrieval_limit": 5, "max_map_images": 2,
        },
        "models": {
            "qwen": {
                "python": "/home/sasha/miniconda3/envs/change-agentic-rag/bin/python",
                "model_id": "Qwen/Qwen3.5-4B", "revision": QWEN_REVISION,
                "device": "cuda", "local_files_only": True,
                "precision": "bfloat16", "attention": "sdpa",
                "enable_thinking": False, "do_sample": False,
                "max_images": 8, "max_input_tokens": 4096, "max_new_tokens": 1024,
                "image_min_pixels": 65536, "image_max_pixels": 262144,
            },
            "sam3": {
                "python": "/home/sasha/miniconda3/envs/sam3_facades_20260907/bin/python",
                "revision": SAM3_REVISION,
                "checkpoint_path": "/home/sasha/Facade_Change_Benchmark/models/sam3.pt",
                "checkpoint_bytes": 3450062241, "device": "cuda",
            },
            "jina": {
                "python": "/home/sasha/miniconda3/envs/change-agentic-rag/bin/python",
                "model_id": "jinaai/jina-clip-v2", "revision": JINA_REVISION,
                "local_files_only": True, "trust_remote_code": True,
                "device": "cuda",
            },
        },
        "availability_policy": copy.deepcopy(AVAILABILITY_POLICY),
    }


def _absolute(value, parent, interpreter=False):
    if not isinstance(value, str) or not value:
        raise ValueError("Configured paths must be nonempty strings")
    path = Path(value).expanduser()
    if interpreter:
        return os.path.abspath(parent / path if not path.is_absolute() else path)
    return str((parent / path).resolve() if not path.is_absolute() else path.resolve())


def load_config(path=None):
    config = defaults()
    if path is not None:
        path = Path(path).expanduser().resolve()
        supplied = read_json(path)
        if not isinstance(supplied, dict) or set(supplied) - set(config):
            raise ValueError("Unknown configuration fields")
        config.update(copy.deepcopy(supplied))
        parent = path.parent
    else:
        parent = ROOT
    if config["schema_version"] != 1:
        raise ValueError("Unsupported configuration schema")
    for key in ("store_root", "knowledge_root", "cache_root"):
        config[key] = _absolute(config[key], parent)
    if config["gpu_mode"] not in {"sequential", "resident"}:
        raise ValueError("gpu_mode must be sequential or resident")
    if config["retrieval_encoder"] not in {None, "jina", "dino"}:
        raise ValueError("retrieval_encoder must be null, jina or dino")
    if type(config["max_region_pixels"]) is not int or config["max_region_pixels"] < 1:
        raise ValueError("max_region_pixels must be a positive integer")
    if not isinstance(config["modes"], list) or len(set(config["modes"])) != len(config["modes"]):
        raise ValueError("modes must be a list of distinct mode names")
    if set(config["modes"]) - set(MODES):
        raise ValueError("Unknown VLM mode")
    methods = config["methods"]
    if (not isinstance(methods, list) or not methods or len(set(methods)) != len(methods)
            or set(methods) - set(ALL_METHODS) or set(methods) & EXCLUDED_METHODS):
        raise ValueError("Select distinct existing methods; GeoSCD/RSCD are outside this integration")
    if set(methods) == {"rgb_diff"}:
        raise ValueError("RGB difference alone is not a change detection baseline")
    if config["alignment"].get("download_weights"):
        raise ValueError("This workflow never downloads model weights")
    if config["alignment"].get("worker_python"):
        config["alignment"]["worker_python"] = _absolute(config["alignment"]["worker_python"], parent, interpreter=True)
    for role, model in config["models"].items():
        if role not in {"qwen", "sam3", "jina", "dino"} or not isinstance(model, dict):
            raise ValueError("Unknown model role or invalid model configuration")
        if model.get("local_files_only", True) is not True:
            raise ValueError("Model loading must use local_files_only=True")
        for key in ("python", "checkpoint_path", "source_root"):
            if model.get(key):
                model[key] = _absolute(model[key], parent, interpreter=key == "python")
    if config["fusion"] is not None and config["fusion"].get("run"):
        config["fusion"]["run"] = _absolute(config["fusion"]["run"], parent)
    if config["availability_policy"] != AVAILABILITY_POLICY:
        raise ValueError("Changing availability policy requires a versioned implementation")
    return config


def import_method_options(config, path, selected=None, semantic_run=None):
    """Read the existing methods JSON. Never invent paths to weights or data."""
    result = copy.deepcopy(config)
    if path:
        source = Path(path).expanduser().resolve()
        document = read_json(source)
        options = document.get("methods")
        if not isinstance(options, dict):
            raise ValueError("Expected the existing methods configuration: {'methods': {...}}")
        methods = selected or [name for name in options if name not in EXCLUDED_METHODS]
        result["methods"] = list(methods)
        for name in methods:
            if name not in options and name not in {"ssim", "rgb_diff", "lzma_abs", "lzma_mod256"}:
                raise ValueError(f"Missing existing method configuration: {name}")
            value = copy.deepcopy(options.get(name, {}))
            for key in ("worker_python", "source_root", "checkpoint_path", "dino_root", "dino_checkpoint",
                        "training_run", "backbone_checkpoint", "calibration_checkpoint", "model_config"):
                if value.get(key):
                    value[key] = _absolute(value[key], source.parent, interpreter=key == "worker_python")
            result["method_options"][name] = value
    elif selected:
        result["methods"] = list(selected)
    if semantic_run:
        result["method_options"].setdefault("semantic_dino", {}).update(
            training_run=str(Path(semantic_run).expanduser().resolve()), export_probabilities=True)
        if "semantic_dino" not in result["methods"]:
            result["methods"].append("semantic_dino")
    return result


def write_new_config(path, config):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return path
