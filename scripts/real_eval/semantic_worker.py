"""Isolated semantic inference protocol: native uint8 RGB in, class maps out.

Model imports happen only after init. No GT, labels or support archive fields
are accepted; geometric support remains in the parent evaluator.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from facade_change.methods.remote import _json_value
from scripts.real_eval.robustness_common import atomic_npz, sha256


EXTRA_PREFIX = "extra__"


def validate_rgb(rgb):
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
            or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1):
        raise ValueError("Semantic inference requires native uint8 H,W,3 RGB")
    return rgb


def validate_prediction(result, shape):
    if not isinstance(result, dict):
        raise ValueError("Semantic adapter must return a dictionary")
    names = result.get("class_names")
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(names) != len(set(names))):
        raise ValueError("Semantic output needs distinct nonempty class_names")
    if not isinstance(result.get("score_kind"), str) or not result["score_kind"]:
        raise ValueError("Semantic output needs an explicit score_kind")
    scores = result.get("scores")
    if (not isinstance(scores, np.ndarray) or scores.dtype != np.float32
            or scores.shape != (len(names), *shape)
            or not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1)):
        raise ValueError("Semantic scores must be finite native-grid float32 C,H,W in [0,1]")
    extras = result.get("extra_arrays", {})
    if not isinstance(extras, dict):
        raise ValueError("Semantic extra_arrays must be a dictionary")
    for name, value in extras.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError("Invalid semantic extra array name")
        if (not isinstance(value, np.ndarray) or value.dtype.kind not in "biufc"
                or value.dtype.hasobject or not np.isfinite(value).all()):
            raise ValueError(f"Semantic extra array {name} must be finite numeric/boolean, no pickle")
    return result


def _make_segmenter(config, vocabulary):
    from scripts.real_eval.semantic_adapters import make_segmenter
    return make_segmenter(config, vocabulary)


def _synchronize():
    # Do not import Torch or initialize CUDA in a CPU-only worker.
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def _environment():
    packages = {}
    for name in ("numpy", "Pillow", "torch", "torchvision", "sam3", "mmcv", "mmcv-full",
                 "mmsegmentation", "open-clip-torch", "timm", "omegaconf", "hydra-core",
                 "scipy", "cupy", "cupy-cuda11x", "cupy-cuda12x", "faiss-cpu", "faiss-gpu"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "executable": sys.executable,
            "packages": packages, "worker_source_sha256": sha256(__file__)}


def preflight_once(requests, responses):
    """Resolve assets/package source in this interpreter without model imports."""
    from scripts.real_eval.semantic_adapters import preflight
    config = json.load(requests)
    if not isinstance(config, dict):
        raise ValueError("Preflight config must be a JSON object")
    result = preflight(config)
    if result["config"].get("kind") == "sam3" and not result.get("sources", {}).get("sam3"):
        result["ok"] = False
        result["errors"].append("SAM3 source package unavailable in selected worker environment")
    result["environment"] = _environment()
    responses.write(json.dumps(result, default=_json_value, allow_nan=False) + "\n")
    responses.flush()


def serve(requests, responses):
    """Serve explicit streams; tests patch only the adapter factory."""
    segmenter, worker_environment, init_seconds = None, None, None
    try:
        for line in requests:
            request_id, close = None, False
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("Worker request must be an object")
                request_id = request.get("id")
                operation = request.get("operation")
                if operation == "init":
                    if set(request) != {"id", "operation", "config", "vocabulary"}:
                        raise ValueError("Unexpected semantic init fields")
                    if segmenter is not None:
                        raise ValueError("Worker already initialized; close before changing model")
                    started = time.perf_counter()
                    segmenter = _make_segmenter(request["config"], request["vocabulary"])
                    _synchronize()
                    init_seconds = time.perf_counter() - started
                    worker_environment = _environment()
                    payload = {"metadata": {**segmenter.metadata,
                                            "worker_environment": worker_environment,
                                            "init_seconds": init_seconds}}
                elif operation == "predict":
                    if set(request) != {"id", "operation", "input_path", "output_path"}:
                        raise ValueError("Predict accepts only RGB archive paths")
                    if segmenter is None:
                        raise ValueError("Worker is not initialized")
                    with np.load(request["input_path"], allow_pickle=False) as archive:
                        if archive.files != ["rgb"]:
                            raise ValueError("Semantic input archive must contain only rgb")
                        rgb = validate_rgb(archive["rgb"].copy())
                    _synchronize()
                    started = time.perf_counter()
                    result = segmenter.predict(rgb)
                    _synchronize()
                    elapsed = time.perf_counter() - started
                    validate_prediction(result, rgb.shape[:2])
                    arrays = {"scores": result["scores"],
                              **{EXTRA_PREFIX + name: value
                                 for name, value in result.get("extra_arrays", {}).items()}}
                    atomic_npz(Path(request["output_path"]), arrays)
                    metadata = {**segmenter.metadata, **result.get("metadata", {}),
                                "worker_environment": worker_environment,
                                "init_seconds": init_seconds}
                    if "softmax_scale" in result:
                        metadata["softmax_scale"] = result["softmax_scale"]
                    payload = {"class_names": result["class_names"],
                               "score_kind": result["score_kind"], "metadata": metadata,
                               "elapsed_seconds": elapsed}
                elif operation == "close":
                    if set(request) != {"id", "operation"}:
                        raise ValueError("Unexpected close fields")
                    if segmenter is not None:
                        segmenter.close()
                        segmenter = None
                    payload, close = {}, True
                else:
                    raise ValueError(f"Unknown worker operation: {operation}")
                encoded = json.dumps({"id": request_id, "ok": True, **payload},
                                     default=_json_value, allow_nan=False)
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                encoded = json.dumps({"id": request_id, "ok": False,
                                      "error": f"{type(error).__name__}: {error}"})
            responses.write(encoded + "\n")
            responses.flush()
            if close:
                break
    finally:
        if segmenter is not None:
            segmenter.close()


def main():
    # Both Python prints and native model logs must stay off the JSON control pipe.
    control = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    try:
        if sys.argv[1:] == ["--preflight"]:
            preflight_once(sys.stdin, control)
        elif sys.argv[1:]:
            raise ValueError("Only --preflight is accepted by semantic worker")
        else:
            serve(sys.stdin, control)
    finally:
        control.close()


if __name__ == "__main__":
    main()
