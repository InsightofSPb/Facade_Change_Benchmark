"""JSON-line inference worker. Only RGB pairs/support enter; labels never enter."""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Worker metadata is not JSON serializable: {type(value).__name__}")


def serve(requests, responses):
    """Serve on explicit streams so tests can exercise protocol without models."""
    scorer = None
    worker_environment = None
    try:
        for line in requests:
            request_id = None
            close = False
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("Worker request must be an object")
                request_id = request.get("id")
                operation = request.get("operation")
                if operation == "init":
                    if set(request) - {"id", "operation", "method", "options"}:
                        raise ValueError("Unexpected init fields")
                    if scorer is not None:
                        raise ValueError("Worker already initialized; close before changing method")
                    from .methods.registry import make_method
                    scorer = make_method(request["method"], **request.get("options", {}))
                    from .io import environment
                    worker_environment = {**environment(), "executable": sys.executable}
                    payload = {"metadata": {**scorer.metadata, "worker_environment": worker_environment}}
                elif operation == "score":
                    if set(request) != {"id", "operation", "input_path", "output_path"}:
                        raise ValueError("Score requests accept only RGB/support archive paths")
                    if scorer is None:
                        raise ValueError("Worker is not initialized")
                    with np.load(request["input_path"], allow_pickle=False) as archive:
                        if set(archive.files) != {"reference", "source", "support"}:
                            raise ValueError("Inference input archive must contain only reference, source, support")
                        reference, source, support = (archive[name].copy() for name in
                                                      ("reference", "source", "support"))
                    from .methods.base import validate_rgb_inputs
                    reference, source, support = validate_rgb_inputs(reference, source, support)
                    scores = np.asarray(scorer(reference, source, support))
                    if scores.shape != support.shape or scores.dtype != np.float32:
                        raise ValueError("Scorer must return native-grid float32 scores")
                    arrays = {"scores": scores}
                    for name in ("raw_scores", "native_prediction"):
                        value = getattr(scorer, name, None)
                        if value is not None:
                            value = np.asarray(value)
                            dtype = np.bool_ if name == "native_prediction" else np.float32
                            if value.shape != support.shape or value.dtype != dtype:
                                raise ValueError(f"Scorer {name} must use the native grid and {dtype}")
                            arrays[name] = value
                    np.savez(request["output_path"], **arrays)
                    payload = {"metadata": {**scorer.metadata, "worker_environment": worker_environment}}
                elif operation == "close":
                    if scorer is not None:
                        if hasattr(scorer, "close"):
                            scorer.close()
                        scorer = None
                    payload, close = {}, True
                else:
                    raise ValueError(f"Unknown worker operation: {operation}")
                response = {"id": request_id, "ok": True, **payload}
                encoded = json.dumps(response, default=_json_value, allow_nan=False)
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                encoded = json.dumps({"id": request_id, "ok": False,
                                      "error": f"{type(exc).__name__}: {exc}"})
            responses.write(encoded + "\n")
            responses.flush()
            if close:
                break
    finally:
        if scorer is not None and hasattr(scorer, "close"):
            scorer.close()


def main():
    # Preserve one dedicated control pipe; redirect Python and native library
    # stdout to stderr before importing any model. Model logs cannot corrupt JSON.
    control = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    try:
        serve(sys.stdin, control)
    finally:
        control.close()


if __name__ == "__main__":
    main()
