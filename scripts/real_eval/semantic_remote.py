"""Sequential semantic model workers using the existing RemotePool transport."""
from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from facade_change.methods.remote import RemotePool, _json_value
from scripts.real_eval.semantic_worker import EXTRA_PREFIX, validate_prediction, validate_rgb


def _worker_launch(config):
    executable = Path(config.get("worker_python", sys.executable)).expanduser()
    if not executable.is_absolute() or not executable.is_file():
        raise ValueError(f"worker_python must be an existing absolute Python executable: {executable}")
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    command = [str(executable), "-s", "-u", "-B", "-m", "scripts.real_eval.semantic_worker"]
    return executable, root, env, command


def preflight_remote(config, timeout=120):
    """Hash local assets and versions in the selected CPU-only interpreter.

    This does not construct a segmenter or import Torch/CUDA. Adapter failures
    return ``ok: false``; interpreter/protocol failures raise with their stderr.
    """
    config = dict(config)
    if not math.isfinite(float(timeout)) or timeout <= 0:
        raise ValueError("Preflight timeout must be finite and positive")
    executable, root, env, command = _worker_launch(config)
    try:
        process = subprocess.run(
            command + ["--preflight"], cwd=str(root), env=env,
            input=json.dumps(config, default=_json_value, allow_nan=False),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"Semantic preflight timed out after {timeout:g}s: {executable}") from error
    if process.returncode:
        raise RuntimeError(f"Semantic preflight failed in {executable}: {process.stderr.strip()}")
    try:
        result = json.loads(process.stdout)
        if (not isinstance(result, dict) or not isinstance(result.get("ok"), bool)
                or not isinstance(result.get("config"), dict)
                or not isinstance(result.get("environment", {}).get("packages"), dict)
                or not result.get("environment", {}).get("executable")):
            raise ValueError("Missing preflight contract fields")
    except (ValueError, AttributeError) as error:
        raise RuntimeError(f"Semantic preflight returned invalid JSON in {executable}: "
                           f"{process.stderr.strip()}") from error
    result["config"]["worker_python"] = str(executable)
    return result


class SemanticPool(RemotePool):
    """Keep one semantic model/environment alive until its configuration changes."""

    def activate(self, config, vocabulary):
        config = dict(config)
        executable, root, env, command = _worker_launch(config)
        config.pop("worker_python", None)
        timeout = float(config.pop("worker_timeout", self.timeout))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("worker_timeout must be finite and positive")
        key = json.dumps([str(executable), config, vocabulary], default=_json_value,
                         sort_keys=True, allow_nan=False)
        if self.process is not None and self.process.poll() is None and self._key == key:
            self._active_timeout = timeout
            return self.metadata
        self.release()
        handle = tempfile.NamedTemporaryFile(prefix="facade-semantic-worker-", suffix=".log",
                                             dir=str(self.log_dir) if self.log_dir else None,
                                             delete=False, mode="w", encoding="utf-8")
        self.stderr_path, self._stderr = Path(handle.name), handle
        self._active_timeout = timeout
        started = time.perf_counter()
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(root), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=handle, text=True, encoding="utf-8", bufsize=1,
            )
            self._responses = queue.Queue()
            self._reader = threading.Thread(target=self._read_lines,
                                            args=(self.process, self._responses), daemon=True)
            self._reader.start()
            self._key = key
            response = self._request("init", config=config, vocabulary=vocabulary)
            if not isinstance(response.get("metadata"), dict):
                raise self._error("returned invalid initialization metadata")
            self.metadata = {**response["metadata"],
                             "activation_seconds": time.perf_counter() - started,
                             "worker_python": str(executable),
                             "worker_stderr_log": str(self.stderr_path)}
            return self.metadata
        except BaseException:
            self._dispose()
            raise

    def predict(self, config, vocabulary, rgb):
        rgb = validate_rgb(rgb)
        self.activate(config, vocabulary)
        started = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="facade-semantic-rgb-") as folder:
                source, target = Path(folder) / "input.npz", Path(folder) / "output.npz"
                np.savez(source, rgb=rgb)
                response = self._request("predict", input_path=str(source), output_path=str(target))
                if not target.is_file():
                    raise ValueError("Worker did not save the requested semantic output")
                with np.load(target, allow_pickle=False) as archive:
                    if len(archive.files) != len(set(archive.files)):
                        raise ValueError("Worker returned duplicate NPZ arrays")
                    arrays = {name: archive[name].copy() for name in archive.files}
            if "scores" not in arrays or any(name != "scores" and not name.startswith(EXTRA_PREFIX)
                                              for name in arrays):
                raise ValueError("Worker returned unexpected semantic arrays")
            elapsed = response.get("elapsed_seconds")
            if (isinstance(elapsed, bool) or not isinstance(elapsed, (float, int))
                    or not math.isfinite(elapsed) or elapsed < 0):
                raise ValueError("Worker returned invalid inference timing")
            if not isinstance(response.get("metadata"), dict):
                raise ValueError("Worker returned invalid semantic metadata")
            metadata = {**self.metadata, **response["metadata"]}
            result = {"scores": arrays["scores"], "class_names": response.get("class_names"),
                      "score_kind": response.get("score_kind"),
                      "extra_arrays": {name[len(EXTRA_PREFIX):]: value
                                       for name, value in arrays.items() if name != "scores"},
                      "metadata": metadata, "elapsed_seconds": float(elapsed),
                      "request_elapsed_seconds": time.perf_counter() - started}
            validate_prediction(result, rgb.shape[:2])
            self.metadata = metadata
            return result
        except BaseException:
            self._dispose()
            raise
