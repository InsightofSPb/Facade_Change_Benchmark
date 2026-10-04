"""One persistent, isolated inference worker at a time; no labels cross this API."""
from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

from .base import validate_rgb_inputs


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Worker configuration is not JSON serializable: {type(value).__name__}")


class RemotePool:
    """Keep only the currently selected method's process/model alive."""

    def __init__(self, log_dir=None, timeout=3600):
        self.log_dir = Path(log_dir).expanduser().resolve() if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = float(timeout)
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Worker timeout must be finite and positive")
        self.process = None
        self.metadata = None
        self.stderr_path = None
        self._key = None
        self._responses = None
        self._reader = None
        self._stderr = None
        self._request_id = 0
        self._active_timeout = self.timeout

    def _error(self, message):
        return RuntimeError(f"Inference worker {message}; stderr log: {self.stderr_path}")

    def _dispose(self):
        process, self.process = self.process, None
        self._key = None
        if process is not None:
            if process.poll() is None:
                process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            for pipe in (process.stdin, process.stdout):
                if pipe:
                    pipe.close()
        if self._reader is not None:
            self._reader.join(timeout=1)
            self._reader = None
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None
        self._responses = None

    @staticmethod
    def _read_lines(process, responses):
        try:
            for line in process.stdout:
                responses.put(line)
        finally:
            responses.put(None)

    def _request(self, operation, **payload):
        if self.process is None:
            raise self._error("is not active")
        self._request_id += 1
        request_id = self._request_id
        message = {"id": request_id, "operation": operation, **payload}
        try:
            self.process.stdin.write(json.dumps(message, default=_json_value, allow_nan=False) + "\n")
            self.process.stdin.flush()
        except (OSError, BrokenPipeError) as exc:
            self._dispose()
            raise self._error("closed its input unexpectedly") from exc
        try:
            line = self._responses.get(timeout=self._active_timeout)
        except queue.Empty as exc:
            self._dispose()
            raise self._error(f"timed out after {self._active_timeout:g}s during {operation}") from exc
        if line is None:
            self._dispose()
            raise self._error(f"exited without a response during {operation}")
        try:
            response = json.loads(line)
            if not isinstance(response, dict) or response.get("id") != request_id:
                raise ValueError("response id does not match request")
        except (json.JSONDecodeError, ValueError) as exc:
            self._dispose()
            raise self._error(f"returned invalid JSON protocol during {operation}") from exc
        if not response.get("ok"):
            self._dispose()
            raise self._error(f"failed during {operation}: {response.get('error', 'unknown error')}")
        return response

    def activate(self, method, options):
        options = dict(options)
        executable = Path(options.pop("worker_python", sys.executable)).expanduser()
        if not executable.is_absolute() or not executable.is_file():
            raise ValueError(f"worker_python must be an existing absolute Python executable: {executable}")
        timeout = float(options.pop("worker_timeout", self.timeout))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("worker_timeout must be finite and positive")
        key = json.dumps([method, str(executable), options], default=_json_value,
                         sort_keys=True, allow_nan=False)
        if self.process is not None and self._key == key:
            return self.metadata
        self.release()
        root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        # Add this checkout explicitly; no installation or activation in the parent.
        env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        handle = tempfile.NamedTemporaryFile(prefix=f"facade-worker-{method}-", suffix=".log",
                                             dir=str(self.log_dir) if self.log_dir else None,
                                             delete=False, mode="w", encoding="utf-8")
        self.stderr_path, self._stderr = Path(handle.name), handle
        self._active_timeout = timeout
        try:
            self.process = subprocess.Popen(
                [str(executable), "-s", "-u", "-B", "-m", "facade_change.method_worker"],
                cwd=str(root), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=handle, text=True, encoding="utf-8", bufsize=1,
            )
            self._responses = queue.Queue()
            self._reader = threading.Thread(target=self._read_lines,
                                            args=(self.process, self._responses), daemon=True)
            self._reader.start()
            self._key = key
            response = self._request("init", method=method, options=options)
            self.metadata = response.get("metadata", {})
            return self.metadata
        except BaseException:
            self._dispose()
            raise

    def score(self, method, options, reference, source, support):
        reference, source, support = validate_rgb_inputs(reference, source, support)
        self.activate(method, options)
        with tempfile.TemporaryDirectory(prefix="facade-worker-rgb-") as folder:
            request_path, response_path = Path(folder) / "input.npz", Path(folder) / "output.npz"
            np.savez(request_path, reference=reference, source=source, support=support)
            response = self._request("score", input_path=str(request_path), output_path=str(response_path))
            if not response_path.is_file():
                self._dispose()
                raise self._error("did not save the requested inference output")
            with np.load(response_path, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
        if "scores" not in arrays or set(arrays) - {"scores", "raw_scores", "native_prediction"}:
            raise self._error("returned unexpected output arrays")
        for name, array in arrays.items():
            if array.shape != support.shape:
                raise self._error(f"returned {name} with a different native grid")
            expected_dtype = np.bool_ if name == "native_prediction" else np.float32
            if array.dtype != expected_dtype:
                raise self._error(f"returned invalid {name} dtype")
        if not np.isfinite(arrays["scores"][support]).all():
            raise self._error("returned nonfinite scores inside the base support")
        if not np.isnan(arrays["scores"][~support]).all():
            raise self._error("returned scores outside the base support")
        self.metadata = response.get("metadata", self.metadata)
        return {**arrays, "metadata": self.metadata}

    def release(self):
        if self.process is not None:
            try:
                # Cleanup should not wait the full inference timeout.
                previous = self._active_timeout
                self._active_timeout = min(previous, 10)
                self._request("close")
                self.process.wait(timeout=5)
            except (RuntimeError, OSError, subprocess.TimeoutExpired):
                pass
            finally:
                self._dispose()

    close = release

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class RemoteScorer:
    """Serializable scorer facade; initialization happens in the selected worker."""

    def __init__(self, method, options=None, pool=None, **extra_options):
        self.method = method
        self.options = {**(options or {}), **extra_options}
        self.pool = pool if pool is not None else RemotePool()
        self.raw_scores = None
        self.native_prediction = None
        kind = "native_mask" if method in {"anychange", "geoscd"} else "score"
        self.metadata = {
            "method": method, "output_kind": kind, "initialization": "pending in selected worker",
            "configured_options": json.loads(json.dumps(self.options, default=_json_value)),
            "worker_python": str(self.options.get("worker_python", sys.executable)),
            "input_contract": "native uint8 RGB reference/source and boolean base support only; no labels",
        }

    def activate(self):
        metadata = self.pool.activate(self.method, self.options)
        self.metadata = {**metadata, "worker_python": str(self.options.get("worker_python", sys.executable)),
                         "worker_stderr_log": str(self.pool.stderr_path)}
        return self.metadata

    def __call__(self, reference, source, support):
        result = self.pool.score(self.method, self.options, reference, source, support)
        self.raw_scores = result.get("raw_scores")
        self.native_prediction = result.get("native_prediction")
        self.metadata = {**result["metadata"],
                         "worker_python": str(self.options.get("worker_python", sys.executable)),
                         "worker_stderr_log": str(self.pool.stderr_path)}
        return result["scores"]

    def close(self):
        self.pool.release()
