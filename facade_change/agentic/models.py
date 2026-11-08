"""Local-only model workers and bounded GPU ownership for the change assistant.

No torch/transformers import happens in the parent process. A worker owns its
model and allocator; terminating that worker really releases its live tensors.
The caller owns the complete CPU conversation, so reopening Qwen replays the
same evidence instead of attempting to transfer a KV cache.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid

from ..io import read_json, sha256, write_json


QWEN_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SAM3_REVISION = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"
JINA_REVISION = "e10d47f5691d0454a0fb5d13f46f2199b74cb436"
JINA_CODE_REVISION = "39e6a55ae971b59bea6e44675d237c99762e7ee2"
JINA_TEXT_CONFIG_REVISION = "ab036b023d30b4d1138c4c3bfa9f0c445ab455d6"
JINA_TEXT_CODE_REVISION = "bd55a5ec8e6c0fb1d6c26efb4b6a4a74ce8a88d3"

DEFAULTS = {
    "qwen": {"model_id": "Qwen/Qwen3.5-4B", "revision": QWEN_REVISION,
             "device": "cuda", "precision": "bfloat16", "attention": "sdpa",
             "max_images": 4, "max_image_pixels": 1048576,
             "image_min_pixels": 65536, "image_max_pixels": 262144,
             "max_input_tokens": 8192, "max_new_tokens": 512,
             "enable_thinking": False, "seed": 0, "do_sample": False, "temperature": 0.7, "top_p": 0.8},
    "sam3": {"revision": SAM3_REVISION, "device": "cuda",
             "checkpoint_bytes": 3450062241, "resolution": 1008,
             "confidence_threshold": 0.5, "max_image_pixels": 16777216,
             "strict_source": True},
    "jina": {"model_id": "jinaai/jina-clip-v2", "revision": JINA_REVISION,
             "code_revision": JINA_CODE_REVISION, "device": "cpu",
             "text_config_revision": JINA_TEXT_CONFIG_REVISION,
             "text_code_revision": JINA_TEXT_CODE_REVISION,
             "precision": "float32", "max_images": 16, "max_texts": 32,
             "max_image_pixels": 16777216, "max_text_chars": 8192,
             "truncate_dim": 1024, "trust_remote_code": True},
    "dino": {"device": "cpu", "trust_checkpoint": False,
             "max_images": 16, "max_image_pixels": 16777216},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _timed_sha256(path, deadline=None):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise WorkerError("timeout", "Deadline exceeded while verifying model cache/input hashes")
            block = stream.read(1024 * 1024)
            if not block:
                return checksum.hexdigest()
            checksum.update(block)


def _identity_files(value, deadline=None):
    """Hash model-visible files, never an arbitrary benchmark archive."""
    found = {}
    def visit(node):
        if isinstance(node, list):
            for child in node:
                visit(child)
        elif isinstance(node, dict):
            for key, child in node.items():
                if key in {"image_path", "path"} and isinstance(child, str) and (
                        key == "image_path" or node.get("type") == "image"):
                    path = Path(child).expanduser().resolve()
                    found[str(path)] = _timed_sha256(path, deadline)
                elif key in {"image_paths", "support_paths"}:
                    if not isinstance(child, list):
                        raise ValueError(f"{key} must be a list")
                    for item in child:
                        if item is None and key == "support_paths":
                            continue
                        path = Path(item).expanduser().resolve()
                        found[str(path)] = _timed_sha256(path, deadline)
                else:
                    visit(child)
    visit(value)
    return found


def _resources_valid(resources, deadline=None):
    for row in resources:
        path = Path(row["path"])
        if not path.is_file() or _timed_sha256(path, deadline) != row["sha256"]:
            return False
    return True


class WorkerError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class _Worker:
    def __init__(self, role, options, log_dir, deadline):
        executable = Path(options.get("python", sys.executable)).expanduser()
        # Do not resolve the interpreter symlink: that would escape a venv.
        if not executable.is_absolute() or not executable.is_file():
            raise WorkerError("missing_interpreter", f"Missing absolute worker interpreter: {executable}")
        self.role, self.sequence, self.process = role, 0, None
        self.responses = queue.Queue()
        root = Path(__file__).resolve().parents[2]
        self.log_path = Path(log_dir) / f"{role}-{time.time_ns()}.log"
        self.log = self.log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env.update(PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HF_DATASETS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
        env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        try:
            self.process = subprocess.Popen(
                [str(executable), "-s", "-u", "-B", "-m", "facade_change.agentic.model_worker"],
                cwd=str(root), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.log, text=True, encoding="utf-8", bufsize=1)
            self.reader = threading.Thread(target=self._read, daemon=True)
            self.reader.start()
            response = self.request({"op": "init", "role": role, "options": options}, deadline)
            if response.get("status") != "ok":
                error = response.get("error", {})
                raise WorkerError(error.get("code", "initialization_error"), error.get("message", "Worker initialization failed"))
            self.provenance = response.get("provenance", {})
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            for line in self.process.stdout:
                self.responses.put(line)
        finally:
            self.responses.put(None)

    def request(self, payload, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WorkerError("timeout", "Model call deadline exhausted")
        self.sequence += 1
        identifier = self.sequence
        try:
            self.process.stdin.write(json.dumps({"id": identifier, **payload}, allow_nan=False) + "\n")
            self.process.stdin.flush()
            line = self.responses.get(timeout=remaining)
        except queue.Empty as exc:
            raise WorkerError("timeout", f"{self.role} exceeded call deadline; log: {self.log_path}") from exc
        except (BrokenPipeError, OSError) as exc:
            raise WorkerError("worker_exit", f"{self.role} closed its pipe; log: {self.log_path}") from exc
        if line is None:
            raise WorkerError("worker_exit", f"{self.role} exited; log: {self.log_path}")
        try:
            reply = json.loads(line)
            if not isinstance(reply, dict) or reply.pop("id", None) != identifier:
                raise ValueError("Mismatching worker response ID")
        except (ValueError, TypeError) as exc:
            raise WorkerError("protocol_error", f"Invalid worker response; log: {self.log_path}") from exc
        return reply

    def close(self):
        process = self.process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            for handle in (process.stdin, process.stdout):
                if handle:
                    handle.close()
        if getattr(self, "reader", None):
            self.reader.join(timeout=1)
        if getattr(self, "log", None):
            self.log.close()
        self.process = None


class ModelPool:
    """Small role-specific supervisor; no arbitrary commands or model tools.

    Successful cache entries are immutable and hash-checked. A cache hit records
    zero current inference cost and preserves original cost under source_usage.
    Full messages, images, geometry and processing options bind the key. There
    is at most one OOM retry, using exactly the same payload/configuration.
    """
    def __init__(self, models, log_dir, gpu_mode="sequential", timeout=300, cache_dir=None):
        if gpu_mode not in {"sequential", "resident"}:
            raise ValueError("gpu_mode must be sequential or resident")
        if not isinstance(models, dict) or set(models) - set(DEFAULTS):
            raise ValueError("Only qwen, sam3, jina and dino model roles are allowed")
        self.models = {role: {**DEFAULTS[role], **dict(options)} for role, options in models.items()}
        self.gpu_mode, self.timeout = gpu_mode, positive_number(timeout, "timeout")
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.workers, self.events, self._worker_options = {}, [], {}
        self._pool_id = uuid.uuid4().hex
        self._lock = threading.Lock()

    def _event(self, **values):
        """Persist owned-process lifecycle alongside the run's worker logs."""
        record = {"schema_version": 1, "pool_id": self._pool_id,
                  "event_id": f"{self._pool_id}:{len(self.events) + 1}",
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(), **values}
        self.events.append(record)
        with (self.log_dir / "pool.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return record

    def _key(self, role, payload, deadline=None):
        options = self.models[role]
        resources = {}
        for name in ("checkpoint_path",):
            if options.get(name) and Path(options[name]).is_file():
                resources[name] = _timed_sha256(options[name], deadline)
        semantic = {k: v for k, v in payload.items() if k not in {"out", "timeout_seconds"}}
        source_hashes = {p.name: _timed_sha256(p, deadline) for p in (Path(__file__), Path(__file__).with_name("model_worker.py"))}
        return digest({"schema": 1, "role": role, "options": options, "request": semantic,
                       "inputs": _identity_files(payload, deadline), "weights": resources, "implementation": source_hashes})

    def _cached(self, key, deadline=None):
        path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if path is None or not path.exists():
            return None
        entry = read_json(path)
        if entry.get("key") != key or digest(entry.get("response")) != entry.get("response_sha256"):
            raise ValueError("Corrupted model response cache")
        result = entry["response"]
        if not _resources_valid(result.get("artifacts", []), deadline):
            return None
        if not _resources_valid(result.get("provenance", {}).get("resources", []), deadline):
            return None
        return copy.deepcopy(result)

    def _store(self, key, response):
        if not self.cache_dir:
            return
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            # Reuse never overwrites an old inference or repairs a corrupt entry.
            return
        write_json(path, {"key": key, "response_sha256": digest(response), "response": response})

    def _release(self, role, reason):
        worker = self.workers.pop(role, None)
        self._worker_options.pop(role, None)
        if worker is not None:
            pid = getattr(getattr(worker, "process", None), "pid", None)
            worker.close()
            self._event(event="worker_released", role=role, reason=reason, pid=pid)

    def _activate(self, role, deadline):
        keep = {role}
        if self.gpu_mode == "resident" and role in {"qwen", "sam3"}:
            keep = {"qwen", "sam3"}
        for current in list(self.workers):
            if current not in keep:
                self._release(current, "gpu_handoff")
        if role in self.workers and (
                self._worker_options.get(role) != digest(self.models[role])
                or not _resources_valid(self.workers[role].provenance.get("resources", []), deadline)):
            self._release(role, "model_configuration_or_resources_changed")
        if role not in self.workers:
            self.workers[role] = _Worker(role, self.models[role], self.log_dir, deadline)
            self._worker_options[role] = digest(self.models[role])
            self._event(event="worker_started", role=role, reason="activate",
                        pid=getattr(getattr(self.workers[role], "process", None), "pid", None),
                        log_path=str(self.workers[role].log_path))
            return self.workers[role], 1
        return self.workers[role], 0

    def call(self, role, payload):
        with self._lock:
            return self._call(role, copy.deepcopy(payload))

    def _call(self, role, payload):
        started = time.monotonic()
        if role not in self.models:
            return {"status": "error", "error": {"code": "not_configured", "message": f"Model role is not configured: {role}"},
                    "usage": {"elapsed_seconds": 0., "model_loads": 0, "cache_hits": 0,
                              "peak_allocated_bytes": None, "peak_reserved_bytes": None, "min_free_bytes": None}}
        if not isinstance(payload, dict):
            raise ValueError("Model request must be an object")
        deadline = started + min(self.timeout, positive_number(payload.get("timeout_seconds", self.timeout), "timeout_seconds"))
        totals = {"input_tokens": 0, "output_tokens": 0, "images": 0, "model_loads": 0,
                  "worker_starts": 0, "retries": 0, "cache_hits": 0, "model_calls": 0,
                  "peak_allocated_bytes": None, "peak_reserved_bytes": None}
        minimum_free = None
        attempts = []
        key = None
        try:
            key = self._key(role, payload, deadline)
            cached = None if payload.get("op") == "check" else self._cached(key, deadline)
            if time.monotonic() >= deadline:
                raise WorkerError("timeout", "Model call deadline exhausted during cache validation")
            if cached is not None:
                original = cached.pop("usage", {})
                return {**cached, "cache_key": key, "cache_hit": True, "source_usage": original,
                        "usage": {**totals, "cache_hits": 1, "peak_allocated_bytes": 0,
                                  "peak_reserved_bytes": 0, "min_free_bytes": None,
                                  "elapsed_seconds": time.monotonic() - started}}
            for attempt in range(2):
                worker, created = self._activate(role, deadline)
                totals["worker_starts"] += created
                request = {**payload, "timeout_seconds": max(.001, deadline - time.monotonic())}
                reply = worker.request(request, deadline)
                worker.provenance = reply.get("provenance", worker.provenance)
                usage = reply.get("usage", {})
                for name in totals:
                    if name not in {"retries", "worker_starts", "cache_hits", "peak_allocated_bytes", "peak_reserved_bytes"}:
                        totals[name] += usage.get(name, 0)
                memory = usage.get("memory", {})
                for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
                    measured = memory.get(name, usage.get(name))
                    if measured is not None:
                        totals[name] = measured if totals[name] is None else max(totals[name], measured)
                if memory.get("free_bytes") is not None:
                    minimum_free = memory["free_bytes"] if minimum_free is None else min(minimum_free, memory["free_bytes"])
                attempts.append({"attempt": attempt + 1, "status": reply.get("status"),
                                 "error": reply.get("error"), "usage": usage,
                                 "provenance": reply.get("provenance", {})})
                if reply.get("status") != "error" or reply.get("error", {}).get("code") != "out_of_memory":
                    break
                # Drop only application-owned processes. A second failure is terminal.
                for current in list(self.workers):
                    self._release(current, "oom_recovery")
                if attempt == 1:
                    break
                totals["retries"] += 1
            result = {**reply, "cache_key": key, "cache_hit": False,
                      "usage": {**totals, "elapsed_seconds": time.monotonic() - started,
                                "min_free_bytes": minimum_free, "attempts": attempts}}
            if result.get("status") in {"ok", "empty"} and payload.get("op") != "check":
                self._store(key, result)
            elif result.get("status") == "error":
                self._release(role, "model_request_failed")
            return result
        except (WorkerError, OSError, ValueError, KeyError, TypeError) as exc:
            self._release(role, "request_failed")
            return {"status": "error", "error": {"code": getattr(exc, "code", "validation_error"), "message": str(exc)},
                    "cache_key": key, "cache_hit": False,
                    "usage": {**totals, "elapsed_seconds": time.monotonic() - started, "attempts": attempts,
                              "min_free_bytes": minimum_free,
                              "incomplete": isinstance(exc, WorkerError),
                              "unobserved_worker_cost": isinstance(exc, WorkerError)}}

    def close(self):
        for role in list(self.workers):
            self._release(role, "pool_closed")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
