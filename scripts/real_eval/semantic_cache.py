"""Durable per-computation cache for the semantic routing experiment.

The caller owns the task specification: include input RGB/support hashes, model
and source/checkpoint identity, vocabulary, preprocessing and output contract.
Do not put the run directory, requested method list or split list in that spec.
Evaluation/visualization tasks must also include their GT/metric/render versions.
An optional ``event=run.event`` writes to the existing research event ledger.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from facade_change.io import sync_directory
from scripts.real_eval.robustness_common import atomic_json, atomic_npz, digest, sha256


SCHEMA_VERSION = 1


def _json_copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _array(value):
    array = np.asarray(value)
    if array.dtype.kind not in "biufc" or array.dtype.hasobject:
        raise ValueError("Cache arrays must contain numeric or boolean values, never objects")
    return array


def hash_array(value):
    """Hash dtype, shape and C-order values, independent of memory strides."""
    array = _array(value)
    header = digest({"dtype": array.dtype.str, "shape": list(array.shape)})
    result = hashlib.sha256(header.encode("ascii"))
    result.update(np.ascontiguousarray(array).tobytes())
    return result.hexdigest()


def _arrays(values):
    if not isinstance(values, dict) or not values:
        raise ValueError("Cache output must be a nonempty dict of named arrays")
    if any(not isinstance(key, str) or not key or "/" in key or "\\" in key
           or key in {".", "..", "file"} for key in values):
        raise ValueError("Invalid cache array name")
    return {key: _array(value) for key, value in values.items()}


def _descriptions(arrays):
    return {name: {"dtype": value.dtype.str, "shape": list(value.shape),
                   "sha256": hash_array(value)} for name, value in arrays.items()}


def _utc():
    return datetime.now(timezone.utc).isoformat()


class Cache:
    """Atomic NPZ + JSON commits, with immutable quarantine for failed attempts.

    ``get`` returns ``(arrays, record)`` or ``None``. Only a committed, validated
    JSON marker can make an entry reusable. Corruption is retained in quarantine
    and treated as a miss. ``put`` preserves any existing valid result. Locks
    protect individual cache keys even when different run directories share it.
    """

    def __init__(self, root, event=None):
        self.root = Path(root).expanduser().resolve()
        self.event = event
        for name in ("objects", "records", "locks", "quarantine"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def key(self, spec):
        if not isinstance(spec, dict) or not spec:
            raise ValueError("A nonempty task specification is required")
        return digest({"schema_version": SCHEMA_VERSION, "spec": _json_copy(spec)})

    @contextmanager
    def _locked(self, key):
        import fcntl
        with (self.root / "locks" / (key + ".lock")).open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _paths(self, key):
        return self.root / "records" / (key + ".json"), self.root / "objects" / (key + ".npz")

    def _emit(self, kind, **values):
        if self.event is not None:
            return self.event(kind, **values)
        return None

    def _partials(self, key):
        return [partial for path in self._paths(key)
                for partial in path.parent.glob("." + path.name + "*")]

    def _quarantine(self, key, reason, status="invalidated"):
        target = self.root / "quarantine" / key / str(time.time_ns())
        target.mkdir(parents=True, exist_ok=False)
        marker, artifact = self._paths(key)
        for source, name in ((marker, "record.json"), (artifact, "artifact.npz")):
            if os.path.lexists(source):
                source.replace(target / name)
                sync_directory(source.parent)
        for partial in self._partials(key):
            partial.replace(target / partial.name)
            sync_directory(partial.parent)
        atomic_json(target / "attempt.json", {
            "schema_version": SCHEMA_VERSION, "cache_key": key,
            "status": status, "reason": str(reason), "timestamp_utc": _utc(),
        })
        sync_directory(target)
        self._emit("cache_" + status, cache_key=key, reason=str(reason),
                   attempt_path=str(target.relative_to(self.root)))

    def _read(self, key, spec):
        marker, artifact = self._paths(key)
        if not os.path.lexists(marker) and not os.path.lexists(artifact):
            if self._partials(key):
                self._quarantine(key, "Interrupted temporary artifact without commit")
            return None
        try:
            if marker.is_symlink() or artifact.is_symlink():
                raise ValueError("Cache artifacts cannot be symlinks")
            record = json.loads(marker.read_text(encoding="utf-8"))
            expected_path = str(artifact.relative_to(self.root))
            if (record.get("schema_version") != SCHEMA_VERSION
                    or record.get("cache_key") != key or record.get("spec") != spec
                    or record.get("artifact_path") != expected_path):
                raise ValueError("Cache commit does not match task identity")
            if sha256(artifact) != record.get("artifact_sha256"):
                raise ValueError("Cache artifact SHA-256 mismatch")
            with np.load(artifact, allow_pickle=False) as archive:
                if len(archive.files) != len(set(archive.files)):
                    raise ValueError("Duplicate NPZ array names")
                arrays = _arrays({name: archive[name].copy() for name in archive.files})
            if _descriptions(arrays) != record.get("arrays"):
                raise ValueError("Cache array contract mismatch")
            return arrays, record
        except (OSError, ValueError, TypeError, KeyError, AttributeError, EOFError,
                zipfile.BadZipFile) as error:
            self._quarantine(key, error)
            return None

    def get(self, spec):
        spec = _json_copy(spec)
        key = self.key(spec)
        with self._locked(key):
            result = self._read(key, spec)
        if result is not None:
            self._emit("cache_hit", cache_key=key,
                       artifact_sha256=result[1]["artifact_sha256"])
        return result

    def put(self, spec, arrays, metadata=None):
        spec = _json_copy(spec)
        key = self.key(spec)
        arrays = _arrays(arrays)
        metadata = _json_copy({} if metadata is None else metadata)
        if not isinstance(metadata, dict):
            raise ValueError("Cache metadata must be a dictionary")
        for field in ("elapsed_seconds", "init_seconds"):
            if field in metadata and metadata[field] is not None and (isinstance(metadata[field], bool)
                    or not isinstance(metadata[field], (int, float))
                    or not math.isfinite(metadata[field]) or metadata[field] < 0):
                raise ValueError(f"{field} must be a finite nonnegative number or null when unmeasured")
        with self._locked(key):
            current = self._read(key, spec)
            if current is not None:
                self._emit("cache_hit", cache_key=key,
                           artifact_sha256=current[1]["artifact_sha256"])
                return current[1]
            marker, artifact = self._paths(key)
            try:
                artifact_hash = atomic_npz(artifact, arrays)
                sync_directory(artifact.parent)
                record = {
                    "schema_version": SCHEMA_VERSION, "cache_key": key,
                    "spec": spec, "artifact_path": str(artifact.relative_to(self.root)),
                    "artifact_sha256": artifact_hash, "created_utc": _utc(),
                    "arrays": _descriptions(arrays), "metadata": metadata,
                    **{field: metadata[field] for field in ("elapsed_seconds", "init_seconds")
                       if field in metadata},
                }
                atomic_json(marker, record)
                sync_directory(marker.parent)
            except BaseException as error:
                self._quarantine(key, error, status="write_failed")
                raise
            self._emit("cache_committed", cache_key=key, artifact_sha256=artifact_hash)
            return record
