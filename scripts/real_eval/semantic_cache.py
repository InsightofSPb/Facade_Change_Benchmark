"""Durable per-computation cache for the semantic routing experiment.

The caller owns the task specification: include input RGB/support hashes, model
and source/checkpoint identity, vocabulary, preprocessing and output contract.
Do not put the run directory, requested method list or split list in that spec.
Evaluation/visualization tasks must also include their GT/metric/render versions.
An optional ``event=run.event`` writes to the existing research event ledger.
"""
from __future__ import annotations

import hashlib
import copy
import json
import math
import os
import stat
import threading
import time
import zipfile
from collections import OrderedDict
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
    result.update(np.ascontiguousarray(array))
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

    ``get`` returns ``(arrays, record)`` or ``None``; ``names`` selects arrays.
    ``inspect`` returns just the verified record. Only a committed, validated
    JSON marker can make an entry reusable. Corruption is retained in quarantine
    and treated as a miss. ``put`` preserves any existing valid result. Locks
    protect individual cache keys even when different run directories share it.
    Each process fully verifies an entry once, then checks both file identities
    on every access. Array reuse is bounded by bytes; oversized SAM outputs are
    never retained. Returned arrays/records cannot mutate the private memo.
    """

    def __init__(self, root, event=None, *, max_memory_bytes=64 * 1024 * 1024,
                 max_verified_entries=4096):
        if type(max_memory_bytes) is not int or max_memory_bytes < 0:
            raise ValueError("max_memory_bytes must be a nonnegative integer")
        if type(max_verified_entries) is not int or max_verified_entries < 1:
            raise ValueError("max_verified_entries must be a positive integer")
        self.root = Path(root).expanduser().resolve()
        self.event = event
        self.max_memory_bytes = max_memory_bytes
        self.max_verified_entries = max_verified_entries
        self._verified = OrderedDict()
        self._resident = OrderedDict()
        self._resident_bytes = 0
        self._memo_lock = threading.RLock()
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

    def _fingerprint(self, key):
        values = []
        for path in self._paths(key):
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("Cache artifacts must be regular files, never symlinks")
            values.append((info.st_dev, info.st_ino, info.st_mode, info.st_size,
                           info.st_mtime_ns, info.st_ctime_ns))
        return tuple(values)

    @property
    def memory_bytes(self):
        """Bytes privately retained for array reuse, excluding caller outputs."""
        with self._memo_lock:
            return self._resident_bytes

    def _drop_arrays(self, key):
        resident = self._resident.pop(key, None)
        if resident is not None:
            self._resident_bytes -= sum(value.nbytes for value in resident.values())

    def _forget(self, key):
        with self._memo_lock:
            self._verified.pop(key, None)
            self._drop_arrays(key)

    def _known(self, key, fingerprint):
        with self._memo_lock:
            entry = self._verified.get(key)
            if entry is None or entry[0] != fingerprint:
                self._forget(key)
                return None
            self._verified.move_to_end(key)
            return copy.deepcopy(entry[1])

    def _memory_arrays(self, key, names):
        with self._memo_lock:
            resident = self._resident.get(key)
            if resident is None or not set(names) <= resident.keys():
                return None
            self._resident.move_to_end(key)
            return {name: resident[name].copy() for name in names}

    def _remember(self, key, fingerprint, record, arrays):
        with self._memo_lock:
            self._verified[key] = (fingerprint, copy.deepcopy(record))
            self._verified.move_to_end(key)
            while len(self._verified) > self.max_verified_entries:
                old_key, _ = self._verified.popitem(last=False)
                self._drop_arrays(old_key)
            size = sum(value.nbytes for value in arrays.values())
            if not self.max_memory_bytes or size > self.max_memory_bytes:
                return
            self._drop_arrays(key)
            while self._resident and self._resident_bytes + size > self.max_memory_bytes:
                self._drop_arrays(next(iter(self._resident)))
            self._resident[key] = {name: value.copy() for name, value in arrays.items()}
            self._resident_bytes += size

    @staticmethod
    def _names(record, names):
        wanted = list(record["arrays"]) if names is None else names
        if set(wanted) - record["arrays"].keys():
            raise LookupError("Requested arrays are absent from this cache entry")
        return wanted

    def _emit(self, kind, **values):
        if self.event is not None:
            return self.event(kind, **values)
        return None

    def _partials(self, key):
        return [partial for path in self._paths(key)
                for partial in path.parent.glob("." + path.name + "*")]

    def _quarantine(self, key, reason, status="invalidated"):
        self._forget(key)
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

    def _read(self, key, spec, names=None, metadata_only=False):
        marker, artifact = self._paths(key)
        if not os.path.lexists(marker) and not os.path.lexists(artifact):
            self._forget(key)
            if self._partials(key):
                self._quarantine(key, "Interrupted temporary artifact without commit")
            return None
        try:
            fingerprint = self._fingerprint(key)
            record = self._known(key, fingerprint)
            if record is not None:
                if metadata_only:
                    if self._fingerprint(key) != fingerprint:
                        raise ValueError("Cache changed while reading metadata")
                    return None, record
                wanted = self._names(record, names)
                arrays = self._memory_arrays(key, wanted)
                if arrays is not None:
                    if self._fingerprint(key) != fingerprint:
                        raise ValueError("Cache changed while copying resident arrays")
                    return arrays, record
                # SHA and all array hashes were verified for this exact file
                # identity. Decode only what the caller needs; keep safe dtypes.
                with np.load(artifact, allow_pickle=False) as archive:
                    arrays = _arrays({name: archive[name] for name in wanted})
                for name, value in arrays.items():
                    description = record["arrays"][name]
                    if value.dtype.str != description["dtype"] or list(value.shape) != description["shape"]:
                        raise ValueError("Cache array contract mismatch")
                if self._fingerprint(key) != fingerprint:
                    raise ValueError("Cache changed while reading arrays")
                self._remember(key, fingerprint, record, arrays)
                return arrays, record
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
                arrays = _arrays({name: archive[name] for name in archive.files})
            if _descriptions(arrays) != record.get("arrays"):
                raise ValueError("Cache array contract mismatch")
            if self._fingerprint(key) != fingerprint:
                raise ValueError("Cache changed during integrity validation")
            wanted = self._names(record, names)
            selected = {name: arrays[name] for name in wanted}
            self._remember(key, fingerprint, record, selected)
            return (None if metadata_only else selected), record
        except (OSError, ValueError, TypeError, KeyError, AttributeError, EOFError,
                zipfile.BadZipFile) as error:
            self._quarantine(key, error)
            return None

    def get(self, spec, names=None):
        """Read selected arrays; first access still verifies the entire artifact."""
        if names is not None:
            if (not isinstance(names, (list, tuple)) or not names
                    or any(not isinstance(name, str) for name in names)
                    or len(names) != len(set(names))):
                raise ValueError("names must be a nonempty list of distinct array names")
        spec = _json_copy(spec)
        key = self.key(spec)
        with self._locked(key):
            result = self._read(key, spec, names=names)
        if result is not None:
            self._emit("cache_hit", cache_key=key,
                       artifact_sha256=result[1]["artifact_sha256"])
        return result

    def inspect(self, spec):
        """Return verified metadata without repeated array decoding/hashing.

        The first access performs full integrity validation, exactly as get.
        Later accesses check marker/NPZ file identities before trusting the memo.
        """
        spec = _json_copy(spec)
        key = self.key(spec)
        with self._locked(key):
            result = self._read(key, spec, metadata_only=True)
        if result is None:
            return None
        self._emit("cache_hit", cache_key=key, artifact_sha256=result[1]["artifact_sha256"],
                   metadata_only=True)
        return result[1]

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
            current = self._read(key, spec, metadata_only=True)
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
            # This process just hashed and atomically committed these arrays.
            # Seed verification now so a score-only follow-up need not decode
            # the full SAM instance archive we just wrote. Callers own the
            # input arrays until put returns and must not mutate them mid-write.
            self._remember(key, self._fingerprint(key), record, arrays)
            self._emit("cache_committed", cache_key=key, artifact_sha256=artifact_hash)
            return record
