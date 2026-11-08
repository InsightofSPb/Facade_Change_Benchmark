"""Versioned, deduplicated scientific arrays on the existing RunStore ledger.

Schema 2 uses immutable exact-dtype NPY objects and small map descriptors. It is
an adapter, not a new run manager: RunStore owns locking, events and run status.
Readers hash each immutable file once per reader session, then use read-only
memmaps. Only requested fields are materialized; legacy NPZ runs remain readable.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np

from facade_change.io import read_json, sha256, sync_directory, write_json
from facade_change.research.common import digest

SCHEMA_VERSION = 2
PREFIX = "2026-11-08_"
SCHEMA_FILE = PREFIX + "storage_schema.json"
DEPENDENCIES_FILE = PREFIX + "storage_dependencies.json"


class MissingDependencyError(FileNotFoundError):
    """A saved scientific result cannot be restored without its named artifact."""


def _inside(root, relative):
    root = Path(root).resolve()
    candidate = Path(relative)
    candidate = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Scientific artifact escapes its run: {relative}")
    return candidate


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def _signature(path):
    try:
        stat = path.stat()
    except FileNotFoundError as exc:
        raise MissingDependencyError(f"Missing scientific array/dependency: {path}") from exc
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


def _array_digest(array):
    value = np.ascontiguousarray(array).reshape(array.shape)
    h = hashlib.sha256()
    h.update(json.dumps([value.dtype.str, list(value.shape)], separators=(",", ":")).encode())
    if value.size:
        h.update(memoryview(value).cast("B"))
    return h.hexdigest(), value


class _Reader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.verified = {}
        self.descriptors = OrderedDict()

    def verify_file(self, path, expected):
        signature = _signature(path)
        previous = self.verified.get(str(path))
        if previous != (signature, expected):
            if expected and sha256(path) != expected:
                raise ValueError(f"Saved scientific artifact hash mismatch: {path}")
            self.verified[str(path)] = (signature, expected)
        return signature

    def descriptor(self, record):
        relative = record.get("path", record.get("artifact_path"))
        if not relative:
            raise ValueError("Saved scientific map has no artifact path")
        path = _inside(self.root, relative)
        expected = record.get("artifact_sha256", record.get("sha256"))
        signature = self.verify_file(path, expected)
        cached = self.descriptors.get(str(path))
        if cached is None or cached[0] != signature:
            value = read_json(path)
            if value.get("storage_schema") != SCHEMA_VERSION:
                raise ValueError(f"Unsupported scientific storage schema: {path}")
            self.descriptors[str(path)] = signature, value
            while len(self.descriptors) > 128:
                self.descriptors.popitem(last=False)
        else:
            self.descriptors.move_to_end(str(path))
        return self.descriptors[str(path)][1]

    def reference(self, reference, verify_only=False):
        operation = reference.get("operation")
        if operation == "stack":
            values = [self.reference(item, verify_only) for item in reference["arrays"]]
            return None if verify_only else np.stack(values, axis=reference.get("axis", 0))
        if operation == "mask_unknown":
            value = self.reference(reference["array"], verify_only)
            support = self.reference(reference["support"], verify_only)
            return None if verify_only else np.where(support, value, 255).astype(value.dtype, copy=False)
        if operation == "unknown":
            value = self.reference(reference["array"], verify_only)
            return None if verify_only else value == 255
        if operation:
            raise ValueError(f"Unsupported exact array reconstruction operation: {operation}")
        path = _inside(self.root, reference["path"])
        self.verify_file(path, reference["sha256"])
        if verify_only:
            return None
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.dtype.str != reference["dtype"] or list(value.shape) != reference["shape"]:
            raise ValueError(f"Scientific array shape/dtype contract mismatch: {path}")
        return value


_READERS = OrderedDict()


def _reader(root):
    key = str(Path(root).resolve())
    if key not in _READERS:
        _READERS[key] = _Reader(key)
        while len(_READERS) > 8:
            _READERS.popitem(last=False)
    _READERS.move_to_end(key)
    return _READERS[key]


def clear_reader_cache(root=None):
    """Start a fresh integrity-verification session (e.g. at explicit resume)."""
    if root is None:
        _READERS.clear()
    else:
        _READERS.pop(str(Path(root).resolve()), None)


class ArrayMap(Mapping):
    """Lazy field mapping and NPZ-compatible context manager; never loads all fields."""
    def __init__(self, reader, record, fields=None):
        self.reader, self.record, self.archive = reader, record, None
        self._cache = OrderedDict()
        path = _inside(reader.root, record.get("path", record.get("artifact_path")))
        if record.get("storage_schema") == SCHEMA_VERSION or path.suffix == ".json":
            self.descriptor = reader.descriptor(record)
            self.references = self.descriptor["arrays"]
        else:
            reader.verify_file(path, record.get("artifact_sha256", record.get("sha256")))
            self.archive = np.load(path, allow_pickle=False)
            self.references = dict.fromkeys(self.archive.files)
        self._keys = list(self.references) if fields is None else list(fields)
        missing = set(self._keys) - set(self.references)
        if missing:
            self.close()
            raise KeyError(f"Saved scientific fields are unavailable: {sorted(missing)}")

    @property
    def files(self):
        return list(self._keys)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __contains__(self, key):
        return key in self._keys

    def __getitem__(self, key):
        if key not in self._keys:
            raise KeyError(key)
        if key not in self._cache:
            self._cache[key] = self.archive[key] if self.archive is not None else self.reader.reference(self.references[key])
            while len(self._cache) > 8:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        return self._cache[key]

    def close(self):
        if self.archive is not None:
            self.archive.close()
            self.archive = None
        self._cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


def open_arrays(root, record, fields=None):
    return ArrayMap(_reader(root), record, fields)


read_map = open_arrays


def verify_manifest(root, records, *, fresh=False):
    """Verify every referenced object once without decoding any tensor or RGB."""
    if fresh:
        clear_reader_cache(root)
    reader = _reader(root)
    count = 0
    for record in records:
        with ArrayMap(reader, record) as arrays:
            if arrays.archive is None:
                for reference in arrays.references.values():
                    reader.reference(reference, verify_only=True)
            count += 1
    return dict(storage_schema=SCHEMA_VERSION, maps=count, unique_verified_files=len(reader.verified))


class SharedMapStore:
    """Save immutable scientific maps and references using RunStore unit markers."""
    def __init__(self, out, ledger=None, dataset=None):
        self.out = Path(out).resolve()
        self.ledger, self.dataset = ledger, dataset
        self.reader = _reader(self.out)
        self._records = {}
        self._observations = {}
        self._targets = {}
        self._initialized = False
        self._legacy_maps = False
        for path in sorted((self.out / "units").glob("*.json")):
            record = read_json(path)
            if record.get("storage_schema") == SCHEMA_VERSION and record.get("kind") == "history_scenario_float_maps":
                self._records[(record["method"], record["event_id"])] = record
            elif record.get("kind") == "history_scenario_float_maps":
                self._legacy_maps = True

    @property
    def records(self):
        return list(self._records.values())

    def get(self, method, event_id):
        return self._records.get((method, event_id))

    def read(self, record, fields=None):
        return ArrayMap(self.reader, record, fields)

    def read_reference(self, reference):
        """Read one shared frozen-input field without opening unrelated tensors."""
        return self.reader.reference(reference)

    def references(self, record, fields=None):
        refs = self.reader.descriptor(record)["arrays"]
        return {key: refs[key] for key in (refs if fields is None else fields)}

    def verify(self, records=None):
        return verify_manifest(self.out, self.records if records is None else records)

    def _event(self, kind, **values):
        return self.ledger.event(kind, **values) if self.ledger is not None else None

    def _initialize(self):
        if self._initialized:
            return
        if self._legacy_maps:
            raise ValueError("Schema-1 scientific runs are read-only; use a new output directory for schema 2")
        contract = dict(storage_schema=SCHEMA_VERSION, format="immutable exact-dtype NPY objects + JSON references",
            scientific_dtypes="float32 or original exact dtype; no implicit quantization", array_hash="dtype+shape+contiguous bytes",
            integrity="verify on first open/resume, reuse only while size/mtime/ctime/inode unchanged",
            nonlinear_order="save exact raster when native-grid reconstruction is not declared exact",
            maps="every exported TRAIN/VAL/TEST event; gallery selection does not limit storage",
            reference_operations=["stack", "mask_unknown", "unknown"], legacy="schema-1 NPZ read-only")
        path = self.out / SCHEMA_FILE
        if path.exists() and read_json(path) != contract:
            raise ValueError("Scientific storage schema contract changed")
        if not path.exists():
            _json(path, contract)
            self._event("scientific_storage_schema", artifact_path=SCHEMA_FILE, artifact_sha256=sha256(path), storage_schema=SCHEMA_VERSION)
        self._dependencies()
        self._initialized = True

    def _dependencies(self):
        """Cache inputs remain external/read-only; reports use the copied numeric objects."""
        entries = []
        dataset = self.dataset
        if dataset is not None and hasattr(dataset, "cache_index"):
            for key, item in sorted(dataset.cache_index.items()):
                if item is None:
                    continue
                kind = key.split(":", 1)[0]
                root = item.get("artifact_root") or (dataset.saved.out if kind == "feature" else dataset.frozen.root)
                entries.append(dict(source_key=key, path=str((Path(root) / item["artifact_path"]).resolve()),
                    sha256=item["artifact_sha256"], required_for=["retraining", "recomputing_base_predictions"],
                    report_only_required=False, read_only=True, owned_by_this_run=item.get("origin") == "run_overlay",
                    verified_by="ScenarioData source-cache/overlay audit"))
        value = dict(storage_schema=SCHEMA_VERSION, dependencies=entries,
                     report_only_external_dependencies=[], external_caches_are_not_owned_by_this_run=True)
        if dataset is not None and hasattr(dataset, "geometry"):
            value["source_geometry_contract"] = dataset.geometry
        path = self.out / DEPENDENCIES_FILE
        if path.exists() and read_json(path) != value:
            raise ValueError("External source cache dependencies changed")
        if not path.exists():
            _json(path, value)

    def put(self, array):
        """Store an exact immutable array once, shared by any number of logical fields."""
        array = np.asarray(array)
        if array.dtype.hasobject:
            raise TypeError("Scientific arrays must not contain Python objects")
        content, array = _array_digest(array)
        relative = f"arrays/{content[:2]}/{content}.npy"
        path = self.out / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
            try:
                with temporary.open("xb") as stream:
                    np.save(stream, array, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                sync_directory(path.parent)
            finally:
                if temporary.exists():
                    temporary.unlink()
            checksum = sha256(path)
            self.reader.verified[str(path)] = (_signature(path), checksum)
            self._event("scientific_array_saved", artifact_path=relative, artifact_sha256=checksum,
                        content_sha256=content, shape=list(array.shape), dtype=array.dtype.str, bytes=path.stat().st_size)
        else:
            # The deterministic NPY header + exact bytes are immutable; check once per writer session.
            cached = self.reader.verified.get(str(path))
            checksum = cached[1] if cached else sha256(path)
            self.reader.verify_file(path, checksum)
            existing = np.load(path, mmap_mode="r", allow_pickle=False)
            if existing.dtype != array.dtype or existing.shape != array.shape:
                raise ValueError(f"Content-addressed scientific array collision: {path}")
            if cached is None and _array_digest(existing)[0] != content:
                raise ValueError(f"Content-addressed scientific array hash mismatch: {path}")
        return dict(path=relative, sha256=checksum, content_sha256=content, dtype=array.dtype.str,
                    shape=list(array.shape), bytes=int(array.nbytes))

    def _observation(self, observation):
        identifier = observation["observation_id"]
        if identifier in self._observations:
            return self._observations[identifier]
        path = self.out / "shared" / "observations" / (digest(identifier) + ".json")
        if path.exists():
            value = read_json(path)
        else:
            rgb, support = self.dataset.frame(identifier)
            value = dict(observation_id=identifier, year=observation["year"],
                         rgb=self.put(rgb), geometric_support=self.put(np.asarray(support, bool)))
            _json(path, value)
        self._observations[identifier] = value
        return value

    def _shared_targets(self, row):
        identifier = row["event_id"]
        if identifier not in self._targets:
            path = self.out / "shared" / "targets" / (digest(identifier) + ".json")
            if path.exists():
                value = read_json(path)
            else:
                value = {key: self.put(array) for key, array in self.dataset.targets(row).items()}
                _json(path, value)
            self._targets[identifier] = value
        return self._targets[identifier]

    def _semantic_identity(self, row, method, spec, metadata, key, reference):
        if key not in ("semantic_probability", "semantic_before") or self.dataset is None:
            return
        index = int(row["frame"]) - (key == "semantic_before")
        observations = self.dataset.tracks[row["track_id"]]["observations"][:index + 1]
        # SemanticMemory's repeat control reconstructs each endpoint prefix
        # independently; both controls therefore reuse the same earlier state.
        producer = metadata.get("base_method") or spec.get("parent_method") or method
        state_contract = {name: spec[name] for name in ("scenario", "mode", "control", "hidden", "size", "family")
                          if name in spec}
        identity = dict(model=producer, checkpoint=metadata.get("checkpoint", metadata.get("checkpoint_sha256")),
            track=row["track_id"], observation=observations[-1]["observation_id"],
            prefix_contract=digest(dict(semantic_state_contract=state_contract, prefix=observations)))
        path = self.out / "shared" / "semantic_states" / (digest(identity) + ".json")
        value = dict(identity=identity, array=reference)
        if path.exists():
            if read_json(path) != value:
                raise ValueError("Semantic state changed for the same model/checkpoint/observation/prefix contract")
        else:
            _json(path, value)

    def save(self, row, method, spec, result, metadata=None, refs=None):
        """Save an event, accepting parent references to avoid touching large tensors.

        ``refs`` is a field->reference mapping from ``references(parent)``. Explicit
        result fields override references; repeated content still stores once.
        Exact raster diagnostics are retained unless the caller supplies a proven
        transformation recipe in metadata. No interpolation/nonlinearity is inferred.
        """
        self._initialize()
        old = self.get(method, row["event_id"])
        if old is not None:
            self.reader.descriptor(old)
            return old
        metadata = dict(metadata or {})
        arrays = dict(refs or {})
        if self.dataset is not None:
            arrays.update(self._shared_targets(row))
        arrays.update({key: self.put(value) for key, value in result.get("extras", {}).items()})
        arrays["score"] = self.put(np.asarray(result["score"], np.float32))
        arrays["support"] = self.put(np.asarray(result["support"], bool))
        if spec.get("scenario") == "C":
            if "progression_target" not in arrays:
                raise ValueError("Progression map lacks its annotation-derived target")
            target = self.reader.reference(arrays["progression_target"])
            arrays["target"] = self.put(np.where(target < 0, 255, target).astype(np.uint8))
            arrays["support"] = self.put(np.asarray(result["support"], bool) & (target >= 0))
            if "class_targets" in arrays:
                shape = tuple(arrays["class_targets"]["shape"])
                arrays["class_targets"] = self.put(np.full(shape, 255, np.uint8))
        for key in ("target", "class_targets"):
            if key in arrays:
                arrays[key] = dict(operation="mask_unknown", array=arrays[key], support=arrays["support"])
        if "target" in arrays:
            arrays["unknown"] = dict(operation="unknown", array=arrays["target"])
        if self.dataset is not None:
            observations = self.dataset.tracks[row["track_id"]]["observations"][:int(row["frame"]) + 1]
            frames = [self._observation(item) for item in observations]
            arrays["history_rgb"] = dict(operation="stack", arrays=[item["rgb"] for item in frames], axis=0)
            arrays["years"] = self.put(np.asarray([item["year"] for item in observations]))
            if spec.get("scenario") == "A":
                annotation = self.dataset.annotation_history(row)
                arrays["semantic_target"] = self.put(annotation["labels"][-1].astype(np.uint8))
                arrays["semantic_known"] = self.put(np.asarray(annotation["known"][-1, 0], bool))
                for key in ("semantic_probability", "semantic_before"):
                    if key in arrays:
                        self._semantic_identity(row, method, spec, metadata, key, arrays[key])
        identifier = PREFIX + "scenarios-map-" + digest([method, row["event_id"]])[:24]
        record = dict(unit_id=identifier, kind="history_scenario_float_maps", storage_schema=SCHEMA_VERSION,
            method=method, task=spec["task"], family=spec.get("family"), size=spec.get("size"),
            control=spec.get("control"), scenario=spec.get("scenario", "main"),
            target_type=spec.get("target_type"), spec=spec,
            **{key: row[key] for key in ("event_id", "building_id", "split", "track_id", "frame", "bbox_xywh", "view_id", "previous_year", "current_year", "known_pixels") if key in row})
        record.update(observations=int(row["frame"]) + 1, **metadata)
        relative = f"maps/{identifier}.json"
        descriptor = dict(storage_schema=SCHEMA_VERSION, unit_id=identifier, arrays=arrays,
                          transformation_recipes=metadata.get("transformation_recipes", {}),
                          raster_contract="Stored rasters are exact float32 outputs in the declared original operation order")
        _json(self.out / relative, descriptor)
        record.update(path=relative, artifact_path=relative, artifact_sha256=sha256(self.out / relative))
        record["source_event_id"] = self._event("unit_artifact_saved", unit_id=identifier,
            artifact_path=relative, artifact_sha256=record["artifact_sha256"], storage_schema=SCHEMA_VERSION)
        _json(self.out / "units" / (identifier + ".json"), record)
        self._event("unit_completed", unit_id=identifier, storage_schema=SCHEMA_VERSION)
        self._records[(method, row["event_id"])] = record
        return record

    def storage_summary(self):
        paths = list((self.out / "arrays").rglob("*.npy"))
        return dict(storage_schema=SCHEMA_VERSION, maps=len(self._records), unique_array_files=len(paths),
                    array_file_bytes=sum(path.stat().st_size for path in paths),
                    compression="none; exact NPY", fields_are_references=True)
