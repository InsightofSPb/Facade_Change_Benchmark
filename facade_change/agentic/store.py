"""Persistent physical regions and causal evidence, added 2026-11-08.

Records are immutable. The existing temporal RunStore owns the append-only
journal; JSON files are hash-bound materialized records, not a second ledger.
No model or annotation reader is used by an online lookup.
"""
from __future__ import annotations

import copy
import importlib
import json
import math
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from ..history_features import digest
from ..io import read_json, sha256, write_json


SCHEMA = 1
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
FORBIDDEN = {"gt", "ground_truth", "groundtruth", "annotations", "annotation",
             "packed", "labels", "targets", "target_mask", "gt_mask", "gt_path"}
GEOMETRY_FIELDS = {"source_to_crop", "crop_to_source", "source_to_reference",
    "reference_to_crop", "native_shape", "crop_shape", "bbox_xyxy", "bbox_xywh",
    "coordinate_system", "source_to_anchor", "crop_to_anchor", "anchor_to_crop",
    "source_to_region", "region_to_source", "reference_to_canvas", "source_to_canvas",
    "version", "geometry_sha256", "alignment_run_sha256"}
OBSERVATION_FIELDS = {"observation_id", "region_id", "building_id", "view_id", "split",
    "date", "date_status", "available_at", "original_path", "crop_path", "support_path",
    "original_sha256", "crop_sha256", "support_sha256", "image_id", "geometry",
    "visibility", "correspondence_status", "provenance", "description", "description_source"}
EVIDENCE_FIELDS = {"evidence_id", "region_id", "observation_ids", "kind", "available_at",
    "summary", "artifacts", "model", "revision", "prompt_sha256", "config_sha256",
    "parameters", "provenance", "source", "author", "coordinate_system", "status",
    "source_evidence_ids"}
EVIDENCE_KINDS = {"observation", "prediction", "measurement", "agent_hypothesis",
                  "human_confirmation", "knowledge"}
REGISTERED = {"verified", "aligned", "anchor", "provided", "geometrically_registered"}


def reject_forbidden(value):
    """Reject annotation-bearing objects even when nested under provenance."""
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or key.lower() in FORBIDDEN or key.lower().startswith("gt_"):
                raise ValueError(f"Field is not allowed in online evidence: {key}")
            reject_forbidden(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_forbidden(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite JSON value")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError("Online records require bounded JSON data, not arrays or objects")


def _allow(value, fields, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    reject_forbidden(value)
    unknown = set(value) - fields
    if unknown:
        raise ValueError(f"Unknown {name} fields: {', '.join(sorted(unknown))}")
    if len(json.dumps(value, ensure_ascii=False, allow_nan=False)) > 65536:
        raise ValueError(f"{name} exceeds the bounded metadata size")


def _id(value, name):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid {name}")
    return value


def temporal_value(value):
    """Conservative latest instant for YYYY, ISO date or timezone-aware instant.

    A year-only observation becomes available at the *end* of that year. It
    cannot be silently placed before a precisely dated event in that year.
    Undated objects deliberately return None and never enter causal history.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("A boolean is not a date")
    if isinstance(value, int):
        value = str(value)
    if not isinstance(value, str):
        raise ValueError("Expected YYYY, ISO date, or timezone-aware timestamp")
    if re.fullmatch(r"\d{4}", value):
        return datetime(int(value), 12, 31, 23, 59, 59, 999999, timezone.utc).timestamp()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        moment = datetime.fromisoformat(value).replace(hour=23, minute=59, second=59,
                                                     microsecond=999999, tzinfo=timezone.utc)
    else:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if moment.tzinfo is None:
            raise ValueError("Timestamp needs an explicit timezone")
    return moment.timestamp()


def causally_available(record, cutoff):
    end = temporal_value(cutoff)
    if end is None:
        raise ValueError("A causal query needs a dated cutoff")
    observed, known = temporal_value(record.get("date")), temporal_value(record.get("available_at"))
    return (observed is not None and known is not None and observed <= end and known <= end
            and record.get("date_status") != "unknown")


def checked_artifact(path, expected=None):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing evidence artifact: {path}")
    actual = sha256(path)
    if expected is not None and expected != actual:
        raise ValueError(f"Evidence artifact hash differs: {path}")
    return str(path), actual


def _geometry(value):
    _allow(value, GEOMETRY_FIELDS, "geometry")
    for key, item in value.items():
        if "_to_" in key:
            matrix = np.asarray(item, dtype=np.float64)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
                raise ValueError(f"Invalid invertible coordinate transform: {key}")
        if key in {"native_shape", "crop_shape"}:
            if (not isinstance(item, list) or len(item) != 2
                    or any(type(n) is not int or n <= 0 for n in item)):
                raise ValueError(f"Invalid geometry {key}")
    if not value.get("coordinate_system"):
        raise ValueError("Geometry needs an explicit coordinate_system")
    if "source_to_crop" in value and "crop_to_source" in value:
        product = np.asarray(value["source_to_crop"]) @ np.asarray(value["crop_to_source"])
        if abs(product[2, 2]) < 1e-12 or not np.allclose(product/product[2, 2], np.eye(3), rtol=1e-8, atol=1e-7):
            raise ValueError("Source/crop transforms are not inverses")
    if {"source_to_anchor", "crop_to_anchor", "source_to_crop"}.issubset(value):
        composed = np.asarray(value["crop_to_anchor"]) @ np.asarray(value["source_to_crop"])
        actual = np.asarray(value["source_to_anchor"])
        if (abs(composed[2, 2]) < 1e-12 or abs(actual[2, 2]) < 1e-12
                or not np.allclose(composed/composed[2, 2], actual/actual[2, 2], rtol=1e-8, atol=1e-7)):
            raise ValueError("Crop/anchor transforms disagree")
    return copy.deepcopy(value)


def _same_geometry(left, right):
    """Equivalent stored coordinate contracts may include extra diagnostics."""
    if not {"source_to_crop", "crop_shape"}.issubset(left) or not {"source_to_crop", "crop_shape"}.issubset(right):
        return left == right
    for key in set(left) & set(right):
        if "_to_" in key:
            if not np.allclose(left[key], right[key], rtol=1e-10, atol=1e-8):
                return False
        elif key in {"crop_shape", "native_shape", "bbox_xyxy", "bbox_xywh", "version"} and left[key] != right[key]:
            return False
    return True


class RegionStore:
    """Small local immutable store; use one context at a time per root."""
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        identity_path = self.root / "store_identity.json"
        try:
            with identity_path.open("x", encoding="utf-8") as stream:
                json.dump({"schema_version": SCHEMA, "store_id": "regions-" + uuid.uuid4().hex}, stream)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            pass
        identity = read_json(identity_path)
        self.store_id = _id(identity["store_id"], "store ID")
        journal = self.root / "journal"
        module = importlib.import_module("facade_change.2026-10-14_temporal_experiment")
        namespace = self.store_id
        class RegionJournal(module.RunStore):
            def event(self, kind, **values):
                self.record["run_id"] = namespace
                return super().event(kind, run_id=namespace,
                                     event_id=f"{namespace}:{self.sequence + 1}", **values)
        self.journal = RegionJournal(journal, {"kind": "agentic_region_store", "schema_version": SCHEMA},
                                     resume=journal.exists())
        self.journal.record["kind"] = "agentic_region_store"
        write_json(journal / "run.json", self.journal.record)
        self._committed = {}
        for line in (journal / "events.jsonl").read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("run_id") != self.store_id or not event.get("event_id", "").startswith(self.store_id + ":"):
                self.close()
                raise ValueError("Region journal identity differs from its immutable store identity")
            if event["event_type"] == "record_committed":
                key = (event["collection"], event["identifier"])
                old = self._committed.setdefault(key, (event["payload_sha256"], event["event_id"]))
                if old[0] != event["payload_sha256"]:
                    self.close()
                    raise ValueError("Conflicting immutable record in event journal")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if getattr(self, "journal", None) is not None:
            self.journal.close()
            self.journal = None

    def _path(self, collection, identifier):
        _id(identifier, "record ID")
        if collection not in {"regions", "observations", "evidence", "embeddings"}:
            raise ValueError("Unknown collection")
        return self.root / collection / (identifier + ".json")

    def _read(self, collection, identifier):
        key = (collection, identifier)
        if key not in self._committed:
            raise KeyError(f"No committed {collection} record: {identifier}")
        value = read_json(self._path(collection, identifier))
        expected, event = self._committed[key]
        if digest(value) != expected:
            raise ValueError(f"Immutable {collection} record changed: {identifier}")
        return {**value, "record_sha256": expected, "source_event_id": event}

    def _list(self, collection):
        return [self._read(collection, key) for category, key in sorted(self._committed)
                if category == collection]

    def _save(self, collection, identifier, value):
        value = copy.deepcopy(value)
        reject_forbidden(value)
        expected = digest(value)
        path = self._path(collection, identifier)
        if (collection, identifier) in self._committed:
            existing = self._read(collection, identifier)
            if existing["record_sha256"] != expected:
                raise ValueError(f"Refusing to change immutable {collection}: {identifier}")
            return existing
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and digest(read_json(path)) != expected:
            raise ValueError(f"Uncommitted conflicting record: {path}")
        if not path.exists():
            write_json(path, value)
        event = self.journal.event("record_committed", collection=collection,
                                   identifier=identifier, payload_sha256=expected)
        self._committed[collection, identifier] = (expected, event)
        return {**value, "record_sha256": expected, "source_event_id": event}

    def get_region(self, region_id):
        return self._read("regions", region_id)

    def list_regions(self):
        return self._list("regions")

    def register_region(self, building_id, view_id, anchor_sha256, bbox_xyxy, split="unassigned", region_id=None):
        _id(building_id, "building_id")
        _id(view_id, "view_id")
        if not isinstance(anchor_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", anchor_sha256):
            raise ValueError("Region anchor needs an image SHA-256")
        if (not isinstance(bbox_xyxy, (list, tuple)) or len(bbox_xyxy) != 4
                or any(type(n) is not int for n in bbox_xyxy)
                or bbox_xyxy[2] <= bbox_xyxy[0] or bbox_xyxy[3] <= bbox_xyxy[1]):
            raise ValueError("Region bbox must be integer xyxy in anchor pixels")
        if split not in {"train", "val", "test", "unassigned"}:
            raise ValueError("Invalid region split")
        for region in self.list_regions():
            if region["building_id"] == building_id and region["split"] != split:
                raise ValueError("Building leakage or reassignment across splits")
            if region["view_id"] == view_id and region["building_id"] != building_id:
                raise ValueError("View assigned to different buildings")
            if (region["building_id"] == building_id and region["view_id"] == view_id
                    and region["anchor_sha256"] != anchor_sha256):
                raise ValueError("Canonical anchor is immutable; map the new observation to the existing anchor")
        x0, y0, x1, y1 = bbox_xyxy
        identity = {"building_id": building_id, "view_id": view_id,
                    "anchor_sha256": anchor_sha256, "bbox_xywh": [x0, y0, x1-x0, y1-y0]}
        # Identical to history_features.assemble_tracks: later dates are absent.
        stable = digest(identity)[:24]
        if region_id is not None and region_id != stable:
            raise ValueError("region_id does not match the fixed physical-region identity")
        return self._save("regions", stable, {"schema_version": SCHEMA, "region_id": stable,
            **identity, "bbox_xyxy": list(bbox_xyxy), "split": split,
            "coordinate_system": "native_anchor_pixel_centres_xy", "coordinate_version": 1})

    def add_observation(self, record):
        _allow(record, OBSERVATION_FIELDS, "observation")
        row = copy.deepcopy(record)
        region = self.get_region(row["region_id"])
        for field in ("building_id", "view_id", "split"):
            if field in row and row[field] != region[field]:
                raise ValueError(f"Observation changes region {field}")
            row[field] = region[field]
        for key in ("date", "available_at"):
            temporal_value(row.get(key))
            row[key] = str(row[key]) if row.get(key) is not None else None
        if row.get("date_status") not in {"confirmed", "inferred", "unknown"}:
            raise ValueError("Explicit date_status is required")
        if row["date"] is None and row["date_status"] != "unknown":
            raise ValueError("An undated observation must have date_status=unknown")
        row["geometry"] = _geometry(row["geometry"])
        for box in ("bbox_xyxy", "bbox_xywh"):
            if box in row["geometry"] and row["geometry"][box] != region[box]:
                raise ValueError("Observation geometry changes the fixed physical region")
        for stem in ("original", "crop", "support"):
            row[stem + "_path"], row[stem + "_sha256"] = checked_artifact(
                row[stem + "_path"], row.get(stem + "_sha256"))
        # Only scalar support and raw image files are accepted, never a mixed
        # benchmark NPZ that might hold packed COCO annotations.
        with Image.open(row["crop_path"]) as image:
            if image.mode not in {"RGB", "RGBA"}:
                raise ValueError("Observation crop must be RGB/RGBA")
            shape = (image.height, image.width)
        if Path(row["support_path"]).suffix.lower() == ".npy":
            support = np.load(row["support_path"], allow_pickle=False)
            if support.dtype != bool:
                raise ValueError("Support NPY must be boolean")
        else:
            with Image.open(row["support_path"]) as image:
                support = np.asarray(image)
            if support.ndim != 2 or not np.isin(support, [0, 1, 255]).all():
                raise ValueError("Support image must be binary")
        if support.shape != shape or row["geometry"].get("crop_shape", list(shape)) != list(shape):
            raise ValueError("Crop, geometry and support grids disagree")
        row.setdefault("visibility", "unknown")
        row.setdefault("correspondence_status", "unknown")
        row.setdefault("provenance", {})
        row["schema_version"] = SCHEMA
        identifier = row.get("observation_id") or digest([row["region_id"], row["original_sha256"]])[:24]
        _id(identifier, "observation_id")
        row["observation_id"] = identifier
        for previous in self.list_observations():
            if previous["original_sha256"] == row["original_sha256"]:
                if previous["split"] != row["split"]:
                    raise ValueError("Identical image occurs across splits")
                if any(previous[key] != row[key] for key in ("building_id", "view_id", "date")):
                    raise ValueError("Identical image bytes have conflicting identity/date")
        if ("observations", identifier) in self._committed:
            previous = self.get_observation(identifier)
            # New run directories may contain byte-identical crops. Reuse the
            # original immutable observation, never rewrite its provenance.
            comparable = {k: v for k, v in row.items()
                          if k not in {"original_path", "crop_path", "support_path", "provenance", "geometry"}}
            if (all(previous.get(k) == v for k, v in comparable.items())
                    and _same_geometry(previous["geometry"], row["geometry"])):
                self.verify_observation(previous)
                return previous
        return self._save("observations", identifier, row)

    def get_observation(self, observation_id):
        return self._read("observations", observation_id)

    def list_observations(self):
        return self._list("observations")

    def verify_observation(self, row):
        for stem in ("original", "crop", "support"):
            checked_artifact(row[stem + "_path"], row[stem + "_sha256"])
        return row

    def history(self, region_id, cutoff, exclude_observation_ids=()):
        region = self.get_region(region_id)
        if temporal_value(cutoff) is None:
            raise ValueError("A causal history query needs a dated cutoff")
        selected, excluded = [], {"current_pair": 0, "undated_or_unavailable": 0,
                                 "future": 0, "unresolved_correspondence": 0}
        for row in self.list_observations():
            if row["region_id"] != region_id:
                continue
            if row["observation_id"] in exclude_observation_ids:
                excluded["current_pair"] += 1
            elif row["date"] is None or row["available_at"] is None or row["date_status"] == "unknown":
                excluded["undated_or_unavailable"] += 1
            elif not causally_available(row, cutoff):
                excluded["future"] += 1
            elif row["correspondence_status"] not in REGISTERED:
                excluded["unresolved_correspondence"] += 1
            else:
                selected.append(self.verify_observation(row))
        selected.sort(key=lambda r: (temporal_value(r["date"]), r["observation_id"]))
        available_ids = {row["observation_id"] for row in selected}
        evidence = []
        causal_memo = {}
        for row in self._list("evidence"):
            if (row.get("region_id") != region_id or row.get("kind") not in
                    {"prediction", "measurement", "agent_hypothesis", "human_confirmation"}):
                continue
            known = temporal_value(row.get("available_at"))
            referenced = set(row.get("observation_ids", []))
            if (known is not None and known <= temporal_value(cutoff) and referenced
                    and referenced.issubset(available_ids)
                    and self._evidence_causal(row, cutoff, set(exclude_observation_ids), causal_memo, set(), region)):
                evidence.append(self.get_evidence(row["evidence_id"]))
        evidence.sort(key=lambda row: (temporal_value(row["available_at"]), row["evidence_id"]))
        return {"status": "ok" if selected else "no_comparable_history", "region_id": region_id,
                "cutoff": str(cutoff), "observations": selected, "evidence": evidence, "excluded": excluded}

    def _evidence_causal(self, row, cutoff, excluded, memo, visiting, region):
        identifier = row["evidence_id"]
        if identifier in memo:
            return memo[identifier]
        if identifier in visiting:
            return False
        visiting = visiting | {identifier}
        known = temporal_value(row.get("available_at"))
        if known is None or known > temporal_value(cutoff):
            return False
        source_time = row.get("source", {}).get("available_at") if isinstance(row.get("source"), dict) else None
        if source_time is not None and temporal_value(source_time) > temporal_value(cutoff):
            return False
        observations = row.get("observation_ids", [])
        if not observations and row["kind"] != "knowledge":
            return False
        for observation_id in observations:
            if observation_id in excluded:
                return False
            observation = self.get_observation(observation_id)
            if (not causally_available(observation, cutoff)
                    or observation["correspondence_status"] not in REGISTERED):
                return False
            if (observation["region_id"] != region["region_id"]
                    and (observation["split"] != "train" or observation["building_id"] == region["building_id"])):
                return False
        for parent_id in row.get("source_evidence_ids", []):
            if not self._evidence_causal(self._read("evidence", parent_id), cutoff, excluded, memo, visiting, region):
                return False
        self.get_evidence(identifier)
        memo[identifier] = True
        return True

    def add_evidence(self, record):
        _allow(record, EVIDENCE_FIELDS, "evidence")
        row = copy.deepcopy(record)
        if row.get("kind") not in EVIDENCE_KINDS:
            raise ValueError("Evidence must distinguish observation/prediction/hypothesis/human confirmation")
        if row.get("region_id") is not None:
            self.get_region(row["region_id"])
        for identifier in row.get("observation_ids", []):
            observation = self.get_observation(identifier)
            if row.get("region_id") and observation["region_id"] != row["region_id"]:
                raise ValueError("Evidence references another physical region")
        temporal_value(row.get("available_at"))
        if row.get("available_at") is not None:
            row["available_at"] = str(row["available_at"])
        if row["kind"] == "human_confirmation" and (not row.get("author") or not row.get("source")):
            raise ValueError("Human confirmation requires author and source")
        if row["kind"] == "knowledge" and not row.get("source"):
            raise ValueError("Knowledge requires an attributable source")
        for parent in row.get("source_evidence_ids", []):
            self.get_evidence(parent)
        artifacts = []
        for artifact in row.get("artifacts", []):
            _allow(artifact, {"path", "sha256", "role", "media_type"}, "artifact")
            if not artifact.get("role"):
                raise ValueError("Evidence artifact needs an explicit role")
            if re.search(r"(?:^|[_\W])(gt|ground_truth|annotation|annotations|packed|labels|targets)(?:$|[_\W])",
                         artifact["role"].lower()):
                raise ValueError("Annotation artifacts are not online evidence")
            path, value = checked_artifact(artifact["path"], artifact.get("sha256"))
            if Path(path).suffix.lower() == ".npz":
                with np.load(path, allow_pickle=False) as arrays:
                    if any(key.lower() in FORBIDDEN or key.lower().startswith("gt_") for key in arrays.files):
                        raise ValueError("Mixed annotation archive is not online evidence")
            artifacts.append({**artifact, "path": path, "sha256": value})
        row["artifacts"] = artifacts
        identifier = row.pop("evidence_id", None) or "evidence-" + digest(row)[:32]
        _id(identifier, "evidence_id")
        row.update(schema_version=SCHEMA, evidence_id=identifier)
        return self._save("evidence", identifier, row)

    def get_evidence(self, evidence_id):
        row = self._read("evidence", evidence_id)
        for artifact in row.get("artifacts", []):
            checked_artifact(artifact["path"], artifact["sha256"])
        return row

    def list_evidence(self):
        return [self.get_evidence(row["evidence_id"]) for row in self._list("evidence")]

    def import_histories(self, histories_root, available_at=None, cutoff=None):
        """Import the existing RGB-only canonical history format by allowlist.

        available_at='observation_date' is an explicit retrospective availability
        assumption. Otherwise provide an archive-availability date; None keeps
        imported rows outside causal lookups until their availability is known.
        """
        from ..history_features import Fingerprints, _history_document, _inside
        root = Path(histories_root).expanduser().resolve()
        fingerprints = Fingerprints()
        document, manifest_sha = _history_document(root, fingerprints)
        imported = []
        for track in document["tracks"]:
            x, y, width, height = track["bbox_xywh"]
            region = self.register_region(track["building_id"], track["view_id"],
                track["anchor_sha256"], [x, y, x+width, y+height], track["split"], track["track_id"])
            for observation in track["observations"]:
                if cutoff is not None and temporal_value(observation["year"]) > temporal_value(cutoff):
                    continue
                matrix = np.asarray(observation["alignment"]["matrix"], dtype=float)
                anchor_to_crop = np.array([[1., 0., -x], [0., 1., -y], [0., 0., 1.]])
                transformed = anchor_to_crop @ matrix
                row = self.add_observation({"observation_id": observation["observation_id"],
                    "region_id": region["region_id"], "image_id": str(observation["image_id"]),
                    "date": str(observation["year"]), "date_status": "confirmed",
                    "available_at": str(observation["year"]) if available_at == "observation_date" else available_at,
                    "original_path": observation["original_image_path"],
                    "original_sha256": observation["image_sha256"],
                    "crop_path": str(_inside(root, observation["rgb"])), "crop_sha256": observation["rgb_sha256"],
                    "support_path": str(_inside(root, observation["support"])), "support_sha256": observation["support_sha256"],
                    "geometry": {"source_to_crop": transformed.tolist(),
                        "crop_to_source": np.linalg.inv(transformed).tolist(), "source_to_anchor": matrix.tolist(),
                        "anchor_to_crop": anchor_to_crop.tolist(), "crop_to_anchor": np.linalg.inv(anchor_to_crop).tolist(),
                        "crop_shape": [height, width], "bbox_xywh": [x, y, width, height],
                        "coordinate_system": "native_anchor_pixel_centres_xy"},
                    "visibility": "unknown", "correspondence_status": "aligned",
                    "provenance": {"producer": "history_features", "manifest_sha256": manifest_sha,
                        "availability_policy": "assumed_observation_date" if available_at == "observation_date" else "explicit_or_unknown"}})
                imported.append(row["observation_id"])
        fingerprints.verify()
        return {"status": "ok", "observation_ids": imported, "count": len(imported),
                "source_manifest_sha256": manifest_sha}
