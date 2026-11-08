"""Small, allowlisted evidence tools for the 2026-11-08 change assistant.

Only identifiers exposed by the current case/history/search can be opened.
Benchmark archives and arbitrary model-provided paths are never accepted.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..io import read_json, sha256, write_json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


TOOL_SCHEMAS = {
    "get_history": {"description": "Past observations of this region, excluding the compared pair.",
                    "arguments": {"limit": "optional integer 1..10"}},
    "open_crop": {"description": "Open a permitted observation as an actual image; source preserves detail.",
                  "arguments": {"observation_id": "known ID", "view": "crop or source (default crop)"}},
    "open_pair": {"description": "Reopen the two current RGB crops.", "arguments": {}},
    "get_maps": {"description": "Existing predicted method maps, distinct types/scales and provenance.",
                 "arguments": {"method": "optional exact method ID", "limit": "preview image limit 0..10 (default 2); all summaries retained"}},
    "search_similar": {"description": "Permitted past TRAIN analogues; current building and duplicate images excluded.",
                       "arguments": {"query": "text OR observation_id", "observation_id": "optional known ID",
                                     "representation": "visual, history, or joint_image_text (default visual)",
                                     "limit": "optional integer 1..10"}},
    "get_knowledge": {"description": "Sourced text fragments and related project concepts; not inferred causes.",
                      "arguments": {"query": "text", "concept_id": "optional known concept ID",
                                    "limit": "optional integer 1..10"}},
    "sam3": {"description": "Text-conditioned SAM3 on one permitted RGB crop. Empty is not no change.",
             "arguments": {"observation_id": "known ID", "text": "text concept",
                           "concept_id": "optional project concept ID"}},
}


def _text(value, name, maximum=1000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be nonempty text of at most {maximum} characters")
    return value.strip()


def _limit(value, default=5):
    value = default if value is None else value
    if type(value) is not int or not 1 <= value <= 10:
        raise ValueError("limit must be an integer in 1..10")
    return value


def remap_mask(mask, support, geometry):
    """Nearest-neighbour crop->original mapping, with explicit pixel-centre H."""
    import cv2
    mask, support = np.asarray(mask), np.asarray(support)
    if mask.ndim != 2 or mask.dtype != bool or support.shape != mask.shape or support.dtype != bool:
        raise ValueError("Mask/support must be same-grid boolean arrays")
    matrix = np.asarray(geometry.get("crop_to_source"), dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
        raise ValueError("A valid crop_to_source transform is required")
    if "source_to_crop" in geometry:
        forward = np.asarray(geometry["source_to_crop"], dtype=float)
        product = forward @ matrix if forward.shape == (3, 3) else np.zeros((3, 3))
        if (not np.isfinite(product).all() or abs(product[2, 2]) < 1e-12
                or not np.allclose(product / product[2, 2], np.eye(3), rtol=1e-7, atol=1e-7)):
            raise ValueError("Crop/source transforms are not inverse mappings")
    shape = geometry.get("native_shape")
    if (not isinstance(shape, (list, tuple)) or len(shape) < 2
            or any(type(v) is not int or v <= 0 for v in shape[:2])):
        raise ValueError("A native observation shape is required")
    h, w = shape[:2]
    if h * w > 50_000_000:
        raise ValueError("Unsafe original image size; no implicit resizing")
    result = cv2.warpPerspective((mask & support).astype(np.uint8), matrix, (w, h),
                                 flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0).astype(bool)
    native_support = cv2.warpPerspective(support.astype(np.uint8), matrix, (w, h),
                                         flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=0).astype(bool)
    return result & native_support, native_support


class ToolService:
    """Concrete, local evidence operations; the agent chooses IDs, never paths."""

    def __init__(self, store, search, knowledge, model_pool, case, out,
                 method_evidence=(), embedding_role=None):
        self.store, self.search, self.knowledge, self.pool = store, search, knowledge, model_pool
        self.case = dict(case)
        self.out = Path(out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.method_evidence = list(method_evidence or case.get("method_evidence", ()))
        self.embedding_role = embedding_role
        self.current = [str(case["reference_observation_id"]), str(case["source_observation_id"])]
        if self.current[0] == self.current[1]:
            raise ValueError("The comparison must contain distinct observations")
        self.allowed = set(self.current)
        self.evidence = {}
        self._timeouts = None
        self.remaining_model_calls = None
        self._model_usage = {}
        self._model_source_usage = None
        self.contract = {key: case.get(key) for key in ("case_id", "region_id", "building_id", "view_id", "split",
                         "cutoff", "reference_observation_id", "source_observation_id", "reference_date", "source_date")}
        self.contract.update(retrieval_splits=["train"], exclude_current_building=True,
                             exclude_current_image_hashes=True, schema_version=1)
        path = self.out / "evidence.json"
        if path.exists():
            saved = read_json(path)
            if (saved.get("contract") != self.contract
                    or saved.get("evidence_sha256") != digest(saved.get("evidence", []))):
                raise ValueError("Evidence contract/cutoff or content changed")
            self.evidence = {row["evidence_id"]: row for row in saved.get("evidence", ())}
            for row in self.evidence.values():
                for artifact in row.get("artifacts", ()):
                    if sha256(artifact["path"]) != artifact["sha256"]:
                        raise ValueError("Saved tool evidence artifact changed")
                for identifier in row.get("observation_ids", ()):
                    self._authorize(identifier)
                self.allowed.update(row.get("observation_ids", ()))

    def _save(self):
        records = list(self.evidence.values())
        write_json(self.out / "evidence.json", {"schema_version": 1, "case_id": self.case["case_id"],
                                                "contract": self.contract, "evidence": records,
                                                "evidence_sha256": digest(records)})

    def _remember(self, record):
        record = dict(record)
        identifier = record.setdefault("evidence_id", "evidence-" + digest(record)[:32])
        previous = self.evidence.get(identifier)
        if previous is not None and ({k: v for k, v in previous.items() if k != "source_event_id"}
                                     != {k: v for k, v in record.items() if k != "source_event_id"}):
            raise ValueError("An immutable evidence ID changed")
        for artifact in record.get("artifacts", ()):
            path = Path(artifact["path"])
            if not path.is_file() or sha256(path) != artifact["sha256"]:
                raise ValueError(f"Missing or changed evidence artifact: {path}")
        # Reuse the project's persistent event journal for new tool evidence.
        # Existing method records retain their original producer metadata.
        if hasattr(self.store, "add_evidence"):
            try:
                persisted = self.store.get_evidence(identifier)
            except KeyError:
                allowed = {"evidence_id", "region_id", "observation_ids", "kind", "summary", "artifacts",
                           "model", "revision", "coordinate_system", "source", "status"}
                persisted_record = {key: value for key, value in record.items() if key in allowed and value is not None}
                persisted_record["available_at"] = record.get("available_at", self.case.get("cutoff"))
                persisted_record["parameters"] = {key: value for key, value in record.items()
                    if key not in allowed | {"available_at", "source_event_id"} and value is not None}
                persisted = self.store.add_evidence(persisted_record)
            else:
                for key in ("kind", "region_id", "observation_ids", "summary", "source", "model", "revision", "source_evidence_ids", "author", "coordinate_system"):
                    if key in record and key in persisted and record[key] != persisted[key]:
                        raise ValueError("Evidence changes the meaning of an immutable stored ID")
                fingerprints = lambda artifacts: sorted((a["sha256"], a["role"], a.get("media_type") or "") for a in artifacts)
                if fingerprints(record.get("artifacts", [])) != fingerprints(persisted.get("artifacts", [])):
                    raise ValueError("Evidence changes immutable stored artifacts")
            record["source_event_id"] = persisted.get("source_event_id")
        self.evidence[identifier] = record
        self._save()
        return record

    @staticmethod
    def _artifact(path, role, media_type=None):
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError(f"Missing artifact: {path}")
        return {"path": str(path), "sha256": sha256(path), "role": role,
                **({"media_type": media_type} if media_type else {})}

    def _observation(self, identifier):
        identifier = _text(identifier, "observation_id", 256)
        if identifier not in self.allowed:
            raise ValueError("Observation has not been exposed by the current pair, history or filtered retrieval")
        return self._authorize(identifier)

    def _authorize(self, identifier):
        from .store import REGISTERED, causally_available
        observation = self.store.verify_observation(self.store.get_observation(identifier))
        if identifier in self.current:
            if observation["region_id"] != self.case["region_id"]:
                raise ValueError("Compared observation belongs to another region")
            return observation
        if (not causally_available(observation, self.case["cutoff"])
                or observation.get("correspondence_status") not in REGISTERED):
            raise ValueError("Observation is not causally available/comparable")
        if observation["region_id"] != self.case["region_id"]:
            hashes = {self.store.get_observation(i)[key] for i in self.current for key in ("original_sha256", "crop_sha256")}
            if (observation["split"] != "train" or observation["building_id"] == self.case["building_id"]
                    or observation["original_sha256"] in hashes or observation["crop_sha256"] in hashes):
                raise ValueError("Observation is outside the permitted analogue library")
        return observation

    def _observation_evidence(self, observation, view="crop"):
        if view not in {"crop", "source"}:
            raise ValueError("view must be crop or source")
        path = observation["crop_path" if view == "crop" else "original_path"]
        expected = observation.get("crop_sha256" if view == "crop" else "original_sha256")
        if expected and sha256(path) != expected:
            raise ValueError("Observation image has changed")
        identifier = str(observation["observation_id"])
        with Image.open(path) as source_image:
            image_shape = [source_image.height, source_image.width]
        geometry = dict(observation.get("geometry", {}))
        region = self.store.get_region(observation["region_id"])
        geometry["anchor_sha256"] = region["anchor_sha256"]
        geometry.setdefault("bbox_xyxy", region["bbox_xyxy"])
        if "native_shape" not in geometry:
            with Image.open(observation["original_path"]) as original:
                geometry["native_shape"] = [original.height, original.width]
        crop_frame = {"coordinate_system": "region_crop_pixels", "region_id": observation["region_id"]}
        rgb_frame = crop_frame if view == "crop" else {
            "coordinate_system": "observation_native", "observation_id": identifier}
        record = {"evidence_id": "observation-" + digest([identifier, view, "geometric-support-v3"])[:32], "kind": "observation",
                  "observation_ids": [identifier], "region_id": observation["region_id"],
                  "observation_id": identifier,
                  "date": observation.get("date"), "date_status": observation.get("date_status"),
                  "available_at": observation.get("available_at"),
                  "visibility": observation.get("visibility", "unknown"),
                  "correspondence_status": observation.get("correspondence_status", "unknown"),
                  "coordinate_system": "region_crop_pixels" if view == "crop" else "observation_native",
                  "image_shape": image_shape,
                  "geometry": geometry,
                  "artifact_coordinates": {"rgb": rgb_frame, "geometric_support": crop_frame},
                  "summary": "Unmodified RGB. The separate geometric support artifact is not an occlusion mask.",
                  "artifacts": [self._artifact(path, "rgb", "image/png"),
                                self._artifact(observation["support_path"], "geometric_support")]}
        self._remember(record)
        image = {"type": "image", "path": str(Path(path).resolve()), "evidence_id": record["evidence_id"]}
        return record, image

    @staticmethod
    def _public(record):
        # Artifact links are references, not an invitation to read arbitrary archives.
        keys = ("evidence_id", "kind", "region_id", "observation_ids", "date", "date_status", "available_at",
                "visibility", "correspondence_status", "coordinate_system", "summary", "method",
                "output_kind", "units", "direction", "model", "revision", "concept_id", "text",
                "status", "source", "author", "source_event_id", "source_evidence_ids", "semantic_mapping", "class_binding", "native_shape", "image_shape",
                "observation_id", "geometry", "artifact_coordinates", "input_observation")
        result = {key: record[key] for key in keys if key in record}
        result["artifacts"] = [{k: a[k] for k in ("path", "sha256", "role", "media_type") if k in a}
                               for a in record.get("artifacts", ())]
        return result

    def base_pair(self):
        records, images, observations = [], [], []
        for identifier in self.current:
            observation = self._observation(identifier)
            observations.append(observation)
            record, image = self._observation_evidence(observation)
            records.append(self._public(record))
            images.append(image)
        support_record, support_image = self._pair_support(observations)
        records.append(self._public(support_record))
        images.append(support_image)
        return {"status": "ok", "evidence": records, "images": images,
                "comparability": self.case.get("comparability", "unknown"),
                "limitations": self.case.get("limitations", []), "usage": {}}

    def _pair_support(self, observations):
        """Expose one labelled geometric map, alongside both untouched RGB images."""
        supports = []
        shape = tuple(observations[0]["geometry"]["crop_shape"])
        for observation in observations:
            if tuple(observation["geometry"]["crop_shape"]) != shape:
                raise ValueError("Compared observations need the same fixed region grid")
            path = Path(observation["support_path"])
            if sha256(path) != observation["support_sha256"]:
                raise ValueError("Observation geometric support changed")
            if path.suffix == ".npy":
                support = np.load(path, allow_pickle=False)
                if support.dtype != bool:
                    raise ValueError("Stored numpy geometric support must be boolean")
            else:
                with Image.open(path) as image:
                    support = np.asarray(image)
                if support.ndim != 2 or not np.isin(support, [0, 1, 255]).all():
                    raise ValueError("Stored geometric support must be a binary grid")
                support = support > 0
            if support.shape != shape:
                raise ValueError("Geometric support is not on the region grid")
            supports.append(support)
        common = supports[0] & supports[1]
        source_only = supports[1] & ~supports[0]
        reference_only = supports[0] & ~supports[1]
        if self.case.get("support_path"):
            with Image.open(self.case["support_path"]) as image:
                prepared = np.asarray(image) > 0
            if prepared.shape != common.shape or not np.array_equal(prepared, common):
                raise ValueError("Prepared common support disagrees with observation supports")
        identity = {"version": 1, "region_id": self.case["region_id"], "observations": self.current,
                    "support_sha256": [row["support_sha256"] for row in observations]}
        directory = self.out / "pair_support"
        directory.mkdir(exist_ok=True)
        arrays = {"common_support": common, "source_only_support": source_only,
                  "reference_only_support": reference_only}
        artifacts = []
        for role, mask in arrays.items():
            path = directory / (role + ".npy")
            if path.exists():
                if not np.array_equal(np.load(path, allow_pickle=False), mask):
                    raise ValueError("Saved pair support changed")
            else:
                np.save(path, mask, allow_pickle=False)
            artifacts.append(self._artifact(path, role))
        # Four exclusive categories, on exactly the same pixel grid as the RGB.
        colors = {"common_support": [40, 190, 70], "source_only_first_observation": [240, 40, 180],
                  "reference_only": [40, 100, 230], "neither_observed": [0, 0, 0]}
        visual = np.zeros((*shape, 3), np.uint8)
        visual[common] = colors["common_support"]
        visual[source_only] = colors["source_only_first_observation"]
        visual[reference_only] = colors["reference_only"]
        preview = directory / "geometric_support_categories.png"
        if preview.exists():
            with Image.open(preview) as image:
                if not np.array_equal(np.asarray(image), visual):
                    raise ValueError("Saved support visualization changed")
        else:
            Image.fromarray(visual).save(preview)
        artifacts.append(self._artifact(preview, "geometric_support_categories", "image/png"))
        region = self.store.get_region(self.case["region_id"])
        record = {"evidence_id": "pair-support-" + digest(identity)[:32], "kind": "measurement",
                  "method": "pair_geometric_support", "region_id": self.case["region_id"],
                  "observation_ids": self.current, "available_at": self.case["cutoff"],
                  "coordinate_system": "region_crop_pixels", "image_shape": list(shape),
                  "geometry": {"crop_shape": list(shape), "bbox_xyxy": region["bbox_xyxy"],
                               "anchor_sha256": region["anchor_sha256"]},
                  "summary": {"visualization": "Categorical geometric support; not RGB, not physical occlusion",
                      "legend_rgb": colors, "common_pixels": int(common.sum()),
                      "source_only_pixels": int(source_only.sum()), "reference_only_pixels": int(reference_only.sum()),
                      "neither_pixels": int((~(supports[0] | supports[1])).sum()),
                      "decision_scope": "Only common support can establish temporal change or absence of change",
                      "first_observation_rule": "Source-only pixels have no reference observation; not temporal change",
                      "localization_rule": "A definitive localized verdict requires full common support in the canonical bbox envelope"},
                  "artifacts": artifacts}
        self._remember(record)
        return record, {"type": "image", "path": str(preview), "evidence_id": record["evidence_id"]}

    def execute(self, name, arguments, timeout_seconds=None):
        started = time.monotonic()
        if name not in TOOL_SCHEMAS or not isinstance(arguments, dict):
            return {"status": "error", "error": "Unknown tool or invalid arguments", "evidence": [],
                    "images": [], "usage": {"elapsed_seconds": time.monotonic() - started}}
        allowed = set(TOOL_SCHEMAS[name]["arguments"])
        if set(arguments) - allowed:
            return {"status": "error", "error": "Unexpected tool argument", "evidence": [], "images": [],
                    "usage": {"elapsed_seconds": time.monotonic() - started}}
        self._timeouts = timeout_seconds
        self._model_usage = {}
        self._model_source_usage = None
        try:
            result = getattr(self, "_" + name)(**arguments)
        except (ValueError, KeyError, OSError, TypeError, RuntimeError, ImportError) as exc:
            result = {"status": "error", "error": f"{type(exc).__name__}: {exc}", "evidence": [], "images": [],
                      "usage": self._model_usage, "source_usage": self._model_source_usage}
        finally:
            self._timeouts = None
        result.setdefault("evidence", [])
        result.setdefault("images", [])
        result.setdefault("usage", {})
        result["usage"]["elapsed_seconds"] = time.monotonic() - started
        return result

    def _open_pair(self):
        return self.base_pair()

    def _open_crop(self, observation_id, view="crop"):
        record, image = self._observation_evidence(self._observation(observation_id), view)
        return {"status": "ok", "evidence": [self._public(record)], "images": [image]}

    def _get_history(self, limit=3):
        limit = _limit(limit, 3)
        history = self.store.history(self.case["region_id"], self.case["cutoff"],
                                     exclude_observation_ids=self.current)
        rows = history.get("observations", [])[-limit:]
        records, images = [], []
        for observation in rows:
            self.allowed.add(str(observation["observation_id"]))
            record, image = self._observation_evidence(observation)
            records.append(self._public(record))
            images.append(image)
        for statement in history.get("evidence", [])[-limit:]:
            for identifier in statement.get("observation_ids", ()):
                self._authorize(identifier)
                self.allowed.add(identifier)
            record = {key: statement[key] for key in ("evidence_id", "kind", "region_id", "observation_ids",
                "available_at", "summary", "artifacts", "model", "revision", "source", "source_event_id",
                "source_evidence_ids", "coordinate_system", "author") if key in statement}
            self._remember(record)
            records.append(self._public(record))
        return {"status": history.get("status", "ok" if rows else "no_history"),
                "evidence": records, "images": images, "total_available": len(history.get("observations", [])),
                "selected_count": len(rows), "selection": "latest permitted prior observations", "limit": limit}

    def _get_maps(self, method=None, limit=2):
        if type(limit) is not int or not 0 <= limit <= 10:
            raise ValueError("Map preview limit must be 0..10")
        if method is not None:
            method = _text(method, "method", 256)
        records, images = [], []
        rows = [self.store.get_evidence(item) if isinstance(item, str) else dict(item)
                for item in self.method_evidence]
        # Keep the configured fusion visible even under the fixed-context preview
        # budget; independent methods retain all of their summaries and artifacts.
        for row in sorted(rows, key=lambda row: (
                row.get("method", row.get("parameters", {}).get("method")) != "fusion", str(row.get("evidence_id", "")))):
            if row.get("kind") not in {"prediction", "measurement"}:
                raise ValueError("Only prediction/measurement evidence is available as maps")
            parameters = row.get("parameters", {})
            summary = row.get("summary", {})
            summary = summary if isinstance(summary, dict) else {}
            name = row.get("method", parameters.get("method", summary.get("method", row.get("model"))))
            if method is not None and method != name:
                continue
            if row.get("region_id") != self.case["region_id"]:
                raise ValueError("Method evidence belongs to another region")
            if not set(row.get("observation_ids", ())).issubset(self.current):
                raise ValueError("Method evidence does not describe this pair")
            artifacts = []
            for artifact in row.get("artifacts", ()):
                if any(word in artifact.get("role", "").lower() for word in ("ground_truth", "annotation", "label", "packed")):
                    raise ValueError("Annotation artifacts are not tool inputs")
                artifacts.append(self._artifact(artifact["path"], artifact["role"], artifact.get("media_type")))
                if artifact.get("sha256") and artifact["sha256"] != artifacts[-1]["sha256"]:
                    raise ValueError("Method artifact hash changed")
            record = {"evidence_id": row["evidence_id"], "kind": row["kind"], "region_id": row["region_id"],
                      "observation_ids": list(row.get("observation_ids", ())), "method": name,
                      "summary": row.get("summary", ""), "artifacts": artifacts,
                      "coordinate_system": row.get("coordinate_system", "region_crop_pixels"),
                      "available_at": row.get("available_at")}
            for field in ("output_kind", "units", "direction", "semantic_mapping", "status"):
                if field in row or field in parameters or field in summary:
                    record[field] = row.get(field, parameters.get(field, summary.get(field)))
            for field in ("model", "revision"):
                if field in row:
                    record[field] = row[field]
            for field in ("source", "source_evidence_ids", "geometry"):
                if field in row or field in parameters:
                    record[field] = row.get(field, parameters.get(field))
            self._remember(record)
            records.append(self._public(record))
            for artifact in artifacts:
                if artifact["role"] in {"preview", "overlay", "score_preview", "mask_preview", "preview_path", "overlay_path"}:
                    with Image.open(artifact["path"]) as image:
                        image.verify()
                    images.append({"type": "image", "path": artifact["path"], "evidence_id": record["evidence_id"]})
        return {"status": "ok" if records else "unavailable", "evidence": records, "images": images[:limit],
                "selected_preview_count": min(limit, len(images)), "total_preview_count": len(images),
                "selection": "fusion preview first, then evidence-ID order; all method summaries retained", "limit": limit}

    def _filters(self):
        observations = [self._observation(identifier) for identifier in self.current]
        return {"cutoff": self.case["cutoff"], "allowed_splits": ("train",),
                "exclude_buildings": (self.case["building_id"],), "exclude_observation_ids": tuple(self.current),
                "exclude_hashes": tuple(row[key] for row in observations for key in ("original_sha256", "crop_sha256"))}

    def _search_similar(self, query=None, observation_id=None, representation="visual", limit=5):
        limit = _limit(limit)
        if self.search is None:
            return {"status": "unavailable", "reason": "Search corpus unavailable"}
        if representation not in {"visual", "history", "joint_image_text"}:
            raise ValueError("Unknown retrieval representation")
        filters = self._filters()
        usage = {}
        source_usage = None
        if representation == "joint_image_text":
            if observation_id is not None:
                raise ValueError("Joint text-to-image search expects query, not observation_id")
            if self.remaining_model_calls is not None and self.remaining_model_calls < 1:
                return {"status": "budget_exhausted", "reason": "model_call_budget_exhausted"}
            request = {"op": "embed", "texts": [_text(query, "query")], "task": "retrieval.query"}
            if self._timeouts is not None:
                request["timeout_seconds"] = self._timeouts
            encoded = self.pool.call("jina", request)
            usage = {**encoded.get("usage", {}), "model_requests": 1}
            self._model_usage = usage
            source_usage = self._model_source_usage = encoded.get("source_usage")
            if encoded.get("status") != "ok":
                return {"status": "unavailable", "error": encoded.get("error"), "usage": usage,
                        "reason": "Verified Jina model unavailable; lexical search remains a separate operation"}
            if encoded["space"].get("representation") != "joint_image_text":
                raise ValueError("Text-to-image search requires a verified joint embedding space")
            result = self.search.similar(encoded["vectors"][0], encoded["space"], limit=limit, **filters)
        elif observation_id is not None:
            observation = self._observation(observation_id)
            candidates = self.search.list_embeddings(observation_id=observation["observation_id"])
            if self.embedding_role:
                settings = getattr(self.pool, "models", {}).get(self.embedding_role, {})
                if self.embedding_role == "jina":
                    candidates = [r for r in candidates if r["space"].get("model") == settings.get("model_id", "jinaai/jina-clip-v2")
                                  and r["space"].get("revision") == settings.get("revision")]
                elif self.embedding_role == "dino":
                    candidates = [r for r in candidates if r["space"].get("model") == "dinov2_vits14"]
                    checkpoint = settings.get("checkpoint_path")
                    if checkpoint and Path(checkpoint).is_file():
                        expected_revision = sha256(checkpoint)
                        candidates = [r for r in candidates if r["space"].get("revision") == expected_revision]
            if len(candidates) > 1:
                raise ValueError("Multiple cached spaces match; configure a unique model revision")
            embedding = candidates[0] if candidates else None
            if embedding is None and self.embedding_role:
                if self.remaining_model_calls is not None and self.remaining_model_calls < 1:
                    return {"status": "budget_exhausted", "reason": "model_call_budget_exhausted"}
                request = {"op": "embed", "image_paths": [observation["crop_path"]], "geometry": observation["geometry"]}
                if self.embedding_role == "dino":
                    request["support_paths"] = [observation["support_path"]]
                if self._timeouts is not None:
                    request["timeout_seconds"] = self._timeouts
                encoded = self.pool.call(self.embedding_role, request)
                usage = {**encoded.get("usage", {}), "model_requests": 1}
                self._model_usage = usage
                source_usage = self._model_source_usage = encoded.get("source_usage")
                if encoded.get("status") != "ok":
                    return {"status": "error", "error": encoded.get("error"), "usage": usage}
                embedding = self.search.add_embedding(observation["observation_id"], encoded["vectors"][0], encoded["space"])
            if embedding is None:
                return {"status": "unavailable", "reason": "No verified visual embedding; use text search or configure an encoder"}
            if representation == "history":
                result = self.search.similar_history(observation["region_id"], space=embedding["space"], limit=limit, **filters)
            else:
                result = self.search.similar(embedding["vector"], embedding["space"], limit=limit, **filters)
        else:
            if representation == "history":
                raise ValueError("History search needs observation_id and verified visual features")
            result = self.search.text(_text(query, "query"), limit=limit, **filters)
        records = []
        for match in result.get("matches", ()):
            identifier = match.get("observation_id")
            if identifier is None:
                # Historical matches expose only members selected by the search's causal policy.
                identifiers = match.get("observation_ids", [])
            else:
                identifiers = [identifier]
            for identifier in identifiers:
                self.allowed.add(str(identifier))
                observation = self._observation(identifier)
                record, _ = self._observation_evidence(observation)
                records.append(self._public(record))
        return {"status": result.get("status", "ok"), "evidence": records,
                "matches": [{k: row[k] for k in ("observation_id", "region_id", "observation_ids", "score", "distance") if k in row}
                            for row in result.get("matches", ())],
                "corpus_version": result.get("corpus_version"), "cache_key": result.get("cache_key"),
                "source_usage": source_usage,
                "usage": {**usage, "cache_hits": int(result.get("cache_hit", False)) + int(usage.get("cache_hits", 0))}}

    def _get_knowledge(self, query="facade", concept_id=None, limit=5):
        if self.knowledge is None:
            return {"status": "unavailable", "reason": "No knowledge corpus"}
        result = self.knowledge.search(_text(query, "query"), self.case["cutoff"], limit=_limit(limit),
                                       concept_ids=(() if concept_id is None else (_text(concept_id, "concept_id", 256),)))
        records = []
        for fragment in result.get("fragments", ()):
            source = {**fragment.get("source", {}),
                      **{key: fragment[key] for key in ("document_id", "fragment_id") if key in fragment}}
            if fragment.get("locator") is not None:
                source["fragment_locator"] = fragment["locator"]
            record = {"evidence_id": "knowledge-" + digest([self.case["region_id"], fragment, source])[:32], "kind": "knowledge",
                      "region_id": self.case["region_id"], "summary": fragment.get("text", ""),
                      "available_at": fragment.get("available_at"), "source": source, "artifacts": []}
            self._remember(record)
            records.append(self._public(record))
        concepts = result.get("concepts", [])
        related = self.knowledge.related(concept_id, self.case["cutoff"]) if concept_id else None
        return {"status": result.get("status", "ok"), "evidence": records, "concepts": concepts,
                "related": related, "corpus_version": result.get("corpus_version"),
                "usage": {"cache_hits": int(result.get("cache_hit", False))}}

    def _sam3(self, observation_id, text, concept_id=None):
        if self.remaining_model_calls is not None and self.remaining_model_calls < 1:
            return {"status": "budget_exhausted", "reason": "model_call_budget_exhausted"}
        observation = self._observation(observation_id)
        text = _text(text, "text", 256)
        concept = None
        if concept_id is not None:
            if self.knowledge is None:
                raise ValueError("A concept binding needs the project taxonomy")
            concept_id = _text(concept_id, "concept_id", 256)
            permitted = self.knowledge.search(text, self.case["cutoff"], concept_ids=(concept_id,))
            concept = next((row for row in permitted.get("concepts", ()) if row["concept_id"] == concept_id), None)
            if concept is None:
                raise ValueError("Concept was unavailable at the case cutoff")
            names = [concept.get("name", ""), *concept.get("synonyms", [])]
            if text.casefold() not in {str(name).casefold() for name in names}:
                raise ValueError("Text differs from the bound concept name/aliases")
        signature = digest({"observation": observation["observation_id"], "crop_sha256": observation["crop_sha256"],
                            "geometry": observation["geometry"], "text": text, "concept_id": concept_id,
                            "available_at": self.case["cutoff"],
                            "model": getattr(self.pool, "models", {}).get("sam3", {}),
                            "tool_implementation_sha256": sha256(__file__)})
        destination = self.out / "sam3" / signature[:24]
        request = {"op": "segment", "image_path": observation["crop_path"], "text": text,
                   "out": str(destination), "geometry": observation["geometry"]}
        if self._timeouts is not None:
            request["timeout_seconds"] = self._timeouts
        result = self.pool.call("sam3", request)
        self._model_usage = {**result.get("usage", {}), "model_requests": 1}
        self._model_source_usage = result.get("source_usage")
        if result.get("status") not in {"ok", "empty"}:
            return {"status": "error", "error": result.get("error", "SAM3 failure"),
                    "usage": self._model_usage, "source_usage": self._model_source_usage}
        masks = np.load(result["masks_path"], allow_pickle=False)
        if isinstance(masks, np.lib.npyio.NpzFile):
            with masks as archive:
                masks = archive["masks"].copy()
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        with Image.open(observation["crop_path"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        if masks.ndim != 3 or masks.shape[1:] != rgb.shape[:2] or masks.dtype != bool:
            raise ValueError("SAM3 masks must be boolean instance masks on the native crop grid")
        support_path = Path(observation["support_path"])
        if support_path.suffix == ".npy":
            support = np.load(support_path, allow_pickle=False)
        else:
            with Image.open(support_path) as image:
                support = np.asarray(image) > 0
        if support.dtype != bool or support.shape != rgb.shape[:2]:
            raise ValueError("Stored support is not on the crop grid")
        union = masks.any(axis=0) & support
        geometry = dict(observation["geometry"])
        if "native_shape" not in geometry:
            with Image.open(observation["original_path"]) as original:
                geometry["native_shape"] = [original.height, original.width]
        native, native_support = remap_mask(union, support, geometry)
        destination.mkdir(parents=True, exist_ok=True)
        mapped = destination / "mapped_masks.npz"
        np.savez_compressed(mapped, union_crop=union, union_observation=native,
                            observation_support=native_support,
                            crop_to_source=np.asarray(observation["geometry"]["crop_to_source"]))
        overlay = rgb.copy()
        overlay[union] = np.rint(.5 * rgb[union] + .5 * np.array([255, 40, 40])).astype(np.uint8)
        overlay_path = destination / "overlay.png"
        Image.fromarray(overlay).save(overlay_path)
        provenance = result.get("provenance", {})
        options = provenance.get("options", {})
        crop_frame = {"coordinate_system": "region_crop_pixels", "region_id": observation["region_id"]}
        native_frame = {"coordinate_system": "observation_native", "observation_id": observation["observation_id"]}
        artifact_coordinates = {"instance_masks": crop_frame, "overlay": crop_frame,
            "mapped_masks": {"union_crop": crop_frame, "union_observation": native_frame,
                             "observation_support": native_frame}}
        record = {"evidence_id": "sam3-" + digest({"signature": signature, "provenance": provenance})[:32],
                  "kind": "prediction", "region_id": observation["region_id"],
                  "observation_id": observation["observation_id"],
                  "observation_ids": [observation["observation_id"]], "method": "sam3_text_refinement",
                  "status": "empty" if not masks.shape[0] else "ok", "text": text,
                  "available_at": self.case["cutoff"],
                  "concept_id": concept_id, "class_binding": concept_id if concept else None,
                  "summary": "Text-conditioned mask prediction; not temporal change ground truth",
                  "coordinate_system": "region_crop_pixels", "geometry": geometry,
                  "image_shape": list(rgb.shape[:2]), "artifact_coordinates": artifact_coordinates,
                  "input_observation": {key: observation.get(key) for key in ("observation_id", "region_id",
                      "building_id", "view_id", "date", "date_status", "available_at", "original_sha256",
                      "crop_sha256", "support_sha256", "record_sha256", "source_event_id")},
                  "native_shape": list(native.shape), "model": options.get("model_id", provenance.get("model", "sam3")),
                  "revision": options.get("revision", provenance.get("revision")), "provenance": provenance,
                  "artifacts": [self._artifact(result["masks_path"], "instance_masks"),
                                self._artifact(mapped, "mapped_masks"), self._artifact(overlay_path, "overlay", "image/png")]}
        if result.get("probabilities_path"):
            record["artifacts"].append(self._artifact(result["probabilities_path"], "probabilities"))
            record["artifact_coordinates"]["probabilities"] = crop_frame
        self._remember(record)
        return {"status": record["status"], "evidence": [self._public(record)],
                "images": [{"type": "image", "path": str(overlay_path), "evidence_id": record["evidence_id"]}],
                "scores": result.get("scores", []), "boxes": result.get("boxes", []),
                "usage": self._model_usage, "source_usage": self._model_source_usage}
