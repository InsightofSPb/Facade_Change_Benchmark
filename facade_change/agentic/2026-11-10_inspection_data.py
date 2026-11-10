"""Frozen RGB inventory and original-pixel inspection for the 2026-11-10 pilot."""
from __future__ import annotations

import copy
import importlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..history_features import Fingerprints, _history_document, _inside, digest
from ..io import load_rgb, read_json, sha256, write_json
from .config import QWEN_REVISION
from .models import DEFAULTS
from .pipeline import _matrix, _support, _validate_prepared, resolve_pair_spec
from .store import causally_available, reject_forbidden, temporal_value
from .tools import ToolService


VERSION = "2026-11-10-v1"
POLICIES = ("pair", "fixed_history", "heuristic", "agent")
CAPS = dict(max_actions=2, max_model_calls=5, max_images=8, max_total_images=32,
            max_input_tokens=16384, max_total_input_tokens=65536,
            max_output_tokens=1024, max_total_output_tokens=4096,
            max_invalid_replies=1, max_same_action=1, max_seconds=1800)
SCOPE = "AnyChange-guided inspection inside supplied registered canonical regions"
PROTOCOL_KEYS = {"schema_version", "kind", "version", "scope", "policies", "caps",
                 "history_manifest_sha256", "questions_sha256", "archive_available_at",
                 "task_mode", "min_history_observations", "max_native_crop_pixels",
                 "enable_sam3", "assessability_reviewed_against"}


def protocol_document(history_sha, questions_sha, *, available_at="observation_date"):
    return dict(schema_version=1, kind="facade_inspection_protocol", version=VERSION,
                scope=SCOPE, policies=list(POLICIES), caps=CAPS.copy(),
                history_manifest_sha256=history_sha, questions_sha256=questions_sha,
                archive_available_at=available_at, task_mode="retrospective",
                min_history_observations=2, max_native_crop_pixels=1048576,
                enable_sam3=False, assessability_reviewed_against="frozen_permitted_rgb_history_universe")


def validate_protocol(value):
    reject_forbidden(value)
    if not isinstance(value, dict) or set(value) != PROTOCOL_KEYS:
        raise ValueError("Protocol fields must be exactly: " + ", ".join(sorted(PROTOCOL_KEYS)))
    expected = protocol_document(value["history_manifest_sha256"], value["questions_sha256"])
    for key in ("schema_version", "kind", "version", "scope", "caps", "enable_sam3",
                "min_history_observations", "assessability_reviewed_against"):
        if value[key] != expected[key] or type(value[key]) is not type(expected[key]):
            raise ValueError(f"Protocol {key}: expected {expected[key]!r}, received {value[key]!r}")
    if any(type(value["caps"][key]) is not type(limit) for key, limit in CAPS.items()):
        raise ValueError("Protocol caps must use integer values, never booleans")
    if value["policies"] not in [list(POLICIES), ["heuristic", "agent"]]:
        raise ValueError("Choose all four policies or the two-case heuristic/agent smoke profile")
    for key in ("history_manifest_sha256", "questions_sha256"):
        if not isinstance(value[key], str) or len(value[key]) != 64 or any(c not in "0123456789abcdef" for c in value[key]):
            raise ValueError(f"{key} requires the exact existing input SHA-256")
    if value["task_mode"] not in {"retrospective", "as_of"}:
        raise ValueError("Explicit retrospective/as_of task_mode required")
    if value["archive_available_at"] != "observation_date" and temporal_value(value["archive_available_at"]) is None:
        raise ValueError("Explicit archive availability required")
    if type(value["max_native_crop_pixels"]) is not int or value["max_native_crop_pixels"] < 1:
        raise ValueError("max_native_crop_pixels must be a positive integer")
    return copy.deepcopy(value)


def validate_config(config, protocol, *, check_resources=True):
    """No model loading. Report actual conflicts; never reconstruct a local config."""
    errors = []
    if not isinstance(config.get("method_options"), dict) or not isinstance(config.get("budgets"), dict):
        raise ValueError("method_options and budgets must be explicit JSON objects")
    options = config.get("method_options", {}).get("anychange", {})
    if not isinstance(options, dict):
        raise ValueError("method_options.anychange must be a JSON object")
    if "anychange" not in config.get("methods", []):
        errors.append("methods must include anychange")
    for key in ("source_root", "checkpoint_path"):
        if not options.get(key):
            errors.append("method_options.anychange." + key + " is missing")
    allowed = {"source_root", "checkpoint_path", "device", "trust_checkpoint", "points_per_batch", "worker_python"}
    if set(options) - allowed:
        errors.append("Unknown AnyChange options: " + str(sorted(set(options) - allowed)))
    qwen = {**DEFAULTS["qwen"], **config.get("models", {}).get("qwen", {})}
    unknown = set(qwen) - set(DEFAULTS["qwen"]) - {"python", "local_files_only", "checkpoint_path", "source_root"}
    if unknown:
        errors.append("Unknown models.qwen options: " + str(sorted(unknown)))
    for key in ("max_image_pixels", "image_min_pixels", "image_max_pixels"):
        if type(qwen.get(key)) is not int or qwen[key] < 1:
            errors.append(f"models.qwen.{key}={qwen.get(key)!r}; expected positive integer")
    if all(type(qwen.get(k)) is int for k in ("image_min_pixels", "image_max_pixels")) and qwen["image_min_pixels"] > qwen["image_max_pixels"]:
        errors.append("models.qwen.image_min_pixels exceeds image_max_pixels")
    if not qwen.get("python"):
        errors.append("models.qwen.python is missing")
    pinned = dict(model_id="Qwen/Qwen3.5-4B", revision=QWEN_REVISION, precision="bfloat16",
                  attention="sdpa", device="cuda", enable_thinking=False, do_sample=False, local_files_only=True)
    for key, expected in pinned.items():
        if qwen.get(key) != expected:
            errors.append(f"models.qwen.{key}={qwen.get(key)!r}; required {expected!r}")
    for budget, model in (("max_images", "max_images"), ("max_input_tokens", "max_input_tokens"),
                          ("max_output_tokens", "max_new_tokens")):
        actual, required = qwen.get(model), protocol["caps"][budget]
        if type(actual) is not int or actual < required:
            errors.append(f"models.qwen.{model}={actual!r} < inspection {budget}={required}")
    for key, required in protocol["caps"].items():
        actual = config.get("budgets", {}).get(key)
        if not isinstance(actual, (int, float)) or isinstance(actual, bool) or actual < required:
            errors.append(f"budgets.{key}={actual!r} < inspection {key}={required}")
    if config.get("gpu_mode") != "sequential":
        errors.append("gpu_mode must be sequential for this pilot")
    if config.get("worker_timeout", 0) < protocol["caps"]["max_seconds"]:
        errors.append(f"worker_timeout={config.get('worker_timeout')} < max_seconds={protocol['caps']['max_seconds']}")
    if type(qwen["max_image_pixels"]) is int and protocol["max_native_crop_pixels"] > qwen["max_image_pixels"]:
        errors.append(f"max_native_crop_pixels={protocol['max_native_crop_pixels']} > models.qwen.max_image_pixels={qwen['max_image_pixels']}")
    if check_resources:
        for label, path in (("models.qwen.python", qwen.get("python")),
                            ("method_options.anychange.worker_python", options.get("worker_python")),
                            ("method_options.anychange.checkpoint_path", options.get("checkpoint_path"))):
            if path and not Path(path).is_file():
                errors.append(f"Missing {label}: {path}")
        root = options.get("source_root")
        if root and not (Path(root) / "torchange/models/segment_any_change/anychange.py").is_file():
            errors.append(f"Missing original AnyChange implementation under source_root={root}")
    if errors:
        raise ValueError("Configuration conflicts:\n" + "\n".join(errors))
    return qwen


def _availability(row, protocol):
    value = protocol["archive_available_at"]
    return str(row["year"]) if value == "observation_date" else value


def _observation(root, track, row, protocol, fp):
    x, y, width, height = track["bbox_xywh"]
    matrix = _matrix(row["alignment"]["matrix"])
    shift = np.array([[1., 0, -x], [0, 1., -y], [0, 0, 1.]])
    transform = shift @ matrix
    paths = dict(original=Path(row["original_image_path"]).expanduser().resolve(),
                 crop=_inside(root, row["rgb"]), support=_inside(root, row["support"]))
    hashes = dict(original=row["image_sha256"], crop=row["rgb_sha256"], support=row["support_sha256"])
    for key in paths:
        fp.add(paths[key], hashes[key])
    with Image.open(paths["original"]) as image:
        native_shape = [image.height, image.width]
    with Image.open(paths["crop"]) as image:
        if image.mode != "RGB" or image.size != (width, height):
            raise ValueError("Canonical RGB shape/type mismatch")
    support = _support(paths["support"], (height, width))
    if not support.any():
        raise ValueError("Observation has no geometric coverage")
    if row["observation_id"] != digest([track["track_id"], row["image_sha256"]])[:24]:
        raise ValueError("Observation ID differs from canonical region/image identity")
    return dict(observation_id=row["observation_id"], region_id=track["track_id"],
        building_id=track["building_id"], view_id=track["view_id"], split=track["split"],
        image_id=str(row["image_id"]), date=str(row["year"]), date_status="confirmed",
        available_at=_availability(row, protocol),
        **{key + "_path": str(path) for key, path in paths.items()},
        **{key + "_sha256": value for key, value in hashes.items()},
        geometry=dict(source_to_anchor=matrix.tolist(), source_to_crop=transform.tolist(),
            crop_to_source=np.linalg.inv(transform).tolist(), crop_to_anchor=np.linalg.inv(shift).tolist(),
            native_shape=native_shape, crop_shape=[height, width], bbox_xyxy=[x, y, x+width, y+height],
            coordinate_system="native anchor pixel centres; region crop origin at box top-left"),
        visibility="unknown", correspondence_status="aligned",
        provenance=dict(availability_assumption=protocol["archive_available_at"])), int(support.sum())


def resolve_inputs(questions_path, histories, config, protocol, *, check_resources=True):
    qa = importlib.import_module(".2026-11-10_questions", __package__)
    protocol = validate_protocol(protocol)
    validate_config(config, protocol, check_resources=check_resources)
    root = Path(histories).expanduser().resolve()
    root = root.parent if root.name == "manifest.json" else root
    fp = Fingerprints()
    fp.add(questions_path, protocol["questions_sha256"])
    public = read_json(questions_path)
    reject_forbidden(public)
    qa.validate_public({**public, "questions": []})
    if public["schema_version"] != 1:
        raise ValueError("Unsupported public question schema_version")
    if not isinstance(public["questions"], list):
        raise ValueError("Public questions must be a list")
    if any(not isinstance(q, dict) for q in public["questions"]):
        raise ValueError("Each public question must be a JSON object")
    ids = [q.get("question_id") for q in public["questions"]]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Public question IDs must be distinct nonempty strings")
    document, history_sha = _history_document(root, fp)
    if history_sha != protocol["history_manifest_sha256"]:
        raise ValueError("Canonical history manifest SHA differs from frozen protocol")
    # Read only split/identity metadata across the manifest, never held-out pixels/labels.
    hashes, originals = {}, {}
    for track in document["tracks"]:
        for row in track["observations"]:
            for field in ("image_sha256", "rgb_sha256"):
                if hashes.setdefault(row[field], track["split"]) != track["split"]:
                    raise ValueError("Canonical image/crop hash split conflict")
            identity = (track["building_id"], track["view_id"], row["year"])
            if originals.setdefault(row["image_sha256"], identity) != identity:
                raise ValueError("Canonical original image building/view/date conflict")
    tracks = {t["track_id"]: t for t in document["tracks"]}
    resolved, eligibility = [], []
    if not public["questions"]:
        raise ValueError("An approved, nonempty public cohort is required")
    for question in public["questions"]:
        identifier = question["question_id"]
        report = dict(question_id=identifier, status="ineligible", eligible_history_count=0)
        try:
            qa.validate_public({**public, "questions": [question]})
            if question["task"] != "temporal_change" or len(question["observations"]) != 2:
                raise ValueError("Select temporal_change questions with exactly two compared observations")
            if question["split"] not in {"train", "val"}:
                raise ValueError("TEST/unassigned are forbidden for this development pilot")
            if question["task_mode"] != protocol["task_mode"]:
                raise ValueError("Question task_mode differs from protocol")
            track = tracks[question["region_id"]]
            x, y, width, height = track["bbox_xywh"]
            identity = {k: track[k] for k in ("building_id", "view_id", "anchor_sha256", "bbox_xywh")}
            if track["track_id"] != digest(identity)[:24]:
                raise ValueError("Region ID differs from persistent canonical identity")
            if any(track[k] != question[k] for k in ("building_id", "view_id", "split")):
                raise ValueError("Question region/building/view/split mismatch")
            frame = dict(coordinate_system="region_crop_pixels", region_id=track["track_id"],
                shape=[height, width], anchor_sha256=track["anchor_sha256"], anchor_bbox_xyxy=[x, y, x+width, y+height])
            if question["frame"] != frame:
                raise ValueError("Question frame differs from canonical region")
            lookup = {r["observation_id"]: r for r in track["observations"]}
            observations, current = {}, []
            coverage = {}
            for supplied in question["observations"]:
                row = lookup[supplied["observation_id"]]
                obs, covered = _observation(root, track, row, protocol, fp)
                geometry = {k: obs["geometry"][k] for k in ("source_to_crop", "crop_to_source", "crop_shape")}
                geometry["anchor_sha256"] = track["anchor_sha256"]
                expected = dict(observation_id=obs["observation_id"], date=obs["date"], available_at=obs["available_at"],
                    date_status="confirmed", rgb_path=obs["crop_path"], rgb_sha256=obs["crop_sha256"],
                    support_path=obs["support_path"], support_sha256=obs["support_sha256"], geometry=geometry)
                if supplied != expected:
                    raise ValueError("Public observation ID/hash/date/availability/geometry does not match canonical manifest")
                if not causally_available(obs, question["knowledge_cutoff"]):
                    raise ValueError("Compared observation unavailable at knowledge_cutoff")
                observations[obs["observation_id"]] = obs
                coverage[obs["observation_id"]] = covered
                current.append(obs["observation_id"])
            dates = [observations[i]["date"] for i in current]
            if question["task_dates"] != dates or temporal_value(dates[0]) >= temporal_value(dates[1]):
                raise ValueError("Compared chronology/task_dates disagree")
            history, excluded = [], []
            for row in track["observations"]:
                oid = row["observation_id"]
                if oid in current:
                    continue
                available = _availability(row, protocol)
                if temporal_value(row["year"]) >= temporal_value(dates[0]):
                    excluded.append(dict(observation_id=oid, reason="not_strictly_before_reference"))
                elif available is None or temporal_value(available) > temporal_value(question["knowledge_cutoff"]):
                    excluded.append(dict(observation_id=oid, reason="unavailable_at_knowledge_cutoff"))
                else:
                    obs, covered = _observation(root, track, row, protocol, fp)
                    observations[oid], coverage[oid] = obs, covered
                    history.append(oid)
            history.sort(key=lambda oid: (temporal_value(observations[oid]["date"]), oid), reverse=True)
            report.update(eligible_history_count=len(history), eligible_history_ids=history, excluded=excluded,
                          distinct_history_dates=len({observations[i]["date"] for i in history}))
            if len(history) < 2 or report["distinct_history_dates"] < 2:
                raise ValueError("Date-selection pilot requires two eligible historical dates in addition to the pair")
            if width*height > min(config["max_region_pixels"],
                    {**DEFAULTS["qwen"], **config["models"]["qwen"]}["max_image_pixels"]):
                raise ValueError("Canonical pair images exceed configured region/model pixel limits")
            for oid, obs in observations.items():
                native_limit = int(np.prod(obs["geometry"]["native_shape"]))
                geometry = crop_geometry(obs, [0, 0, width, height], native_limit)
                if "fixed_history" in protocol["policies"] and oid in history[:2] and int(np.prod(geometry["crop_shape"])) > protocol["max_native_crop_pixels"]:
                    raise ValueError("Fixed-history full-region native crop exceeds max_native_crop_pixels: " + oid)
            def image_spec(oid):
                obs = observations[oid]
                return dict(path=obs["original_path"], sha256=obs["original_sha256"], image_id=obs["image_id"],
                    date=obs["date"], date_status="confirmed", available_at=obs["available_at"],
                    native_shape=obs["geometry"]["native_shape"], crop_path=obs["crop_path"], crop_sha256=obs["crop_sha256"],
                    support_path=obs["support_path"], support_sha256=obs["support_sha256"],
                    source_to_anchor=obs["geometry"]["source_to_anchor"])
            pair = resolve_pair_spec(dict(case_id=identifier, building_id=track["building_id"], view_id=track["view_id"],
                split=track["split"], region_id=track["track_id"], anchor_sha256=track["anchor_sha256"],
                bbox_xyxy=frame["anchor_bbox_xyxy"], reference=image_spec(current[0]), source=image_spec(current[1]),
                prepared_provenance=dict(manifest=str(root / "manifest.json"), manifest_sha256=history_sha,
                                        track_id=track["track_id"], projection="canonical RGB/support only")))
            _validate_prepared(pair, frame["anchor_bbox_xyxy"], frame["anchor_sha256"])
            inventory = dict(version=VERSION, question_id=identifier, region_id=track["track_id"], frame=frame,
                knowledge_cutoff=question["knowledge_cutoff"], current_ids=current, history_ids=history,
                observations=observations, coverage_pixels=coverage,
                archive_available_at=protocol["archive_available_at"], caps=protocol["caps"],
                max_native_crop_pixels=protocol["max_native_crop_pixels"],
                allowed_actions={"inspect_region": "current pair together or one listed historical observation; any integer region box",
                                 "stop": "selection only; fresh judge still required"}, enable_sam3=False)
            reject_forbidden(inventory)
            resolved.append(dict(question=question, pair=pair, inventory=inventory))
            report["status"] = "eligible"
        except (ValueError, KeyError, TypeError, OSError) as exc:
            report["reason"] = f"{type(exc).__name__}: {exc}"
        eligibility.append(report)
    fp.verify()
    return dict(public=public, resolved=resolved, eligibility=eligibility, input_sha256=fp.files,
                status="eligible" if all(r["status"] == "eligible" for r in eligibility) else "ineligible",
                scope=SCOPE, counts=dict(questions=len(public["questions"]),
                    buildings=len({q["building_id"] for q in public["questions"]}),
                    manifest_tracks=len(document["tracks"])))


def proposals(mask, support):
    if mask.dtype != bool or mask.shape != support.shape or support.dtype != bool:
        raise ValueError("AnyChange needs a valid binary native map on geometric support")
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask & support).astype(np.uint8), connectivity=8)
    components = []
    for i in range(1, count):
        x, y, width, height, area = map(int, stats[i])
        box = [x, y, x+width, y+height]
        component_id = "component-" + digest([box, area, np.packbits(labels == i).tobytes().hex()])[:20]
        components.append(dict(proposal_id=component_id, bbox_xyxy=box, area_pixels=area,
                               min_envelope_dimension=min(width, height), component_label=i))
    components.sort(key=lambda c: (-c["area_pixels"], c["bbox_xyxy"][1], c["bbox_xyxy"][0],
                                    c["bbox_xyxy"][3], c["bbox_xyxy"][2], c["proposal_id"]))
    selected = components[:3]
    fallback = None if selected else dict(proposal_id="whole-region-fallback", bbox_xyxy=[0, 0, mask.shape[1], mask.shape[0]],
                                         is_anychange_detection=False)
    return dict(component_count=len(components), components=components, selected=selected,
                selected_ids=[c["proposal_id"] for c in selected], omitted=components[3:],
                omitted_pixels=sum(c["area_pixels"] for c in components[3:]), fallback=fallback,
                ordering="8-connected; area descending, top/left/bottom/right/ID ascending; no size filter; not confidence"), labels


def validate_action(inventory, observation_ids, bbox_xyxy):
    if not isinstance(observation_ids, list) or not (
        observation_ids == inventory["current_ids"] or
        len(observation_ids) == 1 and observation_ids[0] in inventory["history_ids"]):
        raise ValueError("inspect_region requires the ordered current pair or one permitted history ID")
    height, width = inventory["frame"]["shape"]
    if (not isinstance(bbox_xyxy, list) or len(bbox_xyxy) != 4 or any(type(v) is not int for v in bbox_xyxy)
            or not 0 <= bbox_xyxy[0] < bbox_xyxy[2] <= width
            or not 0 <= bbox_xyxy[1] < bbox_xyxy[3] <= height):
        raise ValueError("bbox_xyxy must be a nonempty integer half-open box inside region_crop_pixels")


def crop_geometry(observation, box, maximum):
    transform = _matrix(observation["geometry"]["crop_to_source"])
    _matrix(observation["geometry"]["source_to_crop"])
    if not np.allclose(transform @ observation["geometry"]["source_to_crop"], np.eye(3), atol=1e-7):
        raise ValueError("Crop/source transforms are not inverses")
    x0, y0, x1, y1 = box
    corners = np.array([[x0, y0, 1.], [x1, y0, 1.], [x1, y1, 1.], [x0, y1, 1.]]) @ transform.T
    if not np.isfinite(corners).all() or not (np.all(corners[:, 2] > 1e-12) or np.all(corners[:, 2] < -1e-12)):
        raise ValueError("Requested crop crosses a singular projective boundary")
    polygon = corners[:, :2] / corners[:, 2:]
    low, high = np.floor(polygon.min(axis=0)).astype(int), np.ceil(polygon.max(axis=0)).astype(int)
    height, width = observation["geometry"]["native_shape"]
    bounds = [max(0, int(low[0])), max(0, int(low[1])), min(width, int(high[0])), min(height, int(high[1]))]
    left, top, right, bottom = bounds
    if right <= left or bottom <= top or (right-left)*(bottom-top) > maximum:
        raise ValueError(f"Native enclosing crop is empty or exceeds max_native_crop_pixels={maximum}: {bounds}")
    offset = np.array([[1., 0, left], [0, 1., top], [0, 0, 1.]])
    return dict(requested_bbox_xyxy=box, requested_coordinate_system="region_crop_pixels",
        delivered_native_bbox_xyxy=bounds, source_polygon_xy=polygon.tolist(), native_shape=[height, width],
        crop_shape=[bottom-top, right-left], native_offset_xy=[left, top],
        crop_local_to_observation_native=offset.tolist(),
        crop_local_to_region=(_matrix(observation["geometry"]["source_to_crop"]) @ offset).tolist(),
        context="Perspective envelope outside requested polygon is context, never additional evaluated area",
        sampling="original RGB pixel slice; no resize or super-resolution; processor sampling reported separately")


class InspectionTools(ToolService):
    """A frozen allowlist replaces generic store.history and retrieval authorization."""
    def __init__(self, store, case, out, inventory, maximum):
        self.inventory = copy.deepcopy(inventory)
        self.inventory_hash = digest(self.inventory)
        self.maximum = maximum
        if maximum != inventory["max_native_crop_pixels"]:
            raise ValueError("Native pixel limit differs from the frozen action set")
        for path_key, hash_key in (("proposal_preview_path", "proposal_preview_sha256"),):
            if path_key in inventory and sha256(inventory[path_key]) != inventory[hash_key]:
                raise ValueError("Frozen method preview changed")
        method = inventory.get("method", {})
        if method and sha256(method["native_map_path"]) != method["native_map_sha256"]:
            raise ValueError("Frozen AnyChange native map changed")
        super().__init__(store, None, None, None, case, out, method_evidence=[])
        self.contract.update(inventory_sha256=self.inventory_hash, protocol_version=VERSION)

    def _authorize(self, identifier):
        if identifier not in self.inventory["observations"]:
            raise ValueError("Observation is outside the frozen RGB inventory")
        frozen = self.inventory["observations"][identifier]
        current = self.store.verify_observation(self.store.get_observation(identifier))
        for key in ("region_id", "building_id", "view_id", "split", "date", "available_at",
                    "original_sha256", "crop_sha256", "support_sha256"):
            if current[key] != frozen[key]:
                raise ValueError("Frozen observation changed: " + key)
        if current["split"] not in {"train", "val"} or current["region_id"] != self.inventory["region_id"]:
            raise ValueError("Cross-region or held-out observation")
        if not causally_available(current, self.inventory["knowledge_cutoff"]):
            raise ValueError("Observation unavailable at knowledge_cutoff")
        if identifier not in self.current and (identifier not in self.inventory["history_ids"] or
            temporal_value(current["date"]) >= temporal_value(self.case["reference_date"])):
            raise ValueError("History must strictly precede reference")
        for key in ("source_to_crop", "crop_to_source", "native_shape", "crop_shape"):
            if not np.allclose(current["geometry"][key], frozen["geometry"][key], atol=1e-8):
                raise ValueError("Frozen observation geometry changed")
        return current

    def catalogue(self):
        return [dict(observation_id=i, date=self.inventory["observations"][i]["date"],
                     available_at=self.inventory["observations"][i]["available_at"],
                     correspondence_status="aligned", geometric_coverage_pixels=self.inventory["coverage_pixels"][i])
                for i in self.inventory["history_ids"]]

    def preview_action(self, observation_ids, bbox_xyxy):
        validate_action(self.inventory, observation_ids, bbox_xyxy)
        return [crop_geometry(self._authorize(i), bbox_xyxy, self.maximum) for i in observation_ids]

    def execute(self, name, arguments, timeout_seconds=None):
        if name != "inspect_region" or not isinstance(arguments, dict) or set(arguments) != {"observation_ids", "bbox_xyxy"}:
            raise ValueError("Only inspect_region(observation_ids, bbox_xyxy) is permitted; SAM3 is disabled")
        return self.inspect_region(**arguments)

    def inspect_region(self, observation_ids, bbox_xyxy):
        geometries = self.preview_action(observation_ids, bbox_xyxy)
        records, images = [], []
        for oid, geometry in zip(observation_ids, geometries):
            obs = self._authorize(oid)
            rgb, alpha = load_rgb(obs["original_path"])
            if list(rgb.shape[:2]) != geometry["native_shape"]:
                raise ValueError("Native RGB dimensions changed")
            left, top, right, bottom = geometry["delivered_native_bbox_xyxy"]
            cropped = rgb[top:bottom, left:right].copy()
            yy, xx = np.mgrid[top:bottom, left:right]
            points = np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ _matrix(obs["geometry"]["source_to_crop"]).T
            valid_denominator = np.abs(points[..., 2]) > 1e-12
            region = points[..., :2] / np.where(valid_denominator, points[..., 2], 1)[..., None]
            x0, y0, x1, y1 = bbox_xyxy
            requested = valid_denominator & (region[..., 0] >= x0) & (region[..., 0] < x1) & (region[..., 1] >= y0) & (region[..., 1] < y1)
            support = _support(obs["support_path"], tuple(obs["geometry"]["crop_shape"]))
            mapped = cv2.remap(support.astype(np.uint8), region[..., 0].astype(np.float32), region[..., 1].astype(np.float32),
                               cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
            geometric = requested & mapped & alpha[top:bottom, left:right]
            folder = self.out / "crops" / digest([self.inventory_hash, oid, bbox_xyxy])[:24]
            folder.mkdir(parents=True, exist_ok=False)
            Image.fromarray(cropped).save(folder / "rgb.png")
            Image.fromarray(geometric.astype(np.uint8)*255).save(folder / "support.png")
            Image.fromarray((~requested).astype(np.uint8)*255).save(folder / "envelope_context.png")
            Image.fromarray(alpha[top:bottom, left:right].astype(np.uint8)*255).save(folder / "alpha.png")
            geometry.update(region_id=self.case["region_id"], observation_id=oid, original_sha256=obs["original_sha256"],
                            requested_pixels=int(requested.sum()), geometric_pixels=int(geometric.sum()),
                            context_pixels=int((~requested).sum()))
            write_json(folder / "geometry.json", geometry)
            record = dict(evidence_id="inspection-" + digest([self.inventory_hash, oid, bbox_xyxy])[:32],
                kind="observation", observation_id=oid, observation_ids=[oid], region_id=self.case["region_id"],
                date=obs["date"], available_at=obs["available_at"], coordinate_system="inspection_crop_pixels",
                method="native_rgb_inspection", image_shape=list(cropped.shape[:2]), geometry=geometry,
                summary="Native original RGB enclosing crop. Use crop_local_to_region for final region_crop_pixels box. Geometric coverage is not visibility.",
                artifacts=[self._artifact(folder / name, role) for name, role in
                    (("rgb.png", "rgb"), ("support.png", "geometric_support"), ("envelope_context.png", "envelope_context"),
                     ("alpha.png", "alpha"), ("geometry.json", "geometry"))])
            records.append(self._public(self._remember(record)))
            images.append(dict(type="image", path=str(folder / "rgb.png"), evidence_id=record["evidence_id"]))
        return dict(status="ok", evidence=records, images=images, usage={})
