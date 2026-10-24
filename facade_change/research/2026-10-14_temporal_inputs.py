"""Audited TRAIN/VAL histories and native-grid real crops, without semantic GT.

``preflight`` opens metadata and hashes selected input bytes, but decodes no
images. ``load_sequences`` additionally selects real crops using only common
geometric/alpha support. Evaluation truth returned by ``load_frame`` must never
be passed to a scorer. Real observations have no temporal labels.
"""
from __future__ import annotations

import importlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ..derived import _starts
from ..geometry import expanded_canvas
from ..io import load_rgb, read_json, sha256
from ..semantic_training import audit_semantic_leakage
from .common import digest


SPLITS = ("train", "val")
SYNTHETIC_ARRAYS = ("rgb", "support", "visibility", "latent_rgb", "event_labels")


def _remember(path, fingerprints, expected=None):
    path = Path(path).expanduser().resolve()
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Input hash mismatch: {path}")
    if str(path) in fingerprints and fingerprints[str(path)] != actual:
        raise ValueError(f"Input changed during preflight: {path}")
    fingerprints[str(path)] = actual
    return path


def _artifact(root, record, relative, fingerprints, expected=None):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Artifact paths must be relative")
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("Artifact path escapes its run")
    recorded = record.get("artifact_sha256", {}).get(relative)
    if not recorded or (expected is not None and expected != recorded):
        raise ValueError(f"Missing or conflicting artifact provenance: {relative}")
    return _remember(path, fingerprints, recorded)


def _audit(rows, split):
    report = audit_semantic_leakage({"images": rows}, split)
    if report["status"] != "passed":
        raise ValueError(f"Temporal split leakage: {report['violations']}")
    return report


def _synthetic(path, fingerprints):
    root = Path(path).expanduser().resolve()
    record = read_json(_remember(root / "run.json", fingerprints))
    loader = importlib.import_module("facade_change.2026-10-04_msdzip_h0")
    _, _, index, split, _ = loader._history_parent(root, record)
    if digest(record["config"]) != record.get("config_sha256"):
        raise ValueError("History configuration hash mismatch")
    for name in ("manifest.json", "summary.json", "config.json", "split.json"):
        _artifact(root, record, name, fingerprints)
    bases = {base["base_id"]: base for base in index["bases"]}
    rows = [{**base, "image_id": "history:" + base["base_id"],
             "sha256": base["image_sha256"], "metadata_status": "reviewed"}
            for base in bases.values()]
    _audit(rows, split)
    sequences, arms = [], defaultdict(dict)
    for sequence in index["sequences"]:
        base = bases[sequence["base_id"]]
        if sequence["split"] == "test":
            continue  # Metadata audit above includes TEST; artifact bytes do not.
        if sequence["split"] not in SPLITS:
            raise ValueError("Temporal experiments accept TRAIN/VAL only")
        arm = sequence["arm"]
        if arm in arms[base["base_id"]]:
            raise ValueError("Duplicate synthetic arm")
        arms[base["base_id"]][arm] = sequence
        frames = sequence["frames"]
        if len(frames) != 5 or [frame["frame"] for frame in frames] != list(range(1, 6)):
            raise ValueError("Synthetic history requires five chronological frames")
        # Observation renderers can put known synthetic occlusion in per-frame
        # support. That is oracle visibility, not model-visible geometry. Use
        # the unchanged original alpha/geometric support for every observation.
        geometric_support = _artifact(root, record, base["support"], fingerprints)
        geometric_support_hash = fingerprints[str(geometric_support)]
        prepared = []
        for frame in frames:
            paths = {name: str(_artifact(root, record, frame["paths"][name], fingerprints,
                                          frame["artifact_sha256"][name]))
                     for name in SYNTHETIC_ARRAYS if name != "support"}
            paths["support"] = str(geometric_support)
            hashes = {**frame["artifact_sha256"], "support": geometric_support_hash}
            prepared.append({**frame, "paths": paths, "artifact_sha256": hashes,
                             "model_support_source": "base_geometric_support"})
        crop = base["crop"]
        sequences.append({**sequence, "frames": prepared, "domain": "synthetic",
                          "view_id": base.get("view_id"), "image_sha256": base["image_sha256"],
                          "shape": [crop["height"], crop["width"]],
                          "label_scope": index["label_scope"]})
    for paired in arms.values():
        if set(paired) != {"clean", "event"}:
            raise ValueError("Synthetic base lacks matched clean/event arms")
        for clean, event in zip(paired["clean"]["frames"], paired["event"]["frames"]):
            if clean["nuisance"] != event["nuisance"]:
                raise ValueError("Synthetic clean/event nuisance conditions differ")
    coverage = {"sequences": len(sequences), "bases": len(arms),
                "by_split": dict(Counter(s["split"] for s in sequences)),
                "excluded_test_sequences": sum(s["split"] == "test" for s in index["sequences"])}
    return sequences, rows, coverage


def _dataset(path, fingerprints):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "dataset.json"
    config = read_json(_remember(path, fingerprints))
    if config.get("kind") != "semantic_crop_index":
        raise ValueError("Real histories require the current prepared dataset.json")
    root = path.parent
    paths = {}
    for name in ("manifest", "split"):
        candidate = Path(config[name]).expanduser()
        paths[name] = _remember(candidate if candidate.is_absolute() else root / candidate, fingerprints)
    if (root / "run.json").is_file():
        record = read_json(_remember(root / "run.json", fingerprints))
        if not str(record.get("status", "")).startswith("completed"):
            raise ValueError("Current dataset preparation is incomplete")
        for candidate in (path, *paths.values()):
            if root in candidate.parents:
                relative = candidate.relative_to(root).as_posix()
                if relative in record.get("artifact_sha256", {}):
                    _artifact(root, record, relative, fingerprints)
    manifest, split = read_json(paths["manifest"]), read_json(paths["split"])
    if split.get("development_only", False) or split.get("mode", "reviewed") != "reviewed":
        raise ValueError("Real histories require a reviewed building split")
    _audit(manifest["images"], split)
    rows = []
    for item in manifest["images"]:
        row = dict(item)
        if (row.get("metadata_status") == "reviewed" and row.get("building_id")
                and row.get("split") in {"train", "val", "test"}
                and split.get("building_assignments", {}).get(row["building_id"]) != row["split"]):
            raise ValueError("Current temporal image lacks its reviewed building assignment")
        source = Path(row["image_path"]).expanduser()
        row["image_path"] = str(source if source.is_absolute() else (paths["manifest"].parent / source).resolve())
        rows.append(row)
    return rows, manifest.get("pairs", []), split


def _alignments(paths, fingerprints):
    found, attempts = defaultdict(list), []
    for value in paths:
        root = Path(value).expanduser().resolve()
        if (root / "sessions").is_dir():
            from facade_change.alignment_session import resolve_session
            root = resolve_session(root)
        record = read_json(_remember(root / "run.json", fingerprints))
        status = str(record.get("status", ""))
        if not status.startswith("completed") and status not in {"interrupted", "failed"}:
            raise ValueError("Alignment input must be terminal, not still running")
        if record.get("kind") == "incremental_alignment_preparation":
            if len(paths) != 1:
                raise ValueError("Use the selected alignment session alone; mixing runs would override its exclusions")
            from facade_change.alignment_session import load_session
            return load_session(root, fingerprints).matches()
        if record.get("kind") == "pair_alignment":
            candidates = [(root, record, record.get("config", {}).get("method", "unknown"))]
        elif record.get("kind") == "alignment_and_data_batch":
            results = read_json(_artifact(root, record, "results.json", fingerprints))
            candidates = []
            for result in results:
                child = (root / result["path"]).resolve()
                if root not in child.parents:
                    raise ValueError("Alignment child path escapes its batch")
                attempt = {"path": str(child), "pair_id": result["pair_id"],
                           "method": result["method"], "status": result["status"]}
                attempts.append(attempt)
                if not (child / "run.json").is_file():
                    attempt["reason"] = "missing_pair_run"
                    continue
                relative = (child / "run.json").relative_to(root).as_posix()
                child_record = read_json(_artifact(root, record, relative, fingerprints))
                candidates.append((child, child_record, result["method"]))
        else:
            raise ValueError("Expected pair_alignment or alignment_and_data_batch input")
        for child, item, method in candidates:
            observations = item.get("observations", {})
            if not all(observations.get(role, {}).get("sha256") for role in ("reference", "source")):
                attempts.append({"path": str(child), "status": item.get("status"),
                                 "reason": "missing_observation_provenance"})
                continue
            key = tuple(observations[role]["sha256"] for role in ("reference", "source"))
            found[key].append({"path": child, "record": item, "method": method})
    return found, attempts


def _real(config, fingerprints):
    images, pairs, split = _dataset(config["dataset"], fingerprints)
    alignments, attempts = _alignments(config.get("alignment_runs", []), fingerprints)
    pair_keys = {(str(p["reference_id"]), str(p["source_id"])) for p in pairs}
    groups, excluded = defaultdict(list), []
    for row in images:
        if row.get("split") not in SPLITS:
            continue
        if (row.get("metadata_status") != "reviewed" or not row.get("building_id")
                or not row.get("view_id") or row.get("year") is None
                or row.get("temporal_eligible") is False):
            excluded.append({"image_id": row["image_id"], "reason": "ineligible_temporal_metadata"})
            continue
        groups[(row["split"], row["building_id"], row["view_id"])].append(row)
    sequences, coverage = [], []
    for key, observations in sorted(groups.items()):
        observations.sort(key=lambda row: (int(row["year"]), str(row["image_id"])))
        report = {"split": key[0], "building_id": key[1], "view_id": key[2],
                  "eligible_observations": len(observations), "pairs": []}
        coverage.append(report)
        if len(observations) < 2:
            report["status"] = "fewer_than_two_observations"
            continue
        if len({int(row["year"]) for row in observations}) != len(observations):
            report["status"] = "ambiguous_same_year_observations"
            continue
        reference = observations[0]
        frames = [{**reference, "source_to_reference": np.eye(3).tolist(), "frame": 1}]
        for source in observations[1:]:
            pair = {"reference_id": reference["image_id"], "source_id": source["image_id"],
                    "reference_year": reference["year"], "source_year": source["year"]}
            report["pairs"].append(pair)
            if (str(reference["image_id"]), str(source["image_id"])) not in pair_keys:
                pair["status"] = "missing_prepared_first_anchor_pair"
                continue
            matches = alignments.get((reference["sha256"], source["sha256"]), [])
            pair["attempts"] = [{"path": str(m["path"]), "status": m["record"].get("status"),
                                 "method": m["method"]} for m in matches]
            for match in matches:
                for role, current in (("reference", reference), ("source", source)):
                    previous = match["record"]["observations"][role]
                    for field in ("split", "building_id", "view_id", "year", "width", "height", "metadata_status"):
                        if previous.get(field) != current.get(field):
                            raise ValueError(f"Alignment/current dataset {role}.{field} mismatch")
            accepted = [match for match in matches if
                        str(match["record"].get("status", "")).startswith("completed")
                        and (match.get("selection") or match["record"].get("quality_gate", {}).get("passed") is True)]
            if not accepted:
                pair["status"] = "rejected_or_failed_alignment" if matches else "missing_alignment"
                continue
            # Configuration order gives deterministic priority among reusable runs.
            chosen = accepted[0]
            from facade_change.saved_geometry import frame_geometry
            frames.append({**source, **frame_geometry(chosen, fingerprints), "frame": len(frames) + 1})
            pair.update(status="accepted", selected_alignment=str(chosen["path"]))
        report["usable_observations"] = len(frames)
        if len(frames) < 2:
            report["status"] = "no_accepted_source"
            continue
        for frame in frames:
            _remember(frame["image_path"], fingerprints, frame["sha256"])
        report["status"] = "eligible_for_geometric_crops"
        identifier = "real-" + digest([key, [f["sha256"] for f in frames]])[:24]
        sequences.append({"sequence_id": identifier, "domain": "real", "split": key[0],
                          "building_id": key[1], "view_id": key[2], "arm": None,
                          "frames": frames, "shape": [reference["height"], reference["width"]],
                          "label_scope": "unlabelled real history; visual review only",
                          "missing_dates": len(observations) - len(frames)})
    return sequences, images, split, {"views": coverage, "alignment_attempts": attempts,
                                     "excluded_images": excluded}


def preflight(config):
    """Audit metadata and freeze source hashes without decoding RGB or labels."""
    fingerprints, sequences, identity_rows = {}, [], []
    coverage = {"test_rgb_loaded": False, "semantic_labels_loaded": False}
    if config.get("history_run"):
        synthetic, rows, coverage["synthetic"] = _synthetic(config["history_run"], fingerprints)
        sequences.extend(synthetic)
        identity_rows.extend(rows)
    split = {}
    if config.get("dataset"):
        real, rows, split, coverage["real"] = _real(config, fingerprints)
        sequences.extend(real)
        identity_rows.extend({**row, "image_id": "real:" + str(row["image_id"])} for row in rows)
    if not sequences:
        raise ValueError("No eligible TRAIN/VAL sequences in the requested inputs")
    leakage = _audit(identity_rows, split)
    provenance = {"files_sha256": fingerprints, "leakage_check": leakage,
                  "test_rgb_loaded": False, "semantic_labels_loaded": False,
                  "selection_policy": "synthetic frozen selection; real first-anchor geometry/alpha only"}
    provenance["sha256"] = digest(provenance)
    return sequences, provenance, coverage


def _native(frame):
    if sha256(frame["image_path"]) != frame["sha256"]:
        raise ValueError("Native image changed after temporal preflight")
    rgb, support = load_rgb(frame["image_path"])
    if rgb.shape[:2] != (frame["height"], frame["width"]):
        raise ValueError("Native image dimensions changed")
    return rgb, support


def _warp(array, matrix, width, height):
    import cv2
    return cv2.warpPerspective(array, np.asarray(matrix, dtype=float), (width, height),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def load_sequences(config):
    """Prepare metadata for native crops; real crop selection never reads GT."""
    candidates, provenance, coverage = preflight(config)
    tile, stride = config.get("real_tile_size", 512), config.get("real_stride", 256)
    minimum = config.get("real_min_support_fraction", .8)
    if type(tile) is not int or tile < 8 or type(stride) is not int or not 0 < stride <= tile:
        raise ValueError("Real crops require tile >=8 and 0 < stride <= tile")
    if not isinstance(minimum, (int, float)) or isinstance(minimum, bool) or not 0 < minimum <= 1:
        raise ValueError("real_min_support_fraction must be in (0,1]")
    sequences = []
    for sequence in candidates:
        if sequence["domain"] == "synthetic":
            sequences.append(sequence)
            continue
        height, width = sequence["shape"]
        crop_h, crop_w = min(tile, height), min(tile, width)
        boxes = [(x, y, crop_w, crop_h) for y in _starts(height, tile, stride)
                 for x in _starts(width, tile, stride)]
        # Only one native RGB is decoded at once; retain no full RGB canvas.
        intersections = [np.ones((crop_h, crop_w), bool) for _ in boxes]
        for frame in sequence["frames"]:
            native_rgb, support = _native(frame)
            del native_rgb
            from facade_change.saved_geometry import FrameSampler
            sampler = FrameSampler(frame)
            for index, box in enumerate(boxes):
                intersections[index] &= sampler.support(support, box)
        accepted = 0
        for (x, y, w, h), support in zip(boxes, intersections):
            fraction = float(support.mean())
            if fraction < minimum:
                continue
            accepted += 1
            sequences.append({**sequence, "sequence_id": sequence["sequence_id"] + f"-x{x}-y{y}",
                              "parent_sequence_id": sequence["sequence_id"], "crop_xywh": [x, y, w, h],
                              "shape": [h, w], "common_support_fraction": fraction})
        report = next(row for row in coverage["real"]["views"] if
                      (row["split"], row["building_id"], row["view_id"]) ==
                      (sequence["split"], sequence["building_id"], sequence["view_id"]))
        report.update(candidate_crops=len(boxes), accepted_crops=accepted,
                      rejected_low_support_crops=len(boxes) - accepted,
                      status="cropped" if accepted else "no_common_supported_crop")
    provenance["selected_sequences_sha256"] = digest(sequences)
    provenance["sha256"] = digest({key: value for key, value in provenance.items() if key != "sha256"})
    coverage["sequences"] = len(sequences)
    return sequences, provenance, coverage


def load_frame(sequence, index):
    """Load one observed crop; latent/visibility/event arrays are evaluation-only."""
    if sequence["split"] not in SPLITS:
        raise ValueError("TEST frame access is forbidden")
    if type(index) is not int or not 0 <= index < len(sequence["frames"]):
        raise IndexError("Frame index is outside the available causal sequence")
    frame = sequence["frames"][index]
    if sequence["domain"] == "real":
        rgb, support = _native(frame)
        from facade_change.saved_geometry import FrameSampler
        sampler = FrameSampler(frame)
        crop = sequence["crop_xywh"]
        width, height = crop[2:]
        return {"rgb": sampler.warp(rgb, crop),
                "support": sampler.support(support, crop),
                "event_labels": np.full((height, width), 255, np.uint8),
                "event": "unknown", "nuisance": "unknown"}
    output = {}
    for name in SYNTHETIC_ARRAYS:
        path = frame["paths"][name]
        if sha256(path) != frame["artifact_sha256"][name]:
            raise ValueError(f"Synthetic frame changed after preflight: {path}")
        if name in {"rgb", "latent_rgb"}:
            value, opaque = load_rgb(path)
            if not opaque.all():
                raise ValueError("Synthetic RGB artifacts must be fully opaque")
        else:
            with Image.open(path) as image:
                value = np.array(image)
            if value.dtype != np.uint8 or value.ndim != 2:
                raise ValueError("Synthetic masks must be native uint8 single-channel arrays")
            if name in {"support", "visibility"}:
                if not np.isin(value, [0, 255]).all():
                    raise ValueError("Synthetic support/visibility must use 0/255")
                value = value == 255
            elif not np.isin(value, [0, 1, 255]).all():
                raise ValueError("Synthetic event labels must use 0/1/255")
        if list(value.shape[:2]) != sequence["shape"]:
            raise ValueError("Synthetic frame grid disagrees with its source crop")
        output[name] = value
    if np.any(output["visibility"] & ~output["support"]):
        raise ValueError("Synthetic visibility includes pixels outside geometric support")
    output.update(event=frame["event"], nuisance=frame["nuisance"]["kind"])
    return output
