"""Pairwise adapter for the shared COCO annotation-target workflow.

Read existing crop transforms and supports; do not recrop RGB, rematch images,
load DINO, change partitions, or interpret unannotated pixels as healthy.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import importlib
from pathlib import Path
import uuid

import numpy as np
from PIL import Image

from .annotation_targets import SCOPE, event_target, load_annotation_source, project_masks
from .history_features import Fingerprints, _inside, _npz, digest, progress
from .io import read_json, sha256, write_json
from .segmentation.data import rasterize_annotations


history = importlib.import_module("facade_change.research.2026-10-25_history_models")
targets = importlib.import_module("facade_change.research.2026-10-25_history_targets")
PAIR_FIELDS = ("dataset_crop_id", "pair_id", "alignment_method", "reference_id", "source_id",
               "reference_rgb", "source_rgb", "crop_root")


class CropRun(history.RunStore):
    """Use the existing run ledger and atomic unit cache, with a pair-specific kind."""

    def event(self, kind, **values):
        if not hasattr(self, "unique_id"):
            self.unique_id = self.record.get("annotation_run_id") or "crop-targets-" + uuid.uuid4().hex
        self.record.update(run_id=self.unique_id, annotation_run_id=self.unique_id,
                           kind="crop_annotation_targets")
        return super().event(kind, run_id=self.unique_id,
                             event_id=f"{self.unique_id}:{self.sequence + 1}", **values)


@dataclass
class CropData:
    history_root: Path  # Shared audit renderer's RGB root; this is the crop dataset.
    index_path: Path
    index: dict
    rows: list
    tracks: dict
    fingerprints: Fingerprints
    coverage: dict


def _parent_file(root, parent, relative, fp, expected=None):
    path = _inside(root, relative)
    recorded = parent.get("artifact_sha256", {}).get(relative)
    if not recorded or (expected is not None and recorded != expected):
        raise ValueError(f"Crop parent lacks a matching artifact hash: {relative}")
    fp.add(path, recorded)
    return path


def _support(path, shape):
    with Image.open(path) as image:
        raw = np.asarray(image)
    if raw.dtype != np.uint8 or raw.shape != tuple(shape) or not np.isin(raw, [0, 255]).all():
        raise ValueError(f"Invalid binary crop support: {path}")
    return raw == 255


def load_crops(path, splits, annotation_manifest=None):
    """Check the frozen crop snapshot and selected metadata before rasterizing COCO."""
    if not splits or len(set(splits)) != len(splits) or set(splits) - {"train", "val", "test"}:
        raise ValueError("Use unique explicit train/val/test splits")
    root = Path(path).expanduser().resolve()
    fp = Fingerprints()
    fp.add(root / "run.json")
    parent = read_json(root / "run.json")
    if (parent.get("kind") != "crop_dataset"
            or parent.get("status") not in {"completed", "completed_needs_review", "completed_with_issues"}):
        raise ValueError("--crops requires a completed pairwise crop dataset")
    summary = read_json(_parent_file(root, parent, "summary.json", fp))
    index_path = _parent_file(root, parent, summary["index_path"], fp)
    index = read_json(index_path)
    split = read_json(_parent_file(root, parent, "split.json", fp))
    if (index.get("schema_version") != 1 or index.get("split") != split
            or split.get("mode") != "reviewed" or split.get("development_only") is not False):
        raise ValueError("Crop targets require the existing reviewed split snapshot")
    config = parent["config"]
    manifest_path = targets.resolve_path(config["manifest_path"], root)
    expected = config.get("input_sha256", {}).get("manifest") or config.get("input_sha256", {}).get(str(manifest_path))
    if not expected:
        raise ValueError("Crop parent lacks the reviewed manifest hash")
    fp.add(manifest_path, expected)
    source = load_annotation_source(annotation_manifest or manifest_path, fp)
    if [(c["id"], c["name"]) for c in source["categories"]] != sorted((c["id"], c["name"]) for c in index["categories"]):
        raise ValueError("Annotation and crop category mappings disagree")
    if len(index["crops"]) != summary["crop_count"]:
        raise ValueError("Crop summary and index counts disagree")
    seen, buildings, views, hashes, rows, tracks = set(), {}, {}, {}, [], {}
    for row in index["crops"]:
        identifier, building, view, part = (row[k] for k in ("dataset_crop_id", "building_id", "view_id", "split"))
        if identifier in seen or not building or not view or part not in {"train", "val", "test"}:
            raise ValueError("Invalid/duplicate crop identity")
        seen.add(identifier)
        if (split["building_assignments"].get(building) != part
                or buildings.setdefault(building, part) != part
                or views.setdefault(view, building) != building):
            raise ValueError("Crop building/view split leakage or assignment mismatch")
        for value in row["source_image_sha256"].values():
            if hashes.setdefault(value, part) != part:
                raise ValueError("Original image hash leakage across crop splits")
        if part not in splits:
            continue
        if (type(row["reference_year"]) is not int or type(row["source_year"]) is not int
                or row["reference_year"] >= row["source_year"]):
            raise ValueError("Pairwise crop targets require chronological cross-year pairs")
        rows.append(row)
    if not rows:
        raise ValueError("No pairwise crops in the requested splits")
    for row in progress(rows, "Validate saved crop geometry"):
        directory = _inside(root, row["path"])
        metadata_path = _parent_file(root, parent, (directory / "crop.json").relative_to(root).as_posix(), fp)
        metadata = read_json(metadata_path)
        for key in ("crop_id", "building_id", "view_id", "split", "reference_id", "source_id",
                    "valid_fraction", "bbox_canvas_xyxy", "tile_shape", "reference_to_crop",
                    "source_to_crop", "dense_geometry", "source_image_sha256", "artifact_sha256"):
            if metadata.get(key) != row.get(key):
                raise ValueError(f"Crop index disagrees with saved crop.json: {key}")
        shape = row["tile_shape"]
        if len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError("Invalid crop dimensions")
        if (row["pair_id"] != f"{row['reference_id']}-{row['source_id']}"
                or row["dataset_crop_id"] != row["pair_id"] + "/" + row["crop_id"]):
            raise ValueError("Pair/crop ID disagrees with its observations")
        x0, y0, x1, y1 = row["bbox_canvas_xyxy"]
        if (any(type(n) is not int for n in (x0, y0, x1, y1))
                or [y1-y0, x1-x0] != shape):
            raise ValueError("Saved crop rectangle and dimensions disagree")
        if row.get("dense_geometry"):
            dense = row["dense_geometry"]
            dense_path = Path(dense["path"]).expanduser().resolve()
            if str(dense_path) not in fp.files:
                fp.add(dense_path, dense["sha256"])
            elif fp.files[str(dense_path)] != dense["sha256"]:
                raise ValueError("Conflicting saved dense geometry hashes")
        track_id = "pair-" + digest(row["dataset_crop_id"])[:32]
        observations = []
        for role in ("reference", "source"):
            rgb_relative = (directory / f"{role}_rgb.png").relative_to(root).as_posix()
            if row[role + "_rgb"] != rgb_relative:
                raise ValueError("Indexed crop RGB path disagrees with its directory")
            rgb_hash = row["artifact_sha256"][f"{role}_rgb.png"]
            if parent["artifact_sha256"].get(rgb_relative) != rgb_hash:
                raise ValueError("Indexed RGB hash disagrees with its crop parent")
            support_relative = (directory / f"{role}_support.png").relative_to(root).as_posix()
            support_hash = row["artifact_sha256"][f"{role}_support.png"]
            _parent_file(root, parent, support_relative, fp, support_hash)
            image_hash = row["source_image_sha256"][role]
            image = source["images"].get(image_hash)
            if image is not None:
                coco = source["coco_images"].get(str(image["image_id"]))
                if (coco is None or any(coco[k] != image[k] for k in ("height", "width"))
                        or any(image.get(k) != row[k] for k in ("building_id", "view_id", "split"))
                        or image.get("year") != row[role + "_year"]):
                    raise ValueError("Annotation/crop dimensions, identity, split or year mismatch")
                original = Path(row["original_" + role + "_path"]).expanduser().resolve()
                if str(original) not in fp.files:
                    fp.add(original, image_hash)
                elif fp.files[str(original)] != image_hash:
                    raise ValueError("Conflicting original image hashes")
            observations.append({"observation_id": digest([track_id, role]), "role": role,
                                 "year": row[role + "_year"], "image_sha256": image_hash,
                                 "rgb": rgb_relative, "rgb_sha256": rgb_hash,
                                 "support": support_relative, "support_sha256": support_hash})
        overlap = (directory / "geometric_overlap.png").relative_to(root).as_posix()
        _parent_file(root, parent, overlap, fp, row["artifact_sha256"]["geometric_overlap.png"])
        tracks[track_id] = {"track_id": track_id, "observations": observations, "crop": row}
    coverage = {"candidate_crops": len(index["crops"]), "selected_crops": len(rows),
                "buildings": len({r["building_id"] for r in rows}), "splits": list(splits),
                "by_split": dict(Counter(r["split"] for r in rows))}
    return CropData(root, index_path, index, rows, tracks, fp, coverage), source


def export_crop_targets(args):
    """Rasterize each selected original once, then reuse the history target rules."""
    import cv2
    data, source = load_crops(args.crops, args.splits, args.annotation_manifest)
    names = [c["name"] for c in source["categories"]]
    if args.categories and (set(args.categories) - set(names) or len(set(args.categories)) != len(args.categories)):
        raise ValueError(f"Unknown/duplicate target categories; available: {names}")
    channels = [i for i, name in enumerate(names) if not args.categories or name in args.categories]
    out = Path(args.out).expanduser().resolve()
    if out == data.history_root or out in data.history_root.parents:
        raise ValueError("Annotation output must be separate from the input crop dataset")
    if args.resume and (out / "run.json").exists():
        previous = read_json(out / "run.json")
        if previous.get("kind") not in {"crop_annotation_targets", "temporal_experiment"}:
            raise ValueError("Resume output belongs to another workflow")
        if previous.get("status") == "completed":
            raise ValueError("Annotation export already completed; use a new output directory")
    store = CropRun(out, {k: v for k, v in vars(args).items() if k != "resume"}, resume=args.resume)
    observations, processing = {}, Counter()
    try:
        core = importlib.import_module("facade_change.annotation_targets")
        raster = importlib.import_module("facade_change.segmentation.data")
        evaluator = importlib.import_module("scripts.real_eval.real_gt_metrics")
        contract = {"inputs": dict(data.fingerprints.files), "module_sha256": sha256(__file__),
                    "annotation_core_sha256": sha256(core.__file__), "rasterizer_sha256": sha256(raster.__file__),
                    "evaluator_sha256": sha256(evaluator.__file__), "cv2": cv2.__version__,
                    "categories": source["categories"], "target_channels": channels, "target_scope": SCOPE,
                    "category_to_channel": [{"category_id": c["id"], "channel": i}
                                            for i, c in enumerate(source["categories"])],
                    "grid": "saved pairwise native canvas and exact RGB crop transforms",
                    "unknown_policy": "unannotated, unsupported and crowd remain unknown",
                    "rgb_recomputed": False, "dino_recomputed": False, "reviewed_event_gt_created": False}
        store.contract(contract)
        jobs = defaultdict(list)
        for track in data.tracks.values():
            for observation in track["observations"]:
                jobs[observation["image_sha256"]].append((track, observation))
        for image_hash, items in progress(sorted(jobs.items()), "Project crop annotations"):
            image, packed = source["images"].get(image_hash), None
            dense_key, sampler = None, None
            for track, observation in items:
                crop = track["crop"]
                unit_id = "annotation-" + observation["observation_id"]
                cached = store.read_unit(unit_id)
                if cached is not None:
                    record, arrays = cached
                    if (arrays["packed"].dtype != np.uint16 or arrays["support"].dtype != bool
                            or arrays["packed"].shape != tuple(crop["tile_shape"])
                            or arrays["support"].shape != arrays["packed"].shape):
                        raise ValueError("Invalid cached projected crop annotations")
                    observations[observation["observation_id"]] = record
                    processing["reused_observations"] += 1
                    continue
                support = _support(_inside(data.history_root, observation["support"]), crop["tile_shape"])
                if image is None:
                    labels = np.zeros(support.shape, dtype=np.uint16)
                    status = "missing_annotation_image"
                else:
                    if packed is None:
                        packed = rasterize_annotations(source["annotations"].get(str(image["image_id"]), []),
                                                       image["height"], image["width"], source["channel_map"])
                        processing["rasterized_images"] += 1
                    role = observation["role"]
                    if role == "source" and crop.get("dense_geometry"):
                        from .saved_geometry import FrameSampler
                        dense = crop["dense_geometry"]
                        signature = digest(dense)
                        if dense_key != signature:
                            sampler = FrameSampler({"dense_geometry": dense})
                            dense_key = signature
                        x0, y0, x1, y1 = crop["bbox_canvas_xyxy"]
                        labels = sampler.warp(packed, (x0, y0, x1-x0, y1-y0), nearest=True)
                        labels[~support] = 0
                    else:
                        height, width = crop["tile_shape"]
                        labels = project_masks(packed, crop[role + "_to_crop"], (0, 0, width, height), support)
                    status = "projected" if labels.any() else "unannotated_crop"
                record = {"unit_id": unit_id, "observation_id": observation["observation_id"],
                          "track_id": track["track_id"], "image_sha256": image_hash,
                          "source_annotation_image_id": image["image_id"] if image else None,
                          "status": status, "known_pixels": int((labels != 0).sum())}
                observations[observation["observation_id"]] = store.save_unit(record, {"packed": labels, "support": support})
                processing["computed_observations"] += 1
        rows, class_counts = [], Counter()
        for key, track in progress(sorted(data.tracks.items()), "Derive crop target differences"):
            crop, states = track["crop"], []
            for observation in track["observations"]:
                with np.load(store.out / observations[observation["observation_id"]]["artifact_path"], allow_pickle=False) as arrays:
                    states.append({name: arrays[name].copy() for name in ("packed", "support")})
            overlap_path = _inside(data.history_root, crop["path"]) / "geometric_overlap.png"
            overlap = _support(overlap_path, crop["tile_shape"])
            if (not np.array_equal(overlap, states[0]["support"] & states[1]["support"])
                    or not np.isclose(overlap.mean(), crop["valid_fraction"])):
                raise ValueError("Saved crop overlap disagrees with its observation supports")
            result, target, layers = event_target(*states, channels, args.boundary_tolerance,
                args.min_known_pixels, args.min_known_fraction, args.min_changed_pixels)
            event_id = digest(["pairwise-annotation", crop["dataset_crop_id"], crop["source_image_sha256"]])
            target_path = store.out / "targets" / f"{event_id}.npz"
            _npz(target_path, {"target": target, "class_targets": layers,
                               "category_ids": np.array([source["categories"][i]["id"] for i in channels])})
            changed = [names[c] for c, layer in zip(channels, layers) if np.any(layer == 1)]
            class_counts.update(changed)
            rows.append({**result, **{k: crop[k] for k in ("dataset_crop_id", "pair_id", "alignment_method",
                         "reference_id", "source_id", "reference_rgb", "source_rgb", "building_id", "view_id", "split")},
                         "crop_root": str(data.history_root), "track_id": key, "event_id": event_id,
                         "previous_year": crop["reference_year"], "current_year": crop["source_year"],
                         "history_start_year": crop["reference_year"],
                         "review_source": f"annotation-derived:{store.unique_id}:{event_id}",
                         "changed_categories": "|".join(changed), "target_path": target_path.relative_to(store.out).as_posix(),
                         "target_sha256": sha256(target_path), "annotation_run": str(store.out),
                         "notes": "Pairwise annotation difference; not reviewed physical event truth"})
        history.write_csv(store.out / "labels.csv", rows, (*history.LABEL_FIELDS, *targets.EXTRA_FIELDS, *PAIR_FIELDS))
        write_json(store.out / "index.json", {"schema_version": 1, "kind": "crop_annotation_targets",
                    "crop_root": str(data.history_root), "crop_index_sha256": sha256(data.index_path),
                    "contract": contract, "observations": observations, "events": rows,
                    "input_sha256": data.fingerprints.files})
        targets.render_audit(data, source, rows, store, observations, args.gallery_limit,
                             title="Перенос целевых масок на парные кропы")
        summary = {"kind": "crop_annotation_targets", "scope": SCOPE, "crops": data.coverage,
                   "events": len(rows), "derived_positive": sum(r["label"] == 1 for r in rows),
                   "derived_stable": sum(r["label"] == 0 for r in rows), "unknown": sum(r["label"] == "" for r in rows),
                   "reasons": dict(Counter(r["reason"] for r in rows)), "changes_by_category": dict(class_counts),
                   "observations": dict(Counter(r["status"] for r in observations.values())),
                   "processing": dict(processing), "reviewed_event_gt_created": False,
                   "unknown_pixels_are_negative": False, "occluder_categories_added": False,
                   "by_split": {part: dict(Counter(str(r["label"]) if r["label"] != "" else "unknown"
                                                for r in rows if r["split"] == part)) for part in args.splits}}
        history.finish_run(store, data, summary)
        return {**summary, "out": str(store.out)}
    except BaseException as exc:
        store.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        store.close()
