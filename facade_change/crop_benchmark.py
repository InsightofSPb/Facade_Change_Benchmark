"""Benchmark frozen crop controls or a complete aligned RGB crop export.

Fit thresholds on clean TRAIN only, keep them for every evaluation split, and
retain raw RGB as the scorer input. Full exports include explicit TEST results;
the existing augmented development protocol remains TRAIN/VAL only.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from functools import lru_cache
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from .annotation_targets import _thumbnail
from .benchmark_config import load_method_options
from .history_features import Fingerprints, digest, progress
from .hypothesis_benchmark import _case_inputs, _counts, _inside, _mask, _threshold_grid
from .io import environment, load_rgb, read_json, sha256, write_json
from .methods.registry import ALL_METHODS, EXTERNAL_METHODS
from .methods.remote import AUXILIARY_ARRAY_NAMES, RemotePool, RemoteScorer, validate_auxiliary_arrays
from .scorers import make_scorer
from scripts.real_eval.real_gt_metrics import evaluate_mask


PROTOCOL = "crop-development-clean-train-v1"
FULL_PROTOCOL = "crop-export-clean-train-v1"
METRICS = ("f1", "iou", "precision", "recall", "fpr")
FIELDS = ("method", "branch", "case_id", "dataset_crop_id", "building_id", "view_id", "split",
          "scenario_id", "nuisance_kind", "state", "hypothesis", "sham_self_paste", "target_type", "category_id", "category",
          "target_origin", "threshold", "tp", "fp", "fn", "tn", "valid_pixels", "ignored_pixels",
          "positive_pixels", "negative_pixels", *METRICS, "scoring_seconds", "score_path", "score_sha256")
IMMUTABLE_METADATA = ("method", "output_kind", "checkpoint", "checkpoints", "weights", "source", "sources",
                      "backbone", "package", "normalization", "settings", "author_settings", "checkpoint_model_args",
                      "device", "worker_python", "worker_environment", "codec", "implementation_version",
                      "representation", "tile_size", "stride", "score_formula", "category_to_channel",
                      "ontology", "ontology_sha256", "model_config", "evaluation_split_audit",
                      "checkpoint_path", "checkpoint_sha256", "checkpoint_dataset_verified", "input_sha256",
                      "reviewed_split", "buildings", "source_sha256", "source_commit", "model_args",
                      "window_groups", "protocol")


class FrozenCrops:
    """One checked RGB/target provider for both branches; no new preparation."""

    protocol = PROTOCOL
    kind = "crop_development_benchmark"

    def __init__(self, dataset_run, targets_run, branches, max_pairs_per_split):
        self.root = Path(dataset_run).expanduser().resolve()
        self.fp = Fingerprints()
        self.fp.add(self.root / "run.json")
        self.parent = read_json(self.root / "run.json")
        if self.parent.get("kind") != "hypothesis_dataset" or self.parent.get("status") not in {"completed", "completed_needs_review"}:
            raise ValueError("--dataset-run must be a completed augmented crop dataset")
        self.index = read_json(self.file("index.json"))
        self.split = read_json(self.file("split.json"))
        summary = read_json(self.file("summary.json"))
        config = read_json(self.file("config.json"))
        cohort = read_json(self.file("cohort_selection.json"))
        if (self.split.get("mode") != "reviewed" or self.split.get("development_only") is not False
                or self.index.get("split") != self.split or cohort.get("kind") != "crop_selection_plan"
                or self.parent.get("config", {}).get("input_sha256") != self.index.get("input_sha256")
                or summary.get("case_count") != len(self.index["cases"])):
            raise ValueError("Augmented data, reviewed split and frozen cohort disagree")
        if self.parent["config"]["input_sha256"]["config"] != sha256(self.root / "config.json"):
            raise ValueError("Augmentation configuration hash mismatch")
        if cohort.get("input_sha256") != self.index["input_sha256"]:
            raise ValueError("Cohort belongs to another crop snapshot")
        pairs, original_owners = {}, {}
        for row in cohort["selected"]:
            identifier, building, part = row["dataset_crop_id"], row["building_id"], row["split"]
            if identifier in pairs or self.split["building_assignments"].get(building) != part:
                raise ValueError("Duplicate cohort crop or building split mismatch")
            pairs[identifier] = row
            for value in row["source_image_sha256"].values():
                if original_owners.setdefault(value, part) != part:
                    raise ValueError("Original image hash leaks between crop splits")
        chosen = set()
        for part in ("train", "val"):
            candidates = sorted((r for r in pairs.values() if r["split"] == part), key=lambda r: digest(r["dataset_crop_id"]))
            chosen.update(r["dataset_crop_id"] for r in (candidates[:max_pairs_per_split] if max_pairs_per_split else candidates))
        if {pairs[k]["split"] for k in chosen} != {"train", "val"}:
            raise ValueError("The development protocol requires both TRAIN and VAL crop pairs")
        self.pairs = {key: pairs[key] for key in chosen}
        self.bases = {b["base_id"]: b for b in self.index["bases"]}
        if len(self.bases) != len(self.index["bases"]):
            raise ValueError("Duplicate synthetic base IDs")
        self.cases, self.events, self.target_root, self.target_record = [], {}, None, None
        self.categories = []
        seen_ids = set()
        if "synthetic" in branches:
            members = cohort["synthetic_members"]
            selected_bases = {b["base_id"] for b in self.bases.values() if chosen.intersection(members.get(b["dataset_crop_id"], []))}
            expected = {(state, s["id"]) for state in config["states"] for s in config["scenarios"]}
            signatures = defaultdict(list)
            for case in self.index["cases"]:
                if case["base_id"] not in selected_bases:
                    continue
                base = self.bases[case["base_id"]]
                if (any(case.get(k) != base.get(k) for k in ("building_id", "view_id", "split"))
                        or self.split["building_assignments"].get(base["building_id"]) != base["split"]
                        or case["parent_dataset_crop_id"] != base["dataset_crop_id"]):
                    raise ValueError("Synthetic case/base identity mismatch")
                signatures[case["base_id"]].append((case["state"], case["scenario_id"]))
                self._append(case, "synthetic", seen_ids)
            for base in selected_bases:
                if len(signatures[base]) != len(expected) or set(signatures[base]) != expected:
                    raise ValueError("Synthetic base has missing/duplicate state-scenario cases")
        if "real" in branches:
            real = read_json(self.file("real_pairs/index.json"))
            if (real.get("kind") != "real_pair_augmentations" or real.get("artifact_root") != ".."
                    or real.get("natural_change_labels_created") is not False):
                raise ValueError("Invalid real-pair augmentation index")
            signatures = defaultdict(list)
            expected = {s["id"] for s in config["scenarios"]}
            for case in real["cases"]:
                key = case["parent_dataset_crop_id"]
                if key not in chosen:
                    continue
                row = pairs[key]
                if (any(case[k] != row[k] for k in ("pair_id", "building_id", "view_id", "split", "reference_year", "source_year"))
                        or case.get("natural_change_labels_created") is not False):
                    raise ValueError("Real case/cohort identity mismatch")
                signatures[key].append(case["scenario_id"])
                self._append(case, "real", seen_ids)
            if any(len(signatures[k]) != len(expected) or set(signatures[k]) != expected for k in chosen):
                raise ValueError("Real crop pair has missing/duplicate nuisance cases")
            self._load_targets(targets_run)
        self.cases.sort(key=lambda r: (r["split"] != "train", r["branch"], r["case_id"]))
        self.selection = dict(protocol=PROTOCOL, max_pairs_per_split=max_pairs_per_split,
            pair_ids=sorted(chosen), case_ids={branch: [r["case_id"] for r in self.cases if r["branch"] == branch] for branch in branches},
            source_pair_count=len(pairs), selected_pair_count=len(chosen),
            selected_by_split=dict(Counter(r["split"] for r in self.pairs.values())),
            algorithm="SHA256(dataset_crop_id) within TRAIN/VAL before model scores; all scenarios; matching synthetic families")

    def _append(self, case, branch, seen):
        identifier = branch, case["case_id"]
        if identifier in seen or case["split"] not in {"train", "val"}:
            raise ValueError("Duplicate case or TEST case in development benchmark")
        seen.add(identifier)
        self.cases.append(dict(case, branch=branch))

    def file(self, relative, expected=None, target=False):
        root, record = (self.target_root, self.target_record) if target else (self.root, self.parent)
        path = _inside(root, relative)
        recorded = record.get("artifact_sha256", {}).get(relative)
        if not recorded or (expected is not None and expected != recorded):
            raise ValueError(f"Missing/inconsistent artifact hash: {relative}")
        old = self.fp.files.get(str(path))
        if old is None:
            self.fp.add(path, recorded)
        elif old != recorded:
            raise ValueError(f"Conflicting artifact hash: {relative}")
        return path

    def _load_targets(self, path):
        if not path:
            raise ValueError("The real branch requires --targets-run from prepare targets --crops")
        self.target_root = Path(path).expanduser().resolve()
        self.fp.add(self.target_root / "run.json")
        self.target_record = read_json(self.target_root / "run.json")
        if self.target_record.get("kind") != "crop_annotation_targets" or self.target_record.get("status") != "completed":
            raise ValueError("--targets-run must be a completed pairwise annotation export")
        index = read_json(self.file("index.json", target=True))
        expected_index = getattr(self, "crop_index_sha256", None) or self.index["input_sha256"]["parent_index"]
        if index.get("crop_index_sha256") != expected_index:
            raise ValueError("Target masks belong to a different crop snapshot")
        contract = index["contract"]
        channels = contract["target_channels"]
        if (not channels or any(type(c) is not int or not 0 <= c < len(contract["categories"]) for c in channels)
                or len(set(channels)) != len(channels)):
            raise ValueError("Invalid target ontology channels")
        self.categories = [contract["categories"][channel] for channel in contract["target_channels"]]
        if len({c["id"] for c in self.categories}) != len(self.categories):
            raise ValueError("Ambiguous target ontology")
        for event in index["events"]:
            key = event["dataset_crop_id"]
            if key not in self.pairs:
                continue
            row = self.pairs[key]
            if (key in self.events or any(event[k] != row[k] for k in ("pair_id", "building_id", "view_id", "split"))
                    or event["previous_year"] != row["reference_year"] or event["current_year"] != row["source_year"]):
                raise ValueError("Target event and frozen crop identity/year disagree")
            self.events[key] = event
            self.file(event["target_path"], event["target_sha256"], target=True)
        if set(self.events) != set(self.pairs):
            raise ValueError("Targets do not cover every selected real crop pair")
        # Validate all selected maps before allocating a model; retain only a small LRU.
        for key in progress(sorted(self.events), "Validate selected target maps"):
            self.target_arrays(key)

    @lru_cache(maxsize=8)
    def target_arrays(self, key):
        event = self.events[key]
        with np.load(self.file(event["target_path"], event["target_sha256"], target=True), allow_pickle=False) as archive:
            if set(archive.files) != {"target", "class_targets", "category_ids"}:
                raise ValueError("Unexpected crop target archive schema")
            target, layers, ids = (archive[k].copy() for k in ("target", "class_targets", "category_ids"))
        shape = tuple(self.pairs[key]["tile_shape"])
        if (target.shape != shape or target.dtype != np.uint8 or layers.shape != (len(self.categories), *shape)
                or layers.dtype != np.uint8 or not np.isin(layers, [0, 1, 255]).all()
                or ids.tolist() != [c["id"] for c in self.categories]):
            raise ValueError("Target grid, labels or category/channel mapping mismatch")
        combined = np.full(shape, 255, np.uint8)
        combined[np.all(layers == 0, axis=0)] = 0
        combined[np.any(layers == 1, axis=0)] = 1
        if not np.array_equal(target, combined):
            raise ValueError("Combined target disagrees with class targets")
        return target, layers

    def inputs(self, case):
        if case["branch"] == "synthetic":
            checked = {}
            reference, source, support, target = _case_inputs(self.root, self.parent, self.bases[case["base_id"]], case, checked)
            for relative, value in checked.items():
                path = str(_inside(self.root, relative))
                if path in self.fp.files and self.fp.files[path] != value:
                    raise ValueError("Synthetic input changed")
                self.fp.files[path] = value  # _case_inputs just checked these exact bytes.
            return reference, source, support, target, None
        paths = {key: self.file(case[key], case["artifact_sha256"][key]) for key in (
            "reference_rgb", "source_rgb", "reference_support", "source_support", "artificial_visibility",
            "original_geometric_overlap", "geometric_visible_support")}
        reference, opaque_a = load_rgb(paths["reference_rgb"])
        source, opaque_b = load_rgb(paths["source_rgb"])
        shape = reference.shape[:2]
        if source.shape != reference.shape or list(shape) != self.pairs[case["parent_dataset_crop_id"]]["tile_shape"]:
            raise ValueError("Real RGB shape differs from the frozen crop grid")
        support = _mask(paths["original_geometric_overlap"], shape)
        reference_support = _mask(paths["reference_support"], shape)
        transformed_support = _mask(paths["source_support"], shape)
        visibility = _mask(paths["artificial_visibility"], shape)
        visible = _mask(paths["geometric_visible_support"], shape)
        if (np.any(support & ~(reference_support & opaque_a & opaque_b))
                or not np.array_equal(visible, support & transformed_support & visibility)):
            raise ValueError("Real geometric/visibility masks disagree")
        target, layers = self.target_arrays(case["parent_dataset_crop_id"])
        if np.any((target != 255) & ~support) or np.any((layers != 255) & ~support):
            raise ValueError("Annotation targets extend outside original geometry")
        target, layers = target.copy(), layers.copy()
        target[~visible], layers[:, ~visible] = 255, 255
        return reference, source, support, target, layers


class ExportCrops(FrozenCrops):
    """Read every requested native crop, retaining stable and unknown examples."""

    protocol = FULL_PROTOCOL
    kind = "crop_export_benchmark"

    def __init__(self, crops, targets_run, splits, max_pairs_per_split):
        if (not splits or len(set(splits)) != len(splits)
                or set(splits) - {"train", "val", "test"} or "train" not in splits):
            raise ValueError("Full crop evaluation needs unique --splits including train for threshold calibration")
        self.root = Path(crops).expanduser().resolve()
        self.fp = Fingerprints()
        self.fp.add(self.root / "run.json")
        self.parent = read_json(self.root / "run.json")
        if (self.parent.get("kind") != "crop_dataset"
                or self.parent.get("status") not in {"completed", "completed_needs_review", "completed_with_issues"}):
            raise ValueError("--crops requires a completed aligned crop export")
        summary = read_json(self.file("summary.json"))
        index_path = self.file(summary["index_path"])
        self.crop_index_sha256 = sha256(index_path)
        self.index = read_json(index_path)
        self.split = read_json(self.file("split.json"))
        if (self.index.get("schema_version") != 1 or self.index.get("split") != self.split
                or self.split.get("mode") != "reviewed" or self.split.get("development_only") is not False
                or len(self.index["crops"]) != summary["crop_count"]):
            raise ValueError("Crop index, summary and reviewed split disagree")
        pairs, views, images, pair_owners = {}, {}, {}, {}
        for row in self.index["crops"]:
            key, building, view, part = (row[k] for k in ("dataset_crop_id", "building_id", "view_id", "split"))
            if (key in pairs or not building or not view or part not in {"train", "val", "test"}
                    or self.split["building_assignments"].get(building) != part
                    or views.setdefault(view, building) != building):
                raise ValueError("Duplicate crop or building/view split leakage")
            if set(row["source_image_sha256"]) != {"reference", "source"}:
                raise ValueError("Crop lacks original image identities")
            for value in row["source_image_sha256"].values():
                if images.setdefault(value, part) != part:
                    raise ValueError("Original image hash leaks between crop splits")
            owner = (building, view, tuple(sorted(row["source_image_sha256"].items())))
            if pair_owners.setdefault(row["pair_id"], owner) != owner:
                raise ValueError("Photo pair has inconsistent owners or original images")
            if (type(row["reference_year"]) is not int or type(row["source_year"]) is not int
                    or row["reference_year"] >= row["source_year"]):
                raise ValueError("Crop pair must have chronological cross-year observations")
            pairs[key] = row
        self.pairs = {}
        for part in splits:
            selected = sorted((row for row in pairs.values() if row["split"] == part),
                              key=lambda row: digest(row["dataset_crop_id"]))
            if not selected:
                raise ValueError(f"No exported crops in requested split: {part}")
            for row in selected[:max_pairs_per_split] if max_pairs_per_split else selected:
                self.pairs[row["dataset_crop_id"]] = row
        self.cases, self.events, self.target_root, self.target_record = [], {}, None, None
        self.categories = []
        for key, row in self.pairs.items():
            directory = _inside(self.root, row["path"])
            metadata = read_json(self.file((directory / "crop.json").relative_to(self.root).as_posix()))
            for name in ("crop_id", "building_id", "view_id", "split", "reference_id", "source_id", "valid_fraction",
                         "bbox_canvas_xyxy", "tile_shape", "reference_to_crop", "source_to_crop", "dense_geometry",
                         "source_image_sha256", "artifact_sha256"):
                if metadata.get(name) != row.get(name):
                    raise ValueError(f"Crop index differs from saved crop.json: {name}")
            shape, bbox = row["tile_shape"], row["bbox_canvas_xyxy"]
            if (len(shape) != 2 or any(type(n) is not int or n < 8 for n in shape)
                    or len(bbox) != 4 or any(type(n) is not int for n in bbox)
                    or [bbox[3] - bbox[1], bbox[2] - bbox[0]] != shape
                    or not 0 < row["valid_fraction"] <= 1):
                raise ValueError("Invalid native crop rectangle/support")
            if (row["pair_id"] != f"{row['reference_id']}-{row['source_id']}"
                    or key != row["pair_id"] + "/" + row["crop_id"]):
                raise ValueError("Crop identity differs from its observations")
            for name in ("reference_rgb", "source_rgb", "reference_support", "source_support", "geometric_overlap"):
                relative = (directory / (name + ".png")).relative_to(self.root).as_posix()
                if name in row and row[name] != relative:
                    raise ValueError(f"Indexed crop artifact path disagrees: {name}")
                self.file(relative, row["artifact_sha256"][name + ".png"])
            self.cases.append(dict(case_id=key, parent_dataset_crop_id=key, building_id=row["building_id"],
                                   view_id=row["view_id"], split=row["split"], branch="real",
                                   scenario_id="clean", nuisance_kind="identity"))
        self._load_targets(targets_run)
        target_index = read_json(self.file("index.json", target=True))
        crop_categories = sorted((c["id"], c["name"]) for c in self.index["categories"])
        target_categories = [(c["id"], c["name"]) for c in target_index["contract"]["categories"]]
        if crop_categories != target_categories or len({c[0] for c in crop_categories}) != len(crop_categories):
            raise ValueError("Crop and target category mappings disagree")
        self.coverage = {part: dict(crop_pairs=0, known_pairs=0, positive_pairs=0, unknown_pairs=0,
                                   known_pixel_contributions=0, ignored_pixel_contributions=0) for part in splits}
        for case in progress(self.cases, "Validate native RGB, support and targets"):
            _, _, _, target, _ = self.inputs(case)
            counts = self.coverage[case["split"]]
            known = int(np.count_nonzero(target != 255))
            counts["crop_pairs"] += 1
            counts["known_pairs"] += int(known > 0)
            counts["positive_pairs"] += int(np.any(target == 1))
            counts["unknown_pairs"] += int(known == 0)
            counts["known_pixel_contributions"] += known
            counts["ignored_pixel_contributions"] += target.size - known
        self.selection = dict(protocol=self.protocol, max_pairs_per_split=max_pairs_per_split,
            requested_splits=list(splits), pair_ids=sorted(self.pairs), case_ids={"real": [c["case_id"] for c in self.cases]},
            source_pair_count=len(pairs), selected_pair_count=len(self.pairs),
            source_by_split=dict(Counter(r["split"] for r in pairs.values())),
            selected_by_split=dict(Counter(r["split"] for r in self.pairs.values())),
            complete_requested_splits=max_pairs_per_split == 0,
            algorithm="All exported crop IDs in requested splits; optional SHA256-ranked smoke limit; no annotation selection")

    def inputs(self, case):
        row = self.pairs[case["parent_dataset_crop_id"]]
        directory = _inside(self.root, row["path"])
        paths = {name: self.file((directory / (name + ".png")).relative_to(self.root).as_posix(),
                                row["artifact_sha256"][name + ".png"])
                 for name in ("reference_rgb", "source_rgb", "reference_support", "source_support", "geometric_overlap")}
        reference, opaque_a = load_rgb(paths["reference_rgb"])
        source, opaque_b = load_rgb(paths["source_rgb"])
        shape = tuple(row["tile_shape"])
        if reference.shape != (*shape, 3) or source.shape != reference.shape:
            raise ValueError("Crop RGB differs from the native grid")
        a, b, support = (_mask(paths[name], shape) for name in
                         ("reference_support", "source_support", "geometric_overlap"))
        if (not np.array_equal(support, a & b) or np.any(support & ~(opaque_a & opaque_b))
                or not np.isclose(support.mean(), row["valid_fraction"])):
            raise ValueError("Crop geometry, alpha and valid_fraction disagree")
        target, layers = self.target_arrays(case["parent_dataset_crop_id"])
        if np.any((target != 255) & ~support) or np.any((layers != 255) & ~support):
            raise ValueError("Annotation targets extend outside crop geometry")
        return reference, source, support, target, layers


def _identity(metadata):
    result = {key: metadata[key] for key in IMMUTABLE_METADATA if key in metadata}
    if "worker_environment" in result:
        result["worker_environment"] = {k: v for k, v in result["worker_environment"].items()
                                        if k not in {"git_commit", "git_dirty"}}
    return result


def _pin_model_files(value, fp):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            fp.add(value["path"], value["sha256"])
        if isinstance(value.get("checkpoint_path"), str) and isinstance(value.get("checkpoint_sha256"), str):
            fp.add(value["checkpoint_path"], value["checkpoint_sha256"])
        for item in value.values():
            _pin_model_files(item, fp)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _pin_model_files(item, fp)


def _validate_arrays(arrays, support, native, metadata):
    if "scores" not in arrays or set(arrays)-{"scores", "raw_scores", "native_prediction"}-AUXILIARY_ARRAY_NAMES:
        raise ValueError("Unexpected method output archive schema")
    scores = arrays["scores"]
    if (scores.shape != support.shape or scores.dtype != np.float32 or not np.isfinite(scores[support]).all()
            or not np.isnan(scores[~support]).all() or np.any(scores[support] < 0) or np.any(scores[support] > 1)):
        raise ValueError("Invalid native-grid score map")
    if "raw_scores" in arrays:
        raw = arrays["raw_scores"]
        if (raw.shape != support.shape or raw.dtype != np.float32 or not np.isfinite(raw[support]).all()
                or not np.isnan(raw[~support]).all() or np.any(raw[support] < 0)):
            raise ValueError("Invalid native raw score map")
    if "native_prediction" in arrays and (arrays["native_prediction"].dtype != bool or arrays["native_prediction"].shape != support.shape):
        raise ValueError("Invalid native prediction map")
    if native and ("native_prediction" not in arrays or not np.array_equal(scores[support], arrays["native_prediction"][support].astype(np.float32))):
        raise ValueError("Native-mask methods must retain the author's binary prediction")
    validate_auxiliary_arrays({k: v for k, v in arrays.items() if k in AUXILIARY_ARRAY_NAMES}, support, metadata)


def _gpu_report(device):
    if not str(device).startswith("cuda"):
        return None
    if not shutil.which("nvidia-smi"):
        return dict(device=device, nvidia_smi="unavailable; worker validates CUDA")
    result = {}
    for name, query in (("memory", ["--query-gpu=index,name,memory.total,memory.free", "--format=csv"]),
                        ("processes", ["--query-compute-apps=pid,process_name,used_memory", "--format=csv"])):
        command = subprocess.run(["nvidia-smi", *query], capture_output=True, text=True, timeout=20)
        result[name] = command.stdout.strip() or command.stderr.strip()
        print(result[name], flush=True)
    return result


def _calibrate(rows, grid):
    """Building macro over clean TRAIN ROIs; no VAL or nuisance-specific fitting."""
    positive, negative = defaultdict(list), defaultdict(list)
    for case, counts in rows:
        if case["split"] != "train" or case["nuisance_kind"] != "identity":
            raise ValueError("Calibration accepts clean TRAIN only")
        if case.get("sham_self_paste"):
            continue
        tp, fp, fn, tn = counts
        if np.any(tp+fn > 0):
            positive[case["building_id"]].append(2*tp/np.maximum(2*tp+fp+fn, 1))
        if np.any(fp+tn > 0):
            negative[case["building_id"]].append(fp/np.maximum(fp+tn, 1))
    if not positive or not negative:
        raise ValueError("Clean TRAIN needs known positive and negative pixels; increase the explicit smoke subset or inspect masks")
    f1 = np.mean([np.mean(v, axis=0) for v in positive.values()], axis=0)
    fpr = np.mean([np.mean(v, axis=0) for v in negative.values()], axis=0)
    tied = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))
    tied = tied[np.isclose(fpr[tied], fpr[tied].min(), rtol=0, atol=1e-12)]
    selected = int(tied[-1])
    return dict(threshold=float(grid[selected]), selection_split="train", selection_scenario="identity",
                criterion="max building-macro positive-ROI F1; then min building-macro known-negative FPR; then highest threshold",
                prediction_rule="float32 score > threshold", positive_buildings=len(positive), negative_buildings=len(negative),
                curve=[dict(threshold=float(t), f1=float(f1[i]), fpr=float(fpr[i])) for i, t in enumerate(grid)])


class Macro:
    """Stream crop metrics; first average within each building, then across buildings."""
    def __init__(self):
        self.groups = {}

    def add(self, row):
        if row["sham_self_paste"]:
            return
        for scenario in ("all", row["scenario_id"]):
            for hypothesis in (("all", row["hypothesis"]) if row["hypothesis"] else ("all",)):
                key = (row["method"], row["branch"], row["split"], row["target_type"], row["category_id"], scenario, hypothesis)
                group = self.groups.setdefault(key, dict(category=row["category"], buildings={}, cases=0))
                building = group["buildings"].setdefault(row["building_id"], {m: [0., 0] for m in METRICS})
                group["cases"] += 1
                for name in METRICS:
                    if row[name] is not None:
                        building[name][0] += row[name]
                        building[name][1] += 1

    def rows(self):
        rows = []
        for key, group in sorted(self.groups.items(), key=lambda item: str(item[0])):
            row = dict(zip(("method", "branch", "split", "target_type", "category_id", "scenario_id", "hypothesis"), key))
            row.update(category=group["category"], case_count=group["cases"], building_count=len(group["buildings"]))
            for metric in METRICS:
                values = [v[metric][0]/v[metric][1] for v in group["buildings"].values() if v[metric][1]]
                row[metric] = float(np.mean(values)) if values else None
                row[metric+"_building_count"] = len(values)
            rows.append(row)
        return rows


def _gallery_case_ids(cases, limit):
    groups = defaultdict(list)
    for case in sorted(cases, key=lambda c: digest([c["building_id"], c["case_id"]])):
        groups[case["branch"], case["split"]].append(case)
    selected, building_counts = set(), Counter()
    while len(selected) < limit and any(groups.values()):
        for key in sorted(groups):
            if not groups[key] or len(selected) >= limit:
                continue
            case = min(groups[key], key=lambda c: (building_counts[key, c["building_id"]], digest(c["case_id"])))
            groups[key].remove(case)
            selected.add((case["branch"], case["case_id"]))
            building_counts[key, case["building_id"]] += 1
    return selected


def _gallery_row(case, method, reference, source, target, scores, prediction):
    import html
    known = target != 255
    truth = np.full((*target.shape, 3), 120, np.uint8)
    truth[known] = 20
    truth[target == 1] = [245, 110, 35]
    heat = np.zeros_like(truth)
    values = np.nan_to_num(scores, nan=0.)
    heat[..., 0], heat[..., 2] = np.rint(255*values).astype(np.uint8), np.rint(255*(1-values)).astype(np.uint8)
    heat[~np.isfinite(scores)] = 120
    pred = np.full_like(truth, 120)
    pred[known] = 20
    pred[known & prediction] = [245, 110, 35]
    label = html.escape(f"{method} / {case['branch']} / {case['split']} / {case['building_id']} / {case['scenario_id']}")
    return '<tr><td>'+label+'</td>'+''.join('<td>'+_thumbnail(a)+'</td>' for a in (reference, source, truth, heat, pred))+'</tr>'


def run(args):
    raw_export = bool(getattr(args, "crops", None))
    args.branches = args.branches or (["real"] if raw_export else ["synthetic", "real"])
    if (not args.methods or len(set(args.methods)) != len(args.methods) or set(args.methods)-set(ALL_METHODS)
            or not args.branches or set(args.branches)-{"synthetic", "real"} or len(set(args.branches)) != len(args.branches)
            or args.max_pairs_per_split < 0 or args.gallery_limit < 0):
        raise ValueError("Invalid methods, branches or subset/gallery budget")
    if raw_export:
        if args.branches != ["real"]:
            raise ValueError("--crops evaluates original real pairs only; use --branches real")
        args.splits = args.splits or ["train", "val", "test"]
        data = ExportCrops(args.crops, args.targets_run, args.splits, args.max_pairs_per_split)
    else:
        if getattr(args, "splits", None) is not None:
            raise ValueError("--splits belongs to --crops; augmented development data retain TRAIN/VAL")
        data = FrozenCrops(args.dataset_run, args.targets_run, args.branches, args.max_pairs_per_split)
    protocol, run_kind = data.protocol, data.kind
    actual_splits = sorted({case["split"] for case in data.cases})
    out = Path(args.out).expanduser().resolve()
    for root in (data.root, data.target_root):
        if root and (out == root or out in root.parents or root in out.parents):
            raise ValueError("Benchmark output must be separate from its input runs")
    if args.resume and (out / "run.json").exists():
        old = read_json(out / "run.json")
        if old.get("kind") != run_kind or old.get("status") == "completed":
            raise ValueError("Resume requires an interrupted crop benchmark; completed outputs are immutable")
    options = load_method_options(args.methods_config) if args.methods_config else {}
    if args.methods_config:
        data.fp.add(args.methods_config)
    if set(options)-set(ALL_METHODS):
        raise ValueError("Unknown method name in methods configuration")
    for method in args.methods:
        if method.startswith("msdzip_") and not options.get(method, {}).get("checkpoint_path"):
            raise ValueError(f"{method} requires checkpoint_path in --methods-config")
        if raw_export and (method.startswith("arib_bps_") or method == "bcm_net_rgb"):
            raise ValueError(f"{method}: transfer from its training dataset to --crops has not been validated")
    modules = [Path(__file__), Path(__file__).with_name("benchmark_config.py"),
               Path(__file__).with_name("hypothesis_benchmark.py"), Path(__file__).with_name("scorers.py"),
               Path(__file__).with_name("history_features.py"), Path(__file__).with_name("io.py"),
               Path(__file__).with_name("2026-10-14_temporal_experiment.py"),
               Path(__file__).with_name("2026-10-04_msdzip_h0.py"),
               Path(__file__).parent.parent / "scripts/real_eval/real_gt_metrics.py"]
    modules += sorted((Path(__file__).parent / "methods").glob("*.py"))
    source_hashes = {p.relative_to(Path(__file__).parent.parent).as_posix(): sha256(p) for p in modules}
    for path in modules:
        data.fp.add(path)
    snapshot_hash = digest(data.fp.files)
    selection_hash = digest(data.selection)
    config = {k: v for k, v in vars(args).items() if k != "resume"}
    runtime = environment()
    RunStore = importlib.import_module("facade_change.2026-10-14_temporal_experiment").RunStore
    store = RunStore(out, config, resume=args.resume)
    store.record["kind"] = run_kind
    pool = RemotePool(log_dir=out / "worker_logs")
    processed, computed, reused, thresholds, macro, html_rows = 0, 0, 0, {}, Macro(), []
    gallery_ids = _gallery_case_ids(data.cases, args.gallery_limit)
    try:
        store.contract(dict(protocol=protocol, inputs=dict(data.fp.files), sources=source_hashes,
            runtime={k: runtime[k] for k in ("python", "platform", "packages")},
            selection=data.selection, categories=data.categories,
            target_scope=("annotation-derived real changes; unknown pixels ignored" if raw_export else
                          "procedural edit GT and separate annotation-derived real changes; unknown pixels ignored"),
            inference_inputs="RGB pair and original geometry only; no target masks or artificial visibility",
            calibration="clean TRAIN only, one threshold per method/branch; sham excluded; fixed for all evaluation cases",
            statistical_unit=("building; overlapping crops are correlated; means over exported crop pairs, not whole-facade IoU"
                              if raw_export else "building; selected overlapping crops are correlated; exploratory development cohort; no TEST"),
            split_roles=(dict(train="calibration and in-sample diagnostics; semantic checkpoints may have seen these images",
                              val="separate validation", test="separate frozen-threshold evaluation",
                              checkpoint_independence="not established by this crop split; retain checkpoint provenance")
                         if raw_export else dict(train="calibration and diagnostics", val="development evaluation"))))
        write_json(out / "selection.json", data.selection)
        write_json(out / "split.json", data.split)
        writers = {}
        with (out / "cases.csv").open("w", encoding="utf-8", newline="") as case_stream, \
                (out / "classes.csv").open("w", encoding="utf-8", newline="") as class_stream:
            for key, stream in (("event", case_stream), ("class", class_stream)):
                writers[key] = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
                writers[key].writeheader()
            for method in args.methods:
                settings = {"device": args.device, "trust_checkpoint": args.trust_checkpoint, **options.get(method, {})}
                if method == "geoscd":
                    settings.pop("trust_checkpoint", None)
                if raw_export and method.startswith("msdzip_"):
                    settings.pop("dataset_fingerprint", None)
                    settings["evaluation_split"] = data.split
                elif method.startswith(("arib_bps_", "msdzip_")) or method == "bcm_net_rgb":
                    settings["dataset_fingerprint"] = data.index["input_sha256"]
                use_worker = method in EXTERNAL_METHODS or (
                    method.startswith("msdzip_") and bool(settings.get("worker_python")))
                if method not in EXTERNAL_METHODS:
                    allowed = {"device", "trust_checkpoint", "checkpoint_path", "dataset_fingerprint",
                               "compression_tile_size", "compression_stride", "zstd_level", "lzma_preset", "evaluation_split"}
                    if use_worker:
                        allowed.update(("worker_python", "worker_timeout"))
                    settings = {k: v for k, v in settings.items() if k in allowed}
                report = _gpu_report(settings.get("device"))
                if report:
                    store.event("gpu_preflight", method=method, report=report)
                scorer = RemoteScorer(method, settings, pool) if use_worker else make_scorer(method, **settings)
                if isinstance(scorer, RemoteScorer):
                    scorer.activate()
                identity = _identity(scorer.metadata)
                method_contract = out / "method_contracts" / (method+".json")
                if method_contract.exists():
                    if read_json(method_contract) != identity:
                        raise ValueError("Model/weight/worker identity changed since the interrupted run")
                else:
                    method_contract.parent.mkdir(exist_ok=True)
                    write_json(method_contract, identity)
                _pin_model_files(identity, data.fp)
                store.event("method_activated", method=method, identity=identity)
                native = scorer.metadata.get("output_kind") == "native_mask"
                grid = _threshold_grid(method)
                method_hash = digest([snapshot_hash, selection_hash, method, settings, identity])

                def score_case(case):
                    nonlocal computed, reused
                    reference, source, support, target, layers = data.inputs(case)
                    identifier = digest([method, case["branch"], case["case_id"]])
                    input_hash = digest([method_hash, case])
                    cached = store.read_unit(identifier)
                    if cached:
                        record, arrays = cached
                        if record.get("input_sha256") != input_hash:
                            raise ValueError("Cached map differs from this method/case/input contract")
                        reused += 1
                    else:
                        started = time.perf_counter()
                        scores = np.asarray(scorer(reference.copy(), source.copy(), support.copy()))
                        seconds = time.perf_counter()-started
                        if _identity(scorer.metadata) != identity:
                            raise ValueError("Method source/weights/worker settings changed during inference")
                        arrays = {"scores": scores}
                        for name in ("raw_scores", "native_prediction"):
                            value = getattr(scorer, name, None)
                            if value is not None:
                                arrays[name] = np.asarray(value)
                        auxiliary = getattr(scorer, "auxiliary_arrays", {})
                        validate_auxiliary_arrays(auxiliary, support, scorer.metadata)
                        arrays.update(auxiliary)
                        _validate_arrays(arrays, support, native, scorer.metadata)
                        record = store.save_unit(dict(unit_id=identifier, method=method, branch=case["branch"],
                            case_id=case["case_id"], input_sha256=input_hash, scoring_seconds=seconds), arrays)
                        computed += 1
                    _validate_arrays(arrays, support, native, scorer.metadata)
                    return record, arrays, reference, source, target, layers

                calibration = defaultdict(list)
                for case in progress([c for c in data.cases if c["split"] == "train" and c["nuisance_kind"] == "identity"],
                                     f"{method}: clean TRAIN"):
                    record, arrays, _, _, labels, _ = score_case(case)
                    calibration[case["branch"]].append((case, _counts(arrays["scores"], labels, grid)))
                thresholds[method] = {branch: dict(threshold=.5, criterion="author native binary mask; no calibration",
                    selection_split=None, selection_scenario=None) if native else _calibrate(calibration[branch], grid)
                    for branch in args.branches}
                write_json(out / "threshold_selection.json", dict(protocol=protocol, methods=thresholds))
                store.event("thresholds_frozen", method=method, thresholds={b: t["threshold"] for b, t in thresholds[method].items()})
                for case in progress(data.cases, f"{method}: frozen thresholds"):
                    record, arrays, reference, source, target, layers = score_case(case)
                    threshold = thresholds[method][case["branch"]]["threshold"]
                    prediction = arrays["native_prediction"].copy() if native else arrays["scores"] > np.float32(threshold)
                    common = dict(method=method, branch=case["branch"], case_id=case["case_id"],
                        dataset_crop_id=case["parent_dataset_crop_id"], building_id=case["building_id"],
                        view_id=case["view_id"], split=case["split"], scenario_id=case["scenario_id"],
                        nuisance_kind=case["nuisance_kind"], state=case.get("state", "real_observation"),
                        hypothesis=case.get("hypothesis", ""),
                        sham_self_paste=case.get("sham_self_paste", False), threshold=threshold,
                        target_origin="procedural" if case["branch"] == "synthetic" else "annotation_derived",
                        scoring_seconds=record["scoring_seconds"], score_path=record["artifact_path"], score_sha256=record["artifact_sha256"])
                    targets = [("event", None, "all_selected_targets", target)]
                    if layers is not None:
                        targets += [("class", c["id"], c["name"], layer) for c, layer in zip(data.categories, layers)]
                    for kind, cid, category, labels in targets:
                        row = dict(common, target_type=kind, category_id=cid, category=category, **evaluate_mask(prediction, labels))
                        if row["positive_pixels"] == 0:
                            # A stable/unknown ROI has no positive-change localization
                            # task. Its false signals remain in counts and FPR.
                            row.update(f1=None, iou=None, precision=None, recall=None)
                        writers[kind].writerow(row)
                        macro.add(row)
                    processed += 1
                    if (case["branch"], case["case_id"]) in gallery_ids:
                        html_rows.append(_gallery_row(case, method, reference, source, target, arrays["scores"], prediction))
                if hasattr(scorer, "close"):
                    scorer.close()
                pool.release()
        aggregates = macro.rows()
        write_json(out / "metrics.json", dict(protocol=protocol, aggregation="case mean within building, then equally weighted buildings",
            undefined_metrics="null; no known pixels, or zero metric denominator; report contributing building counts",
            positive_metrics="F1/IoU/precision/recall on known-positive ROIs only; negatives remain in counts and FPR",
            sham_policy="saved in cases.csv, excluded from primary aggregates and threshold fitting",
            real_scope="agreement with annotation-derived target-class change maps on known support; not semantic-state segmentation or independent event GT",
            aggregates=aggregates))
        page = ['<!doctype html><meta charset="utf-8"><title>Crop benchmark</title>',
                '<style>body{font:16px system-ui;margin:24px}td{padding:6px;vertical-align:top}</style>',
                '<h1>Frozen crop comparisons</h1><p>Reference / source / target (gray unknown) / score / prediction on known support.</p>',
                '<p>Clean TRAIN thresholds; unchanged across evaluation splits. TRAIN is in-sample diagnostics. '
                'Real targets are annotation-derived; checkpoint independence requires its own provenance.</p><table>', *html_rows, '</table>']
        (out / "gallery.html").write_text('\n'.join(page), encoding="utf-8")
        data.fp.verify()
        summary = dict(kind=run_kind, status="completed", protocol=protocol,
            selected_pair_count=data.selection["selected_pair_count"], methods=args.methods,
            cases_by_branch=dict(Counter(c["branch"] for c in data.cases)),
            case_method_evaluations=processed, computed_maps=computed, reused_map_reads=reused,
            threshold_split="clean TRAIN", evaluation_split=("; ".join("TRAIN diagnostics" if part == "train" else part.upper() + " evaluation"
                                                                      for part in args.splits)
                                                            if raw_export else "VAL; TRAIN diagnostics also saved"),
            test_used="test" in actual_splits,
            targets_are_annotation_derived="real" in args.branches, categories=data.categories,
            metrics_path="metrics.json", cases_path="cases.csv", classes_path="classes.csv", gallery_path="gallery.html",
            val_overall=[r for r in aggregates if r["split"] == "val" and r["target_type"] == "event"
                         and r["scenario_id"] == "all" and r["hypothesis"] == "all"])
        if raw_export:
            summary.update(selected_by_split=data.selection["selected_by_split"], coverage_by_split=data.coverage,
                           complete_requested_splits=data.selection["complete_requested_splits"],
                           overall_by_split={part: [r for r in aggregates if r["split"] == part
                               and r["target_type"] == "event" and r["scenario_id"] == "all" and r["hypothesis"] == "all"]
                               for part in actual_splits})
        write_json(out / "verified_inputs.json", data.fp.files)
        write_json(out / "summary.json", summary)
        store.finish("completed", summary=summary, artifact_sha256={p.relative_to(out).as_posix(): sha256(p)
            for p in sorted(out.rglob("*")) if p.is_file() and p.name not in {"run.json", "events.jsonl", ".run.lock"}})
        return {**summary, "out": str(out)}
    except BaseException as exc:
        store.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        pool.release()
        store.close()


def parser():
    cli = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    inputs = cli.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--dataset-run", help="Completed augmented crop run (existing TRAIN/VAL protocol)")
    inputs.add_argument("--crops", help="Completed native crop export; evaluate all requested crop IDs")
    cli.add_argument("--targets-run", help="Completed prepare targets --crops export; required for the real branch")
    cli.add_argument("--out", required=True)
    cli.add_argument("--branches", nargs="+", choices=("synthetic", "real"),
                     help="Default: both augmented branches, or real only for --crops")
    cli.add_argument("--splits", nargs="+", choices=("train", "val", "test"),
                     help="With --crops: default train val test; train is required for calibration")
    cli.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=["rgb_diff", "ssim"])
    cli.add_argument("--methods-config", help="Existing local source/weights/worker-Python method options")
    cli.add_argument("--max-pairs-per-split", type=int, default=0, help="SHA-ranked smoke subset; 0 uses all pairs in the input protocol")
    cli.add_argument("--device", default="cpu")
    cli.add_argument("--trust-checkpoint", action="store_true")
    cli.add_argument("--gallery-limit", type=int, default=48)
    cli.add_argument("--resume", action="store_true")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"crop benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
