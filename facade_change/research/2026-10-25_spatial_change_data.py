"""Validated spatial tokens, derived targets and optional predicted context."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import importlib
from pathlib import Path

import numpy as np
from PIL import Image

from ..history_features import _inside, digest
from ..io import read_json, sha256


history = importlib.import_module("facade_change.research.2026-10-25_history_models")


def checked_categories(categories):
    if (not categories or any(not isinstance(c, dict) or "id" not in c
                              or not isinstance(c.get("name"), str) or not c["name"] for c in categories)
            or len({str(c["id"]) for c in categories}) != len(categories)
            or len({c["name"] for c in categories}) != len(categories)):
        raise ValueError("Categories require unique stable IDs and names in channel order")
    return categories


class SpatialData:
    def __init__(self, features, targets, splits=("train", "val"), history_root=None,
                 target_features=None, categories=None, auxiliary=None, smoke_buildings=0):
        self.history = history.load_histories(features, splits, 3, history_root)
        if smoke_buildings < 0:
            raise ValueError("smoke_buildings cannot be negative")
        if smoke_buildings:
            selected_buildings = set()
            for split in splits:
                selected_buildings.update(sorted({s["building_id"] for s in self.history.samples
                                                  if s["split"] == split})[:smoke_buildings])
            self.history.samples = [s for s in self.history.samples if s["building_id"] in selected_buildings]
        self.fp = self.history.fingerprints
        index_path = Path(targets).expanduser().resolve()
        if index_path.is_dir():
            index_path /= "index.json"
        self.fp.add(index_path)
        target_index = read_json(index_path)
        self.target_root = index_path.parent
        self.fp.add(self.target_root / "run.json")
        if (target_index.get("kind") != "history_annotation_targets"
                or target_index.get("schema_version") != 1
                or read_json(self.target_root / "run.json").get("status") != "completed"):
            raise ValueError("Expected completed history-targets index.json schema 1")
        source_encoder = self.history.index["encoder"]
        original_sha = target_index["feature_index_sha256"]
        if original_sha != sha256(self.history.index_path):
            candidates = [Path(target_features)] if target_features else [
                Path(p) for p, value in target_index["input_sha256"].items()
                if value == original_sha and Path(p).is_file()]
            if not candidates:
                raise ValueError("New token cache needs original target --target-features index to prove compatibility")
            original_path = candidates[0].expanduser().resolve()
            self.fp.add(original_path, original_sha)
            original = read_json(original_path)
            source_encoder = original["encoder"]
            old = {k: v for k, v in source_encoder.items() if k != "save_patch_tokens"}
            new = {k: v for k, v in self.history.index["encoder"].items() if k != "save_patch_tokens"}
            if old != new or original["history_manifest_sha256"] != self.history.index["history_manifest_sha256"]:
                raise ValueError("Target/token caches differ in encoder, preprocessing or history geometry")
        if not self.history.index["encoder"].get("save_patch_tokens"):
            raise ValueError("Spatial training requires history-features --save-patch-tokens in a new cache")
        contract = target_index["contract"]
        all_categories = checked_categories(contract["categories"])
        selected = contract["target_channels"]
        if (not selected or len(set(selected)) != len(selected)
                or any(not isinstance(i, int) or i < 0 or i >= len(all_categories) for i in selected)):
            raise ValueError("Invalid target channel projection")
        exported = [all_categories[i] for i in selected]
        names = {c["name"] for c in exported}
        if categories and (len(set(categories)) != len(categories) or set(categories) - names):
            raise ValueError(f"Unknown/duplicate target categories; available: {sorted(names)}")
        self.channels = [i for i, c in enumerate(exported) if not categories or c["name"] in categories]
        self.exported_ids = np.asarray([c["id"] for c in exported])
        self.categories = [exported[i] for i in self.channels]
        self.auxiliary, self.auxiliary_root = None, None
        self.extra_spec = {"channels": [], "source": None}
        if auxiliary:
            path = Path(auxiliary).expanduser().resolve()
            self.fp.add(path)
            aux = read_json(path)
            if (aux.get("kind") != "spatial_auxiliary_maps" or aux.get("schema_version") != 1
                    or aux.get("history_manifest_sha256") != self.history.index["history_manifest_sha256"]
                    or aux.get("source_kind") not in ("prediction", "rgb_derived")
                    or not aux.get("source", {}).get("id") or not aux["source"].get("revision")):
                raise ValueError("Auxiliary maps require compatible geometry and versioned prediction/RGB provenance")
            checked_categories(aux["channels"])
            if any(c.get("kind") not in ("context", "signal") for c in aux["channels"]):
                raise ValueError("Auxiliary channel kind must be context or signal")
            self.auxiliary, self.auxiliary_root = aux, path.parent
            self.extra_spec = {k: aux[k] for k in ("channels", "source", "source_kind")}
        self.samples, self.coverage = [], Counter()
        self.class_coverage = {}
        events = {}
        for row in target_index["events"]:
            if row["event_id"] in events:
                raise ValueError("Duplicate target event")
            events[row["event_id"]] = row
        features_by_id = self.history.index.get("features", {})
        for sample in self.history.samples:
            row = events.get(sample["event_id"])
            if row is None:
                self.coverage["missing_target_events"] += 1
                continue
            track = self.history.tracks[sample["track_id"]]
            prefix = [o for o in track["observations"] if o["year"] <= sample["current_year"]]
            vectors = np.concatenate((sample["past"], sample["current"][None]), axis=0)
            expected = digest({"encoder": source_encoder, "track_id": sample["track_id"],
                               "observations": prefix, "years": sample["years"].tolist(),
                               "z_sha256": hashlib.sha256(vectors.tobytes()).hexdigest()})
            if (row["prefix_sha256"] != expected or any(row[k] != sample[k] for k in
                    ("track_id", "building_id", "view_id", "split", "previous_year", "current_year"))):
                raise ValueError("Target prefix identity differs from feature observations/geometry")
            start = int(row.get("history_start_year") or sample["years"][0])
            if start != int(sample["years"][0]):
                raise ValueError("Episode boundaries require a separately validated target export; no implicit reset")
            # One deterministic observation per year, identical for neural and RGB baselines.
            observations = [min((o for o in prefix if o["year"] == year), key=lambda o: o["observation_id"])
                            for year in sample["years"]]
            for observation in observations:
                oid = observation["observation_id"]
                entry = features_by_id.get(oid)
                if not entry or entry.get("signature", {}) != {
                        "encoder_sha256": digest(self.history.index["encoder"]),
                        "rgb_sha256": observation["rgb_sha256"],
                        "support_sha256": observation["support_sha256"]}:
                    raise ValueError(f"Missing/incompatible spatial feature record: {oid}")
                self.fp.add(_inside(self.history.index_path.parent, entry["path"]), entry["sha256"])
                self.fp.add(_inside(self.history.history_root, observation["support"]), observation["support_sha256"])
                self.fp.add(_inside(self.history.history_root, observation["rgb"]), observation["rgb_sha256"])
                if self.auxiliary is not None:
                    width, height = track["bbox_xywh"][2:]
                    self.extra(observation, track["bbox_xywh"], (height // 14, width // 14))
            path = _inside(self.target_root, row["target_path"])
            self.fp.add(path, row["target_sha256"])
            record = {**sample, "observations": observations, "bbox_xywh": track["bbox_xywh"],
                      "target_path": str(path), "target_sha256": row["target_sha256"]}
            target = self.target(record)
            geometry = [self.support(str(_inside(self.history.history_root, o["support"])))
                        for o in observations[-2:]]
            if any(mask.shape != target.shape[-2:] for mask in geometry):
                raise ValueError("Selected pair support does not match target geometry")
            target[:, ~(geometry[0] & geometry[1])] = 255
            count = self.class_coverage.setdefault(sample["split"], np.zeros((len(self.categories), 3), np.int64))
            count[:, 0] += (target == 1).sum((1, 2))
            count[:, 1] += (target == 0).sum((1, 2))
            count[:, 2] += (target == 255).sum((1, 2))
            if not np.any(target != 255):
                self.coverage["all_unknown_events"] += 1
                continue
            self.samples.append(record)
        if not self.samples:
            raise ValueError("No spatial targets with known pixels in requested splits")
        self.coverage["eligible_events"] = len(self.samples)
        self.coverage["smoke_buildings_per_split"] = smoke_buildings
        self.dimension = self.history.index["encoder"]["dimensions"]
        first = self.load(self.samples[0])
        self.identity = {"history_manifest_sha256": self.history.index["history_manifest_sha256"],
                         "encoder": self.history.index["encoder"], "target_categories": self.categories,
                         "auxiliary": self.extra_spec, "same_year_policy": "lexicographically first observation_id",
                         "feature_grid": list(first["features"].shape[-2:]),
                         "target_scope": contract["target_scope"],
                         "supervision": "derived annotation differences; not independently reviewed events"}

    @lru_cache(maxsize=96)
    def arrays(self, path):
        with np.load(path, allow_pickle=False) as archive:
            return {k: archive[k].copy() for k in archive.files}

    def target(self, sample):
        arrays = self.arrays(sample["target_path"])
        width, height = sample["bbox_xywh"][2:]
        labels = arrays["class_targets"]
        if (labels.shape != (len(self.exported_ids), height, width) or labels.dtype != np.uint8
                or not np.array_equal(arrays["category_ids"], self.exported_ids)
                or not np.isin(labels, (0, 1, 255)).all()):
            raise ValueError("Target shape, channel order or 0/1/255 semantics mismatch")
        return labels[self.channels].copy()

    @lru_cache(maxsize=96)
    def support(self, path):
        with Image.open(path) as image:
            array = np.array(image)
        if array.dtype != np.uint8 or array.ndim != 2 or not np.isin(array, (0, 255)).all():
            raise ValueError("Observation support must be native binary uint8")
        return array == 255

    def extra(self, observation, bbox, shape):
        channels = len(self.extra_spec["channels"])
        zeros = np.zeros((channels, *shape), np.float32)
        absent = np.zeros(zeros.shape, bool)
        if self.auxiliary is None or observation["observation_id"] not in self.auxiliary["observations"]:
            return zeros, absent
        row = self.auxiliary["observations"][observation["observation_id"]]
        if any(row.get(k) != observation[k] for k in ("rgb_sha256", "support_sha256")) or row.get("bbox_xywh") != bbox:
            raise ValueError("Auxiliary context observation or coordinate identity mismatch")
        path = _inside(self.auxiliary_root, row["path"])
        self.fp.add(path, row["sha256"])
        arrays = self.arrays(str(path))
        values, available = arrays["values"], arrays["available"]
        if (values.ndim != 3 or values.shape[0] != channels or available.shape != values.shape
                or available.dtype != bool or not np.isfinite(values[available]).all()):
            raise ValueError("Auxiliary values need C,H,W and separate boolean availability")
        values = np.where(available, values, 0).astype(np.float32)
        if row.get("grid") == "rgb":
            width, height = bbox[2:]
            if values.shape[1:] != (height, width) or height % shape[0] or width % shape[1]:
                raise ValueError("RGB auxiliary grid does not match the canonical crop")
            layout = (channels, shape[0], height // shape[0], shape[1], width // shape[1])
            values = values.reshape(layout).mean((2, 4))
            available = available.reshape(layout).all((2, 4))
        elif row.get("grid") != "patch":
            raise ValueError("Auxiliary grid must explicitly be rgb or patch")
        if values.shape != zeros.shape:
            raise ValueError("Auxiliary patch grid mismatch")
        return np.where(available, values, 0), available

    def load(self, sample):
        width, height = sample["bbox_xywh"][2:]
        if width % 14 or height % 14:
            raise ValueError("Existing DINOv2 cache requires crop dimensions divisible by 14")
        shape = (height // 14, width // 14)
        tokens, masks, extras, available, geometries = [], [], [], [], []
        for observation in sample["observations"]:
            entry = self.history.index["features"][observation["observation_id"]]
            arrays = self.arrays(str(_inside(self.history.index_path.parent, entry["path"])))
            value, valid = arrays.get("patch_tokens"), arrays["patch_support"]
            if (value is None or value.shape != (shape[0] * shape[1], self.dimension)
                    or value.dtype.kind != "f" or not np.isfinite(value).all()
                    or valid.shape != (shape[0] * shape[1],) or valid.dtype != bool):
                raise ValueError("Invalid/missing spatial patch_tokens; re-extract with --save-patch-tokens")
            geom = self.support(str(_inside(self.history.history_root, observation["support"])))
            if geom.shape != (height, width):
                raise ValueError("Support does not match crop coordinates")
            expected = geom.reshape(shape[0], 14, shape[1], 14).all((1, 3))
            if not np.array_equal(expected, valid.reshape(shape)):
                raise ValueError("Patch support differs from recorded RGB geometry")
            tokens.append(value.reshape(*shape, self.dimension).transpose(2, 0, 1).astype(np.float32))
            masks.append(valid.reshape(1, *shape))
            geometries.append(geom)
            extra, observed = self.extra(observation, sample["bbox_xywh"], shape)
            extras.append(extra)
            available.append(observed)
        target = self.target(sample)
        geometry = geometries[-2] & geometries[-1]
        target[:, ~geometry] = 255
        return {"features": np.stack(tokens), "support": np.stack(masks), "years": sample["years"].copy(),
                "extra": np.stack(extras), "available": np.stack(available),
                "target": target, "geometry": geometry}

    def rgb_pair(self, sample):
        result = []
        for observation in sample["observations"][-2:]:
            path = _inside(self.history.history_root, observation["rgb"])
            self.fp.add(path, observation["rgb_sha256"])
            with Image.open(path) as image:
                value = np.array(image)
            if value.dtype != np.uint8 or value.shape != (*self.target(sample).shape[-2:], 3):
                raise ValueError("Baseline RGB must retain native crop shape and uint8 channels")
            result.append(value)
        return result
