"""Fixed lossless TRAIN augmentations, bound to audited dataset snapshots.

Preparation is CPU-only. Original crops are materialized first, so CutMix never
requires decoding a second full photograph and can only use current TRAIN data.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version
from tqdm import tqdm

from ..io import finish_record, new_directory, read_json, run_record, sha256, write_json
from ..semantic_training import audit_semantic_leakage
from .data import PreparedData, augment_crop, cutmix_crop, cutout_crop

_COORDS = ("x", "y", "width", "height", "pad_right", "pad_bottom")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _source(data, image_id):
    image = data.images[image_id]
    # Export-generated COCO IDs and annotation ordering do not affect content.
    annotations = [{key: row.get(key, 0 if key == "iscrowd" else None)
                    for key in ("category_id", "segmentation", "iscrowd")}
                   for row in data.annotations_by_image.get(image_id, [])]
    annotations.sort(key=lambda row: json.dumps(row, sort_keys=True))
    return {"image_sha256": image["sha256"], "annotation_sha256": _digest(annotations),
            "width": image["width"], "height": image["height"],
            "file_name": image.get("resolved_file_name") or Path(image["image_path"]).name,
            "category_sha256": _digest([(row["id"], row["name"]) for row in data.categories])}


def _crop_key(source, crop):
    return _digest({"source": source, "crop": {key: crop[key] for key in _COORDS}})


def _policy(seed, cutmix_prob, cutout_prob):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Augmentation seed must be a nonnegative integer")
    for name, value in (("cutmix_prob", cutmix_prob), ("cutout_prob", cutout_prob)):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0,1]")
    return {"schema": 1, "seed": seed, "augmented_versions_per_train_crop": 2,
            "cutmix_prob": float(cutmix_prob), "cutout_prob": float(cutout_prob),
            "cutmix_alpha": 1.0, "cutout_max_fraction": .3,
            "transforms": "outdoor_geometry_photometry_then_cutmix_then_cutout",
            "data_code_sha256": sha256(Path(__file__).with_name("data.py")),
            "prepared_code_sha256": sha256(Path(__file__)),
            "numpy": np.__version__, "Pillow": pillow_version}


def _safe_path(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Prepared asset path leaves augmentation directory")
    return path


class _Artifacts:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.run = read_json(self.path / "run.json")
        if (self.run.get("kind") != "fixed_semantic_augmentations"
                or self.run.get("status") != "completed"):
            raise ValueError("Augmentation preparation is not completed")
        self.hashes = self.run.get("artifact_sha256", {})
        self._verified = {}
        for name in ("config.json", "index.jsonl", "ownership.json", "summary.json"):
            self.verify(name, self.hashes.get(name))
        self.config = read_json(self.path / "config.json")
        self.ownership = read_json(self.path / "ownership.json")
        with (self.path / "index.jsonl").open(encoding="utf-8") as stream:
            self.records = [json.loads(line) for line in stream]
        self.by_sample = {}
        for row in self.records:
            key = (row["crop_key"], row["variant"])
            if key in self.by_sample:
                raise ValueError("Duplicate prepared crop variant")
            self.by_sample[key] = row
            for field in ("rgb", "labels"):
                if self.hashes.get(row[field]) != row[field + "_sha256"]:
                    raise ValueError("Prepared asset/index hash mismatch")

    def verify(self, relative, expected):
        if not expected:
            raise ValueError(f"Prepared artifact has no checksum: {relative}")
        path = _safe_path(self.path, relative)
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        if relative in self._verified:
            if signature != self._verified[relative]:
                raise ValueError(f"Prepared asset changed after verification: {relative}")
        else:
            if sha256(path) != expected:
                raise ValueError(f"Prepared asset hash mismatch: {relative}")
            self._verified[relative] = signature
        return path

    def load(self, record):
        rgb_path = self.verify(record["rgb"], record["rgb_sha256"])
        labels_path = self.verify(record["labels"], record["labels_sha256"])
        with Image.open(rgb_path) as image:
            if image.mode != "RGB":
                raise ValueError("Prepared RGB must be lossless uint8 RGB")
            rgb = np.array(image)
        with np.load(labels_path, allow_pickle=False) as labels:
            if set(labels.files) != {"targets", "valid"}:
                raise ValueError("Unexpected prepared label arrays")
            targets, valid = labels["targets"], labels["valid"]
        shape = (self.config["crop_size"],) * 2
        if (rgb.dtype != np.uint8 or rgb.shape != (*shape, 3)
                or targets.dtype != bool or targets.shape != (len(self.config["categories"]), *shape)
                or valid.dtype != bool or valid.shape != shape
                or int(valid.sum()) != record["valid_pixels"] or targets[:, ~valid].any()
                or not np.array_equal(valid, targets.any(axis=0))):
            raise ValueError("Prepared crop shape, dtype or supervision mismatch")
        return rgb, targets, valid


def _record(data, image_id, crop, source, variant, policy):
    key = _crop_key(source, crop)
    sample_id = _digest({"crop_key": key, "variant": variant,
                         "policy": _digest(policy) if variant else "original-v1"})
    return {"image_id": image_id, "crop_id": crop["crop_id"], "crop_key": key,
            "sample_id": sample_id, "split": data.images[image_id]["split"], "variant": variant,
            "source": source, "crop": {name: crop[name] for name in _COORDS},
            "category_mapping": data.metadata["category_to_channel"],
            "transform_seed": int(sample_id[:16], 16) if variant else None,
            "donor": None, "cutmix_applied": False, "cutout_applied": False,
            "rgb": f"samples/{sample_id}.png", "labels": f"samples/{sample_id}.npz"}


def _write_sample(out, record, arrays):
    rgb, targets, valid = arrays
    Image.fromarray(rgb).save(out / record["rgb"], format="PNG")
    np.savez_compressed(out / record["labels"], targets=targets, valid=valid)
    record["valid_pixels"] = int(valid.sum())
    for name in ("rgb", "labels"):
        record[name + "_sha256"] = sha256(out / record[name])


def _copy_sample(out, record, previous, old):
    for name in ("rgb", "labels"):
        path = previous.verify(old[name], old[name + "_sha256"])
        shutil.copyfile(path, out / record[name])
        record[name + "_sha256"] = old[name + "_sha256"]
    for name in ("valid_pixels", "donor", "cutmix_applied", "cutout_applied"):
        record[name] = old[name]


def _dataset_history(data):
    value = data.config.get("semantic_history")
    if not value:
        return [], None
    path = Path(value).expanduser()
    path = path.resolve() if path.is_absolute() else (data.dataset_path.parent / path).resolve()
    relative = path.relative_to(data.dataset_path.parent).as_posix()
    run = read_json(data.dataset_path.parent / "run.json")
    expected = run.get("artifact_sha256", {}).get(relative)
    if not expected or sha256(path) != expected:
        raise ValueError("Semantic ownership history is missing or changed")
    history = read_json(path)
    if history.get("kind") != "semantic_ownership_history":
        raise ValueError("Invalid semantic ownership history")
    audit = audit_semantic_leakage(data.manifest, {"historical_cohort": history["images"]})
    if audit["status"] != "passed":
        raise ValueError(f"Dataset historical split ownership conflict: {audit['violations']}")
    return history["images"], expected


def _check_history(data, previous):
    history = (previous.ownership["images"] + previous.ownership.get("historical_images", [])
               + _dataset_history(data)[0])
    audit = audit_semantic_leakage(data.manifest, {"historical_cohort": history})
    if audit["status"] != "passed":
        raise ValueError(f"Historical split ownership conflict; cannot reuse augmentations: {audit['violations']}")


def prepare_augmentations(dataset, out, seed=42, cutmix_prob=.25, cutout_prob=.125, reuse=None):
    """Store every TRAIN crop as original+two variants, VAL as original only."""
    data = dataset if isinstance(dataset, PreparedData) else PreparedData(dataset)
    if set(data.splits) != {"train", "val"}:
        raise ValueError("Fixed preparation requires the complete TRAIN and VAL dataset")
    policy = _policy(seed, cutmix_prob, cutout_prob)
    dataset_history, history_hash = _dataset_history(data)
    previous = _Artifacts(reuse) if reuse is not None else None
    if previous is not None:
        _check_history(data, previous)
    config = {"dataset_sha256": data.metadata["sha256"], "dataset_path": str(data.dataset_path),
              "crop_size": data.crop_size, "categories": data.categories, "policy": policy,
              "semantic_history_sha256": history_hash,
              "dataset_provenance": data.metadata,
              "reuse": str(previous.path) if previous else None}
    out = new_directory(out)
    (out / "samples").mkdir()
    run = run_record("fixed_semantic_augmentations", config)
    write_json(out / "run.json", run)
    write_json(out / "config.json", config)
    # Metadata only: no TEST pixels or annotations are materialized.
    ownership = {"images": [{key: row.get(key) for key in
                             ("image_id", "file_name", "sha256", "building_id", "view_id",
                              "metadata_status", "split")} for row in data.manifest["images"]]}
    historical = dataset_history[:]
    if previous is not None:
        historical += previous.ownership["images"] + previous.ownership.get("historical_images", [])
    if historical:
        ownership["historical_images"] = list({_digest(row): row for row in historical}.values())
    write_json(out / "ownership.json", ownership)
    records, originals, reused = [], {}, 0
    try:
        sources = {image_id: _source(data, image_id) for image_id in data.images}
        total = sum(len(rows) for rows in data.crops_by_image.values())
        with tqdm(total=total, desc="Prepare original TRAIN/VAL crops", unit="crop") as progress:
            for image_id in data.images:
                for crop in data.crops_by_image[image_id]:
                    record = _record(data, image_id, crop, sources[image_id], 0, policy)
                    if record["crop_key"] in originals:
                        raise ValueError("Duplicate source photograph/crop identity in prepared dataset")
                    old = previous.by_sample.get((record["crop_key"], 0)) if previous else None
                    if old is not None:
                        _copy_sample(out, record, previous, old)
                        reused += 1
                    else:
                        _write_sample(out, record, data.crop(image_id, crop))
                    originals[record["crop_key"]] = record
                    records.append(record)
                    progress.update()
        data._cache = None  # Release the final full-size photograph before augmentation.
        # Read small lossless originals directly, never a full CutMix photograph.
        scratch = _Artifacts.__new__(_Artifacts)
        scratch.path, scratch.config, scratch._verified = out, config, {}
        train = sorted((row for row in records if row["split"] == "train"), key=lambda row: row["crop_key"])
        policy_matches = previous is not None and previous.config["policy"] == policy
        for original in tqdm(train, desc="Prepare two fixed TRAIN augmentations", unit="crop"):
            receiver = scratch.load(original)
            candidates = [row for row in train if row["source"]["image_sha256"] != original["source"]["image_sha256"]]
            for variant in (1, 2):
                crop = {**original["crop"], "crop_id": original["crop_id"]}
                record = _record(data, original["image_id"], crop, original["source"], variant, policy)
                old = previous.by_sample.get((record["crop_key"], variant)) if policy_matches else None
                old_donor = originals.get(old["donor"]["crop_key"]) if old and old.get("donor") else None
                donor_valid = (not old or not old.get("donor") or
                               (old_donor is not None and old_donor["split"] == "train"
                                and old_donor["source"] == old["donor"]["source"]
                                and old_donor["source"]["image_sha256"] != original["source"]["image_sha256"]))
                if old is not None and donor_valid:
                    _copy_sample(out, record, previous, old)
                    if record["donor"]:
                        record["donor"] = {key: old_donor[key] for key in ("crop_key", "source", "image_id", "crop_id")}
                    reused += 1
                else:
                    rng = np.random.default_rng(record["transform_seed"])
                    arrays = augment_crop(*receiver, rng)
                    if rng.random() < cutmix_prob and candidates:
                        donor = candidates[int(rng.integers(len(candidates)))]
                        arrays = cutmix_crop(arrays, augment_crop(*scratch.load(donor), rng), rng)
                        record["donor"] = {key: donor[key] for key in ("crop_key", "source", "image_id", "crop_id")}
                        record["cutmix_applied"] = True
                    if rng.random() < cutout_prob:
                        arrays = cutout_crop(*arrays, rng)
                        record["cutout_applied"] = True
                    _write_sample(out, record, arrays)
                records.append(record)
        with (out / "index.jsonl").open("w", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        counts = Counter((row["split"], row["variant"] > 0) for row in records)
        summary = {"out": str(out), "status": "completed", "train_originals": counts["train", False],
                   "train_augmented": counts["train", True], "val_originals": counts["val", False],
                   "records": len(records), "reused_records": reused,
                   "empty_samples": sum(row["valid_pixels"] == 0 for row in records),
                   "cutmix_samples": sum(row["cutmix_applied"] for row in records),
                   "cutout_samples": sum(row["cutout_applied"] for row in records),
                   "test_rgb_loaded": False, "test_annotations_rasterized": False}
        write_json(out / "summary.json", summary)
        finish_record(out, run, "completed")
        return summary
    except BaseException as error:
        finish_record(out, run, "failed", error=str(error))
        raise


class PreparedAugmentations(_Artifacts):
    """Strict snapshot reader: immutable cached RGB, targets and supervision."""
    def __init__(self, path, data):
        super().__init__(path)
        if (self.config["dataset_sha256"] != data.metadata["sha256"]
                or self.config["categories"] != data.categories or self.config["crop_size"] != data.crop_size
                or self.config.get("semantic_history_sha256") != _dataset_history(data)[1]):
            raise ValueError("Prepared augmentations belong to a different dataset; prepare a new snapshot")
        _check_history(data, self)
        expected, self._originals = set(), {}
        for image_id in data.images:
            source = _source(data, image_id)
            for crop in data.crops_by_image[image_id]:
                variants = (0, 1, 2) if data.images[image_id]["split"] == "train" else (0,)
                for variant in variants:
                    reference = _record(data, image_id, crop, source, variant, self.config["policy"])
                    key = (reference["crop_key"], variant)
                    expected.add(key)
                    row = self.by_sample.get(key)
                    if row is None or any(row[name] != reference[name] for name in
                            ("image_id", "crop_id", "split", "sample_id", "source", "crop", "category_mapping", "transform_seed")):
                        raise ValueError("Prepared crop identity/completeness mismatch")
                    if variant == 0:
                        self._originals[image_id, crop["crop_id"]] = row
        if expected != set(self.by_sample):
            raise ValueError("Unexpected crops in prepared augmentation index")
        for row in self.records:
            if row.get("donor"):
                donor = self.by_sample.get((row["donor"]["crop_key"], 0))
                if (not row["variant"] or row["split"] != "train" or donor is None
                        or donor["split"] != "train" or donor["source"] != row["donor"]["source"]
                        or donor["source"]["image_sha256"] == row["source"]["image_sha256"]):
                    raise ValueError("Prepared CutMix donor is outside current TRAIN")
        self.data = data
        self.runpath = self.path
        self.metadata = self.provenance = {"path": str(self.path), "run_sha256": sha256(self.path / "run.json"),
                                          "config": self.config, "records": len(self.records)}

    def original_crop(self, image_id, crop_row):
        return self.load(self._originals[image_id, crop_row["crop_id"]])
