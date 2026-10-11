"""Prepared native-grid COCO crops with overlapping semantic supervision.

Only annotated, opaque, non-crowd pixels are supervised. The absence of a
category inside this support is treated as a negative for that category;
unannotated pixels are unknown, not a healthy/background class.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ..augmentations import _encoded, _linear
from ..io import load_rgb, read_json, sha256
from ..semantic_training import audit_semantic_leakage


def _resolve(value, parent):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class PreparedData:
    """Metadata-only construction; lazy RGB and COCO rasterization for TRAIN/VAL.

    A single full image and a uint16 label bitset are cached. Test RGB is never
    opened by this reader, while all split identities are audited as metadata.
    """

    def __init__(self, dataset_path, splits=("train", "val"), raster_cache_dir=None):
        path = Path(dataset_path).expanduser().resolve()
        self.dataset_path = path / "dataset.json" if path.is_dir() else path
        self.splits = tuple(splits)
        if not self.splits or len(set(self.splits)) != len(self.splits) or not set(self.splits) <= {"train", "val"}:
            raise ValueError("Training reader accepts only distinct train/val splits")
        config = read_json(self.dataset_path)
        if config.get("kind") != "semantic_crop_index":
            raise ValueError("Expected a prepared semantic_crop_index dataset")
        self.config = config
        self.crop_size = _integer(config["crop_size"], "crop_size", 1)
        paths = {name: _resolve(config[name], self.dataset_path.parent)
                 for name in ("source_coco", "manifest", "crops", "split")}
        paths["dataset"] = self.dataset_path
        hashes = {name: sha256(path) for name, path in paths.items()}
        expected = config.get("source_coco_sha256")
        if expected and hashes["source_coco"] != expected:
            raise ValueError("Prepared source COCO hash does not match dataset.json")
        run_path = self.dataset_path.parent / "run.json"
        if run_path.is_file():
            record = read_json(run_path)
            if not str(record.get("status", "")).startswith("completed"):
                raise ValueError("Dataset preparation did not complete successfully")
            artifacts = record.get("artifact_sha256", {})
            for name, path in paths.items():
                try:
                    relative = path.relative_to(self.dataset_path.parent).as_posix()
                except ValueError:
                    continue
                if relative in artifacts and artifacts[relative] != hashes[name]:
                    raise ValueError(f"Prepared artifact hash mismatch: {relative}")
        self.manifest = read_json(paths["manifest"])
        split = read_json(paths["split"])
        leakage = audit_semantic_leakage(self.manifest, split)
        if leakage["status"] != "passed":
            raise ValueError(f"Semantic split leakage audit failed: {leakage['violations']}")
        coco = read_json(paths["source_coco"])
        self.categories = sorted(coco["categories"], key=lambda row: row["id"])
        self.category_ids = [_integer(row["id"], "category_id") for row in self.categories]
        if not 1 <= len(self.categories) <= 16 or len(set(self.category_ids)) != len(self.category_ids):
            raise ValueError("Expected 1–16 unique COCO categories for uint16 label bitsets")
        for other in (config.get("categories"), self.manifest.get("categories")):
            if other is not None and {(row["id"], row["name"]) for row in other} != {
                    (row["id"], row["name"]) for row in self.categories}:
                raise ValueError("Category IDs/names differ between prepared artifacts")
        self.category_to_channel = {category: channel for channel, category in enumerate(self.category_ids)}
        all_images = {row["image_id"]: row for row in self.manifest["images"]}
        self.images = {key: dict(row) for key, row in all_images.items() if row.get("split") in self.splits}
        if not self.images:
            raise ValueError("No selected TRAIN/VAL images")
        coco_images = {}
        for row in coco["images"]:
            if row["id"] in coco_images:
                raise ValueError(f"Duplicate COCO image ID: {row['id']}")
            coco_images[row["id"]] = row
        for key, row in self.images.items():
            for axis in ("width", "height"):
                _integer(row[axis], axis, 1)
            source = coco_images.get(key)
            if source is None or any(source[axis] != row[axis] for axis in ("width", "height")):
                raise ValueError(f"COCO/manifest image dimensions mismatch: {key}")
            row["image_path"] = str(_resolve(row["image_path"], paths["manifest"].parent))
        self.annotations_by_image = defaultdict(list)
        annotation_ids = set()
        for annotation in coco.get("annotations", []):
            if annotation["image_id"] not in self.images:
                continue
            if annotation["id"] in annotation_ids:
                raise ValueError(f"Duplicate selected annotation ID: {annotation['id']}")
            annotation_ids.add(annotation["id"])
            if annotation["category_id"] not in self.category_to_channel:
                raise ValueError(f"Unknown annotation category: {annotation['category_id']}")
            self.annotations_by_image[annotation["image_id"]].append(annotation)
        self.crops_by_image = {key: [] for key in self.images}
        crop_ids = set()
        with paths["crops"].open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["crop_id"] in crop_ids:
                    raise ValueError(f"Duplicate crop ID: {row['crop_id']}")
                crop_ids.add(row["crop_id"])
                image = all_images.get(row["image_id"])
                if image is None or row["split"] != image["split"]:
                    raise ValueError(f"Crop image/split mismatch: {row['crop_id']}")
                for key in ("building_id", "view_id"):
                    if row.get(key) != image.get(key):
                        raise ValueError(f"Crop {key} mismatch: {row['crop_id']}")
                if row.get("image_sha256") != image.get("sha256"):
                    raise ValueError(f"Crop image hash mismatch: {row['crop_id']}")
                x, y = (_integer(row[axis], axis) for axis in ("x", "y"))
                width, height = (_integer(row[axis], axis, 1) for axis in ("width", "height"))
                if (width > self.crop_size or height > self.crop_size or x + width > image["width"]
                        or y + height > image["height"] or row["pad_right"] != self.crop_size - width
                        or row["pad_bottom"] != self.crop_size - height):
                    raise ValueError(f"Invalid crop coordinates/padding: {row['crop_id']}")
                if row["image_id"] in self.images:
                    self.crops_by_image[row["image_id"]].append(row)
        if any(not rows for rows in self.crops_by_image.values()):
            raise ValueError("Selected image has no prepared crops")
        self.metadata = {"paths": {name: str(path) for name, path in paths.items()},
                         "sha256": hashes, "leakage_check": leakage,
                         "selected_splits": list(self.splits), "test_rgb_loaded": False,
                         "test_annotations_rasterized": False,
                         "supervision": "multi-label within annotated opaque non-crowd union; outside unknown",
                         "evaluation_scope": config.get("evaluation_scope", "exploratory"),
                         "category_to_channel": [{"category_id": key, "channel": value}
                                                 for key, value in self.category_to_channel.items()]}
        self.provenance = self.metadata
        self.raster_cache_dir = raster_cache_dir
        self._cache = None
        self._verified_images = {}

    def image_ids(self, split):
        return [key for key, row in self.images.items() if row["split"] == split]

    def load_image(self, image_id):
        if image_id not in self.images:
            raise ValueError(f"Image is outside selected TRAIN/VAL: {image_id}")
        row = self.images[image_id]
        path = Path(row["image_path"])
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        if image_id in self._verified_images:
            if self._verified_images[image_id] != signature:
                raise ValueError(f"Image changed after verification: {path}")
        else:
            if sha256(path) != row["sha256"]:
                raise ValueError(f"Image hash mismatch: {path}")
            self._verified_images[image_id] = signature
        if self._cache is not None and self._cache[0] == image_id:
            return self._cache[1:]
        # Release our references before allocating another potentially very
        # large photograph and its masks. Existing caller arrays remain valid.
        self._cache = None
        rgb, opaque = load_rgb(path)
        h, w = rgb.shape[:2]
        if (h, w) != (row["height"], row["width"]):
            raise ValueError(f"Stored image dimensions differ from manifest: {path}")
        cache_path, fingerprint = None, None
        if self.raster_cache_dir is not None:
            directory = Path(self.raster_cache_dir)
            directory.mkdir(parents=True, exist_ok=True)
            key = {"schema": 1, "image_id": image_id, "image_sha256": row["sha256"],
                   "source_coco_sha256": self.metadata["sha256"]["source_coco"],
                   "category_ids": self.category_ids, "shape": [h, w], "crowd": "ignore_union"}
            fingerprint = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
            cache_path = directory / f"{fingerprint}.npz"
        if cache_path is not None and cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as cached:
                if set(cached.files) != {"packed", "fingerprint"} or cached["fingerprint"].item() != fingerprint:
                    raise ValueError(f"Raster cache provenance mismatch: {cache_path}")
                packed = cached["packed"]
            if packed.dtype != np.uint16 or packed.shape != (h, w):
                raise ValueError(f"Raster cache shape/dtype mismatch: {cache_path}")
            if int(packed.max(initial=0)) >= 1 << len(self.category_ids):
                raise ValueError(f"Raster cache contains unknown category bits: {cache_path}")
        else:
            packed = self._rasterize(image_id, h, w)
            if cache_path is not None:
                temporary_path = None
                try:
                    with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix=".tmp", delete=False) as stream:
                        temporary_path = Path(stream.name)
                        np.savez_compressed(stream, packed=packed, fingerprint=np.array(fingerprint))
                    temporary_path.replace(cache_path)
                finally:
                    if temporary_path is not None:
                        temporary_path.unlink(missing_ok=True)
        self._cache = (image_id, rgb, packed, opaque)
        return rgb, packed, opaque

    def _rasterize(self, image_id, h, w):
        try:
            from pycocotools import mask as mask_utils
        except ImportError as error:
            raise RuntimeError("Semantic rasterization requires pycocotools; install the segmentation extra") from error
        packed = np.zeros((h, w), dtype=np.uint16)
        crowd = np.zeros((h, w), dtype=bool)
        for annotation in self.annotations_by_image.get(image_id, []):
            segmentation = annotation.get("segmentation")
            if isinstance(segmentation, list) and segmentation:
                rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, h, w))
            elif isinstance(segmentation, dict):
                if list(segmentation.get("size", [])) != [h, w]:
                    raise ValueError(f"RLE size mismatch: annotation {annotation['id']}")
                rle = (mask_utils.frPyObjects(segmentation, h, w)
                       if isinstance(segmentation.get("counts"), list) else segmentation)
            else:
                raise ValueError(f"Missing/invalid COCO segmentation: annotation {annotation['id']}")
            mask = mask_utils.decode(rle).astype(bool)
            if mask.shape != (h, w):
                raise ValueError(f"Decoded annotation has invalid shape: {annotation['id']}")
            if annotation.get("iscrowd", 0):
                crowd |= mask
            else:
                packed[mask] |= np.uint16(1 << self.category_to_channel[annotation["category_id"]])
        packed[crowd] = 0
        return packed

    def crop(self, image_id, crop_row):
        if crop_row["image_id"] != image_id:
            raise ValueError("Crop does not belong to requested image")
        rgb, packed, opaque = self.load_image(image_id)
        x, y, w, h = (crop_row[key] for key in ("x", "y", "width", "height"))
        # Edge extension avoids inserting black borders into the model input;
        # labels and supervision remain invalid outside the original image.
        tile = np.pad(rgb[y:y+h, x:x+w], ((0, self.crop_size-h), (0, self.crop_size-w), (0, 0)), mode="edge")
        labels = np.zeros((self.crop_size, self.crop_size), dtype=np.uint16)
        labels[:h, :w] = packed[y:y+h, x:x+w]
        valid = np.zeros(labels.shape, dtype=bool)
        valid[:h, :w] = opaque[y:y+h, x:x+w] & (labels[:h, :w] != 0)
        targets = (labels[None] & (np.uint16(1) << np.arange(len(self.categories), dtype=np.uint16)[:, None, None])) != 0
        targets &= valid[None]
        return tile, targets, valid


def augment_crop(rgb, targets, valid, rng):
    """Joint mild geometry plus outdoor photometry; no blur or label synthesis."""
    rgb, targets, valid = rgb.copy(), targets.copy(), valid.copy()
    h, w = valid.shape
    if rng.random() < .5:
        rgb, targets, valid = rgb[:, ::-1].copy(), targets[:, :, ::-1].copy(), valid[:, ::-1].copy()
    if rng.random() < .5:
        angle, scale = np.deg2rad(rng.uniform(-5, 5)), rng.uniform(.95, 1.05)
        shift = rng.uniform(-.03, .03, size=2) * [w, h]
        cosine, sine = np.cos(angle), np.sin(angle)
        linear = scale * np.array([[cosine, -sine], [sine, cosine]])
        center = np.array([w / 2, h / 2])
        forward = np.eye(3)
        forward[:2, :2], forward[:2, 2] = linear, center + shift - linear @ center
        inverse = tuple(np.linalg.inv(forward)[:2].ravel())
        rgb = np.array(Image.fromarray(rgb).transform((w, h), Image.Transform.AFFINE, inverse,
                                                    Image.Resampling.BILINEAR, fillcolor=(0, 0, 0)))
        labels = np.sum(targets.astype(np.uint16) * (np.uint16(1) << np.arange(len(targets), dtype=np.uint16))[:, None, None], axis=0, dtype=np.uint16)
        labels = np.array(Image.fromarray(labels.astype(np.int32)).transform(
            (w, h), Image.Transform.AFFINE, inverse, Image.Resampling.NEAREST, fillcolor=0), dtype=np.uint16)
        # Annotation support is a discrete label mask, not an RGB interpolation
        # footprint. Nearest sampling preserves thin cracks; new borders ignore.
        support = np.array(Image.fromarray(valid.astype(np.uint8) * 255).transform(
            (w, h), Image.Transform.AFFINE, inverse, Image.Resampling.NEAREST, fillcolor=0))
        valid = support == 255
        targets = (labels[None] & (np.uint16(1) << np.arange(len(targets), dtype=np.uint16)[:, None, None])) != 0
    linear_rgb = _linear(rgb)
    linear_rgb *= 2 ** rng.uniform(-.5, .5)
    linear_rgb = .18 + rng.uniform(.85, 1.15) * (linear_rgb - .18)
    linear_rgb *= rng.uniform(.9, 1.1, size=3)
    if rng.random() < .5:
        yy, xx = np.ogrid[:h, :w]
        angle = rng.uniform(0, 2 * np.pi)
        distance = np.cos(angle) * ((xx + .5) / w - .5) + np.sin(angle) * ((yy + .5) / h - .5)
        alpha = np.clip(.5 + (distance - rng.uniform(-.25, .25)) / rng.uniform(.08, .25), 0, 1)
        linear_rgb *= 1 - rng.uniform(.15, .45) * alpha[..., None]
    rgb = _encoded(linear_rgb)
    valid &= targets.any(axis=0)
    targets &= valid[None]
    return np.ascontiguousarray(rgb), np.ascontiguousarray(targets), np.ascontiguousarray(valid)


def cutmix_crop(receiver, donor, rng, alpha=1.0):
    """Copy one rectangle of RGB, all overlapping labels and valid support."""
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("CutMix alpha must be positive")
    rgb, targets, valid = (array.copy() for array in receiver)
    if any(first.shape != second.shape for first, second in zip(receiver, donor)):
        raise ValueError("CutMix crops must have matching shapes")
    h, w = valid.shape
    ratio = np.sqrt(1 - rng.beta(alpha, alpha))
    rw, rh = max(1, int(w * ratio)), max(1, int(h * ratio))
    cx, cy = int(rng.integers(w)), int(rng.integers(h))
    x0, x1 = max(0, cx-rw//2), min(w, cx+(rw+1)//2)
    y0, y1 = max(0, cy-rh//2), min(h, cy+(rh+1)//2)
    rgb[y0:y1, x0:x1] = donor[0][y0:y1, x0:x1]
    targets[:, y0:y1, x0:x1] = donor[1][:, y0:y1, x0:x1]
    valid[y0:y1, x0:x1] = donor[2][y0:y1, x0:x1]
    targets &= valid[None]
    return rgb, targets, valid


def cutout_crop(rgb, targets, valid, rng, max_fraction=.3):
    """Synthetic occlusion is ignored by supervision, never labelled healthy."""
    if not 0 < max_fraction <= 1:
        raise ValueError("CutOut max_fraction must be in (0,1]")
    rgb, targets, valid = rgb.copy(), targets.copy(), valid.copy()
    h, w = valid.shape
    rh = max(1, int(h * rng.uniform(max_fraction / 2, max_fraction)))
    rw = max(1, int(w * rng.uniform(max_fraction / 2, max_fraction)))
    y, x = int(rng.integers(h-rh+1)), int(rng.integers(w-rw+1))
    fill = np.rint(rgb.reshape(-1, 3).mean(axis=0)).astype(np.uint8)
    rgb[y:y+rh, x:x+rw] = fill
    valid[y:y+rh, x:x+rw] = False
    targets[:, y:y+rh, x:x+rw] = False
    return rgb, targets, valid
