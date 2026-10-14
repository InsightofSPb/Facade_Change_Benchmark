"""Temporal probabilities from an existing, frozen multilabel DINO head."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io import read_json, sha256
from .base import file_provenance, integer_option, mask_scores, validate_rgb_inputs


def _category_mapping(config, expected_categories=None):
    categories = config.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError("Semantic checkpoint has no explicit categories")
    ids, names = [], []
    for category in categories:
        category_id, name = category.get("id"), category.get("name")
        if type(category_id) is not int or category_id < 0 or not isinstance(name, str) or not name.strip():
            raise ValueError("Semantic checkpoint category IDs/names are invalid")
        ids.append(category_id)
        names.append(name)
    if len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise ValueError("Semantic checkpoint category IDs/names must be unique")
    mapping = [{"category_id": category_id, "channel": channel}
               for channel, category_id in enumerate(ids)]
    if config.get("category_to_channel") != mapping:
        raise ValueError("Semantic checkpoint category/channel mapping is inconsistent")
    provenance_mapping = config.get("dataset_provenance", {}).get("category_to_channel")
    if provenance_mapping is not None and provenance_mapping != mapping:
        raise ValueError("Semantic checkpoint dataset/channel mapping is inconsistent")
    if expected_categories is not None:
        expected = sorted((row["id"], row["name"]) for row in expected_categories)
        if expected != sorted(zip(ids, names)):
            raise ValueError("Current category IDs/names differ from semantic checkpoint ontology")
    return categories, mapping


def _training_origin(config, current_manifest=None):
    """Compare only checkpoint-bound identity metadata, never evaluation labels."""
    provenance = config.get("dataset_provenance", {})
    path = provenance.get("paths", {}).get("manifest")
    expected = provenance.get("sha256", {}).get("manifest")
    warnings, training = [], []
    manifest_verified = False
    if path and expected and Path(path).is_file():
        if sha256(path) != expected:
            raise ValueError("Semantic checkpoint training manifest hash mismatch")
        original = read_json(path)
        selected = config.get("training_image_ids")
        if not isinstance(selected, list) or not selected:
            warnings.append("Checkpoint training image IDs are missing; training origins unverified")
        else:
            by_id = {str(row["image_id"]): row for row in original.get("images", [])}
            if len(by_id) != len(original.get("images", [])):
                raise ValueError("Semantic training manifest contains duplicate image IDs")
            if any(str(image_id) not in by_id for image_id in selected):
                raise ValueError("Semantic checkpoint training image IDs absent from training manifest")
            training = [by_id[str(image_id)] for image_id in selected]
            if any(row.get("split") != "train" for row in training):
                raise ValueError("Semantic checkpoint trained on a non-TRAIN manifest row")
            manifest_verified = True
    else:
        warnings.append("Checkpoint training manifest is unavailable or lacks a bound hash; origins unverified")
    unknown = [row.get("image_id") for row in training if not row.get("building_id")
               or row.get("metadata_status") != "reviewed" or not row.get("sha256")]
    if unknown:
        warnings.append("Some checkpoint TRAIN identities are unverified; physical holdout independence is unproven")
    overlaps = {split: [] for split in ("train", "val", "test")}
    current = read_json(current_manifest) if current_manifest else None
    if current is not None and (not isinstance(current, dict)
                                or not isinstance(current.get("images"), list)
                                or not current["images"]
                                or any(not isinstance(row, dict) for row in current["images"])):
        raise ValueError("current_manifest must contain a nonempty images list; pass the manifest, not dataset.json")
    if current is None:
        warnings.append("Current manifest not supplied; checkpoint TRAIN overlap with current splits is unverified")
    elif manifest_verified:
        normalize = lambda value: str(value).strip().casefold() if value is not None else ""
        keys = ("building_id", "view_id", "sha256")
        known = {key: {normalize(row.get(key)) for row in training if normalize(row.get(key))}
                 for key in keys}
        for row in current.get("images", []):
            split = row.get("split")
            matched = [key for key in keys if normalize(row.get(key)) in known[key]]
            if split in overlaps and matched:
                overlaps[split].append({"image_id": row.get("image_id"),
                                        "building_id": row.get("building_id"), "matched_on": matched})
        if overlaps["val"] or overlaps["test"]:
            warnings.append("Semantic checkpoint TRAIN overlaps current VAL/TEST; affected results are exploratory")
    independent = bool(manifest_verified and current is not None and not unknown
                       and not overlaps["val"] and not overlaps["test"])
    return {"training_manifest_verified": manifest_verified,
            "training_manifest": file_provenance(path) if manifest_verified else None,
            "current_manifest": file_provenance(current_manifest) if current_manifest else None,
            "training_images": len(training), "unverified_training_image_ids": unknown,
            "train_overlap_by_current_split": overlaps,
            "known_checkpoint_train_holdout_overlap": bool(overlaps["val"] or overlaps["test"]),
            "checkpoint_train_holdout_identity_disjointness_verified": independent,
            "scope": "checkpoint training identities only; VAL checkpoint selection is not independent evaluation",
            "warnings": warnings}


class SemanticDINOScorer:
    """Frozen semantic delta; uncertainty is exported separately and never gates it."""

    def __init__(self, training_run=None, device="cpu", batch_size=4,
                 export_probabilities=False, current_manifest=None, expected_categories=None,
                 crop_size=None, crop_stride=None):
        if not training_run:
            raise ValueError("semantic_dino requires training_run with completed best.pth provenance")
        from ..segmentation.evaluate import _checked_run
        from ..segmentation.model import load_model
        from ..segmentation.train import gpu_preflight

        self.device = str(device)
        self.batch_size = integer_option(batch_size, "batch_size", 1)
        if type(export_probabilities) is not bool:
            raise ValueError("export_probabilities must be boolean")
        self.export_probabilities = export_probabilities
        root, record, config, saved = _checked_run(training_run)
        if current_manifest:
            current_categories = read_json(current_manifest).get("categories")
            if expected_categories is None and current_categories is not None:
                expected_categories = current_categories
        categories, mapping = _category_mapping(config, expected_categories)
        if saved["backbone_provenance"].get("num_classes") != len(categories):
            raise ValueError("Semantic checkpoint head channel count differs from its ontology")
        origin = _training_origin(config, current_manifest)
        dataset_provenance = config.get("dataset_provenance", {})
        dataset_path = dataset_provenance.get("paths", {}).get("dataset") or config.get("dataset")
        geometry = {}
        if dataset_path and Path(dataset_path).is_file():
            expected_hash = dataset_provenance.get("sha256", {}).get("dataset")
            if not expected_hash or sha256(dataset_path) != expected_hash:
                raise ValueError("Semantic checkpoint dataset geometry lacks a matching bound hash")
            geometry = read_json(dataset_path)
        self.crop_size = integer_option(crop_size if crop_size is not None else geometry.get("crop_size", 512),
                                        "crop_size", 1)
        default_stride = geometry.get("crop_stride", geometry.get("stride", min(448, self.crop_size)))
        self.crop_stride = integer_option(crop_stride if crop_stride is not None else default_stride,
                                          "crop_stride", 1, self.crop_size)
        if not geometry:
            origin["warnings"].append("Training crop geometry unavailable; recorded adapter crop geometry is used")
        gpu = gpu_preflight(self.device)
        model, provenance = load_model(config["source_root"], config["checkpoint"], len(categories),
                                       self.device, config.get("trust_checkpoint", False))
        for name, key in (("sources", "python_tree_sha256"), ("checkpoint", "sha256")):
            if provenance[name][key] != saved["backbone_provenance"][name][key]:
                raise ValueError("Loaded semantic backbone/source fingerprint differs from training")
        model.head.load_state_dict(saved["head_state_dict"], strict=True)
        self.model = model.eval().requires_grad_(False)
        self.raw_scores = self.native_prediction = None
        self.auxiliary_arrays = {}
        self.metadata = {
            "method": "semantic_dino", "implementation_version": "2026-10-14-v1", "output_kind": "score",
            "training_run": str(root), "training_status": record["status"],
            "head_checkpoint": file_provenance(root / "best.pth"), "checkpoint_epoch": saved["epoch"],
            "checkpoint_selection": "previously saved best VAL head; no retraining or reselection",
            "categories": categories, "category_to_channel": mapping, "num_classes": len(categories),
            "probability": "independent sigmoid for every class; no softmax",
            "score": "max_c abs(p_source[c] - p_reference[c]); no pair normalization",
            "semantic_delta": "p_source - p_reference; positive appearance, negative disappearance",
            "uncertainty": "mean independent Bernoulli entropy in bits; diagnostic only, never score gating",
            "uncertainty_gating": False, "weights_frozen": True, "downloads": False,
            "crop_size": self.crop_size, "crop_stride": self.crop_stride,
            "spatial_protocol": "native RGB grid; edge-padded semantic crops; fixed midpoint ownership",
            "support_policy": "all output arrays NaN outside geometric support; no label masks enter inference",
            "export_probabilities": export_probabilities, "device": self.device,
            "backbone_provenance": provenance, "checkpoint_origin_audit": origin, "gpu_preflight": gpu,
        }

    def _probabilities(self, rgb):
        import torch
        from ..preparation import semantic_crop_boxes
        from ..segmentation.train import ownership_masks

        height, width = rgb.shape[:2]
        rows = [{"crop_id": str(index), "x": x, "y": y, "width": w, "height": h}
                for index, (x, y, w, h) in enumerate(semantic_crop_boxes(
                    width, height, self.crop_size, self.crop_stride))]
        ownership = ownership_masks(rows, width, height, self.crop_size)
        probabilities = np.empty((self.metadata["num_classes"], height, width), np.float32)
        with torch.inference_mode():
            for start in range(0, len(rows), self.batch_size):
                batch = rows[start:start + self.batch_size]
                tiles = []
                for row in batch:
                    x, y, w, h = (row[key] for key in ("x", "y", "width", "height"))
                    tiles.append(np.pad(rgb[y:y+h, x:x+w],
                                        ((0, self.crop_size-h), (0, self.crop_size-w), (0, 0)), mode="edge"))
                inputs = torch.from_numpy(np.stack(tiles)).permute(0, 3, 1, 2).to(
                    device=self.device, dtype=torch.float32) / 255.
                logits = self.model(inputs)
                expected = (len(batch), self.metadata["num_classes"], self.crop_size, self.crop_size)
                if tuple(logits.shape) != expected or not torch.isfinite(logits).all():
                    raise RuntimeError("Semantic head returned invalid native-crop logits")
                outputs = logits.float().sigmoid().cpu().numpy()
                for row, output in zip(batch, outputs):
                    x, y, w, h = (row[key] for key in ("x", "y", "width", "height"))
                    owner = ownership[row["crop_id"]][:h, :w]
                    probabilities[:, y:y+h, x:x+w][:, owner] = output[:, :h, :w][:, owner]
        return probabilities

    def __call__(self, reference, source, support):
        from ..segmentation.evaluate import bernoulli_entropy

        reference, source, support = validate_rgb_inputs(reference, source, support)
        before, after = self._probabilities(reference), self._probabilities(source)
        delta = after - before
        scores = mask_scores(np.abs(delta).max(axis=0), support)
        arrays = {"semantic_delta": delta,
                  "uncertainty_reference": bernoulli_entropy(before),
                  "uncertainty_source": bernoulli_entropy(after)}
        if self.export_probabilities:
            arrays.update(probabilities_reference=before, probabilities_source=after)
        for value in arrays.values():
            value[..., ~support] = np.nan
        self.auxiliary_arrays = arrays
        return scores

    def close(self):
        if getattr(self, "model", None) is not None:
            self.model.cpu()
            self.model = None
