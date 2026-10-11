"""Fixed augmentation content, replay, source invalidation and leakage guards."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from facade_change.io import sha256, write_json
from facade_change.segmentation.data import PreparedData
from facade_change.segmentation.prepared import PreparedAugmentations, prepare_augmentations


@unittest.skipUnless(importlib.util.find_spec("pycocotools"), "pycocotools optional dependency")
class FixedAugmentationsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "dataset"
        self.data_root.mkdir()
        self.categories = [{"id": 3, "name": "CRACK"}, {"id": 9, "name": "REPAIRS"}]
        self.images, self.annotations, self.crops = [], [], []
        for image_id, split in enumerate(("train", "train", "val", "test"), 1):
            self.add_image(image_id, split)
        self.save()

    def add_image(self, image_id, split):
        path = self.data_root / f"photo{image_id}.png"
        rgb = np.full((8, 16, 3), 40 + image_id * 20, np.uint8)
        rgb[2:6, :7, 0] += 30
        Image.fromarray(rgb).save(path)
        self.images.append({"image_id": image_id, "file_name": path.name, "image_path": str(path),
                            "sha256": sha256(path), "width": 16, "height": 8,
                            "building_id": f"building{image_id}", "view_id": f"view{image_id}",
                            "metadata_status": "reviewed", "split": split})
        self.annotations.append({"id": image_id, "image_id": image_id, "category_id": 3,
                                 "segmentation": [[1, 1, 7, 1, 7, 7, 1, 7]] if split != "test" else None})
        for x in (0, 8):
            self.crops.append({"crop_id": f"{image_id}:{x}:0:8", "image_id": image_id,
                               "image_sha256": sha256(path), "building_id": f"building{image_id}",
                               "view_id": f"view{image_id}", "split": split, "x": x, "y": 0,
                               "width": 8, "height": 8, "pad_right": 0, "pad_bottom": 0})

    def save(self):
        write_json(self.data_root / "coco.json", {
            "images": [{"id": row["image_id"], "file_name": row["file_name"], "width": 16, "height": 8}
                       for row in self.images], "annotations": self.annotations, "categories": self.categories})
        write_json(self.data_root / "manifest.json", {"images": self.images, "categories": self.categories})
        write_json(self.data_root / "split.json", {})
        (self.data_root / "crops.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.crops))
        write_json(self.data_root / "dataset.json", {
            "kind": "semantic_crop_index", "source_coco": "coco.json", "manifest": "manifest.json",
            "split": "split.json", "crops": "crops.jsonl", "crop_size": 8, "categories": self.categories,
            "source_coco_sha256": sha256(self.data_root / "coco.json")})

    def prepare(self, name="prepared", **kwargs):
        data = PreparedData(self.data_root)
        out = self.root / name
        summary = prepare_augmentations(data, out, **kwargs)
        return PreparedAugmentations(out, data), summary

    def test_all_crops_two_augments_val_originals_and_no_test_pixels(self):
        (self.data_root / "photo4.png").unlink()
        prepared, summary = self.prepare(cutmix_prob=1, cutout_prob=1)
        self.assertEqual(summary["train_originals"], 4)
        self.assertEqual(summary["train_augmented"], 8)
        self.assertEqual(summary["val_originals"], 2)
        self.assertEqual(summary["records"], 14)
        self.assertGreater(summary["empty_samples"], 0)
        self.assertEqual(summary["cutmix_samples"], 8)
        self.assertEqual(summary["cutout_samples"], 8)
        for row in prepared.records:
            self.assertNotEqual(row["split"], "test")
            rgb, targets, valid = prepared.load(row)
            self.assertEqual(rgb.dtype, np.uint8)
            self.assertFalse(targets[:, ~valid].any())
            if row["variant"]:
                self.assertNotEqual(row["image_id"], row["donor"]["image_id"])
                self.assertIn(row["donor"]["image_id"], (1, 2))
        with patch.object(prepared.data, "load_image", side_effect=AssertionError("Raw image opened")):
            crop = prepared.data.crops_by_image[1][0]
            rgb, targets, valid = prepared.original_crop(1, crop)
            self.assertEqual(int(valid.sum()), 36)

    def test_seed_replay_and_extension_preserve_prior_donors_and_pixels(self):
        old, _ = self.prepare(cutmix_prob=1)
        again, _ = self.prepare("repeat", cutmix_prob=1)
        self.assertEqual([(r["sample_id"], r["rgb_sha256"], r["labels_sha256"]) for r in old.records],
                         [(r["sample_id"], r["rgb_sha256"], r["labels_sha256"]) for r in again.records])
        self.add_image(5, "train")
        self.save()
        with self.assertRaisesRegex(ValueError, "different dataset"):
            PreparedAugmentations(old.path, PreparedData(self.data_root))
        extended, summary = self.prepare("extended", reuse=old.path, cutmix_prob=1)
        self.assertEqual(summary["reused_records"], len(old.records))
        for record in old.records:
            current = extended.by_sample[record["crop_key"], record["variant"]]
            self.assertEqual(current["rgb_sha256"], record["rgb_sha256"])
            self.assertEqual(current["labels_sha256"], record["labels_sha256"])
            self.assertEqual(current["donor"], record["donor"])
        self.assertNotEqual((old.path / old.records[0]["rgb"]).stat().st_ino,
                            (extended.path / old.records[0]["rgb"]).stat().st_ino)

    def test_coco_ids_do_not_change_augmentation_content(self):
        old, _ = self.prepare(cutmix_prob=1)
        for image in self.images:
            image["image_id"] += 100
        for annotation in self.annotations:
            annotation["image_id"] += 100
            annotation["id"] += 1000
        for crop in self.crops:
            crop["image_id"] += 100
            crop["crop_id"] = f"{crop['image_id']}:{crop['x']}:0:8"
        self.annotations.reverse()
        self.save()
        current, summary = self.prepare("newids", reuse=old.path, cutmix_prob=1)
        self.assertEqual(summary["reused_records"], len(old.records))
        for row in current.records:
            self.assertGreaterEqual(row["image_id"], 101)
            if row["donor"]:
                self.assertGreaterEqual(row["donor"]["image_id"], 101)
            previous = old.by_sample[row["crop_key"], row["variant"]]
            self.assertEqual(row["rgb_sha256"], previous["rgb_sha256"])

    def test_changed_annotation_invalidates_receiver_and_its_donors(self):
        old, _ = self.prepare(cutmix_prob=1)
        self.annotations[0]["category_id"] = 9
        self.save()
        current, summary = self.prepare("reannotated", reuse=old.path, cutmix_prob=1)
        self.assertEqual(summary["reused_records"], 4)  # Two unchanged TRAIN originals + two VAL originals.
        self.assertNotEqual(current.records[0]["source"]["annotation_sha256"],
                            old.records[0]["source"]["annotation_sha256"])

    def test_reuse_rejects_historical_train_to_holdout_assignment(self):
        old, _ = self.prepare()
        self.images[0]["split"] = "val"
        for row in self.crops:
            if row["image_id"] == 1:
                row["split"] = "val"
        self.save()
        with self.assertRaisesRegex(ValueError, "Historical split ownership conflict"):
            self.prepare("leaky", reuse=old.path)
        self.assertFalse((self.root / "leaky").exists())

    def test_historical_ownership_survives_removed_then_reintroduced_image(self):
        old, _ = self.prepare()
        removed = self.images.pop(0)
        removed_crops = [row for row in self.crops if row["image_id"] == 1]
        self.crops = [row for row in self.crops if row["image_id"] != 1]
        self.save()
        intermediate, _ = self.prepare("removed", reuse=old.path)
        self.images.append({**removed, "split": "test"})
        self.crops.extend({**row, "split": "test"} for row in removed_crops)
        self.save()
        with self.assertRaisesRegex(ValueError, "Historical split ownership conflict"):
            self.prepare("reintroduced", reuse=intermediate.path)

    def test_modified_assets_fail_before_and_after_first_load(self):
        prepared, _ = self.prepare()
        row = prepared.records[0]
        prepared.load(row)
        path = prepared.path / row["rgb"]
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed after verification"):
            prepared.load(row)
        fresh = PreparedAugmentations(prepared.path, prepared.data)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            fresh.load(row)

    def test_incomplete_or_edited_index_rejected_and_no_overwrite(self):
        prepared, _ = self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()
        with (prepared.path / "index.jsonl").open("a") as stream:
            stream.write("{}\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            PreparedAugmentations(prepared.path, prepared.data)

    def test_dataset_cumulative_history_is_bound_and_verified(self):
        history = self.data_root / "semantic_history.json"
        write_json(history, {"kind": "semantic_ownership_history", "images": self.images})
        config_path = self.data_root / "dataset.json"
        config = json.loads(config_path.read_text())
        config["semantic_history"] = str(history)
        write_json(config_path, config)
        write_json(self.data_root / "run.json", {
            "status": "completed", "artifact_sha256": {"semantic_history.json": sha256(history)}})
        prepared, _ = self.prepare()
        self.assertEqual(prepared.config["semantic_history_sha256"], sha256(history))
        self.assertEqual(len(prepared.ownership["historical_images"]), 4)
        history.write_text("{}")
        with self.assertRaisesRegex(ValueError, "history is missing or changed"):
            PreparedAugmentations(prepared.path, PreparedData(self.data_root))


if __name__ == "__main__":
    unittest.main()
