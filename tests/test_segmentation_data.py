"""Native-grid overlapping labels, split isolation and training augmentation."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from facade_change.io import sha256, write_json
from facade_change.segmentation.data import PreparedData, augment_crop, cutmix_crop, cutout_crop


class PreparedDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        categories = [{"id": 3, "name": "CRACK"}, {"id": 9, "name": "REPAIRS"}]
        manifest_images, coco_images, crops = [], [], []
        for index, split in enumerate(("train", "val", "test"), 1):
            path = self.root / f"{split}.png"
            rgb = np.full((6, 7, 4), 70 + index, dtype=np.uint8)
            rgb[..., 3] = 255
            rgb[0, 0, 3] = 0
            Image.fromarray(rgb).save(path)
            manifest_images.append({"image_id": index, "file_name": path.name, "image_path": str(path),
                                    "sha256": sha256(path), "height": 6, "width": 7,
                                    "building_id": f"building{index}", "view_id": f"view{index}",
                                    "metadata_status": "reviewed", "split": split})
            coco_images.append({"id": index, "file_name": path.name, "height": 6, "width": 7})
            crops.append({"crop_id": f"{index}:0:0:8", "image_id": index, "image_path": str(path),
                          "image_sha256": sha256(path), "building_id": f"building{index}",
                          "view_id": f"view{index}", "split": split, "x": 0, "y": 0,
                          "width": 7, "height": 6, "pad_right": 1, "pad_bottom": 2})
        annotations = [
            {"id": 1, "image_id": 1, "category_id": 3, "segmentation": [[0, 0, 5, 0, 5, 4, 0, 4]]},
            {"id": 2, "image_id": 1, "category_id": 9, "segmentation": [[2, 1, 6, 1, 6, 5, 2, 5]]},
            {"id": 3, "image_id": 1, "category_id": 3, "iscrowd": 1,
             "segmentation": [[4, 3, 5, 3, 5, 4, 4, 4]]},
            # Test annotation is deliberately invalid: training must not decode it.
            {"id": 4, "image_id": 3, "category_id": 999, "segmentation": None},
        ]
        self.coco = {"images": coco_images, "annotations": annotations, "categories": categories}
        self.manifest = {"images": manifest_images, "categories": categories}
        self.crops = crops
        write_json(self.root / "coco.json", self.coco)
        write_json(self.root / "manifest.json", self.manifest)
        write_json(self.root / "split.json", {})
        (self.root / "crops.jsonl").write_text("".join(json.dumps(row) + "\n" for row in crops))
        self.config = {"kind": "semantic_crop_index", "crop_size": 8, "source_coco": "coco.json",
                       "source_coco_sha256": sha256(self.root / "coco.json"), "manifest": "manifest.json",
                       "crops": "crops.jsonl", "split": "split.json", "categories": categories}
        write_json(self.root / "dataset.json", self.config)

    def test_metadata_does_not_open_rgb_and_rejects_test_access(self):
        (self.root / "test.png").unlink()
        with patch("facade_change.segmentation.data.load_rgb", side_effect=AssertionError("RGB opened")):
            data = PreparedData(self.root)
            self.assertEqual(data.image_ids("train"), [1])
            self.assertEqual(data.image_ids("val"), [2])
            self.assertEqual(data.image_ids("test"), [])
            self.assertNotIn(3, data.annotations_by_image)
            with self.assertRaisesRegex(ValueError, "outside selected"):
                data.load_image(3)

    @unittest.skipUnless(importlib.util.find_spec("pycocotools"), "pycocotools optional dependency")
    def test_overlapping_labels_unknown_alpha_crowd_and_padding(self):
        data = PreparedData(self.root)
        rgb, packed, opaque = data.load_image(1)
        self.assertEqual(packed.dtype, np.uint16)
        self.assertEqual(packed[2, 3], 3)  # Both classes survive the overlap.
        self.assertEqual(packed[3, 4], 0)  # Crowd ignored even inside non-crowd labels.
        self.assertFalse(opaque[0, 0])
        tile, targets, valid = data.crop(1, data.crops_by_image[1][0])
        self.assertEqual(tile.shape, (8, 8, 3))
        self.assertEqual(targets.shape, (2, 8, 8))
        self.assertTrue(targets[:, 2, 3].all())
        self.assertFalse(valid[0, 0])
        self.assertFalse(valid[5, 0])  # Unannotated is unknown, not background.
        self.assertFalse(valid[3, 4])
        self.assertFalse(valid[6:, :].any())
        self.assertFalse(valid[:, 7:].any())
        self.assertFalse(targets[:, ~valid].any())
        self.assertTrue(np.array_equal(tile[-1, -1], rgb[-1, -1]))

    @unittest.skipUnless(importlib.util.find_spec("pycocotools"), "pycocotools optional dependency")
    def test_standard_coco_rle(self):
        from pycocotools import mask as mask_utils
        mask = np.zeros((6, 7), dtype=np.uint8)
        mask[2:4, 3:6] = 1
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("ascii")
        self.coco["annotations"] = [{"id": 11, "image_id": 1, "category_id": 9, "segmentation": rle}]
        write_json(self.root / "coco.json", self.coco)
        self.config["source_coco_sha256"] = sha256(self.root / "coco.json")
        write_json(self.root / "dataset.json", self.config)
        packed = PreparedData(self.root).load_image(1)[1]
        np.testing.assert_array_equal(packed, mask * 2)

    @unittest.skipUnless(importlib.util.find_spec("pycocotools"), "pycocotools optional dependency")
    def test_previous_image_cache_released_before_next_decode(self):
        from facade_change.io import load_rgb
        data = PreparedData(self.root)
        old_rgb, old_labels, _ = data.load_image(1)
        old_rgb_snapshot, old_labels_snapshot = old_rgb.copy(), old_labels.copy()

        def decode(path):
            self.assertIsNone(data._cache)
            return load_rgb(path)

        with patch("facade_change.segmentation.data.load_rgb", side_effect=decode):
            data.load_image(2)
        np.testing.assert_array_equal(old_rgb, old_rgb_snapshot)
        np.testing.assert_array_equal(old_labels, old_labels_snapshot)
        self.assertEqual(data._cache[0], 2)

    @unittest.skipUnless(importlib.util.find_spec("pycocotools"), "pycocotools optional dependency")
    def test_disk_raster_cache_reuses_masks_and_checks_metadata_shape_and_rgb(self):
        directory = self.root / "mask-cache"
        data = PreparedData(self.root, raster_cache_dir=directory)
        expected = data.load_image(1)[1].copy()
        data._cache = None
        with patch("pycocotools.mask.decode", side_effect=AssertionError("Repeated mask decode")):
            np.testing.assert_array_equal(data.load_image(1)[1], expected)
        cache_path = next(directory.glob("*.npz"))
        with np.load(cache_path, allow_pickle=False) as cache:
            fingerprint = cache["fingerprint"].copy()
        for labels, key, message in ((np.zeros((1, 1), np.uint16), fingerprint, "shape/dtype"),
                                     (expected, np.array("wrong"), "provenance")):
            np.savez_compressed(cache_path, packed=labels, fingerprint=key)
            data._cache = None
            with self.assertRaisesRegex(ValueError, message):
                data.load_image(1)
        np.savez_compressed(cache_path, packed=expected, fingerprint=fingerprint)
        (self.root / "train.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Image changed"):
            data.load_image(1)

    def test_leakage_blocks_construction(self):
        self.manifest["images"][1]["building_id"] = "BUILDING1"
        write_json(self.root / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "leakage audit failed"):
            PreparedData(self.root)

    def test_duplicate_crop_and_invalid_bounds_are_rejected(self):
        for rows in ([self.crops[0], self.crops[0], *self.crops[1:]],
                     [{**self.crops[0], "x": 1}, *self.crops[1:]]):
            with self.subTest(rows=rows):
                (self.root / "crops.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.assertRaises(ValueError):
                    PreparedData(self.root)

    def test_hash_validation_before_training(self):
        self.config["source_coco_sha256"] = "wrong"
        write_json(self.root / "dataset.json", self.config)
        with self.assertRaisesRegex(ValueError, "COCO hash"):
            PreparedData(self.root)
        self.config["source_coco_sha256"] = sha256(self.root / "coco.json")
        write_json(self.root / "dataset.json", self.config)
        write_json(self.root / "run.json", {"status": "completed", "artifact_sha256": {"crops.jsonl": "wrong"}})
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            PreparedData(self.root)


class AugmentationTests(unittest.TestCase):
    def fixture(self):
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[8:24, 8:24] = 200
        valid = np.zeros((32, 32), dtype=bool)
        valid[8:24, 8:24] = True
        return rgb, np.stack([valid, valid]), valid

    def test_geometry_preserves_overlap_and_valid_target_alignment(self):
        rgb, targets, valid = self.fixture()
        originals = (rgb.copy(), targets.copy(), valid.copy())
        for seed in range(12):
            output, labels, support = augment_crop(rgb, targets, valid, np.random.default_rng(seed))
            self.assertEqual(output.dtype, np.uint8)
            np.testing.assert_array_equal(labels[0], labels[1])
            np.testing.assert_array_equal(support, labels.any(axis=0))
            self.assertFalse(labels[:, ~support].any())
            self.assertGreater(support.sum(), 0)
            # RGB is bilinear while labels/support use their shared nearest grid.
            self.assertGreater(float(output[support].mean()), float(output[~support].mean()))
        for original, actual in zip(originals, (rgb, targets, valid)):
            np.testing.assert_array_equal(original, actual)

    def test_affine_does_not_erase_one_pixel_crack(self):
        class AffineOnlyRng:
            def __init__(self):
                self.random_values = iter([1., 0., 1.])  # no flip, affine, no shadow
                self.scalar_values = iter([4., 1., 0., 1.])  # angle, scale, EV, contrast

            def random(self):
                return next(self.random_values)

            def uniform(self, low, high, size=None):
                if size == 2:
                    return np.array([.012, -.012])
                if size == 3:
                    return np.ones(3)
                return next(self.scalar_values)

        valid = np.zeros((32, 32), dtype=bool)
        valid[3:29, 16] = True
        rgb = np.repeat(valid[..., None], 3, axis=2).astype(np.uint8) * 200
        _, targets, support = augment_crop(rgb, valid[None], valid, AffineOnlyRng())
        self.assertGreaterEqual(support.sum(), 20)
        np.testing.assert_array_equal(targets[0], support)

    def test_cutmix_copies_all_labels_and_support_with_rgb(self):
        receiver = (np.zeros((32, 32, 3), dtype=np.uint8), np.zeros((2, 32, 32), bool), np.zeros((32, 32), bool))
        donor = (np.full((32, 32, 3), 255, dtype=np.uint8), np.ones((2, 32, 32), bool), np.ones((32, 32), bool))
        donor[2][:4] = False
        result = cutmix_crop(receiver, donor, np.random.default_rng(12))
        copied = result[0][..., 0] == 255
        self.assertTrue(copied.any())
        self.assertFalse(copied.all())
        np.testing.assert_array_equal(result[2], copied & donor[2])
        for channel in result[1]:
            np.testing.assert_array_equal(channel, result[2])
        self.assertFalse(receiver[0].any())
        self.assertFalse(receiver[2].any())

    def test_cutout_removed_region_is_ignored(self):
        rgb = np.full((32, 32, 3), 180, np.uint8)
        targets, valid = np.ones((2, 32, 32), bool), np.ones((32, 32), bool)
        output, labels, support = cutout_crop(rgb, targets, valid, np.random.default_rng(1))
        self.assertLess(support.sum(), valid.sum())
        self.assertGreater(support.sum(), 0)
        self.assertFalse(labels[:, ~support].any())
        self.assertTrue(labels[:, support].all())
        self.assertTrue(valid.all())
        np.testing.assert_array_equal(output, rgb)  # Constant-image mean fill.


if __name__ == "__main__":
    unittest.main()
