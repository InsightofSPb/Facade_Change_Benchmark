import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from facade_change.derived import build_crops, controlled_examples
from facade_change.geometry import expanded_canvas
from facade_change.io import finish_record, load_rgb, read_json, run_record, sha256, write_json


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required for native crop sampling")
class DerivedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pair = self.root / "pair"
        self.pair.mkdir()
        rng = np.random.default_rng(5)
        self.rgb = rng.integers(0, 256, (24, 28, 3), dtype=np.uint8)
        observations = {}
        for index, role in enumerate(("reference", "source")):
            image = self.root / f"{role}.png"
            Image.fromarray(self.rgb).save(image)
            observations[role] = {"image_id": index, "image_path": str(image), "height": 24, "width": 28,
                                  "sha256": sha256(image), "view_id": "facade-view", "building_id": "facade",
                                  "metadata_status": "reviewed", "year": 2010 + index, "split": "test"}
            # Invalid scientific pixel files: crop export must ignore these.
            (self.pair / f"{role}_rgb.png").write_text("do not read a resampled pair image")
        self.matrix = np.array([[1., .015, -3.25], [0, 1, 1.5], [0, 0, 1]])
        self.canvas = expanded_canvas(self.rgb.shape, self.rgb.shape, self.matrix)
        write_json(self.pair / "geometry.json", self.canvas.as_dict())
        record = run_record("pair_alignment", {})
        record["observations"] = observations
        finish_record(self.pair, record, "completed_needs_review")

    def tearDown(self):
        self.temp.cleanup()

    def test_crops_sample_originals_once_and_compose_before_warp(self):
        import cv2

        out = self.root / "crops"
        with patch("facade_change.derived.load_rgb", wraps=load_rgb) as reader:
            result = build_crops(self.pair, out, tile_size=12, stride=6, split="test", group_id="facade")
        self.assertEqual(reader.call_count, 2)
        self.assertGreater(result["crop_count"], 1)
        crops = read_json(out / "crops.json")["crops"]
        for crop in crops:
            x, y, x1, y1 = crop["bbox_canvas_xyxy"]
            self.assertEqual([x1 - x, y1 - y], [12, 12])
            offset = np.array([[1., 0, -x], [0, 1, -y], [0, 0, 1]])
            direct = offset @ self.canvas.reference_to_canvas @ self.matrix
            np.testing.assert_allclose(crop["source_to_crop"], direct)
            expected = cv2.warpPerspective(self.rgb, direct, (12, 12), flags=cv2.INTER_LINEAR)
            actual, _ = load_rgb(out / crop["path"] / "source_rgb.png")
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(crop["split"], "test")
            self.assertEqual(crop["group_id"], "facade")
            self.assertGreaterEqual(crop["valid_fraction"], .8)
        record = read_json(out / "run.json")
        for filename, digest in record["artifact_sha256"].items():
            self.assertEqual(sha256(out / filename), digest)

    def test_unreviewed_scientific_split_and_changed_geometry_fail(self):
        record = read_json(self.pair / "run.json")
        record["observations"]["source"]["metadata_status"] = "inferred"
        write_json(self.pair / "run.json", record)
        with self.assertRaisesRegex(ValueError, "reviewed"):
            build_crops(self.pair, self.root / "invalid", split="train")
        geometry = self.pair / "geometry.json"
        geometry.write_text(geometry.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "geometry changed"):
            build_crops(self.pair, self.root / "changed", split="dev")

    def test_rejected_alignment_and_split_relabel_are_forbidden(self):
        with self.assertRaisesRegex(ValueError, "stored observation splits"):
            build_crops(self.pair, self.root / "wrong-split", split="train")
        record = read_json(self.pair / "run.json")
        del record["observations"]["source"]["split"]
        write_json(self.pair / "run.json", record)
        with self.assertRaisesRegex(ValueError, "stored observation splits"):
            build_crops(self.pair, self.root / "missing-split", split="test")
        record["status"] = "completed_rejected"
        write_json(self.pair / "run.json", record)
        with self.assertRaisesRegex(ValueError, "Rejected alignment"):
            build_crops(self.pair, self.root / "rejected", split="dev")
        record["status"] = "completed_needs_review"
        record["quality_gate"] = {"passed": False}
        write_json(self.pair / "run.json", record)
        with self.assertRaisesRegex(ValueError, "Rejected alignment"):
            build_crops(self.pair, self.root / "gate-rejected", split="dev")

    def test_alpha_and_black_pixels_keep_distinct_support(self):
        import cv2

        rgba = np.concatenate([self.rgb.copy(), np.full((24, 28, 1), 255, np.uint8)], axis=2)
        rgba[6:10, 10:14, :3] = 0
        rgba[12:15, 10:14, 3] = 0
        source = self.root / "source.png"
        Image.fromarray(rgba).save(source)
        record = read_json(self.pair / "run.json")
        record["observations"]["source"]["sha256"] = sha256(source)
        write_json(self.pair / "run.json", record)
        out = self.root / "alpha-crops"
        build_crops(self.pair, out, tile_size=12, stride=6, min_valid_fraction=.5, split="test")
        for crop in read_json(out / "crops.json")["crops"]:
            expected = cv2.warpPerspective((rgba[..., 3] == 255).astype(np.float32),
                                           np.asarray(crop["source_to_crop"]), (12, 12), flags=cv2.INTER_LINEAR) >= 1 - 1e-6
            with Image.open(out / crop["path"] / "source_support.png") as image:
                actual = np.array(image) == 255
            np.testing.assert_array_equal(actual, expected)

    def test_controls_preserve_group_factorial_labels_and_matched_sham(self):
        crops = self.root / "crops"
        build_crops(self.pair, crops, tile_size=12, stride=6, split="test", group_id="facade")
        controls = self.root / "controls"
        summary = controlled_examples(crops, controls, seed=17, max_crops=1)
        self.assertEqual(summary["example_count"], 6)
        examples = read_json(controls / "examples.json")["examples"]
        self.assertEqual({(row["state_change"], row["nuisance"]) for row in examples if not row["sham_self_paste"]},
                         {(False, False), (False, True), (True, False), (True, True)})
        arrays = {}
        for example in examples:
            self.assertEqual((example["group_id"], example["split"]), ("facade", "test"))
            target = controls / example["path"]
            with Image.open(target / "labels_reference.png") as image:
                labels = np.array(image)
            with Image.open(target / "comparable_reference.png") as image:
                comparable = np.array(image) == 255
            self.assertTrue(np.all(labels[~comparable] == 255))
            self.assertEqual(bool(np.any(labels == 1)), example["state_change"])
            matrix = np.asarray(example["source_to_reference"])
            np.testing.assert_allclose(matrix @ np.asarray(example["reference_to_source"]), np.eye(3))
            arrays[(example["state_change"], example["nuisance"], example["sham_self_paste"])] = load_rgb(target / "source_rgb.png")[0]
        for nuisance in (False, True):
            np.testing.assert_array_equal(arrays[False, nuisance, False], arrays[False, nuisance, True])
        second = self.root / "controls-again"
        controlled_examples(crops, second, seed=17, max_crops=1)
        for example in examples:
            self.assertEqual(sha256(controls / example["path"] / "source_rgb.png"),
                             sha256(second / example["path"] / "source_rgb.png"))


if __name__ == "__main__":
    unittest.main()
