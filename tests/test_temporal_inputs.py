"""Causal input provenance, no TEST access, and native real-grid integration."""
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from facade_change.hypothesis_dataset import prepare_history_dataset
from facade_change.io import finish_record, read_json, run_record, sha256, write_json


inputs = importlib.import_module("facade_change.research.2026-10-14_temporal_inputs")


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required")
class TemporalInputsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.images, crops, coco_images = [], [], []
        specifications = [("train", year) for year in range(2001, 2006)] + [
            ("val", 2010), ("val", 2011), ("test", 2012)]
        for identifier, (part, year) in enumerate(specifications, 1):
            yy, xx = np.indices((32, 40))
            rgb = np.stack([(xx * 4 + identifier) % 255, (yy * 5 + 30) % 255,
                            np.full_like(xx, 80 + identifier)], axis=-1).astype(np.uint8)
            path = self.dataset / f"photo-{identifier}.png"
            Image.fromarray(rgb).save(path)
            row = {"image_id": identifier, "image_path": str(path), "file_name": path.name,
                   "sha256": sha256(path), "width": 40, "height": 32,
                   "building_id": f"building-{part}", "view_id": f"view-{part}",
                   "year": year, "split": part, "metadata_status": "reviewed",
                   "temporal_eligible": True}
            self.images.append(row)
            coco_images.append({"id": identifier, "file_name": path.name, "width": 40, "height": 32})
            crops.append({"crop_id": f"crop-{identifier}", "image_id": identifier,
                          "image_sha256": row["sha256"], "building_id": row["building_id"],
                          "view_id": row["view_id"], "split": part, "x": 0, "y": 0,
                          "width": 40, "height": 32, "pad_right": 0, "pad_bottom": 8})
        pairs = []
        for reference, sources in ((self.images[0], self.images[1:5]), (self.images[5], self.images[6:7])):
            for source in sources:
                pairs.append({"pair_id": f"{reference['image_id']}-{source['image_id']}",
                              "reference_id": reference["image_id"], "source_id": source["image_id"],
                              "building_id": reference["building_id"], "view_id": reference["view_id"],
                              "split": reference["split"]})
        categories = [{"id": 1, "name": "CRACK"}]
        write_json(self.dataset / "manifest.json", {"images": self.images, "pairs": pairs,
                                                   "categories": categories})
        write_json(self.dataset / "split.json", {"building_assignments": {
            row["building_id"]: row["split"] for row in self.images}})
        write_json(self.dataset / "coco.json", {"images": coco_images, "categories": categories,
                                               "annotations": []})
        (self.dataset / "crops.jsonl").write_text("".join(json.dumps(row) + "\n" for row in crops))
        write_json(self.dataset / "dataset.json", {"kind": "semantic_crop_index", "crop_size": 40,
                                                  "source_coco": "coco.json", "manifest": "manifest.json",
                                                  "crops": "crops.jsonl", "split": "split.json",
                                                  "source_coco_sha256": sha256(self.dataset / "coco.json"),
                                                  "categories": categories})
        Path(self.images[-1]["image_path"]).unlink()  # TEST access must fail if attempted.

    def history(self):
        out = self.root / "history"
        prepare_history_dataset(self.dataset, out)
        return out

    def alignment(self, reference, source, passed=True):
        out = self.root / f"pair-{reference['image_id']}-{source['image_id']}"
        out.mkdir()
        record = run_record("pair_alignment", {"method": "sift"})
        record["observations"] = {"reference": reference, "source": source}
        record["quality_gate"] = {"passed": passed}
        write_json(out / "geometry.json", {"source_to_reference": np.eye(3).tolist(),
                                           "reference_to_canvas": np.eye(3).tolist(),
                                           "width": reference["width"], "height": reference["height"]})
        finish_record(out, record, "completed_needs_review" if passed else "completed_rejected")
        return str(out)

    def real_config(self):
        return {"dataset": str(self.dataset), "real_tile_size": 16, "real_stride": 16,
                "alignment_runs": [self.alignment(self.images[0], self.images[1]),
                                   self.alignment(self.images[0], self.images[2]),
                                   self.alignment(self.images[0], self.images[3], False),
                                   self.alignment(self.images[5], self.images[6])]}

    def test_synthetic_preflight_is_metadata_only_and_loads_exact_native_artifacts(self):
        config = {"history_run": str(self.history())}
        with patch.object(inputs, "load_rgb", side_effect=AssertionError("decoded in preflight")), \
                patch.object(inputs.Image, "open", side_effect=AssertionError("labels decoded")):
            sequences, provenance, coverage = inputs.preflight(config)
        self.assertEqual(len(sequences), 4)
        self.assertFalse(provenance["test_rgb_loaded"])
        self.assertEqual(coverage["synthetic"]["bases"], 2)
        event = next(row for row in sequences if row["arm"] == "event")
        old, persistent = inputs.load_frame(event, 2), inputs.load_frame(event, 3)
        np.testing.assert_array_equal(old["latent_rgb"], persistent["latent_rgb"])
        self.assertFalse((persistent["event_labels"] == 1).any())
        self.assertEqual(persistent["rgb"].shape, (32, 40, 3))
        self.assertEqual(persistent["support"].dtype, np.bool_)

    def test_procedural_occlusion_is_evaluation_only_not_model_support(self):
        from facade_change.augmentations import apply_nuisance

        def occlude_onset(rgb, support, spec):
            if spec["id"] == "frame_3":
                spec = {"id": spec["id"], "kind": "occlusion", "rectangle": [0., 0., 1., 1.]}
            return apply_nuisance(rgb, support, spec)

        with patch("facade_change.augmentations.apply_nuisance", side_effect=occlude_onset):
            history = self.history()
        sequences, provenance, _ = inputs.load_sequences({"history_run": str(history)})
        manifest = read_json(history / "manifest.json")
        parent_frame = manifest["sequences"][0]["frames"][2]
        with Image.open(history / parent_frame["paths"]["support"]) as image:
            self.assertFalse(np.asarray(image).any())
        for sequence in sequences:
            first, hidden = inputs.load_frame(sequence, 0), inputs.load_frame(sequence, 2)
            np.testing.assert_array_equal(first["support"], hidden["support"])
            self.assertTrue(hidden["support"].all())
            self.assertFalse(hidden["visibility"].any())
            self.assertTrue((hidden["event_labels"] == 255).all())
            self.assertEqual(sequence["frames"][2]["model_support_source"], "base_geometric_support")
            self.assertIn(sequence["frames"][2]["paths"]["support"], provenance["files_sha256"])

    def test_real_chronology_native_pixels_coverage_and_unknown_labels(self):
        config = self.real_config()
        with patch.object(inputs, "load_rgb", side_effect=AssertionError("decoded in preflight")):
            candidates, _, _ = inputs.preflight(config)
        self.assertEqual(sorted(len(s["frames"]) for s in candidates), [2, 3])
        sequences, provenance, coverage = inputs.load_sequences(config)
        train = next(s for s in sequences if s["split"] == "train")
        self.assertEqual([f["year"] for f in train["frames"]], [2001, 2002, 2003])
        self.assertEqual(train["missing_dates"], 2)
        actual = inputs.load_frame(train, 1)
        with Image.open(self.images[1]["image_path"]) as image:
            native = np.array(image)
        x, y, w, h = train["crop_xywh"]
        np.testing.assert_array_equal(actual["rgb"], native[y:y+h, x:x+w])
        self.assertTrue(actual["support"].all())
        self.assertTrue((actual["event_labels"] == 255).all())
        self.assertNotIn("latent_rgb", actual)
        train_coverage = next(v for v in coverage["real"]["views"] if v["split"] == "train")
        self.assertEqual([p["status"] for p in train_coverage["pairs"]],
                         ["accepted", "accepted", "rejected_or_failed_alignment", "missing_alignment"])
        self.assertFalse(any("photo-8" in path for path in provenance["files_sha256"]))
        self.assertEqual(provenance, inputs.load_sequences(config)[1])

    def test_changed_rgb_and_geometry_are_rejected(self):
        config = self.real_config()
        sequences, _, _ = inputs.load_sequences(config)
        train = next(s for s in sequences if s["split"] == "train")
        Image.fromarray(np.zeros((32, 40, 3), np.uint8)).save(self.images[1]["image_path"])
        with self.assertRaisesRegex(ValueError, "changed after"):
            inputs.load_frame(train, 1)
        geometry = Path(config["alignment_runs"][0]) / "geometry.json"
        write_json(geometry, {"source_to_reference": np.eye(3).tolist()})
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            inputs.preflight(config)

    def test_cross_source_split_leakage_is_rejected(self):
        config = self.real_config()
        config["history_run"] = str(self.history())
        manifest = read_json(self.dataset / "manifest.json")
        manifest["images"][-1]["sha256"] = manifest["images"][0]["sha256"]
        write_json(self.dataset / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "split leakage"):
            inputs.preflight(config)

    def test_test_frame_access_is_explicitly_forbidden(self):
        with self.assertRaisesRegex(ValueError, "TEST"):
            inputs.load_frame({"split": "test", "frames": []}, 0)


if __name__ == "__main__":
    unittest.main()
