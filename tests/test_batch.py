import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from facade_change.alignment import SIFTMatcher
from facade_change.batch import run_batch
from facade_change.data import build_manifest
from facade_change.demo import make_fixture
from facade_change.geometry import transform_points
from facade_change.io import read_json, sha256, write_json
from facade_change.pipeline import run_pair
from facade_change.preparation import prepare_dataset


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required for integration")
class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        fixture = make_fixture(self.root / "fixture")
        build_manifest(fixture / "paths.json", self.root / "manifest", fixture / "reviewed.csv")
        prepare_dataset(self.root / "manifest/manifest.json", self.root / "prepared")
        self.manifest = self.root / "prepared/manifest.json"

    def tearDown(self):
        self.temp.cleanup()

    def test_batch_to_native_crops_and_controls_without_unnecessary_loftr(self):
        out = self.root / "batch"
        with patch("facade_change.pipeline.LoFTRMatcher", side_effect=AssertionError("Unnecessary LoFTR")) as loader:
            summary = run_batch(self.manifest, out, methods=["sift", "cascade"], crops=True, controls=1)
        loader.assert_not_called()
        self.assertEqual(summary["passed_routing_gate"], 2)
        self.assertEqual(summary["derivative_failures"], 0)
        derivatives = read_json(out / "derivatives.json")["0-1"]
        self.assertGreater(derivatives["crop_summary"]["crop_count"], 0)
        self.assertEqual(derivatives["control_summary"]["example_count"], 6)
        self.assertTrue((out / "comparison.html").is_file())
        for path, digest in read_json(out / "run.json")["artifact_sha256"].items():
            self.assertEqual(sha256(out / path), digest)

    def test_cascade_preserves_failed_attempt_and_recovers_known_geometry(self):
        class FailedMatcher:
            def match(self, *args):
                raise ValueError("Fixture has no first-stage matches")
        # Routing seam only; pretrained LoFTR is checked separately with actual weights.
        result = run_pair(self.manifest, 0, 1, self.root / "fallback", method="cascade",
                          matchers={"sift": FailedMatcher(), "loftr": SIFTMatcher()})
        self.assertEqual(result["selected_method"], "loftr")
        self.assertEqual([a["status"] for a in result["attempts"]], ["failed", "completed"])
        geometry = read_json(self.root / "fallback/geometry.json")
        probes = np.array([[80., 40.], [400., 340.]])
        np.testing.assert_allclose(transform_points(probes, geometry["source_to_reference"]),
                                   probes + [64, 0], atol=.5)

    def test_failed_method_does_not_remove_successful_comparison_or_denominator(self):
        out = self.root / "partial"
        with patch("facade_change.pipeline.LoFTRMatcher", side_effect=RuntimeError("Unavailable checkpoint")):
            summary = run_batch(self.manifest, out, methods=["loftr", "sift"])
        self.assertEqual(summary["attempted_runs"], 2)
        self.assertEqual(summary["failed_runs"], 1)
        self.assertEqual(summary["passed_routing_gate"], 1)
        self.assertEqual(read_json(out / "run.json")["status"], "completed_with_issues")
        self.assertIn("Unavailable checkpoint", (out / "comparison.html").read_text())

    def test_empty_requested_crops_are_reported_as_issue(self):
        with patch("facade_change.derived.build_crops", return_value={"crop_count": 0}):
            summary = run_batch(self.manifest, self.root / "empty", methods=["sift"], crops=True)
        self.assertEqual(summary["derivative_failures"], 1)
        self.assertEqual(read_json(self.root / "empty/run.json")["status"], "completed_with_issues")


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required for integration")
class UnifiedPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        base = make_fixture(self.root / "base")
        arrays = []
        for path in sorted((base / "images").glob("*.png")):
            with Image.open(path) as image:
                arrays.append(np.array(image))
        inputs = self.root / "inputs"
        (inputs / "images").mkdir(parents=True)
        images, metadata = [], []
        for group in range(10):
            for index, year in enumerate((2010, 2020)):
                image_id = 2 * group + index
                name = f"building{group:02d}_{year}.png"
                # Distinct bytes across groups, with the known within-pair geometry.
                array = arrays[index] // 2 + group * 3
                Image.fromarray(array).save(inputs / "images" / name)
                images.append({"id": image_id, "file_name": name, "width": 512, "height": 384})
                metadata.append({"image_id": image_id, "view_id": f"view{group}",
                                 "building_id": f"building{group}", "year": year,
                                 "reviewed": "true", "notes": "Synthetic test only"})
        write_json(inputs / "coco.json", {"images": images,
                   "annotations": [{"id": 100, "image_id": 0, "category_id": 7,
                                    "segmentation": [[10, 10, 40, 10, 40, 40]], "area": 450}],
                   "categories": [{"id": 7, "name": "synthetic"}]})
        with (inputs / "reviewed.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metadata[0]))
            writer.writeheader()
            writer.writerows(metadata)
        self.config = inputs / "dataset.json"
        write_json(self.config, {"coco_json": "coco.json", "image_roots": ["images"],
                   "metadata_csv": "reviewed.csv", "alignment": {"methods": ["sift"], "device": "cpu"},
                   "crops": {"method": "sift", "stride": 256, "controls": 0}})
        script = Path(__file__).resolve().parents[1] / "scripts/2026-10-03_prepare_dataset.py"
        spec = importlib.util.spec_from_file_location("unified_dataset_runner", script)
        self.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.runner)

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_flow_preserves_groups_paths_and_annotation_scope(self):
        out = self.root / "complete"
        summary = self.runner.run_dataset(self.config, out)
        self.assertEqual(summary["status"], "completed_needs_review")
        self.assertEqual(summary["preparation"]["image_splits"], {"train": 14, "val": 2, "test": 4})
        self.assertEqual(summary["alignment"]["eligible_pairs"], 10)
        self.assertEqual(summary["alignment"]["failed_runs"], 0)
        self.assertGreater(summary["crop_count"], 0)
        index = read_json(out / summary["index_path"])
        self.assertEqual(len(index["pairs"]), 10)
        self.assertEqual(len(index["crops"]), summary["crop_count"])
        self.assertFalse(index["semantic_masks"]["rasterized"])
        self.assertEqual(index["temporal_ground_truth"], "not_created")
        self.assertEqual(index["source_annotations"]["sha256"], sha256(self.config.parent / "coco.json"))
        groups = index["split"]["building_assignments"]
        for crop in index["crops"]:
            self.assertEqual(crop["split"], groups[crop["building_id"]])
            for key in ("reference_rgb", "source_rgb", "geometric_overlap"):
                self.assertTrue((out / crop[key]).is_file())
        self.assertEqual(read_json(out / "run.json")["status"], "completed_needs_review")

    def test_directory_images_outside_coco_are_never_prepared(self):
        inputs = self.config.parent
        Image.fromarray(np.full((384, 512, 3), 177, np.uint8)).save(inputs / "images/outside_2025.png")
        out = self.root / "coco-only"
        summary = self.runner.run_dataset(self.config, out)
        manifest = read_json(out / "prepared/manifest.json")
        self.assertEqual(len(manifest["images"]), 20)
        self.assertTrue(all(row["file_name"] != "outside_2025.png" for row in manifest["images"]))
        self.assertEqual(summary["preparation"]["image_splits"], {"train": 14, "val": 2, "test": 4})
        index = read_json(out / summary["index_path"])
        self.assertEqual(index["image_selection"], "current_coco_images_only")
        self.assertTrue(all(int(crop["source_id"]) < 20 and int(crop["reference_id"]) < 20
                            for crop in index["crops"]))

    def test_expansion_preserves_gold_and_extends_current_coco_through_crops(self):
        first = self.root / "gold"
        self.runner.run_dataset(self.config, first, prepare_only=True)
        original_split = read_json(first / "prepared/split.json")
        building = next(name for name, split in original_split["building_assignments"].items() if split == "train")
        prepared = read_json(first / "prepared/manifest.json")
        reference = next(row for row in prepared["images"] if row["building_id"] == building)
        with Image.open(reference["image_path"]) as image:
            original = np.array(image)
        inputs = self.config.parent
        coco = read_json(inputs / "coco.json")
        with (inputs / "reviewed.csv").open(encoding="utf-8", newline="") as stream:
            metadata = list(csv.DictReader(stream))
        for i in range(10):
            image_id = 20 + i
            name = f"extension{i}_2030.png"
            array = original.copy()
            array[0, 0] = [i, 200, 255]
            Image.fromarray(array).save(inputs / "images" / name)
            coco["images"].append({"id": image_id, "file_name": name, "width": 512, "height": 384})
            metadata.append({"image_id": image_id, "view_id": reference["view_id"] if i == 0 else f"new_view{i}",
                             "building_id": building if i == 0 else f"new_building{i}",
                             "year": 2030, "reviewed": "true", "notes": "Synthetic extension"})
        write_json(inputs / "coco.json", coco)
        with (inputs / "reviewed.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metadata[0]))
            writer.writeheader()
            writer.writerows(metadata)
        config = read_json(self.config)
        config["previous_split"] = str(first / "prepared/split.json")
        write_json(self.config, config)
        out = self.root / "expanded"
        summary = self.runner.run_dataset(self.config, out)
        self.assertTrue(summary["previous_split_applied"])
        self.assertEqual(summary["preparation"]["image_splits"], {"train": 21, "val": 3, "test": 6})
        split = read_json(out / "prepared/split.json")
        self.assertEqual(split["gold_cohort"], original_split["gold_cohort"])
        for group, partition in original_split["building_assignments"].items():
            self.assertEqual(split["building_assignments"][group], partition)
        manifest = read_json(out / "prepared/manifest.json")
        self.assertEqual(sum(row["gold_member"] for row in manifest["images"]), 20)
        added = next(row for row in manifest["images"] if row["image_id"] == 20)
        self.assertEqual(added["split"], "train")
        index = read_json(out / summary["index_path"])
        added_crops = [row for row in index["crops"] if row["source_id"] == 20]
        self.assertGreater(len(added_crops), 0)
        self.assertTrue(all(row["split"] == "train" and row["reference_gold_member"]
                            and not row["source_gold_member"] for row in added_crops))

    def test_reused_manifest_cannot_add_or_remove_coco_members(self):
        first = self.root / "first"
        self.runner.run_dataset(self.config, first, prepare_only=True)
        manifest_path = first / "inventory/manifest.json"
        manifest = read_json(manifest_path)
        original_images = manifest["images"][:]
        config = read_json(self.config)
        config["manifest_path"] = str(manifest_path)
        write_json(self.config, config)
        mutations = [original_images + [{**original_images[0], "image_id": 999}],
                     original_images[:-1],
                     [{**row, "file_name": "different.png"} if i == 0 else row
                      for i, row in enumerate(original_images)]]
        for i, rows in enumerate(mutations):
            with self.subTest(mutation=i):
                manifest["images"] = rows
                write_json(manifest_path, manifest)
                with self.assertRaisesRegex(ValueError, "exactly.*COCO"):
                    self.runner.run_dataset(self.config, self.root / f"invalid-members-{i}", prepare_only=True)

    def test_previous_split_is_rejected_in_dev_mode(self):
        config = read_json(self.config)
        config.update(previous_split="old/split.json", split={"mode": "dev"})
        write_json(self.config, config)
        with self.assertRaisesRegex(ValueError, "previous_split.*reviewed"):
            self.runner.load_config(self.config)

    def test_missing_gold_image_is_reported_as_issue_in_unified_run(self):
        first = self.root / "gold"
        self.runner.run_dataset(self.config, first, prepare_only=True)
        manifest = read_json(first / "prepared/manifest.json")
        missing_hash = manifest["images"][-1]["sha256"]
        inputs = self.config.parent
        coco = read_json(inputs / "coco.json")
        coco["images"].pop()
        write_json(inputs / "coco.json", coco)
        with (inputs / "reviewed.csv").open(encoding="utf-8", newline="") as stream:
            metadata = list(csv.DictReader(stream))[:-1]
        with (inputs / "reviewed.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metadata[0]))
            writer.writeheader()
            writer.writerows(metadata)
        config = read_json(self.config)
        config["previous_split"] = str(first / "prepared/split.json")
        write_json(self.config, config)
        out = self.root / "missing-gold"
        summary = self.runner.run_dataset(self.config, out, prepare_only=True)
        self.assertEqual(summary["status"], "completed_with_issues")
        self.assertEqual(summary["preparation"]["missing_gold_sha256"], [missing_hash])
        self.assertIn(missing_hash, (out / "summary.txt").read_text(encoding="utf-8"))

    def test_reuse_inventory_never_decodes_originals_for_split_changes(self):
        first = self.root / "first"
        self.runner.run_dataset(self.config, first, prepare_only=True)
        config = read_json(self.config)
        config["manifest_path"] = str(first / "inventory/manifest.json")
        write_json(self.config, config)
        with patch("facade_change.data.load_rgb", side_effect=AssertionError("Unexpected RGB scan")):
            summary = self.runner.run_dataset(self.config, self.root / "reused", prepare_only=True)
        self.assertTrue(summary["inventory_reused"])
        self.assertFalse((self.root / "reused/inventory").exists())
        coco = read_json(self.config.parent / "coco.json")
        coco["annotations"][0]["area"] = 451
        write_json(self.config.parent / "coco.json", coco)
        with self.assertRaisesRegex(ValueError, "COCO|coco"):
            self.runner.run_dataset(self.config, self.root / "stale", prepare_only=True)

    def test_unreviewed_metadata_emits_tables_without_alignment(self):
        config = read_json(self.config)
        config["metadata_csv"] = None
        write_json(self.config, config)
        out = self.root / "review"
        with patch.object(self.runner, "run_batch", side_effect=AssertionError("Premature alignment")):
            summary = self.runner.run_dataset(self.config, out)
        self.assertEqual(summary["status"], "needs_metadata_review")
        self.assertIsNone(summary["index_path"])
        self.assertTrue((out / "prepared/metadata_review.csv").is_file())
        self.assertTrue((out / "prepared/group_review.csv").is_file())
        self.assertFalse((out / "batch").exists())
        self.assertTrue(read_json(out / "prepared/split.json")["development_only"])

    def test_invalid_fraction_plan_is_rejected(self):
        config = read_json(self.config)
        config["split"] = {"train": .7, "val": .1, "test": .3}
        write_json(self.config, config)
        with self.assertRaisesRegex(ValueError, "sum|fraction"):
            self.runner.run_dataset(self.config, self.root / "invalid", prepare_only=True)

    def test_missing_input_is_preserved_and_reported_as_issue(self):
        coco_path = self.config.parent / "coco.json"
        coco = read_json(coco_path)
        coco["images"].append({"id": 20, "file_name": "missing_2020.png", "width": 512, "height": 384})
        write_json(coco_path, coco)
        out = self.root / "missing"
        summary = self.runner.run_dataset(self.config, out, prepare_only=True)
        self.assertEqual(summary["status"], "completed_with_issues")
        self.assertEqual(summary["preparation"]["image_splits"]["excluded"], 1)
        manifest = read_json(out / "prepared/manifest.json")
        self.assertEqual(len(manifest["images"]), 21)
        self.assertEqual(manifest["images"][-1]["image_status"], "missing")

    def test_failed_alignment_stays_in_overall_summary(self):
        config = read_json(self.config)
        config["alignment"]["limit"] = 1
        write_json(self.config, config)
        out = self.root / "failed-alignment"
        with patch("facade_change.batch.run_pair", side_effect=RuntimeError("Synthetic matcher failure")):
            summary = self.runner.run_dataset(self.config, out)
        self.assertEqual(summary["status"], "completed_with_issues")
        self.assertEqual(summary["alignment"]["eligible_pairs"], 1)
        self.assertEqual(summary["alignment"]["failed_runs"], 1)
        self.assertEqual(summary["crop_count"], 0)
        self.assertEqual(len(read_json(out / "batch/results.json")), 1)


if __name__ == "__main__":
    unittest.main()
