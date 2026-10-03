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
from facade_change.crop_dataset import prepare_crop_dataset
from facade_change.data import build_manifest
from facade_change.demo import make_fixture
from facade_change.io import read_json, sha256, write_json
from facade_change.pipeline import run_pair
from facade_change.preparation import prepare_dataset


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required for native crop export")
class CropDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        base = make_fixture(self.root / "fixture")
        arrays = []
        for path in sorted((base / "images").glob("*.png")):
            with Image.open(path) as image:
                arrays.append(np.array(image))
        images = self.root / "images"
        images.mkdir()
        coco_images, metadata = [], []
        for group in range(3):
            for index, year in enumerate((2010, 2020)):
                image_id = 2 * group + index
                name = f"wall{group}_{year}.png"
                Image.fromarray(arrays[index] // 2 + group * 3).save(images / name)
                coco_images.append({"id": image_id, "file_name": name, "width": 512, "height": 384})
                metadata.append({"image_id": image_id, "view_id": f"wall{group}", "building_id": f"building{group}",
                                 "year": year, "reviewed": "true", "notes": "Synthetic geometry only"})
        write_json(self.root / "coco.json", {"images": coco_images, "annotations": [], "categories": []})
        write_json(self.root / "paths.json", {"coco_json": "coco.json", "image_roots": ["images"], "metadata_rules": None})
        with (self.root / "review.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metadata[0]))
            writer.writeheader()
            writer.writerows(metadata)
        build_manifest(self.root / "paths.json", self.root / "inventory", self.root / "review.csv")
        prepare_dataset(self.root / "inventory/manifest.json", self.root / "prepared", split_mode="reviewed")
        self.manifest = self.root / "prepared/manifest.json"
        self.batch = self.root / "batch"

        def partial_pair(manifest, reference, source, out, method, **kwargs):
            if int(reference) == 4 or (int(reference) == 2 and method == "loftr"):
                raise RuntimeError("Synthetic unavailable alignment")
            return run_pair(manifest, reference, source, out, method=method, **kwargs)

        # Mock only the model seam: both successful geometry runs use actual SIFT.
        with patch("facade_change.batch.run_pair", side_effect=partial_pair):
            with patch("facade_change.pipeline.LoFTRMatcher", side_effect=lambda *args, **kwargs: SIFTMatcher()):
                run_batch(self.manifest, self.batch, methods=["sift", "loftr"], limit=0)

    def tearDown(self):
        self.temp.cleanup()

    def update_batch_hash(self, path):
        record = read_json(self.batch / "run.json")
        record["artifact_sha256"][path.relative_to(self.batch).as_posix()] = sha256(path)
        write_json(self.batch / "run.json", record)

    def test_priority_fallback_existing_split_and_originals_without_rematching(self):
        out = self.root / "crops"
        before = {path: path.read_bytes() for path in (self.batch / "run.json", self.batch / "results.json", self.manifest)}
        with patch("facade_change.batch.run_pair", side_effect=AssertionError("Crop export must not rematch")):
            with patch("facade_change.pipeline.SIFTMatcher", side_effect=AssertionError("No matcher needed")):
                summary = prepare_crop_dataset(self.batch, out, tile_size=128, stride=128, min_valid_fraction=1.)
        self.assertEqual((summary["selected_pair_count"], summary["exported_pair_count"], summary["excluded_pair_count"]), (3, 2, 1))
        self.assertEqual(summary["pair_method_counts"], {"loftr": 1, "sift": 1})
        self.assertEqual(summary["derivative_failures"], 0)
        self.assertGreater(summary["crop_count"], 0)
        index = read_json(out / summary["index_path"])
        self.assertEqual([(row["pair_id"], row["method"]) for row in index["pairs"]], [("0-1", "loftr"), ("2-3", "sift")])
        self.assertEqual(index["excluded_pairs"][0]["pair_id"], "4-5")
        self.assertEqual(len({row["dataset_crop_id"] for row in index["crops"]}), len(index["crops"]))
        source = read_json(self.manifest)
        owners = {str(row["image_id"]): row for row in source["images"]}
        for crop in index["crops"]:
            reference = owners[str(crop["reference_id"])]
            self.assertEqual((crop["split"], crop["building_id"]), (reference["split"], reference["building_id"]))
            self.assertEqual((crop["reference_year"], crop["source_year"]), (2010, 2020))
            self.assertEqual(crop["reference_file_name"], reference["file_name"])
            self.assertTrue(crop["reference_gold_member"] and crop["source_gold_member"])
            for key in ("reference_rgb", "source_rgb", "geometric_overlap", "not_comparable"):
                self.assertTrue((out / crop[key]).is_file())
            self.assertEqual(crop["valid_fraction"], 1.)
            with Image.open(out / crop["not_comparable"]) as image:
                self.assertFalse(np.asarray(image).any())
        with Image.open(self.batch / "loftr/pair-0-1/source_only.png") as image:
            self.assertGreater(np.asarray(image).sum(), 0)
        self.assertTrue(all(row["rejected_low_overlap"] > 0 for row in index["pairs"]))
        self.assertEqual((out / "split.json").read_bytes(), (self.manifest.parent / "split.json").read_bytes())
        self.assertFalse(index["semantic_masks"]["rasterized"])
        self.assertEqual(index["temporal_ground_truth"], "not_created")
        self.assertIn("loftr", (out / summary["gallery_path"]).read_text(encoding="utf-8"))
        with (out / summary["csv_index_path"]).open(encoding="utf-8", newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        self.assertEqual(len(csv_rows), len(index["crops"]))
        self.assertEqual((csv_rows[0]["reference_year"], csv_rows[0]["source_year"]), ("2010", "2020"))
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_rejected_methods_are_never_exported_as_accepted(self):
        results = read_json(self.batch / "results.json")
        for row in results:
            if row["pair_id"] == "0-1":
                row["status"] = "rejected"
        write_json(self.batch / "results.json", results)
        self.update_batch_hash(self.batch / "results.json")
        summary = prepare_crop_dataset(self.batch, self.root / "rejected", tile_size=128, stride=128)
        self.assertEqual(summary["selected_pair_count"], 3)
        self.assertEqual(summary["exported_pair_count"], 1)
        self.assertEqual(summary["excluded_pair_count"], 2)
        index = read_json(self.root / "rejected" / summary["index_path"])
        self.assertEqual({row["pair_id"] for row in index["crops"]}, {"2-3"})
        self.assertIn("loftr:rejected", index["excluded_pairs"][0]["method_statuses"])

    def test_pair_observation_split_cannot_override_current_manifest(self):
        pair_run = self.batch / "loftr/pair-0-1/run.json"
        record = read_json(pair_run)
        original = record["observations"]["source"]["split"]
        record["observations"]["source"]["split"] = "test" if original != "test" else "train"
        write_json(pair_run, record)
        self.update_batch_hash(pair_run)
        summary = prepare_crop_dataset(self.batch, self.root / "wrong-owner", tile_size=128, stride=128)
        index = read_json(self.root / "wrong-owner" / summary["index_path"])
        excluded = next(row for row in index["excluded_pairs"] if row["pair_id"] == "0-1")
        self.assertIn("source.split disagrees", excluded["reason"])
        self.assertEqual(summary["derivative_failures"], 1)
        self.assertEqual({row["pair_id"] for row in index["crops"]}, {"2-3"})

    def test_changed_manifest_or_results_fail_provenance_check(self):
        results_path = self.batch / "results.json"
        original = results_path.read_bytes()
        results_path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "results changed"):
            prepare_crop_dataset(self.batch, self.root / "changed-results")
        results_path.write_bytes(original)
        self.manifest.write_bytes(self.manifest.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            prepare_crop_dataset(self.batch, self.root / "changed-manifest")

    def test_total_control_budget_and_group_inheritance(self):
        out = self.root / "controls"
        summary = prepare_crop_dataset(self.batch, out, tile_size=128, stride=128, controls=3)
        self.assertEqual(summary["control_source_crop_count"], 3)
        self.assertEqual(summary["control_example_count"], 18)
        self.assertEqual(summary["control_failures"], 0)
        self.assertIn("controls/pair-0-1/gallery.html", (out / "gallery.html").read_text(encoding="utf-8"))
        index = read_json(out / summary["index_path"])
        self.assertEqual(len(index["controls"]["examples"]), 18)
        self.assertIn("no real damage", index["controls"]["label_scope"])
        crop_owners = {(row["pair_id"], row["crop_id"]): (row["split"], row["building_id"]) for row in index["crops"]}
        for example in index["controls"]["examples"]:
            self.assertEqual((example["split"], example["building_id"]), crop_owners[(example["pair_id"], example["parent_crop_id"])])
            self.assertTrue((out / example["path"] / "labels_reference.png").is_file())
        self.assertEqual(sum(example["sham_self_paste"] for example in index["controls"]["examples"]), 6)

    def test_single_method_choice_does_not_silently_fallback(self):
        summary = prepare_crop_dataset(self.batch, self.root / "loftr-only", methods=("loftr",), tile_size=128, stride=128)
        self.assertEqual(summary["pair_method_counts"], {"loftr": 1})
        self.assertEqual(summary["exported_pair_count"], 1)
        self.assertEqual(summary["excluded_pair_count"], 2)


if __name__ == "__main__":
    unittest.main()
