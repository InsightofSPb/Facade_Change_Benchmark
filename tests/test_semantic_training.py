import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from facade_change.semantic_training import include_auxiliary_training, _transition_windows, mine_transition_candidates
from facade_change.io import finish_record, read_json, sha256, write_json


class AuxiliarySemanticTrainingTests(unittest.TestCase):
    def row(self, image_id, split="excluded", **kwargs):
        row = {"image_id": image_id, "file_name": f"IMG_{image_id}.jpg", "sha256": str(image_id),
               "image_status": "ready", "metadata_status": "unknown", "building_id": None,
               "view_id": None, "year": None, "split": split, "semantic_eligible": False,
               "temporal_eligible": False,
               "preparation_exclusion_reasons": ["metadata_not_reviewed", "unknown_building"]}
        if split != "excluded":
            row.update(building_id=f"building{image_id}", metadata_status="reviewed",
                       semantic_eligible=True, preparation_exclusion_reasons=[])
        return {**row, **kwargs}

    def test_unknown_identity_is_preserved_and_originals_are_unchanged(self):
        rows = [self.row(1, "train"), self.row(2), self.row(3, year=2026, view_id="inferred_front")]
        prepared = {"images": rows, "pairs": [], "summary": {"semantic_eligible_image_count": 1}}
        split = {"building_assignments": {"building1": "train"}}
        before, original_split = copy.deepcopy(prepared), copy.deepcopy(split)
        manifest, report = include_auxiliary_training(prepared, split, True)
        self.assertEqual(prepared, before)
        self.assertEqual(split, original_split)
        self.assertEqual([r["image_id"] for r in report["included_images"]], [2, 3])
        self.assertFalse(manifest["physical_disjointness_verified"])
        self.assertEqual(manifest["images"][0]["training_role"], "reviewed")
        for row in manifest["images"][1:]:
            self.assertEqual(row["split"], "train")
            self.assertEqual(row["training_role"], "auxiliary_semantic_train")
            self.assertIsNone(row["building_id"])
            self.assertEqual(row["metadata_status"], "unknown")
            self.assertTrue(row["semantic_eligible"])
            self.assertFalse(row["temporal_eligible"])
        self.assertEqual(manifest["images"][2]["view_id"], "inferred_front")
        self.assertEqual(manifest["images"][2]["year"], 2026)
        self.assertEqual(manifest["pairs"], [])
        self.assertEqual(manifest["summary"]["semantic_eligible_image_count"], 3)

    def test_current_historical_and_absent_holdouts_are_blocked(self):
        rows = [self.row(1, "val", view_id="held_front"), self.row(2, sha256="1"),
                self.row(3, view_id="held_front"), self.row(4, building_id="absent_test"),
                self.row(5, sha256="historical_hash"), self.row(6, view_id="gold_view")]
        split = {"building_assignments": {"absent_test": "test"},
                 "historical_cohort": [{"split": "test", "sha256": "historical_hash"}],
                 "gold_cohort": [{"split": "val", "view_id": "gold_view"}]}
        manifest, report = include_auxiliary_training({"images": rows}, split, True)
        self.assertEqual(len(report["blocked_images"]), 5)
        self.assertFalse(report["included_images"])
        self.assertTrue(report["physical_disjointness_verified"])
        self.assertTrue(all(row["split"] == "excluded" for row in manifest["images"][1:]))

    def test_exact_train_duplicates_and_earlier_auxiliary_duplicates_are_skipped(self):
        rows = [self.row(1, "train"), self.row(2, sha256="1"), self.row(3), self.row(4, sha256="3")]
        manifest, report = include_auxiliary_training({"images": rows}, {}, True)
        self.assertEqual([r["image_id"] for r in report["included_images"]], [3])
        self.assertEqual([r["image_id"] for r in report["duplicate_images"]], [2, 4])
        self.assertEqual(manifest["images"][1]["split"], "excluded")
        self.assertEqual(manifest["images"][3]["auxiliary_training_exclusion_reason"], "duplicate_train_sha256")

    def test_disabled_and_integrity_exclusions_never_add_training_images(self):
        rows = [self.row(1), self.row(2, image_status="missing"), self.row(3, sha256=None),
                self.row(4, preparation_exclusion_reasons=["unknown_building", "image_dimension_mismatch"]),
                self.row(5, preparation_exclusion_reasons=[])]
        disabled, report = include_auxiliary_training({"images": rows}, {}, False)
        self.assertFalse(report["enabled"])
        self.assertFalse(report["included_images"])
        self.assertTrue(all(row["split"] == "excluded" for row in disabled["images"]))
        enabled, report = include_auxiliary_training({"images": rows}, {}, True)
        self.assertEqual([r["image_id"] for r in report["included_images"]], [1])
        self.assertTrue(all(row["split"] == "excluded" for row in enabled["images"][1:]))


@unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("pycocotools"),
                     "alignment and segmentation extras required")
class RealTransitionCandidateTests(unittest.TestCase):
    def setUp(self):
        from pycocotools import mask as mask_utils
        from facade_change.geometry import expanded_canvas

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.alignment = self.root / "alignment"
        self.alignment.mkdir()
        self.categories = [{"id": 1, "name": "DELAMINATION"}, {"id": 2, "name": "SPALLING"},
                           {"id": 3, "name": "REPAIRS"}, {"id": 4, "name": "ORNAMENT_INTACT"}]
        images, crops, annotations = [], [], []
        for image_id, year, split in ((1, 2009, "train"), (2, 2013, "train"),
                                     (3, 2016, "train"), (4, 2019, "test")):
            path = self.dataset / f"{image_id}.png"
            array = np.full((24, 24, 4), 60 + image_id, dtype=np.uint8)
            array[..., 3] = 255
            if image_id == 2:
                array[8:10, 8:10, 3] = 0
            Image.fromarray(array).save(path)
            building, view = ("B", "B-front") if split == "train" else ("C", "C-front")
            images.append({"image_id": image_id, "file_name": path.name, "image_path": str(path),
                           "sha256": sha256(path), "width": 24, "height": 24, "year": year,
                           "split": split, "building_id": building, "view_id": view,
                           "metadata_status": "reviewed", "temporal_eligible": True})
            crops.append({"crop_id": str(image_id), "image_id": image_id, "split": split,
                          "building_id": building, "view_id": view, "image_sha256": sha256(path),
                          "x": 0, "y": 0, "width": 24, "height": 24, "pad_right": 0, "pad_bottom": 0})
            if split == "test":
                path.unlink()  # Mining must never load TEST RGB or this invalid annotation.
                annotations.append({"id": 99, "image_id": 4, "category_id": 999, "segmentation": None})
                continue
            for category in (4, 1 if image_id == 1 else 2):
                mask = np.ones((24, 24), np.uint8) if category == 4 else np.zeros((24, 24), np.uint8)
                if category != 4:
                    x = 6 if image_id == 1 else 4
                    mask[4:16, x:x+12] = 1
                rle = mask_utils.encode(np.asfortranarray(mask))
                rle["counts"] = rle["counts"].decode("ascii")
                annotations.append({"id": len(annotations)+1, "image_id": image_id,
                                    "category_id": category, "segmentation": rle})
        pairs = [{"pair_id": f"1-{source}", "reference_id": 1, "source_id": source,
                  "building_id": "B", "view_id": "B-front", "split": "train"} for source in (2, 3)]
        write_json(self.dataset / "manifest.json", {"images": images, "pairs": pairs,
                                                     "categories": self.categories})
        write_json(self.dataset / "split.json", {"building_assignments": {"B": "train", "C": "test"}})
        write_json(self.dataset / "coco.json", {"categories": self.categories, "annotations": annotations,
                    "images": [{"id": row["image_id"], "file_name": row["file_name"],
                                "width": 24, "height": 24} for row in images]})
        (self.dataset / "crops.jsonl").write_text("".join(json.dumps(row)+"\n" for row in crops))
        write_json(self.dataset / "dataset.json", {"kind": "semantic_crop_index", "crop_size": 24,
                   "source_coco": "coco.json", "source_coco_sha256": sha256(self.dataset / "coco.json"),
                   "manifest": "manifest.json", "split": "split.json", "crops": "crops.jsonl",
                   "categories": self.categories})
        # Native source -> reference translates right; canvas retains ref scale.
        canvas = expanded_canvas((24, 24), (24, 24), [[1, 0, 2], [0, 1, 0], [0, 0, 1]])
        write_json(self.alignment / "geometry.json", canvas.as_dict())
        self.pair_record = {"kind": "pair_alignment", "config": {"method": "sift"},
                            "quality_gate": {"passed": True, "reasons": []},
                            "observations": {"reference": images[0], "source": images[1]}}
        finish_record(self.alignment, self.pair_record, "completed_needs_review")

    def run_mining(self, **kwargs):
        return mine_transition_candidates(self.dataset, self.alignment, self.root / "result",
                                           window_sizes=(8, 12), min_component_pixels=2, **kwargs)

    def test_warp_direction_multilabel_masks_missing_pairs_and_no_test_access(self):
        result = self.run_mining()
        self.assertGreater(result["candidate_count"], 0)
        self.assertEqual(result["pair_status_counts"], {"mined_needs_review": 1, "missing_alignment": 1})
        self.assertEqual(result["views"][0]["usable_observations"], 2)
        self.assertFalse(result["views"][0]["history_of_at_least_3_observations"])
        self.assertFalse(result["test_rgb_loaded"])
        self.assertFalse(result["quantitative_temporal_evaluation_allowed"])
        rows = read_json(self.root / "result/candidates.json")["candidates"]
        for row in rows:
            self.assertEqual((row["from_class"], row["to_class"]), ("DELAMINATION", "SPALLING"))
            self.assertEqual(row["review_status"], "needs_review")
            directory = self.root / "result" / row["path"]
            event = np.array(Image.open(directory / "candidate_mask.png")) > 0
            support = np.array(Image.open(directory / "annotated_support.png")) > 0
            with np.load(directory / "semantic_masks.npz") as masks:
                self.assertTrue(np.all(masks["reference"][event] == 9))  # Delamination + ornament.
                self.assertTrue(np.all(masks["source"][event] == 10))  # Spalling + ornament.
            self.assertTrue(np.all(support[event]))
            self.assertAlmostEqual(row["source_to_crop"][0][2] - row["reference_to_crop"][0][2], 2.)

    def test_unknown_and_overlapping_existing_target_are_not_disappearances(self):
        categories = [{"id": 1, "name": "DELAMINATION", "channel": 0},
                      {"id": 2, "name": "SPALLING", "channel": 1}]
        reference = np.ones((16, 16), np.uint16)
        source = np.full((16, 16), 2, np.uint16)
        common = np.ones((16, 16), bool)
        # Unknown source cannot act as a negative class observation.
        common[:, :8] = False
        # A still present at the source is not an A -> B transition.
        source[:, 8:12] = 3
        # B already present at reference is not a B appearance.
        reference[:, 12:] = 3
        windows = _transition_windows(reference, source, common, [tuple(categories)], (8,), .1, .01, 1)
        self.assertFalse(windows)
        source[4:12, 8:12] = 2
        windows = _transition_windows(reference, source, common, [tuple(categories)], (8,), .1, .01, 1)
        self.assertTrue(windows)
        self.assertTrue(all(row["candidate_pixels"] <= 32 for row in windows))

    def test_rekeyed_old_coco_ids_reuse_only_when_image_sha_matches(self):
        self.pair_record["observations"]["reference"]["image_id"] = 900
        self.pair_record["observations"]["source"]["image_id"] = 901
        finish_record(self.alignment, self.pair_record, "completed_needs_review")
        self.assertGreater(self.run_mining()["candidate_count"], 0)

    def test_old_split_mismatch_fails_instead_of_relabelling(self):
        self.pair_record["observations"]["reference"]["split"] = "val"
        finish_record(self.alignment, self.pair_record, "completed_needs_review")
        with self.assertRaisesRegex(ValueError, "dataset split mismatch"):
            self.run_mining()
        self.assertEqual(read_json(self.root / "result/run.json")["status"], "failed")

    def test_geometry_and_original_image_tamper_fail(self):
        geometry = read_json(self.alignment / "geometry.json")
        geometry["source_to_reference"][0][2] = -2
        write_json(self.alignment / "geometry.json", geometry)
        with self.assertRaisesRegex(ValueError, "geometry hash mismatch"):
            self.run_mining()
        # Separate output for the second attempt; restore known geometry hash.
        geometry["source_to_reference"][0][2] = 2.0
        write_json(self.alignment / "geometry.json", geometry)
        Image.fromarray(np.zeros((24, 24, 3), np.uint8)).save(self.dataset / "2.png")
        with self.assertRaisesRegex(ValueError, "Image hash mismatch"):
            mine_transition_candidates(self.dataset, self.alignment, self.root / "image-tamper",
                                       window_sizes=(8,), min_component_pixels=2)

    def test_rejected_alignment_is_reported_and_not_mined(self):
        self.pair_record["quality_gate"]["passed"] = False
        finish_record(self.alignment, self.pair_record, "completed_rejected")
        result = self.run_mining()
        self.assertEqual(result["candidate_count"], 0)
        self.assertEqual(result["pair_status_counts"], {"rejected_alignment": 1, "missing_alignment": 1})

    def test_negative_shift_honors_expanded_canvas_offset(self):
        from facade_change.geometry import expanded_canvas
        canvas = expanded_canvas((24, 24), (24, 24), [[1, 0, -2], [0, 1, 0], [0, 0, 1]])
        write_json(self.alignment / "geometry.json", canvas.as_dict())
        finish_record(self.alignment, self.pair_record, "completed_needs_review")
        result = self.run_mining()
        self.assertGreater(result["candidate_count"], 0)
        rows = read_json(self.root / "result/candidates.json")["candidates"]
        for row in rows:
            x = row["bbox_canvas_xyxy"][0]
            self.assertAlmostEqual(row["reference_to_crop"][0][2], 2 - x)
            self.assertAlmostEqual(row["source_to_crop"][0][2], -x)

    def test_batch_reuses_child_provenance_and_rejects_modified_child_record(self):
        import shutil
        batch = self.root / "batch"
        child = batch / "sift/pair-1-2"
        shutil.copytree(self.alignment, child)
        write_json(batch / "results.json", [{"pair_id": "1-2", "method": "sift",
                                             "path": "sift/pair-1-2", "status": "passed"}])
        finish_record(batch, {"kind": "alignment_and_data_batch"}, "completed_needs_review")
        result = mine_transition_candidates(self.dataset, batch, self.root / "batch-result",
                                             window_sizes=(8,), min_component_pixels=2)
        self.assertGreater(result["candidate_count"], 0)
        changed = read_json(child / "run.json")
        changed["observations"]["reference"]["year"] = 1900
        write_json(child / "run.json", changed)
        with self.assertRaisesRegex(ValueError, "child provenance hash mismatch"):
            mine_transition_candidates(self.dataset, batch, self.root / "batch-tampered",
                                       window_sizes=(8,), min_component_pixels=2)

    def test_directed_spalling_to_repair_is_not_reversed(self):
        pair = ({"id": 2, "name": "SPALLING", "channel": 1},
                {"id": 3, "name": "REPAIRS", "channel": 2})
        spall = np.full((8, 8), 2, np.uint16)
        repair = np.full((8, 8), 4, np.uint16)
        common = np.ones((8, 8), bool)
        self.assertEqual(len(_transition_windows(spall, repair, common, [pair], (8,), .5, .5, 1)), 1)
        self.assertFalse(_transition_windows(repair, spall, common, [pair], (8,), .5, .5, 1))


if __name__ == "__main__":
    unittest.main()
