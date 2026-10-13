import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from facade_change.semantic_training import (include_auxiliary_training, _transition_windows,
                                             _select_transition_windows, mine_transition_candidates)
from facade_change.io import finish_record, read_json, sha256, write_json, run_record


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
        from facade_change.geometry import warp_pair
        arrays = [np.array(Image.open(self.dataset / f"{i}.png")) for i in (1, 2)]
        warped = warp_pair(arrays[0][..., :3], arrays[0][..., 3] == 255,
                           arrays[1][..., :3], arrays[1][..., 3] == 255, canvas)
        for role in ("reference", "source"):
            Image.fromarray(warped[role]).save(self.alignment / f"{role}_rgb.png")
        Image.fromarray(warped["overlap"].astype(np.uint8)*255).save(self.alignment / "overlap.png")
        self.pair_record = run_record("pair_alignment", {"method": "sift"})
        self.pair_record.update(quality_gate={"passed": True, "reasons": []},
                                observations={"reference": images[0], "source": images[1]})
        finish_record(self.alignment, self.pair_record, "completed_needs_review")

    def run_mining(self, **kwargs):
        options = {"window_sizes": (8, 12), "min_component_pixels": 2, "all_class_pairs": False,
                   "event_types": ("replacement",), "boundary_tolerance_pixels": 0, "crop_mode": "windows"}
        options.update(kwargs)
        return mine_transition_candidates(self.dataset, self.alignment, self.root / "result",
                                           **options)

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
        windows = _transition_windows(reference, source, common, [tuple(categories)], (8,), .1, .01, 1,
                                      boundary_tolerance_pixels=0)
        self.assertFalse(windows)
        source[4:12, 8:12] = 2
        windows = _transition_windows(reference, source, common, [tuple(categories)], (8,), .1, .01, 1,
                                      boundary_tolerance_pixels=0)
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

    def test_all_events_export_raw_robust_masks_nullable_classes_and_cap_counts(self):
        result = self.run_mining(event_types=("replacement", "appearance", "disappearance"),
                                 all_class_pairs=True, max_windows_per_pair=3)
        rows = read_json(self.root / "result/candidates.json")["candidates"]
        self.assertEqual({row["event_type"] for row in rows}, {"replacement", "appearance", "disappearance"})
        self.assertEqual(result["candidate_version"], 2)
        self.assertEqual(result["qualifying_windows"] - result["omitted_windows"], 3)
        gallery = (self.root / "result/gallery.html").read_text(encoding="utf-8")
        self.assertIn("Area fractions are not confidence scores", gallery)
        for row in rows:
            self.assertIn(f'candidate pixels: {row["candidate_pixels"]}/{row["window_size"]**2}', gallery)
            root = self.root / "result" / row["path"]
            self.assertTrue((root / "raw_candidate_mask.png").is_file())
            self.assertTrue((root / "ignored_boundary_mask.png").is_file())
            self.assertEqual(row["raw_candidate_pixels"], row["candidate_pixels"]
                             + row["ignored_boundary_pixels"] + row["other_robust_component_pixels"])
            if row["event_type"] == "appearance":
                self.assertIsNone(row["from_class"])
                self.assertIsNone(row["from_category_id"])
            if row["event_type"] == "disappearance":
                self.assertIsNone(row["to_class"])

    def test_boolean_pixel_parameters_are_invalid(self):
        for name in ("boundary_tolerance_pixels", "min_component_pixels", "max_windows_per_pair", "context_padding"):
            with self.assertRaises(ValueError):
                self.run_mining(**{name: True})

    def test_component_export_and_review_preserve_native_rgb_geometry_and_separate_support(self):
        from facade_change.semantic_training import prepare_transition_review
        result = self.run_mining(crop_mode="component", context_padding=3)
        self.assertEqual(result["candidate_version"], 3)
        rows = read_json(self.root / "result/candidates.json")["candidates"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["selection_bbox_canvas_xyxy"], [6, 4, 18, 16])
        self.assertEqual(row["bbox_canvas_xyxy"], [3, 1, 21, 19])
        self.assertEqual(row["candidate_pixels"], row["component_pixels"])
        directory = self.root / "result" / row["path"]
        for role in ("reference", "source"):
            with Image.open(self.alignment / f"{role}_rgb.png") as native:
                expected = np.array(native.crop(row["bbox_canvas_xyxy"]))
            np.testing.assert_array_equal(np.array(Image.open(directory / f"{role}.png")), expected)
            with Image.open(directory / f"{role}_semantic.png") as overlay:
                self.assertEqual(overlay.size, (18, 18))
        geometric = np.array(Image.open(directory / "geometric_support.png")) > 0
        event = np.array(Image.open(directory / "candidate_mask.png")) > 0
        self.assertTrue(np.all(geometric[event]))
        self.assertGreater(geometric.sum(), event.sum())  # Context is not marked as changed.
        self.assertFalse(geometric[7:9, 7:9].any())  # Transparent source hole remains unsupported.
        review = prepare_transition_review([self.root / "result"], self.root / "review")
        self.assertEqual((review["real_crops"], review["real_crop_histories"]), (1, 1))
        self.assertFalse(review["annotation_files_opened"])
        page = (self.root / "review/gallery.html").read_text()
        self.assertIn("GT классов COCO", page)
        self.assertIn("история одного участка", page)
        self.assertTrue(all(category["name"] in page for category in self.categories))


@unittest.skipUnless(importlib.util.find_spec("cv2"), "alignment extra required")
class RobustTransitionWindowTests(unittest.TestCase):
    categories = [{"id": 1, "name": "ORNAMENT_INTACT", "channel": 0},
                  {"id": 2, "name": "WATER_STAIN", "channel": 1},
                  {"id": 3, "name": "CRACK", "channel": 2}]

    def windows(self, reference, source, common=None, **kwargs):
        options = {"event_types": ("replacement", "appearance", "disappearance"),
                   "boundary_tolerance_pixels": 2, "categories": self.categories}
        options.update(kwargs)
        pairs = [(a, b) for a in self.categories for b in self.categories if a != b]
        if common is None:
            common = (reference != 0) & (source != 0)
        return _transition_windows(reference, source, common, pairs, (50, 100), .5, .02, 25, **options)

    def test_arbitrary_replacement_and_independent_multilabel_appearance_disappearance(self):
        earlier = np.ones((100, 100), np.uint16)
        later = earlier.copy()
        later[25:45, 25:45] |= 2  # Stain appears while ornament remains.
        rows = self.windows(earlier, later)
        self.assertTrue(rows)
        self.assertEqual({(row["event_type"], row["from_class"], row["to_class"]) for row in rows},
                         {("appearance", None, "WATER_STAIN")})
        reverse = self.windows(later, earlier)
        self.assertEqual({row["event_type"] for row in reverse}, {"disappearance"})
        # A class switch outside the legacy pair whitelist is discovered too.
        earlier[25:45, 25:45] = 2
        later[25:45, 25:45] = 4
        pairs = [row for row in self.windows(earlier, later) if row["event_type"] == "replacement"]
        self.assertEqual({(row["from_class"], row["to_class"]) for row in pairs}, {("WATER_STAIN", "CRACK")})

    def test_unknown_pixels_do_not_become_appearance_or_disappearance(self):
        earlier = np.ones((100, 100), np.uint16)
        later = earlier.copy()
        earlier[25:45, 25:45] = 0
        later[25:45, 25:45] = 2
        self.assertFalse(self.windows(earlier, later))
        self.assertFalse(self.windows(later, earlier))

    def test_five_ten_and_scattered_mislabelled_pixels_fail_component_gates(self):
        earlier = np.ones((100, 100), np.uint16)
        for count in (5, 10):
            later = earlier.copy()
            later[20, 20:20+count] |= 2
            self.assertFalse(self.windows(earlier, later))
        later = earlier.copy()
        later[::5, ::5] |= 2  # 400 pixels, but no qualifying connected component.
        self.assertFalse(self.windows(earlier, later))

    def test_small_contour_jitter_is_suppressed_without_morphological_erosion(self):
        earlier = np.ones((100, 100), np.uint16)
        earlier[20:70, 20:70] |= 2
        for shift in (1, 2):
            later = np.ones_like(earlier)
            later[20:70, 20+shift:70+shift] |= 2
            self.assertFalse(self.windows(earlier, later))
            self.assertTrue(self.windows(earlier, later, boundary_tolerance_pixels=0))

    def test_entirely_new_one_pixel_crack_survives_if_window_area_is_sufficient(self):
        earlier = np.ones((100, 100), np.uint16)
        later = earlier.copy()
        later[:, 49] |= 4
        rows = self.windows(earlier, later)
        self.assertTrue(rows)
        self.assertEqual({row["window_size"] for row in rows}, {50})  # 50px meets2%; 100px does not meet200.
        for row in rows:
            self.assertEqual(row["event_type"], "appearance")
            self.assertEqual(row["to_class"], "CRACK")
            self.assertEqual(row["candidate_pixels"], 50)
            self.assertEqual(row["ignored_boundary_pixels"], 0)
            self.assertEqual(int(row["candidate_mask"].any(axis=0).sum()), 1)

    def test_component_crops_keep_whole_region_and_padding_does_not_change_selection(self):
        earlier = np.ones((500, 600), np.uint16)
        later = earlier.copy()
        later[140:300, 180:440] |= 2  # Larger than either legacy window.
        for padding, box in ((0, [180, 140, 440, 300]), (100, [80, 40, 540, 400])):
            rows = self.windows(earlier, later, crop_mode="component", context_padding=padding)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["bbox_canvas_xyxy"], box)
            self.assertEqual(row["candidate_pixels"], 160*260)
            self.assertEqual(row["component_pixels"], row["candidate_pixels"])
            self.assertEqual(row["selection_transition_fraction"], 1.)
            self.assertIsNone(row["window_size"])
        later = earlier.copy()
        later[5:85, 2] |= 4  # Thin crack near the image edge; context must not dilute its gate.
        rows = self.windows(earlier, later, crop_mode="component", context_padding=100)
        self.assertEqual(rows[0]["bbox_canvas_xyxy"], [0, 0, 103, 185])
        self.assertEqual(rows[0]["candidate_pixels"], 80)
        self.assertLess(rows[0]["transition_fraction"], .02)
        later[5:85, 2] = 1
        later[5:15, 2] |= 4
        self.assertFalse(self.windows(earlier, later, crop_mode="component"))

    def test_strong_patch_preserves_raw_boundary_exclusion_and_full_window_fraction(self):
        earlier = np.ones((100, 100), np.uint16)
        later = earlier.copy()
        earlier[20:60, 20:50] |= 2
        later[20:60, 20:70] |= 2
        rows = [row for row in self.windows(earlier, later) if row["event_type"] == "appearance"]
        self.assertTrue(any(row["window_size"] == 100 for row in rows))
        for row in rows:
            self.assertGreater(row["ignored_boundary_pixels"], 0)
            self.assertGreaterEqual(row["candidate_pixels"], 25)
            self.assertGreaterEqual(row["candidate_pixels"] / row["window_size"]**2, .02)
            self.assertEqual(row["raw_candidate_pixels"], row["candidate_pixels"]
                             + row["ignored_boundary_pixels"] + row["other_robust_component_pixels"])
            self.assertLess(row["candidate_pixels"], row["window_size"]**2)

    def test_cap_round_robin_retains_rare_event_class_groups_and_reports_omissions(self):
        rows = [{"event_type": "appearance", "from_class": None, "to_class": "WATER_STAIN", "rank": i}
                for i in range(12)]
        rows += [{"event_type": "replacement", "from_class": "WATER_STAIN", "to_class": "CRACK", "rank": 0},
                 {"event_type": "disappearance", "from_class": "CRACK", "to_class": None, "rank": 0}]
        selected, counts = _select_transition_windows(rows, 3)
        self.assertEqual({row["event_type"] for row in selected}, {"appearance", "replacement", "disappearance"})
        self.assertEqual(sum(row["omitted_windows"] for row in counts), 11)
        self.assertTrue(all(row["selected_windows"] == 1 for row in counts))
        rows = []
        for i in range(11):
            rows.extend([{"event_type": "appearance", "from_class": None, "to_class": f"C{i}"},
                         {"event_type": "disappearance", "from_class": f"C{i}", "to_class": None},
                         {"event_type": "replacement", "from_class": f"C{i}", "to_class": f"C{i+1}"}])
        selected, counts = _select_transition_windows(rows, 20)
        self.assertEqual({row["event_type"] for row in selected}, {"appearance", "replacement", "disappearance"})
        self.assertTrue(all(sum(row["event_type"] == kind for row in selected) >= 6
                            for kind in ("replacement", "appearance", "disappearance")))
        self.assertEqual(sum(row["omitted_windows"] for row in counts), 13)

    def test_cli_defaults_and_explicit_legacy_protocol(self):
        from facade_change.cli import parser
        base = ["mine-transitions", "--dataset", "dataset", "--alignment-run", "alignment", "--out", "out"]
        default = parser().parse_args(base)
        self.assertTrue(default.all_class_pairs)
        self.assertEqual(default.event_types, ["replacement", "appearance", "disappearance"])
        self.assertEqual((default.boundary_tolerance_pixels, default.min_component_pixels), (2, 25))
        self.assertEqual((default.crop_mode, default.context_padding), ("component", 100))
        legacy = parser().parse_args(base + ["--no-all-class-pairs", "--event-types", "replacement",
                                             "--boundary-tolerance-pixels", "0", "--min-component-pixels", "16",
                                             "--crop-mode", "windows"])
        self.assertFalse(legacy.all_class_pairs)
        self.assertEqual(legacy.crop_mode, "windows")
        self.assertEqual(legacy.event_types, ["replacement"])
        self.assertEqual((legacy.boundary_tolerance_pixels, legacy.min_component_pixels), (0, 16))


class StaticHistoryReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        images, self.raw_images = [], {}
        for split_index, split in enumerate(("val", "train", "test")):
            for offset, year in enumerate((2010, 2020, 2015, 2025), 1):
                image_id = split_index * 10 + offset
                path = self.root / f"raw-{split}-{image_id}.png"
                Image.new("RGB", (30, 18), (20+image_id, 50, 80)).save(path)
                row = {"image_id": image_id, "image_path": str(path), "sha256": sha256(path),
                       "width": 30, "height": 18, "year": year, "split": split,
                       "building_id": split + "-B", "view_id": split + "-B-front"}
                images.append(row)
                self.raw_images[split, offset] = row
                if split == "test":
                    path.unlink()  # The metadata may be read; TEST pixels must never be loaded.
        manifest = self.root / "manifest.json"
        write_json(manifest, {"images": images})
        self.identity = {"paths": {"dataset": "/unopened/dataset.json", "manifest": str(manifest)},
                         "sha256": {"dataset": "a"*64, "manifest": sha256(manifest)}}

    def source(self, name="source", split="val", building=None, version=2):
        root = self.root / name
        root.mkdir()
        building = building or split + "-B"
        rows, pairs = [], []
        for index, (year, status) in enumerate(((2020, "mined_needs_review"),
                                               (2015, "mined_needs_review"),
                                               (2025, "rejected_alignment"))):
            reference_row, source_row = self.raw_images[split, 1], self.raw_images[split, index+2]
            pair = {"pair_id": f"{reference_row['image_id']}-{source_row['image_id']}",
                    "reference_id": reference_row["image_id"], "source_id": source_row["image_id"],
                    "reference_year": 2010, "source_year": year, "split": split,
                    "building_id": building, "view_id": building + "-front", "status": status,
                    "candidate_count": int(index == 0)}
            if status == "mined_needs_review":
                child = self.root / f"{name}-alignment-{index}"
                child.mkdir()
                record = run_record("pair_alignment", {"method": "sift"})
                record["observations"] = {role: {**row, "building_id": building, "view_id": building + "-front"}
                    for role, row in (("reference", reference_row), ("source", source_row))}
                record["quality_gate"] = {"passed": True}
                for role in ("reference", "source"):
                    Image.fromarray(np.full((20, 24, 3), 90+index, np.uint8)).save(child / f"{role}_rgb.png")
                if version == 3:
                    from facade_change.geometry import expanded_canvas
                    canvas = expanded_canvas((18, 30), (18, 30),
                                             [[1, 0, -4*index], [0, 1, -2*index], [0, 0, 1]])
                    write_json(child / "geometry.json", canvas.as_dict())
                    shift_x, shift_y = canvas.reference_to_canvas[:2, 2].astype(int)
                    yy, xx = np.indices((canvas.height, canvas.width))
                    pixels = np.zeros((canvas.height, canvas.width, 3), np.uint8)
                    pixels[..., 0] = np.clip((xx-shift_x)*7, 0, 255)
                    pixels[..., 1] = np.clip((yy-shift_y)*11, 0, 255)
                    for role in ("reference", "source"):
                        Image.fromarray(pixels).save(child / f"{role}_rgb.png")
                    Image.new("L", (canvas.width, canvas.height), 255).save(child / "overlap.png")
                    pair.update(reference_to_canvas=canvas.reference_to_canvas.tolist(),
                                geometry_sha256=sha256(child / "geometry.json"))
                finish_record(child, record, "completed_needs_review")
                pair.update(alignment_run=str(child), alignment_run_sha256=sha256(child / "run.json"))
                if index == 0:
                    row = {**pair, "candidate_id": "candidate-00000", "path": "candidate-00000",
                           "event_type": "replacement", "from_class": "DELAMINATION", "to_class": "SPALLING",
                           "bbox_canvas_xyxy": [2, 3, 8, 9],
                           "image_sha256": {role: value["sha256"] for role, value in record["observations"].items()}}
                    rows.append(row)
                    (root / row["path"]).mkdir()
                    Image.fromarray(np.ones((6, 6), np.uint8)*255).save(root / row["path"] / "candidate_mask.png")
                    if version == 3:
                        row.update(bbox_reference_xyxy=[2, 3, 8, 9], context_padding=100)
                        for role in ("reference", "source"):
                            Image.new("RGB", (6, 6), "purple").save(root / row["path"] / f"{role}_semantic.png")
            pairs.append(pair)
        config = {"candidate_version": version, "splits": [split], "dataset": self.identity}
        write_json(root / "pairs.json", pairs)
        write_json(root / "candidates.json", {"candidate_version": version, "categories": [{"id": 1, "name": "DELAMINATION"}], "candidates": rows})
        finish_record(root, run_record("real_semantic_transition_candidates", config), "completed_needs_review")
        return root

    def synthetic(self):
        root = self.root / "synthetic"
        root.mkdir()
        config = {"splits": ["train", "val"], "input_sha256": self.identity["sha256"]}
        sequences, bases = [], []
        for split in ("train", "val"):
            base_id = split + "-base"
            bases.append({"base_id": base_id, "building_id": split + "-B", "view_id": split + "-B-front",
                          "split": split, "image_sha256": self.raw_images[split, 1]["sha256"]})
            for arm in ("clean", "event"):
                frames = []
                for frame in range(1, 6):
                    paths = {}
                    for kind in ("rgb", "damage_present", "new_change", "repaired"):
                        filename = f"{split}-{arm}-{frame}-{kind}.png"
                        image = Image.new("RGB" if kind == "rgb" else "L", (12, 8), 80 if kind == "rgb" else 0)
                        image.save(root / filename)
                        paths[kind] = filename
                    frames.append({"frame": frame, "paths": paths, "event": "none", "nuisance": {"kind": "shadow"}})
                sequences.append({"base_id": base_id, "building_id": split + "-B", "split": split,
                                  "arm": arm, "frames": frames})
        write_json(root / "manifest.json", {"kind": "persistent_procedural_histories", "config": config,
                                           "bases": bases, "sequences": sequences})
        finish_record(root, run_record("history_dataset", config), "completed_exploratory")
        return root

    def review(self, sources, name="review", **kwargs):
        from facade_change.semantic_training import prepare_transition_review
        return prepare_transition_review(sources, self.root / name, **kwargs)

    def test_complete_static_gallery_synthetic_pairs_gaps_and_unchanged_sources(self):
        val, train, synthetic = self.source(), self.source("train", "train"), self.synthetic()
        before = {str(p): sha256(p) for p in self.root.rglob("*") if p.is_file()}
        result = self.review([val, train], synthetic_run=synthetic)
        self.assertEqual((result["synthetic_bases"], result["synthetic_sequences"], result["real_histories"],
                          result["real_pairs"], result["input_candidates"]), (2, 4, 2, 6, 2))
        self.assertEqual(result["pair_status_counts"], {"mined_needs_review": 4, "rejected_alignment": 2})
        page = (self.root / "review/gallery.html").read_text()
        self.assertIn("H0", page)
        self.assertIn("H1", page)
        self.assertIn("ПРОБЕЛ", page)
        self.assertEqual(result["real_original_frames"], 8)
        self.assertIn("2025: ПРОБЕЛ — rejected_alignment", page)
        self.assertNotIn("исходное фото · rejected_alignment", page)
        self.assertEqual((result["real_crops"], result["real_crop_histories"]), (2, 0))
        crops = read_json(self.root / "review/histories.json")["real_crops"]
        for crop in crops:
            for panel in crop["panels"]:
                with Image.open(self.root / "review" / panel["src"]) as image:
                    self.assertEqual(image.size, (6, 6))
        exported_images = read_json(self.root / "review/histories.json")["real_images"]
        self.assertEqual({row["year"] for row in exported_images}, {2010, 2015, 2020, 2025})
        self.assertTrue(all(row["split"] in {"train", "val"} for row in exported_images))
        self.assertIn("кандидатов нет", page)
        self.assertIn("Фиолетовая", page)
        self.assertNotIn("<select", page)
        self.assertNotIn("<input", page)
        self.assertNotIn("review-completed.csv", page)
        self.assertEqual(before, {path: sha256(path) for path in before})
        exported = read_json(self.root / "review/histories.json")["real_pairs"]
        self.assertEqual([p["source_year"] for p in exported], [2015, 2020, 2025]*2)
        self.assertFalse(result["quantitative_temporal_evaluation_allowed"])
        record = read_json(self.root / "review/run.json")
        for relative, digest in record["artifact_sha256"].items():
            self.assertEqual(sha256(self.root / "review" / relative), digest)

    def test_render_preview_only_and_cli_has_no_sampling_or_verdict_flags(self):
        from facade_change.cli import parser
        source = self.source()
        result = self.review([source], preview_max_side=10)
        self.assertEqual(result["preview_max_side"], 10)
        for path in (self.root / "review/images").glob("*.jpg"):
            with Image.open(path) as image:
                self.assertLessEqual(max(image.size), 10)
        options = parser().parse_args(["review-transitions", "--candidate-runs", str(source),
                                      "--synthetic-run", "histories", "--out", "out"])
        self.assertEqual(options.preview_max_side, 0)
        self.assertFalse(hasattr(options, "limit"))
        self.assertFalse(hasattr(options, "seed"))

    def test_same_region_timeline_translates_different_canvases_and_deduplicates_boxes(self):
        source = self.source(version=3)
        document = read_json(source / "candidates.json")
        duplicate = {**document["candidates"][0], "candidate_id": "candidate-duplicate",
                     "event_type": "appearance", "from_class": None}
        document["candidates"].append(duplicate)
        pairs = read_json(source / "pairs.json")
        pairs[0]["candidate_count"] = 2
        write_json(source / "candidates.json", document)
        write_json(source / "pairs.json", pairs)
        finish_record(source, read_json(source / "run.json"), "completed_needs_review")
        result = self.review([source])
        self.assertEqual((result["input_candidates"], result["real_crops"], result["real_crop_histories"]), (2, 1, 1))
        history = read_json(self.root / "review/histories.json")["real_crop_histories"][0]
        self.assertEqual(history["bbox_reference_xyxy"], [2, 3, 8, 9])
        frames = history["frames"]
        self.assertEqual([frame["year"] for frame in frames], [2015, 2020])
        self.assertEqual([frame["bbox_canvas_xyxy"] for frame in frames], [[6, 5, 12, 11], [2, 3, 8, 9]])
        arrays = [np.array(Image.open(self.root / "review" / frame["panel"]["src"])) for frame in frames]
        np.testing.assert_array_equal(*arrays)  # Same anchor coordinates, not same canvas coordinates.

    def test_component_review_rejects_wrong_reference_box_and_marks_missing_coverage(self):
        source = self.source(version=3)
        document = read_json(source / "candidates.json")
        document["candidates"][0]["bbox_reference_xyxy"][0] += 1
        write_json(source / "candidates.json", document)
        finish_record(source, read_json(source / "run.json"), "completed_needs_review")
        with self.assertRaisesRegex(ValueError, "reference crop box"):
            self.review([source], "wrong-offset")
        source = self.source("no-coverage", version=3)
        child = self.root / "no-coverage-alignment-1"
        with Image.open(child / "overlap.png") as image:
            Image.new("L", image.size, 0).save(child / "overlap.png")
        finish_record(child, read_json(child / "run.json"), "completed_needs_review")
        pairs = read_json(source / "pairs.json")
        pairs[1]["alignment_run_sha256"] = sha256(child / "run.json")
        write_json(source / "pairs.json", pairs)
        finish_record(source, read_json(source / "run.json"), "completed_needs_review")
        self.review([source], "no-coverage-gallery")
        frames = read_json(self.root / "no-coverage-gallery/histories.json")["real_crop_histories"][0]["frames"]
        self.assertEqual(frames[0]["status"], "insufficient_geometric_support")
        self.assertNotIn("panel", frames[0])
        self.assertEqual(frames[1]["status"], "aligned_crop_preview")

    def test_reject_test_crosssplit_and_mismatched_dataset(self):
        with self.assertRaisesRegex(ValueError, "TRAIN/VAL"):
            self.review([self.source("test", "test")])
        val = self.source("val")
        train = self.source("train", "train", building=" VAL-b ")
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.review([val, train], "leaked")
        synthetic = self.synthetic()
        record = read_json(synthetic / "run.json")
        record["config"]["input_sha256"] = {"wrong": "dataset"}
        finish_record(synthetic, run_record("history_dataset", record["config"]), "completed_exploratory")
        with self.assertRaisesRegex(ValueError, "same TRAIN/VAL dataset"):
            self.review([val], "different", synthetic_run=synthetic)

    def test_tampered_alignment_rgb_mask_manifest_and_missing_sources_fail(self):
        for index, relative in enumerate(("candidate-00000/candidate_mask.png", "candidates.json", "pairs.json")):
            source = self.source(f"source{index}")
            (source / relative).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.review([source], f"bad{index}")
        source = self.source("alignment-tamper")
        (self.root / "alignment-tamper-alignment-0/source_rgb.png").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.review([source], "badalignment")

    def test_symlinks_and_candidate_traversal_are_rejected(self):
        source = self.source()
        mask = source / "candidate-00000/candidate_mask.png"
        mask.unlink()
        mask.symlink_to(self.root / "nonexistent.png")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.review([source])
        source = self.source("traversal")
        doc = read_json(source / "candidates.json")
        doc["candidates"][0]["path"] = "../outside"
        write_json(source / "candidates.json", doc)
        finish_record(source, read_json(source / "run.json"), "completed_needs_review")
        with self.assertRaisesRegex(ValueError, "leaves"):
            self.review([source], "badtraversal")

    def test_missing_or_modified_original_in_rejected_pair_is_not_silently_omitted(self):
        source = self.source()
        missing = Path(self.raw_images["val", 4]["image_path"])
        missing.unlink()
        with self.assertRaisesRegex(ValueError, "Required original image is unavailable"):
            self.review([source])
        Image.new("RGB", (30, 18), "red").save(missing)
        with self.assertRaisesRegex(ValueError, "Original image hash mismatch"):
            self.review([source], "modified")

    def test_synthetic_incomplete_chain_and_candidate_count_mismatch_fail(self):
        source, synthetic = self.source(), self.synthetic()
        doc = read_json(synthetic / "manifest.json")
        doc["sequences"][0]["frames"].pop()
        write_json(synthetic / "manifest.json", doc)
        finish_record(synthetic, read_json(synthetic / "run.json"), "completed_exploratory")
        with self.assertRaisesRegex(ValueError, "five-frame"):
            self.review([source], synthetic_run=synthetic)
        pairs = read_json(source / "pairs.json")
        pairs[0]["candidate_count"] = 2
        write_json(source / "pairs.json", pairs)
        finish_record(source, read_json(source / "run.json"), "completed_needs_review")
        with self.assertRaisesRegex(ValueError, "candidate count"):
            self.review([source], "bad-count")


if __name__ == "__main__":
    unittest.main()
