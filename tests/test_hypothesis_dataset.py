import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from facade_change.hypothesis_dataset import prepare_history_dataset, prepare_hypothesis_dataset
from facade_change.io import finish_record, read_json, run_record, sha256, write_json


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required for procedural state rendering")
class HistoryDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        categories = [{"id": 1, "name": "CRACK"}]
        coco_images, images, crops = [], [], []
        for identifier, split in enumerate(("train", "train", "val", "train", "test"), 1):
            yy, xx = np.indices((64, 64))
            rgb = np.stack([(xx * 2 + 50 + identifier) % 255, (yy * 2 + 90) % 255,
                            np.full_like(xx, 130 + identifier), np.full_like(xx, 255)], axis=-1).astype(np.uint8)
            rgb[:2, :, 3] = 0
            path = self.dataset / f"photo-{identifier}.png"
            Image.fromarray(rgb).save(path)
            building = None if identifier == 4 else f"building-{identifier}"
            view = None if identifier == 4 else f"view-{identifier}"
            image = {"image_id": identifier, "file_name": path.name, "image_path": str(path),
                     "sha256": sha256(path), "width": 64, "height": 64, "building_id": building,
                     "view_id": view, "metadata_status": "unknown" if identifier == 4 else "reviewed",
                     "split": split}
            images.append(image)
            coco_images.append({"id": identifier, "file_name": path.name, "width": 64, "height": 64})
            crops.append({"crop_id": f"{identifier}-crop", "image_id": identifier, "image_sha256": image["sha256"],
                          "building_id": building, "view_id": view, "split": split, "x": 0, "y": 0,
                          "width": 64, "height": 64, "pad_right": 0, "pad_bottom": 0})
        self.images, self.crops = images, crops
        write_json(self.dataset / "coco.json", {"images": coco_images, "categories": categories,
                                               "annotations": [{"id": 99, "image_id": 5, "category_id": 999,
                                                                "segmentation": None}]})
        write_json(self.dataset / "manifest.json", {"images": images, "categories": categories})
        write_json(self.dataset / "split.json", {"building_assignments": {
            row["building_id"]: row["split"] for row in images if row["building_id"]}})
        (self.dataset / "crops.jsonl").write_text("".join(json.dumps(crop) + "\n" for crop in crops))
        write_json(self.dataset / "dataset.json", {"kind": "semantic_crop_index", "crop_size": 64,
                                                  "source_coco": "coco.json", "manifest": "manifest.json",
                                                  "crops": "crops.jsonl", "split": "split.json",
                                                  "source_coco_sha256": sha256(self.dataset / "coco.json"),
                                                  "categories": categories})

    def prepare(self, name="history", **kwargs):
        out = self.root / name
        # Neither TEST nor unidentified TRAIN may be decoded by selection.
        (self.dataset / "photo-4.png").unlink(missing_ok=True)
        (self.dataset / "photo-5.png").unlink(missing_ok=True)
        with patch("facade_change.segmentation.data.PreparedData._rasterize", side_effect=AssertionError("labels decoded")):
            summary = prepare_history_dataset(self.dataset, out, **kwargs)
        return out, summary, read_json(out / "manifest.json")

    @staticmethod
    def array(out, frame, name):
        with Image.open(out / frame["paths"][name]) as image:
            return np.array(image)

    def test_clean_controls_persistence_contexts_and_native_support(self):
        out, summary, manifest = self.prepare()
        self.assertEqual(summary["sequences"], 6)
        self.assertEqual(summary["event_types"], {"growth": 1, "partial_repair": 1, "delamination": 1})
        self.assertEqual(summary["buildings_by_split"], {"train": 2, "val": 1})
        self.assertFalse(summary["test_rgb_loaded"])
        self.assertFalse(summary["semantic_annotations_rasterized"])
        for base in manifest["bases"]:
            clean, event = [sequence for sequence in manifest["sequences"] if sequence["base_id"] == base["base_id"]]
            for index, (control_frame, frame) in enumerate(zip(clean["frames"], event["frames"])):
                self.assertEqual(control_frame["nuisance"], frame["nuisance"])
                np.testing.assert_array_equal(self.array(out, control_frame, "visibility"),
                                              self.array(out, frame, "visibility"))
                self.assertEqual(control_frame["event_pixels"], 0)
                self.assertNotEqual(frame["nuisance"]["kind"], "blur")
                if index in (0, 1, 3):
                    self.assertEqual(frame["event_pixels"], 0)
                else:
                    self.assertGreater(frame["event_pixels"], 0)
                if index:
                    self.assertEqual(frame["contexts"], {"earliest": [1], "previous": [index],
                                                         "history": list(range(1, index + 1))})
                    self.assertTrue(all(value < frame["frame"] for value in frame["contexts"]["history"]))
                    comparable = self.array(out, frame, "comparable") > 0
                    labels = self.array(out, frame, "event_labels")
                    change = self.array(out, frame, "new_change") > 0
                    self.assertTrue(np.all(labels[~comparable] == 255))
                    np.testing.assert_array_equal(labels[comparable], change[comparable].astype(np.uint8))
                    self.assertFalse(comparable[:2].any())
            np.testing.assert_array_equal(self.array(out, event["frames"][2], "latent_rgb"),
                                          self.array(out, event["frames"][3], "latent_rgb"))
        record = read_json(out / "run.json")
        self.assertEqual(record["status"], "completed_exploratory")
        for path, expected in record["artifact_sha256"].items():
            self.assertEqual(sha256(out / path), expected)
        self.assertTrue((out / "gallery.html").is_file())

    def test_reproducible_without_test_access_and_without_overwrite(self):
        _, _, first = self.prepare("first")
        self.crops.reverse()
        (self.dataset / "crops.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.crops))
        _, _, second = self.prepare("second")
        self.assertEqual(first["sequences"], second["sequences"])
        with self.assertRaises(FileExistsError):
            self.prepare("second")

    def test_hidden_event_is_preserved_in_state_truth_but_ignored_in_labels(self):
        from facade_change.augmentations import apply_nuisance

        def occlude_onset(rgb, support, spec):
            if spec["id"] == "frame_3":
                spec = {"id": spec["id"], "kind": "occlusion", "rectangle": [0., 0., 1., 1.]}
            return apply_nuisance(rgb, support, spec)

        with patch("facade_change.augmentations.apply_nuisance", side_effect=occlude_onset):
            out, _, manifest = self.prepare()
        for sequence in manifest["sequences"]:
            if sequence["arm"] == "event":
                onset = sequence["frames"][2]
                self.assertGreater(onset["event_pixels"], 0)
                self.assertEqual(onset["visible_event_pixels"], 0)
                self.assertTrue(np.all(self.array(out, onset, "event_labels") == 255))
                self.assertTrue(np.all(self.array(out, sequence["frames"][3], "event_labels") == 255))

    def test_explicit_test_and_tampered_selected_image_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "train/val"):
            self.prepare(splits=("test",))
        (self.dataset / "photo-1.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source image hash"):
            self.prepare("changed")
        self.assertEqual(read_json(self.root / "changed/run.json")["status"], "failed")


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required for procedural state rendering")
class HypothesisDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.parent = self.root / "crops"
        self.parent.mkdir()
        self.rows = []
        for building, split, offset in (("b_train", "train", 10), ("b_val", "val", 40), ("b_test", "test", 70)):
            self._add_crop(building, split, "view_a", offset, "a")
            self._add_crop(building, split, "view_a", offset, "b")
            self._add_crop(building, split, "view_b", offset + 1, "c")
        # Same anchor and crop through a later temporal pair; it is not a new sample.
        self._add_crop("b_train", "train", "view_a", 10, "a", duplicate=True)
        self._add_crop("b_train", "train", "view_c", 12, "low", fraction=.8)
        self.split = {"mode": "reviewed", "development_only": False,
                      "building_assignments": {"b_train": "train", "b_val": "val", "b_test": "test"},
                      "gold_cohort": [{"sha256": "immutable-gold-membership"}]}
        write_json(self.parent / "split.json", self.split)
        self.index_name = "2026-10-04_dataset_index.json"
        self.parent_summary = {"selected_pair_count": 210, "exported_pair_count": 179,
                               "excluded_pair_count": 31, "crop_count": len(self.rows),
                               "index_path": self.index_name, "status": "completed_with_issues"}
        self._finish_parent()
        self.config = self.root / "config.json"
        write_json(self.config, {"schema_version": 1,
                                "selection": {"splits": ["train", "val", "test"], "crops_per_building": 2,
                                              "max_buildings_per_split": 0, "min_valid_fraction": .95},
                                "states": ["unchanged", "crack", "paint_patch", "self_paste"],
                                "scenarios": [{"id": "identity", "kind": "identity"},
                                              {"id": "shadow", "kind": "shadow", "strength": .35},
                                              {"id": "occluded", "kind": "occlusion", "rectangle": [0., 0., 1., 1.]}]})

    def tearDown(self):
        self.temp.cleanup()

    def _add_crop(self, building, split, view, anchor, position, duplicate=False, fraction=None):
        pair = f"{anchor}-{anchor + (100 if duplicate else 50)}"
        name = "crop-" + position
        relative = Path("pairs") / pair / "crops" / name
        directory = self.parent / relative
        directory.mkdir(parents=True)
        yy, xx = np.indices((32, 32))
        position_key = sum(position.encode())
        rgb = np.stack([(xx * 7 + anchor + position_key) % 255,
                        (yy * 9 + anchor) % 255,
                        ((xx + yy) * 3 + anchor * 2) % 255], axis=-1).astype(np.uint8)
        support = np.ones((32, 32), bool)
        support[0] = False
        overlap = support.copy()
        if fraction is not None:
            overlap[:6] = False
        source = 255 - rgb
        arrays = {"reference_rgb": rgb, "source_rgb": source, "reference_support": support,
                  "source_support": overlap, "geometric_overlap": overlap, "not_comparable": ~overlap,
                  "absolute_residual_rgb": np.abs(source.astype(int) - rgb).astype(np.uint8)}
        for key, array in arrays.items():
            Image.fromarray(array.astype(np.uint8) * 255 if array.dtype == bool else array).save(directory / (key + ".png"))
        artifacts = {path.name: sha256(path) for path in directory.glob("*.png")}
        original_hashes = {"reference": f"anchor-{anchor}", "source": f"later-{pair}"}
        metadata = {"crop_id": name, "path": name, "bbox_canvas_xyxy": [position_key, 0, position_key + 32, 32],
                    "tile_shape": [32, 32], "valid_fraction": float(overlap.mean()),
                    "reference_to_crop": np.eye(3).tolist(), "source_to_crop": np.eye(3).tolist(),
                    "reference_id": anchor, "source_id": anchor + (100 if duplicate else 50),
                    "source_image_sha256": original_hashes, "artifact_sha256": artifacts,
                    "building_id": building, "group_id": building, "view_id": view, "split": split,
                    "metadata_status": "reviewed"}
        write_json(directory / "crop.json", metadata)
        self.rows.append({**metadata, "path": relative.as_posix(), "dataset_crop_id": pair + "/" + name,
                          "pair_id": pair, "reference_year": 2010, "source_year": 2020,
                          "reference_file_name": f"{view}_2010.png", "source_file_name": f"{view}_2020.png",
                          "original_reference_path": "/unavailable/original.png", "original_source_path": "/unavailable/later.png",
                          **{key: (relative / (key + ".png")).as_posix()
                             for key in ("reference_rgb", "source_rgb", "geometric_overlap", "not_comparable")}})

    def _finish_parent(self):
        write_json(self.parent / self.index_name, {"split": self.split, "crops": self.rows,
                                                 "categories": [{"id": 1, "name": "CRACK"}, {"id": 2, "name": "REPAIRS"}],
                                                 "excluded_pairs": [{"pair_id": f"failed-{n}", "reason": "no_passed_requested_method"}
                                                                    for n in range(31)]})
        write_json(self.parent / "summary.json", self.parent_summary)
        finish_record(self.parent, run_record("crop_dataset", {}), "completed_with_issues")

    def _export(self, name="controls"):
        out = self.root / name
        summary = prepare_hypothesis_dataset(self.parent, self.config, out)
        return out, summary, read_json(out / "index.json")

    def _mask(self, out, relative):
        with Image.open(out / relative) as image:
            return np.asarray(image)

    def test_balanced_selection_inherits_groups_deduplicates_anchors_and_preserves_parent(self):
        before = {path: path.read_bytes() for path in self.parent.rglob("*") if path.is_file()}
        with patch("facade_change.pipeline.run_pair", side_effect=AssertionError("No real alignment needed")):
            out, summary, index = self._export()
        self.assertEqual(summary["selected_crop_count"], 6)
        self.assertEqual(summary["case_count"], 72)
        self.assertEqual(summary["selected_crops_by_building"], {"b_train": 2, "b_val": 2, "b_test": 2})
        self.assertEqual(summary["cases_by_split"], {"train": 24, "val": 24, "test": 24})
        self.assertEqual(summary["cases_by_hypothesis"], {"H0": 36, "H1": 36})
        selected = read_json(out / "selection.json")["selected"]
        for building in ("b_train", "b_val", "b_test"):
            samples = [row for row in selected if row["building_id"] == building]
            self.assertEqual({row["view_id"] for row in samples}, {"view_a", "view_b"})
            self.assertEqual(len({row["artifact_sha256"]["reference_rgb.png"] for row in samples}), 2)
        reasons = {row["reason"] for row in read_json(out / "selection.json")["excluded_crops"]}
        self.assertIn("duplicate_reference_rgb", reasons)
        self.assertIn("below_min_valid_fraction", reasons)
        self.assertEqual(summary["parent_excluded_pair_count"], 31)
        self.assertEqual(index["parent_summary"]["selected_pair_count"], 210)
        self.assertEqual(index["parent_summary"]["exported_pair_count"], 179)
        self.assertEqual(len(read_json(out / "selection.json")["parent_excluded_pairs"]), 31)
        self.assertEqual((out / "split.json").read_bytes(), (self.parent / "split.json").read_bytes())
        self.assertEqual(len(list(out.glob("bases/*/reference_rgb.png"))), 6)
        self.assertFalse(list(out.glob("cases/*/reference_rgb.png")))
        for case in index["cases"]:
            self.assertEqual(case["split"], self.split["building_assignments"][case["building_id"]])
            self.assertEqual((case["reference_year"], case["source_year"]), (2010, 2020))
            self.assertEqual(case["source_to_reference"], np.eye(3).tolist())
        with (out / "index.csv").open(newline="", encoding="utf-8") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 72)
        self.assertIn("families/crack.html", (out / "gallery.html").read_text())
        self.assertIn('loading="lazy"', (out / "families/crack.html").read_text())
        self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_h0_uses_reference_and_sham_is_identical_with_matching_nuisance(self):
        out, _, index = self._export()
        cases = {(row["base_id"], row["state"], row["scenario_id"]): row for row in index["cases"]}
        for base in index["bases"]:
            identity = cases[base["base_id"], "unchanged", "identity"]
            self.assertEqual(self._mask(out, identity["reference_rgb"]).tobytes(), self._mask(out, identity["source_rgb"]).tobytes())
            self.assertFalse(self._mask(out, identity["true_edit_mask_reference"]).any())
            for scenario in ("identity", "shadow", "occluded"):
                h0, sham = cases[base["base_id"], "unchanged", scenario], cases[base["base_id"], "self_paste", scenario]
                self.assertEqual(h0["artifact_sha256"]["source_rgb"], sham["artifact_sha256"]["source_rgb"])
                self.assertEqual(sham["hypothesis"], "H0")
                self.assertFalse(sham["independent_photo_sample"])
                self.assertTrue(sham["sham_self_paste"])
            shadow = cases[base["base_id"], "unchanged", "shadow"]
            labels = self._mask(out, shadow["labels_reference"])
            self.assertEqual(set(np.unique(labels)), {0, 255})
            self.assertTrue(self._mask(out, shadow["nuisance_mask"]).any())

    def test_full_and_visible_edit_gt_remain_distinct_for_known_occlusion(self):
        out, _, index = self._export()
        for case in index["cases"]:
            full = self._mask(out, case["true_edit_mask_reference"]) > 0
            visible = self._mask(out, case["visible_edit_mask_reference"]) > 0
            comparable = self._mask(out, case["comparable"]) > 0
            labels = self._mask(out, case["labels_reference"])
            np.testing.assert_array_equal(visible, full & comparable)
            self.assertEqual(case["full_edit_pixel_count"], int(full.sum()))
            np.testing.assert_array_equal(labels[comparable], full[comparable].astype(np.uint8))
            self.assertTrue(np.all(labels[~comparable] == 255))
            if case["scenario_id"] == "occluded":
                self.assertFalse(comparable.any())
                self.assertGreater(case["source_occlusion_fraction"], 0)
                if case["state"] in {"crack", "paint_patch"}:
                    self.assertTrue(full.any())
                    self.assertEqual(case["hypothesis"], "H1")
                    self.assertEqual(case["retained_visible_edit_fraction"], 0.)
                    self.assertTrue(case["exclude_from_visible_recall"])

    def test_all_eighteen_single_nuisance_specs_match_sham_and_preserve_state_gt(self):
        supplied = Path(__file__).resolve().parents[1] / "configs/2026-10-04_h0h1_controls.json"
        config = read_json(supplied)
        self.assertEqual(len(config["scenarios"]), 18)
        write_json(self.config, config)
        out, summary, index = self._export("all-nuisances")
        self.assertEqual(summary["case_count"], 6 * 4 * 18)
        cases = {(row["base_id"], row["state"], row["scenario_id"]): row for row in index["cases"]}
        for base in index["bases"]:
            for scenario in config["scenarios"]:
                unchanged = cases[base["base_id"], "unchanged", scenario["id"]]
                sham = cases[base["base_id"], "self_paste", scenario["id"]]
                self.assertEqual(unchanged["artifact_sha256"]["source_rgb"], sham["artifact_sha256"]["source_rgb"])
                self.assertEqual(unchanged["artifact_sha256"]["labels_reference"], sham["artifact_sha256"]["labels_reference"])
            for state in config["states"]:
                masks = {case["artifact_sha256"]["true_edit_mask_reference"] for case in index["cases"]
                         if case["base_id"] == base["base_id"] and case["state"] == state}
                self.assertEqual(len(masks), 1)
        shadow = next(case for case in index["cases"] if case["nuisance_kind"] == "shadow")
        intensity = np.load(out / shadow["nuisance_intensity"], allow_pickle=False)
        self.assertEqual(intensity.shape, (32, 32))
        self.assertGreater(float(intensity.max()), 0.)

    def test_case_bytes_are_deterministic_and_selection_ignores_index_order(self):
        out, _, first = self._export("first")
        self.rows.reverse()
        self._finish_parent()
        second_out, _, second = self._export("second")
        self.assertEqual([row["dataset_crop_id"] for row in read_json(out / "selection.json")["selected"]],
                         [row["dataset_crop_id"] for row in read_json(second_out / "selection.json")["selected"]])
        self.assertEqual({row["case_id"]: row["artifact_sha256"] for row in first["cases"]},
                         {row["case_id"]: row["artifact_sha256"] for row in second["cases"]})
        # Appending a low-overlap candidate changes the parent ledger but no eligible selection.
        self._add_crop("b_train", "train", "new_view", 20, "new-low", fraction=.8)
        self._finish_parent()
        _, _, third = self._export("third")
        self.assertEqual([row["case_id"] for row in first["cases"]], [row["case_id"] for row in third["cases"]])

    def test_changed_selected_crop_or_index_fails_provenance(self):
        selected_reference = self.parent / self.rows[2]["reference_rgb"]
        original = selected_reference.read_bytes()
        selected_reference.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "Parent artifact changed"):
            self._export("changed-photo")
        self.assertEqual(read_json(self.root / "changed-photo/run.json")["status"], "failed")
        selected_reference.write_bytes(original)
        index_path = self.parent / self.index_name
        index_path.write_bytes(index_path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Parent artifact changed"):
            self._export("changed-index")
        self.assertFalse((self.root / "changed-index").exists())

    def test_cross_split_identity_and_empty_selection_are_rejected(self):
        self.rows[0]["split"] = "test"
        self._finish_parent()
        with self.assertRaisesRegex(ValueError, "building split"):
            self._export("leakage")
        self.rows[0]["split"] = "train"
        self._finish_parent()
        config = read_json(self.config)
        config["selection"]["min_valid_fraction"] = 1.
        write_json(self.config, config)
        with self.assertRaisesRegex(ValueError, "No indexed crops"):
            self._export("empty")
        self.assertFalse((self.root / "empty").exists())

    def test_actual_crop_dataset_contract_end_to_end(self):
        # Reuse the established real SIFT synthetic batch fixture; no learned models.
        from test_crop_dataset import CropDatasetTests
        from facade_change.crop_dataset import prepare_crop_dataset

        fixture = CropDatasetTests("test_single_method_choice_does_not_silently_fallback")
        fixture.setUp()
        try:
            crop_out = fixture.root / "hypothesis-input"
            prepare_crop_dataset(fixture.batch, crop_out, tile_size=128, stride=128, min_valid_fraction=1.)
            summary = prepare_hypothesis_dataset(crop_out, self.config, fixture.root / "hypotheses")
            self.assertGreater(summary["selected_crop_count"], 0)
            self.assertEqual(summary["case_count"], summary["selected_crop_count"] * 12)
            self.assertEqual(summary["parent_excluded_pair_count"], 1)
        finally:
            fixture.tearDown()

    def _annotation_statistics(self, different_reference_support=False):
        """A hash-bound console-statistics snapshot with partial, multilabel targets."""
        from facade_change import annotation_targets
        from facade_change.crop_selection import DEFAULTS, VERSION
        # Same reference with a genuinely different late image: retain the real pair,
        # but do not generate the identical synthetic reference family twice.
        duplicate = next(r for r in self.rows if r["source_id"]-r["reference_id"] == 100)
        directory = self.parent / duplicate["path"]
        with Image.open(directory / "source_rgb.png") as image:
            rgb = np.asarray(image).copy()
        Image.fromarray((rgb.astype(int)+17).astype(np.uint8)).save(directory / "source_rgb.png")
        duplicate["artifact_sha256"]["source_rgb.png"] = sha256(directory / "source_rgb.png")
        metadata = read_json(directory / "crop.json")
        if different_reference_support:
            with Image.open(directory / "reference_support.png") as image:
                support = np.asarray(image).copy()
            support[-1, -1] = 0
            with Image.open(directory / "geometric_overlap.png") as image:
                overlap = np.asarray(image).copy()
            overlap[-1, -1] = 0
            for name, array in (("reference_support", support), ("geometric_overlap", overlap), ("not_comparable", 255-overlap)):
                Image.fromarray(array).save(directory / (name+".png"))
                duplicate["artifact_sha256"][name+".png"] = sha256(directory / (name+".png"))
            duplicate["valid_fraction"] = float((overlap > 0).mean())
            metadata["valid_fraction"] = duplicate["valid_fraction"]
        metadata["artifact_sha256"] = duplicate["artifact_sha256"]
        write_json(directory / "crop.json", metadata)
        self._finish_parent()
        config = read_json(self.config)
        config.update(include_real_pairs=True)
        config["selection"] = dict(DEFAULTS, strategy=VERSION, max_pairs=50, max_same_pair_iou=1.)
        write_json(self.config, config)
        statistics = self.root / "statistics"
        statistics.mkdir()
        crop_rows, class_rows = [], []
        for row in self.rows:
            if row["split"] == "test":
                continue
            meta = {k: row[k] for k in ("dataset_crop_id", "pair_id", "building_id", "view_id", "split",
                                        "reference_year", "source_year")}
            meta["alignment_method"] = "sift"
            row["alignment_method"] = "sift"
            stable = row["crop_id"] == "crop-b"
            known_fraction = 512/round(row["valid_fraction"]*1024)
            crop_rows.append(dict(meta, status="unknown" if stable else "changed",
                reason="target_annotations_without_comparable_support" if stable else "target_difference_on_annotated_support",
                known_fraction=known_fraction, known_pixels=512, changed_pixels=0 if stable else 64,
                uncomparable_target_pixels=32))
            for cid, category in ((1, "CRACK"), (2, "REPAIRS")):
                changed = 0 if stable else 64
                # REPAIRS absent on the entire known ROI is not a stable-present case.
                absent = cid == 2 and row["view_id"] != "view_b"
                class_rows.append(dict(meta, category_id=cid, category=category,
                    status="unknown" if stable or absent else "changed", known_pixels=512,
                    known_fraction=known_fraction, changed_pixels=0 if absent else changed,
                    change_fraction=0 if absent else changed/512,
                    before_fraction=0 if absent else .25,
                    after_fraction=0 if absent else .25+(0 if stable else 32/512),
                    area_delta=0 if absent or stable else 32/512, uncomparable_target_pixels=32))
        self._finish_parent()
        for name, rows in (("crops.csv", crop_rows), ("class_changes.csv", class_rows)):
            with (statistics / name).open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        write_json(statistics / "summary.json", dict(all_crop_pairs=len(self.rows), processed_crop_pairs=len(crop_rows),
            classes=[dict(category_id=1, category="CRACK"), dict(category_id=2, category="REPAIRS")]))
        write_json(statistics / "experiment_contract.json", dict(inputs={str(self.parent / name): sha256(self.parent / name)
            for name in ("run.json", "summary.json", self.index_name, "split.json")},
            annotation_core_sha256=sha256(annotation_targets.__file__)))
        record = run_record("crop_statistics", dict(crops=str(self.parent)))
        finish_record(statistics, record, "completed")
        return statistics, config, crop_rows, class_rows

    def test_annotation_plan_is_rgb_free_frozen_and_matches_all_generated_file_counts(self):
        statistics, config, _, _ = self._annotation_statistics()
        before = {p: p.read_bytes() for p in self.parent.rglob("*") if p.is_file()}
        with patch("facade_change.hypothesis_dataset.load_rgb", side_effect=AssertionError("Plan decoded RGB")), \
                patch("PIL.Image.open", side_effect=AssertionError("Plan opened an image")):
            report = prepare_hypothesis_dataset(self.parent, self.config, self.root / "plan",
                                                statistics_run=statistics, plan_only=True)
        self.assertFalse(list((self.root / "plan").rglob("*.png")))
        self.assertEqual(report["selected_building_count"], 2)
        self.assertGreater(report["selected_pair_count"], report["synthetic_base_count"])
        self.assertEqual(report["selected_view_count"], report["eligible_view_count"])
        self.assertEqual({row["split"] for row in read_json(self.root / "plan/selection.json")["selected"]}, {"train", "val"})
        with patch("facade_change.crop_selection.select_pairs", side_effect=AssertionError("Frozen plan was reranked")):
            result = prepare_hypothesis_dataset(self.parent, self.config, self.root / "combined",
                                                selection_plan=self.root / "plan")
        out, budget = self.root / "combined", report["generation_budget"]
        pngs = list(out.rglob("*.png"))
        rgb_count = 0
        for path in pngs:
            with Image.open(path) as image:
                rgb_count += image.mode == "RGB"
        self.assertEqual(rgb_count, budget["rgb_png"])
        self.assertEqual(len(pngs)-rgb_count, budget["auxiliary_png"])
        self.assertEqual(len(list(out.rglob("*.npy"))), budget["intensity_npy"])
        self.assertEqual(result["case_count"], budget["synthetic_comparisons"])
        real = read_json(out / "real_pairs/index.json")
        self.assertEqual(len(real["cases"]), budget["real_comparisons"])
        self.assertTrue(any(c["original_pair_status"] == "unknown" for c in real["cases"]))
        by_id = {r["dataset_crop_id"]: r for r in self.rows}
        for case in real["cases"]:
            row = by_id[case["parent_dataset_crop_id"]]
            self.assertFalse(case["natural_change_labels_created"])
            self.assertNotIn("hypothesis", case)
            self.assertNotIn("labels_reference", case)
            if case["scenario_id"] == "identity":
                np.testing.assert_array_equal(self._mask(out, case["source_rgb"]), self._mask(self.parent, row["source_rgb"]))
                self.assertFalse(np.array_equal(self._mask(out, case["source_rgb"]), self._mask(out, case["reference_rgb"])))
            if case["scenario_id"] == "occluded":
                self.assertFalse(self._mask(out, case["geometric_visible_support"]).any())
        for path, expected in read_json(out / "run.json")["artifact_sha256"].items():
            self.assertEqual(sha256(out / path), expected)
        self.assertEqual(before, {p: p.read_bytes() for p in self.parent.rglob("*") if p.is_file()})

    def test_annotation_selection_is_order_independent_and_does_not_label_absent_classes_stable(self):
        from facade_change.crop_selection import select_pairs
        _, config, crop_rows, class_rows = self._annotation_statistics()
        categories = read_json(self.parent / self.index_name)["categories"]
        first = select_pairs(self.rows, self.split, categories, crop_rows, class_rows, config["selection"])
        second = select_pairs(list(reversed(self.rows)), self.split, categories, list(reversed(crop_rows)),
                              list(reversed(class_rows)), config["selection"])
        self.assertEqual(first, second)
        self.assertTrue(any(item["role"] == "stable_present_on_common_support" for detail in first["selected_details"].values() for item in detail["evidence"]))
        self.assertFalse(any(item["category_id"] == 2 and item["role"] == "stable_present_on_common_support"
                            for detail in first["selected_details"].values() for item in detail["evidence"]))
        with self.assertRaisesRegex(ValueError, "retain every"):
            select_pairs(self.rows, self.split, categories, crop_rows, class_rows, dict(config["selection"], max_pairs=1))

    def test_annotation_statistics_tampering_and_different_crop_snapshot_fail_before_rgb(self):
        statistics, _, _, _ = self._annotation_statistics()
        original = (statistics / "class_changes.csv").read_bytes()
        (statistics / "class_changes.csv").write_bytes(original+b"\n")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            prepare_hypothesis_dataset(self.parent, self.config, self.root / "bad", statistics_run=statistics, plan_only=True)
        self.assertFalse((self.root / "bad").exists())
        (statistics / "class_changes.csv").write_bytes(original)
        self.rows.reverse()
        self._finish_parent()
        with self.assertRaisesRegex(ValueError, "another crop snapshot"):
            prepare_hypothesis_dataset(self.parent, self.config, self.root / "bad-index", statistics_run=statistics, plan_only=True)

    def test_annotation_frozen_plan_rejects_changed_config_and_output_inside_inputs(self):
        statistics, config, _, _ = self._annotation_statistics()
        with self.assertRaisesRegex(ValueError, "Output must be separate"):
            prepare_hypothesis_dataset(self.parent, self.config, self.parent / "bad", statistics_run=statistics, plan_only=True)
        prepare_hypothesis_dataset(self.parent, self.config, self.root / "plan", statistics_run=statistics, plan_only=True)
        config["selection"]["max_pairs"] += 1
        write_json(self.config, config)
        with self.assertRaisesRegex(ValueError, "differs"):
            prepare_hypothesis_dataset(self.parent, self.config, self.root / "bad-config", selection_plan=self.root / "plan")

    def test_annotation_round_robin_retains_thin_rare_class_and_enforces_spatial_and_building_caps(self):
        from copy import deepcopy
        from facade_change.crop_selection import select_pairs, _iou
        _, config, crop_rows, class_rows = self._annotation_statistics()
        prototype = self.rows[0]
        meta = crop_rows[0]
        class_meta = class_rows[0]
        rows, crop_stats, class_stats = [], [], []
        for pair in range(2):
            for number, x in enumerate((0, 16, 64, 80, 128)):
                row = deepcopy(prototype)
                identifier = f"p{pair}/crop-{number}"
                row.update(pair_id=f"p{pair}", dataset_crop_id=identifier, bbox_canvas_xyxy=[x, 0, x+32, 32],
                           source_image_sha256=dict(reference="original-A", source=f"original-B{pair}"),
                           artifact_sha256={k: identifier+"/"+k for k in row["artifact_sha256"]})
                rows.append(row)
                identity = dict(dataset_crop_id=identifier, pair_id=row["pair_id"])
                crop_stats.append(dict(meta, **identity))
                class_stats.append(dict(class_meta, **identity))
                if pair == 1 and number == 4:
                    n = round(row["valid_fraction"]*1024)
                    class_stats.append(dict(class_meta, **identity, category_id=2, category="REPAIRS",
                        known_pixels=n, known_fraction=1., changed_pixels=8, change_fraction=8/n,
                        before_fraction=128/n, after_fraction=136/n, area_delta=8/n))
        cfg = dict(config["selection"], max_pairs=5, max_pairs_per_building=3,
                   max_crops_per_photo_pair=2, max_same_pair_iou=.25)
        result = select_pairs(rows, self.split, [{"id": 1, "name": "CRACK"}, {"id": 2, "name": "REPAIRS"}],
                              crop_stats, class_stats, cfg)
        self.assertEqual(len(result["selected"]), 3)
        self.assertTrue(any(item["category_id"] == 2 and item["magnitude_band"] == 0
                            for details in result["selected_details"].values() for item in details["evidence"]))
        for row in result["selected"]:
            same = [r for r in result["selected"] if r["pair_id"] == row["pair_id"]]
            self.assertLessEqual(len(same), 2)
            for other in same:
                if row != other:
                    self.assertLessEqual(_iou(row["bbox_canvas_xyxy"], other["bbox_canvas_xyxy"]), .25)
        self.assertIn("building_budget", result["report"]["excluded_reasons"])

    def test_annotation_identical_rgb_with_different_support_exports_distinct_synthetic_families(self):
        statistics, _, _, _ = self._annotation_statistics(different_reference_support=True)
        out = self.root / "different-support"
        result = prepare_hypothesis_dataset(self.parent, self.config, out, statistics_run=statistics)
        index = read_json(out / "index.json")
        self.assertEqual(len({b["base_id"] for b in index["bases"]}), result["selected_crop_count"])
        references = {}
        for case in index["cases"]:
            if case["state"] == "unchanged" and case["scenario_id"] == "identity":
                references.setdefault(case["artifact_sha256"]["reference_rgb"], set()).add(case["artifact_sha256"]["reference_support"])
        self.assertTrue(any(len(supports) == 2 for supports in references.values()))

    def test_supplied_annotation_config_runs_all_eighteen_scenarios_for_both_branches(self):
        statistics, _, _, _ = self._annotation_statistics()
        supplied = Path(__file__).resolve().parents[1] / "configs/2026-10-25_crop_controls.json"
        write_json(self.config, read_json(supplied))
        plan = prepare_hypothesis_dataset(self.parent, self.config, self.root / "full-matrix-plan",
                                         statistics_run=statistics, plan_only=True)
        out = self.root / "full-matrix"
        with patch("facade_change.segmentation.data.rasterize_annotations", side_effect=AssertionError("No new masks")):
            report = prepare_hypothesis_dataset(self.parent, self.config, out,
                                                selection_plan=self.root / "full-matrix-plan")
        self.assertEqual(report["case_count"], 72*plan["synthetic_base_count"])
        real = read_json(out / "real_pairs/index.json")
        self.assertEqual(len(real["cases"]), 18*plan["selected_pair_count"])
        self.assertEqual(len({c["scenario_id"] for c in real["cases"]}), 18)
        self.assertEqual(len(list(out.rglob("*.png"))), plan["generation_budget"]["rgb_png"]+plan["generation_budget"]["auxiliary_png"])
        self.assertEqual(len(list(out.rglob("*.npy"))), plan["generation_budget"]["intensity_npy"])


if __name__ == "__main__":
    unittest.main()
