import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from facade_change.hypothesis_dataset import prepare_hypothesis_dataset
from facade_change.io import finish_record, read_json, run_record, sha256, write_json


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


if __name__ == "__main__":
    unittest.main()
