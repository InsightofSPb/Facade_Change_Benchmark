import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from facade_change.hypothesis_benchmark import (
    THRESHOLDS, _building_macro, _calibrate, _case_metrics, _counts, run_hypothesis_benchmark,
)
from facade_change.io import finish_record, read_json, sha256, write_json


class BenchmarkMetricTests(unittest.TestCase):
    def test_strict_grid_counts_match_prediction_at_float32_boundary(self):
        middle = np.float32(.2)
        scores = np.array([middle, np.nextafter(middle, np.float32(np.inf)),
                           np.nextafter(middle, np.float32(-np.inf)), np.nan], np.float32)
        labels = np.array([1, 0, 1, 255], np.uint8)
        tp, fp, fn, tn = _counts(scores, labels, THRESHOLDS)
        for i, threshold in enumerate(THRESHOLDS):
            prediction = scores > threshold
            self.assertEqual((int(tp[i]), int(fp[i]), int(fn[i]), int(tn[i])),
                             (int((prediction & (labels == 1)).sum()), int((prediction & (labels == 0)).sum()),
                              int((~prediction & (labels == 1)).sum()), int((~prediction & (labels == 0)).sum())))

    def test_macro_weights_buildings_equally_after_within_building_case_means(self):
        rows = [{"building_id": "many", "h0_pixel_fpr": 1., "fp": 10000, "tn": 0} for _ in range(10)]
        rows.append({"building_id": "one", "h0_pixel_fpr": 0., "fp": 0, "tn": 1})
        result = _building_macro(rows)
        self.assertEqual(result["means"]["h0_pixel_fpr"], .5)
        self.assertEqual(result["metric_building_counts"]["h0_pixel_fpr"], 2)
        self.assertNotEqual(result["means"]["h0_pixel_fpr"], 10 / 11)
        self.assertIsNone(result["means"]["f1"])

    def test_calibration_excludes_sham_and_selects_highest_exact_tie(self):
        positive = np.array([.6], np.float32)
        negative = np.array([.1], np.float32)
        h1_counts = _counts(np.concatenate([positive, negative]), np.array([1, 0]), THRESHOLDS)
        h0_counts = _counts(negative, np.array([0]), THRESHOLDS)
        rows = [({"building_id": "val", "hypothesis": "H1", "sham_self_paste": False}, h1_counts),
                ({"building_id": "val", "hypothesis": "H0", "sham_self_paste": False}, h0_counts)]
        chosen = _calibrate(rows)
        self.assertAlmostEqual(chosen["threshold"], .59, places=6)
        sham_counts = _counts(np.array([.99], np.float32), np.array([0]), THRESHOLDS)
        rows += [({"building_id": "extra-sham", "hypothesis": "H0", "sham_self_paste": True}, sham_counts)] * 100
        self.assertEqual(_calibrate(rows), chosen)

    def test_ignored_labels_never_become_false_positives(self):
        case = {"hypothesis": "H0"}
        metrics = _case_metrics(case, np.array([1., .1, .9], np.float32), np.array([255, 0, 255]), .2)
        self.assertEqual((metrics["fp"], metrics["tn"], metrics["ignored_pixel_count"]), (0, 1, 2))
        self.assertEqual(metrics["h0_pixel_fpr"], 0.)
        self.assertIsNone(metrics["f1"])


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required for the existing H0/H1 fixture")
class HypothesisBenchmarkTests(unittest.TestCase):
    def setUp(self):
        # Established exporter fixture is the parent; no original images/models are reread.
        from test_hypothesis_dataset import HypothesisDatasetTests
        self.fixture = HypothesisDatasetTests("test_h0_uses_reference_and_sham_is_identical_with_matching_nuisance")
        self.fixture.setUp()
        self.root = self.fixture.root
        self.dataset, _, self.index = self.fixture._export("hypotheses")

    def tearDown(self):
        self.fixture.tearDown()

    def _run(self, name="smoke", methods=None, limit=1):
        out = self.root / name
        summary = run_hypothesis_benchmark(self.dataset, out, methods=methods, max_bases_per_split=limit)
        return out, summary, read_json(out / "metrics.json")

    def _rewrite_parent(self):
        write_json(self.dataset / "index.json", self.index)
        parent = read_json(self.dataset / "run.json")
        finish_record(self.dataset, parent, "completed_needs_review")

    def _save_case_array(self, case, key, array):
        path = self.dataset / case[key]
        Image.fromarray(array).save(path)
        case["artifact_sha256"][key] = sha256(path)

    def test_end_to_end_frozen_selection_primary_vs_sham_and_standalone_gallery(self):
        before = {path: sha256(path) for path in self.dataset.rglob("*") if path.is_file()}
        out, summary, metrics = self._run()
        self.assertEqual((summary["selected_base_count"], summary["selected_case_count"], summary["scored_case_method_count"]), (3, 36, 72))
        self.assertEqual(summary["buildings_by_split"], {"train": 1, "val": 1, "test": 1})
        self.assertEqual(read_json(out / "run.json")["status"], "completed_exploratory")
        self.assertEqual((out / "split.json").read_bytes(), (self.dataset / "split.json").read_bytes())
        for method in ("rgb_diff", "ssim"):
            for part in ("train", "val", "test"):
                primary, sham = metrics["primary"][method]["by_split"][part], metrics["sham_controls"][method]["by_split"][part]
                self.assertEqual((primary["case_count"], sham["case_count"]), (9, 3))
                self.assertEqual((primary["building_count"], sham["building_count"]), (1, 1))
            thresholds = {row["threshold"] for row in metrics["cases"] if row["method"] == method}
            self.assertEqual(thresholds, {summary["thresholds"][method]})
        calibration = read_json(out / "threshold_selection.json")
        self.assertTrue(calibration["exclude_self_paste"])
        self.assertTrue(all(len(record["curve"]) == 101 for record in calibration["methods"].values()))
        text = (out / "summary.txt").read_text()
        self.assertIn("TEST rgb_diff: f1=", text)
        self.assertIn("h0_pixel_fpr=", text)
        self.assertIn("EXPLORATORY", text)
        gallery = (out / "gallery.html").read_text()
        self.assertIn("gallery_inputs/", gallery)
        self.assertNotIn("../hypotheses", gallery)
        self.assertTrue(list(out.glob("gallery_inputs/**/reference_rgb.png")))
        self.assertEqual({path: sha256(path) for path in before}, before)

    def test_scorer_receives_only_base_support_even_when_source_is_fully_occluded(self):
        from facade_change.scorers import score_change
        calls = []
        def audited(reference, source, geometric_support, method="rgb_diff"):
            self.assertEqual(geometric_support.dtype, bool)
            self.assertFalse(geometric_support[0].any())
            self.assertTrue(geometric_support[1:].all())
            calls.append(geometric_support.copy())
            return score_change(reference, source, geometric_support, method=method)
        with patch("facade_change.scorers.score_change", side_effect=audited):
            out, _, metrics = self._run(methods=["rgb_diff"])
        self.assertEqual(len(calls), 36)
        occluded = [row for row in metrics["cases"] if row["scenario_id"] == "occluded"]
        self.assertTrue(all(row["evaluated_pixel_count"] == 0 and row["fp"] == 0 for row in occluded))
        for row in occluded:
            scores = np.load(out / row["score_path"], allow_pickle=False)
            self.assertTrue(np.isfinite(scores[1:]).all())
            self.assertTrue(np.isnan(scores[0]).all())
        self.assertTrue(any(np.asarray(Image.open(out / row["prediction_path"]))[1:].any() for row in occluded))

    def test_test_oracle_mask_changes_leave_scores_and_validation_thresholds_unchanged(self):
        first_out, first_summary, first = self._run("before-oracle", methods=["rgb_diff"])
        for case in self.index["cases"]:
            if case["split"] != "test" or case["scenario_id"] != "occluded":
                continue
            with Image.open(self.dataset / case["reference_support"]) as image:
                support = np.asarray(image) == 255
            with Image.open(self.dataset / case["true_edit_mask_reference"]) as image:
                edit = np.asarray(image) == 255
            for key in ("source_support", "true_visibility", "comparable"):
                self._save_case_array(case, key, support.astype(np.uint8) * 255)
            self._save_case_array(case, "source_occlusion", np.zeros(support.shape, np.uint8))
            self._save_case_array(case, "visible_edit_mask_reference", (edit & support).astype(np.uint8) * 255)
            labels = np.full(support.shape, 255, np.uint8)
            labels[support] = edit[support].astype(np.uint8)
            self._save_case_array(case, "labels_reference", labels)
            case["comparable_fraction"] = float(support.mean())
            case["visible_edit_pixel_count"] = int(edit.sum())
            case["retained_visible_edit_fraction"] = 1. if edit.any() else None
            case["exclude_from_visible_recall"] = False
        self._rewrite_parent()
        second_out, second_summary, second = self._run("after-oracle", methods=["rgb_diff"])
        self.assertEqual(first_summary["thresholds"], second_summary["thresholds"])
        self.assertEqual({row["case_id"]: row["score_sha256"] for row in first["cases"]},
                         {row["case_id"]: row["score_sha256"] for row in second["cases"]})
        self.assertEqual(read_json(first_out / "threshold_selection.json"), read_json(second_out / "threshold_selection.json"))
        self.assertNotEqual(first_summary["primary"]["rgb_diff"]["test"], second_summary["primary"]["rgb_diff"]["test"])

    def test_case_maps_are_deterministic_and_zero_limit_selects_all_frozen_bases(self):
        _, first_summary, first = self._run("first", methods=["rgb_diff"])
        _, second_summary, second = self._run("second", methods=["rgb_diff"])
        self.assertEqual(first_summary["thresholds"], second_summary["thresholds"])
        self.assertEqual([(row["case_id"], row["score_sha256"], row["prediction_sha256"]) for row in first["cases"]],
                         [(row["case_id"], row["score_sha256"], row["prediction_sha256"]) for row in second["cases"]])
        _, full_summary, _ = self._run("all", methods=["rgb_diff"], limit=0)
        self.assertEqual((full_summary["selected_base_count"], full_summary["selected_case_count"]), (6, 72))

    def test_changed_selected_bytes_and_split_leakage_fail_provenance(self):
        selected = run_hypothesis_benchmark(self.dataset, self.root / "pick", methods=["rgb_diff"])
        selection = read_json(self.root / "pick/selection.json")
        identifier = selection["case_ids"][0]
        case = next(row for row in self.index["cases"] if row["case_id"] == identifier)
        path = self.dataset / case["source_rgb"]
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "Dataset artifact changed"):
            self._run("changed", methods=["rgb_diff"])
        self.assertEqual(read_json(self.root / "changed/run.json")["status"], "failed")
        path.write_bytes(original)
        self.index["cases"][0]["split"] = "test" if self.index["cases"][0]["split"] != "test" else "val"
        self._rewrite_parent()
        with self.assertRaisesRegex(ValueError, "Case identity disagrees"):
            self._run("leak", methods=["rgb_diff"])
        self.assertFalse((self.root / "leak").exists())

    def test_default_real_eighteen_scenario_export_scores_same_216_cases(self):
        from facade_change.hypothesis_dataset import prepare_hypothesis_dataset
        config = Path(__file__).resolve().parents[1] / "configs/2026-10-04_h0h1_controls.json"
        full_dataset = self.root / "full-hypotheses"
        prepare_hypothesis_dataset(self.fixture.parent, config, full_dataset)
        summary = run_hypothesis_benchmark(full_dataset, self.root / "full-smoke")
        self.assertEqual((summary["selected_base_count"], summary["selected_case_count"], summary["scored_case_method_count"]), (3, 216, 432))
        out = self.root / "full-smoke"
        for method in ("rgb_diff", "ssim"):
            self.assertEqual(len(list((out / "heatmaps" / method).glob("*.png"))), 36)
        metrics = read_json(out / "metrics.json")
        self.assertEqual({row["case_id"] for row in metrics["cases"] if row["method"] == "rgb_diff"},
                         {row["case_id"] for row in metrics["cases"] if row["method"] == "ssim"})


if __name__ == "__main__":
    unittest.main()
