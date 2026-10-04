import importlib.util
import contextlib
import io
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

    def _run(self, name="smoke", methods=None, limit=1, **options):
        out = self.root / name
        summary = run_hypothesis_benchmark(self.dataset, out, methods=methods, max_bases_per_split=limit, **options)
        return out, summary, read_json(out / "metrics.json")

    def _rewrite_parent(self):
        write_json(self.dataset / "index.json", self.index)
        parent = read_json(self.dataset / "run.json")
        finish_record(self.dataset, parent, "completed_needs_review")

    def _save_case_array(self, case, key, array):
        path = self.dataset / case[key]
        Image.fromarray(array).save(path)
        case["artifact_sha256"][key] = sha256(path)

    def test_quick_ten_distinct_buildings_calibrate_before_test_and_save_live_metrics(self):
        for part, count, anchor in (("val", 2, 90), ("test", 6, 150)):
            for i in range(count):
                building = f"extra_{part}_{i}"
                self.fixture.split["building_assignments"][building] = part
                self.fixture._add_crop(building, part, "view", anchor + i, "only")
        write_json(self.fixture.parent / "split.json", self.fixture.split)
        self.fixture._finish_parent()
        dataset, _, _ = self.fixture._export("larger-hypotheses")
        from facade_change.hypothesis_benchmark import _case_inputs
        out = self.root / "quick-ten"
        phase_order = []

        def checked_inputs(root, parent, base, case, checked):
            phase_order.append(case["split"])
            if case["split"] == "test":
                self.assertTrue((out / "threshold_selection.json").is_file())
                choice = read_json(out / "threshold_selection.json")
                self.assertEqual(choice["methods"]["rgb_diff"]["h1_building_count"], 3)
            return _case_inputs(root, parent, base, case, checked)

        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch("facade_change.benchmark_progress._load_tqdm", return_value=None), \
                patch("facade_change.hypothesis_benchmark._case_inputs", side_effect=checked_inputs):
            summary = run_hypothesis_benchmark(dataset, out, methods=["rgb_diff"], quick_bases=10, selection_seed=42)
        self.assertEqual((summary["selected_base_count"], summary["selected_case_count"], summary["scored_case_method_count"]), (10, 60, 60))
        self.assertEqual(summary["buildings_by_split"], {"train": 0, "val": 3, "test": 7})
        selection = read_json(out / "selection.json")
        self.assertEqual(len({base["building_id"] for base in selection["bases"]}), 10)
        self.assertEqual(selection["quick_selection"]["selected_case_count"], 60)
        self.assertEqual(set(selection["quick_selection"]["states"]), {"unchanged", "crack", "paint_patch"})
        live = read_json(out / "live_metrics.json")
        self.assertTrue(live["thresholds_frozen"])
        self.assertEqual(live["thresholds"], summary["thresholds"])
        self.assertEqual(live["completed_base_count"], 10)
        self.assertEqual(live["primary"]["rgb_diff"]["by_split"]["test"]["means"], summary["primary"]["rgb_diff"]["test"]["means"])
        self.assertIn("предварительные пороги", output.getvalue())
        self.assertIn("Накопленный TEST", output.getvalue())
        self.assertEqual((out / "progress.txt").read_text().count("Каждая выбранная пара:"), 10)

    def test_quick_interruption_preserves_completed_validation_and_frozen_threshold(self):
        from facade_change.hypothesis_benchmark import _case_inputs
        out = self.root / "quick-interrupted"

        def interrupted_inputs(root, parent, base, case, checked):
            if case["split"] == "test":
                self.assertTrue((out / "threshold_selection.json").is_file())
                raise KeyboardInterrupt("stop before test")
            return _case_inputs(root, parent, base, case, checked)

        with contextlib.redirect_stdout(io.StringIO()), patch("facade_change.benchmark_progress._load_tqdm", return_value=None), \
                patch("facade_change.hypothesis_benchmark._case_inputs", side_effect=interrupted_inputs):
            with self.assertRaises(KeyboardInterrupt):
                run_hypothesis_benchmark(self.dataset, out, methods=["rgb_diff"], quick_bases=10)
        self.assertEqual(read_json(out / "run.json")["status"], "interrupted")
        live = read_json(out / "live_metrics.json")
        self.assertEqual(live["completed_base_count"], 1)
        self.assertTrue(live["thresholds_frozen"])
        self.assertEqual(len(live["cases"]), 6)
        self.assertTrue((out / "progress.txt").is_file())

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("zstandard"), "PyTorch/zstandard required")
    def test_quick_all_eight_methods_use_existing_checkpoints_and_identical_cases(self):
        import importlib
        import torch
        msdzip = importlib.import_module("facade_change.2026-10-04_msdzip_h0")
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fit = msdzip.train_msdzip_h0(self.dataset, self.root / "quick-fit", epochs=1,
                    max_train_bytes=32, max_val_bytes=16, model_batch_size=4, timesteps=2,
                    hidden_dim=4, ffn_dim=8, vocab_dim=2, window_groups=2)
                checkpoints = {rep: self.root / "quick-fit" / fit["results"][rep]["checkpoint_path"] for rep in ("abs", "mod256")}
                before = {rep: sha256(path) for rep, path in checkpoints.items()}
                methods = ["rgb_diff", "ssim", "zstd_abs", "zstd_mod256", "lzma_abs", "lzma_mod256", "msdzip_abs", "msdzip_mod256"]
                with patch("facade_change.benchmark_progress._load_tqdm", return_value=None):
                    out, summary, metrics = self._run("quick-eight", methods=methods, quick_bases=10,
                        msdzip_abs_checkpoint=checkpoints["abs"], msdzip_mod256_checkpoint=checkpoints["mod256"],
                        compression_tile_size=16, compression_stride=8)
            self.assertEqual(summary["selected_case_count"], 12)
            self.assertEqual(summary["scored_case_method_count"], 96)
            expected = set(read_json(out / "selection.json")["case_ids"])
            for method in methods:
                rows = [row for row in metrics["cases"] if row["method"] == method]
                self.assertEqual({row["case_id"] for row in rows}, expected)
                self.assertEqual({row["threshold"] for row in rows}, {summary["thresholds"][method]})
            self.assertEqual({rep: sha256(path) for rep, path in checkpoints.items()}, before)
            self.assertEqual(read_json(out / "live_metrics.json")["completed_case_method_count"], 96)
        finally:
            torch.set_num_threads(old_threads)

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

    @unittest.skipUnless(importlib.util.find_spec("zstandard"), "Optional zstandard codec required")
    def test_compression_uses_existing_protocol_and_preserves_native_bpb(self):
        methods = ["rgb_diff", "ssim", "zstd_abs", "zstd_mod256", "lzma_abs", "lzma_mod256"]
        out, summary, metrics = self._run("compression", methods=methods, compression_tile_size=16, compression_stride=8)
        self.assertEqual(summary["scored_case_method_count"], 36 * len(methods))
        selection = read_json(out / "threshold_selection.json")
        self.assertEqual(set(selection["methods"]), set(methods))
        record = read_json(out / "run.json")
        for row in metrics["cases"]:
            if row["method"] in {"rgb_diff", "ssim"}:
                self.assertIsNone(row["native_bpb_path"])
                continue
            raw = np.load(out / row["native_bpb_path"], allow_pickle=False)
            scores = np.load(out / row["score_path"], allow_pickle=False)
            self.assertEqual(raw.dtype, np.float32)
            self.assertTrue(np.isnan(raw[0]).all())
            np.testing.assert_allclose(scores[1:], -np.expm1(-raw[1:] / 8), rtol=1e-6, atol=1e-7)
            self.assertGreater(row["native_bpb_supported_mean"], 0)
            self.assertEqual(record["config"]["scorers"][row["method"]]["tile_size"], 16)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Optional PyTorch predictor required")
    def test_original_msdzip_checkpoint_runs_through_shared_evaluator(self):
        import importlib
        import torch
        msdzip = importlib.import_module("facade_change.2026-10-04_msdzip_h0")
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            fit = self.root / "original-fit"
            result = msdzip.train_msdzip_h0(
                self.dataset, fit, representations=["abs"], epochs=1,
                max_train_bytes=32, max_val_bytes=16, model_batch_size=32,
                timesteps=2, vocab_dim=2, hidden_dim=4, ffn_dim=8, window_groups=16,
                max_bases_per_split=1,
            )
            checkpoint = fit / result["results"]["abs"]["checkpoint_path"]
            out, summary, metrics = self._run("original-evaluation", methods=["msdzip_abs"],
                                             msdzip_abs_checkpoint=checkpoint)
            self.assertEqual(summary["scored_case_method_count"], 36)
            metadata = read_json(out / "run.json")["config"]["scorers"]["msdzip_abs"]
            self.assertTrue(metadata["checkpoint_dataset_verified"])
            self.assertEqual(metadata["source_sha256"], msdzip.SOURCE_SHA256)
            self.assertEqual(len({row["threshold"] for row in metrics["cases"]}), 1)
            self.assertTrue(all((out / row["native_bpb_path"]).is_file() for row in metrics["cases"]))
        finally:
            torch.set_num_threads(previous_threads)

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
