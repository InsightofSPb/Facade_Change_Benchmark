"""Leave-facade-out calibration on authenticated, saved real-data predictions."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_real_gt_metrics import hashes, metrics, saved_map_fixture, sha


from scripts.real_eval import real_gt_calibration as calibration

SCRIPT = Path(calibration.__file__)


def metric_fixture(directory, unknown_building=None):
    root, snapshot, review, adapter, frames, payloads = saved_map_fixture(directory)
    if unknown_building:
        for (sequence, _), frame in frames.items():
            if sequence == f"real-{unknown_building.lower()}-x0-y0":
                frame["packed"][:] = 0
    # A history gate has a genuinely smaller observed support. Matched cohorts
    # must use this support during both calibration and held-out evaluation.
    identifier = "msdzip_abs-B-history-1"
    record, arrays = payloads[identifier]
    arrays["support"] = arrays["support"].copy()
    arrays["scores"] = arrays["scores"].copy()
    arrays["support"][0, 2] = False
    arrays["scores"][0, 2] = np.nan
    path = root / record["artifact_path"]
    np.savez_compressed(path, **arrays)
    record["artifact_sha256"] = sha(path)
    (root / "units" / f"{identifier}.json").write_text(json.dumps(record))
    units_path = snapshot / "2026-10-17_source_units.json"
    units = json.loads(units_path.read_text())
    for item in units:
        if item["unit_id"] == identifier:
            item.update(record)
    units_path.write_text(json.dumps(units))
    gt_dir = directory / "gt_metrics"
    with contextlib.redirect_stdout(io.StringIO()):
        metrics.run(root, snapshot, review, gt_dir, input_adapter=adapter)
    return root, snapshot, review, gt_dir, frames, payloads


def training_row(building, frame, scores, labels=None):
    scores = np.asarray(scores, dtype=np.float32)
    if labels is None:
        labels = np.zeros(scores.shape, np.uint8)
    return dict(building_id=building, sequence_id="seq-" + building, frame=frame,
                scores=scores, labels=np.asarray(labels, dtype=np.uint8))


class RealCalibrationPureTests(unittest.TestCase):
    def test_fpr_quantile_weights_facades_then_frames_and_preserves_ties(self):
        rows = [training_row("A", 1, [.1, .1]), training_row("A", 2, [.9]),
                training_row("B", 1, [.5] * 100)]
        result = calibration.calibrate_threshold(rows, "held-out", .25)
        self.assertEqual(result["status"], "calibrated")
        self.assertEqual(result["threshold"], .5)
        self.assertAlmostEqual(result["achieved_fpr"], .25)
        self.assertEqual(set(result["train_buildings"]), {"A", "B"})
        # Strict > excludes the complete tied .5 block, not a selected fraction.
        cutoff = result["threshold"]
        fpr_a = (np.mean(rows[0]["scores"] > cutoff) + np.mean(rows[1]["scores"] > cutoff)) / 2
        fpr_b = np.mean(rows[2]["scores"] > cutoff)
        self.assertAlmostEqual((fpr_a + fpr_b) / 2, .25)
        lower = np.nextafter(np.float32(cutoff), np.float32(-np.inf))
        self.assertGreater((np.mean(rows[0]["scores"] > lower) / 2 + np.mean(rows[1]["scores"] > lower) / 2 + np.mean(rows[2]["scores"] > lower)) / 2, .25)

    def test_heldout_scores_labels_and_facade_size_cannot_change_threshold(self):
        rows = [training_row("train-a", 1, [.1, .2, .5, .9]),
                training_row("train-b", 1, [.2, .4, .6, .8]),
                training_row("held-out", 1, [.01, .99], [0, 1])]
        before = copy.deepcopy(rows)
        first = calibration.calibrate_threshold(rows, "held-out", .25)
        changed = rows[:2] + [training_row("held-out", 999, [1e9] * 10000, [1] * 10000),
                             training_row("held-out", 888, [-1e9] * 10000, [0] * 10000)]
        second = calibration.calibrate_threshold(changed, "held-out", .25)
        for key in ("status", "threshold", "achieved_fpr", "train_buildings", "train_pair_keys"):
            self.assertEqual(first[key], second[key], key)
        self.assertNotIn("held-out", first["train_buildings"])
        for actual, expected in zip(rows, before):
            np.testing.assert_array_equal(actual["scores"], expected["scores"])
            np.testing.assert_array_equal(actual["labels"], expected["labels"])

    def test_average_precision_uses_tie_blocks_and_ignores_input_order(self):
        scores = np.array([.8, .8, .1], np.float32)
        labels = np.array([1, 0, 1], np.uint8)
        self.assertAlmostEqual(calibration.average_precision(scores, labels), 7 / 12)
        order = [1, 0, 2]
        self.assertAlmostEqual(calibration.average_precision(scores[order], labels[order]), 7 / 12)
        self.assertAlmostEqual(calibration.average_precision(np.ones(3, np.float32), labels), 2 / 3)
        self.assertIsNone(calibration.average_precision(scores, np.zeros(3, np.uint8)))
        self.assertIsNone(calibration.average_precision(np.array([], np.float32), np.array([], np.uint8)))

    def test_no_training_negatives_or_no_training_data_is_unsupported(self):
        for rows in ([training_row("held-out", 1, [.1, .8], [0, 1])],
                     [training_row("train", 1, [.2, .3], [1, 1])],
                     [training_row("train", 1, [], [])]):
            with self.subTest(rows=len(rows)):
                result = calibration.calibrate_threshold(rows, "held-out", .05)
                self.assertEqual(result["status"], "unsupported")
                self.assertIsNone(result.get("threshold"))

    def test_matching_uses_exact_pair_and_pixel_intersection_before_calibration(self):
        left = dict(training_row("A", 1, [.1, .8, .4], [0, 1, 0]), shape=[1, 4],
                    pixels=np.array([0, 1, 3], np.int32), native=np.array([0, 1, 1], bool))
        right = dict(training_row("A", 1, [.7, .9, .4], [1, 0, 0]), shape=[1, 4],
                     pixels=np.array([1, 2, 3], np.int32), native=None)
        missing = dict(left, frame=2)
        before = copy.deepcopy((left, right))
        matched_left, matched_right = calibration.match_samples([left, missing], [right])
        self.assertEqual((len(matched_left), len(matched_right)), (1, 1))
        for row in (matched_left[0], matched_right[0]):
            np.testing.assert_array_equal(row["pixels"], [1, 3])
            np.testing.assert_array_equal(row["labels"], [1, 0])
        np.testing.assert_array_equal(matched_left[0]["native"], [True, True])
        self.assertEqual(matched_left[0]["support_sha256"], matched_right[0]["support_sha256"])
        first = calibration.calibrate_threshold(matched_left, "B", .05)
        second = calibration.calibrate_threshold(matched_right, "B", .05)
        self.assertEqual(first["train_pair_keys"], second["train_pair_keys"])
        self.assertEqual(first["train_support_sha256"], second["train_support_sha256"])
        for row, original in zip((left, right), before):
            for key in ("pixels", "scores", "labels"):
                np.testing.assert_array_equal(row[key], original[key])
        contradictory = dict(right, labels=np.array([0, 0, 0], np.uint8))
        with self.assertRaises(ValueError):
            calibration.match_samples([left], [contradictory])


class RealCalibrationIntegrationTests(unittest.TestCase):
    def test_saved_maps_lofo_fusions_matched_support_and_authentication(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, snapshot, review, gt, frames, payloads = metric_fixture(base)
            protected = (root, snapshot, review, gt)
            before = {path: hashes(path) for path in protected}
            family = "fusion/geoscd:frozen:previous+msdzip_abs:current_train:history"
            predictions = {}

            def capture(**values):
                if values["family"] == family and values["goal"] == .01:
                    row = values["sample"]
                    key = row["sequence_id"], row["frame"]
                    predictions.setdefault(key, {})[values["predictor"]] = values["prediction"].copy()

            out = base / "calibrated"
            with contextlib.redirect_stdout(io.StringIO()), patch("subprocess.Popen", side_effect=AssertionError("No model inference")):
                summary = calibration.run(gt, out, prediction_hook=capture)
            self.assertEqual(summary["status"], "completed")
            self.assertTrue((out / "tables.md").is_file())
            self.assertTrue((out / "calibration_audit.csv").is_file())
            self.assertTrue((out / "pr_curves.csv").is_file())
            provenance = json.loads((out / "provenance.json").read_text())
            self.assertTrue(provenance["no_model_inference"])
            self.assertTrue(provenance["unchanged_history_memory_update_policy"])
            self.assertEqual({path: hashes(path) for path in protected}, before)
            native, gate = "geoscd:frozen:previous", "msdzip_abs:current_train:history"
            self.assertEqual(len(predictions), 3, "Missing Geo pair must stay missing")
            self.assertNotIn(("real-b-x0-y0", 2), predictions)
            for masks in predictions.values():
                np.testing.assert_array_equal(masks[native + "&" + gate + "@fpr0.01"],
                                              masks[native + "@fpr0.01"] & masks[gate + "@fpr0.01"])
                np.testing.assert_array_equal(masks[native + "|" + gate + "@fpr0.01"],
                                              masks[native + "@fpr0.01"] | masks[gate + "@fpr0.01"])
            pair_rows = [row for row in summary["per_pair"] if row["family"] == family and row["goal"] == .01]
            for key in predictions:
                rows = [row for row in pair_rows if (row["sequence_id"], row["frame"]) == key]
                self.assertEqual(len({row["support_sha256"] for row in rows}), 1)
                self.assertEqual(len({row["valid_pixels"] for row in rows}), 1)
            coverage = {row["family"]: row for row in summary["coverage"]}
            self.assertEqual(coverage[family]["matched_pairs"], 3)
            self.assertEqual(coverage[family.replace("geoscd:frozen", "anychange:frozen")]["matched_pairs"], 4)
            history_family = "matched_history/msdzip_abs:current_train"
            fits = [row for row in summary["thresholds"] if row["family"] == history_family and row["heldout_building"] == "A" and row["goal"] == .01]
            self.assertEqual(len(fits), 2)
            self.assertEqual(fits[0]["train_pair_keys"], fits[1]["train_pair_keys"])
            self.assertEqual(fits[0]["train_support_sha256"], fits[1]["train_support_sha256"])
            self.assertEqual([row["train_negative_pixels"] for row in fits], [17, 17])
            matched = [row for row in summary["per_pair"] if row["family"] == history_family and row["goal"] == .01 and row["building_id"] == "B" and row["frame"] == 1]
            self.assertEqual({row["valid_pixels"] for row in matched}, {9})
            self.assertEqual(len({row["support_sha256"] for row in matched}), 1)
            for fit in summary["thresholds"]:
                self.assertNotIn(fit["heldout_building"], fit["train_buildings"])
            positive = next(row for row in summary["per_pair"] if
                            (row["family"], row["component"], row["building_id"], row["frame"], row["goal"]) ==
                            ("standalone/ssim:frozen:previous", "always_positive", "A", 1, .01))
            self.assertEqual(tuple(positive[key] for key in ("tp", "fp", "fn", "tn")), (2, 8, 0, 0))
            self.assertEqual((positive["precision"], positive["recall"], positive["fpr"]), (.2, 1, 1))
            with self.assertRaises(FileExistsError):
                calibration.run(gt, out)
            record, _ = payloads["ssim-A-previous-1"]
            path = root / record["artifact_path"]
            path.write_bytes(path.read_bytes() + b"tampered")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                calibration.Samples(gt).load(("ssim", "frozen", "previous"))

    def test_unknown_other_facade_skips_calibration_fold_without_zero_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, snapshot, review, gt, _, _ = metric_fixture(base, unknown_building="B")
            before = {path: hashes(path) for path in (root, snapshot, review, gt)}
            with contextlib.redirect_stdout(io.StringIO()), patch("subprocess.Popen", side_effect=AssertionError("No model inference")):
                summary = calibration.run(gt, base / "unsupported")
            family = "standalone/ssim:frozen:previous"
            fits = [row for row in summary["thresholds"] if row["family"] == family]
            self.assertEqual(len(fits), 2)
            self.assertTrue(all(row["status"] == "unsupported" and row["threshold"] is None and row["heldout_building"] == "A" for row in fits))
            self.assertFalse(any(row["family"] == family for row in summary["per_pair"]))
            self.assertTrue(any(row["family"] == family and row["status"] == "skipped_evaluation" for row in summary["calibration_audit"]))
            coverage = next(row for row in summary["coverage"] if row["family"] == family)
            self.assertEqual((coverage["matched_pairs"], coverage["annotated_pairs"], coverage["buildings"]), (4, 2, ["A"]))
            self.assertEqual({path: hashes(path) for path in before}, before)


if __name__ == "__main__":
    unittest.main()
