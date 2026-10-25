"""Saved-only reports: calibration isolation, facade weighting and honest targets."""
from __future__ import annotations

import csv
import importlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from facade_change.io import sha256


reports = importlib.import_module("facade_change.research.2026-10-14_temporal_reports")


def item(identifier, scores, **changes):
    scores = np.array(scores, dtype=np.float32, ndmin=2)
    record = {"unit_id": identifier, "method": "rgb_diff", "variant": "default",
              "mode": "previous", "target": "step_event", "domain": "synthetic",
              "split": "val", "building_id": identifier, "sequence_id": identifier,
              "frame": 1, "arm": "clean", "event": "none", "nuisance": "identity",
              "hypothesis": "H0", "output_kind": "continuous", "status": "completed",
              "elapsed_seconds": .01, **changes}
    arrays = {"scores": scores, "labels": np.zeros(scores.shape, dtype=np.uint8),
              "support": np.ones(scores.shape, dtype=bool),
              "reference": np.zeros((*scores.shape, 3), dtype=np.uint8),
              "source": np.full((*scores.shape, 3), 100, dtype=np.uint8)}
    return {"record": record, "arrays": arrays}


def save(root, value):
    record = value["record"].copy()
    path = root / "maps" / (record["unit_id"] + ".npz")
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, **value["arrays"])
    record.update(artifact_path=str(path.relative_to(root)), artifact_sha256=sha256(path))
    return record


class TemporalReportsTest(unittest.TestCase):
    def test_h0_only_selection_keeps_nuisances_and_excludes_train_h1_test_real(self):
        controls = [item("clean", [0., 0.]), item("shadow", [.8, .8], nuisance="shadow")]
        ignored = [item("train", [100., 100.], split="train"),
                   item("test", [100., 100.], split="test"),
                   item("real", [100., 100.], domain="real"),
                   item("event", [100., 100.], arm="event", hypothesis="H1")]
        result = reports.calibrate_thresholds(controls + ignored)
        self.assertEqual(result["h0_unit_count"], 2)
        self.assertEqual(result["source_unit_ids"], ["clean", "shadow"])
        self.assertAlmostEqual(result["choices"][0]["threshold"], .8)
        self.assertEqual(result["choices"][0]["h0_pixel_fpr"], 0.)

    def test_facade_macro_not_pixel_pooling_and_units_equal_within_facade(self):
        # Facade A's two units have FPR 1 and 0 at t=0: mean .5.
        # Facade B's much larger unit has FPR 0: facade-macro FPR .25.
        values = [item("a1", [1.], building_id="A"),
                  item("a2", [0.], building_id="A"),
                  item("b", np.zeros(200), building_id="B")]
        result = reports.calibrate_thresholds(values, (.2, .3))
        self.assertEqual(result["choices"][0]["threshold"], 1.)
        self.assertEqual(result["choices"][1]["threshold"], 0.)
        self.assertEqual(result["choices"][1]["h0_pixel_fpr"], .25)

    def test_safe_grid_has_no_detection_threshold_for_ties_and_large_scores(self):
        value = item("control", [500., 500., 900., 900.])
        result = reports.calibrate_thresholds([value], (0., .01), max_grid=2)
        self.assertTrue(all(choice["threshold"] == 900 for choice in result["choices"]))
        self.assertTrue(all(choice["h0_pixel_fpr"] == 0 for choice in result["choices"]))

    def test_mixed_targets_cannot_calibrate_together(self):
        with self.assertRaisesRegex(ValueError, "separately"):
            reports.calibrate_thresholds([item("p", [0.]), item("e", [1.], target="earliest_state")])

    def test_rebuild_has_real_gallery_only_native_fixed_metrics_and_partial_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = item("control", [0., .1])
            changed = item("change", [.9, 0.], arm="event", hypothesis="H1")
            changed["arrays"]["labels"][0, 0] = 1
            real = item("real", [.9, 0.], domain="real", hypothesis="H1", arm="event")
            real["arrays"]["labels"][0, 0] = 1
            native = item("native", [0., 1.], method="anychange", output_kind="native_binary",
                          hypothesis="H1", arm="event")
            native["arrays"]["labels"][0, 0] = 1
            native["arrays"]["native_prediction"] = np.array([[False, True]])
            records = [save(root, value) for value in (control, changed, real, native)]
            records.append({**item("failure", [0.])["record"], "status": "failed", "error": "fixture failure"})
            summary = reports.build_reports(root, records)
            self.assertEqual(summary["completed_unit_count"], 4)
            self.assertEqual(summary["unreviewed_real_unit_count"], 1)
            self.assertEqual(summary["status_counts"], {"completed": 4, "failed": 1})
            self.assertFalse(any(row["domain"] == "real" for row in summary["metrics"]))
            native_metrics = [row for row in summary["metrics"] if row["method"] == "anychange"]
            self.assertEqual(len(native_metrics), 1)
            self.assertEqual(native_metrics[0]["recall"], 0.)
            self.assertIsNone(native_metrics[0]["fpr_target"])
            threshold = [entry for entry in summary["calibration"] if entry["method"] == "anychange"][0]
            self.assertEqual(threshold["curve"], [])
            gallery = (root / summary["gallery_path"]).read_text()
            self.assertIn("data:image/png;base64,", gallery)
            self.assertIn("Unreviewed: no temporal GT", gallery)
            self.assertIn("fixture failure", gallery)
            again = reports.build_reports(root, records)
            self.assertEqual(summary, again)
            with (root / "2026-10-14_temporal_units.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 5)

    def test_completed_corruption_and_mismatched_history_support_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = item("previous", [0., 1.], sequence_id="series")
            record = save(root, value)
            (root / record["artifact_path"]).write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                reports.build_reports(root, [record])
            record = save(root, value)
            history = item("history", [0., 1.], sequence_id="series", mode="history")
            history["arrays"]["support"][0, 1] = False
            history["arrays"]["labels"][0, 1] = 255
            history_record = save(root, history)
            with self.assertRaisesRegex(ValueError, "labels or support disagree"):
                reports.build_reports(root, [record, history_record])

    def test_missing_calibration_is_qualitative_and_test_reports_are_forbidden(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = item("train", [.5, .9], split="train")
            summary = reports.build_reports(root, [save(root, value)])
            self.assertEqual(summary["metrics"], [])
            self.assertEqual(summary["calibration"][0]["status"], "missing_calibration")
            value["record"]["split"] = "test"
            with self.assertRaisesRegex(ValueError, "excludes TEST"):
                reports.build_reports(root, [save(root, value)])

    def test_paired_history_uses_same_labels_and_intersection_of_completed_units(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = [item("previous_control", [0., .1]),
                      item("history_control", [0., .1], mode="history")]
            for mode in ("previous", "history"):
                value = item(mode, [.9 if mode == "history" else 0., 0.], mode=mode,
                             sequence_id="same-series", building_id="same-building", arm="event", hypothesis="H1")
                value["arrays"]["labels"][0, 0] = 1
                values.append(value)
            summary = reports.build_reports(root, [save(root, value) for value in values])
            self.assertEqual(len(summary["history_comparisons"]), 2)
            self.assertTrue(all(row["recall_delta"] == 1. for row in summary["history_comparisons"]))
            self.assertTrue(all(row["matched_unit_count"] == 1 for row in summary["history_comparisons"]))
            self.assertEqual(json.loads((root / "2026-10-14_temporal_summary.json").read_text()), summary)

    def test_history_initialization_is_coverage_not_failure_and_reference_caption_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            caption = ("Previous observation for context; history compares current-episode exemplars "
                       "selected by reference_frame")
            value = item("history", [0., .1], mode="history", reference_display=caption)
            initialized = {**value["record"], "unit_id": "initial-only", "frame": 0, "status": "initialized"}
            summary = reports.build_reports(root, [initialized, save(root, value)])
            self.assertEqual(summary["initialized_unit_count"], 1)
            self.assertEqual(summary["failed_or_unfinished_unit_count"], 0)
            self.assertEqual(summary["completed_unit_count"], 1)
            gallery = (root / summary["gallery_path"]).read_text()
            self.assertIn(caption, gallery)
            self.assertNotIn("Unfinished or failed units", gallery)
            self.assertNotIn("initial-only", gallery)
            with (root / summary["coverage_path"]).open() as stream:
                coverage = list(csv.DictReader(stream))
            initialized_coverage = next(row for row in coverage if row["unit_id"] == "initial-only")
            self.assertEqual(initialized_coverage["status"], "initialized")
            self.assertEqual(initialized_coverage["initialization_only"], "True")
            self.assertEqual(initialized_coverage["numeric_temporal_labels"], "False")


if __name__ == "__main__":
    unittest.main()
