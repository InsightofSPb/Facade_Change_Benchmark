import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from facade_change.segmentation.metrics import (
    aggregate_metrics,
    confusion_counts,
    write_reports,
)


class SegmentationMetricsTests(unittest.TestCase):
    def setUp(self):
        self.categories = [{"id": 8, "name": "CRACK"}, {"id": 2, "name": "REPAIRS"}]

    def test_overlapping_labels_and_ignored_pixels(self):
        target = np.array([[[1, 1], [0, 0]], [[1, 0], [1, 0]]], dtype=bool)
        prediction = np.array([[[1, 0], [1, 1]], [[1, 1], [1, 1]]], dtype=bool)
        valid = np.array([[1, 1], [0, 0]], dtype=bool)
        counts = confusion_counts(prediction, target, valid)
        np.testing.assert_array_equal(counts, [[1, 0, 1], [1, 1, 0]])
        self.assertEqual(counts.dtype, np.int64)

    def test_equal_facades_and_count_pooling_inside_facade(self):
        records = [
            {"image_id": 1, "building_id": "large", "counts": [[90, 0, 0], [0, 0, 0]]},
            {"image_id": 2, "building_id": "large", "counts": [[0, 0, 10], [0, 0, 0]]},
            {"image_id": 3, "building_id": "small", "counts": [[0, 0, 1], [0, 0, 0]]},
        ]
        result = aggregate_metrics(records, self.categories)
        self.assertAlmostEqual(result["primary"]["miou"], 0.45)
        self.assertAlmostEqual(result["pixel_pooled_secondary"]["miou"], 90 / 101)
        self.assertEqual(result["primary"]["building_count"], 2)
        self.assertEqual(result["per_building"][0]["image_count"], 2)
        self.assertEqual(result["per_class"][0]["category_id"], 8)
        self.assertEqual(result["per_class"][1]["eligible_buildings"]["iou"], 0)

    def test_absent_ground_truth_with_false_positive_is_zero_iou(self):
        result = aggregate_metrics([
            {"image_id": 1, "building_id": "facade", "counts": [[4, 0, 0], [0, 2, 0]]}
        ], self.categories)
        self.assertEqual(result["primary"]["miou"], 0.5)
        self.assertEqual(result["per_class"][1]["iou"], 0)
        self.assertIsNone(result["per_class"][1]["recall"])
        self.assertEqual(result["per_class"][1]["precision"], 0)

    def test_all_invalid_is_null_and_json_safe(self):
        labels = np.ones((2, 3, 3), dtype=bool)
        counts = confusion_counts(labels, labels, np.zeros((3, 3), dtype=bool))
        result = aggregate_metrics([
            {"image_id": 1, "building_id": "facade", "counts": counts.tolist()}
        ], self.categories)
        self.assertIsNone(result["primary"]["miou"])
        self.assertEqual(result["primary"]["eligible_buildings"]["miou"], 0)
        json.dumps(result, allow_nan=False)
        self.assertIsNone(aggregate_metrics([], self.categories)["primary"]["miou"])

    def test_native_images_must_not_be_counted_twice(self):
        record = {"image_id": 1, "building_id": "facade", "counts": [[1, 0, 0], [0, 0, 0]]}
        with self.assertRaisesRegex(ValueError, "duplicate native image"):
            aggregate_metrics([record, record], self.categories)
        with self.assertRaisesRegex(ValueError, "confirmed building_id"):
            aggregate_metrics([{**record, "building_id": None}], self.categories)

    def test_invalid_counts_and_unthresholded_scores_rejected(self):
        record = {"image_id": 1, "building_id": "facade", "counts": [[1.5, 0, 0], [0, 0, 0]]}
        with self.assertRaisesRegex(ValueError, "nonnegative integers"):
            aggregate_metrics([record], self.categories)
        with self.assertRaisesRegex(ValueError, "boolean"):
            confusion_counts(np.zeros((2, 3, 3)), np.zeros((2, 3, 3), bool), np.ones((3, 3), bool))

    def test_csv_reports_keep_undefined_values_empty(self):
        metrics = aggregate_metrics([
            {"image_id": 1, "building_id": "facade", "counts": [[1, 0, 1], [0, 0, 0]]}
        ], self.categories)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_reports(directory, metrics)
            with Path(paths["per_class"]).open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["category_id"], "8")
            self.assertEqual(rows[1]["iou"], "")
            self.assertEqual(rows[1]["iou_buildings"], "0")
            with Path(paths["per_building"]).open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["miou"], "0.5")


if __name__ == "__main__":
    unittest.main()
