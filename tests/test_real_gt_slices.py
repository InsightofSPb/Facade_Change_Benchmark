import hashlib
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.real_eval.real_gt_slices import SliceCollector, aggregate_slices, evaluate_class_slices


CATEGORIES = [{"id": 7, "name": "REPAIRS"}, {"id": 3, "name": "DELAMINATION"}]


class SliceTests(unittest.TestCase):
    def fixture_arrays(self):
        old = np.array([1, 2, 1, 1, 2, 3], np.uint16)
        new = np.array([1, 2, 2, 3, 1, 1], np.uint16)
        labels = np.array([0, 0, 1, 1, 1, 1], np.uint8)
        pred = np.array([1, 1, 1, 1, 0, 1], bool)
        return pred, old, new, np.ones(6, bool), labels

    def test_other_class_changes_and_stable_other_classes_are_not_false_positives(self):
        rows = evaluate_class_slices(*self.fixture_arrays(), CATEGORIES)
        row = next(row for row in rows if row["class_id"] == 7 and row["direction"] == "change")
        self.assertEqual([row[key] for key in ("tp", "fp", "fn", "tn")], [1, 1, 1, 0])
        self.assertEqual(row["valid_pixels"], 3)
        # Channel order follows metadata, not category_id - 1.
        second = next(row for row in rows if row["class_id"] == 3 and row["direction"] == "change")
        self.assertEqual([second[key] for key in ("tp", "fp", "fn", "tn")], [3, 1, 1, 0])

    def test_gain_and_loss_ignore_opposite_direction(self):
        rows = evaluate_class_slices(*self.fixture_arrays(), CATEGORIES)
        gain = next(row for row in rows if row["class_id"] == 7 and row["direction"] == "gain")
        loss = next(row for row in rows if row["class_id"] == 7 and row["direction"] == "loss")
        self.assertEqual((gain["tp"], gain["fn"], gain["fp"]), (0, 1, 1))
        self.assertEqual((loss["tp"], loss["fn"], loss["fp"]), (1, 0, 1))

    def test_unknown_and_explicit_region_remain_excluded(self):
        arrays = list(self.fixture_arrays())
        arrays[-1][2] = 255
        region = np.array([0, 1, 1, 1, 1, 1], bool)
        rows = evaluate_class_slices(*arrays, CATEGORIES, region_mask=region)
        first = rows[0]
        self.assertEqual((first["tp"], first["fp"], first["fn"], first["valid_pixels"]), (0, 0, 1, 1))
        self.assertEqual(first["precision"], 0.0)

    def make_collector_fixture(self, root):
        pred, old, new, support, labels = self.fixture_arrays()
        path = root / "gt.npz"
        np.savez_compressed(path, reference_bits=old.reshape(2, 3), source_bits=new.reshape(2, 3),
                            labels=labels.reshape(2, 3), support=support.reshape(2, 3))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        (root / "provenance.json").write_text(json.dumps({"output_gt_sha256": {"gt.npz": digest}}))
        sample = dict(gt_artifact="gt.npz", sequence_id="real-a-x0-y0", frame=1,
                      building_id="A", shape=(2, 3), pixels=np.arange(6, dtype=np.int32), labels=labels)
        return pred, sample

    def test_collector_fails_changed_labels_and_exports_no_invented_sky(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pred, sample = self.make_collector_fixture(root)
            collector = SliceCollector(root, CATEGORIES)
            kwargs = dict(family="fusion", mode="previous", predictor="anychange+mod256", goal="f1",
                          heldout_building="A", sample=sample, prediction=pred)
            collector.record(**kwargs)
            protocol = collector.finalize(root / "out")
            self.assertEqual(protocol["sky_status"], "unavailable_no_explicit_sky_masks")
            with (root / "out" / "class_per_pair.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["region"] for row in rows}, {"annotated_support"})
            self.assertEqual(protocol["output_rows"]["class_per_pair.csv"], 6)
            sample["labels"] = sample["labels"].copy()
            sample["labels"][0] = 1
            with self.assertRaisesRegex(ValueError, "labels differ"):
                collector.record(**kwargs)

    def test_explicit_sky_requires_hash_and_known_mask_without_extending_gt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pred, sample = self.make_collector_fixture(root)
            np.savez_compressed(root / "region.npz", sky=np.array([[1, 0, 0], [0, 0, 0]], bool), known=np.ones((2, 3), bool))
            digest = hashlib.sha256((root / "region.npz").read_bytes()).hexdigest()
            manifest = root / "regions.json"
            manifest.write_text(json.dumps(dict(schema_version=1, entries=[dict(sequence_id=sample["sequence_id"], mode="previous", frame=1, path="region.npz", sha256=digest)])))
            collector = SliceCollector(root, CATEGORIES, manifest)
            collector.record(family="fusion", mode="previous", predictor="anychange+mod256", goal="f1", heldout_building="A", sample=sample, prediction=pred)
            collector.finalize(root / "out")
            with (root / "out" / "class_per_pair.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            without = next(row for row in rows if row["region"] == "without_sky" and row["class_id"] == "7" and row["direction"] == "change")
            self.assertEqual(int(without["fp"]), 0)
            self.assertEqual(int(without["tp"]), 1)
            manifest_data = json.loads(manifest.read_text())
            manifest_data["entries"][0]["sha256"] = "wrong"
            manifest.write_text(json.dumps(manifest_data))
            broken = SliceCollector(root, CATEGORIES, manifest)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                broken.record(family="fusion", mode="previous", predictor="x", goal="f1", heldout_building="A", sample=sample, prediction=pred)
            broken.finalize(root / "failed-out")

    def test_facade_macro_is_not_pixel_pooled(self):
        base = dict(family="fusion", mode="previous", predictor="x", goal="f1", region="annotated_support",
                    class_id=7, class_name="REPAIRS", direction="change")
        a = dict(base, building_id="A", tp=100, fp=0, fn=0, tn=0, valid_pixels=100, positive_pixels=100, negative_pixels=0)
        b = dict(base, building_id="B", tp=0, fp=0, fn=1, tn=0, valid_pixels=1, positive_pixels=1, negative_pixels=0)
        facades, macro = aggregate_slices([a, b])
        self.assertEqual(len(facades), 2)
        self.assertEqual(macro[0]["recall"], 0.5)
        self.assertEqual(macro[0]["f1"], 0.5)

    def test_absent_classes_remain_visible_with_null_metrics(self):
        row = dict(family="fusion", mode="previous", predictor="x", goal="f1", region="annotated_support",
                   class_id=7, class_name="REPAIRS", direction="change", building_id="A",
                   tp=0, fp=0, fn=0, tn=0, valid_pixels=0, positive_pixels=0, negative_pixels=0)
        facades, macro = aggregate_slices([row])
        self.assertEqual(facades, [])
        self.assertEqual(len(macro), 1)
        self.assertEqual(macro[0]["status"], "no_evaluable_pixels")
        self.assertEqual(macro[0]["class_name"], "REPAIRS")
        self.assertIsNone(macro[0]["f1"])

    def test_frame_means_precede_facade_means_and_negative_frames_do_not_dilute_prfi(self):
        base = dict(family="fusion", mode="previous", predictor="x", goal="f1", region="annotated_support",
                    class_id=7, class_name="REPAIRS", direction="change")
        rows = [dict(base, building_id="A", tp=100, fp=0, fn=0, tn=10, valid_pixels=110, positive_pixels=100, negative_pixels=10),
                dict(base, building_id="A", tp=0, fp=0, fn=1, tn=10, valid_pixels=11, positive_pixels=1, negative_pixels=10),
                dict(base, building_id="A", tp=0, fp=10, fn=0, tn=0, valid_pixels=10, positive_pixels=0, negative_pixels=10),
                dict(base, building_id="B", tp=1, fp=0, fn=0, tn=1, valid_pixels=2, positive_pixels=1, negative_pixels=1)]
        facades, macro = aggregate_slices(rows)
        a = next(row for row in facades if row["building_id"] == "A")
        self.assertEqual(a["f1"], 0.5)
        self.assertEqual(a["f1_frames"], 2)
        self.assertAlmostEqual(a["fpr"], 1 / 3)
        self.assertEqual(macro[0]["f1"], 0.75)
        self.assertAlmostEqual(macro[0]["fpr"], 1 / 6)
        self.assertEqual(macro[0]["aggregation"], "frame_then_facade_macro")

    def test_zero_valid_pair_rows_not_exported_but_macro_keeps_class(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pred, sample = self.make_collector_fixture(root)
            categories = CATEGORIES + [{"id": 9, "name": "ABSENT"}]
            collector = SliceCollector(root, categories)
            for _ in range(10):
                collector.record(family="fusion", mode="previous", predictor="x", goal="f1", heldout_building="A", sample=sample, prediction=pred)
            protocol = collector.finalize(root / "out")
            self.assertFalse(hasattr(collector, "rows"))
            self.assertEqual(protocol["output_rows"]["class_per_pair.csv"], 60)
            with (root / "out" / "class_macro.csv").open() as stream:
                absent = [row for row in csv.DictReader(stream) if row["class_name"] == "ABSENT"]
            self.assertEqual(len(absent), 3)
            self.assertEqual({row["status"] for row in absent}, {"no_evaluable_pixels"})
            self.assertEqual({row["f1"] for row in absent}, {""})


if __name__ == "__main__":
    unittest.main()
