"""Canonical count parity, paired inference and no-inference report reconstruction."""
from contextlib import redirect_stdout
import importlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from facade_change.io import read_json, sha256, write_json

report = importlib.import_module("data_analysis.2026-11-08_history_scenarios_report")


class Fixture:
    def __init__(self):
        self.categories = [dict(id=19, name="CRACK"), dict(id=91, name="REPAIRS")]
        self.groups = [dict(name="damage", categories=["CRACK"])]
        self.samples, self.arrays = [], {}
        for split in ("train", "val", "test"):
            for building in ("positive", "negative"):
                # Two crops overlap in source coordinates and must count overlap once.
                for crop in range(2):
                    event = f"{split}-{building}-{crop}"
                    target = np.zeros((4, 4), np.uint8)
                    if building == "positive":
                        target[1:3, 1:3] = 1
                    target[0, 0] = 255
                    before = np.ones_like(target, np.uint32)
                    after = before.copy()
                    after[target == 1] = 2
                    class_targets = np.repeat(target[None], 2, axis=0)
                    self.arrays[event] = dict(target=target, support=np.ones((4, 4), bool),
                        class_targets=class_targets, before=before, after=after)
                    self.samples.append(dict(event_id=event, split=split, building_id=split + building,
                        view_id=building, previous_year=2001, current_year=2010, bbox_xywh=[0, 0, 4, 4],
                        known_pixels=int((target != 255).sum()), history_length=3))
        self.canonical = self

    def target(self, event):
        return self.arrays[event]


def maps(root, data):
    manifest = []
    for method, control in (("history", "history"), ("pair", "pair-only")):
        for row in data.samples:
            arrays = data.target(row["event_id"])
            scores = np.where(arrays["target"] == 1, .9, .1).astype(np.float32)
            if method == "pair":
                scores[1, 1] = .1
            name = method + "-" + row["event_id"] + ".npz"
            np.savez(root / name, **arrays, score=scores, U=np.full((4, 4), 2., np.float32),
                l1=np.full((4, 4), .2, np.float32), forecast_support=np.ones((4, 4), bool))
            manifest.append(dict(**row, path=name, sha256=sha256(root / name), method=method,
                task="direct", family="convgru", size="medium", control=control, observations=3))
    return manifest


class ReportTests(unittest.TestCase):
    def test_canonical_baseline_count_parity_and_negative_only_fpr(self):
        data = Fixture()
        prediction = lambda row: dict(score=np.full((4, 4), .2, np.float32), support=np.ones((4, 4), bool))
        actual = report.evaluate_predictor(data, prediction, "val", threshold=.5)
        old, _ = report.metrics.collect(data, [s for s in data.samples if s["split"] == "val"],
            lambda row: (prediction(row)["score"], prediction(row)["support"]))
        for building in old:
            expected = report.metrics.metrics(old[building][0], .5)
            found = next(r for r in actual["per_building"] if r["building_id"] == building and r["slice_type"] == "overall")
            for key in ("tp", "fp", "fn", "tn", "ap_binned"):
                self.assertEqual(expected[key], found[key])
        self.assertEqual(actual["f1_buildings"], 1)
        self.assertEqual(actual["fpr_buildings"], 2)
        self.assertEqual(actual["known_pixels"], 30)  # Not 60 from overlapping crops.
        self.assertEqual(actual["f1"], 0.)
        self.assertIsNone(actual["precision"])

    def test_undefined_metrics_and_val_with_no_positives(self):
        histogram = np.zeros((1, 2, report.metrics.BINS), np.int64)
        histogram[0, 0, 1] = 100
        with self.assertRaises(report.InsufficientData):
            report.select_threshold({"building": histogram})
        result = report.fixed_macro([report.metrics.metrics(histogram[0], .5)])
        self.assertIsNone(result["f1"])
        self.assertEqual(result["fpr"], 0.)
        self.assertEqual(result["fpr_buildings"], 1)

    def test_threshold_selects_recall_then_lower_fpr(self):
        hist = np.zeros((1, 2, report.metrics.BINS), np.int64)
        hist[0, 0, 100] = 95
        hist[0, 0, 700] = 5
        hist[0, 1, 1000] = 10
        selected = report.select_threshold({"building": hist}, .05)
        self.assertEqual(selected["calibration_recall"], 1.)
        self.assertEqual(selected["calibration_fpr"], 0.)
        self.assertEqual(selected["threshold"], 999 / 2048)

    def test_bootstrap_paired_buildings_and_support_rejection(self):
        rows = []
        for method, values in (("history", [.7, .4, None]), ("pair", [.5, .4, None])):
            for building, value in enumerate(values):
                rows.append(dict(method=method, building_id=str(building), f1=value,
                                 positive_pixels=10 if building < 2 else 0, negative_pixels=10))
        result = report.paired_bootstrap(rows, "history", "pair", repetitions=2000)
        self.assertAlmostEqual(result["delta"], .1)
        self.assertEqual(result["buildings"], 3)
        self.assertEqual(result["eligible_buildings"], 2)
        self.assertEqual(result["ci_low"], 0.)
        self.assertFalse(result["bootstrap_supports_direction"])
        rows[-1]["negative_pixels"] = 9
        with self.assertRaisesRegex(ValueError, "support differ"):
            report.paired_bootstrap(rows, "history", "pair")

    def test_practical_status_requires_prediction_and_detection(self):
        baseline = dict(f1=.2, iou=.1, recall=.4, fpr=.04)
        better = dict(f1=.23, iou=.13, recall=.4, fpr=.04)
        self.assertEqual(report.practical_status(better, baseline), "PRACTICALLY_USEFUL")
        self.assertEqual(report.practical_status(better, baseline, predictive=True, prediction_improved=False),
                         "BELOW_PRACTICAL_THRESHOLD")
        self.assertEqual(report.practical_status(dict(baseline, f1=.2005), baseline), "BELOW_PRACTICAL_THRESHOLD")
        self.assertEqual(report.practical_status(baseline, baseline), "NO_GAIN")
        self.assertEqual(report.practical_status({}, {}, execution_status="interrupted"), "INCOMPLETE")
        self.assertEqual(report.practical_status({}, {}, execution_status="error"), "ERROR")
        self.assertEqual(report.practical_status({}, {}), "INSUFFICIENT_DATA")

    def test_full_maps_report_rebuild_and_common_scales(self):
        data = Fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = maps(root, data)
            records = [dict(id="history", task="direct", family="convgru", size="medium", control="history", status="completed"),
                       dict(id="pair", task="direct", family="convgru", size="medium", control="pair-only", status="completed")]
            with redirect_stdout(io.StringIO()):
                result = report.build_report(data, manifest, root, dict(bootstrap_repetitions=100), records)
            before = {row["path"]: sha256(root / row["path"]) for row in manifest}
            with mock.patch.object(report, "evaluate_predictor", side_effect=AssertionError("No inference")), redirect_stdout(io.StringIO()):
                again = report.report_only(root)
            self.assertEqual(result, again)
            self.assertEqual(before, {row["path"]: sha256(root / row["path"]) for row in manifest})
            main = result["main_table"][0]
            self.assertEqual(main["control"], "pair")
            self.assertEqual(main["status"], "PRACTICALLY_USEFUL")
            self.assertTrue(result["cohorts"]["history"]["cohorts_identical"])
            gallery = report.build_gallery(root, manifest, result["thresholds"]["0.05"], limit=2)
            self.assertEqual(gallery["scales"]["score"], [0., 1.])
            self.assertGreater(gallery["examples"], 0)
            self.assertIn("not saved", (root / (report.PREFIX + "gallery") / (report.PREFIX + "index.html")).read_text())

    def test_incomplete_report_without_maps_can_rebuild(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / (report.PREFIX + "resolved_config.json"), {})
            write_json(root / (report.PREFIX + "resolved_matrix.json"), dict(variants=[dict(id="future", task="h0")]))
            with redirect_stdout(io.StringIO()):
                result = report.report_only(root)
            self.assertEqual(result["main_table"][0]["status"], "INCOMPLETE")
            self.assertIsNone(result["main_table"][0]["f1"])
            self.assertIn("NA", (root / (report.PREFIX + "main_table.csv")).read_text())

    def test_histogram_spool_is_read_only_mmap_and_retains_only_descriptors(self):
        with tempfile.TemporaryDirectory() as temporary:
            spool = report._HistogramSpool(Path(temporary))
            source = {"b": np.arange(2 * report.metrics.BINS, dtype=np.int64).reshape(1, 2, -1),
                      "a": np.zeros((1, 2, report.metrics.BINS), np.int64)}
            expected = source["b"].copy()
            spool["history", "test"] = source
            source["b"][:] = -1
            mapped = spool["history", "test"]
            self.assertIsInstance(mapped["b"], np.memmap)
            self.assertFalse(mapped["b"].flags.writeable)
            np.testing.assert_array_equal(mapped["b"], expected)
            self.assertEqual(tuple(mapped), ("a", "b"))
            for path, buildings in spool.index.values():
                self.assertIsInstance(path, Path)
                self.assertEqual(buildings, ("a", "b"))
            self.assertEqual(report.select_threshold(mapped), report.select_threshold(dict(mapped)))

    def test_report_uses_mmap_spool_and_cleans_it_after_success_or_error(self):
        data = Fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = maps(root, data)
            records = [dict(id="history", task="direct", family="convgru", size="medium", control="history", status="completed"),
                       dict(id="pair", task="direct", family="convgru", size="medium", control="pair-only", status="completed")]
            original_load = np.load
            mmap_loads = []
            def load(*args, **kwargs):
                value = original_load(*args, **kwargs)
                if kwargs.get("mmap_mode") == "r":
                    self.assertIsInstance(value, np.memmap)
                    mmap_loads.append(str(args[0]))
                return value
            with mock.patch.object(report.np, "load", side_effect=load), redirect_stdout(io.StringIO()):
                first = report.build_report(data, manifest, root, dict(bootstrap_repeats=8, gallery_limit=0), records)
                expected = (root / (report.PREFIX + "main_table.csv")).read_bytes()
                second = report.report_only(root)
            self.assertEqual(first, second)
            self.assertEqual(expected, (root / (report.PREFIX + "main_table.csv")).read_bytes())
            self.assertGreater(len(mmap_loads), 0)
            self.assertEqual(list(root.glob(report.PREFIX + "histograms-*")), [])
            with mock.patch.object(report, "collect_results", side_effect=ValueError("broken map")):
                with self.assertRaisesRegex(ValueError, "broken map"):
                    report.build_report(data, manifest, root, {}, records)
            self.assertEqual(list(root.glob(report.PREFIX + "histograms-*")), [])

    def test_explicit_nondefault_practical_and_bootstrap_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, data = Path(temporary), Fixture()
            manifest = maps(root, data)
            records = [dict(id="history", task="direct", family="convgru", size="medium", control="history", status="completed"),
                       dict(id="pair", task="direct", family="convgru", size="medium", control="pair-only", status="completed")]
            config = dict(bootstrap_repeats=17, practical=dict(f1_gain=1., iou_gain=1.), gallery_limit=0)
            with redirect_stdout(io.StringIO()):
                result = report.build_report(data, manifest, root, config, records)
            self.assertEqual(result["main_table"][0]["status"], "BELOW_PRACTICAL_THRESHOLD")
            import csv
            with (root / (report.PREFIX + "comparisons.csv")).open(encoding="utf-8-sig") as stream:
                self.assertEqual({row["repetitions"] for row in csv.DictReader(stream)}, {"17"})
            self.assertEqual(report.practical_status(dict(f1=.22,iou=.1,recall=.4,fpr=.04),
                                                     dict(f1=.20,iou=.1,recall=.4,fpr=.04)), "PRACTICALLY_USEFUL")

    def test_stack_nonlinearity_and_native_or_follow_overlap_merge(self):
        data = Fixture()
        samples = [row for row in data.samples if row["split"] == "val"]
        contract = dict(normalizer=dict(low=[0.], high=[1.]), parameters=[-.8, 1.], center=[0.], scale=[1.],
                        method_names=["anychange"])
        def prediction(row):
            raw = np.full((1,4,4), float(row["event_id"].endswith("-1")), np.float32)
            return dict(score=np.zeros((4,4),np.float32), support=np.ones((4,4),bool),
                        extras=dict(stack_raw=raw), stack_contract=contract)
        hist, coverage = report.collect_results(data, samples, prediction)
        result = report.metrics.metrics(hist["valpositive"][0], .5)
        self.assertEqual(result["tp"], 4)
        self.assertEqual(result["fp"], 11)
        self.assertEqual(coverage["known_pixels"], 30)
        self.assertIn("before nonlinear", coverage["stack_merge"])

    def test_progression_callback_uses_proxy_instead_of_last_change(self):
        data = Fixture()
        def predict(row):
            proxy = np.zeros((4,4), np.float32)
            proxy[3,3] = 1
            return dict(score=np.where(proxy == 1,.9,.1).astype(np.float32), support=np.ones((4,4),bool),
                        extras=dict(progression_target=proxy))
        result = report.evaluate_predictor(data,predict,"val",spec=dict(task="progression",scenario="C"),threshold=.5)
        self.assertEqual(result["f1"],1.)
        self.assertEqual(result["positive_pixels"],2)

    def test_missing_forecast_support_excluded_from_prediction_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.savez(root / "map.npz", target=np.zeros((2, 2), np.uint8), support=np.ones((2, 2), bool),
                     forecast_support=np.array([[1, 0], [0, 0]], bool), U=np.array([[2., 0.], [0., 0.]]))
            summary, rows = report._prediction_diagnostics(root, [dict(path="map.npz", building_id="one")])
            self.assertEqual(summary["U_all"], 2.)
            self.assertEqual(rows[0]["valid_pixel_crops"], 1)


if __name__ == "__main__":
    unittest.main()
