import contextlib
import io
import unittest
from unittest.mock import patch

from facade_change.benchmark_progress import (
    format_case_table, format_crop_table, format_cumulative_test_table, progress_bars,
)


class BenchmarkTableTests(unittest.TestCase):
    def setUp(self):
        self.base = {"base_id": "crop-1", "building_id": "facade-1", "split": "val"}
        self.summaries = {
            "rgb_diff": {"f1": 0.7521, "iou": 0.60, "precision": 0.81,
                         "recall": 0.70, "h0_pixel_fpr": 0.0123,
                         "scoring_seconds": 12.34, "comparable_fraction": 0.95,
                         "retained_visible_edit_fraction": 0.90,
                         "threshold": 0.32, "case_count": 6},
            "msdzip_abs": {"f1": None, "iou": float("nan"), "precision": None,
                           "recall": None, "h0_pixel_fpr": 0, "case_count": 2},
        }

    def test_formats_all_metrics_and_missing_values_without_changing_order(self):
        text = format_crop_table(self.base, self.summaries, calibrated=False)
        self.assertIn("75.21%", text)
        self.assertIn("1.23%", text)
        self.assertIn("95.00%", text)
        self.assertIn("90.00%", text)
        self.assertIn("12.3", text)
        self.assertIn("0.3200", text)
        self.assertIn("—", text)
        self.assertNotIn("nan", text)
        self.assertLess(text.index("rgb_diff"), text.index("msdzip_abs"))
        self.assertIn("предварительные пороги", text)

    def test_test_table_explicitly_identifies_fixed_validation_thresholds(self):
        base = dict(self.base, split="test")
        text = format_crop_table(base, self.summaries, calibrated=True, completed_bases=4)
        self.assertIn("TEST: пороги зафиксированы по всей выбранной validation", text)
        self.assertNotIn("предварительные", text)
        self.assertIn("завершено кропов: 4", text)
        cumulative = format_cumulative_test_table(self.summaries, 2)
        self.assertIn("завершено тестовых кропов: 2", cumulative)
        self.assertIn("по фасадам", cumulative)

    def test_empty_table_and_missing_base_fields_are_readable(self):
        text = format_crop_table({}, {}, calibrated=False)
        self.assertIn("Кроп —", text)
        self.assertIn("Оценок пока нет", text)

    def test_case_table_uses_h1_f1_and_h0_false_positive_rate(self):
        cases = [
            {"case_id": "h0", "state": "unchanged", "scenario_id": "blur_08", "hypothesis": "H0"},
            {"case_id": "h1", "state": "crack", "scenario_id": "shadow_band_35", "hypothesis": "H1"},
            {"case_id": "hidden", "state": "paint_patch", "scenario_id": "occlusion", "hypothesis": "H1"},
        ]
        metrics = {"rgb_diff": {"h0": {"f1": None, "h0_pixel_fpr": 0.0123},
                                "h1": {"f1": 0.75, "h0_pixel_fpr": None},
                                "hidden": {"f1": None}},
                   "msdzip_abs": {}}
        text = format_case_table(cases, metrics)
        self.assertIn("1.23%", text)
        self.assertIn("75.00%", text)
        self.assertIn("shadow_band_35", text)
        self.assertIn("H0 — FPR (ниже лучше)", text)
        self.assertIn("H1 — F1 (выше лучше)", text)
        self.assertEqual(len([line for line in text.splitlines() if line.startswith("paint_patch")]), 1)
        self.assertIn("—", text.splitlines()[-1])


class BenchmarkProgressTests(unittest.TestCase):
    def test_case_method_units_update_both_bars_and_close_after_failure(self):
        instances = []

        class FakeTqdm:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.count = 0
                self.closed = False
                self.postfix = None
                self.postfix_refresh = None
                instances.append(self)

            def update(self, amount):
                self.count += amount

            def set_postfix_str(self, postfix, refresh):
                self.postfix = postfix
                self.postfix_refresh = refresh

            def close(self):
                self.closed = True

            @staticmethod
            def write(message, file):
                pass

        with patch("facade_change.benchmark_progress._load_tqdm", return_value=FakeTqdm):
            with self.assertRaisesRegex(RuntimeError, "scorer failed"):
                with progress_bars(12, 2) as progress:
                    progress.start_base(1, 6, "facade-1 / crop-1")
                    progress.job_started("msdzip_abs", "crack/blur_08")
                    self.assertEqual(progress.completed_jobs, 0)
                    self.assertEqual([bar.count for bar in instances], [0, 0])
                    self.assertEqual(instances[1].postfix, "msdzip_abs / crack/blur_08")
                    self.assertTrue(instances[1].postfix_refresh)
                    for method in ("rgb_diff", "ssim", "msdzip_abs"):
                        progress.job_finished(method, "case-1")
                    raise RuntimeError("scorer failed")
        self.assertEqual([bar.count for bar in instances], [3, 3])
        self.assertTrue(all(bar.closed for bar in instances))
        self.assertEqual(instances[0].kwargs["total"], 12)
        self.assertEqual(instances[1].kwargs["total"], 6)
        self.assertEqual(instances[1].postfix, "msdzip_abs / case-1")

    def test_text_fallback_counts_jobs_and_restarts_crop_count(self):
        output = io.StringIO()
        with patch("facade_change.benchmark_progress._load_tqdm", return_value=None):
            with contextlib.redirect_stdout(output):
                with progress_bars(4, 2) as progress:
                    for index in (1, 2):
                        progress.start_base(index, 2, f"crop-{index}")
                        progress.job_started("rgb_diff", "unchanged/blur_08")
                        self.assertEqual(progress.completed_base_jobs, 0)
                        progress.job_finished("rgb_diff", "case-1")
                        progress.job_finished("ssim", "case-1")
                    self.assertEqual(progress.completed_jobs, 4)
                    self.assertEqual(progress.completed_base_jobs, 2)
        self.assertIn("tqdm недоступен", output.getvalue())
        self.assertIn("Прогресс: 4/4", output.getvalue())


if __name__ == "__main__":
    unittest.main()
