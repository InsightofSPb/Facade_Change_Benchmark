import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from facade_change.benchmark_config import benchmark_arguments
from facade_change.cli import parser
from facade_change.hypothesis_benchmark import run_hypothesis_benchmark
from facade_change.io import read_json, write_json
import test_hypothesis_benchmark as _fixtures


class BenchmarkConfigTests(unittest.TestCase):
    def test_config_paths_resolve_from_config_and_explicit_flags_override(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            configs = root / "configs"
            configs.mkdir()
            write_json(configs / "methods.json", {"methods": {"dinov2": {
                "worker_python": "/usr/bin/python3", "source_root": "../sources/dino"}}})
            write_json(configs / "run.json", {"dataset_run": "../runs/data", "out": "../runs/new",
                "methods_config": "methods.json", "selection_path": "../runs/old/selection.json",
                "reuse_run": "../runs/old", "device": "cuda:0", "methods": ["dinov2"]})
            flags = vars(parser().parse_args(["h0h1-benchmark", "--config", str(configs / "run.json")]))
            flags.pop("command")
            result = benchmark_arguments(flags)
            self.assertEqual(result["dataset_run"], str(root / "runs/data"))
            self.assertEqual(result["out"], str(root / "runs/new"))
            self.assertEqual(result["method_options"]["dinov2"]["source_root"], str(root / "sources/dino"))
            self.assertEqual(result["device"], "cuda:0")
            override = benchmark_arguments({**flags, "out": "runs/override", "methods": ["lpips"]})
            self.assertEqual(override["out"], "runs/override")
            self.assertEqual(override["methods"], ["lpips"])

    def test_unknown_keys_and_missing_required_arguments_fail_before_inference(self):
        with self.assertRaisesRegex(ValueError, "Unknown benchmark"):
            benchmark_arguments({"dataset_run": "x", "out": "y", "invented": 2})
        with self.assertRaisesRegex(ValueError, "requires dataset_run"):
            benchmark_arguments({"methods": ["lpips"]})


class FakeRemoteScorer:
    instances = []

    def __init__(self, method, options, pool):
        self.method, self.options = method, options
        self.metadata = {"method": method, "output_kind": "native_mask" if method == "anychange" else "score"}
        self.raw_scores, self.native_prediction = None, None
        self.activations, self.closed = 0, False
        self.instances.append(self)

    def activate(self):
        self.activations += 1
        self.metadata["loaded"] = True

    def __call__(self, reference, source, support):
        assert reference.dtype == source.dtype == np.uint8 and support.dtype == bool
        distance = np.abs(reference.astype(np.float32) - source.astype(np.float32)).mean(axis=2) / 255
        self.native_prediction = (distance > .1) & support
        scores = self.native_prediction.astype(np.float32) if self.method == "anychange" else distance.astype(np.float32)
        scores[~support] = np.nan
        if self.method != "anychange":
            self.raw_scores = distance.copy()
            self.raw_scores[~support] = np.nan
            self.metadata["raw_units"] = "fixture distance"
        return scores

    def close(self):
        self.closed = True


class CommonRunnerTests(unittest.TestCase):
    setUp = _fixtures.HypothesisBenchmarkTests.setUp
    tearDown = _fixtures.HypothesisBenchmarkTests.tearDown

    def test_merge_reuses_maps_thresholds_and_times_and_preserves_native_decisions(self):
        previous, combined, replayed = (self.root / name for name in ("previous", "combined", "replayed"))
        FakeRemoteScorer.instances = []
        with contextlib.redirect_stdout(io.StringIO()), patch("facade_change.benchmark_progress._load_tqdm", return_value=None):
            original = run_hypothesis_benchmark(self.dataset, previous, methods=["rgb_diff", "ssim"], quick_bases=10)
            with patch("facade_change.scorers.make_scorer", side_effect=AssertionError("cached method loaded")), \
                    patch("facade_change.methods.remote.RemoteScorer", FakeRemoteScorer):
                result = run_hypothesis_benchmark(self.dataset, combined,
                    methods=["rscd_cmu", "anychange"], reuse_run=previous)
            # A combined run can itself be reused, including generic raw maps and
            # fixed native decisions, without loading any method again.
            with patch("facade_change.scorers.make_scorer", side_effect=AssertionError("cached method loaded")):
                again = run_hypothesis_benchmark(self.dataset, replayed, methods=["rgb_diff"], reuse_run=combined)
        self.assertEqual(result["selected_case_count"], original["selected_case_count"])
        self.assertEqual(result["primary"]["rgb_diff"], original["primary"]["rgb_diff"])
        self.assertEqual(result["scoring"]["rgb_diff"]["seconds"], original["scoring"]["rgb_diff"]["seconds"])
        self.assertEqual(result["thresholds"]["anychange"], .5)
        choices = read_json(combined / "threshold_selection.json")["methods"]
        self.assertIsNone(choices["anychange"]["selection_split"])
        self.assertEqual(choices["rscd_cmu"]["selection_split"], "val")
        self.assertEqual(result["native_primary"]["anychange"], result["primary"]["anychange"])
        self.assertIn("rscd_cmu", result["native_primary"])
        self.assertEqual(again["primary"], result["primary"])
        self.assertEqual(again["native_primary"], result["native_primary"])
        self.assertEqual(again["new_inference_case_method_count"], 0)
        self.assertTrue(all(instance.closed for instance in FakeRemoteScorer.instances))
        self.assertTrue(all(instance.activations == result["selected_base_count"] for instance in FakeRemoteScorer.instances))
        old_selection, new_selection = read_json(previous / "selection.json"), read_json(combined / "selection.json")
        self.assertEqual(old_selection["case_ids"], new_selection["case_ids"])
        self.assertEqual(old_selection["quick_selection"], new_selection["quick_selection"])
        rows = read_json(combined / "metrics.json")["cases"]
        self.assertTrue(all(row["raw_score_path"] for row in rows if row["method"] == "rscd_cmu"))

    def test_worker_failure_records_interruption_and_closes_adapter(self):
        out = self.root / "worker-failure"
        FakeRemoteScorer.instances = []
        with contextlib.redirect_stdout(io.StringIO()), \
                patch("facade_change.methods.remote.RemoteScorer", FakeRemoteScorer), \
                patch.object(FakeRemoteScorer, "__call__", side_effect=KeyboardInterrupt("fixture")), \
                patch("facade_change.benchmark_progress._load_tqdm", return_value=None):
            with self.assertRaises(KeyboardInterrupt):
                run_hypothesis_benchmark(self.dataset, out, methods=["anychange"], quick_bases=10)
        record = read_json(out / "run.json")
        self.assertEqual(record["status"], "interrupted")
        self.assertTrue(record["config"]["scorers"]["anychange"]["loaded"])
        self.assertTrue(FakeRemoteScorer.instances[0].closed)

    def test_recompute_replaces_only_requested_cached_method_and_keeps_selection(self):
        previous, replaced = self.root / "old-rscd", self.root / "new-rscd"
        FakeRemoteScorer.instances = []
        with contextlib.redirect_stdout(io.StringIO()), \
                patch("facade_change.benchmark_progress._load_tqdm", return_value=None), \
                patch("facade_change.methods.remote.RemoteScorer", FakeRemoteScorer):
            old = run_hypothesis_benchmark(self.dataset, previous,
                methods=["rgb_diff", "rscd_cmu"], quick_bases=10)
            old_record = (previous / "run.json").read_bytes()
            FakeRemoteScorer.instances = []
            with patch("facade_change.scorers.make_scorer", side_effect=AssertionError("cached RGB loaded")):
                new = run_hypothesis_benchmark(self.dataset, replaced,
                    methods=["rscd_cmu"], recompute_methods=["rscd_cmu"], reuse_run=previous)
        self.assertEqual(old["primary"]["rgb_diff"], new["primary"]["rgb_diff"])
        self.assertEqual(old["scoring"]["rgb_diff"]["seconds"], new["scoring"]["rgb_diff"]["seconds"])
        self.assertEqual((previous / "run.json").read_bytes(), old_record)
        self.assertEqual(new["new_inference_case_method_count"], new["selected_case_count"])
        self.assertEqual([item.method for item in FakeRemoteScorer.instances], ["rscd_cmu"])
        self.assertEqual(read_json(previous / "selection.json")["case_ids"],
                         read_json(replaced / "selection.json")["case_ids"])

    def test_reloaded_model_cannot_change_sources_between_validation_and_test(self):
        out = self.root / "changed-model"
        original = FakeRemoteScorer.activate

        def changed(scorer):
            original(scorer)
            scorer.metadata["source"] = {"sha256": str(scorer.activations)}

        with contextlib.redirect_stdout(io.StringIO()), \
                patch("facade_change.methods.remote.RemoteScorer", FakeRemoteScorer), \
                patch.object(FakeRemoteScorer, "activate", changed), \
                patch("facade_change.benchmark_progress._load_tqdm", return_value=None):
            with self.assertRaisesRegex(ValueError, "changed during benchmark"):
                run_hypothesis_benchmark(self.dataset, out, methods=["anychange"], quick_bases=10)
        self.assertEqual(read_json(out / "run.json")["status"], "failed")
