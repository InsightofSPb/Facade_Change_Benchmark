"""One CLI across stages; preserve scientific results and legacy exit behavior."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from facade_change import cli, workflow
from facade_change.io import read_json


ROOT = Path(__file__).resolve().parents[1]


class WorkflowCLITests(unittest.TestCase):
    def test_agentic_route_keeps_backend_arguments_and_exit_status(self):
        backend = Mock()
        backend.main.return_value = 3
        arguments = ["www", "run", "--questions", "path with spaces", "--out", "new"]
        with patch.object(workflow, "import_module", return_value=backend) as load:
            self.assertEqual(cli.main(["workflow", "agentic", *arguments]), 3)
        load.assert_called_once_with("facade_change.agentic.cli")
        backend.main.assert_called_once_with(arguments)

    def test_dry_run_never_imports_or_executes_backend(self):
        for arguments, expected in (
                (["benchmark", "real", "--prepared", "missing", "--out", "new"],
                 ["run", "--prepared", "missing", "--out", "new"]),
                (["agentic", "inspection", "run", "--out", "new"],
                 ["inspection", "run", "--out", "new"]),
                (["benchmark", "metrics", "--run", "saved", "--out", "new"],
                 ["--run", "saved", "--out", "new"])):
            with self.subTest(arguments=arguments), \
                    patch.object(workflow, "import_module") as load, \
                    patch.object(workflow.subprocess, "run") as run, \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(workflow.main(["--dry-run", *arguments]), 0)
            self.assertEqual(json.loads(output.getvalue())["arguments"], expected)
            self.assertEqual(json.loads(output.getvalue())["execution"], "not_started")
            load.assert_not_called()
            run.assert_not_called()

    def test_stage_help_is_lazy_and_invalid_stage_never_loads_models(self):
        for argv, expected in ((["--help"], 0), (["prepare", "--help"], 0),
                               (["train", "--help"], 0), (["unknown", "dataset"], 2)):
            with self.subTest(argv=argv), patch.object(workflow, "import_module") as load, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as caught:
                cli.main(["workflow", *argv])
            self.assertEqual(caught.exception.code, expected)
            load.assert_not_called()

    def test_every_documented_task_has_working_help_without_input_data(self):
        for stage, commands in workflow.COMMANDS.items():
            for task in commands:
                with self.subTest(stage=stage, task=task):
                    result = subprocess.run(
                        [sys.executable, "-B", "-m", "facade_change", "workflow", stage, task, "--help"],
                        cwd=ROOT, text=True, capture_output=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)

    def test_segmentation_stage_mismatch_is_rejected_before_dispatch(self):
        for argv in (["train", "segmentation", "--evaluate-run=existing"],
                     ["train", "segmentation", "--prepare-only"],
                     ["prepare", "augmentations", "--evaluate-run", "existing"],
                     ["benchmark", "segmentation", "--dataset", "dataset"]):
            with self.subTest(argv=argv), patch.object(workflow, "import_module") as load, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                workflow.main(argv)
            self.assertEqual(caught.exception.code, 2)
            load.assert_not_called()

    def test_training_adapter_preserves_arguments_and_failure(self):
        backend = Mock()
        backend.main.return_value = 2
        arguments = ["--dataset", "data with spaces", "--augmentations", "frozen", "--device", "cpu"]
        with patch.object(workflow, "import_module", return_value=backend) as load:
            self.assertEqual(cli.main(["workflow", "train", "segmentation", *arguments]), 2)
        load.assert_called_once_with("facade_change.segmentation.__main__")
        backend.main.assert_called_once_with(arguments)

    def test_real_modes_are_not_reinterpreted_or_reordered(self):
        backend = Mock()
        backend.main.return_value = None
        with patch.object(workflow, "import_module", return_value=backend):
            self.assertEqual(workflow.main(["benchmark", "real", "--prepared", "saved",
                                            "--modes", "history", "earliest"]), 0)
        backend.main.assert_called_once_with(["run", "--prepared", "saved", "--modes", "history", "earliest"])

    def test_frozen_legacy_evaluator_uses_same_python_and_returns_failure(self):
        with patch.object(workflow.subprocess, "run", return_value=Mock(returncode=7)) as run:
            self.assertEqual(workflow.main(["benchmark", "metrics", "--run", "a run", "--out", "new"]), 7)
        run.assert_called_once_with([sys.executable, "-m", "scripts.real_eval.real_gt_metrics",
                                     "--run", "a run", "--out", "new"], check=False)


class CrossCheckoutParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "scripts/2026-11-17-compare-parity.py"
        spec = importlib.util.spec_from_file_location("cross_checkout_parity", path)
        cls.parity = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.parity)

    def test_only_declared_timing_is_excluded_from_metrics(self):
        artifact = "rgb/metrics.json"
        left = {"cases": [{"scoring_seconds": 1.0, "f1": 0.5,
                           "source_year": 2010, "score_sha256": "original"}]}
        right = json.loads(json.dumps(left))
        right["cases"][0]["scoring_seconds"] = 2.0
        expected = self.parity.normalize(left, artifact, ())
        self.assertEqual(expected, self.parity.normalize(right, artifact, ()))
        for key, changed in (("f1", 0.6), ("source_year", 2020),
                             ("score_sha256", "different")):
            altered = json.loads(json.dumps(right))
            altered["cases"][0][key] = changed
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.parity.require_equal(
                    expected, self.parity.normalize(altered, artifact, ()), "metrics")

    def test_full_exposure_preserves_order_geometry_and_pixel_identity(self):
        artifact = "agent/questions/q/policy/exposure.json"
        value = {"evidence": {"photo": {
            "image_sha256": "pixels", "observation_ids": ["2009", "2015"],
            "geometry": {"crop_shape": [10, 20]}}}, "tools": []}
        expected = self.parity.normalize(value, artifact, ())
        for key, replacement in (("image_sha256", "other pixels"),
                                 ("observation_ids", ["2015", "2009"]),
                                 ("geometry", {"crop_shape": [20, 10]})):
            altered = json.loads(json.dumps(value))
            altered["evidence"]["photo"][key] = replacement
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.parity.require_equal(
                    expected, self.parity.normalize(altered, artifact, ()), "exposure")

    def test_saved_verification_fails_for_modified_added_or_deleted_fixture(self):
        for mutation in ("modify", "add", "delete"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                fixtures = out / "fixtures"
                fixtures.mkdir()
                source = fixtures / "rgb.png"
                source.write_bytes(b"immutable fixture")
                self.parity.write(out / "fixture-sha256.json",
                                  self.parity.file_snapshot(fixtures))
                self.parity.write(out / self.parity.REPORT, {"commits": {}})
                if mutation == "modify":
                    source.write_bytes(b"changed pixels")
                elif mutation == "add":
                    (fixtures / "unexpected.png").write_bytes(b"extra")
                else:
                    source.unlink()
                with contextlib.redirect_stderr(io.StringIO()) as error:
                    status = self.parity.main(["--out", str(out), "--verify-only"])
                self.assertEqual(status, 1)
                self.assertIn("fixture mutation", error.getvalue())


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required for the established RGB fixture")
class WorkflowSyntheticTests(unittest.TestCase):
    def test_preparation_benchmark_gallery_preserve_legacy_numbers_and_inputs(self):
        from test_hypothesis_dataset import HypothesisDatasetTests
        from facade_change.hypothesis_benchmark import run_hypothesis_benchmark
        fixture = HypothesisDatasetTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        before = {p: p.read_bytes() for p in fixture.parent.rglob("*") if p.is_file()}
        data, result, legacy = (fixture.root / name for name in ("prepared", "benchmark", "legacy"))
        with contextlib.redirect_stdout(io.StringIO()), \
                patch("facade_change.benchmark_progress._load_tqdm", return_value=None):
            self.assertEqual(cli.main(["workflow", "prepare", "synthetic", "--crop-run", str(fixture.parent),
                                       "--config", str(fixture.config), "--out", str(data)]), 0)
            self.assertEqual(cli.main(["workflow", "benchmark", "synthetic", "--dataset-run", str(data),
                                       "--methods", "rgb_diff", "ssim", "--out", str(result)]), 0)
            run_hypothesis_benchmark(data, legacy, methods=["rgb_diff", "ssim"])
        self.assertEqual(read_json(result / "threshold_selection.json"), read_json(legacy / "threshold_selection.json"))
        new_rows, old_rows = (read_json(p / "metrics.json")["cases"] for p in (result, legacy))
        keys = ("method", "case_id", "building_id", "split", "threshold", "tp", "fp", "fn", "tn",
                "score_sha256", "prediction_sha256")
        self.assertEqual([{k: r[k] for k in keys} for r in new_rows],
                         [{k: r[k] for k in keys} for r in old_rows])
        self.assertTrue((result / "gallery.html").is_file())
        self.assertTrue(list((result / "heatmaps").rglob("*.png")))
        self.assertEqual({p: p.read_bytes() for p in before}, before)


def load_tests(loader, tests, pattern):
    # unittest discovery skips ISO-dated filenames because '-' is not an identifier.
    filenames = (
        "test_2026-11-04_history_benchmark.py",
        "test_2026-11-10_inspection.py",
        "test_2026-11-16_inspection.py",
        "test_2026-11-16_resource_verification.py",
    )
    for filename in filenames:
        path = Path(__file__).with_name(filename)
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tests.addTests(loader.loadTestsFromModule(module))
    return tests


if __name__ == "__main__":
    unittest.main()
