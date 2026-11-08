"""The real full launch must be a deliberate second step after profiling."""
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

runner = importlib.import_module("data_analysis.2026-11-08_history_scenarios")
progress = importlib.import_module("facade_change.research.2026-11-08_history_scenarios_progress")


class ReducedRunnerTests(unittest.TestCase):
    def test_default_matrix_and_fusion_budget(self):
        training = importlib.import_module("facade_change.research.2026-11-08_history_scenarios_training")
        config = runner.resolve_config(device="cpu")
        matrix = runner._reduced_matrix(training, config)
        self.assertEqual(len(matrix["variants"]), 18)
        self.assertEqual(len(matrix["fusion_variants"]), 56)
        self.assertEqual(config["fusion_epochs"], 5)
        self.assertEqual(config["effective_batch"], 8)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            path.write_text(json.dumps({"fusion_epochs": 11}))
            with self.assertRaisesRegex(ValueError, "Fusion epochs"):
                runner.resolve_config(path)

    def test_full_without_profile_only_profiles(self):
        preflight = importlib.import_module("facade_change.research.2026-11-08_history_scenarios_preflight")
        data, models, training, reporter = runner.modules()
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            args = runner.parser().parse_args(["full", "--out", str(out), "--device", "cpu"])
            config = runner.resolve_config(device="cpu")
            def profile(dataset, config, root, **kwargs):
                report = {"status": "MEASURED"}
                (root / "2026-11-08_resource_report.json").write_text(json.dumps(report))
                return report
            with patch.object(data, "ScenarioData", return_value=data.SyntheticData()), \
                    patch.object(preflight, "run_preflight", side_effect=profile) as measured, \
                    patch.object(preflight, "compatible", return_value=(True, [])), \
                    patch.object(training, "train_all", side_effect=AssertionError("training forbidden")), \
                    patch.object(training, "export_all", side_effect=AssertionError("TEST forbidden")):
                self.assertEqual(runner.run_reduced(args, config)["status"], "MEASURED")
            measured.assert_called_once()
            self.assertEqual(json.loads((out / "run.json").read_text())["status"], "preflight_completed")
            command = runner._continue_command(args, out / "2026-11-08_resource_report.json")
            self.assertIn("--preflight-report", command)
            self.assertIn(str(out), command)
            # A separate explicit full invocation can reuse that same directory.
            args.preflight_report = str(out / "2026-11-08_resource_report.json")
            with patch.object(data, "ScenarioData", return_value=data.SyntheticData()), \
                    patch.object(preflight, "compatible", return_value=(True, [])), \
                    patch.object(preflight, "check_resources", return_value={"passed": True}), \
                    patch.object(preflight, "run_preflight", side_effect=AssertionError("must reuse profile")), \
                    patch.object(training, "train_all", return_value=[]) as fit, \
                    patch.object(training, "export_all", return_value=[]) as export, \
                    patch.object(reporter, "build_report", return_value={"fixture": True}):
                self.assertEqual(runner.run_reduced(args, config), {"fixture": True})
                fit.assert_called_once()
                export.assert_called_once()
            self.assertEqual(json.loads((out / "run.json").read_text())["status"], "completed")

    def test_plain_progress_preserves_exceptions_and_labels_old_val(self):
        stream = io.StringIO()
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with progress.Progress(stage="train", total=2, epoch=3, stream=stream) as bar:
                bar.update(loss=.3, last_val_epoch=2, last_val_f1=.4)
                raise RuntimeError("failure")
        output = stream.getvalue()
        self.assertIn("status=failed", output)
        self.assertIn("last_val_epoch=2", output)
        self.assertNotIn("\x1b", output)
        self.assertNotIn("\r", output)

    def test_legacy_configuration_remains_explicit(self):
        config = runner.resolve_config(runner.LEGACY_CONFIG, device="cpu")
        self.assertEqual(config["protocol"], "history-three-scenarios-v1")
        self.assertNotIn("matrix_protocol", config)

    def test_reprofiling_a_resume_prints_resume_without_restart(self):
        args = runner.parser().parse_args(["resume", "--out", "/tmp/history-partial"])
        command = runner._continue_command(args, "/tmp/history-partial/profile.json")
        self.assertIn("history-scenarios resume", command)
        self.assertNotIn("history-scenarios full", command)

    def test_frozen_json_preserves_bytes_mtime_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "matrix.json"
            path.write_text('{"matrix":18}')
            before = path.read_bytes(), path.stat().st_mtime_ns
            runner._frozen_json(path, {"matrix": 18})
            self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))
            with self.assertRaisesRegex(ValueError, "Frozen scientific contract differs"):
                runner._frozen_json(path, {"matrix": 71})
            self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))
