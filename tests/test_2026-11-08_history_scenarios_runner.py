"""CLI safety, one-seed policy and immutable source-run boundaries."""
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from facade_change.io import write_json

runner = importlib.import_module("data_analysis.2026-11-08_history_scenarios")


class RunnerContractTests(unittest.TestCase):
    def test_exact_seed_and_unknown_practical_keys_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            for value in (29, [13], 13.0):
                write_json(path, {"seed": value})
                with self.assertRaisesRegex(ValueError, "seed"):
                    runner.resolve_config(path)
            write_json(path, {"practical": {"f1_gain": .02, "typo": 0}})
            with self.assertRaisesRegex(ValueError, "Practical"):
                runner.resolve_config(path)

    def test_cli_modes_are_explicit_and_smoke_does_not_shrink_full(self):
        config = runner.resolve_config(device="cpu")
        smoke = runner.resolve_config(device="cpu", smoke=True)
        self.assertEqual(config["epochs"], 60)
        self.assertEqual(config["seed"], smoke["seed"])
        self.assertLess(smoke["epochs"], config["epochs"])
        for mode in ("preflight", "smoke", "full", "resume", "report-only"):
            self.assertEqual(runner.parser().parse_args([mode, "--out", "/tmp/run"]).mode, mode)

    def test_sources_and_ancestors_cannot_be_overwritten(self):
        config = runner.resolve_config()
        for path in (config["memory_run"], str(Path(config["memory_run"]).parent),
                     str(Path(config["memory_run"]) / "child")):
            with self.assertRaisesRegex(ValueError, "protected"):
                runner.safe_output(path, config)
        self.assertEqual(runner.safe_output("/tmp/2026-11-08-new-run", config),
                         Path("/tmp/2026-11-08-new-run"))

    def test_report_only_does_not_open_dataset_or_training(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            write_json(out / "run.json", {"status": "interrupted"})
            (out / ".run.lock").touch()
            reporter = importlib.import_module("data_analysis.2026-11-08_history_scenarios_report")
            with patch.object(reporter, "report_only", return_value={"status": "INCOMPLETE"}) as report, \
                    patch.object(runner, "modules", side_effect=AssertionError("models must not load")):
                self.assertEqual(runner.report_only(out)["status"], "INCOMPLETE")
                report.assert_called_once_with(out)

    def test_local_cpu_smoke_restores_enclosing_cuda_determinism_policy(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Optional CPU Torch is not installed")
        old_enabled = torch.are_deterministic_algorithms_enabled()
        old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            with tempfile.TemporaryDirectory() as folder:
                out = Path(folder)
                def smoke(args):
                    torch.use_deterministic_algorithms(True, warn_only=False)
                    Path(args.out).mkdir()
                    write_json(Path(args.out) / "run.json", {"status": "completed"})
                with patch.object(runner, "run", side_effect=smoke):
                    runner.local_smoke(out, {}, unittest.mock.Mock())
                self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())
        finally:
            torch.use_deterministic_algorithms(old_enabled, warn_only=old_warn)


if __name__ == "__main__":
    unittest.main()
