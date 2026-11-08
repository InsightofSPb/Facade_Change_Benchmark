"""Saved-head evaluation parity and failure cases on procedural CPU data."""
from argparse import Namespace
import contextlib
import csv
import importlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from facade_change.io import read_json, sha256, write_json

evaluation = importlib.import_module("data_analysis.2026-11-08_memory_evaluate")
memory, fusion = evaluation.memory, evaluation.fusion
previous = importlib.import_module("tests.test_2026-11-06_history_fusion")

try:
    import torch
except ImportError:
    torch = None


class OperatingPointTests(unittest.TestCase):
    def test_val_budgets_change_thresholds_but_keep_native_mask_and_frozen_fpr05(self):
        hist = np.zeros((1, 2, memory.metrics.BINS), np.int64)
        for score, count in ((.1, 85), (.4, 5), (.7, 5), (.9, 5)):
            hist[0, 0, int(np.ceil(score * 2048))] = count
        for score, count in ((.2, 20), (.5, 30), (.8, 30), (.95, 20)):
            hist[0, 1, int(np.ceil(score * 2048))] = count
        native = np.zeros_like(hist)
        native[0, 0, [0, -1]] = [92, 8]
        native[0, 1, [0, -1]] = [40, 60]
        val = dict(stack=dict(building=hist.copy()), pair=dict(building=hist.copy()),
                   anychange=dict(building=native))
        frozen = Namespace(fpr_threshold=dict(threshold=.75, selection_split="val"))
        selected = evaluation.fpr_thresholds(val, frozen, [.05, .10, .15])
        for budget, expected_recall in ((.05, .5), (.10, .8), (.15, 1.)):
            for method in ("pair", "stack"):
                point = selected[budget][method]
                self.assertAlmostEqual(point["calibration_recall"], expected_recall)
                self.assertLessEqual(point["calibration_fpr"], budget)
                self.assertTrue(point["val_budget_met"])
                self.assertIn(f"<= {budget:g};", point["criterion"])
            self.assertEqual(selected[budget]["anychange"]["threshold"], .5)
            self.assertAlmostEqual(selected[budget]["anychange"]["calibration_fpr"], .08)
            self.assertEqual(selected[budget]["anychange"]["val_budget_met"], budget >= .08)
        self.assertEqual(selected[.05]["stack"]["threshold"], .75)
        self.assertEqual(frozen.fpr_threshold, dict(threshold=.75, selection_split="val"))
        self.assertGreater(selected[.10]["stack"]["threshold"], selected[.15]["stack"]["threshold"])


@unittest.skipIf(torch is None, "PyTorch is optional for offline analysis environments")
class SavedEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        root = cls.root
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fusion.data_module.prepare(previous.original.fixture(root))
            with mock.patch.object(fusion.benchmark, "RemoteScorer", previous.SemanticFake), mock.patch.object(
                    fusion.benchmark, "make_scorer", previous.SemanticFake):
                fusion.benchmark.run(previous.original.arguments(root, methods=list(fusion.METHODS)))
            args = Namespace(run=str(root / "full"), out=str(root / "fusion"), sample_per_building=200,
                             seed=13, l2=.01, semantic_device="cpu", all_pairs=False, resume=False)
            with mock.patch.object(fusion, "RemoteScorer", previous.SemanticFake):
                fusion.run(args)

            def fake_features(source, store, device):
                encoder = store.out / (memory.PREFIX + "encoder.json")
                write_json(encoder, dict(procedural_test_double=True))
                for identifier in source.data.observations:
                    rgb, _ = source.data.frame(identifier)
                    store.save_unit(dict(unit_id=memory.PREFIX + "features-" + identifier,
                        kind="frozen_observation_features", observation_id=identifier,
                        encoder_sha256=sha256(encoder), target_labels_used=False),
                        dict(patch_tokens=np.full((256, 384), rgb.mean() / 255, np.float16),
                             patch_support=np.ones(256, bool)))
                return dict(observations=len(source.data.observations), procedural_test_double=True)

            args = Namespace(run=str(root / "full"), fusion=str(root / "fusion"), out=str(root / "memory"),
                             device="cpu", epochs=2, hidden=4, batch_size=2, lr=.001,
                             seeds=[13], history_shift_tokens=[1, 7], resume=False)
            with mock.patch.object(memory, "prepare_features", side_effect=fake_features):
                cls.reference = memory.run(args)
        # Reproduce the stopped three-seed plan: only the four seed-13 units exist.
        cls.saved = root / "memory"
        config = read_json(cls.saved / "request.json")
        config["seeds"] = [13, 29, 47]
        write_json(cls.saved / "request.json", config)
        record = read_json(cls.saved / "run.json")
        record.update(status="interrupted", config=config)
        write_json(cls.saved / "run.json", record)
        contract = read_json(cls.saved / "experiment_contract.json")
        contract["config"] = config
        write_json(cls.saved / "experiment_contract.json", contract)

    def setUp(self):
        self.args = Namespace(run=str(self.saved), out=str(self.root / self._testMethodName),
                              seeds=[13], device=None, resume=False)

    def replace_json(self, path, change):
        original = path.read_bytes()
        self.addCleanup(path.write_bytes, original)
        value = read_json(path)
        change(value)
        write_json(path, value)

    def test_interrupted_subset_matches_complete_run_without_training_or_rgb(self):
        self.args.fpr_budgets = [.05, .10, .15]
        before = {str(p): sha256(p) for p in self.saved.rglob("*") if p.is_file()}
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            for target, name in ((memory, "train"), (memory, "prepare_features"),
                                 (memory.history_features, "DinoEncoder"), (torch.optim, "AdamW"),
                                 (fusion, "RemoteScorer"), (fusion.data_module.Dataset, "frame")):
                stack.enter_context(mock.patch.object(target, name, side_effect=AssertionError(f"Unexpected {name}")))
            with mock.patch.object(memory, "evaluate", wraps=memory.evaluate) as forward:
                result = evaluation.run(self.args)
            self.assertEqual(forward.call_count, 5)  # Three splits plus two history shifts, not per budget.
            self.assertEqual(result["results"], self.reference["results"])
            self.assertEqual(result["comparisons"], self.reference["comparisons"])
            self.assertEqual(result["evaluated_seeds"], [13])
            self.assertEqual(result["originally_requested_seeds"], [13, 29, 47])
            self.assertFalse(result["training_performed"])
            self.assertEqual(result["features"]["newly_computed"], 0)
            self.assertEqual(result["fpr_budgets"], [.05, .10, .15])
            self.assertEqual(result["checkpoint_selection_fpr_budget"], .05)
            for suffix in ("metrics.csv", "per_building.csv", "comparisons.csv", "history_shift.csv",
                           "return_to_past.csv", "thresholds-13.json"):
                self.assertEqual((Path(self.args.out) / (memory.PREFIX + suffix)).read_bytes(),
                                 (self.saved / (memory.PREFIX + suffix)).read_bytes())
            for ordinary, budget_file in (("metrics.csv", "fpr_metrics.csv"),
                                          ("per_building.csv", "fpr_per_building.csv"),
                                          ("return_to_past.csv", "fpr_return_to_past.csv")):
                with (Path(self.args.out) / (memory.PREFIX + ordinary)).open(encoding="utf-8-sig") as stream:
                    expected = list(csv.DictReader(stream))
                with (Path(self.args.out) / (memory.PREFIX + budget_file)).open(encoding="utf-8-sig") as stream:
                    actual = [r for r in csv.DictReader(stream) if float(r["val_fpr_budget"]) == .05]
                self.assertEqual([{k: r[k] for k in expected[0]} for r in actual], expected)
            points = read_json(Path(self.args.out) / (memory.PREFIX + "fpr_thresholds-13.json"))
            for row in result["fpr_results"]:
                selected = points[str(row["val_fpr_budget"])][row["method"]]
                self.assertEqual(row["threshold"], selected["threshold"])
                self.assertEqual(row["overall_val_fpr"], selected["calibration_fpr"])
                if row["method"] != "anychange":
                    self.assertTrue(row["val_budget_met"])
            self.args.resume = True
            with mock.patch.object(memory, "evaluate", side_effect=AssertionError("Repeated evaluation")):
                self.assertEqual(evaluation.run(self.args), result)
            report = Path(self.args.out) / (memory.PREFIX + "report.txt")
            report.write_text("damaged report", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Completed evaluation output is corrupt"):
                evaluation.run(self.args)
        after = {str(p): sha256(p) for p in self.saved.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(read_json(self.saved / "run.json")["status"], "interrupted")
        self.assertEqual(read_json(Path(self.args.out) / "run.json")["status"], "completed")

    def test_invalid_budgets_fail_before_opening_inputs(self):
        for budgets in ([5, 10, 15], [0], [1], [.1, .1], [float("nan")], [float("inf")]):
            self.args.fpr_budgets = budgets
            with mock.patch.object(evaluation, "SavedMemory", side_effect=AssertionError("Read inputs")):
                with self.assertRaisesRegex(ValueError, "FPR budgets must be distinct fractions"):
                    evaluation.run(self.args)

    def test_fpr_sweep_can_skip_completed_shift_diagnostics(self):
        self.args.fpr_budgets = [.05, .10, .15]
        self.args.skip_history_shifts = True
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                mock.patch.object(memory, "evaluate", wraps=memory.evaluate) as forward:
            evaluation.run(self.args)
        self.assertEqual(forward.call_count, 3)
        self.assertFalse((Path(self.args.out) / (memory.PREFIX + "history_shift.csv")).exists())
        self.assertTrue((Path(self.args.out) / (memory.PREFIX + "fpr_report.txt")).exists())

    def test_missing_last_head_never_evaluates_or_trains(self):
        unit = self.saved / "units" / (memory.PREFIX + "model-convgru-last-13.json")
        original = unit.read_bytes()
        unit.unlink()
        self.addCleanup(unit.write_bytes, original)
        with mock.patch.object(fusion, "Source", side_effect=AssertionError("Started evaluation")):
            with self.assertRaisesRegex(ValueError, "Missing completed unit.*convgru-last-13"):
                evaluation.run(self.args)
        self.assertFalse(Path(self.args.out).exists())

    def test_corrupt_weights_and_mismatched_encoder_are_rejected(self):
        unit = self.saved / "units" / (memory.PREFIX + "model-pair-13.json")
        artifact = self.saved / read_json(unit)["artifact_path"]
        original = artifact.read_bytes()
        artifact.write_bytes(b"corrupt")
        try:
            with self.assertRaisesRegex(ValueError, "Corrupt committed unit"):
                evaluation.run(self.args)
        finally:
            artifact.write_bytes(original)
        feature = next((self.saved / "units").glob(memory.PREFIX + "features-*.json"))
        self.replace_json(feature, lambda r: r.update(encoder_sha256="another-encoder"))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "Feature encoder contract differs"):
                evaluation.run(self.args)

    def test_active_training_lock_and_nonexistent_seed_are_rejected(self):
        with (self.saved / ".run.lock").open("rb") as lock:
            evaluation.fcntl.flock(lock, evaluation.fcntl.LOCK_EX | evaluation.fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "still running"):
                evaluation.run(self.args)
        self.args.seeds = [29]
        with self.assertRaisesRegex(ValueError, "Missing completed unit.*pair-29"):
            evaluation.run(self.args)

    def test_scientific_code_change_rejected_progress_only_change_accepted(self):
        path = self.saved / "experiment_contract.json"
        runner = Path(memory.__file__).name
        migration = read_json(Path(memory.__file__).with_name(memory.PREFIX + "memory_progress_compatibility.json"))
        self.replace_json(path, lambda c: c["code"].update({runner: migration["before_sha256"]}))
        with evaluation.SavedMemory(self.saved) as saved:
            self.assertEqual(evaluation.validate_code(saved)[runner], migration["after_sha256"])
        self.replace_json(path, lambda c: c["code"].update({"2026-11-08_variability_memory_model.py": "changed"}))
        with self.assertRaisesRegex(ValueError, "implementation changed"):
            evaluation.run(self.args)

    def test_cli_selects_original_worker_without_training_options(self):
        source = dict(config=dict(options=dict(dinov2=dict(worker_python="/original/env/bin/python"))))
        with mock.patch.object(evaluation, "read_json", side_effect=[dict(source=str(self.root / "full")), source]), \
                mock.patch.object(evaluation.subprocess, "run", return_value=Namespace(returncode=0)) as launch:
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exit_code:
                evaluation.main(["--run", str(self.saved), "--out", self.args.out, "--seeds", "13", "--resume",
                                 "--fpr-budgets", "0.05", "0.10", "0.15", "--skip-history-shifts"])
        self.assertEqual(exit_code.exception.code, 0)
        command = launch.call_args.args[0]
        self.assertEqual(command[0], "/original/env/bin/python")
        self.assertIn("memory-evaluate", command)
        self.assertIn("--worker", command)
        self.assertNotIn("--epochs", command)
        self.assertNotIn("--fusion", command)
        self.assertEqual(command[command.index("--fpr-budgets") + 1:command.index("--fpr-budgets") + 4],
                         ["0.05", "0.1", "0.15"])
        self.assertIn("--skip-history-shifts", command)


if __name__ == "__main__":
    unittest.main()
