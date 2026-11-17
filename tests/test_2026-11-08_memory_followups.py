"""Causal controls and end-to-end offline follow-ups on procedural CPU data."""
from argparse import Namespace
import contextlib
import importlib
import io
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch

from facade_change.io import read_json, sha256

followup = importlib.import_module("data_analysis.2026-11-08_memory_followups")
bank_module = followup.bank_module
upstream = importlib.import_module("tests.test_2026-11-08_memory_evaluate")


class FollowupContractTests(unittest.TestCase):
    def test_same_checkpoint_intervention_preserves_pair_dates_scores_and_support(self):
        class Recorder(torch.nn.Module):
            def forward(self, features, support, years, scores):
                self.received = (features.clone(), support.clone(), years.clone(), scores.clone())
                return features[0].sum()
        model = Recorder()
        features = torch.arange(4 * 3 * 2 * 2).reshape(4, 3, 2, 2).float()
        support = torch.ones(4, 1, 2, 2, dtype=torch.bool)
        support[0, :, 0, 1] = False
        support[-2, :, 1, 0] = False
        years, scores = torch.tensor([2000, 2003, 2009, 2012]), torch.rand(7, 4, 4)
        original = features.clone(), support.clone()
        followup.HistoryInput(model)(features, support, years, scores)
        actual = model.received
        followup.HistoryInput(model, repeat=True)(features, support, years, scores)
        repeated = model.received
        torch.testing.assert_close(repeated[0][:-2], features[-2:-1].expand_as(features[:-2]))
        torch.testing.assert_close(repeated[0][-2:], actual[0][-2:])
        for a, b in zip(actual[1:], repeated[1:]):
            torch.testing.assert_close(a, b)
        torch.testing.assert_close(repeated[1][-2:], support[-2:])
        torch.testing.assert_close(repeated[2], years)
        torch.testing.assert_close(repeated[3], scores)
        torch.testing.assert_close(features, original[0])
        torch.testing.assert_close(support, original[1])

    def test_short_histories_are_identical_under_both_interventions(self):
        module = importlib.import_module("facade_change.research.2026-11-08_variability_memory_model")
        model = module.VariabilityMemory(channels=3, hidden=2, mode="convgru")
        torch.nn.init.normal_(model.decoder[-1].weight)
        values = (torch.rand(2, 3, 2, 2), torch.ones(2, 1, 2, 2, dtype=torch.bool),
                  torch.tensor([2000, 2003]), torch.rand(7, 4, 4))
        a = followup.HistoryInput(model)(*values)
        b = followup.HistoryInput(model, repeat=True)(*values)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_mlp_initializes_to_the_linear_function_and_has_145_parameters(self):
        rng = np.random.default_rng(5)
        parameters = rng.normal(size=8)
        model = followup.ScoreMLP(parameters)
        x = torch.from_numpy(rng.normal(size=(100, 7)).astype(np.float32))
        actual = model(x).detach().numpy()
        np.testing.assert_allclose(actual, parameters[0] + x.numpy() @ parameters[1:], atol=1e-6, rtol=1e-5)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 145)

    def test_bank_distance_is_about_transitions_and_preserves_direction(self):
        nuisance = torch.tensor([[0., .1], [.1, 0.]])
        transitions = torch.tensor([[0., .1], [1., 0.], [-1., 0.]])
        distances = bank_module.TransitionBank.nearest(transitions, nuisance).numpy()
        self.assertAlmostEqual(float(distances[0]), 0., places=6)
        self.assertGreater(distances[1], .2)
        self.assertGreater(distances[2], .2)  # Return to an old appearance is still a transition.

    def test_prefix_and_foreign_memory_never_use_current_future_or_query_building(self):
        args = Namespace(seed=13, personal_size=8)
        rows = {str(i): dict(event_id=str(i), track_id="track", frame=i, current_year=2000 + i,
                             split="test", building_id="query") for i in range(1, 5)}
        inputs = Namespace(source=Namespace(data=Namespace(events=rows)))
        bank = bank_module.TransitionBank(inputs, None, args, lambda x, **kw: x)
        bank.stable_threshold = .2
        visited = []
        def transition(event_id):
            visited.append(event_id)
            return np.full((2, 384), int(event_id), np.float32), np.ones(2, bool), np.array([.1, .9])
        bank.transition = transition
        bank.foreign_banks = {
            "query": dict(vectors=np.full((2, 384), 666, np.float32), year=np.array([1990, 1991])),
            "donor": dict(vectors=np.full((3, 384), 99, np.float32), year=np.array([2000, 2001, 2020]))}
        own, foreign = bank.matched_memories(rows["3"])
        self.assertEqual(set(visited), {"1", "2"})
        self.assertEqual(len(own), len(foreign))
        self.assertEqual(len(own), 2)
        self.assertEqual(set(own[:, 0]), {1, 2})
        self.assertEqual(set(foreign[:, 0]), {99})
        self.assertEqual(bank.coverage["3"]["donor"], "donor")
        bank.foreign_banks["donor"]["year"][:] = 2020
        own, foreign = bank.matched_memories(rows["3"])
        self.assertEqual(len(own), 0)
        self.assertEqual(len(foreign), 0)

    def test_global_reference_excludes_the_whole_training_building(self):
        inputs = Namespace(device="cpu", source=Namespace(data=Namespace(events={})))
        bank = bank_module.TransitionBank(inputs, None, Namespace(), lambda x, **kw: x)
        bank.global_banks = {"a": dict(vectors=np.full((3, 384), 1, np.float32)),
                             "b": dict(vectors=np.full((2, 384), 2, np.float32))}
        self.assertEqual(set(bank.reference("a")[:, 0].numpy()), {2})
        self.assertEqual(set(bank.reference("b")[:, 0].numpy()), {1})
        self.assertEqual(len(bank.reference(None)), 5)


class FollowupEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        upstream.SavedEvaluationTests.setUpClass.__func__(cls)

    def arguments(self):
        return Namespace(run=str(self.saved), out=str(self.root / self._testMethodName), seed=13, epochs=1,
                         device="cpu", batch_size=32768, lr=.001, l2=.01, fpr_budgets=[.05, .1, .15],
                         bank_per_building=8, foreign_per_building=16, personal_size=8,
                         min_known_fraction=.5, stable_quantile=.5, resume=False)

    def test_all_followups_reuse_inputs_keep_parity_and_resume(self):
        args = self.arguments()
        before = {str(p): sha256(p) for p in self.saved.rglob("*") if p.is_file()}
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            for target, name in ((followup.memory, "train"), (followup.memory, "prepare_features"),
                                 (followup.memory.history_features, "DinoEncoder"),
                                 (followup.fusion, "RemoteScorer"), (followup.fusion.data_module.Dataset, "frame")):
                stack.enter_context(mock.patch.object(target, name, side_effect=AssertionError(f"Unexpected {name}")))
            result = followup.run(args)
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["feature_extraction_performed"])
        self.assertFalse(result["detector_inference_performed"])
        methods = {r["method"] for r in result["results"]}
        self.assertTrue({"stack", "linear-all", "mlp-16", "transition-global", "transition-own",
                         "transition-foreign", "transition-distance"} <= methods)
        self.assertEqual(result["mlp"]["parameters"], 145)
        self.assertIn(result["mlp"]["selected_epoch"], (0, 1))
        self.assertGreaterEqual(result["correction_parameters"][1], 0)
        self.assertTrue(result["ablation"]["three_observation_cohort_equals_full"])
        self.assertEqual({r["cohort"] for r in result["ablation"]["results"]},
                         {"all", "at_least_three_observations"})
        thresholds = read_json(Path(args.out) / (followup.PREFIX + "ablation_thresholds.json"))
        for selected in thresholds.values():
            self.assertEqual(selected["convgru-common"]["threshold"], selected["convgru-original"]["threshold"])
            self.assertEqual(selected["convgru-repeat"]["threshold"], selected["convgru-original"]["threshold"])
        old = {(r["split"], r["slice_type"], r["slice"]): r for r in self.reference["results"] if r["method"] == "stack"}
        for row in result["results"]:
            if row["method"] == "stack" and row["val_fpr_budget"] == .05:
                for metric in followup.metrics.METRICS:
                    self.assertEqual(row[metric], old[row["split"], row["slice_type"], row["slice"]][metric])
        after = {str(p): sha256(p) for p in self.saved.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        args.resume = True
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(followup, "followups", side_effect=AssertionError("Recomputed completed run")):
            self.assertEqual(followup.run(args), result)
        report = Path(args.out) / (followup.PREFIX + "followup_report.txt")
        report.write_text("corrupt", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(ValueError, "Corrupt follow-up output"):
            followup.run(args)

    def test_bank_reads_only_train_labels_and_predicts_first_prefix_without_detector_maps(self):
        args = self.arguments()
        class Store:
            def read_unit(self, identifier):
                return None
            def save_unit(self, record, arrays):
                return record
        with followup.evaluation.SavedMemory(self.saved) as saved:
            source = followup.fusion.Source(saved.config["source"])
            frozen = followup.memory.FrozenStack(saved.config["fusion"], source)
            inputs = followup.memory.Inputs(source, saved, frozen, "cpu")
            bank = bank_module.TransitionBank(inputs, Store(), args, lambda x, **kw: x)
            original_target = source.data.target
            def train_target(event_id):
                self.assertEqual(source.data.events[event_id]["split"], "train")
                return original_target(event_id)
            with mock.patch.object(source.data, "target", side_effect=train_target):
                bank.prepare()
            row = next(r for r in source.data.samples if r["split"] == "test" and r["frame"] == 2)
            with mock.patch.object(source.data, "target", side_effect=AssertionError("Prediction reads GT")), \
                    mock.patch.object(inputs, "scores", side_effect=AssertionError("Requested first-pair detector maps")):
                maps = bank.maps(row["event_id"])
            self.assertEqual(maps.shape, (4, 224, 224))
            self.assertGreater(bank.coverage[row["event_id"]]["available_own"], 0)
            self.assertGreater(bank.coverage[row["event_id"]]["matched"], 0)

    def test_invalid_budget_fails_before_opening_inputs(self):
        args = self.arguments()
        args.fpr_budgets = [5, 10]
        with mock.patch.object(followup.evaluation, "SavedMemory", side_effect=AssertionError("Opened inputs")):
            with self.assertRaisesRegex(ValueError, "fractional FPR"):
                followup.run(args)


if __name__ == "__main__":
    unittest.main()
