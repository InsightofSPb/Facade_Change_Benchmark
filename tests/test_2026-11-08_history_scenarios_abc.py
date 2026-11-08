"""A/B/C implementation fixtures; these are not evidence on real facades."""
import importlib
import inspect
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is optional in the analysis-only environment")
class ScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.m = importlib.import_module("facade_change.research.2026-11-08_history_scenarios_abc")

    def setUp(self):
        torch.manual_seed(13)

    def inputs(self):
        return (torch.randn(2, 4, 6, 4, 5), torch.ones(2, 4, 4, 5, dtype=torch.bool),
                torch.tensor([[2000., 2003., 2012., 2020.], [2001., 2007., 2013., 2022.]]))

    def test_semantic_repair_and_reappearance_are_not_frozen_out(self):
        values = torch.tensor([.97, .96, .02, .98]).view(1, 4, 1, 1, 1)
        support = torch.ones(1, 4, 1, 1, dtype=torch.bool)
        times = torch.tensor([[2000., 2005., 2010., 2018.]])
        for mode in ("confidence", "learned", "repeat"):
            model = self.m.SemanticMemory(1, mode, hidden=4)
            result = model(values, support, times)
            self.assertGreater(float(result[0, 1]), .9)
            self.assertLess(float(result[0, 2]), .25)
            self.assertGreater(float(result[0, 3]), .75)
            self.assertGreater(float(self.m.semantic_change(result[:, 1], result[:, 2])), .5)
            self.assertGreater(float(self.m.semantic_change(result[:, 2], result[:, 3])), .5)

    def test_semantic_causality_masks_and_repeat_capacity(self):
        _, support, times = self.inputs()
        values = torch.rand(2, 4, 3, 4, 5)
        for mode in self.m.SEMANTIC_MODES:
            model = self.m.SemanticMemory(3, mode, hidden=4)
            original = model(values, support, times)
            changed = values.clone()
            changed[:, -1] = 1 - changed[:, -1]
            torch.testing.assert_close(original[:, :-1], model(changed, support, times)[:, :-1])
            support[:, 1, :2] = False
            expected = model(values, support, times)
            values[:, 1, :, :2] = float("nan")
            torch.testing.assert_close(expected, model(values, support, times))
            values[:, 1, :, :2] = .5
        models = [self.m.SemanticMemory(3, mode, hidden=4) for mode in ("learned", "repeat")]
        self.assertEqual([sum(p.numel() for p in m.parameters()) for m in models][0],
                         sum(p.numel() for p in models[1].parameters()))
        repeat = models[1]
        expected = repeat(values, support, times)[:, -1]
        values[:, :-2] = torch.rand_like(values[:, :-2])
        torch.testing.assert_close(expected, repeat(values, support, times)[:, -1])
        self.assertNotIn("target", inspect.signature(repeat.forward).parameters)

    def test_padding_is_valid_right_prefix_and_last_valid_is_current(self):
        features, support, times = self.inputs()
        probabilities = torch.rand(2, 4, 3, 4, 5)
        padding = torch.tensor([[True, True, True, True], [True, True, False, False]])
        times[1, 2:] = 0
        for scenario, mode in (("A", "learned"), ("A", "repeat"), ("B", "pair"),
                               ("C", "latest"), ("C", "first"), ("C", "history")):
            spec = dict(scenario=scenario, mode=mode, hidden=4)
            model = self.m.build_scenario_model(spec, 6, 3)
            batched = self.m.scenario_forward(model, spec, features, support, times,
                                              semantic_probs=probabilities, padding=padding)
            trimmed = self.m.scenario_forward(model, spec, features[1:2, :2], support[1:2, :2],
                times[1:2, :2], semantic_probs=probabilities[1:2, :2])
            torch.testing.assert_close(batched["logits"][1:2], trimmed["logits"])
            self.assertEqual(batched["last_indices"].tolist(), [3, 1])

    def test_semantic_image_grid_and_dense_target_grid(self):
        features, support, times = self.inputs()
        probabilities = torch.rand(2, 4, 3, 16, 20)
        semantic_support = torch.ones(2, 4, 16, 20, dtype=torch.bool)
        spec = dict(scenario="A", mode="learned", hidden=4)
        model = self.m.build_scenario_model(spec, 6, 3)
        out = self.m.scenario_forward(model, spec, features, semantic_support, times,
                                     semantic_probs=probabilities)
        self.assertEqual(out["logits"].shape, (2, 1, 16, 20))
        semantic_gt = (probabilities > .5).float()
        semantic_known = semantic_support[:, :, None]
        loss = self.m.scenario_loss(model, spec, out, features, semantic_support, times,
            torch.zeros(2, 16, 20), semantic_support[:, -1],
            semantic_gt=semantic_gt, semantic_known=semantic_known)
        loss.backward()
        for scenario, mode in (("B", "history"), ("C", "history")):
            spec = dict(scenario=scenario, mode=mode, hidden=4)
            model = self.m.build_scenario_model(spec, 6, 3)
            out = self.m.scenario_forward(model, spec, features, support, times)
            loss = self.m.scenario_loss(model, spec, out, features, support, times,
                torch.zeros(2, 16, 20), semantic_support[:, -1],
                semantic_gt=semantic_gt, semantic_known=semantic_known)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))

    def test_semantic_unknown_loss_and_training_step(self):
        features, support, times = self.inputs()
        probabilities = torch.rand(2, 4, 3, 4, 5)
        gt = (probabilities > .5).float()
        known = torch.ones_like(gt, dtype=torch.bool)
        spec = dict(scenario="A", mode="learned", hidden=4)
        model = self.m.build_scenario_model(spec, 6, 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        out = self.m.scenario_forward(model, spec, features, support, times, semantic_probs=probabilities)
        loss = self.m.scenario_loss(model, spec, out, features, support, times,
                                   torch.zeros(2, 4, 5), support[:, -1], semantic_gt=gt, semantic_known=known)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        logits = torch.randn(1, 1, 2, 2, requires_grad=True)
        unknown = torch.full((1, 2, 2), 255.)
        empty = self.m.masked_bce(logits, unknown, torch.zeros_like(unknown, dtype=torch.bool))
        self.assertEqual(float(empty), 0.)
        empty.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_pair_inference_signature_and_identical_capacity(self):
        features, support, times = self.inputs()
        models = [self.m.build_scenario_model(dict(scenario="B", mode=mode, hidden=4), 6, 3)
                  for mode in self.m.PAIR_MODES]
        self.assertEqual(len({sum(p.numel() for p in m.parameters()) for m in models}), 1)
        self.assertEqual(set(inspect.signature(models[0].forward).parameters),
                         {"previous", "current", "previous_support", "current_support", "gap"})
        spec = dict(scenario="B", mode="history", hidden=4)
        out = self.m.scenario_forward(models[0], spec, features, support, times)["logits"]
        features[:, :-2] = 100 * torch.randn_like(features[:, :-2])
        support[:, :-2] = False
        times[:, :-2] -= 100
        other = self.m.scenario_forward(models[0], spec, features, support, times)["logits"]
        torch.testing.assert_close(out, other)

    def test_history_auxiliary_requires_known_stability_and_does_not_equalize_changes(self):
        features, support, _ = self.inputs()
        model = self.m.PairAuxDetector(6, hidden=4)
        semantic_gt = torch.zeros(2, 4, 2, 4, 5)
        semantic_gt[:, -1, 0, :, 2:] = 1
        known = torch.ones_like(semantic_gt, dtype=torch.bool)
        known[:, 0, :, 0] = False
        stable = self.m.stable_correspondences(semantic_gt, known, support)
        self.assertFalse(stable[:, :, :, 2:].any())
        self.assertFalse(stable[:, 0, 0].any())
        loss = self.m.pair_auxiliary_loss(model, "history", features, support,
                                        stable_history=stable, changed_mask=~stable[:, -1])
        self.assertGreater(float(loss["reconstruction"]), 0)
        loss["total"].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        with self.assertRaisesRegex(ValueError, "stable TRAIN"):
            self.m.pair_auxiliary_loss(model, "history", features, support)
        empty = self.m.pair_auxiliary_loss(model, "history", features, support,
                                         stable_history=torch.zeros_like(stable))
        self.assertEqual(float(empty["consistency"]), 0)

    def progression(self, damage, repair=None, missing=None):
        t = len(damage)
        names = ["REPAIRS", "ADVERTISEMENTS", "CRACK", "ORNAMENT_INTACT", "TEXT_OR_IMAGES"]
        gt = np.zeros((t, len(names), 1, 1), np.int8)
        gt[:, names.index("CRACK"), 0, 0] = damage
        if repair is not None:
            gt[:, names.index("REPAIRS"), 0, 0] = repair
        known = np.ones((t, 1, 1), bool)
        if missing is not None:
            known[missing] = False
        return self.m.progression_target(gt, known, np.ones_like(known), names,
                                         np.arange(t) * 3 + 2000, damage_classes=("CRACK",))

    def test_progression_stable_worsening_recovery_reappearance_and_unknown(self):
        cases = [([0, 0, 0], 0, 0), ([1, 1, 1], 0, 0),
                 ([0, 1, 1], 1, 1), ([1, 1, 0], 0, 2), ([0, 0, 1], -1, 3),
                 ([1, 0, 1, 1], 1, 1)]
        for damage, target, state in cases:
            actual = self.progression(damage)
            self.assertEqual(int(actual["target"][0, 0]), target, damage)
            self.assertEqual(int(actual["state"][0, 0]), state, damage)
        recovered = self.progression([1, 0, 0], repair=[0, 1, 0])
        self.assertEqual(int(recovered["state"][0, 0]), 2)
        reappeared = self.progression([1, 0, 1, 1], repair=[0, 1, 0, 0])
        self.assertEqual(int(reappeared["target"][0, 0]), 1)
        missing = self.progression([0, 1, 1], missing=1)
        self.assertEqual(missing["coverage"]["status"], "INSUFFICIENT_DATA")
        self.assertEqual(int(missing["target"][0, 0]), -1)
        self.assertEqual(self.progression([0, 1])["coverage"]["status"], "INSUFFICIENT_DATA")

    def test_progression_ontology_not_positional_and_nondamage_not_progression(self):
        names = ["CRACK", "REPAIRS", "ADVERTISEMENTS", "TEXT_OR_IMAGES", "ORNAMENT_INTACT"]
        gt = np.zeros((3, 5, 2, 2), np.int8)
        gt[1:, 2:] = 1
        support = np.ones((3, 2, 2), bool)
        result = self.m.progression_target(gt, support, support, names, [2000, 2001, 2005],
                                           damage_classes=("CRACK",))
        self.assertTrue((result["target"] == 0).all())
        with self.assertRaisesRegex(ValueError, "ontology"):
            self.m.progression_target(gt, support, support, names, [2000, 2001, 2005])
        duplicate = ["CRACK", "CRACK", "ADVERTISEMENTS", "TEXT_OR_IMAGES", "ORNAMENT_INTACT"]
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            self.m.progression_target(gt, support, support, duplicate, [2000, 2001, 2005], damage_classes=("CRACK",))

    def test_progression_trainable_common_target_and_review_coverage(self):
        features, support, times = self.inputs()
        counts = []
        target = torch.zeros(2, 4, 5)
        target[:, :, :2] = 1
        for mode in self.m.PROGRESSION_MODES:
            spec = dict(scenario="C", mode=mode, hidden=4)
            model = self.m.build_scenario_model(spec, 6, 3)
            counts.append(sum(p.numel() for p in model.parameters()))
            output = self.m.scenario_forward(model, spec, features, support, times)
            self.assertEqual(output["logits"].shape, (2, 1, 4, 5))
            loss = self.m.scenario_loss(model, spec, output, features, support, times,
                                       target, support[:, -1])
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(len(set(counts)), 1)
        manifest = self.m.progression_review_manifest([
            dict(event_id="x", building_id="building-1", split="test",
                 coverage=self.progression([0, 1, 1])["coverage"])])
        self.assertEqual(manifest["status"], "INSUFFICIENT_DATA")
        self.assertEqual(manifest["events"][0]["event_id"], "x")

    def test_actual_training_and_prediction_paths_all_abc_controls(self):
        training = importlib.import_module("facade_change.research.2026-11-08_history_scenarios_training")
        data_module = importlib.import_module("facade_change.research.2026-11-08_history_scenarios_data")
        dataset = data_module.SyntheticData(channels=8, grid=4, buildings_per_split=2)
        normalizer = data_module.fit_normalizer(dataset)
        config = dict(smoke=True, epochs=1, patience=1, effective_batch=2, device="cpu", amp=False)
        specs = [dict(s, hidden=4) for s in training.resolve_matrix(config) if s["scenario"] != "main"]
        self.assertEqual(len(specs), 11)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            for spec in specs:
                record = training.train_variant(spec, dataset, directory, config, normalizer=normalizer)
                self.assertEqual(record["status"], "completed", spec["id"])
                if record["parameters"]:
                    self.assertGreater(record["optimizer_steps"], 0)
                selected = training.load_selected(dataset, record, config)
                row = dataset.split_samples("test")[0]
                # Forward includes semantic observations when needed, but no GT.
                with mock.patch.object(dataset, "annotation_history", side_effect=AssertionError("GT in forward")):
                    training._forward(selected, dataset, row, spec, normalizer, "cpu")
                prediction = training.prediction(selected, dataset, row, spec, normalizer, config)
                self.assertEqual(prediction["score"].shape, (8, 8))
                self.assertTrue(np.isfinite(prediction["score"]).all())
                if spec["scenario"] == "A":
                    self.assertEqual(prediction["extras"]["semantic_probability"].shape,
                                     (len(dataset.categories), 8, 8))
                if spec["scenario"] == "C":
                    self.assertIn("progression_target", prediction["extras"])
                with mock.patch.object(training, "_loss", side_effect=AssertionError("retraining completed model")):
                    resumed = training.train_variant(spec, dataset, directory, config,
                                                     normalizer=normalizer, resume=True)
                self.assertEqual(resumed["checkpoint"], record["checkpoint"])
                self.assertTrue(Path(record["checkpoint"]["path"]).is_file())

    def test_training_adapter_uses_old_to_current_stability_not_adjacent_stability(self):
        training = importlib.import_module("facade_change.research.2026-11-08_history_scenarios_training")
        data_module = importlib.import_module("facade_change.research.2026-11-08_history_scenarios_data")
        dataset = data_module.SyntheticData(channels=8, grid=4, buildings_per_split=2)
        normalizer = data_module.fit_normalizer(dataset)
        row = dataset.split_samples("train")[0]
        spec = dict(id="B-history", scenario="B", task="direct", mode="history", hidden=4)
        model = training.build_model(spec, dataset, dict(device="cpu"))
        gt = np.zeros((3, len(dataset.categories), 8, 8), bool)
        gt[-1, 0] = True  # First pair stable, but both older frames differ from current.
        annotation = dict(labels=gt, known=np.ones((3, 1, 8, 8), bool),
                          support=np.ones((3, 1, 8, 8), bool))
        with mock.patch.object(dataset, "annotation_history", return_value=annotation), \
                mock.patch.object(self.m, "pair_auxiliary_loss", wraps=self.m.pair_auxiliary_loss) as call:
            loss = training._loss(model, dataset, row, spec, normalizer, training.options(dict(device="cpu")))
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(call.call_args.kwargs["stable_history"].any())


if __name__ == "__main__":
    unittest.main()
