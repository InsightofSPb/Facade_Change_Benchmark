"""Spatial temporal and probability contracts; synthetic checks are not science evidence."""
import importlib
import math
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch optional; run these contracts in the history environment")
class TemporalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.module = importlib.import_module("facade_change.research.2026-11-08_history_scenarios_models")

    def inputs(self):
        generator = torch.Generator().manual_seed(13)
        value = torch.randn(2, 4, 3, 3, 4, generator=generator)
        support = torch.ones(2, 4, 1, 3, 4, dtype=torch.bool)
        years = torch.tensor([[2000., 2005., 2010., 2020.], [2000., 2010., 2020., float("nan")]])
        padding = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
        return value, support, years, padding

    def model(self, family="convgru", task="actual-next", **kwargs):
        torch.manual_seed(13)
        return self.module.TemporalSpatialModel(channels=3, family=family, size="smoke", task=task, **kwargs)

    def test_both_families_all_tasks_train_dense_outputs(self):
        value, support, years, padding = self.inputs()
        for family in ("convgru", "transformer"):
            for task in self.module.TASKS:
                with self.subTest(family=family, task=task):
                    model = self.model(family, task)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
                    target_year = torch.tensor([2020., 2020.]) if task == "direct" else torch.tensor([2025., 2025.])
                    output = model(value, support, years, target_year, padding)
                    if task == "direct":
                        self.assertEqual(tuple(output["logits"].shape), (2, 1, 3, 4))
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(output["logits"], torch.ones_like(output["logits"]))
                    else:
                        self.assertEqual(tuple(output["mean"].shape), (2, 3, 3, 4))
                        self.assertTrue(bool((output["scale"] >= .05).all()))
                        loss = self.module.quantized_logistic_nll(value[:, -1], output["mean"], output["scale"]).mean()
                    before = next(model.parameters()).detach().clone()
                    loss.backward()
                    optimizer.step()
                    self.assertFalse(torch.equal(before, next(model.parameters())))
                    self.assertTrue(math.isfinite(float(loss.detach())))

    def test_padding_and_unsupported_values_cannot_leak(self):
        value, support, years, padding = self.inputs()
        support[:, :, :, 0, 0] = False
        for family in ("convgru", "transformer"):
            model = self.model(family)
            first = model(value, support, years, torch.tensor([2025., 2025.]), padding)
            changed = value.clone()
            changed[1, 3] = float("nan")
            changed[:, :, :, 0, 0] = float("nan")
            second = model(changed, support, years, torch.tensor([2025., 2025.]), padding)
            torch.testing.assert_close(first["mean"], second["mean"], rtol=0, atol=0)
            self.assertFalse(bool(first["support"][:, :, 0, 0].any()))
            single = model(value[1:2, :3], support[1:2, :3], years[1:2, :3], torch.tensor([2025.]))
            torch.testing.assert_close(first["mean"][1:], single["mean"], rtol=1e-5, atol=1e-6)

    def test_absent_whole_frame_transformer_no_nan(self):
        value, support, years, padding = self.inputs()
        support[:, 1] = False
        support[0] = False
        result = self.model("transformer")(value, support, years, torch.tensor([2025., 2025.]), padding)
        self.assertTrue(bool(torch.isfinite(result["mean"]).all()))
        self.assertFalse(bool(result["support"][0].any()))

    def test_no_cross_building_attention_or_batch_statistic_leakage(self):
        value, support, years, padding = self.inputs()
        for family in ("convgru", "transformer"):
            model = self.model(family)
            first = model(value, support, years, torch.tensor([2025., 2025.]), padding)
            changed, changed_support = value.clone(), support.clone()
            changed[0] = 1000
            changed_support[0, :, :, :, 0] = False
            second = model(changed, changed_support, years, torch.tensor([2025., 2025.]), padding)
            torch.testing.assert_close(first["mean"][1], second["mean"][1], rtol=0, atol=0)

    def test_trainer_preprocessing_forecast_ignores_deleted_current_future_and_gt(self):
        data = importlib.import_module("facade_change.research.2026-11-08_history_scenarios_data")
        training = importlib.import_module("facade_change.research.2026-11-08_history_scenarios_training")
        dataset = data.SyntheticData(channels=3, grid=3)
        normalizer = data.fit_normalizer(dataset)
        row = dataset.split_samples("val")[0]
        selected = dataset.tracks[row["track_id"]]
        for task in ("actual-next", "h0"):
            for family in ("convgru", "transformer"):
                for control in ("history", "last-only"):
                    spec = dict(task=task, scenario="main", family=family, control=control,
                                size="smoke", delta="signed", time_mode="real")
                    model = self.model(family, task, control=control).eval()
                    first, initial_inputs = training._forward(model, dataset, row, spec, normalizer, "cpu")
                    removed = {}
                    for observation in selected["observations"]:
                        if observation["year"] >= row["current_year"]:
                            identifier = observation["observation_id"]
                            removed[identifier] = dataset._features.pop(identifier)
                    # Any target, GT, semantic or pair map lookup here is a bug.
                    try:
                        with mock.patch.object(dataset, "targets", side_effect=AssertionError("GT entered forecast")), \
                             mock.patch.object(dataset, "annotation", side_effect=AssertionError("GT entered forecast")), \
                             mock.patch.object(dataset, "feature_target", side_effect=AssertionError("Current feature entered forecast")), \
                             mock.patch.object(dataset, "scores", side_effect=AssertionError("Current detector map entered forecast")):
                            second, altered_inputs = training._forward(model, dataset, row, spec, normalizer, "cpu")
                        for key in initial_inputs:
                            torch.testing.assert_close(initial_inputs[key], altered_inputs[key], rtol=0, atol=0)
                        torch.testing.assert_close(first["mean"], second["mean"], rtol=0, atol=0)
                    finally:
                        dataset._features.update(removed)

    def test_predictor_rejects_current_and_invalid_time_contract(self):
        value, support, years, padding = self.inputs()
        model = self.model()
        with self.assertRaisesRegex(ValueError, "strictly past"):
            model(value, support, years, torch.tensor([2020., 2020.]), padding)
        with self.assertRaisesRegex(ValueError, "right-padded"):
            invalid_padding = padding.clone()
            invalid_padding[0, 1] = False
            model(value, support, years, torch.tensor([2025., 2025.]), invalid_padding)

    def test_last_only_and_pair_only_ignore_older_content(self):
        value, support, years, padding = self.inputs()
        changed = value.clone()
        changed[:, 0] += 100
        for family in ("convgru", "transformer"):
            for task, control, target in (("direct", "pair-only", 2020.), ("actual-next", "last-only", 2025.)):
                model = self.model(family, task, control=control)
                key = "logits" if task == "direct" else "mean"
                first = model(value, support, years, torch.tensor([target, target]), padding)
                second = model(changed, support, years, torch.tensor([target, target]), padding)
                torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

    def test_mean_is_orderless_and_index_time_removes_physical_intervals(self):
        value, support, years, padding = self.inputs()
        for family in ("convgru", "transformer"):
            model = self.model(family, control="mean")
            altered = value.clone()
            altered[:, :2] = altered[:, :2].flip(1)
            first = model(value, support, years, torch.tensor([2025., 2025.]), padding)
            second = model(altered, support, years, torch.tensor([2025., 2025.]), padding)
            torch.testing.assert_close(first["mean"], second["mean"], rtol=1e-6, atol=1e-6)
            model = self.model(family, time_mode="index")
            first = model(value, support, years, torch.tensor([2025., 2025.]), padding)
            second = model(value, support, 1900. + (years - 2000.) * 3,
                           torch.tensor([1975., 1975.]), padding)
            torch.testing.assert_close(first["mean"], second["mean"], rtol=0, atol=0)

    def test_intervention_keeps_last_pair_and_recomputes_directional_delta(self):
        value, support, years, padding = self.inputs()
        altered, actual_support = self.module.intervene_history(value, support, padding, mode="repeat", preserve=2)
        for row, n in enumerate(padding.sum(1).tolist()):
            torch.testing.assert_close(altered[row, n - 2:n], value[row, n - 2:n], rtol=0, atol=0)
        self.assertTrue(torch.equal(support, actual_support))
        old, _ = self.module.recompute_deltas(value, support & padding[:, :, None, None, None])
        new, valid = self.module.recompute_deltas(altered, support & padding[:, :, None, None, None])
        self.assertFalse(torch.equal(old[0, 1], new[0, 1]))
        self.assertTrue(torch.equal(new[0, 1], torch.zeros_like(new[0, 1])))
        absolute, _ = self.module.recompute_deltas(altered, support, "absolute")
        torch.testing.assert_close(new[0].abs(), absolute[0])
        self.assertFalse(bool(valid[1, 3].any()))

    def test_repeat_and_shuffle_fail_closed_on_incompatible_support(self):
        value, support, years, padding = self.inputs()
        support[0, 2, :, 0, 0] = False
        with self.assertRaisesRegex(ValueError, "common source support"):
            self.module.intervene_history(value, support, padding, mode="repeat", preserve=2)

    def test_meaningful_scales_parameter_matching(self):
        for size in ("medium", "large"):
            for task in ("direct", "actual-next"):
                counts = []
                for family in ("convgru", "transformer"):
                    # Real DINO cache shape: exercise the primary sizes too,
                    # rather than relying on smoke modules and parameter counts.
                    model = self.module.TemporalSpatialModel(channels=384, family=family, size=size, task=task)
                    counts.append(self.module.count_parameters(model))
                    value = torch.randn(1, 3, 384, 16, 16)
                    support = torch.ones(1, 3, 1, 16, 16, dtype=torch.bool)
                    support[:, :, :, 0, 0] = False
                    years = torch.tensor([[2001., 2008., 2020.]])
                    target_year = torch.tensor([2020. if task == "direct" else 2025.])
                    output = model(value, support, years, target_year)
                    if task == "direct":
                        self.assertEqual(tuple(output["logits"].shape), (1, 1, 16, 16))
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            output["logits"], torch.zeros_like(output["logits"]))
                    else:
                        self.assertEqual(tuple(output["mean"].shape), (1, 384, 16, 16))
                        loss = self.module.quantized_logistic_nll(
                            torch.randn_like(output["mean"]), output["mean"], output["scale"]).mean()
                    loss.backward()
                    self.assertTrue(bool(torch.isfinite(loss)))
                    self.assertTrue(all(bool(torch.isfinite(parameter.grad).all())
                                        for parameter in model.parameters() if parameter.grad is not None))
                    del model, output, loss
                self.assertGreater(min(counts), 1_000_000)
                self.assertLess(abs(counts[0] - counts[1]) / min(counts), .25)

    def test_quantized_probability_known_mass_and_positive_history_gain(self):
        target = torch.zeros(1, 1, 1, 1, dtype=torch.float64, requires_grad=True)
        mean = torch.zeros_like(target, requires_grad=True)
        scale = torch.ones_like(target, requires_grad=True)
        bits = self.module.quantized_logistic_nll(target, mean, scale, step=2.)
        self.assertAlmostEqual(float(bits.detach()), -math.log2(math.tanh(.5)), places=12)
        bits.sum().backward()
        self.assertIsNone(target.grad)
        gain = self.module.history_gain(target, mean, scale, mean + 2, scale, torch.ones_like(target, dtype=torch.bool), step=2.)
        self.assertGreater(float(gain.detach()), 0.)
        # Every real value falls into a quantizer bin; masses sum to one.
        centers = torch.arange(-40, 41, dtype=torch.float64).reshape(1, 1, 1, -1)
        masses = 2 ** -self.module.quantized_logistic_nll(centers, torch.zeros_like(centers), torch.ones_like(centers), step=1.)
        self.assertAlmostEqual(float(masses.sum()), 1., places=12)

    def test_quantized_extreme_tail_and_variance_cost_are_finite(self):
        target = torch.tensor([[[[10000.]]]])
        mean = torch.zeros_like(target, requires_grad=True)
        scale = torch.full_like(target, .05, requires_grad=True)
        bits = self.module.quantized_logistic_nll(target, mean, scale)
        bits.sum().backward()
        self.assertTrue(bool(torch.isfinite(bits).all() & torch.isfinite(mean.grad).all() & torch.isfinite(scale.grad).all()))
        zero = torch.zeros_like(target)
        small = self.module.quantized_logistic_nll(zero, zero, torch.ones_like(zero))
        inflated = self.module.quantized_logistic_nll(zero, zero, torch.ones_like(zero) * 100)
        self.assertGreater(float(inflated), float(small))
        stats = self.module.probability_diagnostics(zero, zero, torch.ones_like(zero), torch.zeros_like(zero, dtype=torch.bool))
        self.assertIsNone(stats["nll_bits_per_component"])

    def test_probability_calibration_and_saturation_diagnostics(self):
        generator = torch.Generator().manual_seed(13)
        u = torch.rand((1, 1, 100, 100), generator=generator).clamp(.00001, .99999)
        target = torch.log(u) - torch.log1p(-u)
        scale, mean = torch.ones_like(target), torch.zeros_like(target)
        stats = self.module.probability_diagnostics(target, mean, scale, torch.ones_like(target, dtype=torch.bool))
        self.assertAlmostEqual(stats["interval_95_coverage"], .95, delta=.012)
        self.assertEqual(stats["count"], 10000)
        self.assertEqual(stats["scale_floor_fraction"], 0.)
        tail = self.module.probability_diagnostics(target * 0 + 10000, mean, scale * .05,
                                                  torch.ones_like(target, dtype=torch.bool))
        self.assertEqual(tail["numerical_saturation_fraction"], 1.)
        self.assertEqual(tail["scale_floor_fraction"], 1.)


if __name__ == "__main__":
    unittest.main()
