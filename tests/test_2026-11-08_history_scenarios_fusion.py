"""Scientific plumbing tests, not evidence of a temporal-history benefit."""
import importlib
import json
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch optional")
class FusionHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.f = importlib.import_module("facade_change.research.2026-11-08_history_scenarios_fusion")

    def setUp(self):
        torch.manual_seed(13)

    def scores(self, shape=(2, 1, 9, 11)):
        stack = torch.rand(shape, requires_grad=True)
        method = torch.rand(shape, requires_grad=True)
        support = torch.ones(shape, dtype=torch.bool)
        support[0, :, 2:4, 3:6] = False
        if shape[0] > 1:
            support[1] = False
        return stack, method, support

    def test_exact_score_architectures_and_finite_backward(self):
        for kind, count in (("logreg", 3), ("score-mlp", 129), ("score-cnn", 2641)):
            with self.subTest(kind=kind):
                head = self.f.build_head(kind)
                self.assertEqual(self.f.parameter_count(head), count)
                stack, method, support = self.scores()
                logits = self.f.score_logits(head, stack, method, support)
                loss = self.f.masked_building_bce(logits, torch.zeros_like(logits), support, support)
                loss.backward()
                self.assertTrue(bool(torch.isfinite(logits).all()))
                self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters()))
                self.assertIsNone(stack.grad)
                self.assertIsNone(method.grad)
                self.assertTrue(bool((self.f.score_prediction(head, stack, method, support)[~support] == 0).all()))

    def test_common_clipping_logit_and_invalid_nan_policy(self):
        stack, method, support = self.scores()
        stack = stack.detach()
        method = method.detach()
        stack[0, 0, 0, 0], method[0, 0, 0, 0] = 0., 1.
        baseline = self.f.score_inputs(stack, method, support)
        stack[~support], method[~support] = float("nan"), float("inf")
        torch.testing.assert_close(self.f.score_inputs(stack, method, support), baseline, rtol=0, atol=0)
        self.assertTrue(bool(torch.isfinite(baseline).all()))
        self.assertTrue(bool((baseline[~support.expand_as(baseline)] == 0).all()))
        stack[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.f.score_inputs(stack, method, support)

    def test_cnn_halo_equals_whole_raster_forward_and_backward(self):
        for kind in self.f.SCORE_KINDS:
            head = self.f.build_head(kind)
            stack, method, support = self.scores((1, 1, 9, 11))
            whole = self.f.score_prediction(head, stack, method, support)
            tiled = self.f.apply_score_blocks(head, stack, method, support, block_size=3)
            torch.testing.assert_close(tiled, whole, rtol=1e-6, atol=1e-7)
            head.zero_grad()
            whole.sum().backward()
            expected = [p.grad.clone() for p in head.parameters()]
            head.zero_grad()
            self.f.apply_score_blocks(head, stack, method, support, block_size=3).sum().backward()
            for actual, reference in zip(head.parameters(), expected):
                torch.testing.assert_close(actual.grad, reference, rtol=2e-5, atol=2e-6)

    def test_feature_heads_identical_available_data_frozen_and_finite_empty_support(self):
        for kind in self.f.FEATURE_KINDS:
            head = self.f.build_head(kind, channels=7)
            current = torch.randn(2, 7, 4, 4, requires_grad=True)
            forecast = torch.randn(2, 7, 4, 4, requires_grad=True)
            stack = torch.rand(2, 1, 4, 4, requires_grad=True)
            support = torch.ones(2, 1, 4, 4, dtype=torch.bool)
            support[1] = False
            support[0, :, 1, 2] = False
            logits = self.f.feature_logits(head, current, forecast, support, stack)
            loss = self.f.masked_building_bce(logits, torch.zeros_like(logits), support, support)
            loss.backward()
            self.assertTrue(bool(torch.isfinite(logits).all()))
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters()))
            self.assertIsNone(current.grad)
            self.assertIsNone(forecast.grad)
            self.assertIsNone(stack.grad)
            probability, resized_support = self.f.feature_prediction(head, current, forecast, support, stack, (13, 15))
            self.assertEqual(tuple(probability.shape), (2, 1, 13, 15))
            self.assertFalse(bool(resized_support[1].any()))
            self.assertTrue(bool((probability[~resized_support] == 0).all()))
        left = self.f.head_contract(self.f.build_head("feature-mlp", 7))
        right = self.f.head_contract(self.f.build_head("cross-attention", 7))
        for field in ("feature_inputs", "order", "feature_native_grid", "score_clip", "frozen_base"):
            self.assertEqual(left[field], right[field])

    def test_feature_invalid_values_and_other_buildings_do_not_leak(self):
        current, forecast = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)
        support = torch.ones(2, 1, 4, 4, dtype=torch.bool)
        support[:, :, 1:3, 1:3] = False
        stack = torch.rand(2, 1, 4, 4)
        for kind in self.f.FEATURE_KINDS:
            head = self.f.build_head(kind, channels=3)
            baseline = self.f.feature_logits(head, current, forecast, support, stack)
            changed_current, changed_forecast, changed_stack = current.clone(), forecast.clone(), stack.clone()
            changed_current[~support.expand_as(current)] = float("nan")
            changed_forecast[~support.expand_as(forecast)] = float("inf")
            changed_stack[~support] = float("nan")
            actual = self.f.feature_logits(head, changed_current, changed_forecast, support, changed_stack)
            torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
            changed_current[0], changed_forecast[0], changed_stack[0] = 1000., -1000., .999
            actual = self.f.feature_logits(head, changed_current, changed_forecast, support, changed_stack)
            torch.testing.assert_close(actual[1], baseline[1], rtol=0, atol=0)

    def test_all_empty_support_has_zero_differentiable_loss(self):
        for kind in self.f.KINDS:
            head = self.f.build_head(kind, channels=3)
            support = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
            if kind in self.f.SCORE_KINDS:
                logits = self.f.score_logits(head, torch.rand(1, 1, 4, 4), torch.rand(1, 1, 4, 4), support)
            else:
                logits = self.f.feature_logits(head, torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4), support,
                    torch.rand(1, 1, 4, 4))
            loss = self.f.masked_building_bce(logits, torch.full_like(logits, 255), support, support)
            self.assertEqual(float(loss.detach()), 0.)
            loss.backward()
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters()))

    def test_equal_building_loss_and_known_only_in_loss(self):
        logits = torch.tensor([0., 0., 2.]).reshape(3, 1, 1, 1).requires_grad_()
        target = torch.zeros_like(logits)
        support = torch.ones_like(logits, dtype=torch.bool)
        actual = self.f.masked_building_bce(logits, target, support, support, ["a", "a", "b"])
        expected = (torch.nn.functional.softplus(logits[0]) + torch.nn.functional.softplus(logits[2])) / 2
        torch.testing.assert_close(actual, expected.squeeze())
        known = support.clone()
        known[2] = False
        actual = self.f.masked_building_bce(logits, target, known, support, ["a", "a", "b"])
        torch.testing.assert_close(actual, torch.nn.functional.softplus(logits[0]).squeeze())

    def test_export_restore_exact_and_preserves_rng(self):
        for kind in self.f.KINDS:
            head = self.f.build_head(kind, channels=5)
            payload = json.loads(json.dumps(self.f.serialize_head(head)))
            before = torch.get_rng_state().clone()
            restored = self.f.restore_head(payload)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            for name, value in head.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
            self.assertEqual(self.f.parameter_count(restored), payload["contract"]["parameters"])
        payload["contract"]["order"] = "average nonlinear crop outputs"
        with self.assertRaisesRegex(ValueError, "order"):
            self.f.restore_head(payload)

    def test_fixed_training_budget(self):
        self.assertEqual(self.f.fusion_training_contract()["epochs"], 5)
        self.assertEqual(self.f.fusion_training_contract(10)["epochs"], 10)
        for epochs in (0, 11, 5.5):
            with self.assertRaises(ValueError):
                self.f.fusion_training_contract(epochs)

    def test_reduced_matrix_has_only_declared_full_and_short_fits(self):
        training = importlib.import_module("facade_change.research.2026-11-08_history_scenarios_training")
        config = dict(matrix_protocol="reduced-v2")
        primary = training.resolve_matrix(config)
        fusion = training.resolve_fusion_matrix(config)
        self.assertEqual(len(primary), 18)
        self.assertEqual(len({r["id"] for r in primary + fusion}), 74)
        self.assertEqual(sum(r["fit_kind"] == "score-fusion" for r in fusion), 48)
        self.assertEqual(sum(r["fit_kind"] == "feature-fusion" for r in fusion), 8)
        self.assertEqual({r["size"] for r in primary if r["scenario"] == "main"}, {"large"})
        self.assertEqual({r["control"] for r in primary if r["scenario"] == "B"}, {"history", "regularized"})
        self.assertFalse(any(r["scenario"] == "C" or "residual" in r["parent_method"] for r in fusion))
        self.assertTrue(all(r["task"] == "h0" for r in fusion if r["fit_kind"] == "feature-fusion"))
        config["fpr_budgets"] = [.05, .10, .15]
        self.assertEqual(training.resolve_fusion_matrix(config), fusion)

    def test_canonical_overlap_inputs_anychange_or_and_unknown_masks(self):
        class Data:
            categories = [dict(name="damage")]
            groups = []

            def target(self, event_id):
                target = np.zeros((4, 5), np.uint8)
                target[:, 3] = 255
                return dict(target=target, class_targets=target[None], support=np.ones_like(target, bool),
                    before=np.ones_like(target, np.uint32), after=np.ones_like(target, np.uint32))

        samples = [dict(event_id=str(i), building_id="b", view_id="v", previous_year=2000,
            current_year=2010, bbox_xywh=[3 * i, 0, 5, 4], known_pixels=16) for i in range(2)]
        contract = dict(method_names=["anychange", "distance"], normalizer=dict(low=[0., 0.], high=[1., 1.]),
            model=dict(parameters=[-1., 2., .3], center=[0., 0.], scale=[1., 1.]))
        calls = []

        def predict(row):
            calls.append(row["event_id"])
            self.assertEqual(row["bbox_xywh"], samples[int(row["event_id"])]["bbox_xywh"])
            i = int(row["event_id"])
            return dict(stack_raw=np.stack((np.full((4, 5), i, np.float32), np.full((4, 5), .5, np.float32))),
                stack_contract=contract, score=np.full((4, 5), .2 + .6 * i, np.float32),
                support=np.ones((4, 5), bool))

        blocks = list(self.f.iter_score_blocks(Data(), samples, predict, halo=2, block_size=4))
        self.assertEqual(len(blocks), 2)
        self.assertEqual(calls, ["0", "1"])
        first = blocks[0][1]
        self.assertEqual(first["origin_xy"], (-2, -2))
        # Global x=3 lies in both crops: binary channel OR=1, method mean=.5.
        self.assertAlmostEqual(float(first["method"][0, 2, 5]), .5)
        self.assertAlmostEqual(float(first["stack"][0, 2, 5]), 1 / (1 + np.exp(-1.15)), places=6)
        # Unknown annotation does not remove geometric inputs.
        self.assertTrue(first["support"][0, 2, 5])
        np.testing.assert_array_equal(first["method"][0, :, 4:8], blocks[1][1]["method"][0, :, :4])

        # Canonical CNN output equals one complete merged raster including
        # neighbouring pixels at a block boundary, even where labels are unknown.
        whole_stack = np.zeros((1, 1, 12, 16), np.float32)
        whole_method = np.zeros_like(whole_stack)
        whole_support = np.zeros_like(whole_stack, dtype=bool)
        for (_, (bx, by)), block in blocks:
            sy, sx = block["core_slices"]
            destination = (slice(None), slice(None), slice(2 + by * 4, 6 + by * 4), slice(2 + bx * 4, 6 + bx * 4))
            whole_stack[destination] = block["stack"][:, sy, sx]
            whole_method[destination] = block["method"][:, sy, sx]
            whole_support[destination] = block["support"][:, sy, sx]
        head = self.f.build_head("score-cnn")
        reference = self.f.score_prediction(head, torch.from_numpy(whole_stack),
            torch.from_numpy(whole_method), torch.from_numpy(whole_support))
        for (_, (bx, by)), block in blocks:
            actual = self.f.apply_score_tile(head, torch.from_numpy(block["stack"])[None],
                torch.from_numpy(block["method"])[None], torch.from_numpy(block["support"])[None], block["core_slices"])
            torch.testing.assert_close(actual, reference[..., 2 + by * 4:6 + by * 4, 2 + bx * 4:6 + bx * 4],
                rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
