"""Bounded CPU checks of real reference-only conditioning and byte alignment."""
import importlib
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from facade_change.research.context import ReferenceHead, bias_for_positions, extract_reference
    from facade_change.methods.base import pad_rgb_pair

    class FakeDino(torch.nn.Module):
        patch_size = 14

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.inputs = []

        def forward_features(self, values):
            self.inputs.append(values.detach().clone())
            pooled = torch.nn.functional.avg_pool2d(values, 14, 14)
            return {"x_norm_patchtokens": pooled.flatten(2).transpose(1, 2) * self.weight}


@unittest.skipIf(torch is None, "Optional torch dependency is unavailable")
class ReferenceContextTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.shape = (25, 39)
        self.features = torch.randn(3, 2, 3)
        self.reference = np.arange(25 * 39 * 3, dtype=np.uint16).reshape(25, 39, 3).astype(np.uint8)
        self.support = np.ones(self.shape, bool)
        self.support[:2, :3] = False

    def test_provider_is_reference_only_frozen_and_uses_existing_preparation(self):
        model = FakeDino()
        features, info = extract_reference(self.reference, self.support, model)
        known = self.reference.copy()
        known[~self.support] = 0
        padded, _, padding = pad_rgb_pair(known, known)
        expected = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float() / 255
        expected = (expected - torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1)) / torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1)
        torch.testing.assert_close(model.inputs[0], expected)
        self.assertEqual(tuple(features.shape), (3, 2, 3))
        self.assertEqual(info["padding"], padding.as_dict())
        self.assertFalse(features.requires_grad)
        self.assertFalse(model.training)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        changed = self.reference.copy()
        changed[~self.support] = 255
        again, _ = extract_reference(changed, self.support, model)
        torch.testing.assert_close(features, again, rtol=0, atol=0)
        with self.assertRaises(TypeError):
            extract_reference(self.reference, self.support, model, source=self.reference)

    def test_zero_heads_have_exact_original_logits_and_parameter_match(self):
        torch.manual_seed(7)
        spatial = ReferenceHead("spatial", 3, width=4, tokens=2)
        torch.manual_seed(7)
        spectral = ReferenceHead("spectral", 3, width=4, tokens=2)
        self.assertEqual(sum(p.numel() for p in spatial.parameters()), sum(p.numel() for p in spectral.parameters()))
        for a, b in zip(spatial.state_dict().values(), spectral.state_dict().values()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        positions = np.array([100, 0, 8, -1, 2, 2924], np.int64)
        original = torch.randn(len(positions), 256)
        for head in (spatial, spectral):
            bias = bias_for_positions(head, self.features, self.shape, positions)
            torch.testing.assert_close(bias, torch.zeros_like(bias), rtol=0, atol=0)
            torch.testing.assert_close(original + bias, original, rtol=0, atol=0)

    def test_native_positions_channels_padding_and_sparse_interpolation(self):
        head = ReferenceHead("spatial", 3, width=4, tokens=2)
        with torch.no_grad():
            head.logit_head.weight.normal_(std=.1)
            head.logit_head.bias.copy_(torch.arange(768) / 768)
        offsets = np.array([2924, 1, 80, 0, -1, 279, 2, 80], np.int64)
        native = head(self.features, self.shape).reshape(-1, 256)
        actual = bias_for_positions(head, self.features, self.shape, offsets)
        active = offsets >= 0
        torch.testing.assert_close(actual[active], native[offsets[active]], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(actual[~active], torch.zeros_like(actual[~active]), rtol=0, atol=0)
        for invalid in ([2925], [-2], [1.5], [True]):
            with self.assertRaises(ValueError):
                bias_for_positions(head, self.features, self.shape, invalid)

    def test_spectral_full_grid_is_finite_for_odd_even_and_dc_inputs(self):
        for grid in ((1, 1), (3, 5), (2, 4)):
            for values in (torch.zeros(3, *grid), torch.ones(3, *grid), torch.randn(3, *grid)):
                head = ReferenceHead("spectral", 3, width=4, tokens=2)
                with torch.no_grad():
                    head.logit_head.weight.normal_(std=.01)
                shape = (grid[0] * 14 - 1, grid[1] * 14 - 2)
                bias = bias_for_positions(head, values, shape, np.array([0, 2, shape[0] * shape[1] * 3 - 1]))
                self.assertTrue(torch.isfinite(bias).all())
                bias.square().mean().backward()
                for parameter in head.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_real_head_training_changes_only_head_and_uses_both_modes(self):
        model = FakeDino()
        features, _ = extract_reference(self.reference, self.support, model)
        frozen = {key: value.clone() for key, value in model.state_dict().items()}
        original = importlib.import_module("facade_change.2026-10-04_msdzip_h0")
        predictor = original._new_model(original._model_args(4, 2, 8, 16, 4), torch.device("cpu")).eval()
        for parameter in predictor.parameters():
            parameter.requires_grad_(False)
        predictor_before = {key: value.clone() for key, value in predictor.state_dict().items()}
        with torch.no_grad():
            originals = original._predict_groups(predictor, torch.randint(0, 256, (3, 4, 2))).reshape(12, 256)
        targets = torch.arange(12) % 256
        offsets = np.arange(12, dtype=np.int64)
        corrected = []
        for mode in ("spatial", "spectral"):
            torch.manual_seed(5)
            head = ReferenceHead(mode, 3, width=4, tokens=2)
            optimizer = torch.optim.Adam(head.parameters(), lr=.01)
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                bias = bias_for_positions(head, features, self.shape, offsets)
                torch.nn.functional.cross_entropy(originals + bias, targets).backward()
                optimizer.step()
            self.assertGreater(float(head.logit_head.weight.detach().abs().sum()), 0)
            corrected.append(bias_for_positions(head, features, self.shape, offsets).detach())
        self.assertFalse(torch.equal(corrected[0], corrected[1]))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for key, value in predictor.state_dict().items():
            torch.testing.assert_close(value, predictor_before[key], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in predictor.parameters()))

    def test_cached_grid_is_detached_eval_only_and_preserves_sparse_bias(self):
        head = ReferenceHead("spectral", 3, width=4, tokens=2).eval()
        with torch.no_grad():
            head.logit_head.weight.normal_(std=.01)
            grid = head.grid_logits(self.features)
        offsets = [0, 5, -1, 2924]
        actual = bias_for_positions(head, self.features, self.shape, offsets, grid_logits=grid)
        expected = bias_for_positions(head, self.features, self.shape, offsets)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        invalids = (grid.clone().requires_grad_(), grid[:, :, :-1], torch.full_like(grid, float("nan")),
                    grid.to(dtype=torch.float64), grid.to(device="meta"))
        for invalid in invalids:
            with self.assertRaises(ValueError):
                bias_for_positions(head, self.features, self.shape, offsets, grid_logits=invalid)
        head.train()
        for positions in (offsets, [-1]):
            with self.assertRaises(ValueError):
                bias_for_positions(head, self.features, self.shape, positions, grid_logits=grid)

    def test_byte_probabilities_do_not_see_target_or_future_source(self):
        original = importlib.import_module("facade_change.2026-10-04_msdzip_h0")
        args = original._model_args(4, 2, 8, 16, 4)
        predictor = original._new_model(args, torch.device("cpu")).eval()
        for parameter in predictor.parameters():
            parameter.requires_grad_(False)
        reference = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        source = np.flip(reference, axis=1).copy()
        support = np.ones((4, 5), bool)
        offset = 8
        changed = source.copy()
        changed.reshape(-1)[offset:] ^= 127
        dino = FakeDino()
        features, _ = extract_reference(reference, support, dino)
        head = ReferenceHead("spectral", 3, width=4, tokens=2)
        with torch.no_grad():
            head.logit_head.weight.normal_(std=.01)
        probabilities = []
        predictor_before = {key: value.clone() for key, value in predictor.state_dict().items()}
        for current in (source, changed):
            contexts, labels = original._windows(original.residual_bytes(reference, current, "mod256"), support, [offset], 2)
            batch, _, indices = next(original._batch_groups(contexts, labels, np.array([offset]), 4, 2))
            with torch.no_grad():
                logits = original._predict_groups(predictor, torch.as_tensor(batch))
                active = indices >= 0
                probabilities.append(logits[active] + bias_for_positions(head, features, (4, 5), [offset]))
        torch.testing.assert_close(probabilities[0], probabilities[1], rtol=0, atol=0)
        for key, value in predictor.state_dict().items():
            torch.testing.assert_close(value, predictor_before[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
