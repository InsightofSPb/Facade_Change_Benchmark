"""CPU contracts for the frozen feature extractor and masked supervision."""
import importlib.util
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from facade_change.segmentation.model import FrozenDinoSegmenter, masked_multilabel_loss

    class TinyBackbone(nn.Module):
        patch_size = 14
        embed_dim = 6

        def __init__(self):
            super().__init__()
            self.project = nn.Conv2d(3, self.embed_dim, kernel_size=14, stride=14)
            self.dropout = nn.Dropout(0.5)
            self.last_input_shape = None

        def get_intermediate_layers(self, value, n, reshape):
            assert reshape
            self.last_input_shape = tuple(value.shape)
            features = self.dropout(self.project(value))
            return tuple(features * (index + 1) for index in range(n))


@unittest.skipUnless(HAS_TORCH, "optional torch dependency is unavailable")
class FrozenDinoModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_forward_backward_freezes_backbone_and_restores_shape(self):
        model = FrozenDinoSegmenter(TinyBackbone(), 3).train()
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.head.training)
        before = {key: tensor.clone() for key, tensor in model.backbone.state_dict().items()}
        rgb = torch.rand(2, 3, 29, 35)
        logits = model(rgb)
        self.assertEqual(tuple(logits.shape), (2, 3, 29, 35))
        self.assertEqual(model.backbone.last_input_shape, (2, 3, 42, 42))
        target = torch.zeros_like(logits)
        target[:, 0] = 1
        loss = masked_multilabel_loss(logits, target, torch.ones(2, 29, 35, dtype=torch.bool))
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.backbone.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.backbone.parameters()))
        self.assertTrue(all(parameter.grad is not None for parameter in model.head.parameters()))
        for key, value in model.backbone.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))

    def test_output_coordinates_upsample_before_removing_padding(self):
        model = FrozenDinoSegmenter(TinyBackbone(), 2).eval()
        rgb = torch.rand(1, 3, 29, 35)
        with torch.no_grad():
            actual = model(rgb)
            normalized = (rgb - model.rgb_mean) / model.rgb_std
            padded = F.pad(normalized, (0, 7, 0, 13), mode="reflect")
            features = model.backbone.get_intermediate_layers(padded, n=4, reshape=True)
            expected = F.interpolate(model.head(torch.cat(features, dim=1)),
                                     size=(42, 42), mode="bilinear", align_corners=False)[..., :29, :35]
        torch.testing.assert_close(actual, expected)

    def test_tiny_dimension_uses_replication_and_valid_shape(self):
        model = FrozenDinoSegmenter(TinyBackbone(), 2).eval()
        with torch.no_grad():
            logits = model(torch.rand(1, 3, 1, 2))
        self.assertEqual(tuple(logits.shape), (1, 2, 1, 2))
        self.assertTrue(torch.isfinite(logits).all())

    def test_loss_ignores_unknown_pixels_and_preserves_overlap(self):
        logits = torch.randn(1, 3, 4, 4, requires_grad=True)
        targets = torch.zeros_like(logits)
        targets[:, :2, 1:3, 1:3] = 1  # Two classes share the same pixels.
        valid = torch.zeros(1, 4, 4, dtype=torch.bool)
        valid[:, 1:3, 1:3] = True
        changed = logits.detach().clone()
        changed.masked_fill_(~valid.unsqueeze(1), 50)
        loss = masked_multilabel_loss(logits, targets, valid)
        torch.testing.assert_close(loss, masked_multilabel_loss(changed, targets, valid))
        loss.backward()
        self.assertTrue((logits.grad.masked_select(~valid.unsqueeze(1))).eq(0).all())
        self.assertTrue((logits.grad[:, :2, 1:3, 1:3] < 0).all())
        self.assertTrue((logits.grad[:, 2, 1:3, 1:3] > 0).all())

    def test_empty_support_has_zero_loss_and_gradient(self):
        logits = torch.randn(2, 3, 4, 4, requires_grad=True)
        loss = masked_multilabel_loss(logits, torch.zeros_like(logits),
                                     torch.zeros(2, 1, 4, 4, dtype=torch.bool))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_head_can_learn_a_small_multilabel_batch(self):
        model = FrozenDinoSegmenter(TinyBackbone(), 3).train()
        rgb = torch.rand(2, 3, 28, 28)
        targets = torch.zeros(2, 3, 28, 28)
        targets[:, :2] = 1
        valid = torch.ones(2, 28, 28, dtype=torch.bool)
        optimizer = torch.optim.Adam(model.head.parameters(), lr=0.05)
        initial = masked_multilabel_loss(model(rgb), targets, valid).item()
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            loss = masked_multilabel_loss(model(rgb), targets, valid)
            loss.backward()
            optimizer.step()
        final = masked_multilabel_loss(model(rgb), targets, valid).item()
        self.assertLess(final, initial * 0.6)


if __name__ == "__main__":
    unittest.main()
