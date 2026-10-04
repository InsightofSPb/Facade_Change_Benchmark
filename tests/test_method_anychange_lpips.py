"""CPU contracts for imported methods without downloading model weights."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from facade_change.methods.anychange import AnyChangeScorer, _construct_model, _union_masks
from facade_change.methods.lpips import LPIPSScorer, _local_weights, distance_to_scores

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch unavailable")
class AnyChangeContractTests(unittest.TestCase):
    def scorer(self, forward):
        model = SimpleNamespace(clears=0, cached=None)
        def clear():
            model.clears += 1
            model.cached = None
        model.clear_cached_embedding = clear
        model.forward = lambda a, b: forward(model, a, b)
        scorer = AnyChangeScorer.__new__(AnyChangeScorer)
        scorer.model = model
        scorer.decode_rle = lambda rle: rle
        scorer.native_prediction = None
        scorer.raw_scores = None
        return scorer

    def test_sequential_pairs_never_reuse_embeddings_and_union_native_masks(self):
        seen = []
        def forward(model, a, b):
            self.assertIsNone(model.cached)
            model.cached = int(b[0, 0, 0])
            seen.append(model.cached)
            one = np.zeros(a.shape[:2], bool)
            two = np.zeros(a.shape[:2], bool)
            one[0, 0] = True
            two[1, 1] = model.cached > 0
            return {"rles": [one, two]}, None, None
        scorer = self.scorer(forward)
        reference = np.zeros((3, 4, 3), np.uint8)
        support = np.ones((3, 4), bool)
        support[0, 3] = False
        first = scorer(reference, reference.copy(), support)
        second = scorer(reference, np.ones_like(reference), support)
        self.assertEqual(seen, [0, 1])
        self.assertEqual(scorer.model.clears, 4)
        self.assertIsNone(scorer.model.cached)
        self.assertEqual(first[1, 1], 0)
        self.assertEqual(second[1, 1], 1)
        self.assertTrue(np.isnan(second[0, 3]))
        self.assertEqual(second.dtype, np.float32)
        self.assertEqual(scorer.native_prediction.dtype, np.bool_)
        self.assertTrue(scorer.native_prediction[0, 0])

    def test_cache_is_cleared_when_author_forward_fails(self):
        def fail(model, a, b):
            model.cached = "partial embedding"
            raise RuntimeError("author inference failure")
        scorer = self.scorer(fail)
        image = np.zeros((3, 4, 3), np.uint8)
        with self.assertRaisesRegex(RuntimeError, "author inference"):
            scorer(image, image, np.ones(image.shape[:2], bool))
        self.assertEqual(scorer.model.clears, 2)
        self.assertIsNone(scorer.model.cached)
        self.assertIsNone(scorer.native_prediction)

    def test_invalid_mask_grid_fails_instead_of_resizing(self):
        with self.assertRaisesRegex(RuntimeError, "original RGB grid"):
            _union_masks({"rles": [np.zeros((2, 3), bool)]}, lambda x: x, (3, 4))
        np.testing.assert_array_equal(
            _union_masks({"rles": []}, lambda x: x, (3, 4)), np.zeros((3, 4), bool))

    def test_support_is_not_passed_to_author_model(self):
        scorer = self.scorer(lambda model, a, b: ({"rles": []}, None, None))
        image = np.zeros((3, 4, 3), np.uint8)
        with self.assertRaisesRegex(ValueError, "boolean"):
            scorer(image, image, np.ones((3, 4), np.uint8))
        self.assertEqual(scorer.model.clears, 0)

    def test_requested_device_and_neck_transform_share_same_sam(self):
        class SAM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.image_encoder = torch.nn.Module()
                self.image_encoder.neck = torch.nn.ModuleList(
                    [torch.nn.Identity() for _ in range(3)] + [torch.nn.LayerNorm(2)])
        sam = SAM()
        state = {key: value.clone() for key, value in sam.state_dict().items()}
        state["image_encoder.neck.3.weight"] = torch.tensor([2., 4.])
        state["image_encoder.neck.3.bias"] = torch.tensor([1., 2.])
        class Author:
            def __init__(self, *args, **kwargs):
                raise AssertionError("Author auto-device constructor must not run")
            def set_hyperparameters(self, **settings):
                self.change_settings = settings
        modules = (SimpleNamespace(AnyChange=Author),
                   SimpleNamespace(sam_model_registry={"vit_h": lambda checkpoint=None: sam}),
                   SimpleNamespace(SimpleMaskGenerator=lambda model, **kw: SimpleNamespace(sam=model, settings=kw)),
                   SimpleNamespace(rle_to_mask=lambda x: x))
        with patch("facade_change.methods.anychange._author_modules", return_value=modules), \
             patch("facade_change.methods.anychange._safe_state", return_value=state):
            model, decoder = _construct_model(Path("unused"), Path("unused"), "cpu", False, 16)
        self.assertEqual(model.device, torch.device("cpu"))
        self.assertIs(model.maskgen.sam, sam)
        self.assertEqual(model.maskgen.settings["points_per_batch"], 16)
        self.assertEqual(model.change_settings["change_confidence_threshold"], 145)
        self.assertFalse(model.change_settings["auto_threshold"])
        np.testing.assert_allclose(model.inv_transform(torch.tensor([[[5.]], [[10.]]])).numpy(), [[[2.]], [[2.]]])
        self.assertTrue(all(not p.requires_grad for p in sam.parameters()))


class LPIPSScoreTests(unittest.TestCase):
    def test_fixed_distance_mapping_is_not_per_pair_normalized(self):
        first = distance_to_scores(np.array([[0., 1., 3.]], np.float32))
        second = distance_to_scores(np.array([[0., 1., 100.]], np.float32))
        np.testing.assert_array_equal(first[0, :2], second[0, :2])
        np.testing.assert_allclose(first, [[0., .5, .75]])
        self.assertEqual(first.dtype, np.float32)
        for bad in (np.array([1.]), np.array([[-1.]]), np.array([[np.nan]])):
            with self.assertRaises(RuntimeError):
                distance_to_scores(bad)


@unittest.skipIf(torch is None, "PyTorch unavailable")
class LPIPSContractTests(unittest.TestCase):
    def test_native_rgb_normalization_raw_distances_and_support(self):
        received = []
        class FakeLPIPS:
            def __call__(self, a, b, normalize=False):
                received.append((a.clone(), b.clone(), normalize))
                return torch.tensor([[[[0., 1.], [3., 8.]]]])
        scorer = LPIPSScorer.__new__(LPIPSScorer)
        scorer.device = "cpu"
        scorer.model = FakeLPIPS()
        scorer.raw_scores = scorer.native_prediction = None
        reference = np.zeros((2, 2, 3), np.uint8)
        source = np.full_like(reference, 255)
        support = np.array([[True, True], [True, False]])
        score = scorer(reference, source, support)
        a, b, normalize = received[0]
        self.assertEqual(tuple(a.shape), (1, 3, 2, 2))
        self.assertTrue(torch.all(a == -1))
        self.assertTrue(torch.all(b == 1))
        self.assertFalse(normalize)
        np.testing.assert_allclose(score[support], [0., .5, .75])
        np.testing.assert_allclose(scorer.raw_scores[support], [0., 1., 3.])
        self.assertTrue(np.isnan(score[1, 1]))
        self.assertTrue(np.isnan(scorer.raw_scores[1, 1]))
        self.assertIsNone(scorer.native_prediction)

    def test_grid_mismatch_rejected(self):
        scorer = LPIPSScorer.__new__(LPIPSScorer)
        scorer.device = "cpu"
        scorer.model = lambda a, b, **kw: torch.zeros(1, 1, 3, 3)
        image = np.zeros((4, 4, 3), np.uint8)
        with self.assertRaisesRegex(RuntimeError, "unchanged native"):
            scorer(image, image, np.ones((4, 4), bool))
        self.assertIsNone(scorer.raw_scores)

    def test_missing_local_weights_fail_before_model_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            package = SimpleNamespace(__file__=str(Path(directory) / "__init__.py"))
            with patch("torch.hub.get_dir", return_value=directory):
                with self.assertRaisesRegex(FileNotFoundError, "automatic downloads disabled"):
                    _local_weights(package, None, None)
                backbone = Path(directory) / "backbone.pth"
                calibration = Path(directory) / "calibration.pth"
                backbone.write_bytes(b"local backbone")
                calibration.write_bytes(b"local calibration")
                self.assertEqual(_local_weights(package, backbone, calibration),
                                 (backbone.resolve(), calibration.resolve()))

    def test_all_learned_tensors_use_pretrained_state_and_no_random_fallback(self):
        from facade_change.methods.lpips import _load_pretrained_states
        class MiniLPIPS(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Module()
                for i in range(5):
                    sequence = torch.nn.Sequential()
                    sequence.add_module(str(i), torch.nn.Conv2d(1, 1, 1))
                    setattr(self.net, f"slice{i + 1}", sequence)
                    linear = torch.nn.Module()
                    linear.model = torch.nn.Sequential(torch.nn.Dropout(), torch.nn.Conv2d(1, 1, 1, bias=False))
                    setattr(self, f"lin{i}", linear)
                self.lins = torch.nn.ModuleList([getattr(self, f"lin{i}") for i in range(5)])
        model = MiniLPIPS()
        backbone = {}
        for i in range(5):
            backbone[f"features.{i}.weight"] = torch.full((1, 1, 1, 1), i + 1.)
            backbone[f"features.{i}.bias"] = torch.full((1,), i + 1.)
        calibration = {f"lin{i}.model.1.weight": torch.full((1, 1, 1, 1), i + .5) for i in range(5)}
        _load_pretrained_states(model, backbone, calibration)
        for name, parameter in model.named_parameters():
            expected = calibration[name] if name in calibration else backbone["features." + ".".join(name.split(".")[2:])]
            self.assertTrue(torch.equal(parameter, expected), name)
        with self.assertRaisesRegex(RuntimeError, "feature layout"):
            _load_pretrained_states(model, {**backbone, "features.99.weight": torch.ones(1)}, calibration)
        bad = dict(calibration)
        bad.pop("lin3.model.1.weight")
        with self.assertRaisesRegex(RuntimeError, "five official"):
            _load_pretrained_states(model, backbone, bad)


if __name__ == "__main__":
    unittest.main()
