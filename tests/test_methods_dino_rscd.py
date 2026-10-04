"""Geometry and original preprocessing/loading contracts without real weights."""
import argparse
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from facade_change.methods.base import pad_rgb_pair
from facade_change.methods.dinov2 import (DINOv2Scorer, _assert_package_source,
                                         _checkpoint, _load_dino, _source_provenance)
from facade_change.methods.rscd import RSCDScorer


if torch is not None:
    class FakeDino(torch.nn.Module):
        patch_size = 14

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.inputs = []

        def forward_features(self, image):
            self.inputs.append(image.detach().clone())
            pooled = torch.nn.functional.avg_pool2d(image, 14, 14)
            values = pooled.flatten(2).transpose(1, 2)
            return {"x_norm_patchtokens": values * self.weight}

    class FakeRSCD(torch.nn.Module):
        def __init__(self, dino):
            super().__init__()
            self.backbone = dino
            for parameter in dino.parameters():
                parameter.requires_grad_(False)
            self.head = torch.nn.Linear(1, 1, bias=False)
            self.upsample = torch.nn.Upsample(size=(504, 504), mode="bilinear")
            self.inputs = []

        def forward(self, image0, image1):
            self.inputs = [image0.detach().clone(), image1.detach().clone()]
            height, width = self.upsample.size
            y = torch.arange(height).reshape(height, 1)
            x = torch.arange(width).reshape(1, width)
            changed = (y + x - height) * self.head.weight[0, 0] / 100
            return torch.stack([torch.zeros_like(changed), changed], dim=-1).unsqueeze(0)


@unittest.skipIf(torch is None, "Optional torch dependency is unavailable")
class DinoRSCDTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dino_root = self.root / "dinov2"
        self.dino_root.mkdir()
        (self.dino_root / "hubconf.py").write_text("# fake local source\n")
        self.weight = self.root / "dinov2_vits14_pretrain.pth"
        torch.save(FakeDino().state_dict(), self.weight)
        y, x = np.indices((256, 256))
        self.reference = np.stack([x % 256, y % 256, (x + y) % 256], axis=2).astype(np.uint8)
        self.source = np.flip(self.reference, axis=1).copy()
        self.support = np.ones((256, 256), dtype=bool)
        self.support[:3, :4] = False

    def test_dino_loader_is_local_strict_and_does_not_download(self):
        model = FakeDino()
        with mock.patch("torch.hub.load", return_value=model) as hub:
            loaded, metadata = _load_dino(self.dino_root, self.weight)
        self.assertIs(loaded, model)
        hub.assert_called_once_with(str(self.dino_root), "dinov2_vits14",
                                    source="local", pretrained=False)
        self.assertEqual(metadata["checkpoint_loading"], "weights_only")
        self.assertEqual(metadata["checkpoint_keys"], 1)
        torch.save({"wrong_key": torch.ones(1)}, self.weight)
        with mock.patch("torch.hub.load", return_value=FakeDino()):
            with self.assertRaises(RuntimeError):
                _load_dino(self.dino_root, self.weight)

    def test_dino_geometry_normalization_and_frozen_native_grid(self):
        fake = FakeDino()
        with mock.patch("torch.hub.load", return_value=fake):
            scorer = DINOv2Scorer(source_root=self.dino_root, checkpoint_path=self.weight,
                                  device="cpu")
        actual = scorer(self.reference, self.source, self.support)
        padded0, padded1, padding = pad_rgb_pair(self.reference, self.source)
        mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
        std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
        features = []
        for image, seen in zip((padded0, padded1), fake.inputs):
            expected = (torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255 - mean) / std
            torch.testing.assert_close(seen, expected)
            features.append(torch.nn.functional.avg_pool2d(expected, 14, 14).flatten(2).transpose(1, 2))
        token_map = (1 - torch.nn.functional.cosine_similarity(*features, dim=-1)) / 2
        full = torch.nn.functional.interpolate(token_map.reshape(1, 1, 19, 19),
                                                size=(266, 266), mode="bilinear",
                                                align_corners=False)[0, 0].clamp(0, 1).numpy()
        expected_native = full[5:261, 5:261]
        np.testing.assert_allclose(actual[self.support], expected_native[self.support], atol=1e-7)
        self.assertEqual(actual.shape, (256, 256))
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(np.isnan(actual[~self.support]).all())
        self.assertFalse(fake.training)
        self.assertFalse(any(parameter.requires_grad for parameter in fake.parameters()))
        self.assertEqual(scorer.metadata["last_padding"], padding.as_dict())
        self.assertIsNone(scorer.raw_scores)
        self.assertIsNone(scorer.native_prediction)
        scorer.close()
        self.assertIsNone(scorer.model)

    def test_dino_rejects_wrong_backbone_patch_size(self):
        fake = FakeDino()
        fake.patch_size = 16
        with mock.patch("torch.hub.load", return_value=fake):
            with self.assertRaisesRegex(ValueError, "patch size 14"):
                _load_dino(self.dino_root, self.weight)

    def test_restricted_checkpoint_loading_requires_explicit_trust(self):
        path = self.root / "namespace.pth"
        torch.save({"args": argparse.Namespace(value=3)}, path)
        with self.assertRaisesRegex(RuntimeError, "trust_checkpoint=True"):
            _checkpoint(path)
        record, mode = _checkpoint(path, trust_checkpoint=True)
        self.assertEqual(record["args"].value, 3)
        self.assertEqual(mode, "trusted_pickle")

    def test_numpy_metadata_is_allowed_without_unrestricted_pickle(self):
        path = self.root / "numpy.pth"
        torch.save({"metric": np.float64(.5), "model": {"head": torch.ones(1)}}, path)
        record, mode = _checkpoint(path, numpy_metadata=True)
        self.assertEqual(record["metric"], .5)
        self.assertEqual(mode, "weights_only_numpy_metadata")

    def test_numpy_compatibility_namespace_without_multiarray_does_not_block_loading(self):
        path = self.root / "numpy-compatibility.pth"
        torch.save({"metric": np.float64(.5), "model": {"head": torch.ones(1)}}, path)
        before = list(torch.serialization.get_safe_globals())
        # NumPy 1.26 has a _core compatibility package whose presence does not
        # guarantee a multiarray attribute. Exercise that exact failure shape.
        with mock.patch.object(np, "_core", types.SimpleNamespace(), create=True):
            record, mode = _checkpoint(path, numpy_metadata=True)
        self.assertEqual(record["metric"], .5)
        torch.testing.assert_close(record["model"]["head"], torch.ones(1))
        self.assertEqual(mode, "weights_only_numpy_metadata")
        self.assertEqual(torch.serialization.get_safe_globals(), before)

    def test_old_torch_never_implicitly_uses_unrestricted_loading(self):
        def old_load(path, **kwargs):
            if "weights_only" in kwargs:
                raise TypeError("unexpected keyword weights_only")
            return {"value": 4}
        with mock.patch("torch.load", side_effect=old_load) as loading:
            with self.assertRaisesRegex(RuntimeError, "trust_checkpoint=True"):
                _checkpoint(self.weight)
            self.assertEqual(loading.call_count, 1)
            record, mode = _checkpoint(self.weight, trust_checkpoint=True)
        self.assertEqual(record["value"], 4)
        self.assertEqual(mode, "trusted_pickle")

    def make_rscd(self, method="rscd_cmu", invalid=None, unused=False):
        root = self.root / "rscd"
        root.mkdir(exist_ok=True)
        (root / "model.py").write_text("# original reference source\n")
        utils_root = self.root / "py_utils"
        utils_root.mkdir(exist_ok=True)
        (utils_root / "utils_torch.py").write_text("# original reference source\n")
        head_path = self.root / "rscd.pth"
        args = {"name": "dino2 + cross_attention", "dino-model": "dinov2_vits14",
                "freeze-dino": True, "unfreeze-dino-last-n-layer": 0,
                "num-blocks": 1, "target-shp-row": 504, "target-shp-col": 504}
        if invalid:
            args.update(invalid)
        torch.save({"args": {"model": args}, "model": {"module.head.weight": torch.tensor([[2.]])}}, head_path)
        dino = FakeDino()
        backbone = types.SimpleNamespace(_get_dino=lambda: "original")
        original_loader = backbone._get_dino
        created = []
        def get_model(**kwargs):
            self.assertEqual(kwargs, args)
            model = FakeRSCD(backbone._get_dino())
            created.append(model)
            return model
        def load_state(model, state, verbose, return_details):
            self.assertFalse(verbose)
            self.assertTrue(return_details)
            self.assertFalse(model.module.backbone.weight.requires_grad)
            self.assertTrue(model.module.head.weight.requires_grad)
            model.module.head.load_state_dict({"weight": state["module.head.weight"]}, strict=True)
            return model, {"unexpected": torch.ones(1)} if unused else {}
        models = types.SimpleNamespace(get_model=get_model)
        utils = types.SimpleNamespace(load_grad_required_state=mock.Mock(side_effect=load_state))
        with mock.patch("facade_change.methods.rscd._load_dino", return_value=(dino, {})), \
                mock.patch("facade_change.methods.rscd._local_imports", return_value=(models, backbone, utils)):
            scorer = RSCDScorer(method=method, source_root=root, dino_root=self.dino_root,
                                py_utils_root=utils_root, checkpoint_path=head_path,
                                dino_checkpoint=self.weight, device="cpu")
        self.assertIs(backbone._get_dino, original_loader)
        return scorer, created[0], utils, args

    def test_rscd_uses_author_head_loader_then_freezes_all_parameters(self):
        scorer, model, utils, args = self.make_rscd()
        self.assertEqual(utils.load_grad_required_state.call_count, 1)
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
        self.assertFalse(model.training)
        self.assertEqual(scorer.metadata["checkpoint_model_args"], args)
        self.assertEqual(scorer.metadata["checkpoint_key_audit"], {"head_keys": 1, "unused_keys": []})

    def test_rscd_probability_and_author_mask_are_cropped_to_native_grid(self):
        scorer, model, _, _ = self.make_rscd("rscd_diff_cmu")
        actual = scorer(self.reference, self.source, self.support)
        for image, seen in zip(pad_rgb_pair(self.reference, self.source)[:2], model.inputs):
            expected = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255
            torch.testing.assert_close(seen, expected)
        y, x = np.indices((256, 256))
        changed = (y + 5 + x + 5 - 266) * 2 / 100
        expected = 1 / (1 + np.exp(-changed))
        np.testing.assert_allclose(actual[self.support], expected[self.support], atol=1e-7)
        np.testing.assert_array_equal(scorer.native_prediction, (changed > 0) & self.support)
        self.assertEqual(scorer.metadata["method"], "rscd_diff_cmu")
        self.assertEqual(model.upsample.size, (266, 266))
        self.assertEqual(scorer.native_prediction.dtype, np.bool_)
        self.assertTrue(np.isnan(actual[~self.support]).all())
        scorer.close()
        self.assertIsNone(scorer.native_prediction)

    def test_rscd_three_checkpoint_variants_preserve_original_model_args(self):
        for method in ("rscd_cmu", "rscd_diff_cmu", "rscd_pscd"):
            with self.subTest(method=method):
                scorer, _, _, _ = self.make_rscd(method)
                self.assertEqual(scorer.metadata["method"], method)
        with self.assertRaisesRegex(ValueError, "different DINO backbone"):
            self.make_rscd(invalid={"dino-model": "dinov2_vitb14"})
        with self.assertRaisesRegex(ValueError, "frozen RSCD"):
            self.make_rscd(invalid={"freeze-dino": False})

    def test_rscd_rejects_unconsumed_checkpoint_head_keys(self):
        with self.assertRaisesRegex(ValueError, "Unused RSCD checkpoint keys"):
            self.make_rscd(unused=True)

    def test_source_fingerprint_detects_content_changes_and_import_conflicts(self):
        original = _source_provenance(self.dino_root)
        (self.dino_root / "hubconf.py").write_text("# changed source\n")
        self.assertNotEqual(original["python_tree_sha256"],
                            _source_provenance(self.dino_root)["python_tree_sha256"])
        foreign = types.ModuleType("dinov2")
        foreign.__path__ = [str(self.root / "foreign")]
        with mock.patch.dict("sys.modules", {"dinov2": foreign}):
            with self.assertRaisesRegex(RuntimeError, "Conflicting imported"):
                _assert_package_source("dinov2", self.dino_root / "dinov2")


if __name__ == "__main__":
    unittest.main()
