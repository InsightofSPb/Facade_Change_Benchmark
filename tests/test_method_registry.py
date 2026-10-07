"""Native-grid preservation and lazy common method entry-point contracts."""
import importlib.util
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from facade_change.methods.base import mask_scores, pad_rgb_pair, restore_map
from facade_change.methods.registry import EXTERNAL_METHODS, LEGACY_METHODS, make_method
from facade_change.scorers import make_scorer


class NativeGridPaddingTests(unittest.TestCase):
    def test_256_grid_preserved_with_symmetric_reflection_and_inverse_crop(self):
        reference = np.random.default_rng(7).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        source = 255 - reference
        before = reference.copy(), source.copy()
        reference.flags.writeable = source.flags.writeable = False
        padded_reference, padded_source, padding = pad_rgb_pair(reference, source)
        self.assertEqual(padded_reference.shape, (266, 266, 3))
        self.assertEqual((padding.top, padding.bottom, padding.left, padding.right), (5, 5, 5, 5))
        np.testing.assert_array_equal(padded_reference[5:261, 5:261], reference)
        np.testing.assert_array_equal(padded_source[5:261, 5:261], source)
        np.testing.assert_array_equal(padded_reference[0, 5:261], reference[5])
        np.testing.assert_array_equal(padded_reference[-1, 5:261], reference[-6])
        grid = np.arange(266 * 266).reshape(266, 266)
        cropped = restore_map(grid, padding)
        np.testing.assert_array_equal(cropped, grid[5:261, 5:261])
        self.assertEqual(cropped.shape, (256, 256))
        json.dumps(padding.as_dict(), allow_nan=False)
        np.testing.assert_array_equal(before[0], reference)
        np.testing.assert_array_equal(before[1], source)

    def test_divisible_odd_and_singleton_sizes_have_exact_native_inverse(self):
        for shape in ((28, 42), (255, 257), (1, 1), (1, 9)):
            rgb = np.arange(np.prod(shape) * 3, dtype=np.uint8).reshape(*shape, 3)
            padded, _, padding = pad_rgb_pair(rgb, rgb)
            self.assertEqual(padded.shape[0] % 14, 0)
            self.assertEqual(padded.shape[1] % 14, 0)
            self.assertLessEqual(abs(padding.top - padding.bottom), 1)
            self.assertLessEqual(abs(padding.left - padding.right), 1)
            np.testing.assert_array_equal(restore_map(padded[..., 0], padding), rgb[..., 0])
        with self.assertRaisesRegex(ValueError, "integer"):
            pad_rgb_pair(rgb, rgb, multiple=True)
        with self.assertRaisesRegex(ValueError, "full padded"):
            restore_map(np.zeros((14, 13)), padding)

    def test_output_masking_preserves_fixed_score_scale_and_rejects_invalid_values(self):
        scores = np.array([[.4, .7], [np.nan, .2]], np.float64)
        support = np.array([[True, True], [False, True]])
        restored = mask_scores(scores, support)
        self.assertEqual(restored.dtype, np.float32)
        np.testing.assert_array_equal(restored[support], scores[support].astype(np.float32))
        self.assertTrue(np.isnan(restored[~support]).all())
        for invalid in (np.inf, -.1, 1.01, np.nan):
            changed = scores.copy()
            changed[0, 0] = invalid
            with self.assertRaisesRegex(RuntimeError, "invalid supported"):
                mask_scores(changed, support)


class MethodRegistryTests(unittest.TestCase):
    def test_existing_entry_point_and_registry_preserve_codec_and_rgb_outputs(self):
        reference = np.arange(9 * 13 * 3, dtype=np.uint8).reshape(9, 13, 3)
        source = np.roll(reference, 1, axis=0)
        support = np.ones(reference.shape[:2], bool)
        support[:2, :3] = False
        methods = ["rgb_diff", "lzma_abs", "lzma_mod256"]
        if importlib.util.find_spec("cv2"):
            methods.append("ssim")
        if importlib.util.find_spec("zstandard"):
            methods.extend(("zstd_abs", "zstd_mod256"))
        for name in methods:
            with self.subTest(method=name):
                current = make_method(name, compression_tile_size=4, compression_stride=2)
                legacy = make_scorer(name, compression_tile_size=4, compression_stride=2)
                np.testing.assert_array_equal(current(reference, source, support),
                                              legacy(reference, source, support))
                if current.raw_scores is not None:
                    np.testing.assert_array_equal(current.raw_scores, legacy.raw_scores)
                self.assertEqual(current.metadata["output_kind"], "score")

    def test_external_constructors_are_lazy_and_rscd_keeps_its_checkpoint_variant(self):
        class FakeAdapter:
            def __init__(self, **options):
                self.options = options

        for name in EXTERNAL_METHODS:
            module = SimpleNamespace(**{class_name: FakeAdapter for class_name in (
                "LPIPSScorer", "DINOv2Scorer", "RSCDScorer", "AnyChangeScorer", "GeoSCDScorer",
                "ClassicalCodecScorer", "ArIBScorer")})
            with patch("facade_change.methods.registry.importlib.import_module", return_value=module) as importer:
                scorer = make_method(name, checkpoint_path="local.pt", device="cpu")
            self.assertEqual(importer.call_count, 1)
            self.assertEqual(scorer.options["checkpoint_path"], "local.pt")
            if name.startswith(("rscd_", "jpegls_", "arib_bps_")) or name == "h264_rgb":
                self.assertEqual(scorer.options["method"], name)
            else:
                self.assertNotIn("method", scorer.options)

    def test_legacy_registry_never_imports_external_adapters_and_rejects_labels(self):
        with patch("facade_change.methods.registry.importlib.import_module", side_effect=AssertionError("unexpected model import")):
            make_method("rgb_diff")
            make_method("lzma_abs")
        self.assertEqual(len(LEGACY_METHODS), 8)
        self.assertEqual(len(EXTERNAL_METHODS), 12)
        for name, options in (("bad", {}), ("rgb_diff", {"edit_mask": None}),
                              ("lzma_abs", {"occlusion_mask": None})):
            with self.assertRaises(ValueError):
                make_method(name, **options)


if __name__ == "__main__":
    unittest.main()
