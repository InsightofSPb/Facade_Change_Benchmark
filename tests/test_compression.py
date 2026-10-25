"""Residual representation and actual codec-score contracts, independent of GT."""
import builtins
import importlib.util
import json
import lzma
import unittest
from unittest.mock import patch

import numpy as np

from facade_change.scorers import make_scorer, rgb_residual, score_change, scorer_metadata


class ResidualRepresentationTests(unittest.TestCase):
    def test_signed_subtraction_and_modulo_reconstruction_cover_byte_extremes(self):
        reference = np.array([0, 255, 100, 100, 0, 255], np.uint8).reshape(1, 2, 3)
        source = np.array([255, 0, 110, 90, 0, 255], np.uint8).reshape(1, 2, 3)
        absolute = rgb_residual(reference, source, "abs")
        modular = rgb_residual(reference, source, "mod256")
        np.testing.assert_array_equal(absolute.ravel(), [255, 255, 10, 10, 0, 0])
        np.testing.assert_array_equal(modular.ravel(), [255, 1, 10, 246, 0, 0])
        restored = (reference.astype(np.int16) + modular.astype(np.int16)) % 256
        np.testing.assert_array_equal(restored.astype(np.uint8), source)
        self.assertEqual(absolute.dtype, np.uint8)
        self.assertEqual(modular.dtype, np.uint8)

    def test_all_rgb_byte_values_roundtrip_and_layout_is_interleaved(self):
        reference = np.arange(256, dtype=np.uint8)[None, :, None].repeat(3, axis=2)
        source = reference[:, ::-1].copy()
        modular = rgb_residual(reference, source, "mod256")
        restored = np.remainder(reference.astype(np.int16) + modular, 256).astype(np.uint8)
        np.testing.assert_array_equal(restored, source)
        np.testing.assert_array_equal(np.frombuffer(modular.tobytes(order="C"), np.uint8).reshape(modular.shape), modular)
        with self.assertRaisesRegex(ValueError, "representation"):
            rgb_residual(reference, source, "signed")
        with self.assertRaisesRegex(ValueError, "uint8"):
            rgb_residual(reference.astype(float), source, "abs")


class LZMAScorerTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(941)
        self.reference = rng.integers(0, 256, (11, 13, 3), dtype=np.uint8)
        self.source = self.reference.copy()
        self.source[2:8, 4:10] = [24, 80, 172]
        self.support = np.ones(self.reference.shape[:2], bool)
        self.support[:3, :2] = False

    def test_codec_roundtrip_preserves_both_residual_byte_streams(self):
        for representation in ("abs", "mod256"):
            scorer = make_scorer(f"lzma_{representation}")
            payload = rgb_residual(self.reference, self.source, representation).tobytes(order="C")
            self.assertEqual(lzma.decompress(scorer.compress(payload)), payload)

    def test_support_neutralization_is_independent_of_invalid_rgb(self):
        scorer = make_scorer("lzma_abs", compression_tile_size=8, compression_stride=4)
        scores = scorer(self.reference, self.source, self.support)
        native = scorer.raw_scores.copy()
        reference, source = self.reference.copy(), self.source.copy()
        reference[~self.support] = [255, 0, 180]
        source[~self.support] = [0, 255, 255]
        reference.flags.writeable = source.flags.writeable = self.support.flags.writeable = False
        second = scorer(reference, source, self.support)
        np.testing.assert_array_equal(scores, second)
        np.testing.assert_array_equal(native, scorer.raw_scores)
        self.assertEqual(second.shape, self.support.shape)
        self.assertEqual(second.dtype, np.float32)
        self.assertEqual(scorer.raw_scores.dtype, np.float32)
        self.assertTrue(np.isnan(second[~self.support]).all())
        self.assertTrue(np.isfinite(second[self.support]).all())
        self.assertTrue((second[self.support] >= 0).all() and (second[self.support] <= 1).all())

    def test_overlap_header_and_padding_match_independent_stream_length(self):
        reference = np.zeros((3, 3, 3), np.uint8)
        source = np.array([[[10, 246, 0], [10, 246, 0], [30, 20, 70]],
                           [[10, 246, 0], [10, 246, 0], [30, 20, 70]],
                           [[40, 50, 90], [40, 50, 90], [30, 20, 70]]], np.uint8)
        support = np.ones((3, 3), bool)
        support[0, 0] = False
        residual = source.copy()
        residual[~support] = 0
        total, count = np.zeros((3, 3)), np.zeros((3, 3))
        for row in (0, 1, 2):
            for col in (0, 1, 2):
                tile = np.zeros((2, 2, 3), np.uint8)
                block = residual[row:row + 2, col:col + 2]
                tile[:block.shape[0], :block.shape[1]] = block
                stream = lzma.compress(tile.tobytes(), format=lzma.FORMAT_XZ,
                                       check=lzma.CHECK_CRC64, preset=3)
                total[row:row + 2, col:col + 2] += len(stream) * 8 / 12
                count[row:row + 2, col:col + 2] += 1
        expected = total / count
        scorer = make_scorer("lzma_mod256", compression_tile_size=2, compression_stride=1)
        scores = scorer(reference, source, support)
        np.testing.assert_allclose(scorer.raw_scores[support], expected[support], rtol=1e-7)
        np.testing.assert_allclose(scores[support], -np.expm1(-expected[support] / 8), rtol=1e-7)
        self.assertTrue((scorer.raw_scores[support] > 8).all())

    def test_identical_black_rgb_keeps_codec_overhead_and_deterministic_scores(self):
        reference = np.zeros((7, 9, 3), np.uint8)
        support = np.ones((7, 9), bool)
        scorer = make_scorer("lzma_abs", compression_tile_size=8, compression_stride=4)
        first = scorer(reference, reference.copy(), support)
        second = scorer(reference, reference.copy(), support)
        np.testing.assert_array_equal(first, second)
        self.assertTrue((first > 0).all())
        self.assertTrue((scorer.raw_scores > 0).all())
        np.testing.assert_array_equal(first, score_change(reference, reference, support, "lzma_abs",
                                                         compression_tile_size=8, compression_stride=4))

    def test_native_scale_does_not_change_unaffected_tile_after_distant_edit(self):
        reference = np.zeros((16, 16, 3), np.uint8)
        support = np.ones((16, 16), bool)
        source = reference.copy()
        scorer = make_scorer("lzma_abs", compression_tile_size=4, compression_stride=4)
        first = scorer(reference, source, support).copy()
        source[12:] = np.random.default_rng(4).integers(0, 256, (4, 16, 3), dtype=np.uint8)
        second = scorer(reference, source, support)
        np.testing.assert_array_equal(first[:12], second[:12])
        self.assertGreater(float(second[-1, -1]), float(first[-1, -1]))

    def test_metadata_and_options_are_explicit_serializable_and_validated(self):
        metadata = scorer_metadata("lzma_mod256", compression_tile_size=8, compression_stride=3)
        self.assertEqual(metadata["representation"], "mod256")
        self.assertEqual(metadata["codec_options"]["format"], "XZ")
        self.assertIn("header", metadata["native_units"])
        self.assertIn("no per-map", metadata["score_formula"])
        self.assertEqual(metadata["stride"], 3)
        json.dumps(metadata, allow_nan=False)
        metadata["codec_options"]["preset"] = 0
        self.assertEqual(scorer_metadata("lzma_mod256")["codec_options"]["preset"], 3)
        for options in ({"compression_stride": 33}, {"compression_tile_size": True},
                        {"lzma_preset": 10}, {"compression_stride": 0}, {"edit_mask": None}):
            with self.assertRaises(ValueError):
                make_scorer("lzma_abs", **options)
        with self.assertRaises(ValueError):
            score_change(self.reference[:, :-1], self.source, self.support, "lzma_abs")


class ZstdScorerTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("zstandard"), "Optional zstandard not installed")
    def test_codec_roundtrip_and_score_determinism(self):
        import zstandard
        reference = np.zeros((9, 12, 3), np.uint8)
        source = np.random.default_rng(2).integers(0, 256, reference.shape, dtype=np.uint8)
        support = np.ones(reference.shape[:2], bool)
        for representation in ("abs", "mod256"):
            scorer = make_scorer(f"zstd_{representation}", compression_tile_size=8, compression_stride=4)
            residual = rgb_residual(reference, source, representation)
            stream = scorer.compress(residual.tobytes(order="C"))
            self.assertEqual(zstandard.ZstdDecompressor().decompress(stream), residual.tobytes(order="C"))
            first = scorer(reference, source, support).copy()
            np.testing.assert_array_equal(first, scorer(reference, source, support))

    def test_missing_optional_zstandard_raises_actionable_error(self):
        original_import = builtins.__import__

        def unavailable(name, *args, **kwargs):
            if name == "zstandard":
                raise ImportError("test unavailable codec")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=unavailable):
            with self.assertRaisesRegex(RuntimeError, "pip install zstandard"):
                make_scorer("zstd_abs")


if __name__ == "__main__":
    unittest.main()
