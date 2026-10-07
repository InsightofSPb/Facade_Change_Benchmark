import copy
import importlib
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from facade_change.hypothesis_benchmark import _counts, _calibrate, _threshold_grid, THRESHOLDS
from facade_change.methods.lossless import FFmpegCodec, ClassicalCodecScorer
from facade_change.benchmark_results import ReuseResults
from facade_change.io import read_json
from test_benchmark_results import save_run


class CalibrationAndReuseTests(unittest.TestCase):
    def test_unknown_tile_option_fails_before_codec_initialization(self):
        with patch("facade_change.methods.lossless.FFmpegCodec", side_effect=AssertionError("codec loaded")):
            with self.assertRaisesRegex(ValueError, "Unknown classical codec options"):
                ClassicalCodecScorer("jpegls_mod256", tile_stride=32)

    def test_tiny_probabilities_can_separate_h0_h1_without_map_normalization(self):
        scores = np.array([1e-12, 1e-12, 1e-8, 1e-8], dtype=np.float32)
        labels = np.array([0, 0, 1, 1], dtype=np.uint8)
        h1 = {"building_id": "val", "hypothesis": "H1", "sham_self_paste": False}
        h0 = {**h1, "hypothesis": "H0"}
        coarse = _calibrate([(h1, _counts(scores, labels, THRESHOLDS)),
                              (h0, _counts(scores[:2], labels[:2], THRESHOLDS))])
        grid = _threshold_grid("rscd_cmu")
        choice = _calibrate([(h1, _counts(scores, labels, grid)),
                             (h0, _counts(scores[:2], labels[:2], grid))], grid)
        self.assertEqual(coarse["threshold"], 0)
        self.assertGreater(choice["threshold"], 1e-12)
        self.assertLess(choice["threshold"], 1e-8)
        self.assertEqual(choice["curve"][choice["grid_index"]]["h1_building_macro_f1"], 1)
        self.assertEqual(choice["selection_split"], "val")
        for method in ("rgb_diff", "ssim", "msdzip_mod256"):
            np.testing.assert_array_equal(_threshold_grid(method), THRESHOLDS)

    def test_explicit_cache_exclusion_validates_source_and_preserves_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            *_, method = save_run(root)
            before = (root / "run.json").read_bytes()
            reuse = ReuseResults(root, exclude_methods=[method])
            self.assertEqual(reuse.methods, ())
            self.assertEqual(reuse.provenance["recomputed_methods"], [method])
            self.assertEqual(before, (root / "run.json").read_bytes())
            with self.assertRaisesRegex(ValueError, "Recomputed"):
                ReuseResults(root, exclude_methods=["unknown"])
            row = next(iter(reuse.rows.values()))
            (root / row["score_path"]).write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                ReuseResults(root, exclude_methods=[method])


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class RealCodecTests(unittest.TestCase):
    def test_exact_rgb_streams_and_conditional_p_frame_accounting(self):
        rng = np.random.default_rng(5)
        a = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        b = a.copy()
        b[5:13, 8:16] = [255, 0, 1]
        for kind in ("jpegls", "h264"):
            codec = FFmpegCodec(kind)
            unchanged, info0 = codec.encode(a, a)
            changed, info1 = codec.encode(a, b)
            self.assertTrue(info0["roundtrip_verified"] and info1["roundtrip_verified"])
            self.assertGreater(changed, 0)
            if kind == "h264":
                self.assertLess(unchanged, changed)
                self.assertEqual(info1["stream_bytes"], info1["reference_i_bytes"]+changed)
                self.assertGreater(info1["reference_i_bytes"], changed)
            else:
                self.assertEqual(changed, info1["stream_bytes"])

    def test_signed_residual_wrap_support_and_raw_header_costs(self):
        a = np.full((18, 21, 3), 255, dtype=np.uint8)
        b = np.zeros_like(a)
        support = np.ones((18, 21), dtype=bool)
        support[:, 3:5] = False
        scorer = ClassicalCodecScorer("jpegls_mod256", tile_size=32, stride=32)
        scores = scorer(a, b, support)
        raw = scorer.raw_scores.copy()
        self.assertEqual(scores.shape, support.shape)
        self.assertTrue(np.isnan(scores[~support]).all())
        changed = b.copy()
        changed[~support] = 180
        np.testing.assert_array_equal(scorer(a, changed, support), scores)
        np.testing.assert_allclose(scores[support], -np.expm1(-raw[support]/8), atol=1e-7)
        self.assertEqual(scorer.metadata["last_codec_stats"]["tiles"], 1)
        length = scorer.metadata["last_codec_stats"]["charged_bytes"]
        np.testing.assert_allclose(raw[support], 8*length/(32*32*3))


if __name__ == "__main__":
    unittest.main()
