import contextlib
import importlib.util
import io
import shutil
import sys
import unittest
from unittest.mock import patch

import numpy as np

from facade_change.hypothesis_benchmark import run_hypothesis_benchmark
from facade_change.io import read_json, sha256
from facade_change.methods.lossless import FFmpegCodec
import test_hypothesis_benchmark as _fixtures


@unittest.skipUnless(
    importlib.util.find_spec("cv2") and shutil.which("ffmpeg") and shutil.which("ffprobe"),
    "OpenCV and FFmpeg/ffprobe required for real classical codec workers",
)
class ClassicalRunnerTests(unittest.TestCase):
    setUp = _fixtures.HypothesisBenchmarkTests.setUp
    tearDown = _fixtures.HypothesisBenchmarkTests.tearDown

    def test_real_codec_costs_roundtrip_and_reuse_preserve_verified_artifacts(self):
        for kind in ("jpegls", "h264"):
            try:
                FFmpegCodec(kind)
            except RuntimeError as exc:
                if "lacks required encoder" in str(exc):
                    self.skipTest(str(exc))
                raise

        methods = ["jpegls_mod256", "h264_rgb"]
        original, replayed = self.root / "classical-original", self.root / "classical-replayed"
        options = {name: {"worker_python": sys.executable, "tile_size": 32, "stride": 32}
                   for name in methods}
        with contextlib.redirect_stdout(io.StringIO()), \
                patch("facade_change.benchmark_progress._load_tqdm", return_value=None):
            first = run_hypothesis_benchmark(self.dataset, original, methods=methods,
                                            quick_bases=2, method_options=options)

        self.assertEqual(read_json(original / "run.json")["status"], "completed_exploratory")
        self.assertEqual(first["selected_case_count"], 12)
        costs = read_json(original / "codec_stats.json")
        metrics = read_json(original / "metrics.json")["cases"]
        self.assertEqual(set(costs), set(methods))
        for row in metrics:
            stats = costs[row["method"]][row["case_id"]]
            self.assertEqual(stats["cost_mode"], "bitstream")
            self.assertTrue(stats["all_tiles_roundtrip_verified"])
            self.assertEqual(stats["tiles"], 1)
            self.assertGreater(stats["charged_bytes"], 0)
            self.assertEqual(stats["all_stream_bytes"],
                             stats["charged_bytes"] + stats["reference_i_bytes"])
            if row["method"] == "h264_rgb":
                self.assertGreater(stats["reference_i_bytes"], 0)
            else:
                self.assertEqual(stats["reference_i_bytes"], 0)
            scores = np.load(original / row["score_path"], allow_pickle=False)
            raw = np.load(original / row["raw_score_path"], allow_pickle=False)
            support = np.isfinite(scores)
            self.assertTrue(np.isnan(raw[~support]).all())
            np.testing.assert_allclose(raw[support], 8 * stats["charged_bytes"] / (32 * 32 * 3))
            np.testing.assert_allclose(scores[support], -np.expm1(-raw[support] / 8), atol=1e-7)

        before = {path.relative_to(original): sha256(path)
                  for path in original.rglob("*") if path.is_file()}
        with contextlib.redirect_stdout(io.StringIO()), \
                patch("facade_change.benchmark_progress._load_tqdm", return_value=None), \
                patch("facade_change.methods.remote.RemoteScorer", side_effect=AssertionError("cached codec loaded")), \
                patch("facade_change.scorers.make_scorer", side_effect=AssertionError("cached scorer loaded")):
            again = run_hypothesis_benchmark(self.dataset, replayed, methods=methods, reuse_run=original)
            self.assertEqual(again["new_inference_case_method_count"], 0)
            self.assertEqual(again["thresholds"], first["thresholds"])
            self.assertEqual(read_json(replayed / "threshold_selection.json"),
                             read_json(original / "threshold_selection.json"))
            self.assertEqual(read_json(replayed / "codec_stats.json"), costs)
            replay_rows = {(row["method"], row["case_id"]): row
                           for row in read_json(replayed / "metrics.json")["cases"]}
            for row in metrics:
                cached = replay_rows[row["method"], row["case_id"]]
                self.assertEqual(cached["prediction_sha256"], row["prediction_sha256"])
                self.assertEqual(sha256(replayed / cached["prediction_path"]),
                                 sha256(original / row["prediction_path"]))
                self.assertEqual(cached["score_sha256"], row["score_sha256"])

            # Tampering with the cost artifact must fail before a worker loads.
            cost_path = original / "codec_stats.json"
            cost_bytes = cost_path.read_bytes()
            try:
                cost_path.write_bytes(b"{}\n")
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    run_hypothesis_benchmark(self.dataset, self.root / "classical-corrupt",
                                            methods=methods, reuse_run=original)
            finally:
                cost_path.write_bytes(cost_bytes)

        self.assertEqual({path.relative_to(original): sha256(path)
                          for path in original.rglob("*") if path.is_file()}, before)


if __name__ == "__main__":
    unittest.main()
