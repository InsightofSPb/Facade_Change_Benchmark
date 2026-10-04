"""Isolated CPU worker contracts; no model download or pretrained inference."""
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from facade_change.method_worker import serve
from facade_change.methods.geoscd import GeoSCDScorer
from facade_change.methods.remote import RemotePool, RemoteScorer


FAKE_WORKER = r'''
import os
import sys
import time
import numpy as np
import facade_change.methods.registry as registry
from facade_change.method_worker import main

class FakeScorer:
    def __init__(self, method, **options):
        if method == "init_error":
            raise ValueError("fake load failure")
        self.method = method
        self.metadata = {"method": method, "pid": os.getpid(), "output_kind": "score"}
        self.raw_scores = None
        self.native_prediction = None
        print("Python model loading log")
        os.write(1, b"Native model loading log\n")
    def __call__(self, reference, source, support):
        if self.method == "crash":
            os._exit(7)
        if self.method == "slow":
            time.sleep(2)
        scores = np.mean(np.abs(source.astype(np.float32)-reference.astype(np.float32)), axis=2)/255
        scores = scores.astype(np.float32)
        scores[~support] = np.nan
        self.raw_scores = scores.copy()
        self.native_prediction = np.nan_to_num(scores) > .5
        return scores
    def close(self):
        print("model released")

registry.make_method = FakeScorer
main()
'''


class GeoSCDAdapterTests(unittest.TestCase):
    def test_uses_native_rgb_and_author_mask_only(self):
        a = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        b = np.flip(a, axis=1).copy()
        support = np.ones((6, 8), bool)
        support[0, 0] = False
        output = np.zeros((512, 512), bool)
        output[:, 256:] = True
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        detector = Mock(metadata={"model_grid": [512, 512]})
        detector.torch = torch

        def predict(left, right):
            with Image.open(left) as image:
                np.testing.assert_array_equal(np.asarray(image), a)
                self.assertEqual(image.size, (8, 6))
            with Image.open(right) as image:
                np.testing.assert_array_equal(np.asarray(image), b)
            return {"final_reference_mask": output,
                    "reference_detection": {"score": np.full((512, 512), 999.0)}}

        detector.side_effect = predict
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            vggt, sam = root / "vggt.pt", root / "sam.pth"
            vggt.write_bytes(b"vggt")
            sam.write_bytes(b"sam")
            with patch("facade_change.methods.geoscd.source_provenance", return_value={"commit": "pinned"}), \
                 patch("facade_change.geoscd_full.OfficialFull", return_value=detector):
                scorer = GeoSCDScorer(source_root=root, checkpoint_path=vggt, sam_checkpoint=sam)
                scores = scorer(a, b, support)
            self.assertEqual(scores.dtype, np.float32)
            self.assertTrue(np.isnan(scores[0, 0]))
            expected = np.zeros((6, 8), bool)
            expected[:, 4:] = True
            np.testing.assert_array_equal(scorer.native_prediction, expected)
            np.testing.assert_array_equal(scores[support], expected[support].astype(np.float32))
            self.assertIsNone(scorer.raw_scores)
            self.assertEqual(scorer.metadata["output_kind"], "native_mask")
            self.assertEqual(len(scorer.metadata["checkpoints"]["sam"]["sha256"]), 64)
            self.assertFalse(Path(detector.call_args.args[0]).exists())
            scorer.close()
            detector.geometry.model.cpu.assert_called_once()
            detector.sam.cpu.assert_called_once()
            scorer.close()

    def test_rejects_incomplete_checkpoint_configuration(self):
        with self.assertRaisesRegex(ValueError, "requires source_root"):
            GeoSCDScorer()
        with self.assertRaisesRegex(ValueError, "Unknown GeoSCD options"):
            GeoSCDScorer(labels="forbidden")


class RemoteWorkerTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.script = self.root / "fake_worker.py"
        self.script.write_text(FAKE_WORKER, encoding="utf-8")
        real_popen = subprocess.Popen

        def fake_launch(command, **kwargs):
            return real_popen([sys.executable, "-u", "-B", str(self.script)], **kwargs)

        self.launch = patch("facade_change.methods.remote.subprocess.Popen", side_effect=fake_launch)
        self.launch.start()
        self.pool = RemotePool(log_dir=self.root / "logs", timeout=20)
        self.a = np.zeros((5, 7, 3), np.uint8)
        self.b = np.full_like(self.a, 255)
        self.support = np.ones(self.a.shape[:2], bool)
        self.support[0, 0] = False

    def tearDown(self):
        self.pool.close()
        self.launch.stop()
        self.folder.cleanup()

    def test_worker_reuses_process_and_switch_releases_previous(self):
        scorer = RemoteScorer("first", {"worker_python": sys.executable}, self.pool)
        self.assertIsNone(self.pool.process)
        score = scorer(self.a, self.b, self.support)
        first_process, first_pid = self.pool.process, scorer.metadata["pid"]
        self.assertTrue(np.isnan(score[0, 0]))
        np.testing.assert_array_equal(score[self.support], 1)
        self.assertEqual(scorer.raw_scores.dtype, np.float32)
        self.assertEqual(scorer.native_prediction.dtype, bool)
        scorer(self.b, self.a, self.support)
        self.assertEqual(first_pid, scorer.metadata["pid"])
        log = self.pool.stderr_path.read_text(encoding="utf-8")
        self.assertIn("Python model loading log", log)
        self.assertIn("Native model loading log", log)
        other = RemoteScorer("second", {"worker_python": sys.executable}, self.pool)
        other(self.a, self.b, self.support)
        self.assertNotEqual(first_pid, other.metadata["pid"])
        self.assertIsNotNone(first_process.poll())
        self.pool.close()
        self.assertIsNone(self.pool.process)

    def test_model_init_error_keeps_clear_log(self):
        scorer = RemoteScorer("init_error", {"worker_python": sys.executable}, self.pool)
        with self.assertRaisesRegex(RuntimeError, "fake load failure.*stderr log"):
            scorer(self.a, self.b, self.support)
        self.assertIsNone(self.pool.process)
        self.assertIn("fake load failure", self.pool.stderr_path.read_text())

    def test_crashed_process_does_not_hang(self):
        scorer = RemoteScorer("crash", {"worker_python": sys.executable}, self.pool)
        with self.assertRaisesRegex(RuntimeError, "exited without a response.*stderr log"):
            scorer(self.a, self.b, self.support)
        self.assertIsNone(self.pool.process)

    def test_timeout_kills_worker(self):
        options = {"worker_python": sys.executable}
        self.pool.activate("slow", options)
        self.pool._active_timeout = .05
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "request.npz"
            np.savez(source, reference=self.a, source=self.b, support=self.support)
            with self.assertRaisesRegex(RuntimeError, "timed out.*stderr log"):
                self.pool._request("score", input_path=str(source), output_path=str(Path(folder) / "out.npz"))
        self.assertIsNone(self.pool.process)

    def test_rejects_relative_python_path(self):
        with self.assertRaisesRegex(ValueError, "absolute Python"):
            self.pool.activate("first", {"worker_python": "python"})
        self.assertIsNone(self.pool.process)


class WorkerInputTests(unittest.TestCase):
    def test_archive_with_labels_is_rejected_before_scoring(self):
        fake = SimpleNamespace(metadata={"method": "fake"}, close=Mock())
        fake.__call__ = Mock()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "with-labels.npz"
            a = np.zeros((2, 2, 3), np.uint8)
            np.savez(path, reference=a, source=a, support=np.ones((2, 2), bool),
                     labels=np.ones((2, 2), np.uint8))
            requests = [
                {"id": 1, "operation": "init", "method": "fake", "options": {}},
                {"id": 2, "operation": "score", "input_path": str(path),
                 "output_path": str(Path(folder) / "output.npz")},
                {"id": 3, "operation": "close"},
            ]
            output = io.StringIO()
            with patch("facade_change.methods.registry.make_method", return_value=fake), \
                 patch("sys.stderr", new=io.StringIO()):
                serve(io.StringIO("\n".join(map(json.dumps, requests))), output)
            rows = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertTrue(rows[0]["ok"])
            self.assertFalse(rows[1]["ok"])
            self.assertIn("only reference, source, support", rows[1]["error"])
            self.assertTrue(rows[2]["ok"])
            fake.close.assert_called_once()
            fake.__call__.assert_not_called()


if __name__ == "__main__":
    unittest.main()
