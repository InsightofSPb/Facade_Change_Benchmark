"""CPU-only protocol tests with a patched adapter, never real model weights."""
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

from scripts.real_eval.semantic_remote import SemanticPool, preflight_remote
from scripts.real_eval.semantic_worker import serve, validate_prediction


FAKE_WORKER = r'''
import os
import sys
import numpy as np
import scripts.real_eval.semantic_worker as worker

class TestSegmenter:
    def __init__(self, config, vocabulary):
        if config['name'] == 'init_error':
            raise ValueError('test model load failure')
        self.config, self.vocabulary = config, vocabulary
        self.metadata = {'pid': os.getpid(), 'torch_loaded': 'torch' in sys.modules}
        print('Python loading log')
        os.write(1, b'Native loading log\n')
    def predict(self, rgb):
        if self.config['name'] == 'crash':
            os._exit(9)
        scores = np.mean(rgb.astype(np.float32), axis=2)[None] / 255
        scores = np.repeat(scores, len(self.vocabulary), axis=0)
        if self.config['name'] == 'bad_probability':
            scores[0, 0, 0] = 2
        return {'scores': scores, 'class_names': self.vocabulary,
                'score_kind': 'test_independent_scores',
                'extra_arrays': {'raw_scores': scores * 20 - 10},
                'metadata': {'prompt_texts': ['wall', 'crack']}, 'softmax_scale': 100}
    def close(self):
        print('model released')

worker._make_segmenter = TestSegmenter
worker.main()
'''


class SemanticPoolTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        script = self.root / "test_worker.py"
        script.write_text(FAKE_WORKER, encoding="utf-8")
        real_popen = subprocess.Popen

        def launch(command, **kwargs):
            self.assertEqual(command[-1], "scripts.real_eval.semantic_worker")
            return real_popen([sys.executable, "-u", "-B", str(script)], **kwargs)

        self.launch = patch("scripts.real_eval.semantic_remote.subprocess.Popen", side_effect=launch)
        self.popen = self.launch.start()
        self.pool = SemanticPool(self.root / "logs", timeout=15)
        self.rgb = np.full((4, 7, 3), 127, dtype=np.uint8)
        self.config = {"name": "test_model", "worker_python": sys.executable}

    def tearDown(self):
        self.pool.close()
        self.launch.stop()
        self.folder.cleanup()

    def test_reuses_model_and_releases_on_vocabulary_or_model_change(self):
        first = self.pool.predict(self.config, ["crack"], self.rgb)
        process = self.pool.process
        repeated = self.pool.predict(self.config, ["crack"], self.rgb)
        self.assertEqual(first["metadata"]["pid"], repeated["metadata"]["pid"])
        self.assertEqual(self.popen.call_count, 1)
        np.testing.assert_array_equal(first["scores"], np.full((1, 4, 7), 127 / 255, np.float32))
        self.assertEqual(first["extra_arrays"]["raw_scores"].shape, (1, 4, 7))
        self.assertEqual(first["metadata"]["softmax_scale"], 100)
        self.assertEqual(first["metadata"]["prompt_texts"], ["wall", "crack"])
        self.assertGreaterEqual(first["elapsed_seconds"], 0)
        self.assertGreaterEqual(first["metadata"]["init_seconds"], 0)
        self.assertFalse(first["metadata"]["torch_loaded"])
        log = self.pool.stderr_path.read_text()
        self.assertIn("Python loading log", log)
        self.assertIn("Native loading log", log)
        second = self.pool.predict(self.config, ["crack", "wall"], self.rgb)
        self.assertEqual(second["scores"].shape, (2, 4, 7))
        self.assertIsNotNone(process.poll())
        self.assertNotEqual(first["metadata"]["pid"], second["metadata"]["pid"])
        process = self.pool.process
        third = self.pool.predict({**self.config, "name": "other_model"}, ["crack", "wall"], self.rgb)
        self.assertIsNotNone(process.poll())
        self.assertNotEqual(second["metadata"]["pid"], third["metadata"]["pid"])

    def test_initialization_failure_preserves_log_and_disposes(self):
        with self.assertRaisesRegex(RuntimeError, "test model load failure.*stderr log"):
            self.pool.activate({**self.config, "name": "init_error"}, ["crack"])
        self.assertIsNone(self.pool.process)
        self.assertIn("test model load failure", self.pool.stderr_path.read_text())

    def test_invalid_probability_disposes_worker(self):
        with self.assertRaisesRegex(RuntimeError, r"native-grid float32 C,H,W in \[0,1\]"):
            self.pool.predict({**self.config, "name": "bad_probability"}, ["crack"], self.rgb)
        self.assertIsNone(self.pool.process)

    def test_process_crash_uses_existing_transport_failure(self):
        with self.assertRaisesRegex(RuntimeError, "exited without a response"):
            self.pool.predict({**self.config, "name": "crash"}, ["crack"], self.rgb)
        self.assertIsNone(self.pool.process)

    def test_bad_inputs_do_not_start_worker(self):
        with self.assertRaisesRegex(ValueError, "uint8"):
            self.pool.predict(self.config, ["crack"], self.rgb.astype(np.float32))
        with self.assertRaisesRegex(ValueError, "absolute Python"):
            self.pool.activate({**self.config, "worker_python": "python"}, ["crack"])
        self.assertEqual(self.popen.call_count, 0)


class SemanticWorkerContractTests(unittest.TestCase):
    def test_preflight_runs_in_selected_interpreter_without_model_initialization(self):
        result = preflight_remote({"kind": "no-such-model", "worker_python": sys.executable})
        self.assertFalse(result["ok"])
        self.assertIn("Unknown segmentation adapter", result["errors"][0])
        self.assertEqual(result["config"]["worker_python"], sys.executable)
        self.assertEqual(result["environment"]["executable"], sys.executable)
        self.assertEqual(result["environment"]["packages"]["numpy"], np.__version__)
        self.assertEqual(len(result["environment"]["worker_source_sha256"]), 64)

    def test_gt_or_support_cannot_cross_worker_boundary(self):
        for forbidden in ("labels", "support"):
            fake = SimpleNamespace(metadata={}, predict=Mock(), close=Mock())
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "input.npz"
                np.savez(path, rgb=np.zeros((2, 3, 3), np.uint8),
                         **{forbidden: np.ones((2, 3), np.uint8)})
                requests = [
                    {"id": 1, "operation": "init", "config": {}, "vocabulary": []},
                    {"id": 2, "operation": "predict", "input_path": str(path),
                     "output_path": str(Path(folder) / "output.npz")},
                    {"id": 3, "operation": "close"},
                ]
                output = io.StringIO()
                with patch("scripts.real_eval.semantic_worker._make_segmenter", return_value=fake), \
                        patch("sys.stderr", new=io.StringIO()):
                    serve(io.StringIO("\n".join(map(json.dumps, requests))), output)
                rows = [json.loads(line) for line in output.getvalue().splitlines()]
                self.assertTrue(rows[0]["ok"])
                self.assertFalse(rows[1]["ok"])
                self.assertIn("only rgb", rows[1]["error"])
                self.assertTrue(rows[2]["ok"])
                fake.predict.assert_not_called()
                fake.close.assert_called_once()

    def test_geometry_nonfinite_and_pickle_arrays_rejected(self):
        good = {"scores": np.zeros((1, 2, 3), np.float32), "class_names": ["crack"],
                "score_kind": "independent", "extra_arrays": {}}
        for bad in ({**good, "scores": np.zeros((1, 3, 2), np.float32)},
                    {**good, "scores": np.full((1, 2, 3), np.nan, np.float32)},
                    {**good, "extra_arrays": {"mask": np.array([object()], dtype=object)}},
                    {**good, "extra_arrays": {"prompts": np.array(["crack"])}}):
            with self.assertRaises(ValueError):
                validate_prediction(bad, (2, 3))


if __name__ == "__main__":
    unittest.main()
