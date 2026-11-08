"""CPU contracts only. Fake workers/processors are not a real model smoke."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from contextlib import nullcontext, redirect_stderr
from types import SimpleNamespace

import numpy as np
from PIL import Image

from facade_change.agentic.models import ModelPool, WorkerError, _Worker
from facade_change.agentic.model_worker import Adapter, DinoAdapter, QwenAdapter, SAM3Adapter, serve


class FakeWorker:
    made = []
    replies = []

    def __init__(self, role, options, log_dir, deadline):
        self.role, self.options, self.closed = role, options, False
        self.log_path = Path(log_dir) / "fake.log"
        self.requests = []
        self.provenance = {"test_double": True}
        self.made.append(self)

    def request(self, payload, deadline):
        self.requests.append(copy.deepcopy(payload))
        if self.replies:
            value = self.replies.pop(0)
            if isinstance(value, Exception):
                raise value
            return copy.deepcopy(value)
        return {"status": "ok", "text": "contract test double", "provenance": {"test_double": True},
                "usage": {"model_loads": 1 if len(self.requests) == 1 else 0, "model_calls": 1,
                          "input_tokens": 10, "output_tokens": 3, "images": 2}}

    def close(self):
        self.closed = True


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image = self.root / "image.png"
        Image.new("RGB", (16, 16), "red").save(self.image)
        FakeWorker.made, FakeWorker.replies = [], []
        self.patch = patch("facade_change.agentic.models._Worker", FakeWorker)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def payload(self, text="compare", **extra):
        return {"op": "generate", "messages": [{"role": "user", "content": [
            {"type": "text", "text": text}, {"type": "image", "path": str(self.image)}]}],
            "geometry": {"crop_to_observation": [[1, 0, 2], [0, 1, 3], [0, 0, 1]]}, **extra}

    def pool(self, **kwargs):
        return ModelPool({role: {"python": sys.executable} for role in ("qwen", "sam3", "jina")},
                         self.root / "logs", **kwargs)

    def test_sequential_handoff_replays_cpu_messages(self):
        with self.pool() as pool:
            message = self.payload()
            pool.call("qwen", message)
            pool.call("sam3", {"op": "check"})
            pool.call("qwen", message)
            self.assertEqual([w.role for w in FakeWorker.made], ["qwen", "sam3", "qwen"])
            self.assertTrue(FakeWorker.made[0].closed)
            self.assertTrue(FakeWorker.made[1].closed)
            self.assertEqual(FakeWorker.made[0].requests[0]["messages"], FakeWorker.made[2].requests[0]["messages"])
        self.assertTrue(all(w.closed for w in FakeWorker.made))
        records = [json.loads(line) for line in (self.root / "logs" / "pool.jsonl").read_text().splitlines()]
        self.assertEqual([row["event"] for row in records],
                         ["worker_started", "worker_released", "worker_started", "worker_released", "worker_started", "worker_released"])
        self.assertEqual([row["role"] for row in records], ["qwen", "qwen", "sam3", "sam3", "qwen", "qwen"])
        self.assertEqual(records[1]["reason"], "gpu_handoff")
        self.assertTrue(all(row["timestamp_utc"] and "pid" in row for row in records))
        self.assertEqual(len({row["event_id"] for row in records}), len(records))

    def test_resident_pair_does_not_keep_third_heavy_model(self):
        with self.pool(gpu_mode="resident") as pool:
            pool.call("qwen", self.payload())
            pool.call("sam3", {"op": "check"})
            self.assertFalse(any(w.closed for w in FakeWorker.made))
            pool.call("qwen", self.payload("second"))
            self.assertEqual(len(FakeWorker.made), 2)
            pool.call("jina", {"op": "embed", "texts": ["crack"]})
            self.assertTrue(all(w.closed for w in FakeWorker.made[:2]))

    def test_oom_retries_identical_request_once_and_counts_both(self):
        FakeWorker.replies = [
            {"status": "error", "error": {"code": "out_of_memory", "message": "test OOM"},
             "usage": {"model_loads": 1, "model_calls": 1, "input_tokens": 10,
                       "memory": {"peak_allocated_bytes": 100, "peak_reserved_bytes": 200, "free_bytes": 30}}},
            {"status": "error", "error": {"code": "out_of_memory", "message": "test OOM again"},
             "usage": {"model_loads": 1, "model_calls": 1, "input_tokens": 10,
                       "memory": {"peak_allocated_bytes": 110, "peak_reserved_bytes": 190, "free_bytes": 50}}}]
        with self.pool() as pool:
            result = pool.call("qwen", self.payload())
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["usage"]["retries"], 1)
        self.assertEqual(result["usage"]["model_loads"], 2)
        self.assertEqual(result["usage"]["input_tokens"], 20)
        self.assertEqual(result["usage"]["peak_allocated_bytes"], 110)
        self.assertEqual(result["usage"]["peak_reserved_bytes"], 200)
        self.assertEqual(result["usage"]["min_free_bytes"], 30)
        self.assertEqual(len(FakeWorker.made), 2)
        left, right = [dict(w.requests[0]) for w in FakeWorker.made]
        left.pop("timeout_seconds"); right.pop("timeout_seconds")
        self.assertEqual(left, right)
        self.assertTrue(all(w.closed for w in FakeWorker.made))

    def test_cache_binds_image_geometry_prompt_config_and_artifacts(self):
        with self.pool(cache_dir=self.root / "cache") as pool:
            first = pool.call("qwen", self.payload())
            second = pool.call("qwen", self.payload(timeout_seconds=10))
            self.assertTrue(second["cache_hit"])
            self.assertEqual(second["usage"]["model_calls"], 0)
            self.assertEqual(second["source_usage"]["model_calls"], 1)
            self.assertIsNone(first["usage"]["peak_allocated_bytes"])
            self.assertIsNone(second["source_usage"]["peak_allocated_bytes"])
            self.assertEqual(second["usage"]["peak_allocated_bytes"], 0)
            self.assertEqual(second["usage"]["peak_reserved_bytes"], 0)
            self.assertFalse(pool.call("qwen", self.payload("other prompt"))["cache_hit"])
            changed = self.payload(); changed["geometry"]["crop_to_observation"][0][2] = 4
            self.assertFalse(pool.call("qwen", changed)["cache_hit"])
            Image.new("RGB", (16, 16), "blue").save(self.image)
            self.assertFalse(pool.call("qwen", self.payload())["cache_hit"])
            pool.models["qwen"]["seed"] = 7
            self.assertFalse(pool.call("qwen", self.payload())["cache_hit"])
            self.assertFalse(first["cache_hit"])

    def test_artifact_corruption_invalidates_successful_cache(self):
        artifact = self.root / "mask.npy"
        np.save(artifact, np.zeros((1, 16, 16), bool))
        from facade_change.io import sha256
        FakeWorker.replies = [{"status": "empty", "artifacts": [{"path": str(artifact), "sha256": sha256(artifact)}], "usage": {}}]
        with self.pool(cache_dir=self.root / "cache") as pool:
            request = {"op": "segment", "image_path": str(self.image), "text": "crack", "out": str(self.root / "out")}
            pool.call("sam3", request)
            self.assertTrue(pool.call("sam3", request)["cache_hit"])
            np.save(artifact, np.ones((1, 16, 16), bool))
            self.assertFalse(pool.call("sam3", request)["cache_hit"])

    def test_deadline_failure_is_visible_and_not_retried(self):
        FakeWorker.replies = [WorkerError("timeout", "synthetic timeout")]
        with self.pool() as pool:
            result = pool.call("qwen", self.payload(timeout_seconds=.1))
            self.assertEqual(result["error"]["code"], "timeout")
            self.assertTrue(result["usage"]["incomplete"])
            self.assertEqual(result["usage"]["retries"], 0)
            self.assertTrue(FakeWorker.made[0].closed)
            self.assertIsNone(result["usage"]["peak_allocated_bytes"])
            self.assertIsNone(result["usage"]["peak_reserved_bytes"])

    def test_startup_and_import_failures_leave_gpu_memory_unknown(self):
        with self.pool() as pool:
            with patch("facade_change.agentic.models._Worker", side_effect=WorkerError("missing_interpreter", "missing")):
                startup = pool.call("qwen", self.payload())
            self.assertEqual(startup["error"]["code"], "missing_interpreter")
            self.assertIsNone(startup["usage"]["peak_allocated_bytes"])
            self.assertIsNone(startup["usage"]["peak_reserved_bytes"])
            FakeWorker.replies = [{"status": "error", "error": {"code": "model_error", "message": "No module named torch"},
                                   "usage": {"model_loads": 0, "memory": {}}}]
            failed_import = pool.call("qwen", self.payload())
            self.assertIsNone(failed_import["usage"]["peak_allocated_bytes"])
            self.assertIsNone(failed_import["usage"]["peak_reserved_bytes"])
            self.assertTrue(FakeWorker.made[-1].closed)
            self.assertNotIn("qwen", pool.workers)


class ProcessorInputs:
    def __init__(self, tokens, images):
        self.values = {"input_ids": np.ones((1, tokens), np.int64)}
        if images:
            self.values.update(pixel_values=np.ones((1, 12)), image_grid_thw=np.array([[1, 1, 1]]))

    def __getitem__(self, name):
        return self.values[name]

    def __contains__(self, name):
        return name in self.values


class FakeProcessor:
    def __init__(self, tokens=10, materialize=True):
        self.tokens, self.materialize = tokens, materialize

    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.kwargs = messages, kwargs
        return ProcessorInputs(self.tokens, self.materialize)


class AdapterContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.image = Path(self.temp.name) / "rgb.png"
        Image.new("RGB", (16, 16), "white").save(self.image)

    def tearDown(self):
        self.temp.cleanup()

    def qwen(self, **options):
        adapter = QwenAdapter("qwen", options)
        adapter.checked = True
        adapter.processor = FakeProcessor()
        adapter._begin()
        return adapter

    def request(self):
        return {"op": "measure", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "compare"}, {"type": "image", "path": str(self.image), "evidence_id": "e1"}]}]}

    def test_dino_live_space_matches_shared_contract_and_tracks_encoder_resources(self):
        from facade_change.agentic.retrieval import dino_embedding_space
        checkpoint = Path(self.temp.name) / "synthetic-checkpoint.pth"
        checkpoint.write_bytes(b"contract fixture, not model weights")
        source = Path(self.temp.name) / "synthetic-source"
        source.mkdir()
        (source / "model.py").write_text("# Source fingerprint fixture\n")
        adapter = DinoAdapter("dino", {"checkpoint_path": str(checkpoint), "source_root": str(source)})
        with patch.dict(sys.modules, {"torch": SimpleNamespace()}):
            adapter._check()
        self.assertEqual(adapter.provenance["space"], dino_embedding_space(
            adapter.contract["checkpoint_sha256"], adapter.contract["python_tree_sha256"]))
        names = {Path(row["path"]).name for row in adapter.provenance["resources"]}
        self.assertTrue({"history_features.py", "dinov2.py", "retrieval.py"}.issubset(names))
        self.assertIsNone(adapter.model)

    def test_qwen_materializes_image_objects_and_disables_thinking(self):
        adapter = self.qwen()
        result = adapter.handle(self.request())
        image = adapter.processor.messages[0]["content"][1]["image"]
        self.assertIsInstance(image, Image.Image)
        self.assertFalse(adapter.processor.kwargs["enable_thinking"])
        self.assertEqual(result["input_tokens"], 10)
        self.assertFalse(result["model_loaded"])

    def test_limits_reject_without_silent_truncation_or_resize(self):
        adapter = self.qwen(max_input_tokens=9)
        with self.assertRaisesRegex(ValueError, "no truncation"):
            adapter.handle(self.request())
        adapter = self.qwen(max_image_pixels=100)
        with self.assertRaisesRegex(ValueError, "no implicit downscale"):
            adapter.handle(self.request())
        adapter = self.qwen(max_images=1)
        request = self.request(); request["messages"][0]["content"].append(request["messages"][0]["content"][1])
        with self.assertRaisesRegex(ValueError, "max_images"):
            adapter.handle(request)

    def test_text_paths_are_not_images_and_gt_fields_are_rejected(self):
        adapter = self.qwen()
        request = self.request(); request["messages"][0]["content"] = [{"type": "text", "text": str(self.image)}]
        with self.assertRaisesRegex(ValueError, "actual images"):
            adapter.handle(request)
        request = self.request(); request["labels"] = [[1]]
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            adapter.handle(request)
        adapter.processor = FakeProcessor(materialize=False)
        with self.assertRaisesRegex(RuntimeError, "image tensors"):
            adapter.handle(self.request())

    def test_worker_protocol_rejects_commands_and_survives_invalid_json(self):
        requests = io.StringIO('not-json\n' + json.dumps({"id": 2, "op": "shell", "command": "ls"}) + '\n')
        output = io.StringIO()
        with redirect_stderr(io.StringIO()):
            serve(requests, output)
        replies = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(replies), 2)
        self.assertTrue(all(row["status"] == "error" for row in replies))

    def test_sam_post_sigmoid_values_and_empty_output_are_preserved(self):
        class Tensor:
            def __init__(self, value): self.value = value
            def detach(self): return self
            def float(self): return Tensor(self.value.astype(np.float32))
            def cpu(self): return self
            def numpy(self): return self.value
        for count in (0, 1):
            probabilities = np.full((count, 1, 16, 16), .7, np.float32)
            state = {"masks_logits": Tensor(probabilities), "masks": Tensor(probabilities > .5),
                     "scores": Tensor(np.full(count, .8)), "boxes": Tensor(np.zeros((count, 4)))}
            processor = SimpleNamespace(set_image=lambda image: {}, set_text_prompt=lambda **kwargs: state)
            adapter = SAM3Adapter("sam3", {})
            adapter.processor = processor
            adapter._load = lambda: None
            adapter._begin()
            torch = SimpleNamespace(inference_mode=nullcontext, autocast=lambda *a, **kw: nullcontext(), bfloat16="bf16")
            out = Path(self.temp.name) / f"sam-{count}"
            with patch.dict(sys.modules, {"torch": torch}):
                result = adapter.handle({"op": "segment", "image_path": str(self.image), "text": "crack", "out": str(out)})
            self.assertEqual(result["status"], "ok" if count else "empty")
            self.assertEqual(result["native_shape"], [16, 16])
            np.testing.assert_array_equal(np.load(result["probabilities_path"]), probabilities[:, 0])
            # Applying sigmoid again would change .7 to ~.668; reject that drift.
            self.assertEqual(np.load(result["masks_path"]).dtype, np.bool_)

    def test_real_subprocess_protocol_starts_outside_cwd_without_torch(self):
        # Init imports only lightweight adapter modules. Unknown op is rejected
        # before any model import/download; this is a process contract smoke.
        with tempfile.TemporaryDirectory() as folder:
            worker = _Worker("qwen", {"python": sys.executable}, folder, time.monotonic() + 10)
            try:
                response = worker.request({"op": "shell", "command": "anything"}, time.monotonic() + 10)
                self.assertEqual(response["status"], "error")
                self.assertIn("Unsupported Qwen request", response["error"]["message"])
            finally:
                worker.close()


if __name__ == "__main__":
    unittest.main()
