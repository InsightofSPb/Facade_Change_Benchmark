"""Temporal inference queue: CPU fakes exercise causal and resumable contracts."""
import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.real_eval import night_run as runner
from scripts.real_eval.robustness_common import atomic_json, atomic_npz, read_json, sha256


def metadata(method):
    return {"method": method, "fixture": "temporal-test",
            "output_kind": "native_mask" if method in runner.NATIVE_METHODS else "score"}


def fixture(root, methods=("rgb_diff", "dinov2", "geoscd")):
    source, prepared = root/"source", root/"prepared"
    (source/"method_provenance").mkdir(parents=True)
    (source/"maps").mkdir()
    (prepared/"frames").mkdir(parents=True)
    for method in methods:
        atomic_json(source/"method_provenance"/f"full_{method}.json", metadata(method))
    frames = []
    for index, value in enumerate((10, 30, 60, 90)):
        path = prepared/"frames"/f"{index}.npz"
        support = np.ones((3, 4), bool)
        support[-1, -1] = False
        atomic_npz(path, {"rgb": np.full((3, 4, 3), value, np.uint8), "support": support,
                          "packed": np.full((3, 4), 1 << (index % 2), np.uint16)})
        frames.append({"frame": index, "year": 2000+index, "image_id": index,
                       "artifact_path": str(path.relative_to(prepared)), "artifact_sha256": sha256(path)})
    manifest = {"schema_version": 1, "kind": "real_temporal_prepared", "source_run": str(source),
                "source_config": {"device": "cpu", "method_options": {}}, "source_records": [],
                "source_summary": {"calibration": [{"method": "dinov2", "variant": "frozen",
                    "mode": "previous", "target": "step_event", "status": "calibrated",
                    "selection_split": "val", "selection_domain": "synthetic", "selection_arm": "clean",
                    "choices": [{"fpr_target": .01, "threshold": 1.}]}]},
                "categories": [{"id": 1, "name": "one"}, {"id": 2, "name": "two"}],
                "sequences": [{"sequence_id": "real-fixture-x0-y0", "building_id": "facade-test",
                               "split": "test", "view_id": "view", "crop_xywh": [0, 0, 4, 3],
                               "frames": frames}]}
    atomic_json(prepared/"manifest.json", manifest)
    return prepared, source


class Factory:
    def __init__(self, fail=False, interrupt_at=None):
        self.calls, self.activations, self.releases = [], [], 0
        self.fail, self.interrupt_at = fail, interrupt_at

    def __call__(self, **kwargs):
        owner = self

        class Pool:
            process = None
            stderr_path = None
            last_worker_retained = False

            def activate(self, method, options):
                owner.activations.append(method)
                return metadata(method)

            def score(self, method, options, reference, source, support):
                # A strict signature and raw uint8 assertions ensure neither
                # annotations, thresholds nor future frames reach the scorer.
                assert reference.dtype == source.dtype == np.uint8
                assert support.dtype == bool
                owner.calls.append((method, int(reference[0, 0, 0]), int(source[0, 0, 0]), support.copy()))
                if owner.interrupt_at == len(owner.calls):
                    owner.interrupt_at = None
                    raise KeyboardInterrupt()
                if method == "geoscd" and owner.fail:
                    owner.fail = False
                    self.last_worker_retained = True
                    raise RuntimeError("Inference worker failed during score: "+runner.RECOVERABLE_SCORE_ERROR+"; stderr log: fixture")
                values = np.abs(reference.astype(np.float32)-source).mean(axis=2)/255
                output = {"metadata": metadata(method)}
                if method in runner.NATIVE_METHODS:
                    output["native_prediction"] = values > .1
                    values = output["native_prediction"].astype(np.float32)
                values[~support] = np.nan
                output["scores"] = values
                return output

            def release(self):
                owner.releases += 1

        return Pool()


class TemporalRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.prepared, self.source = fixture(self.root)
        self.out = self.root/"out"
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.gpu = patch.object(runner, "_snapshot_gpu")
        self.gpu.start()

    def tearDown(self):
        self.gpu.stop()
        self.quiet.__exit__(None, None, None)
        self.temp.cleanup()

    def execute(self, factory, **kwargs):
        return runner.run(self.prepared, self.out, pool_factory=factory,
                          methods=kwargs.pop("methods", ("rgb_diff", "dinov2")), **kwargs)

    def test_mode_order_references_causal_memory_and_resume(self):
        factory = Factory()
        result = self.execute(factory)
        self.assertEqual((result["status"], result["completed"]), ("completed", 15))
        self.assertEqual(factory.activations, ["rgb_diff", "dinov2", "rgb_diff", "dinov2", "dinov2"])
        self.assertEqual(len(factory.calls), 16)
        self.assertEqual([(a, b) for _, a, b, _ in factory.calls[:3]], [(10, 30), (30, 60), (60, 90)])
        self.assertEqual([(a, b) for _, a, b, _ in factory.calls[6:8]], [(10, 60), (10, 90)])
        history_calls = factory.calls[-6:]
        self.assertEqual([(a, b) for _, a, b, _ in history_calls],
                         [(10, 30), (10, 60), (30, 60), (10, 90), (30, 90), (60, 90)])
        history_rows = [r for r in result["results"] if r["mode"] == "history"]
        self.assertEqual(history_rows[1]["previous_history_artifact"], history_rows[0]["artifact_sha256"])
        with np.load(self.out/history_rows[-1]["artifact_path"], allow_pickle=False) as arrays:
            self.assertEqual(int(arrays["memory__next_frame"]), 4)
            self.assertNotIn("packed", arrays)
            self.assertNotIn("labels", arrays)
            np.testing.assert_array_equal(arrays["reference"], np.full((3, 4, 3), 60, np.uint8))
        resumed = Factory()
        self.assertEqual(self.execute(resumed, resume=True)["completed"], 15)
        self.assertEqual(resumed.calls, [])

    def test_first_earliest_pair_reuses_previous_without_changing_target(self):
        factory = Factory()
        result = self.execute(factory, methods=("rgb_diff",), modes=("previous", "earliest"))
        self.assertEqual(result["completed"], 6)
        self.assertEqual(len(factory.calls), 5)
        previous = next(r for r in result["results"] if r["mode"] == "previous" and r["frame"] == 1)
        earliest = next(r for r in result["results"] if r["mode"] == "earliest" and r["frame"] == 1)
        self.assertEqual(earliest["origin"], "reused_previous_first_pair")
        self.assertEqual(earliest["target"], "earliest_state")
        self.assertEqual(earliest["source_result_unit_id"], previous["unit_id"])
        with np.load(self.out/previous["artifact_path"], allow_pickle=False) as left:
            with np.load(self.out/earliest["artifact_path"], allow_pickle=False) as right:
                np.testing.assert_array_equal(left["scores"], right["scores"])
                np.testing.assert_array_equal(left["support"], right["support"])

    def test_history_interruption_restores_previous_checkpoint(self):
        factory = Factory(interrupt_at=2)
        result = self.execute(factory, methods=("dinov2",), modes=("history",))
        self.assertEqual((result["status"], result["completed"]), ("interrupted", 1))
        resumed = Factory()
        result = self.execute(resumed, methods=("dinov2",), modes=("history",), resume=True)
        self.assertEqual((result["status"], result["completed"]), ("completed", 3))
        self.assertEqual(len(resumed.calls), 5)
        self.assertEqual([(a, b) for _, a, b, _ in resumed.calls[:2]], [(10, 60), (30, 60)])

    def test_soft_budget_resume_does_not_change_contract(self):
        factory = Factory()
        result = self.execute(factory, max_hours=0)
        self.assertEqual((result["status"], result["unprocessed"]), ("budget_exhausted", 15))
        self.assertFalse(factory.activations)
        self.assertEqual(self.execute(Factory(), resume=True, max_hours=1)["completed"], 15)

    def test_geoscd_failed_pair_remains_missing_and_retry_is_one_call(self):
        factory = Factory(fail=True)
        result = self.execute(factory, methods=("geoscd",), modes=("previous", "earliest"))
        self.assertEqual((result["completed"], result["failed"]), (5, 1))
        failed = next(r for r in result["results"] if r["status"] == "failed")
        self.assertNotIn("artifact_path", failed)
        self.assertEqual(len(factory.calls), 6)
        no_retry = Factory()
        self.execute(no_retry, methods=("geoscd",), modes=("previous", "earliest"), resume=True)
        self.assertFalse(no_retry.calls)
        retry = Factory()
        result = self.execute(retry, methods=("geoscd",), modes=("previous", "earliest"), resume=True, retry_failed=True)
        self.assertEqual((result["completed"], result["failed"]), (6, 0))
        self.assertEqual(len(retry.calls), 1)

    def test_cache_reused_only_for_exact_rgb_support_and_provenance(self):
        manifest = read_json(self.prepared/"manifest.json")
        seq = manifest["sequences"][0]
        inputs, _, _ = runner._input(self.prepared, seq, 1, "previous")
        scores = np.where(inputs["support"], .2, np.nan).astype(np.float32)
        path = self.source/"maps"/"cache.npz"
        atomic_npz(path, {**inputs, "scores": scores})
        manifest["source_records"] = [{"unit_id": "cache", "method": "rgb_diff", "variant": "frozen",
            "domain": "real", "mode": "previous", "split": "test", "status": "completed",
            "sequence_id": seq["sequence_id"], "building_id": seq["building_id"], "frame": 1,
            "artifact_path": "maps/cache.npz", "artifact_sha256": sha256(path),
            "method_metadata": metadata("rgb_diff")}]
        atomic_json(self.prepared/"manifest.json", manifest)
        factory = Factory()
        result = self.execute(factory, methods=("rgb_diff",), modes=("previous",))
        self.assertEqual(len(factory.calls), 2)
        self.assertEqual(result["results"][0]["origin"], "reused_source")
        measured = Factory()
        pilot = runner.run(self.prepared, self.root/"pilot", methods=("rgb_diff",),
                           modes=("previous",), pool_factory=measured, pilot=True, pilot_pairs=1)
        self.assertEqual(len(measured.calls), 1)
        self.assertEqual(pilot["results"][0]["origin"], "inference")
        changed = copy.deepcopy(manifest)
        changed["source_records"][0]["artifact_sha256"] = "0"*64
        atomic_json(self.prepared/"manifest.json", changed)
        factory = Factory()
        result = runner.run(self.prepared, self.root/"uncached", methods=("rgb_diff",),
                            modes=("previous",), pool_factory=factory)
        self.assertEqual(len(factory.calls), 3)
        self.assertEqual(result["results"][0]["origin"], "inference")

    def test_artifact_corruption_rejected_before_gpu(self):
        (self.prepared/"frames/1.npz").write_bytes(b"changed")
        factory = Factory()
        with self.assertRaisesRegex(ValueError, "Prepared frame changed"):
            self.execute(factory)
        self.assertEqual(factory.activations, [])

    def test_one_unavailable_model_does_not_block_other_methods(self):
        factory = Factory()
        result = self.execute(factory, methods=("rgb_diff", "msdzip_abs"), modes=("previous",))
        self.assertEqual(result["status"], "completed_with_failures")
        self.assertEqual((result["completed"], result["unprocessed"]), (3, 3))
        self.assertIn("previous/msdzip_abs", result["method_failures"])
        self.assertEqual(factory.activations, ["rgb_diff"])

    def test_pilot_is_bounded_and_projection_separates_initialization(self):
        factory = Factory()
        result = self.execute(factory, pilot=True, pilot_pairs=1, pilot_history_frames=2)
        self.assertEqual(result["completed"], 6)
        self.assertEqual(len(factory.calls), 7)
        timing = read_json(self.out/"timing.json")
        self.assertFalse(timing["unmeasured"])
        self.assertEqual(len(timing["methods"]), 5)
        self.assertIsNotNone(timing["estimated_hours_high"])
        self.assertTrue(all("initialization_seconds" in row for row in timing["methods"]))


if __name__ == "__main__":
    unittest.main()
