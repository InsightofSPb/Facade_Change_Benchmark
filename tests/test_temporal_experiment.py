"""CPU integration of persistent workers, saved reports and resumable histories."""
import contextlib
import copy
import importlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from facade_change.io import read_json, sha256
from facade_change.methods.remote import RemotePool
from facade_change.methods.rgb import RGBScorer


runner = importlib.import_module("facade_change.2026-10-14_temporal_experiment")
spec = importlib.util.spec_from_file_location(
    "temporal_input_fixtures", Path(__file__).with_name("test_temporal_inputs.py"))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


class TemporalSmokeWindowTests(unittest.TestCase):
    def frames(self, count=5, shape=(512, 512)):
        y, x = np.indices(shape)
        frames = []
        for index in range(count):
            rgb = np.stack(((x + index) % 256, (y + 7 * index) % 256,
                            (x + y + 13 * index) % 256), axis=2).astype(np.uint8)
            support = np.zeros(shape, bool)
            support[200:260, 300:360] = True
            if index >= 2:
                support[:] = False
                support[221:241, 323:343] = True
            frames.append({"rgb": rgb, "support": support})
        return frames

    def test_window_is_native_deterministic_and_common_to_all_frames(self):
        frames = self.frames()
        original = copy.deepcopy(frames)
        cropped, selection = runner._smoke_window(frames)
        again, repeated = runner._smoke_window(frames)
        self.assertEqual(selection, repeated)
        x, y, width, height = selection["crop_xywh"]
        self.assertGreaterEqual(x, 323)
        self.assertGreaterEqual(y, 221)
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)
        self.assertLessEqual(width, 16)
        self.assertLessEqual(height, 16)
        self.assertEqual(len(cropped), 5)
        for full, before, small, duplicate in zip(frames, original, cropped, again):
            np.testing.assert_array_equal(small["rgb"], full["rgb"][y:y+height, x:x+width])
            np.testing.assert_array_equal(small["support"], full["support"][y:y+height, x:x+width])
            self.assertTrue(small["support"].all())
            for key in ("rgb", "support"):
                np.testing.assert_array_equal(small[key], duplicate[key])
                np.testing.assert_array_equal(full[key], before[key])
        # Returned samples must not expose mutable views of the prepared input.
        cropped[0]["rgb"][:] = 0
        cropped[0]["support"][:] = False
        np.testing.assert_array_equal(frames[0]["rgb"], original[0]["rgb"])
        np.testing.assert_array_equal(frames[0]["support"], original[0]["support"])

    def test_small_inputs_are_not_resized_and_disjoint_support_is_rejected(self):
        frames = [{"rgb": np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3),
                   "support": np.ones((7, 11), bool)} for _ in range(2)]
        cropped, selection = runner._smoke_window(frames)
        self.assertEqual(selection["crop_xywh"], [0, 0, 11, 7])
        for full, small in zip(frames, cropped):
            np.testing.assert_array_equal(small["rgb"], full["rgb"])
        frames[0]["support"][:, 5:] = False
        frames[1]["support"][:, :5] = False
        with self.assertRaises(ValueError):
            runner._smoke_window(frames)

    def test_msdzip_smoke_bounds_pair_and_every_history_call(self):
        native = self.frames()
        sequences = [{"domain": domain, "sequence_id": domain + "-example",
                      "frames": list(range(5 if domain == "synthetic" else 2))}
                     for domain in ("synthetic", "real")]
        inputs = runner._module("inputs")
        for method in ("msdzip_abs", "msdzip_mod256"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                calls, events = [], []
                scorer = self.recording_scorer(calls)
                store = SimpleNamespace(out=Path(directory), event=lambda *args, **kwargs: events.append((args, kwargs)))
                with patch.object(inputs, "load_frame", side_effect=lambda sequence, index: native[index]):
                    runner._smoke(store, sequences, {"id": method, "method": method}, scorer,
                                  {"history_methods": [method], "memory_exemplars": 3})
                self.assertGreater(len(calls), 2, "history must invoke real scoring in addition to both pairs")
                for reference, source, support in calls:
                    self.assertLessEqual(source.shape[0], 16)
                    self.assertLessEqual(source.shape[1], 16)
                    self.assertEqual(reference.shape, source.shape)
                    self.assertEqual(support.shape, source.shape[:2])
                    self.assertTrue(support.any())
                marker = read_json(Path(directory) / "smoke" / f"{method}.json")
                for domain in ("synthetic", "real"):
                    selection = marker["smoke_inputs"][domain]
                    self.assertIn("selection_policy", selection)
                    self.assertIn("native_shape", selection)
                    self.assertIn("scored_shape", selection)
                    x, y, width, height = selection["crop_xywh"]
                    pair = calls[0 if domain == "synthetic" else 1]
                    np.testing.assert_array_equal(pair[0], native[0]["rgb"][y:y+height, x:x+width])
                    np.testing.assert_array_equal(pair[1], native[1]["rgb"][y:y+height, x:x+width])
                with np.load(Path(directory) / marker["artifact"], allow_pickle=False) as saved:
                    self.assertEqual(int(saved["memory__next_frame"]), 5)
                    self.assertLessEqual(saved["memory__rgb"].shape[-3], 16)
                    self.assertLessEqual(saved["memory__rgb"].shape[-2], 16)

    @staticmethod
    def recording_scorer(calls):
        class RecordingScorer(RGBScorer):
            def __call__(self, reference, source, support):
                calls.append((reference.copy(), source.copy(), support.copy()))
                return super().__call__(reference, source, support)
        return RecordingScorer("rgb_diff")

    def test_other_smoke_methods_and_full_msdzip_pairs_keep_native_crops(self):
        frames = self.frames(count=2)
        sequence = {"domain": "real", "frames": [0, 1], "sequence_id": "native",
                    "split": "val", "building_id": "building"}
        synthetic = {**sequence, "domain": "synthetic"}
        inputs = runner._module("inputs")
        calls = []
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(inputs, "load_frame", side_effect=lambda sequence, index: frames[index]):
            store = SimpleNamespace(out=Path(directory), event=lambda *args, **kwargs: None)
            runner._smoke(store, [synthetic, sequence], {"id": "rgb_diff", "method": "rgb_diff"},
                          self.recording_scorer(calls), {"history_methods": [], "memory_exemplars": 3})
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(source.shape == (512, 512, 3) for _, source, _ in calls))
            calls.clear()
            committed = []

            def stop_after_one(record, arrays):
                committed.append(arrays)
                raise KeyboardInterrupt("inspect first actual full-resolution inference unit")

            store.read_unit = lambda identifier: None
            store.save_unit = stop_after_one
            with self.assertRaises(KeyboardInterrupt):
                runner._pairs(store, [sequence], {"id": "msdzip_abs", "method": "msdzip_abs", "variant": "current"},
                              self.recording_scorer(calls))
            self.assertEqual(len(calls), 1)
            np.testing.assert_array_equal(calls[0][0], frames[0]["rgb"])
            np.testing.assert_array_equal(calls[0][1], frames[1]["rgb"])
            self.assertEqual(committed[0]["scores"].shape, (512, 512))


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required")
class TemporalExperimentIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TemporalInputsTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.history = self.fixture.history()
        # Isolate concurrent working-tree edits during the CPU test while still
        # testing the runner's exact contract comparison and source refusal.
        self.environment = runner.environment()
        self.options = {"history_run": str(self.history), "device": "cpu",
                        "methods": ["rgb_diff", "ssim"], "history_methods": [],
                        "methods_config": "", "semantic_run": "",
                        "method_options": {name: {"worker_python": sys.executable}
                                           for name in ("rgb_diff", "ssim")}}

    def run_experiment(self, name="run", **kwargs):
        options = dict(self.options) if not kwargs.get("resume") and not kwargs.get("report_only") else {}
        options.update(kwargs)
        with patch.object(runner, "environment", return_value=copy.deepcopy(self.environment)), \
                contextlib.redirect_stdout(io.StringIO()):
            return runner.run_temporal_experiment(self.root / name, **options)

    def records(self, name):
        return [read_json(path) for path in sorted((self.root / name / "units").glob("*.json"))]

    def arrays(self, name, record):
        with np.load(self.root / name / record["artifact_path"], allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}

    def test_persistent_cpu_workers_and_standalone_report_rebuild(self):
        calls = []
        activate = RemotePool.activate

        def observe(pool, method, options):
            result = activate(pool, method, options)
            calls.append((method, pool.process.pid))
            return result

        with patch.object(RemotePool, "activate", observe):
            result = self.run_experiment()
        self.assertEqual(result["status"], "completed_exploratory",
                         {key: value.get("error") for key, value in result["methods"].items()})
        self.assertEqual(Counter(row["method"] for row in self.records("run")), {"rgb_diff": 32, "ssim": 32})
        counts = Counter(calls)
        for method in self.options["methods"]:
            self.assertEqual(len({pid for candidate, pid in counts if candidate == method}), 2)
            self.assertGreater(max(n for (candidate, _), n in counts.items() if candidate == method), 8)
        output = self.root / "run"
        reports = ["2026-10-14_temporal_thresholds.json", "2026-10-14_temporal_summary.json",
                   "2026-10-14_temporal_gallery.html"]
        before = {name: sha256(output / name) for name in reports}
        self.history.rename(self.root / "history-unavailable")
        with patch.object(runner, "_scorer", side_effect=AssertionError("inference during report rebuild")), \
                patch.object(runner._module("inputs"), "load_sequences", side_effect=AssertionError("input reload")):
            self.run_experiment(report_only=True)
        self.assertEqual(before, {name: sha256(output / name) for name in reports})
        self.assertEqual(read_json(output / "run.json")["status"], "reports_rebuilt")

    def test_interruption_resume_preserves_committed_units_and_matches_fresh_run(self):
        self.options["methods"] = ["rgb_diff"]
        save = runner.RunStore.save_unit
        completed = []

        def interrupt(store, record, arrays):
            result = save(store, record, arrays)
            completed.append(record["unit_id"])
            if len(completed) == 7:
                raise KeyboardInterrupt("controlled interruption after atomic commit")
            return result

        with patch.object(runner.RunStore, "save_unit", interrupt), self.assertRaises(KeyboardInterrupt):
            self.run_experiment("resumed")
        self.assertEqual(read_json(self.root / "resumed/run.json")["status"], "interrupted")
        saved = {row["unit_id"]: row["artifact_sha256"] for row in self.records("resumed")}
        self.assertEqual(len(saved), 7)
        interrupted_tail = b'{"sequence":999,"event_type":"incomplete'
        with (self.root / "resumed/events.jsonl").open("ab") as stream:
            stream.write(interrupted_tail)
        self.assertEqual(self.run_experiment("resumed", resume=True)["status"], "completed_exploratory")
        fragments = list((self.root / "resumed").glob("2026-10-14-interrupted-ledger-*.bin"))
        self.assertEqual(len(fragments), 1)
        self.assertEqual(fragments[0].read_bytes(), interrupted_tail)
        self.run_experiment("fresh")
        resumed = {row["unit_id"]: row for row in self.records("resumed")}
        fresh = {row["unit_id"]: row for row in self.records("fresh")}
        self.assertEqual(resumed.keys(), fresh.keys())
        self.assertEqual(saved, {key: resumed[key]["artifact_sha256"] for key in saved})
        for key in resumed:
            a, b = self.arrays("resumed", resumed[key]), self.arrays("fresh", fresh[key])
            self.assertEqual(a.keys(), b.keys())
            for name in a:
                np.testing.assert_array_equal(a[name], b[name])
        self.assertEqual(read_json(self.root / "resumed/2026-10-14_temporal_thresholds.json"),
                         read_json(self.root / "fresh/2026-10-14_temporal_thresholds.json"))
        events = [json.loads(line) for line in (self.root / "resumed/events.jsonl").read_text().splitlines()]
        self.assertEqual(len({event["sequence"] for event in events}), len(events))
        self.assertEqual(sum(event["event_type"] == "unit_completed" for event in events), 32)

    def test_resume_refuses_changed_configuration_source_and_artifact(self):
        self.options["methods"] = ["rgb_diff"]
        self.run_experiment(preflight_only=True)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            self.run_experiment(resume=True, **{**self.options, "seed": 99})
        changed = copy.deepcopy(self.environment)
        changed["source_sha256"]["2026-10-14_temporal_experiment.py"] = "changed-source"
        with patch.object(runner, "environment", return_value=changed), \
                self.assertRaisesRegex(ValueError, "source code or environment changed"):
            runner.run_temporal_experiment(self.root / "run", resume=True)
        manifest = read_json(self.history / "manifest.json")
        relative = manifest["sequences"][0]["frames"][0]["paths"]["rgb"]
        path = self.history / relative
        path.write_bytes(path.read_bytes() + b"changed-source-artifact")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.run_experiment(resume=True)

    def test_one_inference_failure_keeps_other_method_and_partial_outputs(self):
        original = runner._scorer

        class LimitedScorer:
            def __init__(self, wrapped):
                self.wrapped = wrapped
                self.calls = 0

            def __getattr__(self, name):
                return getattr(self.wrapped, name)

            def __call__(self, reference, source, support):
                self.calls += 1
                if self.calls == 3:
                    raise RuntimeError("injected method failure")
                return self.wrapped(reference, source, support)

        def scorer(variant, pool):
            wrapped = original(variant, pool)
            return LimitedScorer(wrapped) if variant["method"] == "ssim" else wrapped

        with patch.object(runner, "_scorer", scorer):
            result = self.run_experiment()
        self.assertEqual(result["status"], "completed_with_failures")
        self.assertEqual(result["methods"]["rgb_diff"]["status"], "completed")
        self.assertEqual(result["methods"]["ssim"]["stage"], "inference")
        self.assertEqual(Counter(row["method"] for row in self.records("run")), {"rgb_diff": 32, "ssim": 2})
        self.assertTrue((self.root / "run/2026-10-14_temporal_gallery.html").is_file())

    def test_history_uses_only_past_frames_and_future_train_edit_cannot_change_prefix(self):
        self.options.update(methods=["dinov2"], history_methods=["dinov2"], method_options={})
        inputs = runner._module("inputs")
        real_loader = inputs.load_frame

        def rgb_backend(variant, pool):
            scorer = RGBScorer("rgb_diff")
            scorer.metadata.update(method="dinov2", test_backend="rgb_diff")
            scorer.activate = lambda: scorer.metadata
            return scorer

        def future_edit(sequence, index):
            frame = real_loader(sequence, index)
            if sequence["split"] == "train" and sequence["arm"] == "event" and index == 4:
                frame["rgb"] = 255 - frame["rgb"]
            return frame

        with patch.object(runner, "_scorer", rgb_backend):
            self.assertEqual(self.run_experiment("history-run")["status"], "completed_exploratory")
            with patch.object(inputs, "load_frame", future_edit):
                self.assertEqual(self.run_experiment("future-edited")["status"], "completed_exploratory")
        first = {row["unit_id"]: row for row in self.records("history-run") if row["mode"] == "history"}
        second = {row["unit_id"]: row for row in self.records("future-edited") if row["mode"] == "history"}
        self.assertEqual(len(first), 20)
        for identifier, record in first.items():
            arrays = self.arrays("history-run", record)
            self.assertEqual(int(arrays["memory__next_frame"]), record["frame"] + 1)
            self.assertTrue(np.all(arrays["memory__frame_ids"] <= record["frame"]))
            self.assertTrue(np.all(arrays["reference_frame"][arrays["support"]] < record["frame"]))
            if record["frame"] < 4:
                changed = self.arrays("future-edited", second[identifier])
                for key in arrays:
                    np.testing.assert_array_equal(arrays[key], changed[key])

    def test_history_resume_restores_committed_memory_chain(self):
        self.options.update(methods=["dinov2"], history_methods=["dinov2"], method_options={})
        save = runner.RunStore.save_unit
        history_count = 0

        def rgb_backend(variant, pool):
            scorer = RGBScorer("rgb_diff")
            scorer.metadata.update(method="dinov2", test_backend="rgb_diff")
            scorer.activate = lambda: scorer.metadata
            return scorer

        def interrupt(store, record, arrays):
            nonlocal history_count
            result = save(store, record, arrays)
            history_count += record["mode"] == "history"
            if history_count == 7:
                raise KeyboardInterrupt("interrupt after saved memory state")
            return result

        with patch.object(runner, "_scorer", rgb_backend):
            with patch.object(runner.RunStore, "save_unit", interrupt), self.assertRaises(KeyboardInterrupt):
                self.run_experiment("history-resumed")
            before = {row["unit_id"]: row["artifact_sha256"] for row in self.records("history-resumed")}
            self.run_experiment("history-resumed", resume=True)
            self.run_experiment("history-fresh")
        resumed = {row["unit_id"]: row for row in self.records("history-resumed")}
        fresh = {row["unit_id"]: row for row in self.records("history-fresh")}
        self.assertEqual(resumed.keys(), fresh.keys())
        self.assertEqual(before, {key: resumed[key]["artifact_sha256"] for key in before})
        for key, record in resumed.items():
            if record["mode"] != "history":
                continue
            a, b = self.arrays("history-resumed", record), self.arrays("history-fresh", fresh[key])
            for name in a:
                np.testing.assert_array_equal(a[name], b[name])


if __name__ == "__main__":
    unittest.main()
