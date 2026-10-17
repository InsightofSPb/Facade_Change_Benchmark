"""CPU-only GeoSCD retry orchestration, input identity, and resume checks."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import queue
import sys
import tempfile
import unittest

import numpy as np


from scripts.real_eval import geoscd_real_retry as retry

SCRIPT = Path(retry.__file__)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(root):
    return {str(path.relative_to(root)): sha(path) for path in root.rglob("*") if path.is_file()}


def fixture(directory):
    root, review, snapshot = (directory / name for name in ("run", "review", "snapshot"))
    for folder in (root / "maps", root / "units", review, snapshot, directory / "upstream"):
        folder.mkdir(parents=True)
    for name in ("vggt.pt", "sam.pt"):
        (directory / name).write_text("fixture checkpoint; never loaded by a model")
    options = dict(worker_python=sys.executable, source_root=str(directory / "upstream"),
                   checkpoint_path=str(directory / "vggt.pt"), sam_checkpoint=str(directory / "sam.pt"),
                   device="cuda:2", seed=17, mode="occupy", points_per_side=7, iou_thresh=.61)
    config = dict(device="cuda:0", method_options={"geoscd": options})
    (root / "resolved_config.json").write_text(json.dumps(config))
    (root / "run.json").write_text(json.dumps(dict(config=config, status="completed_exploratory")))
    rows, inputs = [], {}
    for number, building in enumerate(("A", "B", "C")):
        for mode in ("previous", "earliest"):
            for frame in (1, 2):
                reference_frame = frame - 1 if mode == "previous" else 0
                arrays = dict(reference=np.full((3, 4, 3), 40 * number + 10 * reference_frame, np.uint8),
                              source=np.full((3, 4, 3), 40 * number + 10 * frame, np.uint8),
                              support=np.ones((3, 4), bool), scores=np.zeros((3, 4), np.float32),
                              labels=np.full((3, 4), 255, np.uint8))
                arrays["support"][-1, -1] = False
                arrays["scores"][~arrays["support"]] = np.nan
                identifier = f"rgb-{building}-{mode}-{frame}"
                path = root / "maps" / f"{identifier}.npz"
                np.savez_compressed(path, **arrays)
                row = dict(unit_id=identifier, method="rgb_diff", variant="frozen", variant_id="rgb_diff",
                           mode=mode, target="earliest_state" if mode == "earliest" else "step_event",
                           domain="real", split="val", building_id=building, sequence_id=f"{building}-view-x0-y0",
                           frame=frame, status="completed", output_kind="score", label_verified=False,
                           hypothesis="H1", arm=None, event="unknown", nuisance="unknown", elapsed_seconds=.01,
                           artifact_path=str(path.relative_to(root)), artifact_sha256=sha(path))
                rows.append(row)
                inputs[identifier] = (copy.deepcopy(row), arrays)
    cached_row, cached_arrays = inputs["rgb-A-previous-1"]
    cached_arrays = {key: value.copy() for key, value in cached_arrays.items()}
    cached_arrays["native_prediction"] = np.zeros((3, 4), bool)
    cached_arrays["native_prediction"][0, 0] = True
    cached_arrays["scores"] = cached_arrays["native_prediction"].astype(np.float32)
    cached_arrays["scores"][~cached_arrays["support"]] = np.nan
    path = root / "maps" / "cached-geoscd.npz"
    np.savez_compressed(path, **cached_arrays)
    rows.append(dict(cached_row, unit_id="cached-geoscd", method="geoscd", variant_id="geoscd",
                     output_kind="native_mask", artifact_path=str(path.relative_to(root)), artifact_sha256=sha(path),
                     method_metadata=dict(method="geoscd", output_kind="native_mask", configured_options=options)))
    for domain, split in (("real", "test"), ("real", "train"), ("synthetic", "val")):
        rows.append(dict(rows[0], unit_id=f"FORBIDDEN-{domain}-{split}", domain=domain, split=split,
                         sequence_id=f"FORBIDDEN-{domain}-{split}", building_id="FORBIDDEN", artifact_path="missing.npz"))
    for row in rows:
        (root / "units" / f"{row['unit_id']}.json").write_text(json.dumps(row))
    (review / "summary.json").write_text(json.dumps(dict(selected_sequences=[f"{b}-view-x0-y0" for b in ("A", "B", "C")])))
    (review / "provenance.json").write_text(json.dumps(dict(source_run=str(root), snapshot=str(snapshot))))
    (snapshot / "2026-10-17_source_units.json").write_text(json.dumps(rows))
    (snapshot / "2026-10-17_quick_summary.json").write_text(json.dumps(dict(source_run=str(root))))
    return root, review, snapshot, options, inputs, cached_arrays


class FakePool:
    def __init__(self, calls, failures, **kwargs):
        self.calls, self.failures = calls, failures
        self.metadata = dict(method="geoscd", output_kind="native_mask", fixture=True)
        self.stderr_path = Path(kwargs.get("log_dir", "/tmp")) / "fake-geoscd.log"

    def activate(self, method, options):
        assert method == "geoscd"
        return self.metadata

    def score(self, method, options, reference, source, support):
        self.calls.append(dict(method=method, options=copy.deepcopy(options),
                               reference=reference.copy(), source=source.copy(), support=support.copy()))
        identity = int(reference[0, 0, 0]), int(source[0, 0, 0])
        if identity in self.failures:
            self.failures.remove(identity)
            raise RuntimeError("Inference worker failed during score: ValueError: No finite valid SAM correspondence similarities; stderr log: fake-geoscd.log")
        native = np.indices(support.shape).sum(axis=0) % 2 == 0
        scores = native.astype(np.float32)
        scores[~support] = np.nan
        return dict(scores=scores, native_prediction=native, metadata=self.metadata)

    def release(self):
        pass

    def close(self):
        self.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


class GeoSCDRealRetryTests(unittest.TestCase):
    def protocol_pool(self, response=None, *, timeout=False, dead=False):
        repository = str(Path(__file__).resolve().parents[1])
        if repository not in sys.path:
            sys.path.insert(0, repository)
        from facade_change.methods.remote import RemotePool, _json_value

        class Process:
            def __init__(self):
                self.stdin, self.stdout = io.StringIO(), io.StringIO()
                self.dead, self.killed = dead, False

            def poll(self):
                return 1 if self.dead else None

            def kill(self):
                self.dead, self.killed = True, True

            def wait(self, timeout=None):
                return 0

        pool = retry.make_geoscd_pool(RemotePool, _json_value)()
        process = Process()
        pool.process, pool._key = process, "original-worker-and-model-options"
        pool._responses, pool._active_timeout = queue.Queue(), .001
        if not timeout:
            pool._responses.put(response)
        return pool, process

    def test_exact_score_correspondence_error_keeps_worker_for_next_request(self):
        message = "ValueError: No finite valid SAM correspondence similarities"
        pool, process = self.protocol_pool(json.dumps(dict(id=1, ok=False, error=message)))
        responses, key = pool._responses, pool._key
        try:
            with self.assertRaisesRegex(RuntimeError, "No finite valid SAM correspondence similarities"):
                pool._request("score", input_path="pair-one.npz", output_path="result-one.npz")
            self.assertIs(pool.process, process)
            self.assertIs(pool._responses, responses)
            self.assertEqual(pool._key, key)
            self.assertFalse(process.killed)
            responses.put(json.dumps(dict(id=2, ok=True, metadata={"next_pair": "worked"})))
            result = pool._request("score", input_path="pair-two.npz", output_path="result-two.npz")
            self.assertTrue(result["ok"])
            self.assertIs(pool.process, process)
            self.assertEqual(pool._key, key)
            self.assertEqual([json.loads(line)["id"] for line in process.stdin.getvalue().splitlines()], [1, 2])
        finally:
            pool._dispose()

    def test_other_worker_failures_still_dispose(self):
        exact = "ValueError: No finite valid SAM correspondence similarities"
        cases = [
            ("unrelated", "score", json.dumps(dict(id=1, ok=False, error="ValueError: invalid input")), False, False),
            ("different_type", "score", json.dumps(dict(id=1, ok=False, error=exact.replace("ValueError", "RuntimeError"))), False, False),
            ("extra_text", "score", json.dumps(dict(id=1, ok=False, error=exact + "; another failure")), False, False),
            ("initialization", "init", json.dumps(dict(id=1, ok=False, error=exact)), False, False),
            ("malformed", "score", "not JSON", False, False),
            ("mismatched_id", "score", json.dumps(dict(id=9, ok=True)), False, False),
            ("eof", "score", None, False, False),
            ("timeout", "score", None, True, False),
            ("dead_worker", "score", json.dumps(dict(id=1, ok=False, error=exact)), False, True),
        ]
        for label, operation, response, timeout, dead in cases:
            with self.subTest(case=label):
                pool, process = self.protocol_pool(response, timeout=timeout, dead=dead)
                with self.assertRaises(RuntimeError):
                    pool._request(operation)
                self.assertIsNone(pool.process)
                self.assertIsNone(pool._key)
                self.assertIsNone(pool._responses)
                self.assertTrue(dead or process.killed)

    def test_cached_missing_failed_and_resume_preserve_inputs_and_successes(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            root, review, snapshot, options, inputs, cached_arrays = fixture(directory)
            calls, failures = [], {(10, 20)}
            factory = lambda **kwargs: FakePool(calls, failures, **kwargs)
            out = directory / "retry"
            before = hashes(root), hashes(review), hashes(snapshot)
            with contextlib.redirect_stdout(io.StringIO()):
                retry.run(root, review, out, snapshot_path=snapshot, pool_factory=factory)
            self.assertEqual(len(calls), 11, "Only the 11 uncached pairs should invoke GeoSCD")
            self.assertTrue(all(call["method"] == "geoscd" and call["options"] == options for call in calls))
            expected_payloads = [(value[1]["reference"], value[1]["source"], value[1]["support"]) for value in inputs.values()]
            for call in calls:
                self.assertTrue(any(all(np.array_equal(call[key], array) for key, array in zip(("reference", "source", "support"), payload))
                                    for payload in expected_payloads))
            units = [json.loads(path.read_text()) for path in sorted((out / "units").glob("*.json"))]
            self.assertEqual(len(units), 12)
            self.assertTrue(all((row["domain"], row["split"], row["method"]) == ("real", "val", "geoscd") for row in units))
            success = [row for row in units if row["status"] == "completed"]
            failed = [row for row in units if row["status"] == "failed"]
            self.assertEqual((len(success), len(failed)), (11, 1))
            self.assertNotIn("artifact_path", failed[0], "Failed correspondence must never produce a fake zero map")
            self.assertEqual(failed[0]["source_input_unit_id"], "rgb-A-previous-2")
            cached = next(row for row in success if row["source_input_unit_id"] == "rgb-A-previous-1")
            self.assertEqual(cached["origin"], "reused_source")
            with np.load(out / cached["artifact_path"], allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["native_prediction"], cached_arrays["native_prediction"])
            for row in units:
                baseline, arrays = inputs[row["source_input_unit_id"]]
                self.assertEqual(row["source_input_artifact_sha256"], baseline["artifact_sha256"])
                self.assertEqual(row["source_input_record_sha256"], sha(root / "units" / f"{baseline['unit_id']}.json"))
                self.assertEqual(row["source_input_sha256"], retry.payload_sha256(arrays))
            success_before = {row["unit_id"]: (row["artifact_sha256"], sha(out / row["artifact_path"])) for row in success}
            attempts_before = hashes(out / "attempts")
            with contextlib.redirect_stdout(io.StringIO()):
                retry.run(root, review, out, snapshot_path=snapshot, pool_factory=factory, resume=True)
            self.assertEqual(len(calls), 11)
            self.assertEqual(hashes(out / "attempts"), attempts_before)
            with contextlib.redirect_stdout(io.StringIO()):
                retry.run(root, review, out, snapshot_path=snapshot, pool_factory=factory, resume=True, retry_failed=True)
            self.assertEqual(len(calls), 12)
            units = [json.loads(path.read_text()) for path in (out / "units").glob("*.json")]
            self.assertEqual(sum(row["status"] == "completed" for row in units), 12)
            for row in units:
                if row["unit_id"] in success_before:
                    self.assertEqual((row["artifact_sha256"], sha(out / row["artifact_path"])), success_before[row["unit_id"]])
            self.assertEqual(len(hashes(out / "attempts")), len(attempts_before) + 1)
            self.assertEqual((hashes(root), hashes(review), hashes(snapshot)), before)

    def test_top_level_device_fallback_and_payload_identity(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            root, review, snapshot, options, inputs, _ = fixture(directory)
            config_path = root / "resolved_config.json"
            config = json.loads(config_path.read_text())
            del config["method_options"]["geoscd"]["device"]
            config_path.write_text(json.dumps(config))
            (root / "run.json").write_text(json.dumps(dict(config=config, status="completed_exploratory")))
            cached_path = root / "units/cached-geoscd.json"
            cached_record = json.loads(cached_path.read_text())
            cached_record["method_metadata"]["configured_options"]["device"] = "cuda:0"
            cached_path.write_text(json.dumps(cached_record))
            snapshot_units = snapshot / "2026-10-17_source_units.json"
            recorded = json.loads(snapshot_units.read_text())
            snapshot_units.write_text(json.dumps([cached_record if row["unit_id"] == "cached-geoscd" else row
                                                  for row in recorded]))
            calls = []
            with contextlib.redirect_stdout(io.StringIO()):
                retry.run(root, review, directory / "retry", modes=("previous",), snapshot_path=snapshot,
                          pool_factory=lambda **kwargs: FakePool(calls, set(), **kwargs))
            self.assertEqual(len(calls), 5)
            self.assertTrue(all(call["options"] == {**options, "device": "cuda:0"} for call in calls))
            arrays = inputs["rgb-A-previous-2"][1]
            original = retry.payload_sha256(arrays)
            changed = {key: value.copy() for key, value in arrays.items()}
            changed["labels"][:] = 0
            changed["scores"][:] = 8
            self.assertEqual(retry.payload_sha256(changed), original, "Only RGB/support belong in inference identity")
            changed["source"][0, 0, 0] += 1
            self.assertNotEqual(retry.payload_sha256(changed), original)
            changed["source"] = arrays["source"].copy()
            changed["support"][0, 0] = False
            self.assertNotEqual(retry.payload_sha256(changed), original)

    def test_explicit_known_runner_upgrade_keeps_saved_results_and_later_resume(self):
        for old_sha in retry.KNOWN_PREVIOUS_RUNNER_SHA256:
            with self.subTest(old_sha=old_sha):
                self._check_known_runner_upgrade(old_sha)

    def _check_known_runner_upgrade(self, old_sha):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            root, review, snapshot, _, _, _ = fixture(directory)
            calls, out = [], directory / "retry"
            factory = lambda **kwargs: FakePool(calls, set(), **kwargs)
            with contextlib.redirect_stdout(io.StringIO()):
                retry.run(root, review, out, snapshot_path=snapshot, pool_factory=factory)
            old_run = json.loads((out / "run.json").read_text())
            old_run["script_sha256"] = old_run["contract"]["script_sha256"] = old_sha
            (out / "run.json").write_text(json.dumps(old_run))
            saved = hashes(out / "maps"), hashes(out / "units"), hashes(out / "attempts")
            source_before = hashes(root), hashes(review), hashes(snapshot)
            initial_calls = len(calls)
            with self.assertRaises(ValueError):
                retry.run(root, review, out, snapshot_path=snapshot, pool_factory=factory, resume=True)
            self.assertEqual(json.loads((out / "run.json").read_text()), old_run)
            with contextlib.redirect_stdout(io.StringIO()):
                retry.run(root, review, out, snapshot_path=snapshot, pool_factory=factory,
                          resume=True, upgrade_runner=True)
            self.assertEqual(len(calls), initial_calls)
            self.assertEqual(json.loads((out / "run.json").read_text())["contract"]["script_sha256"], sha(SCRIPT))
            upgrades = list((out / "runner_upgrades").glob("*.json"))
            self.assertEqual(len(upgrades), 1)
            upgrade = json.loads(upgrades[0].read_text())
            evidence = json.dumps(upgrade)
            self.assertIn(old_sha, evidence)
            self.assertIn(sha(SCRIPT), evidence)
            self.assertIn(json.dumps(old_run["contract"], sort_keys=True),
                          json.dumps(upgrade, sort_keys=True), "Keep the previous frozen contract as upgrade evidence")
            with contextlib.redirect_stdout(io.StringIO()):
                retry.run(root, review, out, snapshot_path=snapshot, pool_factory=factory, resume=True)
            self.assertEqual(len(calls), initial_calls)
            self.assertEqual(len(list((out / "runner_upgrades").glob("*.json"))), 1)
            self.assertEqual((hashes(out / "maps"), hashes(out / "units"), hashes(out / "attempts")), saved)
            self.assertEqual((hashes(root), hashes(review), hashes(snapshot)), source_before)

    def test_runner_upgrade_does_not_allow_other_contract_changes(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            root, review, snapshot, _, _, _ = fixture(directory)
            calls, out = [], directory / "retry"
            factory = lambda **kwargs: FakePool(calls, set(), **kwargs)
            with contextlib.redirect_stdout(io.StringIO()):
                retry.run(root, review, out, snapshot_path=snapshot, pool_factory=factory)
            original = json.loads((out / "run.json").read_text())
            initial_calls = len(calls)
            for change in ("unknown_script", "method_options", "input_records", "repository_code_sha256",
                           "selected_sequences", "snapshot_quick_summary_sha256"):
                with self.subTest(change=change):
                    old = copy.deepcopy(original)
                    old["script_sha256"] = old["contract"]["script_sha256"] = (
                        "0" * 64 if change == "unknown_script" else
                        "c1a30439d1f2daaf90e0fb1d0cf6cecf52c4ded5811c112effe7c1ba8e80ba08")
                    if change == "method_options":
                        old["contract"][change]["seed"] += 1
                    elif change == "input_records":
                        old["contract"][change][0]["artifact_sha256"] = "0" * 64
                    elif change == "repository_code_sha256":
                        old["contract"][change]["facade_change/methods/geoscd.py"] = "0" * 64
                    elif change == "selected_sequences":
                        old["contract"][change] = list(reversed(old["contract"][change]))
                    elif change == "snapshot_quick_summary_sha256":
                        old["contract"][change] = "0" * 64
                    (out / "run.json").write_text(json.dumps(old))
                    before = hashes(out)
                    with self.assertRaises(ValueError):
                        retry.run(root, review, out, snapshot_path=snapshot, pool_factory=factory,
                                  resume=True, upgrade_runner=True)
                    self.assertEqual(hashes(out), before)
                    self.assertEqual(len(calls), initial_calls)


if __name__ == "__main__":
    unittest.main()
