"""Exact scientific-array sharing, selective reads and legacy report contracts."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

storage = importlib.import_module("facade_change.research.2026-11-08_history_scenarios_storage")


class TinyData:
    def __init__(self):
        self.tracks = {"track": {"observations": [dict(observation_id=f"o{i}", year=2000 + i) for i in range(3)]}}
        self.frame_reads = []

    def frame(self, identifier):
        self.frame_reads.append(identifier)
        return np.full((4, 4, 3), int(identifier[-1]), np.uint8), np.ones((4, 4), bool)

    def targets(self, row):
        target = np.zeros((4, 4), np.uint8)
        target[0, 0] = 1
        return dict(target=target, class_targets=target[None], support=np.ones((4, 4), bool),
                    before=np.ones((4, 4), np.uint16), after=np.full((4, 4), 2, np.uint16))

    def annotation_history(self, row):
        return dict(labels=np.ones((row["frame"] + 1, 1, 4, 4), bool),
                    known=np.ones((row["frame"] + 1, 1, 4, 4), bool))


class SharedScientificArrays(unittest.TestCase):
    def setUp(self):
        storage.clear_reader_cache()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = TinyData()
        self.store = storage.SharedMapStore(self.root, dataset=self.data)
        self.row = dict(event_id="event1", building_id="building", split="train", track_id="track", frame=1,
                        bbox_xywh=[0, 0, 4, 4], view_id="view", previous_year=2000, current_year=2001)
        self.spec = dict(task="h0", scenario="main", control="history")
        self.support = np.ones((4, 4), bool)
        self.support[-1, -1] = False
        self.result = dict(score=np.full((4, 4), .4, np.float32), support=self.support,
            extras=dict(forecast=np.full((8, 2, 2), .3, np.float32), scale=np.full((8, 2, 2), .5, np.float32),
                        target_features=np.full((8, 2, 2), .7, np.float32), residual_score=np.full((4, 4), .2, np.float32)))

    def tearDown(self):
        storage.clear_reader_cache()
        self.temporary.cleanup()

    def test_residual_and_fusion_reference_the_same_forecast_without_copies(self):
        parent = self.store.save(self.row, "predictor", self.spec, self.result)
        before = set(self.root.rglob("*.npy"))
        refs = self.store.references(parent)
        residual = self.store.save(self.row, "predictor-residual", self.spec,
            dict(score=self.result["extras"]["residual_score"], support=self.support), refs=refs)
        self.assertEqual(before, set(self.root.rglob("*.npy")))
        fused = self.store.save(self.row, "predictor-score-mlp", self.spec,
            dict(score=np.full((4, 4), .6, np.float32), support=self.support), refs=refs)
        self.assertEqual(len(set(self.root.rglob("*.npy")) - before), 1)
        for record in (residual, fused):
            for key in ("forecast", "scale", "target_features"):
                self.assertEqual(refs[key], self.store.references(record)[key])
        self.assertEqual(self.data.frame_reads, ["o0", "o1"])
        self.assertEqual(len(self.store.records), 3)

    def test_selective_fields_never_open_forecast_and_hash_once(self):
        record = self.store.save(self.row, "predictor", self.spec, self.result)
        storage.clear_reader_cache()
        paths = []
        original_load = np.load
        original_hash = storage.sha256
        hashes = []
        def load(path, **kwargs):
            paths.append(str(path))
            return original_load(path, **kwargs)
        def checksum(path):
            hashes.append(str(path))
            return original_hash(path)
        with mock.patch.object(storage.np, "load", side_effect=load), mock.patch.object(storage, "sha256", side_effect=checksum):
            for _ in range(3):
                with storage.read_map(self.root, record, ("score", "support", "target")) as arrays:
                    self.assertNotIn("forecast", arrays)
                    self.assertIsInstance(arrays["score"], np.memmap)
                    self.assertEqual(arrays["target"][-1, -1], 255)
                    self.assertEqual(arrays["target"][0, 0], 1)
        forecast = self.store.references(record)["forecast"]["path"]
        self.assertFalse(any(path.endswith(forecast) for path in paths))
        self.assertEqual(len(hashes), len(set(hashes)))

    def test_targets_unknown_and_history_virtual_reconstruction_are_exact(self):
        record = self.store.save(self.row, "predictor", self.spec, self.result)
        arrays = self.store.read(record)
        np.testing.assert_array_equal(arrays["unknown"], ~self.support)
        np.testing.assert_array_equal(arrays["class_targets"][0], arrays["target"])
        self.assertEqual(arrays["before"].dtype, np.uint16)
        np.testing.assert_array_equal(arrays["history_rgb"], np.stack([self.data.frame("o0")[0], self.data.frame("o1")[0]]))
        self.assertEqual(self.store.references(record)["history_rgb"]["operation"], "stack")
        row2 = {**self.row, "event_id": "event2", "frame": 2}
        self.store.save(row2, "predictor", self.spec, self.result)
        self.assertEqual(len(list((self.root / "shared/observations").glob("*.json"))), 3)

    def test_missing_dependency_and_tampered_array_are_clear_errors(self):
        record = self.store.save(self.row, "predictor", self.spec, self.result)
        ref = self.store.references(record)["forecast"]
        path = self.root / ref["path"]
        payload = path.read_bytes()
        path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.store.read(record)["forecast"]
        path.unlink()
        with self.assertRaisesRegex(storage.MissingDependencyError, "Missing scientific"):
            self.store.read(record)["forecast"]

    def test_resume_inventory_and_report_reads_do_not_mutate_scientific_files(self):
        record = self.store.save(self.row, "predictor", self.spec, self.result)
        before = {str(path): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        resumed = storage.SharedMapStore(self.root, dataset=self.data)
        self.assertEqual(resumed.get("predictor", "event1"), record)
        self.assertEqual(resumed.verify()["maps"], 1)
        with storage.read_map(self.root, record) as values:
            for name in values:
                np.asarray(values[name])
        after = {str(path): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_legacy_npz_is_read_only_and_selective(self):
        path = self.root / "legacy.npz"
        np.savez_compressed(path, score=self.result["score"], forecast=self.result["extras"]["forecast"])
        record = dict(path="legacy.npz", artifact_sha256=storage.sha256(path))
        before = path.read_bytes()
        with storage.read_map(self.root, record, fields=("score",)) as arrays:
            self.assertEqual(arrays.files, ["score"])
            np.testing.assert_array_equal(arrays["score"], self.result["score"])
            with self.assertRaises(KeyError):
                arrays["forecast"]
        self.assertEqual(before, path.read_bytes())

    def test_semantic_state_prefix_identity_and_dtype_are_preserved(self):
        spec = dict(task="direct", scenario="A", control="learned")
        p0 = np.full((1, 4, 4), .1, np.float32)
        p1 = np.full((1, 4, 4), .2, np.float32)
        p2 = np.full((1, 4, 4), .3, np.float32)
        first = self.store.save(self.row, "semantic", spec, dict(score=p1[0], support=self.support,
            extras=dict(semantic_probability=p1, semantic_before=p0)), metadata=dict(checkpoint="weights-one"))
        second = self.store.save({**self.row, "event_id": "event2", "frame": 2}, "semantic", spec,
            dict(score=p2[0], support=self.support, extras=dict(semantic_probability=p2, semantic_before=p1)),
            metadata=dict(checkpoint="weights-one"))
        self.assertEqual(self.store.references(first)["semantic_probability"], self.store.references(second)["semantic_before"])
        self.assertEqual(len(list((self.root / "shared/semantic_states").glob("*.json"))), 3)
        self.assertEqual(self.store.read(second)["semantic_probability"].dtype, np.float32)
        before = set(self.root.rglob("*.npy"))
        self.store.save(self.row, "semantic+score-mlp", {**spec, "id": "semantic+score-mlp",
            "parent_method": "semantic", "fusion_kind": "score-mlp"}, dict(score=p1[0], support=self.support),
            metadata=dict(base_method="semantic", checkpoint="weights-one"), refs=self.store.references(first))
        self.assertEqual(len(list((self.root / "shared/semantic_states").glob("*.json"))), 3)
        self.assertEqual(before, set(self.root.rglob("*.npy")))

    def test_all_splits_and_all_events_are_saved_independent_of_gallery(self):
        for split in ("train", "val", "test"):
            for event in range(3):
                row = {**self.row, "split": split, "event_id": f"{split}-{event}"}
                self.store.save(row, "predictor", self.spec, self.result)
        self.assertEqual(len(self.store.records), 9)
        self.assertEqual({row["split"] for row in self.store.records}, {"train", "val", "test"})
        self.assertEqual(self.store.verify()["maps"], 9)
        self.assertEqual(self.store.storage_summary()["maps"], 9)

    def test_existing_runstore_ledger_and_unit_markers_are_used(self):
        temporal = importlib.import_module("facade_change.2026-10-14_temporal_experiment")
        out = self.root / "run"
        ledger = temporal.RunStore(out, {"scientific_storage": 2})
        try:
            adapter = storage.SharedMapStore(out, ledger, self.data)
            record = adapter.save(self.row, "predictor", self.spec, self.result)
            events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
            self.assertIn("scientific_array_saved", {event["event_type"] for event in events})
            self.assertEqual(ledger.records(), [record])
            self.assertEqual(record["storage_schema"], 2)
        finally:
            ledger.close()

    def test_write_does_not_accept_tampered_existing_object_after_reopen(self):
        ref = self.store.put(self.result["extras"]["forecast"])
        path = self.root / ref["path"]
        payload = path.read_bytes()
        path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
        storage.clear_reader_cache()
        reopened = storage.SharedMapStore(self.root, dataset=self.data)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            reopened.put(self.result["extras"]["forecast"])

    def test_legacy_unit_cannot_be_overwritten_by_schema_two(self):
        (self.root / "units").mkdir()
        (self.root / "units/legacy.json").write_text(json.dumps(dict(kind="history_scenario_float_maps")))
        reopened = storage.SharedMapStore(self.root, dataset=self.data)
        with self.assertRaisesRegex(ValueError, "read-only"):
            reopened.save(self.row, "predictor", self.spec, self.result)

    def test_scalar_and_original_dtypes_remain_exact(self):
        for value in (np.array(1.25, np.float64), np.zeros((0, 2), np.float32), np.array([1, 2], np.int64)):
            ref = self.store.put(value)
            restored = self.store.read_reference(ref)
            self.assertEqual(restored.dtype, value.dtype)
            self.assertEqual(restored.shape, value.shape)
            np.testing.assert_array_equal(restored, value)


if __name__ == "__main__":
    unittest.main()
