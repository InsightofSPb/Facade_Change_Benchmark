import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.real_eval.semantic_cache import Cache, hash_array
from scripts.real_eval.robustness_common import atomic_json, atomic_npz


class SemanticCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.events = []
        self.cache = Cache(self.temporary.name,
                           event=lambda kind, **values: self.events.append((kind, values)))
        self.spec = {
            "rgb": hash_array(np.zeros((4, 5, 3), dtype=np.uint8)),
            "support": hash_array(np.ones((4, 5), dtype=bool)),
            "model": {"name": "example", "checkpoint_sha256": "a" * 64},
            "vocabulary": ["crack", "wall"], "preprocessing": {"resize": False},
            "output_schema": "semantic_scores_v1",
        }
        self.arrays = {"scores": np.arange(40, dtype=np.float32).reshape(2, 4, 5),
                       "support": np.ones((4, 5), dtype=bool)}

    def test_roundtrip_is_durable_and_put_never_overwrites(self):
        record = self.cache.put(self.spec, self.arrays, {"elapsed_seconds": 2.5, "init_seconds": 1})
        fresh = Cache(self.temporary.name)
        arrays, loaded = fresh.get(self.spec)
        self.assertEqual(record, loaded)
        np.testing.assert_array_equal(arrays["scores"], self.arrays["scores"])
        again = fresh.put(self.spec, {"scores": np.zeros((1,), dtype=np.float32)})
        self.assertEqual(record, again)
        self.assertFalse(Path(record["artifact_path"]).is_absolute())
        self.assertEqual(record["elapsed_seconds"], 2.5)

    def test_key_is_local_and_changes_with_every_computation_input(self):
        key = self.cache.key(self.spec)
        self.assertEqual(key, self.cache.key(dict(reversed(list(self.spec.items())))))
        self.cache.put(self.spec, self.arrays)
        for field, value in (("rgb", "new-rgb"), ("support", "new-support"),
                             ("model", {"name": "msdzip_abs", "checkpoint_sha256": "b" * 64}),
                             ("model", {"name": "example", "checkpoint_sha256": "c" * 64}),
                             ("vocabulary", ["crack", "wall", "window"]),
                             ("preprocessing", {"resize": True}), ("output_schema", "v2")):
            changed = {**self.spec, field: value}
            self.assertNotEqual(key, self.cache.key(changed))
            self.assertIsNone(self.cache.get(changed))
            self.cache.put(changed, self.arrays)
        self.assertIsNotNone(self.cache.get(self.spec))

    def test_corrupt_artifact_is_preserved_and_recomputed(self):
        record = self.cache.put(self.spec, self.arrays)
        artifact = self.cache.root / record["artifact_path"]
        artifact.write_bytes(b"interrupted/corrupt")
        self.assertIsNone(self.cache.get(self.spec))
        preserved = list((self.cache.root / "quarantine").rglob("artifact.npz"))
        self.assertEqual(preserved[0].read_bytes(), b"interrupted/corrupt")
        self.cache.put(self.spec, self.arrays)
        self.assertIsNotNone(self.cache.get(self.spec))
        self.assertIn("cache_invalidated", [kind for kind, values in self.events])

    def test_partial_write_is_not_a_cache_hit(self):
        key = self.cache.key(self.spec)
        atomic_npz(self.cache.root / "objects" / (key + ".npz"), self.arrays)
        self.assertIsNone(self.cache.get(self.spec))
        self.assertEqual(len(list((self.cache.root / "quarantine").rglob("attempt.json"))), 1)

    def test_imported_unmeasured_timing_is_null_not_zero(self):
        record = self.cache.put(self.spec, self.arrays, {"elapsed_seconds": None, "init_seconds": None})
        self.assertIsNone(record["metadata"]["elapsed_seconds"])
        self.assertIsNone(record["metadata"]["init_seconds"])

    def test_commit_failure_keeps_attempt_and_allows_retry(self):
        def interrupted_commit(path, value):
            if Path(path).parent.name == "records":
                raise OSError("disk")
            return atomic_json(path, value)

        with mock.patch("scripts.real_eval.semantic_cache.atomic_json", side_effect=interrupted_commit):
            with self.assertRaisesRegex(OSError, "disk"):
                self.cache.put(self.spec, self.arrays)
        self.assertIsNone(self.cache.get(self.spec))
        self.assertTrue(list((self.cache.root / "quarantine").rglob("artifact.npz")))
        self.assertTrue(list((self.cache.root / "quarantine").rglob("attempt.json")))
        self.cache.put(self.spec, self.arrays)
        self.assertIsNotNone(self.cache.get(self.spec))

    def test_unsafe_dtype_and_forged_npz_are_rejected(self):
        with self.assertRaises(ValueError):
            self.cache.put(self.spec, {"scores": np.array([{"unsafe": True}], dtype=object)})
        record = self.cache.put(self.spec, self.arrays)
        artifact = self.cache.root / record["artifact_path"]
        record["artifact_sha256"] = atomic_npz(artifact, {"scores": np.array([object()], dtype=object)})
        atomic_json(self.cache.root / "records" / (record["cache_key"] + ".json"), record)
        self.assertIsNone(self.cache.get(self.spec))

    def test_interrupted_temporary_file_is_isolated(self):
        key = self.cache.key(self.spec)
        partial = self.cache.root / "objects" / ("." + key + ".npz-abandoned")
        partial.write_bytes(b"partial ZIP")
        self.assertIsNone(self.cache.get(self.spec))
        self.assertFalse(partial.exists())
        retained = list((self.cache.root / "quarantine").rglob(partial.name))
        self.assertEqual(retained[0].read_bytes(), b"partial ZIP")

    def test_shared_cache_commits_once_across_instances(self):
        def writer(value):
            return Cache(self.temporary.name).put(self.spec, {"scores": np.array([value])})
        with ThreadPoolExecutor(max_workers=2) as executor:
            records = list(executor.map(writer, [1, 2]))
        self.assertEqual(records[0], records[1])
        self.assertIsNotNone(self.cache.get(self.spec))

    def test_hash_has_shape_dtype_and_layout_contract(self):
        array = np.arange(12, dtype=np.uint8).reshape(3, 4)
        self.assertEqual(hash_array(array), hash_array(np.asfortranarray(array)))
        self.assertNotEqual(hash_array(array), hash_array(array.reshape(4, 3)))
        self.assertNotEqual(hash_array(array), hash_array(array.astype(np.uint16)))


if __name__ == "__main__":
    unittest.main()
