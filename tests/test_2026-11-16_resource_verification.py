"""File integrity and actual read counts, independent of hardware/timing thresholds."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from facade_change.agentic import models
from facade_change.io import sha256
from tests.test_2026_11_08_agentic_models import FakeWorker


def counters():
    return {"seconds": 0., "bytes_hashed": 0, "files_hashed": 0, "stat_cache_hits": 0}


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.resource = self.root / "weights"
        self.resource.write_bytes(b"model-resource-A")
        self.row = {"path": str(self.resource), "sha256": sha256(self.resource)}
        self.verifier, self.stats = models._ResourceVerifier(), counters()

    def valid(self, verifier=None):
        return (verifier or self.verifier).valid([self.row], None, self.stats)

    def test_unchanged_files_are_read_once_and_new_verifier_or_pid_rechecks(self):
        with patch.object(models, "_timed_sha256", wraps=models._timed_sha256) as reads:
            self.assertTrue(self.valid())
            self.assertTrue(self.valid())
            self.assertEqual(reads.call_count, 1)
            self.assertEqual(self.stats["bytes_hashed"], self.resource.stat().st_size)
            self.assertEqual(self.stats["stat_cache_hits"], 1)
            self.assertTrue(self.valid(models._ResourceVerifier()))
            self.assertEqual(reads.call_count, 2)
            with patch.object(models.os, "getpid", return_value=os.getpid() + 1):
                self.assertTrue(self.valid())
            self.assertEqual(reads.call_count, 3)

    def test_same_size_mutation_even_with_restored_mtime_rehashes_and_rejects(self):
        self.assertTrue(self.valid())
        before = self.resource.stat()
        self.resource.write_bytes(b"model-resource-B")
        os.utime(self.resource, ns=(before.st_atime_ns, before.st_mtime_ns))
        with patch.object(models, "_timed_sha256", wraps=models._timed_sha256) as reads:
            self.assertFalse(self.valid())
        self.assertEqual(reads.call_count, 1)
        self.assertEqual(self.stats["files_hashed"], 2)

    def test_replacement_and_symlink_retargeting_cannot_reuse_approval(self):
        link = self.root / "link"
        link.symlink_to(self.resource)
        self.row["path"] = str(link)
        self.assertTrue(self.valid())
        other = self.root / "other"
        other.write_bytes(self.resource.read_bytes())
        # Even identical content is fully reverified on a different resolved path.
        link.unlink()
        link.symlink_to(other)
        with patch.object(models, "_timed_sha256", wraps=models._timed_sha256) as reads:
            self.assertTrue(self.valid())
        self.assertEqual(reads.call_count, 1)
        replacement = self.root / "replacement"
        replacement.write_bytes(b"model-resource-C")
        replacement.replace(other)
        self.assertFalse(self.valid())
        self.assertEqual(self.stats["files_hashed"], 3)

    def test_change_during_verification_fails_and_does_not_seed_cache(self):
        original_hash = models._timed_sha256
        def mutate(path, deadline=None, stats=None):
            checksum = original_hash(path, deadline, stats)
            self.resource.write_bytes(b"model-resource-B")
            return checksum
        with patch.object(models, "_timed_sha256", side_effect=mutate):
            with self.assertRaisesRegex(models.WorkerError, "changed during verification"):
                self.valid()
        self.assertFalse(self.valid())
        self.assertEqual(self.stats["files_hashed"], 2)
        self.assertEqual(self.stats["stat_cache_hits"], 0)

    def test_deadline_and_missing_resource_do_not_become_cached_success(self):
        self.assertTrue(self.valid())
        with self.assertRaisesRegex(models.WorkerError, "Deadline"):
            self.verifier.valid([self.row], 0, self.stats)
        self.resource.unlink()
        self.assertFalse(self.valid())

    def test_pool_warm_calls_avoid_reads_but_response_cache_still_full_checks(self):
        resource = self.row
        class ResourceWorker(FakeWorker):
            def request(self, payload, deadline):
                result = super().request(payload, deadline)
                result["provenance"] = {"resources": [copy.deepcopy(resource)]}
                return result
        FakeWorker.made, FakeWorker.replies = [], []
        with patch.object(models, "_Worker", ResourceWorker):
            with models.ModelPool({"qwen": {}}, self.root / "logs", cache_dir=self.root / "cache") as pool:
                first = pool.call("qwen", {"op": "generate", "messages": []})
                second = pool.call("qwen", {"op": "generate", "messages": [], "tag": 2})
                self.assertEqual(first["status"], "ok")
                self.assertEqual(first["usage"]["resource_validation"]["bytes_hashed"], self.resource.stat().st_size)
                self.assertEqual(second["usage"]["resource_validation"]["bytes_hashed"], 0)
                self.assertEqual(second["usage"]["resource_validation"]["stat_cache_hits"], 2)
                self.assertFalse(second["cache_hit"])
                cached = pool.call("qwen", {"op": "generate", "messages": []})
                self.assertTrue(cached["cache_hit"])
                self.assertEqual(cached["usage"]["resource_validation"]["files_hashed"], 1)
                self.assertIn("resource_validation", pool.events[-1])
                self.resource.write_bytes(b"model-resource-B")
                changed = pool.call("qwen", {"op": "generate", "messages": []})
                self.assertFalse(changed["cache_hit"])
                self.assertEqual(changed["error"]["code"], "resource_changed")
                self.assertTrue(all(w.closed for w in FakeWorker.made))
            with models.ModelPool({"qwen": {}}, self.root / "next-run") as pool:
                resource["sha256"] = sha256(self.resource)
                fresh = pool.call("qwen", {"op": "generate", "messages": []})
                self.assertEqual(fresh["usage"]["resource_validation"]["files_hashed"], 1)

    def test_configured_checkpoint_key_uses_same_checked_digest(self):
        with models.ModelPool({"qwen": {"checkpoint_path": str(self.resource)}}, self.root / "logs") as pool:
            pool._resource_checks = counters()
            original = pool._key("qwen", {"op": "generate"})
            self.assertEqual(pool._key("qwen", {"op": "generate"}), original)
            self.assertEqual(pool._resource_checks["files_hashed"], 1)
            self.resource.write_bytes(b"model-resource-B")
            self.assertNotEqual(pool._key("qwen", {"op": "generate"}), original)
            self.assertEqual(pool._resource_checks["files_hashed"], 2)


if __name__ == "__main__":
    unittest.main()
