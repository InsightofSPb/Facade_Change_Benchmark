"""Real quick-report projection and checksum-bound restoration, without models."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.real_eval.source_unit_metadata import _verified_marker, source_rows


class SourceUnitMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.marker = self.root / "units" / "sample-unit.json"
        self.marker.parent.mkdir()
        self.original = {
            "unit_id": "sample-unit", "method": "msdzip_abs", "variant": "current_train",
            "status": "completed", "domain": "real", "split": "val", "mode": "previous",
            "sequence_id": "real-history", "frame": 1, "building_id": "facade",
            "artifact_path": "maps/sample-unit.npz", "artifact_sha256": "a" * 64,
            "method_metadata": {"method": "msdzip_abs", "checkpoint_path": "/weights/abs.pt",
                                "checkpoint_sha256": "b" * 64},
        }
        payload = json.dumps(self.original, ensure_ascii=False).encode("utf-8")
        self.marker.write_bytes(payload)
        # This is exactly the quick report's snapshot projection and hash key.
        self.row = {key: value for key, value in self.original.items() if key != "method_metadata"}
        self.manifest = {
            "source_run": str(self.root), "source_records": [self.row],
            "source_summary": {"source_json_sha256": {
                "units/sample-unit.json": hashlib.sha256(payload).hexdigest()}},
        }
        _verified_marker.cache_clear()

    def test_actual_quick_snapshot_restores_metadata_without_mutating_manifest(self):
        before = copy.deepcopy(self.manifest)
        self.assertEqual(source_rows(self.manifest), [self.original])
        self.assertEqual(self.manifest, before)

    def test_dictionary_rows_and_existing_metadata_are_preserved(self):
        self.manifest["source_records"] = {"sample-unit": self.original}
        self.marker.unlink()
        self.assertIs(source_rows(self.manifest)[0], self.original)

    def test_legacy_without_marker_hash_does_not_read_or_invent_metadata(self):
        self.manifest["source_summary"] = {}
        self.marker.unlink()
        self.assertEqual(source_rows(self.manifest), [self.row])
        self.assertNotIn("method_metadata", source_rows(self.manifest)[0])

    def test_marker_missing_from_hash_map_keeps_unverified_row(self):
        self.manifest["source_summary"]["source_json_sha256"] = {"run.json": "c" * 64}
        self.marker.unlink()
        self.assertEqual(source_rows(self.manifest), [self.row])

    def test_changed_marker_fails_closed_even_after_cached_read(self):
        source_rows(self.manifest)
        self.marker.write_bytes(self.marker.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Изменился исходный marker"):
            source_rows(self.manifest)

    def test_snapshot_projection_mismatch_fails_closed(self):
        self.row["frame"] = 2
        with self.assertRaisesRegex(ValueError, "расходится с записью снимка"):
            source_rows(self.manifest)

    def test_repeated_preflights_share_verified_reads(self):
        original_reader = Path.read_bytes
        with patch.object(Path, "read_bytes", autospec=True, side_effect=original_reader) as reader:
            source_rows(self.manifest)
            source_rows(self.manifest)
        self.assertEqual(reader.call_count, 1)

    def test_malformed_rows_and_hash_mapping_are_rejected(self):
        for rows in ("not-rows", [42], {"unit": None}):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "source_records"):
                source_rows({"source_records": rows})
        self.manifest["source_summary"]["source_json_sha256"] = []
        with self.assertRaisesRegex(ValueError, "source_json_sha256"):
            source_rows(self.manifest)

    def test_path_escape_is_rejected_before_marker_read(self):
        self.row["unit_id"] = "../../outside"
        self.manifest["source_summary"]["source_json_sha256"]["units/../../outside.json"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "unit_id"):
            source_rows(self.manifest)


if __name__ == "__main__":
    unittest.main()
