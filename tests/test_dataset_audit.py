import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from facade_change.data import build_manifest
from facade_change.io import read_json, write_json


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required by dataset runner")
class DatasetAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        names = ["place_2024.png", "place_2025.png", "place_2026_v1.png", "place_2026_v2.png",
                 "unknown.png", "single_2020.png", "directory_only_2026.png"]
        for i, name in enumerate(names):
            Image.fromarray(np.full((16, 20, 3), 20 + i, np.uint8)).save(self.images / name)
        self.coco = self.root / "coco.json"
        write_json(self.coco, {"images": [{"id": 70 + i, "file_name": name, "width": 20, "height": 16}
                   for i, name in enumerate(names[:-1])], "annotations": [], "categories": []})
        self.config = self.root / "config.json"
        write_json(self.config, {"coco_json": str(self.coco), "image_roots": [str(self.images)]})
        build_manifest(self.config, self.root / "old")
        spec = importlib.util.spec_from_file_location("audit_script", Path(__file__).resolve().parents[1]
                                                   / "scripts/2026-10-03_audit_dataset.py")
        self.audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.audit)
        self.fixes = self.root / "fixes.json"
        write_json(self.fixes, [{"file_name": name, "view_id": "place", "building_id": "place",
                                "year": year, "notes": "Human confirmed"}
                               for name, year in zip(names[:4], [2024, 2025, 2026, 2026])])

    def tearDown(self):
        self.temp.cleanup()

    def test_audit_uses_current_ids_preserves_two_same_year_observations_and_review_scope(self):
        coco = read_json(self.coco)
        for row in coco["images"]:
            row["id"] += 1000
        coco["annotations"] = [{"id": 1, "image_id": 1070, "category_id": 5, "area": 10}]
        coco["categories"] = [{"id": 5, "name": "new_annotation"}]
        write_json(self.coco, coco)
        out = self.root / "audit"
        with patch("facade_change.data.load_rgb", side_effect=AssertionError("Unchanged RGB decoded")):
            audit, status = self.audit.run_audit(self.config, self.root / "old/manifest.json", out, self.fixes)
        self.assertEqual(status, "completed")
        self.assertEqual(audit["inventory"]["image_count"], 6)
        self.assertEqual(audit["inventory"]["reused_image_count"], 6)
        self.assertEqual(audit["inventory"]["annotation_count"], 1)
        self.assertEqual(audit["preparation"]["pair_count"], 3)
        prepared = read_json(out / "dataset/prepared/manifest.json")
        self.assertEqual({(p["reference_id"], p["source_id"]) for p in prepared["pairs"]},
                         {(1070, 1071), (1071, 1072), (1071, 1073)})
        self.assertEqual(audit["unreviewed_ready_image_count"], 2)
        self.assertEqual(audit["pairing_status"], {"pair_candidate": 4, "metadata_or_image_unresolved": 1,
                                                "single_year_view": 1})
        with (out / "metadata_review_with_names.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["image_id"], "1070")
        self.assertEqual(rows[0]["annotation_count"], "1")
        self.assertEqual(rows[-2]["reviewed"], "false")
        self.assertEqual(read_json(out / "run.json")["status"], "completed")
        with self.assertRaises(FileExistsError):
            self.audit.run_audit(self.config, self.root / "old/manifest.json", out, self.fixes)

    def test_missing_fix_is_an_explicit_failure(self):
        fixes = read_json(self.fixes)
        fixes[0]["file_name"] = "absent.png"
        write_json(self.fixes, fixes)
        with self.assertRaisesRegex(ValueError, "Missing, ambiguous or duplicate"):
            self.audit.run_audit(self.config, self.root / "old/manifest.json", self.root / "failed", self.fixes)
        self.assertEqual(read_json(self.root / "failed/run.json")["status"], "failed")

    def test_zero_area_filter_uses_cleaned_counts_without_decoding_or_removing_images(self):
        coco = read_json(self.coco)
        coco["annotations"] = [{"id": i, "image_id": 70, "category_id": 5, "area": area}
                               for i, area in enumerate([10, 0, -1], 1)]
        coco["categories"] = [{"id": 5, "name": "synthetic"}]
        write_json(self.coco, coco)
        before = self.coco.read_bytes()
        with patch("facade_change.data.load_rgb", side_effect=AssertionError("Unchanged RGB decoded")):
            audit, status = self.audit.run_audit(self.config, self.root / "old/manifest.json",
                                                 self.root / "clean", self.fixes)
        self.assertEqual(status, "completed")
        self.assertEqual(self.coco.read_bytes(), before)
        self.assertEqual(audit["inventory"]["image_count"], 6)
        self.assertEqual(audit["inventory"]["annotation_count"], 1)
        self.assertEqual(audit["inventory"]["decoded_image_count"], 0)
        self.assertEqual(audit["annotation_preprocessing"]["summary"]["removed_annotation_count"], 2)
        self.assertEqual(audit["annotation_issues"]["annotation_nonpositive_area_ids"], [])


if __name__ == "__main__":
    unittest.main()
