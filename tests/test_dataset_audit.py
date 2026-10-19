import csv
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from facade_change.data import DEFAULT_METADATA_RULES, build_manifest
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
                         {(1070, 1071), (1070, 1072), (1070, 1073)})
        self.assertEqual(audit["unreviewed_ready_image_count"], 2)
        self.assertEqual(audit["unreviewed_eligible_image_count"], 1)
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


SCRIPT = Path(__file__).resolve().parents[1] / "data/dataset_statistics.py"
SPEC = importlib.util.spec_from_file_location("dataset_statistics", SCRIPT)
STATS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STATS)


class DatasetStatisticsTests(unittest.TestCase):
    @staticmethod
    def coco(names):
        return {"images": [{"id": index, "file_name": name} for index, name in enumerate(names)],
                "annotations": [], "categories": []}

    def rows(self, names):
        return STATS.image_metadata(self.coco(names), DEFAULT_METADATA_RULES)

    def fixture(self):
        return self.coco([
            "deadbeef-Ulitsa_Lenina_46_facade_2010.jpg",
            "Ulitsa_Lenina_46_balcony_2010.jpg",
            "Ulitsa_Lenina_46_side_2015.jpg",
            "Ulitsa_Lenina_46_2015_balcony.jpg",
            "Ulitsa_Lenina_46_balcony_right_2025.jpg",
            "Ulitsa_Lenina_46_facade.jpg",
            "Antonenko8_2022.jpg", "Antonenko8_2026.jpg",
            "Pushkinskaya_5_2026.jpg", "Marata_8.jpg",
            "IMG_0001.jpg", "PXL_20260926_123549548.jpg",
        ])

    def test_pool_views_and_count_distinct_years_separately_from_span(self):
        report = STATS.summarize(STATS.image_metadata(self.fixture(), DEFAULT_METADATA_RULES))
        summary = report["summary"]
        self.assertEqual((summary["images"], summary["identified_facades"], summary["histories"]), (12, 4, 2))
        self.assertEqual((summary["history_images"], summary["other_images"], summary["unidentified_images"]), (8, 4, 2))
        self.assertEqual(summary["images_without_year"], 4)
        self.assertEqual(summary["inferred_group_images"], 3)
        self.assertEqual(summary["mean_history_span_years"], 9.5)
        self.assertEqual(summary["median_history_span_years"], 9.5)
        self.assertEqual(summary["mean_history_observation_years"], 2.5)
        self.assertEqual(summary["mean_history_images"], 4)
        self.assertEqual(report["distributions"]["observation_years"], [
            {"observation_years": 0, "facades": 1, "images": 1, "undated_images": 1},
            {"observation_years": 1, "facades": 1, "images": 1, "undated_images": 0},
            {"observation_years": 2, "facades": 1, "images": 2, "undated_images": 0},
            {"observation_years": 3, "facades": 1, "images": 6, "undated_images": 1},
        ])
        self.assertEqual(report["distributions"]["span_years"], [
            {"span_years": 4, "facades": 1, "images": 2, "undated_images": 0},
            {"span_years": 15, "facades": 1, "images": 6, "undated_images": 1},
        ])
        facade = next(row for row in report["facades"] if row["building_id"] == "Ulitsa_Lenina_46")
        self.assertEqual(facade["images_per_year"], {2010: 2, 2015: 2, 2025: 1})
        self.assertEqual(summary["history_facades_with_undated_images"], 1)

    def test_same_year_variants_are_images_not_extra_years(self):
        rows = self.rows(["Voskova_12_2026_v1.jpg", "Voskova_12_2026_v2.jpg",
                          "Voskova_12_2026_side.jpg"])
        report = STATS.summarize(rows)
        self.assertEqual(report["summary"]["histories"], 0)
        self.assertEqual(report["facades"][0]["image_count"], 3)
        self.assertEqual(report["facades"][0]["observation_years"], 1)
        self.assertEqual(report["facades"][0]["span_years"], 0)
        self.assertIsNone(report["summary"]["mean_history_span_years"])

    def test_filename_exceptions_and_hash_prefixes_keep_reviewed_years(self):
        rows = self.rows(["nested/deadbeef-ryileeva_26_2.png", "ryileeva_26_3.png", "Marata_8.jpg"])
        self.assertEqual([row["year"] for row in rows], [2024, 2023, None])
        self.assertTrue(all(row["metadata_status"] == "reviewed" for row in rows))
        self.assertEqual(STATS.summarize(rows)["summary"]["mean_history_span_years"], 1)

    def test_unknown_addresses_and_numeric_house_suffixes_are_not_merged(self):
        rows = self.rows(["Ulitsa_Lenina_46_2_2024.jpg", "NewStreet_55_2024.jpg",
                          "IMG_2024.jpg", "Ulitsa_Lenina_46_unknown_2024.jpg",
                          "Ulitsa_Lenina_46_side_2010_2025.jpg"])
        self.assertTrue(all(row["building_id"] is None for row in rows))
        self.assertEqual(STATS.summarize(rows)["summary"]["unidentified_images"], 5)

    def test_explicit_unknown_date_in_review_is_not_guessed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            path.write_text("image_id,building_id,view_id,year,reviewed\n0,Antonenko8,,,true\n")
            rows = STATS.image_metadata(self.coco(["Antonenko8_2025.jpg"]), DEFAULT_METADATA_RULES, path)
        self.assertIsNone(rows[0]["year"])
        self.assertEqual(rows[0]["metadata_source"], "override_csv")

    def test_rename_map_uses_canonical_name_and_preserves_coco_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image_rename_map.csv"
            path.write_text("old_name,new_name\nIMG_0001.jpg,ryileeva_26_2.png\n")
            rows = STATS.image_metadata(self.coco(["deadbeef-IMG_0001.jpg"]), DEFAULT_METADATA_RULES,
                                        rename_maps=[path])
        self.assertEqual(rows[0]["file_name"], "deadbeef-IMG_0001.jpg")
        self.assertEqual(rows[0]["resolved_file_name"], "ryileeva_26_2.png")
        self.assertEqual((rows[0]["building_id"], rows[0]["year"]), ("ryileeva_26", 2024))

    def test_invalid_or_duplicate_image_ids_fail_before_counting(self):
        for coco in ({}, {"images": {}}, {"images": [{"file_name": "a.jpg"}]},
                     {"images": [{"id": True, "file_name": "a.jpg"}]},
                     {"images": [{"id": 1, "file_name": "a.jpg"}, {"id": "1", "file_name": "b.jpg"}]}):
            with self.subTest(coco=coco), self.assertRaises(ValueError):
                STATS.image_metadata(coco, DEFAULT_METADATA_RULES)

    def test_repeated_filenames_are_flagged_but_counted(self):
        report = STATS.summarize(self.rows(["IMG_0001.jpg", "IMG_0001.jpg"]))
        self.assertEqual(report["summary"]["images"], 2)
        self.assertEqual(report["summary"]["duplicate_file_names"], {"IMG_0001.jpg": 2})
        self.assertIn("WARNING", STATS.format_report(report))

    def test_empty_dataset_has_no_nan_or_fictitious_history(self):
        report = STATS.summarize(self.rows([]))
        self.assertEqual(report["summary"]["images"], 0)
        self.assertEqual(report["summary"]["histories"], 0)
        self.assertIsNone(report["summary"]["mean_history_span_years"])
        self.assertIn("N/A", STATS.format_report(report))
        json.dumps(report, allow_nan=False)

    def test_metadata_only_run_writes_reconcilable_reports_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coco = root / "coco.json"
            coco.write_text(json.dumps(self.fixture()))
            original = coco.read_bytes()
            with redirect_stdout(io.StringIO()), patch("facade_change.data.load_rgb", side_effect=AssertionError("RGB decoded")):
                report, out = STATS.run(coco, root / "out")
            self.assertEqual(coco.read_bytes(), original)
            self.assertEqual(report["summary"]["images"], 12)
            record = json.loads(next(out.glob("*_run.json")).read_text())
            self.assertEqual(record["status"], "completed")
            self.assertEqual(len(record["artifact_sha256"]), 4)
            with next(out.glob("*_images.csv")).open(encoding="utf-8-sig") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 12)
            with self.assertRaises(FileExistsError):
                STATS.run(coco, out)

    def test_cli_works_from_another_directory_without_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coco = root / "coco.json"
            coco.write_text(json.dumps(self.fixture()))
            result = subprocess.run([sys.executable, str(SCRIPT), str(coco)], cwd=root,
                                    capture_output=True, text=True, check=True)
            self.assertIn("Histories (at least 2 distinct years): 2 (8 images)", result.stdout)
            self.assertEqual(len(list(root.glob("dataset_statistics/*/*_dataset_statistics.json"))), 1)


if __name__ == "__main__":
    unittest.main()
