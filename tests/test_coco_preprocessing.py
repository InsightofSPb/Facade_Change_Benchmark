import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from facade_change.data import preprocess_coco
from facade_change.io import read_json, sha256, write_json


class CocoPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "coco.json"
        self.coco = {
            "info": {"description": "preserve custom metadata"}, "custom": [1, {"value": "unchanged"}],
            "images": [{"id": 4, "file_name": "wall.png", "width": 10, "height": 8},
                       {"id": 7, "file_name": "empty.png", "width": 10, "height": 8}],
            "categories": [{"id": 3, "name": "CRACK", "supercategory": "damage"}],
            "annotations": [
                {"id": 21, "image_id": 4, "category_id": 3, "area": 2.5,
                 "bbox": [1, 2, 3, 4], "segmentation": [[1, 2, 3, 4, 5, 6]], "iscrowd": 0},
                {"id": 30, "image_id": 7, "category_id": 3, "area": 0, "bbox": [0, 0, 0, 0]},
                {"id": 31, "image_id": 7, "category_id": 3, "area": -2, "bbox": [0, 0, 1, 1]},
                {"id": 45, "image_id": 4, "category_id": 3, "bbox": [2, 2, 1, 1]},
            ],
        }
        write_json(self.source, self.coco)

    def tearDown(self):
        self.temp.cleanup()

    def test_only_explicit_nonpositive_areas_removed_without_changing_source_or_ids(self):
        before = self.source.read_bytes()
        result = preprocess_coco(self.source, self.root / "cleaned")
        self.assertEqual(self.source.read_bytes(), before)
        cleaned = read_json(result["coco_path"])
        expected = {**self.coco, "annotations": [self.coco["annotations"][0], self.coco["annotations"][3]]}
        self.assertEqual(cleaned, expected)
        self.assertEqual(result["summary"]["removed_annotation_count"], 2)
        self.assertEqual(result["summary"]["original_annotation_count"], 4)
        self.assertEqual(result["summary"]["retained_annotation_count"], 2)
        self.assertEqual(result["summary"]["images_without_annotations_count"], 1)
        report = read_json(result["report_path"])
        self.assertEqual([row["annotation_id"] for row in report["removed_annotations"]], [30, 31])
        self.assertEqual(report["removed_annotations"][0]["file_name"], "empty.png")
        self.assertEqual(report["removed_annotations"][0]["category_name"], "CRACK")
        self.assertEqual(report["removed_annotations"][1]["area"], -2)
        self.assertEqual(report["source"]["sha256"], sha256(self.source))
        self.assertEqual(report["output"]["sha256"], sha256(result["coco_path"]))
        record = read_json(self.root / "cleaned/run.json")
        self.assertEqual(record["status"], "completed")
        for filename, digest in record["artifact_sha256"].items():
            self.assertEqual(sha256(self.root / "cleaned" / filename), digest)
        with self.assertRaises(FileExistsError):
            preprocess_coco(self.source, self.root / "cleaned")

    def test_no_removals_keeps_every_annotation(self):
        self.coco["annotations"] = [row for row in self.coco["annotations"] if row.get("area", 1) > 0]
        write_json(self.source, self.coco)
        result = preprocess_coco(self.source, self.root / "already_clean")
        self.assertEqual(read_json(result["coco_path"]), self.coco)
        self.assertEqual(result["summary"]["removed_annotation_count"], 0)
        self.assertEqual(read_json(result["report_path"])["removed_annotations"], [])

    def test_source_changed_during_preprocessing_records_failure(self):
        def write_and_modify(path, value):
            write_json(path, value)
            if Path(path).name == "annotations.json":
                self.source.write_text(self.source.read_text(encoding="utf-8") + "\n", encoding="utf-8")

        with patch("facade_change.data.write_json", side_effect=write_and_modify):
            with self.assertRaisesRegex(ValueError, "Source COCO changed"):
                preprocess_coco(self.source, self.root / "changed")
        self.assertEqual(read_json(self.root / "changed/run.json")["status"], "failed")


if __name__ == "__main__":
    unittest.main()
