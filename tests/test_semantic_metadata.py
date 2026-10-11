import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from facade_change.data import build_manifest, filename_metadata, read_filename_rules, read_overrides
from facade_change.io import read_json, write_json


class SemanticMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.review = self.root / "review.csv"

    def tearDown(self):
        self.temp.cleanup()

    def overrides(self, **fields):
        row = {"image_id": "1", "building_id": "house", "reviewed": "true", **fields}
        with self.review.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        return read_overrides(self.review, {"1"})

    def inventory(self, name="house_2024.png"):
        images = self.root / "images"
        images.mkdir()
        Image.new("RGB", (10, 8), (10, 20, 30)).save(images / name)
        coco = self.root / "coco.json"
        write_json(coco, {"images": [{"id": 1, "file_name": name, "width": 10, "height": 8}],
                          "annotations": [], "categories": []})
        config = self.root / "paths.json"
        write_json(config, {"coco_json": str(coco), "image_roots": [str(images)], "metadata_rules": None})
        return config, coco, images / name

    def test_building_only_confirmation_clears_temporal_guesses(self):
        self.assertEqual(self.overrides()["1"], {
            "building_id": "house", "view_id": None, "year": None,
            "metadata_status": "reviewed", "metadata_source": "override_csv", "metadata_notes": ""})
        config, _, _ = self.inventory()
        result = build_manifest(config, self.root / "inventory", overrides=self.review)
        row = result["images"][0]
        self.assertEqual((row["building_id"], row["view_id"], row["year"]), ("house", None, None))
        self.assertEqual(row["metadata_status"], "reviewed")

    def test_confirmed_view_without_year_and_year_without_view_are_supported(self):
        row = self.overrides(view_id=" house_front ", year=" ")["1"]
        self.assertEqual((row["view_id"], row["year"]), ("house_front", None))
        row = self.overrides(view_id="", year="2024")["1"]
        self.assertEqual((row["view_id"], row["year"]), (None, 2024))

    def test_building_is_required_even_when_temporal_fields_exist(self):
        with self.assertRaisesRegex(ValueError, "needs building_id"):
            self.overrides(building_id=" ", view_id="house_front", year="2024")

    def test_nonempty_invalid_year_never_becomes_unknown(self):
        for year in ("not known", "2024.0", "1799", "2101", "true"):
            with self.subTest(year=year), self.assertRaisesRegex(ValueError, "Invalid year"):
                self.overrides(year=year)

    def test_unreviewed_candidate_does_not_confirm_building(self):
        self.assertEqual(self.overrides(reviewed="false"), {})
        self.assertIsNone(filename_metadata("IMG_1234.jpg")["building_id"])

    def test_variant_year_is_inferred_without_guessing_shared_view(self):
        rows = []
        for index in (1, 2):
            name = f"deadbeef-house_2026_v{index}.jpg"
            candidate = filename_metadata(name)
            self.assertEqual((candidate["year"], candidate["view_id"], candidate["building_id"]),
                             (2026, None, None))
            self.assertEqual(candidate["metadata_status"], "inferred")
            rows.append({"image_id": index, "file_name": name})
        rules = self.root / "rules.json"
        write_json(rules, [{"view_id": "house", "building_id": "confirmed_house"}])
        self.assertEqual(read_filename_rules(rules, rows), {})

    def test_verified_cache_inherits_missing_temporal_fields_without_decoding(self):
        config, coco_path, _ = self.inventory()
        self.overrides()
        previous = self.root / "first/manifest.json"
        build_manifest(config, previous.parent, overrides=self.review)
        # Old building-only exports may omit optional keys instead of using null.
        cached = read_json(previous)
        del cached["images"][0]["view_id"]
        del cached["images"][0]["year"]
        write_json(previous, cached)
        coco = read_json(coco_path)
        coco["images"][0]["id"] = 99
        write_json(coco_path, coco)
        with patch("facade_change.data.load_rgb", side_effect=AssertionError("verified RGB must reuse cache")):
            result = build_manifest(config, self.root / "refreshed", previous_manifest=previous)
        row = result["images"][0]
        self.assertEqual((row["image_id"], row["building_id"], row["view_id"], row["year"]),
                         (99, "house", None, None))
        self.assertEqual(row["metadata_source"], "previous_manifest")
        self.assertEqual(result["summary"]["inherited_reviewed_metadata_count"], 1)

    def test_changed_pixels_do_not_inherit_building_only_confirmation(self):
        config, _, image_path = self.inventory()
        self.overrides()
        previous = self.root / "first/manifest.json"
        build_manifest(config, previous.parent, overrides=self.review)
        Image.new("RGB", (10, 8), (40, 50, 60)).save(image_path)
        result = build_manifest(config, self.root / "changed", previous_manifest=previous)
        row = result["images"][0]
        self.assertIsNone(row["building_id"])
        self.assertEqual(row["metadata_status"], "inferred")
        self.assertEqual(result["summary"]["inherited_reviewed_metadata_count"], 0)


if __name__ == "__main__":
    unittest.main()
