import importlib.util
import tempfile
import unittest
from pathlib import Path

from facade_change.data import DEFAULT_METADATA_RULES, read_filename_rules
from facade_change.io import read_json, write_json


spec = importlib.util.spec_from_file_location("building_review", Path(__file__).resolve().parents[1]
                                            / "scripts/2026-10-03_review_buildings.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class BuildingReviewTests(unittest.TestCase):
    def test_confirmed_aliases_share_building_but_confirmed_address_groups_stay_separate(self):
        views = ["Kaznacheiskaya", "Kaznacheiskaya_2",
                 "9-ya_linia_VO_16-18", "9-ya_linia_VO_18"]
        rows = [{"image_id": 500 + i, "file_name": f"{view}_2025.png", "image_status": "ready",
                 "year": 2025, "view_id": view, "metadata_status": "inferred", "building_id": None}
                for i, view in enumerate(views)]
        rules = read_filename_rules(DEFAULT_METADATA_RULES, rows)
        for row in rows:
            row.update(rules[str(row["image_id"])])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / "manifest.json"
            write_json(manifest, {"images": rows})
            groups, summary = review.review_buildings(manifest, root / "review")
        by_building = {group["building_id"]: group for group in groups}
        self.assertEqual(set(by_building), {"Kaznacheiskaya_2", "9-ya_linia_VO_16-18", "9-ya_linia_VO_18"})
        self.assertEqual(by_building["Kaznacheiskaya_2"]["view_ids"], "Kaznacheiskaya;Kaznacheiskaya_2")
        self.assertEqual(by_building["Kaznacheiskaya_2"]["image_count"], 2)
        for view in views[2:]:
            self.assertEqual(by_building[view]["view_ids"], view)
            self.assertEqual(by_building[view]["image_count"], 1)
        self.assertTrue(all(group["reviewed"] == "true" for group in groups))
        self.assertEqual(summary["eligible_images"], 4)

    def test_detail_groups_preserve_address_numbers_and_house_letters(self):
        for view, expected in [("Bolshoi_10b_balcony_bottom", "Bolshoi_10b"),
                               ("Kamenoostrovskii_13_2_ornament", "Kamenoostrovskii_13_2"),
                               ("ProfPopova4v_balconyv2", "ProfPopova4v"),
                               ("marata_54_34", "marata_54_34"),
                               ("9-ya_linia_VO_16-18", "9-ya_linia_VO_16-18"),
                               ("Voskova_full_4", "Voskova_4")]:
            self.assertEqual(review.proposed_building(view), expected)
        self.assertIsNone(review.proposed_building("Kaznacheiskaya"))

    def test_proposals_do_not_confirm_metadata_or_hide_unknown_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / "manifest.json"
            rows = [{"image_id": i, "file_name": f"file{i}.png", "image_status": "ready", "year": 2025,
                     "view_id": view, "metadata_status": "inferred", "building_id": None}
                    for i, view in enumerate(["house_10b_balcony", "house_10b_right", "Kaznacheiskaya", None])]
            write_json(manifest, {"images": rows})
            before = manifest.read_bytes()
            groups, summary = review.review_buildings(manifest, root / "review")
            self.assertEqual(summary["eligible_images"], 3)
            self.assertEqual(summary["unresolved_images"], 1)
            self.assertEqual(len(groups), 2)
            self.assertTrue(all(r["reviewed"] == "false" for r in groups))
            self.assertEqual(manifest.read_bytes(), before)
            self.assertEqual(read_json(root / "review/run.json")["status"], "completed")


if __name__ == "__main__":
    unittest.main()
