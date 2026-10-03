import importlib.util
import tempfile
import unittest
from pathlib import Path

from facade_change.io import read_json, write_json


spec = importlib.util.spec_from_file_location("building_review", Path(__file__).resolve().parents[1]
                                            / "scripts/2026-10-03_review_buildings.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class BuildingReviewTests(unittest.TestCase):
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
