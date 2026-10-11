import copy
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from facade_change.io import read_json, write_json
from facade_change.preparation import prepare_dataset, split_reviewed_buildings


class FacadeSplitTests(unittest.TestCase):
    @staticmethod
    def rows(medium=20, singles=22, long=10):
        result = []
        for prefix, count, years in (("medium", medium, 5), ("single", singles, 1), ("long", long, 8)):
            for building_index in range(count):
                building = f"{prefix}_{building_index}"
                for year in range(2000, 2000 + years):
                    image_id = len(result) + 1
                    result.append({"image_id": image_id, "file_name": f"image_{image_id}.png",
                                   "sha256": f"{image_id:064x}", "width": 512, "height": 512,
                                   "image_status": "ready", "metadata_status": "reviewed",
                                   "building_id": building, "view_id": building, "year": year})
        return result

    @staticmethod
    def split(rows, **kwargs):
        return split_reviewed_buildings(rows, val_fraction=.08, test_fraction=.12,
                                        balance_unit="buildings", split_policy="facades-v3", **kwargs)

    def test_fifty_two_buildings_target_seven_test_four_val_regardless_of_crops(self):
        rows = self.rows()
        owners, balance = self.split(rows)
        self.assertEqual(Counter(owners.values()), {"train": 41, "val": 4, "test": 7})
        self.assertEqual(balance["target_unit"], "buildings")
        self.assertEqual(balance["series_constraints"]["desired_building_counts"], {"val": 4, "test": 7})
        self.assertEqual(balance["series_constraints"]["status"], "satisfied")
        changed = copy.deepcopy(rows)
        test_group = next(group for group, owner in owners.items() if owner == "test")
        for row in changed:
            if row["building_id"] == test_group:
                row["width"] = 512 + 448 * 19
        changed_owners, changed_balance = self.split(changed)
        self.assertEqual(changed_owners, owners)
        self.assertEqual(changed_balance["series_constraints"]["buildings"][test_group]["crop_count"], 100)
        self.assertGreater(changed_balance["partitions"]["test"]["crop_count"], balance["partitions"]["test"]["crop_count"])
        self.assertTrue(all(owner == "train" for group, owner in owners.items()
                            if group.startswith(("single", "long"))))

    def test_large_population_caps_fresh_test_at_ten_buildings(self):
        owners, balance = self.split(self.rows(medium=22, singles=78, long=0))
        self.assertEqual(Counter(owners.values()), {"train": 82, "val": 8, "test": 10})
        self.assertEqual(balance["series_constraints"]["test_building_range"], [7, 10])

    def test_shortage_reports_actual_available_medium_buildings(self):
        owners, balance = self.split(self.rows(medium=4, singles=5, long=2))
        self.assertEqual(balance["series_constraints"]["status"], "insufficient")
        self.assertLess(Counter(owners.values())["test"], 7)
        self.assertTrue(any("requested at least 7" in issue for issue in balance["series_constraints"]["issues"]))
        self.assertTrue(all(owner == "train" for group, owner in owners.items()
                            if group.startswith(("single", "long"))))

    def test_semantic_undated_train_and_same_policy_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.rows()
            rows[-1].update(year=None, view_id=None)
            source = root / "manifest.json"
            write_json(source, {"images": rows})
            options = dict(split_mode="reviewed", val_fraction=.08, test_fraction=.12,
                           balance_unit="buildings", split_policy="facades-v3")
            first = prepare_dataset(source, root / "first", **options)
            undated = first["images"][-1]
            self.assertEqual(undated["split"], "train")
            self.assertTrue(undated["semantic_eligible"])
            self.assertFalse(undated["temporal_eligible"])
            prepare_dataset(source, root / "second", previous_split=root / "first/split.json", **options)
            old, current = read_json(root / "first/split.json"), read_json(root / "second/split.json")
            self.assertEqual(old["building_assignments"], current["building_assignments"])
            self.assertEqual(old["gold_cohort"], current["gold_cohort"])
            self.assertEqual(current["split_policy"], "facades-v3")
            with self.assertRaisesRegex(ValueError, "Previous split policy must match"):
                prepare_dataset(source, root / "other", previous_split=root / "first/split.json",
                                **{**options, "split_policy": "series-v2", "balance_unit": "crops"})


if __name__ == "__main__":
    unittest.main()
