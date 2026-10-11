import copy
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from facade_change.io import read_json, write_json
from facade_change.preparation import prepare_dataset, split_reviewed_buildings, validate_partitions


class FacadeSplitV4Tests(unittest.TestCase):
    @staticmethod
    def rows(medium=25, singles=20, long=10):
        rows = []
        for prefix, count, years in (("medium", medium, 5), ("single", singles, 1), ("long", long, 8)):
            for index in range(count):
                building = f"{prefix}_{index}"
                for year in range(2000, 2000 + years):
                    image_id = len(rows) + 1
                    rows.append({"image_id": image_id, "file_name": f"image_{image_id}.png",
                                 "sha256": f"{image_id:064x}", "width": 512, "height": 512,
                                 "image_status": "ready", "metadata_status": "reviewed",
                                 "building_id": building, "view_id": building, "year": year})
        return rows

    @staticmethod
    def split(rows, **kwargs):
        return split_reviewed_buildings(rows, val_fraction=.10, test_fraction=.20,
                                        balance_unit="buildings", split_policy="facades-v4", **kwargs)

    def test_fifty_five_buildings_round_to_38_6_11(self):
        owners, balance = self.split(self.rows())
        expected = {"train": 38, "val": 6, "test": 11}
        self.assertEqual(Counter(owners.values()), expected)
        constraints = balance["series_constraints"]
        self.assertEqual(constraints["desired_building_counts"], expected)
        self.assertEqual(constraints["status"], "satisfied")
        self.assertIsNone(constraints["test_building_range"])
        self.assertTrue(all(owner == "train" for building, owner in owners.items()
                            if building.startswith(("single", "long"))))

    def test_hundred_buildings_have_twenty_test_without_v3_cap(self):
        owners, balance = self.split(self.rows(medium=40, singles=50, long=10))
        self.assertEqual(Counter(owners.values()), {"train": 70, "val": 10, "test": 20})
        self.assertEqual(balance["series_constraints"]["status"], "satisfied")

    def test_small_populations_do_not_expand_holdouts_for_history_preferences(self):
        for total, expected in ((10, {"train": 7, "val": 1, "test": 2}),
                                (5, {"train": 4, "test": 1})):
            with self.subTest(total=total):
                owners, balance = self.split(self.rows(medium=total, singles=0, long=0))
                self.assertEqual(Counter(owners.values()), expected)
                self.assertEqual(balance["series_constraints"]["status"], "satisfied")

    def test_image_dimensions_and_repeated_photos_do_not_change_owners(self):
        rows = self.rows()
        owners, initial = self.split(rows)
        changed = copy.deepcopy(rows)
        building = next(group for group, owner in owners.items() if owner == "test")
        for row in changed:
            if row["building_id"] == building:
                row["width"] = 512 + 448 * 19
        extra = [row for row in changed if row["building_id"] == building]
        for row in extra * 10:
            image_id = len(changed) + 1
            changed.append({**row, "image_id": image_id, "file_name": f"image_{image_id}.png",
                            "sha256": f"{image_id:064x}"})
        repeated_owners, repeated = self.split(changed)
        self.assertEqual(repeated_owners, owners)
        self.assertGreater(repeated["partitions"]["test"]["crop_count"],
                           initial["partitions"]["test"]["crop_count"])
        self.assertGreater(repeated["partitions"]["test"]["image_count"],
                           initial["partitions"]["test"]["image_count"])

    def test_all_views_share_one_building_owner_and_cross_split_is_rejected(self):
        rows = self.rows()
        building = "medium_0"
        for year in range(2000, 2004):
            image_id = len(rows) + 1
            rows.append({**rows[0], "image_id": image_id, "file_name": f"image_{image_id}.png",
                         "sha256": f"{image_id:064x}", "view_id": f"{building}_rear", "year": year})
        owners, _ = self.split(rows)
        for row in rows:
            row["split"] = owners[row["building_id"]]
        validate_partitions(rows)
        self.assertEqual({row["split"] for row in rows if row["building_id"] == building}, {owners[building]})
        rows[-1]["split"] = "test" if owners[building] != "test" else "train"
        with self.assertRaisesRegex(ValueError, "Split leakage: building_id="):
            validate_partitions(rows)

    def test_series_shortage_is_explicit_without_promoting_short_or_long(self):
        owners, balance = self.split(self.rows(medium=4, singles=41, long=10))
        constraints = balance["series_constraints"]
        self.assertEqual(constraints["status"], "insufficient")
        self.assertTrue(any("desired 11" in issue for issue in constraints["issues"]))
        self.assertTrue(all(owner == "train" for building, owner in owners.items()
                            if building.startswith(("single", "long"))))
        self.assertGreaterEqual(constraints["medium_buildings_by_split"]["train"], 1)

    def test_count_targets_precede_preferred_five_year_coverage(self):
        rows = self.rows(medium=25, singles=20, long=10)
        rows = [row for row in rows if not row["building_id"].startswith("medium") or row["year"] <= 2002]
        owners, balance = self.split(rows)
        self.assertEqual(Counter(owners.values()), {"train": 38, "val": 6, "test": 11})
        self.assertTrue(any("5–7 reviewed years" in issue for issue in balance["series_constraints"]["issues"]))

    def test_same_policy_extension_keeps_every_previous_owner_and_gold_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.rows()
            source = root / "manifest.json"
            write_json(source, {"images": rows})
            options = dict(split_mode="reviewed", val_fraction=.10, test_fraction=.20,
                           balance_unit="buildings", split_policy="facades-v4")
            prepare_dataset(source, root / "first", **options)
            first = read_json(root / "first/split.json")
            for year in range(2000, 2005):
                image_id = len(rows) + 1
                rows.append({**rows[0], "image_id": image_id, "file_name": f"image_{image_id}.png",
                             "sha256": f"{image_id:064x}", "building_id": "new_medium",
                             "view_id": "new_medium", "year": year})
            write_json(source, {"images": rows})
            prepared = prepare_dataset(source, root / "second", previous_split=root / "first/split.json", **options)
            second = read_json(root / "second/split.json")
            for building, owner in first["building_assignments"].items():
                self.assertEqual(second["building_assignments"][building], owner)
            self.assertEqual(first["gold_cohort"], second["gold_cohort"])
            validate_partitions(prepared["images"], allow_missing_views=True)
            self.assertEqual(second["split_policy"], "facades-v4")

    def test_v3_keeps_previous_seven_to_ten_rule(self):
        owners, balance = split_reviewed_buildings(self.rows(), val_fraction=.08, test_fraction=.12,
                                                   balance_unit="buildings", split_policy="facades-v3")
        self.assertEqual(Counter(owners.values()), {"train": 44, "val": 4, "test": 7})
        self.assertEqual(balance["series_constraints"]["test_building_range"], [7, 10])

    def test_v4_rejects_image_balancing(self):
        with self.assertRaisesRegex(ValueError, "facades-v4 requires balance_unit=buildings"):
            split_reviewed_buildings(self.rows(), split_policy="facades-v4")


if __name__ == "__main__":
    unittest.main()
