import copy
import tempfile
import unittest
from pathlib import Path

from facade_change.io import read_json, write_json
from facade_change.preparation import prepare_dataset, split_reviewed_buildings, validate_partitions


class SeriesSplitTests(unittest.TestCase):
    def setUp(self):
        self.next_id = 1
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def series(self, building, years, crops=1, view=None):
        rows = []
        for year in years:
            image_id = self.next_id
            self.next_id += 1
            rows.append({"image_id": image_id, "file_name": f"image_{image_id}.png",
                         "sha256": f"{image_id:064x}", "width": 512 + 448 * (crops - 1), "height": 512,
                         "image_status": "ready", "metadata_status": "reviewed",
                         "building_id": building, "view_id": view or building, "year": year})
        return rows

    def split(self, rows, **kwargs):
        return split_reviewed_buildings(rows, val_fraction=.08, test_fraction=.12,
                                        balance_unit="crops", split_policy="series-v2", **kwargs)

    def prepare(self, rows, name="prepared", **kwargs):
        manifest = self.root / f"{name}-input.json"
        write_json(manifest, {"images": rows})
        return prepare_dataset(manifest, self.root / name, split_mode="reviewed", balance_unit="crops",
                               split_policy="series-v2", val_fraction=.08, test_fraction=.12, **kwargs)

    def balanced_rows(self):
        rows = self.series("long_train", range(2000, 2008), crops=10)
        rows += self.series("single_train", [2026], crops=17)
        rows += self.series("short_medium", range(2000, 2003))
        for index in range(5):
            rows += self.series(f"history{index}", range(2000, 2005))
        return rows

    def test_native_crop_balance_and_holdout_history_coverage_are_deterministic(self):
        rows = self.balanced_rows()
        assignments, balance = self.split(rows)
        self.assertEqual(assignments["long_train"], "train")
        self.assertEqual(assignments["single_train"], "train")
        self.assertEqual(assignments["short_medium"], "train")
        self.assertEqual({name: stats["crop_count"] for name, stats in balance["partitions"].items()},
                         {"train": 100, "val": 10, "test": 15})
        self.assertEqual(balance["series_constraints"]["status"], "satisfied")
        self.assertEqual(balance["series_constraints"]["preferred_5_to_7_year_buildings_per_holdout"], 2)
        self.assertEqual(self.split(list(reversed(rows))), (assignments, balance))

    def test_length_counts_years_within_views_and_all_building_views_stay_together(self):
        rows = self.series("mixed", [2000, 2000, 2001], view="mixed_left")
        rows += self.series("mixed", [2000, 2001, 2002], view="mixed_right")
        rows += self.series("two_views", [2000, 2001], view="two_left")
        rows += self.series("two_views", [2002, 2003], view="two_right")
        assignments, balance = self.split(rows)
        details = balance["series_constraints"]["buildings"]
        self.assertEqual(details["mixed"]["max_distinct_years_per_view"], 3)
        self.assertEqual(details["mixed"]["series_class"], "medium")
        self.assertEqual(details["two_views"]["max_distinct_years_per_view"], 2)
        self.assertEqual(assignments["two_views"], "train")
        self.assertEqual(balance["series_constraints"]["status"], "insufficient")

    def test_missing_temporal_metadata_is_semantic_train_and_never_becomes_a_pair(self):
        rows = self.balanced_rows()
        rows += self.series("undated", [None, 2026])
        rows += self.series("unknown_view", [2026])
        rows[-1]["view_id"] = None
        rows += self.series("unreviewed", [2026])
        rows[-1]["metadata_status"] = "inferred"
        rows += self.series("unknown_building", [2026])
        rows[-1]["building_id"] = None
        result = self.prepare(rows)
        by_building = {row["building_id"]: row for row in result["images"]}
        undated = next(row for row in result["images"] if row["year"] is None)
        self.assertEqual(undated["split"], "train")
        self.assertTrue(undated["semantic_eligible"])
        self.assertFalse(undated["temporal_eligible"])
        self.assertEqual(undated["temporal_exclusion_reasons"], ["unknown_or_invalid_year"])
        self.assertEqual(by_building["unknown_view"]["split"], "train")
        self.assertTrue(by_building["unknown_view"]["semantic_eligible"])
        self.assertEqual(by_building["unreviewed"]["split"], "excluded")
        self.assertEqual(by_building[None]["split"], "excluded")
        ids = {undated["image_id"], by_building["unknown_view"]["image_id"]}
        self.assertFalse(any(pair["source_id"] in ids or pair["reference_id"] in ids for pair in result["pairs"]))
        validate_partitions(result["images"], allow_missing_views=True)
        self.assertEqual(read_json(self.root / "prepared/split.json")["split_policy"], "series-v2")

    def test_no_medium_buildings_reports_empty_holdouts_instead_of_using_long_series(self):
        rows = self.series("single", [2026]) + self.series("long", range(2000, 2010))
        assignments, balance = self.split(rows)
        self.assertEqual(set(assignments.values()), {"train"})
        self.assertEqual(balance["series_constraints"]["status"], "insufficient")
        self.assertTrue(balance["series_constraints"]["issues"])
        result = self.prepare(rows)
        self.assertEqual(result["summary"]["image_splits"], {"train": 11})

    def test_extension_keeps_owners_gold_and_completed_metadata(self):
        rows = self.balanced_rows() + self.series("undated", [None])
        self.prepare(rows, "first")
        previous = read_json(self.root / "first/split.json")
        updated = copy.deepcopy(rows)
        updated[-1]["year"] = 2026
        updated += self.series("new_medium", range(2010, 2017))
        self.prepare(updated, "second", previous_split=self.root / "first/split.json")
        current = read_json(self.root / "second/split.json")
        self.assertEqual(current["gold_cohort"], previous["gold_cohort"])
        for group, owner in previous["building_assignments"].items():
            self.assertEqual(current["building_assignments"][group], owner)
        old_undated = next(row for row in current["historical_cohort"] if row["building_id"] == "undated")
        self.assertEqual(old_undated["year"], 2026)
        self.prepare(updated, "third", previous_split=self.root / "second/split.json")
        changed = copy.deepcopy(updated)
        next(row for row in changed if row["building_id"] == "undated")["year"] = 2025
        with self.assertRaisesRegex(ValueError, "Historical image ownership changed"):
            self.prepare(changed, "changed", previous_split=self.root / "third/split.json")

    def test_extension_preserves_holdout_that_grows_long_and_reports_exception(self):
        rows = self.balanced_rows()
        self.prepare(rows, "first")
        previous = read_json(self.root / "first/split.json")
        group = next(group for group, owner in previous["building_assignments"].items() if owner == "test")
        rows += self.series(group, range(2005, 2008))
        self.prepare(rows, "second", previous_split=self.root / "first/split.json")
        current = read_json(self.root / "second/split.json")
        self.assertEqual(current["building_assignments"][group], "test")
        self.assertIn(group, current["balance"]["series_constraints"]["historical_policy_exceptions"])

    def test_version_transition_cannot_extend_legacy_ownership(self):
        rows = self.balanced_rows()
        manifest = self.root / "inventory.json"
        write_json(manifest, {"images": rows})
        prepare_dataset(manifest, self.root / "legacy", split_mode="reviewed", balance_unit="crops",
                        val_fraction=.08, test_fraction=.12)
        with self.assertRaisesRegex(ValueError, "Previous split policy must match"):
            self.prepare(rows, previous_split=self.root / "legacy/split.json")
        undated = self.series("undated", [None])
        write_json(manifest, {"images": rows + undated})
        result = prepare_dataset(manifest, self.root / "legacy2", split_mode="reviewed")
        self.assertEqual(result["images"][-1]["split"], "excluded")

    def test_missing_view_never_relaxes_building_hash_or_known_view_leakage(self):
        rows = self.balanced_rows()
        assignment, _ = self.split(rows)
        train = next(row for row in rows if assignment[row["building_id"]] == "train")
        test = next(row for row in rows if assignment[row["building_id"]] == "test")
        test["sha256"] = train["sha256"]
        with self.assertRaisesRegex(ValueError, "Split leakage: sha256"):
            self.prepare(rows)
        first, second = self.series("a", [None]) + self.series("b", [None])
        first.update(view_id=None, split="train")
        second.update(view_id=None, split="test", sha256=first["sha256"])
        with self.assertRaisesRegex(ValueError, "Split leakage: sha256"):
            validate_partitions([first, second], allow_missing_views=True)


if __name__ == "__main__":
    unittest.main()
