import csv
import tempfile
import unittest
from pathlib import Path

from facade_change.io import read_json, sha256, write_json
from facade_change.preparation import possible_group_overlaps, prepare_dataset, validate_partitions


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "inventory.json"

    def tearDown(self):
        self.temp.cleanup()

    def row(self, image_id, view="wall", year=2010, building=None, reviewed=False):
        return {"image_id": image_id, "file_name": f"{view}_{year}.png",
                "image_path": str(self.root / "photos_not_available_here" / f"{image_id}.png"),
                "sha256": f"{image_id:064x}", "width": 20, "height": 10,
                "image_status": "ready", "metadata_status": "reviewed" if reviewed else "inferred",
                "view_id": view, "year": year, "building_id": building}

    def save(self, images):
        write_json(self.manifest, {"schema_version": 1, "source": {"coco_sha256": "sourcehash"},
                                   "categories": [{"id": 1, "name": "CRACK"}],
                                   "summary": {"annotation_count": 7}, "images": images})

    def test_dev_reuses_inventory_without_photos_and_preserves_unknowns(self):
        self.save([self.row(1), self.row(2, year=2015), self.row(3, year=2020),
                   self.row(4, view=None, year=None)])
        before = self.manifest.read_bytes()
        result = prepare_dataset(self.manifest, self.root / "prepared")
        self.assertEqual(self.manifest.read_bytes(), before)
        self.assertEqual([(p["reference_id"], p["source_id"]) for p in result["pairs"]], [(1, 2), (2, 3)])
        self.assertEqual([row["split"] for row in result["images"]], ["dev", "dev", "dev", "excluded"])
        self.assertIn("unknown_view", result["images"][-1]["preparation_exclusion_reasons"])
        self.assertEqual(result["inventory_summary"]["annotation_count"], 7)
        self.assertFalse(result["summary"]["source_images_revalidated"])
        record = read_json(self.root / "prepared/run.json")
        self.assertEqual(record["status"], "completed_dev_only")
        for name, digest in record["artifact_sha256"].items():
            self.assertEqual(sha256(self.root / "prepared" / name), digest)

    def test_pair_policies_handle_multiple_observations_same_year(self):
        self.save([self.row(1), self.row(2, year=2015), self.row(3, year=2020), self.row(4, year=2015)])
        expected = {"adjacent": {(1, 2), (1, 4), (2, 3), (4, 3)},
                    "first-anchor": {(1, 2), (1, 4), (1, 3)},
                    "all": {(1, 2), (1, 4), (1, 3), (2, 3), (4, 3)}}
        for policy, pairs in expected.items():
            with self.subTest(policy=policy):
                result = prepare_dataset(self.manifest, self.root / policy, pair_policy=policy)
                self.assertEqual({(p["reference_id"], p["source_id"]) for p in result["pairs"]}, pairs)
                self.assertTrue(all(p["source_year"] > p["reference_year"] for p in result["pairs"]))

    def test_reviewed_overrides_split_physical_buildings_and_quarantine_rest(self):
        rows = [self.row(1, "front"), self.row(2, "back"), self.row(3, "other"),
                self.row(4, "third"), self.row(5, "unconfirmed")]
        self.save(rows)
        overrides = self.root / "review.csv"
        overrides.write_text("image_id,view_id,building_id,year,reviewed,notes\n"
                             "1,front,A,2010,true,confirmed\n2,back,A,2010,true,confirmed\n"
                             "3,other,B,2010,true,confirmed\n4,third,C,2010,true,confirmed\n", encoding="utf-8")
        first = prepare_dataset(self.manifest, self.root / "one", overrides, split_mode="reviewed", test_fraction=.2)
        second = prepare_dataset(self.manifest, self.root / "two", overrides, split_mode="reviewed", test_fraction=.2)
        splits = [r["split"] for r in first["images"]]
        self.assertEqual(splits[0], splits[1])
        self.assertEqual(set(splits[:4]), {"train", "val", "test"})
        self.assertEqual(splits[-1], "excluded")
        self.assertEqual(splits, [r["split"] for r in second["images"]])
        self.assertIsNone(read_json(self.manifest)["images"][0]["building_id"])

    def test_optional_train_val_targets_images_with_uneven_buildings(self):
        rows = []
        for group, size in (("large", 8), ("small1", 1), ("small2", 1)):
            for _ in range(size):
                rows.append(self.row(len(rows) + 1, group, building=group, reviewed=True))
        self.save(rows)
        result = prepare_dataset(self.manifest, self.root / "split", split_mode="reviewed", val_fraction=.2, test_fraction=0.)
        self.assertEqual(result["summary"]["image_splits"], {"train": 8, "val": 2})
        balance = result["summary"]["split_balance"]["partitions"]
        self.assertEqual(balance["val"]["target_image_fraction"], .2)
        self.assertEqual(balance["val"]["image_count"], 2)
        self.assertEqual(balance["val"]["building_count"], 2)
        self.assertEqual(balance["val"]["image_fraction_deviation"], 0.)
        self.assertEqual(balance["test"]["image_count"], 0)
        self.assertEqual(balance["test"]["building_count"], 0)
        validate_partitions(result["images"])

    def test_default_train_val_test_uses_70_10_20_image_targets(self):
        rows = []
        for group, size in (("large", 14), ("two1", 2), ("one1", 1), ("two2", 2), ("one2", 1)):
            for _ in range(size):
                rows.append(self.row(len(rows) + 1, group, building=group, reviewed=True))
        self.save(rows)
        result = prepare_dataset(self.manifest, self.root / "split", split_mode="reviewed")
        balance = result["summary"]["split_balance"]["partitions"]
        for name, target in {"train": .7, "val": .10, "test": .20}.items():
            self.assertAlmostEqual(balance[name]["target_image_fraction"], target)
        self.assertEqual({name: stats["image_count"] for name, stats in balance.items()},
                         {"train": 14, "val": 2, "test": 4})
        buildings = {}
        for row in result["images"]:
            buildings.setdefault(row["building_id"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in buildings.values()))
        validate_partitions(result["images"])

    def test_indivisible_buildings_report_achieved_image_fraction(self):
        rows = []
        for group in ("first", "second"):
            for _ in range(5):
                rows.append(self.row(len(rows) + 1, group, building=group, reviewed=True))
        self.save(rows)
        first = prepare_dataset(self.manifest, self.root / "split", split_mode="reviewed", val_fraction=.2, test_fraction=0.)
        second = prepare_dataset(self.manifest, self.root / "same", split_mode="reviewed", val_fraction=.2, test_fraction=0.)
        balance = first["summary"]["split_balance"]["partitions"]["val"]
        self.assertEqual(balance["target_image_count"], 2.)
        self.assertEqual(balance["image_count"], 5)
        self.assertEqual(balance["image_count_deviation"], 3.)
        self.assertAlmostEqual(balance["image_fraction_deviation"], .3)
        self.assertEqual([r["split"] for r in first["images"]], [r["split"] for r in second["images"]])

    def test_reviewed_split_rejects_byte_duplicate_across_partitions(self):
        rows = [self.row(i, f"view{i}", building=f"building{i}", reviewed=True) for i in (1, 2, 3)]
        rows[1]["sha256"] = rows[0]["sha256"]
        self.save(rows)
        with self.assertRaisesRegex(ValueError, "Split leakage: sha256"):
            prepare_dataset(self.manifest, self.root / "failed", split_mode="reviewed", test_fraction=.2)
        self.assertEqual(read_json(self.root / "failed/run.json")["status"], "failed")

    def test_view_and_unknown_group_leakage_are_rejected(self):
        one = self.row(1, building="one", reviewed=True)
        two = self.row(2, building="two", reviewed=True)
        one["split"] = two["split"] = "train"
        with self.assertRaisesRegex(ValueError, "multiple building_id"):
            validate_partitions([one, two])
        two["building_id"] = None
        with self.assertRaisesRegex(ValueError, "Unreviewed physical group"):
            validate_partitions([two])
        two.update(building_id="one", split="test")
        with self.assertRaisesRegex(ValueError, "Split leakage: building_id"):
            validate_partitions([one, two])

    def test_group_overlap_is_warning_and_never_fabricates_building(self):
        views = ["9-ya_linia_VO_16-18", "9-ya_linia_VO_18", "9-ya_linia_VO_20"]
        self.save([self.row(i, view=view) for i, view in enumerate(views, 1)])
        result = prepare_dataset(self.manifest, self.root / "prepared")
        self.assertTrue(all(row["building_id"] is None for row in result["images"]))
        self.assertEqual(possible_group_overlaps(views), {views[0]: [views[1]], views[1]: [views[0]]})
        with (self.root / "prepared/group_review.csv").open(newline="", encoding="utf-8") as stream:
            groups = {row["view_id"]: row for row in csv.DictReader(stream)}
        self.assertEqual(groups[views[0]]["possible_overlapping_views"], views[1])
        self.assertEqual(groups[views[2]]["possible_overlapping_views"], "")

    def test_identical_image_pair_is_explicitly_excluded(self):
        rows = [self.row(1), self.row(2, year=2020)]
        rows[1]["sha256"] = rows[0]["sha256"]
        self.save(rows)
        result = prepare_dataset(self.manifest, self.root / "prepared")
        self.assertEqual(result["pairs"], [])
        self.assertEqual(result["excluded_pairs"][0]["reason"], "identical_image_bytes")


if __name__ == "__main__":
    unittest.main()
