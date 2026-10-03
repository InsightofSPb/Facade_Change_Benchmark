import csv
import tempfile
import unittest
from pathlib import Path

from facade_change.io import read_json, sha256, write_json
from facade_change.preparation import possible_group_overlaps, prepare_dataset, split_reviewed_buildings, validate_partitions


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

    def test_gold_split_extends_without_moving_old_buildings(self):
        rows = [self.row(i + 1, f"view{i}", building=f"b{i:02}", reviewed=True) for i in range(10)]
        self.save(rows)
        prepare_dataset(self.manifest, self.root / "gold", split_mode="reviewed")
        first = read_json(self.root / "gold/split.json")
        rows += [self.row(i + 1, f"view{i}", building=f"b{i:02}", reviewed=True) for i in range(10, 20)]
        rebuilt, _ = split_reviewed_buildings(rows)
        self.assertNotEqual(rebuilt["b01"], first["building_assignments"]["b01"])
        self.save(rows)
        result = prepare_dataset(self.manifest, self.root / "extended", split_mode="reviewed",
                                 previous_split=self.root / "gold/split.json")
        extended = read_json(self.root / "extended/split.json")
        for building, owner in first["building_assignments"].items():
            self.assertEqual(extended["building_assignments"][building], owner)
        self.assertEqual(extended["gold_cohort"], first["gold_cohort"])
        self.assertEqual(len(extended["extended_cohort"]), 20)
        self.assertEqual(len(extended["historical_cohort"]), 20)
        self.assertEqual(result["summary"]["image_splits"], {"train": 14, "val": 2, "test": 4})
        self.assertEqual(result["summary"]["current_gold_image_count"], 10)
        self.assertEqual(result["summary"]["missing_gold_sha256"], [])
        self.assertEqual([r["gold_member"] for r in result["images"]], [True] * 10 + [False] * 10)
        self.assertEqual(extended["provenance"]["previous_split_sha256"], sha256(self.root / "gold/split.json"))
        self.assertEqual(extended["provenance"]["gold_manifest_sha256"], first["provenance"]["gold_manifest_sha256"])
        validate_partitions(result["images"])

    def test_known_building_new_image_and_coco_id_reindex_keep_gold_ownership(self):
        rows = [self.row(i, f"view{i}", building=f"building{i}", reviewed=True) for i in range(1, 11)]
        self.save(rows)
        prepare_dataset(self.manifest, self.root / "gold", split_mode="reviewed")
        first = read_json(self.root / "gold/split.json")
        for row in rows:
            row["image_id"] += 100
        new = self.row(200, "view1", year=2020, building="building1", reviewed=True)
        rows.append(new)
        self.save(rows)
        result = prepare_dataset(self.manifest, self.root / "extended", split_mode="reviewed",
                                 previous_split=self.root / "gold/split.json")
        second = read_json(self.root / "extended/split.json")
        self.assertEqual(second["building_assignments"], first["building_assignments"])
        self.assertEqual(second["gold_cohort"], first["gold_cohort"])
        self.assertEqual(result["images"][-1]["split"], first["building_assignments"]["building1"])
        self.assertFalse(result["images"][-1]["gold_member"])
        self.assertTrue(all(row["gold_member"] for row in result["images"][:-1]))
        self.assertEqual(result["summary"]["missing_gold_sha256"], [])

    def test_missing_building_is_retained_and_restored_after_two_extensions(self):
        rows = [self.row(i, f"view{i}", building=f"building{i}", reviewed=True) for i in range(1, 11)]
        self.save(rows)
        prepare_dataset(self.manifest, self.root / "gold", split_mode="reviewed")
        first = read_json(self.root / "gold/split.json")
        missing = rows.pop()
        self.save(rows)
        result = prepare_dataset(self.manifest, self.root / "partial", split_mode="reviewed",
                                 previous_split=self.root / "gold/split.json")
        partial = read_json(self.root / "partial/split.json")
        self.assertEqual(partial["building_assignments"], first["building_assignments"])
        self.assertEqual(result["summary"]["missing_gold_sha256"], [missing["sha256"]])
        self.assertEqual(result["summary"]["split_balance"]["absent_historical_building_count"], 1)
        self.assertEqual(len(partial["historical_cohort"]), 10)
        self.save([missing])
        restored = prepare_dataset(self.manifest, self.root / "restored", split_mode="reviewed",
                                   previous_split=self.root / "partial/split.json")
        self.assertEqual(restored["images"][0]["split"], first["building_assignments"][missing["building_id"]])
        self.assertTrue(restored["images"][0]["gold_member"])
        self.assertEqual(read_json(self.root / "restored/split.json")["gold_cohort"], first["gold_cohort"])

    def test_extension_rejects_seed_targets_dev_and_legacy_split(self):
        self.save([self.row(i, f"view{i}", building=f"building{i}", reviewed=True) for i in range(1, 11)])
        prepare_dataset(self.manifest, self.root / "gold", split_mode="reviewed")
        previous = self.root / "gold/split.json"
        for index, changed in enumerate(({"seed": 43}, {"val_fraction": .2}, {"split_mode": "dev"})):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(ValueError, "must match|requires split_mode=reviewed"):
                    prepare_dataset(self.manifest, self.root / f"changed{index}", previous_split=previous,
                                    **{"split_mode": "reviewed", **changed})
        legacy = read_json(previous)
        del legacy["gold_cohort"]
        write_json(self.root / "legacy.json", legacy)
        with self.assertRaisesRegex(ValueError, "recreate its initial split"):
            prepare_dataset(self.manifest, self.root / "legacy", split_mode="reviewed",
                            previous_split=self.root / "legacy.json")

    def test_extension_rejects_changed_metadata_and_replaced_original_bytes(self):
        originals = [self.row(i, f"view{i}", building=f"building{i}", reviewed=True) for i in range(1, 11)]
        self.save(originals)
        prepare_dataset(self.manifest, self.root / "gold", split_mode="reviewed")
        for key, value in (("building_id", "replacement"), ("view_id", "replacement"),
                           ("year", 2020), ("sha256", "f" * 64)):
            with self.subTest(key=key):
                rows = [dict(row) for row in originals]
                rows[0][key] = value
                self.save(rows)
                with self.assertRaisesRegex(ValueError, "Historical image ownership changed|Original image bytes changed"):
                    prepare_dataset(self.manifest, self.root / key, split_mode="reviewed",
                                    previous_split=self.root / "gold/split.json")
                self.assertEqual(read_json(self.root / key / "run.json")["status"], "failed")

    def test_absent_historical_images_still_prevent_hash_and_view_leakage(self):
        originals = [self.row(i, f"view{i}", building=f"building{i}", reviewed=True) for i in range(1, 11)]
        self.save(originals)
        prepare_dataset(self.manifest, self.root / "gold", split_mode="reviewed")
        first = read_json(self.root / "gold/split.json")
        owner = first["building_assignments"][originals[0]["building_id"]]
        other = next(row for row in originals if first["building_assignments"][row["building_id"]] != owner)
        for key in ("sha256", "view_id"):
            with self.subTest(key=key):
                new = self.row(100, "new_view", building=other["building_id"], reviewed=True)
                new[key] = originals[0][key]
                self.save([new])
                with self.assertRaisesRegex(ValueError, "Historical image ownership changed|Split leakage: view_id"):
                    prepare_dataset(self.manifest, self.root / key, split_mode="reviewed",
                                    previous_split=self.root / "gold/split.json")

    def test_upload_prefix_changes_keep_identity_and_cannot_hide_replaced_bytes(self):
        originals = [self.row(i, f"view{i}", building=f"building{i}", reviewed=True) for i in range(1, 11)]
        originals[0]["file_name"] = "1234abcd-facade_2010.png"
        self.save(originals)
        prepare_dataset(self.manifest, self.root / "gold", split_mode="reviewed")
        rows = [dict(row) for row in originals]
        rows[0].update(image_id=100, file_name="8765dcba-facade_2010.png")
        self.save(rows)
        result = prepare_dataset(self.manifest, self.root / "renamed", split_mode="reviewed",
                                 previous_split=self.root / "gold/split.json")
        self.assertTrue(result["images"][0]["gold_member"])
        self.assertEqual(result["summary"]["missing_gold_sha256"], [])
        rows[0]["sha256"] = "f" * 64
        self.save(rows)
        with self.assertRaisesRegex(ValueError, "Original image bytes changed"):
            prepare_dataset(self.manifest, self.root / "replaced", split_mode="reviewed",
                            previous_split=self.root / "gold/split.json")

    def test_distinct_originals_with_ambiguous_upload_prefix_names_are_not_merged(self):
        rows = [self.row(i, f"view{i}", building=f"building{i}", reviewed=True) for i in range(1, 11)]
        rows[0]["file_name"] = "1234abcd-facade_2010.png"
        rows[1]["file_name"] = "8765dcba-facade_2010.png"
        self.save(rows)
        with self.assertRaisesRegex(ValueError, "Ambiguous normalized filename"):
            prepare_dataset(self.manifest, self.root / "ambiguous", split_mode="reviewed")

    def test_reused_manifest_clears_gold_flags_on_dev_and_excluded_images(self):
        rows = [self.row(i, f"view{i}", building=f"building{i}", reviewed=True) for i in range(1, 11)]
        self.save(rows)
        prepare_dataset(self.manifest, self.root / "gold", split_mode="reviewed")
        dev = prepare_dataset(self.root / "gold/manifest.json", self.root / "dev")
        self.assertTrue(all(not row["gold_member"] for row in dev["images"]))
        reused = read_json(self.root / "gold/manifest.json")
        reused["images"][0]["metadata_status"] = "inferred"
        write_json(self.manifest, reused)
        result = prepare_dataset(self.manifest, self.root / "excluded", split_mode="reviewed",
                                 previous_split=self.root / "gold/split.json")
        self.assertEqual(result["images"][0]["split"], "excluded")
        self.assertFalse(result["images"][0]["gold_member"])
        self.assertEqual(result["summary"]["current_gold_image_count"], 9)

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
