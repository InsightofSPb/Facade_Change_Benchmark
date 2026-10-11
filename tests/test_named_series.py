import csv
import io
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from facade_change.data import DEFAULT_METADATA_RULES, build_manifest, read_filename_rules
from facade_change.dataset_setup import prepare_from_paths
from facade_change.io import read_json, write_json
from facade_change.preparation import prepare_dataset, validate_partitions


class NamedSeriesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.photos = self.root / "photos"
        self.photos.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def observations(self):
        series = {
            "Kamennoostrovskii_57": [2013, 2016, 2019, 2021, 2022, 2023, 2025, 2026],
            "Sablinskaya_16": [2013, 2016, 2019, 2021, 2022, 2023, 2025],
            "Kamenoostrovskii_prospect_25_2": [2013, 2016, 2019, 2021, 2022, 2023, 2024, 2025, 2026],
        }
        rows = [(f"{building}_{year}.jpg", building, year)
                for building, years in series.items() for year in years]
        rows += [(f"Sablinskaya_16_2026_v{variant}.jpg", "Sablinskaya_16", 2026)
                 for variant in (1, 2)]
        return rows

    def fixture(self):
        rows, renames = [], []
        names = [name for name, _, _ in self.observations()] + ["PXL_20260926_123549548.jpg"]
        for index, name in enumerate(names, 1):
            Image.new("RGB", (10, 8), (index * 7, index * 3, 20)).save(self.photos / name)
            # Exercise both hash-prefix resolution and explicit renaming from
            # an opaque camera name to a series member.
            old = f"{index:08x}-{name}" if index % 2 else f"IMG_{index:04d}.jpg"
            rows.append({"id": 1000 + index, "file_name": old, "width": 10, "height": 8})
            renames.append((old, name))
        mapping = self.root / "image_rename_map.csv"
        with mapping.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["old_name", "new_name"])
            writer.writerows(renames)
        coco = self.root / "coco.json"
        write_json(coco, {"images": rows, "annotations": [], "categories": []})
        config = self.root / "paths.json"
        write_json(config, {"coco_json": str(coco), "image_roots": [str(self.photos)],
                            "rename_maps": [str(mapping)], "metadata_rules": str(DEFAULT_METADATA_RULES)})
        return config

    def test_explicit_rules_cover_all_dates_and_only_named_variants(self):
        observations = self.observations()
        images = [{"id": index, "file_name": f"deadbeef-{name}"}
                  for index, (name, _, _) in enumerate(observations, 1)]
        images += [{"id": 100, "file_name": "PXL_20260926_123549548.jpg"},
                   {"id": 101, "file_name": "Sablinskaya_16_2026_v3.jpg"},
                   {"id": 102, "file_name": "unreviewed_house_2026.jpg"}]
        rules = read_filename_rules(DEFAULT_METADATA_RULES, images)
        self.assertEqual(len(rules), 26)
        for index, (_, building, year) in enumerate(observations, 1):
            self.assertEqual((rules[str(index)]["building_id"], rules[str(index)]["view_id"],
                              rules[str(index)]["year"], rules[str(index)]["metadata_status"]),
                             (building, building, year, "reviewed"))
        self.assertTrue({"100", "101", "102"}.isdisjoint(rules))

    def test_inventory_and_prepare_recover_23_first_anchor_pairs(self):
        inventory = build_manifest(self.fixture(), self.root / "inventory")
        self.assertEqual(inventory["summary"]["image_status"], {"ready": 27})
        prepared = prepare_dataset(self.root / "inventory/manifest.json", self.root / "prepared",
                                   split_mode="reviewed", split_policy="series-v2", balance_unit="crops")
        reviewed = [row for row in prepared["images"] if row["metadata_status"] == "reviewed"]
        self.assertEqual(len(reviewed), 26)
        self.assertEqual({row["split"] for row in reviewed}, {"train"})
        self.assertTrue(all(row["temporal_eligible"] for row in reviewed))
        self.assertEqual(Counter(pair["building_id"] for pair in prepared["pairs"]),
                         {"Kamennoostrovskii_57": 7, "Sablinskaya_16": 8,
                          "Kamenoostrovskii_prospect_25_2": 8})
        self.assertTrue(all(pair["reference_year"] == 2013 < pair["source_year"]
                            for pair in prepared["pairs"]))
        variants = {row["image_id"] for row in reviewed
                    if row["resolved_file_name"].startswith("Sablinskaya_16_2026_v")}
        self.assertEqual(len(variants), 2)
        self.assertTrue(variants.issubset({pair["source_id"] for pair in prepared["pairs"]}))
        self.assertTrue(all(pair["reference_id"] not in variants for pair in prepared["pairs"]))
        anonymous = inventory["images"][-1]
        self.assertIsNone(anonymous["building_id"])
        self.assertNotEqual(anonymous["metadata_status"], "reviewed")
        validate_partitions(prepared["images"], allow_missing_views=True)

    def test_refresh_applies_new_rules_to_old_unreviewed_rows_without_decoding(self):
        config = self.fixture()
        current = read_json(config)
        write_json(config, {**current, "metadata_rules": None})
        previous = build_manifest(config, self.root / "previous")
        self.assertEqual(previous["summary"]["metadata_status"].get("reviewed", 0), 0)
        write_json(config, current)
        with patch("facade_change.data.load_rgb", side_effect=AssertionError("verified RGB is cached")):
            refreshed = build_manifest(config, self.root / "refreshed",
                                       previous_manifest=self.root / "previous/manifest.json")
        self.assertEqual(refreshed["summary"]["reused_image_count"], 27)
        self.assertEqual(refreshed["summary"]["metadata_status"].get("reviewed"), 26)
        self.assertIsNone(refreshed["images"][-1]["building_id"])

    @staticmethod
    def added_observations():
        rows = [("Pushkinskaya_5_2026.jpg", "Pushkinskaya_5", 2026),
                ("Marata_8.jpg", "Marata_8", None),
                ("Chkalovskii_prospect_58_2026.jpg", "Chkalovskii_prospect_58", 2026),
                ("Chkalovskii_prospect_54_2026.jpg", "Chkalovskii_prospect_54", 2026),
                ("Bolshaya_Monetnay_11_2026.jpg", "Bolshaya_Monetnay_11", 2026),
                ("Kamenoostrovskii_prospect_26_28_2026.jpg", "Kamenoostrovskii_prospect_26_28", 2026)]
        return rows + [(f"Poltavskaya_14_{year}.jpg", "Poltavskaya_14", year)
                       for year in (2009, 2013, 2016, 2019, 2023, 2024, 2025)]

    def test_new_named_facades_resolve_hash_prefix_without_inventing_dates_or_aliases(self):
        observations = self.added_observations()
        images = [{"id": index, "file_name": f"{index:08x}-{name}"}
                  for index, (name, _, _) in enumerate(observations, 1)]
        images += [{"id": 100, "file_name": "Bolshaya_Monetnaya_11_2026.jpg"},
                   {"id": 101, "file_name": "Kamennoostrovskii_prospect_26_28_2026.jpg"},
                   {"id": 102, "file_name": "Poltavskaya_14_2025_v2.jpg"}]
        rules = read_filename_rules(DEFAULT_METADATA_RULES, images)
        self.assertEqual(len(rules), 13)
        for index, (_, building, year) in enumerate(observations, 1):
            self.assertEqual((rules[str(index)]["building_id"], rules[str(index)]["view_id"],
                              rules[str(index)]["year"], rules[str(index)]["metadata_status"]),
                             (building, building, year, "reviewed"))
        self.assertTrue({"100", "101", "102"}.isdisjoint(rules))

    def test_recognized_old_auxiliary_series_stays_train_and_exports_six_temporal_pairs(self):
        observations = self.added_observations()
        current_rules = read_json(DEFAULT_METADATA_RULES)
        added_buildings = {building for _, building, _ in observations}
        old_rules = self.root / "old_rules.json"
        write_json(old_rules, [rule for rule in current_rules
                              if rule["building_id"] not in added_buildings])
        images = []
        # An existing reviewed building makes the initial semantic preparation
        # valid; all thirteen new observations initially have auxiliary TRAIN ownership.
        for index, name in enumerate(["Antonenko8_2019.jpg", "Antonenko8_2025.jpg"] +
                                     [name for name, _, _ in observations], 1):
            Image.new("RGB", (24, 24), (index * 7, index * 3, 20)).save(self.photos / name)
            images.append({"id": index, "file_name": f"{index:08x}-{name}",
                           "width": 24, "height": 24})
        coco = self.root / "coco.json"
        write_json(coco, {"images": images, "annotations": [],
                          "categories": [{"id": 1, "name": "CRACK"}]})
        runs = self.root / "runs"
        runs.mkdir()
        first, second = runs / "first", runs / "second"
        with redirect_stdout(io.StringIO()):
            prepare_from_paths(coco, self.photos, out=first, runs_root=runs,
                               metadata_rules=old_rules, crop_size=24, crop_stride=20)
            prepare_from_paths(coco, self.photos, out=second, runs_root=runs,
                               metadata_rules=DEFAULT_METADATA_RULES, crop_size=24, crop_stride=20,
                               previous_split=first / "prepared/split.json")
        manifest = read_json(second / "semantic_manifest.json")
        recognized = [row for row in manifest["images"] if row["building_id"] in added_buildings]
        self.assertEqual(len(recognized), 13)
        self.assertEqual({row["split"] for row in recognized}, {"train"})
        self.assertEqual({row["metadata_status"] for row in recognized}, {"reviewed"})
        marata = next(row for row in recognized if row["building_id"] == "Marata_8")
        self.assertIsNone(marata["year"])
        self.assertTrue(marata["semantic_eligible"])
        self.assertFalse(marata["temporal_eligible"])
        pairs = [row for row in read_json(second / "prepared/manifest.json")["pairs"]
                 if row["building_id"] in added_buildings]
        self.assertEqual(len(pairs), 6)
        self.assertEqual({row["building_id"] for row in pairs}, {"Poltavskaya_14"})
        self.assertEqual({row["reference_year"] for row in pairs}, {2009})
        self.assertEqual({row["source_year"] for row in pairs}, {2013, 2016, 2019, 2023, 2024, 2025})
        self.assertEqual(read_json(second / "leakage_check.json")["status"], "passed")

    def test_exact_filename_rule_can_confirm_undated_building_without_weakening_year_validation(self):
        images = [{"id": 1, "file_name": "deadbeef-Marat_8.jpg"},
                  {"id": 2, "file_name": "dated_2025.jpg"},
                  {"id": 3, "file_name": "unknown_date_2025.jpg"}]
        mappings = [{"file_name": "Marat_8.jpg", "view_id": "Marat_8", "building_id": "Marat_8"},
                    {"file_name": "dated_2025.jpg", "view_id": "dated", "building_id": "dated"},
                    {"file_name": "unknown_date_2025.jpg", "view_id": "unknown_date",
                     "building_id": "unknown_date", "year": None}]
        path = self.root / "rules.json"
        write_json(path, mappings)
        rules = read_filename_rules(path, images)
        self.assertEqual([rules[str(index)]["year"] for index in (1, 2, 3)], [None, 2025, None])
        for invalid in (True, False, "2025", 2025.0, 1700, 2200):
            with self.subTest(invalid=invalid):
                write_json(path, [{**mappings[0], "year": invalid}])
                with self.assertRaisesRegex(ValueError, "integer year"):
                    read_filename_rules(path, images)


if __name__ == "__main__":
    unittest.main()
