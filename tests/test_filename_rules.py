import tempfile
import unittest
from pathlib import Path

from facade_change.data import DEFAULT_METADATA_RULES, read_filename_rules
from facade_change.io import write_json


class FilenameRulesTests(unittest.TestCase):
    def test_view_rules_use_current_ids_and_only_terminal_filename_years(self):
        rows = [{"image_id": 701, "file_name": "deadbeef-house_left_2024.png"},
                {"id": 903, "file_name": "house_left_2026.png"},
                {"image_id": 904, "file_name": "house_left_2026_v1.png"},
                {"image_id": 905, "file_name": "house_left_1800.png"}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rules.json"
            write_json(path, [{"view_id": "house_left", "building_id": "house"},
                              {"view_id": "absent_view", "building_id": "absent_house"}])
            rules = read_filename_rules(path, rows)
        self.assertEqual(set(rules), {"701", "903"})
        self.assertEqual([rules[key]["year"] for key in ("701", "903")], [2024, 2026])
        self.assertEqual(rules["903"]["metadata_source"], "view_rules")

    def test_distinct_views_share_building_and_conflicting_view_rules_fail(self):
        rows = [{"image_id": 1, "file_name": "house_left_2024.png"},
                {"image_id": 2, "file_name": "house_right_2024.png"}]
        mappings = [{"view_id": "house_left", "building_id": "house"},
                    {"view_id": "house_right", "building_id": "house"}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rules.json"
            write_json(path, mappings)
            rules = read_filename_rules(path, rows)
            self.assertEqual([rules[key]["view_id"] for key in ("1", "2")],
                             ["house_left", "house_right"])
            self.assertEqual({rule["building_id"] for rule in rules.values()}, {"house"})
            write_json(path, mappings + [{"view_id": "house_left", "building_id": "other"}])
            with self.assertRaisesRegex(ValueError, "Conflicting building"):
                read_filename_rules(path, rows)

    def test_filename_exceptions_override_view_rules_in_either_order(self):
        rows = [{"image_id": 1, "file_name": "house_left_2024.png"},
                {"image_id": 2, "file_name": "camera.png"}]
        view = {"view_id": "house_left", "building_id": "house"}
        exception = {"file_name": "house_left_2024.png", "view_id": "corrected_view",
                     "building_id": "corrected_house", "year": 2023}
        camera = {"file_name": "camera.png", "view_id": "house_left",
                  "building_id": "house", "year": 2011}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rules.json"
            results = []
            for mappings in ([view, exception, camera], [camera, exception, view]):
                write_json(path, mappings)
                results.append(read_filename_rules(path, rows))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0]["1"]["view_id"], "corrected_view")
        self.assertEqual(results[0]["1"]["building_id"], "corrected_house")
        self.assertEqual(results[0]["1"]["year"], 2023)
        self.assertEqual(results[0]["2"]["year"], 2011)
        self.assertEqual(results[0]["1"]["metadata_source"], "filename_rules")

    def test_confirmed_dates_follow_names_after_coco_id_changes(self):
        rows = [{"id": 999, "file_name": "deadbeef-ryileeva_26_2.png"},
                {"id": 888, "file_name": "ryileeva_26_3.png"},
                {"id": 777, "file_name": "unknown.png"}]
        rules = read_filename_rules(DEFAULT_METADATA_RULES, rows)
        self.assertEqual(set(rules), {"999", "888"})
        self.assertEqual(rules["999"]["year"], 2024)
        self.assertEqual(rules["888"]["year"], 2023)
        self.assertEqual(rules["999"]["building_id"], "ryileeva_26")
        self.assertEqual(rules["999"]["metadata_status"], "reviewed")

    def test_ambiguous_current_names_cannot_receive_confirmations(self):
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            read_filename_rules(DEFAULT_METADATA_RULES, [
                {"image_id": 1, "file_name": "deadbeef-ryileeva_26_2.png"},
                {"image_id": 2, "file_name": "ryileeva_26_2.png"}])

    def test_duplicate_or_invalid_rules_fail_and_none_disables_rules(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rules.json"
            rule = {"file_name": "one.png", "view_id": "one", "building_id": "one", "year": 2024}
            write_json(path, [rule, rule])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_filename_rules(path, [])
            write_json(path, [{**rule, "year": True}])
            with self.assertRaisesRegex(ValueError, "integer year"):
                read_filename_rules(path, [])
        self.assertEqual(read_filename_rules(None, []), {})


if __name__ == "__main__":
    unittest.main()
