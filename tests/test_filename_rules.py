import tempfile
import unittest
from pathlib import Path

from facade_change.data import DEFAULT_METADATA_RULES, read_filename_rules
from facade_change.io import write_json


class FilenameRulesTests(unittest.TestCase):
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
