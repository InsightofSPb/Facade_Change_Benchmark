import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from facade_change.data import ImageResolver, build_manifest, read_rename_maps
from facade_change.io import load_rgb, read_json, sha256, write_json


class RenameMapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.photos = self.root / "photos"
        self.photos.mkdir()
        self.map = self.root / "image_rename_map.csv"
        self.coco = self.root / "coco.json"
        self.config = self.root / "paths.json"

    def tearDown(self):
        self.temp.cleanup()

    def image(self, name, root=None, color=(20, 30, 40)):
        path = (root or self.photos) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (10, 8), color).save(path)
        return path.resolve()

    def mapping(self, pairs, path=None):
        path = path or self.map
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["old_name", "new_name"])
            writer.writerows(pairs)
        return path

    def sources(self, images, rules=None):
        write_json(self.coco, {"images": images, "annotations": [], "categories": []})
        write_json(self.config, {"coco_json": self.coco.name, "image_roots": [self.photos.name],
                                 "rename_maps": [self.map.name], "metadata_rules": rules})

    def row(self, name, image_id=1, **values):
        return {"id": image_id, "file_name": name, "width": 10, "height": 8, **values}

    def test_forward_and_reverse_mapping_preserve_exact_names(self):
        old, new = "deadbeef-camera.png", "house_2021.png"
        self.mapping([(old, new)])
        for disk, query in [(new, old), (old, new)]:
            with self.subTest(disk=disk):
                folder = self.root / disk.removesuffix(".png")
                folder.mkdir()
                expected = self.image(disk, folder)
                resolver = ImageResolver([folder], [self.map])
                path, rule, candidates = resolver.resolve({"file_name": query})
                self.assertEqual((path, rule, candidates), (expected, "rename_map", []))
                self.assertEqual(resolver.canonical_name(query, path), new)

    def test_explicit_and_exact_matches_win_and_exact_ambiguity_is_not_bypassed(self):
        old, new = "camera.png", "house_2021.png"
        self.mapping([(old, new)])
        expected = self.image("nested/" + old)
        self.image(new)
        resolver = ImageResolver([self.photos], [self.map])
        self.assertEqual(resolver.resolve({"file_name": old, "path": "nested/" + old})[:2],
                         (expected, "explicit_path"))
        self.assertEqual(resolver.resolve({"file_name": old})[:2], (expected, "exact_basename"))
        second = self.image("another/" + old)
        path, rule, candidates = ImageResolver([self.photos], [self.map]).resolve({"file_name": old})
        self.assertIsNone(path)
        self.assertEqual(rule, "ambiguous_exact_basename")
        self.assertEqual(candidates, sorted([str(expected), str(second)]))

    def test_map_precedes_prefix_and_casefold_fallback_uses_mapped_names(self):
        old, new = "deadbeef-camera.png", "house_2021.png"
        self.mapping([(old, new)])
        self.image("camera.png")
        expected = self.image(new)
        self.assertEqual(ImageResolver([self.photos], [self.map]).resolve({"file_name": old})[:2],
                         (expected, "rename_map"))
        expected.unlink()
        (self.photos / "camera.png").unlink()
        expected = self.image(new.upper())
        self.assertEqual(ImageResolver([self.photos], [self.map]).resolve({"file_name": old})[:2],
                         (expected, "casefold"))

    def test_mapped_disk_collisions_are_reported(self):
        self.mapping([("camera.png", "house_2021.png")])
        first = self.image("one/house_2021.png")
        second = self.image("two/house_2021.png")
        path, rule, candidates = ImageResolver([self.photos], [self.map]).resolve({"file_name": "camera.png"})
        self.assertIsNone(path)
        self.assertEqual(rule, "ambiguous_rename_map")
        self.assertEqual(candidates, sorted([str(first), str(second)]))

    def test_changed_coco_upload_prefix_uses_unique_map_alias_and_reports_collisions(self):
        self.mapping([("deadbeef-camera.png", "house_2021.png")])
        expected = self.image("house_2021.png")
        resolver = ImageResolver([self.photos], [self.map])
        query = "cafef00d-camera.png"
        self.assertEqual(resolver.resolve({"file_name": query})[:2], (expected, "rename_map"))
        self.assertEqual(resolver.canonical_name(query), "house_2021.png")
        self.mapping([("deadbeef-camera.png", "house_2021.png"),
                      ("abcdef12-camera.png", "other_2022.png")])
        resolver = ImageResolver([self.photos], [self.map])
        self.assertEqual(resolver.resolve({"file_name": query})[:2], (None, "ambiguous_rename_map"))
        self.assertEqual(resolver.canonical_name(query), query)
        self.assertEqual(resolver.resolve({"file_name": "deadbeef-camera.png"})[:2], (expected, "rename_map"))

    def test_qualified_map_names_do_not_fall_back_to_other_directories(self):
        self.mapping([("camera.png", "nested/house_2021.png")])
        self.image("other/house_2021.png")
        resolver = ImageResolver([self.photos], [self.map])
        self.assertEqual(resolver.resolve({"file_name": "camera.png"}), (None, "not_found", []))
        expected = self.image("nested/house_2021.png")
        self.assertEqual(ImageResolver([self.photos], [self.map]).resolve({"file_name": "camera.png"})[:2],
                         (expected, "rename_map"))

    def test_qualified_alias_basename_ambiguity_is_not_guessed(self):
        self.mapping([("a/camera.png", "a/house_2021.png"),
                      ("b/camera.png", "b/other_2022.png")])
        expected = self.image("a/house_2021.png")
        resolver = ImageResolver([self.photos], [self.map])
        self.assertEqual(resolver.resolve({"file_name": "camera.png"})[:2],
                         (None, "ambiguous_rename_map"))
        self.assertEqual(resolver.resolve({"file_name": "a/camera.png"})[:2], (expected, "rename_map"))

    def test_unicode_quoted_names_and_acyclic_chains(self):
        old, middle, new = "кроп, старый.png", "intermediate.png", "дом_2021.png"
        self.mapping([(old, middle), (middle, new), (old, middle)])
        expected = self.image(old)
        resolver = ImageResolver([self.photos], [self.map])
        self.assertEqual(resolver.resolve({"file_name": new})[:2], (expected, "rename_map"))
        self.assertEqual(resolver.canonical_name(old), new)

    def test_invalid_conflicting_and_cyclic_rows_fail_explicitly(self):
        invalid = [[("a.png", "b.png"), ("a.png", "c.png")],
                   [("a.png", "c.png"), ("b.png", "c.png")],
                   [("a.png", "b.png"), ("b.png", "a.png")],
                   [("a.png", "b.png"), ("b.png", "c.png"), ("c.png", "a.png")],
                   [("", "b.png")], [("../a.png", "b.png")],
                   [("a.png", "/outside/b.png")], [("a.png", "C:\\outside\\b.png")],
                   [("a.png", "nested/../b.png")], [("a.txt", "b.png")]]
        for pairs in invalid:
            with self.subTest(pairs=pairs):
                self.mapping(pairs)
                with self.assertRaises(ValueError):
                    ImageResolver([self.photos], [self.map])
        self.map.write_text("old,new\na.png,b.png\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "old_name,new_name"):
            ImageResolver([self.photos], [self.map])

    def test_identity_rows_are_noops_but_conflicting_identity_rows_fail(self):
        self.mapping([("unchanged.png", "unchanged.png"), ("a.png", "b.png")])
        groups, canonical, sources = read_rename_maps([self.map])
        self.assertEqual(groups["unchanged.png"], ("unchanged.png",))
        self.assertEqual(groups["a.png"], ("a.png", "b.png"))
        self.assertEqual(canonical["a.png"], "b.png")
        self.assertEqual(sources, [{"path": str(self.map), "sha256": sha256(self.map)}])
        for pairs in [[("a.png", "a.png"), ("a.png", "b.png")],
                      [("a.png", "b.png"), ("b.png", "b.png")]]:
            self.mapping(pairs)
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                read_rename_maps([self.map])

    def test_symlink_outside_roots_is_not_a_mapped_candidate(self):
        outside = self.image("outside.png", self.root)
        (self.photos / "house_2021.png").symlink_to(outside)
        self.mapping([("camera.png", "house_2021.png")])
        self.assertEqual(ImageResolver([self.photos], [self.map]).resolve({"file_name": "camera.png"}),
                         (None, "not_found", []))

    def test_manifest_infers_new_name_retains_coco_and_records_relative_csv_hash(self):
        old, new = "deadbeef-camera.png", "house_2021.png"
        self.mapping([(old, new), ("not_in_coco.png", "extra_2020.png")])
        expected = self.image(new)
        self.image("extra_2020.png")
        self.sources([self.row(old, 91, path="old/export/" + old)])
        result = build_manifest(self.config, self.root / "manifest")
        row = result["images"][0]
        self.assertEqual((row["image_id"], row["file_name"], row["coco_path"]),
                         (91, old, "old/export/" + old))
        self.assertEqual((row["resolved_file_name"], row["image_path"]), (new, str(expected)))
        self.assertEqual((row["view_id"], row["year"], row["metadata_status"]), ("house", 2021, "inferred"))
        self.assertEqual(result["summary"]["image_count"], 1)
        self.assertEqual(result["source"]["rename_maps"], [{"path": str(self.map), "sha256": sha256(self.map)}])
        self.assertEqual(read_json(self.root / "manifest/run.json")["config"]["rename_maps"],
                         result["source"]["rename_maps"])

    def test_mapped_filename_rules_and_id_override_precedence(self):
        old, new = "camera.png", "house_2021.png"
        self.mapping([(old, new)])
        self.image(new)
        rules = self.root / "rules.json"
        write_json(rules, [{"file_name": old, "view_id": "confirmed_view", "building_id": "confirmed",
                            "year": 2020}])
        self.sources([self.row(old, 7)], rules.name)
        row = build_manifest(self.config, self.root / "rule_manifest")["images"][0]
        self.assertEqual((row["building_id"], row["year"], row["metadata_source"]),
                         ("confirmed", 2020, "filename_rules"))
        override = self.root / "review.csv"
        override.write_text("image_id,view_id,building_id,year,reviewed,notes\n"
                            "7,override_view,override_building,2022,true,confirmed\n", encoding="utf-8")
        row = build_manifest(self.config, self.root / "override_manifest", override)["images"][0]
        self.assertEqual((row["building_id"], row["year"], row["metadata_source"]),
                         ("override_building", 2022, "override_csv"))
        write_json(rules, [{"file_name": name, "view_id": "view", "building_id": "building", "year": 2020}
                           for name in (old, new)])
        with self.assertRaisesRegex(ValueError, "Duplicate or ambiguous"):
            build_manifest(self.config, self.root / "conflicting_rules")

    def test_distinct_qualified_groups_cannot_share_metadata_or_cache_basename(self):
        self.mapping([("old_a.png", "a/shared.png"), ("old_b.png", "b/shared.png")])
        image = self.image("b/shared.png")
        rules = self.root / "rules.json"
        write_json(rules, [{"file_name": "old_a.png", "view_id": "view_a", "building_id": "building_a",
                            "year": 2000}])
        self.sources([self.row("old_b.png", 99)], rules.name)
        with self.assertRaisesRegex(ValueError, "Ambiguous canonical rename-map basename"):
            build_manifest(self.config, self.root / "ambiguous_rule")
        self.sources([self.row("old_b.png", 99)])
        previous = self.root / "cached.json"
        write_json(previous, {"images": [{"image_id": 1, "file_name": "old_a.png", "image_status": "ready",
                                         "sha256": sha256(image), "width": 10, "height": 8,
                                         "opaque_fraction": 1., "metadata_status": "reviewed",
                                         "view_id": "view_a", "building_id": "building_a", "year": 2000}]})
        with patch("facade_change.data.load_rgb", side_effect=AssertionError("guard must run before decoding")):
            with self.assertRaisesRegex(ValueError, "Ambiguous canonical rename-map basename"):
                build_manifest(self.config, self.root / "ambiguous_cache", previous_manifest=previous)
        self.assertEqual(ImageResolver([self.photos], [self.map]).resolve({"file_name": "old_b.png"})[:2],
                         (image, "rename_map"))

    def previous(self, old, new):
        self.mapping([(old, new)])
        original = self.image(old)
        self.sources([self.row(old)])
        override = self.root / "review.csv"
        override.write_text("image_id,view_id,building_id,year,reviewed,notes\n"
                            "1,reviewed_view,reviewed_building,2019,true,confirmed\n", encoding="utf-8")
        build_manifest(self.config, self.root / "previous", override)
        previous = self.root / "previous/manifest.json"
        legacy = read_json(previous)
        del legacy["images"][0]["resolved_file_name"]
        write_json(previous, legacy)
        original.rename(self.photos / new)
        self.sources([self.row(new, 99)])
        return previous

    def test_substantive_rename_and_id_change_reuse_verified_legacy_cache(self):
        previous = self.previous("camera.png", "house_2021.png")
        with patch("facade_change.data.load_rgb", side_effect=AssertionError("unchanged RGB must not decode")):
            result = build_manifest(self.config, self.root / "refreshed", previous_manifest=previous)
        row = result["images"][0]
        self.assertEqual((row["image_id"], row["file_name"]), (99, "house_2021.png"))
        self.assertEqual((row["building_id"], row["year"], row["metadata_source"]),
                         ("reviewed_building", 2019, "previous_manifest"))
        self.assertEqual(row["reused_from_image_id"], 1)
        self.assertEqual(result["summary"]["decoded_image_count"], 0)

    def test_changed_bytes_or_dimensions_do_not_inherit_reviewed_metadata(self):
        previous = self.previous("camera.png", "house_2021.png")
        self.image("house_2021.png", color=(99, 80, 60))
        with patch("facade_change.data.load_rgb", wraps=load_rgb) as decode:
            row = build_manifest(self.config, self.root / "changed", previous_manifest=previous)["images"][0]
        self.assertEqual(decode.call_count, 1)
        self.assertIsNone(row["building_id"])
        self.assertEqual(row["year"], 2021)
        self.sources([self.row("house_2021.png", 99, width=11)])
        with patch("facade_change.data.load_rgb", wraps=load_rgb) as decode:
            row = build_manifest(self.config, self.root / "dimensions", previous_manifest=previous)["images"][0]
        self.assertEqual(decode.call_count, 1)
        self.assertEqual(row["image_status"], "dimension_mismatch")
        self.assertIsNone(row["building_id"])

    def test_duplicate_coco_aliases_remain_separate_and_are_not_cache_identity(self):
        old, new = "camera.png", "house_2021.png"
        previous = self.previous(old, new)
        self.sources([self.row(old, 1), self.row(new, 2)])
        with patch("facade_change.data.load_rgb", wraps=load_rgb) as decode:
            result = build_manifest(self.config, self.root / "duplicates", previous_manifest=previous)
        self.assertEqual(decode.call_count, 2)
        self.assertEqual([row["file_name"] for row in result["images"]], [old, new])
        self.assertTrue(all(row["image_status"] == "duplicate_resolved_path" for row in result["images"]))
        self.assertEqual(result["summary"]["reused_image_count"], 0)
        self.assertEqual(result["summary"]["inherited_reviewed_metadata_count"], 0)

    def test_csv_mutation_during_manifest_is_a_failed_run(self):
        self.mapping([("camera.png", "house_2021.png")])
        self.image("house_2021.png")
        self.sources([self.row("camera.png")])

        def decode_and_mutate(path):
            rgb = load_rgb(path)
            with self.map.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            return rgb

        with patch("facade_change.data.load_rgb", side_effect=decode_and_mutate):
            with self.assertRaisesRegex(ValueError, "Rename map changed during inspection"):
                build_manifest(self.config, self.root / "mutated")
        self.assertEqual(read_json(self.root / "mutated/run.json")["status"], "failed")


if __name__ == "__main__":
    unittest.main()
