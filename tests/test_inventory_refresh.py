import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from facade_change.data import build_manifest
from facade_change.io import load_rgb, read_json, sha256, write_json
from facade_change.preparation import prepare_dataset


class InventoryRefreshTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.photos = self.root / "photos"
        self.photos.mkdir()
        self.config = self.root / "paths.json"
        self.coco = self.root / "coco.json"

    def tearDown(self):
        self.temp.cleanup()

    def image(self, image_id, name, root=None, size=(10, 8), color=None):
        path = (root or self.photos) / name
        Image.new("RGB", size, color or (image_id % 256, 20, 30)).save(path)
        return {"id": image_id, "file_name": name, "width": size[0], "height": size[1]}

    def sources(self, images, root=None):
        write_json(self.coco, {"images": images, "annotations": [], "categories": []})
        write_json(self.config, {"coco_json": str(self.coco), "image_roots": [str(root or self.photos)]})

    def initial(self, images):
        self.sources(images)
        review = self.root / "review.csv"
        with review.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image_id", "view_id", "building_id", "year", "reviewed", "notes"])
            writer.writeheader()
            writer.writerows({"image_id": row["id"], "view_id": f"reviewed_view{row['id']}",
                              "building_id": f"building{row['id']}", "year": 2000 + row["id"],
                              "reviewed": "true", "notes": "confirmed"} for row in images)
        build_manifest(self.config, self.root / "initial", review)
        return self.root / "initial/manifest.json"

    def test_coco_id_renumbering_preserves_reviewed_metadata_by_name_and_hash(self):
        images = [self.image(1, "one_2010.png"), self.image(2, "two_2010.png"), self.image(3, "three_2010.png")]
        previous = self.initial(images)
        prepare_dataset(previous, self.root / "prepared", split_mode="reviewed")
        previous = self.root / "prepared/manifest.json"
        images[0]["id"], images[1]["id"] = images[1]["id"], images[0]["id"]
        self.sources(images)
        with patch("facade_change.data.load_rgb", side_effect=AssertionError("unchanged images must not decode")):
            result = build_manifest(self.config, self.root / "refreshed", previous_manifest=previous)
        self.assertEqual([row["building_id"] for row in result["images"]], ["building1", "building2", "building3"])
        self.assertEqual([row["reused_from_image_id"] for row in result["images"]], [1, 2, 3])
        self.assertEqual(result["summary"]["reused_image_count"], 3)
        self.assertEqual(result["summary"]["decoded_image_count"], 0)
        self.assertTrue(all(row["split"] == "unassigned" and "gold_member" not in row for row in result["images"]))
        self.assertEqual(result["source"]["previous_manifest_sha256"], sha256(previous))

    def test_prefix_cleanup_and_root_move_reuse_cache_and_current_paths(self):
        images = [self.image(1, "1234abcd-wall_2010.png")]
        previous = self.initial(images)
        moved = self.root / "moved"
        moved.mkdir()
        (moved / "wall_2010.png").write_bytes((self.photos / images[0]["file_name"]).read_bytes())
        images[0].update(id=200, file_name="wall_2010.png")
        self.sources(images, moved)
        with patch("facade_change.data.load_rgb", side_effect=AssertionError("no decoding expected")):
            result = build_manifest(self.config, self.root / "moved_inventory", previous_manifest=previous)
        row = result["images"][0]
        self.assertEqual(row["image_path"], str(moved / "wall_2010.png"))
        self.assertEqual(row["building_id"], "building1")
        self.assertEqual(row["inventory_validation"], "reused_sha256_checked")
        self.assertEqual(row["metadata_inherited_from_image_id"], 1)

    def test_only_current_coco_membership_with_removed_and_new_observations(self):
        old = [self.image(i, f"wall{i}_2010.png") for i in range(1, 4)]
        previous = self.initial(old)
        new = self.image(4, "new_wall_2020.png")
        self.image(5, "outside_coco_2020.png")
        current = [old[1], old[2], new]
        self.sources(current)
        with patch("facade_change.data.load_rgb", wraps=load_rgb) as decode:
            result = build_manifest(self.config, self.root / "extended", previous_manifest=previous)
        self.assertEqual(decode.call_count, 1)
        self.assertEqual([row["image_id"] for row in result["images"]], [2, 3, 4])
        self.assertEqual(result["summary"]["reused_image_count"], 2)
        self.assertEqual(result["summary"]["decoded_image_count"], 1)
        self.assertEqual(result["summary"]["inherited_reviewed_metadata_count"], 2)
        self.assertEqual(result["images"][-1]["metadata_status"], "inferred")
        self.assertIsNone(result["images"][-1]["building_id"])

    def test_changed_bytes_same_numeric_id_do_not_inherit_reviewed_metadata(self):
        images = [self.image(1, "wall_2010.png")]
        previous = self.initial(images)
        self.image(1, "wall_2010.png", color=(100, 100, 100))
        self.sources(images)
        with patch("facade_change.data.load_rgb", wraps=load_rgb) as decode:
            result = build_manifest(self.config, self.root / "replaced", previous_manifest=previous)
        self.assertEqual(decode.call_count, 1)
        row = result["images"][0]
        self.assertEqual(row["image_status"], "ready")
        self.assertEqual(row["metadata_status"], "inferred")
        self.assertEqual(row["year"], 2010)
        self.assertIsNone(row["building_id"])
        self.assertNotIn("reused_from_image_id", row)

    def test_coco_dimension_change_forces_decode_and_metadata_review(self):
        images = [self.image(1, "wall_2010.png")]
        previous = self.initial(images)
        images[0]["width"] = 11
        self.sources(images)
        with patch("facade_change.data.load_rgb", wraps=load_rgb) as decode:
            result = build_manifest(self.config, self.root / "dimensions", previous_manifest=previous)
        self.assertEqual(decode.call_count, 1)
        self.assertEqual(result["images"][0]["image_status"], "dimension_mismatch")
        self.assertEqual(result["images"][0]["metadata_status"], "inferred")
        self.assertIsNone(result["images"][0]["building_id"])

    def test_ambiguous_previous_normalized_names_never_reuse_or_inherit(self):
        images = [self.image(1, "wall_2010.png")]
        previous = self.initial(images)
        cached = read_json(previous)
        duplicate = dict(cached["images"][0], image_id=9, file_name="1234abcd-wall_2010.png", building_id="other")
        cached["images"].append(duplicate)
        write_json(previous, cached)
        with patch("facade_change.data.load_rgb", wraps=load_rgb) as decode:
            result = build_manifest(self.config, self.root / "ambiguous_previous", previous_manifest=previous)
        self.assertEqual(decode.call_count, 1)
        self.assertEqual(result["images"][0]["metadata_status"], "inferred")
        self.assertIsNone(result["images"][0]["building_id"])

    def test_ambiguous_current_normalized_names_never_reuse_or_inherit(self):
        images = [self.image(1, "1234abcd-wall_2010.png")]
        previous = self.initial(images)
        images.append(self.image(2, "5678dcba-wall_2010.png"))
        self.sources(images)
        with patch("facade_change.data.load_rgb", wraps=load_rgb) as decode:
            result = build_manifest(self.config, self.root / "ambiguous_current", previous_manifest=previous)
        self.assertEqual(decode.call_count, 2)
        self.assertEqual(result["summary"]["reused_image_count"], 0)
        self.assertTrue(all(row["metadata_status"] == "inferred" and row["building_id"] is None for row in result["images"]))

    def test_current_explicit_overrides_take_precedence_over_old_review(self):
        images = [self.image(1, "wall_2010.png")]
        previous = self.initial(images)
        override = self.root / "updated_review.csv"
        override.write_text("image_id,view_id,building_id,year,reviewed,notes\n"
                            "1,corrected_view,corrected_building,2012,true,corrected\n", encoding="utf-8")
        result = build_manifest(self.config, self.root / "override", override, previous_manifest=previous)
        row = result["images"][0]
        self.assertEqual((row["view_id"], row["building_id"], row["year"]), ("corrected_view", "corrected_building", 2012))
        self.assertEqual(row["metadata_source"], "override_csv")

    def test_incomplete_cache_decodes_but_preserves_verified_review(self):
        images = [self.image(1, "wall_2010.png")]
        previous = self.initial(images)
        cached = read_json(previous)
        del cached["images"][0]["opaque_fraction"]
        write_json(previous, cached)
        with patch("facade_change.data.load_rgb", wraps=load_rgb) as decode:
            result = build_manifest(self.config, self.root / "incomplete", previous_manifest=previous)
        self.assertEqual(decode.call_count, 1)
        self.assertEqual(result["images"][0]["building_id"], "building1")
        self.assertEqual(result["images"][0]["metadata_source"], "previous_manifest")

    def test_changed_coco_during_inspection_is_reported_as_failed(self):
        images = [self.image(1, "wall_2010.png")]
        self.sources(images)

        def decode_and_modify(path):
            rgb = load_rgb(path)
            self.coco.write_text(self.coco.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            return rgb

        with patch("facade_change.data.load_rgb", side_effect=decode_and_modify):
            with self.assertRaisesRegex(ValueError, "Source COCO changed"):
                build_manifest(self.config, self.root / "changed_source")
        self.assertEqual(read_json(self.root / "changed_source/run.json")["status"], "failed")

    def test_changed_previous_manifest_during_inspection_is_reported_as_failed(self):
        images = [self.image(1, "wall_2010.png")]
        previous = self.initial(images)
        cached = read_json(previous)
        del cached["images"][0]["opaque_fraction"]
        write_json(previous, cached)

        def decode_and_modify(path):
            rgb = load_rgb(path)
            previous.write_text(previous.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            return rgb

        with patch("facade_change.data.load_rgb", side_effect=decode_and_modify):
            with self.assertRaisesRegex(ValueError, "Previous manifest changed"):
                build_manifest(self.config, self.root / "changed_parent", previous_manifest=previous)
        self.assertEqual(read_json(self.root / "changed_parent/run.json")["status"], "failed")


if __name__ == "__main__":
    unittest.main()
