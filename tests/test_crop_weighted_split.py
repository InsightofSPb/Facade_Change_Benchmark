import tempfile
import unittest
from pathlib import Path

from facade_change.io import read_json, sha256, write_json
from facade_change.preparation import (
    prepare_dataset, semantic_crop_boxes, semantic_crop_count,
    split_reviewed_buildings, validate_partitions,
)


class CropWeightedSplitTests(unittest.TestCase):
    @staticmethod
    def image(image_id, building, crops=1):
        return {"image_id": image_id, "file_name": f"{building}_view{image_id}_2010.png",
                "image_path": f"/unavailable/photos/{image_id}.png",
                "sha256": f"{image_id:064x}", "width": 512 + (crops - 1) * 448, "height": 512,
                "image_status": "ready", "metadata_status": "reviewed",
                "building_id": building, "view_id": f"{building}_view{image_id}", "year": 2010}

    def test_small_observation_is_one_partial_crop_and_keeps_native_dimensions(self):
        self.assertEqual(list(semantic_crop_boxes(333, 200)), [(0, 0, 333, 200)])
        self.assertEqual(semantic_crop_count(333, 200), 1)
        self.assertEqual(list(semantic_crop_boxes(777, 200)),
                         [(0, 0, 512, 200), (265, 0, 512, 200)])

    def test_final_origins_are_edge_anchored_without_duplicate_regular_origins(self):
        self.assertEqual(list(semantic_crop_boxes(960, 512)),
                         [(0, 0, 512, 512), (448, 0, 512, 512)])
        self.assertEqual(list(semantic_crop_boxes(1024, 512)),
                         [(0, 0, 512, 512), (448, 0, 512, 512), (512, 0, 512, 512)])
        self.assertEqual(semantic_crop_count(1024, 1024), 9)
        self.assertEqual(semantic_crop_count(513, 513), 4)

    def test_crop_grid_covers_every_native_pixel_without_resizing(self):
        boxes = list(semantic_crop_boxes(23, 11, crop_size=8, crop_stride=6))
        covered = {(x + dx, y + dy) for x, y, width, height in boxes
                   for dy in range(height) for dx in range(width)}
        self.assertEqual(covered, {(x, y) for x in range(23) for y in range(11)})
        self.assertEqual(semantic_crop_count(23, 11, 8, 6), len(boxes))
        self.assertEqual(len(boxes), 8)

    def test_crop_geometry_and_recorded_dimensions_fail_closed(self):
        for width, height, size, stride in ((0, 512, 512, 448), (True, 512, 512, 448),
                                             (512, 2.5, 512, 448), (512, 512, 0, 448),
                                             (512, 512, 512, 513), (512, 512, 512, True)):
            with self.subTest(values=(width, height, size, stride)):
                with self.assertRaises(ValueError):
                    semantic_crop_count(width, height, size, stride)
                with self.assertRaises(ValueError):
                    list(semantic_crop_boxes(width, height, size, stride))
        with self.assertRaisesRegex(ValueError, "valid dimensions for crop balancing"):
            split_reviewed_buildings([{"building_id": "a", "image_id": 1}],
                                     val_fraction=0, test_fraction=0, balance_unit="crops")

    def test_crop_balancing_targets_weighted_observations_not_photo_counts(self):
        rows = [self.image(1, "large", 14), self.image(2, "medium", 4), self.image(3, "small", 2)]
        assignments, balance = split_reviewed_buildings(rows, balance_unit="crops")
        self.assertEqual(assignments, {"large": "train", "medium": "test", "small": "val"})
        self.assertEqual(balance["eligible_image_count"], 3)
        self.assertEqual(balance["eligible_crop_count"], 20)
        self.assertEqual(balance["target_unit"], "crops")
        self.assertEqual({key: row["crop_count"] for key, row in balance["partitions"].items()},
                         {"train": 14, "val": 2, "test": 4})
        for part, fraction in (("train", .7), ("val", .1), ("test", .2)):
            stats = balance["partitions"][part]
            self.assertEqual(stats["image_count"], 1)
            self.assertAlmostEqual(stats["image_fraction"], 1 / 3)
            self.assertAlmostEqual(stats["crop_fraction"], fraction)
            self.assertAlmostEqual(stats["crop_fraction_deviation"], 0)

    def test_frozen_buildings_and_missing_historical_owners_survive_crop_balance(self):
        fixed = {"old_train": "train", "old_val": "val", "old_test": "test", "absent": "test"}
        rows = [self.image(i, name, count) for i, (name, count) in enumerate(
            (("old_train", 7), ("old_val", 1), ("old_test", 2),
             ("new_large", 7), ("new_small", 1), ("new_medium", 2)), 1)]
        assignments, balance = split_reviewed_buildings(rows, fixed_assignments=fixed, balance_unit="crops")
        self.assertEqual({name: assignments[name] for name in fixed}, fixed)
        self.assertEqual(assignments["new_large"], "train")
        self.assertEqual(assignments["new_small"], "val")
        self.assertEqual(assignments["new_medium"], "test")
        self.assertEqual(balance["fixed_building_count"], 3)
        self.assertEqual(balance["absent_historical_building_count"], 1)
        self.assertEqual({key: row["crop_count"] for key, row in balance["partitions"].items()},
                         {"train": 14, "val": 2, "test": 4})
        self.assertEqual(split_reviewed_buildings(list(reversed(rows)), fixed_assignments=fixed,
                                                 balance_unit="crops")[0], assignments)

    def test_unreachable_fraction_reports_drift_without_moving_existing_buildings(self):
        rows = [self.image(1, "huge_val", 15), self.image(2, "old_train", 4),
                self.image(3, "old_test", 1), self.image(4, "new", 1)]
        fixed = {"huge_val": "val", "old_train": "train", "old_test": "test"}
        assignments, balance = split_reviewed_buildings(rows, fixed_assignments=fixed, balance_unit="crops")
        self.assertEqual({name: assignments[name] for name in fixed}, fixed)
        self.assertEqual(balance["partitions"]["val"]["crop_count"], 15)
        self.assertGreater(balance["partitions"]["val"]["crop_fraction_deviation"], .6)

    def test_image_default_remains_usable_without_crop_metadata(self):
        rows = [{"building_id": name} for name in ("a", "b", "c")]
        assignments, balance = split_reviewed_buildings(rows)
        self.assertEqual(set(assignments.values()), {"train", "val", "test"})
        self.assertEqual(balance["target_unit"], "images")
        self.assertFalse(balance["crop_metadata_complete"])
        self.assertIsNone(balance["eligible_crop_count"])
        self.assertTrue(all(row["crop_count"] is None for row in balance["partitions"].values()))

    def test_prepare_extends_old_image_balanced_gold_with_recorded_crop_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "inventory.json"
            rows = [self.image(i, f"building{i}") for i in range(1, 11)]
            write_json(manifest, {"images": rows})
            prepare_dataset(manifest, root / "old", split_mode="reviewed")
            previous = root / "old/split.json"
            original = read_json(previous)
            known_val = next(name for name, split in original["building_assignments"].items() if split == "val")
            rows += [self.image(11, known_val, 5), self.image(12, "new_large", 7),
                     self.image(13, "new_small", 1)]
            write_json(manifest, {"images": rows})
            result = prepare_dataset(manifest, root / "new", split_mode="reviewed", previous_split=previous,
                                     balance_unit="crops", crop_size=512, crop_stride=448)
            current = read_json(root / "new/split.json")
            self.assertEqual(current["gold_cohort"], original["gold_cohort"])
            for building, owner in original["building_assignments"].items():
                self.assertEqual(current["building_assignments"][building], owner)
            self.assertEqual(result["images"][10]["split"], "val")
            self.assertEqual(current["seed"], original["seed"])
            self.assertEqual(current["fractions"], original["fractions"])
            self.assertEqual(current["balance_unit"], "crops")
            self.assertEqual(current["crop_geometry"]["overlap"], 64)
            self.assertEqual(result["summary"]["eligible_crop_count"], 23)
            self.assertEqual(sum(stats["crop_count"] for stats in current["balance"]["partitions"].values()), 23)
            self.assertEqual(sum(stats["image_count"] for stats in current["balance"]["partitions"].values()), 13)
            self.assertEqual(current["provenance"]["previous_split_sha256"], sha256(previous))
            self.assertEqual(current["provenance"]["gold_manifest_sha256"],
                             original["provenance"]["gold_manifest_sha256"])
            self.assertEqual(result["summary"]["current_gold_image_count"], 10)
            self.assertEqual(result["pairs"], [])  # Single observations still contribute semantic crops.
            self.assertIn("Single-observation semantic crops", (root / "new/summary.txt").read_text())
            record = read_json(root / "new/run.json")
            self.assertEqual(record["status"], "completed_needs_alignment_and_label_review")
            self.assertEqual(record["config"]["balance_unit"], "crops")
            for name, digest in record["artifact_sha256"].items():
                self.assertEqual(sha256(root / "new" / name), digest)
            validate_partitions(result["images"])


if __name__ == "__main__":
    unittest.main()
