"""Semantic split extension must retain auxiliary TRAIN exposure history."""
import io
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from PIL import Image

from facade_change.dataset_setup import prepare_from_paths
from facade_change.io import read_json, write_json


class SemanticExtensionGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.photos = self.root / "photos"
        self.photos.mkdir()
        self.runs = self.root / "runs"
        self.runs.mkdir()
        self.rules = self.root / "rules.json"
        self.images, self.views = [], []
        for group in range(10):
            for year in range(2020, 2023):
                self.image(f"front_{group}_{year}.png", f"house_{group}")
        self.unknown = self.image("camera_unknown.png")

    def tearDown(self):
        self.temp.cleanup()

    def image(self, name, building=None):
        index = len(self.images) + 1
        Image.new("RGB", (24, 24), (index, index * 3 % 256, index * 7 % 256)).save(self.photos / name)
        row = {"id": index, "file_name": name, "width": 24, "height": 24}
        self.images.append(row)
        if building:
            rule = {"view_id": name.rsplit("_", 1)[0], "building_id": building}
            if rule not in self.views:
                self.views.append(rule)
        return row

    def prepare(self, name, images=None, **kwargs):
        coco = self.root / f"{name}.json"
        write_json(coco, {"images": self.images if images is None else images,
                          "annotations": [], "categories": [{"id": 1, "name": "CRACK"}]})
        write_json(self.rules, self.views)
        with redirect_stdout(io.StringIO()):
            prepare_from_paths(coco, self.photos, out=self.runs / name, runs_root=self.runs,
                               metadata_rules=self.rules, crop_size=24, crop_stride=20,
                               split_policy="facades-v4", **kwargs)
        return self.runs / name

    def test_rekey_add_two_facades_keeps_all_frozen_owners(self):
        first = self.prepare("first")
        original = read_json(first / "prepared/split.json")
        for group in (10, 11):
            for year in range(2020, 2023):
                self.image(f"front_{group}_{year}.png", f"house_{group}")
        rekeyed = [{**row, "id": row["id"] + 1000} for row in self.images]
        second = self.prepare("second", rekeyed, previous_split=first / "prepared/split.json")
        extended = read_json(second / "prepared/split.json")
        for building, owner in original["building_assignments"].items():
            self.assertEqual(extended["building_assignments"][building], owner)
        self.assertEqual(len(extended["building_assignments"]), 12)
        self.assertEqual(extended["gold_cohort"], original["gold_cohort"])
        self.assertEqual(read_json(second / "leakage_check.json")["status"], "passed")

    def test_omitted_auxiliary_photo_readded_and_recognized_forces_whole_building_train(self):
        first = self.prepare("first")
        old_sha = next(row["sha256"] for row in read_json(first / "semantic_manifest.json")["images"]
                       if row["image_id"] == self.unknown["id"])
        without = [row for row in self.images if row != self.unknown]
        second = self.prepare("without", without)
        self.assertIn(old_sha, {row["sha256"] for row in read_json(second / "semantic_history.json")["images"]})
        new = self.image("recognized_2010.png", "recognized_house")
        shutil.copyfile(self.photos / self.unknown["file_name"], self.photos / new["file_name"])
        self.image("recognized_2015.png", "recognized_house")
        self.image("recognized_2025.png", "recognized_house")
        third = self.prepare("recognized", [row for row in self.images if row != self.unknown])
        split = read_json(third / "prepared/split.json")
        self.assertEqual(split["building_assignments"]["recognized_house"], "train")
        semantic = read_json(third / "semantic_manifest.json")
        recognized = [row for row in semantic["images"] if row["building_id"] == "recognized_house"]
        self.assertEqual(len(recognized), 3)
        self.assertEqual({row["split"] for row in recognized}, {"train"})
        self.assertEqual(read_json(third / "extension_guard.json")["inherited_building_owners"],
                         {"recognized_house": "train"})
        # The next extension discovers the cumulative history, not the older full COCO.
        fourth = self.prepare("repeated", [row for row in self.images if row != self.unknown])
        self.assertEqual(read_json(fourth / "prepared/split.json")["building_assignments"],
                         split["building_assignments"])

    def test_auxiliary_train_recognition_as_frozen_holdout_fails(self):
        first = self.prepare("first")
        assignments = read_json(first / "prepared/split.json")["building_assignments"]
        held = next(building for building, owner in assignments.items() if owner == "test")
        review = self.root / "review.csv"
        review.write_text("image_id,building_id,view_id,year,reviewed\n"
                          f"{self.unknown['id']},{held},previous_unknown,2026,true\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Historical semantic ownership conflict"):
            self.prepare("conflict", metadata_csv=review)
        self.assertFalse((self.runs / "conflict/dataset.json").exists())

    def test_previous_release_semantic_manifest_bootstraps_auxiliary_history(self):
        first = self.prepare("old_release")
        # The user's existing 392-image preparation predates semantic_history.json.
        (first / "semantic_history.json").unlink()
        record = read_json(first / "run.json")
        record["artifact_sha256"].pop("semantic_history.json")
        write_json(first / "run.json", record)
        assignments = read_json(first / "prepared/split.json")["building_assignments"]
        held = next(building for building, owner in assignments.items() if owner == "val")
        review = self.root / "old_release_review.csv"
        review.write_text("image_id,building_id,view_id,year,reviewed\n"
                          f"{self.unknown['id']},{held},previous_unknown,2026,true\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Historical semantic ownership conflict"):
            self.prepare("old_release_conflict", previous_split=first / "prepared/split.json",
                         metadata_csv=review)

    def test_renamed_train_duplicate_cannot_enter_existing_val_building(self):
        first = self.prepare("first")
        rows = read_json(first / "semantic_manifest.json")["images"]
        train = next(row for row in rows if row["split"] == "train" and row["building_id"])
        val = next(row for row in rows if row["split"] == "val")
        alias = self.image(f"{val['view_id']}_2026.png", val["building_id"])
        shutil.copyfile(train["image_path"], self.photos / alias["file_name"])
        with self.assertRaisesRegex(ValueError, "Historical semantic ownership conflict"):
            self.prepare("alias_conflict")
        self.assertFalse((self.runs / "alias_conflict/dataset.json").exists())

    def test_tampered_semantic_history_is_rejected(self):
        first = self.prepare("first")
        history = first / "semantic_history.json"
        content = read_json(history)
        content["images"] = []
        write_json(history, content)
        with self.assertRaisesRegex(ValueError, "Semantic ownership artifact is missing or changed"):
            self.prepare("tampered", previous_split=first / "prepared/split.json")


if __name__ == "__main__":
    unittest.main()
