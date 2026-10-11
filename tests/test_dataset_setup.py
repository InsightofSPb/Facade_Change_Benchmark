import csv
import io
import json
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

from PIL import Image

from facade_change.dataset_setup import discover_previous_split, prepare_from_paths
from facade_change.io import read_json, sha256, write_json
from facade_change.preparation import prepare_dataset, validate_partitions


class DatasetSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.photos = self.root / "photos"
        self.photos.mkdir()
        self.runs = self.root / "runs"
        self.runs.mkdir()
        self.rules = self.root / "metadata_rules.json"
        self.view_rules = []
        self.last_stdout = ""

    def tearDown(self):
        self.temp.cleanup()

    def image(self, image_id, view, year, building=None, size=(24, 20),
              coco_name=None, present=True):
        actual_name = f"{view}_{year}.png"
        if present:
            Image.new("RGB", size, ((image_id * 43) % 256,
                                    (image_id * 61) % 256,
                                    (image_id * 73) % 256)).save(self.photos / actual_name)
        if building is not None:
            rule = {"view_id": view, "building_id": building}
            if rule not in self.view_rules:
                self.view_rules.append(rule)
        return {"id": image_id, "file_name": coco_name or f"deadbeef-{actual_name}",
                "width": size[0], "height": size[1]}

    def buildings(self, count=3):
        return [self.image(2 * group + date + 1, f"front_{group}", year,
                           building=f"building_{group}")
                for group in range(count) for date, year in enumerate((2010, 2020))]

    def sources(self, images, name="coco.json", annotations=None):
        path = self.root / name
        write_json(path, {"images": images, "annotations": annotations or [],
                          "categories": [{"id": 3, "name": "CRACK"}]})
        write_json(self.rules, self.view_rules)
        return path

    def prepare(self, coco, name, **kwargs):
        kwargs.setdefault("split_policy", "legacy")
        output = io.StringIO()
        with redirect_stdout(output):
            result = prepare_from_paths(coco, self.photos, out=self.runs / name,
                                        runs_root=self.runs, metadata_rules=self.rules, **kwargs)
        self.last_stdout = output.getvalue()
        return result

    def crops(self, out):
        return [json.loads(line) for line in (out / "semantic_crops.jsonl").read_text(
            encoding="utf-8").splitlines()]

    def csv_rows(self, path):
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_current_coco_membership_ids_and_physical_group_ownership(self):
        images = self.buildings(5)
        images += [self.image(91, "side_0", 2012, "building_0"),
                   self.image(92, "side_0", 2022, "building_0")]
        outside = self.image(700, "outside_coco", 2026, "not_in_export")
        annotation = {"id": 9007, "image_id": images[0]["id"], "category_id": 3,
                      "area": 24, "bbox": [2, 2, 6, 4], "iscrowd": 0,
                      "segmentation": [[2, 2, 8, 2, 8, 6, 2, 6]]}
        coco = self.sources(images, annotations=[annotation])
        source_before = coco.read_bytes()
        photo_hashes = {p: sha256(p) for p in self.photos.glob("*.png")}

        summary = self.prepare(coco, "complete")
        out = self.runs / "complete"
        manifest = read_json(out / "prepared/manifest.json")
        split = read_json(out / "prepared/split.json")
        indexed = self.crops(out)
        expected_ids = {row["id"] for row in images}

        self.assertEqual(summary["status"], "completed")
        self.assertTrue(summary["all_images_ready"])
        self.assertTrue(summary["all_files_present"])
        self.assertEqual(summary["image_count"], len(images))
        self.assertEqual(summary["ready_image_count"], len(images))
        self.assertEqual({r["image_id"] for r in manifest["images"]}, expected_ids)
        self.assertEqual({r["image_id"] for r in indexed}, expected_ids)
        self.assertNotIn(outside["id"], expected_ids)
        self.assertEqual(len(indexed), len(images))
        self.assertEqual(summary["crop_count"], len(indexed))
        self.assertEqual(summary["crop_splits"], {
            name: sum(r["split"] == name for r in indexed) for name in ("train", "val", "test")})
        self.assertEqual(split["balance_unit"], "crops")
        self.assertEqual(split["balance"]["eligible_crop_count"], len(indexed))
        self.assertEqual({r["split"] for r in manifest["images"]}, {"train", "val", "test"})
        validate_partitions(manifest["images"])
        by_id = {r["image_id"]: r for r in manifest["images"]}
        for crop in indexed:
            source = by_id[crop["image_id"]]
            self.assertEqual(crop["building_id"], source["building_id"])
            self.assertEqual(crop["split"], source["split"])
            self.assertEqual((crop["x"], crop["y"], crop["width"], crop["height"]), (0, 0, 24, 20))
            self.assertEqual((crop["pad_right"], crop["pad_bottom"]), (488, 492))
        owners = {r["building_id"]: r["split"] for r in manifest["images"]}
        self.assertEqual(by_id[91]["split"], owners["building_0"])
        self.assertEqual(by_id[92]["split"], owners["building_0"])
        for pair in manifest["pairs"]:
            self.assertIn(pair["reference_id"], expected_ids)
            self.assertIn(pair["source_id"], expected_ids)
            self.assertLess(pair["reference_year"], pair["source_year"])
            self.assertEqual(pair["split"], by_id[pair["reference_id"]]["split"])
            self.assertEqual(pair["split"], by_id[pair["source_id"]]["split"])
        dataset = read_json(out / "dataset.json")
        self.assertEqual(read_json(dataset["source_coco"])["annotations"], [annotation])
        self.assertEqual(dataset["temporal_alignment"], "not_run")
        self.assertEqual(dataset["temporal_change_ground_truth"], "not_created")
        self.assertEqual(coco.read_bytes(), source_before)
        self.assertTrue(all(sha256(path) == digest for path, digest in photo_hashes.items()))

    def test_edge_anchored_grid_matches_reported_crop_counts(self):
        images = self.buildings()
        coco = self.sources(images)
        summary = self.prepare(coco, "grid", crop_size=16, crop_stride=12)
        out = self.runs / "grid"
        indexed = self.crops(out)
        self.assertEqual(summary["crop_count"], 4 * len(images))
        self.assertEqual(Counter(r["image_id"] for r in indexed),
                         Counter({row["id"]: 4 for row in images}))
        for image in images:
            crops = [r for r in indexed if r["image_id"] == image["id"]]
            self.assertEqual({(r["x"], r["y"]) for r in crops}, {(0, 0), (8, 0), (0, 4), (8, 4)})
            self.assertTrue(all(r["width"] == r["height"] == 16 for r in crops))
            self.assertTrue(all(r["pad_right"] == r["pad_bottom"] == 0 for r in crops))
        balance = read_json(out / "prepared/split.json")["balance"]
        self.assertEqual(balance["eligible_crop_count"], len(indexed))
        for name, count in summary["crop_splits"].items():
            self.assertEqual(balance["partitions"][name]["crop_count"], count)

    def test_missing_file_reports_original_name_id_and_prepares_valid_subset(self):
        images = self.buildings()
        missing = self.image(999, "front_0", 2029, "building_0", present=False,
                             coco_name="folder/deadbeef-front_0_2029.png")
        coco = self.sources([*images, missing])
        summary = self.prepare(coco, "missing")
        out = self.runs / "missing"
        self.assertEqual(summary["status"], "completed_with_issues")
        self.assertFalse(summary["all_files_present"])
        self.assertFalse(summary["all_images_ready"])
        self.assertEqual(summary["image_count"], len(images) + 1)
        self.assertEqual(summary["ready_image_count"], len(images))
        self.assertEqual(summary["missing_image_names"], [missing["file_name"]])
        self.assertIn(missing["file_name"], self.last_stdout)
        self.assertIn("COCO image_id=999", self.last_stdout)
        exported = self.csv_rows(out / "missing_images.csv")
        self.assertEqual([(r["image_id"], r["file_name"]) for r in exported],
                         [("999", missing["file_name"])])
        manifest = read_json(out / "prepared/manifest.json")
        row = next(r for r in manifest["images"] if r["image_id"] == 999)
        self.assertEqual(row["split"], "excluded")
        self.assertNotIn(999, {r["image_id"] for r in self.crops(out)})
        self.assertEqual(read_json(out / "run.json")["status"], "completed_with_issues")

    def test_unconfirmed_metadata_is_visible_but_excluded_from_crops_and_pairs(self):
        images = self.buildings()
        unknown = self.image(909, "unconfirmed_view", 2026)
        coco = self.sources([*images, unknown])
        summary = self.prepare(coco, "metadata")
        out = self.runs / "metadata"
        self.assertEqual(summary["status"], "needs_metadata_review")
        self.assertTrue(summary["all_files_present"])
        self.assertEqual(summary["ready_image_count"], len(images) + 1)
        self.assertEqual(summary["metadata_review_names"], [unknown["file_name"]])
        manifest = read_json(out / "prepared/manifest.json")
        row = next(r for r in manifest["images"] if r["image_id"] == 909)
        self.assertEqual(row["split"], "excluded")
        self.assertIn("metadata_not_reviewed", row["preparation_exclusion_reasons"])
        self.assertNotIn(909, {r["image_id"] for r in self.crops(out)})
        self.assertTrue(all(909 not in (p["reference_id"], p["source_id"]) for p in manifest["pairs"]))
        self.assertIn(unknown["file_name"], self.last_stdout)

    def test_auto_previous_split_preserves_gold_after_id_reindex_and_new_dates(self):
        images = self.buildings()
        first_coco = self.sources(images, "first_coco.json")
        self.prepare(first_coco, "first", crop_size=16, crop_stride=12)
        first_path = self.runs / "first/prepared/split.json"
        first_split = read_json(first_path)
        first_artifacts = {p: sha256(p) for p in (self.runs / "first").rglob("*") if p.is_file()}
        current = [{**row, "id": row["id"] + 100} for row in images]
        current += [self.image(300, "front_0", 2026, "building_0"),
                    self.image(400, "front_new", 2011, "building_new"),
                    self.image(401, "front_new", 2025, "building_new")]
        self.image(998, "not_in_current_export", 2026, "outside")
        second_coco = self.sources(current, "second_coco.json")
        summary = self.prepare(second_coco, "extended", crop_size=16, crop_stride=12)
        out = self.runs / "extended"
        second_split = read_json(out / "prepared/split.json")
        manifest = read_json(out / "prepared/manifest.json")

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["previous_split"], str(first_path))
        for building, owner in first_split["building_assignments"].items():
            self.assertEqual(second_split["building_assignments"][building], owner)
        self.assertEqual(second_split["gold_cohort"], first_split["gold_cohort"])
        self.assertEqual(second_split["provenance"]["previous_split_sha256"], sha256(first_path))
        self.assertEqual({r["image_id"] for r in manifest["images"]}, {r["id"] for r in current})
        self.assertEqual(sum(r["gold_member"] for r in manifest["images"]), len(images))
        by_id = {r["image_id"]: r for r in manifest["images"]}
        self.assertFalse(by_id[300]["gold_member"])
        self.assertEqual(by_id[300]["split"], first_split["building_assignments"]["building_0"])
        self.assertEqual({(p["reference_id"], p["source_id"]) for p in manifest["pairs"]
                          if p["building_id"] == "building_0"}, {(101, 102), (101, 300)})
        self.assertEqual(summary["crop_count"], 4 * len(current))
        self.assertEqual({r["image_id"] for r in self.crops(out)}, {r["id"] for r in current})
        validate_partitions(manifest["images"])
        self.assertTrue(all(sha256(path) == digest for path, digest in first_artifacts.items()))

    def test_automatic_rename_csv_preserves_coco_name_and_recovers_metadata(self):
        images = self.buildings()
        original_name = "export_only_observation.png"
        images[0]["file_name"] = original_name
        mapping = self.photos / "2026_image_rename_map.csv"
        mapping.write_text("old_name,new_name\nexport_only_observation.png,front_0_2010.png\n", encoding="utf-8")
        coco = self.sources(images)
        summary = self.prepare(coco, "renamed")
        out = self.runs / "renamed"
        manifest = read_json(out / "prepared/manifest.json")
        row = next(r for r in manifest["images"] if r["image_id"] == 1)
        self.assertEqual(summary["status"], "completed")
        self.assertTrue(summary["all_files_present"])
        self.assertEqual(row["file_name"], original_name)
        self.assertEqual(row["image_path"], str(self.photos / "front_0_2010.png"))
        self.assertEqual(row["building_id"], "building_0")
        self.assertEqual((row["view_id"], row["year"]), ("front_0", 2010))
        self.assertEqual(row["resolution_rule"], "rename_map")
        self.assertEqual(read_json(out / "run.json")["config"]["rename_maps"], [str(mapping)])
        self.assertEqual(next(r for r in self.crops(out) if r["image_id"] == 1)["file_name"], original_name)

    def test_discovery_ignores_newer_failed_dev_and_benchmark_copies(self):
        images = self.buildings()
        coco = self.sources(images)
        self.prepare(coco, "original")
        original = self.runs / "original/prepared"
        with redirect_stdout(io.StringIO()):
            prepare_dataset(self.runs / "original/inventory/manifest.json", self.runs / "newer_dev")
        for name, kind, status in (("newer_failed", "dataset_preparation", "failed"),
                                   ("newer_benchmark", "hypothesis_benchmark", "completed_exploratory")):
            directory = self.runs / name
            directory.mkdir()
            for artifact in ("split.json", "manifest.json"):
                (directory / artifact).write_bytes((original / artifact).read_bytes())
            write_json(directory / "run.json", {"kind": kind, "status": status,
                                                 "finished_utc": "2099-01-01T00:00:00Z"})
        discovered = discover_previous_split(self.runs, {"front_0_2010.png"})
        self.assertEqual(discovered, original / "split.json")
        summary = self.prepare(coco, "after_ignored")
        self.assertEqual(summary["previous_split"], str(original / "split.json"))

    def test_conflicting_histories_require_review_and_still_export_missing_names(self):
        images = self.buildings()
        coco = self.sources(images)
        self.prepare(coco, "first")
        with redirect_stdout(io.StringIO()):
            prepare_dataset(self.runs / "first/inventory/manifest.json", self.runs / "conflicting",
                            split_mode="reviewed", seed=99)
        missing = self.image(999, "front_0", 2029, "building_0", present=False)
        current = self.sources([*images, missing], "current_coco.json")
        summary = self.prepare(current, "requires_choice")
        out = self.runs / "requires_choice"
        self.assertEqual(summary["status"], "needs_split_review")
        self.assertIn("--previous-split", summary["split_selection_error"])
        self.assertIn(missing["file_name"], self.last_stdout)
        self.assertEqual(self.csv_rows(out / "missing_images.csv")[0]["image_id"], "999")
        self.assertTrue((out / "inventory/manifest.json").is_file())
        self.assertFalse((out / "prepared/split.json").exists())
        self.assertFalse((out / "dataset.json").exists())
        self.assertEqual(summary["crop_count"], 0)

        previous = self.runs / "first/prepared/split.json"
        selected = self.prepare(current, "chosen", previous_split=previous)
        selected_split = read_json(self.runs / "chosen/prepared/split.json")
        self.assertEqual(selected["status"], "completed_with_issues")
        self.assertEqual(selected["previous_split"], str(previous))
        self.assertIsNone(selected["split_selection_error"])
        self.assertEqual(selected_split["building_assignments"], read_json(previous)["building_assignments"])

    def test_explicit_previous_split_rejects_tampered_artifacts_and_keeps_audit(self):
        images = self.buildings()
        first_coco = self.sources(images, "first_coco.json")
        self.prepare(first_coco, "first")
        previous = self.runs / "first/prepared/split.json"
        missing = self.image(999, "front_0", 2029, "building_0", present=False)
        current = self.sources([*images, missing], "current_coco.json")
        for artifact_name in ("split.json", "manifest.json"):
            artifact = previous.with_name(artifact_name)
            original = artifact.read_bytes()
            try:
                artifact.write_bytes(original + b"\n")
                for explicit in (False, True):
                    with self.subTest(artifact=artifact_name, explicit=explicit):
                        name = f"tampered_{artifact.stem}_{explicit}"
                        kwargs = {"previous_split": previous} if explicit else {}
                        result = self.prepare(current, name, **kwargs)
                        out = self.runs / name
                        self.assertEqual(result["status"], "needs_split_review")
                        self.assertIn(artifact_name, result["split_selection_error"])
                        self.assertEqual(result["image_count"], len(images) + 1)
                        self.assertEqual(result["ready_image_count"], len(images))
                        self.assertEqual(result["missing_image_names"], [missing["file_name"]])
                        self.assertIn("COCO image_id=999", self.last_stdout)
                        self.assertTrue((out / "inventory/manifest.json").is_file())
                        self.assertEqual(self.csv_rows(out / "missing_images.csv")[0]["image_id"], "999")
                        self.assertFalse((out / "prepared/split.json").exists())
                        self.assertFalse((out / "dataset.json").exists())
                        self.assertEqual(read_json(out / "run.json")["status"], "needs_split_review")
            finally:
                artifact.write_bytes(original)

    def test_latest_subset_matches_gold_history_and_preserves_absent_owners(self):
        images = self.buildings()
        first_coco = self.sources(images, "first_coco.json")
        self.prepare(first_coco, "first")
        new_images = [self.image(500, "front_new", 2011, "building_new"),
                      self.image(501, "front_new", 2025, "building_new")]
        extended_coco = self.sources([*images, *new_images], "extended_coco.json")
        self.prepare(extended_coco, "extended")
        subset_coco = self.sources(new_images, "subset_coco.json")
        subset = self.prepare(subset_coco, "latest_subset")
        latest = self.runs / "latest_subset/prepared/split.json"
        latest_manifest = read_json(latest.with_name("manifest.json"))
        latest_split = read_json(latest)
        self.assertEqual(subset["status"], "completed_with_issues")
        self.assertEqual({r["image_id"] for r in latest_manifest["images"]}, {500, 501})
        self.assertEqual({r["image_id"] for r in latest_split["gold_cohort"]},
                         {r["id"] for r in images})

        discovered = discover_previous_split(self.runs, {"front_0_2010.png"})
        self.assertEqual(discovered, latest)
        resumed = self.prepare(first_coco, "resumed")
        resumed_split = read_json(self.runs / "resumed/prepared/split.json")
        resumed_manifest = read_json(self.runs / "resumed/prepared/manifest.json")
        self.assertEqual(resumed["previous_split"], str(latest))
        self.assertEqual(resumed_split["building_assignments"], latest_split["building_assignments"])
        self.assertEqual(resumed_split["historical_cohort"], latest_split["historical_cohort"])
        self.assertEqual(resumed_split["gold_cohort"], latest_split["gold_cohort"])
        self.assertEqual({r["image_id"] for r in resumed_manifest["images"]}, {r["id"] for r in images})
        self.assertIn("building_new", resumed_split["building_assignments"])
        validate_partitions(resumed_manifest["images"])

    def test_full_rename_chains_discover_original_split_after_all_names_change(self):
        images = self.buildings()
        first_coco = self.sources(images, "first_coco.json")
        self.prepare(first_coco, "first")
        previous = self.runs / "first/prepared/split.json"
        first_split = read_json(previous)
        current, aliases = [], []
        for row in images:
            old_name = row["file_name"].removeprefix("deadbeef-")
            middle, final = "middle_" + old_name, "renamed_" + old_name
            (self.photos / old_name).rename(self.photos / final)
            current.append({**row, "id": row["id"] + 1000, "file_name": final})
            # The current name C is encountered only on the second A->B, B->C
            # row. A one-pass alias expansion would omit every old name A.
            aliases.extend(((old_name, middle), (middle, final)))
        mapping = self.photos / "2026_image_rename_map.csv"
        with mapping.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("old_name", "new_name"))
            writer.writerows(aliases)
        current_coco = self.sources(current, "current_coco.json")
        result = self.prepare(current_coco, "all_renamed")
        out = self.runs / "all_renamed"
        split = read_json(out / "prepared/split.json")
        manifest = read_json(out / "prepared/manifest.json")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["previous_split"], str(previous))
        self.assertEqual(split["building_assignments"], first_split["building_assignments"])
        self.assertEqual(split["gold_cohort"], first_split["gold_cohort"])
        self.assertTrue(all(r["gold_member"] for r in manifest["images"]))
        self.assertEqual({r["image_id"] for r in manifest["images"]}, {r["id"] for r in current})
        self.assertEqual({r["file_name"] for r in manifest["images"]}, {r["file_name"] for r in current})
        validate_partitions(manifest["images"])


if __name__ == "__main__":
    unittest.main()
