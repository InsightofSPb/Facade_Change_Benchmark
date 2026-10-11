import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from facade_change.dataset_setup import discover_previous_split, prepare_from_paths
from facade_change.io import read_json, sha256, write_json
from facade_change.preparation import validate_partitions


class SeriesDatasetSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.photos = self.root / "photos"
        self.photos.mkdir()
        self.runs = self.root / "runs"
        self.runs.mkdir()
        self.rules = self.root / "rules.json"
        self.views = []
        self.images = []
        self.review = self.root / "review.csv"

    def tearDown(self):
        self.temp.cleanup()

    def image(self, name, building=None, size=(32, 32)):
        image_id = len(self.images) + 1
        Image.new("RGB", size, (image_id, (image_id * 3) % 256, (image_id * 5) % 256)).save(self.photos / name)
        row = {"id": image_id, "file_name": f"deadbeef-{name}", "width": size[0], "height": size[1]}
        self.images.append(row)
        if building:
            view = name.rsplit("_", 1)[0]
            rule = {"view_id": view, "building_id": building}
            if rule not in self.views:
                self.views.append(rule)
        return row

    def dataset(self, medium_count=6, undated=False, unknown=False):
        for year in range(2000, 2009):
            self.image(f"long_front_{year}.png", "long_house", size=(64, 40))
        for group in range(medium_count):
            for year in range(2015, 2020):
                self.image(f"medium_{group}_{year}.png", f"medium_house_{group}")
        self.singleton = self.image("singleton_2026.png", "singleton_house")
        self.undated = self.image("camera_undated.png") if undated else None
        self.unknown = self.image("IMG_001234.png") if unknown else None
        if self.undated:
            self.review.write_text("image_id,building_id,reviewed\n"
                                   f"{self.undated['id']},undated_house,true\n", encoding="utf-8")
        return self.sources()

    def sources(self, name="coco.json"):
        path = self.root / name
        annotations = [{"id": row["id"], "image_id": row["id"], "category_id": 1, "area": 4,
                        "bbox": [1, 1, 2, 2], "segmentation": [[1, 1, 3, 1, 3, 3, 1, 3]]}
                       for row in self.images]
        annotations.append({"id": 9999, "image_id": self.singleton["id"], "category_id": 2, "area": 1,
                            "bbox": [1, 1, 1, 1], "segmentation": [[1, 1, 2, 1, 2, 2, 1, 2]]})
        write_json(path, {"images": self.images, "annotations": annotations,
                          "categories": [{"id": 1, "name": "CRACK"}, {"id": 2, "name": "DELAMINATION"}]})
        write_json(self.rules, self.views)
        return path

    def prepare(self, coco, name, **kwargs):
        kwargs.setdefault("split_policy", "series-v2")
        kwargs.setdefault("metadata_rules", self.rules)
        if self.review.is_file():
            kwargs.setdefault("metadata_csv", self.review)
        output = io.StringIO()
        with redirect_stdout(output):
            result = prepare_from_paths(coco, self.photos, out=self.runs / name, runs_root=self.runs,
                                        crop_size=32, crop_stride=24, **kwargs)
        self.stdout = output.getvalue()
        return result

    def crops(self, name):
        return [json.loads(line) for line in (self.runs / name / "semantic_crops.jsonl").read_text(
            encoding="utf-8").splitlines()]

    def test_series_v2_keeps_undated_semantic_images_in_train_and_temporal_pairs_separate(self):
        coco = self.dataset(undated=True)
        summary = self.prepare(coco, "first")
        out = self.runs / "first"
        manifest = read_json(out / "prepared/manifest.json")
        split = read_json(out / "prepared/split.json")
        by_id = {row["image_id"]: row for row in manifest["images"]}
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["split_policy"], "series-v2")
        self.assertFalse(summary["new_split_version"])
        for name, target in (("train", .80), ("val", .08), ("test", .12)):
            self.assertAlmostEqual(split["fractions"][name], target)
        for building in ("long_house", "singleton_house", "undated_house"):
            self.assertEqual(split["building_assignments"][building], "train")
        undated = by_id[self.undated["id"]]
        self.assertTrue(undated["semantic_eligible"])
        self.assertFalse(undated["temporal_eligible"])
        self.assertIsNone(undated["view_id"])
        self.assertIsNone(undated["year"])
        self.assertIn(self.undated["id"], {row["image_id"] for row in self.crops("first")})
        for pair in manifest["pairs"]:
            self.assertNotIn(self.undated["id"], (pair["reference_id"], pair["source_id"]))
            self.assertTrue(by_id[pair["reference_id"]]["temporal_eligible"])
            self.assertTrue(by_id[pair["source_id"]]["temporal_eligible"])
        validate_partitions(manifest["images"], allow_missing_views=True)
        self.assertEqual(summary["balance"]["series_constraints"]["status"], "satisfied")
        self.assertEqual(summary["crop_count"], len(self.crops("first")))
        self.assertEqual(summary["class_coverage"]["missing_classes"],
                         {"train": [], "val": ["DELAMINATION"], "test": ["DELAMINATION"]})
        with (out / "class_coverage.csv").open(encoding="utf-8", newline="") as stream:
            coverage = list(csv.DictReader(stream))
        self.assertEqual(sum(int(row["annotation_count"]) for row in coverage), len(self.images) + 1)

    def test_legacy_migration_then_auto_v2_extension_preserves_artifacts_and_reindexed_identity(self):
        coco = self.dataset()
        self.prepare(coco, "legacy", split_policy="legacy")
        old_split_path = self.runs / "legacy/prepared/split.json"
        old_split = read_json(old_split_path)
        originals = {path: sha256(path) for path in (self.runs / "legacy").rglob("*") if path.is_file()}
        originals.update({coco: sha256(coco), self.rules: sha256(self.rules)})

        migrated = self.prepare(coco, "v2")
        v2_path = self.runs / "v2/prepared/split.json"
        v2_split = read_json(v2_path)
        self.assertEqual(migrated["status"], "completed")
        self.assertTrue(migrated["new_split_version"])
        self.assertEqual(migrated["previous_split"], str(old_split_path))
        self.assertEqual(migrated["balance"]["fixed_building_count"], 0)
        self.assertEqual(read_json(self.runs / "v2/run.json")["config"]["previous_split_use"],
                         "metadata_cache_and_transition_only")
        transition = read_json(self.runs / "v2/split_transition.json")
        self.assertEqual((transition["previous_policy"], transition["current_policy"]), ("legacy", "series-v2"))
        expected_moves = [{"building_id": building, "previous": owner,
                           "current": v2_split["building_assignments"][building]}
                          for building, owner in sorted(old_split["building_assignments"].items())
                          if owner != v2_split["building_assignments"][building]]
        self.assertEqual(transition["building_changes"], expected_moves)
        self.assertEqual(transition["old_train_now_holdout"],
                         [row["building_id"] for row in expected_moves
                          if row["previous"] == "train" and row["current"] in {"val", "test"}])
        self.assertTrue(all(sha256(path) == digest for path, digest in originals.items()))
        v2_artifacts = {path: sha256(path) for path in (self.runs / "v2").rglob("*") if path.is_file()}

        self.images = [{**row, "id": row["id"] + 1000} for row in self.images]
        self.singleton = next(row for row in self.images if row["file_name"] == self.singleton["file_name"])
        updated = self.sources("reindexed.json")
        with patch("facade_change.data.load_rgb", side_effect=AssertionError("unchanged RGB should reuse verified cache")):
            extended = self.prepare(updated, "extended", metadata_rules=None)
        self.assertEqual(extended["status"], "completed")
        self.assertFalse(extended["new_split_version"])
        self.assertEqual(extended["previous_split"], str(v2_path))
        self.assertEqual(extended["inventory"]["decoded_image_count"], 0)
        self.assertEqual(extended["inventory"]["inherited_reviewed_metadata_count"], len(self.images))
        final_split = read_json(self.runs / "extended/prepared/split.json")
        self.assertEqual(final_split["building_assignments"], v2_split["building_assignments"])
        self.assertEqual(final_split["gold_cohort"], v2_split["gold_cohort"])
        self.assertEqual({row["image_id"] for row in self.crops("extended")}, {row["id"] for row in self.images})
        self.assertTrue(all(sha256(path) == digest for path, digest in v2_artifacts.items()))
        names = {row["file_name"].removeprefix("deadbeef-") for row in self.images}
        self.assertEqual(discover_previous_split(self.runs, names, "legacy"), old_split_path)
        self.assertEqual(discover_previous_split(self.runs, names, "series-v2"),
                         self.runs / "extended/prepared/split.json")

    def test_insufficient_medium_series_exports_audit_and_reports_split_review(self):
        coco = self.dataset(medium_count=1)
        summary = self.prepare(coco, "insufficient")
        out = self.runs / "insufficient"
        self.assertEqual(summary["status"], "needs_split_review")
        constraints = summary["balance"]["series_constraints"]
        self.assertEqual(constraints["status"], "insufficient")
        self.assertTrue(constraints["issues"])
        self.assertTrue((out / "metadata_review_with_names.csv").is_file())
        self.assertTrue((out / "inventory/manifest.json").is_file())
        self.assertEqual(read_json(out / "run.json")["status"], "needs_split_review")
        assignments = read_json(out / "prepared/split.json")["building_assignments"]
        self.assertEqual(assignments["long_house"], "train")
        self.assertEqual(assignments["singleton_house"], "train")
        self.assertIn("Ограничение состава split", self.stdout)

    def test_chronology_enrichment_preserves_gold_and_auto_discovers_latest_history(self):
        coco = self.dataset(undated=True)
        self.prepare(coco, "before_review")
        initial_path = self.runs / "before_review/prepared/split.json"
        initial = read_json(initial_path)
        initial_hash = sha256(initial_path)
        digest = next(row["sha256"] for row in initial["gold_cohort"]
                      if row["image_id"] == self.undated["id"])
        reviewed = self.root / "confirmed_chronology.csv"
        reviewed.write_text("image_id,building_id,view_id,year,reviewed\n"
                            f"{self.undated['id']},undated_house,confirmed_front,2026,true\n", encoding="utf-8")
        second = self.prepare(coco, "after_review", metadata_csv=reviewed)
        enriched_path = self.runs / "after_review/prepared/split.json"
        enriched = read_json(enriched_path)
        self.assertEqual(second["status"], "completed")
        self.assertEqual(second["previous_split"], str(initial_path))
        self.assertEqual(enriched["building_assignments"], initial["building_assignments"])
        self.assertEqual(enriched["gold_cohort"], initial["gold_cohort"])
        self.assertEqual(sha256(initial_path), initial_hash)
        enriched_hash = sha256(enriched_path)

        names = {row["file_name"].removeprefix("deadbeef-") for row in self.images}
        self.assertEqual(discover_previous_split(self.runs, names, "series-v2"), enriched_path)
        with patch("facade_change.data.load_rgb", side_effect=AssertionError("chronology does not change RGB")):
            third = self.prepare(coco, "inherited_review", metadata_csv=None, metadata_rules=None)
        final = read_json(self.runs / "inherited_review/prepared/split.json")
        self.assertEqual(third["status"], "completed")
        self.assertEqual(third["previous_split"], str(enriched_path))
        self.assertEqual(final["building_assignments"], initial["building_assignments"])
        self.assertEqual(final["gold_cohort"], initial["gold_cohort"])
        self.assertEqual(sha256(enriched_path), enriched_hash)
        for split in (enriched, final):
            historical = next(row for row in split["historical_cohort"] if row["sha256"] == digest)
            self.assertEqual((historical["view_id"], historical["year"], historical["split"]),
                             ("confirmed_front", 2026, "train"))
            gold = next(row for row in split["gold_cohort"] if row["sha256"] == digest)
            self.assertEqual((gold["view_id"], gold["year"]), (None, None))
        manifest = read_json(self.runs / "inherited_review/prepared/manifest.json")
        row = next(row for row in manifest["images"] if row["sha256"] == digest)
        self.assertEqual((row["view_id"], row["year"]), ("confirmed_front", 2026))
        self.assertTrue(row["temporal_eligible"])

    def test_unknown_building_stays_excluded_while_reviewed_undated_photo_trains(self):
        coco = self.dataset(undated=True, unknown=True)
        summary = self.prepare(coco, "unknown")
        self.assertEqual(summary["status"], "needs_metadata_review")
        rows = read_json(self.runs / "unknown/prepared/manifest.json")["images"]
        unknown = next(row for row in rows if row["image_id"] == self.unknown["id"])
        self.assertEqual(unknown["split"], "excluded")
        self.assertFalse(unknown["semantic_eligible"])
        self.assertIn("unknown_building", unknown["preparation_exclusion_reasons"])
        self.assertIn(self.unknown["file_name"], summary["metadata_review_names"])
        crop_ids = {row["image_id"] for row in self.crops("unknown")}
        self.assertNotIn(self.unknown["id"], crop_ids)
        self.assertIn(self.undated["id"], crop_ids)

    def test_explicit_auxiliary_train_inclusion_does_not_relabel_physical_split(self):
        coco = self.dataset(undated=True, unknown=True)
        originals = {path: sha256(path) for path in [coco, self.rules, self.review, *self.photos.glob("*.png")]}
        summary = self.prepare(coco, "auxiliary", include_unreviewed_train=True)
        out = self.runs / "auxiliary"
        self.assertEqual(summary["status"], "completed_exploratory_train_only")
        self.assertEqual(summary["auxiliary_train_image_count"], 1)
        physical = read_json(out / "prepared/manifest.json")
        unknown = next(row for row in physical["images"] if row["image_id"] == self.unknown["id"])
        self.assertEqual(unknown["split"], "excluded")
        self.assertIsNone(unknown["building_id"])
        self.assertNotEqual(unknown["metadata_status"], "reviewed")
        self.assertTrue(all(self.unknown["id"] not in (row["reference_id"], row["source_id"])
                            for row in physical["pairs"]))
        semantic = read_json(out / "semantic_manifest.json")
        auxiliary = next(row for row in semantic["images"] if row["image_id"] == self.unknown["id"])
        self.assertEqual(auxiliary["split"], "train")
        self.assertIsNone(auxiliary["building_id"])
        self.assertEqual(auxiliary["metadata_status"], unknown["metadata_status"])
        self.assertEqual(auxiliary["training_role"], "auxiliary_semantic_train")
        dataset = read_json(out / "dataset.json")
        self.assertFalse(dataset["physical_disjointness_verified"])
        crops = self.crops("auxiliary")
        extra_crops = [row for row in crops if row["image_id"] == self.unknown["id"]]
        self.assertEqual(len(extra_crops), 1)
        self.assertEqual(extra_crops[0]["split"], "train")
        self.assertIsNone(extra_crops[0]["building_id"])
        self.assertEqual(summary["crop_count"], summary["balance"]["eligible_crop_count"] + 1)
        self.assertEqual(summary["crop_count"], len(crops))
        self.assertEqual(summary["crop_splits"]["train"],
                         summary["balance"]["partitions"]["train"]["crop_count"] + 1)
        for name in ("val", "test"):
            self.assertEqual(summary["crop_splits"][name], summary["balance"]["partitions"][name]["crop_count"])
        self.assertTrue(all(sha256(path) == digest for path, digest in originals.items()))

    def test_explicit_false_keeps_unreviewed_images_out_of_training(self):
        coco = self.dataset(undated=True, unknown=True)
        summary = self.prepare(coco, "strict", include_unreviewed_train=False)
        self.assertEqual(summary["status"], "needs_metadata_review")
        self.assertEqual(summary.get("auxiliary_train_image_count", 0), 0)
        self.assertEqual(summary["crop_count"], summary["balance"]["eligible_crop_count"])
        self.assertNotIn(self.unknown["id"], {row["image_id"] for row in self.crops("strict")})
        self.assertIn(self.undated["id"], {row["image_id"] for row in self.crops("strict")})

    def test_v2_to_v3_migration_then_extension_preserves_old_files_and_new_owners(self):
        coco = self.dataset(medium_count=12, unknown=True)
        self.prepare(coco, "v2_original")
        v2_path = self.runs / "v2_original/prepared/split.json"
        original_hashes = {path: sha256(path) for path in (self.runs / "v2_original").rglob("*")
                           if path.is_file()}
        migrated = self.prepare(coco, "v3_migration", split_policy="facades-v3")
        v3_path = self.runs / "v3_migration/prepared/split.json"
        v3 = read_json(v3_path)
        self.assertEqual(migrated["status"], "completed_exploratory_train_only")
        self.assertTrue(migrated["new_split_version"])
        self.assertEqual(migrated["previous_split"], str(v2_path))
        self.assertEqual(migrated["auxiliary_train_image_count"], 1)
        self.assertEqual(sum(owner == "test" for owner in v3["building_assignments"].values()), 7)
        self.assertTrue(all(sha256(path) == digest for path, digest in original_hashes.items()))
        v3_hashes = {path: sha256(path) for path in (self.runs / "v3_migration").rglob("*") if path.is_file()}

        extended = self.prepare(coco, "v3_extension", split_policy="facades-v3")
        final = read_json(self.runs / "v3_extension/prepared/split.json")
        self.assertEqual(extended["status"], "completed_exploratory_train_only")
        self.assertFalse(extended["new_split_version"])
        self.assertEqual(extended["previous_split"], str(v3_path))
        self.assertEqual(extended["auxiliary_train_image_count"], 1)
        self.assertEqual(final["building_assignments"], v3["building_assignments"])
        self.assertEqual(final["gold_cohort"], v3["gold_cohort"])
        for name in ("v3_migration", "v3_extension"):
            extra = [row for row in self.crops(name) if row["image_id"] == self.unknown["id"]]
            self.assertEqual(len(extra), 1)
            self.assertEqual(extra[0]["split"], "train")
            self.assertIsNone(extra[0]["building_id"])
        self.assertTrue(all(sha256(path) == digest for path, digest in v3_hashes.items()))

    def test_v3_to_v4_migration_uses_new_fractions_and_then_freezes_owners(self):
        coco = self.dataset(medium_count=20, unknown=True)
        self.prepare(coco, "old_v3", split_policy="facades-v3")
        old_path = self.runs / "old_v3/prepared/split.json"
        old_hashes = {path: sha256(path) for path in (self.runs / "old_v3").rglob("*") if path.is_file()}
        with patch("facade_change.data.load_rgb", side_effect=AssertionError("reuse verified RGB")):
            migrated = self.prepare(coco, "new_v4", split_policy="facades-v4")
        new_path = self.runs / "new_v4/prepared/split.json"
        split = read_json(new_path)
        self.assertEqual(migrated["status"], "completed_exploratory_train_only")
        self.assertTrue(migrated["new_split_version"])
        self.assertEqual(migrated["previous_split"], str(old_path))
        self.assertAlmostEqual(split["fractions"]["train"], .70)
        self.assertEqual((split["fractions"]["val"], split["fractions"]["test"]), (.10, .20))
        owners = split["building_assignments"]
        self.assertEqual([list(owners.values()).count(name) for name in ("train", "val", "test")], [16, 2, 4])
        report = read_json(self.runs / "new_v4/leakage_check.json")
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["violations"], [])
        self.assertTrue(report["known_identity_disjointness_verified"])
        self.assertFalse(report["physical_disjointness_verified"])
        self.assertEqual(len(report["unverified_train_images"]), 1)
        self.assertTrue(all(sha256(path) == digest for path, digest in old_hashes.items()))
        extended = self.prepare(coco, "extend_v4", split_policy="facades-v4")
        final = read_json(self.runs / "extend_v4/prepared/split.json")
        self.assertFalse(extended["new_split_version"])
        self.assertEqual(extended["previous_split"], str(new_path))
        self.assertEqual(final["building_assignments"], owners)
        self.assertEqual(final["gold_cohort"], split["gold_cohort"])

    def test_leakage_failure_saves_report_and_prevents_semantic_index_export(self):
        coco = self.dataset(medium_count=12, unknown=True)
        report = {"status": "failed", "violations": [{"type": "cross_split_identity"}]}
        with patch("facade_change.dataset_setup.audit_semantic_leakage", return_value=report):
            with self.assertRaisesRegex(ValueError, "Split leakage check failed"):
                self.prepare(coco, "leaking", split_policy="facades-v4")
        out = self.runs / "leaking"
        self.assertEqual(read_json(out / "leakage_check.json"), report)
        self.assertEqual(read_json(out / "run.json")["status"], "failed")
        self.assertFalse((out / "dataset.json").exists())
        self.assertFalse((out / "semantic_crops.jsonl").exists())

    def test_cli_defaults_to_facades_v4_with_unknown_images_in_semantic_train(self):
        coco = self.dataset(medium_count=12, undated=True, unknown=True)
        script = Path(__file__).resolve().parents[1] / "scripts/prepare_dataset.py"
        out = self.runs / "cli"
        process = subprocess.run([sys.executable, "-s", str(script), str(coco), str(self.photos),
                                  "--runs-root", str(self.runs), "--out", str(out),
                                  "--metadata-rules", str(self.rules), "--metadata-csv", str(self.review),
                                  "--crop-size", "32", "--crop-stride", "24"],
                                 capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        summary = read_json(out / "summary.json")
        self.assertEqual(summary["split_policy"], "facades-v4")
        self.assertEqual(summary["auxiliary_train_image_count"], 1)
        self.assertEqual(read_json(out / "dataset.json")["split_policy"], "facades-v4")
        self.assertEqual(read_json(out / "leakage_check.json")["status"], "passed")
        self.assertFalse(read_json(out / "dataset.json")["physical_disjointness_verified"])
        unknown = next(row for row in self.crops("cli") if row["image_id"] == self.unknown["id"])
        self.assertEqual(unknown["split"], "train")
        self.assertIsNone(unknown["building_id"])
        physical = read_json(out / "prepared/manifest.json")
        row = next(row for row in physical["images"] if row["image_id"] == self.unknown["id"])
        self.assertEqual(row["split"], "excluded")
        self.assertNotEqual(row["metadata_status"], "reviewed")


if __name__ == "__main__":
    unittest.main()
