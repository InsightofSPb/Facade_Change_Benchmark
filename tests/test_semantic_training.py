import copy
import unittest

from facade_change.semantic_training import include_auxiliary_training


class AuxiliarySemanticTrainingTests(unittest.TestCase):
    def row(self, image_id, split="excluded", **kwargs):
        row = {"image_id": image_id, "file_name": f"IMG_{image_id}.jpg", "sha256": str(image_id),
               "image_status": "ready", "metadata_status": "unknown", "building_id": None,
               "view_id": None, "year": None, "split": split, "semantic_eligible": False,
               "temporal_eligible": False,
               "preparation_exclusion_reasons": ["metadata_not_reviewed", "unknown_building"]}
        if split != "excluded":
            row.update(building_id=f"building{image_id}", metadata_status="reviewed",
                       semantic_eligible=True, preparation_exclusion_reasons=[])
        return {**row, **kwargs}

    def test_unknown_identity_is_preserved_and_originals_are_unchanged(self):
        rows = [self.row(1, "train"), self.row(2), self.row(3, year=2026, view_id="inferred_front")]
        prepared = {"images": rows, "pairs": [], "summary": {"semantic_eligible_image_count": 1}}
        split = {"building_assignments": {"building1": "train"}}
        before, original_split = copy.deepcopy(prepared), copy.deepcopy(split)
        manifest, report = include_auxiliary_training(prepared, split, True)
        self.assertEqual(prepared, before)
        self.assertEqual(split, original_split)
        self.assertEqual([r["image_id"] for r in report["included_images"]], [2, 3])
        self.assertFalse(manifest["physical_disjointness_verified"])
        self.assertEqual(manifest["images"][0]["training_role"], "reviewed")
        for row in manifest["images"][1:]:
            self.assertEqual(row["split"], "train")
            self.assertEqual(row["training_role"], "auxiliary_semantic_train")
            self.assertIsNone(row["building_id"])
            self.assertEqual(row["metadata_status"], "unknown")
            self.assertTrue(row["semantic_eligible"])
            self.assertFalse(row["temporal_eligible"])
        self.assertEqual(manifest["images"][2]["view_id"], "inferred_front")
        self.assertEqual(manifest["images"][2]["year"], 2026)
        self.assertEqual(manifest["pairs"], [])
        self.assertEqual(manifest["summary"]["semantic_eligible_image_count"], 3)

    def test_current_historical_and_absent_holdouts_are_blocked(self):
        rows = [self.row(1, "val", view_id="held_front"), self.row(2, sha256="1"),
                self.row(3, view_id="held_front"), self.row(4, building_id="absent_test"),
                self.row(5, sha256="historical_hash"), self.row(6, view_id="gold_view")]
        split = {"building_assignments": {"absent_test": "test"},
                 "historical_cohort": [{"split": "test", "sha256": "historical_hash"}],
                 "gold_cohort": [{"split": "val", "view_id": "gold_view"}]}
        manifest, report = include_auxiliary_training({"images": rows}, split, True)
        self.assertEqual(len(report["blocked_images"]), 5)
        self.assertFalse(report["included_images"])
        self.assertTrue(report["physical_disjointness_verified"])
        self.assertTrue(all(row["split"] == "excluded" for row in manifest["images"][1:]))

    def test_exact_train_duplicates_and_earlier_auxiliary_duplicates_are_skipped(self):
        rows = [self.row(1, "train"), self.row(2, sha256="1"), self.row(3), self.row(4, sha256="3")]
        manifest, report = include_auxiliary_training({"images": rows}, {}, True)
        self.assertEqual([r["image_id"] for r in report["included_images"]], [3])
        self.assertEqual([r["image_id"] for r in report["duplicate_images"]], [2, 4])
        self.assertEqual(manifest["images"][1]["split"], "excluded")
        self.assertEqual(manifest["images"][3]["auxiliary_training_exclusion_reason"], "duplicate_train_sha256")

    def test_disabled_and_integrity_exclusions_never_add_training_images(self):
        rows = [self.row(1), self.row(2, image_status="missing"), self.row(3, sha256=None),
                self.row(4, preparation_exclusion_reasons=["unknown_building", "image_dimension_mismatch"]),
                self.row(5, preparation_exclusion_reasons=[])]
        disabled, report = include_auxiliary_training({"images": rows}, {}, False)
        self.assertFalse(report["enabled"])
        self.assertFalse(report["included_images"])
        self.assertTrue(all(row["split"] == "excluded" for row in disabled["images"]))
        enabled, report = include_auxiliary_training({"images": rows}, {}, True)
        self.assertEqual([r["image_id"] for r in report["included_images"]], [1])
        self.assertTrue(all(row["split"] == "excluded" for row in enabled["images"][1:]))


if __name__ == "__main__":
    unittest.main()
