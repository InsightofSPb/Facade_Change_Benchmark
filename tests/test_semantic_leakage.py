import copy
import unittest

from facade_change.semantic_training import audit_semantic_leakage


class SemanticLeakageTests(unittest.TestCase):
    @staticmethod
    def row(image_id, split="train", **kwargs):
        return {"image_id": image_id, "file_name": f"photo_{image_id}.jpg", "split": split,
                "building_id": f"building_{image_id}", "view_id": f"view_{image_id}",
                "sha256": f"hash_{image_id}", "metadata_status": "reviewed",
                "temporal_eligible": True, **kwargs}

    @staticmethod
    def pair(**kwargs):
        return {"pair_id": "1-2", "reference_id": 1, "source_id": 2, "split": "train",
                "building_id": "building_1", "view_id": "view_1", **kwargs}

    def test_clean_rows_and_temporal_pair_pass_without_mutation(self):
        rows = [self.row(1), self.row(2, building_id="building_1", view_id="view_1"),
                self.row(3, "val"), self.row(4, "test")]
        manifest = {"images": rows, "pairs": [self.pair()]}
        original = copy.deepcopy(manifest)
        report = audit_semantic_leakage(manifest, {"building_assignments": {"building_1": "train"}})
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["physical_disjointness_verified"])
        self.assertEqual(report["image_counts"], {"train": 2, "val": 1, "test": 1})
        self.assertEqual(report["pair_count"], 1)
        self.assertEqual(manifest, original)

    def test_building_view_and_sha_cross_split_overlaps_fail(self):
        for key in ("building_id", "view_id", "sha256"):
            with self.subTest(key=key):
                rows = [self.row(1), self.row(2, "val", **{key: self.row(1)[key].upper()})]
                report = audit_semantic_leakage({"images": rows})
                self.assertEqual(report["status"], "failed")
                self.assertTrue(any(v["type"] == "cross_split_identity" and v["key"] == key
                                    for v in report["violations"]))

    def test_one_view_cannot_be_assigned_two_buildings_even_within_train(self):
        report = audit_semantic_leakage({"images": [self.row(1), self.row(2, view_id="view_1")]})
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["violations"][0]["type"], "view_has_multiple_buildings")

    def test_historical_identity_and_building_owner_are_checked(self):
        for split in ({"historical_cohort": [{"split": "test", "sha256": "hash_1"}]},
                      {"gold_cohort": [{"split": "val", "view_id": "view_1"}]},
                      {"building_assignments": {"BUILDING_1": "test"}}):
            with self.subTest(split=split):
                report = audit_semantic_leakage({"images": [self.row(1)]}, split)
                self.assertEqual(report["status"], "failed")

    def test_unknown_train_is_allowed_without_physical_disjointness_claim(self):
        report = audit_semantic_leakage({"images": [
            self.row(1, building_id=None, view_id=None, metadata_status="unknown", temporal_eligible=False),
            self.row(2, "val")]})
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["known_identity_disjointness_verified"])
        self.assertFalse(report["physical_disjointness_verified"])
        self.assertEqual(report["unverified_train_images"][0]["image_id"], 1)

    def test_unknown_or_unreviewed_holdout_is_rejected(self):
        for owner in ("val", "test"):
            for changes in ({"building_id": None}, {"metadata_status": "inferred"}):
                with self.subTest(owner=owner, changes=changes):
                    report = audit_semantic_leakage({"images": [self.row(1, owner, **changes)]})
                    self.assertEqual(report["status"], "failed")
                    self.assertEqual(report["violations"][0]["type"], "unreviewed_holdout_image")

    def test_same_split_byte_duplicates_reported_but_not_cross_split_leakage(self):
        rows = [self.row(1), self.row(2, sha256="hash_1")]
        report = audit_semantic_leakage({"images": rows})
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["same_split_duplicate_hashes"][0]["image_ids"], [1, 2])
        rows[1]["split"] = "test"
        self.assertEqual(audit_semantic_leakage({"images": rows})["status"], "failed")

    def test_pair_cannot_cross_splits_or_misstate_ownership(self):
        rows = [self.row(1), self.row(2, "val")]
        report = audit_semantic_leakage({"images": rows, "pairs": [self.pair()]})
        self.assertTrue(any(v["type"] == "pair_split_mismatch" for v in report["violations"]))
        rows[1].update(split="train", building_id="building_1", view_id="view_1")
        report = audit_semantic_leakage({"images": rows, "pairs": [self.pair(split="test")]})
        self.assertEqual(report["status"], "failed")

    def test_pair_cannot_use_auxiliary_or_missing_or_duplicate_endpoint(self):
        rows = [self.row(1), self.row(2, building_id="building_1", view_id="view_1")]
        for pair, changed_row, violation in (
            (self.pair(), {"temporal_eligible": False}, "pair_ineligible_endpoint"),
            (self.pair(), {"training_role": "auxiliary_semantic_train"}, "pair_ineligible_endpoint"),
            (self.pair(source_id=99), {}, "pair_missing_endpoint"),
            (self.pair(), {"sha256": "hash_1"}, "pair_identical_observations"),
            (self.pair(view_id="other"), {}, "pair_identity_mismatch"),
        ):
            with self.subTest(violation=violation):
                report = audit_semantic_leakage({"images": [rows[0], {**rows[1], **changed_row}], "pairs": [pair]})
                self.assertEqual(report["status"], "failed")
                self.assertTrue(any(v["type"] == violation for v in report["violations"]))


if __name__ == "__main__":
    unittest.main()
