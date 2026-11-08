"""Geometric support is visible context and a verdict boundary, never occlusion GT."""
import importlib
import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from facade_change.agentic.reasoning import run_reasoning, validate_final
from facade_change.agentic.tools import digest
from facade_change.io import read_json, sha256


fixtures = importlib.import_module("tests.test_2026_11_08_agentic_reasoning")


class PairSupportRegressionTests(unittest.TestCase):
    def setUp(self):
        # Reuse the real RegionStore/SearchIndex and the explicit inference double.
        self.fixture = fixtures.ToolReasoningTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    def pair(self, reference_support, source_support, *, anchor_bbox=None):
        region = self.fixture.region
        if anchor_bbox is not None:
            region = self.fixture.store.register_region("negative-anchor", "negative-view", "b" * 64,
                                                        anchor_bbox, "test")
        observations = []
        for role, date, support in (("support-reference", "2011", reference_support),
                                    ("support-source", "2021", source_support)):
            crop = np.full((4, 4, 3), 12, np.uint8)
            if role == "support-source":
                crop[:, 2:] = 220  # New appearance exists only where the reference may be absent.
            original = np.full((8, 8, 3), 12, np.uint8)
            original[1:5, 2:6] = crop
            original_path = self.root / (role + "-original.png")
            crop_path = self.root / (role + "-crop.png")
            support_path = self.root / (role + "-support.npy")
            Image.fromarray(original).save(original_path)
            Image.fromarray(crop).save(crop_path)
            np.save(support_path, support)
            geometry = dict(self.fixture.reference["geometry"])
            geometry["bbox_xyxy"] = region["bbox_xyxy"]
            x, y = region["bbox_xyxy"][:2]
            geometry["crop_to_anchor"] = [[1., 0., x], [0., 1., y], [0., 0., 1.]]
            observations.append(self.fixture.store.add_observation({
                "observation_id": role, "region_id": region["region_id"], "date": date,
                "date_status": "confirmed", "available_at": date,
                "original_path": str(original_path), "crop_path": str(crop_path),
                "support_path": str(support_path), "geometry": geometry,
                "correspondence_status": "aligned", "visibility": "unknown"}))
        case = {**self.fixture.case, "case_id": "support-pair", "region_id": region["region_id"],
                "building_id": region["building_id"], "view_id": region["view_id"],
                "reference_observation_id": observations[0]["observation_id"],
                "source_observation_id": observations[1]["observation_id"],
                "reference_date": "2011", "source_date": "2021", "cutoff": "2021"}
        return case, observations

    @staticmethod
    def reply(bbox=None, coordinate="region_crop_pixels", decision="change", **owner):
        def respond(request):
            value = fixtures.final_reply(request, decision)
            if bbox is not None:
                identifier = next(block["evidence_id"] for message in request["messages"]
                                  for block in message["content"] if block["type"] == "image")
                value["localization"] = {"evidence_id": identifier, "coordinate_system": coordinate,
                                         "bbox_xyxy": bbox, **owner}
            return value
        return respond

    def test_all_modes_receive_support_and_reject_first_observation_as_localized_change(self):
        left = np.zeros((4, 4), bool)
        left[:, :2] = True
        case, observations = self.pair(left, np.ones((4, 4), bool))
        original_hashes = [sha256(row["crop_path"]) for row in observations]
        for mode in ("pair", "fixed_context", "agent"):
            with self.subTest(mode=mode):
                pool = fixtures.FakePool([self.reply([2, 0, 4, 4])])
                service, out = self.fixture.service("outside-" + mode, pool, case)
                result = run_reasoning(mode, case, service, pool, out, {"max_invalid_replies": 0})
                self.assertEqual(result["status"], "error", result)
                self.assertIn("outside common geometric support", result["error"])
                request = pool.requests[-1][1]
                pair_packet = next(json.loads(block["text"])["observation"]
                    for message in request["messages"] for block in message["content"]
                    if block["type"] == "text" and block["text"].startswith('{"observation":')
                    and json.loads(block["text"]).get("operation") == "current_pair")
                support = next(row for row in pair_packet["evidence"] if row.get("method") == "pair_geometric_support")
                self.assertEqual(support["summary"]["common_pixels"], 8)
                self.assertEqual(support["summary"]["source_only_pixels"], 8)
                preview = next(a["path"] for a in support["artifacts"] if a["role"] == "geometric_support_categories")
                pixels = np.asarray(Image.open(preview))
                np.testing.assert_array_equal(pixels[0, 0], support["summary"]["legend_rgb"]["common_support"])
                np.testing.assert_array_equal(pixels[0, 3], support["summary"]["legend_rgb"]["source_only_first_observation"])
                images = [block["path"] for message in request["messages"] for block in message["content"]
                          if block["type"] == "image"]
                self.assertEqual(images[:2], [row["crop_path"] for row in observations])
                self.assertIn(preview, images)
                self.assertEqual(read_json(out / "costs.json")["images"], len(images))
                if mode == "pair":
                    transcript = json.dumps(request["messages"])
                    self.assertNotIn('"operation": "get_maps"', transcript)
                    self.assertNotIn('"operation": "get_history"', transcript)
                    self.assertEqual(len(images), 3)
        self.assertEqual([sha256(row["crop_path"]) for row in observations], original_hashes)

    def test_partial_boxes_reject_definitive_verdict_but_uncertainty_keeps_fraction(self):
        left = np.zeros((4, 4), bool)
        left[:, :2] = True
        case, _ = self.pair(left, np.ones((4, 4), bool))
        service, _ = self.fixture.service(case=case)
        packet = service.base_pair()
        identifiers = [row["evidence_id"] for row in packet["evidence"]]
        value = fixtures.final_reply({"messages": []}, identifiers=identifiers)
        value["localization"] = {"evidence_id": identifiers[0], "coordinate_system": "canonical_crop", "bbox_xyxy": [1, 0, 3, 4]}
        for decision in ("change", "no_change"):
            value["decision"] = decision
            with self.assertRaisesRegex(ValueError, "Partially supported"):
                validate_final(value, case, set(identifiers), service.evidence)
        value["decision"] = "insufficient_data"
        result = validate_final(value, case, set(identifiers), service.evidence)
        self.assertEqual(result["localization"]["support_validation"]["common_fraction"], .5)
        self.assertEqual(result["localization"]["coordinate_system"], "region_crop_pixels")
        self.assertEqual(result["status"], "abstained")
        value["decision"] = "change"
        del value["localization"]["bbox_xyxy"]
        with self.assertRaisesRegex(ValueError, "verifiable bbox"):
            validate_final(value, case, set(identifiers), service.evidence)

    def test_empty_overlap_rejects_change_and_no_change_in_every_mode(self):
        case, _ = self.pair(np.zeros((4, 4), bool), np.ones((4, 4), bool))
        # Deliberately stale comparable flag must not override actual support.
        self.assertEqual(case["comparability"], "comparable")
        for mode in ("pair", "fixed_context", "agent"):
            for decision in ("change", "no_change"):
                with self.subTest(mode=mode, decision=decision):
                    pool = fixtures.FakePool([self.reply(decision=decision)])
                    service, out = self.fixture.service("empty-" + mode + decision, pool, case)
                    result = run_reasoning(mode, case, service, pool, out, {"max_invalid_replies": 0})
                    self.assertEqual(result["decision"], "insufficient_data")
                    self.assertIn("No common geometric support", result["error"])

    def test_native_box_is_transformed_for_support_and_keeps_its_owner(self):
        left = np.zeros((4, 4), bool)
        left[:, :2] = True
        case, rows = self.pair(left, np.ones((4, 4), bool))
        service, _ = self.fixture.service(case=case)
        packet = service.base_pair()
        identifiers = [row["evidence_id"] for row in packet["evidence"]]
        value = fixtures.final_reply({"messages": []}, identifiers=identifiers)
        # source_to_crop translates native (-2,-1); this is exactly the supported half.
        value["localization"] = {"evidence_id": identifiers[0], "coordinate_system": "observation_native",
                                 "bbox_xyxy": [2, 1, 4, 5]}
        result = validate_final(value, case, set(identifiers), service.evidence)
        self.assertEqual(result["localization"]["coordinate_system"], "observation_native")
        self.assertEqual(result["localization"]["observation_id"], rows[0]["observation_id"])
        self.assertEqual(result["localization"]["support_validation"]["envelope_xyxy"], [0, 0, 2, 4])
        value["localization"]["observation_id"] = rows[1]["observation_id"]
        with self.assertRaisesRegex(ValueError, "owner differs"):
            validate_final(value, case, set(identifiers), service.evidence)
        del value["localization"]["observation_id"]
        value["localization"]["bbox_xyxy"] = [4, 1, 6, 5]
        with self.assertRaisesRegex(ValueError, "outside common"):
            validate_final(value, case, set(identifiers), service.evidence)

    def test_negative_anchor_coordinates_use_actual_region_transform(self):
        case, _ = self.pair(np.ones((4, 4), bool), np.ones((4, 4), bool), anchor_bbox=[-3, -2, 1, 2])
        service, _ = self.fixture.service(case=case)
        packet = service.base_pair()
        identifiers = [row["evidence_id"] for row in packet["evidence"]]
        value = fixtures.final_reply({"messages": []}, identifiers=identifiers)
        value["localization"] = {"evidence_id": identifiers[0], "coordinate_system": "anchor_native",
                                 "bbox_xyxy": [-3, -2, -1, 1]}
        result = validate_final(value, case, set(identifiers), service.evidence)
        self.assertEqual(result["localization"]["anchor_sha256"], "b" * 64)
        self.assertEqual(result["localization"]["support_validation"]["envelope_xyxy"], [0, 0, 2, 3])

    def test_support_image_counts_towards_budget_and_support_mutation_is_rejected(self):
        case, _ = self.pair(np.ones((4, 4), bool), np.ones((4, 4), bool))
        pool = fixtures.FakePool()
        service, out = self.fixture.service(pool=pool, case=case)
        result = run_reasoning("pair", case, service, pool, out, {"max_images": 2})
        self.assertEqual(result["status"], "abstained")
        self.assertIn("image_budget_exhausted", result["reason"])
        self.assertEqual(pool.requests, [])
        identifiers = list(service.evidence)
        value = fixtures.final_reply({"messages": []}, identifiers=identifiers[:2])
        support = next(row for row in service.evidence.values() if row.get("method") == "pair_geometric_support")
        path = next(row["path"] for row in support["artifacts"] if row["role"] == "common_support")
        np.save(path, np.zeros((4, 4), bool))
        with self.assertRaisesRegex(ValueError, "support is missing or changed"):
            validate_final(value, case, set(identifiers), service.evidence)

    def test_new_support_evidence_preserves_old_immutable_rgb_only_records(self):
        case, rows = self.pair(np.ones((4, 4), bool), np.ones((4, 4), bool))
        legacy_id = "observation-" + digest([rows[0]["observation_id"], "crop"])[:32]
        legacy = self.fixture.store.add_evidence({"evidence_id": legacy_id, "kind": "observation",
            "region_id": case["region_id"], "observation_ids": [rows[0]["observation_id"]],
            "available_at": "2011", "coordinate_system": "canonical_crop",
            "artifacts": [{"path": rows[0]["crop_path"], "sha256": rows[0]["crop_sha256"],
                           "role": "rgb", "media_type": "image/png"}]})
        service, _ = self.fixture.service(case=case)
        packet = service.base_pair()
        self.assertNotIn(legacy_id, [row["evidence_id"] for row in packet["evidence"]])
        self.assertEqual(self.fixture.store.get_evidence(legacy_id), legacy)
        new = packet["evidence"][0]
        self.assertEqual(new["coordinate_system"], "region_crop_pixels")
        self.assertEqual({row["role"] for row in new["artifacts"]}, {"rgb", "geometric_support"})


if __name__ == "__main__":
    unittest.main()
