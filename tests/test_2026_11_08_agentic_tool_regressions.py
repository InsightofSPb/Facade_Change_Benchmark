"""Public-tool regressions using the real store/search and only fake heavy inference."""
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tests import test_2026_11_08_agentic_reasoning as fixtures


SPACE = {"model": "fixture-visual", "revision": "test-only", "representation": "visual",
         "preprocessing": {"purpose": "explicit contract fixture, not a real encoder"}}


class PublicToolRegressions(unittest.TestCase):
    setUp = fixtures.ToolReasoningTests.setUp
    observe = fixtures.ToolReasoningTests.observe
    service = fixtures.ToolReasoningTests.service

    def analogue(self):
        region = self.store.register_region("analogue-building", "analogue-view", "b" * 64,
                                            [0, 0, 4, 4], "train")
        original, crop, support = (self.root / ("analogue-2015-" + name)
                                   for name in ("original.png", "crop.png", "support.npy"))
        Image.fromarray(np.full((8, 8, 3), 60, np.uint8)).save(original)
        Image.fromarray(np.full((4, 4, 3), 60, np.uint8)).save(crop)
        np.save(support, np.ones((4, 4), bool))
        transform = [[1, 0, -3], [0, 1, -2], [0, 0, 1]]
        observation = self.store.add_observation({"observation_id": "analogue-2015",
            "region_id": region["region_id"], "date": "2015", "date_status": "confirmed", "available_at": "2015",
            "original_path": str(original), "crop_path": str(crop), "support_path": str(support),
            "geometry": {"source_to_crop": transform, "crop_to_source": np.linalg.inv(transform).tolist(),
                         "native_shape": [8, 8], "crop_shape": [4, 4], "bbox_xyxy": [0, 0, 4, 4],
                         "coordinate_system": "native_pixel_centres_xy"}, "correspondence_status": "aligned"})
        self.index.add_embedding(self.source["observation_id"], [1., 0.], SPACE)
        self.index.add_embedding(observation["observation_id"], [1., .1], SPACE)
        return region, observation

    def test_retrieved_analogue_can_be_opened_segmented_and_persisted_in_its_own_region(self):
        region, observation = self.analogue()
        service, _ = self.service()
        found = service.execute("search_similar", {"observation_id": "source"})
        self.assertEqual(found["status"], "ok", found)
        self.assertEqual([row["observation_id"] for row in found["matches"]], [observation["observation_id"]])
        opened = service.execute("open_crop", {"observation_id": observation["observation_id"]})
        self.assertEqual(opened["status"], "ok", opened)
        self.assertEqual(opened["images"][0]["path"], observation["crop_path"])
        result = service.execute("sam3", {"observation_id": observation["observation_id"], "text": "crack"})
        self.assertEqual(result["status"], "ok", result)
        record = result["evidence"][0]
        self.assertEqual(record["region_id"], region["region_id"])
        self.assertNotEqual(record["region_id"], self.case["region_id"])
        self.assertEqual(record["observation_ids"], [observation["observation_id"]])
        self.assertEqual(record["coordinate_system"], "region_crop_pixels")
        self.assertEqual(record["observation_id"], observation["observation_id"])
        self.assertEqual(record["available_at"], self.case["cutoff"])
        self.assertEqual(record["input_observation"]["available_at"], "2015")
        self.assertEqual(record["input_observation"]["crop_sha256"], observation["crop_sha256"])
        frames = record["artifact_coordinates"]
        self.assertEqual(frames["overlay"], {"coordinate_system": "region_crop_pixels", "region_id": region["region_id"]})
        native_frame = {"coordinate_system": "observation_native", "observation_id": observation["observation_id"]}
        self.assertEqual(frames["mapped_masks"]["union_observation"], native_frame)
        mapped = next(item["path"] for item in record["artifacts"] if item["role"] == "mapped_masks")
        with np.load(mapped, allow_pickle=False) as arrays:
            expected = np.zeros((8, 8), bool)
            expected[3:5, 4:6] = True
            np.testing.assert_array_equal(arrays["union_observation"], expected)
        self.assertEqual(self.pool.requests[-1][1]["image_path"], observation["crop_path"])
        persisted = self.store.get_evidence(record["evidence_id"])
        self.assertEqual(persisted["region_id"], region["region_id"])
        self.assertEqual(persisted["parameters"]["artifact_coordinates"], frames)
        self.assertEqual(persisted["parameters"]["geometry"], record["geometry"])
        with self.assertRaisesRegex(ValueError, "another physical region"):
            self.store.add_evidence({"region_id": self.case["region_id"], "kind": "prediction",
                "observation_ids": [observation["observation_id"]], "available_at": "2020", "artifacts": []})

    def test_visual_tool_ignores_unavailable_excluded_rgb_but_checks_used_rgb(self):
        region, used = self.analogue()
        service, _ = self.service()
        query = {"observation_id": "source"}
        before = service.execute("search_similar", query)
        self.assertEqual(before["status"], "ok", before)
        future = self.observe("analogue-2025", "2025", region, 70)
        wrong_space = self.observe("analogue-other-space", "2016", region, 90)
        no_embedding = self.observe("analogue-no-embedding", "2017", region, 100)
        self.index.add_embedding(wrong_space["observation_id"], [1., 0.], {**SPACE, "revision": "other-space"})
        excluded_region = self.store.register_region("excluded-building", "excluded-view", "c" * 64,
                                                     [0, 0, 4, 4], "val")
        excluded = self.observe("excluded-2015", "2015", excluded_region, 80)
        for row in (future, excluded):
            self.index.add_embedding(row["observation_id"], [1., 0.], SPACE)
        for row in (future, excluded, wrong_space, no_embedding):
            Path(row["crop_path"]).unlink()
        after = service.execute("search_similar", query)
        self.assertEqual(after["status"], "ok", after)
        for key in ("matches", "evidence", "corpus_version", "cache_key"):
            self.assertEqual(after[key], before[key], key)
        Image.fromarray(np.full((4, 4, 3), 99, np.uint8)).save(used["crop_path"])
        damaged = service.execute("search_similar", query)
        self.assertEqual(damaged["status"], "error", damaged)
        self.assertIn("hash", damaged["error"].lower())


if __name__ == "__main__":
    unittest.main()
