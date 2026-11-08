"""CPU contract tests; injected responses do not validate actual Qwen/SAM3."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from facade_change.agentic.reasoning import run_reasoning, validate_final
from facade_change.agentic.retrieval import KnowledgeBase, SearchIndex
from facade_change.agentic.store import RegionStore
from facade_change.agentic.tools import ToolService, remap_mask
from facade_change.io import read_json, sha256


def final_reply(request, decision="change", identifiers=None):
    evidence = [block["evidence_id"] for message in request["messages"]
                for block in message["content"] if block["type"] == "image"]
    return {"type": "final", "decision": decision, "visible_difference": "CPU contract fixture",
            "semantic_interpretation": None, "localization": None,
            "evidence_ids": list(dict.fromkeys(evidence)) if identifiers is None else identifiers,
            "limitations": ["Injected test model; no actual Qwen quality claim"], "missing_information": []}


class FakePool:
    """Deliberate test double, never offered as a production backend."""
    models = {"qwen": {"revision": "test-double"}, "sam3": {"revision": "test-double"}}

    def __init__(self, replies=()):
        self.replies = list(replies)
        self.requests = []

    def call(self, role, payload):
        self.requests.append((role, payload))
        if role == "sam3":
            out = Path(payload["out"])
            out.mkdir(parents=True, exist_ok=True)
            with Image.open(payload["image_path"]) as image:
                shape = (image.height, image.width)
            masks = np.zeros((1, *shape), bool)
            masks[0, 1:3, 1:3] = True
            np.save(out / "masks.npy", masks)
            np.save(out / "probabilities.npy", masks.astype(np.float32))
            return {"status": "ok", "masks_path": str(out / "masks.npy"),
                    "probabilities_path": str(out / "probabilities.npy"),
                    "scores": [.9], "boxes": [[1, 1, 3, 3]], "native_shape": list(shape),
                    "provenance": {"model": "sam3", "revision": "test-double"},
                    "usage": {"images": 1, "model_loads": 1, "model_calls": 1}}
        value = self.replies.pop(0) if self.replies else final_reply
        if isinstance(value, BaseException):
            raise value
        value = value(payload) if callable(value) else value
        return {"status": "ok", "text": json.dumps(value),
                "usage": {"input_tokens": 30, "output_tokens": 20,
                          "images": sum(block["type"] == "image" for message in payload["messages"]
                                        for block in message["content"]), "model_loads": 1, "model_calls": 1},
                "provenance": {"model": "injected-test-double"}}


class ToolReasoningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = RegionStore(self.root / "store")
        self.addCleanup(self.store.close)
        self.pool = FakePool()
        self.index = SearchIndex(self.store)
        self.knowledge = KnowledgeBase(self.root / "knowledge")
        self.region = self.store.register_region("building", "view", "a" * 64, [0, 0, 4, 4], "test")
        self.old = self.observe("old", "2000", self.region, 5)
        self.reference = self.observe("reference", "2010", self.region, 10)
        self.source = self.observe("source", "2020", self.region, 20)
        self.future = self.observe("future", "2030", self.region, 30)
        self.case = {"case_id": "pair", "region_id": self.region["region_id"], "building_id": "building",
                     "view_id": "view", "split": "test", "cutoff": "2020",
                     "reference_observation_id": "reference", "source_observation_id": "source",
                     "reference_date": "2010", "source_date": "2020", "comparability": "comparable",
                     "limitations": ["Visibility unknown"]}

    def observe(self, identifier, date, region, value):
        original = self.root / (identifier + "-original.png")
        crop = self.root / (identifier + "-crop.png")
        support = self.root / (identifier + "-support.npy")
        Image.fromarray(np.full((8, 8, 3), value, np.uint8)).save(original)
        Image.fromarray(np.full((4, 4, 3), value, np.uint8)).save(crop)
        np.save(support, np.ones((4, 4), bool))
        transform = [[1, 0, -2], [0, 1, -1], [0, 0, 1]]
        return self.store.add_observation({"observation_id": identifier, "region_id": region["region_id"],
            "date": date, "date_status": "confirmed", "available_at": date,
            "original_path": str(original), "crop_path": str(crop), "support_path": str(support),
            "geometry": {"source_to_crop": transform, "crop_to_source": np.linalg.inv(transform).tolist(),
                         "native_shape": [8, 8], "crop_shape": [4, 4], "bbox_xyxy": [0, 0, 4, 4],
                         "coordinate_system": "native_pixel_centres_xy"},
            "correspondence_status": "aligned", "visibility": "unknown",
            "description": "facade crack", "description_source": "model_prediction"})

    def service(self, name="run", pool=None, case=None):
        path = self.root / name
        return ToolService(self.store, self.index, self.knowledge, pool or self.pool,
                           case or self.case, path), path

    def test_pair_transmits_actual_images_and_drops_unlisted_case_data(self):
        case = {**self.case, "packed": [1], "annotations": "hidden marker", "future": "future secret"}
        service, out = self.service(case=case)
        result = run_reasoning("pair", case, service, self.pool, out)
        self.assertEqual(result["status"], "completed")
        request = self.pool.requests[0][1]
        self.assertEqual(len([b for m in request["messages"] for b in m["content"] if b["type"] == "image"]), 2)
        text = json.dumps(request)
        self.assertNotIn("hidden marker", text)
        self.assertNotIn("future secret", text)
        self.assertEqual(read_json(out / "costs.json")["tool_calls"], 0)
        self.assertTrue((out / "messages.json").is_file())
        self.assertTrue((out / "trace.jsonl").is_file())
        self.assertEqual(self.store.get_evidence(result["hypothesis_evidence_id"])["kind"], "agent_hypothesis")

    def test_history_excludes_current_future_and_unexposed_id_access(self):
        service, _ = self.service()
        response = service.execute("get_history", {})
        self.assertEqual([r["observation_ids"] for r in response["evidence"]], [["old"]])
        self.assertEqual(service.execute("open_crop", {"observation_id": "future"})["status"], "error")
        self.assertEqual(service.execute("open_crop", {"observation_id": "old"})["status"], "ok")
        self.assertEqual(service.execute("open_crop", {"observation_id": "old", "path": "/etc/passwd"})["status"], "error")

    def test_history_preserves_hypothesis_and_confirmation_types(self):
        hypothesis = self.store.add_evidence({"region_id": self.region["region_id"], "observation_ids": ["old"],
            "kind": "agent_hypothesis", "available_at": "2001", "summary": "Possible crack", "artifacts": []})
        confirmed = self.store.add_evidence({"region_id": self.region["region_id"], "observation_ids": ["old"],
            "kind": "human_confirmation", "available_at": "2002", "summary": "Reviewed surface observation",
            "author": "fixture-expert", "source": {"locator": "fixture-review"}, "artifacts": []})
        future = self.store.add_evidence({"region_id": self.region["region_id"], "observation_ids": ["old"],
            "kind": "agent_hypothesis", "available_at": "2030", "summary": "Future secret", "artifacts": []})
        service, _ = self.service()
        result = service.execute("get_history", {})
        evidence = {row["evidence_id"]: row for row in result["evidence"]}
        self.assertEqual(evidence[hypothesis["evidence_id"]]["kind"], "agent_hypothesis")
        self.assertEqual(evidence[confirmed["evidence_id"]]["author"], "fixture-expert")
        self.assertNotIn(future["evidence_id"], evidence)

    def test_knowledge_evidence_preserves_precise_fragment_citation(self):
        self.knowledge.import_document({"document_id": "fixture-handbook",
            "source": {"title": "Fixture handbook", "kind": "user_document", "locator": "fixture://handbook"},
            "available_at": "2000", "concepts": [], "relations": [],
            "fragments": [{"fragment_id": "section-3-2", "text": "Fixture facade crack description",
                           "concept_ids": [], "locator": "p. 42, section 3.2"}]})
        service, _ = self.service()
        result = service.execute("get_knowledge", {"query": "crack"})
        self.assertEqual(result["status"], "ok", result)
        evidence = result["evidence"][0]
        self.assertEqual(evidence["source"], {"title": "Fixture handbook", "kind": "user_document",
            "locator": "fixture://handbook", "document_id": "fixture-handbook",
            "fragment_id": "section-3-2", "fragment_locator": "p. 42, section 3.2"})
        self.assertEqual(self.store.get_evidence(evidence["evidence_id"])["source"], evidence["source"])

    def test_fixed_context_is_deterministic_and_pair_forbids_tools(self):
        service, out = self.service()
        result = run_reasoning("fixed_context", self.case, service, self.pool, out)
        self.assertEqual(result["status"], "completed")
        trace = [json.loads(line) for line in (out / "trace.jsonl").read_text().splitlines()]
        self.assertEqual([r["name"] for r in trace if r["event_type"] == "tool_requested"],
                         ["get_history", "get_maps", "get_knowledge"])
        pool = FakePool([{"type": "tool", "name": "get_history", "arguments": {}}])
        service, out = self.service("pair-no-tools", pool)
        result = run_reasoning("pair", self.case, service, pool, out, {"max_invalid_replies": 0})
        self.assertEqual(result["status"], "error")
        self.assertEqual(read_json(out / "costs.json")["tool_calls"], 0)

    def test_agent_action_budget_and_same_request_limit(self):
        action = {"type": "tool", "name": "get_history", "arguments": {}}
        pool = FakePool([action, action])
        service, out = self.service(pool=pool)
        result = run_reasoning("agent", self.case, service, pool, out)
        self.assertEqual(result["reason"], "same_action_limit")
        self.assertEqual(read_json(out / "costs.json")["tool_calls"], 1)
        pool = FakePool([action])
        service, out = self.service("no-action", pool)
        result = run_reasoning("agent", self.case, service, pool, out, {"max_actions": 0})
        self.assertEqual(result["reason"], "action_budget_exhausted")
        self.assertEqual(read_json(out / "costs.json")["tool_calls"], 0)

    def test_unknown_evidence_and_shell_requests_are_rejected(self):
        for index, reply in enumerate((lambda req: final_reply(req, identifiers=["invented"]),
                                      {"type": "tool", "name": "shell", "arguments": {"command": "anything"}})):
            pool = FakePool([reply])
            service, out = self.service("invalid-" + str(index), pool)
            result = run_reasoning("agent", self.case, service, pool, out, {"max_invalid_replies": 0})
            self.assertEqual(result["status"], "error")
            self.assertEqual(read_json(out / "costs.json")["tool_calls"], 0)

    def test_image_budget_does_not_resize_or_drop_images(self):
        service, out = self.service()
        result = run_reasoning("pair", self.case, service, self.pool, out, {"max_images": 1})
        self.assertEqual(result["status"], "abstained")
        self.assertIn("image_budget_exhausted", result["reason"])
        self.assertEqual(self.pool.requests, [])

    def test_resume_keeps_cpu_images_and_rejects_changed_contract_completed_state(self):
        pool = FakePool([KeyboardInterrupt()])
        service, out = self.service(pool=pool)
        with self.assertRaises(KeyboardInterrupt):
            run_reasoning("pair", self.case, service, pool, out)
        state = read_json(out / "state.json")
        self.assertEqual(state["phase"], "qwen_pending")
        self.assertEqual(len(state["image_sha256"]), 2)
        with self.assertRaises(ValueError):
            run_reasoning("pair", self.case, service, pool, out, {"max_actions": 3}, resume=True)
        result = run_reasoning("pair", self.case, service, pool, out, resume=True)
        self.assertEqual(result["status"], "completed")
        with self.assertRaises(FileExistsError):
            run_reasoning("pair", self.case, service, pool, out, resume=True)

    def test_changed_input_cannot_resume(self):
        pool = FakePool([KeyboardInterrupt()])
        service, out = self.service(pool=pool)
        with self.assertRaises(KeyboardInterrupt):
            run_reasoning("pair", self.case, service, pool, out)
        Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(self.source["crop_path"])
        with self.assertRaises(ValueError):
            run_reasoning("pair", self.case, service, pool, out, resume=True)

    def test_sam_returns_separate_crop_native_masks_and_preserves_input(self):
        service, out = self.service()
        before = sha256(self.source["crop_path"])
        response = service.execute("sam3", {"observation_id": "source", "text": "crack"})
        self.assertEqual(response["status"], "ok", response)
        record = response["evidence"][0]
        mapped = next(a["path"] for a in record["artifacts"] if a["role"] == "mapped_masks")
        with np.load(mapped, allow_pickle=False) as arrays:
            expected = np.zeros((8, 8), bool)
            expected[2:4, 3:5] = True
            np.testing.assert_array_equal(arrays["union_observation"], expected)
            self.assertEqual(arrays["union_crop"].sum(), 4)
        self.assertEqual(sha256(self.source["crop_path"]), before)
        self.assertEqual(record["method"], "sam3_text_refinement")
        self.assertIsNone(record["class_binding"])
        self.assertEqual(self.store.get_evidence(record["evidence_id"])["kind"], "prediction")

    def test_remap_distinguishes_support_from_negative_mask(self):
        mask = np.ones((2, 2), bool)
        support = np.array([[1, 0], [1, 1]], bool)
        native, valid = remap_mask(mask, support, {"crop_to_source": [[1, 0, 1], [0, 1, 2], [0, 0, 1]],
                                                   "native_shape": [5, 5]})
        self.assertEqual(native.sum(), 3)
        self.assertFalse(valid[2, 2])
        with self.assertRaises(ValueError):
            remap_mask(mask, support, {"crop_to_source": np.zeros((3, 3)).tolist(), "native_shape": [5, 5]})

    def test_false_change_on_unresolved_pair_rejected(self):
        case = {**self.case, "comparability": "alignment_rejected"}
        service, out = self.service(case=case)
        result = run_reasoning("pair", case, service, self.pool, out, {"max_invalid_replies": 0})
        self.assertEqual(result["status"], "error")
        self.assertIn("correspondence", result["error"])

    def test_explicit_localization_grid_and_artifact_integrity(self):
        service, _ = self.service()
        pair = service.base_pair()
        ids = [row["evidence_id"] for row in pair["evidence"]]
        value = final_reply({"messages": []}, identifiers=ids)
        value["localization"] = {"evidence_id": ids[0], "coordinate_system": "canonical_crop", "bbox_xyxy": [0, 0, 8, 8]}
        with self.assertRaises(ValueError):
            validate_final(value, self.case, set(ids), service.evidence)
        value["localization"] = None
        Path(self.reference["crop_path"]).unlink()
        with self.assertRaises(ValueError):
            validate_final(value, self.case, set(ids), service.evidence)

    def test_saved_tool_permissions_cannot_be_rebound_to_an_earlier_cutoff(self):
        service, out = self.service()
        service.base_pair()
        service.execute("get_history", {})
        with self.assertRaises(ValueError):
            ToolService(self.store, self.index, self.knowledge, self.pool, {**self.case, "cutoff": "1990"}, out)
        reloaded = ToolService(self.store, self.index, self.knowledge, self.pool, self.case, out)
        self.assertEqual(reloaded.execute("open_crop", {"observation_id": "old"})["status"], "ok")

    def test_joint_text_image_search_executes_encoder_and_filters_library(self):
        region = self.store.register_region("other", "other-view", "b" * 64, [0, 0, 4, 4], "train")
        analogue = self.observe("analogue", "2005", region, 99)
        space = {"model": "jinaai/jina-clip-v2", "revision": "pinned-test-revision",
                 "representation": "joint_image_text", "preprocessing": {"normalize": True}}
        self.index.add_embedding("analogue", [1., 0.], space)
        calls = []
        class JointPool(FakePool):
            def call(self, role, payload):
                calls.append((role, payload))
                return {"status": "ok", "vectors": [[1., 0.]], "space": space,
                        "usage": {"model_calls": 1, "model_loads": 1}}
        pool = JointPool()
        service, _ = self.service(pool=pool)
        result = service.execute("search_similar", {"representation": "joint_image_text", "query": "crack"})
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(calls[0][0], "jina")
        self.assertEqual(calls[0][1]["texts"], ["crack"])
        self.assertEqual(result["matches"][0]["observation_id"], analogue["observation_id"])
        self.assertEqual(result["usage"]["model_calls"], 1)
        self.assertEqual(result["usage"]["model_requests"], 1)
        self.assertEqual(service.execute("open_crop", {"observation_id": "analogue"})["status"], "ok")

    def test_maps_preserve_all_summaries_with_explicit_preview_budget(self):
        rows = []
        for index in range(5):
            rows.append(self.store.add_evidence({"kind": "prediction", "region_id": self.case["region_id"],
                "observation_ids": self.service()[0].current, "available_at": "2020",
                "summary": {"method": "method-" + str(index)},
                "parameters": {"method": "method-" + str(index), "output_kind": "score", "units": "distance"},
                "artifacts": [{"path": self.source["crop_path"], "role": "score_preview", "media_type": "image/png"}]}))
        service, out = self.service()
        service.method_evidence = rows
        result = service.execute("get_maps", {"limit": 2})
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(len(result["evidence"]), 5)
        self.assertEqual(len(result["images"]), 2)
        self.assertEqual(result["total_preview_count"], 5)
        self.assertEqual(result["selected_preview_count"], 2)

    def test_native_calls_cache_and_memory_are_not_confused_with_requests(self):
        class CachedPool(FakePool):
            def call(self, role, payload):
                result = super().call(role, payload)
                result["usage"] = {"model_calls": 0, "images": 0, "input_tokens": 0, "output_tokens": 0,
                    "cache_hits": 1, "memory": {"peak_allocated_bytes": 123, "peak_reserved_bytes": 456}}
                result["source_usage"] = {"input_tokens": 100, "output_tokens": 200, "model_calls": 1}
                return result
        pool = CachedPool()
        service, out = self.service(pool=pool)
        run_reasoning("pair", self.case, service, pool, out)
        costs = read_json(out / "costs.json")
        self.assertEqual(costs["model_requests"], 1)
        self.assertEqual(costs["model_calls"], 0)
        self.assertEqual(costs["cache_hits"], 1)
        self.assertEqual(costs["peak_reserved_bytes"], 456)
        self.assertEqual(costs["input_tokens"], 0)
        self.assertEqual(costs["calls"][0]["source_usage"]["input_tokens"], 100)


if __name__ == "__main__":
    unittest.main()
