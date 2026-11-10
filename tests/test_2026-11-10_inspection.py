"""2026-11-10 bounded inspection acceptance; procedural RGB and explicit model stubs."""
import contextlib
import copy
import importlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from facade_change.agentic.config import load_config
from facade_change.agentic.store import RegionStore, PrivateEvidenceError
from facade_change.history_features import digest
from facade_change.io import read_json, sha256, write_json


data = importlib.import_module("facade_change.agentic.2026-11-10_inspection_data")
policy = importlib.import_module("facade_change.agentic.2026-11-10_inspection_policy")
app = importlib.import_module("facade_change.agentic.2026-11-10_inspection")
demo = importlib.import_module("facade_change.agentic.2026-11-10_inspection_demo")


class InspectionAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory()
        cls.root = Path(cls.workspace.name)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.summary = demo.demo(cls.root / "demo")
        cls.base = cls.root / "demo"
        cls.questions = cls.base / "export/questions.json"
        cls.answers = cls.base / "export/answers.private.json"
        cls.public = read_json(cls.questions)
        cls.protocol = read_json(cls.base / "protocol.json")
        cls.config = load_config(cls.base / "synthetic-config.json")
        cls.question = next(q for q in cls.public["questions"] if q["building_id"] == "b1" and q["frame"]["anchor_bbox_xyxy"][0] == 80)
        cls.folder = cls.base / "analysis/cases" / cls.question["question_id"]
        cls.case = read_json(cls.folder / "case.json")
        cls.inventory = read_json(cls.folder / "inventory.json")

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(self.temporary.cleanup)
        self.out = Path(self.temporary.name)
        self.store = RegionStore(self.out / "store")
        self.addCleanup(self.store.close)
        frame = self.inventory["frame"]
        self.store.register_region("b1", "b1-front", frame["anchor_sha256"], frame["anchor_bbox_xyxy"], "val", frame["region_id"])
        for obs in self.inventory["observations"].values():
            self.store.add_observation(obs)

    def service(self, name="service"):
        return data.InspectionTools(self.store, self.case, self.out / name, self.inventory, self.protocol["max_native_crop_pixels"])

    def run_policy(self, name="agent", pool=None, caps=None, suffix="policy"):
        service = self.service(suffix)
        pool = pool or demo.QwenStub(self.config["models"])
        protocol = copy.deepcopy(self.protocol)
        if caps:
            protocol["caps"].update(caps)
        result = policy.run_policy(name, self.question, self.case, service, pool, service.out, protocol)
        return result, service.out, pool

    def resolve(self, public=None, protocol=None, histories=None):
        document = copy.deepcopy(public or self.public)
        path = self.out / "questions.json"
        write_json(path, document)
        contract = copy.deepcopy(protocol or self.protocol)
        contract["questions_sha256"] = sha256(path)
        with contextlib.redirect_stdout(io.StringIO()):
            return data.resolve_inputs(path, histories or self.base / "fixture/histories", self.config, contract, check_resources=False)

    def test_demo_end_to_end_expected_building_counts_and_artifacts(self):
        self.assertEqual(self.summary["status"], "completed")
        for name in data.POLICIES:
            self.assertEqual(self.summary["counters"][name], dict(completed=4, abstained=1, error=0, missing=0))
            wrapper = read_json(self.base / "analysis" / (name + "-predictions.json"))
            self.assertEqual(wrapper["model_manifest_sha256"], sha256(self.questions))
            self.assertEqual(wrapper["ranking_source"]["kind"], "single_box")
            evaluated = read_json(self.base / "evaluation" / name / "evaluation.json")
            one = app._at(evaluated["pooled"])
            ten = next(r for r in evaluated["pooled"]["localization"] if r["k"] == 10 and r["iou_threshold"] == .25)
            self.assertEqual(one["available_predictions"], ten["available_predictions"])
            self.assertEqual(evaluated["protocol"]["protocol"], "supplied_region_qa")
        comparison = read_json(self.base / "evaluation/2026-11-10-comparison.json")
        self.assertAlmostEqual(comparison["primary"]["difference"], 2/3)
        self.assertEqual(comparison["primary"]["positive_event_buildings"], ["b1", "b2"])
        self.assertEqual(comparison["stable_only_buildings"], ["b3"])
        self.assertEqual(comparison["proposal_diagnostics"]["agent"]["recovered_outside_proposals"], 1)
        self.assertTrue((self.base / "analysis/report.html").is_file())
        self.assertTrue((self.base / "evaluation/2026-11-10-comparison.csv").is_file())

    def test_native_crop_original_pixels_offset_polygon_and_round_trip(self):
        service = self.service()
        box = [5, 6, 12, 14]
        result = service.inspect_region(self.inventory["current_ids"], box)
        for record, image in zip(result["evidence"], result["images"]):
            obs = self.inventory["observations"][record["observation_id"]]
            native = np.asarray(Image.open(obs["original_path"]))
            geometry = record["geometry"]
            x0, y0, x1, y1 = geometry["delivered_native_bbox_xyxy"]
            actual = np.asarray(Image.open(image["path"]))
            np.testing.assert_array_equal(actual, native[y0:y1, x0:x1])
            self.assertGreater(actual.shape[1], box[2]-box[0])
            self.assertEqual(geometry["native_offset_xy"], [x0, y0])
            point = np.array([2., 3., 1.])
            direct = np.array(geometry["crop_local_to_region"]) @ point
            round_trip = np.array(obs["geometry"]["crop_to_source"]) @ direct
            native_point = np.array(geometry["crop_local_to_observation_native"]) @ point
            np.testing.assert_allclose(round_trip/round_trip[2], native_point, atol=1e-8)
            self.assertGreater(geometry["context_pixels"], 0)
            self.assertEqual(record["coordinate_system"], "inspection_crop_pixels")
            self.assertTrue((Path(image["path"]).parent / "alpha.png").is_file())

    def test_invalid_boxes_wrong_ids_singular_and_excessive_native_crop(self):
        service = self.service()
        for box in ([0, 0, 0, 2], [-1, 0, 2, 2], [0, 0, 33, 32], [1., 1, 2, 2], [True, 1, 2, 2]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                service.preview_action(self.inventory["current_ids"], box)
        for ids in (["unknown"], self.inventory["current_ids"][:1], self.inventory["current_ids"][::-1],
                    [self.public["questions"][-1]["observations"][0]["observation_id"]]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                service.preview_action(ids, [1, 1, 2, 2])
        obs = copy.deepcopy(next(iter(self.inventory["observations"].values())))
        obs["geometry"]["crop_to_source"] = np.zeros((3, 3)).tolist()
        with self.assertRaisesRegex(ValueError, "nonsingular"):
            data.crop_geometry(obs, [0, 0, 4, 4], 10000)
        service.maximum = 1
        with self.assertRaisesRegex(ValueError, "max_native_crop_pixels"):
            service.preview_action(self.inventory["current_ids"], [0, 0, 32, 32])

    def test_changed_original_hash_fails_and_is_not_loaded(self):
        service = self.service()
        path = Path(self.inventory["observations"][self.inventory["current_ids"][0]]["original_path"])
        saved = path.read_bytes()
        try:
            path.write_bytes(saved + b"changed")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                service.preview_action(self.inventory["current_ids"], [0, 0, 2, 2])
        finally:
            path.write_bytes(saved)

    def test_catalogue_two_choices_no_rgb_until_selected_and_store_pollution_isolated(self):
        service = self.service()
        catalogue = service.catalogue()
        self.assertEqual(len(catalogue), 2)
        self.assertNotIn("path", json.dumps(catalogue))
        self.assertFalse((service.out / "crops").exists())
        old = copy.deepcopy(self.inventory["observations"][self.inventory["history_ids"][0]])
        old["observation_id"] = "unlisted-prior-case"
        self.store.add_observation(old)
        self.store.add_evidence(dict(kind="agent_hypothesis", region_id=self.case["region_id"],
            observation_ids=self.inventory["history_ids"][:1], available_at="2026-11-10", summary="prior policy secret", artifacts=[]))
        self.assertEqual(service.catalogue(), catalogue)
        with self.assertRaisesRegex(ValueError, "frozen RGB inventory"):
            service._authorize("unlisted-prior-case")
        pair = service.base_pair()
        self.assertEqual(len(pair["images"]), 3)
        text = json.dumps(pair)
        for oid in self.inventory["history_ids"]:
            self.assertNotIn(oid, text)
        result = service.inspect_region([catalogue[-1]["observation_id"]], [1, 1, 3, 3])
        self.assertEqual(len(result["images"]), 1)
        for forbidden in ("sam3", "get_history", "open_crop", "get_maps"):
            with self.assertRaises(ValueError):
                service.execute(forbidden, {})

    def test_availability_after_source_before_cutoff_and_no_future_history(self):
        resolved = self.resolve()
        self.assertEqual(resolved["status"], "eligible")
        self.assertTrue(all(r["eligible_history_count"] == 2 for r in resolved["eligibility"]))
        self.assertEqual(self.case["cutoff"], "2026-11-10")
        self.assertEqual(self.case["observation_cutoff"], "2015")
        public = copy.deepcopy(self.public)
        public["questions"][0]["knowledge_cutoff"] = "2020"
        resolved = self.resolve(public)
        self.assertEqual(resolved["status"], "ineligible")
        self.assertEqual(resolved["eligibility"][0]["question_id"], public["questions"][0]["question_id"])
        self.assertIn("unavailable", resolved["eligibility"][0]["reason"])
        # Compared 2005/2010 has only one earlier date; 2015 never becomes history.
        original = read_json(self.base / "candidates/candidates.json")
        earlier = next(c for c in original["candidates"] if c["region_id"] == self.case["region_id"] and c["task_dates"] == ["2005", "2010"])
        qa = importlib.import_module("facade_change.agentic.2026-11-10_questions")
        public["questions"] = [{key: earlier[key] for key in qa.PUBLIC_KEYS}]
        resolved = self.resolve(public)
        self.assertEqual(resolved["eligibility"][0]["eligible_history_count"], 1)
        self.assertIn("not_strictly_before_reference", json.dumps(resolved["eligibility"][0]["excluded"]))

    def test_public_join_wrong_owner_hash_date_and_split_are_ineligible(self):
        for key, value in (("region_id", "unknown"), ("split", "test"), ("split", "unassigned"), ("view_id", "other")):
            public = copy.deepcopy(self.public)
            public["questions"][0][key] = value
            with self.subTest(key=key, value=value):
                self.assertEqual(self.resolve(public)["status"], "ineligible")
        for key, value in (("rgb_sha256", "0"*64), ("date", "2008"), ("available_at", "2026-11-09")):
            public = copy.deepcopy(self.public)
            public["questions"][0]["observations"][0][key] = value
            with self.subTest(key=key):
                self.assertEqual(self.resolve(public)["status"], "ineligible")

    def test_private_fields_never_enter_inference(self):
        public = copy.deepcopy(self.public)
        public["questions"][0]["private_notes"] = "private marker"
        with self.assertRaises(PrivateEvidenceError):
            self.resolve(public)
        for request in demo.QwenStub.requests:
            text = json.dumps(request)
            self.assertNotIn("Explicit synthetic acceptance label", text)
            self.assertNotIn("synthetic_fixture_not_human", text)
            self.assertNotIn("generation_reasons", text)

    def test_components_area_order_ties_tiny_loss_empty_failed_distinct(self):
        mask = np.zeros((20, 20), bool)
        mask[10:14, 10:14] = True
        mask[0, 0:2] = True
        mask[4, 0:2] = True
        mask[19, 0] = True
        selected, _ = data.proposals(mask, np.ones_like(mask))
        self.assertEqual([c["area_pixels"] for c in selected["selected"]], [16, 2, 2])
        self.assertEqual([c["bbox_xyxy"][1] for c in selected["selected"]], [10, 0, 4])
        self.assertEqual(selected["component_count"], 4)
        self.assertEqual(selected["omitted_pixels"], 1)
        empty, _ = data.proposals(np.zeros_like(mask), np.ones_like(mask))
        self.assertEqual(empty["component_count"], 0)
        self.assertFalse(empty["fallback"]["is_anychange_detection"])
        with self.assertRaises((ValueError, AttributeError)):
            data.proposals(None, np.ones_like(mask))

    def test_heuristic_exact_order_agent_different_date_then_stop_fresh_judge(self):
        c, cpath, _ = self.run_policy("heuristic", suffix="c")
        d, dpath, _ = self.run_policy("agent", suffix="d")
        cs, ds = read_json(cpath / "state.json"), read_json(dpath / "state.json")
        self.assertEqual(cs["actions"][0]["selected_ids"], self.inventory["current_ids"])
        self.assertEqual(cs["actions"][1]["selected_ids"], self.inventory["history_ids"][:1])
        self.assertEqual(ds["actions"][0]["selected_ids"], self.inventory["history_ids"][-1:])
        self.assertEqual(ds["selection_termination"], "planner_stop")
        self.assertNotEqual(cs["actions"][0]["requested_bbox_xyxy"], ds["actions"][0]["requested_bbox_xyxy"])
        self.assertEqual(c["inventory_sha256"], d["inventory_sha256"])
        self.assertEqual(cs["contract"]["caps"], ds["contract"]["caps"])
        self.assertEqual(c["judge_prompt_sha256"], d["judge_prompt_sha256"])
        for path in (cpath, dpath):
            judge = read_json(path / "final_judge_input.json")["messages"]
            self.assertEqual([m["role"] for m in judge], ["system", "user"])
            self.assertEqual(judge[0]["content"][0]["text"], policy.JUDGE_PROMPT)
            self.assertNotIn('"type": "stop"', json.dumps(judge))
            self.assertNotIn("catalogue", json.dumps(judge))
            self.assertNotIn("prior policy secret", json.dumps(judge))

    def test_action_cap_goes_straight_to_judge_no_extra_stop_turn(self):
        inventory = self.inventory
        class TwoActions(demo.QwenStub):
            def reply(self, payload):
                if payload["messages"][0]["content"][0]["text"].startswith("Select visual evidence"):
                    previous = sum(m["role"] == "assistant" for m in payload["messages"])
                    ids = inventory["current_ids"] if previous == 0 else inventory["history_ids"][-1:]
                    return dict(type="tool", name="inspect_region", arguments=dict(observation_ids=ids, bbox_xyxy=[5, 6, 12, 14]))
                return super().reply(payload)
        result, path, _ = self.run_policy(pool=TwoActions(self.config["models"]))
        self.assertEqual(result["status"], "completed")
        self.assertIn("direct_fresh_judge", result["selection_termination"])
        costs = read_json(path / "costs.json")
        self.assertEqual(costs["actions"], 2)
        self.assertEqual(costs["model_requests"], 3)

    def test_cached_calls_and_repeated_images_count_toward_all_stages(self):
        class Cached(demo.QwenStub):
            def call(self, role, payload):
                response = super().call(role, payload)
                response["source_usage"] = response["usage"]
                response["cache_hit"] = True
                response["usage"] = dict(input_tokens=0, output_tokens=0, images=0, model_calls=0, cache_hits=1)
                return response
        result, path, _ = self.run_policy(pool=Cached(self.config["models"]))
        self.assertEqual(result["status"], "completed")
        costs = read_json(path / "costs.json")
        self.assertEqual(costs["model_requests"], 3)
        self.assertEqual(costs["images"], 4+5+4)
        self.assertEqual(costs["input_tokens"], 600+700+600)
        self.assertEqual(costs["cache_hits"], 3)
        self.assertEqual(costs["model_calls"], 0)
        self.assertIsNone(costs["cold_loading_seconds"])

    def test_final_judge_reservation_budget_termination_distinct_from_abstention(self):
        result, path, _ = self.run_policy(caps={"max_model_calls": 2})
        self.assertEqual(result["status"], "completed")
        self.assertIn("budget_limited", result["selection_termination"])
        self.assertEqual(read_json(path / "costs.json")["model_requests"], 2)
        result, path, _ = self.run_policy(caps={"max_total_images": 1}, suffix="too-small")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["reason"], "budget_limited")

    def test_retries_malformed_json_and_planner_prose_are_errors(self):
        class Invalid(demo.QwenStub):
            def call(self, role, payload):
                response = super().call(role, payload)
                response["text"] = '{"type":'
                response["generation"]["reached_max_new_tokens"] = True
                return response
        result, path, _ = self.run_policy(pool=Invalid(self.config["models"]))
        self.assertEqual(result["status"], "error")
        self.assertIn("invalid_reply_limit", result["error"])
        self.assertEqual(read_json(path / "costs.json")["model_requests"], 2)
        self.assertTrue(read_json(path / "response-1.json")["generation"]["reached_max_new_tokens"])
        self.assertFalse((path / "final_judge_input.json").exists())
        result, path, _ = self.run_policy(pool=Invalid(self.config["models"]), caps={"max_model_calls": 2}, suffix="retry-reservation")
        self.assertEqual(result["status"], "error")
        self.assertIn("invalid_reply_retry_budget_exhausted", result["error"])
        class Prose(demo.QwenStub):
            def reply(self, payload):
                return dict(type="stop", verdict="change", reasoning="do not forward this")
        result, path, _ = self.run_policy(pool=Prose(self.config["models"]), suffix="prose")
        self.assertEqual(result["status"], "error")

    def test_configuration_conflicts_strict_protocol_and_no_answers_run_flag(self):
        config = copy.deepcopy(self.config)
        config["models"]["qwen"]["max_input_tokens"] = 4096
        config["method_options"]["anychange"].pop("checkpoint_path")
        with self.assertRaises(ValueError) as caught:
            data.validate_config(config, self.protocol, check_resources=False)
        self.assertIn("4096", str(caught.exception))
        self.assertIn("16384", str(caught.exception))
        self.assertIn("checkpoint_path", str(caught.exception))
        protocol = copy.deepcopy(self.protocol)
        protocol["enable_sam3"] = True
        with self.assertRaises(ValueError):
            data.validate_protocol(protocol)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            app.parser().parse_args(["run", "--questions", "q", "--histories", "h", "--config", "c", "--protocol", "p", "--out", "o", "--answers", "private"])

    def test_missing_error_predictions_keep_primary_macro_membership(self):
        path = self.folder / "policies/agent/result.json"
        saved = path.read_bytes()
        path.unlink()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                app.evaluate(self.questions, self.answers, self.base / "analysis", self.out / "missing-evaluation")
        finally:
            path.write_bytes(saved)
        result = read_json(self.out / "missing-evaluation/2026-11-10-comparison.json")
        self.assertEqual(result["primary"]["positive_event_buildings"], ["b1", "b2"])
        self.assertEqual(result["primary"]["agent"], .5)
        self.assertEqual(result["coverage"]["agent"]["missing"], 1)
        path.write_text("{}", encoding="utf-8")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                app.evaluate(self.questions, self.answers, self.base / "analysis", self.out / "error-evaluation")
        finally:
            path.write_bytes(saved)
        result = read_json(self.out / "error-evaluation/2026-11-10-comparison.json")
        self.assertEqual(result["coverage"]["agent"]["technical_errors"], 1)
        self.assertEqual(result["primary"]["agent"], .5)

    def test_protocol_mutation_is_fatal_but_model_failure_is_per_case(self):
        path = self.base / "analysis/protocol.json"
        saved = path.read_bytes()
        path.write_bytes(saved + b" ")
        try:
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                app.evaluate(self.questions, self.answers, self.base / "analysis", self.out / "mutated")
        finally:
            path.write_bytes(saved)
        class Failed(demo.QwenStub):
            def call(self, role, payload):
                raise RuntimeError("Injected worker runtime failure")
        result, path, _ = self.run_policy(pool=Failed(self.config["models"]))
        self.assertEqual(result["status"], "error")
        self.assertIn("runtime failure", result["error"])
        self.assertTrue(read_json(path / "costs.json")["incomplete"])
        self.assertIsNone(read_json(path / "costs.json")["input_tokens"])

    def test_preflight_records_ids_without_loading_models_or_filtering(self):
        check = data.validate_config
        def no_resource_check(config, protocol, **kwargs):
            return check(config, protocol, check_resources=False)
        with patch.object(data, "validate_config", side_effect=no_resource_check), patch.object(app, "ModelPool", side_effect=AssertionError("must not load")), contextlib.redirect_stdout(io.StringIO()):
            result = app.preflight(self.questions, self.base / "fixture/histories", self.base / "synthetic-config.json",
                                   self.base / "protocol.json", self.out / "preflight")
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["model_loading"])
        self.assertEqual(len(read_json(self.out / "preflight/rgb_inventory.json")), 5)
        with self.assertRaises((FileExistsError, ValueError)):
            app.preflight(self.questions, self.base / "fixture/histories", self.base / "synthetic-config.json",
                          self.base / "protocol.json", self.out / "preflight")

    def test_detector_failure_is_not_empty_fallback(self):
        resolved = self.resolve()["resolved"][0]
        config = copy.deepcopy(self.config)
        config.update(cache_root=str(self.out / "detector-cache"), diagnostic_baseline=None, fusion=None)
        class Broken(demo.DetectorStub):
            failed = True
        with patch("facade_change.agentic.pipeline._model_signature", side_effect=demo.stub_signature):
            with self.assertRaisesRegex(ValueError, "preparation failed"):
                app.prepare_inventory(resolved, config, self.out / "failure", self.store, Broken)
        self.assertFalse((self.out / "failure/proposals.json").exists())
        class Empty(demo.DetectorStub):
            empty = True
        with patch("facade_change.agentic.pipeline._model_signature", side_effect=demo.stub_signature):
            case, inventory = app.prepare_inventory(resolved, config, self.out / "empty", self.store, Empty)
        self.assertEqual(inventory["proposals"]["component_count"], 0)
        self.assertFalse(inventory["proposals"]["fallback"]["is_anychange_detection"])

    def test_degenerate_final_box_is_rejected_not_repaired(self):
        class Degenerate(demo.QwenStub):
            def reply(self, payload):
                value = super().reply(payload)
                if value.get("type") == "final":
                    value["localization"]["bbox_xyxy"] = [1, 1, 1, 2]
                return value
        result, path, _ = self.run_policy("heuristic", pool=Degenerate(self.config["models"]))
        self.assertEqual(result["status"], "error")
        self.assertIn("strictly positive", result["error"])

    def test_oom_attempts_and_processing_are_retained_and_measured_overflow_errors(self):
        class Retried(demo.QwenStub):
            def call(self, role, payload):
                response = super().call(role, payload)
                initial = copy.deepcopy(response["usage"])
                response["usage"].update(images=initial["images"]*2, input_tokens=initial["input_tokens"]*2,
                    model_calls=2, retries=1, attempts=[dict(status="error", usage=initial), dict(status="ok", usage=initial)])
                return response
        result, path, _ = self.run_policy("heuristic", pool=Retried(self.config["models"]))
        self.assertEqual(result["status"], "completed")
        costs = read_json(path / "costs.json")
        self.assertEqual(costs["images"], 12)
        self.assertEqual(costs["model_requests"], 1)
        self.assertEqual(costs["model_calls"], 2)
        self.assertEqual(costs["retries"], 1)
        self.assertIn("image_grid_thw", costs["calls"][0]["processing"])
        class Overflow(demo.QwenStub):
            def call(self, role, payload):
                response = super().call(role, payload)
                response["usage"]["input_tokens"] = 65537
                return response
        result, path, _ = self.run_policy("heuristic", pool=Overflow(self.config["models"]), suffix="overflow")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["reason"], "budget_limited")

    def test_native_method_artifacts_and_frozen_geometry_cannot_change(self):
        inventory = copy.deepcopy(self.inventory)
        inventory["proposal_preview_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "preview changed"):
            data.InspectionTools(self.store, self.case, self.out / "bad-preview", inventory, 1048576)
        service = self.service()
        oid = self.inventory["current_ids"][0]
        service.inventory["observations"][oid]["geometry"]["source_to_crop"][0][2] += 1
        with self.assertRaisesRegex(ValueError, "geometry changed"):
            service.preview_action(self.inventory["current_ids"], [0, 0, 2, 2])

    def test_action_image_limit_prevents_opening_unpresentable_evidence(self):
        result, path, _ = self.run_policy("heuristic", caps={"max_images": 3})
        self.assertEqual(result["status"], "completed")
        self.assertIn("budget_limited", result["selection_termination"])
        self.assertEqual(read_json(path / "costs.json")["actions"], 0)
        self.assertFalse((path / "crops").exists())

    def test_elapsed_limit_includes_worker_shutdown(self):
        real_clock = time.monotonic
        delay = [0.]
        class SlowClose(demo.QwenStub):
            def close(self):
                delay[0] += 1801.
        with patch("facade_change.agentic.pipeline._model_signature", side_effect=demo.stub_signature), \
                patch.object(app.time, "monotonic", side_effect=lambda: real_clock()+delay[0]), \
                contextlib.redirect_stdout(io.StringIO()):
            result = app.run(self.questions, self.base / "fixture/histories", self.base / "synthetic-config.json",
                             self.base / "protocol.json", self.out / "slow-close", pool_factory=SlowClose,
                             detector_factory=demo.DetectorStub, check_resources=False)
        self.assertEqual(result["status"], "completed_with_errors")
        self.assertTrue(all(r["status"] == "error" for r in result["outcomes"]))
        self.assertTrue(all("worker_lifecycle" in r["error"] for r in result["outcomes"]))


if __name__ == "__main__":
    unittest.main()
