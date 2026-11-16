"""CPU acceptance for public-only WWW inference; scripted models are explicit fixtures."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from facade_change.history_features import Fingerprints, digest
from facade_change.io import read_json, sha256, write_json
from facade_change.agentic import models

demo = importlib.import_module("facade_change.agentic.2026-11-15_www_demo")
runner = importlib.import_module("facade_change.agentic.2026-11-15_www_runner")
archive = importlib.import_module("facade_change.agentic.2026-11-15_www_archive")


class WWWRunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.questions, self.package, self.config = demo.fixture(self.root / "fixture")
        archive.import_archive(self.package, self.root / "archive")
        archive.index_archive(self.root / "archive/corpus.json", self.config, self.root / "index",
                              pool_factory=demo.SyntheticPool, synthetic=True)
        self.index = self.root / "index/index.json"

    def run_fixture(self, name="run", **kwargs):
        return runner.run(self.questions, self.config, self.root / name, archive_index=self.index,
                          synthetic=True, **{"pool_factory": demo.SyntheticPool, **kwargs})

    def test_three_tasks_identical_cohort_exposure_and_resume(self):
        self.assertEqual(self.run_fixture()["outcome_counts"], {"completed": 8, "abstained": 1, "error": 0})
        records = read_json(self.root / "run/results.json")["results"]
        self.assertEqual({r["task"] for r in records}, runner.TASKS)
        for condition in runner.CONDITIONS:
            rows = [r for r in records if r["condition"] == condition]
            self.assertEqual({r["question_id"] for r in rows}, {r["question_id"] for r in read_json(self.questions)["questions"]})
            for row in rows:
                folder = self.root / "run/questions" / row["question_id"] / condition
                exposed = read_json(folder / "exposure.json")["evidence"]
                self.assertTrue(set(row["answer"]["citations"]) <= set(exposed))
                self.assertNotIn("DO_NOT_EXPOSE", (folder / "messages.json").read_text())
                self.assertLessEqual(row["costs"]["retrieval_rounds"], 2)
                if condition == "photos_only":
                    self.assertTrue(all(r["origin"] == "local_photo" for r in exposed.values()))
        for question in read_json(self.questions)["questions"]:
            messages = [read_json(self.root / "run/questions" / question["question_id"] / c / "messages.json") for c in runner.CONDITIONS]
            self.assertTrue(all(m[:2] == messages[0][:2] for m in messages))
        self.run_fixture("resume", resume_from=self.root / "run")
        self.run_fixture("resume-again", resume_from=self.root / "resume")
        self.assertTrue(all(r["costs"]["model_calls"] == 0 for r in read_json(self.root / "resume/results.json")["results"]))
        resumed_costs = [read_json(p) for p in (self.root / "resume/questions").glob("*/*/costs.json")]
        self.assertTrue(all(r["model_calls"] == 0 and r["source_costs"]["model_calls"] > 0 for r in resumed_costs))
        contract = read_json(self.root / "run/contract.json")
        self.assertEqual(contract["prompt_revision"], runner.PROMPT_REVISION)
        self.assertEqual(contract["prompt_sha256"], digest(runner.SYSTEM))
        with patch.object(runner, "SYSTEM", runner.SYSTEM + "\nChanged fixture instruction"), self.assertRaisesRegex(ValueError, "Resume .*changed"):
            self.run_fixture("changed-prompt", resume_from=self.root / "run")
        self.assertFalse((self.root / "changed-prompt").exists())
        self.config["budgets"]["max_output_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "Resume .*changed"):
            self.run_fixture("bad-resume", resume_from=self.root / "run")

    def test_agent_can_finish_locally_without_any_retrieval_dispatch(self):
        class LocalOnly(demo.SyntheticPool):
            def call(self, role, payload):
                if role == "jina":
                    raise AssertionError("A scripted local-only final must not dispatch retrieval")
                # Script an answer/abstention choice without retrieval. This
                # does not test whether a real model follows the new instruction.
                scripted = copy.deepcopy(payload)
                control = json.loads(scripted["messages"][-1]["content"][0]["text"])
                control["allowed_action"] = "decide" if control["stage"] == "choose_action" else "final"
                scripted["messages"][-1]["content"][0]["text"] = json.dumps(control)
                response = super().call(role, scripted)
                if control["stage"] == "choose_action":
                    value = json.loads(response["text"])
                    value["action"] = value.pop("type")
                    value["stage"] = "choose_action"
                    value.setdefault("reason", "Synthetic local sufficiency; not factual evidence")
                    response["text"] = json.dumps(value)
                return response
        modes = (runner.CONDITIONS[0], runner.CONDITIONS[2])
        result = self.run_fixture(pool_factory=LocalOnly, conditions=modes)
        self.assertEqual(result["outcome_counts"], {"completed": 4, "abstained": 2, "error": 0})
        for row in read_json(self.root / "run/results.json")["results"]:
            self.assertEqual(row["costs"]["retrieval_rounds"], 0)
            self.assertEqual(row["costs"]["qwen_requests"], 1 if row["condition"] == "photos_only" else 2)
            folder = self.root / "run/questions" / row["question_id"] / row["condition"]
            exposed = read_json(folder / "exposure.json")
            self.assertEqual(exposed["tools"], [])
            self.assertTrue(all(r["origin"] == "local_photo" for r in exposed["evidence"].values()))
        for question in read_json(self.questions)["questions"]:
            messages = [read_json(self.root / "run/questions" / question["question_id"] / c / "messages.json") for c in modes]
            self.assertEqual(messages[0][:2], messages[1][:2])  # Same factual instructions, question and RGB.
            controls = [json.loads(m[2]["content"][0]["text"]) for m in messages]
            self.assertEqual([c["allowed_action"] for c in controls], ["final", "retrieve_or_decide"])
            self.assertEqual([c["remaining_retrieval_rounds"] for c in controls], [0, 2])

    def test_private_or_unknown_fields_cannot_enter_public_manifest(self):
        value = read_json(self.questions)
        value["questions"][0]["reviewed_labels"] = {"assessment": "earliest_presence"}
        write_json(self.questions, value)
        with self.assertRaises(ValueError):
            self.run_fixture()

    def test_later_archive_does_not_replace_local_task_in_any_condition(self):
        public = read_json(self.questions)
        question = public["questions"][1]
        write_json(self.questions, {**public, "questions": [question]})
        package = read_json(self.package)
        record = package["records"][0]
        record["observation_date"].update(start="2024", end="2024")
        record.update(publication_date="2024", available_at="2024")
        write_json(self.package, package)
        archive.import_archive(self.package, self.root / "later-archive")
        archive.index_archive(self.root / "later-archive/corpus.json", self.config, self.root / "later-index",
                              pool_factory=demo.SyntheticPool, synthetic=True)
        self.index = self.root / "later-index/index.json"
        self.assertNotIn("2024", question["task_dates"])
        summary = self.run_fixture()
        self.assertEqual(summary["outcome_counts"], {"completed": 3, "abstained": 0, "error": 0})
        scopes = []
        for row in read_json(self.root / "run/results.json")["results"]:
            folder = self.root / "run/questions" / row["question_id"] / row["condition"]
            calls = [read_json(p) for p in sorted((folder / "calls").glob("*-qwen.json"))]
            controls = [json.loads(c["request"]["messages"][-1]["content"][0]["text"]) for c in calls]
            initial = json.loads(calls[0]["request"]["messages"][1]["content"][0]["text"])
            scope = initial["task_scope"]
            scopes.append(scope)
            self.assertEqual(scope["task"], question["task"])
            self.assertEqual(scope["question"], question["question"])
            self.assertEqual(scope["task_dates"], question["task_dates"])
            self.assertEqual(set(scope["local_observation_ids"]), {o["observation_id"] for o in question["observations"]})
            self.assertTrue(all(c["task_scope"] == scope for c in controls))
            self.assertTrue(all("required_decisions" not in c for c in controls))
            self.assertTrue(set(row["answer"]["citations"]) <= set(scope["local_observation_ids"]))
            if row["condition"] != "photos_only":
                self.assertIn("fixture-archive-b1", controls[-1]["archive_state"]["exposed_archive_evidence_ids"])
                self.assertEqual(row["costs"]["retrieval_rounds"], 1)
        self.assertTrue(all(scope == scopes[0] for scope in scopes))

    def test_local_uncertainty_can_still_abstain_after_retrieval(self):
        reason = "Синтетический пример: локальная область закрыта лесами; сравнение недостаточно"

        class UncertainLocal(demo.SyntheticPool):
            def call(self, role, payload):
                response = super().call(role, payload)
                if role == "qwen":
                    control = json.loads(payload["messages"][-1]["content"][0]["text"])
                    if control["stage"] == "choose_action" and control["archive_state"]["searches_performed"]:
                        response["text"] = json.dumps({"type": "abstain", "reason": reason})
                return response

        public = read_json(self.questions)
        write_json(self.questions, {**public, "questions": [public["questions"][1]]})
        summary = self.run_fixture(pool_factory=UncertainLocal, conditions=("bounded_agentic_retrieval",))
        self.assertEqual(summary["outcome_counts"], {"completed": 0, "abstained": 1, "error": 0})
        row = read_json(self.root / "run/results.json")["results"][0]
        self.assertEqual(row["answer"]["decision"], "insufficient_data")
        self.assertEqual(row["answer"]["limitations"], [reason])
        self.assertEqual(row["costs"]["retrieval_rounds"], 1)

    def test_action_selection_has_no_final_scaffold_and_preserves_archive_state(self):
        self.run_fixture(conditions=("bounded_agentic_retrieval",))
        states = set()
        for row in read_json(self.root / "run/results.json")["results"]:
            folder = self.root / "run/questions" / row["question_id"] / row["condition"]
            calls = [read_json(path) for path in sorted((folder / "calls").glob("*-qwen.json"))]
            controls = [json.loads(call["request"]["messages"][-1]["content"][0]["text"]) for call in calls]
            self.assertEqual([c["stage"] for c in controls], ["choose_action", "choose_action", "answer"])
            self.assertEqual(controls[0]["archive_state"]["search_status"], "not_searched")
            self.assertEqual(controls[0]["archive_state"]["searches_performed"], 0)
            self.assertNotIn("response_contract", calls[0]["request"]["messages"][1]["content"][0]["text"])
            self.assertNotIn("response_contract", controls[0])
            self.assertNotIn("response_contract", controls[1])
            self.assertIn("response_contract", controls[2])
            self.assertEqual(controls[1]["archive_state"], controls[2]["archive_state"])
            states.add(controls[1]["archive_state"]["search_status"])
            self.assertEqual(row["costs"]["qwen_requests"], 3)
            self.assertEqual(row["costs"]["output_tokens"], 3 * 60)
            self.assertEqual(row["costs"]["input_tokens"], sum(c["usage"]["input_tokens"] for c in row["costs"]["calls"]))
        self.assertEqual(states, {"no_matches", "records_returned"})
        events = [json.loads(line) for line in (self.root / "run/events.jsonl").read_text().splitlines()]
        choices = [event for event in events if event["event_type"] == "agent_action_selected"]
        self.assertEqual(len(choices), 6)

    def test_archival_answer_rejects_local_only_support_even_when_archive_is_exposed(self):
        question = read_json(self.questions)["questions"][2]
        folder = self.root / "prepared-archive"
        folder.mkdir()
        prepared, evidence, common, support = runner._prepare(question, folder, Fingerprints(), runner.PROTOCOL)
        search = archive.ArchiveSearch(self.index, self.config, Fingerprints(), synthetic=True)
        identifier = next(iter(search.records))
        evidence[identifier], _ = runner._expose_archive(search, {"evidence_id": identifier}, 0)
        answer = {"type": "final", "decision": "supported", "answer_text": "Synthetic claim",
                  "citations": [next(iter(evidence))], "observation_ids": [], "date_interval": None,
                  "boxes": [], "limitations": []}
        with self.assertRaisesRegex(ValueError, "cited archive record"):
            runner.validate_answer(answer, question, evidence, prepared, common, support, runner.PROTOCOL)
        answer["citations"] = [identifier]
        self.assertEqual(runner.validate_answer(answer, question, evidence, prepared, common, support,
                                               runner.PROTOCOL)["status"], "completed")

    def test_stage_action_envelope_dispatches_and_keeps_raw_choice(self):
        class EnvelopePool(demo.SyntheticPool):
            def call(self, role, payload):
                response = super().call(role, payload)
                if role != "qwen":
                    return response
                value = json.loads(response["text"])
                if value["type"] == "final":
                    return response
                value["action"] = value.pop("type")
                value["stage"] = "choose_action"
                value["reason"] = "Synthetic reasoning, not an evidence record"
                value.pop("observation_id", None)
                response["text"] = json.dumps(value)
                return response

        result = self.run_fixture(pool_factory=EnvelopePool, conditions=("bounded_agentic_retrieval",))
        self.assertEqual(result["outcome_counts"], {"completed": 3, "abstained": 0, "error": 0})
        for row in read_json(self.root / "run/results.json")["results"]:
            folder = self.root / "run/questions" / row["question_id"] / row["condition"]
            reply = read_json(folder / "reply-001.json")
            self.assertEqual(reply["parsed"]["action"], "retrieve")
            self.assertNotIn("type", reply["parsed"])
            self.assertEqual(reply["normalized_action"]["type"], "retrieve")
            self.assertEqual(reply["normalized_action"]["reason"], reply["parsed"]["reason"])
            self.assertIn("action_key_to_type", reply["normalizations"])
            tool = read_json(folder / "retrieval-01.json")
            self.assertEqual(tool["arguments"]["query"], reply["parsed"]["query"])
            self.assertIsNone(tool["arguments"]["observation_id"])
            self.assertEqual(row["costs"]["retrieval_rounds"], 1)
            self.assertEqual(row["costs"]["qwen_requests"], 3)

    def test_unsupported_archive_answer_choice_can_retry_with_real_tool_dispatch(self):
        public = read_json(self.questions)
        question = public["questions"][2]
        write_json(self.questions, {**public, "questions": [question]})
        self.config["budgets"]["max_model_calls"] = 5

        class PrematureAnswer(demo.SyntheticPool):
            sent_bad_choice = False

            def call(self, role, payload):
                result = super().call(role, payload)
                if role == "qwen" and not self.sent_bad_choice:
                    self.sent_bad_choice = True
                    result["text"] = '{"type":"answer"}'
                return result

        result = self.run_fixture(pool_factory=PrematureAnswer, conditions=("bounded_agentic_retrieval",))
        self.assertEqual(result["outcome_counts"], {"completed": 1, "abstained": 0, "error": 0})
        row = read_json(self.root / "run/results.json")["results"][0]
        self.assertEqual(row["costs"]["qwen_requests"], 4)
        self.assertEqual(row["costs"]["retrieval_rounds"], 1)
        first = read_json(self.root / "run/questions" / question["question_id"] / row["condition"] / "reply-001.json")
        self.assertIn("No archive evidence", first["validation_error"])
        self.assertEqual(row["answer"]["citations"], ["fixture-archive-b1"])

    def test_stage_only_draft_preserves_content_and_rejects_conflicting_discriminators(self):
        draft = {"type": "archival_qa", "answer": {"source": "connected archive",
                 "documented_shoot_date": "2009", "evidence_id": "local-fixture-photo"}}
        value = {"stage": "answer", "response_contract": draft}
        original = copy.deepcopy(value)
        changes = []
        normalized = runner.normalize_action(value, changes)
        self.assertEqual(normalized, {"type": "answer", "response_contract": draft})
        self.assertEqual(value, original)
        self.assertIn("answer_stage_draft_to_answer_choice", changes)
        for conflicting in (
            {**value, "action": "retrieve", "query": "fixture query"},
            {**value, "type": "archival_qa"},
            {**value, "action": "answer", "type": "retrieve"},
            {**value, "response": draft},
            {"stage": "answer", "response_contract": None},
            {"response_contract": draft},
        ):
            with self.subTest(reply=conflicting), self.assertRaises(ValueError):
                runner.normalize_action(conflicting)

    def test_stage_only_archive_draft_requires_model_selected_retrieval(self):
        public = read_json(self.questions)
        question = public["questions"][2]
        write_json(self.questions, {**public, "questions": [question]})
        self.config["budgets"]["max_model_calls"] = 5
        draft = {"stage": "answer", "response_contract": {"type": "archival_qa", "answer": {
            "source": "connected archive", "documented_shoot_date": "2009",
            "evidence_id": question["observations"][0]["observation_id"]}}}

        for corrected in (False, True):
            class DraftPool(demo.SyntheticPool):
                sent_draft = False

                def call(self, role, payload):
                    response = super().call(role, payload)
                    if role == "qwen" and (not self.sent_draft or not corrected):
                        self.sent_draft = True
                        response["text"] = json.dumps(draft)
                    return response

            with self.subTest(model_corrects_reply=corrected):
                name = "corrected-draft" if corrected else "repeated-draft"
                summary = self.run_fixture(name, pool_factory=DraftPool, conditions=("bounded_agentic_retrieval",))
                self.assertEqual(summary["outcome_counts"], {
                    "completed": int(corrected), "abstained": 0, "error": int(not corrected)})
                row = read_json(self.root / name / "results.json")["results"][0]
                self.assertEqual(row["costs"]["qwen_requests"], 4 if corrected else 2)
                self.assertEqual(row["costs"]["retrieval_rounds"], int(corrected))
                folder = self.root / name / "questions" / question["question_id"] / row["condition"]
                first = read_json(folder / "reply-001.json")
                self.assertEqual(first["parsed"], draft)
                self.assertEqual(first["normalized_action"]["response_contract"], draft["response_contract"])
                self.assertIsNone(first["normalized_final"])
                self.assertIn("local_photo IDs are not archive sources", first["validation_error"])
                call = read_json(folder / "calls/000-qwen.json")
                photos = [json.loads(b["text"]) for b in call["request"]["messages"][1]["content"][1:]
                          if b["type"] == "text"]
                self.assertTrue(all(p["origin"] == "local_photo" for p in photos))
                if corrected:
                    self.assertEqual(row["answer"]["citations"], ["fixture-archive-b1"])
                else:
                    self.assertIsNone(row["answer"])
                    self.assertFalse(list(folder.glob("retrieval-*.json")))

    def test_incomplete_final_cannot_bypass_answer_validation(self):
        class PrematureFinal(demo.SyntheticPool):
            def call(self, role, payload):
                if role == "jina":
                    raise AssertionError("An invalid final must not cause automatic retrieval")
                result = super().call(role, payload)
                result["text"] = '{"type":"final"}'
                return result

        result = self.run_fixture(pool_factory=PrematureFinal, conditions=("bounded_agentic_retrieval",))
        self.assertEqual(result["outcome_counts"], {"completed": 0, "abstained": 0, "error": 3})
        for row in read_json(self.root / "run/results.json")["results"]:
            self.assertIn("missing fields", row["error"])
            self.assertEqual(row["costs"]["qwen_requests"], 2)
            self.assertEqual(row["costs"]["retrieval_rounds"], 0)

    def test_explicit_nested_final_completes_locally_without_another_model_call(self):
        public = read_json(self.questions)
        question = public["questions"][1]
        write_json(self.questions, {**public, "questions": [question]})
        envelopes = ({"type": "answer"}, {"stage": "choose_action", "action": "answer"},
                     {"stage": "answer"}, {}, {"type": "final"})
        for number, envelope in enumerate(envelopes):
            class NestedFinal(demo.SyntheticPool):
                def call(self, role, payload):
                    if role == "jina":
                        raise AssertionError("A complete local answer must not trigger retrieval")
                    scripted = copy.deepcopy(payload)
                    control = json.loads(scripted["messages"][-1]["content"][0]["text"])
                    control.update(stage="answer", allowed_action="final")
                    scripted["messages"][-1]["content"][0]["text"] = json.dumps(control)
                    response = super().call(role, scripted)
                    value = json.loads(response["text"])
                    del value["type"]
                    response["text"] = json.dumps({**envelope, "final": value})
                    return response

            with self.subTest(envelope=envelope):
                name = f"nested-{number}"
                summary = self.run_fixture(name, pool_factory=NestedFinal, conditions=("bounded_agentic_retrieval",))
                self.assertEqual(summary["outcome_counts"], {"completed": 1, "abstained": 0, "error": 0})
                row = read_json(self.root / name / "results.json")["results"][0]
                self.assertEqual(row["costs"]["qwen_requests"], 1)
                self.assertEqual(row["costs"]["retrieval_rounds"], 0)
                folder = self.root / name / "questions" / question["question_id"] / row["condition"]
                raw = read_json(folder / "final_raw.json")
                for returned, given in zip(row["answer"]["boxes"], raw["final"]["boxes"]):
                    self.assertEqual({k: returned[k] for k in given}, given)
                prepared = read_json(self.root / name / "questions" / question["question_id"] / "preparation.json")
                evidence = read_json(folder / "exposure.json")["evidence"]
                common = np.load(Path(prepared["common_support_path"]), allow_pickle=False)
                support = {"evidence_id": "fixture-support", "geometry": {"crop_shape": list(common.shape)}}
                for invalid in ({"type": "retrieve", "query": "q", "final": raw["final"]},
                                {"type": "answer", "final": {}}, {"type": "answer", "final": None},
                                {**raw, "final": {**raw["final"], "citations": ["unseen"]}}):
                    with self.assertRaises(ValueError):
                        runner.validate_answer(invalid, question, evidence, prepared, common, support, runner.PROTOCOL)

    def test_complete_stage_final_can_finish_without_redundant_choice(self):
        class DirectFinal(demo.SyntheticPool):
            def call(self, role, payload):
                if role == "jina":
                    raise AssertionError("A complete local final must not cause retrieval")
                scripted = copy.deepcopy(payload)
                control = json.loads(scripted["messages"][-1]["content"][0]["text"])
                control.update(stage="answer", allowed_action="final")
                scripted["messages"][-1]["content"][0]["text"] = json.dumps(control)
                response = super().call(role, scripted)
                value = json.loads(response["text"])
                del value["type"]
                value["stage"] = "answer"
                response["text"] = json.dumps(value)
                return response

        summary = self.run_fixture(pool_factory=DirectFinal, conditions=("bounded_agentic_retrieval",))
        self.assertEqual(summary["outcome_counts"], {"completed": 2, "abstained": 1, "error": 0})
        for row in read_json(self.root / "run/results.json")["results"]:
            self.assertEqual(row["costs"]["qwen_requests"], 1)
            self.assertEqual(row["costs"]["retrieval_rounds"], 0)
            folder = self.root / "run/questions" / row["question_id"] / row["condition"]
            reply = read_json(folder / "reply-001.json")
            self.assertIsNone(reply["normalized_action"])
            self.assertEqual(reply["normalized_final"], {
                **{k: v for k, v in reply["parsed"].items() if k != "stage"}, "type": "final"})
            self.assertEqual(read_json(folder / "final_raw.json"), reply["parsed"])

    def test_archive_draft_selects_answer_but_only_complete_final_is_scored(self):
        draft = {"type": "archival_qa", "answer": {"source": "https://example.invalid/draft",
                 "evidence_id": "unverified-draft-id", "date": "1900-01-01", "precision": "year"}}

        class DraftThenFinal(demo.SyntheticPool):
            def call(self, role, payload):
                response = super().call(role, payload)
                if role == "qwen":
                    value = json.loads(response["text"])
                    if value["type"] == "answer":
                        value = {"stage": "answer", "action": "answer", "response_contract": draft}
                    elif value["type"] == "final":
                        del value["type"]
                        value["stage"] = "answer"
                    response["text"] = json.dumps(value)
                return response

        public = read_json(self.questions)
        question = public["questions"][2]
        write_json(self.questions, {**public, "questions": [question]})
        summary = self.run_fixture(pool_factory=DraftThenFinal, conditions=("bounded_agentic_retrieval",))
        self.assertEqual(summary["outcome_counts"], {"completed": 1, "abstained": 0, "error": 0})
        row = read_json(self.root / "run/results.json")["results"][0]
        self.assertEqual(row["costs"]["qwen_requests"], 3)
        self.assertEqual(row["costs"]["retrieval_rounds"], 1)
        self.assertEqual(row["answer"]["citations"], ["fixture-archive-b1"])
        folder = self.root / "run/questions" / question["question_id"] / row["condition"]
        choice = read_json(folder / "reply-002.json")
        self.assertIsNone(choice["normalized_final"])
        self.assertIsNone(choice["validation_error"])
        self.assertEqual(choice["normalized_action"]["response_contract"], draft)
        last = read_json(folder / "reply-003.json")
        self.assertEqual(row["answer"], {**last["normalized_final"], "status": "completed"})

    def test_retrieval_cap_goes_directly_to_final_with_unchanged_call_accounting(self):
        class SearchUntilCap(demo.SyntheticPool):
            def call(self, role, payload):
                response = super().call(role, payload)
                if role == "qwen":
                    control = json.loads(payload["messages"][-1]["content"][0]["text"])
                    if control["stage"] == "choose_action":
                        query = "synthetic search " + str(control["archive_state"]["searches_performed"])
                        response["text"] = json.dumps({"type": "retrieve", "query": query})
                return response

        public = read_json(self.questions)
        question = public["questions"][2]
        write_json(self.questions, {**public, "questions": [question]})
        for cap, expected_rounds in ((4, 2), (1, 1)):
            with self.subTest(max_archive_records=cap):
                name = f"cap-{cap}"
                summary = self.run_fixture(name, pool_factory=SearchUntilCap,
                    protocol={**runner.PROTOCOL, "max_archive_records": cap}, conditions=("bounded_agentic_retrieval",))
                self.assertEqual(summary["outcome_counts"], {"completed": 1, "abstained": 0, "error": 0})
                row = read_json(self.root / name / "results.json")["results"][0]
                self.assertEqual(row["costs"]["retrieval_rounds"], expected_rounds)
                self.assertEqual(row["costs"]["qwen_requests"], expected_rounds + 1)
                self.assertEqual(row["costs"]["budget_output_tokens"], (expected_rounds + 1) * 60)
                folder = self.root / name / "questions" / question["question_id"] / row["condition"]
                calls = [read_json(p) for p in sorted((folder / "calls").glob("*-qwen.json"))]
                controls = [json.loads(c["request"]["messages"][-1]["content"][0]["text"]) for c in calls]
                self.assertEqual([c["stage"] for c in controls], ["choose_action"] * expected_rounds + ["answer"])
                self.assertIn("output_instruction", controls[-1])
                events = [json.loads(line) for line in (self.root / name / "events.jsonl").read_text().splitlines()]
                self.assertEqual(sum(e["event_type"] == "agent_final_required" for e in events), 1)

    def test_identity_freeze_and_asset_hash(self):
        value = read_json(self.package)
        base = value["records"][0]
        for status in ("probable", "unresolved", "distractor"):
            value["records"].append({**copy.deepcopy(base), "evidence_id": status, "identity_status": status})
        write_json(self.package, value)
        archive.import_archive(self.package, self.root / "archive2")
        archive.index_archive(self.root / "archive2/corpus.json", self.config, self.root / "index2",
                              pool_factory=demo.SyntheticPool, synthetic=True)
        search = archive.ArchiveSearch(self.root / "index2/index.json", self.config, Fingerprints(), synthetic=True)
        question = read_json(self.questions)["questions"][1]
        allowed, excluded = search.eligible(question)
        self.assertEqual(allowed, [base["evidence_id"]])
        self.assertEqual(len(excluded), 3)
        with self.assertRaisesRegex(ValueError, "consistently frozen"):
            search.eligible({**question, "knowledge_cutoff": "2026-11-16"})
        (self.root / "archive/assets" / Path(base["assets"][0]["path"]).name).exists()
        corpus = read_json(self.root / "archive/corpus.json")
        asset = self.root / "archive" / corpus["records"][0]["assets"][0]["path"]
        asset.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            archive.ArchiveSearch(self.index, self.config, Fingerprints(), synthetic=True)

    def test_no_synthetic_index_in_production(self):
        with self.assertRaisesRegex(ValueError, "synthetic"):
            archive.ArchiveSearch(self.index, self.config, Fingerprints())

    def test_archive_metadata_names_only_actually_shown_observations(self):
        search = archive.ArchiveSearch(self.index, self.config, Fingerprints(), synthetic=True)
        identifier = next(iter(search.records))
        question = read_json(self.questions)["questions"][1]
        for remaining in (0, 1):
            with self.subTest(remaining_images=remaining):
                record, blocks = runner._expose_archive(search, {"evidence_id": identifier}, remaining)
                metadata = json.loads(blocks[0]["text"])
                self.assertEqual(metadata["origin"], "archive")
                self.assertEqual(metadata["observation_id"], identifier if remaining else None)
                self.assertEqual(metadata["image_exposed"], bool(remaining))
                self.assertEqual(sum(b["type"] == "image" for b in blocks), remaining)
                contract = runner.response_contract(question, {identifier: record}, {"compared_observations": []})
                self.assertEqual(contract["exposed_citation_ids"], [identifier])
                self.assertEqual(set(contract["shown_photo_capture_intervals"]), {identifier} if remaining else set())

    def test_retrieved_contract_exposes_exact_dates_before_answering_in_both_modes(self):
        public = read_json(self.questions)
        question = next(q for q in public["questions"] if q["task"] == "earliest_evidence")
        write_json(self.questions, {**public, "questions": [question]})
        package = read_json(self.package)
        row = package["records"][0]
        row.update(building_id=question["building_id"], view_id=question["view_id"])

        class ArchiveAnswer(demo.SyntheticPool):
            def call(self, role, payload):
                response = super().call(role, payload)
                if role != "qwen":
                    return response
                value = json.loads(response["text"])
                if value["type"] == "final":
                    tool = next(m for m in reversed(payload["messages"])
                                if '"retrieval_status"' in m["content"][0].get("text", ""))
                    contract = json.loads(payload["messages"][-1]["content"][0]["text"])["response_contract"]
                    photo = json.loads(tool["content"][1]["text"])
                    identifier = photo["observation_id"]
                    value.update(citations=[identifier], observation_ids=[identifier], date_interval={
                        **contract["shown_photo_capture_intervals"][identifier], "uncertain": False})
                    response["text"] = json.dumps(value)
                return response

        for precision, start, end in (("day", "2008-09-28", "2008-09-28"),
                                      ("month", "2008-09", "2008-09"),
                                      ("interval", "2007", "2008")):
            with self.subTest(precision=precision):
                row["observation_date"].update(start=start, end=end, precision=precision)
                write_json(self.package, package)
                archive.import_archive(self.package, self.root / (precision + "-archive"))
                archive.index_archive(self.root / (precision + "-archive/corpus.json"), self.config,
                                      self.root / (precision + "-index"), pool_factory=demo.SyntheticPool, synthetic=True)
                result = runner.run(self.questions, self.config, self.root / precision,
                    archive_index=self.root / (precision + "-index/index.json"),
                    conditions=runner.CONDITIONS[1:], pool_factory=ArchiveAnswer, synthetic=True)
                self.assertEqual(result["outcome_counts"], {"completed": 2, "abstained": 0, "error": 0})
                for answer in read_json(self.root / precision / "results.json")["results"]:
                    self.assertEqual(answer["answer"]["date_interval"], {
                        "start": start, "end": end, "precision": precision, "uncertain": False})
                    folder = self.root / precision / "questions" / question["question_id"] / answer["condition"]
                    exposed = read_json(folder / "exposure.json")["evidence"]
                    prepared = {"compared_observations": []}
                    bad = copy.deepcopy(answer["answer"])
                    del bad["status"]
                    bad["date_interval"].update(start="2008", end="2008", precision="year")
                    original = copy.deepcopy(bad)
                    with self.assertRaisesRegex(ValueError, "documented date precision"):
                        runner.validate_answer(bad, question, exposed, prepared, None, None, runner.PROTOCOL)
                    self.assertEqual(bad, original)

    def resource_index(self):
        weights, tokenizer = self.root / "weights.bin", self.root / "tokenizer.json"
        weights.write_bytes(b"synthetic-weights-A")
        tokenizer.write_bytes(b"synthetic-tokenizer")
        alias = self.root / "weights-alias.bin"
        alias.symlink_to(weights)
        value = read_json(self.index)
        value["entries"] = [copy.deepcopy(value["entries"][i % len(value["entries"])]) for i in range(77)]
        value["resource_contracts"] = [[{"path": str(weights if i % 2 else alias), "sha256": sha256(weights)},
                                       {"path": str(tokenizer), "sha256": sha256(tokenizer)}] for i in range(77)]
        write_json(self.index, value)
        return value, weights, tokenizer

    def test_archive_load_hashes_repeated_resolved_resources_once_per_load(self):
        original, weights, tokenizer = self.resource_index()
        index_sha = sha256(self.index)
        with patch.object(models, "_timed_sha256", wraps=models._timed_sha256) as reads:
            search = archive.ArchiveSearch(self.index, self.config, Fingerprints(), synthetic=True)
            self.assertCountEqual([Path(call.args[0]) for call in reads.call_args_list], [weights, tokenizer])
            self.assertEqual(search.index["resource_contracts"], original["resource_contracts"])
            self.assertEqual(len(search.index["entries"]), 77)
            archive.ArchiveSearch(self.index, self.config, Fingerprints(), synthetic=True)
            self.assertEqual(reads.call_count, 4)  # No approval survives into the next load.
        self.assertEqual(sha256(self.index), index_sha)

    def test_archive_load_rejects_conflicting_hashes_for_symlink_alias(self):
        value, weights, _ = self.resource_index()
        value["resource_contracts"][-1][0]["sha256"] = "0" * 64
        write_json(self.index, value)
        with patch.object(models, "_timed_sha256", wraps=models._timed_sha256) as reads:
            with self.assertRaisesRegex(ValueError, "Conflicting Jina resource hashes") as error:
                archive.ArchiveSearch(self.index, self.config, Fingerprints(), synthetic=True)
            self.assertIn(str(weights), str(error.exception))
            reads.assert_not_called()

    def test_archive_load_rejects_modified_or_missing_repeated_resource(self):
        _, weights, _ = self.resource_index()
        archive.ArchiveSearch(self.index, self.config, Fingerprints(), synthetic=True)
        weights.write_bytes(b"synthetic-weights-B")  # Same size, different actual bytes.
        with self.assertRaisesRegex(ValueError, "resources are missing or changed"):
            archive.ArchiveSearch(self.index, self.config, Fingerprints(), synthetic=True)
        weights.unlink()
        with self.assertRaisesRegex(ValueError, "resources are missing or changed"):
            archive.ArchiveSearch(self.index, self.config, Fingerprints(), synthetic=True)

    def test_archive_dedup_preserves_required_resources_for_each_real_embedding(self):
        value, _, _ = self.resource_index()
        value["synthetic_fixture"] = False
        value["resource_contracts"][-1] = []
        write_json(self.index, value)
        with self.assertRaisesRegex(ValueError, "real Jina index requires recorded model resources"):
            archive.ArchiveSearch(self.index, self.config, Fingerprints())

    def test_no_backdating_and_month_precision(self):
        value = read_json(self.package)
        row = value["records"][0]
        row.update(availability_basis="collection_time", available_at="2010")
        with self.assertRaisesRegex(ValueError, "never backdate"):
            archive.validate_package(value, self.package.parent, Fingerprints())
        row.update(availability_basis="documented_publication", collected_at="2026-11-16T10:00:00+03:00")
        with self.assertRaisesRegex(ValueError, "new consistent"):
            archive.validate_package(value, self.package.parent, Fingerprints())
        self.assertLess(archive._date("2010-02", "month"), archive._date("2010-03", "month"))

    def test_runtime_failure_is_error_without_answer_retry(self):
        class BrokenJina(demo.SyntheticPool):
            def call(self, role, payload):
                return {"status": "error", "error": "fixture GPU unavailable", "usage": {}} if role == "jina" else super().call(role, payload)
        self.run_fixture(pool_factory=BrokenJina)
        rows = read_json(self.root / "run/results.json")["results"]
        affected = next(r for r in rows if r["task"] == "temporal_change" and r["condition"] == "bounded_agentic_retrieval")
        self.assertEqual(affected["status"], "error")
        self.assertIn("fixture GPU unavailable", affected["error"])
        self.assertEqual(affected["costs"]["qwen_requests"], 1)

    def test_model_cannot_exceed_two_adaptive_rounds(self):
        class RepeatedSearch(demo.SyntheticPool):
            def call(self, role, payload):
                result = super().call(role, payload)
                if role == "qwen":
                    result["text"] = json.dumps({"type": "retrieve", "query": "Synthetic question-specific query", "observation_id": None})
                return result
        self.run_fixture(pool_factory=RepeatedSearch, conditions=("bounded_agentic_retrieval",))
        rows = read_json(self.root / "run/results.json")["results"]
        self.assertTrue(all(r["costs"]["retrieval_rounds"] == 2 for r in rows))
        self.assertTrue(all(r["status"] == "error" and "round budget exhausted" in r["error"] for r in rows))
        self.assertTrue(all(r["costs"]["qwen_requests"] == 4 for r in rows))

    def test_retrieval_does_not_grant_extra_model_or_token_budget(self):
        for budget, limit, error in (("max_model_calls", 1, "model request budget"),
                                     ("max_total_output_tokens", 60, "token/image budget")):
            with self.subTest(budget=budget):
                config = copy.deepcopy(self.config)
                config["budgets"][budget] = limit
                runner.run(self.questions, config, self.root / budget, archive_index=self.index,
                    pool_factory=demo.SyntheticPool, synthetic=True, conditions=(runner.CONDITIONS[2],))
                rows = read_json(self.root / budget / "results.json")["results"]
                self.assertTrue(all(r["status"] == "error" and error in r["error"] for r in rows))
                self.assertTrue(all(r["costs"]["retrieval_rounds"] == 1 and r["costs"]["qwen_requests"] == 1 for r in rows))

    def test_invalid_citations_and_json_keep_bounded_errors(self):
        class Invalid(demo.SyntheticPool):
            def call(self, role, payload):
                result = super().call(role, payload)
                if role == "qwen":
                    result["text"] = '{"type":"final","answer_text":"truncated'
                    result["generation"]["reached_max_new_tokens"] = True
                return result
        summary = self.run_fixture(pool_factory=Invalid)
        self.assertEqual(summary["outcome_counts"]["error"], 9)
        rows = read_json(self.root / "run/results.json")["results"]
        self.assertTrue(all(r["costs"]["qwen_requests"] == 2 for r in rows))
        self.assertTrue(all("Unterminated string" in r["error"] for r in rows))
        event_text = (self.root / "run/events.jsonl").read_text()
        self.assertIn('"reached_max_new_tokens": true', event_text)

    def test_support_and_coordinate_owner_are_strict(self):
        question = read_json(self.questions)["questions"][1]
        folder = self.root / "prepared"
        folder.mkdir()
        prepared, evidence, common, support = runner._prepare(question, folder, Fingerprints(), runner.PROTOCOL)
        ids = list(evidence)
        answer = {"type": "final", "decision": "change", "answer_text": "Fixture", "citations": ids, "observation_ids": ids,
            "date_interval": {"start": "2009", "end": "2015", "precision": "interval", "uncertain": True}, "limitations": [],
            "boxes": [{"evidence_id": ids[-1], "coordinate_system": "region_crop_pixels", "bbox_xyxy": [3, 1, 6, 4]}]}
        self.assertEqual(runner.validate_answer(answer, question, evidence, prepared, common, support, runner.PROTOCOL)["status"], "completed")
        common[2, 3] = False
        with self.assertRaisesRegex(ValueError, "full saved geometric support"):
            runner.validate_answer(answer, question, evidence, prepared, common, support, runner.PROTOCOL)
        common[:] = True
        answer["boxes"][0]["observation_id"] = "wrong-owner"
        with self.assertRaisesRegex(ValueError, "owner"):
            runner.validate_answer(answer, question, evidence, prepared, common, support, runner.PROTOCOL)
        answer["boxes"] = []
        answer["citations"] = ["not-exposed"]
        with self.assertRaisesRegex(ValueError, "actually exposed"):
            runner.validate_answer(answer, question, evidence, prepared, common, support, runner.PROTOCOL)

    def test_box_shape_is_checked_against_its_photo_without_rescaling(self):
        question = read_json(self.questions)["questions"][1]
        folder = self.root / "prepared"
        folder.mkdir()
        prepared, evidence, common, support = runner._prepare(question, folder, Fingerprints(), runner.PROTOCOL)
        local_ids = list(evidence)
        search = archive.ArchiveSearch(self.index, self.config, Fingerprints(), synthetic=True)
        archive_id = next(iter(search.records))
        record, _ = runner._expose_archive(search, {"evidence_id": archive_id}, 1)
        evidence[archive_id] = record
        for identifier, coordinate, shape in (
            (local_ids[-1], "region_crop_pixels", question["frame"]["shape"]),
            (archive_id, "observation_native", record["shape"]),
        ):
            with self.subTest(coordinate=coordinate):
                answer = {"type": "final", "decision": "change", "answer_text": "Synthetic change.",
                    "citations": local_ids + [archive_id], "observation_ids": local_ids,
                    "date_interval": {"start": "2009", "end": "2015", "precision": "interval", "uncertain": True},
                    "limitations": [], "boxes": [{"evidence_id": identifier, "coordinate_system": coordinate,
                                                  "bbox_xyxy": [3, 1, 6, 4], "shape": list(shape)}]}
                original = copy.deepcopy(answer)
                result = runner.validate_answer(answer, question, evidence, prepared, common, support, runner.PROTOCOL)
                self.assertEqual(result["boxes"][0]["shape"], shape)
                self.assertEqual(result["boxes"][0]["bbox_xyxy"], [3, 1, 6, 4])
                self.assertEqual(answer, original)
                for invalid in (None, [shape[0], shape[1] + 1], [float(shape[0]), shape[1]],
                                [True, shape[1]], [shape[0]], {"height": shape[0], "width": shape[1]}):
                    with self.subTest(shape=invalid), self.assertRaisesRegex(ValueError, "shape must match"):
                        answer["boxes"][0]["shape"] = invalid
                        runner.validate_answer(answer, question, evidence, prepared, common, support, runner.PROTOCOL)
                answer["boxes"][0]["shape"] = list(shape)
                answer["boxes"][0]["bbox_xyxy"] = [3, 1, shape[1] + 1, 4]
                with self.assertRaises(ValueError):
                    runner.validate_answer(answer, question, evidence, prepared, common, support, runner.PROTOCOL)
                answer["boxes"][0]["bbox_xyxy"] = [3, 1, 6, 4]
                answer["boxes"][0]["observation_id"] = "wrong-owner"
                with self.assertRaisesRegex(ValueError, "owner"):
                    runner.validate_answer(answer, question, evidence, prepared, common, support, runner.PROTOCOL)
                del answer["boxes"][0]["observation_id"]
                answer["boxes"][0]["invented_label"] = "not a supported field"
                with self.assertRaisesRegex(ValueError, "Unexpected localization fields:.*invented_label"):
                    runner.validate_answer(answer, question, evidence, prepared, common, support, runner.PROTOCOL)

    def test_final_boxes_with_shape_survive_all_conditions_without_retry(self):
        public = read_json(self.questions)
        question = public["questions"][1]
        write_json(self.questions, {**public, "questions": [question]})

        class ShapePool(demo.SyntheticPool):
            def call(self, role, payload):
                response = super().call(role, payload)
                if role == "qwen":
                    value = json.loads(response["text"])
                    for box in value.get("boxes", []):
                        box["shape"] = question["frame"]["shape"]
                    response["text"] = json.dumps(value)
                return response

        summary = self.run_fixture(pool_factory=ShapePool)
        self.assertEqual(summary["outcome_counts"], {"completed": 3, "abstained": 0, "error": 0})
        rows = read_json(self.root / "run/results.json")["results"]
        for row, calls in zip(rows, (1, 1, 3)):
            self.assertEqual(row["costs"]["qwen_requests"], calls)
            self.assertEqual(row["answer"]["boxes"][0]["shape"], question["frame"]["shape"])
            self.assertEqual(row["answer"]["boxes"][0]["bbox_xyxy"], [3, 1, 6, 4])
            folder = self.root / "run/questions" / question["question_id"] / row["condition"]
            self.assertEqual(read_json(folder / "final_raw.json")["boxes"][0]["shape"], question["frame"]["shape"])
            self.assertTrue(all(read_json(p)["validation_error"] is None for p in folder.glob("reply-*.json")))

    def test_empty_archive_is_explicit_plumbing(self):
        with self.assertRaisesRegex(ValueError, "explicit plumbing"):
            runner.run(self.questions, self.config, self.root / "empty", pool_factory=demo.SyntheticPool, synthetic=True)
        result = runner.run(self.questions, self.config, self.root / "empty", allow_empty_archive=True,
                            pool_factory=demo.SyntheticPool, synthetic=True)
        self.assertEqual(result["comparison_scope"], "plumbing_no_archive")

    def test_required_dates_are_not_silently_dropped(self):
        question = read_json(self.questions)["questions"][0]
        with self.assertRaisesRegex(ValueError, "no silent temporal truncation"):
            runner._prepare(question, self.root, Fingerprints(), {**runner.PROTOCOL, "max_local_images": 1})

    def test_condition_major_keeps_all_a_then_c_and_checkpoints_before_b(self):
        order = ("photos_only", "bounded_agentic_retrieval", "fixed_retrieval")
        original = runner._condition

        def interrupt_before_b(question, condition, *args, **kwargs):
            if condition == "fixed_retrieval":
                raise KeyboardInterrupt("synthetic interruption before B")
            return original(question, condition, *args, **kwargs)

        with patch.object(runner, "_condition", side_effect=interrupt_before_b), self.assertRaises(KeyboardInterrupt):
            self.run_fixture("partial", conditions=order, condition_major=True)
        partial = self.root / "partial"
        rows = read_json(partial / "results.json")["results"]
        self.assertEqual([r["condition"] for r in rows], [order[0]] * 3 + [order[1]] * 3)
        self.assertEqual(read_json(partial / "run.json")["status"], "interrupted")
        self.assertEqual(read_json(partial / "contract.json")["execution_order"], "conditions_first")
        self.run_fixture("resumed-phases", conditions=order, condition_major=True, resume_from=partial)
        resumed = read_json(self.root / "resumed-phases/results.json")["results"]
        self.assertEqual([r["condition"] for r in resumed], [c for c in order for _ in range(3)])
        self.assertTrue(all(r["costs"]["model_calls"] == 0 for r in resumed[:6]))
        with self.assertRaisesRegex(ValueError, "Resume .*changed"):
            self.run_fixture("different-order", conditions=order, resume_from=partial)


if __name__ == "__main__":
    unittest.main()
