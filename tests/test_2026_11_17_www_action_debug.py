"""Transport/dispatch and C-only freeze checks; synthetic workers, never Qwen accuracy."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

from facade_change.history_features import digest
from facade_change.io import read_json, sha256
from tests import test_2026_11_15_www as runner_fixtures
from tests import test_2026_11_15_www_curation as review_fixtures

demo, runner, metrics = runner_fixtures.demo, runner_fixtures.runner, review_fixtures.metrics


class ActionDebugTests(unittest.TestCase):
    setUp = runner_fixtures.WWWRunnerTest.setUp
    run_fixture = runner_fixtures.WWWRunnerTest.run_fixture

    def test_saved_null_query_is_transport_only_and_nonnull_never_triggers_search(self):
        # Exact action structure present in seven supplied raw responses.
        raw = '{\n  "stage": "choose_action",\n  "action": "answer",\n  "query": null\n}\n'
        parsed = runner.parse_reply(raw)
        before, changes = copy.deepcopy(parsed), []
        self.assertEqual(runner.normalize_action(parsed, changes), {"type": "answer"})
        self.assertEqual(parsed, before)
        self.assertIn("removed_null_query_from_answer", changes)
        refused = {"type": "abstain", "reason": "Synthetic occlusion", "query": None}
        self.assertEqual(runner.normalize_action(refused), {"type": "abstain", "reason": refused["reason"]})
        for query in ("", "archive query", False, [], {}):
            with self.assertRaisesRegex(ValueError, "conflicts with the chosen action"):
                runner.normalize_action({**before, "query": query})
        with self.assertRaises(ValueError):
            runner.normalize_action({**before, "type": "retrieve"})
        with self.assertRaises(ValueError):
            runner.normalize_action({**before, "observation_id": "unrequested-extra-argument"})
        with self.assertRaises(ValueError):
            runner.normalize_action({"type": "retrieve", "query": None})
        with self.assertRaisesRegex(ValueError, "explicit final envelope"):
            runner.normalize_stage_reply({"stage": "answer", "final": {"type": "answer", "boxes": []}}, "choose_action")

    def test_null_query_advances_local_answer_without_retry_and_keeps_stage_contracts(self):
        class NullQueryPool(demo.SyntheticPool):
            def call(self, role, payload):
                if role == "qwen":
                    control = json.loads(payload["messages"][-1]["content"][0]["text"])
                    if control["stage"] == "choose_action" and control["task_scope"]["task"] != "archival_qa":
                        return {"status": "ok", "text": '{"stage":"choose_action","action":"answer","query":null}',
                                "usage": {"model_calls": 1, "input_tokens": 32, "output_tokens": 29,
                                          "images": sum(b["type"] == "image" for m in payload["messages"] for b in m["content"])}}
                return super().call(role, payload)
        self.run_fixture(pool_factory=NullQueryPool)
        self.assertEqual(digest(runner.SYSTEM), "46b84a65c29c8674fc6dbaa7675fb73580175bf0c93b140c6a2dc5058244d96c")
        for row in read_json(self.root / "run/results.json")["results"]:
            folder = self.root / "run/questions" / row["question_id"] / row["condition"]
            for call_path in (folder / "calls").glob("*-qwen.json"):
                messages = read_json(call_path)["request"]["messages"]
                control = json.loads(messages[-1]["content"][0]["text"])
                expected = runner.stage_system(row["condition"], control["stage"])
                self.assertEqual(messages[0]["content"][0]["text"], expected)
                self.assertEqual("response_contract" in control, control["stage"] == "answer")
                self.assertEqual("action_contract" in control, control["stage"] == "choose_action")
            if row["condition"] == "bounded_agentic_retrieval" and row["task"] != "archival_qa":
                self.assertEqual(row["costs"]["qwen_requests"], 2)
                self.assertEqual(row["costs"]["retrieval_rounds"], 0)
                trace = read_json(folder / "reply-001.json")
                self.assertIsNone(trace["parsed"]["query"])
                self.assertEqual(trace["normalized_action"], {"type": "answer"})
                self.assertIsNone(trace["validation_error"])
                self.assertEqual(trace["allowed_action"], "retrieve_or_decide")

    def test_replay_missing_pixel_support_is_never_filled_in(self):
        path = Path(__file__).resolve().parents[1] / "scripts/2026-11-17-www-action-replay.py"
        spec = importlib.util.spec_from_file_location("www_action_replay_fixture", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        question = read_json(self.questions)["questions"][0]
        coverage = module.RecordedCoverage({"common_pixels": 1}, question)
        self.assertTrue(coverage.any())
        with self.assertRaises(module.MissingSupport):
            coverage[0:1, 0:1]
        with self.assertRaises(ValueError):
            module.RecordedCoverage({"common_pixels": -1}, question)
        with self.assertRaises(ValueError):
            runner.parse_reply('{"type":"final","decision":"change"')


class SingleConditionScoreTests(unittest.TestCase):
    setUp = review_fixtures.ReviewAndMetricsTest.setUp
    approve = review_fixtures.ReviewAndMetricsTest.approve

    def test_c_only_plan_scores_same_frozen_questions_and_rejects_old_abc_contract(self):
        questions, answers, index = self.approve()
        before = {p: sha256(p) for p in (questions, answers, self.candidates)}
        modes = ("bounded_agentic_retrieval",)
        metrics.plan(questions, answers, self.inventory, self.config, self.root / "c-plan",
                     archive_index=index, synthetic=True, conditions=modes, condition_major=True)
        runner.run(questions, self.config, self.root / "c-run", archive_index=index, synthetic=True,
                   pool_factory=demo.SyntheticPool, conditions=modes, condition_major=True)
        with self.assertRaisesRegex(ValueError, "differ from the predeclared plan"):
            metrics.score(questions, answers, self.root / "c-run", self.root / "plan/plan.private.json", self.root / "wrong-score")
        result = metrics.score(questions, answers, self.root / "c-run", self.root / "c-plan/plan.private.json", self.root / "c-score")
        report = read_json(self.root / "c-score/metrics.private.json")
        self.assertEqual(result["conditions"], list(modes))
        self.assertEqual(result["cohort_condition_rows"], 5)
        self.assertEqual({r["condition"] for r in report["cases"]}, set(modes))
        self.assertTrue(all(r["status"] == "not_planned_in_this_execution"
                            for rows in report["paired_building_differences"].values() for r in rows))
        self.assertEqual({p: sha256(p) for p in before}, before)


if __name__ == "__main__":
    unittest.main()
