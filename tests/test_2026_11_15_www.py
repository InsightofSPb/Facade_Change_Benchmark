"""CPU acceptance for public-only WWW inference; scripted models are explicit fixtures."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from facade_change.history_features import Fingerprints
from facade_change.io import read_json, write_json

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
        self.assertEqual(self.run_fixture()["outcome_counts"], {"completed": 9, "abstained": 0, "error": 0})
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
        self.config["budgets"]["max_output_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "Resume .*changed"):
            self.run_fixture("bad-resume", resume_from=self.root / "run")

    def test_private_or_unknown_fields_cannot_enter_public_manifest(self):
        value = read_json(self.questions)
        value["questions"][0]["reviewed_labels"] = {"assessment": "earliest_presence"}
        write_json(self.questions, value)
        with self.assertRaises(ValueError):
            self.run_fixture()

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


if __name__ == "__main__":
    unittest.main()
