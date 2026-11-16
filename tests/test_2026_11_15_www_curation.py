"""Focused CPU checks for inventory, review boundaries and fixed-cohort metrics."""
import copy
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from facade_change.history_features import Fingerprints
from facade_change.io import read_json, sha256, write_json

demo = importlib.import_module("facade_change.agentic.2026-11-15_www_demo")
qa = demo.qa
curation = importlib.import_module("facade_change.agentic.2026-11-15_www_curation")
metrics = importlib.import_module("facade_change.agentic.2026-11-15_www_evaluation")


class ReviewAndMetricsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        _, self.package, self.config = demo.fixture(self.root / "fixture")
        self.candidates = self.root / "fixture/candidates/candidates.json"
        self.rows = read_json(self.candidates)["candidates"]
        self.histories = self.root / "fixture/inputs/histories"
        self.audit = self.root / "audit.json"
        write_json(self.audit, {"kind": "www_exposure_audit", "buildings": [
            {"building_id": b, "cohort": "development", "reason": "explicit synthetic fixture"} for b in {r["building_id"] for r in self.rows}]})
        curation.inventory(self.histories, self.root / "inventory", candidates_path=self.candidates, exposure_path=self.audit)
        self.inventory = self.root / "inventory/inventory.json"

    def test_all_tracks_unknown_exposure_and_deterministic_overlap_cap(self):
        result = curation.inventory(self.histories, self.root / "unknown", candidates_path=self.candidates)
        self.assertEqual(result["tracks"], 8)
        self.assertEqual(result["eligible_views"], 0)
        self.assertTrue(all(r["address_metadata"] is None and r["cohort"] == "unassigned" for r in read_json(self.root / "unknown/inventory.json")["views"]))
        inv = read_json(self.inventory)
        first, exclusions = curation.select_batch(self.rows, inv)
        second, _ = curation.select_batch(list(reversed(self.rows)), inv)
        self.assertEqual([r["question_id"] for r in first], [r["question_id"] for r in second])
        self.assertIn("overlapping_anchor_region", {r["reason"] for r in exclusions})
        self.assertEqual(len({r["building_id"] for r in first}), 7)

    def test_pending_proposals_public_export_and_review_ui(self):
        curation.batch(self.candidates, self.inventory, self.root / "batch", size=100, per_building=12)
        selection = self.root / "batch/selection.private.json"
        summary = curation.review_page(self.candidates, selection, self.root / "ui")
        self.assertEqual(summary["approved"], 0)
        self.assertGreater(summary["automatic_proposals"], 0)
        drafts = read_json(self.root / "ui/drafts.private.json")
        self.assertTrue(all(r["label"]["events"] == [] for r in drafts["items"]))
        self.assertTrue(all(not b["confirmed"] and not b["event_id"] for r in drafts["items"] for b in r["proposed_boxes"]))
        self.assertTrue(all(not f["temporal_gold"] for r in drafts["items"] for f in r["positive_annotation_facts"]))
        curation.public_candidates(self.candidates, selection, self.root / "public", limit=3)
        value = read_json(self.root / "public/questions.json")
        self.assertEqual(len(value["questions"]), 3)
        self.assertNotIn("annotation_evidence", json.dumps(value))
        self.assertNotIn("provisional_answer", json.dumps(value))
        qa.validate_public(value)
        bad = {"assessment": "confirmed_change", "observability": "visible", "reference_answer": "Draft only", "events": []}
        with self.assertRaisesRegex(ValueError, "reviewed event identities"):
            qa._checked_label(next(r for r in self.rows if r["region_id"] == "change"), bad, Fingerprints())

    def approve(self):
        selected = [next(r for r in self.rows if r["region_id"] == region and (region != "earliest" or r["task"] == "earliest_evidence"))
                    for region in ("earliest", "change", "stable", "occluded", "error")]
        decisions = []
        for row in selected:
            label = {"assessment": "confirmed_change", "observability": "visible", "reference_answer": "Explicit synthetic reference",
                "limitations": [], "archival_status": "unknown", "supporting_evidence_ids": [], "events": [],
                "archival_stratum": row["region_id"] == "change", "known_support_exhaustive": False}
            if row["region_id"] == "earliest":
                label.update(assessment="earliest_presence", earliest_observation_id=row["observations"][0]["observation_id"], present_by="2009", onset="unknown")
            elif row["region_id"] == "stable":
                label["assessment"] = "confirmed_stability"
            elif row["region_id"] == "occluded":
                label.update(assessment="insufficient_evidence", observability="occluded", limitations=["Fixture occlusion"])
            else:
                label["events"] = [{"event_id": "fixture-" + row["region_id"], "bbox_xyxy": [3, 1, 6, 4],
                                    "coordinate_system": "region_crop_pixels", "region_id": row["region_id"]}]
                if row["region_id"] == "change":
                    label.update(supporting_evidence_ids=["fixture-archive-b1"], archival_status="verified_useful")
            decisions.append({"question_id": row["question_id"], "expected_revision": 0, "status": "approved",
                              "reviewer": "synthetic_only", "private_notes": "Explicit fixture, no human data", "label": label})
        decision_path = self.root / "decisions.private.json"
        write_json(decision_path, {"candidate_sha256": sha256(self.candidates), "decisions": decisions})
        qa.review(self.candidates, decision_path, self.root / "review", reviewed_gallery_only=True)
        self.assertEqual(len(read_json(self.root / "review/reviewed.json")["states"]), 5)
        qa.export(self.root / "review/reviewed.json", self.root / "export")
        questions, answers = self.root / "export/questions.json", self.root / "export/answers.private.json"
        demo.archive.import_archive(self.package, self.root / "archive")
        demo.archive.index_archive(self.root / "archive/corpus.json", self.config, self.root / "index", pool_factory=demo.SyntheticPool, synthetic=True)
        index = self.root / "index/index.json"
        metrics.plan(questions, answers, self.inventory, self.config, self.root / "plan", archive_index=index, synthetic=True)
        return questions, answers, index

    def test_all_reviewed_rgb_plan_preserves_unknown_exposure_and_scores_every_case(self):
        questions, answers, index = self.approve()
        order = ("photos_only", "bounded_agentic_retrieval", "fixed_retrieval")
        with self.assertRaisesRegex(ValueError, "explicit RGB QA policy"):
            metrics.plan(questions, answers, self.inventory, self.config, self.root / "invalid-legacy",
                         archive_index=index, synthetic=True, include_reviewed=True)
        curation.inventory(self.histories, self.root / "rgb", candidates_path=self.candidates,
                           qa_policy=curation.rgb_policy.POLICY)
        inventory = self.root / "rgb/inventory.json"
        before = sha256(inventory)
        with self.assertRaisesRegex(ValueError, "unaudited/ineligible"):
            metrics.plan(questions, answers, inventory, self.config, self.root / "strict",
                         archive_index=index, synthetic=True)
        metrics.plan(questions, answers, inventory, self.config, self.root / "all-plan",
                     archive_index=index, synthetic=True, include_reviewed=True, conditions=order, condition_major=True)
        plan = self.root / "all-plan/plan.private.json"
        frozen = read_json(plan)
        self.assertEqual(frozen["cohort_scope"], "all_reviewed")
        self.assertEqual({r["cohort"] for r in frozen["cases"]}, {"unassigned"})
        self.assertTrue(all(r["qa_exposure_status"] == "unknown" and not r["evaluation_eligible"] for r in frozen["cases"]))
        self.assertEqual(sha256(inventory), before)
        demo.runner.run(questions, self.config, self.root / "run", archive_index=index,
                        pool_factory=demo.SyntheticPool, synthetic=True, conditions=order, condition_major=True)
        metrics.score(questions, answers, self.root / "run", plan, self.root / "all-score")
        report = read_json(self.root / "all-score/metrics.private.json")
        self.assertEqual(len(report["cases"]), 15)
        for condition in demo.runner.CONDITIONS:
            self.assertEqual(report["summaries"]["all_reviewed/all"][condition]["questions"], 5)
        self.assertEqual(len(report["paired_building_differences"]["all_reviewed/all"]), 3)
        self.assertTrue(all(not r["evaluation_eligible"] for r in report["frozen_exposure"]))
        bad = read_json(inventory)
        for view in bad["views"]:
            view["eligible"] = False
        write_json(self.root / "ineligible.json", bad)
        with self.assertRaisesRegex(ValueError, "unaudited/ineligible"):
            metrics.plan(questions, answers, self.root / "ineligible.json", self.config, self.root / "bad-plan",
                         archive_index=index, synthetic=True, include_reviewed=True)

    def test_cpu_launcher_preserves_approved_ids_and_stops_before_models(self):
        questions, _, _ = self.approve()
        identity = self.root / "identity.json"
        write_json(identity, {"schema_version": 1, "kind": "www_physical_identity_review", "groups": []})
        curation.inventory(self.histories, self.root / "rgb", candidates_path=self.candidates,
                           qa_policy=curation.rgb_policy.POLICY, identity_map=identity)
        config = self.root / "config.json"
        write_json(config, self.config)
        script = Path(__file__).resolve().parents[1] / "scripts/2026-11-17-run-approved-www.py"
        source = self.root / "decisions.private.json"
        original = sha256(source)
        argv = [sys.executable, str(script), "--decisions", str(source), "--candidates", str(self.candidates),
                "--corpus", str(self.root / "archive/corpus.json"), "--inventory", str(self.root / "rgb/inventory.json"),
                "--config", str(config), "--prepare-only"]
        out = self.root / "launch"
        result = subprocess.run(argv + ["--out", str(out), "--expected-approved", "5"],
                                text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(sha256(source), original)
        self.assertEqual(sha256(out / "inputs/2026-11-17-decisions.private.json"), original)
        expected = {q["question_id"] for q in read_json(questions)["questions"]}
        self.assertEqual({q["question_id"] for q in read_json(out / "public/questions.json")["questions"]}, expected)
        self.assertFalse((out / "archive-index").exists())
        self.assertFalse((out / "inference").exists())
        bad = self.root / "wrong-count"
        result = subprocess.run(argv + ["--out", str(bad), "--expected-approved", "4"],
                                text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(bad.exists())

    def test_one_case_export_keeps_reference_and_supports_full_e2e_scoring(self):
        questions, answers, index = self.approve()
        original = read_json(answers)
        identifier = next(q["question_id"] for q in read_json(questions)["questions"] if q["region_id"] == "change")
        qa.export(self.root / "review/reviewed.json", self.root / "smoke-public", question_ids=[identifier])
        subset = read_json(self.root / "smoke-public/answers.private.json")
        self.assertEqual(subset["cases"], [r for r in original["cases"] if r["question_id"] == identifier])
        self.assertEqual(subset["review_source"], original["review_source"])
        smoke_q = self.root / "smoke-public/questions.json"
        smoke_a = self.root / "smoke-public/answers.private.json"
        order = ("photos_only", "bounded_agentic_retrieval", "fixed_retrieval")
        metrics.plan(smoke_q, smoke_a, self.inventory, self.config, self.root / "smoke-plan", archive_index=index,
                     synthetic=True, conditions=order, condition_major=True)
        demo.runner.run(smoke_q, self.config, self.root / "smoke", archive_index=index,
                        pool_factory=demo.SyntheticPool, synthetic=True, conditions=order, condition_major=True)
        metrics.score(smoke_q, smoke_a, self.root / "smoke", self.root / "smoke-plan/plan.private.json", self.root / "smoke-score")
        report = read_json(self.root / "smoke-score/metrics.private.json")
        self.assertEqual(len(report["cases"]), 3)
        self.assertTrue(all(row["correct"] for row in report["cases"]))
        with self.assertRaisesRegex(ValueError, "human-approved"):
            qa.export(self.root / "review/reviewed.json", self.root / "bad-subset", question_ids=["not-approved"])

    def test_launcher_reuses_frozen_inputs_for_new_smoke_and_continuation(self):
        _, _, index = self.approve()
        curation.inventory(self.histories, self.root / "rgb", candidates_path=self.candidates,
                           qa_policy=curation.rgb_policy.POLICY)
        inventory = self.root / "rgb/inventory.json"
        config = self.root / "config.json"
        write_json(config, self.config)
        script = Path(__file__).resolve().parents[1] / "scripts/2026-11-17-run-approved-www.py"
        spec = importlib.util.spec_from_file_location("www_launch_fixture", script)
        launch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launch)
        args = ["--decisions", str(self.root / "decisions.private.json"), "--candidates", str(self.candidates),
                "--corpus", str(self.root / "archive/corpus.json"), "--inventory", str(inventory),
                "--config", str(config), "--expected-approved", "5"]
        prior = self.root / "prior"
        self.assertEqual(launch.main(args + ["--out", str(prior), "--prepare-only"]), 0)
        (prior / "archive-index").mkdir()
        shutil.copyfile(index, prior / "archive-index/index.json")
        smoke_id = next(q["question_id"] for q in read_json(prior / "public/questions.json")["questions"]
                        if q["region_id"] == "change")
        qa.export(prior / "review/reviewed.json", prior / "smoke-public", question_ids=[smoke_id])
        order = ("photos_only", "bounded_agentic_retrieval", "fixed_retrieval")
        for folder, plan in (("public", "plan"), ("smoke-public", "smoke-plan")):
            metrics.plan(prior / folder / "questions.json", prior / folder / "answers.private.json", inventory,
                         self.config, prior / plan, archive_index=prior / "archive-index/index.json", synthetic=True,
                         include_reviewed=True, conditions=order, condition_major=True)
        originals = {str(p): sha256(p) for p in prior.rglob("*") if p.is_file()}
        calls = []
        cli = importlib.import_module("facade_change.agentic.2026-11-15_www")

        def command(repo, *arguments, allow_errors=False):
            calls.append(arguments[:2])
            self.assertEqual(arguments[0], "www")
            values = cli.parser().parse_args(list(map(str, arguments[1:])))
            if values.action == "plan":
                result = metrics.plan(values.questions, values.answers, values.inventory, self.config, values.out,
                    archive_index=values.archive_index, synthetic=True, include_reviewed=values.include_reviewed,
                    conditions=values.conditions, condition_major=values.condition_major)
            elif values.action == "run":
                result = demo.runner.run(values.questions, self.config, values.out, archive_index=values.archive_index,
                    pool_factory=demo.SyntheticPool, synthetic=True, conditions=values.conditions, condition_major=values.condition_major)
            elif values.action == "score":
                result = metrics.score(values.questions, values.answers, values.run, values.plan, values.out)
            else:
                self.fail("Prepared inputs must not rebuild the index or export reviews")
            return int(result["status"] == "completed_with_errors")

        out = self.root / "reused"
        with patch.object(launch, "command", side_effect=command):
            self.assertEqual(launch.main(args + ["--prepared-from", str(prior), "--out", str(out), "--smoke-only"]), 0)
            self.assertFalse((out / "inference").exists())
            self.assertEqual(launch.main(args + ["--out", str(out), "--continue"]), 0)
        for folder in ("public", "smoke-public"):
            for filename in ("questions.json", "answers.private.json"):
                self.assertEqual(sha256(prior / folder / filename), sha256(out / folder / filename))
        self.assertEqual(originals, {str(p): sha256(p) for p in prior.rglob("*") if p.is_file()})
        self.assertEqual(calls.count(("www", "run")), 2)
        self.assertEqual(calls.count(("www", "score")), 2)
        rows = read_json(out / "inference/results.json")["results"]
        self.assertEqual([row["condition"] for row in rows], [c for c in order for _ in range(5)])
        self.assertEqual(len(read_json(out / "score/metrics.private.json")["cases"]), 15)
        with (prior / "archive-index/index.json").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "индекс"):
            launch.main(args + ["--prepared-from", str(prior), "--out", str(self.root / "changed"), "--prepare-only"])
        self.assertFalse((self.root / "changed").exists())

    def test_interrupted_condition_pass_scores_ac_and_marks_b_comparisons_pending(self):
        questions, answers, index = self.approve()
        order = ("photos_only", "bounded_agentic_retrieval", "fixed_retrieval")
        metrics.plan(questions, answers, self.inventory, self.config, self.root / "phased-plan",
                     archive_index=index, synthetic=True, conditions=order, condition_major=True)
        original = demo.runner._condition

        def before_b(question, condition, *args, **kwargs):
            if condition == "fixed_retrieval":
                raise KeyboardInterrupt("synthetic stop before B")
            return original(question, condition, *args, **kwargs)

        with patch.object(demo.runner, "_condition", side_effect=before_b), self.assertRaises(KeyboardInterrupt):
            demo.runner.run(questions, self.config, self.root / "partial", archive_index=index,
                            pool_factory=demo.SyntheticPool, synthetic=True, conditions=order, condition_major=True)
        metrics.score(questions, answers, self.root / "partial", self.root / "phased-plan/plan.private.json",
                      self.root / "partial-score")
        report = read_json(self.root / "partial-score/metrics.private.json")
        self.assertEqual(len(report["cases"]), 15)
        self.assertEqual(report["summaries"]["development/all"]["fixed_retrieval"]["outcome_counts"], {"missing": 5})
        pairs = report["paired_building_differences"]["development/all"]
        self.assertEqual(sum(p.get("status") == "pending_condition" for p in pairs), 2)
        ac = next(p for p in pairs if p["contrast"] == "bounded_agentic_retrieval minus photos_only")
        self.assertIsNotNone(ac["mean_paired_difference"])

    def test_reviewed_endpoint_costs_support_and_all_error_denominators(self):
        questions, answers, index = self.approve()
        class Responses(demo.SyntheticPool):
            def call(self, role, payload):
                if role == "qwen" and payload["geometry"]["frame"]["region_id"] == "error":
                    return {"status": "error", "error": "explicit fixture worker failure", "usage": {"model_calls": 1}}
                result = super().call(role, payload)
                if role == "qwen":
                    value = json.loads(result["text"])
                    region = payload["geometry"]["frame"]["region_id"]
                    if value["type"] == "final" and region in {"stable", "occluded"}:
                        value.update(decision="no_change" if region == "stable" else "insufficient_data", boxes=[], limitations=["Fixture occlusion"] if region == "occluded" else [])
                        result["text"] = json.dumps(value)
                return result
        demo.runner.run(questions, self.config, self.root / "run", archive_index=index, pool_factory=Responses, synthetic=True)
        metrics.score(questions, answers, self.root / "run", self.root / "plan/plan.private.json", self.root / "score")
        result = read_json(self.root / "score/metrics.private.json")
        for condition in demo.runner.CONDITIONS:
            summary = result["summaries"]["development/all"][condition]
            self.assertEqual(summary["questions"], 5)
            self.assertEqual(summary["structured_accuracy"], .8)
            self.assertEqual(summary["building_macro_accuracy"], .8)
            self.assertEqual(summary["outcome_counts"], {"completed": 3, "abstained": 1, "error": 1})
            self.assertEqual(summary["legitimate_abstentions"], 1)
            self.assertEqual(summary["known_support_recall"], 0. if condition == "photos_only" else 1.)
        self.assertEqual(len(result["cases"]), 15)
        contrasts = {"fixed_retrieval minus photos_only", "bounded_agentic_retrieval minus fixed_retrieval",
                     "bounded_agentic_retrieval minus photos_only"}
        for comparisons in result["paired_building_differences"].values():
            self.assertEqual({r["contrast"] for r in comparisons}, contrasts)
        direct = next(r for r in result["paired_building_differences"]["development/all"]
                      if r["contrast"] == "bounded_agentic_retrieval minus photos_only")
        self.assertEqual(direct["mean_paired_difference"], 0.)
        self.assertEqual(direct["paired_confidence_intervals"]["building_macro_accuracy"]["interval"], [0., 0.])
        self.assertEqual(direct["paired_confidence_intervals"]["change_stability_macro_f1"]["interval"], [0., 0.])
        self.assertTrue(all("physical_event_f1" in r for r in result["localization"]["photos_only"]))
        self.assertEqual(result["localization_by_cohort_and_stratum"]["photos_only"]["evaluation/all"][0]["eligible_events"], 0)
        diagnostic = read_json(self.root / "score/annotation-diagnostics.private.json")
        self.assertEqual(diagnostic["access"], qa.PRIVATE)
        self.assertIn("change_proposal_precision", diagnostic)
        self.assertTrue((self.root / "score/reviewed-overlays.private/gallery.html").is_file())
        blind_header = (self.root / "score/blind-review.private.csv").read_text().splitlines()[0]
        self.assertNotIn("condition", blind_header)
        self.assertEqual(len(read_json(self.root / "score/blind-key.private.json")["cases"]), 15)

    def test_interrupted_run_keeps_missing_questions_in_denominator(self):
        questions, answers, index = self.approve()
        class Interrupted(demo.SyntheticPool):
            def call(self, role, payload):
                if role == "qwen" and payload["geometry"]["frame"]["region_id"] == "change":
                    raise KeyboardInterrupt("synthetic interruption")
                return super().call(role, payload)
        with self.assertRaises(KeyboardInterrupt):
            demo.runner.run(questions, self.config, self.root / "run", archive_index=index, pool_factory=Interrupted, synthetic=True)
        metrics.score(questions, answers, self.root / "run", self.root / "plan/plan.private.json", self.root / "score")
        result = read_json(self.root / "score/metrics.private.json")
        for condition in demo.runner.CONDITIONS:
            summary = result["summaries"]["development/all"][condition]
            self.assertEqual(summary["questions"], 5)
            self.assertGreater(summary["outcome_counts"].get("missing", 0), 0)
        direct = next(r for r in result["paired_building_differences"]["development/all"]
                      if r["contrast"] == "bounded_agentic_retrieval minus photos_only")
        all_conditions = result["summaries"]["development/all"]
        expected = all_conditions["bounded_agentic_retrieval"]["building_macro_accuracy"] - all_conditions["photos_only"]["building_macro_accuracy"]
        self.assertAlmostEqual(direct["mean_paired_difference"], expected)
        self.assertEqual(direct["paired_confidence_intervals"], metrics.paired_intervals(
            result["cases"], "photos_only", "bounded_agentic_retrieval"))

    def test_equivalent_observation_and_archival_answer_contract(self):
        row = next(r for r in self.rows if r["task"] == "earliest_evidence")
        label = {"assessment": "earliest_presence", "observability": "unknown", "reference_answer": "Fixture",
            "earliest_observation_id": row["observations"][0]["observation_id"], "present_by": "2009", "onset": "unknown",
            "equivalent_observation_ids": [row["observations"][-1]["observation_id"]]}
        with self.assertRaisesRegex(ValueError, "preserve the same date precision"):
            qa._checked_label(row, label, Fingerprints())
        case = {**row, "task": "archival_qa"}
        label = {"assessment": "archival_fact", "observability": "unknown", "reference_answer": "Fixture archive answer",
            "supporting_evidence_ids": ["fixture-source"], "acceptable_structured_answers": [
                {"decision": "supported", "observation_ids": [], "date_interval": {"start": "2010-02", "end": "2010-03", "precision": "month", "uncertain": True}}]}
        checked = qa._checked_label(case, label, Fingerprints())
        self.assertEqual(checked["events"], [])
        self.assertTrue(metrics.structured_correct(case, checked, {"status": "completed", "answer": label["acceptable_structured_answers"][0]}))


if __name__ == "__main__":
    unittest.main()
