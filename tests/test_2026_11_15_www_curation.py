"""Focused CPU checks for inventory, review boundaries and fixed-cohort metrics."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest

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
        qa.review(self.candidates, decision_path, self.root / "review")
        qa.export(self.root / "review/reviewed.json", self.root / "export")
        questions, answers = self.root / "export/questions.json", self.root / "export/answers.private.json"
        demo.archive.import_archive(self.package, self.root / "archive")
        demo.archive.index_archive(self.root / "archive/corpus.json", self.config, self.root / "index", pool_factory=demo.SyntheticPool, synthetic=True)
        index = self.root / "index/index.json"
        metrics.plan(questions, answers, self.inventory, self.config, self.root / "plan", archive_index=index, synthetic=True)
        return questions, answers, index

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
