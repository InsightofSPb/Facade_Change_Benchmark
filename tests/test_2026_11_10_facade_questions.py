"""Hand-checkable CPU acceptance for private curation and fixed-cohort QA metrics."""
import copy
import csv
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from facade_change.agentic.reasoning import _public_result
from facade_change.agentic.retrieval import KnowledgeBase
from facade_change.agentic.store import reject_forbidden
from facade_change.io import read_json, sha256, write_json


qa = importlib.import_module("facade_change.agentic.2026-11-10_questions")
evaluation = importlib.import_module("facade_change.agentic.2026-11-10_qa_evaluation")
demo = importlib.import_module("facade_change.agentic.2026-11-10_qa_demo")


class FacadeQuestionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "demo"
        demo.demo(self.run)
        self.questions_path = self.run / "export/questions.json"
        self.answers_path = self.run / "export/answers.private.json"
        self.public = read_json(self.questions_path)
        self.private = read_json(self.answers_path)
        self.questions = {r["region_id"]: r for r in self.public["questions"]}
        self.predictions = read_json(self.run / "saved-predictions.json")
        self.result = read_json(self.run / "evaluation/evaluation.json")

    def metric(self, result=None, k=10, iou=.25):
        result = self.result if result is None else result
        return next(r for r in result["pooled"]["localization"] if r["k"] == k and r["iou_threshold"] == iou)

    def evaluate(self, predictions, name="evaluation", questions=None, answers=None):
        path = self.root / (name + "-predictions.json")
        write_json(path, predictions)
        evaluation.evaluate(questions or self.questions_path, answers or self.answers_path, path, self.root / name)
        return read_json(self.root / name / "evaluation.json")

    def test_hand_checked_counts_deduplicate_events_and_do_not_pad_rankings(self):
        row = self.metric()
        self.assertEqual([row[k] for k in ("eligible_events", "matched", "missed", "available_predictions", "unmatched_predictions")],
                         [4, 1, 3, 4, 3])
        self.assertEqual((row["region_recall"], row["region_precision"]), (.25, .25))
        first = self.metric(k=1)
        self.assertEqual((first["matched"], first["available_predictions"], first["unmatched_predictions"]), (1, 2, 1))
        self.assertEqual(self.result["buildings"]["b1"]["localization"][3]["eligible_events"], 2)
        self.assertEqual(len(self.result["buildings"]), 7)
        for k in (3, 5, 10):
            self.assertEqual(self.metric(k=k)["available_predictions"], 4)

    def test_earliest_presence_is_not_an_onset_or_temporal_change(self):
        cases = read_json(self.run / "candidates/candidates.json")["candidates"]
        earliest = next(r for r in cases if r["candidate_type"] == "earliest_damage")
        self.assertIn("2009", earliest["provisional_answer"])
        self.assertIn("неизвестно", earliest["provisional_answer"])
        label = next(r["reviewed_labels"] for r in self.private["cases"] if r["question_id"] == earliest["question_id"])
        self.assertEqual((label["present_by"], label["onset"], label["events"]), ("2009", "unknown", []))
        self.assertEqual(self.result["pooled"]["coverage"]["earliest_evidence_correct"], 1)
        self.assertNotIn(earliest["question_id"], [r["question_id"] for e in self.result["pooled"]["events"] for r in e["representations"]])
        wrong = copy.deepcopy(self.predictions)
        next(r for r in wrong["cases"] if r["question_id"] == earliest["question_id"])["decision"] = "no_change"
        self.assertEqual(self.evaluate(wrong)["pooled"]["coverage"]["earliest_evidence_correct"], 0)

    def test_observability_is_separate_from_underlying_state_errors_and_missing(self):
        counts = self.result["pooled"]["coverage"]
        self.assertEqual((counts["missing"], counts["technical_errors"], counts["justified_abstentions"],
                          counts["unsupported_confident_decisions"], counts["false_alarm_cases"]), (2, 1, 1, 1, 1))
        self.assertEqual(counts["false_alarm_rate"], 1.)
        hidden = next(r["reviewed_labels"] for r in self.private["cases"]
                      if r["question_id"] == self.questions["occluded"]["question_id"])
        self.assertEqual(hidden["underlying_state"]["assessment"], "confirmed_change")
        self.assertEqual(hidden["events"], [])
        changed = copy.deepcopy(self.predictions)
        row = next(r for r in changed["cases"] if r["question_id"] == self.questions["occluded"]["question_id"])
        row.update(status="error", decision="insufficient_data", error="worker failed")
        result = self.evaluate(changed)
        self.assertEqual(result["pooled"]["coverage"]["justified_abstentions"], 0)
        self.assertEqual(result["pooled"]["coverage"]["technical_errors"], 2)
        self.assertEqual(self.metric(result)["eligible_events"], 4)

    def test_mask_area_uses_fixed_reference_and_never_box_area(self):
        row = self.metric()
        one, five = row["area_subsets"]
        self.assertEqual((one["minimum_fraction"], one["eligible_events"], one["matched"]), (.01, 3, 1))
        self.assertEqual((five["minimum_fraction"], five["eligible_events"], five["matched"]), (.05, 1, 1))
        self.assertEqual(self.result["pooled"]["area_availability"], {"available": 3, "unavailable_box_only": 1})
        events = {r["event_id"]: r for r in self.result["pooled"]["events"]}
        self.assertEqual(events["e1"]["mask_area"]["reference_pixels"], 100)
        self.assertEqual(events["e1"]["mask_area"]["shape"], [10, 10])
        self.assertIsNone(events["e3"]["mask_area"])
        self.assertEqual(row["eligible_events"], 4)

    def test_no_private_fields_or_archival_photo_reach_public_manifest_or_online_evidence(self):
        text = self.questions_path.read_text()
        for value in ("provisional_answer", "reference_answer", "private_notes", "generation_reasons",
                      "annotation_evidence", "underlying_state", "private-underlying-state", "fixture-private-archive"):
            self.assertNotIn(value, text)
        qa.validate_public(self.public)
        with self.assertRaises(ValueError):
            reject_forbidden({"provenance": self.private})
        with self.assertRaises(ValueError):
            _public_result({"status": "ok", "evidence": [self.private]})
        with self.assertRaises(ValueError):
            _public_result({"status": "ok", "limitations": [json.dumps({"reference_answer": "private answer"})]})
        knowledge = KnowledgeBase(self.root / "knowledge")
        with self.assertRaises(ValueError):
            knowledge.import_document({"document_id": "leak", "available_at": "2026", "concepts": [], "relations": [],
                "source": {"kind": "user_document", "title": "Forbidden fixture", "locator": "fixture"},
                "fragments": [{"fragment_id": "f", "text": json.dumps(self.private)}]})
        dirty = copy.deepcopy(self.public)
        dirty["questions"][0]["observations"][0]["geometry"]["private_payload"] = "not allowed"
        with self.assertRaises(ValueError):
            qa.validate_public(dirty)

    def test_review_revisions_do_not_overwrite_candidates_and_only_approved_export(self):
        candidates = self.run / "candidates/candidates.json"
        previous = self.run / "review/reviewed.json"
        original = (sha256(candidates), sha256(previous))
        templates = {r["question_id"]: r["question"] for r in read_json(candidates)["candidates"]}
        self.assertTrue(all(r["question"] == templates[r["question_id"]] for r in self.public["questions"]))
        identifier = self.questions["stable"]["question_id"]
        decisions = self.root / "decisions.json"
        write_json(decisions, {"candidate_sha256": sha256(candidates), "decisions": [
            {"question_id": identifier, "expected_revision": 1, "status": "rejected", "reviewer": "fixture-reviewer-2",
             "private_notes": "Synthetic correction"}]})
        qa.review(candidates, decisions, self.root / "revision", previous)
        revised = read_json(self.root / "revision/reviewed.json")
        self.assertEqual(revised["states"][identifier]["revision"], 2)
        self.assertEqual(revised["history"], [{"path": str(previous), "sha256": original[1]}])
        qa.export(self.root / "revision/reviewed.json", self.root / "export-2")
        self.assertNotIn(identifier, [r["question_id"] for r in read_json(self.root / "export-2/questions.json")["questions"]])
        self.assertEqual(original, (sha256(candidates), sha256(previous)))
        with self.assertRaisesRegex(ValueError, "Stale"):
            qa.review(candidates, decisions, self.root / "stale", self.root / "revision/reviewed.json")

    def test_coordinate_mismatch_and_duplicate_overlap_predictions_cannot_add_hits(self):
        wrong = copy.deepcopy(self.predictions)
        row = next(r for r in wrong["cases"] if r["question_id"] == self.questions["change"]["question_id"])
        for prediction in row["regions"]:
            prediction.update(coordinate_system="observation_native", observation_id="change-2015")
        result = self.evaluate(wrong, "wrong-frame")
        self.assertEqual((self.metric(result)["matched"], self.metric(result)["missed"]), (0, 4))
        self.assertEqual(result["pooled"]["coverage"]["coordinate_issue_predictions"], 3)
        repeated = copy.deepcopy(self.predictions)
        repeated["cases"].append({"question_id": self.questions["overlap"]["question_id"], "status": "completed", "decision": "change",
            "regions": [{"prediction_id": "overlapping-crop-duplicate", "rank": 1, "coordinate_system": "region_crop_pixels",
                         "region_id": "overlap", "bbox_xyxy": [1, 1, 4, 4]}]})
        result = self.evaluate(repeated, "duplicate")
        self.assertEqual((self.metric(result)["eligible_events"], self.metric(result)["matched"], self.metric(result)["unmatched_predictions"]), (4, 1, 4))

    def test_single_saved_agentic_box_is_one_prediction_at_every_k(self):
        case = self.questions["change"]
        path = self.root / "result.json"
        write_json(path, {"status": "completed", "decision": "change", "localization": {
            "coordinate_system": "region_crop_pixels", "region_id": "change", "bbox_xyxy": [3, 1, 6, 4],
            "kind": "vlm_bbox_hypothesis", "evidence_id": "saved-fixture-evidence"}})
        document = {"kind": "facade_qa_predictions", "model_manifest_sha256": sha256(self.questions_path),
            "ranking_source": {"kind": "single_box", "source": "existing saved agentic result.json"},
            "cases": [{"question_id": case["question_id"], "result_path": str(path), "result_sha256": sha256(path)}]}
        result = self.evaluate(document)
        for row in result["pooled"]["localization"]:
            self.assertEqual((row["available_predictions"], row["matched"], row["missed"]), (1, 1, 3))

    def test_matching_is_one_to_one_maximum_cardinality_and_broad_box_is_not_a_hit(self):
        frame = {"coordinate_system": "region_crop_pixels", "region_id": "r", "question_id": "q"}
        events = {"a": {"representations": [{**frame, "bbox_xyxy": [0, 0, 2, 2]}]},
                  "b": {"representations": [{**frame, "bbox_xyxy": [1, 0, 3, 2]}]}}
        broad = [{**frame, "prediction_id": "broad", "bbox_xyxy": [0, 0, 10, 10]}]
        self.assertEqual(evaluation.match_one_to_one(broad, events, .25), [])
        one = [{**frame, "prediction_id": "one", "bbox_xyxy": [0, 0, 3, 2]}]
        self.assertEqual(len(evaluation.match_one_to_one(one, events, .5)), 1)
        two = [*one, {**frame, "prediction_id": "two", "bbox_xyxy": [0, 0, 2, 2]}]
        matched = evaluation.match_one_to_one(two, events, .5)
        self.assertEqual(len(matched), 2)
        self.assertEqual(matched, evaluation.match_one_to_one(two, events, .5))

    def test_pixel_scores_require_actual_compatible_temporal_masks(self):
        document = copy.deepcopy(self.predictions)
        row = next(r for r in document["cases"] if r["question_id"] == self.questions["change"]["question_id"])
        event = next(r["reviewed_labels"]["events"][0] for r in self.private["cases"] if r["question_id"] == row["question_id"])
        row["regions"][0].update({key: event[key] for key in ("mask_path", "mask_sha256", "mask_task", "support_path", "support_sha256")})
        result = self.evaluate(document, "masks")
        self.assertEqual(self.metric(result)["matches"][0]["pixels"]["status"], "evaluated")
        self.assertEqual(self.metric(result)["matches"][0]["pixels"]["iou"], 1.)
        row["regions"][0]["mask_task"] = "semantic_damage"
        result = self.evaluate(document, "semantic-mask")
        self.assertEqual(self.metric(result)["matches"][0]["pixels"]["reason"], "mask_is_not_explicit_temporal_change")
        self.assertEqual(self.metric()["matches"][0]["pixels"]["status"], "not_available")

    def test_fixed_pixel_domain_requires_complete_prediction_support(self):
        identifier = self.questions["change"]["question_id"]
        event = next(r["reviewed_labels"]["events"][0] for r in self.private["cases"] if r["question_id"] == identifier)
        mask = np.zeros((10, 10), bool)
        mask[1:4, 3:9] = True
        path = self.root / "prediction.npy"
        np.save(path, mask)
        document = {**self.predictions, "cases": [{"question_id": identifier, "status": "completed", "decision": "change",
            "regions": [{"prediction_id": "wider-mask", "rank": 1, "bbox_xyxy": [3, 1, 9, 4],
                         "coordinate_system": event["coordinate_system"], "region_id": event["region_id"],
                         "mask_task": "temporal_change", "mask_path": str(path), "mask_sha256": sha256(path)}]}]}
        area_subsets = None
        for name, count in (("full", 100), ("tiny", 1), ("empty", 0)):
            support = np.zeros((10, 10), bool)
            if name == "full":
                support[:] = True
            elif name == "tiny":
                support[1, 3] = True
            support_path = self.root / (name + ".npy")
            np.save(support_path, support)
            document["cases"][0]["regions"][0].update(support_path=str(support_path), support_sha256=sha256(support_path))
            result = self.evaluate(document, "domain-" + name)
            row = self.metric(result)
            self.assertEqual((row["matched"], row["missed"], row["available_predictions"]), (1, 3, 1))
            self.assertEqual(row["matches"][0]["iou"], .5)
            if area_subsets is None:
                area_subsets = row["area_subsets"]
            self.assertEqual(row["area_subsets"], area_subsets)
            pixels = row["matches"][0]["pixels"]
            if name == "full":
                self.assertEqual((pixels["tp"], pixels["fp"], pixels["valid_pixels"], pixels["iou"]), (9, 9, 100, .5))
            else:
                self.assertEqual((pixels["status"], pixels["reason"]), ("not_available", "incomplete_prediction_support"))
                self.assertEqual((pixels["required_pixels"], pixels["covered_pixels"], pixels["coverage_fraction"]), (100, count, count/100))
                self.assertNotIn("iou", pixels)
                self.assertNotIn("f1", pixels)

    def test_mixed_saved_result_failures_are_isolated_from_the_fixed_cohort(self):
        document = copy.deepcopy(self.predictions)
        valid = self.root / "valid-result.json"
        write_json(valid, {"status": "completed", "decision": "change", "localization": {
            "coordinate_system": "region_crop_pixels", "region_id": "change", "bbox_xyxy": [3, 1, 6, 4]}})
        malformed = self.root / "malformed-result.json"
        write_json(malformed, {"status": "completed", "decision": "change", "localization": "not an object"})
        missing = self.root / "missing-result.json"
        for region, path in (("change", valid), ("missing", missing), ("error", malformed)):
            identifier = self.questions[region]["question_id"]
            document["cases"] = [r for r in document["cases"] if r["question_id"] != identifier]
            document["cases"].append({"question_id": identifier, "result_path": str(path),
                "ranking_source": {"kind": "single_box", "source": "saved test result"}})
        result = self.evaluate(document, "mixed-results")
        coverage = result["pooled"]["coverage"]
        self.assertEqual((coverage["cohort_cases"], coverage["completed"], coverage["abstained"],
                          coverage["missing"], coverage["technical_errors"]), (8, 4, 1, 2, 1))
        self.assertEqual(coverage["justified_abstentions"], 1)
        self.assertEqual(coverage["response_coverage"], 5/8)
        metric = self.metric(result)
        self.assertEqual((metric["eligible_events"], metric["matched"], metric["missed"], metric["unmatched_predictions"]), (4, 1, 3, 1))
        cases = {r["question_id"]: r for r in result["pooled"]["cases"]}
        for region, path, status in (("missing", missing, "missing"), ("error", malformed, "error")):
            case = cases[self.questions[region]["question_id"]]
            self.assertEqual(case["status"], status)
            self.assertIn(str(path), case["error"])
            self.assertIn(case["question_id"], case["error"])

    def test_reviewed_question_is_versioned_exported_and_preserved_when_omitted(self):
        candidates = self.run / "candidates/candidates.json"
        previous = self.run / "review/reviewed.json"
        identifier = self.questions["change"]["question_id"]
        original = self.questions["change"]["question"]
        wording = "Какие видимые различия есть между этими наблюдениями участка?"
        hashes = (sha256(candidates), sha256(previous))
        state = read_json(previous)["states"][identifier]
        for revision, override in ((1, wording), (2, None), (3, original)):
            decision = {"question_id": identifier, "expected_revision": revision, "status": "approved",
                        "reviewer": "synthetic-wording-reviewer", "private_notes": "Synthetic wording revision", "label": state["label"]}
            if override is not None:
                decision["reviewed_question"] = override
            decisions = self.root / f"wording-{revision}.json"
            write_json(decisions, {"candidate_sha256": sha256(candidates), "decisions": [decision]})
            out = self.root / f"review-{revision}"
            qa.review(candidates, decisions, out, previous)
            previous = out / "reviewed.json"
            effective = wording if revision < 3 else original
            self.assertEqual(read_json(previous)["states"][identifier]["reviewed_question"], effective)
            with (out / "review.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual(next(r for r in csv.DictReader(stream) if r["question_id"] == identifier)["question"], effective)
            self.assertIn(effective, (out / "gallery.html").read_text())
            exported = self.root / f"export-wording-{revision}"
            qa.export(previous, exported)
            public = read_json(exported / "questions.json")
            self.assertEqual(next(r for r in public["questions"] if r["question_id"] == identifier)["question"], effective)
            self.assertNotIn("reviewed_question", json.dumps(public))
            self.assertNotIn("private_notes", json.dumps(public))
            if revision == 1:
                self.assertNotEqual(sha256(exported / "questions.json"), sha256(self.questions_path))
                with self.assertRaisesRegex(ValueError, "identical fixed"):
                    self.evaluate(self.predictions, "stale-wording-predictions", exported / "questions.json", exported / "answers.private.json")
        self.assertEqual(hashes, (sha256(candidates), sha256(self.run / "review/reviewed.json")))

    def test_malformed_individual_outputs_are_errors_not_abstentions(self):
        identifier = self.questions["occluded"]["question_id"]
        payloads = ["{\"status\":", "[]", '{"status":"completed","decision":"change","regions":{}}',
                    '{"status":"completed","decision":"no_change","confidence":NaN}']
        for index, payload in enumerate(payloads):
            with self.subTest(payload=payload):
                path = self.root / f"bad-result-{index}.json"
                path.write_text(payload)
                document = copy.deepcopy(self.predictions)
                document["cases"] = [r for r in document["cases"] if r["question_id"] != identifier]
                document["cases"].append({"question_id": identifier, "result_path": str(path),
                    "ranking_source": {"kind": "single_box", "source": "malformed fixture"}})
                result = self.evaluate(document, f"bad-json-{index}")
                coverage = result["pooled"]["coverage"]
                self.assertEqual((coverage["technical_errors"], coverage["justified_abstentions"]), (2, 0))
                self.assertEqual((self.metric(result)["matched"], self.metric(result)["missed"]), (1, 3))
                error = next(r for r in result["pooled"]["cases"] if r["question_id"] == identifier)["error"]
                self.assertIn(str(path), error)
        for index, regions in enumerate(({}, [None])):
            document = copy.deepcopy(self.predictions)
            row = next(r for r in document["cases"] if r["question_id"] == identifier)
            row.update(status="completed", decision="change", regions=regions)
            result = self.evaluate(document, f"bad-inline-{index}")
            self.assertEqual(result["pooled"]["coverage"]["technical_errors"], 2)
        path = self.root / "unreadable.json"
        write_json(path, {"status": "abstained", "decision": "insufficient_data"})
        document["cases"] = [r for r in document["cases"] if r["question_id"] != identifier]
        document["cases"].append({"question_id": identifier, "result_path": str(path),
                                "ranking_source": {"kind": "single_box", "source": "unreadable fixture"}})
        def unreadable(value):
            if Path(value) == path:
                raise PermissionError("Synthetic denied read")
            return read_json(value)
        with patch.object(evaluation, "read_json", side_effect=unreadable):
            result = self.evaluate(document, "unreadable-case")
        self.assertEqual((result["pooled"]["coverage"]["technical_errors"], result["pooled"]["coverage"]["justified_abstentions"]), (2, 0))

    def test_optional_prediction_artifact_failures_keep_valid_box_matches(self):
        identifier = self.questions["change"]["question_id"]
        event = next(r["reviewed_labels"]["events"][0] for r in self.private["cases"] if r["question_id"] == identifier)
        for index, failure in enumerate(("missing_mask", "wrong_hash", "corrupt_mask", "wrong_grid", "missing_support")):
            document = copy.deepcopy(self.predictions)
            prediction = next(r for r in document["cases"] if r["question_id"] == identifier)["regions"][0]
            prediction.update({k: event[k] for k in ("mask_path", "mask_sha256", "mask_task", "support_path", "support_sha256")})
            path = self.root / (failure + ".npy")
            if failure == "missing_mask":
                prediction["mask_path"] = str(path)
            elif failure == "missing_support":
                prediction["support_path"] = str(path)
            elif failure == "wrong_hash":
                prediction["mask_sha256"] = "0" * 64
            else:
                if failure == "wrong_grid":
                    np.save(path, np.ones((3, 3), bool))
                else:
                    path.write_bytes(b"not a mask")
                prediction.update(mask_path=str(path), mask_sha256=sha256(path))
            result = self.evaluate(document, f"optional-artifact-{index}")
            row = self.metric(result, k=1)
            self.assertEqual((row["matched"], row["missed"]), (1, 3))
            pixels = row["matches"][0]["pixels"]
            self.assertEqual((pixels["status"], pixels["reason"]), ("error", "prediction_pixel_artifact_error"))
            self.assertNotIn("iou", pixels)
            case = next(r for r in result["pooled"]["cases"] if r["question_id"] == identifier)
            self.assertEqual(case["status"], "completed")
            self.assertTrue(case["pixel_diagnostics"])

    def test_missing_reviewed_pixel_domain_is_unavailable_without_losing_box_match(self):
        event = next(r["reviewed_labels"]["events"][0] for r in self.private["cases"] if r["question_id"] == self.questions["change"]["question_id"])
        truth = {**event, "question_id": "q"}
        truth.pop("support_path")
        prediction = {**event, "question_id": "q", "prediction_id": "p"}
        match = evaluation.match_one_to_one([prediction], {"e1": {"representations": [truth]}}, .25)[0]
        self.assertEqual(match["iou"], 1.)
        self.assertEqual(match["pixels"], {"status": "not_available", "reason": "evaluation_domain_unavailable"})

    def test_global_integrity_and_private_leakage_still_abort(self):
        duplicate = copy.deepcopy(self.predictions)
        duplicate["cases"].append(duplicate["cases"][0])
        unknown = copy.deepcopy(self.predictions)
        unknown["cases"][0]["question_id"] = "not-in-cohort"
        invalid_global = {**self.predictions, "ranking_source": {"kind": "gt_iou", "source": "forbidden"}, "cases": []}
        leak = copy.deepcopy(self.predictions)
        leak["cases"][0]["private_notes"] = "not a per-case error"
        for index, document in enumerate((duplicate, unknown, invalid_global, leak)):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.evaluate(document, f"fatal-{index}")
        path = self.root / "leaked-result.json"
        write_json(path, {"status": "completed", "decision": "no_change", "reference_answer": "private"})
        document = {**self.predictions, "ranking_source": {"kind": "single_box", "source": "fixture"},
                    "cases": [{"question_id": self.questions["change"]["question_id"], "result_path": str(path)}]}
        with self.assertRaisesRegex(ValueError, "reference_answer"):
            self.evaluate(document, "fatal-saved-leak")
        # A malformed value must not hide a later private field during validation.
        path.write_text('{"confidence":NaN,"reference_answer":"private"}')
        with self.assertRaisesRegex(ValueError, "reference_answer"):
            self.evaluate(document, "fatal-saved-leak-after-malformed-value")

    def test_inconsistent_reviewed_truth_and_changed_gt_artifacts_remain_fatal(self):
        identifier = self.questions["change"]["question_id"]
        private = copy.deepcopy(self.private)
        row = next(r for r in private["cases"] if r["question_id"] == identifier)
        row["reviewed_labels"]["events"][0]["region_id"] = "wrong-owner"
        row["review"]["label"] = copy.deepcopy(row["reviewed_labels"])
        path = self.root / "altered-answers.private.json"
        write_json(path, private)
        with self.assertRaisesRegex(ValueError, "immutable review"):
            self.evaluate(self.predictions, "bad-gt", answers=path)
        event = next(r["reviewed_labels"]["events"][0] for r in self.private["cases"] if r["question_id"] == identifier)
        np.save(event["mask_path"], np.zeros((10, 10), bool))
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            self.evaluate(self.predictions, "changed-gt-mask")

    def test_empty_unknown_and_stale_question_edits_are_rejected(self):
        candidates = self.run / "candidates/candidates.json"
        previous = self.run / "review/reviewed.json"
        identifier = self.questions["stable"]["question_id"]
        base = {"question_id": identifier, "expected_revision": 1, "status": "rejected",
                "reviewer": "synthetic-reviewer", "private_notes": "Synthetic correction"}
        invalid = [({"reviewed_question": value}, "nonempty string") for value in ("", "  ", None, 12)]
        invalid += [({"question": "silently lost?"}, "use reviewed_question"), ({"unexpected": True}, "unexpected"),
                    ({"expected_revision": 0, "reviewed_question": "Новая формулировка?"}, "Stale")]
        for index, (changes, message) in enumerate(invalid):
            decisions = self.root / f"bad-review-{index}.json"
            write_json(decisions, {"candidate_sha256": sha256(candidates), "decisions": [{**base, **changes}]})
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                qa.review(candidates, decisions, self.root / f"bad-review-out-{index}", previous)
        with self.assertRaises(ValueError):
            reject_forbidden({"reviewed_question": "Private review metadata"})

    def test_unknown_visibility_and_missing_annotations_do_not_create_stability_gt(self):
        cases = read_json(self.run / "candidates/candidates.json")["candidates"]
        self.assertTrue(all(r["observability"] == "unknown" and r["status"] == "candidate" for r in cases))
        uncertain = next(r for r in cases if r["region_id"] == "occluded")
        self.assertEqual(uncertain["candidate_type"], "uncertain_control")
        stable = next(r for r in cases if r["region_id"] == "stable")
        self.assertIn("do_not_prove_stability", stable["generation_reasons"][0])
        bad = {"assessment": "confirmed_change", "observability": "occluded", "reference_answer": "Hidden truth", "events": []}
        with self.assertRaisesRegex(ValueError, "underlying_state"):
            qa._checked_label(uncertain, bad, qa.Fingerprints())
        case = next(r for r in cases if r["region_id"] == "change")
        for box in ([1, 1, 1, 1], [False, 1, 2, 3], ["1", 1, 2, 3], [-1, 0, 2, 2]):
            label = {"assessment": "confirmed_change", "observability": "visible", "reference_answer": "Visible change",
                     "events": [{"event_id": "bad-box", "coordinate_system": "region_crop_pixels",
                                 "region_id": "change", "bbox_xyxy": box}]}
            with self.assertRaises(ValueError):
                qa._checked_label(case, label, qa.Fingerprints())

    def test_knowledge_availability_is_separate_from_task_dates(self):
        case = self.questions["change"]
        self.assertEqual((case["task_dates"], case["knowledge_cutoff"], case["task_mode"]), (["2009", "2015"], "2026", "retrospective"))
        inputs = self.run / "fixture"
        qa.generate(inputs / "histories", self.root / "as-of", annotation_targets=inputs / "annotation-targets/index.json",
                    knowledge_cutoff="2009", task_mode="as_of", available_at="observation_date")
        earlier = read_json(self.root / "as-of/candidates.json")["candidates"]
        self.assertTrue(earlier)
        self.assertTrue(all(o["date"] == "2009" for c in earlier for o in c["observations"]))
        qa.generate(inputs / "histories", self.root / "not-yet-archived", annotation_targets=inputs / "annotation-targets/index.json",
                    knowledge_cutoff="2009", task_mode="as_of", available_at="2026")
        self.assertEqual(read_json(self.root / "not-yet-archived/candidates.json")["candidates"], [])

    def test_spatial_expansion_with_identical_present_classes_is_a_change_candidate(self):
        histories, annotation_path, _ = demo.fixture(self.root / "expansion-fixture")
        index = read_json(annotation_path)
        for year, stop in ((2009, 3), (2015, 5)):
            mapping = index["observations"][f"stable-{year}"]
            artifact = annotation_path.parent / mapping["artifact_path"]
            packed = np.ones((10, 10), np.uint16)
            packed[1:stop, 1:stop] |= 2
            np.savez_compressed(artifact, packed=packed, support=np.ones((10, 10), bool))
            mapping["artifact_sha256"] = sha256(artifact)
        write_json(annotation_path, index)
        write_json(annotation_path.parent / "run.json", {"status": "completed", "synthetic_fixture": True,
                                                         "artifact_sha256": {"index.json": sha256(annotation_path)}})
        qa.generate(histories, self.root / "expansion-candidates", annotation_targets=annotation_path,
                    knowledge_cutoff="2026", task_mode="retrospective", available_at="observation_date")
        case = next(r for r in read_json(self.root / "expansion-candidates/candidates.json")["candidates"]
                    if r["region_id"] == "stable" and r["task"] == "temporal_change")
        classes = [r["categories_present"] for r in case["annotation_evidence"].values()]
        self.assertEqual(classes[0], classes[1])
        self.assertEqual(case["candidate_type"], "possible_change")
        self.assertEqual(case["status"], "candidate")

    def test_ranking_cannot_be_selected_from_gt_and_cohort_mismatch_is_rejected(self):
        document = copy.deepcopy(self.predictions)
        document["ranking_source"]["kind"] = "gt_iou"
        with self.assertRaisesRegex(ValueError, "ranking"):
            self.evaluate(document)
        document = copy.deepcopy(self.predictions)
        document["model_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "identical fixed"):
            self.evaluate(document, "different-cohort")

    def test_empty_positive_cohort_has_undefined_recall_and_retains_missing_case(self):
        identifier = self.questions["stable"]["question_id"]
        public = {**self.public, "questions": [self.questions["stable"]]}
        questions = self.root / "stable-only.json"
        write_json(questions, public)
        private = {**self.private, "model_manifest_sha256": sha256(questions),
                   "cases": [r for r in self.private["cases"] if r["question_id"] == identifier]}
        answers = self.root / "stable-only.private.json"
        write_json(answers, private)
        document = {**self.predictions, "model_manifest_sha256": sha256(questions), "cases": []}
        result = self.evaluate(document, questions=questions, answers=answers)
        self.assertEqual(result["pooled"]["coverage"]["cohort_cases"], 1)
        self.assertEqual(result["pooled"]["coverage"]["missing"], 1)
        self.assertIsNone(self.metric(result)["region_recall"])
        self.assertTrue(all(r["region_recall"] is None for r in self.metric(result)["area_subsets"]))
