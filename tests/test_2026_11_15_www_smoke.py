"""CPU transport/date regressions from the first WWW smoke; no model-quality assertions."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest

from facade_change.history_features import Fingerprints
from facade_change.io import read_json, write_json

demo = importlib.import_module("facade_change.agentic.2026-11-15_www_demo")
runner = demo.runner


class TransportTests(unittest.TestCase):
    def test_action_envelopes_preserve_query_reason_and_input(self):
        # Same envelope shapes as the real probe; these strings are synthetic.
        actions = [
            {"stage": "choose_action", "action": "retrieve", "query": "building_id:fixture date:2008-2009"},
            {"stage": "choose_action", "action": "retrieve", "query": "building_id=fixture region_id=synthetic"},
            {"stage": "choose_action", "action": "answer", "reason": "Достаточно показанных фото"},
            {"action": "abstain", "reason": "Синтетическая окклюзия"},
            {"type": "retrieve", "query": "  Preserve spacing  ", "observation_id": "shown-id", "reason": "Check source"},
            {"type": "answer"},
            {"type": "answer", "action": "answer"},
            {"stage": "answer", "action": "answer", "response": {"answer": "Incomplete synthetic draft"}},
            {"stage": "choose_action", "action": "answer", "response_contract": {"type": "archival_qa", "answer": {}}},
        ]
        for original in actions:
            with self.subTest(original=original):
                before = copy.deepcopy(original)
                changes = []
                parsed = runner.parse_reply(json.dumps(original, ensure_ascii=False), changes)
                normalized = runner.normalize_action(parsed, changes)
                expected = {k: v for k, v in original.items() if k not in {"stage", "action"}}
                expected["type"] = original.get("type", original.get("action"))
                if expected["type"] == "retrieve":
                    expected.setdefault("observation_id", None)
                self.assertEqual(normalized, expected)
                self.assertEqual(original, before)
                self.assertEqual(parsed, original)
                if "action" in original:
                    self.assertIn("action_key_to_type", changes)

    def test_ambiguous_or_semantically_incomplete_actions_are_not_repaired(self):
        invalid = [
            {"type": "answer", "action": "retrieve"},
            {"type": "retrieve", "action": "abstain", "query": "x", "reason": "conflict"},
            {"stage": "answer", "action": "retrieve", "query": "x"},
            {"action": "retrieve"},
            {"action": "retrieve", "query": ""},
            {"action": "retrieve", "query": 12},
            {"action": "retrieve", "query": "x", "observation_id": []},
            {"action": "retrieve", "query": "x", "observation_ids": []},
            {"action": "answer", "reason": None},
            {"action": "abstain"},
            {"action": "abstain", "reason": " "},
            {"action": "answer", "citations": ["invented"]},
            {"action": "unsupported"},
            {"action": ["retrieve", "answer"]},
            {"retrieve": {"query": "x"}, "answer": {}},
            {"action": "final", "decision": "supported"},
            {"stage": "answer", "action": "answer", "response": "not an object"},
            {"stage": "answer", "action": "answer", "response": {}, "response_contract": {}},
        ]
        for original in invalid:
            with self.subTest(original=original), self.assertRaises(ValueError):
                runner.normalize_action(original)

    def test_plain_fence_and_observed_punctuation_preserve_object(self):
        value = {"type": "final", "answer_text": "Русский ответ с ` и {} внутри строки", "boxes": []}
        raw = json.dumps(value, ensure_ascii=False)
        for text in (raw, "```json\n" + raw + "\n```", "```json\n" + raw + "`\n```\n"):
            with self.subTest(text=text):
                changes = []
                self.assertEqual(runner.parse_reply(text, changes), value)
                self.assertEqual(bool(changes), text != raw)
        self.assertIn("removed_trailing_backticks_inside_fence", changes)

    def test_omitted_boxes_only_defaults_for_final_and_does_not_mutate(self):
        value = {"type": "final", "date_interval": {"uncertain": False}}
        original = copy.deepcopy(value)
        changes = []
        self.assertEqual(runner.parse_reply(value, changes), {**value, "boxes": []})
        self.assertEqual(value, original)
        self.assertEqual(changes, ["defaulted_omitted_optional_boxes"])
        for boxes in (None, "", {}, [None]):
            self.assertEqual(runner.parse_reply({**value, "boxes": boxes})["boxes"], boxes)
        self.assertEqual(runner.parse_reply('{"type":"retrieve","query":"x","observation_id":null}'),
                         {"type": "retrieve", "query": "x", "observation_id": None})

    def test_ambiguous_prose_multiple_answers_and_invalid_json_fail_closed(self):
        for text in ('{} {}', 'answer: {}', '{} done', '```json\n{} prose\n```',
                     '```json\n{}\n```\n```json\n{}\n```', '{} `', '```json\n{}',
                     '```json\n{"answer_text":"cut\n```', '{"answer_text":"cut',
                     '{"a":1,}', '{"a":1,"a":2}', '{"a":NaN}', '[]', 'null', ' ' * 100_001):
            with self.subTest(text=text[:60]), self.assertRaises(ValueError):
                runner.parse_reply(text)


class AnswerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.questions, _, self.config = demo.fixture(self.root / "fixture")
        self.public = read_json(self.questions)

    def prepare(self, task):
        question = next(q for q in self.public["questions"] if q["task"] == task)
        folder = self.root / task
        folder.mkdir()
        prepared, evidence, common, support = runner._prepare(question, folder, Fingerprints(), runner.PROTOCOL)
        return question, evidence, prepared, common, support

    def answer(self, args):
        question, evidence, prepared, _, _ = args
        ids = list(evidence)
        earliest = question["task"] == "earliest_evidence"
        return {"type": "final", "decision": "earliest_presence" if earliest else "change", "answer_text": "Fixture only",
            "citations": [ids[0]] if earliest else ids, "observation_ids": [ids[0]] if earliest else ids,
            "date_interval": {**(runner._photo_interval(evidence[ids[0]]) if earliest else
                {"start": question["task_dates"][0], "end": question["task_dates"][-1], "precision": "interval"}),
                "uncertain": not earliest}, "limitations": []}

    def test_temporal_errors_name_bad_values_without_changing_them(self):
        args = self.prepare("temporal_change")
        value = self.answer(args)
        self.assertEqual(runner.validate_answer(value, *args, runner.PROTOCOL)["boxes"], [])
        value["date_interval"].update(precision="year", uncertain=False)
        original = copy.deepcopy(value)
        with self.assertRaises(ValueError) as caught:
            runner.validate_answer(value, *args, runner.PROTOCOL)
        self.assertIn('precision: expected "interval", received "year"', str(caught.exception))
        self.assertIn('uncertain: expected true, received false', str(caught.exception))
        self.assertEqual(value, original)
        del value["date_interval"]["uncertain"]
        with self.assertRaisesRegex(ValueError, "missing fields=.*uncertain"):
            runner.validate_answer(value, *args, runner.PROTOCOL)
        value = self.answer(args)
        value["date_interval"]["start"] = "2008"
        with self.assertRaisesRegex(ValueError, 'start: expected "2009", received "2008"'):
            runner.validate_answer(value, *args, runner.PROTOCOL)

    def test_final_envelopes_preserve_content_and_cannot_repair_incomplete_drafts(self):
        args = self.prepare("temporal_change")
        final = self.answer(args)
        expected = runner.validate_answer(final, *args, runner.PROTOCOL)
        content = {key: value for key, value in final.items() if key != "type"}
        for envelope in ({"stage": "answer", **content},
                         {"stage": "answer", "action": "answer", "response": content}):
            before = copy.deepcopy(envelope)
            changes = []
            self.assertEqual(runner.validate_answer(envelope, *args, runner.PROTOCOL,
                                                   normalizations=changes), expected)
            self.assertIn("answer_stage_to_final_type", changes)
            self.assertEqual(envelope, before)
        drafts = [
            {"stage": "answer", "action": "answer", "response": {"answer": "Draft", "evidence_id": "missing"}},
            {"stage": "answer", "action": "answer", "response_contract": {"type": "archival_qa", "answer": {}}},
            {"stage": "answer", "decision": "change"},
            {"stage": "answer", "action": "retrieve", **content},
            {"stage": "answer", **content, "citations": ["not-shown"]},
            {"stage": "answer", **content, "date_interval": {**content["date_interval"], "start": "1800"}},
        ]
        for draft in drafts:
            before = copy.deepcopy(draft)
            with self.subTest(draft=draft), self.assertRaises(ValueError):
                runner.validate_answer(draft, *args, runner.PROTOCOL)
            self.assertEqual(draft, before)

    def test_earliest_canonical_year_unknown_uncertainty_and_malformed_boxes(self):
        args = self.prepare("earliest_evidence")
        value = self.answer(args)
        normalized = runner.validate_answer(json.dumps(value), *args, runner.PROTOCOL)
        self.assertFalse(normalized["date_interval"]["uncertain"])
        self.assertEqual(normalized["date_interval"]["start"], "2009")
        value["date_interval"].update(start="2009-01-01", end="2009-12-31")
        with self.assertRaisesRegex(ValueError, "canonical source dates"):
            runner.validate_answer(value, *args, runner.PROTOCOL)
        for boxes in (None, {}, "", [None]):
            with self.subTest(boxes=boxes), self.assertRaises(ValueError):
                runner.validate_answer({**self.answer(args), "boxes": boxes}, *args, runner.PROTOCOL)
        value = self.answer(args)
        del value["date_interval"]["uncertain"]
        with self.assertRaisesRegex(ValueError, "missing fields=.*uncertain"):
            runner.validate_answer(value, *args, runner.PROTOCOL)

    def test_coordinates_remain_absolute_and_never_rescaled_or_repaired(self):
        args = self.prepare("temporal_change")
        value = self.answer(args)
        identifier = value["citations"][-1]
        box = {"evidence_id": identifier, "coordinate_system": "region_crop_pixels", "bbox_xyxy": [0.5, 0.5, 0.75, 0.75]}
        value["boxes"] = [box]
        result = runner.validate_answer(value, *args, runner.PROTOCOL)
        self.assertEqual(result["boxes"][0]["bbox_xyxy"], box["bbox_xyxy"])
        for coords in ([1, 1, 1, 1], [0, 0, 1000, 1000], [0, 0, float("nan"), 1]):
            with self.subTest(coords=coords), self.assertRaises(ValueError):
                runner.validate_answer({**value, "boxes": [{**box, "bbox_xyxy": coords}]}, *args, runner.PROTOCOL)
        with self.assertRaises(ValueError):
            runner.validate_answer({**value, "boxes": [{**box, "coordinate_system": "observation_native"}]}, *args, runner.PROTOCOL)

    def test_live_retry_audit_and_offline_validation_use_same_normalization(self):
        question = next(q for q in self.public["questions"] if q["task"] == "temporal_change")
        write_json(self.root / "public.json", {**self.public, "questions": [question]})

        class Wrapped(demo.SyntheticPool):
            calls = 0
            def call(self, role, payload):
                response = super().call(role, payload)
                if role == "qwen":
                    self.calls += 1
                    value = json.loads(response["text"])
                    del value["boxes"]
                    if self.calls == 1:
                        value["date_interval"].update(precision="year", uncertain=False)
                    response["text"] = "```json\n" + json.dumps(value) + "`\n```"
                return response

        result = runner.run(self.root / "public.json", self.config, self.root / "run",
            conditions=("photos_only",), allow_empty_archive=True, pool_factory=Wrapped, synthetic=True)
        self.assertEqual(result["outcome_counts"], {"completed": 1, "abstained": 0, "error": 0})
        folder = self.root / "run/questions" / question["question_id"] / "photos_only"
        first, last = read_json(folder / "reply-001.json"), read_json(folder / "reply-002.json")
        self.assertIn("precision", first["validation_error"])
        self.assertIn("uncertain", first["validation_error"])
        self.assertIsNone(last["validation_error"])
        for row in (first, last):
            self.assertIn("defaulted_omitted_optional_boxes", row["normalizations"])
            self.assertIn("removed_trailing_backticks_inside_fence", row["normalizations"])
        call = read_json(folder / "calls/001-qwen.json")
        feedback = next(m["content"][0]["text"] for m in call["request"]["messages"]
                        if m["role"] == "user" and m["content"][0]["text"].startswith("Ответ не прошёл"))
        self.assertIn('"precision": "interval"', feedback)
        self.assertIn('"start": "2009"', feedback)
        args = self.prepare("temporal_change")
        self.assertEqual(runner.validate_answer(call["response"]["text"], *args, runner.PROTOCOL),
                         read_json(folder / "result.json")["answer"])
        self.assertEqual(read_json(folder / "messages.json")[-1]["content"][0]["text"], call["response"]["text"])


if __name__ == "__main__":
    unittest.main()
