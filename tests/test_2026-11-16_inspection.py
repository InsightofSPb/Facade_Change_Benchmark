"""CPU regressions for transport, admission and unchanged approved review chains."""
import contextlib
import copy
import importlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from facade_change.agentic.cli import TerminalProgress
from facade_change.history_features import Fingerprints, digest
from facade_change.io import read_json, sha256, write_json


old = importlib.import_module("tests.test_2026-11-10_inspection")
app, data, demo = old.app, old.data, old.demo
cohort = importlib.import_module("facade_change.agentic.2026-11-16_inspection_cohort")
transport = importlib.import_module("facade_change.agentic.2026-11-16_inspection_reply")
evaluator = importlib.import_module("facade_change.agentic.2026-11-10_qa_evaluation")


class SmokePreparationTests(unittest.TestCase):
    def test_shared_transport_keeps_distinct_response_contracts(self):
        www = importlib.import_module("facade_change.agentic.2026-11-15_www_runner")
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}', '[]'):
            for parse in (www.parse_reply, transport.parse_reply):
                with self.subTest(text=text, parser=parse.__module__), self.assertRaises(ValueError):
                    parse(text, [])
        raw = '{"type":"final"}'
        self.assertEqual(www.parse_reply(raw)["boxes"], [])
        self.assertNotIn("boxes", transport.parse_reply(raw, []))
        # WWW historically tolerates a repeated trailing fence; inspection's
        # selector requires a single strict fence. Sharing JSON must keep this.
        extra_fence = '```json\n{"type":"final"}\n```\n```'
        self.assertEqual(www.parse_reply(extra_fence)["type"], "final")
        with self.assertRaises(ValueError):
            transport.parse_reply(extra_fence, [])

    def test_shared_identity_validator_keeps_pending_admission_explicit(self):
        shared = importlib.import_module("facade_change.agentic.2026-11-16_www_identity")
        self.assertFalse(shared.Identities().pending("Voskova_10"))
        inspection = cohort.identity.Identities()
        self.assertTrue(inspection.pending("Voskova_10"))
        self.assertEqual(inspection.physical("Voskova_12"), "Voskova_12")
        self.assertEqual(inspection.problem("Voskova_10", []),
                         "physical_identity_review_pending")

    setUpClass = classmethod(old.InspectionAcceptance.setUpClass.__func__)
    tearDownClass = classmethod(old.InspectionAcceptance.tearDownClass.__func__)
    setUp = old.InspectionAcceptance.setUp
    service = old.InspectionAcceptance.service
    run_policy = old.InspectionAcceptance.run_policy
    resolve = old.InspectionAcceptance.resolve

    def make_inventory(self, source=None, **kwargs):
        out = self.out / "inventory"
        with contextlib.redirect_stdout(io.StringIO()):
            cohort.inventory(source or self.questions, self.base / "fixture/histories", out,
                             available_at="2026-11-10", **kwargs)
        return out

    def confirmation(self, folder, review=None):
        review = review or self.base / "review/reviewed.json"
        result = read_json(folder / "assessability-template.private.json")
        states = read_json(review)["states"]
        for row in result["cases"]:
            row.update(decision="compatible", reviewer="synthetic_fixture_not_human",
                       rationale="Procedural CPU fixture only; all permitted RGB checked by construction",
                       basis="narrow_human_scope_review", review_source_sha256=sha256(review),
                       review_revision=states[row["question_id"]]["revision"],
                       sources=[dict(path=str(review), sha256=sha256(review))])
        path = self.out / "synthetic-confirmation.private.json"
        write_json(path, result)
        return path

    def prepare(self, folder, confirmation, review=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return cohort.prepare(review or self.base / "review/reviewed.json", folder / "selection.json",
                                  confirmation, self.out / "prepared")

    def test_plain_and_fenced_selector_and_judge_have_same_calls_and_decision(self):
        class Fenced(demo.QwenStub):
            def call(self, role, payload):
                response = super().call(role, payload)
                response["text"] = " \n```json\n" + response["text"] + "\n```\t"
                return response
        for name in ("heuristic", "agent"):
            plain, first, _ = self.run_policy(name, suffix=name + "-plain")
            fenced, second, _ = self.run_policy(name, pool=Fenced(self.config["models"]), suffix=name + "-fenced")
            for key in ("status", "decision"):
                self.assertEqual(plain[key], fenced[key])
            self.assertEqual(plain["localization"]["bbox_xyxy"], fenced["localization"]["bbox_xyxy"])
            a, b = read_json(first / "costs.json"), read_json(second / "costs.json")
            self.assertEqual(a["model_requests"], b["model_requests"])
            self.assertEqual(b["invalid_replies"], 0)
            for reply in second.glob("reply-*.json"):
                audit = read_json(reply)
                self.assertEqual(audit["cosmetic_transformations"],
                                 ["removed_outer_whitespace", "removed_single_json_fence"])
                self.assertIsNone(audit["validation_error"])
                self.assertIsInstance(audit["parsed_object"], dict)
                self.assertTrue(audit["raw_text"].startswith(" \n```json"))

    def test_transport_rejects_multiple_truncated_prose_duplicate_and_nonfinite(self):
        for text in ('{} {}', 'text {}', '{} trailing', '```json\n{}', '```json\n{}\n````',
                     '{"x":1,"x":2}', '{"x":{"y":1,"y":2}}', '{"x":NaN}',
                     '{"x":Infinity}', '{"x":-Infinity}', '{"x":1e400}', '[]', '{'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                transport.parse_reply(text, [])

    def test_malformed_json_consumes_repair_but_records_raw_and_validation_error(self):
        class Once(demo.QwenStub):
            count = 0
            def call(self, role, payload):
                response = super().call(role, payload)
                self.count += 1
                if self.count == 1:
                    response["text"] = '{"decision":'
                return response
        result, folder, _ = self.run_policy("heuristic", pool=Once(self.config["models"]))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(read_json(folder / "costs.json")["invalid_replies"], 1)
        self.assertEqual(read_json(folder / "costs.json")["model_requests"], 2)
        audit = read_json(folder / "reply-1.json")
        self.assertEqual(audit["raw_text"], '{"decision":')
        self.assertIsNone(audit["parsed_object"])
        self.assertIn("JSONDecodeError", audit["validation_error"])

    def test_numeric_strings_and_boolean_coordinates_are_not_repaired(self):
        class BadBox(demo.QwenStub):
            def reply(self, payload):
                reply = super().reply(payload)
                reply["localization"]["bbox_xyxy"][0] = "20"
                return reply
        result, folder, _ = self.run_policy("heuristic", pool=BadBox(self.config["models"]))
        self.assertEqual(result["status"], "error")
        self.assertIn("finite numeric", read_json(folder / "reply-1.json")["validation_error"])
        for box in ([True, 0, 1, 2], [0, 0, float("inf"), 2], [0, 0, 1]):
            with self.assertRaises(ValueError):
                transport.numeric_box(dict(bbox_xyxy=box))

    def test_inventory_candidates_needs_no_approvals_models_or_predictions(self):
        original_open = Path.open
        def guarded(path, *args, **kwargs):
            if "prediction" in path.name:
                raise AssertionError("Inventory opened a prediction")
            return original_open(path, *args, **kwargs)
        with patch.object(Path, "open", guarded), patch.object(app, "ModelPool", side_effect=AssertionError), \
                patch.object(data, "validate_config", side_effect=AssertionError):
            folder = self.make_inventory(self.base / "candidates/candidates.json")
        report = read_json(folder / "inventory.private.json")
        self.assertEqual(report["counts"]["eligible"], 5)
        self.assertGreater(report["counts"]["ineligible"], 0)
        self.assertEqual(report["filters"][-1]["after"], 5)
        self.assertFalse(report["models_loaded"])
        self.assertFalse(report["predictions_read"])
        self.assertEqual(report["touched_alias_groups"], [])
        template = read_json(folder / "assessability-template.private.json")
        self.assertTrue(all(r["decision"] == "pending" for r in template["cases"]))

    def test_missing_native_file_is_pending_not_verified_or_selected(self):
        path = Path(self.inventory_original())
        moved = path.with_suffix(".temporarily-unavailable")
        path.rename(moved)
        try:
            folder = self.make_inventory()
            report = read_json(folder / "inventory.private.json")
            row = next(r for r in report["rows"] if r["question_id"] == self.question["question_id"])
            self.assertEqual(row["status"], "pending")
            self.assertNotIn("files", row["passed"])
            self.assertNotIn(row["question_id"], read_json(folder / "selection.json")["question_ids"])
        finally:
            moved.rename(path)

    def inventory_original(self):
        return next(iter(self.__class__.inventory["observations"].values()))["original_path"]

    def test_test_split_wrong_frame_date_sha_and_one_history_date_fail_admission(self):
        fp = Fingerprints()
        root, document, _, identities = cohort.history_context(self.base / "fixture/histories", fp)
        for kind in ("split", "frame", "date", "sha", "history"):
            question = copy.deepcopy(self.question)
            histories = copy.deepcopy(document)
            if kind == "split":
                question["split"] = "test"
            elif kind == "frame":
                question["frame"]["shape"][0] += 1
            elif kind == "date":
                question["observations"][0]["date"] = "2009"
            elif kind == "sha":
                question["observations"][0]["rgb_sha256"] = "0" * 64
            else:
                track = next(t for t in histories["tracks"] if t["track_id"] == question["region_id"])
                track["observations"] = track["observations"][1:]
            rows, _ = cohort.compatibility([question], root, histories, "2026-11-10", identities, fp)
            self.assertEqual(rows[0]["status"], "ineligible", kind)

    def test_full_smoke_preserve_rows_and_review_chain_and_evaluate_missing(self):
        folder = self.make_inventory()
        confirmation = self.confirmation(folder)
        result = self.prepare(folder, confirmation)
        self.assertEqual((result["full"], result["smoke"]), (5, 2))
        original_public = {r["question_id"]: r for r in self.public["questions"]}
        original_private = {r["question_id"]: r for r in read_json(self.answers)["cases"]}
        for name in ("full", "smoke"):
            export = self.out / "prepared" / name / "export"
            questions = read_json(export / "questions.json")
            for row in questions["questions"]:
                self.assertEqual(row, original_public[row["question_id"]])
            for row in read_json(export / "answers.private.json")["cases"]:
                self.assertEqual(row, original_private[row["question_id"]])
            wrapper = self.out / (name + "-predictions.json")
            write_json(wrapper, app.prediction_wrapper(sha256(export / "questions.json"), []))
            with contextlib.redirect_stdout(io.StringIO()):
                scored = evaluator.evaluate(export / "questions.json", export / "answers.private.json", wrapper, self.out / (name + "-evaluation"))
            self.assertEqual(scored["coverage"]["missing"], len(questions["questions"]))
        self.assertNotEqual(result["manifests"]["full"]["questions_sha256"], result["manifests"]["smoke"]["questions_sha256"])

    def test_pending_assessability_or_wrong_universe_cannot_set_attestation(self):
        folder = self.make_inventory()
        with self.assertRaisesRegex(ValueError, "pending or incompatible"):
            self.prepare(folder, folder / "assessability-template.private.json")
        path = self.confirmation(folder)
        confirmation = read_json(path)
        confirmation["cases"][0]["universe_sha256"] = "0" * 64
        write_json(path, confirmation)
        with self.assertRaisesRegex(ValueError, "pending or incompatible"):
            self.prepare(folder, path)
        self.assertFalse((self.out / "prepared").exists())
        selection = read_json(folder / "selection.json")
        selection["question_ids"] = selection["question_ids"][::-1]
        write_json(folder / "selection.json", selection)
        with self.assertRaisesRegex(ValueError, "Selection differs"):
            self.prepare(folder, path)

    def test_unknown_duplicate_or_pending_ids_are_rejected_by_subset_export(self):
        pending = next(r["question_id"] for r in read_json(self.base / "candidates/candidates.json")["candidates"]
                       if r["question_id"] not in {q["question_id"] for q in self.public["questions"]})
        for ids in (["unknown"], [self.question["question_id"]] * 2, [pending]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                cohort.qa.export(self.base / "review/reviewed.json", self.out / "bad", question_ids=ids)

    def identity_map(self, decision="same_building"):
        path = self.out / "identity.json"
        write_json(path, dict(schema_version=1, kind="www_physical_identity_review", groups=[dict(
            physical_building_id="b1", building_ids=["b1", "b2"], decision=decision,
            reviewer="synthetic_fixture_not_human", rationale="Synthetic identity contract fixture only",
            sources=[dict(path=str(self.questions), sha256=sha256(self.questions))])]))
        return path

    def test_physical_grouping_and_event_deduplication_precede_macro(self):
        mapping = self.identity_map()
        positive = [q for q in self.public["questions"] if q["building_id"] in {"b1", "b2"}
                    and q["frame"]["anchor_bbox_xyxy"][0] == 80]
        questions = {q["question_id"]: {**q, "physical_building_id": "b1"} for q in positive}
        labels = {r["question_id"]: copy.deepcopy(r["reviewed_labels"]) for r in read_json(self.answers)["cases"]
                  if r["question_id"] in questions}
        for row in labels.values():
            row["events"][0]["event_id"] = "one-physical-episode"
        self.assertEqual(len(evaluator._events(questions, labels)), 1)
        for question in questions.values():
            question["physical_building_id"] = question["building_id"]
        with self.assertRaisesRegex(ValueError, "spans buildings"):
            evaluator._events(questions, labels)
        wrapper = self.out / "missing-predictions.json"
        write_json(wrapper, app.prediction_wrapper(sha256(self.questions), []))
        with contextlib.redirect_stdout(io.StringIO()):
            evaluator.evaluate(self.questions, self.answers, wrapper, self.out / "grouped", identity_map=mapping)
        grouped = read_json(self.out / "grouped/evaluation.json")
        self.assertEqual(set(grouped["buildings"]), {"b1", "b3", "b4"})
        self.assertEqual(app._at(grouped["buildings"]["b1"])["eligible_events"], 2)

    def test_pending_identity_split_conflict_and_changed_source_sha_fail(self):
        mapping = self.identity_map("pending")
        ids = cohort.identity.Identities(mapping, Fingerprints())
        tracks = [dict(building_id="b1", split="val"), dict(building_id="b2", split="train")]
        self.assertIn("pending", ids.problem("b1", tracks))
        mapping = self.identity_map()
        ids = cohort.identity.Identities(mapping, Fingerprints())
        self.assertIn("crosses_original_splits", ids.problem("b1", tracks))
        default = cohort.identity.Identities()
        self.assertTrue(default.pending("Voskova_10"))
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            cohort.identity.Identities(mapping, Fingerprints(), "0" * 64)

    def test_prepared_smoke_run_does_not_open_private_answers_or_selection(self):
        folder = self.make_inventory()
        self.prepare(folder, self.confirmation(folder))
        smoke = self.out / "prepared/smoke"
        native_open = Path.open
        def guarded(path, *args, **kwargs):
            if path.suffix == ".json" and ("private" in path.name or path.name == "reviewed.json"):
                raise AssertionError("Inference opened private data: " + str(path))
            return native_open(path, *args, **kwargs)
        with patch.object(Path, "open", guarded), patch("facade_change.agentic.pipeline._model_signature", demo.stub_signature), \
                contextlib.redirect_stdout(io.StringIO()):
            result = app.run(smoke / "export/questions.json", self.base / "fixture/histories",
                             self.base / "synthetic-config.json", smoke / "protocol.json", self.out / "smoke-run",
                             pool_factory=demo.QwenStub, detector_factory=demo.DetectorStub, check_resources=False)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["policies"], ["heuristic", "agent"])
        with contextlib.redirect_stdout(io.StringIO()):
            evaluated = app.evaluate(smoke / "export/questions.json", smoke / "export/answers.private.json",
                                     self.out / "smoke-run", self.out / "smoke-evaluation")
        self.assertEqual(evaluated["status"], "completed")

    def test_progress_is_visible_before_final_json_and_does_not_show_raw_reply(self):
        terminal = io.StringIO()
        observer = TerminalProgress(stream=terminal)
        self.addCleanup(observer.close)
        with patch("facade_change.agentic.pipeline._model_signature", demo.stub_signature), contextlib.redirect_stdout(io.StringIO()):
            app.run(self.questions, self.base / "fixture/histories", self.base / "synthetic-config.json",
                    self.base / "protocol.json", self.out / "progress-run", pool_factory=demo.QwenStub,
                    detector_factory=demo.DetectorStub, check_resources=False, progress=observer)
        log = terminal.getvalue()
        for stage in ("preflight_started", "preparation_started", "policy_started", "model_requested planner",
                      "model_returned judge", "inspection_delivered", "selector_stop", "policy_finished"):
            self.assertIn(stage, log)
        self.assertNotIn("Explicit synthetic acceptance label", log)
        self.assertNotIn("visible_difference", log)

    def test_config_derivative_preserves_source_and_reports_actual_conflicts(self):
        source = self.base / "synthetic-config.json"
        before = source.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            result = cohort.check_config(source, self.base / "protocol.json", self.out / "config-check", derive=True)
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(result["gpu_smoke"], "not_run")
        self.assertEqual(result["status"], "blocked")  # Synthetic source/checkpoint do not exist.
        self.assertTrue((self.out / "config-check/2026-11-16-inspection.local.json").exists())

    def test_cost_summary_preserves_unknown_and_charges_shared_preparation_once(self):
        result = app.cost_summary(self.base / "analysis", data.POLICIES,
                                  [q["question_id"] for q in self.public["questions"]])
        self.assertEqual(result["shared_preparation_once"]["expected"], 5)
        self.assertIsNone(result["policies"]["agent"]["warm_inference_seconds"]["total"])
        self.assertGreater(result["policies"]["agent"]["stages"]["planner"]["observed_requests"], 0)
        self.assertEqual(result["policies"]["heuristic"]["stages"]["planner"]["observed_requests"], 0)
        absent = app.cost_summary(self.out / "missing-run", ["agent"], ["missing-case"])
        self.assertIsNone(absent["policies"]["agent"]["stages"]["planner"]["actual"]["input_tokens"]["total"])


if __name__ == "__main__":
    unittest.main()
