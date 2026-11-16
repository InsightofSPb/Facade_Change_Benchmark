"""CPU regression: legacy parent dates and one reviewed archive earliest target for A/B/C."""
import copy
import importlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from facade_change.history_features import Fingerprints, digest
from facade_change.io import read_json, sha256, write_json
from facade_change.agentic.runner import complete_run, new_run
from facade_change.agentic.store import reject_forbidden

demo = importlib.import_module("facade_change.agentic.2026-11-15_www_demo")
qa, archive, runner = demo.qa, demo.archive, demo.runner
curation = importlib.import_module("facade_change.agentic.2026-11-15_www_curation")
metrics = importlib.import_module("facade_change.agentic.2026-11-15_www_evaluation")


class ExposedEarliestPool(demo.SyntheticPool):
    """Scripted acceptance only: choose an earlier dated image only after it was exposed."""
    def call(self, role, payload):
        result = super().call(role, payload)
        if role != "qwen":
            return result
        value = json.loads(result["text"])
        if value["type"] != "final":
            return result
        for message in payload["messages"]:
            for block in message["content"]:
                if block["type"] != "text":
                    continue
                try:
                    metadata = json.loads(block["text"])
                except ValueError:
                    continue
                if isinstance(metadata, dict) and metadata.get("image_exposed") and metadata.get("observation_date", {}).get("start"):
                    date = metadata["observation_date"]
                    value.update(observation_ids=[metadata["evidence_id"]], citations=[metadata["evidence_id"]],
                        date_interval={**{k: date[k] for k in ("start", "end", "precision")}, "uncertain": True},
                        answer_text="Синтетический пример: свидетельство уже есть к указанной дате; начало неизвестно.")
        result["text"] = json.dumps(value, ensure_ascii=False)
        return result


class ArchiveReviewTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        _, self.base_package, self.config = demo.fixture(self.root / "fixture")
        self.candidates = self.root / "fixture/candidates/candidates.json"
        self.candidate_doc = read_json(self.candidates)
        self.case = next(r for r in self.candidate_doc["candidates"] if r["task"] == "earliest_evidence")
        self.number = 0

    def make_archive(self, date=None, *, parent=False, changes=None):
        self.number += 1
        package = read_json(self.base_package)
        record = package["records"][0]
        # The synthetic image depicts the same known damaged crop as the local 2009 input.
        photo = self.root / "fixture/earliest-archive.png"
        shutil.copyfile(self.case["observations"][0]["rgb_path"], photo)
        record.update(evidence_id="fixture-earliest-archive", building_id=self.case["building_id"], view_id=self.case["view_id"],
            observation_date={**(date or {"start": "2008", "end": "2008", "precision": "year"}), "basis": "explicit synthetic capture metadata"},
            publication_date="2020", available_at="2020")
        record["assets"][0].update(path=photo.name, sha256=sha256(photo))
        if parent:
            pdf = self.root / "fixture/parent.pdf"
            with Image.open(photo) as image:
                image.save(pdf, format="PDF")
            document = {**copy.deepcopy(record), "evidence_id": "fixture-parent", "record_type": "document",
                "observation_date": {"start": None, "end": None, "precision": "mixed_document", "basis": "Several capture dates in the PDF"},
                "assets": [{"path": pdf.name, "sha256": sha256(pdf), "media_type": "application/pdf", "width": None, "height": None, "page_index": None}]}
            record.update(record_type="image_in_document", parent_evidence_id=document["evidence_id"])
            record["assets"][0]["page_index"] = 1
            package["records"].insert(0, document)
        record.update(changes or {})
        source = self.root / "fixture" / f"package-{self.number}.json"
        write_json(source, package)
        destination = self.root / f"archive-{self.number}"
        archive.import_archive(source, destination)
        corpus = destination / "corpus.json"
        row = next(r for r in read_json(corpus)["records"] if r["evidence_id"] == record["evidence_id"])
        return source, corpus, row

    def label(self, corpus, record):
        return {"assessment": "earliest_presence", "observability": "visible", "reference_answer": "PRIVATE synthetic reviewed match",
            "limitations": ["Capture interval is not damage onset"], "earliest_observation_id": record["evidence_id"], "onset": "unknown",
            "present_by_interval": {k: record["observation_date"][k] for k in ("start", "end", "precision")},
            "archival_status": "verified_useful", "supporting_evidence_ids": [record["evidence_id"]], "archival_stratum": True,
            "earliest_archive": {"corpus_path": str(corpus), "corpus_sha256": sha256(corpus), "record_sha256": digest(record),
                "asset_sha256": archive.image_assets(record)[0]["sha256"], **{k: self.case[k] for k in ("building_id", "view_id", "region_id")},
                "match_basis": "PRIVATE synthetic reviewer matched this image to the supplied physical region"}, "events": []}

    def test_mixed_parent_preserves_original_bytes_dates_hashes_and_page_lineage(self):
        source, corpus, child = self.make_archive(parent=True)
        parent = read_json(corpus)["records"][0]
        original = read_json(source)
        self.assertEqual(parent["observation_date"], original["records"][0]["observation_date"])
        self.assertEqual(parent["observation_date"]["precision"], "mixed_document")
        self.assertIsNone(parent["observation_date"]["start"])
        self.assertEqual(child["observation_date"], original["records"][1]["observation_date"])
        self.assertEqual(child["parent_evidence_id"], parent["evidence_id"])
        self.assertEqual(child["assets"][0]["page_index"], 1)
        self.assertEqual((corpus.parent / "source_package.json").read_bytes(), source.read_bytes())
        self.assertEqual(read_json(corpus.parent / "config.json")["package_sha256"], sha256(source))
        for saved, supplied in zip(read_json(corpus)["records"], original["records"]):
            self.assertEqual(saved["assets"][0]["sha256"], supplied["assets"][0]["sha256"])
            for key in ("date_source", "publication_date", "collected_at", "available_at", "availability_basis"):
                self.assertEqual(saved[key], supplied[key])
        # Canonical v3 unknown has exactly the same undated semantics.
        original["records"][0]["observation_date"]["precision"] = "unknown"
        archive.validate_package(original, source.parent, Fingerprints())

    def test_legacy_precision_is_not_available_to_photos_or_dated_documents(self):
        _, corpus, _ = self.make_archive(parent=True)
        original = read_json(corpus)
        for record_index, precision, start, end in ((1, "mixed_document", None, None), (1, "century", "2008", "2008"),
                                                    (0, "mixed_document", "2008", "2008")):
            with self.subTest(record=record_index, precision=precision):
                value = copy.deepcopy(original)
                value["records"][record_index]["observation_date"].update(precision=precision, start=start, end=end)
                with self.assertRaises(ValueError):
                    archive.validate_package(value, corpus.parent, Fingerprints())

    def test_reference_rejects_missing_ids_hashes_owners_and_implicit_support(self):
        _, corpus, record = self.make_archive()
        good = self.label(corpus, record)
        qa._checked_label(self.case, good, Fingerprints())
        mutations = [({"earliest_observation_id": "missing-record"}, "does not exist"),
            ({"onset": "2008"}, "onset must remain unknown"), ({"supporting_evidence_ids": []}, "supporting evidence IDs"),
            ({"archival_status": "unknown"}, "explicit human-reviewed"), ({"present_by": "2008"}, "scalar present_by")]
        for changes, message in mutations:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                qa._checked_label(self.case, {**copy.deepcopy(good), **changes}, Fingerprints())
        for key, value in (("corpus_sha256", "0"*64), ("record_sha256", "0"*64), ("asset_sha256", "0"*64),
                           ("building_id", "wrong"), ("view_id", "wrong"), ("region_id", "wrong"), ("match_basis", "")):
            with self.subTest(binding=key), self.assertRaises(ValueError):
                label = copy.deepcopy(good)
                label["earliest_archive"][key] = value
                qa._checked_label(self.case, label, Fingerprints())
        for changes in ({"building_id": "different-building"}, {"view_id": "different-view"}, {"identity_status": "probable"}):
            _, other, row = self.make_archive(changes=changes)
            with self.subTest(record=changes), self.assertRaisesRegex(ValueError, "identity"):
                qa._checked_label(self.case, self.label(other, row), Fingerprints())
        with self.assertRaisesRegex(ValueError, "same retrospective"):
            qa._checked_label({**self.case, "knowledge_cutoff": "2026-11-16"}, good, Fingerprints())
        (corpus.parent / record["assets"][0]["path"]).write_bytes(b"changed after review")
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            qa._checked_label(self.case, good, Fingerprints())

    def test_unknown_capture_cannot_borrow_publication_as_earliest(self):
        _, corpus, record = self.make_archive({"start": None, "end": None, "precision": "unknown"})
        label = self.label(corpus, record)
        label["present_by_interval"] = {"start": "2020", "end": "2020", "precision": "year"}
        with self.assertRaisesRegex(ValueError, "publication is not capture"):
            qa._checked_label(self.case, label, Fingerprints())

    def freeze(self, corpus, record, identity_map=None):
        label = self.label(corpus, record)
        path = self.root / "decisions.private.json"
        write_json(path, qa._private("facade_qa_review_decisions", candidate_sha256=sha256(self.candidates), decisions=[{
            "question_id": self.case["question_id"], "expected_revision": 0, "status": "approved", "reviewer": "synthetic_fixture_only",
            "private_notes": "Explicit synthetic reference frozen before all model responses", "label": label}]))
        qa.review(self.candidates, path, self.root / "review")
        qa.export(self.root / "review/reviewed.json", self.root / "export")
        questions, answers = self.root / "export/questions.json", self.root / "export/answers.private.json"
        public = read_json(questions)
        self.assertEqual(public["questions"], [{k: self.case[k] for k in qa.PUBLIC_KEYS}])
        self.assertNotIn("earliest_archive", json.dumps(public))
        self.assertEqual(read_json(answers)["cases"][0]["reviewed_labels"], label)
        write_json(self.root / "audit.json", {"kind": "www_exposure_audit", "buildings": [{"building_id": self.case["building_id"], "cohort": "development"}]})
        curation.inventory(self.root / "fixture/inputs/histories", self.root / "inventory", candidates_path=self.candidates,
                           exposure_path=self.root / "audit.json", identity_map=identity_map)
        archive.index_archive(corpus, self.config, self.root / "index", pool_factory=ExposedEarliestPool, synthetic=True)
        metrics.plan(questions, answers, self.root / "inventory/inventory.json", self.config, self.root / "plan",
                     archive_index=self.root / "index/index.json", synthetic=True, identity_map=identity_map)
        self.assertEqual(read_json(self.root / "plan/plan.private.json")["archive_corpus_sha256"], sha256(corpus))
        return questions, answers, label

    def exercise_pipeline(self, date, *, later_local_only=False):
        if later_local_only:
            # A separate synthetic question permits only the existing 2015 observation.
            self.case = copy.deepcopy(self.case)
            self.case.update(question_id="fixture-earliest-local-2015", task_dates=["2015"], observations=self.case["observations"][-1:])
            self.case["annotation_previews"] = []
            self.case["annotation_evidence"] = {self.case["observations"][0]["observation_id"]: self.case["annotation_evidence"][self.case["observations"][0]["observation_id"]]}
            journal = new_run(self.root / "later-candidates", {"synthetic_fixture": True}, "facade_qa_candidates")
            write_json(journal.out / "candidates.json", {**self.candidate_doc, "candidates": [self.case]})
            complete_run(journal, "completed")
            journal.close()
            self.candidates = journal.out / "candidates.json"
        _, corpus, record = self.make_archive(date, parent=True)
        questions, answers, label = self.freeze(corpus, record)
        with patch.object(qa, "_checked_label", side_effect=AssertionError("inference read private labels")):
            result = runner.run(questions, self.config, self.root / "run", archive_index=self.root / "index/index.json",
                                pool_factory=ExposedEarliestPool, synthetic=True)
        self.assertEqual(result["outcome_counts"], {"completed": 3, "abstained": 0, "error": 0})
        predictions = read_json(self.root / "run/results.json")["results"]
        for row in predictions:
            messages = read_json(self.root / "run/questions" / self.case["question_id"] / row["condition"] / "messages.json")
            self.assertNotIn("PRIVATE synthetic", json.dumps(messages))
            self.assertNotIn("earliest_archive", json.dumps(messages))
            if row["condition"] == "photos_only":
                self.assertFalse(metrics.structured_correct(self.case, label, row))
                self.assertEqual(row["answer"]["observation_ids"], [self.case["observations"][0]["observation_id"]])
            else:
                self.assertTrue(metrics.structured_correct(self.case, label, row))
                self.assertEqual({k: row["answer"]["date_interval"][k] for k in date}, date)
        metrics.score(questions, answers, self.root / "run", self.root / "plan/plan.private.json", self.root / "score")
        report = read_json(self.root / "score/metrics.private.json")
        for condition in runner.CONDITIONS:
            self.assertEqual(report["summaries"]["development/all"][condition]["structured_accuracy"], float(condition != "photos_only"))
        return questions, answers, corpus, record

    def test_reviewed_archive_2008_beats_valid_but_later_local_2009(self):
        questions, answers, corpus, record = self.exercise_pipeline({"start": "2008", "end": "2008", "precision": "year"})
        with self.assertRaisesRegex(ValueError, "frozen retrieval corpus"):
            metrics.plan(questions, answers, self.root / "inventory/inventory.json", self.config, self.root / "empty-plan",
                         allow_empty_archive=True, synthetic=True)
        _, other, _ = self.make_archive(changes={"title": "A different frozen corpus version"})
        archive.index_archive(other, self.config, self.root / "other-index", pool_factory=ExposedEarliestPool, synthetic=True)
        with self.assertRaisesRegex(ValueError, "frozen retrieval corpus"):
            metrics.plan(questions, answers, self.root / "inventory/inventory.json", self.config, self.root / "other-plan",
                         archive_index=self.root / "other-index/index.json", synthetic=True)

    def test_month_interval_roundtrips_review_export_plan_and_scoring(self):
        self.exercise_pipeline({"start": "2011-03-01", "end": "2011-03-31", "precision": "month"}, later_local_only=True)

    def test_day_and_documented_interval_preserve_precision(self):
        for date in ({"start": "2008-04-17", "end": "2008-04-17", "precision": "day"},
                     {"start": "2007", "end": "2008", "precision": "interval"}):
            with self.subTest(date=date):
                _, corpus, record = self.make_archive(date)
                label = qa._checked_label(self.case, self.label(corpus, record), Fingerprints())
                result = {"status": "completed", "answer": {"decision": "earliest_presence", "observation_ids": [record["evidence_id"]],
                    "date_interval": {**date, "uncertain": True}}}
                self.assertTrue(metrics.structured_correct(self.case, label, result))
                result["answer"]["date_interval"]["precision"] = "year"
                self.assertFalse(metrics.structured_correct(self.case, label, result))

    def test_private_binding_cannot_enter_public_sources_or_queries(self):
        _, corpus, record = self.make_archive()
        label = self.label(corpus, record)
        for value in ({"earliest_archive": label["earliest_archive"]}, {"present_by_interval": label["present_by_interval"]}):
            with self.assertRaises(ValueError):
                reject_forbidden(value)
            with self.assertRaises(ValueError):
                reject_forbidden(json.dumps(value))

    def test_reviewed_identity_is_shared_by_archive_reference_retrieval_plan_and_score(self):
        alias = self.case["building_id"] + "-alias"
        _, corpus, record = self.make_archive(changes={"building_id": alias})
        label = self.label(corpus, record)
        with self.assertRaisesRegex(ValueError, "identity"):
            qa._checked_label(self.case, label, Fingerprints())
        identity_path = self.root / "reviewed-identity.json"
        write_json(identity_path, {"schema_version": 1, "kind": "www_physical_identity_review", "groups": [{
            "physical_building_id": self.case["building_id"], "building_ids": [self.case["building_id"], alias],
            "decision": "same_building", "reviewer": "synthetic_human", "rationale": "Explicit synthetic alias review",
            "sources": [{"path": str(corpus), "sha256": sha256(corpus)}]}]})
        label["earliest_archive"]["identity_source"] = {"path": str(identity_path), "sha256": sha256(identity_path)}
        with patch.object(self, "label", return_value=label):
            questions, answers, _ = self.freeze(corpus, record, identity_map=identity_path)
        with self.assertRaisesRegex(ValueError, "same reviewed identity"):
            metrics.plan(questions, answers, self.root / "inventory/inventory.json", self.config, self.root / "bad-plan",
                         archive_index=self.root / "index/index.json", synthetic=True)
        result = runner.run(questions, self.config, self.root / "run", archive_index=self.root / "index/index.json",
                            identity_map=identity_path, pool_factory=ExposedEarliestPool, synthetic=True)
        self.assertEqual(result["outcome_counts"]["completed"], 3)
        metrics.score(questions, answers, self.root / "run", self.root / "plan/plan.private.json", self.root / "score")
        report = read_json(self.root / "score/metrics.private.json")
        self.assertEqual(report["identity_source"], label["earliest_archive"]["identity_source"])
        self.assertTrue(all(r["physical_building_id"] == self.case["building_id"] for r in report["cases"]))
        self.assertEqual(report["summaries"]["development/all"]["fixed_retrieval"]["structured_accuracy"], 1.)
        identities = archive.identity.Identities(identity_path, Fingerprints())
        self.assertFalse(identities.compatible({**record, "view_id": "wrong-view"}, self.case))


if __name__ == "__main__":
    unittest.main()
