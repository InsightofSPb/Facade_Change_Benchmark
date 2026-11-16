"""Prospective RGB QA policy and saved-role costs: CPU fixtures, no model validation."""
import copy
import importlib
from pathlib import Path
import tempfile
import unittest

from facade_change.history_features import Fingerprints, digest
from facade_change.io import read_json, sha256, write_json

demo = importlib.import_module("facade_change.agentic.2026-11-15_www_demo")
curation = importlib.import_module("facade_change.agentic.2026-11-15_www_curation")
policy = curation.rgb_policy
factual = importlib.import_module("facade_change.agentic.2026-11-16_www_factual")
costs = factual.costs


def rgb_fixture(root):
    """Construct a NEW synthetic history revision with three splits and nine dates."""
    _, package, config = demo.fixture(root / "fixture")
    histories = root / "fixture/inputs/histories"
    manifest = read_json(histories / "manifest.json")
    for track in manifest["tracks"]:
        track["split"] = ("train", "val", "test")[int(track["building_id"][1:]) % 3]
        if track["track_id"] == "earliest":
            for year in range(2016, 2023):
                # Same visible fixture RGB at additional dates, not evidence of new damage.
                track["observations"].append({**copy.deepcopy(track["observations"][1]),
                    "observation_id": "synthetic-extra-" + str(year), "year": year})
            track["distinct_years"] = 9
    write_json(histories / "manifest.json", manifest)
    write_json(histories / manifest["report"] / "manifest.json", manifest)
    write_json(histories / manifest["report"] / "run.json", {"status": "completed", "synthetic_fixture": True,
        "manifest_sha256": sha256(histories / "manifest.json")})
    source = read_json(root / "fixture/candidates/candidates.json")["source"]
    annotation_path = Path(source["annotation_targets"])
    annotations = read_json(annotation_path)
    annotations["input_sha256"] = {str(histories / "manifest.json"): sha256(histories / "manifest.json")}
    write_json(annotation_path, annotations)
    write_json(annotation_path.parent / "run.json", {"status": "completed", "synthetic_fixture": True,
        "artifact_sha256": {"index.json": sha256(annotation_path)}})
    demo.qa.generate(histories, root / "candidates", annotation_targets=annotation_path,
        knowledge_cutoff="2026-11-15", task_mode="retrospective", available_at="observation_date", splits=("train", "val", "test"))
    return root / "candidates/candidates.json", histories, package, config


class RGBPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.candidates, self.histories, self.package, self.config = rgb_fixture(self.root)
        self.cases = read_json(self.candidates)["candidates"]

    def inventory(self, name="inventory", **kwargs):
        curation.inventory(self.histories, self.root / name, candidates_path=self.candidates, qa_policy=policy.POLICY, **kwargs)
        return read_json(self.root / name / "inventory.json")

    def test_train_allowed_unknown_qa_exposure_is_not_audited_evaluation_and_legacy_stays(self):
        before = {p: sha256(p) for p in (self.candidates, self.histories / "manifest.json")}
        inv = self.inventory()
        train = [v for v in inv["views"] if v["split"] == "train"]
        self.assertTrue(train and all(v["eligible"] for v in train))
        self.assertTrue(all(v["original_split"] == "train" and v["legacy_exposure_status"] == "training_exposure" for v in train))
        self.assertTrue(all(v["qa_exposure_status"] == "unknown" and not v["evaluation_eligible"] for v in inv["views"]))
        curation.inventory(self.histories, self.root / "legacy", candidates_path=self.candidates)
        legacy = read_json(self.root / "legacy/inventory.json")
        self.assertTrue(all(not v["eligible"] for v in legacy["views"] if v["split"] == "train"))
        with self.assertRaisesRegex(ValueError, "recorded inventory"):
            curation.select_batch(self.cases, legacy, qa_policy=policy.POLICY)
        self.assertEqual({p: sha256(p) for p in before}, before)

    def test_building_passes_precede_types_and_missing_categories_do_not_block(self):
        inv = self.inventory()
        rows = []
        for i, case in enumerate(next(c for c in self.cases if c["building_id"] == b) for b in ("b0", "b1", "b2")):
            for n in range(10 if i == 0 else 2):
                row = copy.deepcopy(case)
                row.update(question_id=f"fixture-{i}-{n}", task="temporal_change", candidate_type="possible_change")
                row["observations"] = row["observations"][:2]
                row["task_dates"] = [r["date"] for r in row["observations"]]
                row["frame"]["anchor_bbox_xyxy"] = [n*20, 0, n*20+10, 10]  # Independent synthetic regions.
                rows.append(row)
        selected, excluded = curation.select_batch(rows, inv, size=9, targets=curation.REVIEW_TARGETS)
        self.assertEqual(len({r["building_id"] for r in selected[:3]}), 3)
        self.assertEqual(len(selected), 9)
        self.assertGreater(sum(r["building_id"] == "b0" for r in selected), 2)
        self.assertTrue(all(r["candidate_type"] == "possible_change" for r in selected))
        self.assertNotIn("provisional_target_quota", {r["reason"] for r in excluded})
        repeat, _ = curation.select_batch(list(reversed(rows)), inv, size=9, targets=curation.REVIEW_TARGETS)
        self.assertEqual([r["question_id"] for r in selected], [r["question_id"] for r in repeat])
        selected, _ = curation.select_batch(rows, inv, size=100, per_building=3)
        self.assertLessEqual(sum(r["building_id"] == "b0" for r in selected), 3)

    def test_long_earliest_is_reported_but_independent_intervals_remain(self):
        inv = self.inventory()
        earliest = next(c for c in self.cases if c["task"] == "earliest_evidence")
        frozen = copy.deepcopy(earliest)
        check = policy.image_budget(earliest)
        self.assertFalse(check["compatible"])
        self.assertTrue(check["requires_reviewed_scope_revision"])
        selected, excluded = curation.select_batch(self.cases, inv)
        self.assertTrue(any(r["question_id"] == earliest["question_id"] and r["reason"] == "local_image_budget_incompatible" for r in excluded))
        intervals = [c for c in selected if c["region_id"] == "earliest" and c["task"] == "temporal_change"]
        self.assertGreater(len(intervals), 2)
        self.assertEqual(earliest, frozen)
        case = intervals[0]
        out = self.root / "preparation"
        out.mkdir()
        prepared, *_ = demo.runner._prepare(case, out, Fingerprints(), demo.runner.PROTOCOL)
        self.assertEqual(policy.image_budget(case)["selected_observation_ids"], prepared["selected_observations"])

    def test_reviewed_aliases_allow_old_split_overlap_but_preserve_qa_partition_conflict(self):
        mapping = demo.archive.identity.template()
        path = self.root / "identity.json"
        proof = self.root / "identity-note.txt"
        proof.write_text("Synthetic human identity decision")
        base = [{"building_id": name, "view_id": name, "split": split, "cohort": "unassigned", "exposure_status": "unknown",
            "exclusion_reasons": [], "technical_exclusion_reasons": [], "original_sha_cross_physical_building": False}
            for name, split in (("Voskova_10", "train"), ("Voskova_12", "test"))]
        write_json(path, mapping)
        pending = demo.archive.identity.Identities(path, Fingerprints())
        self.assertTrue(all(not r["eligible"] for r in policy.annotate_inventory(copy.deepcopy(base), pending, {}, {})))
        mapping["groups"][0].update(decision="same_building", reviewer="fixture-reviewer", rationale="Synthetic same building",
            sources=[{"path": str(proof), "sha256": sha256(proof)}])
        write_json(path, mapping)
        identities = demo.archive.identity.Identities(path, Fingerprints())
        allowed = policy.annotate_inventory(copy.deepcopy(base), identities, {}, {})
        self.assertTrue(all(r["eligible"] and r["segmentation_split_overlap"] for r in allowed))
        self.assertEqual({r["physical_building_id"] for r in allowed}, {"Voskova_10"})
        exposure = {"exposure_domain": "rgb_qa", "qa_policy": policy.POLICY, "protocol_sha256": sha256(proof), "buildings": [
            {"building_id": "Voskova_10", "cohort": "development", "prior_exposure": "used_for_qa_development"},
            {"building_id": "Voskova_12", "cohort": "evaluation", "prior_exposure": "none_found", "reviewer": "fixture",
             "audit_sources": [{"path": str(proof), "sha256": sha256(proof)}]}]}
        conflict = policy.annotate_inventory(copy.deepcopy(base), identities, exposure, {})
        self.assertTrue(all(r["qa_development_evaluation_overlap"] and not r["evaluation_eligible"] for r in conflict))
        self.assertEqual([r["original_split"] for r in conflict], ["train", "test"])

    def test_review_package_preserves_handoff_and_excludes_used_or_overlapping_questions(self):
        case = next(c for c in self.cases if c["region_id"] == "change")
        target = {"question_id": case["question_id"], "group": "visible_change", "answer_ru": "Synthetic original human decision",
                  "clarification_verbatim": "Synthetic case 03 text stays byte-identical"}
        handoff, selection = self.root / "handoff.private.json", self.root / "selection.private.json"
        write_json(handoff, {"source_candidates_sha256": sha256(self.candidates), "cases": [target]})
        write_json(selection, demo.qa._private("www_question_selection", candidate_sha256=sha256(self.candidates), question_ids=[case["question_id"]]))
        before = {p: p.read_bytes() for p in (self.candidates, handoff, selection)}
        result = curation.review_package(self.candidates, self.histories, None, self.root / "review",
            qa_policy=policy.POLICY, handoff_path=handoff, preserved_selection=selection)
        self.assertEqual(result["approved"], 0)
        self.assertFalse(result["full_target_cohort_created"])
        self.assertEqual(result["shortfall"], 100-result["selected"])
        selected = read_json(self.root / "review/batch/selection.private.json")
        used = {c["question_id"] for c in self.cases if c["region_id"] in {"change", "overlap"}}
        self.assertFalse(set(selected["question_ids"]) & used)
        self.assertEqual(read_json(self.root / "review/preserved-review/drafts.private.json")["items"][0]["human_text_judgment"], target)
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        self.assertGreater(result["distributions"]["original_split"]["train"], 0)
        coverage = read_json(self.root / "review/archive-coverage.private.json")
        self.assertTrue(all(r["search_status"] == ["not_documented"] and not r["source_evidence_ids"] for r in coverage["buildings"]))
        self.assertTrue(all(r["human_usefulness_status"] == "unreviewed" for r in coverage["questions"]))
        curation.public_candidates(self.candidates, self.root / "review/batch/selection.private.json", self.root / "public")
        public = demo.qa.validate_public(read_json(self.root / "public/questions.json"))
        self.assertEqual(public["retrieval_contract"]["qa_preparation_policy"], {"id": policy.POLICY, "sha256": digest(policy.contract(policy.POLICY))})
        self.assertTrue(all("human_support_review" not in r for r in public["questions"]))

    def test_archive_presence_search_and_human_usefulness_are_separate_private_facts(self):
        inv = self.inventory()
        demo.archive.import_archive(self.package, self.root / "archive")
        corpus = self.root / "archive/corpus.json"
        identities = demo.archive.identity.Identities()
        empty = policy.archive_coverage(self.cases, inv, corpus, identities, Fingerprints())
        self.assertTrue(any(r["source_evidence_ids"] and r["search_status"] == ["not_documented"] for r in empty["buildings"]))
        note = self.root / "search-note.txt"
        note.write_text("Synthetic manually recorded archive search")
        case = next(c for c in self.cases if c["building_id"] == "b1")
        review = self.root / "coverage-review.private.json"
        write_json(review, demo.qa._private("www_archive_coverage_review", corpus_sha256=sha256(corpus), buildings=[
            {"building_id": "b2", "search_status": "searched_no_sources", "reviewer": "fixture", "notes": "Synthetic search found no source",
             "audit_sources": [{"path": str(note), "sha256": sha256(note)}]},
            {"building_id": "b3", "search_status": "not_searched", "reviewer": "fixture", "notes": "Synthetic search not yet done"}],
            support=[{"question_id": case["question_id"], "evidence_id": "fixture-archive-b1", "verdict": "useful",
                      "reviewer": "fixture-human", "notes": "Explicit synthetic support judgment"}]))
        report = policy.archive_coverage(self.cases, inv, corpus, identities, Fingerprints(), review)
        by_id = {r["physical_building_id"]: r for r in report["buildings"]}
        self.assertEqual(by_id["b2"]["search_status"], ["searched_no_sources"])
        self.assertEqual(by_id["b3"]["search_status"], ["not_searched"])
        self.assertEqual(by_id["b1"]["human_confirmed_useful"], 1)
        self.assertFalse(report["selection_filter"])
        from facade_change.agentic.store import PrivateEvidenceError, reject_forbidden
        with self.assertRaises(PrivateEvidenceError):
            reject_forbidden(report)


class SavedCostTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "calls").mkdir()

    def read(self, path, expected=None):
        if expected:
            self.assertEqual(sha256(path), expected)
        return read_json(path)

    def make_call(self, index, role, *, inputs=0, outputs=0, count=1, elapsed=1., status="ok", **extra):
        path = self.root / "calls" / f"{index:03d}-{role}.json"
        usage = {**dict.fromkeys(costs.FIELDS, 0), "input_tokens": inputs, "output_tokens": outputs,
                 "model_calls": count, "elapsed_seconds": elapsed, **extra.pop("usage", {})}
        response = {"status": status, "usage": usage, **extra}
        write_json(path, {"role": role, "request": {}, "response": response})
        return {"role": role, "artifact": str(path), "sha256": sha256(path), "usage": usage}

    def test_role_totals_errors_retries_cache_and_unlisted_calls_are_not_double_counted(self):
        entries = [self.make_call(0, "jina", inputs=17, elapsed=.4),
            self.make_call(1, "qwen", inputs=100, outputs=20, count=2, elapsed=2., usage={"retries": 1,
                "attempts": [{"status": "error", "usage": {"input_tokens": 20}}, {"status": "ok", "usage": {"input_tokens": 80}}]}),
            self.make_call(2, "qwen", inputs=30, status="error", elapsed=.5, usage={"incomplete": True}),
            self.make_call(3, "qwen", count=0, elapsed=.1, cache_hit=True, usage={"cache_hits": 1}, source_usage={"input_tokens": 300, "output_tokens": 50})]
        write_json(self.root / "costs.json", {"calls": entries, "qwen_requests": 3, "model_calls": 4,
            "retrieval_rounds": 1, "elapsed_seconds": 3.1})
        write_json(self.root / "retrieval-01.json", {"result": {"status": "matches"}})
        result = costs.case_costs({"status": "error"}, self.root, self.read)
        qwen, jina = result["roles"]["qwen"], result["roles"]["jina"]
        self.assertEqual((qwen["input_tokens"], qwen["output_tokens"], qwen["model_calls"], qwen["requests"]), (130, 20, 3, 3))
        self.assertEqual((jina["input_tokens"], jina["output_tokens"], jina["requests"]), (17, 0, 1))
        self.assertEqual(qwen["cached_source_input_tokens"], 300)
        self.assertEqual(qwen["attempt_status_counts"]["error"], 1)
        self.assertEqual(qwen["status_counts"]["error"], 1)
        self.assertTrue(result["incomplete"])
        self.make_call(4, "jina", inputs=5)
        extra = costs.case_costs({"status": "error"}, self.root, self.read)
        self.assertEqual(extra["roles"]["jina"]["input_tokens"], 22)
        self.assertTrue(any("absent_from_cost_ledger" in i for i in extra["issues"]))

    def test_resume_does_not_rebill_copied_calls_and_missing_rows_stay_in_denominator(self):
        self.make_call(0, "qwen", inputs=100, outputs=20)
        write_json(self.root / "costs.json", {"resumed_from": {"path": "original"}, "elapsed_seconds": 0,
            "cache_hits": 1, "source_costs": {"input_tokens": 100, "output_tokens": 20}})
        reused = costs.case_costs({"status": "completed"}, self.root, self.read)
        self.assertEqual(reused["roles"]["qwen"]["requests"], 0)
        self.assertEqual(reused["source_costs"]["input_tokens"], 100)
        missing = costs.case_costs({"status": "missing"}, self.root / "absent", self.read)
        rows = [{"condition": "photos_only", "status": "completed", "costs": reused},
                {"condition": "photos_only", "status": "missing", "costs": missing}]
        result = costs.aggregate(rows, ["photos_only"])["photos_only"]
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["status_counts"], {"completed": 1, "missing": 1})
        self.assertEqual(result["incomplete_rows"], 1)
        self.assertEqual(result["reused_result_rows"], 1)

    def test_missing_call_keeps_role_usage_and_unattributed_tokens_remain_explicit(self):
        entry = self.make_call(0, "qwen", inputs=71, outputs=9, status="error")
        Path(entry["artifact"]).unlink()
        write_json(self.root / "costs.json", {"calls": [entry], "qwen_requests": 1,
            "model_calls": 1, "input_tokens": 71, "output_tokens": 9, "retrieval_rounds": 0, "elapsed_seconds": 2})
        result = costs.case_costs({"status": "error"}, self.root, self.read)
        self.assertEqual(result["roles"]["qwen"]["input_tokens"], 71)
        self.assertEqual(result["roles"]["qwen"]["model_calls"], 1)
        self.assertEqual(result["roles"]["qwen"]["status_counts"], {"unknown": 1})
        self.assertTrue(result["incomplete"])
        self.assertIn("missing_call_artifact_using_recorded_usage:000-qwen.json", result["issues"])
        # An old aggregate without role/call artifacts cannot be declared a zero-cost run.
        write_json(self.root / "costs.json", {"input_tokens": 71, "output_tokens": 9,
            "retrieval_rounds": 0, "elapsed_seconds": 2})
        unknown = costs.case_costs({"status": "error"}, self.root, self.read)
        self.assertTrue(unknown["incomplete"])
        self.assertIn("saved_total_not_fully_attributed_by_role:input_tokens", unknown["issues"])
        self.assertEqual(unknown["ledger_totals_all_roles"]["input_tokens"], 71)


if __name__ == "__main__":
    unittest.main()
