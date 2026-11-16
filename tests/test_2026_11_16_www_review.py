"""CPU checks for prospective review selection, reviewed identities and building resampling."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest

from facade_change.history_features import Fingerprints
from facade_change.io import read_json, sha256, write_json

demo = importlib.import_module("facade_change.agentic.2026-11-15_www_demo")
curation = importlib.import_module("facade_change.agentic.2026-11-15_www_curation")
metrics = importlib.import_module("facade_change.agentic.2026-11-15_www_evaluation")
identity = demo.archive.identity


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "identity.json"
        self.source = self.root / "identity-source.txt"
        self.source.write_text("Synthetic human identity review source")
        self.document = identity.template()

    def load(self, reviewed=False):
        if reviewed:
            for group in self.document["groups"]:
                group.update(decision="same_building", reviewer="synthetic_human", rationale="Explicit fixture review",
                             sources=[{"path": str(self.source), "sha256": sha256(self.source)}])
        write_json(self.path, self.document)
        return identity.Identities(self.path, Fingerprints())

    def test_pending_aliases_do_not_merge_or_erase_known_view_restrictions(self):
        ids = self.load()
        record = {"building_id": "Voskova_12", "view_id": "front"}
        question = {"building_id": "Voskova_10", "view_id": "front"}
        self.assertFalse(ids.compatible(record, question))
        self.assertEqual(ids.physical("Voskova_12"), "Voskova_12")
        ids = self.load(reviewed=True)
        self.assertTrue(ids.compatible(record, question))
        self.assertFalse(ids.compatible({**record, "view_id": "rear"}, question))
        self.assertEqual(record["building_id"], "Voskova_12")
        self.assertEqual(ids.physical("Voskova_12"), "Voskova_10")

    def test_review_requires_sources_and_revalidates_their_bytes(self):
        self.document["groups"][0]["decision"] = "same_building"
        with self.assertRaisesRegex(ValueError, "human reviewer"):
            self.load()
        self.load(reviewed=True)
        self.source.write_text("changed review source")
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            identity.Identities(self.path, Fingerprints())

    def test_split_and_exposure_history_survives_grouping(self):
        ids = self.load(reviewed=True)
        rows = [{"building_id": name, "view_id": "front", "split": split, "cohort": cohort,
                 "exposure_status": cohort, "eligible": True, "exclusion_reasons": []}
                for name, split, cohort in (("Voskova_10", "val", "development"), ("Voskova_12", "test", "evaluation"))]
        ids.annotate_inventory(rows)
        self.assertEqual([r["split"] for r in rows], ["val", "test"])
        self.assertEqual([r["original_cohort"] for r in rows], ["development", "evaluation"])
        self.assertTrue(all(r["cohort"] == "development" and not r["eligible"] for r in rows))
        self.assertTrue(all("physical_building_crosses_original_splits" in r["exclusion_reasons"] for r in rows))
        self.assertEqual({r["physical_building_id"] for r in rows}, {"Voskova_10"})

    def test_pending_or_incompletely_audited_group_is_not_evaluation(self):
        row = {"building_id": "Voskova_10", "view_id": "front", "split": "test", "cohort": "evaluation",
               "exposure_status": "audited", "eligible": True, "exclusion_reasons": []}
        self.assertFalse(self.load().annotate_inventory([copy.deepcopy(row)])[0]["eligible"])
        checked = self.load(reviewed=True).annotate_inventory([row])[0]
        self.assertFalse(checked["eligible"])
        self.assertIn("physical_group_exposure_not_fully_audited", checked["exclusion_reasons"])

    def test_explicit_audit_of_absent_alias_can_complete_group_exposure_review(self):
        row = {"building_id": "Voskova_10", "view_id": "front", "split": "test", "cohort": "evaluation",
               "exposure_status": "audited", "eligible": True, "exclusion_reasons": []}
        ids = self.load(reviewed=True)
        checked = ids.annotate_inventory([row], audited_buildings={"Voskova_12"})[0]
        self.assertTrue(checked["eligible"])
        self.assertEqual(checked["original_cohort"], "evaluation")


class SelectionTests(unittest.TestCase):
    def fixtures(self):
        rows, views = [], []
        for b in range(10):
            building = f"building-{b}"
            views.append({"building_id": building, "view_id": "front", "physical_building_id": building,
                          "eligible": True, "cohort": "development"})
            for family in curation.REVIEW_FAMILIES:
                for n in range(6):
                    offset = (n + list(curation.REVIEW_FAMILIES).index(family)*10)*100
                    rows.append({"question_id": f"q-{b}-{family}-{n}", "building_id": building, "view_id": "front",
                        "task": "earliest_evidence" if family == "earliest_damage" else "temporal_change",
                        "task_dates": ["2010", "2020"], "candidate_type": family,
                        "frame": {"anchor_sha256": building, "anchor_bbox_xyxy": [offset, 0, offset+80, 80]}})
        return rows, {"views": views}

    def test_targets_balance_buildings_and_are_order_independent(self):
        rows, inventory = self.fixtures()
        selected, _ = curation.select_batch(rows, inventory, targets=curation.REVIEW_TARGETS)
        reversed_selection, _ = curation.select_batch(rows[::-1], inventory, targets=curation.REVIEW_TARGETS)
        self.assertEqual([r["question_id"] for r in selected], [r["question_id"] for r in reversed_selection])
        self.assertEqual(len(selected), 100)
        from collections import Counter
        self.assertEqual(Counter(curation.REVIEW_FAMILIES[r["candidate_type"]] for r in selected), curation.REVIEW_TARGETS)
        self.assertLessEqual(max(Counter(r["building_id"] for r in selected).values()), 12)
        self.assertEqual(len({r["building_id"] for r in selected}), 10)
        self.assertTrue(all("assessment" not in r for r in selected))

    def test_overlapping_same_event_is_removed_but_later_episode_survives(self):
        rows, inventory = self.fixtures()
        first = rows[6]  # A temporal case.
        duplicate = {**copy.deepcopy(first), "question_id": "overlapping-crop"}
        duplicate["frame"]["anchor_bbox_xyxy"][2] -= 1
        later = {**copy.deepcopy(first), "question_id": "later-episode", "task_dates": ["2021", "2022"]}
        targets = {k: int(k == "visible_change")*3 for k in curation.REVIEW_TARGETS}
        selected, excluded = curation.select_batch([first, duplicate, later], inventory, targets=targets, size=3)
        self.assertEqual(len(selected), 2)
        self.assertIn("later-episode", {r["question_id"] for r in selected})
        self.assertEqual(excluded[0]["reason"], "overlapping_same_interval_event_proposal")
        other_task = {**copy.deepcopy(first), "task": "earliest_evidence"}
        self.assertTrue(curation.same_event_candidate(first, other_task, .5))

    def test_exposed_questions_and_physical_alias_caps_cannot_fill_shortfall(self):
        rows, inventory = self.fixtures()
        for view in inventory["views"]:
            view["physical_building_id"] = "one-physical-building"
        selected, excluded = curation.select_batch(rows, inventory, targets=curation.REVIEW_TARGETS, excluded_cases=[rows[0]])
        self.assertLessEqual(len(selected), 12)
        self.assertNotIn(rows[0]["question_id"], {r["question_id"] for r in selected})
        self.assertIn("per_building_cap", {r["reason"] for r in excluded})
        self.assertIn("previously_exposed_question_or_overlapping_event", {r["reason"] for r in excluded})


class AggregationTests(unittest.TestCase):
    def rows(self):
        rows = []
        for condition in demo.runner.CONDITIONS:
            for q, building, physical, assessment, decision, status in (
                ("q1", "alias-a", "physical-a", "confirmed_change", "change", "completed"),
                ("q2", "alias-b", "physical-a", "confirmed_change", None, "missing"),
                ("q3", "building-b", "physical-b", "confirmed_stability", "no_change", "completed"),
                ("q4", "building-b", "physical-b", "insufficient_evidence", None, "error")):
                rows.append({"question_id": q, "building_id": building, "physical_building_id": physical,
                    "assessment": assessment, "condition": condition, "decision": decision, "status": status,
                    "correct": status == "completed", "costs": {}, "legitimate_abstention": False,
                    "known_archive_support_count": 0, "known_support_hits": 0, "valid_citations": 0, "citations": 0})
        return rows

    def test_macro_f1_includes_missing_and_errors_without_relabeling_insufficient(self):
        rows = self.rows()[:4]
        result = metrics.aggregate(rows)
        self.assertEqual(result["questions"], 4)
        self.assertEqual(result["independent_buildings"], 2)
        self.assertEqual(result["structured_accuracy"], .5)
        self.assertEqual(result["change_stability"]["questions"], 3)
        self.assertAlmostEqual(result["change_stability"]["macro_f1"], 5/6)
        self.assertEqual(result["change_stability"]["classes"]["change"]["fn"], 1)
        for row in rows:
            row["status"] = "error"
        self.assertEqual(metrics.change_stability(rows)["macro_f1"], 0.)

    def test_paired_seed_reproducibility_and_physical_cluster_unit(self):
        rows = self.rows()
        a, b = demo.runner.CONDITIONS[:2]
        first = metrics.paired_intervals(rows, a, b)
        self.assertEqual(first, metrics.paired_intervals(rows[::-1], a, b))
        self.assertEqual(first["independent_buildings"], 2)
        self.assertEqual(first["building_macro_accuracy"]["interval"], [0., 0.])
        self.assertEqual(first["change_stability_macro_f1"]["interval"], [0., 0.])
        for row in rows:
            if row["condition"] == b:
                row.update(correct=False, status="error", decision=None)
        changed = metrics.paired_intervals(rows, a, b)
        self.assertEqual(changed["building_macro_accuracy"]["difference"], -.5)
        self.assertEqual(changed["building_macro_accuracy"]["interval"], [-.5, -.5])
        singleton = [r for r in rows if r["physical_building_id"] == "physical-a"]
        self.assertIsNone(metrics.paired_intervals(singleton, a, b)["building_macro_accuracy"]["interval"])
        with self.assertRaisesRegex(ValueError, "same frozen"):
            metrics.paired_intervals(rows[1:], a, b)

    def test_non_binary_buildings_do_not_supply_fake_zero_f1_resamples(self):
        rows = self.rows()
        for row in rows:
            if row["physical_building_id"] == "physical-b":
                row["assessment"] = "earliest_presence"
        a, b = demo.runner.CONDITIONS[:2]
        result = metrics.paired_intervals(rows, a, b)
        self.assertEqual(result["independent_buildings"], 2)
        self.assertIsNotNone(result["building_macro_accuracy"]["interval"])
        self.assertEqual(result["change_stability_macro_f1"]["independent_buildings"], 1)
        self.assertIsNone(result["change_stability_macro_f1"]["interval"])


class PackageTests(unittest.TestCase):
    def test_absent_reviewed_alias_propagates_known_exposure_without_erasing_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            questions, _, _ = demo.fixture(root / "fixture")
            mapping = identity.template()
            mapping["groups"] = [{"physical_building_id": "b1", "building_ids": ["b1", "previous-id"],
                "decision": "same_building", "reviewer": "synthetic_human", "rationale": "Explicit fixture only",
                "sources": [{"path": str(questions), "sha256": sha256(questions)}]}]
            write_json(root / "identity.json", mapping)
            write_json(root / "audit.json", {"kind": "www_exposure_audit", "buildings": [
                {"building_id": "previous-id", "cohort": "development"}]})
            exposed = read_json(questions)
            exposed["questions"] = [q for q in exposed["questions"] if q["building_id"] == "b1"]
            for question in exposed["questions"]:
                question["building_id"] = "previous-id"
            write_json(root / "exposed.json", exposed)
            for source in ("audit", "public"):
                options = {"exposure_path": root / "audit.json"} if source == "audit" else {"exposed_questions": [root / "exposed.json"]}
                curation.inventory(root / "fixture/inputs/histories", root / source,
                    identity_map=root / "identity.json", **options)
                rows = read_json(root / source / "inventory.json")["views"]
                row = next(r for r in rows if r["building_id"] == "b1")
                self.assertEqual(row["cohort"], "development")
                self.assertEqual(row["original_cohort"], "unassigned")
                self.assertTrue(row["eligible"])
                self.assertIn("physical_building_id", (root / source / "gallery.html").read_text())
                if source == "public":
                    self.assertTrue(row["known_exposure_manifests"])

    def test_export_shortfall_archive_metadata_and_preserved_human_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, package, config = demo.fixture(root / "fixture")
            candidates_path = root / "fixture/candidates/candidates.json"
            cases = read_json(candidates_path)["candidates"]
            old = cases[:2]
            handoff = root / "handoff.private.json"
            original = {"source_candidates_sha256": sha256(candidates_path), "cases": [
                {"question_id": r["question_id"], "group": "visible_change", "answer_ru": "UNCHANGED human text",
                 "clarification_verbatim": "CASE 03 preserved clarification"} for r in old]}
            write_json(handoff, original)
            selection = root / "selection.private.json"
            write_json(selection, demo.qa._private("www_question_selection", candidate_sha256=sha256(candidates_path),
                                                  question_ids=[r["question_id"] for r in old]))
            audit = root / "audit.json"
            write_json(audit, {"kind": "www_exposure_audit", "buildings": [
                {"building_id": b, "cohort": "development"} for b in {r["building_id"] for r in cases}]})
            demo.archive.import_archive(package, root / "archive")
            events = []
            summary = curation.review_package(candidates_path, root / "fixture/inputs/histories", root / "archive/corpus.json", root / "review-package",
                exposure_path=audit, handoff_path=handoff, preserved_selection=selection,
                progress=lambda event, **values: events.append(event))
            self.assertEqual(summary["approved"], 0)
            self.assertFalse(summary["models_executed"])
            self.assertFalse(summary["full_target_cohort_created"])
            self.assertEqual(summary["shortfall"], 100-summary["selected"])
            self.assertEqual(sum(summary["target_shortfalls"].values()), summary["shortfall"])
            self.assertIn("review_stage", events)
            self.assertIn("case_started", events)
            saved = root / "review-package"
            self.assertEqual((saved / "preserved-handoff.private.json").read_bytes(), handoff.read_bytes())
            kept = read_json(saved / "preserved-review/drafts.private.json")
            self.assertEqual([r["human_text_judgment"] for r in kept["items"]], original["cases"])
            fresh = read_json(saved / "review/drafts.private.json")
            all_items = fresh["items"] + kept["items"]
            self.assertTrue(all(r["label"]["events"] == [] and not r["label"]["supporting_evidence_ids"] for r in all_items))
            self.assertTrue(all(not b["confirmed"] for r in all_items for b in r["proposed_boxes"]))
            archive = [a for r in all_items for a in r["archive_evidence"]]
            self.assertTrue(archive)
            self.assertTrue(all(a["record_sha256"] and a["observation_date"] and a["source_page_url"] for a in archive))
            page = (saved / "review/review.html").read_text()
            self.assertIn("www_archive_correspondence_proposals", page)
            self.assertTrue((saved / "index.html").is_file())
            demo.qa.validate_public(read_json(saved / "preserved-public/questions.json"))
            partial = read_json(selection)
            partial["question_ids"] = partial["question_ids"][:1]
            write_json(root / "partial-selection.json", partial)
            with self.assertRaisesRegex(ValueError, "every original human decision"):
                curation.review_package(candidates_path, root / "fixture/inputs/histories", root / "archive/corpus.json", root / "partial-review",
                    handoff_path=handoff, preserved_selection=root / "partial-selection.json")


if __name__ == "__main__":
    unittest.main()
