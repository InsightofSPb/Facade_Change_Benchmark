"""Synthetic private audit fixtures; no validation of the inaccessible owner package."""
import base64
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image

from facade_change.history_features import Fingerprints
from facade_change.io import read_json, sha256, write_json
from facade_change.agentic.store import PrivateEvidenceError, reject_forbidden

demo = importlib.import_module("facade_change.agentic.2026-11-15_www_demo")
curation = importlib.import_module("facade_change.agentic.2026-11-15_www_curation")
private = importlib.import_module("facade_change.agentic.2026-11-16_www_private_audit")
factual = importlib.import_module("facade_change.agentic.2026-11-16_www_factual")


def private_fixture(root, case):
    root.mkdir()
    image = root / "synthetic-extracted.png"
    Image.new("RGB", (1110, 640), (190, 150, 110)).save(image)
    assets = {"synthetic-extracted-photo": {"sha256": sha256(image), "width": 1110, "height": 640,
        "provenance": {"kind": "synthetic_pdf_extract", "parent": "synthetic-page", "locator": "page 46 photo 2"}}}
    write_json(root / private.REGISTER, {"access": "private_evaluation", "assets": assets})
    rows = [{"review_id": f"synthetic-review-{i:02d}", "kind": "region_audit" if i < 7 else "shortlisted_comparison",
        "building_id": case["building_id"] if i < 7 else "Voskova_10", "view_id": case["view_id"] if i == 0 else None,
        "region_id": case["region_id"] if i == 0 else None, "question_id": case["question_id"] if i == 0 else None,
        "event_category": "architectural_alteration" if i >= 7 else "unreviewed",
        "evidence_ids": ["fixture-archive-b1"], "uncertainty": "Synthetic pending correspondence",
        "panels": [{"asset": "synthetic-extracted-photo", "box_xyxy": [10, 20, 1100, 630]}]} for i in range(13)]
    write_json(root / private.PROPOSALS, {"access": "private_evaluation", "synthetic_fixture": True, "rows": rows})
    data = {"synthetic-extracted-photo": "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()}
    (root / private.GALLERY).write_text('<html><script>throw Error("DO_NOT_EXECUTE_PACKAGE_CODE")</script>'
        '<script id="asset-data" type="application/json">' + json.dumps(data) + '</script></html>')
    write_json(root / "2026-11-16-alias-audit.private.json", {"decision": "pending", "synthetic_fixture": True})
    write_json(root / "2026-11-16-derivative-audit.private.json", {"derivative": assets, "synthetic_fixture": True})
    (root / "unsafe.sh").write_text("exit 99\n")
    return root


def flat_private_fixture(root, cases):
    """Real reported schema/counts, entirely synthetic images, IDs and provenance."""
    root.mkdir()
    assets, data = {}, {}
    for i in range(41):
        key = f"synthetic-flat-{i:02d}"
        path = root / (key + ".png")
        size = (1110, 640) if i == 0 else (32, 24)
        Image.new("RGB", size, (i, 80, 160)).save(path)
        assets[key] = {"sha256": sha256(path), "width": size[0], "height": size[1],
            "label": "Synthetic native photo " + str(i), "evidence_id": "fixture-archive-b1",
            "capture_date": "2013", "precision": "year", "source_url": "https://example.invalid/synthetic",
            "note": "Fixture metadata, not a real survey image", "source_workspace_path": str(path),
            "delivery_location": private.GALLERY + "#asset-data", "extra_field": {"retained": i}}
        data[key] = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
    write_json(root / private.REGISTER, assets)
    (root / private.GALLERY).write_text('<script id="asset-data" type="application/json">' + json.dumps(data) + '</script>')
    rows = []
    for i in range(13):
        case = cases[i % 7]
        rows.append({"review_id": f"synthetic-flat-R-{i:02d}", "building_group": "Synthetic-proposed-group",
            "inventory_ids": [case["building_id"]], "local_view_ids": [case["view_id"]],
            "local_region_ids": [case["region_id"]] if i < 8 else [],
            "question_ids": [case["question_id"]] if i < 8 else [],
            "category": "damage_or_repair_question" if i < 8 else "architectural_alteration",
            "proposed_question": "A different proposed comparison; never replace the existing context question",
            "uncertainty": "Synthetic pending correspondence", "panels": [
                {"asset": f"synthetic-flat-{3*i+j:02d}", "box_xyxy": [1, 2, 20, 22]} for j in range(3)]})
    write_json(root / private.PROPOSALS, {"synthetic_fixture": True, "rows": rows})
    return root


class PrivateArchiveAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        _, package, self.config = demo.fixture(self.root / "fixture")
        self.candidates = self.root / "fixture/candidates/candidates.json"
        self.cases = read_json(self.candidates)["candidates"]
        self.case = next(c for c in self.cases if c["building_id"] == "b1")
        self.private = private_fixture(self.root / "private", self.case)
        # Same evidence ID, a full page with different dimensions/bytes. Never
        # substitute this page for the 1110×640 registered extracted photograph.
        Image.new("RGB", (1013, 1432), "white").save(self.root / "fixture/archive.png")
        document = read_json(package)
        document["records"][0]["assets"][0].update(width=1013, height=1432, sha256=sha256(self.root / "fixture/archive.png"))
        write_json(package, document)
        demo.archive.import_archive(package, self.root / "public-archive")
        self.corpus = self.root / "public-archive/corpus.json"

    def test_thirteen_pending_rows_exact_derivative_asset_and_private_boundary(self):
        proposals = read_json(self.private / private.PROPOSALS)
        proposals["corpus_sha256"] = "0" * 64  # A private audit of another corpus revision.
        write_json(self.private / private.PROPOSALS, proposals)
        target = {"question_id": self.case["question_id"], "group": "visible_change", "answer_ru": "Original synthetic human decision",
                  "clarification_verbatim": "Original synthetic case 03 clarification"}
        handoff = self.root / "handoff.private.json"
        write_json(handoff, {"source_candidates_sha256": sha256(self.candidates), "cases": [target]})
        before = handoff.read_bytes()
        summary = private.export(self.private, self.root / "audit", candidates_path=self.candidates,
            archive_corpus=self.corpus, handoff_path=handoff)
        self.assertEqual((summary["proposals"], summary["pending"], summary["approved"]), (13, 13, 0))
        self.assertEqual(summary["different_public_assets"], 13)
        result = read_json(self.root / "audit/audit.private.json")
        self.assertEqual(result["archive_corpus_sha256"], sha256(self.corpus))
        self.assertEqual(result["audit_claimed_corpus_sha256"], "0" * 64)
        self.assertEqual(result["corpus_binding"], "different_revision_pending_recheck")
        panel = result["rows"][0]["panels"][0]
        self.assertEqual(panel["shape"], [640, 1110])
        self.assertEqual(panel["public_assets"][0]["width"], 1013)
        with Image.open(self.root / "audit/assets/proposal-0-0.png") as image:
            self.assertEqual(image.size, (1110, 640))
            self.assertEqual(image.getpixel((1100, 30)), (30, 160, 255))
        self.assertEqual(result["rows"][0]["candidate_link"]["status"], "exact_identifiers")
        edited = read_json(self.root / "audit/candidate-review/drafts.private.json")
        self.assertEqual(edited["items"][0]["human_text_judgment"], target)
        self.assertEqual(handoff.read_bytes(), before)
        self.assertEqual(result["rows"][7]["event_category"], "architectural_alteration")
        self.assertIsNone(result["rows"][7]["candidate_link"]["question_id"])
        self.assertTrue(result["rows"][7]["identity_review_pending"])
        self.assertFalse(any(r["approved"] for r in result["rows"]))
        self.assertEqual((self.root / "audit/source.private" / private.PROPOSALS).read_bytes(), (self.private / private.PROPOSALS).read_bytes())
        self.assertIn("2026-11-16-derivative-audit.private.json", result["source_files"])
        self.assertFalse((self.root / "audit/source.private/unsafe.sh").exists())
        self.assertNotIn("DO_NOT_EXECUTE_PACKAGE_CODE", (self.root / "audit/gallery.html").read_text())
        with self.assertRaises(PrivateEvidenceError):
            reject_forbidden(result)
        with self.assertRaises(PrivateEvidenceError):
            demo.archive.validate_package(result, self.root / "audit", Fingerprints())
        with self.assertRaises((ValueError, PrivateEvidenceError)):
            demo.runner.public_questions(self.root / "audit/audit.private.json")

    def test_box_owner_mismatch_invalid_box_and_transfer_are_reported_without_repair(self):
        original = read_json(self.private / private.PROPOSALS)
        original["rows"][0]["panels"][0]["asset_sha256"] = "0"*64
        original["rows"][1]["panels"][0]["box_xyxy"] = [10, 20, 10, 20]
        original["rows"][2]["panels"][0]["display_asset"] = "public-page"
        write_json(self.private / private.PROPOSALS, original)
        summary = private.export(self.private, self.root / "audit", archive_corpus=self.corpus)
        self.assertEqual(summary["invalid_panels"], 3)
        result = read_json(self.root / "audit/audit.private.json")
        for i in range(3):
            panel = result["rows"][i]["panels"][0]
            self.assertEqual(panel["box_validation"], "error")
            self.assertEqual(panel["original_panel"], original["rows"][i]["panels"][0])
            self.assertFalse((self.root / f"audit/assets/proposal-{i}-0.png").exists())

    def test_asset_bytes_dimensions_and_zip_member_uniqueness(self):
        register = read_json(self.private / private.REGISTER)
        for key, value in (("sha256", "0"*64), ("width", 1013)):
            changed = copy.deepcopy(register)
            changed["assets"]["synthetic-extracted-photo"][key] = value
            write_json(self.private / private.REGISTER, changed)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "mismatch"):
                private.export(self.private, self.root / key)
        write_json(self.private / private.REGISTER, register)
        zipped = self.root / "private.zip"
        with zipfile.ZipFile(zipped, "w") as bundle:
            for path in self.private.iterdir():
                bundle.write(path, "package/" + path.name)
        self.assertEqual(private.export(zipped, self.root / "zip-review")["pending"], 13)

    def test_pending_review_survives_zero_batch_and_explains_cross_split_guard(self):
        proposals = read_json(self.private / private.PROPOSALS)
        for row in proposals["rows"][7:]:
            row["inventory_ids"] = [row.pop("building_id")]
        write_json(self.private / private.PROPOSALS, proposals)
        identity_path = self.root / "identities.json"
        mapping = demo.archive.identity.template()
        mapping["groups"][0].update(decision="same_building", reviewer="synthetic_human", rationale="Fixture only",
            sources=[{"path": str(self.private / private.REGISTER), "sha256": sha256(self.private / private.REGISTER)}])
        write_json(identity_path, mapping)
        identities = demo.archive.identity.Identities(identity_path, Fingerprints())
        views = [{"building_id": name, "view_id": "front", "split": split, "cohort": "development",
            "exposure_status": "fixture", "eligible": True, "exclusion_reasons": []} for name, split in (("Voskova_10", "val"), ("Voskova_12", "test"))]
        identities.annotate_inventory(views)
        self.assertTrue(all(not v["eligible"] for v in views))
        write_json(self.root / "inventory.json", {"identity_source": identities.source, "views": views})
        summary = private.export(self.private, self.root / "audit", inventory_path=self.root / "inventory.json", identity_map=identity_path)
        self.assertEqual(summary["pending"], 13)
        self.assertEqual(summary["cross_split_review_rows"], 6)
        item = read_json(self.root / "audit/audit.private.json")["rows"][7]
        self.assertIn("BOTH development and evaluation", item["cross_split_consequence"])
        self.assertEqual(item["decision"], "pending")
        # The existing package has no eligible fixture views without an exposure
        # audit. Its independent pending section must still retain all 13 rows.
        supplement = self.root / "supplement"
        supplement.mkdir()
        (supplement / "comparison-cards.md").write_text("Synthetic V4.1 comparison card, pending")
        result = curation.review_package(self.candidates, self.root / "fixture/inputs/histories", self.corpus,
            self.root / "package-review", private_audit=self.private, private_audit_supplement=supplement)
        self.assertEqual(result["selected"], 0)
        self.assertEqual(result["shortfall"], 100)
        self.assertEqual(result["pending_private_audit"]["pending"], 13)
        self.assertEqual(result["pending_private_audit"]["supplement_files"], 1)
        self.assertIn("pending-audit/index.html", (self.root / "package-review/index.html").read_text())

    def test_flat_register_plural_ids_link_only_existing_question_context(self):
        package = flat_private_fixture(self.root / "flat", self.cases)
        zipped = self.root / "2026-11-16-www-v4-private-region-review.zip"
        with zipfile.ZipFile(zipped, "w") as bundle:
            for name in (private.PROPOSALS, private.REGISTER, private.GALLERY):
                bundle.write(package / name, name)
        before = {p: p.read_bytes() for p in (self.candidates, package / private.REGISTER, package / private.PROPOSALS)}
        summary = private.export(zipped, self.root / "flat-review", candidates_path=self.candidates, archive_corpus=self.corpus)
        self.assertEqual([summary[k] for k in ("assets", "panels", "invalid_panels", "pending", "linked_candidate_rows",
            "unique_linked_questions", "unresolved_candidate_links", "approved")], [41, 39, 0, 13, 8, 7, 5, 0])
        result = read_json(self.root / "flat-review/audit.private.json")
        for key, original in read_json(package / private.REGISTER).items():
            normalized = result["assets"][key]
            self.assertEqual({k: normalized[k] for k in original}, original)
            self.assertEqual(normalized["provenance"]["source_fields"]["source_workspace_path"], original["source_workspace_path"])
            self.assertEqual(sha256(self.root / "flat-review" / normalized["src"]), original["sha256"])
        self.assertEqual((self.root / "flat-review/source.private" / private.REGISTER).read_bytes(), before[package / private.REGISTER])
        self.assertTrue(all(r["decision"] == "pending" and not r["approved"] for r in result["rows"]))
        self.assertTrue(all(r["candidate_link"]["context_only"] for r in result["rows"]))
        self.assertEqual(result["rows"][0]["category"], "damage_or_repair_question")
        self.assertEqual(result["rows"][8]["category"], "architectural_alteration")
        self.assertEqual(result["rows"][8]["event_category"], "unreviewed")
        editor = read_json(self.root / "flat-review/candidate-review/drafts.private.json")
        self.assertEqual({r["question_id"]: r["question"] for r in editor["items"]},
                         {r["question_id"]: r["question"] for r in self.cases[:7]})
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        with self.assertRaises(PrivateEvidenceError):
            reject_forbidden(result)

    def test_flat_metadata_does_not_bypass_source_or_asset_validation(self):
        package = flat_private_fixture(self.root / "flat", self.cases)
        raw = private.package_bytes(package, Fingerprints())
        register = json.loads(raw[private.REGISTER])
        key = "synthetic-flat-40"  # Unused by the 39 panels, still fully verified.
        out = self.root / "loaded"
        (out / "assets").mkdir(parents=True)
        for change in ({"sha256": "0"*64}, {"width": 31}, {"provenance": {}},
                       {field: None for field in ("evidence_id", "source_url", "source_workspace_path", "delivery_location")}):
            changed = copy.deepcopy(register)
            changed[key].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                private.load_assets({**raw, private.REGISTER: json.dumps(changed).encode()}, out)
        bmp = self.root / "unsupported.bmp"
        Image.new("RGB", (32, 24)).save(bmp)
        register[key]["sha256"] = sha256(bmp)
        html = raw[private.GALLERY].decode()
        parser = private.AssetData()
        parser.feed(html)
        embedded = json.loads("".join(parser.parts))
        embedded[key] = "data:image/png;base64," + base64.b64encode(bmp.read_bytes()).decode()
        with self.assertRaisesRegex(ValueError, "format mismatch"):
            private.load_assets({**raw, private.REGISTER: json.dumps(register).encode(),
                private.GALLERY: ('<script id="asset-data" type="application/json">' + json.dumps(embedded) + '</script>').encode()}, out)

    def test_plural_identifier_ambiguity_conflicts_and_proposed_group(self):
        # Reported identifier structure, tested independently from any real image.
        row = {"review_id": "R-V1", "building_group": "Voskova12-proposed-group", "inventory_ids": ["Voskova_10"],
            "local_view_ids": ["Voskova_10"], "local_region_ids": ["5c246d589c516955e16a23dd"],
            "question_ids": ["qa-8ed445b0c2faaca6b3f4265b"], "category": "damage_or_repair_question"}
        case = {**self.case, "building_id": "Voskova_10", "view_id": "Voskova_10", "region_id": row["local_region_ids"][0],
                "question_id": row["question_ids"][0]}
        linked = private.candidate_link(row, [case])
        self.assertEqual(linked["question_id"], case["question_id"])
        self.assertNotIn("building_group", linked["supplied_identifiers"])
        alternative = {**case, "question_id": "another-explicit-question"}
        both = {**row, "question_ids": [case["question_id"], alternative["question_id"]]}
        ambiguous = private.candidate_link(both, [case, alternative])
        self.assertTrue(ambiguous["ambiguous"])
        self.assertIsNone(ambiguous["question_id"])
        conflict = private.candidate_link({**row, "question_id": alternative["question_id"]}, [case, alternative])
        self.assertEqual(conflict["identifier_conflicts"], ["question_id"])
        self.assertIsNone(conflict["question_id"])
        self.assertIsNone(private.candidate_link({**row, "local_view_ids": ["other-view"]}, [case])["question_id"])
        self.assertIsNone(private.candidate_link({"review_id": "group-only", "building_group": "Voskova_10"}, [case])["question_id"])
        with self.assertRaisesRegex(ValueError, "list of nonempty IDs"):
            private.candidate_link({**row, "question_ids": case["question_id"]}, [case])

    def test_v41_supplement_exposes_every_member_without_projecting_or_overriding(self):
        supplement = self.root / "2026-11-16-v4.1-correction-private-handoff.zip"
        files = {"coordinate-mappings.json": json.dumps({"source_asset": "synthetic-extracted-photo",
                    "source_sha256": "0"*64, "preview_box": [1, 2, 3, 4], "status": "needs_verification"}).encode(),
            "comparison-cards.html": b'<h1>Pending comparison</h1><script>NEVER_EXECUTE_SUPPLEMENT()</script>',
            "nested/notes.md": b'Synthetic correction note', private.REGISTER: b'{"must_not_override_base":true}',
            "unknown.bin": b'\x80\x00\xff'}
        with zipfile.ZipFile(supplement, "w") as bundle:
            for name, content in files.items():
                bundle.writestr(name, content)
        before = supplement.read_bytes()
        summary = private.export(self.private, self.root / "supplement-review", supplement_path=supplement)
        self.assertEqual((summary["supplement_files"], summary["invalid_panels"], summary["approved"]), (5, 0, 0))
        self.assertFalse(summary["supplemental_coordinates_applied"])
        result = read_json(self.root / "supplement-review/audit.private.json")
        for name, content in files.items():
            target = self.root / "supplement-review" / result["supplement_files"][name]["saved_as"]
            self.assertEqual(target.read_bytes(), content)
            self.assertEqual(sha256(target), result["supplement_files"][name]["sha256"])
            self.assertEqual(target.suffix, ".txt")
        page = (self.root / "supplement-review/supplement/gallery.html").read_text()
        self.assertIn("&lt;script&gt;NEVER_EXECUTE_SUPPLEMENT()&lt;/script&gt;", page)
        self.assertNotIn("<script>", page)
        self.assertIn("supplement/gallery.html", (self.root / "supplement-review/index.html").read_text())
        with Image.open(self.root / "supplement-review/assets/proposal-0-0.png") as image:
            self.assertEqual(image.size, (1110, 640))
            self.assertEqual(image.getpixel((1100, 30)), (30, 160, 255))
        self.assertEqual(supplement.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "private-audit-supplement"):
            private.export(supplement, self.root / "wrong-base")
        with self.assertRaisesRegex(ValueError, "requires the original"):
            curation.review_page(self.candidates, "unused-selection", self.root / "no-base", private_audit_supplement=supplement)


class FactualAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.questions, package, self.config = demo.fixture(self.root / "fixture")
        demo.archive.import_archive(package, self.root / "archive")
        demo.archive.index_archive(self.root / "archive/corpus.json", self.config, self.root / "index", pool_factory=demo.SyntheticPool, synthetic=True)
        self.index = self.root / "index/index.json"

    def test_unscored_interrupted_48_row_fixture_preserves_statuses_and_blank_judgments(self):
        document = read_json(self.questions)
        originals = document["questions"]
        document["questions"] = [{**copy.deepcopy(originals[i % len(originals)]), "question_id": f"synthetic-question-{i:02d}"} for i in range(16)]
        questions = self.root / "16-questions.json"
        write_json(questions, document)
        class Outcomes(demo.SyntheticPool):
            def call(self, role, payload):
                if role == "qwen":
                    identifier = payload["geometry"]["question_id"]
                    if identifier.endswith("15"):
                        raise KeyboardInterrupt("Synthetic interruption, not the owner's pilot")
                    if identifier.endswith("03"):
                        return {"status": "error", "error": "Synthetic worker failure", "usage": {}}
                result = super().call(role, payload)
                if role == "qwen" and identifier.endswith("04"):
                    value = json.loads(result["text"])
                    if value["type"] == "final":
                        value.update(decision="insufficient_data", observation_ids=[], date_interval=None, boxes=[], limitations=["Synthetic ambiguous view"])
                        result["text"] = json.dumps(value)
                return result
        with self.assertRaises(KeyboardInterrupt):
            demo.runner.run(questions, self.config, self.root / "run", archive_index=self.index, pool_factory=Outcomes, synthetic=True)
        before = {p: sha256(p) for p in (self.root / "run").rglob("*") if p.is_file()}
        summary = factual.audit(self.root / "run", self.root / "audit")
        self.assertEqual(summary["expected_rows"], 48)
        self.assertEqual(summary["exported_rows"], 48)
        # Five archival questions now abstain in photos_only: local citations
        # cannot substantiate archive contents. All 48 rows remain in the audit.
        self.assertEqual(summary["execution_status_counts"], {"completed": 34, "error": 3, "abstained": 8, "missing": 3})
        self.assertEqual(summary["human_reviewed"], 0)
        self.assertFalse(summary["models_executed"])
        payload = read_json(self.root / "audit/blind-review.private.json")
        self.assertTrue(all(row[k] == "" for row in payload["cases"] for k in factual.JUDGMENTS))
        self.assertNotIn('"condition"', json.dumps(payload))
        self.assertNotIn("photos_only", (self.root / "audit/factual-review.html").read_text())
        self.assertEqual(len(read_json(self.root / "audit/blind-key.private.json")["cases"]), 48)
        self.assertFalse((self.root / "audit/plan.private.json").exists())
        self.assertEqual({p: sha256(p) for p in before}, before)

    def test_staged_archive_is_not_shown_without_a_subsequent_qwen_request(self):
        self.config["budgets"]["max_model_calls"] = 1
        demo.runner.run(self.questions, self.config, self.root / "run", archive_index=self.index,
            conditions=("bounded_agentic_retrieval",), pool_factory=demo.SyntheticPool, synthetic=True)
        factual.audit(self.root / "run", self.root / "audit")
        keys = read_json(self.root / "audit/blind-key.private.json")["cases"]
        self.assertTrue(any("fixture-archive-b1" in r["exposure_ids_not_in_saved_requests"] for r in keys))
        payload = read_json(self.root / "audit/blind-review.private.json")
        self.assertTrue(all(s["evidence_id"] != "fixture-archive-b1" for r in payload["cases"] for s in r["evidence"]))

    def test_shown_uncited_archive_and_handoff_target_are_private_and_never_auto_judged(self):
        demo.runner.run(self.questions, self.config, self.root / "run", archive_index=self.index,
            pool_factory=demo.SyntheticPool, synthetic=True)
        candidates = self.root / "fixture/candidates/candidates.json"
        case = read_json(self.questions)["questions"][0]
        target = {"question_id": case["question_id"], "answer_ru": "Different synthetic target event", "clarification_verbatim": "Keep this exact clarification"}
        handoff = self.root / "handoff.private.json"
        write_json(handoff, {"source_candidates_sha256": sha256(candidates), "cases": [target]})
        factual.audit(self.root / "run", self.root / "audit", candidates_path=candidates, handoff_path=handoff)
        payload = read_json(self.root / "audit/blind-review.private.json")
        self.assertEqual(sum(r["target_event"] == target for r in payload["cases"]), 3)
        self.assertTrue(any(s["evidence_id"] == "fixture-archive-b1" and not s["cited"] for r in payload["cases"] for s in r["evidence"]))
        self.assertTrue(all(r["claim_support"] == "" for r in payload["cases"]))
        with self.assertRaises(PrivateEvidenceError):
            reject_forbidden(payload)


if __name__ == "__main__":
    unittest.main()
