"""Private transfer keeps source bytes, geometry and annotation channel order intact."""

import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from facade_change.io import read_json, sha256, write_json

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/2026-11-16-package-www-review.py"
SPEC = importlib.util.spec_from_file_location("review_bundle", SCRIPT)
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)
demo = importlib.import_module("facade_change.agentic.2026-11-15_www_demo")


class ReviewBundleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        demo.fixture(self.root / "fixture")
        self.candidates = self.root / "fixture/candidates/candidates.json"
        rows = read_json(self.candidates)["candidates"]
        self.case = next(r for r in rows if r["region_id"] == "change")
        self.package = self.root / "review-package"
        (self.package / "batch").mkdir(parents=True)
        self.selection = self.package / "batch/selection.private.json"
        write_json(self.selection, {"kind": "www_question_selection", "candidate_sha256": sha256(self.candidates),
                                   "question_ids": [self.case["question_id"]]})

    def test_complete_inputs_are_bound_to_unchanged_candidates(self):
        original = self.candidates.read_bytes()
        out = self.root / "2026-11-16-bundle.zip"
        summary = bundle.collect(self.candidates, self.package, out)
        self.assertEqual(summary["questions"], 1)
        self.assertEqual(summary["automatic_approvals"], 0)
        with zipfile.ZipFile(out) as archive:
            packet = json.loads(archive.read("2026-11-16-review-inputs.private.json"))
            mapping = packet["archive_paths"]
            self.assertEqual(archive.read(mapping[str(self.candidates)]), original)
            self.assertEqual(packet["cases"], [self.case])
            self.assertEqual(packet["annotation_contract"]["category_to_channel"],
                             [{"category_id": 4, "channel": 0}, {"category_id": 9, "channel": 1}])
            for observation in packet["selected_history_tracks"][0]["observations"]:
                original_path = observation["original_image_path"]
                self.assertEqual(archive.read(mapping[original_path]), Path(original_path).read_bytes())
            for evidence in self.case["annotation_evidence"].values():
                self.assertEqual(archive.read(mapping[evidence["path"]]), Path(evidence["path"]).read_bytes())
            self.assertEqual(archive.testzip(), None)
        self.assertEqual(self.candidates.read_bytes(), original)
        with self.assertRaises(FileExistsError):
            bundle.collect(self.candidates, self.package, out)

    def test_other_selection_is_rejected_before_output(self):
        selection = read_json(self.selection)
        selection["candidate_sha256"] = "0" * 64
        write_json(self.selection, selection)
        out = self.root / "2026-11-16-wrong.zip"
        with self.assertRaisesRegex(ValueError, "разным исходным"):
            bundle.collect(self.candidates, self.package, out)
        self.assertFalse(out.exists())

    def test_changed_rgb_is_not_packaged_as_verified(self):
        path = Path(self.case["observations"][0]["rgb_path"])
        path.write_bytes(path.read_bytes() + b"modified")
        out = self.root / "2026-11-16-stale.zip"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            bundle.collect(self.candidates, self.package, out)
        self.assertFalse(out.exists())
