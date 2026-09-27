import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from facade_change.alignment import SIFTMatcher
from facade_change.batch import run_batch
from facade_change.data import build_manifest
from facade_change.demo import make_fixture
from facade_change.geometry import transform_points
from facade_change.io import read_json, sha256
from facade_change.pipeline import run_pair
from facade_change.preparation import prepare_dataset


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required for integration")
class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        fixture = make_fixture(self.root / "fixture")
        build_manifest(fixture / "paths.json", self.root / "manifest", fixture / "reviewed.csv")
        prepare_dataset(self.root / "manifest/manifest.json", self.root / "prepared")
        self.manifest = self.root / "prepared/manifest.json"

    def tearDown(self):
        self.temp.cleanup()

    def test_batch_to_native_crops_and_controls_without_unnecessary_loftr(self):
        out = self.root / "batch"
        with patch("facade_change.pipeline.LoFTRMatcher", side_effect=AssertionError("Unnecessary LoFTR")) as loader:
            summary = run_batch(self.manifest, out, methods=["sift", "cascade"], crops=True, controls=1)
        loader.assert_not_called()
        self.assertEqual(summary["passed_routing_gate"], 2)
        self.assertEqual(summary["derivative_failures"], 0)
        derivatives = read_json(out / "derivatives.json")["0-1"]
        self.assertGreater(derivatives["crop_summary"]["crop_count"], 0)
        self.assertEqual(derivatives["control_summary"]["example_count"], 6)
        self.assertTrue((out / "comparison.html").is_file())
        for path, digest in read_json(out / "run.json")["artifact_sha256"].items():
            self.assertEqual(sha256(out / path), digest)

    def test_cascade_preserves_failed_attempt_and_recovers_known_geometry(self):
        class FailedMatcher:
            def match(self, *args):
                raise ValueError("Fixture has no first-stage matches")
        # Routing seam only; pretrained LoFTR is checked separately with actual weights.
        result = run_pair(self.manifest, 0, 1, self.root / "fallback", method="cascade",
                          matchers={"sift": FailedMatcher(), "loftr": SIFTMatcher()})
        self.assertEqual(result["selected_method"], "loftr")
        self.assertEqual([a["status"] for a in result["attempts"]], ["failed", "completed"])
        geometry = read_json(self.root / "fallback/geometry.json")
        probes = np.array([[80., 40.], [400., 340.]])
        np.testing.assert_allclose(transform_points(probes, geometry["source_to_reference"]),
                                   probes + [64, 0], atol=.5)

    def test_failed_method_does_not_remove_successful_comparison_or_denominator(self):
        out = self.root / "partial"
        with patch("facade_change.pipeline.LoFTRMatcher", side_effect=RuntimeError("Unavailable checkpoint")):
            summary = run_batch(self.manifest, out, methods=["loftr", "sift"])
        self.assertEqual(summary["attempted_runs"], 2)
        self.assertEqual(summary["failed_runs"], 1)
        self.assertEqual(summary["passed_routing_gate"], 1)
        self.assertEqual(read_json(out / "run.json")["status"], "completed_with_issues")
        self.assertIn("Unavailable checkpoint", (out / "comparison.html").read_text())

    def test_empty_requested_crops_are_reported_as_issue(self):
        with patch("facade_change.derived.build_crops", return_value={"crop_count": 0}):
            summary = run_batch(self.manifest, self.root / "empty", methods=["sift"], crops=True)
        self.assertEqual(summary["derivative_failures"], 1)
        self.assertEqual(read_json(self.root / "empty/run.json")["status"], "completed_with_issues")


if __name__ == "__main__":
    unittest.main()
