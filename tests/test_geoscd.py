"""CPU contract tests only; they do not validate pretrained VGGT facade quality."""
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from facade_change.data import build_manifest
from facade_change.demo import make_fixture
from facade_change.geoscd import GEOSCD_COMMIT, dense_warp, native_remap, relative_camera_transform, run_geoscd
from facade_change.io import read_json, sha256, write_json
from facade_change.preparation import prepare_dataset


def grid(size=518):
    yy, xx = np.mgrid[:size, :size]
    return np.stack((xx, yy), axis=2).astype(np.float32)


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required")
class DenseCoordinateTests(unittest.TestCase):
    def test_identity_keeps_every_native_border_pixel_on_unequal_grids(self):
        x, y, valid = native_remap(grid(), (7, 13), (11, 19))
        np.testing.assert_allclose(x, np.broadcast_to((np.arange(13) + .5) * 19 / 13 - .5, (7, 13)), atol=2e-6)
        np.testing.assert_allclose(y, np.broadcast_to(((np.arange(7) + .5) * 11 / 7 - .5)[:, None], (7, 13)), atol=2e-6)
        self.assertTrue(valid.all())

    def test_inverse_direction_translation_and_native_scale_are_explicit(self):
        coordinates = grid() + [2.5, -1.25]
        x, y, valid = native_remap(coordinates, (15, 21), (30, 42))
        np.testing.assert_allclose(x[0], (np.arange(21) + .5) * 2 - .5 + 2.5 * 42 / 518, atol=4e-6)
        np.testing.assert_allclose(y[:, 0], (np.arange(15) + .5) * 2 - .5 - 1.25 * 30 / 518, atol=4e-6)
        self.assertTrue(valid.all())

    def test_nonfinite_and_behind_camera_fields_are_not_support(self):
        coordinates = grid(8)
        coordinates[2, 3] = np.nan
        z_valid = np.ones((8, 8), dtype=bool)
        z_valid[4, 5] = False
        x, y, valid = native_remap(coordinates, (8, 8), (8, 8), z_valid)
        self.assertFalse(valid[2, 3])
        self.assertFalse(valid[4, 5])
        self.assertTrue(np.isfinite(x).all() and np.isfinite(y).all())

    def test_native_translation_samples_original_once_and_respects_alpha_and_black(self):
        source = np.zeros((8, 9, 3), dtype=np.uint8)
        source[..., 0] = np.arange(9)[None, :]
        reference = np.zeros_like(source)
        opaque = np.ones((8, 9), bool)
        source_opaque = opaque.copy()
        source_opaque[3, 4] = False
        # Reference x maps to original source x+1.
        coordinates = grid() + [518 / 9, 0]
        warped = dense_warp(reference, opaque, source, source_opaque, coordinates)
        np.testing.assert_array_equal(warped["source"][:, :8], source[:, 1:])
        self.assertFalse(warped["overlap"][3, 3])
        self.assertFalse(warped["overlap"][:, 8].any())
        self.assertTrue(warped["overlap"][0, 0])
        np.testing.assert_array_equal(warped["source"][0, 0], [1, 0, 0])

    def test_relative_camera_does_not_assume_identity_first_camera(self):
        reference = np.column_stack((np.eye(3), [10., 2., 0.]))
        source = np.column_stack((np.eye(3), [12., 5., 0.]))
        result = relative_camera_transform(reference, source)
        np.testing.assert_allclose(result, np.column_stack((np.eye(3), [2., 3., 0.])))
        with self.assertRaises(ValueError):
            relative_camera_transform(np.eye(4), source)


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required")
class GeometryBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        fixture = make_fixture(self.root / "fixture")
        build_manifest(fixture / "paths.json", self.root / "manifest", fixture / "reviewed.csv")
        prepare_dataset(self.root / "manifest/manifest.json", self.root / "prepared")
        self.manifest = self.root / "prepared/manifest.json"
        self.checkpoint = self.root / "model.pt"
        self.checkpoint.write_bytes(b"CPU fake checkpoint; never deserialized")
        self.source_patch = patch("facade_change.geoscd.source_provenance", return_value={"commit": GEOSCD_COMMIT, "source_sha256": {}})
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)
        self.addCleanup(self.temp.cleanup)

    def prediction(self):
        resolution = 518
        coordinates = grid(resolution)
        return {"reference_to_source": coordinates + [-64 * resolution / 512, 0],
                "source_to_reference": coordinates + [64 * resolution / 512, 0],
                "reference_occlusion": np.ones((resolution, resolution), bool),
                "source_occlusion": np.zeros((resolution, resolution), bool),
                "reference_projected_z": np.ones((resolution, resolution)),
                "source_projected_z": np.ones((resolution, resolution)),
                "camera_reference_to_source": np.column_stack((np.eye(3), [-64., 0., 0.])),
                "predicted_depth_confidence": np.ones((2, resolution, resolution)),
                "inference_seconds": .25, "peak_cuda_allocated_bytes": 0}

    def execute(self, out, backend, **kwargs):
        return run_geoscd(self.manifest, out, self.root / "external-geoscd", self.checkpoint,
                          _backend=backend, **kwargs)

    def test_native_rgb_masks_provenance_and_pending_review_without_occlusion_filtering(self):
        calls = []
        def backend(reference, source):
            calls.append((reference, source))
            return self.prediction()
        out = self.root / "geometry"
        summary = self.execute(out, backend)
        self.assertEqual(summary["computed_runs"], 1)
        self.assertEqual(summary["accepted_by_manual_review"], 0)
        self.assertEqual(summary["failed_runs"], 0)
        self.assertEqual(len(calls), 1)
        manifest = read_json(self.manifest)
        images = {str(row["image_id"]): row for row in manifest["images"]}
        self.assertEqual(calls[0], (images["0"]["image_path"], images["1"]["image_path"]))
        child = out / "geoscd/pair-0-1"
        with Image.open(child / "reference_rgb.png") as image:
            reference_rgb = np.asarray(image)
        with Image.open(images["0"]["image_path"]) as image:
            np.testing.assert_array_equal(reference_rgb, np.asarray(image))
        with Image.open(child / "overlap.png") as image:
            overlap = np.asarray(image) > 0
        with Image.open(child / "predicted_reference_occlusion.png") as image:
            self.assertTrue((np.asarray(image) == 255).all())
        self.assertGreater(int(overlap.sum()), 10000)  # Predicted all-occluded does not remove valid geometry.
        geometry = read_json(child / "geometry.json")
        self.assertEqual(geometry["type"], "dense_reference_grid")
        self.assertFalse(geometry["predicted_occlusion_used_as_support"])
        self.assertNotIn("source_to_reference", geometry)  # No fabricated H.
        self.assertEqual(geometry["originals"]["source"]["sha256"], images["1"]["sha256"])
        with np.load(child / "correspondences.npz", allow_pickle=False) as archive:
            self.assertIn("source_to_reference", archive.files)
            self.assertIn("predicted_depth_confidence", archive.files)
        self.assertTrue(np.isnan(np.load(child / "rgb_mean_absolute_difference.npy")[~overlap]).all())
        with (out / "manual_review.csv").open(encoding="utf-8") as handle:
            self.assertEqual(next(csv.DictReader(handle))["review_status"], "pending")
        row = read_json(out / "results.json")[0]
        self.assertEqual(row["reference_file"], images["0"]["file_name"])
        self.assertEqual(row["status"], "computed_needs_review")
        for name, digest in read_json(out / "run.json")["artifact_sha256"].items():
            self.assertEqual(sha256(out / name), digest)
        with self.assertRaises(FileExistsError):
            self.execute(out, backend)

    def test_failure_keeps_selected_denominator_and_diagnostic_child_record(self):
        def failed(*args):
            raise RuntimeError("Fake projection unavailable")
        out = self.root / "failed"
        summary = self.execute(out, failed)
        self.assertEqual(summary["eligible_pairs"], 1)
        self.assertEqual(summary["failed_runs"], 1)
        self.assertEqual(summary["computed_runs"], 0)
        self.assertEqual(read_json(out / "run.json")["status"], "completed_with_issues")
        self.assertEqual(read_json(out / "geoscd/pair-0-1/run.json")["status"], "failed")
        self.assertIn("Fake projection unavailable", (out / "comparison.html").read_text())

    def test_interrupted_pair_is_pending_not_a_completed_failure(self):
        def interrupted(*args):
            raise KeyboardInterrupt
        out = self.root / "interrupted"
        with self.assertRaises(KeyboardInterrupt):
            self.execute(out, interrupted)
        summary = read_json(out / "summary.json")
        self.assertEqual(summary["not_attempted"], 1)
        self.assertEqual(summary["failed_runs"], 0)
        self.assertEqual(read_json(out / "geoscd/pair-0-1/run.json")["status"], "interrupted")
        self.assertEqual(read_json(out / "run.json")["status"], "interrupted")

    def test_comparison_validates_manifest_and_links_existing_same_pair_outputs(self):
        previous = self.root / "previous"
        previous.mkdir()
        pair = {"pair_id": "0-1", "reference_id": "0", "source_id": "1", "split": "dev", "view_id": "demo"}
        actual = read_json(self.manifest)["images"][0]
        pair["view_id"] = actual["view_id"]
        write_json(previous / "selected_pairs.json", [pair])
        write_json(previous / "results.json", [{"pair_id": "0-1", "method": "loftr", "status": "passed", "path": "loftr/pair-0-1"}])
        write_json(previous / "run.json", {"status": "completed_needs_review",
                   "config": {"manifest_sha256": sha256(self.manifest)},
                   "artifact_sha256": {name: sha256(previous / name) for name in ("results.json", "selected_pairs.json")}})
        out = self.root / "comparison"
        summary = self.execute(out, lambda *args: self.prediction(), comparison_run=previous)
        self.assertEqual(summary["previous_routing_gates_on_same_selected_pairs"]["method_counts"]["loftr"]["passed"], 1)
        self.assertIn("../previous/loftr/pair-0-1/overlay_preview.jpg", (out / "comparison.html").read_text())
        write_json(previous / "run.json", {"status": "completed_needs_review", "config": {"manifest_sha256": "different"}})
        with self.assertRaisesRegex(ValueError, "different prepared manifest"):
            self.execute(self.root / "invalid-comparison", lambda *args: self.prediction(), comparison_run=previous)

    def test_changed_original_and_unsafe_resolution_are_explicit_failures(self):
        image = read_json(self.manifest)["images"][0]
        Path(image["image_path"]).write_bytes(b"Changed after manifest")
        called = []
        summary = self.execute(self.root / "changed", lambda *args: called.append(args))
        self.assertEqual(summary["failed_runs"], 1)
        self.assertEqual(called, [])
        with self.assertRaisesRegex(ValueError, "resolution=518"):
            self.execute(self.root / "wrong-grid", lambda *args: self.prediction(), resolution=512)
