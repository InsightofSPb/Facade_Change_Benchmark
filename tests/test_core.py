import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from facade_change.alignment import Matches, align, ratio_matches, valid_locations
from facade_change.data import ImageResolver, build_manifest, read_overrides
from facade_change.demo import make_fixture, run_demo
from facade_change.geometry import absolute_residual, expanded_canvas, proxy_transform, transform_points, warp_pair
from facade_change.io import load_rgb, read_json, sha256, write_json
from facade_change.pipeline import run_pair, select_pair

HAS_CV2 = importlib.util.find_spec("cv2") is not None


class GeometryTests(unittest.TestCase):
    def test_rounded_proxy_centres_and_direction(self):
        native, proxy = (3024, 4033), (768, 1024)
        p = proxy_transform(native, proxy)
        points = np.array([[0., 0.], [3017., 2911.], [4032., 3023.]])
        expected = (points + .5) * np.array([1024 / 4033, 768 / 3024]) - .5
        np.testing.assert_allclose(transform_points(points, p), expected)
        np.testing.assert_allclose(transform_points(expected, np.linalg.inv(p)), points, atol=1e-10)

    def test_union_canvas_keeps_native_scale(self):
        identity = expanded_canvas((12, 16), (12, 16), np.eye(3))
        self.assertEqual((identity.width, identity.height), (16, 12))
        h = np.array([[1., 0, -5], [0, 1, 3], [0, 0, 1]])
        union = expanded_canvas((12, 16), (12, 16), h)
        self.assertEqual((union.width, union.height), (21, 15))
        np.testing.assert_array_equal(union.reference_to_canvas, [[1, 0, 5], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(transform_points([[0, 0]], union.reference_to_canvas @ h), [[0, 3]])

    def test_unsafe_homographies_rejected_before_allocation(self):
        for h in (np.zeros((3, 3)), np.full((3, 3), np.nan),
                  [[1, 0, 0], [0, 1, 0], [.2, 0, -1]],
                  [[1, 0, 1e8], [0, 1, 0], [0, 0, 1]]):
            with self.subTest(h=h), self.assertRaises(ValueError):
                expanded_canvas((12, 16), (12, 16), h)

    def test_abs_residual_does_not_wrap(self):
        a = np.array([[[0, 255, 3]]], np.uint8)
        b = np.array([[[255, 0, 250]]], np.uint8)
        np.testing.assert_array_equal(absolute_residual(a, b), [[[255, 255, 247]]])

    def test_knn_short_rows(self):
        m = SimpleNamespace(distance=1.)
        n = SimpleNamespace(distance=2.)
        self.assertEqual(ratio_matches([[], [m], [m, n]]), [m])

    def test_padding_nonfinite_and_transparency_are_invalid(self):
        support = np.ones((8, 8), bool)
        support[3, 2] = False
        points = np.array([[0, 0], [7, 7], [8, 4], [-1, 2], [2, 3], [np.nan, 0]])
        np.testing.assert_array_equal(valid_locations(points, support), [True, True, False, False, False, False])

    @unittest.skipUnless(HAS_CV2, "OpenCV required for warping")
    def test_black_pixels_valid_alpha_invalid_new_area_preserved(self):
        rgb = np.zeros((12, 16, 3), np.uint8)
        ref_opaque = np.ones((12, 16), bool)
        src_opaque = ref_opaque.copy()
        src_opaque[2, 2] = False
        canvas = expanded_canvas(rgb.shape, rgb.shape, [[1, 0, 5], [0, 1, 0], [0, 0, 1]])
        result = warp_pair(rgb, ref_opaque, rgb, src_opaque, canvas)
        self.assertTrue(result["overlap"][0, 5])
        self.assertFalse(result["overlap"][2, 7])
        self.assertEqual(int(result["source_only"].sum()), 12 * 5)
        self.assertEqual(result["source"].shape, (12, 21, 3))

    @unittest.skipUnless(HAS_CV2, "OpenCV required for estimator")
    def test_alternate_matcher_uses_shared_native_estimator(self):
        # A matcher seam test on unequal native/proxy sizes, independent of SIFT.
        source_shape, reference_shape = (333, 511), (399, 617)
        h = np.array([[1.05, .01, 10], [-.01, 1.03, 20], [0, 0, 1.]])
        xs, ys = np.meshgrid([20, 130, 300, 480], [20, 100, 200, 310])
        native_source = np.column_stack([xs.ravel(), ys.ravel()])
        native_reference = transform_points(native_source, h)
        class FixedMatcher:
            def match(self, source, reference, source_support, reference_support):
                sp = proxy_transform(source_shape, source.shape)
                rp = proxy_transform(reference_shape, reference.shape)
                return Matches(transform_points(native_source, sp), transform_points(native_reference, rp), {"id": "fixture"})
        result = align(np.zeros((*reference_shape, 3), np.uint8), np.ones(reference_shape, bool),
                       np.zeros((*source_shape, 3), np.uint8), np.ones(source_shape, bool), FixedMatcher(), max_side=256)
        np.testing.assert_allclose(transform_points(native_source, result.source_to_reference), native_reference, atol=1e-3)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def save(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((8, 10, 3), np.uint8)).save(path)
        return path

    def test_exact_prefix_and_ambiguity(self):
        a = self.save("one/12345678-facade_2010.png")
        resolver = ImageResolver([self.root])
        self.assertEqual(resolver.resolve({"file_name": a.name})[0], a)
        self.assertEqual(resolver.resolve({"file_name": "abcdabcd-facade_2010.png"})[0], a)
        self.save("two/87654321-facade_2010.png")
        path, status, candidates = ImageResolver([self.root]).resolve({"file_name": "facade_2010.png"})
        self.assertIsNone(path)
        self.assertEqual(status, "ambiguous_stripped_prefix")
        self.assertEqual(len(candidates), 2)

    def test_explicit_path_and_labelstudio_namespace(self):
        a = self.save("abc00000-photo.png")
        resolver = ImageResolver([self.root])
        self.assertEqual(resolver.resolve({"path": str(a), "file_name": "unrelated.png"})[0], a)
        self.assertEqual(resolver.resolve({"path": "/data/upload/1/" + a.name, "file_name": a.name})[0], a)

    def test_rgb_channel_order_and_alpha(self):
        rgba = np.array([[[255, 0, 9, 255], [0, 0, 0, 254]]], np.uint8)
        path = self.root / "rgba.png"
        Image.fromarray(rgba).save(path)
        rgb, opaque = load_rgb(path)
        np.testing.assert_array_equal(rgb, rgba[..., :3])
        np.testing.assert_array_equal(opaque, [[True, False]])

    def test_manifest_keeps_missing_unknown_and_bad_dimensions(self):
        self.save("normal.png")
        self.save("bbbbbbbb-view_2020.png")
        write_json(self.root / "coco.json", {"images": [
            {"id": 0, "file_name": "normal.png", "width": 10, "height": 8},
            {"id": 1, "file_name": "bbbbbbbb-view_2020.png", "width": 999, "height": 8},
            {"id": 2, "file_name": "absent.png", "width": 10, "height": 8}], "annotations": [], "categories": []})
        write_json(self.root / "paths.json", {"coco_json": "coco.json", "image_roots": ["."]})
        result = build_manifest(self.root / "paths.json", self.root / "manifest")
        self.assertEqual(len(result["images"]), 3)
        self.assertEqual([r["image_status"] for r in result["images"]], ["ready", "dimension_mismatch", "missing"])
        self.assertEqual(result["images"][0]["metadata_status"], "unknown")
        self.assertEqual(read_json(self.root / "manifest/run.json")["status"], "completed_with_issues")
        with self.assertRaises(FileExistsError):
            build_manifest(self.root / "paths.json", self.root / "manifest")

    def test_duplicate_overrides_rejected(self):
        path = self.root / "override.csv"
        path.write_text("image_id,reviewed\n1,false\n1,false\n")
        with self.assertRaises(ValueError):
            read_overrides(path, {"1"})

    def test_pair_requires_review_same_view_and_forward_time(self):
        fixture = make_fixture(self.root / "fixture")
        manifest = build_manifest(fixture / "paths.json", self.root / "manifest")
        with self.assertRaises(ValueError):
            select_pair(manifest, 0, 1)
        self.assertEqual(select_pair(manifest, 0, 1, True)[1]["year"], 2020)
        with self.assertRaises(ValueError):
            select_pair(manifest, 1, 0, True)
        manifest["images"][1]["view_id"] = "other"
        with self.assertRaises(ValueError):
            select_pair(manifest, 0, 1, True)

    def test_changed_input_records_failure(self):
        fixture = make_fixture(self.root / "fixture")
        manifest_dir = self.root / "manifest"
        result = build_manifest(fixture / "paths.json", manifest_dir, fixture / "reviewed.csv")
        image_path = Path(result["images"][0]["image_path"])
        image_path.write_bytes(image_path.read_bytes() + b"changed")
        with self.assertRaises(ValueError):
            run_pair(manifest_dir / "manifest.json", 0, 1, self.root / "pair")
        record = read_json(self.root / "pair/run.json")
        self.assertEqual(record["status"], "failed")
        self.assertIn("changed since manifest", record["error"])

    @unittest.skipUnless(HAS_CV2, "OpenCV required for synthetic integration run")
    def test_end_to_end_sift_known_geometry_and_provenance(self):
        result = run_demo(self.root / "demo")
        self.assertTrue(result["passed"])
        record = read_json(self.root / "demo/sift/run.json")
        self.assertEqual(record["status"], "completed_needs_review")
        for name, digest in record["artifact_sha256"].items():
            self.assertEqual(sha256(self.root / "demo/sift" / name), digest)
        self.assertTrue((self.root / "demo/sift/gallery.html").is_file())


if __name__ == "__main__":
    unittest.main()
