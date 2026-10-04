"""CPU full-adapter contracts; no pretrained VGGT/SAM inference is claimed."""
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from facade_change.data import build_manifest
from facade_change.demo import make_fixture
from facade_change.geoscd import GEOSCD_COMMIT
from facade_change.geoscd_full import merge_directional_masks, native_camera_warp, run_geoscd_full, skip_unused_sam_attention, validate_sam_vit_h_state
from facade_change.io import read_json, sha256
from facade_change.preparation import prepare_dataset


def window_partition(x, window_size):
    b, h, w, c = x.shape
    ph, pw = (-h) % window_size, (-w) % window_size
    x = np.pad(x, ((0, 0), (0, ph), (0, pw), (0, 0)))
    hp, wp = h + ph, w + pw
    windows = x.reshape(b, hp // window_size, window_size, wp // window_size, window_size, c)
    return windows.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, c), (hp, wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    hp, wp = pad_hw
    h, w = hw
    b = windows.shape[0] // (hp * wp // window_size // window_size)
    x = windows.reshape(b, hp // window_size, wp // window_size, window_size, window_size, -1)
    return x.transpose(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, -1)[:, :h, :w].copy()


class ParityBlock:
    """Pinned Block.forward body with deterministic CPU-array layer seams."""
    def __init__(self, window_size):
        self.window_size = window_size
        self.calls = []
        self.norm1 = lambda x: x * 2 + 1
        self.norm2 = lambda x: x * .5 - 2
        self.mlp = lambda x: x * .25 + 3

    def attn(self, x, return_qkv=False):
        self.calls.append((x.shape, return_qkv))
        return np.stack((x, x * 2, x * 3)) if return_qkv else x * 4 + .5

    def forward(self, x, return_qkv=False):
        qkv = self.attn(self.norm1(x), return_qkv)
        if return_qkv:
            return qkv
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class FullContractTests(unittest.TestCase):
    def test_sam1_vit_h_architecture_is_required_not_sam3_or_smaller_sam(self):
        state = {"image_encoder.patch_embed.proj.weight": SimpleNamespace(shape=(1280, 3, 16, 16)),
                 "image_encoder.blocks.31.attn.qkv.weight": SimpleNamespace(shape=(3840, 1280))}
        validate_sam_vit_h_state(state)
        for wrong in ({}, {"model": state}, {"image_encoder.patch_embed.proj.weight": SimpleNamespace(shape=(768, 3, 16, 16))}):
            with self.assertRaisesRegex(ValueError, "SAM 1 ViT-H"):
                validate_sam_vit_h_state(wrong)

    def test_official_directional_fusion_gathers_source_mask_and_keeps_reference_union(self):
        yy, xx = np.mgrid[:4, :5]
        coordinates = np.stack((xx + 1, yy), axis=2).astype(float)
        reference = np.zeros((4, 5), bool)
        source = np.zeros_like(reference)
        source[2, 3] = True
        reference[0, 4] = True
        coordinates[1, 0] = np.nan
        final = merge_directional_masks(reference, source, coordinates)
        expected = reference.copy()
        expected[2, 2] = True
        np.testing.assert_array_equal(final, expected)

    def test_unused_attention_removal_preserves_normal_and_qkv_paths_with_padding(self):
        optimized = skip_unused_sam_attention(ParityBlock.forward)
        x = np.arange(1 * 7 * 5 * 3, dtype=float).reshape(1, 7, 5, 3)
        for window in (0, 2, 4):
            original_block, optimized_block = ParityBlock(window), ParityBlock(window)
            expected = original_block.forward(x)
            result = optimized(optimized_block, x)
            np.testing.assert_array_equal(result, expected)
            self.assertEqual(len(original_block.calls), 2)
            self.assertEqual(len(optimized_block.calls), 1)
            np.testing.assert_array_equal(optimized(optimized_block, x, True), original_block.forward(x, True))


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required")
class FullNativeTests(unittest.TestCase):
    def test_identity_native_grid_and_black_support_are_preserved(self):
        yy, xx = np.mgrid[:512, :512]
        grid = np.stack((xx, yy), axis=2).astype(np.float32)
        image = np.zeros((7, 13, 3), np.uint8)
        image[..., 0] = np.arange(13)
        opaque = np.ones((7, 13), bool)
        warped, support, overlap = native_camera_warp(image, opaque, image, opaque, grid, np.ones((512, 512)))
        np.testing.assert_array_equal(warped, image)
        self.assertTrue(support.all() and overlap.all())

    def test_zero_origin_camera_grid_lift_and_positive_z_are_explicit(self):
        yy, xx = np.mgrid[:512, :512]
        grid = np.stack((xx, yy), axis=2).astype(np.float32)
        reference = np.zeros((8, 9, 3), np.uint8)
        source = np.zeros((8, 18, 3), np.uint8)
        source[..., 0] = np.arange(18)
        grid += [512 / 18, 0]  # One native source pixel shift.
        z = np.ones((512, 512))
        warped, support, overlap = native_camera_warp(reference, np.ones((8, 9), bool), source, np.ones((8, 18), bool), grid, z)
        # Affine native baseline x -> (x+.5)*2-.5, plus exactly one source pixel.
        self.assertEqual(int(warped[2, 2, 0]), 6)
        self.assertTrue(overlap[2, 2])
        z[:] = -1
        _, _, overlap = native_camera_warp(reference, np.ones((8, 9), bool), source, np.ones((8, 18), bool), grid, z)
        self.assertFalse(overlap.any())


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required")
class FullBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        fixture = make_fixture(self.root / "fixture")
        build_manifest(fixture / "paths.json", self.root / "manifest", fixture / "reviewed.csv")
        prepare_dataset(self.root / "manifest/manifest.json", self.root / "prepared")
        self.manifest = self.root / "prepared/manifest.json"
        self.checkpoint = self.root / "vggt.pt"
        self.sam = self.root / "sam_vit_h_4b8939.pth"
        self.checkpoint.write_bytes(b"Fake VGGT; never loaded")
        self.sam.write_bytes(b"Fake SAM; never loaded")
        patched = patch("facade_change.geoscd_full.source_provenance", return_value={"commit": GEOSCD_COMMIT})
        patched.start()
        self.addCleanup(patched.stop)
        self.addCleanup(self.temp.cleanup)

    def prediction(self):
        yy, xx = np.mgrid[:512, :512]
        grid = np.stack((xx, yy), axis=2).astype(np.float32)
        mask = np.zeros((512, 512), bool)
        mask[200:240, 220:250] = True
        left = {"mask": mask, "score": (xx / 256).astype(np.float32), "feature_valid": np.ones((512, 512), bool)}
        right = {"mask": np.zeros_like(mask), "score": (yy / 256).astype(np.float32), "feature_valid": np.ones_like(mask)}
        geometry = lambda delta: {"coordinates": grid + [delta, 0], "projected_z": np.ones((512, 512)),
                                  "occlusion": np.ones((512, 512), bool), "camera_extrinsic": np.column_stack((np.eye(3), [delta, 0, 0])),
                                  "scattered_depth": np.ones((512, 512))}
        return {"reference_geometry": geometry(-64), "source_geometry": geometry(64), "reference_detection": left,
                "source_detection": right, "final_reference_mask": mask, "geometry_seconds": .5, "sam_seconds": 1.,
                "inference_seconds": 1.5, "peak_cuda_allocated_bytes": 0}

    def execute(self, out, backend, **kwargs):
        return run_geoscd_full(self.manifest, out, self.root / "external", self.checkpoint, self.sam, _backend=backend, **kwargs)

    def test_full_exports_native_binary_scores_originals_and_provenance_without_occ_support_filter(self):
        calls = []
        def backend(*args):
            calls.append(args)
            return self.prediction()
        out = self.root / "full"
        summary = self.execute(out, backend)
        self.assertEqual(summary["computed_runs"], 1)
        self.assertFalse(summary["evaluation_ready"])
        self.assertEqual(len(calls), 1)
        child = out / "geoscd-full/pair-0-1"
        images = read_json(self.manifest)["images"]
        with Image.open(child / "reference_rgb.png") as saved, Image.open(images[0]["image_path"]) as original:
            np.testing.assert_array_equal(np.asarray(saved), np.asarray(original))
        with Image.open(child / "final_change_mask_reference_native.png") as mask:
            self.assertEqual(mask.size, (images[0]["width"], images[0]["height"]))
            self.assertGreater(int(np.asarray(mask).sum()), 0)
        with Image.open(child / "predicted_reference_occlusion.png") as occ:
            self.assertTrue((np.asarray(occ) == 255).all())
        with Image.open(child / "overlap.png") as overlap:
            comparable = np.asarray(overlap) > 0
        self.assertGreater(int(comparable.sum()), 10000)
        score = np.load(child / "reference_sam_key_change_score_native.npy", allow_pickle=False)
        self.assertTrue(np.isnan(score[~comparable]).all())
        self.assertTrue(np.isfinite(score[comparable]).any())
        with np.load(child / "full_predictions_model_grid.npz", allow_pickle=False) as archive:
            self.assertEqual(archive["final_reference_mask"].shape, (512, 512))
            self.assertIn("source_score", archive.files)
        contract = read_json(child / "prediction_contract.json")
        self.assertIn("not probability", contract["scores"])
        self.assertEqual(contract["originals"]["source"]["sha256"], images[1]["sha256"])
        row = read_json(out / "results.json")[0]
        self.assertEqual(row["reference_file"], images[0]["file_name"])
        self.assertEqual(row["status"], "computed_needs_review")
        with (out / "manual_review.csv").open(encoding="utf-8") as handle:
            self.assertEqual(next(csv.DictReader(handle))["review_status"], "pending")
        record = read_json(out / "run.json")
        self.assertEqual(record["config"]["sam_checkpoint_sha256"], sha256(self.sam))
        for path, digest in record["artifact_sha256"].items():
            self.assertEqual(sha256(out / path), digest)
        with self.assertRaises(FileExistsError):
            self.execute(out, backend)

    def test_real_sam3_filename_is_rejected_without_loading_any_backend(self):
        sam3 = self.root / "sam3.pt"
        sam3.write_bytes(b"wrong architecture")
        with self.assertRaisesRegex(ValueError, "sam3.pt is incompatible"):
            run_geoscd_full(self.manifest, self.root / "wrong", self.root / "external", self.checkpoint, sam3,
                            _backend=lambda *args: self.fail("Must not load models"))
        self.assertFalse((self.root / "wrong").exists())

    def test_failed_pair_and_interruption_keep_denominator_and_status(self):
        def failure(*args):
            raise RuntimeError("SAM stage unavailable")
        out = self.root / "failed"
        summary = self.execute(out, failure)
        self.assertEqual(summary["failed_runs"], 1)
        self.assertEqual(summary["eligible_pairs"], 1)
        self.assertEqual(read_json(out / "run.json")["status"], "completed_with_issues")
        self.assertIn("SAM stage unavailable", (out / "comparison.html").read_text())
        def interrupted(*args):
            raise KeyboardInterrupt
        out = self.root / "interrupted"
        with self.assertRaises(KeyboardInterrupt):
            self.execute(out, interrupted)
        self.assertEqual(read_json(out / "summary.json")["not_attempted"], 1)
        self.assertEqual(read_json(out / "geoscd-full/pair-0-1/run.json")["status"], "interrupted")

    def test_changed_original_fails_before_model_inference(self):
        image = read_json(self.manifest)["images"][0]
        Path(image["image_path"]).write_bytes(b"changed")
        summary = self.execute(self.root / "changed", lambda *args: self.fail("Do not infer changed input"))
        self.assertEqual(summary["failed_runs"], 1)


if __name__ == "__main__":
    unittest.main()
