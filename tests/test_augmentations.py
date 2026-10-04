"""Deterministic pixel and label contracts; no claim of realistic facade damage."""
import importlib.util
import unittest

import numpy as np

from facade_change.augmentations import apply_nuisance, render_state, validate_scenarios


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required")
class AugmentationTests(unittest.TestCase):
    def setUp(self):
        yy, xx = np.mgrid[:256, :256]
        self.rgb = np.stack((30 + (xx * 5 + yy * 2) % 210,
                             40 + (xx + yy * 7) % 200,
                             25 + (xx * 9 + yy * 3) % 220), axis=2).astype(np.uint8)
        self.support = np.ones((256, 256), bool)
        self.support[:8, :30] = False

    def test_identity_and_self_paste_keep_exact_bytes_and_zero_state_gt(self):
        for state in ("unchanged", "self_paste"):
            result = render_state(self.rgb, self.support, state)
            np.testing.assert_array_equal(result["rgb"], self.rgb)
            self.assertFalse(result["edit_mask"].any())
            self.assertFalse(np.shares_memory(result["rgb"], self.rgb))
        sham = render_state(self.rgb, self.support, "self_paste")
        paint = render_state(self.rgb, self.support, "paint_patch")
        np.testing.assert_array_equal(sham["insertion_boundary"], paint["insertion_boundary"])
        self.assertTrue(sham["insertion_boundary"].any())
        identity = apply_nuisance(self.rgb, self.support, {"id": "identity", "kind": "identity"})
        np.testing.assert_array_equal(identity["rgb"], self.rgb)
        np.testing.assert_array_equal(identity["support"], self.support)
        np.testing.assert_array_equal(identity["true_visibility"], self.support)
        self.assertFalse(identity["nuisance_mask"].any())

    def test_h1_is_local_actual_rgb_edit_and_preserves_unsupported_pixels(self):
        for state in ("crack", "paint_patch"):
            result = render_state(self.rgb, self.support, state)
            changed = np.any(result["rgb"] != self.rgb, axis=2)
            np.testing.assert_array_equal(result["edit_mask"], changed)
            self.assertGreater(int(changed.sum()), 100)
            self.assertLess(int(changed.sum()), self.support.size // 8)
            self.assertFalse(changed[~self.support].any())
            np.testing.assert_array_equal(result["rgb"][~changed], self.rgb[~changed])
            self.assertFalse(result["insertion_boundary"][~self.support].any())
        crack = render_state(self.rgb, self.support, "crack")
        self.assertTrue((crack["rgb"] <= self.rgb).all())
        self.assertTrue(np.any(crack["rgb"] < self.rgb))

    def test_invisible_state_edits_are_rejected(self):
        black = np.zeros_like(self.rgb)
        with self.assertRaisesRegex(ValueError, "no visible supported RGB edit"):
            render_state(black, self.support, "crack")
        support = np.zeros_like(self.support)
        support[:10, :10] = True  # The fixed H1 region is absent from valid support.
        with self.assertRaisesRegex(ValueError, "no visible supported RGB edit"):
            render_state(self.rgb, support, "paint_patch")

    def test_shadow_strength_keeps_template_footprint_and_monotonic_attenuation(self):
        for template in ("diagonal", "band"):
            rendered = [apply_nuisance(self.rgb, self.support,
                                       {"id": f"shadow_{template}_{index}", "kind": "shadow", "template": template,
                                        "strength": strength, "edge_width": .035})
                        for index, strength in enumerate((.15, .35, .55))]
            for before, after in zip(rendered, rendered[1:]):
                np.testing.assert_array_equal(before["nuisance_mask"], after["nuisance_mask"])
                np.testing.assert_array_equal(before["alpha"], after["alpha"])
                self.assertTrue((after["rgb"] <= before["rgb"]).all())
                self.assertTrue((after["intensity"] >= before["intensity"]).all())
            for result in rendered:
                np.testing.assert_array_equal(result["support"], self.support)
                np.testing.assert_array_equal(result["true_visibility"], self.support)
                self.assertFalse(result["nuisance_mask"][~self.support].any())
                np.testing.assert_array_equal(result["rgb"][~result["nuisance_mask"]], self.rgb[~result["nuisance_mask"]])
            self.assertTrue(np.any((rendered[0]["alpha"] > 0) & (rendered[0]["alpha"] < 1)))

    def test_exposure_is_explicit_linear_srgb_and_white_balance_is_channel_specific(self):
        gray = np.full_like(self.rgb, 128)
        exposed = apply_nuisance(gray, self.support, {"id": "exposure", "kind": "exposure", "gain": .75})
        np.testing.assert_array_equal(exposed["rgb"][self.support], np.full((self.support.sum(), 3), 112, dtype=np.uint8))
        self.assertEqual(exposed["parameters"]["color_space"], "linear-light sRGB")
        balance = apply_nuisance(gray, self.support, {"id": "white_balance", "kind": "white_balance", "gains": [1.12, 1., .88]})
        self.assertTrue((balance["rgb"][self.support, 0] > 128).all())
        self.assertTrue((balance["rgb"][self.support, 1] == 128).all())
        self.assertTrue((balance["rgb"][self.support, 2] < 128).all())
        for result in (exposed, balance):
            np.testing.assert_array_equal(result["rgb"][~self.support], gray[~self.support])

    def test_known_occlusion_hides_only_its_region_and_part_of_h1(self):
        state = render_state(self.rgb, self.support, "paint_patch")
        result = apply_nuisance(state["rgb"], self.support, {"id": "occlusion", "kind": "occlusion",
                                                             "rectangle": [.48, .25, .68, .8], "color": [62, 68, 58]})
        region = result["nuisance_mask"]
        self.assertTrue(region.any())
        np.testing.assert_array_equal(result["rgb"][region], np.broadcast_to([62, 68, 58], (region.sum(), 3)))
        np.testing.assert_array_equal(result["support"], self.support & ~region)
        np.testing.assert_array_equal(result["true_visibility"], self.support & ~region)
        self.assertTrue((state["edit_mask"] & region).any())
        self.assertTrue((state["edit_mask"] & result["true_visibility"]).any())
        np.testing.assert_array_equal(result["rgb"][~region], state["rgb"][~region])

    def test_blur_and_jpeg_do_not_bleed_invalid_black_pixels_into_supported_color(self):
        uniform = np.full_like(self.rgb, 100)
        support = np.ones_like(self.support)
        support[:, :30] = False
        with_hole = uniform.copy()
        with_hole[~support] = 0
        for kind, parameters in (("blur", {"sigma": 1.6}), ("jpeg", {"quality": 40})):
            spec = {"id": kind, "kind": kind, **parameters}
            full = apply_nuisance(uniform, np.ones_like(support), spec)
            partial = apply_nuisance(with_hole, support, spec)
            np.testing.assert_array_equal(partial["rgb"][support], full["rgb"][support])
            np.testing.assert_array_equal(partial["rgb"][~support], with_hole[~support])
            np.testing.assert_array_equal(partial["support"], support)
            self.assertIn("nearest supported", partial["parameters"]["invalid_pixel_handling"])

    def test_jpeg_is_actual_codec_roundtrip_and_deterministic(self):
        spec = {"id": "jpeg_40", "kind": "jpeg", "quality": 40}
        first = apply_nuisance(self.rgb, self.support, spec)
        second = apply_nuisance(self.rgb, self.support, spec)
        self.assertGreater(int(first["actual_change_mask"].sum()), self.support.sum() // 2)
        self.assertGreater(first["parameters"]["encoded_jpeg_bytes"], 100)
        self.assertEqual(len(first["parameters"]["encoded_jpeg_sha256"]), 64)
        self.assertEqual(first["parameters"], second["parameters"])
        np.testing.assert_array_equal(first["rgb"], second["rgb"])

    def test_every_variant_is_fresh_deterministic_uint8_on_the_same_grid(self):
        scenarios = validate_scenarios([{"id": kind, "kind": kind} for kind in
                                       ("identity", "shadow", "exposure", "contrast", "white_balance", "blur", "jpeg", "occlusion")])
        rgb_before, support_before = self.rgb.copy(), self.support.copy()
        for state in ("unchanged", "crack", "paint_patch", "self_paste"):
            one = render_state(self.rgb, self.support, state)
            two = render_state(self.rgb, self.support, state)
            np.testing.assert_array_equal(one["rgb"], two["rgb"])
            np.testing.assert_array_equal(one["edit_mask"], two["edit_mask"])
            for spec in scenarios:
                first, second = apply_nuisance(one["rgb"], self.support, spec), apply_nuisance(one["rgb"], self.support, spec)
                np.testing.assert_array_equal(first["rgb"], second["rgb"])
                self.assertEqual(first["parameters"], second["parameters"])
                self.assertEqual(first["rgb"].dtype, np.uint8)
                self.assertEqual(first["rgb"].shape, self.rgb.shape)
                if spec["kind"] != "occlusion":
                    np.testing.assert_array_equal(first["support"], self.support)
                    np.testing.assert_array_equal(first["true_visibility"], self.support)
        np.testing.assert_array_equal(self.rgb, rgb_before)
        np.testing.assert_array_equal(self.support, support_before)

    def test_unknown_and_invalid_scenarios_fail_explicitly(self):
        invalid = [{"id": "../unsafe", "kind": "identity"}, {"id": "wrong", "kind": "noise"},
                   {"id": "wrong", "kind": "shadow", "strength": float("nan")},
                   {"id": "wrong", "kind": "jpeg", "quality": 70.5},
                   {"id": "wrong", "kind": "exposure", "gain": True},
                   {"id": "wrong", "kind": "white_balance", "gains": [1, 1]},
                   {"id": "wrong", "kind": "occlusion", "rectangle": [.8, .2, .7, .9]},
                   {"id": "wrong", "kind": "identity", "hidden_gamma": 2}]
        for spec in invalid:
            with self.assertRaises(ValueError):
                validate_scenarios([spec])
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_scenarios([{"id": "duplicate", "kind": "identity"}] * 2)
        with self.assertRaises(ValueError):
            apply_nuisance(self.rgb.astype(np.float32), self.support, {"id": "identity", "kind": "identity"})


if __name__ == "__main__":
    unittest.main()
