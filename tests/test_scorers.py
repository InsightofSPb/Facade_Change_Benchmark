"""Numerical scorer contracts, independent of procedural GT and thresholds."""
import importlib.util
import inspect
import unittest

import numpy as np

from facade_change.scorers import score_change, scorer_metadata


class RGBDifferenceTests(unittest.TestCase):
    def test_rgb_scale_is_channel_mean_without_uint8_wraparound(self):
        reference = np.zeros((3, 4, 3), np.uint8)
        source = reference.copy()
        source[0, 0] = [255, 255, 255]
        source[0, 1] = [255, 0, 0]
        source[0, 2] = [25, 50, 75]
        support = np.ones((3, 4), bool)
        score = score_change(reference, source, support, "rgb_diff")
        self.assertEqual(score.dtype, np.float32)
        self.assertEqual(float(score[0, 0]), 1)
        self.assertAlmostEqual(float(score[0, 1]), 1 / 3, places=7)
        self.assertAlmostEqual(float(score[0, 2]), 50 / 255, places=7)
        np.testing.assert_array_equal(score[1:], 0)
        np.testing.assert_array_equal(score_change(source, reference, support, "rgb_diff"), score)

    def test_identity_supported_black_and_nan_outside_support(self):
        reference = np.zeros((7, 9, 3), np.uint8)
        support = np.ones((7, 9), bool)
        support[:2, :3] = False
        score = score_change(reference, reference.copy(), support, "rgb_diff")
        self.assertTrue(np.isfinite(score[support]).all())
        self.assertTrue((score[support] == 0).all())
        self.assertTrue(np.isnan(score[~support]).all())

    def test_scorer_inputs_exclude_oracle_labels_and_are_never_mutated(self):
        self.assertEqual(list(inspect.signature(score_change).parameters),
                         ["reference_rgb", "source_rgb", "geometric_support", "method"])
        reference = np.zeros((4, 6, 3), np.uint8)
        source = np.full_like(reference, 80)
        support = np.ones((4, 6), bool)
        reference.flags.writeable = source.flags.writeable = support.flags.writeable = False
        score_change(reference, source, support, "rgb_diff")
        np.testing.assert_array_equal(reference, 0)
        np.testing.assert_array_equal(source, 80)
        self.assertTrue(support.all())

    def test_validation_rejects_unknown_method_and_wrong_signal_or_support(self):
        reference = np.zeros((3, 5, 3), np.uint8)
        support = np.ones((3, 5), bool)
        for method in ("unknown", [], None):
            with self.assertRaises(ValueError):
                score_change(reference, reference, support, method)
        for source, mask in ((reference.astype(np.float32), support), (reference[:, :4], support),
                             (reference, support.astype(np.uint8)), (reference, np.zeros_like(support)),
                             (reference, support[:, :4])):
            with self.assertRaises(ValueError):
                score_change(reference, source, mask, "rgb_diff")

    def test_metadata_is_serializable_fresh_and_declares_support_adaptation(self):
        import json
        first = scorer_metadata("ssim")
        self.assertEqual(first["constants"]["C1"], .01 ** 2)
        self.assertEqual(first["gaussian"]["window_size"], [11, 11])
        self.assertIn("population", first["moments"])
        self.assertIn("geometric support", first["support"])
        self.assertIn("ssim.pdf", first["primary_source"])
        json.dumps(first, allow_nan=False)
        first["gaussian"]["window_size"][0] = 1
        self.assertEqual(scorer_metadata("ssim")["gaussian"]["window_size"], [11, 11])


def hand_ssim(reference, source, support, cy, cx):
    """Direct centered weighted sums, not the scorer's E[x²]−E[x]² filters."""
    weights, x, y = [], [], []
    h, w = support.shape
    for dy in range(-5, 6):
        for dx in range(-5, 6):
            row, col = cy + dy, cx + dx
            if 0 <= row < h and 0 <= col < w and support[row, col]:
                weights.append(np.exp(-(dx * dx + dy * dy) / (2 * 1.5 ** 2)))
                x.append(reference[row, col].astype(float) / 255)
                y.append(source[row, col].astype(float) / 255)
    weights = np.asarray(weights)
    weights /= weights.sum()
    x, y = np.asarray(x), np.asarray(y)
    mx, my = (weights[:, None] * x).sum(axis=0), (weights[:, None] * y).sum(axis=0)
    vx = (weights[:, None] * (x - mx) ** 2).sum(axis=0)
    vy = (weights[:, None] * (y - my) ** 2).sum(axis=0)
    covariance = (weights[:, None] * (x - mx) * (y - my)).sum(axis=0)
    similarity = ((2 * mx * my + .01 ** 2) * (2 * covariance + .03 ** 2)
                  / ((mx * mx + my * my + .01 ** 2) * (vx + vy + .03 ** 2)))
    return float(np.clip((1 - similarity.mean()) / 2, 0, 1))


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required for Gaussian SSIM")
class SSIMTests(unittest.TestCase):
    def setUp(self):
        yy, xx = np.mgrid[:29, :35]
        self.reference = np.stack(((xx * 5 + yy * 9) % 256, (xx * 13 + yy * 3) % 256,
                                   (xx * 2 + yy * 7) % 256), axis=2).astype(np.uint8)
        self.source = self.reference.copy()
        self.source[9:19, 10:25] = [172, 164, 151]
        self.support = np.ones((29, 35), bool)
        self.support[10:12, 12:14] = False

    def test_ssim_parity_with_direct_weighted_population_moments(self):
        score = score_change(self.reference, self.source, self.support, "ssim")
        for cy, cx in ((14, 18), (12, 16), (8, 11), (0, 0), (28, 34)):
            self.assertAlmostEqual(float(score[cy, cx]), hand_ssim(self.reference, self.source, self.support, cy, cx), places=7)
        self.assertEqual(score.dtype, np.float32)

    def test_constant_patches_match_luminance_formula_per_rgb_channel(self):
        x, y = np.array([80, 100, 180], np.uint8), np.array([120, 120, 60], np.uint8)
        reference = np.broadcast_to(x, (18, 23, 3)).copy()
        source = np.broadcast_to(y, reference.shape).copy()
        support = np.ones((18, 23), bool)
        support[:4, :7] = False
        xv, yv = x.astype(float) / 255, y.astype(float) / 255
        per_channel = (2 * xv * yv + .01 ** 2) / (xv * xv + yv * yv + .01 ** 2)
        expected = (1 - per_channel.mean()) / 2
        score = score_change(reference, source, support, "ssim")
        np.testing.assert_allclose(score[support], expected, rtol=0, atol=1e-7)

    def test_invalid_colors_cannot_spill_into_supported_scores(self):
        first = score_change(self.reference, self.source, self.support, "ssim")
        reference, source = self.reference.copy(), self.source.copy()
        reference[~self.support] = [255, 0, 180]
        source[~self.support] = [0, 255, 255]
        second = score_change(reference, source, self.support, "ssim")
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isnan(second[~self.support]).all())

    def test_identity_is_exact_even_when_invalid_colors_differ(self):
        source = self.reference.copy()
        source[~self.support] = 255 - source[~self.support]
        score = score_change(self.reference, source, self.support, "ssim")
        np.testing.assert_array_equal(score[self.support], 0)
        self.assertTrue(np.isnan(score[~self.support]).all())

    def test_local_edit_has_a_finite_five_pixel_ssim_neighborhood(self):
        reference = np.full((40, 50, 3), 80, np.uint8)
        source = reference.copy()
        source[17:22, 25:30] = 180
        score = score_change(reference, source, np.ones(reference.shape[:2], bool), "ssim")
        self.assertGreater(float(score[19, 27]), .01)
        self.assertGreater(float(score[16, 27]), 0)  # Context reacts outside the planted RGB pixels.
        outside = np.ones(score.shape, bool)
        outside[12:27, 20:35] = False
        np.testing.assert_array_equal(score[outside], 0)

    def test_single_supported_pixel_is_finite_and_black_is_valid(self):
        reference = np.zeros((7, 8, 3), np.uint8)
        source = np.full_like(reference, 255)
        support = np.zeros((7, 8), bool)
        support[3, 4] = True
        score = score_change(reference, source, support, "ssim")
        self.assertAlmostEqual(float(score[3, 4]), (1 - .0001 / 1.0001) / 2, places=7)
        self.assertTrue(np.isnan(score[~support]).all())

    def test_negative_ssim_has_more_than_half_change_and_is_symmetric_deterministic(self):
        yy, xx = np.mgrid[:20, :20]
        reference = np.repeat((((xx + yy) % 2) * 255).astype(np.uint8)[..., None], 3, axis=2)
        source = 255 - reference
        support = np.ones((20, 20), bool)
        score = score_change(reference, source, support, "ssim")
        self.assertGreater(float(score[10, 10]), .9)
        np.testing.assert_array_equal(score, score_change(reference, source, support, "ssim"))
        np.testing.assert_array_equal(score, score_change(source, reference, support, "ssim"))
        self.assertTrue(np.isfinite(score).all())
        self.assertTrue((score >= 0).all() and (score <= 1).all())
        np.testing.assert_array_equal(source, 255 - reference)


if __name__ == "__main__":
    unittest.main()
