"""Route geometry/projection tests independent of checkpoints and GPUs."""
import unittest

import numpy as np

from scripts.real_eval.semantic_geometry import regions, semantic_prediction, stitch, target_union


VOCAB = [
    {"name": "CRACK", "group": "domain"},
    {"name": "SPALLING", "group": "domain"},
    {"name": "WALL", "group": "structure"},
    {"name": "WINDOW", "group": "structure"},
]
CATEGORIES = [{"name": "CRACK"}, {"name": "SPALLING"}, {"name": "ADVERTISEMENTS"}]


def native(scores, names, kind, **extra):
    return {"scores": np.asarray(scores, np.float32).reshape(len(names), 1, 1),
            "class_names": names, "score_kind": kind, **extra}


class SemanticGeometryTests(unittest.TestCase):
    def test_fixed_dino_keeps_overlapping_independent_labels(self):
        value = native([.8, .9], ["CRACK", "SPALLING"], "independent_sigmoid")
        scores, predicted, names = semantic_prediction(value, VOCAB, "fixed", confidence=.5)
        np.testing.assert_array_equal(scores, value["scores"])
        self.assertTrue(predicted.all())
        self.assertEqual(names, ["CRACK", "SPALLING"])

    def test_fixed_maskclip_background_competes_and_missing_class_stays_missing(self):
        value = native([.6, .3, .1], ["BACKGROUND", "CRACK", "SPALLING"], "native_exclusive_probability")
        _, predicted, names = semantic_prediction(value, VOCAB, "fixed", confidence=.25)
        np.testing.assert_array_equal(predicted[:, 0, 0], [True, False, False])
        self.assertNotIn("ADVERTISEMENTS", names)
        self.assertFalse(target_union(predicted, names, CATEGORIES).any())

    def test_sam_domain_and_layered_keep_overlap_flat_resolves_winner(self):
        value = native([.8, .2, .9, .1], ["CRACK", "SPALLING", "WALL", "WINDOW"],
                       "max_instance_score_times_mask_probability")
        for variant in ("domain", "expanded_layered"):
            _, predicted, _ = semantic_prediction(value, VOCAB, variant, confidence=.5)
            np.testing.assert_array_equal(predicted[:, 0, 0], [True, False, True, False])
        _, flat, _ = semantic_prediction(value, VOCAB, "expanded_flat", confidence=.5)
        np.testing.assert_array_equal(flat[:, 0, 0], [False, False, True, False])

    def test_stock_layered_recomputes_target_group_from_raw_scores(self):
        raw = np.array([.15, .10, .90, .80], np.float32).reshape(4, 1, 1)
        value = native([.001, .001, .90, .098], ["CRACK", "SPALLING", "WALL", "WINDOW"],
                       "vocabulary_softmax_uncalibrated", extra_arrays={"raw_scores": raw},
                       metadata={"softmax_scale": 100.0})
        scores, predicted, _ = semantic_prediction(value, VOCAB, "expanded_layered", confidence=.5)
        np.testing.assert_array_equal(predicted[:, 0, 0], [True, False, True, False])
        self.assertGreater(float(scores[0, 0, 0]), .99)
        _, flat, _ = semantic_prediction(value, VOCAB, "expanded_flat", confidence=.5)
        self.assertFalse(flat[0].any())
        self.assertTrue(flat[2].all())

    def test_unlisted_classes_do_not_enter_target_union(self):
        predicted = np.ones((2, 3, 4), bool)
        result = target_union(predicted, ["WALL", "BACKGROUND"], CATEGORIES)
        self.assertEqual(result.shape, (3, 4))
        self.assertFalse(result.any())

    def test_regions_four_connected_and_component_area_filter(self):
        mask = np.zeros((8, 9), bool)
        mask[1:3, 1:3] = True
        mask[3:5, 3:5] = True  # touches first only at a diagonal
        mask[6, 7] = True
        self.assertEqual(regions(mask, min_area=2, max_regions=9, padding=0, min_size=1),
                         [(1, 1, 3, 3), (3, 3, 5, 5)])

    def test_region_limits_largest_first_and_padding_clamped(self):
        mask = np.zeros((12, 15), bool)
        mask[:3, :4] = True
        mask[9:11, 12:14] = True
        boxes = regions(mask, min_area=1, max_regions=1, padding=2, min_size=8)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0], (0, 0, 8, 8))

    def test_regions_bridge_merges_runs_and_handles_small_grid(self):
        mask = np.array([[1, 0, 1], [1, 1, 1]], bool)
        self.assertEqual(regions(mask, min_area=5, max_regions=2, padding=0, min_size=1), [(0, 0, 3, 2)])
        self.assertEqual(regions(mask, min_area=1, max_regions=2, padding=16, min_size=56), [(0, 0, 3, 2)])
        self.assertEqual(regions(np.zeros((2, 3), bool)), [])
        with self.assertRaises(ValueError):
            regions(mask.astype(np.uint8))

    def test_stitch_max_scores_union_and_false_negatives_outside_regions(self):
        parts = [((0, 0, 2, 2), np.full((2, 2), .4, np.float32), np.ones((2, 2), bool)),
                 ((1, 1, 3, 3), np.full((2, 2), .8, np.float32), np.ones((2, 2), bool))]
        scores, predicted, covered = stitch(parts, (4, 4))
        self.assertAlmostEqual(float(scores[1, 1]), .8)
        self.assertAlmostEqual(float(scores[0, 0]), .4)
        self.assertEqual(int(covered.sum()), 7)
        self.assertEqual(int(predicted.sum()), 7)
        self.assertEqual(float(scores[3, 3]), 0.)
        truth = np.zeros((4, 4), bool)
        truth[1, 1] = truth[3, 3] = True
        # Evaluate on original support, not covered: missed ROI remains FN.
        self.assertEqual(int((truth & ~predicted).sum()), 1)

    def test_unknown_semantic_variant_fails_explicitly(self):
        value = native([.8], ["CRACK"], "independent_sigmoid")
        with self.assertRaisesRegex(ValueError, "Unknown semantic variant"):
            semantic_prediction(value, VOCAB, "layerd_typo")

    def test_regions_reject_fractional_or_boolean_options(self):
        mask = np.ones((2, 3), bool)
        for option in ({"padding": .5}, {"min_area": True}, {"max_regions": 1.5}, {"min_size": 0}):
            with self.assertRaises(ValueError):
                regions(mask, **option)

    def test_stitch_rejects_out_of_bounds_bad_shapes_and_infinity(self):
        scores = np.ones((2, 2), np.float32)
        prediction = np.ones((2, 2), bool)
        invalid = [
            ((-1, 0, 1, 2), scores, prediction),
            ((0, 0, 4, 2), scores, prediction),
            ((0, 0, 2, 2), scores[:, :1], prediction),
            ((0, 0, 2, 2), scores * np.inf, prediction),
            ((0, 0, 2, 2), scores, prediction.astype(np.uint8)),
        ]
        for part in invalid:
            with self.subTest(box=part[0]):
                with self.assertRaises(ValueError):
                    stitch([part], (3, 3))
        # NaN unsupported pixels map to zero, not to excluded evaluation pixels.
        scores[0, 0] = np.nan
        prediction[0, 0] = False
        output, predicted, _ = stitch([((0, 0, 2, 2), scores, prediction)], (3, 3))
        self.assertEqual(float(output[0, 0]), 0)
        self.assertFalse(predicted[0, 0])

    def test_stitch_multiclass_and_empty_routes(self):
        local = np.zeros((2, 2, 2), np.float32)
        local[1] = .9
        scores, predicted, covered = stitch([((1, 0, 3, 2), local, local > .5)], (3, 4), channels=2)
        self.assertEqual(scores.shape, (2, 3, 4))
        self.assertFalse(predicted[0].any())
        self.assertEqual(int(predicted[1].sum()), 4)
        self.assertFalse(covered[2].any())
        scores, predicted, covered = stitch([], (3, 4), channels=2)
        self.assertFalse(predicted.any())
        self.assertFalse(covered.any())
        self.assertEqual(float(scores.sum()), 0.)


if __name__ == "__main__":
    unittest.main()
