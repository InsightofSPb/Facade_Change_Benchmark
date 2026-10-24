"""Scientific invariants for calibration of saved night-run maps."""
import copy
import json
import unittest

import numpy as np

from scripts.real_eval import night_pair_calibration as calibration


def row(building, left, right=None, labels=None, frame=1, native_left=False, native_right=False):
    left = np.asarray(left, dtype=bool if native_left else np.float32)
    result = dict(sequence_id='seq-' + building, frame=frame, building_id=building,
                  pixels=np.arange(len(left), dtype=np.int32), left=left,
                  labels=np.asarray(labels, np.uint8) if labels is not None else np.zeros(len(left), np.uint8))
    if right is not None:
        result['right'] = np.asarray(right, dtype=bool if native_right else np.float32)
    return result


def weighted_fpr(rows, model):
    buildings = {}
    for sample in rows:
        if sample['building_id'] == model['choice']['heldout_building']:
            continue
        negative = sample['labels'] == 0
        if negative.any():
            prediction = calibration.predict(sample, model)
            buildings.setdefault(sample['building_id'], []).append(float(np.mean(prediction[negative])))
    return float(np.mean([np.mean(values) for values in buildings.values()]))


class NightPairCalibrationTests(unittest.TestCase):
    def test_heldout_arrays_are_not_read_and_positive_scores_do_not_select_thresholds(self):
        rows = [row('A', [0, 3, 10], [9, 2, 99], [0, 0, 1]),
                row('B', [1, 2], [3, 5]), row('H', [99, 99], [99, 99])]
        before = calibration.prepare(rows, 'H')
        changed = copy.deepcopy(rows)
        changed[0]['left'][-1], changed[0]['right'][-1] = -1000, -2000
        # Even malformed heldout arrays cannot affect the fitted distributions.
        changed[-1] = dict(sequence_id='other-H', frame=77, building_id='H',
                           labels='not an array', left=object())
        after = calibration.prepare(changed, 'H')
        for op in ('and', 'or'):
            a, b = calibration.choose(before, .3, op), calibration.choose(after, .3, op)
            self.assertEqual(a['choice'], b['choice'])
            np.testing.assert_array_equal(calibration.predict(rows[-1], a), calibration.predict(rows[-1], b))
        self.assertEqual(before['audit']['train_buildings'], ['A', 'B'])
        self.assertEqual(before['audit']['train_negative_pixels'], 4)

    def test_final_union_budget_differs_from_separate_branch_budgets(self):
        rows = [row('A', [0, 1, 2, 3], [3, 2, 1, 0])]
        left_only = [dict(rows[0])]
        del left_only[0]['right']
        right_only = [dict(left_only[0], left=rows[0]['right'])]
        left = calibration.predict(left_only[0], calibration.fit(left_only, 'H', .25))
        right = calibration.predict(right_only[0], calibration.fit(right_only, 'H', .25))
        self.assertEqual(np.mean(left), .25)
        self.assertEqual(np.mean(right), .25)
        self.assertEqual(np.mean(left | right), .5)
        profile = calibration.prepare(rows, 'H')
        model = calibration.choose(profile, .25, 'or')
        self.assertEqual(weighted_fpr(rows, model), 0.)  # Ties forbid a .25 union.
        self.assertEqual(model['choice']['threshold'], .75)
        self.assertEqual(model['choice']['achieved_train_fpr'], 0.)
        self.assertEqual(model['choice']['rank_normalization'], 'weighted_training_negative_strict_left_ecdf')
        self.assertFalse(calibration.predict(rows[0], model).any())

    def test_strict_ecdf_retains_new_extremes_at_zero_training_fpr(self):
        rows = [row('A', [1, 1, 3], [2, 2, 4])]
        probe = row('H', [3, 4, -100, 4], [4, 5, 5, -100])
        for operation, expected in [('and', [False, True, False, False]),
                                    ('or', [False, True, True, True])]:
            model = calibration.fit(rows, 'H', 0., operation)
            self.assertEqual(model['choice']['achieved_train_fpr'], 0.)
            np.testing.assert_array_equal(calibration.predict(probe, model), expected)
        # Strict-left ECDF: the tied minimum has rank zero, above maximum rank one.
        ecdf = calibration.prepare(rows, 'H')['rank_models'][0]
        np.testing.assert_allclose(calibration._rank(np.array([0, 1, 2, 3, 4]), ecdf),
                                   [0, 0, 2 / 3, 2 / 3, 1])

    def test_json_recipe_reproduces_rank_fusion_without_calibration_data(self):
        rows = [row('A', [1, 1, 3, 5], [7, 5, 5, 2]),
                row('A', [2, 9], [1, 8], frame=2), row('B', [4, 6, 8], [4, 2, 9])]
        probe = row('H', [-100, 1, 2, 3, 4, 5, 6, 9, 100],
                    [100, 7, 6, 5, 4, 3, 2, 1, -100])
        profile = calibration.prepare(rows, 'H')
        left_rank, right_rank = [calibration._rank(probe[name], ecdf)
                                 for name, ecdf in zip(('left', 'right'), profile['rank_models'])]
        for operation in ('and', 'or'):
            for budget in (0., .05, .25, .5, .9):
                model = calibration.choose(profile, budget, operation)
                choice = json.loads(json.dumps(model['choice'], allow_nan=False))
                fused = np.minimum(left_rank, right_rank) if operation == 'and' else np.maximum(left_rank, right_rank)
                expected = fused > choice['threshold']
                np.testing.assert_array_equal(calibration.predict_choice(probe, choice), expected)
                np.testing.assert_array_equal(calibration.predict(probe, model), expected)
                self.assertEqual(set(choice['raw_gates']), {'left', 'right'})

    def test_facades_and_negative_bearing_frames_have_equal_weight(self):
        rows = [row('A', [0, 0], [0, 0], frame=1), row('A', [9], [9], frame=2),
                row('B', [5] * 100, [5] * 100),
                row('A', [-999], [-999], [1], frame=3), row('P', [999], [999], [1])]
        for op in ('and', 'or'):
            model = calibration.fit(rows, 'H', .25, op)
            self.assertEqual(model['choice']['train_buildings'], ['A', 'B'])
            self.assertEqual(len(model['choice']['train_pair_keys']), 3)
            self.assertEqual(model['choice']['train_negative_pixels'], 103)
            self.assertAlmostEqual(model['choice']['achieved_train_fpr'], .25)
            self.assertAlmostEqual(weighted_fpr(rows, model), .25)
            self.assertTrue(calibration.predict(rows[1], model)[0])
            self.assertFalse(calibration.predict(rows[2], model).any())
            reverse = calibration.fit(list(reversed(rows)), 'H', .25, op)
            self.assertEqual(model['choice'], reverse['choice'])

    def test_native_or_floor_is_explicit_and_native_and_all_gate_keeps_low_positive(self):
        rows = [row('A', [1, 0, 1], [1, 2, -100], [0, 0, 1], native_left=True)]
        profile = calibration.prepare(rows, 'H')
        impossible = calibration.choose(profile, .1, 'or')
        self.assertEqual(impossible['choice']['status'], 'budget_unattainable')
        self.assertFalse(impossible['choice']['feasible'])
        self.assertEqual(impossible['choice']['minimum_train_fpr'], .5)
        with self.assertRaises(ValueError):
            calibration.predict(rows[0], impossible)
        model = calibration.choose(profile, .5, 'and')
        self.assertEqual(model['choice']['gate_mode'], 'all')
        self.assertIsNone(model['choice']['threshold'])
        np.testing.assert_array_equal(calibration.predict(rows[0], model), [True, False, True])
        # Order of a scalar and a native branch changes no behavior.
        swapped = [dict(rows[0], left=rows[0]['right'], right=rows[0]['left'])]
        other = calibration.fit(swapped, 'H', .5, 'and')
        np.testing.assert_array_equal(calibration.predict(rows[0], model), calibration.predict(swapped[0], other))

    def test_fixed_native_masks_are_reported_outside_budget_and_not_suppressed(self):
        rows = [row('A', [1, 0, 1, 0], [0, 1, 1, 0], native_left=True, native_right=True)]
        for operation, expected in [('and', [0, 0, 1, 0]), ('or', [1, 1, 1, 0])]:
            model = calibration.fit(rows, 'H', .1, operation)
            self.assertEqual(model['choice']['status'], 'fixed_reference_outside_budget')
            self.assertFalse(model['choice']['feasible'])
            np.testing.assert_array_equal(calibration.predict(rows[0], model), expected)
            self.assertEqual(model['choice']['achieved_train_fpr'], np.mean(expected))
        single = [dict(rows[0])]
        del single[0]['right']
        model = calibration.fit(single, 'H', .5)
        self.assertEqual(model['choice']['status'], 'fixed_reference_within_budget')
        np.testing.assert_array_equal(calibration.predict(single[0], model), rows[0]['left'])

    def test_unknown_gt_duplicates_misaligned_and_nonfinite_training_inputs_rejected(self):
        base = row('A', [1, 2], [3, 4])
        bad_values = [dict(base, labels=np.array([0, 255], np.uint8)),
                      dict(base, labels=np.array([0, 2], np.uint8)),
                      dict(base, labels=np.array([0, 1], bool)),
                      dict(base, pixels=np.array([1, 1], np.int32)),
                      dict(base, pixels=np.array([1], np.int32)),
                      dict(base, left=np.array([1, 2], np.float64)),
                      dict(base, right=np.array([3, np.nan], np.float32))]
        for bad in bad_values:
            with self.subTest(bad=repr(bad)), self.assertRaises(ValueError):
                calibration.prepare([bad], 'H')
        with self.assertRaises(ValueError):
            calibration.prepare([base, base], 'H')

    def test_no_other_negatives_is_unsupported_even_for_native_masks(self):
        for native in (False, True):
            rows = [row('A', [1], [1], [1], native_left=native, native_right=native),
                    row('H', [1], [1], native_left=native, native_right=native)]
            for op in ('and', 'or'):
                model = calibration.fit(rows, 'H', .05, op)
                self.assertEqual(model['choice']['status'], 'unsupported')
                self.assertIsNone(model['choice']['achieved_train_fpr'])
                with self.assertRaises(ValueError):
                    calibration.predict(rows[1], model)

    def test_serialized_audit_is_finite_and_actual_final_fpr_matches_all_profiles(self):
        rng = np.random.default_rng(4827)
        for kinds in [(False,), (True,), (False, False), (True, False), (False, True), (True, True)]:
            rows = []
            for building, frame, size in [('A', 1, 7), ('A', 2, 31), ('B', 1, 19), ('H', 1, 13)]:
                left = rng.integers(0, 5, size)
                right = rng.integers(0, 5, size) if len(kinds) == 2 else None
                rows.append(row(building, left, right, rng.integers(0, 2, size), frame,
                                native_left=kinds[0], native_right=kinds[-1]))
            profile = calibration.prepare(rows, 'H')
            for operation in (('single',) if len(kinds) == 1 else ('and', 'or')):
                for budget in (0., .01, .05, .25, .8):
                    model = calibration.choose(profile, budget, operation)
                    choice = model['choice']
                    json.dumps(choice, allow_nan=False)
                    if choice['status'] == 'budget_unattainable':
                        self.assertGreater(choice['minimum_train_fpr'], budget)
                        continue
                    self.assertAlmostEqual(weighted_fpr(rows, model), choice['achieved_train_fpr'], places=12)
                    if choice['status'] == 'calibrated':
                        self.assertLessEqual(weighted_fpr(rows, model), budget + 1e-12)


if __name__ == '__main__':
    unittest.main()
