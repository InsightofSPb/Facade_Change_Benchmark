"""Final-output FPR constraints must not be confused with scalar gate budgets."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.real_eval import real_final_fpr as final
from test_real_gt_calibration import metric_fixture
from test_real_gt_metrics import hashes, sha


def pair(building, frame, scores, native, labels=None):
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.zeros(len(scores), np.uint8) if labels is None else np.asarray(labels, np.uint8)
    common = dict(building_id=building, sequence_id='seq-' + building, frame=frame,
                  labels=labels, pixels=np.arange(len(scores), dtype=np.int32), shape=[1, len(scores)],
                  support_sha256=sha_bytes(labels.tobytes()))
    return dict(common, native=np.asarray(native, bool), scores=scores.copy()), dict(common, native=None, scores=scores)


def sha_bytes(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def rows(*pairs):
    return [value[0] for value in pairs], [value[1] for value in pairs]


def actual_weighted_fpr(left, right, choice):
    values = {}
    for a, b in zip(left, right):
        if a['building_id'] == choice['heldout_building'] or not np.any(a['labels'] == 0):
            continue
        p = final.predict(a['native'], b['scores'], choice)
        values.setdefault(a['building_id'], []).append(float(np.mean(p[a['labels'] == 0])))
    return float(np.mean([np.mean(v) for v in values.values()]))


class FinalFPRProfileTests(unittest.TestCase):
    def test_or_floor_and_cap_and_scalar_use_full_negative_denominator(self):
        left, right = rows(pair('A', 1, [.1, .3, .6, .9], [1, 0, 0, 0]))
        profile = final.final_fpr_profiles(left, right, 'heldout')
        self.assertEqual(profile['native_fpr'], .25)
        # OR permits one extra false positive in the full four-pixel support.
        choice = final.choose_final_threshold(profile, 'heldout', .5, 'or')
        self.assertAlmostEqual(choice['threshold'], .6, places=6)
        self.assertEqual(choice['achieved_train_fpr'], .5)
        self.assertEqual(actual_weighted_fpr(left, right, choice), .5)
        infeasible = final.choose_final_threshold(profile, 'heldout', .05, 'or')
        self.assertFalse(infeasible['feasible'])
        self.assertEqual(infeasible['status'], 'budget_unattainable')
        self.assertIsNone(infeasible['threshold'])
        self.assertEqual(infeasible['minimum_train_fpr'], .25)
        with self.assertRaises(ValueError):
            final.predict(left[0]['native'], right[0]['scores'], infeasible)
        native = final.choose_final_threshold(profile, 'heldout', .05, 'native')
        self.assertEqual(native['status'], 'fixed_reference_outside_budget')
        np.testing.assert_array_equal(final.predict(left[0]['native'], right[0]['scores'], native), [1, 0, 0, 0])

    def test_and_uses_all_gate_endpoint_including_positive_below_negative_minimum(self):
        left, right = rows(pair('A', 1, [.2, .8, -10.], [1, 0, 1], [0, 0, 1]))
        profile = final.final_fpr_profiles(left, right, 'heldout')
        choice = final.choose_final_threshold(profile, 'heldout', .5, 'and')
        self.assertEqual(choice['gate_mode'], 'all')
        self.assertIsNone(choice['threshold'])
        np.testing.assert_array_equal(final.predict(left[0]['native'], right[0]['scores'], choice), [1, 0, 1])
        self.assertEqual(choice['achieved_train_fpr'], .5)
        json.dumps(choice, allow_nan=False)  # No -infinity sentinel in JSON.
        empty = dict(choice, operation='scalar', gate_mode='empty')
        self.assertFalse(final.predict(left[0]['native'], right[0]['scores'], empty).any())

    def test_ties_and_equal_facade_then_frame_weights(self):
        left, right = rows(pair('A', 1, [.1, .1], [0, 1]), pair('A', 2, [.9], [1]),
                           pair('B', 1, [.5] * 100, [0] * 100))
        profile = final.final_fpr_profiles(left, right, 'heldout')
        scalar = final.choose_final_threshold(profile, 'heldout', .25, 'scalar')
        self.assertEqual(scalar['threshold'], .5)
        self.assertAlmostEqual(profile['native_fpr'], .375)
        self.assertAlmostEqual(actual_weighted_fpr(left, right, scalar), .25)
        # All .5 ties disappear together; one pixel cannot be admitted to fill budget.
        less = dict(scalar, threshold=float(np.nextafter(np.float32(.5), np.float32(-np.inf))))
        self.assertGreater(actual_weighted_fpr(left, right, less), .25)
        for op, budget in [('and', .2), ('or', .5), ('scalar', .3)]:
            choice = final.choose_final_threshold(profile, 'heldout', budget, op)
            self.assertAlmostEqual(choice['achieved_train_fpr'], actual_weighted_fpr(left, right, choice), places=12)
            self.assertLessEqual(choice['achieved_train_fpr'], budget + 1e-12)

    def test_holdout_and_positive_score_changes_do_not_affect_fit(self):
        left, right = rows(pair('A', 1, [.2, .8, 100], [1, 0, 1], [0, 0, 1]),
                           pair('B', 1, [.3, .7], [0, 0]), pair('H', 1, [.1, .9], [1, 1]))
        before = final.final_fpr_profiles(left, right, 'H')
        changed_left, changed_right = copy.deepcopy((left, right))
        changed_right[0]['scores'][2] = -100
        changed_left[-1], changed_right[-1] = pair('H', 77, [999] * 100, [1] * 100, [1] * 100)
        after = final.final_fpr_profiles(changed_left, changed_right, 'H')
        for op in final.OPERATIONS:
            self.assertEqual(final.choose_final_threshold(before, 'H', .3, op), final.choose_final_threshold(after, 'H', .3, op))
        self.assertNotIn('H', before['audit']['train_buildings'])

    def test_no_other_negatives_is_unsupported_not_all_negative_prediction(self):
        left, right = rows(pair('A', 1, [.4], [0], [1]), pair('H', 1, [.7], [1]))
        profile = final.final_fpr_profiles(left, right, 'H')
        for op in final.OPERATIONS:
            choice = final.choose_final_threshold(profile, 'H', .05, op)
            self.assertEqual(choice['status'], 'unsupported')
            self.assertIsNone(choice['achieved_train_fpr'])
        bad = copy.deepcopy(left)
        bad[0]['native'] = bad[0]['native'].astype(np.uint8)
        with self.assertRaises(ValueError):
            final.final_fpr_profiles(bad, right, 'H')

    def test_or_at_floor_keeps_strict_threshold_without_splitting_ties(self):
        left, right = rows(pair('A', 1, [1, 1, 2, 9], [1, 0, 0, 0], [0, 0, 0, 1]))
        profile = final.final_fpr_profiles(left, right, 'H')
        choice = final.choose_final_threshold(profile, 'H', 1 / 3, 'or')
        self.assertEqual(choice['threshold'], 2)
        # Exact floor does not mean forcibly removing high-scoring positives.
        np.testing.assert_array_equal(final.predict(left[0]['native'], right[0]['scores'], choice), [1, 0, 0, 1])


class FinalFPRIntegrationTests(unittest.TestCase):
    def test_saved_val_maps_all_operations_shared_support_and_frozen_class_slices(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, snapshot, review, gt, _, _ = metric_fixture(base)
            before = {path: hashes(path) for path in (root, snapshot, review, gt)}
            masks = {}
            def capture(**data):
                masks[(data['family'], data['goal'], data['sample']['sequence_id'], data['sample']['frame'], data['predictor'])] = data['prediction'].copy()
            out = base / 'final_fpr'
            with contextlib.redirect_stdout(io.StringIO()), patch('subprocess.Popen', side_effect=AssertionError('No GPU inference')):
                result = final.run(gt, out, budgets=(.01, .05, .15, .5), scalar_methods=('msdzip_abs',), prediction_hook=capture)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(before, {path: hashes(path) for path in before})
            self.assertTrue((out / 'class_macro.csv').exists())
            self.assertTrue((out / 'prediction_recipes.json').exists())
            provenance = json.loads((out / 'provenance.json').read_text())
            self.assertTrue(provenance['no_class_specific_thresholds'])
            self.assertTrue(provenance['no_model_inference'])
            grouped = {}
            for row in result['per_pair']:
                key = row['family'], row['goal'], row['sequence_id'], row['frame']
                grouped.setdefault(key, []).append(row)
            for records in grouped.values():
                self.assertEqual(len({r['support_sha256'] for r in records}), 1)
                self.assertEqual(len({r['valid_pixels'] for r in records}), 1)
                for row in records:
                    self.assertEqual(row['heldout_building'], row['building_id'])
            for fit in result['calibration_audit']:
                self.assertNotIn(fit['heldout_building'], fit['train_buildings'])
                if fit['status'] == 'calibrated':
                    self.assertLessEqual(fit['achieved_train_fpr'], fit['goal'] + 1e-12)
                if fit['status'] == 'budget_unattainable':
                    self.assertEqual(fit['evaluated_pairs'], 0)
                    self.assertIsNone(fit['actual_heldout_fpr'])
                    self.assertFalse(any(r['family'] == fit['family'] and r['goal'] == fit['goal']
                                         and r['operation'] == fit['operation'] and r['building_id'] == fit['heldout_building']
                                         for r in result['per_pair']))
            comparisons = result['comparisons']
            self.assertTrue(any(r['candidate'] == 'and' and r['baseline'] == 'scalar' for r in comparisons))
            self.assertTrue(any(r['candidate'] == 'or' and r['baseline'] == 'native' for r in comparisons))
            for row in comparisons:
                self.assertLessEqual(row['evaluated_pairs'], row['full_family_pairs'])
            family = 'final_fpr/geoscd:frozen:previous+msdzip_abs:current_train:history'
            coverage = next(r for r in result['coverage'] if r['family'] == family)
            self.assertEqual(coverage['matched_pairs'], 3)
            with self.assertRaises(FileExistsError):
                final.run(gt, out)
            # Every class prediction has exactly an already frozen overall recipe.
            with (out / 'class_per_pair.csv').open() as stream:
                class_rows = list(csv.DictReader(stream))
            self.assertTrue(class_rows)
            for row in class_rows:
                original_family = row['family'].split('/matched_', 1)[0]
                self.assertTrue((original_family, float(row['goal']), row['sequence_id'], int(row['frame']), row['predictor']) in masks)
            with (out / 'class_comparisons.csv').open() as stream:
                class_comparisons = list(csv.DictReader(stream))
            self.assertTrue(class_comparisons)
            partial_family = 'final_fpr/anychange:frozen:previous+msdzip_abs:current_train:previous/matched_or'
            partial = [r for r in class_rows if r['family'] == partial_family and float(r['goal']) == .15]
            self.assertEqual({r['building_id'] for r in partial}, {'B'})
            self.assertEqual(len({r['predictor'] for r in partial}), 3)
            matched = [r for r in class_comparisons if r['family'] == partial_family and float(r['goal']) == .15]
            self.assertTrue(matched)
            self.assertTrue(all(int(r['facades']) <= 1 for r in matched))
            pair_comparisons = [r for r in comparisons if r['family'] == partial_family.rsplit('/', 1)[0]
                                and r['goal'] == .15 and r['candidate'] == 'or' and r['subset'] == 'all']
            self.assertEqual(len(pair_comparisons), 2)
            self.assertTrue(all(r['evaluated_pairs'] == 2 and r['full_family_pairs'] == 4 for r in pair_comparisons))

    def test_class_initialization_failure_records_failed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _, _, _, gt, _, _ = metric_fixture(base)
            out = base / 'failed'
            with patch.object(final, 'SliceCollector', side_effect=ValueError('bad ontology')):
                with self.assertRaisesRegex(ValueError, 'bad ontology'):
                    final.run(gt, out)
            self.assertEqual(json.loads((out / 'run.json').read_text())['status'], 'failed')

    def test_authenticated_non_val_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, snapshot, _, gt, _, _ = metric_fixture(base)
            source_units = snapshot / '2026-10-17_source_units.json'
            units = json.loads(source_units.read_text())
            for unit in units:
                unit['split'] = 'test'
            source_units.write_text(json.dumps(units))
            # Remove authoritative duplicate per-unit records, then update only
            # the test fixture's authentication chain; split must still fail.
            provenance_path = gt / 'provenance.json'
            provenance = json.loads(provenance_path.read_text())
            provenance['source_metadata_sha256'] = {
                path: sha(Path(path)) for path in provenance['source_metadata_sha256'] if Path(path).parent.name != 'units'}
            provenance_path.write_text(json.dumps(provenance))
            with self.assertRaisesRegex(ValueError, 'VAL'):
                final.run(gt, base / 'forbidden', class_slices=False)
            self.assertFalse((base / 'forbidden').exists())

    def test_cli_help_from_other_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(Path(final.__file__).resolve()), '--help'], cwd=directory,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--budgets', result.stdout)
            self.assertIn('--scalar-methods', result.stdout)


if __name__ == '__main__':
    unittest.main()
