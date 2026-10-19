"""Frozen replay, exact event partition, facade weighting, and a read-only cache audit."""
import contextlib
import importlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

base = importlib.import_module('test_2026-10-19_continuous_semantic_updates')
audit = importlib.import_module('scripts.real_eval.2026-10-20_semantic_error_audit')
pilot, runner = audit.pilot, audit.continuous


class SemanticErrorAuditTests(unittest.TestCase):
    def test_partition_sign_zero_strict_threshold_and_unknown(self):
        truth = np.ones((1, 6), bool)
        delta = np.array([[-.5, 0, .25, .5, .75, -.5]], np.float32)
        predicted = delta > .5  # equality must be a miss
        known = np.array([[1, 1, 1, 1, 1, 0]], bool)
        row = audit.decompose(truth, delta, predicted, known)
        self.assertEqual([row[k] for k in audit.PARTS], [1, 1, 2, 1])
        self.assertEqual(row['gt_events'], 5)
        self.assertEqual(row['sign_ceiling'], .6)
        self.assertEqual(row['recall'], .2)
        self.assertEqual(sum(row[k + '_fraction'] for k in audit.PARTS), 1.)
        invalid = predicted.copy()
        invalid[0, 0] = True
        with self.assertRaisesRegex(ValueError, 'direction gate'):
            audit.decompose(truth, delta, invalid, known)

    def test_binary_overlap_does_not_credit_wrong_class_or_direction(self):
        p = base.pair(positives=1)
        truth = runner.event_truth(p)
        # GT: CRACK gain + REPAIRS loss. Predict the opposite class directions.
        signed = np.ones(truth.shape, np.float32)
        signed[0, 0, 0, 0] = signed[1, 1, 0, 0] = -1
        prediction = np.zeros(truth.shape, bool)
        prediction[1, 0, 0, 0] = prediction[0, 1, 0, 0] = True
        row = audit.decompose(truth, signed, prediction, p['labels'] != 255)
        self.assertEqual(row['gt_events'], 2)
        self.assertEqual(row['wrong_sign'], 2)
        self.assertEqual(row['detected'], 0)
        self.assertEqual(row['fp'], 2)
        self.assertEqual(row['sign_ceiling'], 0.)

    def test_no_gt_is_na_and_false_positives_remain(self):
        truth = np.zeros((2, 2), bool)
        row = audit.decompose(truth, np.ones((2, 2)), np.ones((2, 2), bool), np.ones((2, 2), bool))
        self.assertIsNone(row['sign_ceiling'])
        self.assertIsNone(row['recall'])
        self.assertEqual(row['fp'], 4)
        self.assertEqual(row['fpr'], 1.)

    def test_multilabel_transitions_keep_exact_sets_and_exclude_unknown(self):
        pair = dict(reference=dict(packed=np.array([[3, 1, 0, 1]], np.uint16)),
                    source=dict(packed=np.array([[2, 3, 1, 1]], np.uint16)),
                    labels=np.array([[1, 1, 255, 0]], np.uint8))
        rows = audit.gt_transitions(pair, ['A', 'B'])
        found = {(r['before_classes'], r['after_classes']): r['pixels'] for r in rows}
        self.assertEqual(found, {('A|B', 'B'): 1, ('A', 'A|B'): 1, ('A', 'A'): 1})
        self.assertEqual(sum(r['pixels'] for r in rows if r['changed']), 2)

    def test_positive_pairs_then_equal_facades_not_pooled_counts(self):
        rows = []
        for building, size, hit in [('A', 1, True), ('A', 99, False), ('B', 100, True), ('C', 0, False)]:
            shape = (1, size)
            result = audit.decompose(np.ones(shape, bool), np.ones(shape),
                                     np.full(shape, hit, bool), np.ones(shape, bool))
            rows.append(dict(split='val', variant='semantic', class_name='__all__', direction='gain_loss',
                             building_id=building, **result))
        summary, facades = audit.aggregate(rows)
        self.assertEqual(summary[0]['recall'], .75)  # (mean(1,0) + 1)/2, not 101/200
        self.assertEqual(summary[0]['facades'], 3)
        self.assertEqual(summary[0]['recall_facades'], 2)
        self.assertEqual(summary[0]['gt_events'], 200)
        self.assertIsNone(next(r for r in facades if r['building_id'] == 'C')['recall'])

    def test_complete_frozen_run_no_fitting_and_readonly_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, semantic, wide = base.fixture(root)
            frozen, out = root / 'evaluation', root / 'audit'
            with contextlib.redirect_stdout(io.StringIO()):
                status = runner.run(semantic, wide, frozen, feature_design='directed_delta')
            self.assertEqual(status['status'], 'completed', status.get('error'))
            inputs = (prepared, semantic, wide, root / 'cache', frozen)
            before = base.old.hashes(inputs)
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(runner, 'fit_bundle', side_effect=AssertionError('No fitting allowed')), \
                    patch.object(runner.calibration, 'calibrate_threshold', side_effect=AssertionError('No calibration allowed')):
                state = audit.run(frozen, out, selected=(('facade-val', 2020, 2021),))
            self.assertEqual(state['status'], 'completed', state.get('error'))
            self.assertEqual(state['evaluation_contexts'], 8)
            self.assertEqual(state['gallery_pairs'], 1)
            self.assertEqual(before, base.old.hashes(inputs))
            self.assertTrue(state['replay_passed'])
            replay = audit.read_csv(out / 'replay_checks.csv')
            self.assertEqual(len(replay), 40)
            self.assertTrue(all(r['passed'] == 'True' for r in replay))
            self.assertEqual(sum(r['saved_masks_checked'] == 'True' for r in replay), 25)
            selected = pilot.read_json(out / 'selected_pairs.json')
            with np.load(out / selected[0]['artifact_path'], allow_pickle=False) as data:
                self.assertEqual(data['before__probabilities'].dtype, np.float32)
                self.assertEqual(data['before__packed'].dtype, np.uint16)
                self.assertIn('semantic_lpips__gain', data)
                self.assertIn('lpips__support', data)
            for item in pilot.read_json(out / 'artifacts.json'):
                self.assertEqual(pilot.sha256(out / item['path']), item['sha256'])
            self.assertTrue((out / 'report.html').is_file())
            self.assertIn('DINO ≥ 0.5', (out / 'report.html').read_text())
            # An altered model blocks instead of silently refitting it.
            index = pilot.read_json(frozen / '2026-10-19_artifacts.json')
            model_path = frozen / index['models'][0]['path']
            model_path.write_text(model_path.read_text() + ' ')
            with contextlib.redirect_stdout(io.StringIO()):
                blocked = audit.run(frozen, root / 'corrupt', selected=())
            self.assertEqual(blocked['status'], 'blocked')
            self.assertIn('model checksum mismatch', blocked['error'])
            self.assertFalse((root / 'corrupt' / 'summary.csv').exists())
            with self.assertRaisesRegex(ValueError, 'separate'):
                audit.run(frozen, frozen / 'child')

    def test_same_counts_different_mask_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, semantic, wide = base.fixture(root)
            frozen = root / 'evaluation'
            original_score = runner.score_pair

            def initial_score(pair, name, model=None):
                events, scores = original_score(pair, name, model)
                if pair['context']['split'] == 'val' and name == 'lpips_only':
                    scores = scores.copy()
                    scores[0, 1] = .9  # an extra FP for this synthetic replay test
                return events, scores

            with contextlib.redirect_stdout(io.StringIO()), patch.object(runner, 'score_pair', initial_score):
                runner.run(semantic, wide, frozen, feature_design='directed_delta')

            def changed_score(pair, name, model=None):
                events, scores = initial_score(pair, name, model)
                if pair['context']['split'] == 'val' and name == 'lpips_only':
                    # This swaps two unchanged pixels, preserving confusion counts.
                    scores[0, 1], scores[0, 2] = scores[0, 2], scores[0, 1]
                return events, scores

            with contextlib.redirect_stdout(io.StringIO()), patch.object(runner, 'score_pair', changed_score):
                blocked = audit.run(frozen, root / 'audit', selected=())
            self.assertEqual(blocked['status'], 'blocked', blocked)
            self.assertIn('mask replay mismatch', blocked['error'])
            self.assertFalse((root / 'audit' / 'summary.csv').exists())


if __name__ == '__main__':
    unittest.main()
