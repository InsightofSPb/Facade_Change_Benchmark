"""Fusion leakage, event semantics and a complete read-only-cache smoke run."""
import contextlib
import copy
import csv
import importlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

continuous = importlib.import_module('scripts.real_eval.2026-10-19_continuous_semantic_updates')
old = importlib.import_module('test_semantic_updates')
pilot = continuous.pilot


def fixture(root):
    prepared, semantic, wide = old.fixture(root)
    manifest = pilot.read_json(prepared / 'manifest.json')
    # More TRAIN facades, sharing synthetic arrays only to keep the fixture small.
    for name in ('B', 'C'):
        sequence = copy.deepcopy(manifest['sequences'][0])
        sequence.update(sequence_id='sequence-train-' + name, building_id='facade-train-' + name)
        manifest['sequences'].append(sequence)
        for folder in (semantic, wide):
            for path in list((folder / 'units').glob('train-*.json')):
                unit = pilot.read_json(path)
                if unit['sequence_id'] != 'sequence-train':
                    continue
                unit.update(sequence_id=sequence['sequence_id'], building_id=sequence['building_id'])
                pilot.atomic_json(folder / 'units' / (name + '-' + path.name), unit)
    pilot.atomic_json(prepared / 'manifest.json', manifest)
    run = pilot.read_json(semantic / 'run.json')
    run['config_revision'] = pilot.digest(dict(config=run['config'],
        prepared=pilot.sha256(prepared / 'manifest.json'), schema=run['schema']))
    pilot.atomic_json(semantic / 'run.json', run)
    for path in (semantic / 'units').glob('*.json'):
        unit = pilot.read_json(path)
        unit['config_revision'] = run['config_revision']
        pilot.atomic_json(path, unit)
    run = pilot.read_json(wide / 'run.json')
    run['contract']['manifest_sha256'] = pilot.sha256(prepared / 'manifest.json')
    run['config_hash'] = pilot.digest(run['contract'])
    pilot.atomic_json(wide / 'run.json', run)
    return prepared, semantic, wide


def pair(building='A', frame=1, positives=2, split='train'):
    before = np.full((8, 8), 2, np.uint16)
    after = before.copy()
    after.ravel()[:positives] = 1
    probabilities = lambda bits: pilot.unpack(bits, 2).astype(np.float32) * .8 + .1
    score = np.where(before != after, .9, .1).astype(np.float32)
    support = np.ones_like(score, bool)
    return dict(context=dict(building_id=building, sequence_id=building, frame=frame, split=split),
                reference=dict(packed=before, probabilities=probabilities(before)),
                source=dict(packed=after, probabilities=probabilities(after)),
                labels=(before != after).astype(np.uint8), support=support,
                support_sha256=pilot.hash_array(support),
                maps={m: dict(scores=score.copy()) for m in pilot.METHODS})


class ContinuousUpdateTests(unittest.TestCase):
    def test_population_weights_and_equal_facades_survive_sampling(self):
        pairs = [pair('A', 1, 1), pair('A', 2, 9), pair('B', 1, 20)]
        x, y, w, audit = continuous.training_sample(pairs, max_events=8)
        expected = (.5 * (1 + 9) / 2 + .5 * 20) / (2 * 64)
        self.assertAlmostEqual(float(w @ y), expected)
        self.assertEqual([r['pair_weight'] for r in audit], [.25, .25, .5])
        again = continuous.training_sample(pairs, max_events=8)
        for first, second in zip((x, y, w), again[:3]):
            np.testing.assert_array_equal(first, second)
        with self.assertRaisesRegex(ValueError, 'TRAIN'):
            continuous.training_sample([pair(split='val')])

    def test_regularized_fit_learns_and_handles_constant_features(self):
        x = np.column_stack((np.arange(20), np.ones(20)))
        y = np.arange(20) >= 10
        model = continuous.fit_logistic(x, y, np.ones(20))
        self.assertTrue(model['converged'])
        self.assertGreater(model['coefficients'][0], 0)
        self.assertAlmostEqual(model['coefficients'][1], 0)
        prediction = continuous.sigmoid(x @ model['coefficients'] + model['intercept'])
        self.assertLess(prediction[0], .1)
        self.assertGreater(prediction[-1], .9)

    def test_heldout_facade_and_val_test_cannot_affect_fitting_or_thresholds(self):
        pairs = [pair('A'), pair('B'), pair('V', split='val'), pair('T', split='test')]
        with contextlib.redirect_stdout(io.StringIO()):
            before = continuous.fit_bundle(pairs, excluded='B')
        altered = copy.deepcopy(pairs)
        for p in altered[1:]:
            p['reference']['probabilities'][:] = .8
            p['source']['packed'][:] = 1
            p['labels'][:] = 1
            for m in pilot.METHODS:
                p['maps'][m]['scores'][:] = 100
        with contextlib.redirect_stdout(io.StringIO()):
            after = continuous.fit_bundle(altered, excluded='B')
        self.assertEqual(before, after)
        self.assertEqual(before['training_buildings'], ['A'])
        for choice in before['thresholds'].values():
            self.assertEqual(choice['train_buildings'], ['A'])
            self.assertLessEqual(choice['achieved_fpr'], .05 + 1e-12)

    def test_continuous_events_do_not_require_state_crossing_or_gt(self):
        p = pair()
        p['reference']['probabilities'][:] = .2
        p['source']['probabilities'][:] = .4
        model = dict(coefficients=[-3., 3.], intercept=0.)
        events, scores = continuous.score_pair(p, 'semantic', model)
        self.assertTrue((events[0] > .5).all())  # both classes; no 0.5 state crossing
        self.assertFalse(events[1].any())
        p['labels'][:] = 255
        p['source']['packed'][:] = 0
        np.testing.assert_array_equal(continuous.score_pair(p, 'semantic', model)[0], events)
        p['source'], p['reference'] = p['reference'], p['source']
        reversed_events, reversed_scores = continuous.score_pair(p, 'semantic', model)
        np.testing.assert_array_equal(reversed_events, events[::-1])
        np.testing.assert_array_equal(reversed_scores, scores)
        p['source']['probabilities'][:] = p['reference']['probabilities']
        self.assertFalse(continuous.score_pair(p, 'semantic', model)[0].any())

    def test_wrong_direction_counts_fp_fn_and_unknowns_are_excluded(self):
        p = pair(positives=1)
        truth = continuous.event_truth(p)
        wrong = truth[::-1].astype(np.float32)
        # Another prediction lies at an unknown pixel and must not count as FP.
        p['labels'][0, 1] = 255
        wrong[0, 0, 0, 1] = 1
        rows, _, _ = continuous.evaluate(p, wrong, wrong.max(axis=(0, 1)), .5,
                                         [{'name': 'CRACK'}, {'name': 'REPAIRS'}])
        self.assertEqual(rows[0]['f1'], 1.)
        self.assertEqual(rows[1]['f1'], 0.)
        self.assertEqual((rows[1]['fp'], rows[1]['fn']), (2, 2))

    def test_average_precision_uses_equal_facades_and_preserves_undefined(self):
        base = dict(split='val', variant='semantic', target='binary_change', class_name='__all__', direction='change')
        measured = continuous.metrics.evaluate_mask(np.array([True]), np.array([1], np.uint8))
        rows = [dict(base, **measured, building_id=b, average_precision=ap)
                for b, ap in [('A', 0.), ('A', 1.), ('A', None), ('B', 1.)]]
        summaries, _ = continuous.aggregate(rows)
        self.assertEqual(summaries[0]['average_precision'], .75)
        self.assertEqual(summaries[0]['ap_pairs'], 3)
        p = pair(positives=0)
        events, scores = continuous.score_pair(p, 'semantic_delta')
        rows, _, _ = continuous.evaluate(p, events, scores, .5, [{'name': 'C'}, {'name': 'R'}])
        self.assertIsNone(rows[0]['average_precision'])
        self.assertIsNone(rows[1]['average_precision'])

    def test_complete_synthetic_run_all_cohorts_without_input_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, semantic, wide = fixture(root)
            inputs = (prepared, semantic, wide, root / 'cache')
            before = old.hashes(inputs)
            with contextlib.redirect_stdout(io.StringIO()):
                result = continuous.run(semantic, wide, root / 'out')
            self.assertEqual(result['status'], 'completed', result.get('error'))
            self.assertEqual(result['temporal_train_facades'], 3)
            self.assertEqual(before, old.hashes(inputs))
            with (root / 'out' / 'summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({r['split'] for r in rows}, {'train_fit', 'train_fusion_oof', 'val', 'test'})
            self.assertEqual({r['variant'] for r in rows}, set(continuous.VARIANTS))
            index = pilot.read_json(root / 'out' / '2026-10-19_artifacts.json')
            self.assertEqual(len(index['models']), 4)
            self.assertEqual(len(index['maps']), 5)
            for entry in index['models']:
                model = pilot.read_json(root / 'out' / entry['path'])
                self.assertNotIn(model['excluded_building'], model['training_buildings'])
                self.assertEqual(pilot.sha256(root / 'out' / entry['path']), entry['sha256'])
            self.assertTrue((root / 'out' / '2026-10-19_report.html').exists())
            self.assertEqual(len(list((root / 'out' / 'gallery').glob('*.png'))), 50)
            with self.assertRaises(FileExistsError):
                continuous.run(semantic, wide, root / 'out')
            with self.assertRaisesRegex(ValueError, 'separate'):
                continuous.run(semantic, wide, semantic / 'unsafe-output')

    def test_missing_cache_blocks_without_partial_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, semantic, wide = fixture(root)
            (wide / 'maps' / 'val-lpips.npz').unlink()
            with contextlib.redirect_stdout(io.StringIO()):
                result = continuous.run(semantic, wide, root / 'out')
            self.assertEqual(result['status'], 'blocked')
            self.assertTrue((root / 'out' / 'coverage.csv').exists())
            self.assertFalse((root / 'out' / 'summary.csv').exists())


if __name__ == '__main__':
    unittest.main()
