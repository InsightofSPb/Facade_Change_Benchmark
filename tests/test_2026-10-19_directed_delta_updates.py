"""Controls for the state-confidence failure and the isolated difference-feature revision."""
import contextlib
import copy
import csv
import importlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

base = importlib.import_module('test_2026-10-19_continuous_semantic_updates')
runner = base.continuous


class DirectedDeltaTests(unittest.TestCase):
    def test_large_transition_outranks_high_confidence_persistence(self):
        p = base.pair()
        p['reference']['probabilities'][:, 0, :4] = [.1, .999, .125, .75]
        p['source']['probabilities'][:, 0, :4] = [.9, 1., .25, .875]
        model = dict(feature_design='directed_delta', coefficients=[2.], intercept=-3.)
        events, scores = runner.score_pair(p, 'semantic', model)
        self.assertGreater(scores[0, 0], scores[0, 1])
        self.assertEqual(scores[0, 2], scores[0, 3])  # same difference, different state confidence
        self.assertEqual(scores.dtype, np.float64)
        p['source'], p['reference'] = p['reference'], p['source']
        backwards, score_backwards = runner.score_pair(p, 'semantic', model)
        np.testing.assert_array_equal(backwards, events[::-1])
        np.testing.assert_array_equal(score_backwards, scores)

    def test_nonpositive_semantic_slope_is_rejected_without_sign_flip(self):
        x = np.linspace(-1, 1, 100)[:, None]
        with self.assertRaisesRegex(ValueError, 'positive semantic slope'):
            runner.fit_event_model(x, x[:, 0] < 0, np.ones(len(x)),
                                   ['p_to_minus_p_from'], 'directed_delta')
        with self.assertRaisesRegex(ValueError, 'positive semantic slope'):
            runner.score_pair(base.pair(), 'semantic', dict(feature_design='directed_delta',
                              coefficients=[-1.], intercept=0.))

    def test_heldout_inputs_cannot_affect_difference_fits(self):
        pairs = [base.pair('A'), base.pair('B'), base.pair('V', split='val'), base.pair('T', split='test')]
        with contextlib.redirect_stdout(io.StringIO()):
            before = runner.fit_bundle(pairs, excluded='B', feature_design='directed_delta')
        altered = copy.deepcopy(pairs)
        for p in altered[1:]:
            p['source']['probabilities'][:] = .7
            p['reference']['probabilities'][:] = .8
            p['source']['packed'][:] = 1
            p['labels'][:] = 1
            for method in runner.pilot.METHODS:
                p['maps'][method]['scores'][:] = 10
        with contextlib.redirect_stdout(io.StringIO()):
            after = runner.fit_bundle(altered, excluded='B', feature_design='directed_delta')
        self.assertEqual(before, after)
        self.assertEqual(before['models']['semantic']['features'], ['p_to_minus_p_from'])

    def test_independently_calibrated_semantic_model_matches_delta_ranking(self):
        pairs = [base.pair('A', positives=4), base.pair('B', positives=9), base.pair('V', positives=7, split='val')]
        rng = np.random.default_rng(20261019)
        for p in pairs:
            for key in ('reference', 'source'):
                probabilities = p[key]['probabilities']
                p[key]['probabilities'] = (probabilities + rng.uniform(-.07, .07, probabilities.shape)).astype(np.float32)
        with contextlib.redirect_stdout(io.StringIO()):
            bundle = runner.fit_bundle(pairs, feature_design='directed_delta')
        for p in pairs:
            results = []
            for variant in ('semantic_delta', 'semantic'):
                events, scores = runner.score_pair(p, variant, bundle['models'].get(variant))
                results.append(runner.evaluate(p, events, scores, bundle['thresholds'][variant]['threshold'],
                                               [{'name': 'CRACK'}, {'name': 'REPAIRS'}]))
            self.assertTrue(runner.check_parity(*results)['passed'])
        # The guard must detect a genuine mismatch, not just write a success field.
        changed = list(results[1])
        changed[1] = changed[1].copy()
        changed[1][0, 0] = ~changed[1][0, 0]
        self.assertFalse(runner.check_parity(results[0], changed)['passed'])

    def test_complete_readonly_run_records_all_fold_parity_and_design(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, semantic, wide = base.fixture(root)
            inputs = (prepared, semantic, wide, root / 'cache')
            before = base.old.hashes(inputs)
            with contextlib.redirect_stdout(io.StringIO()):
                status = runner.run(semantic, wide, root / 'out', feature_design='directed_delta')
            self.assertEqual(status['status'], 'completed', status.get('error'))
            self.assertTrue(status['parity_passed'])
            self.assertEqual(status['parity_pairs'], 8)
            self.assertEqual(before, base.old.hashes(inputs))
            config = runner.pilot.read_json(root / 'out' / 'resolved_config.json')
            self.assertEqual(config['schema'], runner.DELTA_SCHEMA)
            self.assertEqual(config['features'], list(runner.DELTA_FEATURES))
            with (root / 'out' / '2026-10-19_parity.csv').open() as stream:
                checks = list(csv.DictReader(stream))
            self.assertEqual(len(checks), 8)
            self.assertTrue(all(r['passed'] == 'True' for r in checks))
            self.assertEqual({r['split'] for r in checks}, {'train_fit', 'train_fusion_oof', 'val', 'test'})
            index = runner.pilot.read_json(root / 'out' / '2026-10-19_artifacts.json')
            for item in index['models']:
                model = runner.pilot.read_json(root / 'out' / item['path'])
                self.assertEqual(model['feature_design'], 'directed_delta')
                self.assertTrue(all(m['coefficients'][0] > 0 for m in model['models'].values()))
            for item in index['maps']:
                with np.load(root / 'out' / item['artifact_path'], allow_pickle=False) as data:
                    self.assertEqual(data['semantic__change_score'].dtype, np.float64)
                    np.testing.assert_array_equal(data['semantic__prediction'], data['semantic_delta__prediction'])


if __name__ == '__main__':
    unittest.main()
