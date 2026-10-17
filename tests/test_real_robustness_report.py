"""Frozen-threshold robustness reporting from actual saved fake-runner outputs."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.real_eval import real_gt_metrics as metrics
from scripts.real_eval import robustness_report as report
from scripts.real_eval import robustness_run as runner
from scripts.real_eval.robustness_common import atomic_json, atomic_npz, read_json, sha256
from tests.test_real_robustness_run import FakeFactory, fixture as run_fixture


METHODS = ('rgb_diff', 'ssim', 'geoscd')


def fixture(root):
    source, prepared, selection = run_fixture(root, methods=METHODS)
    path = prepared / 'manifest.json'
    manifest = read_json(path)
    manifest['categories'] = [dict(id=1, name='surface'), dict(id=2, name='damage'),
                              dict(id=3, name='absent')]
    manifest['boundary_tolerance'] = 0
    manifest['source_summary'] = dict(source_run=str(source), calibration=[
        dict(method=method, variant='frozen', mode='previous', target='step_event',
             status='calibrated', selection_split='val', selection_domain='synthetic',
             selection_arm='clean', choices=[dict(fpr_target=.01, threshold=.5),
                                              dict(fpr_target=.05, threshold=.2)])
        for method in METHODS if method != 'geoscd'])
    for row in manifest['pairs']:
        artifact = prepared / row['artifact_path']
        with np.load(artifact, allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in data.files}
        # Real semantic bitsets agree exactly with the existing binary GT.
        arrays['source_bits'][:] = 1
        arrays['source_bits'][0, 0] = 3
        atomic_npz(artifact, arrays)
        row['artifact_sha256'] = sha256(artifact)
    atomic_json(path, manifest)
    chosen = read_json(selection)
    chosen['manifest_sha256'] = sha256(path)
    atomic_json(selection, chosen)
    return source, prepared, selection


def select(rows, **criteria):
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    if len(matches) != 1:
        raise AssertionError(f'Expected exactly one row for {criteria}; got {len(matches)}')
    return matches[0]


def csv_rows(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


class RobustnessReportIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source, self.prepared, self.selection = fixture(self.root)
        self.inference = self.root / 'inference'
        self.silence = contextlib.redirect_stdout(io.StringIO())
        self.silence.__enter__()
        self.gpu = patch.object(runner, '_snapshot_gpu')
        self.gpu.start()

    def tearDown(self):
        self.gpu.stop()
        self.silence.__exit__(None, None, None)
        self.temporary.cleanup()

    def execute(self, fail=None):
        return runner.run(self.prepared, self.selection, self.inference,
                          methods=METHODS, pool_factory=FakeFactory(fail=fail))

    def test_saved_runner_results_native_scores_fusions_and_frozen_h0(self):
        self.assertEqual(self.execute()['completed'], 30)
        source_hashes = {str(p): sha256(p) for p in self.source.rglob('*') if p.is_file()}
        output = self.root / 'report'
        # Reporting must consume the cache and must not create an inference worker.
        with patch('subprocess.Popen', side_effect=AssertionError('No report inference')):
            summary = report.report(self.inference, output)
        self.assertEqual(summary['calibration_scope'], 'synthetic_VAL_H0_frozen')
        self.assertEqual(summary['inference_status'], 'completed')
        self.assertEqual(len(summary['coverage']), 30)
        rows = summary['aggregate']
        strict = select(rows, cohort='available', hypothesis='H1', condition='original',
                        predictor='rgb_diff', point=.01)
        loose = select(rows, cohort='available', hypothesis='H1', condition='original',
                       predictor='rgb_diff', point=.05)
        self.assertEqual((strict['tp'], strict['fp'], strict['fn'], strict['tn']), (0, 0, 2, 20))
        self.assertEqual((loose['tp'], loose['fp'], loose['fn'], loose['tn']), (2, 20, 0, 0))
        self.assertEqual((strict['f1'], strict['recall'], strict['fpr']), (0., 0., 0.))
        self.assertAlmostEqual(loose['precision'], 1 / 11)
        self.assertAlmostEqual(loose['f1'], 1 / 6)
        self.assertEqual(loose['recall'], 1.)
        h0 = select(rows, cohort='available', hypothesis='H0', condition='exposure',
                    predictor='rgb_diff', point=.05)
        self.assertEqual((h0['positive_pairs'], h0['tp'], h0['fp'], h0['fn'], h0['tn']), (0, 0, 22, 0, 0))
        self.assertEqual(h0['fpr'], 1.)
        self.assertIsNone(h0['f1'], 'H0-only frames must not become false zero-F1 observations')
        native = select(rows, cohort='available', hypothesis='H1', condition='shadow',
                        predictor='geoscd', point='native')
        self.assertEqual((native['tp'], native['fp'], native['fn'], native['tn']), (2, 10, 0, 10))
        self.assertAlmostEqual(native['f1'], 2 / 7)
        baseline = select(rows, cohort='fusion/geoscd&ssim', hypothesis='H1', condition='shadow',
                          predictor='geoscd', point=.01)
        gate = select(rows, cohort='fusion/geoscd&ssim', hypothesis='H1', condition='shadow',
                      predictor='geoscd&ssim', point=.01)
        self.assertEqual((baseline['pairs'], gate['pairs'], baseline['valid_pixels'], gate['valid_pixels']),
                         (2, 2, 22, 22))
        self.assertEqual((gate['tp'], gate['fp'], gate['fn'], gate['tn']), (0, 0, 2, 20))
        self.assertEqual(baseline['f1'], native['f1'])
        for row in summary['deltas']:
            self.assertEqual(row['delta_f1'], 0.)
            self.assertEqual(row['delta_fpr'], 0.)
        frozen = read_json(output / 'thresholds.json')
        self.assertEqual({row['threshold'] for row in frozen}, {.2, .5})
        self.assertEqual({row['scope'] for row in frozen}, {'synthetic_VAL_H0_frozen'})
        self.assertEqual(source_hashes, {str(p): sha256(p) for p in self.source.rglob('*') if p.is_file()})
        classes = csv_rows(output / 'class_aggregate.csv')
        absent = select(classes, cohort='available', hypothesis='H1', condition='original',
                        predictor='geoscd', point='native', class_name='absent', direction='change')
        self.assertEqual((absent['pairs'], absent['facades'], absent['valid_pixels']), ('0', '0', '0'))
        self.assertEqual(absent['f1'], '')
        damage = select(classes, cohort='available', hypothesis='H1', condition='original',
                        predictor='geoscd', point='native', class_name='damage', direction='gain')
        self.assertEqual((damage['tp'], damage['fp'], damage['fn']), ('2', '0', '0'))
        self.assertEqual(float(damage['f1']), 1.)

    def test_failure_changes_coverage_but_not_available_or_matched_cohorts_silently(self):
        result = self.execute(fail='geoscd')
        self.assertEqual((result['completed'], result['failed']), (29, 1))
        summary = report.report(self.inference, self.root / 'report')
        self.assertEqual(summary['inference_status'], 'completed_with_failures')
        failed = [row for row in summary['coverage'] if row['status'] == 'failed']
        self.assertEqual(len(failed), 1)
        self.assertEqual((failed[0]['pair_id'], failed[0]['hypothesis'], failed[0]['condition']),
                         ('pair-1', 'H1', 'original'))
        rows = summary['aggregate']
        available = select(rows, cohort='available', hypothesis='H1', condition='original',
                           predictor='rgb_diff', point=.01)
        common = select(rows, cohort='common_methods', hypothesis='H1', condition='original',
                        predictor='rgb_diff', point=.01)
        self.assertEqual((available['pairs'], common['pairs']), (2, 1))
        self.assertEqual((available['fn'], common['fn']), (2, 1))
        delta = select(summary['deltas'], condition='exposure', predictor='geoscd', point='native')
        self.assertEqual(delta['pairs'], 1)
        self.assertEqual(delta['delta_f1'], 0.)
        standalone = select(rows, cohort='available', hypothesis='H1', condition='exposure',
                            predictor='geoscd', point='native')
        self.assertEqual(standalone['pairs'], 2)
        for side in ('original', 'exposure'):
            matched = select(rows, cohort='matched_original', comparison='exposure', hypothesis='H1',
                             condition=side, predictor='geoscd&ssim', point=.05)
            self.assertEqual((matched['pairs'], matched['tp'], matched['fp']), (1, 1, 5))

    def test_matched_delta_is_perturbed_minus_original_with_same_threshold(self):
        class ChangingFactory(FakeFactory):
            def __call__(self, **kwargs):
                pool = super().__call__(**kwargs)
                original_score = pool.score

                def score(method, options, reference, source, support):
                    output = original_score(method, options, reference, source, support)
                    if method == 'ssim' and source[0, 0, 0] < 80:
                        output['scores'][support] = .75
                    return output

                pool.score = score
                return pool

        runner.run(self.prepared, self.selection, self.inference,
                   methods=METHODS, pool_factory=ChangingFactory())
        summary = report.report(self.inference, self.root / 'delta-report')
        delta = select(summary['deltas'], condition='shadow', predictor='ssim', point=.01)
        self.assertEqual((delta['pairs'], delta['facades']), (2, 2))
        self.assertEqual(delta['original_f1'], 0.)
        self.assertAlmostEqual(delta['perturbed_f1'], 1 / 6)
        self.assertAlmostEqual(delta['delta_f1'], 1 / 6)
        self.assertEqual((delta['original_fpr'], delta['perturbed_fpr'], delta['delta_fpr']), (0., 1., 1.))
        fusion = select(summary['deltas'], condition='shadow', predictor='geoscd&ssim', point=.01)
        self.assertAlmostEqual(fusion['delta_f1'], 2 / 7)
        self.assertAlmostEqual(fusion['delta_fpr'], .5)

    def test_real_lofo_uses_four_part_group_and_original_heldout_thresholds(self):
        self.execute()
        metrics_run, calibration = self.root / 'gt', self.root / 'lofo'
        metrics_run.mkdir()
        calibration.mkdir()
        atomic_json(metrics_run / 'provenance.json', dict(source_run=str(self.source), boundary_tolerance=0))
        atomic_json(calibration / 'provenance.json', dict(metrics_run=str(metrics_run),
                    input_sha256={str(metrics_run / 'provenance.json'): sha256(metrics_run / 'provenance.json')}))
        thresholds = [dict(family_kind='standalone', group=f'{method}:frozen:previous:step_event',
                           heldout_building=facade, goal=goal, status='calibrated', threshold=.25,
                           train_buildings=[other], operator='>')
                      for method in ('rgb_diff', 'ssim') for goal in (.01, .05)
                      for facade, other in (('facade-1', 'facade-2'), ('facade-2', 'facade-1'))]
        atomic_json(calibration / 'thresholds.json', thresholds)
        summary = report.report(self.inference, self.root / 'lofo-report', calibration_run=calibration)
        self.assertEqual(summary['calibration_scope'], 'real_VAL_standalone_LOFO_frozen')
        selected = select(summary['aggregate'], cohort='available', hypothesis='H0', condition='shadow',
                          predictor='rgb_diff', point=.05)
        self.assertEqual((selected['fp'], selected['tn']), (0, 22), 'Threshold ties must use >, not >=')
        loaded = report.Thresholds(read_json(self.prepared / 'manifest.json'), calibration)
        loaded.rows[0]['train_buildings'].append(loaded.rows[0]['heldout_building'])
        with self.assertRaisesRegex(ValueError, 'held-out'):
            loaded.get('rgb_diff', 'frozen', 'facade-1', .01)


class RobustnessAggregateTests(unittest.TestCase):
    def test_frames_then_facades_not_pooled_pixels_or_all_frames(self):
        header = dict(cohort='available', comparison='', hypothesis='H1', condition='original',
                      predictor='fixture', point=.01)
        rows = []
        examples = [('A', [True, False], [1, 0]),
                    ('A', [False, False], [1, 0]),
                    ('A', [True, False], [0, 0]),
                    ('B', [False] * 100, [1] + [0] * 99)]
        for building, prediction, truth in examples:
            counts = metrics.evaluate_mask(np.array(prediction, bool), np.array(truth, np.uint8))
            counts['marked_fraction'] = (counts['tp'] + counts['fp']) / counts['valid_pixels']
            rows.append({**header, 'building_id': building, **counts})
        aggregates, facades = report.aggregate(rows)
        self.assertEqual(len(aggregates), 1)
        row = aggregates[0]
        self.assertAlmostEqual(row['f1'], .25)  # A: (1 + 0)/2; B: 0; equal facade mean.
        self.assertAlmostEqual(row['recall'], .25)
        self.assertAlmostEqual(row['fpr'], 1 / 12)  # A: (0 + 0 + .5)/3; B: 0.
        self.assertEqual((row['pairs'], row['positive_pairs'], row['facades']), (4, 3, 2))
        self.assertEqual((row['tp'], row['fp'], row['fn'], row['tn']), (1, 1, 2, 102))
        self.assertAlmostEqual(select(facades, building_id='A')['f1'], .5)


if __name__ == '__main__':
    unittest.main()
