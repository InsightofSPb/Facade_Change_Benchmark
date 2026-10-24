"""Saved-map evaluation: real bitset GT, honest coverage and matched history gates."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.real_eval import night_report
from scripts.real_eval.robustness_common import atomic_json, atomic_npz, read_json, sha256


def select(rows, **criteria):
    found = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    if len(found) != 1:
        raise AssertionError(f'{criteria}: expected one row, got {len(found)}')
    return found[0]


def fixture(base):
    prepared, run = base / 'prepared', base / 'run'
    prepared.mkdir()
    (run / 'units').mkdir(parents=True)
    (run / 'maps').mkdir()
    sequences, originals = [], {}
    methods = ['rgb_diff', 'geoscd', 'msdzip_abs']
    for building, split in [('A', 'train'), ('B', 'test')]:
        frames = []
        for index in range(3):
            packed = np.array([[1, 1], [1, 0]], np.uint16)
            if index:
                packed[0, index - 1] = 3
            rgb = np.full((2, 2, 3), 20 * index + (1 if building == 'A' else 2), np.uint8)
            arrays = dict(rgb=rgb, packed=packed, support=np.ones((2, 2), bool))
            relative = f'{building}-{index}.npz'
            checksum = atomic_npz(prepared / relative, arrays)
            frames.append(dict(frame=index, year=2010 + index, artifact_path=relative, artifact_sha256=checksum))
            originals[(building, index)] = arrays
        sequences.append(dict(sequence_id=building, building_id=building, split=split,
                              view_id=building, crop_xywh=[0, 0, 2, 2], frames=frames))
    calibration = []
    for method in ('rgb_diff', 'msdzip_abs'):
        for mode in ('previous', 'earliest', 'history'):
            calibration.append(dict(method=method, variant='current_train' if method == 'msdzip_abs' else 'frozen',
                                    mode=mode, target='earliest_state' if mode == 'earliest' else 'step_event',
                                    status='calibrated', selection_split='val', selection_domain='synthetic', selection_arm='clean',
                                    choices=[dict(fpr_target=.01, threshold=.5), dict(fpr_target=.05, threshold=.2)]))
    manifest = dict(schema_version=1, kind='real_temporal_prepared', source_run=str(base / 'original'),
                    snapshot=str(base / 'snapshot'), source_config={}, source_summary={'calibration': calibration},
                    categories=[dict(id=1, name='surface'), dict(id=2, name='damage'), dict(id=3, name='absent')],
                    sequences=sequences)
    atomic_json(prepared / 'manifest.json', manifest)
    contracts = {method: {'expected_metadata': {}} for method in methods}
    contracts['msdzip_abs']['expected_metadata'] = dict(buildings={'train': ['A']}, checkpoint_sha256='x',
                                                     reviewed_split={'mode': 'reviewed', 'building_assignments': {'A': 'train'}})
    atomic_json(run / 'run.json', dict(prepared_path=str(prepared), manifest_sha256=sha256(prepared / 'manifest.json'),
                                      methods=methods, history_methods=['msdzip_abs'], modes=list(night_report.MODES), contracts=contracts))
    for sequence in sequences:
        building = sequence['building_id']
        for frame in (1, 2):
            for mode in night_report.MODES:
                for method in ['msdzip_abs'] if mode == 'history' else methods:
                    key = f'{building}-{frame}-{mode}-{method}'
                    row = dict(unit_id=key, sequence_id=building, building_id=building, split=sequence['split'], frame=frame,
                               method=method, mode=mode, variant='current_train' if method == 'msdzip_abs' else 'frozen',
                               target='earliest_state' if mode == 'earliest' else 'step_event', status='completed')
                    old = originals[(building, 0 if mode == 'earliest' else frame - 1)]
                    new = originals[(building, frame)]
                    support = np.ones((2, 2), bool)
                    scores = np.ones((2, 2), np.float32)
                    if mode == 'history':
                        support[1, 0] = False
                        scores[~support] = np.nan
                        scores[0, 1] = .3
                    arrays = dict(reference=old['rgb'], source=new['rgb'], support=support, scores=scores)
                    if method == 'geoscd':
                        arrays['native_prediction'] = np.array([[True, True], [False, False]])
                    relative = f'maps/{key}.npz'
                    row['artifact_path'] = relative
                    row['artifact_sha256'] = atomic_npz(run / relative, arrays)
                    atomic_json(run / 'units' / f'{key}.json', row)
    return prepared, run


class RealTemporalReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.prepared, self.run = fixture(self.base)
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()

    def tearDown(self):
        self.quiet.__exit__(None, None, None)
        self.temporary.cleanup()

    def test_three_modes_targets_and_facade_exposure_are_distinct(self):
        with patch('subprocess.Popen', side_effect=AssertionError('Reporting must not infer')):
            result = night_report.report(self.run)
        previous = select(result['tables'], predictor='rgb_diff', mode='previous', point=.01,
                          split_scope='test', training_scope='all')
        earliest = select(result['tables'], predictor='rgb_diff', mode='earliest', point=.01,
                          split_scope='test', training_scope='all')
        self.assertEqual((previous['tp'], previous['fp'], previous['positive_pixels']), (3, 3, 3))
        self.assertEqual((earliest['tp'], earliest['fp'], earliest['positive_pixels']), (2, 4, 2))
        self.assertAlmostEqual(previous['precision'], .5)
        self.assertAlmostEqual(previous['f1'], (.5 + .8) / 2)
        unseen = select(result['tables'], predictor='msdzip_abs', mode='previous', point=.01,
                        split_scope='all_selected', training_scope='verified_unseen')
        seen = select(result['tables'], predictor='msdzip_abs', mode='previous', point=.01,
                      split_scope='all_selected', training_scope='seen')
        self.assertEqual((unseen['facades'], unseen['evaluated_pairs']), (1, 2))
        self.assertEqual((seen['facades'], seen['evaluated_pairs']), (1, 2))
        with (self.run / 'reports' / 'per_class.csv').open() as stream:
            import csv
            classes = list(csv.DictReader(stream))
        absent = select(classes, predictor='rgb_diff', mode='earliest', point='0.01', split_scope='test',
                        training_scope='all', class_name='absent', direction='change')
        self.assertEqual((absent['valid_pixels'], absent['f1']), ('0', ''))
        damage_loss = select(classes, predictor='rgb_diff', mode='previous', point='0.01', split_scope='test',
                             training_scope='all', class_name='damage', direction='loss')
        self.assertEqual((damage_loss['tp'], damage_loss['fn']), ('1', '0'))
        self.assertEqual(len(list((self.run / 'reports').iterdir())), 7)
        provenance = read_json(self.run / 'reports' / 'provenance.json')
        self.assertTrue(provenance['no_model_inference'])
        self.assertEqual(provenance['metrics_csv_units'], 'fractions')

    def test_fusion_and_history_deltas_use_exact_shared_support(self):
        result = night_report.report(self.run)
        comparison = select(result['comparisons'], comparison='fusion/rgb_diff&msdzip_abs_history',
                            split_scope='test', training_scope='all', point=.01, region='overall')
        self.assertEqual((comparison['evaluated_pairs'], comparison['valid_pixels']), (2, 4))
        # Each history comparison ignores [1,0], so the matched baseline FPR is
        # 1 in the first positive frame, absent in the second (no negatives).
        self.assertEqual(comparison['baseline_fpr'], 1.)
        self.assertEqual(comparison['candidate_fpr'], 0.)
        self.assertAlmostEqual(comparison['baseline_precision'], .75)
        self.assertEqual(comparison['candidate_precision'], 1.)
        self.assertAlmostEqual(comparison['delta_f1'], 0.)
        fusion = select(result['tables'], predictor='rgb_diff&msdzip_abs_history', mode='history', point=.01,
                        split_scope='all_selected', training_scope='verified_unseen')
        self.assertEqual((fusion['facades'], fusion['evaluated_pairs']), (1, 2))
        self.assertFalse(any(row['mode'] == 'earliest' and '&' in row['predictor'] for row in result['tables']))

    def test_or_and_and_compare_both_components_on_shared_support(self):
        # Complementary predictions: PREVIOUS marks [0,1], HISTORY [0,0].
        # A PREVIOUS false positive at [1,0] lies outside HISTORY support and
        # must be excluded from every component and combination comparison.
        for frame in (1, 2):
            unit = self.run / 'units' / f'B-{frame}-previous-rgb_diff.json'
            record = read_json(unit)
            path = self.run / record['artifact_path']
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            arrays['scores'] = np.array([[0., 1.], [1., 0.]], np.float32)
            record['artifact_sha256'] = atomic_npz(path, arrays)
            atomic_json(unit, record)
        with patch('subprocess.Popen', side_effect=AssertionError('Reporting must not infer')):
            result = night_report.report(self.run)
        union = 'rgb_diff|msdzip_abs_history'
        intersection = 'rgb_diff&msdzip_abs_history'
        for predictor, counts in ((union, (3, 1, 0, 0)), (intersection, (0, 0, 3, 1))):
            for suffix, baseline, base_counts in (
                    ('', 'rgb_diff', (1, 1, 2, 0)),
                    ('/vs_history', 'msdzip_abs_history', (2, 0, 1, 1))):
                row = select(result['comparisons'], comparison='fusion/' + predictor + suffix,
                             split_scope='test', training_scope='all', point=.01, region='overall')
                self.assertEqual(row['baseline'], baseline)
                self.assertEqual((row['evaluated_pairs'], row['valid_pixels']), (2, 4))
                self.assertEqual(tuple(row['baseline_' + name] for name in ('tp', 'fp', 'fn', 'tn')), base_counts)
                self.assertEqual(tuple(row['candidate_' + name] for name in ('tp', 'fp', 'fn', 'tn')), counts)
                self.assertEqual(row['baseline_valid_pixels'], row['candidate_valid_pixels'])
        vs_previous = select(result['comparisons'], comparison='fusion/' + union,
                             split_scope='test', training_scope='all', point=.01, region='overall')
        vs_history = select(result['comparisons'], comparison='fusion/' + union + '/vs_history',
                            split_scope='test', training_scope='all', point=.01, region='overall')
        self.assertAlmostEqual(vs_previous['delta_f1'], .5)
        self.assertAlmostEqual(vs_history['delta_f1'], 0.)
        self.assertEqual((vs_history['baseline_fpr'], vs_history['candidate_fpr']), (0., 1.))
        self.assertFalse(any(row['mode'] == 'earliest' and '|' in row['predictor'] for row in result['tables']))
        for training_scope, split_scope in (('seen', 'train'), ('verified_unseen', 'test')):
            for predictor in (union, intersection):
                row = select(result['tables'], predictor=predictor, mode='history', point=.01,
                             split_scope=split_scope, training_scope=training_scope)
                self.assertEqual((row['facades'], row['evaluated_pairs']), (1, 2))
        provenance = read_json(self.run / 'reports' / 'provenance.json')
        self.assertEqual(provenance['fusion_operations'], {'&': 'AND', '|': 'OR'})
        self.assertEqual(provenance['fusion_comparison_bases'], ['previous_component', 'history_component'])

    def test_or_class_comparisons_keep_joint_training_exposure_and_missing_cohort(self):
        # One native result is missing. The history component comparison must
        # use the same remaining pair, not its full two-pair standalone cohort.
        (self.run / 'units' / 'B-1-previous-geoscd.json').unlink()
        result = night_report.report(self.run)
        for operator in ('&', '|'):
            for suffix in ('', '/vs_history'):
                row = select(result['comparisons'],
                             comparison=f'fusion/geoscd{operator}msdzip_abs_history{suffix}',
                             split_scope='test', training_scope='verified_unseen', point=.01,
                             region='class', class_name='damage', direction='change')
                self.assertEqual((row['facades'], row['evaluated_pairs']), (1, 1))
                self.assertEqual((row['baseline_valid_pixels'], row['candidate_valid_pixels']), (2, 2))
                self.assertEqual(row['baseline_positive_pixels'], row['candidate_positive_pixels'])
                seen = select(result['comparisons'],
                              comparison=f'fusion/geoscd{operator}msdzip_abs_history{suffix}',
                              split_scope='train', training_scope='seen', point=.01,
                              region='class', class_name='damage', direction='change')
                self.assertEqual((seen['facades'], seen['evaluated_pairs']), (1, 2))
        with (self.run / 'reports' / 'per_class.csv').open() as stream:
            import csv
            classes = list(csv.DictReader(stream))
        union = select(classes, predictor='geoscd|msdzip_abs_history', mode='history', point='0.01',
                       split_scope='test', training_scope='verified_unseen', class_name='damage', direction='change')
        self.assertEqual((union['evaluated_pairs'], union['tp'], union['fn']), ('1', '2', '0'))

    def test_class_deltas_use_history_support_for_both_masks(self):
        # Make this a surface-to-damage transition rather than merely adding
        # the damage bit. Both membership directions then belong to the
        # surface class slice; its stable background remains evaluable.
        manifest_path = self.prepared / 'manifest.json'
        manifest = read_json(manifest_path)
        for sequence in manifest['sequences']:
            for frame in sequence['frames']:
                path = self.prepared / frame['artifact_path']
                with np.load(path, allow_pickle=False) as archive:
                    arrays = {key: archive[key].copy() for key in archive.files}
                arrays['packed'][arrays['packed'] == 3] = 2
                frame['artifact_sha256'] = atomic_npz(path, arrays)
        atomic_json(manifest_path, manifest)
        run = read_json(self.run / 'run.json')
        run['manifest_sha256'] = sha256(manifest_path)
        atomic_json(self.run / 'run.json', run)
        result = night_report.report(self.run)
        row = select(result['comparisons'], comparison='fusion/rgb_diff&msdzip_abs_history',
                     split_scope='test', training_scope='all', point=.01,
                     region='class', class_name='surface', direction='change')
        # The ordinary PREVIOUS mask has six known pixel observations. HISTORY
        # excludes one stable pixel per frame, so BOTH compared class masks have
        # four observations, not six versus four.
        self.assertEqual((row['baseline_valid_pixels'], row['candidate_valid_pixels']), (4, 4))
        self.assertEqual((row['baseline_positive_pixels'], row['candidate_positive_pixels']), (3, 3))
        self.assertEqual((row['baseline_tp'], row['baseline_fp'], row['baseline_fn'], row['baseline_tn']), (3, 1, 0, 0))
        self.assertEqual((row['candidate_tp'], row['candidate_fp'], row['candidate_fn'], row['candidate_tn']), (2, 0, 1, 1))
        self.assertAlmostEqual(row['delta_precision'], .25)
        self.assertAlmostEqual(row['delta_recall'], -.25)
        self.assertAlmostEqual(row['delta_f1'], 0.)
        self.assertEqual(row['delta_fpr'], -1.)
        with (self.run / 'reports' / 'comparisons.csv').open() as stream:
            import csv
            csv_rows = list(csv.DictReader(stream))
        self.assertEqual({value['region'] for value in csv_rows}, {'overall', 'class'})

    def test_failure_and_missing_calibration_are_never_zero_predictions(self):
        failed_path = self.run / 'units' / 'B-1-previous-geoscd.json'
        failed = read_json(failed_path)
        failed.update(status='failed', error='No finite valid correspondences')
        atomic_json(failed_path, failed)
        manifest_path = self.prepared / 'manifest.json'
        manifest = read_json(manifest_path)
        manifest['source_summary']['calibration'] = [row for row in manifest['source_summary']['calibration']
                                                      if not (row['method'] == 'msdzip_abs' and row['mode'] == 'history')]
        atomic_json(manifest_path, manifest)
        run = read_json(self.run / 'run.json')
        run['manifest_sha256'] = sha256(manifest_path)
        atomic_json(self.run / 'run.json', run)
        result = night_report.report(self.run)
        available = select(result['tables'], predictor='geoscd', mode='previous', point='native',
                           split_scope='test', training_scope='all')
        self.assertEqual((available['evaluated_pairs'], available['tp'], available['fp']), (1, 2, 0))
        failed_coverage = select(result['coverage'], split='test', method='geoscd', mode='previous', point='native')
        self.assertEqual((failed_coverage['completed_pairs'], failed_coverage['failed_pairs']), (1, 1))
        missing = select(result['coverage'], split='test', method='msdzip_abs', mode='history', point=.01)
        self.assertEqual((missing['status'], missing['evaluable_pairs']), ('missing_calibration', 0))
        self.assertFalse(any(row['mode'] == 'history' for row in result['tables']))

    def test_unknown_training_is_not_inferred_unseen_from_test(self):
        run = read_json(self.run / 'run.json')
        run['contracts']['msdzip_abs']['expected_metadata'] = {}
        atomic_json(self.run / 'run.json', run)
        result = night_report.report(self.run)
        unknown = select(result['tables'], predictor='msdzip_abs', mode='previous', point=.01,
                         split_scope='test', training_scope='unknown')
        self.assertEqual(unknown['unknown_facades'], 1)
        self.assertFalse(any(row['predictor'] == 'msdzip_abs' and row['training_scope'] == 'verified_unseen'
                             for row in result['tables']))

    def test_regeneration_owned_output_and_integrity(self):
        night_report.report(self.run)
        night_report.report(self.run)
        other = self.base / 'unowned'
        other.mkdir()
        (other / 'keep').write_text('keep')
        with self.assertRaises(FileExistsError):
            night_report.report(self.run, other)
        self.assertEqual((other / 'keep').read_text(), 'keep')
        path = self.run / 'maps' / 'A-1-previous-rgb_diff.npz'
        path.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            night_report.report(self.run)

    def test_actual_runner_contract_round_trip(self):
        from scripts.real_eval import night_run
        from tests.test_real_temporal_run import Factory, fixture as runner_fixture
        integration = self.base / 'integration'
        prepared, _ = runner_fixture(integration)
        path = prepared / 'manifest.json'
        manifest = read_json(path)
        manifest['source_summary']['calibration'] = [
            dict(method=method, variant='frozen', mode=mode,
                 target='earliest_state' if mode == 'earliest' else 'step_event',
                 status='calibrated', selection_split='val', selection_domain='synthetic', selection_arm='clean',
                 choices=[dict(fpr_target=.01, threshold=.1)])
            for method in ('rgb_diff', 'dinov2') for mode in night_report.MODES]
        atomic_json(path, manifest)
        out = integration / 'inference'
        with patch.object(night_run, '_snapshot_gpu'):
            inference = night_run.run(prepared, out, methods=('rgb_diff', 'dinov2', 'geoscd'),
                                       pool_factory=Factory())
        self.assertEqual(inference['status'], 'completed')
        result = night_report.report(out, budgets=(.01,))
        self.assertEqual({row['mode'] for row in result['tables']}, set(night_report.MODES))
        row = select(result['coverage'], split='test', method='dinov2', mode='history', point=.01)
        self.assertEqual((row['completed_pairs'], row['evaluable_pairs']), (3, 3))
        self.assertTrue(any(row['predictor'] == 'geoscd&dinov2_history' for row in result['tables']))
        self.assertTrue(all(row['training_scope'] != 'unknown' for row in result['tables']))


if __name__ == '__main__':
    unittest.main()
