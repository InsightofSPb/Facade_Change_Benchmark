"""CPU-only, saved-map VAL calibration and matched pairwise report contract."""
import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.real_eval.robustness_common import atomic_json, atomic_npz, read_json, sha256
from scripts.real_eval.night_common import unit_id


def saved_night_fixture(base):
    """Three calibration facades; TRAIN/TEST are deliberately unreadable pixels."""
    prepared, run = base / 'prepared', base / 'night'
    prepared.mkdir()
    (run / 'units').mkdir(parents=True)
    (run / 'maps').mkdir()
    methods = ['rgb_diff', 'ssim', 'anychange', 'geoscd', 'msdzip_abs']
    modes = ['previous', 'earliest', 'history']
    sequences, images = [], {}
    shape = (16, 16)
    for number, (building, split) in enumerate((('A', 'val'), ('B', 'val'), ('C', 'val'),
                                                ('TRAIN', 'train'), ('TEST', 'test'))):
        frames = []
        for index in range(3):
            bits = np.ones(shape, np.uint16)
            bits[-1, -4:] = 0  # Unknown annotation remains ignored.
            if index:
                bits[index:index + 2, 2:6] = 3
            rgb = np.full((*shape, 3), number * 20 + index + 1, np.uint8)
            arrays = dict(rgb=rgb, packed=bits, support=np.ones(shape, bool))
            relative = f'{building}-{index}.npz'
            checksum = atomic_npz(prepared / relative, arrays)
            frames.append(dict(frame=index, year=2000 + index, artifact_path=relative,
                               artifact_sha256=checksum))
            images[building, index] = arrays
        sequences.append(dict(sequence_id=building, building_id=building, split=split,
                              view_id=building, crop_xywh=[0, 0, 16, 16], frames=frames))
    calibration = []
    for method in ('rgb_diff', 'ssim', 'msdzip_abs'):
        for mode in modes:
            calibration.append(dict(method=method,
                                    variant='current_train' if method == 'msdzip_abs' else 'frozen',
                                    mode=mode, target='earliest_state' if mode == 'earliest' else 'step_event',
                                    status='calibrated', selection_split='val', selection_domain='synthetic',
                                    selection_arm='clean', choices=[dict(fpr_target=.01, threshold=.99),
                                                                   dict(fpr_target=.05, threshold=.95)]))
    manifest = dict(schema_version=1, kind='real_temporal_prepared', source_run=str(base / 'original'),
                    snapshot=str(base / 'snapshot'), source_config={},
                    source_summary={'calibration': calibration},
                    categories=[dict(id=1, name='surface'), dict(id=2, name='damage'),
                                dict(id=3, name='absent')], sequences=sequences)
    atomic_json(prepared / 'manifest.json', manifest)
    contracts = {method: {'expected_metadata': {}} for method in methods}
    contracts['msdzip_abs']['expected_metadata'] = dict(
        buildings={'train': ['TRAIN']}, checkpoint_sha256='synthetic-checkpoint',
        reviewed_split={'mode': 'reviewed', 'building_assignments': {'TRAIN': 'train'}})
    atomic_json(run / 'run.json', dict(prepared_path=str(prepared),
                                      manifest_sha256=sha256(prepared / 'manifest.json'),
                                      methods=methods, history_methods=['msdzip_abs'], modes=modes,
                                      contracts=contracts, status='completed'))
    for sequence in sequences:
        building = sequence['building_id']
        for frame in (1, 2):
            for mode in modes:
                for method in ['msdzip_abs'] if mode == 'history' else methods:
                    key = f'{building}-{frame}-{mode}-{method}'
                    identifier = unit_id(method, mode, building, frame)
                    record = dict(unit_id=identifier, sequence_id=building, building_id=building,
                                  split=sequence['split'], frame=frame, method=method, mode=mode,
                                  variant='current_train' if method == 'msdzip_abs' else 'frozen',
                                  target='earliest_state' if mode == 'earliest' else 'step_event',
                                  status='completed', origin='inference', elapsed_seconds=.02,
                                  scorer_seconds=[.01], initialization_seconds=.005, inference_calls=1)
                    old = images[building, 0 if mode == 'earliest' else frame - 1]
                    new = images[building, frame]
                    support = np.ones(shape, bool)
                    if method == 'msdzip_abs' and mode == 'history':
                        support[:, 0] = False
                    offset = methods.index(method) * 17 + ord(building[0]) + frame
                    scores = ((np.arange(256).reshape(shape) + offset) % 251 / 251).astype(np.float32)
                    change = (old['packed'] != new['packed']) & (old['packed'] != 0) & (new['packed'] != 0)
                    scores[change] += 1.
                    scores[~support] = np.nan
                    arrays = dict(reference=old['rgb'], source=new['rgb'], support=support, scores=scores)
                    if method in ('anychange', 'geoscd'):
                        native = change.copy()
                        native[0, 1:3 if method == 'geoscd' else 2] = True
                        arrays['native_prediction'] = native
                    relative = f'maps/{key}.npz'
                    record['artifact_path'] = relative
                    record['artifact_sha256'] = atomic_npz(run / relative, arrays)
                    atomic_json(run / 'units' / f'{identifier}.json', record)
    # Reading the manifest/records of other splits is fine; loading their pixel
    # artifacts during VAL analysis is not. Remove those blobs to enforce it.
    for path in list(prepared.glob('TRAIN-*.npz')) + list(prepared.glob('TEST-*.npz')):
        path.unlink()
    for path in list((run / 'maps').glob('TRAIN-*.npz')) + list((run / 'maps').glob('TEST-*.npz')):
        path.unlink()
    return prepared, run


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def hashes(root):
    return {str(path.relative_to(root)): sha256(path) for path in root.rglob('*') if path.is_file()}


class NightCalibrationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.prepared, self.run_path = saved_night_fixture(self.base)
        self.out = self.base / 'calibration'
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()

    def tearDown(self):
        self.quiet.__exit__(None, None, None)
        self.temporary.cleanup()

    def run_analysis(self, **kwargs):
        from scripts.real_eval import night_calibration
        # A Git metadata lookup is allowed, model/worker process launches are
        # not. Return no Git metadata and prohibit Popen for everything else.
        def git_only(command, *args, **options):
            if not isinstance(command, (list, tuple)) or not command or command[0] != 'git':
                raise AssertionError(f'Unexpected external command: {command!r}')
            return subprocess.CompletedProcess(command, 0, '', '')
        with patch('subprocess.run', side_effect=git_only), \
                patch('platform.platform', return_value='synthetic-test-platform'), \
                patch('subprocess.Popen', side_effect=AssertionError('Saved-map analysis cannot launch workers')):
            return night_calibration.run(self.run_path, self.out, **kwargs)

    def test_val_only_cpu_report_has_all_pairs_modes_and_holdout_audit(self):
        before = hashes(self.run_path), hashes(self.prepared)
        result = self.run_analysis()
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(before, (hashes(self.run_path), hashes(self.prepared)))
        expected = ('tables.csv', 'per_pair.csv', 'per_facade.csv', 'per_class.csv', 'comparisons.csv',
                    'comparison_per_facade.csv', 'calibration_audit.csv', 'coverage.csv', 'timing.csv',
                    'report.html', 'summary.txt', 'run.json', 'provenance.json')
        for name in expected:
            self.assertTrue((self.out / name).is_file(), name)
        rows = read_csv(self.out / 'per_pair.csv')
        self.assertTrue(rows)
        self.assertEqual({row['split'] for row in rows}, {'val'})
        self.assertEqual({row['building_id'] for row in rows}, {'A', 'B', 'C'})
        for row in rows:
            self.assertEqual(row['heldout_building'], row['building_id'])
        audit = read_csv(self.out / 'calibration_audit.csv')
        self.assertTrue(audit)
        for row in audit:
            training = json.loads(row['train_buildings'])
            self.assertNotIn(row['heldout_building'], training)
            self.assertTrue(set(training) <= {'A', 'B', 'C'})
            if row['status'] == 'calibrated':
                self.assertLessEqual(float(row['achieved_train_fpr']), float(row['point']) + 1e-12)
        families = {row['family'] for row in rows}
        self.assertEqual(len(families), 21)  # Six channels plus all fifteen pairs.
        for left, right in (('rgb_diff', 'ssim'), ('anychange', 'geoscd'), ('ssim', 'msdzip_abs')):
            self.assertTrue(any(left in family and right in family for family in families), (left, right))
        self.assertTrue(any('history' in family and 'previous' in family for family in families))
        # EARLIEST measures another target; it must never be mixed with HISTORY.
        self.assertFalse(any('earliest' in family and 'history' in family for family in families))
        self.assertTrue(any(row['role'] == 'and' for row in rows))
        self.assertTrue(any(row['role'] == 'or' for row in rows))
        comparisons = read_csv(self.out / 'comparisons.csv')
        self.assertEqual({row['comparison_kind'] for row in comparisons}, {'fusion', 'previous_history'})
        classes = read_csv(self.out / 'per_class.csv')
        self.assertTrue(classes)
        absent = [row for row in classes if row['class_name'] == 'absent']
        self.assertTrue(absent)
        self.assertTrue(all(row['valid_pixels'] == '0' and row['f1'] == '' for row in absent))
        self.assertTrue(read_json(self.out / 'provenance.json')['no_model_inference'])

    def test_missing_map_preserves_matching_and_common_support(self):
        (self.run_path / 'units' / (unit_id('ssim', 'previous', 'A', 1) + '.json')).unlink()
        self.run_analysis()
        rows = read_csv(self.out / 'per_pair.csv')
        grouped = {}
        for row in rows:
            grouped.setdefault((row['family'], row['point'], row['sequence_id'], row['frame']), []).append(row)
        for records in grouped.values():
            self.assertEqual(len({row['support_sha256'] for row in records}), 1)
            self.assertEqual(len({row['valid_pixels'] for row in records}), 1)
        affected = [row for row in rows if 'ssim' in row['family'] and 'previous' in row['family']]
        self.assertTrue(affected)
        self.assertFalse(any(row['sequence_id'] == 'A' and row['frame'] == '1' for row in affected))
        histories = [row for row in rows if 'history' in row['family'] and row['role'] in ('left', 'right', 'and', 'or')]
        self.assertTrue(histories)
        self.assertTrue(all(int(row['valid_pixels']) <= 236 for row in histories))
        coverage = read_csv(self.out / 'coverage.csv')
        self.assertTrue(any('ssim' in row['family'] and int(row['input_completed_pairs']) < int(row['expected_pairs'])
                            for row in coverage))

    def test_resume_reuses_family_cache_but_revalidates_sources(self):
        from scripts.real_eval import night_calibration
        self.run_analysis(methods=('rgb_diff', 'ssim'))
        first = (self.out / 'tables.csv').read_bytes()
        with patch.object(night_calibration, '_evaluate_family', side_effect=AssertionError('Recomputed completed family')):
            self.run_analysis(methods=('rgb_diff', 'ssim'), resume=True)
        self.assertEqual(first, (self.out / 'tables.csv').read_bytes())
        path = self.run_path / 'maps' / 'A-1-previous-rgb_diff.npz'
        with path.open('ab') as stream:
            stream.write(b'changed source')
        with self.assertRaises((ValueError, RuntimeError)):
            self.run_analysis(methods=('rgb_diff', 'ssim'), resume=True)

    def test_tampered_cache_is_rejected(self):
        self.run_analysis(methods=('rgb_diff', 'ssim'))
        caches = sorted(path for path in (self.out / 'families').glob('*.json')
                        if not path.name.endswith('.done.json'))
        self.assertTrue(caches)
        with caches[0].open('a') as stream:
            stream.write('\n ')
        with self.assertRaises((ValueError, RuntimeError)):
            self.run_analysis(methods=('rgb_diff', 'ssim'), resume=True)

    def test_existing_output_and_source_overlap_are_rejected(self):
        self.run_analysis(methods=('rgb_diff',))
        with self.assertRaises(FileExistsError):
            self.run_analysis(methods=('rgb_diff',))
        original = hashes(self.run_path), hashes(self.prepared)
        for destination in (self.run_path, self.prepared, self.base):
            self.out = destination
            with self.assertRaises((ValueError, FileExistsError)):
                self.run_analysis(methods=('rgb_diff',))
        self.assertEqual(original, (hashes(self.run_path), hashes(self.prepared)))


if __name__ == '__main__':
    unittest.main()
