"""Night calibration reads only hash-verified, shared known VAL pixels."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.real_eval import night_calibration_data as data
from scripts.real_eval.night_common import unit_id
from scripts.real_eval.robustness_common import atomic_json, atomic_npz, read_json, sha256


def fixture(base):
    prepared, run = base / 'prepared', base / 'run'
    prepared.mkdir()
    (run / 'units').mkdir(parents=True)
    (run / 'maps').mkdir()
    methods, history, modes = ['rgb_diff', 'geoscd', 'msdzip_abs'], ['msdzip_abs'], ['previous', 'earliest', 'history']
    sequences, originals = [], {}
    for building, split in (('A', 'val'), ('B', 'val'), ('T', 'test')):
        frames = []
        for index in range(3):
            packed = np.array([[1, 1, 1], [1, 1, 0]], np.uint16)
            if index:
                packed[0, index - 1] = 3
            arrays = dict(rgb=np.full((2, 3, 3), 20 * index + ord(building), np.uint8),
                          support=np.ones((2, 3), bool), packed=packed)
            path = f'{building}-{index}.npz'
            frames.append(dict(frame=index, year=2010 + index, artifact_path=path,
                               artifact_sha256=atomic_npz(prepared / path, arrays)))
            originals[building, index] = arrays
        sequences.append(dict(sequence_id=building, building_id=building, split=split, frames=frames))
    manifest = dict(schema_version=1, kind='real_temporal_prepared', source_run=str(base / 'source'),
                    categories=[dict(id=1, name='surface'), dict(id=2, name='damage')], sequences=sequences)
    atomic_json(prepared / 'manifest.json', manifest)
    atomic_json(run / 'run.json', dict(schema_version=1, kind='real_temporal_run', status='completed',
                                      prepared_path=str(prepared), manifest_sha256=sha256(prepared / 'manifest.json'),
                                      methods=methods, history_methods=history, modes=modes,
                                      contract={'training_exposure': {method: {'status': 'not_project_trained',
                                                                             'complete': True, 'building_ids': []}
                                                                       for method in methods}}))
    for sequence in sequences:
        building = sequence['building_id']
        for frame in (1, 2):
            for mode in modes:
                for method in history if mode == 'history' else methods:
                    identifier = unit_id(method, mode, building, frame)
                    old = originals[building, 0 if mode == 'earliest' else frame - 1]
                    current = originals[building, frame]
                    support = np.ones((2, 3), bool)
                    if mode == 'history':
                        support[1, 0] = False
                    scores = np.array([[.1, .6, .3], [.9, .2, .8]], np.float32)
                    scores[~support] = np.nan
                    arrays = dict(reference=old['rgb'], source=current['rgb'], support=support, scores=scores)
                    if method == 'geoscd':
                        arrays['native_prediction'] = scores > .5
                    artifact = f'maps/{identifier}.npz'
                    row = dict(unit_id=identifier, method=method, mode=mode, sequence_id=building,
                               building_id=building, split=sequence['split'], frame=frame,
                               status='completed', origin='inference', elapsed_seconds=2.,
                               initialization_seconds=.2, scorer_seconds=[.8, .9],
                               artifact_path=artifact, artifact_sha256=atomic_npz(run / artifact, arrays))
                    atomic_json(run / 'units' / f'{identifier}.json', row)
    return prepared, run


class NightCalibrationDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.prepared, self.run = fixture(self.base)

    def tearDown(self):
        self.temporary.cleanup()

    def unit(self, method='rgb_diff', mode='previous', building='A', frame=1):
        return self.run / 'units' / (unit_id(method, mode, building, frame) + '.json')

    def change_map(self, **values):
        path = self.unit()
        row = read_json(path)
        artifact = self.run / row['artifact_path']
        with np.load(artifact, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        arrays.update(values)
        row['artifact_sha256'] = atomic_npz(artifact, arrays)
        atomic_json(path, row)

    def update_manifest(self, mutate):
        path = self.prepared / 'manifest.json'
        manifest = read_json(path)
        mutate(manifest)
        atomic_json(path, manifest)
        run = read_json(self.run / 'run.json')
        run['manifest_sha256'] = sha256(path)
        atomic_json(self.run / 'run.json', run)

    def test_shared_known_val_support_and_types_without_test_decoding(self):
        # Even checksumming these TEST artifacts would fail; the adapter must
        # inspect TEST identity metadata only, not its RGB, maps or GT arrays.
        for path in self.prepared.glob('T-*.npz'):
            path.unlink()
        for path in (self.run / 'units').glob('*.json'):
            row = read_json(path)
            if row['split'] == 'test':
                (self.run / row['artifact_path']).unlink()
        with patch('subprocess.Popen', side_effect=AssertionError('No model process permitted')):
            samples = data.Samples(self.run)
            rows = samples.rows(('geoscd:previous', 'msdzip_abs:history'))
        self.assertEqual(samples.buildings, ['A', 'B'])
        self.assertEqual(len(rows), 4)
        self.assertEqual(samples.channels, ['rgb_diff:previous', 'geoscd:previous',
                                           'msdzip_abs:previous', 'msdzip_abs:history'])
        for row in rows:
            self.assertEqual(row['split'], 'val')
            self.assertEqual(row['pixels'].tolist(), [0, 1, 2, 4])
            self.assertEqual(row['left'].dtype, bool)
            self.assertEqual(row['right'].dtype, np.float32)
            self.assertEqual(row['reference_bits'].dtype, np.uint16)
            self.assertTrue(np.isin(row['labels'], (0, 1)).all())
            self.assertFalse(row['left'].flags.writeable)
            self.assertEqual(len(row['support_sha256']), 64)
        # HISTORY's second target remains previous->current, not first->current.
        self.assertEqual(rows[1]['labels'].tolist(), [1, 1, 0, 0])
        self.assertEqual(len(samples.provenance['frame_sha256']), 6)
        self.assertEqual(len(samples.provenance['artifact_sha256']), 16)
        samples.validate_snapshot()

    def test_coverage_failures_missing_and_empty_known_support(self):
        path = self.unit('geoscd')
        row = read_json(path)
        row.update(status='failed', error='No finite correspondence')
        atomic_json(path, row)
        self.unit('geoscd', building='B', frame=2).unlink()
        def empty(manifest):
            frame = manifest['sequences'][0]['frames'][2]
            path = self.prepared / frame['artifact_path']
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            arrays['packed'][:] = 0
            frame['artifact_sha256'] = atomic_npz(path, arrays)
        self.update_manifest(empty)
        samples = data.Samples(self.run)
        rows = samples.rows(('geoscd:previous', 'rgb_diff:previous'))
        self.assertEqual([(row['building_id'], row['frame']) for row in rows], [('B', 1)])
        coverage = samples.coverage(('geoscd:previous', 'rgb_diff:previous'))
        a, b = coverage
        self.assertEqual((a['expected_pairs'], a['failed_pairs'], a['empty_gt_pairs'], a['evaluated_pairs']),
                         (2, 1, 1, 0))
        self.assertEqual((b['expected_pairs'], b['missing_pairs'], b['valid_pixels'], b['negative_pixels']),
                         (2, 1, 5, 4))
        self.assertIn('No finite correspondence', a['reason'])

    def test_terminal_run_and_manifest_checksum_required(self):
        path = self.run / 'run.json'
        run = read_json(path)
        run['status'] = 'running'
        atomic_json(path, run)
        with self.assertRaisesRegex(ValueError, 'terminal'):
            data.Samples(self.run)
        run['status'] = 'completed'
        run['manifest_sha256'] = '0' * 64
        atomic_json(path, run)
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            data.Samples(self.run)

    def test_map_hash_is_validated_before_cached_report_reuse(self):
        row = read_json(self.unit())
        with (self.run / row['artifact_path']).open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            data.Samples(self.run)

    def test_map_rgb_support_and_dtype_contract(self):
        with self.subTest('RGB'):
            self.change_map(source=np.zeros((2, 3, 3), np.uint8))
            with self.assertRaisesRegex(ValueError, 'RGB/support'):
                data.Samples(self.run).rows(('rgb_diff:previous',))
        # Restore correct RGB, then separately test score and support types.
        with self.subTest('score dtype'):
            self.change_map(source=np.full((2, 3, 3), 20 + ord('A'), np.uint8),
                            scores=np.zeros((2, 3), np.float64))
            with self.assertRaisesRegex(ValueError, 'score grid'):
                data.Samples(self.run).rows(('rgb_diff:previous',))
        with self.subTest('support dtype'):
            self.change_map(scores=np.zeros((2, 3), np.float32), support=np.ones((2, 3), np.uint8))
            with self.assertRaisesRegex(ValueError, 'RGB/support'):
                data.Samples(self.run).rows(('rgb_diff:previous',))

    def test_split_leak_and_unit_identity_rejected(self):
        self.update_manifest(lambda manifest: manifest['sequences'][2].update(building_id='A'))
        with self.assertRaisesRegex(ValueError, 'multiple splits'):
            data.Samples(self.run)
        self.update_manifest(lambda manifest: manifest['sequences'][2].update(building_id='T'))
        row = read_json(self.unit())
        row['split'] = 'test'
        atomic_json(self.unit(), row)
        with self.assertRaisesRegex(ValueError, 'identity'):
            data.Samples(self.run)

    def test_mutated_source_after_loading_rejected(self):
        samples = data.Samples(self.run)
        samples.rows(('rgb_diff:previous',))
        row = read_json(self.unit())
        row['elapsed_seconds'] = 40
        atomic_json(self.unit(), row)
        with self.assertRaisesRegex(ValueError, 'input changed'):
            samples.validate_snapshot()

    def test_method_filter_ignores_other_map_pixels_and_timings_separate_reuse(self):
        path = self.unit()
        row = read_json(path)
        row.update(origin='reused_source', elapsed_seconds=.01, initialization_seconds=0., scorer_seconds=[])
        atomic_json(path, row)
        excluded = read_json(self.unit('geoscd'))
        (self.run / excluded['artifact_path']).unlink()
        samples = data.Samples(self.run, methods=['rgb_diff'])
        self.assertEqual(samples.channels, ['rgb_diff:previous'])
        self.assertEqual(len(samples.rows(('rgb_diff:previous',))), 4)
        timing = samples.timings[0]
        self.assertEqual((timing['computed_units'], timing['reused_units']), (3, 1))
        self.assertEqual(timing['elapsed_seconds_sum'], 6.)
        self.assertEqual(timing['scorer_seconds_observations'], 6)
        self.assertAlmostEqual(timing['scorer_seconds_median'], .85)

    def test_unknown_gt_bits_and_invalid_unit_mode_rejected(self):
        def extra_bit(manifest):
            frame = manifest['sequences'][0]['frames'][0]
            path = self.prepared / frame['artifact_path']
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            arrays['packed'][0, 0] = 4
            frame['artifact_sha256'] = atomic_npz(path, arrays)
        self.update_manifest(extra_bit)
        with self.assertRaisesRegex(ValueError, 'неизвестный бит'):
            data.Samples(self.run).rows(('rgb_diff:previous',))
        row = read_json(self.unit())
        row['mode'] = 'history'
        atomic_json(self.unit(), row)
        with self.assertRaisesRegex(ValueError, 'outside'):
            data.Samples(self.run)


if __name__ == '__main__':
    unittest.main()
