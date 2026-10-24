"""An active-run snapshot evaluates complete VAL groups and preserves inputs."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from facade_change import workflow
from facade_change.research import temporal_quick_report as quick
from facade_change.research.common import digest
from scripts.real_eval.robustness_common import atomic_json, read_json
from test_night_calibration import hashes
from test_temporal_reports import item, save


def snapshot_fixture(root):
    root.mkdir()
    (root/'units').mkdir()
    config = dict(methods=['rgb_diff'], history_methods=['rgb_diff'])
    atomic_json(root/'run.json', dict(status='running', config=config, config_sha256=digest(config)))
    atomic_json(root/'experiment_contract.json', dict(config=config, sequence_ids=['clean', 'event']))
    atomic_json(root/'preflight.json', dict(coverage={'synthetic': {'by_split': {'val': 2}}}))
    for mode in ('previous', 'history', 'earliest'):
        for arm in ('clean', 'event'):
            for frame in range(1, 5):
                if mode=='earliest' and (arm!='clean' or frame!=1):
                    continue
                value = item(f'{mode}-{arm}-{frame}', [0., 1. if arm=='event' else .1],
                             variant='frozen', mode=mode, target='earliest_state' if mode=='earliest' else 'step_event',
                             sequence_id=arm, building_id='A', frame=frame, arm=arm,
                             hypothesis='H1' if arm=='event' else 'H0')
                if arm=='event':
                    value['arrays']['labels'][0, 1] = 1
                record = save(root, value)
                atomic_json(root/'units'/f"{record['unit_id']}.json", record)


class TemporalSnapshotTests(unittest.TestCase):
    def test_complete_groups_only_facade_metrics_and_previous_history_parity(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source, out = root/'source', root/'out'
            snapshot_fixture(source)
            before = hashes(source)
            with patch('subprocess.Popen', side_effect=AssertionError('Snapshot cannot launch inference')):
                self.assertEqual(workflow.main(['benchmark', 'temporal-snapshot',
                                               '--run', str(source), '--out', str(out)]), 0)
            result = read_json(out/'2026-10-17_quick_summary.json')
            self.assertEqual(hashes(source), before)
            self.assertEqual(result['verified_archive_count'], 16)
            self.assertEqual(len(result['metrics']), 4)
            self.assertEqual({r['mode'] for r in result['metrics']}, {'previous', 'history'})
            for row in result['metrics']:
                self.assertEqual((row['building_count'], row['recall'], row['h0_pixel_fpr']), (1, 1., 0.))
            missing = [r for r in result['coverage'] if r['status']=='incomplete_skipped']
            self.assertEqual(len(missing), 1)
            self.assertEqual((missing[0]['completed_units'], missing[0]['expected_units']), (1, 8))
            self.assertTrue(all(r['recall_delta']==0. for r in result['history_comparisons']))
            with self.assertRaises(FileExistsError):
                quick.run(source, out)

    def test_committed_corruption_and_incomplete_reference_are_not_scored(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source = root/'source'
            snapshot_fixture(source)
            (source/'maps/previous-clean-1.npz').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'Corrupt committed artifact'):
                quick.run(source, root/'corrupt-report')
            (source/'units/previous-clean-1.json').unlink()
            with self.assertRaisesRegex(ValueError, 'No complete five-frame'):
                quick.run(source, root/'incomplete-report')


if __name__ == '__main__':
    unittest.main()
