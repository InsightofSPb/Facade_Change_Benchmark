"""Diagnostics count report snapshots, not stale native/unit caches."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.real_eval.semantic_diagnostics import diagnose, format_summary


class SemanticDiagnosticsTests(unittest.TestCase):
    def save(self, out, name, value):
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    def coverage(self, out, rows):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (out / 'coverage.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def row(self, **fields):
        return dict(config_revision='current', status='failed', kind='segmentation',
                    split='val', sequence_id='s1', frame=1, reference_frame=None,
                    segmenter='sam', method=None, variant='domain', route='baseline',
                    error='model failed', **fields)

    def test_current_report_excludes_stale_units_even_same_revision(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            current = self.row()
            self.save(out, 'run.json', dict(config_revision='current', status='completed_with_failures', records=1, failed=1))
            self.coverage(out, [current])
            self.save(out, 'units/current.json', current)
            self.save(out, 'units/stale_same_revision.json', {**current, 'frame': 4, 'error': 'old invocation'})
            self.save(out, 'units/stale_revision.json', {**current, 'config_revision': 'old', 'error': 'old revision'})
            result = diagnose(out)
            self.assertEqual(result['failed_rows'], 1)
            self.assertEqual(result['record_rows'], 1)
            self.assertEqual(result['errors'][0]['unit_file'], str(out / 'units/current.json'))
            self.assertNotIn('old invocation', format_summary(result))

    def test_coverage_revision_filter_and_exact_error_groups(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            baseline = self.row()
            dependency = {**baseline, 'kind': 'change', 'reference_frame': 0,
                          'method': 'dino', 'route': 'semantic_first_gate'}
            independent = {**dependency, 'route': 'change_first', 'error': 'ROI shape error'}
            stale = {**baseline, 'config_revision': 'old', 'error': 'old'}
            self.save(out, 'run.json', dict(config_revision='current', status='completed_with_failures', records=3, failed=3))
            self.coverage(out, [baseline, dependency, independent, stale])
            result = diagnose(out)
            self.assertEqual(result['failed_rows'], 3)
            self.assertEqual({row['error']: row['count'] for row in result['errors']},
                             {'model failed': 2, 'ROI shape error': 1})
            self.assertEqual(sum(result['failure_breakdown'].values()), 3)
            self.assertTrue(result['warnings'])

    def test_missing_report_does_not_count_old_unit_files(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            self.save(out, 'run.json', dict(config_revision='current', status='running'))
            self.save(out, 'units/stale.json', self.row())
            with self.assertRaisesRegex(ValueError, 'дождитесь первого отчёта'):
                diagnose(out)

    def test_successful_measured_timings_exclude_cache_inventory_and_failed_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            self.save(out, 'run.json', dict(config_revision='current', status='completed', records=1, failed=0))
            self.coverage(out, [{**self.row(), 'status': 'completed', 'error': ''}])
            measured = dict(stage='segment_full', method=None, segmenter='sam', variant='domain',
                            cache_key='one', status='completed', source='measured', elapsed_seconds=4., init_seconds=7.)
            self.save(out, 'timing_observations.json', [measured,
                      {**measured, 'source': 'cache', 'inventory': True},
                      {**measured, 'source': 'cache'},
                      {**measured, 'source': 'imported', 'cache_key': 'two', 'elapsed_seconds': None},
                      {**measured, 'status': 'failed', 'cache_key': 'failed', 'elapsed_seconds': 999.},
                      {**measured, 'stage': 'segment_roi', 'cache_key': 'probe', 'elapsed_seconds': 2.,
                       'init_seconds': None, 'timing_only': True}])
            self.save(out, 'estimate.json', dict(wall_seconds=120., status='estimated', total_lower_seconds=10., total_upper_seconds=20.))
            result = diagnose(out)
            self.assertEqual(sum(row['inference_seconds'] for row in result['timing']), 6.)
            self.assertEqual(sum(row['init_seconds'] for row in result['timing']), 7.)
            self.assertEqual(sum(row['timing_only_samples'] for row in result['timing']), 1)
            self.assertEqual(result['observation_sources'], dict(measured=2, cache=1, imported=1))
            self.assertIn('WALL_MINUTES: 2.00', format_summary(result))

    def test_error_tail_bounded_and_cli_stdlib_only_read_only(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            error = 'Traceback start\n' + 'noise\n' * 500 + 'TypeError: precise root cause'
            self.save(out, 'run.json', dict(config_revision='current', status='completed_with_failures', records=1, failed=1))
            self.coverage(out, [{**self.row(), 'error': error}])
            before = {str(path): path.read_bytes() for path in out.rglob('*') if path.is_file()}
            result = subprocess.run([sys.executable, '-S', '-m', 'scripts.real_eval.semantic_diagnostics',
                                     '--out', str(out), '--max-error-chars', '200'],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('TypeError: precise root cause', result.stdout)
            self.assertIn('середина сокращена', result.stdout)
            self.assertIn('ETA при ошибках неполон', result.stdout)
            self.assertNotIn('noise\n' * 100, result.stdout)
            after = {str(path): path.read_bytes() for path in out.rglob('*') if path.is_file()}
            self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
