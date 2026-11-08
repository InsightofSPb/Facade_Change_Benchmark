"""Moved standalone tools remain runnable from outside their checkout."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class RealEvalCLITests(unittest.TestCase):
    def test_standalone_help_from_unrelated_working_directory(self):
        scripts = Path(__file__).resolve().parents[1] / 'scripts' / 'real_eval'
        with tempfile.TemporaryDirectory() as directory:
            for name in ('geoscd_real_retry', 'real_temporal_gallery', 'real_gt_metrics',
                         'real_gt_calibration', 'real_final_fpr', 'temporal_fusion', 'temporal_tables'):
                with self.subTest(script=name):
                    result = subprocess.run(
                        [sys.executable, str(scripts / (name + '.py')), '--help'],
                        cwd=directory, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('--out', result.stdout)

    def test_robustness_subcommands_from_unrelated_working_directory(self):
        script = Path(__file__).resolve().parents[1] / 'scripts' / 'real_eval' / 'real_robustness.py'
        with tempfile.TemporaryDirectory() as directory:
            for command in ('prepare', 'review', 'run', 'report'):
                with self.subTest(command=command):
                    result = subprocess.run([sys.executable, str(script), command, '--help'],
                                            cwd=directory, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('--out', result.stdout)

    def test_temporal_night_subcommands_from_unrelated_working_directory(self):
        script = Path(__file__).resolve().parents[1] / 'scripts' / 'real_eval' / 'real_temporal_night.py'
        with tempfile.TemporaryDirectory() as directory:
            for command in ('audit', 'prepare', 'pilot', 'run', 'report', 'status'):
                with self.subTest(command=command):
                    result = subprocess.run([sys.executable, str(script), command, '--help'],
                                            cwd=directory, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('--run' if command == 'status' else '--out', result.stdout)


if __name__ == '__main__':
    unittest.main()
