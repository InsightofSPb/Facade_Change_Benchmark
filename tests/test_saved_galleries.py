"""Saved galleries preserve identities, calibrated decisions and input bytes."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from facade_change import workflow
from facade_change.gallery import lofo, report_sheets
from scripts.real_eval import night_calibration
from scripts.real_eval.night_common import unit_id
from scripts.real_eval.robustness_common import atomic_json, read_json, sha256
from test_night_calibration import hashes, saved_night_fixture


def report_fixture(root):
    root.mkdir()
    (root / 'gallery').mkdir()
    articles, maps = [], []
    for index in (1, 2):
        prefix = f'view-{index}'
        figures = []
        for name, caption, color in (
                ('before', 'RGB раньше', 30 * index),
                ('after', 'RGB позже', 30 * (index + 1)),
                ('gt', 'GT', 120), ('semantic', 'semantic_delta', 150),
                ('fusion', 'semantic_lpips', 180)):
            relative = f'gallery/{prefix}-{name}.png'
            Image.new('RGB', (16, 12), (color, color, color)).save(root / relative)
            figures.append(f'<figure><img src="{relative}"><figcaption>{caption}</figcaption></figure>')
        articles.append(f'<article><h3>train · facade · {2000+index} → {2001+index}</h3>'
                        + ''.join(figures) + '</article>')
        maps.append(dict(artifact_path=prefix+'.npz', split='train',
                         building_id='facade', sequence_id='one-view'))
    report = root / 'report.html'
    report.write_text('<html><h1>Saved report</h1>' + ''.join(articles) + '</html>', encoding='utf-8')
    atomic_json(root / 'artifacts.json', dict(maps=maps))
    return report


class ReportSheetTests(unittest.TestCase):
    def test_report_and_archive_produce_same_panels_without_modifying_inputs(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            report = report_fixture(root / 'source')
            before = hashes(report.parent)
            archive = root / 'review.tar.gz'
            with tarfile.open(archive, 'w:gz') as stream:
                stream.add(report.parent, arcname='saved')
            for option, source, out in (('--report', report, root/'direct'),
                                        ('--archive', archive, root/'archived')):
                self.assertEqual(workflow.main(['gallery', 'sheets', option, str(source),
                                               '--out', str(out), '--tile', '256']), 0)
                provenance = read_json(out / 'provenance.json')
                self.assertEqual(len(provenance['pairs']), 2)
                self.assertEqual({p['sequence_id'] for p in provenance['pairs']}, {'one-view'})
                self.assertEqual(provenance['source_report_sha256'], before['report.html'])
                self.assertEqual(provenance['source_image_sha256'],
                                 {k: v for k, v in before.items() if k.endswith('.png')})
                self.assertEqual(len(list((out/'sequences').glob('*.png'))), 1)
                self.assertTrue((out/'index.html').is_file())
            a, b = (hashes(root/p) for p in ('direct', 'archived'))
            self.assertEqual({k:v for k,v in a.items() if k.endswith('.png')},
                             {k:v for k,v in b.items() if k.endswith('.png')})
            self.assertEqual(hashes(report.parent), before)

    def test_conflicting_year_rgb_and_escaping_images_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            root = Path(directory)
            report = report_fixture(root / 'source')
            Image.new('RGB', (16, 12), 'red').save(report.parent/'gallery/view-2-before.png')
            self.assertEqual(report_sheets.main(['--report', str(report), '--out', str(root/'out')]), 1)
            report.write_text(report.read_text().replace('gallery/view-1-before.png', '../outside.png'))
            with self.assertRaisesRegex(ValueError, 'leaves the report directory'):
                report_sheets.load_report(report)

    def test_archive_traversal_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'bad.tar.gz'
            with tarfile.open(archive, 'w:gz') as stream:
                info = tarfile.TarInfo('../outside')
                info.size = 1
                stream.addfile(info, io.BytesIO(b'x'))
            with self.assertRaisesRegex(ValueError, 'Unsafe archive member'):
                report_sheets.extract_archive(archive, root/'extract')
            self.assertFalse((root/'outside').exists())


class SavedLOFOGalleryTests(unittest.TestCase):
    def test_saved_recipes_counts_support_and_native_masks_survive_migration(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            base = Path(directory)
            prepared, run = saved_night_fixture(base)
            rename = {'rgb_diff': 'lpips', 'msdzip_abs': 'semantic_dino'}
            config = read_json(run/'run.json')
            config['methods'] = [rename.get(m, m) for m in config['methods']]
            config['history_methods'] = ['semantic_dino']
            config['contracts'] = {rename.get(k, k): v for k, v in config['contracts'].items()}
            atomic_json(run/'run.json', config)
            for path in list((run/'units').glob('*.json')):
                record = read_json(path)
                record['method'] = rename.get(record['method'], record['method'])
                record['variant'] = 'frozen'
                record['domain'] = 'real'
                record['unit_id'] = unit_id(record['method'], record['mode'],
                                            record['sequence_id'], record['frame'])
                path.unlink()
                atomic_json(run/'units'/f"{record['unit_id']}.json", record)
            calibration = base / 'calibration'
            night_calibration.run(run, calibration, budgets=(.05,), methods=['anychange', 'lpips', 'semantic_dino'])
            before = hashes(run), hashes(prepared), hashes(calibration)
            out = base/'gallery'
            with patch('subprocess.Popen', side_effect=AssertionError('Gallery cannot launch inference')):
                self.assertEqual(workflow.main(['gallery', 'lofo', '--run', str(run),
                                               '--calibration-run', str(calibration), '--out', str(out),
                                               '--facade', 'A', '--tile', '128']), 0)
            self.assertEqual((hashes(run), hashes(prepared), hashes(calibration)), before)
            provenance = read_json(out/'provenance.json')
            self.assertEqual(provenance['verified_method_masks'], 10)
            self.assertTrue(provenance['native_anychange_unchanged'])
            self.assertTrue(provenance['no_recalibration'])
            self.assertEqual(set(provenance['dependency_source_sha256']),
                             {'night_pair_calibration.py', 'real_gt_metrics.py'})
            with (out/'per_pair_verification.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({r['parity'] for r in rows}, {'passed'})
            self.assertEqual({r['method'] for r in rows}, {s[0] for s in lofo.SPECS})
            # A semantically corrupted table still fails after its file hash is
            # no longer involved: the renderer checks reconstructed mask counts.
            from argparse import Namespace
            inputs = lofo.Inputs(Namespace(run=run, calibration_run=calibration, prepared=None))
            old, new = inputs.frame('A', 0), inputs.frame('A', 1)
            labels = lofo.derive_labels(old['packed'], new['packed'],
                                       old['support'] & new['support'], 3)
            saved = next(r for r in inputs.families['semantic']['per_pair']
                         if r['sequence_id']=='A' and r['frame']==1 and r['role']=='single')
            saved['tp'] += 1
            with self.assertRaisesRegex(ValueError, 'Маска отличается'):
                inputs.prediction('semantic', 'A', 1, .05, labels)


if __name__ == '__main__':
    unittest.main()
