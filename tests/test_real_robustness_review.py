"""Cached review preserves original selections, GT semantics and source artifacts."""
import builtins
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.real_eval.robustness_common import atomic_json, load_selection, read_json, sha256
from scripts.real_eval.robustness_prepare import prepare
from scripts.real_eval.robustness_review import review
from tests.test_real_robustness_prepare import fixture, tree_hashes


class ReviewTests(unittest.TestCase):
    def prepared(self, base):
        run, snapshot, adapter, *_ = fixture(base)
        with contextlib.redirect_stdout(io.StringIO()):
            manifest = prepare(run, snapshot, base / 'prepared', input_adapter=adapter)
        return manifest

    def test_cached_review_gt_masks_selection_and_immutable_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest = self.prepared(base)
            sources = [base / name for name in ('source', 'snapshot', 'prepared')]
            hashes = [tree_hashes(root) for root in sources]
            original_import = builtins.__import__

            def guarded(name, *args, **kwargs):
                if name == 'torch' or name.startswith(('torch.', 'transformers', 'facade_change.methods')):
                    raise AssertionError('Review imported a model stack: ' + name)
                return original_import(name, *args, **kwargs)

            out = base / 'review'
            with patch('builtins.__import__', side_effect=guarded), contextlib.redirect_stdout(io.StringIO()):
                result = review(base / 'prepared', out)
            self.assertEqual(hashes, [tree_hashes(root) for root in sources])
            self.assertEqual(result['summary']['ready_pairs'], 6)
            self.assertEqual(result['summary']['unknown_pairs'], 2)
            row = next(row for row in result['pairs'] if row.get('stats', {}).get('changed_pixels'))
            self.assertEqual(row['stats']['changed_pixels'], 9)
            # Wall stays present while crack is added; global change must not imply wall loss.
            wall = np.array(Image.open(out / row['transition_masks']['0']))
            crack = np.array(Image.open(out / row['transition_masks']['1']))
            combined = np.array(Image.open(out / row['transition_masks']['all']))
            self.assertEqual(wall[2, 3, 3], 0)
            np.testing.assert_array_equal(crack[2, 3], [25, 200, 95, 190])
            np.testing.assert_array_equal(combined[2, 3], [255, 40, 40, 175])
            for item in result['pairs']:
                if item['status'] == 'ready':
                    for relative in item['previews'].values():
                        self.assertTrue((out / relative).is_file())
            selection = dict(schema_version=1, kind='real_robustness_selection', reviewed=True,
                             pair_ids=[row['pair_id']], manifest_sha256=result['manifest_sha256'])
            atomic_json(out / 'reviewed_selection.json', selection)
            _, selected = load_selection(out / 'reviewed_selection.json', base / 'prepared', manifest)
            self.assertEqual(selected[0]['pair_id'], row['pair_id'])
            self.assertEqual(result['manifest_sha256'], sha256(base / 'prepared' / 'manifest.json'))
            self.assertEqual(read_json(out / 'run.json')['status'], 'completed')
            self.assertTrue((out / 'pair_statistics.csv').is_file())
            self.assertTrue((out / 'class_statistics.csv').is_file())
            self.assertTrue((out / 'threshold_statistics.csv').is_file())

    def test_reject_source_output_and_corrupt_cached_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest = self.prepared(base)
            with self.assertRaises(FileExistsError):
                review(base / 'prepared', base / 'prepared' / 'nested')
            with self.assertRaises(FileExistsError):
                review(base / 'prepared', base / 'source' / 'nested')
            row = next(row for row in manifest['pairs'] if row['status'] == 'ready')
            with (base / 'prepared' / row['artifact_path']).open('ab') as stream:
                stream.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'Prepared artifact changed'), contextlib.redirect_stdout(io.StringIO()):
                review(base / 'prepared', base / 'bad-review')
            self.assertEqual(read_json(base / 'bad-review' / 'run.json')['status'], 'failed')
            self.assertFalse((base / 'bad-review' / 'gallery.html').exists())


if __name__ == '__main__':
    unittest.main()
