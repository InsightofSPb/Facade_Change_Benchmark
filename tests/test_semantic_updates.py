"""Scientific regression checks and a complete synthetic cache-only CLI run."""
import contextlib
import copy
import importlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.real_eval.robustness_common import atomic_json, atomic_npz, digest, read_json, sha256
from scripts.real_eval.robustness_run import payload_sha256
from scripts.real_eval.semantic_cache import Cache, hash_array

pilot = importlib.import_module('scripts.real_eval.2026-10-19_semantic_updates')


def fixture(root):
    prepared, semantic, wide = (root / name for name in ('prepared', 'semantic', 'wide'))
    for path in (prepared, semantic / 'units', wide / 'units'):
        path.mkdir(parents=True)
    cache = Cache(root / 'cache')
    categories = [{'id': 6, 'name': 'CRACK'}, {'id': 7, 'name': 'REPAIRS'}]
    mapping = [{'category_id': 6, 'channel': 0}, {'category_id': 7, 'channel': 1}]
    vocabulary = [{'name': c['name'], 'prompts': [c['name']], 'group': 'target'} for c in categories]
    model = {'ok': True, 'config': {'kind': 'dino_semantic'}, 'files': {'head': {'sha256': 'a' * 64}}}
    config = dict(prepared=str(prepared), segmenters={'dino_semantic': {'kind': 'dino_semantic', 'confidence': .5}},
                  vocabularies={'domain': vocabulary}, modes=['previous'])
    sequences, frames = [], {}
    for split, base in (('train', 20), ('val', 40), ('test', 60)):
        sequence = dict(sequence_id='sequence-' + split, building_id='facade-' + split, split=split, frames=[])
        for frame in (0, 1):
            bits = np.full((4, 4), 2, np.uint16)
            if frame:
                bits[0, 0] = 1
            rgb = np.full((4, 4, 3), base + frame, np.uint8)
            support = np.ones((4, 4), bool)
            data = dict(rgb=rgb, support=support, packed=bits)
            path = f'{split}-{frame}.npz'
            checksum = atomic_npz(prepared / path, data)
            sequence['frames'].append(dict(frame=frame, year=2020 + frame, artifact_path=path, artifact_sha256=checksum))
            frames[split, frame] = data
        sequences.append(sequence)
    manifest = dict(schema_version=1, kind='real_temporal_prepared', categories=categories,
                    category_to_channel=mapping, sequences=sequences)
    atomic_json(prepared / 'manifest.json', manifest)
    revision = digest(dict(config=config, prepared=sha256(prepared / 'manifest.json'), schema='fixture-schema'))
    atomic_json(semantic / 'run.json', dict(kind='semantic_routes', smoke=False, status='completed',
                config=config, config_revision=revision, schema='fixture-schema', cache_root=str(cache.root)))
    atomic_json(semantic / 'preflight.json', dict(segmenters={'dino_semantic': model}))
    metadata = {m: {'method': m, 'checkpoint_sha256': ('b' if m == 'lpips' else 'c') * 64} for m in pilot.METHODS}
    contract = dict(manifest_sha256=sha256(prepared / 'manifest.json'),
                    models={m: {'expected_metadata': metadata[m]} for m in pilot.METHODS})
    atomic_json(wide / 'run.json', dict(contract=contract, config_hash=digest(contract), status='completed'))
    for sequence in sequences:
        split = sequence['split']
        for frame in (0, 1):
            data = frames[split, frame]
            # Reversed channels ensure matching is by class name, never index.
            scores = pilot.unpack(data['packed'], 2)[::-1].astype(np.float32) * .8 + .1
            # A false semantic transition on a stable region.
            if frame:
                scores[:, 0, 1] = [.1, .9]
            spec = dict(stage='segmentation_native', schema='fixture-schema', model=model,
                        vocabulary=vocabulary, rgb=hash_array(data['rgb']))
            record = cache.put(spec, dict(scores=scores), metadata=dict(class_names=['REPAIRS', 'CRACK'],
                                score_kind='independent_sigmoid', method_metadata={'provenance': model}))
            row = dict(kind='segmentation', segmenter='dino_semantic', variant='fixed', route='baseline',
                       sequence_id=sequence['sequence_id'], building_id=sequence['building_id'], split=split,
                       frame=frame, confidence=.5, status='completed', config_revision=revision,
                       native_cache_key=record['cache_key'])
            atomic_json(semantic / 'units' / f'{split}-{frame}.json', row)
        payload = dict(reference=frames[split, 0]['rgb'], source=frames[split, 1]['rgb'],
                       support=frames[split, 0]['support'])
        for method in pilot.METHODS:
            scores = np.full((4, 4), .1, np.float32)
            scores[0, 0] = .9
            # LPIPS rejects the false semantic transition; MSDZip rejects a true event on VAL.
            if method == 'msdzip_abs' and split == 'val':
                scores[0, 0] = .05
            relative = f'maps/{split}-{method}.npz'
            checksum = atomic_npz(wide / relative, dict(payload, scores=scores))
            row = dict(method=method, mode='previous', variant='current_train' if method == 'msdzip_abs' else 'frozen',
                       sequence_id=sequence['sequence_id'], building_id=sequence['building_id'], split=split,
                       frame=1, reference_frame=0, status='completed', input_sha256=payload_sha256(payload),
                       method_metadata=metadata[method], artifact_path=relative, artifact_sha256=checksum)
            atomic_json(wide / 'units' / f'{split}-{method}.json', row)
    return prepared, semantic, wide


def hashes(roots):
    return {str(p): sha256(p) for root in roots for p in root.rglob('*') if p.is_file()}


class DirectedUpdateTests(unittest.TestCase):
    def test_wrong_direction_is_not_a_correct_semantic_prediction(self):
        old, new = np.array([[1]], np.uint16), np.array([[2]], np.uint16)
        prediction = dict(C=np.ones((1, 1), bool), V=old, reconstructed=old)
        rows = pilot.evaluate(new, prediction, old, new, np.ones((1, 1), bool),
                              [{'name': 'CRACK'}, {'name': 'REPAIRS'}])
        self.assertEqual(rows[0]['f1'], 1.)
        self.assertEqual(rows[1]['f1'], 0.)
        self.assertEqual(rows[1]['fp'], 2)
        self.assertEqual(rows[1]['fn'], 2)
        self.assertEqual(rows[0]['exact_v_recall'], 0.)

    def test_unknown_pixels_never_become_healthy_and_suppressed_rois_are_fn(self):
        old = np.array([[1, 0, 1]], np.uint16)
        new = np.array([[2, 2, 2]], np.uint16)
        support = np.array([[True, True, False]])
        result = pilot.updates(old, new, np.zeros((1, 3)), np.zeros((1, 3)),
                               dict(lpips=.5, msdzip_abs=.5), support)
        rows = pilot.evaluate(old, result['semantic_lpips'], old, new, support,
                              [{'name': 'CRACK'}, {'name': 'REPAIRS'}])
        self.assertEqual(rows[0]['valid_pixels'], 1)
        self.assertEqual(rows[0]['fn'], 1)
        self.assertEqual(rows[1]['fn'], 2)
        self.assertEqual(rows[0]['ignored_pixels'], 2)
        self.assertIsNone(rows[0]['exact_v_precision'])
        self.assertEqual(rows[0]['exact_v_recall'], 0.)

    def test_multilabel_cv_reconstruction_and_gate_subsets(self):
        before = np.array([[1, 2, 3, 0]], np.uint16)
        after = np.array([[3, 1, 2, 1]], np.uint16)
        support = np.array([[True, True, True, False]])
        results = pilot.updates(before, after, np.array([[1, 1, 0, 1]]), np.array([[1, 0, 1, 1]]),
                                dict(lpips=.5, msdzip_abs=.5), support)
        np.testing.assert_array_equal(results['semantic']['reconstructed'][support], after[support])
        for result in results.values():
            np.testing.assert_array_equal(np.where(result['C'], result['V'], before), result['reconstructed'])
        self.assertEqual([int(r['C'].sum()) for r in results.values()], [3, 2, 1])
        self.assertEqual(int(results['semantic']['V'][0, 0]), 3)  # Overlap preserved.

    def test_test_labels_and_scores_cannot_change_train_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            _, semantic, wide = fixture(Path(directory))
            source = pilot.Inputs(semantic, wide)
            pairs = [source.pair(s, 1) for s in source.manifest['sequences']]
            initial = pilot.fit_thresholds(pairs)
            for pair in pairs:
                if pair['context']['split'] != 'train':
                    pair['labels'][:] = 1
                    for values in pair['maps'].values():
                        values['scores'][:] = 1e8
            self.assertEqual(initial, pilot.fit_thresholds(pairs))
            self.assertEqual(initial['lpips']['train_buildings'], ['facade-train'])


class CachePilotTests(unittest.TestCase):
    def test_end_to_end_readonly_and_truthful_harmful_compression_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, semantic, wide = fixture(root)
            before = hashes((prepared, semantic, wide, root / 'cache'))
            out = root / 'out'
            with patch('subprocess.Popen', side_effect=AssertionError('No subprocess/inference')), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pilot.main(['--semantic-run', str(semantic), '--change-run', str(wide), '--out', str(out)]), 0)
            self.assertEqual(before, hashes((prepared, semantic, wide, root / 'cache')))
            self.assertEqual(read_json(out / 'run.json')['status'], 'completed')
            self.assertTrue((out / '2026-10-19_report.html').is_file())
            self.assertTrue((out / '2026-10-19_input_hashes.json').is_file())
            import csv
            with (out / 'summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            val = {r['variant']: r for r in rows if r['split'] == 'val' and r['target'] == 'binary_change'
                   and r['protocol'] == 'all_frames'}
            self.assertAlmostEqual(float(val['semantic']['f1']), 2 / 3)
            self.assertEqual(float(val['semantic_lpips']['f1']), 1.)
            self.assertEqual(float(val['semantic_lpips_msdzip']['recall']), 0.)
            directed = next(r for r in rows if r['split'] == 'val' and r['target'] == 'directed_events'
                            and r['variant'] == 'semantic_lpips' and r['protocol'] == 'all_frames')
            self.assertEqual(float(directed['f1']), 1.)
            with self.assertRaises(FileExistsError):
                pilot.run(semantic, wide, out)

    def test_missing_cache_blocks_without_inference_or_partial_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, semantic, wide = fixture(root)
            (wide / 'maps' / 'val-lpips.npz').unlink()
            with contextlib.redirect_stdout(io.StringIO()):
                result = pilot.run(semantic, wide, root / 'out')
            self.assertEqual(result['status'], 'blocked')
            self.assertTrue((root / 'out' / 'coverage.csv').exists())
            self.assertFalse((root / 'out' / 'summary.csv').exists())

    def test_corrupt_cache_is_rejected_without_quarantine_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, semantic, wide = fixture(root)
            key = read_json(semantic / 'units' / 'val-1.json')['native_cache_key']
            path = root / 'cache' / 'objects' / (key + '.npz')
            path.write_bytes(path.read_bytes() + b'corruption')
            before = hashes((root / 'cache',))
            with contextlib.redirect_stdout(io.StringIO()):
                result = pilot.run(semantic, wide, root / 'out')
            self.assertEqual(result['status'], 'blocked')
            self.assertEqual(before, hashes((root / 'cache',)))

    def test_audit_all_inputs_and_mismatched_model_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, semantic, wide = fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pilot.run(semantic, wide, root / 'audit', True)['status'], 'audit_passed')
            path = wide / 'units' / 'val-msdzip_abs.json'
            row = read_json(path)
            row['method_metadata']['checkpoint_sha256'] = 'd' * 64
            atomic_json(path, row)
            with contextlib.redirect_stdout(io.StringIO()):
                result = pilot.run(semantic, wide, root / 'out')
            self.assertEqual(result['status'], 'blocked')

    def test_both_averaging_protocols_keep_equal_facade_weights(self):
        base = dict(variant='semantic', split='val', target='binary_change', class_name='__all__', direction='change')
        positive = dict(base, **pilot.real_gt_metrics.evaluate_mask(np.array([True]), np.array([1], np.uint8)))
        false_alarm = dict(base, **pilot.real_gt_metrics.evaluate_mask(np.array([True]), np.array([0], np.uint8)))
        rows = [dict(positive, building_id='A'), dict(false_alarm, building_id='A'), dict(positive, building_id='B')]
        summaries, _ = pilot.aggregate(rows)
        by_protocol = {r['protocol']: r for r in summaries}
        self.assertEqual(by_protocol['all_frames']['f1'], .75)
        self.assertEqual(by_protocol['positive_frames']['f1'], 1.)
        self.assertEqual(by_protocol['all_frames']['fpr'], 1.)


if __name__ == '__main__':
    unittest.main()
