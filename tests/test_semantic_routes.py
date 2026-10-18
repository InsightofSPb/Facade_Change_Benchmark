"""Small end-to-end CPU routes: real RGB-DIFF/LZMA workers and durable cache."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.real_eval import semantic_routes as routes
from scripts.real_eval.robustness_common import atomic_json, atomic_npz, read_json


class TestOnlySegmenterPool:
    """A predictable test fixture; never registered as a production adapter."""
    calls = []

    def __init__(self, log_dir=None):
        self.process = None

    def predict(self, config, vocabulary, rgb):
        self.calls.append((copy.deepcopy(config), copy.deepcopy(vocabulary), rgb.copy()))
        names = [row['name'] for row in vocabulary]
        scores = np.zeros((len(names), *rgb.shape[:2]), np.float32)
        for index, name in enumerate(names):
            if name == 'CRACK':
                scores[index] = (rgb[..., 0] > 200).astype(np.float32) * .9
            elif name == 'WALL':
                scores[index] = .99  # Flat context wins; layered target survives.
        return {'scores': scores, 'class_names': names, 'score_kind': 'independent_sigmoid',
                'extra_arrays': {}, 'metadata': {'activation_seconds': .1, 'provenance': test_preflight(config)},
                'elapsed_seconds': .02, 'request_elapsed_seconds': .025}

    def release(self):
        self.process = None


def test_preflight(config):
    return {'ok': True, 'errors': [], 'config': copy.deepcopy(config),
            'files': {'checkpoint': {'sha256': config['test_checkpoint_sha256']}},
            'sources': {'adapter': 'test-only-frozen'}, 'implementation_version': 'fixture-v1',
            'environment': {'executable': 'test-only', 'packages': {'numpy': np.__version__}}}


class SemanticRoutesIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prepared = self.root / 'prepared'
        self.prepared.mkdir()
        self.categories = [{'id': 6, 'name': 'CRACK', 'color': '#ff0000'},
                           {'id': 7, 'name': 'REPAIRS', 'color': '#00ff00'}]
        self.mapping = [{'category_id': 6, 'channel': 0}, {'category_id': 7, 'channel': 1}]
        sequences = []
        # Intentionally reversed manifest order: the runner must execute VAL first.
        for split, base in (('train', 20), ('test', 30), ('val', 40)):
            sequence = {'sequence_id': 'sequence-' + split, 'building_id': 'facade-' + split,
                        'split': split, 'shape': [16, 16], 'frames': []}
            for index in range(3):
                rgb = np.full((16, 16, 3), base, np.uint8)
                packed = np.full((16, 16), 2, np.uint16)  # Known REPAIRS everywhere.
                if index:
                    rgb[1:5, 1:5, 0] = 255 if index == 1 else 220
                    rgb[10:14, 10:14, 1] = 255 if index == 1 else 220
                    packed[1:5, 1:5] = 1
                    packed[10:14, 10:14] = 1
                relative = f'{split}-{index}.npz'
                checksum = atomic_npz(self.prepared / relative, {
                    'rgb': rgb, 'packed': packed, 'support': np.ones((16, 16), bool)})
                sequence['frames'].append({'frame': index, 'year': 2020 + index,
                                           'artifact_path': relative, 'artifact_sha256': checksum})
            sequences.append(sequence)
        manifest = {'schema_version': 1, 'kind': 'real_temporal_prepared',
                    'sequences': sequences, 'categories': self.categories,
                    'category_to_channel': self.mapping, 'source_run': str(self.root / 'old'),
                    'source_records': [], 'source_summary': {}}
        atomic_json(self.prepared / 'manifest.json', manifest)
        domain = [{'name': name, 'prompts': [name.lower()], 'group': 'target'}
                  for name in ('CRACK', 'REPAIRS')]
        self.config = {'prepared': str(self.prepared), 'cache_root': str(self.root / 'cache'),
                       'splits': ['train', 'test', 'val'], 'modes': ['previous'],
                       'routes': list(routes.ROUTES),
                       'change_methods': {'rgb_diff': {'options': {}, 'threshold': .1}},
                       'segmenters': {'frozen': {'kind': 'sam3', 'test_checkpoint_sha256': 'a' * 64}},
                       'vocabularies': {'domain': domain, 'expanded': domain + [
                           {'name': 'WALL', 'prompts': ['wall'], 'group': 'context'}]},
                       'roi': {'min_area': 1, 'max_regions': 4, 'padding': 0, 'min_size': 4},
                       'semantic': {'confidence': .5}, 'smoke': {'pairs_per_split': 1}}
        TestOnlySegmenterPool.calls = []
        for target, value in (('SemanticPool', TestOnlySegmenterPool),
                              ('preflight_remote', test_preflight), ('_snapshot_gpu', lambda out: None)):
            patcher = patch.object(routes, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_queue(self, name, config=None, **kwargs):
        with redirect_stdout(io.StringIO()):
            runner = routes.Runner(config or self.config, self.root / name, **kwargs)
            status = runner.run()
        failures = [r.get('error') for r in runner.records if r['status'] == 'failed']
        self.assertEqual(status, 0, failures)
        return runner

    def test_smoke_full_resume_all_routes_and_full_support_false_negatives(self):
        smoke = self.run_queue('smoke', smoke=True)
        self.assertTrue(TestOnlySegmenterPool.calls)
        smoke_count = len(TestOnlySegmenterPool.calls)
        self.assertEqual({r['split'] for r in smoke.records}, {'val', 'test', 'train'})
        self.assertEqual({r['variant'] for r in smoke.records if r['kind'] == 'segmentation'},
                         {'domain', 'expanded_flat', 'expanded_layered'})
        events = [json.loads(line) for line in (smoke.out / 'events.jsonl').read_text().splitlines()]
        self.assertEqual([r['split'] for r in events if r['event_type'] == 'split_completed'],
                         ['val', 'test', 'train'])
        full = self.run_queue('full')
        self.assertGreater(len(TestOnlySegmenterPool.calls), smoke_count)
        self.assertTrue(any(o['source'] == 'cache' for o in full.observations))
        relevant = {r['route']: r for r in full.records if r['kind'] == 'change' and r['split'] == 'val'
                    and r['frame'] == 1 and r['variant'] in ('baseline', 'domain')}
        self.assertTrue(set(routes.ROUTES) <= relevant.keys())
        for route in ('semantic_first_gate', 'semantic_first_rerun'):
            metric = next(m for m in relevant[route]['metrics'] if m['class_name'] == '__all_changes__')
            self.assertEqual(metric['valid_pixels'], 256)
            self.assertEqual(metric['tp'], 16)
            self.assertEqual(metric['fn'], 16)  # Missed region remains in denominator.
        metric = next(m for m in relevant['change_first']['metrics'] if m['class_name'] == '__all_changes__')
        self.assertEqual(metric['tp'], 32)
        self.assertEqual(metric['fn'], 0)
        self.assertTrue(any(r['route'] == 'change_first_cached_clip' for r in full.records))
        before = len(TestOnlySegmenterPool.calls)
        rerun = self.run_queue('full', resume=True)
        self.assertEqual(len(TestOnlySegmenterPool.calls), before)
        self.assertTrue(all(o['source'] == 'cache' for o in rerun.observations))
        self.assertTrue((rerun.out / 'report.html').is_file())
        self.assertTrue((rerun.out / 'estimate.json').is_file())
        self.assertEqual(read_json(rerun.out / 'run.json')['failed'], 0)

    def test_extending_method_reuses_existing_native_tasks_and_checkpoint_change_is_selective(self):
        initial = self.run_queue('initial')
        extended = copy.deepcopy(self.config)
        extended['change_methods']['lzma_abs'] = {
            'options': {'compression_tile_size': 4, 'compression_stride': 4, 'lzma_preset': 0},
            'expected_metadata': {'method': 'lzma_abs'}, 'threshold': .1}
        added = self.run_queue('extended', config=extended)
        old_rgb = {o['cache_key'] for o in initial.observations if o['method'] == 'rgb_diff'}
        new_rgb = [o for o in added.observations if o['method'] == 'rgb_diff']
        self.assertTrue(new_rgb)
        self.assertTrue(all(o['source'] == 'cache' and o['cache_key'] in old_rgb for o in new_rgb))
        self.assertTrue(any(o['method'] == 'lzma_abs' and o['source'] == 'measured' for o in added.observations))
        # A new detector may request different ROIs, but full frame inference is reusable.
        self.assertTrue(all(o['source'] == 'cache' for o in added.observations if o['stage'] == 'segment_full'))
        changed = copy.deepcopy(self.config)
        changed['segmenters']['frozen']['test_checkpoint_sha256'] = 'b' * 64
        prior = len(TestOnlySegmenterPool.calls)
        updated = self.run_queue('new-checkpoint', config=changed)
        self.assertGreater(len(TestOnlySegmenterPool.calls), prior)
        self.assertTrue(all(o['source'] == 'cache' for o in updated.observations if o['stage'] == 'change_full'))
        self.assertTrue(all(o['source'] == 'measured' for o in updated.observations if o['stage'] == 'segment_full'))

    def test_all_change_methods_blocked_stops_before_inference_and_full_run(self):
        config_path = self.root / 'config.json'
        atomic_json(config_path, self.config)
        output = self.root / 'blocked'
        with patch.object(routes, 'change_contract', side_effect=ValueError('missing provenance')), \
                redirect_stdout(io.StringIO()):
            result = routes.main(['run', '--config', str(config_path), '--out', str(output),
                                  '--smoke-first', '--resume'])
        self.assertEqual(result, 2)
        self.assertEqual(TestOnlySegmenterPool.calls, [])
        self.assertFalse((output / 'run.json').exists())
        smoke = read_json(output / 'smoke' / 'run.json')
        self.assertEqual(smoke['status'], 'blocked')
        self.assertIn('change:rgb_diff', smoke['blocked'])
        coverage = read_json(output / 'smoke' / 'coverage_matrix.json')
        self.assertTrue(any(row['not_attempted'] for row in coverage))

    def test_threshold_failure_is_blocked_even_with_resolved_contract(self):
        with patch.object(routes, 'fixed_threshold', side_effect=ValueError('no threshold')), \
                redirect_stdout(io.StringIO()):
            runner = routes.Runner(self.config, self.root / 'threshold-blocked', smoke=True)
            self.assertEqual(runner.run(), 2)
        self.assertIn('rgb_diff', runner.contracts)
        self.assertEqual(TestOnlySegmenterPool.calls, [])

    def test_full_pair_reads_request_only_scores(self):
        original_get = routes.Cache.get
        requested = []

        def tracked_get(cache, spec, **kwargs):
            if kwargs.get('names') is not None:
                requested.append(kwargs['names'])
            return original_get(cache, spec, **kwargs)

        with patch.object(routes.Cache, 'get', tracked_get):
            self.run_queue('selective', smoke=True)
        self.assertTrue(requested)
        self.assertTrue(all(names == ['scores'] for names in requested))

    def test_every_runtime_change_failure_stops_before_segmentation(self):
        with patch.object(routes.Runner, 'change', side_effect=RuntimeError('model cannot load')), \
                redirect_stdout(io.StringIO()):
            runner = routes.Runner(self.config, self.root / 'runtime-blocked', smoke=True)
            self.assertEqual(runner.run(), 2)
        self.assertTrue(runner.records)
        self.assertTrue(all(row['status'] == 'failed' for row in runner.records))
        self.assertEqual(TestOnlySegmenterPool.calls, [])
        self.assertEqual(read_json(runner.out / 'run.json')['status'], 'blocked')

    def test_interrupt_records_state_without_starting_full_queue(self):
        config_path = self.root / 'interrupt-config.json'
        atomic_json(config_path, self.config)
        output = self.root / 'interrupt'
        with patch.object(routes.Runner, 'run_split', side_effect=KeyboardInterrupt), \
                redirect_stdout(io.StringIO()):
            result = routes.main(['run', '--config', str(config_path), '--out', str(output), '--smoke-first'])
        self.assertEqual(result, 130)
        self.assertEqual(read_json(output / 'smoke' / 'run.json')['status'], 'interrupted')
        self.assertFalse((output / 'run.json').exists())


if __name__ == '__main__':
    unittest.main()
