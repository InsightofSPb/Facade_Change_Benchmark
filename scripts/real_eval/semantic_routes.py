#!/usr/bin/env python3
"""Frozen VAL → TEST → TRAIN semantic/change routes with shared task caches.

Run as ``python -m scripts.real_eval.semantic_routes --help``. All model work
is isolated in serial workers. Annotations stay in this parent evaluator.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from contextlib import contextmanager
import copy
import fcntl
import json
from pathlib import Path
import sys
import time

import numpy as np

from facade_change.methods.registry import ALL_METHODS, _EXTERNAL_ADAPTERS
from facade_change.methods.remote import RemotePool
from . import night_common
from .night_run import _caches
from .real_gt_metrics import calibration
from .robustness_common import atomic_json, digest, read_json, sha256
from .robustness_run import (NATIVE_METHODS, _event, _file_hashes, _load_output,
                             _method_contract, _snapshot_gpu, _verify_metadata,
                             model_identity, payload_sha256, validate_output)
from .semantic_cache import Cache, hash_array
from .semantic_geometry import regions, semantic_prediction, stitch, target_union
from .semantic_remote import SemanticPool, preflight_remote
from .semantic_report import change_metrics, render_report, segmentation_metrics
from .semantic_timing import estimate

SCHEMA = '2026-10-18-semantic-routes-v1'
ROUTES = ('change_first', 'semantic_first_gate', 'semantic_first_rerun')
SPLITS = ('val', 'test', 'train')


@contextmanager
def run_lock(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'run.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another runner owns {out}; its processes were not touched') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def variants(config):
    default = ['fixed'] if config['kind'] in ('finetuned_maskclip', 'dino_semantic', 'semantic_dino') else [
        'domain', 'expanded_flat', 'expanded_layered']
    result = config.get('variants', default)
    if not result or set(result) - {'fixed', 'domain', 'expanded_flat', 'expanded_layered'}:
        raise ValueError(f'Invalid variants: {result}')
    if ('fixed' in result) != (default == ['fixed']):
        raise ValueError('Frozen trained heads must retain their fixed ontology')
    return list(dict.fromkeys(result))


def vocabulary_key(variant):
    return 'expanded' if variant.startswith('expanded_') else 'domain'


def selected_pairs(manifest, split, modes, smoke=False, count=2):
    pairs = [(s, f) for s in manifest['sequences'] if s['split'] == split
             for f in range(1, len(s['frames']))]
    if smoke and pairs:
        indices = np.linspace(0, len(pairs)-1, min(count, len(pairs))).astype(int)
        pairs = [pairs[int(i)] for i in indices]
    return [(s, f, (f-1 if mode == 'previous' else 0), mode)
            for mode in modes for s, f in pairs]


def fixed_threshold(manifest, method, entry, mode):
    if method in NATIVE_METHODS:
        return {'kind': 'native_prediction', 'value': None}
    value = entry.get('threshold')
    if isinstance(value, dict):
        value = value.get(mode)
    if value is not None:
        if not isinstance(value, (float, int)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{method}: threshold must lie in [0,1]')
        return {'kind': 'explicit_frozen', 'value': float(value)}
    selected = calibration(manifest.get('source_summary', {}),
                           (method, 'current_train' if method.startswith('msdzip_') else 'frozen',
                            mode, 'step_event' if mode == 'previous' else 'earliest_state'), .01)
    if not selected:
        raise ValueError(f'{method}/{mode}: no frozen threshold; set change_methods.{method}.threshold explicitly')
    return {'kind': 'source_calibration', 'value': float(selected['threshold']), 'provenance': selected}


def binary_change(native, threshold, support):
    if threshold['kind'] == 'native_prediction':
        predicted = native['native_prediction'].copy()
    else:
        predicted = native['scores'] > threshold['value']
    return predicted & support


def change_contract(method, entry, manifest, device):
    if method not in ALL_METHODS:
        raise ValueError(f'Unregistered change method {method!r}; extend facade_change.methods.registry first')
    if 'options' not in entry:
        return _method_contract(method, manifest, Path(manifest['source_run']), device)
    # Explicit extension path for a registry method absent from the source run.
    # Every learned method must supply its existing provenance, not just a name.
    options = copy.deepcopy(entry['options'])
    expected = entry.get('expected_metadata', {})
    if entry.get('provenance_file'):
        expected = read_json(Path(entry['provenance_file']))
    if method not in ('rgb_diff', 'ssim') and not expected:
        raise ValueError(f'{method}: explicit options require expected_metadata or provenance_file')
    if any(options.get(key) for key in ('train', 'auto_train', 'download', 'download_weights', 'download-weights')):
        raise ValueError('Training and downloads are disabled')
    files = _file_hashes(expected)
    for path, checksum in files.items():
        if not Path(path).is_file() or sha256(Path(path)) != checksum:
            raise ValueError(f'{method}: missing or changed model file {path}')
    if device is not None and method not in ('rgb_diff', 'ssim'):
        options['device'] = device
    return {'options': options, 'expected_metadata': expected, 'model_files_sha256': files}


def method_source_identity(method):
    """Bind native cache to the selected adapter, not the entire method list.

    Source-run contracts additionally retain their recorded package provenance.
    Classical explicitly configured methods also need current code hashes.
    """
    package = Path(__file__).resolve().parents[2] / 'facade_change'
    module = (_EXTERNAL_ADAPTERS[method][0] if method in _EXTERNAL_ADAPTERS else
              'rgb' if method in ('rgb_diff', 'ssim') else
              'msdzip' if method.startswith('msdzip_') else 'compression')
    paths = ['method_worker.py', 'methods/base.py', f'methods/{module}.py']
    if method.startswith('msdzip_'):
        paths.extend(['2026-10-04_msdzip_h0.py', 'methods/compression.py'])
    return {name: sha256(package / name) for name in paths if (package / name).is_file()}


class Runner:
    def __init__(self, config, out, *, smoke=False, resume=False, cache_dir=None, pairs_per_split=None):
        self.config = copy.deepcopy(config)
        self.out = Path(out).expanduser().resolve()
        self.smoke = smoke
        self.resume = resume
        self.prepared, self.manifest = night_common.load_prepared(config['prepared'])
        self.categories = self.manifest['categories']
        self.mapping = self.manifest.get('category_to_channel', [
            {'category_id': c['id'], 'channel': i} for i, c in enumerate(self.categories)])
        self.splits = [s for s in SPLITS if s in config.get('splits', SPLITS)]
        self.modes = config.get('modes', ['previous'])
        self.routes = config.get('routes', list(ROUTES))
        self.roi = {**dict(min_area=16, max_regions=8, padding=16, min_size=56), **config.get('roi', {})}
        self.roi.pop('threshold', None)  # CD uses its frozen/native threshold, not a new ROI threshold.
        self.confidence = float(config.get('semantic', {}).get('confidence', .25))
        self.pairs_per_split = int(pairs_per_split or config.get('smoke', {}).get('pairs_per_split', 2))
        self.methods = config['change_methods']
        self.segmenters = config['segmenters']
        if not self.splits or set(config.get('splits', SPLITS)) - set(SPLITS):
            raise ValueError('Select val/test/train splits')
        if set(self.modes) - {'previous', 'earliest'} or not self.modes:
            raise ValueError('Pair routes support previous/earliest; history has a separate causal runner')
        if set(self.routes) - set(ROUTES) or self.pairs_per_split < 1:
            raise ValueError('Invalid routes or smoke sample count')
        if not self.methods or not self.segmenters or not 0 <= self.confidence <= 1:
            raise ValueError('Need methods, segmenters and a valid frozen confidence')
        regions(np.zeros((1, 1), bool), **self.roi)  # CPU validation before opening any worker.
        for cfg in self.segmenters.values():
            for variant in variants(cfg):
                if not config['vocabularies'].get(vocabulary_key(variant)):
                    raise ValueError(f'Missing vocabulary for {variant}')
        self.cache = Cache(cache_dir or config.get('cache_root') or self.prepared.parent / 'semantic-cache-v1',
                           event=self.event)
        self.records, self.observations, self.coverage = [], [], []
        self.completed_tasks = defaultdict(set)
        self.contracts, self.preflights, self.blocked, self.thresholds = {}, {}, {}, {}
        self.method_sources = {m: method_source_identity(m) for m in self.methods if m in ALL_METHODS}
        self.frames = {}
        self.source_cache = defaultdict(list)
        self.cd_pool = RemotePool(log_dir=self.out / 'logs')
        self.sem_pool = SemanticPool(log_dir=self.out / 'logs')
        self.revision = digest({'config': self.config, 'prepared': sha256(self.prepared / 'manifest.json'),
                                'schema': SCHEMA})
        self.report_code = {name: sha256(Path(__file__).with_name(name)) for name in (
            'semantic_routes.py', 'semantic_geometry.py', 'semantic_report.py')}
        self.started = time.time()

    def event(self, kind, **fields):
        _event(self.out, kind, **fields)

    def frame(self, sequence, frame):
        key = (sequence['sequence_id'], frame)
        if key not in self.frames:
            self.frames[key] = night_common.load_frame(self.prepared, sequence['frames'][frame])
        return self.frames[key]

    def context(self, sequence, frame, reference_frame=None, mode='previous', **fields):
        return dict(run_id=self.out.name, config_revision=self.revision, split=sequence['split'],
                    building_id=sequence['building_id'], sequence_id=sequence['sequence_id'],
                    frame=frame, reference_frame=reference_frame, mode=mode, **fields)

    def record(self, context, arrays=None, metrics=None, **fields):
        record = {**context, **fields, 'status': fields.get('status', 'completed'), 'metrics': metrics or []}
        if arrays is not None:
            spec = {'stage': 'evaluation', 'schema': SCHEMA, 'code': self.report_code,
                    'context': {k: v for k, v in context.items() if k not in ('run_id', 'config_revision')},
                    'arrays': {k: hash_array(v) for k, v in arrays.items()}}
            saved = self.cache.put(spec, arrays)
            record['cache_artifact'] = str(self.cache.root / saved['artifact_path'])
            record['cache_key'] = saved['cache_key']
        # Keep earlier configuration revisions reviewable when a method is added
        # or a frozen operating point changes inside the same output directory.
        unit_id = digest({k: v for k, v in context.items() if k != 'run_id'})
        record['unit_id'] = unit_id
        atomic_json(self.out / 'units' / (unit_id + '.json'), record)
        self.records.append(record)
        return record

    def fail(self, context, error):
        self.event('unit_failed', context=context, error=str(error))
        return self.record(context, status='failed', error=str(error))

    def preflight(self):
        # Verify selected prepared RGB/support/GT artifacts once before any model.
        for sequence in self.manifest['sequences']:
            if sequence['split'] in self.splits:
                for frame in sequence['frames']:
                    night_common.load_frame(self.prepared, frame)
        for method, entry in self.methods.items():
            try:
                self.contracts[method] = change_contract(method, entry, self.manifest, self.config.get('device'))
                for mode in self.modes:
                    self.thresholds[method, mode] = fixed_threshold(self.manifest, method, entry, mode)
            except Exception as exc:
                self.blocked['change:' + method] = str(exc)
        for name, cfg in self.segmenters.items():
            try:
                adapter = {k: v for k, v in cfg.items() if k not in ('variants', 'name', 'description', 'confidence')}
                result = preflight_remote(adapter)
                self.preflights[name] = result
                if not result['ok']:
                    raise ValueError('; '.join(result['errors']))
            except Exception as exc:
                self.blocked['segment:' + name] = str(exc)
        # Use the proven source/night cache loader; every candidate is revalidated
        # against RGB, support, checksums, method options and model provenance.
        for source in [None, *self.config.get('reuse_runs', [])]:
            try:
                index = _caches(self.manifest, None, source)
                for key, rows in index.items():
                    for row in rows:
                        if row not in self.source_cache[key]:
                            self.source_cache[key].append(row)
            except Exception as exc:
                self.event('source_cache_unavailable', path=source, error=str(exc))
        exposure = night_common.training_exposure(self.manifest, self.contracts)
        atomic_json(self.out / 'preflight.json', dict(
            contracts=self.contracts, segmenters=self.preflights, blocked=self.blocked,
            thresholds={f'{m}/{mode}': v for (m, mode), v in self.thresholds.items()},
            training_exposure=exposure,
            legacy_training_exposure='unverified; all outputs exploratory on this selected subset'))
        self.event('preflight', blocked=self.blocked)
        for name, error in self.blocked.items():
            print(f'BLOCKED {name}: {error}', flush=True)
        self.inventory_full_cache()

    @staticmethod
    def timing_group(stage, method=None, segmenter=None, variant=None):
        return (stage, method, segmenter, variant)

    def change_spec(self, method, inputs):
        contract = self.contracts[method]
        return dict(stage='change_native', schema=SCHEMA, method=method,
                    model=model_identity(contract['expected_metadata']), options=contract['options'],
                    files=contract['model_files_sha256'], source=self.method_sources[method],
                    payload=payload_sha256(inputs))

    def segment_spec(self, name, vocab_key, rgb):
        return dict(stage='segmentation_native', schema=SCHEMA, model=self.preflights[name],
                    vocabulary=self.config['vocabularies'][vocab_key], rgb=hash_array(rgb))

    def import_change(self, method, inputs, context, import_key):
        contract = self.contracts[method]
        for root, row, _ in self.source_cache.get(import_key, []):
            if row.get('building_id') != context['building_id'] or row.get('split') != context['split']:
                continue
            try:
                _verify_metadata(contract['expected_metadata'], row.get('method_metadata', {}), method)
                recorded_code = row.get('method_metadata', {}).get('worker_environment', {}).get('source_sha256', {})
                if any(name in recorded_code and recorded_code[name] != value
                       for name, value in self.method_sources[method].items()):
                    raise ValueError('Source cache was computed with a different selected adapter implementation')
                native = _load_output(root, row, inputs, method)
                arrays = {k: native[k] for k in ('scores', 'raw_scores', 'native_prediction') if k in native}
                metadata = dict(method_metadata=row.get('method_metadata', {}), imported_from=str(root),
                                source_record=row.get('artifact_path'), elapsed_seconds=None, init_seconds=None)
                record = self.cache.put(self.change_spec(method, inputs), arrays, metadata=metadata)
                return arrays, record
            except Exception as exc:
                self.event('source_cache_rejected', method=method, path=str(root), error=str(exc))
        return None

    def inventory_full_cache(self):
        """CPU-only verification of all reusable full-image work before ETA.

        A smoke run must not forecast 56 fresh inferences when those 56 maps
        already exist in the wide run. No missing task is inferred here.
        """
        print('Verifying full-image cache coverage (CPU only)', flush=True)
        for split in self.splits:
            for sequence, frame, ref, mode in selected_pairs(self.manifest, split, self.modes):
                reference, source = self.frame(sequence, ref), self.frame(sequence, frame)
                inputs = dict(reference=reference['rgb'], source=source['rgb'], support=reference['support'] & source['support'])
                for method in self.methods:
                    if 'change:' + method in self.blocked:
                        continue
                    context = self.context(sequence, frame, ref, mode, method=method)
                    saved = self.cache.get(self.change_spec(method, inputs))
                    origin = 'cache'
                    if saved is None:
                        saved = self.import_change(method, inputs, context, (method, mode, sequence['sequence_id'], frame))
                        origin = 'imported'
                    if saved is not None:
                        self.observe('change_full', saved[1], context, origin, inputs['support'].shape, inventory=True)
            for sequence in self.manifest['sequences']:
                if sequence['split'] != split:
                    continue
                for frame in range(len(sequence['frames'])):
                    rgb = self.frame(sequence, frame)['rgb']
                    for name, cfg in self.segmenters.items():
                        if 'segment:' + name in self.blocked:
                            continue
                        for variant in {vocabulary_key(v) for v in variants(cfg)}:
                            saved = self.cache.get(self.segment_spec(name, variant, rgb))
                            if saved is not None:
                                self.observe('segment_full', saved[1], self.context(sequence, frame,
                                             segmenter=name, native_variant=variant), 'cache', rgb.shape[:2], inventory=True)
            self.frames.clear()

    def observe(self, stage, record, context, source, shape, **extra):
        metadata = record.get('metadata', {})
        row = dict(stage=stage, method=context.get('method'), segmenter=context.get('segmenter'),
                   variant=context.get('native_variant'), split=context['split'], shape=list(shape),
                   cache_key=record['cache_key'], source=source, status='completed',
                   elapsed_seconds=metadata.get('elapsed_seconds'), init_seconds=metadata.get('init_seconds'), **extra)
        row['logical_id'] = digest([context.get(k) for k in (
            'sequence_id', 'frame', 'reference_frame', 'mode', 'method', 'segmenter', 'native_variant')])
        if stage in ('change_full', 'segment_full') and not row.get('timing_only'):
            self.completed_tasks[self.timing_group(stage, row['method'], row['segmenter'], row['variant'])].add(row['logical_id'])
        self.observations.append(row)
        return row

    def change(self, method, inputs, context, *, import_key=None):
        if 'change:' + method in self.blocked:
            raise RuntimeError(self.blocked['change:' + method])
        contract = self.contracts[method]
        spec = self.change_spec(method, inputs)
        saved = self.cache.get(spec)
        origin = 'cache'
        if saved is None:
            if import_key:
                saved = self.import_change(method, inputs, context, import_key)
                if saved is not None:
                    origin = 'imported'
            if saved is None:
                self.sem_pool.release()
                before = self.cd_pool.process
                start = time.perf_counter()
                actual = self.cd_pool.activate(method, contract['options'])
                init = time.perf_counter()-start if self.cd_pool.process is not before else None
                _verify_metadata(contract['expected_metadata'], actual, method)
                start = time.perf_counter()
                native = self.cd_pool.score(method, contract['options'], **inputs)
                elapsed = time.perf_counter()-start
                validate_output(native, inputs['support'], method)
                metadata = dict(method_metadata=native['metadata'], elapsed_seconds=elapsed, init_seconds=init)
                origin = 'measured'
                arrays = {k: native[k] for k in ('scores', 'raw_scores', 'native_prediction') if k in native}
                record = self.cache.put(spec, arrays, metadata=metadata)
                saved = arrays, record
        arrays, record = saved
        validate_output(arrays, inputs['support'], method)
        self.observe(context['stage'], record, context, origin, inputs['support'].shape,
                     timing_only=context.get('timing_only', False))
        return arrays, record

    def segment(self, name, vocab_key, rgb, context):
        if 'segment:' + name in self.blocked:
            raise RuntimeError(self.blocked['segment:' + name])
        checked = self.preflights[name]
        vocabulary = self.config['vocabularies'][vocab_key]
        spec = self.segment_spec(name, vocab_key, rgb)
        saved = self.cache.get(spec)
        origin = 'cache'
        if saved is None:
            self.cd_pool.release()
            before = self.sem_pool.process
            native = self.sem_pool.predict(checked['config'], vocabulary, rgb)
            actual_provenance = native.get('metadata', {}).get('provenance', {})
            if any(actual_provenance.get(key) != checked.get(key)
                   for key in ('files', 'sources', 'implementation_version')):
                raise ValueError('Semantic model/source provenance changed between preflight and loading')
            arrays = {'scores': native['scores'], **{
                'extra__' + k: v for k, v in native.get('extra_arrays', {}).items()}}
            metadata = dict(class_names=native['class_names'], score_kind=native['score_kind'],
                            method_metadata=native.get('metadata', {}),
                            elapsed_seconds=native.get('request_elapsed_seconds', native['elapsed_seconds']),
                            init_seconds=(native.get('metadata', {}).get('activation_seconds')
                                          if self.sem_pool.process is not before else None))
            record = self.cache.put(spec, arrays, metadata=metadata)
            saved, origin = (arrays, record), 'measured'
        arrays, record = saved
        metadata = record['metadata']
        native = dict(scores=arrays['scores'], class_names=metadata['class_names'],
                      score_kind=metadata['score_kind'], metadata=metadata['method_metadata'],
                      extra_arrays={k[len('extra__'):]: v for k, v in arrays.items() if k.startswith('extra__')})
        self.observe(context['stage'], record, {**context, 'native_variant': vocab_key}, origin, rgb.shape[:2],
                     timing_only=context.get('timing_only', False))
        return native, record

    def sem_metrics(self, values, frame):
        scores, predicted, names = values
        return segmentation_metrics(scores, predicted, names, frame['packed'], frame['support'],
                                    self.categories, self.mapping)

    def cd_metrics(self, prediction, scores, reference, source, support):
        return change_metrics(prediction, scores, reference['packed'], source['packed'], support,
                              self.categories, self.mapping)

    def save_sem(self, context, values, frame, **fields):
        scores, predicted, names = values
        return self.record(context, dict(scores=scores, predictions=predicted),
                           self.sem_metrics(values, frame), class_names=names, **fields)

    def save_change(self, context, prediction, scores, reference, source, support, **fields):
        return self.record(context, dict(scores=scores, prediction=prediction),
                           self.cd_metrics(prediction, scores, reference, source, support), **fields)

    def workload(self):
        """Exact full-image counts and explicit worst-case ROI call counts."""
        pair_count = sum(len(selected_pairs(self.manifest, s, self.modes)) for s in self.splits)
        frame_count = sum(len(s['frames']) for s in self.manifest['sequences'] if s['split'] in self.splits)
        rows = []
        for method in self.methods:
            rows.append(dict(stage='change_full', method=method, segmenter=None, variant=None,
                             units=pair_count, model_initializations=len(self.splits)))
        for name, cfg in self.segmenters.items():
            native_variants = sorted({vocabulary_key(v) for v in variants(cfg)})
            for variant in native_variants:
                rows.append(dict(stage='segment_full', method=None, segmenter=name, variant=variant,
                                 units=frame_count, model_initializations=len(self.splits)))
                if 'change_first' in self.routes:
                    maximum = pair_count * len(self.methods) * 2 * self.roi['max_regions']
                    rows.append(dict(stage='segment_roi', method=None, segmenter=name, variant=variant,
                                     units=maximum, roi_upper_bound=maximum, model_initializations=0))
            if 'semantic_first_rerun' in self.routes:
                for variant in variants(cfg):
                    for method in self.methods:
                        maximum = pair_count * self.roi['max_regions']
                        rows.append(dict(stage='change_roi', method=method, segmenter=name, variant=variant,
                                         units=maximum, roi_upper_bound=maximum,
                                         model_initializations=0))
        return rows

    def run_split(self, split):
        pairs = selected_pairs(self.manifest, split, self.modes, self.smoke, self.pairs_per_split)
        if not pairs:
            return
        print(f'{split.upper()}: {len(pairs)} pairs; fixed settings; {len(self.methods)} change methods', flush=True)
        baseline, rerun_jobs = {}, []
        for method in self.methods:
            for sequence, frame, ref, mode in pairs:
                context = self.context(sequence, frame, ref, mode, kind='change', method=method,
                                       segmenter=None, variant='baseline', route='baseline')
                try:
                    reference, source = self.frame(sequence, ref), self.frame(sequence, frame)
                    inputs = dict(reference=reference['rgb'], source=source['rgb'], support=reference['support'] & source['support'])
                    native, saved = self.change(method, inputs, {**context, 'stage': 'change_full'},
                                                import_key=(method, mode, sequence['sequence_id'], frame))
                    prediction = binary_change(native, self.thresholds[method, mode], inputs['support'])
                    baseline[method, mode, sequence['sequence_id'], frame] = (saved, regions(prediction, **self.roi))
                    self.save_change(context, prediction, native['scores'], reference, source, inputs['support'],
                                     native_cache_key=saved['cache_key'], threshold=self.thresholds[method, mode])
                except Exception as exc:
                    self.fail(context, exc)
        self.cd_pool.release()
        frames = {(s['sequence_id'], f): s for s, frame, ref, _ in pairs for f in (ref, frame)}
        for name, cfg in self.segmenters.items():
            for vocab_key in sorted({vocabulary_key(v) for v in variants(cfg)}):
                selected_variants = [v for v in variants(cfg) if vocabulary_key(v) == vocab_key]
                vocabulary = self.config['vocabularies'][vocab_key]
                confidence = float(cfg.get('confidence', self.confidence))
                full = {}
                print(f'{split.upper()}: segmenter={name}, vocabulary={vocab_key}', flush=True)
                for (seqid, frame), sequence in frames.items():
                    context = self.context(sequence, frame, kind='segmentation', method=None,
                                           segmenter=name, variant=vocab_key, route='baseline')
                    try:
                        native, saved = self.segment(name, vocab_key, self.frame(sequence, frame)['rgb'],
                                                     {**context, 'stage': 'segment_full'})
                        full[seqid, frame] = saved
                        for variant in selected_variants:
                            values = semantic_prediction(native, vocabulary, variant, confidence)
                            self.save_sem({**context, 'variant': variant}, values, self.frame(sequence, frame),
                                          native_cache_key=saved['cache_key'],
                                          native_cache_artifact=str(self.cache.root / saved['artifact_path']),
                                          native_method_metadata=native.get('metadata', {}), confidence=confidence)
                    except Exception as exc:
                        for variant in selected_variants:
                            self.fail({**context, 'variant': variant}, exc)
                        self.blocked['segment:' + name] = str(exc)
                for sequence, frame, ref, mode in pairs:
                    if (sequence['sequence_id'], frame) not in full or (sequence['sequence_id'], ref) not in full:
                        for variant in selected_variants:
                            for method in self.methods:
                                for route in self.routes:
                                    self.fail(self.context(sequence, frame, ref, mode, kind='change', method=method,
                                                           segmenter=name, variant=variant, route=route),
                                              self.blocked.get('segment:' + name, 'Full segmentation unavailable'))
                        continue
                    reference, source = self.frame(sequence, ref), self.frame(sequence, frame)
                    support = reference['support'] & source['support']
                    # Read cached full semantic outputs without re-entering the worker.
                    full_values = {}
                    for index in (ref, frame):
                        saved = full[sequence['sequence_id'], index]
                        arrays, checked = self.cache.get(saved['spec'])
                        meta = checked['metadata']
                        native = dict(scores=arrays['scores'], class_names=meta['class_names'], score_kind=meta['score_kind'],
                                      metadata=meta['method_metadata'],
                                      extra_arrays={k[7:]: v for k, v in arrays.items() if k.startswith('extra__')})
                        full_values[index] = {v: semantic_prediction(native, vocabulary, v, confidence)
                                              for v in selected_variants}
                    for method in self.methods:
                        base = baseline.get((method, mode, sequence['sequence_id'], frame))
                        if base is None:
                            for variant in selected_variants:
                                for route in self.routes:
                                    self.fail(self.context(sequence, frame, ref, mode, kind='change', method=method,
                                                           segmenter=name, variant=variant, route=route), 'Baseline change unavailable')
                            continue
                        saved, boxes = base
                        native_change, _ = self.cache.get(saved['spec'])
                        prediction = binary_change(native_change, self.thresholds[method, mode], support)
                        roi_native, roi_error = {}, None
                        if 'change_first' in self.routes:
                            self.coverage.append(dict(stage='segment_roi', method=None, segmenter=name, variant=vocab_key,
                                split=split, pair=[sequence['sequence_id'], frame, mode], change_method=method,
                                calls=2*len(boxes), status='regions' if boxes else 'no_regions'))
                            try:
                                for index, data in ((ref, reference), (frame, source)):
                                    roi_native[index] = []
                                    for box in boxes:
                                        x0, y0, x1, y1 = box
                                        native, _ = self.segment(name, vocab_key, data['rgb'][y0:y1, x0:x1],
                                            self.context(sequence, frame, ref, mode, stage='segment_roi', method=None, segmenter=name))
                                        roi_native[index].append((box, native))
                            except Exception as exc:
                                roi_error = str(exc)
                        for variant in selected_variants:
                            context = self.context(sequence, frame, ref, mode, kind='change', method=method,
                                                   segmenter=name, variant=variant)
                            old, new = full_values[ref][variant], full_values[frame][variant]
                            gate = (target_union(old[1], old[2], self.categories) |
                                    target_union(new[1], new[2], self.categories)) & support
                            if 'semantic_first_gate' in self.routes:
                                scores = np.where(gate, native_change['scores'], 0).astype(np.float32)
                                scores[~support] = np.nan
                                self.save_change({**context, 'route': 'semantic_first_gate'}, prediction & gate, scores,
                                                 reference, source, support, gate_pixels=int(gate.sum()),
                                                 inference_reused=True, compute_saving_claim=False)
                            if 'semantic_first_rerun' in self.routes:
                                semantic_boxes = regions(gate, **self.roi)
                                rerun_jobs.append((context, sequence, frame, ref, semantic_boxes))
                                self.coverage.append(dict(stage='change_roi', method=method, segmenter=name, variant=variant,
                                    split=split, pair=[sequence['sequence_id'], frame, mode], calls=len(semantic_boxes),
                                    status='regions' if semantic_boxes else 'no_regions'))
                            if 'change_first' in self.routes:
                                if roi_error:
                                    self.fail({**context, 'route': 'change_first'}, roi_error)
                                else:
                                    roi_values = {}
                                    for index in (ref, frame):
                                        parts = []
                                        names = full_values[index][variant][2]
                                        for box, native in roi_native[index]:
                                            scores, masks, local_names = semantic_prediction(native, vocabulary, variant, confidence)
                                            if local_names != names:
                                                raise ValueError('ROI segmentation changed its ontology')
                                            parts.append((box, scores, masks))
                                        scores, masks, covered = stitch(parts, support.shape, len(names))
                                        roi_values[index] = (scores, masks, names)
                                        self.save_sem({**context, 'kind': 'segmentation', 'frame': index,
                                                       'pair_frame': frame, 'route': 'change_first_roi'},
                                                      roi_values[index], self.frame(sequence, index),
                                                      boxes=[list(b) for b in boxes], covered_pixels=int(covered.sum()))
                                        # Cheap clipping of whole-crop semantics is kept as a
                                        # separate ablation; it is never called ROI inference.
                                        full_scores, full_masks, _ = full_values[index][variant]
                                        clipped = (np.where(covered[None], full_scores, 0).astype(np.float32),
                                                   full_masks & covered[None], names)
                                        self.save_sem({**context, 'kind': 'segmentation', 'frame': index,
                                                       'pair_frame': frame, 'route': 'change_first_cached_clip'},
                                                      clipped, self.frame(sequence, index), inference_reused=True)
                                    before_masks, after_masks = roi_values[ref][1], roi_values[frame][1]
                                    names = roi_values[frame][2]
                                    interpretation = [{
                                        'class_name': n, 'before_pixels': int((before_masks[i] & prediction).sum()),
                                        'after_pixels': int((after_masks[i] & prediction).sum()),
                                        'gain_pixels': int((~before_masks[i] & after_masks[i] & prediction).sum()),
                                        'loss_pixels': int((before_masks[i] & ~after_masks[i] & prediction).sum())}
                                        for i, n in enumerate(names)]
                                    self.save_change({**context, 'route': 'change_first'}, prediction,
                                                     native_change['scores'], reference, source, support,
                                                     interpretation=interpretation,
                                                     prediction_policy='preserve original change mask; attach ROI semantics',
                                                     boxes=[list(b) for b in boxes])
                                    relevant = (target_union(before_masks, names, self.categories) |
                                                target_union(after_masks, names, self.categories)) & support
                                    refined_scores = np.where(relevant, native_change['scores'], 0).astype(np.float32)
                                    refined_scores[~support] = np.nan
                                    self.save_change({**context, 'route': 'change_first_refined'}, prediction & relevant,
                                                     refined_scores, reference, source, support,
                                                     prediction_policy='separate ablation: require a target semantic class in either date',
                                                     boxes=[list(b) for b in boxes])
                # Timing-only deterministic probe exercises the implementation even
                # when no predicted change region exists. It never enters metrics.
                if self.smoke and 'change_first' in self.routes and 'segment:' + name not in self.blocked:
                    if not any(o['stage'] == 'segment_roi' and o['segmenter'] == name and o['variant'] == vocab_key
                               and o['split'] == split for o in self.observations):
                        sequence, frame, ref, mode = pairs[0]
                        rgb = self.frame(sequence, frame)['rgb']
                        h, w = rgb.shape[:2]
                        y0, x0 = h//4, w//4
                        try:
                            self.segment(name, vocab_key, rgb[y0:h-y0, x0:w-x0],
                                self.context(sequence, frame, ref, mode, stage='segment_roi', method=None,
                                             segmenter=name, timing_only=True))
                        except Exception as exc:
                            self.event('timing_probe_failed', segmenter=name, variant=vocab_key, error=str(exc))
                self.sem_pool.release()
        # Group RGB crop rescoring by CD method after every segmenter is released.
        for method in self.methods:
            for context, sequence, frame, ref, boxes in (j for j in rerun_jobs if j[0]['method'] == method):
                context = {**context, 'route': 'semantic_first_rerun'}
                reference, source = self.frame(sequence, ref), self.frame(sequence, frame)
                support = reference['support'] & source['support']
                try:
                    parts = []
                    for box in boxes:
                        x0, y0, x1, y1 = box
                        local_support = support[y0:y1, x0:x1]
                        if not local_support.any():
                            continue
                        inputs = dict(reference=reference['rgb'][y0:y1, x0:x1],
                                      source=source['rgb'][y0:y1, x0:x1], support=local_support)
                        native, _ = self.change(method, inputs, {**context, 'stage': 'change_roi',
                                                                 'native_variant': context['variant']})
                        parts.append((box, native['scores'], binary_change(native, self.thresholds[method, context['mode']], local_support)))
                    scores, prediction, covered = stitch(parts, support.shape)
                    scores[~support] = np.nan
                    prediction &= support
                    self.save_change(context, prediction, scores, reference, source, support,
                                     boxes=[list(b) for b in boxes], covered_pixels=int(covered.sum()),
                                     evaluation_domain='full common geometric support; no ROI-conditioned denominator')
                    if self.smoke and not boxes:
                        # Fixed, label-free RGB probe; explicitly excluded from metrics.
                        h, w = support.shape
                        y0, x0 = h//4, w//4
                        local_support = support[y0:h-y0, x0:w-x0]
                        if local_support.any():
                            self.change(method, dict(reference=reference['rgb'][y0:h-y0, x0:w-x0],
                                source=source['rgb'][y0:h-y0, x0:w-x0], support=local_support),
                                {**context, 'stage': 'change_roi', 'native_variant': context['variant'], 'timing_only': True})
                except Exception as exc:
                    self.fail(context, exc)
            self.cd_pool.release()
        self.frames.clear()
        self.event('split_completed', split=split)
        self.write_reports()

    def write_reports(self, final=False):
        atomic_json(self.out / 'timing_observations.json', self.observations)
        workload = self.workload()
        # Samples imported from another run have no measured timing. Count only
        # unique, verified native cache entries observed here as already done.
        for group in workload:
            matching = [o for o in self.observations if all(o.get(k) == group.get(k)
                        for k in ('stage', 'method', 'segmenter', 'variant'))]
            key = self.timing_group(group['stage'], group.get('method'), group.get('segmenter'), group.get('variant'))
            completed = (len(self.completed_tasks[key]) if group['stage'] in ('change_full', 'segment_full')
                         else len({o['cache_key'] for o in matching if not o.get('timing_only')}))
            group['cached_units'] = min(group['units'], completed)
            if group['stage'] in ('segment_roi', 'change_roi'):
                estimates = []
                for split in self.splits:
                    selected = [row for row in self.coverage if row['split'] == split
                                and all(row.get(k) == group.get(k) for k in ('stage', 'method', 'segmenter', 'variant'))]
                    pair_ids = {tuple(row['pair']) for row in selected}
                    if pair_ids:
                        estimates.append(len(selected_pairs(self.manifest, split, self.modes)) *
                                         sum(row['calls'] for row in selected) / len(pair_ids))
                    elif selected_pairs(self.manifest, split, self.modes):
                        break
                else:
                    group['estimated_units'] = min(group['units'], sum(estimates))
        # A model may have all full-frame maps cached but still require a new
        # process for uncached ROI tasks. Charge its start to the pending stage.
        for name, cfg in self.segmenters.items():
            for variant in {vocabulary_key(v) for v in variants(cfg)}:
                groups = [g for g in workload if g['stage'] in ('segment_full', 'segment_roi')
                          and g['segmenter'] == name and g['variant'] == variant]
                for group in groups:
                    group['model_initializations'] = 0
                pending = [g for g in groups if g['cached_units'] < g['units']]
                if pending:
                    pending[0]['model_initializations'] = len(self.splits)
        # CD is released before semantics, and loaded once more for ROI rescoring.
        # Charge those extra starts once per method, rather than once per variant.
        for method in self.methods:
            pending = [g for g in workload if g['stage'] == 'change_roi' and g['method'] == method
                       and g['cached_units'] < g['units']]
            if pending:
                pending[0]['model_initializations'] = len(self.splits)
        failures = sum(r['status'] == 'failed' for r in self.records)
        if final and not self.smoke and not failures:
            for group in workload:
                group['cached_units'] = group['units']
        timing = estimate(workload, self.observations, self.coverage)
        timing.update(scope='remaining full requested split queue after verified tasks observed by this invocation',
                      roi_bounds='upper task count assumes max_regions everywhere; not a statistical confidence interval',
                      roi_estimation='central ROI count extrapolated per split from predicted region counts; timing-only probes excluded',
                      wall_seconds=time.time()-self.started, smoke=self.smoke)
        atomic_json(self.out / 'estimate.json', timing)
        expected = self.expected_coverage()
        atomic_json(self.out / 'coverage_matrix.json', expected)
        render_report(self.out, self.manifest, self.records, prepared_path=self.prepared)
        # Native inference timings have a different grain from evaluation units.
        # Do not duplicate a reused full-frame inference once per reported route.
        for name, rows in (('timing', self.observations), ('coverage_matrix', expected)):
            fields = list(dict.fromkeys(k for row in rows for k in row)) or ['status']
            with (self.out / (name + '.csv')).open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        atomic_json(self.out / 'run.json', dict(schema_version=1, kind='semantic_routes', schema=SCHEMA,
                    config_revision=self.revision, smoke=self.smoke, config=self.config,
                    status=('completed_with_failures' if failures else 'completed') if final else 'running',
                    records=len(self.records), failed=failures,
                    cache_root=str(self.cache.root), fixed_settings=True))

    def expected_coverage(self):
        keys = ('kind', 'split', 'method', 'segmenter', 'variant', 'route', 'mode')
        actual = defaultdict(lambda: defaultdict(int))
        for record in self.records:
            actual[tuple(record.get(k) for k in keys)][record['status']] += 1
        output = []

        def add(count, **group):
            counts = actual[tuple(group.get(k) for k in keys)]
            output.append({**group, 'expected': count, 'completed': counts['completed'],
                           'failed': counts['failed'],
                           'not_attempted': max(0, count-sum(counts.values()))})

        for split in self.splits:
            pairs = selected_pairs(self.manifest, split, self.modes, self.smoke, self.pairs_per_split)
            frames = {(s['sequence_id'], index) for s, frame, ref, _ in pairs for index in (frame, ref)}
            for name, cfg in self.segmenters.items():
                for variant in variants(cfg):
                    add(len(frames), kind='segmentation', split=split, method=None, segmenter=name,
                        variant=variant, route='baseline', mode='previous')
            for mode in self.modes:
                count = sum(p[3] == mode for p in pairs)
                for method in self.methods:
                    add(count, kind='change', split=split, method=method, segmenter=None,
                        variant='baseline', route='baseline', mode=mode)
                    for name, cfg in self.segmenters.items():
                        for variant in variants(cfg):
                            group = dict(split=split, method=method, segmenter=name, variant=variant, mode=mode)
                            for route in self.routes:
                                add(count, kind='change', route=route, **group)
                            if 'change_first' in self.routes:
                                add(count, kind='change', route='change_first_refined', **group)
                                for route in ('change_first_roi', 'change_first_cached_clip'):
                                    add(count*2, kind='segmentation', route=route, **group)
        return output

    def run(self):
        with run_lock(self.out):
            if (self.out / 'run.json').exists() and not self.resume:
                raise ValueError(f'{self.out} already contains a run; pass --resume to reuse it')
            request = dict(schema=SCHEMA, config=self.config, revision=self.revision, smoke=self.smoke,
                           created_at=time.time(), prepared_sha256=sha256(self.prepared / 'manifest.json'))
            atomic_json(self.out / 'requests' / f'{time.time_ns()}-{self.revision[:12]}.json', request)
            atomic_json(self.out / 'resolved_config.json', self.config)
            self.event('request_started', revision=self.revision, smoke=self.smoke, splits=self.splits)
            _snapshot_gpu(self.out)
            try:
                self.preflight()
                for split in self.splits:
                    self.run_split(split)
                self.write_reports(final=True)
            finally:
                self.sem_pool.release()
                self.cd_pool.release()
            failures = sum(row['status'] == 'failed' for row in self.records)
            print(f'REPORT: {self.out / "report.html"}\nESTIMATE: {self.out / "estimate.json"}\nFAILED UNITS: {failures}', flush=True)
            return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    from .semantic_config import add_configure_parser, configure
    add_configure_parser(sub)
    for name in ('smoke', 'run'):
        child = sub.add_parser(name)
        child.add_argument('--config', required=True)
        child.add_argument('--out', required=True)
        child.add_argument('--splits', nargs='+', choices=SPLITS)
        child.add_argument('--methods', nargs='+', choices=ALL_METHODS)
        child.add_argument('--segmenters', nargs='+')
        child.add_argument('--cache-dir')
        child.add_argument('--resume', action='store_true')
        child.add_argument('--pairs-per-split', type=int)
        if name == 'run':
            child.add_argument('--smoke-first', action='store_true',
                               help='Run all-route smoke under OUT/smoke, then the full fixed queue')
    args = parser.parse_args(argv)
    if args.command == 'configure':
        configure(args)
        return 0
    config = read_json(Path(args.config).expanduser().resolve())
    if args.splits:
        config['splits'] = args.splits
    if args.methods:
        config['change_methods'] = {name: config['change_methods'].get(name, {}) for name in args.methods}
    if args.segmenters:
        unknown = set(args.segmenters) - config['segmenters'].keys()
        if unknown:
            parser.error(f'Unknown segmenters: {sorted(unknown)}')
        config['segmenters'] = {name: config['segmenters'][name] for name in args.segmenters}
    if args.command == 'run' and args.smoke_first:
        Runner(config, Path(args.out) / 'smoke', smoke=True, resume=args.resume,
               cache_dir=args.cache_dir, pairs_per_split=args.pairs_per_split).run()
    return Runner(config, args.out, smoke=args.command == 'smoke', resume=args.resume,
                  cache_dir=args.cache_dir, pairs_per_split=args.pairs_per_split).run()


if __name__ == '__main__':
    sys.exit(main())
