"""Read-only VAL samples from a frozen night run; no inference imports.

Every family uses the intersection of its available maps and known annotation
pixels. HISTORY is evaluated against the immediately preceding observation;
the saved memory and its synthetic update threshold are never changed here.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
from pathlib import Path

import numpy as np

from . import night_common as common
from .night_report import NATIVE, _load_map, _variant
from .real_gt_metrics import derive_labels
from .robustness_common import digest, read_json, safe_path, sha256


TERMINAL = {'completed', 'completed_with_failures', 'budget_exhausted',
            'interrupted', 'failed_validation'}


def _readonly(value):
    value.setflags(write=False)
    return value


def _timing(records):
    completed = [row for row in records if row.get('status') == 'completed']
    computed = [row for row in completed if row.get('origin') == 'inference']
    output = dict(completed_units=len(completed), computed_units=len(computed),
                  reused_units=sum(str(row.get('origin', '')).startswith('reused')
                                   for row in completed),
                  unknown_origin_units=sum(row.get('origin') != 'inference'
                                           and not str(row.get('origin', '')).startswith('reused')
                                           for row in completed))
    for field in ('elapsed_seconds', 'initialization_seconds', 'scorer_seconds'):
        values = []
        measured_units = 0
        for row in computed:
            raw = row.get(field)
            raw = raw if isinstance(raw, list) else [raw]
            valid = [float(value) for value in raw if isinstance(value, (int, float))
                     and not isinstance(value, bool) and np.isfinite(value) and value >= 0]
            if valid:
                measured_units += 1
                values.extend(valid)
        output[field + '_measured_units'] = measured_units
        output[field + '_observations'] = len(values)
        output[field + '_sum'] = float(sum(values)) if values else None
        output[field + '_median'] = float(np.median(values)) if values else None
    return output


class Samples:
    """Hash-bound metadata and lazy, flattened PREVIOUS/HISTORY VAL arrays.

    Initialisation checks checksums of selected VAL files without decoding
    TRAIN/TEST pixels. A completed report can therefore verify its inputs before
    reusing cached CPU results. Channel caches retain only known supported pixels;
    at most four raw prepared frames remain in memory.
    """

    def __init__(self, run_path, methods=None):
        self.root = Path(run_path).expanduser().resolve()
        self.run = read_json(self.root / 'run.json')
        if self.run.get('status') not in TERMINAL:
            raise ValueError('Night run must have an explicit terminal status, not running')
        self.prepared, self.manifest = common.load_prepared(self.run['prepared_path'])
        manifest_hash = sha256(self.prepared / 'manifest.json')
        if manifest_hash != self.run.get('manifest_sha256'):
            raise ValueError('Prepared manifest changed since inference')
        configured = self.run.get('methods', [])
        history = self.run.get('history_methods', [])
        modes = self.run.get('modes', [])
        if (not configured or len(set(configured)) != len(configured)
                or set(configured) - set(common.DEFAULT_METHODS)
                or len(set(history)) != len(history) or set(history) - set(configured)
                or set(history) - set(common.HISTORY_METHODS)
                or not modes or len(set(modes)) != len(modes)
                or set(modes) - set(common.MODES)):
            raise ValueError('Invalid frozen night-run methods or modes')
        selected = list(configured) if methods is None else list(methods)
        if not selected or len(set(selected)) != len(selected) or set(selected) - set(configured):
            raise ValueError('Selected methods must be distinct configured night-run methods')
        self.channels = [method + ':' + mode for mode in ('previous', 'history') if mode in modes
                         for method in configured if method in selected
                         and (mode != 'history' or method in history)]
        if not self.channels:
            raise ValueError('No selected PREVIOUS/HISTORY channels')
        self.categories = self.manifest['categories']
        self.tolerance = self.manifest.get('boundary_tolerance', 0)
        if type(self.tolerance) is not int or self.tolerance < 0:
            raise ValueError('Boundary tolerance must be a nonnegative integer')
        contract = self.run.get('contract', {})
        if self.run.get('config_hash') is not None and self.run['config_hash'] != digest(contract):
            raise ValueError('Night-run frozen contract checksum differs')
        for field in ('methods', 'history_methods', 'modes', 'manifest_sha256'):
            if field in contract and contract[field] != self.run.get(field):
                raise ValueError('Run metadata differs from frozen contract: ' + field)
        self.origins = contract.get('training_exposure')
        if self.origins is None:
            self.origins = common.training_exposure(
                self.manifest, contract.get('models', self.run.get('contracts', {})))
        self._sequences = {row['sequence_id']: row for row in self.manifest['sequences']}
        self._val = {key: row for key, row in self._sequences.items() if row['split'] == 'val'}
        self.buildings = sorted({row['building_id'] for row in self._val.values()})
        if not self.buildings:
            raise ValueError('No real VAL facades in the prepared night run')
        self._pairs = [(key, frame) for key, sequence in sorted(self._val.items())
                       for frame in range(1, len(sequence['frames']))]
        self._records, unit_hashes, artifact_hashes, frame_hashes = {}, {}, {}, {}
        scheduled = {tuple(job) for job in contract['jobs']} if 'jobs' in contract else None
        self._snapshot = {self.root / 'run.json': sha256(self.root / 'run.json'),
                          self.prepared / 'manifest.json': manifest_hash}
        self._unit_paths = tuple(sorted((self.root / 'units').glob('*.json')))
        for path in self._unit_paths:
            row = read_json(path)
            frame = row.get('frame')
            method, mode, sequence_id = (row.get(key) for key in ('method', 'mode', 'sequence_id'))
            sequence = self._sequences.get(sequence_id)
            if (sequence is None or method not in configured or mode not in modes
                    or (mode == 'history' and method not in history)
                    or type(frame) is not int or not 1 <= frame < len(sequence['frames'])):
                raise ValueError('Inference unit is outside the frozen run')
            target = 'earliest_state' if mode == 'earliest' else 'step_event'
            reference_frame = 0 if mode == 'earliest' else frame - 1
            if (row.get('building_id') != sequence['building_id']
                    or row.get('split') != sequence['split']
                    or row.get('target', target) != target
                    or row.get('variant', _variant(method)) != _variant(method)
                    or row.get('reference_frame', reference_frame) != reference_frame
                    or row.get('reference_year', sequence['frames'][reference_frame]['year'])
                    != sequence['frames'][reference_frame]['year']
                    or row.get('source_year', sequence['frames'][frame]['year'])
                    != sequence['frames'][frame]['year']
                    or row.get('domain', 'real') != 'real'):
                raise ValueError('Inference unit identity/target differs from prepared sequence')
            if scheduled is not None and (method, mode, sequence_id, frame) not in scheduled:
                raise ValueError('Inference unit is outside the frozen job schedule')
            identifier = common.unit_id(method, mode, sequence_id, frame)
            if row.get('unit_id') != identifier or path.stem != identifier:
                raise ValueError('Invalid inference unit identifier')
            key = (method + ':' + mode, sequence_id, frame)
            if key in self._records:
                raise ValueError('Duplicate inference unit')
            if row.get('status') not in ('completed', 'failed'):
                raise ValueError('Inference unit has no terminal status')
            self._records[key] = row
            checksum = sha256(path)
            unit_hashes[str(path.relative_to(self.root))] = checksum
            self._snapshot[path] = checksum
            if sequence_id in self._val and key[0] in self.channels and row['status'] == 'completed':
                artifact = safe_path(self.root, row['artifact_path'])
                checksum = sha256(artifact)
                if checksum != row.get('artifact_sha256'):
                    raise ValueError(f'Result artifact changed: {artifact}')
                artifact_hashes[row['artifact_path']] = checksum
                self._snapshot[artifact] = checksum
        for sequence in self._val.values():
            for frame in sequence['frames']:
                path = safe_path(self.prepared, frame['artifact_path'])
                checksum = sha256(path)
                if checksum != frame['artifact_sha256']:
                    raise ValueError(f'Prepared frame changed: {path}')
                frame_hashes[frame['artifact_path']] = checksum
                self._snapshot[path] = checksum
        self.provenance = dict(source_run=str(self.root), prepared_path=str(self.prepared),
                               run_sha256=self._snapshot[self.root / 'run.json'],
                               manifest_sha256=manifest_hash, unit_sha256=unit_hashes,
                               artifact_sha256=artifact_hashes, frame_sha256=frame_hashes,
                               channels=self.channels, val_buildings=self.buildings,
                               inference_status=self.run['status'], training_exposure=self.origins,
                               history_update_thresholds=contract.get('history_thresholds', {}),
                               no_model_inference=True, split='val')
        self.provenance['input_digest'] = digest(self.provenance)
        self.timings = [dict(channel=channel, method=channel.split(':')[0],
                             mode=channel.split(':')[1], split='val',
                             **_timing([row for (name, sequence_id, _), row in self._records.items()
                                        if name == channel and sequence_id in self._val]))
                        for channel in self.channels]
        self._frames, self._gt, self._channel_cache = OrderedDict(), {}, {}

    def validate_snapshot(self):
        """Refuse changed/deleted/new source units, including during long reports."""
        if tuple(sorted((self.root / 'units').glob('*.json'))) != self._unit_paths:
            raise ValueError('Night-run unit list changed during analysis')
        for path, expected in self._snapshot.items():
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f'Night-run input changed during analysis: {path}')

    def _frame(self, sequence_id, index):
        key = (sequence_id, index)
        if key not in self._frames:
            self._frames[key] = common.load_frame(self.prepared, self._val[sequence_id]['frames'][index])
            if len(self._frames) > 4:
                self._frames.popitem(last=False)
        self._frames.move_to_end(key)
        return self._frames[key]

    def _ground_truth(self, sequence_id, frame):
        key = (sequence_id, frame)
        if key not in self._gt:
            previous, current = self._frame(sequence_id, frame - 1), self._frame(sequence_id, frame)
            if previous['support'].shape != current['support'].shape:
                raise ValueError('Prepared chronology frames must share the same grid')
            support = previous['support'] & current['support']
            labels = derive_labels(previous['packed'], current['packed'], support,
                                   len(self.categories), self.tolerance)
            self._gt[key] = dict(labels=_readonly(labels.ravel()), shape=support.shape,
                                 reference_bits=_readonly(previous['packed'].ravel().copy()),
                                 source_bits=_readonly(current['packed'].ravel().copy()))
        return self._gt[key]

    def _channel(self, channel):
        if channel not in self.channels:
            raise ValueError('Unknown selected analysis channel: ' + str(channel))
        if channel not in self._channel_cache:
            rows = {}
            for sequence_id, frame in self._pairs:
                record = self._records.get((channel, sequence_id, frame))
                if record is None or record['status'] != 'completed':
                    continue
                truth = self._ground_truth(sequence_id, frame)
                previous, current = self._frame(sequence_id, frame - 1), self._frame(sequence_id, frame)
                value = _load_map(self.root, record, previous, current,
                                  previous['support'] & current['support'])
                pixels = np.flatnonzero(value['support'].ravel() & (truth['labels'] != 255))
                prediction = value['native'] if record['method'] in NATIVE else value['scores']
                rows[(sequence_id, frame)] = dict(pixels=_readonly(pixels),
                                                  values=_readonly(prediction.ravel()[pixels]))
            self._channel_cache[channel] = rows
        return self._channel_cache[channel]

    def _channels(self, channels):
        channels = tuple(channels)
        if len(channels) not in (1, 2) or len(set(channels)) != len(channels):
            raise ValueError('A family requires one or two distinct selected channels')
        if set(channels) - set(self.channels):
            raise ValueError('Unknown selected analysis channel')
        return channels

    def rows(self, channels):
        channels = self._channels(channels)
        caches = [self._channel(channel) for channel in channels]
        output = []
        for sequence_id, frame in self._pairs:
            key = (sequence_id, frame)
            if any(key not in cache for cache in caches):
                continue
            left = caches[0][key]
            pixels, values = left['pixels'], {'left': left['values']}
            if len(caches) == 2:
                right = caches[1][key]
                pixels, first, second = np.intersect1d(pixels, right['pixels'], assume_unique=True,
                                                     return_indices=True)
                values = dict(left=_readonly(left['values'][first]), right=_readonly(right['values'][second]))
            if not pixels.size:
                continue
            truth = self._ground_truth(sequence_id, frame)
            support_hash = hashlib.sha256(np.asarray(truth['shape'], dtype='<i8').tobytes()
                                          + np.asarray(pixels, dtype='<i8').tobytes()).hexdigest()
            output.append(dict(sequence_id=sequence_id, frame=frame,
                               building_id=self._val[sequence_id]['building_id'], split='val',
                               pixels=_readonly(pixels), labels=_readonly(truth['labels'][pixels]),
                               reference_bits=_readonly(truth['reference_bits'][pixels]),
                               source_bits=_readonly(truth['source_bits'][pixels]),
                               support_sha256=support_hash, **values))
        return output

    def coverage(self, channels):
        channels = self._channels(channels)
        rows = {(row['sequence_id'], row['frame']): row for row in self.rows(channels)}
        output = []
        for building in self.buildings:
            keys = [(sequence_id, frame) for sequence_id, frame in self._pairs
                    if self._val[sequence_id]['building_id'] == building]
            counts = dict(expected_pairs=len(keys), completed_common_pairs=0, evaluated_pairs=0,
                          failed_pairs=0, missing_pairs=0, empty_gt_pairs=0,
                          valid_pixels=0, negative_pixels=0, positive_pixels=0)
            records = []
            reasons = set()
            for sequence_id, frame in keys:
                available = [self._records.get((channel, sequence_id, frame)) for channel in channels]
                records.extend(record for record in available if record is not None)
                if any(record is not None and record['status'] == 'failed' for record in available):
                    counts['failed_pairs'] += 1
                    reasons.update(str(record.get('error', 'Inference failed')) for record in available
                                   if record is not None and record['status'] == 'failed')
                elif any(record is None for record in available):
                    counts['missing_pairs'] += 1
                    preflight = self.run.get('contract', {}).get('preflight_failures', {})
                    for channel, record in zip(channels, available):
                        if record is None:
                            method, mode = channel.split(':')
                            reason = preflight.get(mode + '/' + method) or preflight.get(method)
                            reasons.add(str(reason) if reason else 'Missing inference unit: ' + channel)
                else:
                    counts['completed_common_pairs'] += 1
                    row = rows.get((sequence_id, frame))
                    if row is None:
                        counts['empty_gt_pairs'] += 1
                    else:
                        counts['evaluated_pairs'] += 1
                        counts['valid_pixels'] += int(row['labels'].size)
                        counts['negative_pixels'] += int((row['labels'] == 0).sum())
                        counts['positive_pixels'] += int((row['labels'] == 1).sum())
            output.append(dict(channels=list(channels), building_id=building, split='val',
                               reason=' | '.join(sorted(reasons)), **counts, **_timing(records)))
        return output
