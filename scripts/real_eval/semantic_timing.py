"""Transparent runtime estimates from completed native-route smoke tasks.

These are engineering planning ranges, not confidence intervals or guaranteed
bounds. Inference and model initialization are separate. Missing measurements
are unknown, never zero; imported results do not constitute timing probes.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict

KEYS = ('stage', 'method', 'segmenter', 'variant')
STAGES = ('change_full', 'segment_full', 'segment_roi', 'change_roi')


def _key(row):
    return tuple(row.get(name, '') for name in KEYS)


def _count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name + ' must be a nonnegative integer')
    return value


def _duration(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0)


def _stats(values):
    values = list(values)
    if not values:
        return {'samples': 0, 'median_seconds': None, 'min_seconds': None, 'max_seconds': None}
    return {'samples': len(values), 'median_seconds': statistics.median(values),
            'min_seconds': min(values), 'max_seconds': max(values)}


def _coverage_rows(coverage):
    if isinstance(coverage, list):
        return coverage
    if isinstance(coverage, dict):
        values = coverage.get('groups', coverage.get('rows', []))
        return values if isinstance(values, list) else []
    return []


def estimate(full_workload, observations, smoke_coverage=None):
    """Estimate remaining runtime for explicit workload groups.

    Workload rows require stage and units, plus method/segmenter/variant as
    applicable. `units` includes already cached work. `roi_upper_bound` is an
    optional maximum total number of native ROI tasks, not a multiplier.
    `estimated_units` optionally supplies an expected total from observed ROI
    frequency. `cached_units` counts only verified reusable outputs; this is
    subtracted from all totals. `model_initializations` counts new initializations
    for this group and defaults to one when any task could remain, else zero.

    Observations require matching stage/method/segmenter/variant, completed
    status, source measured/cache, and finite elapsed_seconds *excluding init*.
    Cache timings are a labeled fallback if no newly measured timing exists.
    `init_seconds` measures one model/process start, not a per-image penalty;
    zeros for tasks that reuse a loaded model are ignored if positive starts
    exist. Imported timings are not timing probes. The same cache_key is counted
    once. Timing samples may come from a different split; quality metrics must
    remain split-specific in the reporting module.

    Ranges use 0.8 × observed minimum and 1.5 × observed maximum. A single
    sample, unseen ROI shapes, I/O, cache loading and contention remain important
    uncertainty. Missing timing for any possible pending task or required init
    makes total bounds null; known subtotals are still exposed. Entirely cached
    groups legitimately contribute zero *remaining model compute*, not zero
    wall-clock reporting or verification work.
    """
    full_workload, observations = list(full_workload), list(observations)
    coverage = defaultdict(list)
    for row in _coverage_rows(smoke_coverage):
        if isinstance(row, dict):
            coverage[_key(row)].append(row)
    grouped = defaultdict(list)
    for index, row in enumerate(observations):
        if not isinstance(row, dict):
            raise ValueError('Timing observations must be dictionaries')
        grouped[_key(row)].append((index, row))
    details = []
    for workload in full_workload:
        if not isinstance(workload, dict) or workload.get('stage') not in STAGES:
            raise ValueError('Workload requires a supported native inference stage')
        key = _key(workload)
        units = _count(workload.get('units'), 'units')
        maximum = _count(workload.get('roi_upper_bound', units), 'roi_upper_bound')
        if maximum < units:
            raise ValueError('roi_upper_bound cannot be smaller than units')
        cached = _count(workload.get('cached_units', 0), 'cached_units')
        if cached > maximum:
            raise ValueError('cached_units exceeds the maximum total workload')
        expected = workload.get('estimated_units', units)
        if not _duration(expected) or expected > maximum or expected < 0:
            raise ValueError('estimated_units must be finite and within the workload upper bound')
        pending, pending_max, pending_expected = max(0, units - cached), maximum - cached, max(0., expected - cached)
        pending_lower = min(pending, pending_expected)
        starts = _count(workload.get('model_initializations', int(pending_max > 0)), 'model_initializations')
        if not pending_max:
            starts = 0
        members = grouped.get(key, [])
        unique = {}
        priority = {'measured': 2, 'cache': 1}
        for index, row in members:
            if (row.get('status') != 'completed' or row.get('source') not in priority
                    or not _duration(row.get('elapsed_seconds'))):
                continue
            identity = row.get('cache_key') or ('observation', index)
            previous = unique.get(identity)
            if previous is None or priority[row['source']] > priority[previous['source']]:
                unique[identity] = row
        measured = [row for row in unique.values() if row['source'] == 'measured']
        chosen = measured or [row for row in unique.values() if row['source'] == 'cache']
        timing_source = 'measured' if measured else 'cache_fallback' if chosen else 'unmeasured'
        inference = _stats(row['elapsed_seconds'] for row in chosen)
        init_values = [row['init_seconds'] for row in chosen if _duration(row.get('init_seconds'))]
        positive_starts = [value for value in init_values if value > 0]
        initialization_source = timing_source if init_values else 'unmeasured'
        if workload['stage'] in ('change_full', 'change_roi') and not positive_starts:
            # One config binds a method to one weights/options contract. Its CD
            # worker startup is reusable timing evidence across ROI selectors;
            # inference timings remain strictly stage/variant matched.
            candidates = [row for _, members in grouped.items() for _, row in members
                          if row.get('stage') in ('change_full', 'change_roi')
                          and row.get('method') == workload.get('method')
                          and row.get('status') == 'completed'
                          and row.get('source') in ('measured', 'cache')
                          and _duration(row.get('init_seconds')) and row['init_seconds'] > 0]
            current = [row for row in candidates if row['source'] == 'measured']
            selected = current or candidates
            if selected:
                positive_starts = [row['init_seconds'] for row in selected]
                initialization_source = ('same_change_method_measured_fallback' if current
                                         else 'same_change_method_cache_fallback')
        initialization = _stats(positive_starts or init_values)
        shapes = sorted({tuple(row['shape']) for row in chosen
                         if isinstance(row.get('shape'), (list, tuple)) and len(row['shape']) == 2
                         and all(type(value) is int and value > 0 for value in row['shape'])})
        coverage_statuses = sorted({str(row.get('status', 'unknown')) for row in coverage.get(key, [])})
        failed = sum(row.get('status') == 'failed' for _, row in members)
        unresolved = []
        if pending_max and not chosen:
            unresolved.append('no_completed_inference_timing')
        if starts and not initialization['samples']:
            unresolved.append('model_initialization_not_measured')
        if not pending_max:
            lower = upper = central = 0.
            probe_status = 'not_required_all_cached' if maximum else 'not_required_no_tasks'
        elif unresolved:
            lower = upper = central = None
            probe_status = ('no_roi_observed' if any('no_roi' in status or 'no_region' in status
                                                   for status in coverage_statuses)
                            else 'failed' if failed else 'unmeasured')
        else:
            init_min, init_max, init_median = (initialization[name] or 0. for name in
                                             ('min_seconds', 'max_seconds', 'median_seconds'))
            lower = .8 * (pending_lower * inference['min_seconds'] + starts * init_min)
            upper = 1.5 * (pending_max * inference['max_seconds'] + starts * init_max)
            central = pending_expected * inference['median_seconds'] + starts * init_median
            probe_status = ('timing_only_probe' if chosen and all(row.get('timing_only') for row in chosen)
                            else 'measured' if measured else 'cache_timing_fallback')
        warnings = []
        if inference['samples'] == 1:
            warnings.append('single_timing_sample')
        if workload['stage'].endswith('_roi'):
            warnings.append('ROI_counts_and_shapes_can_differ_from_smoke')
        if maximum != units:
            warnings.append('upper_bound_uses_maximum_ROI_workload')
        if timing_source == 'cache_fallback':
            warnings.append('reused_historical_timing_current_contention_not_measured')
        details.append({**{name: workload.get(name, '') for name in (*KEYS, 'split')},
                        'units': units, 'maximum_units': maximum, 'estimated_units': expected,
                        'cached_units': cached, 'pending_units': pending, 'pending_lower_units': pending_lower,
                        'pending_maximum_units': pending_max,
                        'model_initializations': starts, 'timing_source': timing_source,
                        'timing_probe_status': probe_status, 'smoke_coverage_statuses': coverage_statuses,
                        'failed_observations': failed, 'inference': inference, 'initialization': initialization,
                        'initialization_source': initialization_source,
                        'timing_only_samples': sum(bool(row.get('timing_only')) for row in chosen),
                        'task_samples': sum(not row.get('timing_only') for row in chosen),
                        'observed_shapes': [list(shape) for shape in shapes],
                        'lower_seconds': lower, 'upper_seconds': upper, 'central_seconds': central,
                        'unresolved': unresolved, 'warnings': warnings})
    unresolved = [row for row in details if row['unresolved']]
    known_lower = sum(row['lower_seconds'] for row in details if row['lower_seconds'] is not None)
    known_upper = sum(row['upper_seconds'] for row in details if row['upper_seconds'] is not None)
    known_central = sum(row['central_seconds'] for row in details if row['central_seconds'] is not None)
    return {'schema_version': 1, 'status': 'incomplete_timing_coverage' if unresolved else 'estimated',
            'estimate_kind': 'heuristic_remaining_model_compute_not_statistical_confidence_interval',
            'total_lower_seconds': None if unresolved else known_lower,
            'total_upper_seconds': None if unresolved else known_upper,
            'total_central_seconds': None if unresolved else known_central,
            'known_lower_seconds': known_lower, 'known_upper_seconds': known_upper,
            'known_central_seconds': known_central, 'unresolved_groups': len(unresolved),
            'unresolved_maximum_units': sum(row['pending_maximum_units'] for row in unresolved),
            'groups': details, 'smoke_coverage': smoke_coverage,
            'assumptions': ['elapsed_seconds excludes model initialization',
                            'groups and model_initializations describe sequential work without double counting',
                            'bounds are 0.8 times observed minimum and 1.5 times observed maximum',
                            'ROI upper bounds count total native tasks, not a multiplier',
                            'verification, cache loading, report writing, scheduling and resource contention are excluded',
                            'zero remaining model compute for cached groups does not mean zero wall-clock time']}
