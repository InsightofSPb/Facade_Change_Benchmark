"""Read-only evaluation of saved original real crop histories.

No model inference or calibration is performed here. Annotation bitset changes
are a diagnostic target, not independently reviewed temporal ground truth.
"""
from __future__ import annotations

import csv
import html
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from . import night_common as common
    from . import real_gt_metrics as metrics
    from .real_gt_slices import evaluate_class_slices
    from .robustness_common import atomic_json, read_json, safe_path, sha256
except ImportError:
    import night_common as common
    import real_gt_metrics as metrics
    from real_gt_slices import evaluate_class_slices
    from robustness_common import atomic_json, read_json, safe_path, sha256

NATIVE = {'geoscd', 'anychange'}
BASELINES = ('geoscd', 'anychange', 'rgb_diff', 'ssim', 'lpips')
MODES = ('previous', 'earliest', 'history')
METRICS = ('precision', 'recall', 'f1', 'iou', 'fpr', 'marked_fraction')
COUNTS = ('tp', 'fp', 'fn', 'tn', 'valid_pixels', 'positive_pixels', 'negative_pixels')
GROUP = ('split_scope', 'training_scope', 'mode', 'target', 'comparison', 'predictor', 'point')


def _variant(method):
    return 'current_train' if method.startswith('msdzip_') else 'frozen'


def _exposure(methods, building, origins):
    values = []
    for method in methods:
        origin = origins.get(method, {})
        if building in origin.get('building_ids', []):
            values.append('seen')
        elif origin.get('status') == 'not_project_trained':
            values.append('verified_unseen')
        elif origin.get('status') == 'verified' and origin.get('complete'):
            values.append('verified_unseen')
        else:
            values.append('unknown')
    return 'seen' if 'seen' in values else 'unknown' if 'unknown' in values else 'verified_unseen'


def aggregate(rows, extra=()):
    """Equal crop-frame observations within facade, then equal facades.

    Only positive observations contribute P/R/F1/IoU. Stable annotated pixels
    from all observations contribute FPR. Counts are summed observations.
    """
    groups = defaultdict(list)
    for row in rows:
        for split_scope in (row['split'], 'all_selected'):
            for training_scope in ('all', row['training_exposure']):
                key = tuple({'split_scope': split_scope, 'training_scope': training_scope,
                             **row}.get(name, '') for name in (*GROUP, *extra))
                groups[key].append(row)
    output, facades = [], []
    for key, members in sorted(groups.items(), key=lambda item: str(item[0])):
        header = dict(zip((*GROUP, *extra), key))
        evaluated = [r for r in members if r['valid_pixels']]
        by_facade = defaultdict(list)
        for row in evaluated:
            by_facade[row['building_id']].append(row)
        facade_values = []
        for building, observations in sorted(by_facade.items()):
            current = dict(header, building_id=building, evaluated_pairs=len(observations),
                           positive_pairs=sum(r['positive_pixels'] > 0 for r in observations),
                           **{name: sum(r[name] for r in observations) for name in COUNTS})
            for metric in METRICS:
                selected = [r[metric] for r in observations if r.get(metric) is not None
                            and (metric in ('fpr', 'marked_fraction') or r['positive_pixels'] > 0)]
                current[metric] = float(np.mean(selected)) if selected else None
            facade_values.append(current)
            facades.append(current)
        current = dict(header, status='evaluated' if evaluated else 'no_evaluable_pixels',
                       facades=len(by_facade), evaluated_pairs=len(evaluated),
                       positive_pairs=sum(r['positive_pixels'] > 0 for r in evaluated),
                       **{name: sum(r[name] for r in evaluated) for name in COUNTS})
        for exposure in ('seen', 'verified_unseen', 'unknown'):
            current[exposure + '_facades'] = len({r['building_id'] for r in evaluated
                                                if r['training_exposure'] == exposure})
        for metric in METRICS:
            values = [r[metric] for r in facade_values if r[metric] is not None]
            current[metric] = float(np.mean(values)) if values else None
            current[metric + '_facades'] = len(values)
        output.append(current)
    return output, facades


def _thresholds(manifest, methods, history_methods, modes, budgets):
    saved = manifest['source_summary']
    if isinstance(saved, str):
        saved = read_json(saved)
    thresholds, unavailable = {}, {}
    for mode in modes:
        target = 'earliest_state' if mode == 'earliest' else 'step_event'
        for method in history_methods if mode == 'history' else methods:
            for point in ('native',) if method in NATIVE else budgets:
                key = (method, mode, point)
                if method in NATIVE:
                    thresholds[key] = None
                    continue
                try:
                    choice = metrics.calibration(saved, (method, _variant(method), mode, target), point)
                    if choice is None:
                        raise ValueError('No calibrated synthetic VAL clean H0 threshold')
                    if choice.get('operator', '>') != '>':
                        raise ValueError('Only the saved strict > operator is supported')
                    thresholds[key] = float(choice['threshold'])
                except (ValueError, KeyError, TypeError) as error:
                    unavailable[key] = str(error)
    return thresholds, unavailable


def _load_map(root, record, reference, source, pair_support):
    path = safe_path(root, record['artifact_path'])
    if sha256(path) != record['artifact_sha256']:
        raise ValueError(f'Result artifact changed: {path}')
    with np.load(path, allow_pickle=False) as archive:
        support = archive['support'].copy()
        scores = archive['scores'].copy()
        if (support.dtype != bool or support.shape != pair_support.shape
                or np.any(support & ~pair_support)
                or not np.array_equal(archive['source'], source['rgb'])
                or not np.array_equal(archive['reference'], reference['rgb'])):
            raise ValueError(f'Saved result does not match chronology RGB/support: {path}')
        if (scores.dtype != np.float32 or scores.shape != support.shape
                or not np.isfinite(scores[support]).all() or not np.isnan(scores[~support]).all()):
            raise ValueError(f'Invalid saved score grid: {path}')
        native = archive['native_prediction'].copy() if record['method'] in NATIVE else None
        if native is not None and (native.dtype != bool or native.shape != support.shape):
            raise ValueError(f'Invalid native prediction: {path}')
    return {'support': support, 'scores': scores, 'native': native}


def _prediction(value, threshold):
    return (value['native'] if threshold is None else value['scores'] > threshold) & value['support']


def _evaluate(prediction, support, reference, source, categories, tolerance):
    labels = metrics.derive_labels(reference['packed'], source['packed'], support, len(categories), tolerance)
    result = metrics.evaluate_mask(prediction, labels)
    result['marked_fraction'] = ((result['tp'] + result['fp']) / result['valid_pixels']
                                 if result['valid_pixels'] else None)
    classes = evaluate_class_slices(prediction, reference['packed'], source['packed'], support,
                                    labels, categories)
    return result, classes


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ['status']
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def _table(rows, columns):
    parts = ['<table><thead><tr>' + ''.join('<th>' + html.escape(key) + '</th>' for key in columns)
             + '</tr></thead><tbody>']
    for row in rows:
        values = []
        for key in columns:
            value = row.get(key)
            if value is None:
                value = '—'
            elif key in METRICS or key.startswith('delta_'):
                value = f'{100 * value:.2f}'
            values.append('<td>' + html.escape(str(value)) + '</td>')
        parts.append('<tr>' + ''.join(values) + '</tr>')
    return ''.join(parts) + '</tbody></table>'


def _html(tables, coverage, comparisons, warnings):
    parts = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Реальные временные изменения</title>',
             '<style>body{font:15px system-ui;margin:2rem;color:#222}table{border-collapse:collapse;margin:1rem 0;font-size:13px}'
             'td,th{border:1px solid #ccc;padding:.35rem;text-align:right}th{background:#eee}p{max-width:1000px}'
             'summary{cursor:pointer;font-weight:bold}td:first-child{text-align:left}</style>',
             '<h1>Реальные временные изменения — выбранные кропы</h1>',
             '<p>Метрики в таблицах — %. В CSV — доли. Рабочая точка 0.01 / 0.05 обозначает целевой FPR '
             'на исходной synthetic VAL H0 при выборе фиксированного порога; реальный FPR не ограничен этим числом.</p>',
             '<p>Сочетания: &amp; = AND, подтверждение обоими сигналами; | = OR, срабатывание любого сигнала. '
             'В каждом случае объединяются PREVIOUS и HISTORY с целью нового события. '
             'Сравнение fusion/... использует PREVIOUS как базу, fusion/.../vs_history — HISTORY; '
             'обе базы и сочетание оцениваются на одинаковых парах и пикселях.</p>']
    parts += ['<p>' + html.escape(warning) + '</p>' for warning in warnings]
    cols = ('predictor', 'point', 'training_scope', 'precision', 'recall', 'f1', 'iou', 'fpr',
            'marked_fraction', 'facades', 'evaluated_pairs', 'positive_pairs', 'tp', 'fp', 'fn', 'tn')
    for mode in MODES:
        parts.append('<h2>' + mode.upper() + '</h2>')
        for split in ('train', 'val', 'test', 'all_selected'):
            current = [r for r in tables if r['mode'] == mode and r['split_scope'] == split
                       and r['training_scope'] in ('all', 'verified_unseen')]
            all_rows = {(r['predictor'], r['comparison'], r['point']): r for r in current
                        if r['training_scope'] == 'all'}
            current = [r for r in current if r['training_scope'] == 'all'
                       or any(all_rows[(r['predictor'], r['comparison'], r['point'])][key]
                              for key in ('seen_facades', 'unknown_facades'))]
            if current:
                parts.append('<details open><summary>' + split.upper() + '</summary>' + _table(current, cols) + '</details>')
        if not any(r['mode'] == mode for r in tables):
            parts.append('<p>Нет доступных результатов с фиксированной рабочей точкой.</p>')
    parts.append('<h2>Сопоставления на одинаковых парах и пикселях</h2>')
    parts.append(_table([r for r in comparisons if r['region'] == 'overall'
                        and r['split_scope'] != 'all_selected' and r['training_scope'] == 'all'],
                        ('split_scope', 'comparison', 'baseline', 'predictor', 'point', 'facades',
                         'evaluated_pairs', 'delta_precision', 'delta_recall', 'delta_f1', 'delta_iou', 'delta_fpr')))
    parts.append('<details><summary>Покрытие и отказы</summary>' + _table(coverage,
                 ('split', 'mode', 'method', 'point', 'expected_pairs', 'completed_pairs', 'failed_pairs',
                  'missing_pairs', 'evaluable_pairs', 'status')) + '</details>')
    parts.append('<p>Полные срезы: <a href="tables.csv">таблицы</a>, <a href="per_facade.csv">фасады</a>, '
                 '<a href="per_class.csv">классы</a>, <a href="comparisons.csv">сравнения</a>, '
                 '<a href="coverage.csv">покрытие</a>, <a href="provenance.json">происхождение</a>.</p></html>')
    return ''.join(parts)


def report(run_path, out_path=None, budgets=(.01, .05)):
    root = Path(run_path).expanduser().resolve()
    out = Path(out_path).expanduser().resolve() if out_path else root / 'reports'
    run = read_json(root / 'run.json')
    prepared, manifest = common.load_prepared(run['prepared_path'])
    prepared = Path(prepared).resolve()
    if sha256(prepared / 'manifest.json') != run['manifest_sha256']:
        raise ValueError('Prepared manifest changed since inference')
    if (not budgets or len(set(budgets)) != len(budgets)
            or any(not np.isfinite(b) or not 0 < b < 1 for b in budgets)):
        raise ValueError('Budgets must be distinct finite values between zero and one')
    if out == root or out in root.parents or out == prepared or prepared in out.parents or out in prepared.parents:
        raise ValueError('Report output overlaps protected input data')
    for value in (manifest.get('source_run'), manifest.get('snapshot')):
        if value:
            protected = Path(value).expanduser().resolve()
            if out == protected or out in protected.parents or protected in out.parents:
                raise ValueError('Report output overlaps frozen source runs')
    if out.exists() and any(out.iterdir()):
        provenance_path = out / 'provenance.json'
        previous = read_json(provenance_path) if provenance_path.is_file() else {}
        if previous.get('kind') != 'real_temporal_report' or previous.get('source_run') != str(root):
            raise FileExistsError('Only this run\'s owned report directory can be regenerated')
    methods = list(run['methods'])
    history_methods = list(run.get('history_methods', []))
    modes = list(run.get('modes', MODES))
    if not set(modes) <= set(MODES):
        raise ValueError('Unknown temporal mode')
    thresholds, unavailable = _thresholds(manifest, methods, history_methods, modes, budgets)
    contract = run.get('contract', {})
    origins = contract.get('training_exposure')
    if origins is None:
        origins = common.training_exposure(manifest, contract.get('models', run.get('contracts', {})))
    sequences = {row['sequence_id']: row for row in manifest['sequences']}
    if len(sequences) != len(manifest['sequences']):
        raise ValueError('Duplicate prepared sequence')
    records, unit_hashes = {}, {}
    for path in sorted((root / 'units').glob('*.json')):
        row = read_json(path)
        key = (row['method'], row['mode'], row['sequence_id'], int(row['frame']))
        if key in records:
            raise ValueError('Duplicate inference unit')
        if (row['sequence_id'] not in sequences or row['method'] not in methods + history_methods
                or row['mode'] not in modes or not 1 <= int(row['frame']) < len(sequences[row['sequence_id']]['frames'])):
            raise ValueError('Inference unit is outside the frozen run')
        sequence = sequences[row['sequence_id']]
        if row.get('building_id') != sequence['building_id'] or row.get('split') != sequence['split']:
            raise ValueError('Inference unit identity differs from prepared sequence')
        records[key] = row
        unit_hashes[str(path.relative_to(root))] = sha256(path)
    rows, class_rows, matched_rows, matched_class_rows = [], [], [], []
    coverage_counts, artifacts = defaultdict(lambda: defaultdict(int)), {}
    failure_reasons = defaultdict(set)
    categories = manifest['categories']
    tolerance = int(manifest.get('boundary_tolerance', 0))

    def append_result(destination, prediction, support, old, new, header,
                      include_classes=False, matched_classes=False):
        values, slices = _evaluate(prediction, support, old, new, categories, tolerance)
        destination.append({**header, **values})
        if include_classes:
            class_rows.extend({**header, **value} for value in slices)
        if matched_classes:
            matched_class_rows.extend({**header, **value} for value in slices)
        return values

    for sequence in manifest['sequences']:
        frames = sequence['frames']
        first = common.load_frame(prepared, frames[0])
        previous = first
        for frame in range(1, len(frames)):
            current = common.load_frame(prepared, frames[frame])
            key_header = dict(sequence_id=sequence['sequence_id'], building_id=sequence['building_id'],
                              split=sequence['split'], frame=frame)
            predictions = {}
            for mode in modes:
                reference = first if mode == 'earliest' else previous
                pair_support = reference['support'] & current['support']
                target = 'earliest_state' if mode == 'earliest' else 'step_event'
                for method in history_methods if mode == 'history' else methods:
                    record = records.get((method, mode, sequence['sequence_id'], frame))
                    points = ('native',) if method in NATIVE else budgets
                    value = None
                    if record and record.get('status') == 'completed':
                        if record.get('variant', _variant(method)) != _variant(method) or record.get('target', target) != target:
                            raise ValueError('Inference variant/target differs from report protocol')
                        value = _load_map(root, record, reference, current, pair_support)
                        artifacts[record['artifact_path']] = record['artifact_sha256']
                    for point in points:
                        coverage_key = (sequence['split'], mode, method, point)
                        coverage = coverage_counts[coverage_key]
                        coverage['expected_pairs'] += 1
                        coverage['completed_pairs' if value is not None else 'failed_pairs'
                                 if record and record.get('status') == 'failed' else 'missing_pairs'] += 1
                        if record and record.get('status') == 'failed':
                            failure_reasons[coverage_key].add(str(record.get('error', 'Inference failed')))
                        preflight = contract.get('preflight_failures', {})
                        reason = preflight.get(method) or preflight.get(mode + '/' + method)
                        if reason:
                            failure_reasons[coverage_key].add(str(reason))
                        threshold_key = (method, mode, point)
                        if value is None or threshold_key not in thresholds:
                            continue
                        prediction = _prediction(value, thresholds[threshold_key])
                        header = dict(key_header, mode=mode, target=target, comparison='standalone', predictor=method,
                                      point=point, training_exposure=_exposure([method], sequence['building_id'], origins))
                        result = append_result(rows, prediction, value['support'], reference, current, header, True)
                        coverage['evaluable_pairs'] += bool(result['valid_pixels'])
                        predictions[(method, mode, point)] = (prediction, value['support'])
            # History has a step-event target. AND and OR can combine it with
            # PREVIOUS; an EARLIEST state mask has a different target.
            for gate in history_methods:
                for point in budgets:
                    historical = predictions.get((gate, 'history', point))
                    if historical is None:
                        continue
                    for baseline in BASELINES:
                        baseline_point = 'native' if baseline in NATIVE else point
                        original = predictions.get((baseline, 'previous', baseline_point))
                        if original is None:
                            continue
                        shared = original[1] & historical[1]
                        exposure = _exposure([baseline, gate], sequence['building_id'], origins)
                        for operation, combined in (('&', original[0] & historical[0]),
                                                    ('|', original[0] | historical[0])):
                            predictor = baseline + operation + gate + '_history'
                            header = dict(key_header, mode='history', target='step_event',
                                          comparison='previous_plus_history', predictor=predictor,
                                          point=point, training_exposure=exposure)
                            append_result(rows, combined, shared, previous, current, header, True)
                            # Keep the PREVIOUS comparison ID stable. The second
                            # contrast makes any gain over the stronger component
                            # visible without comparing different standalone cohorts.
                            for suffix, component, component_name in (
                                    ('', original[0], baseline),
                                    ('/vs_history', historical[0], gate + '_history')):
                                comparison = 'fusion/' + predictor + suffix
                                base_header = dict(header, comparison=comparison,
                                                   predictor=component_name, role='baseline')
                                test_header = dict(header, comparison=comparison, role='candidate')
                                append_result(matched_rows, component, shared, previous, current,
                                              base_header, matched_classes=True)
                                append_result(matched_rows, combined, shared, previous, current,
                                              test_header, matched_classes=True)
                    original = predictions.get((gate, 'previous', point))
                    if original is not None:
                        shared = original[1] & historical[1]
                        header = dict(key_header, mode='history', target='step_event', comparison='history_vs_previous/' + gate,
                                      point=point, training_exposure=_exposure([gate], sequence['building_id'], origins))
                        append_result(matched_rows, original[0], shared, previous, current,
                                      dict(header, predictor=gate + '_previous', role='baseline'), matched_classes=True)
                        append_result(matched_rows, historical[0], shared, previous, current,
                                      dict(header, predictor=gate + '_history', role='candidate'), matched_classes=True)
            previous = current
    tables, per_facade = aggregate(rows)
    per_class, _ = aggregate(class_rows, ('class_id', 'class_name', 'direction'))
    matched, _ = aggregate(matched_rows, ('role',))
    matched_classes, _ = aggregate(matched_class_rows, ('role', 'class_id', 'class_name', 'direction'))
    comparisons = []
    for region, source, extra in (('overall', matched, ()),
                                  ('class', matched_classes, ('class_id', 'class_name', 'direction'))):
        grouped = defaultdict(dict)
        for row in source:
            key = tuple(row[name] for name in (*GROUP, *extra) if name != 'predictor')
            grouped[key][row['role']] = row
        for group in grouped.values():
            if set(group) != {'baseline', 'candidate'}:
                raise ValueError('Incomplete matched comparison')
            base, candidate = group['baseline'], group['candidate']
            if any(base[name] != candidate[name] for name in ('facades', 'evaluated_pairs', 'valid_pixels', 'positive_pixels', 'negative_pixels')):
                raise ValueError('Matched comparison cohort/support differs')
            comparisons.append({**{name: candidate[name] for name in (*GROUP, *extra)},
                                'region': region, 'baseline': base['predictor'],
                                **{name: candidate[name] for name in ('facades', 'evaluated_pairs', 'valid_pixels')},
                                **{prefix + metric: value[metric] for prefix, value in (('baseline_', base), ('candidate_', candidate))
                                   for metric in (*METRICS, *COUNTS)},
                                **{'delta_' + metric: candidate[metric] - base[metric]
                                   if candidate[metric] is not None and base[metric] is not None else None for metric in METRICS}})
    coverage = []
    for (split, mode, method, point), counts in sorted(coverage_counts.items(), key=lambda item: str(item[0])):
        key = (method, mode, point)
        reasons = sorted(failure_reasons[(split, mode, method, point)])
        if key in unavailable:
            reasons.insert(0, unavailable[key])
        coverage.append(dict(split=split, mode=mode, method=method, point=point,
                             **{field: counts[field] for field in ('expected_pairs', 'completed_pairs', 'failed_pairs', 'missing_pairs', 'evaluable_pairs')},
                             status='missing_calibration' if key in unavailable else 'available' if counts['evaluable_pairs'] else 'no_evaluable_results',
                             reason=' | '.join(reasons)))
    warnings = [
        'Отбор фасадов и кропов по изменениям GT: это обогащённая диагностическая выборка, включая выбранный TEST; не полная несмещённая оценка TEST.',
        'GT означает изменение набора аннотированных классов между датами. Неизвестная область исключается; это не независимая экспертная разметка временного события.',
        'PREVIOUS/HISTORY оценивают новое событие между соседними наблюдениями. EARLIEST оценивает состояние относительно первого наблюдения; соседние GT-маски не объединяются.',
        'P/R/F1/IoU усредняются по положительным кропам-кадрам внутри фасада, затем по фасадам. FPR учитывает известные отрицательные пиксели всех кадров и не называется H0 FPR.',
        'Счётчики — суммы пиксельных наблюдений; перекрывающиеся кропы могут учитывать одно место несколько раз. Это не оценка целого фасада без повторов.',
        'Доля отмеченных пикселей измеряется только внутри оцениваемой области; она не измеряет время эксперта.',
        'Основные строки используют доступные результаты: покрытие может различаться. Дельты сравнений рассчитаны на одинаковых парах и поддержке.',
        'AND и OR сравниваются отдельно с PREVIOUS-компонентом и HISTORY-компонентом на общей поддержке; для обеих сторон действует совместная принадлежность обучению. OR не ограничен FPR рабочей точки скалярной ветви.',
        'verified_unseen означает подтверждённое отсутствие фасада в проектном обучении компонента; неизвестное происхождение не считается unseen. Внешнее предобучение не аудировалось.',
        'В классовых срезах класс задан аннотацией; бинарный детектор сам не предсказывает тип повреждения. Интервалы неопределённости и статистическая значимость не рассчитаны.'
    ]
    provenance = dict(schema_version=1, kind='real_temporal_report', source_run=str(root), prepared_path=str(prepared),
                      manifest_sha256=run['manifest_sha256'], run_sha256=sha256(root / 'run.json'),
                      unit_sha256=unit_hashes, artifact_sha256=artifacts, script_sha256=sha256(__file__),
                      calibration_scope='synthetic_VAL_H0_frozen', budgets=list(budgets),
                      inference_status=run.get('status', 'unknown'),
                      thresholds=[dict(method=k[0], mode=k[1], point=k[2], threshold=v) for k, v in thresholds.items()],
                      history_update_thresholds=contract.get('history_thresholds', {}),
                      fusion_operations={'&': 'AND', '|': 'OR'},
                      fusion_comparison_bases=['previous_component', 'history_component'],
                      training_exposure=origins, boundary_tolerance=tolerance, no_model_inference=True,
                      metrics_csv_units='fractions', selected_gt_enriched_cohort=True, warnings=warnings)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.temporal-report-', dir=out.parent) as temporary:
        staging = Path(temporary)
        for name, values in (('tables.csv', tables), ('per_facade.csv', per_facade), ('per_class.csv', per_class),
                             ('coverage.csv', coverage), ('comparisons.csv', comparisons)):
            _write_csv(staging / name, values)
        (staging / 'report.html').write_text(_html(tables, coverage, comparisons, warnings), encoding='utf-8')
        atomic_json(staging / 'provenance.json', provenance)
        out.mkdir(parents=True, exist_ok=True)
        # Each file replacement is atomic; provenance is the final completion marker.
        for path in sorted(staging.iterdir(), key=lambda p: p.name == 'provenance.json'):
            os.replace(path, out / path.name)
    result = dict(out=str(out), report=str(out / 'report.html'), tables=tables, coverage=coverage,
                  comparisons=comparisons, calibration_scope='synthetic_VAL_H0_frozen')
    print(f'Отчёт: {out / "report.html"}; строк таблиц: {len(tables)}; сопоставлений: {len(comparisons)}', flush=True)
    return result
