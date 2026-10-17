"""Evaluate saved robustness maps; thresholds are never fitted on perturbations."""
from __future__ import annotations

import csv
import html
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from .robustness_common import (NATIVE, atomic_json, load_pair, load_prepared,
                                    read_json, safe_path, sha256)
    from . import real_gt_metrics as metrics
    from .real_gt_slices import evaluate_class_slices
except ImportError:
    from robustness_common import (NATIVE, atomic_json, load_pair, load_prepared,
                                   read_json, safe_path, sha256)
    import real_gt_metrics as metrics
    from real_gt_slices import evaluate_class_slices


METRICS = ('precision', 'recall', 'f1', 'iou', 'fpr', 'marked_fraction')
GROUP = ('cohort', 'comparison', 'hypothesis', 'condition', 'predictor', 'point')


def write_csv(path, rows):
    names = list(dict.fromkeys(key for row in rows for key in row)) or ['status']
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, names)
        writer.writeheader()
        writer.writerows(rows)


class Thresholds:
    """Use synthetic frozen thresholds or already-computed standalone real LOFO."""
    def __init__(self, manifest, calibration_run=None):
        self.saved = manifest['source_summary']
        self.rows = None
        self.hashes = {}
        self.scope = 'synthetic_VAL_H0_frozen'
        if calibration_run is not None:
            root = Path(calibration_run).expanduser().resolve()
            provenance = read_json(root / 'provenance.json')
            metrics_root = Path(provenance['metrics_run']).expanduser().resolve()
            parent = read_json(metrics_root / 'provenance.json')
            if Path(parent['source_run']).resolve() != Path(manifest['source_run']).resolve():
                raise ValueError('LOFO calibration belongs to another source run')
            if parent.get('boundary_tolerance', 0) != manifest.get('boundary_tolerance', 0):
                raise ValueError('LOFO calibration uses different GT boundary tolerance')
            for name in ('thresholds.json', 'provenance.json'):
                self.hashes[str(root / name)] = sha256(root / name)
            self.hashes[str(metrics_root / 'provenance.json')] = sha256(metrics_root / 'provenance.json')
            expected = provenance.get('input_sha256', {}).get(str(metrics_root / 'provenance.json'))
            if expected and self.hashes[str(metrics_root / 'provenance.json')] != expected:
                raise ValueError('Original LOFO metadata changed')
            self.rows = read_json(root / 'thresholds.json')
            self.scope = 'real_VAL_standalone_LOFO_frozen'

    def get(self, method, variant, building, goal):
        if method in NATIVE:
            return None
        if self.rows is None:
            value = metrics.calibration(self.saved, (method, variant, 'previous', 'step_event'), goal)
            if value is None:
                raise ValueError(f'Missing frozen synthetic threshold: {method}/{variant}/{goal}')
            return float(value['threshold'])
        group = f'{method}:{variant}:previous:step_event'
        rows = [r for r in self.rows if r.get('family_kind') == 'standalone'
                and r.get('group') == group and r.get('heldout_building') == building
                and abs(float(r.get('goal', -1)) - goal) < 1e-12]
        if len(rows) != 1 or rows[0].get('status') != 'calibrated':
            raise ValueError(f'Missing frozen standalone LOFO threshold: {group}/{building}/{goal}')
        row = rows[0]
        if building in row.get('train_buildings', []) or not row.get('train_buildings'):
            raise ValueError('Invalid held-out facade in frozen LOFO threshold')
        value = row['threshold']
        if not np.isfinite(value) or row.get('operator', '>') != '>':
            raise ValueError('Invalid frozen threshold/operator')
        return float(value)


def aggregate(rows, extra=(), retain_empty=False):
    """Positive-frame P/R/F1/IoU; all-frame FPR; equal frames then facades."""
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row.get(k, '') for k in (*GROUP, *extra))
        if retain_empty:
            groups.setdefault(key, [])
        if row['valid_pixels']:
            groups[key].append(row)
    output, facade_rows = [], []
    for key, members in sorted(groups.items(), key=lambda item: str(item[0])):
        header = dict(zip((*GROUP, *extra), key))
        by_facade = defaultdict(list)
        for row in members:
            by_facade[row['building_id']].append(row)
        facades = []
        for facade, values in sorted(by_facade.items()):
            current = dict(header, building_id=facade, pairs=len(values),
                           positive_pairs=sum(v['positive_pixels'] > 0 for v in values))
            for metric in METRICS:
                selected = [v[metric] for v in values if v.get(metric) is not None
                            and (metric in ('fpr', 'marked_fraction') or v['positive_pixels'] > 0)]
                current[metric] = float(np.mean(selected)) if selected else None
            facades.append(current)
            facade_rows.append(current)
        row = dict(header, facades=len(facades), pairs=len(members),
                   positive_pairs=sum(r['positive_pixels'] > 0 for r in members))
        for name in ('tp', 'fp', 'fn', 'tn', 'valid_pixels', 'positive_pixels', 'negative_pixels'):
            row[name] = sum(r[name] for r in members)
        for metric in METRICS:
            values = [r[metric] for r in facades if r[metric] is not None]
            row[metric] = float(np.mean(values)) if values else None
            row[metric + '_facades'] = len(values)
        output.append(row)
    return output, facade_rows


def _load_result(root, row, pair, condition, hypothesis):
    path = safe_path(root, row['artifact_path'])
    if sha256(path) != row['artifact_sha256']:
        raise ValueError(f'Result artifact changed: {row["unit_id"]}')
    with np.load(path, allow_pickle=False) as archive:
        support = archive['support'].copy()
        scores = archive['scores'].copy()
        reference = pair['source'] if hypothesis == 'H0' else pair['reference']
        source = pair['source'] if condition == 'original' else pair[condition]
        labels = np.where(pair['support'], 0, 255).astype(np.uint8) if hypothesis == 'H0' else pair['labels']
        if (support.dtype != bool or not np.array_equal(support, pair['support'])
                or not np.array_equal(archive['reference'], reference)
                or not np.array_equal(archive['source'], source)
                or not np.array_equal(archive['labels'], labels)):
            raise ValueError('Saved predictions do not match reviewed RGB/GT/support')
        if (scores.dtype != np.float32 or scores.shape != support.shape
                or not np.isfinite(scores[support]).all() or not np.isnan(scores[~support]).all()):
            raise ValueError('Invalid saved scores')
        native = archive['native_prediction'].copy() if row['method'] in NATIVE else None
        if native is not None and (native.dtype != bool or native.shape != support.shape):
            raise ValueError('Invalid native prediction')
    return dict(scores=scores, support=support, native=native, record=row)


def report(run_path, out_path, calibration_run=None, budgets=(.01, .05)):
    root, out = (Path(p).expanduser().resolve() for p in (run_path, out_path))
    run = read_json(root / 'run.json')
    prepared, manifest = load_prepared(run['prepared_path'])
    if sha256(prepared / 'manifest.json') != run['manifest_sha256']:
        raise ValueError('Prepared manifest changed since inference')
    if not budgets or any(not np.isfinite(b) or not 0 < b < 1 for b in budgets) or len(budgets) != len(set(budgets)):
        raise ValueError('Budgets must be distinct finite values between zero and one')
    for protected in (root, prepared, Path(manifest['source_run']), Path(manifest['snapshot'])):
        if out == protected or protected in out.parents or out in protected.parents:
            raise ValueError('Report output must be separate from input runs')
    if out.exists():
        raise FileExistsError('Use a new report output directory')
    threshold_source = Thresholds(manifest, calibration_run)
    methods = list(run['methods'])
    selection = read_json(run['selection_path'])
    if sha256(run['selection_path']) != run['selection_sha256']:
        raise ValueError('Reviewed selection changed since inference')
    chosen = set(selection['pair_ids'])
    pairs = [p for p in manifest['pairs'] if p['pair_id'] in chosen]
    units, unit_hashes = {}, {}
    for path in sorted((root / 'units').glob('*.json')):
        row = read_json(path)
        if row.get('pair_id') not in chosen or row.get('method') not in methods:
            raise ValueError('Unexpected result outside the reviewed run')
        key = (row['pair_id'], row['hypothesis'], row['condition'], row['method'])
        if key in units:
            raise ValueError('Duplicate result identity')
        units[key] = row
        unit_hashes[str(path)] = sha256(path)
    per_pair, class_rows, coverage, threshold_rows = [], [], [], []
    tasks = [('H1', 'original'), ('H1', 'exposure'), ('H1', 'shadow'), ('H0', 'exposure'), ('H0', 'shadow')]
    for pair_row in pairs:
        pair = load_pair(prepared, pair_row)
        identifier, building = pair_row['pair_id'], pair_row['building_id']
        loaded = {}
        for hypothesis, condition in tasks:
            for method in methods:
                row = units.get((identifier, hypothesis, condition, method))
                coverage.append(dict(pair_id=identifier, building_id=building, hypothesis=hypothesis,
                                     condition=condition, method=method,
                                     status=row.get('status', 'unknown') if row else 'unprocessed',
                                     error=row.get('error', '') if row else ''))
                if row and row['status'] == 'completed':
                    loaded[(hypothesis, condition, method)] = _load_result(root, row, pair, condition, hypothesis)

        def predict(item, method, goal):
            if method in NATIVE:
                return item['native']
            record = item['record']
            threshold = threshold_source.get(method, record['variant'], building, goal)
            threshold_rows.append(dict(method=method, variant=record['variant'], building_id=building,
                                       point=goal, threshold=threshold, scope=threshold_source.scope))
            return item['scores'] > threshold

        def emit(cohort, comparison, hypothesis, condition, predictor, components, goal):
            items = [loaded.get((hypothesis, condition, m)) for m in components]
            if any(item is None for item in items):
                return
            support = np.logical_and.reduce([item['support'] for item in items])
            if comparison:
                for m in components:
                    for cond in ('original', comparison):
                        other = loaded.get(('H1', cond, m))
                        if other is None:
                            return
                        support &= other['support']
            prediction = np.logical_and.reduce([predict(item, method, goal) for item, method in zip(items, components)])
            labels = np.where(support, 0, 255).astype(np.uint8) if hypothesis == 'H0' else pair['labels'].copy()
            labels[~support] = 255
            counts = metrics.evaluate_mask(prediction, labels)
            counts['marked_fraction'] = (counts['tp'] + counts['fp']) / counts['valid_pixels'] if counts['valid_pixels'] else None
            header = dict(cohort=cohort, comparison=comparison, hypothesis=hypothesis, condition=condition,
                          predictor=predictor, point='native' if goal is None else goal,
                          pair_id=identifier, building_id=building, sequence_id=pair_row['sequence_id'], frame=pair_row['frame'])
            per_pair.append({**header, **counts})
            if hypothesis == 'H1' and not comparison:
                for value in evaluate_class_slices(prediction, pair['reference_bits'], pair['source_bits'],
                                                  support, labels, manifest['categories']):
                    class_rows.append({**header, **value})

        recipes = [(m, [m]) for m in methods]
        for native in sorted(NATIVE & set(methods)):
            for gate in ('ssim', 'dinov2', 'semantic_dino', 'msdzip_abs', 'msdzip_mod256'):
                if gate in methods:
                    recipes.append((native + '&' + gate, [native, gate]))
        for hypothesis, condition in tasks:
            all_ready = all((hypothesis, condition, m) in loaded for m in methods)
            for name, components in recipes:
                goals = (None,) if len(components) == 1 and components[0] in NATIVE else budgets
                for goal in goals:
                    emit('available', '', hypothesis, condition, name, components, goal)
                    if all_ready:
                        emit('common_methods', '', hypothesis, condition, name, components, goal)
                    if len(components) == 2:
                        # Recompute the native baseline on exactly the fusion cohort.
                        if all((hypothesis, condition, m) in loaded for m in components):
                            emit('fusion/' + name, '', hypothesis, condition, components[0], [components[0]], goal)
                            emit('fusion/' + name, '', hypothesis, condition, name, components, goal)
        for name, components in recipes:
            goals = (None,) if len(components) == 1 and components[0] in NATIVE else budgets
            for condition in ('exposure', 'shadow'):
                for goal in goals:
                    for side in ('original', condition):
                        emit('matched_original', condition, 'H1', side, name, components, goal)
        print(f'Метрики: {identifier}', flush=True)
    aggregate_rows, facade_rows = aggregate(per_pair)
    class_aggregate, _ = aggregate(class_rows, ('class_id', 'class_name', 'direction'), retain_empty=True)
    deltas = []
    base_index = {tuple(row[k] for k in ('comparison', 'predictor', 'point')): row for row in aggregate_rows
                  if row['cohort'] == 'matched_original' and row['condition'] == 'original'}
    for row in aggregate_rows:
        if row['cohort'] != 'matched_original' or row['condition'] == 'original':
            continue
        before = base_index[tuple(row[k] for k in ('comparison', 'predictor', 'point'))]
        delta = {k: row[k] for k in ('condition', 'predictor', 'point', 'pairs', 'facades')}
        for metric in METRICS:
            delta['original_' + metric] = before[metric]
            delta['perturbed_' + metric] = row[metric]
            delta['delta_' + metric] = row[metric] - before[metric] if row[metric] is not None and before[metric] is not None else None
        deltas.append(delta)
    out.mkdir(parents=True)
    tables = dict(per_pair=per_pair, per_facade=facade_rows, aggregate=aggregate_rows,
                  class_per_pair=class_rows, class_aggregate=class_aggregate, coverage=coverage, deltas=deltas)
    for name, rows in tables.items():
        write_csv(out / (name + '.csv'), rows)
    unique_thresholds = list({(r['method'], r['variant'], r['building_id'], r['point']): r for r in threshold_rows}.values())
    atomic_json(out / 'thresholds.json', unique_thresholds)
    notes = [
        'Exploratory real VAL robustness on human-selected pairs; GT-enriched selection is not full-dataset performance.',
        'PREVIOUS only. H1 labels are source COCO class-bitset transitions on known annotation support at both dates.',
        'H0 is the same real source image versus its photometric perturbation; all geometric-support pixels are unchanged.',
        'H0 FPR and FPR on negative pixels of real changing pairs are separate rows.',
        'Thresholds are frozen from unperturbed calibration; they are not fitted on this reviewed or perturbed subset.',
        'P/R/F1/IoU average positive frames within each facade, then facades. FPR/marked fraction use all evaluable frames.',
        'Counts sum pixel observations; class tables measure class-conditioned change localisation, not predicted semantic class accuracy.',
        'Failures are missing results, never zero masks. Compare available, common_methods, fusion families and matched_original separately.',
        'No TEST, retraining or HISTORY-memory reconstruction. Percent differences in deltas.csv are fractions; multiply by 100 for percentage points.',
    ]
    summary = dict(status='completed', inference_status=run.get('status', 'unknown'),
                   source_run=str(root), calibration_scope=threshold_source.scope,
                   selected_pairs=len(pairs), methods=methods, notes=notes, aggregate=aggregate_rows,
                   deltas=deltas, coverage=coverage)
    atomic_json(out / 'summary.json', summary)
    atomic_json(out / 'run.json', dict(schema_version=1, kind='real_robustness_report',
                status='completed', source_run=str(root), inference_status=summary['inference_status'],
                calibration_scope=threshold_source.scope, budgets=list(budgets), no_model_inference=True))
    atomic_json(out / 'provenance.json', dict(manifest_sha256=sha256(prepared / 'manifest.json'),
                selection_sha256=run['selection_sha256'], unit_record_sha256=unit_hashes,
                calibration_sha256=threshold_source.hashes, script_sha256=sha256(__file__)))
    _html(out, summary, class_aggregate)
    print(f'Готово: {out / "report.html"}', flush=True)
    return summary


def _html(out, summary, classes):
    def table(rows, fields):
        cells = ['<table><thead><tr>' + ''.join('<th>' + html.escape(k) + '</th>' for k in fields) + '</tr></thead><tbody>']
        for row in rows:
            values = []
            for field in fields:
                value = row.get(field)
                if field in METRICS or field.startswith(('original_', 'perturbed_', 'delta_')):
                    value = '—' if value is None else f'{100 * value:.2f}'
                values.append('<td>' + html.escape(str(value if value is not None else '—')) + '</td>')
            cells.append('<tr>' + ''.join(values) + '</tr>')
        return ''.join(cells) + '</tbody></table>'
    content = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Устойчивость реальных пар</title>',
               '<style>body{font:16px system-ui;margin:24px;color:#172334;background:#f5f7fa}table{border-collapse:collapse;background:white;margin:16px 0;font-size:13px}td,th{padding:7px;border:1px solid #dce2e8;text-align:right}th{position:sticky;top:0;background:#e9eef5}td:first-child{text-align:left}section{overflow:auto}details{margin:18px 0}li{margin:7px 0}</style>',
               '<h1>Устойчивость на реальных VAL-парах</h1><p>Метрики в %. Разности — процентные пункты. Порог не меняется после аугментации.</p>',
               '<p>Состояние расчёта: ' + html.escape(summary['inference_status']) + '. Покрытие и отказы приведены отдельно.</p>',
               '<p>H0: один снимок и его изменённое освещение. H1: реальная временная пара с прежней разметкой. Отказы не считаются отсутствием изменений.</p>',
               '<p>Выбор по видимой разметке — диагностическая выборка, не оценка всего датасета.</p>',
               '<ul>' + ''.join('<li>' + html.escape(note) + '</li>' for note in summary['notes']) + '</ul>']
    fields = ['hypothesis', 'condition', 'predictor', 'point', *METRICS, 'pairs', 'positive_pairs', 'facades', 'tp', 'fp', 'fn', 'tn']
    for cohort in ('common_methods', 'available'):
        content += ['<h2>' + html.escape(cohort) + '</h2><section>', table([r for r in summary['aggregate'] if r['cohort'] == cohort], fields), '</section>']
    content += ['<h2>Изменение качества на совпадающих парах</h2><section>',
                table(summary['deltas'], ['condition', 'predictor', 'point', 'pairs', 'facades', 'original_f1', 'perturbed_f1', 'delta_f1', 'original_fpr', 'perturbed_fpr', 'delta_fpr']), '</section>',
                '<details><summary>По классам: локализация изменений в размеченных классах</summary><section>',
                table([r for r in classes if r['cohort'] == 'available' and r['direction'] == 'change'],
                      ['class_name', *fields]), '</section></details>',
                '<details><summary>Покрытие и отказы</summary><section>',
                table(summary['coverage'], ['pair_id', 'method', 'hypothesis', 'condition', 'status', 'error']), '</section></details>',
                '<p>Полные таблицы: <a href="aggregate.csv">aggregate.csv</a> · <a href="deltas.csv">deltas.csv</a> · <a href="class_aggregate.csv">class_aggregate.csv</a> · <a href="coverage.csv">coverage.csv</a></p></html>']
    (out / 'report.html').write_text('\n'.join(content), encoding='utf-8')
