"""Resumable CPU-only real-VAL LOFO analysis of frozen night-run maps."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import html
import itertools
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from . import night_pair_calibration as calibration
from . import night_report as report
from .night_calibration_data import Samples
from .real_gt_metrics import evaluate_mask
from .real_gt_slices import evaluate_class_slices
from .robustness_common import atomic_json, digest, read_json, sha256


WARNINGS = [
    'Разведочная LOFO-оценка на выбранном реальном VAL. TEST/TRAIN не участвуют в подборе или оценке.',
    'Целевые 1%/5% ограничивают итоговый FPR на калибровочных фасадах. Фактический FPR оставленного фасада может быть выше.',
    'Для двух scalar-карт используются ранги по отрицательным пикселям калибровки: AND=min рангов, OR=max рангов; затем калибруется итоговая карта.',
    'AnyChange/GeoSCD сохраняют фиксированные бинарные маски. Недостижимый OR не заменяется пустой маской; фиксированные маски вне бюджета явно отмечены.',
    'Карты и обновления памяти HISTORY заморожены. Меняется только порог итоговой маски; это сравнение готовых режимов, а не повторное обучение памяти.',
    'GT — изменение набора аннотированных классов; пустая аннотация неизвестна. Выборка обогащена изменениями, классовые срезы не означают распознавание класса детектором.',
    'P/R/F1/IoU: среднее положительных кадров внутри фасада, затем равный вес фасадов. FPR: кадры с известными отрицательными пикселями, затем фасады.',
    'Сочетания сравниваются с обоими компонентами на одинаковых парах и пикселях; неполное покрытие не равно оценке всей выборки. Интервалы и статистическая значимость не рассчитаны.',
]
CLASS_FIELDS = ('class_id', 'class_name', 'channel', 'direction')


def _aggregate(rows, classes=False):
    extra = ('role', *CLASS_FIELDS) if classes else ('role',)
    tables, facades = report.aggregate(rows, extra)
    keep = lambda r: r['split_scope'] == 'val' and r['training_scope'] == 'all'
    return ([dict(r, family=r['comparison']) for r in tables if keep(r)],
            [dict(r, family=r['comparison']) for r in facades if keep(r)])


def _mode(channels):
    modes = {channel.split(':')[1] for channel in channels}
    return next(iter(modes)) if len(modes) == 1 else 'previous_history'


def _pair_key(row):
    return row['sequence_id'], row['frame']


def _difference(before, after):
    if any(before[k] != after[k] for k in ('facades', 'evaluated_pairs', *report.COUNTS[4:])):
        raise ValueError('Matched comparison coverage differs')
    return dict(facades=before['facades'], evaluated_pairs=before['evaluated_pairs'],
                valid_pixels=before['valid_pixels'], positive_pixels=before['positive_pixels'],
                negative_pixels=before['negative_pixels'],
                **{f'baseline_{k}': before[k] for k in (*report.METRICS, *report.COUNTS)},
                **{f'candidate_{k}': after[k] for k in (*report.METRICS, *report.COUNTS)},
                **{f'delta_{k}': after[k] - before[k]
                   if after[k] is not None and before[k] is not None else None for k in report.METRICS})


def _comparisons(rows, class_rows, channels, budgets):
    if len(channels) == 1:
        return [], []
    pairs = [(base, candidate, 'fusion') for candidate in ('and', 'or') for base in ('left', 'right')]
    if channels[0].split(':')[0] == channels[1].split(':')[0]:
        previous = 'left' if channels[0].endswith(':previous') else 'right'
        pairs.append((previous, 'right' if previous == 'left' else 'left', 'previous_history'))
    comparisons, per_facade = [], []
    for point, (base, candidate, kind) in itertools.product(budgets, pairs):
        before = {_pair_key(r): r for r in rows if r['point'] == point and r['role'] == base}
        after = {_pair_key(r): r for r in rows if r['point'] == point and r['role'] == candidate}
        keys = before.keys() & after.keys()
        if not keys:
            continue
        for key in keys:
            if before[key]['support_sha256'] != after[key]['support_sha256']:
                raise ValueError('Matched pair supports differ')
        raw = [r for key in sorted(keys) for r in (before[key], after[key])]
        header = dict(family=raw[0]['family'], point=point, mode=_mode(channels), target='step_event',
                      comparison_kind=kind, baseline=before[next(iter(keys))]['predictor'],
                      candidate=after[next(iter(keys))]['predictor'], baseline_role=base, candidate_role=candidate,
                      full_family_pairs=len(before),
                      baseline_outside_training_budget=any(before[key]['calibration_feasible'] is False for key in keys),
                      candidate_outside_training_budget=any(after[key]['calibration_feasible'] is False for key in keys))
        for is_class, source in ((False, raw), (True, [r for r in class_rows if r['point'] == point
                                                     and r['role'] in (base, candidate) and _pair_key(r) in keys])):
            tables, facades = _aggregate(source, is_class)
            groups, facade_groups = {}, {}
            for row in tables:
                key = tuple(row[field] for field in CLASS_FIELDS) if is_class else ()
                groups.setdefault(key, {})[row['role']] = row
            for row in facades:
                key = (tuple(row[field] for field in CLASS_FIELDS) if is_class else ()) + (row['building_id'],)
                facade_groups.setdefault(key, {})[row['role']] = row
            for key, members in groups.items():
                if not {base, candidate} <= members.keys():
                    continue
                current = dict(header, region='class' if is_class else 'overall',
                               **(dict(zip(CLASS_FIELDS, key)) if is_class else {}),
                               **_difference(members[base], members[candidate]))
                deltas = []
                for fkey, fmembers in facade_groups.items():
                    if fkey[:-1] != key or not {base, candidate} <= fmembers.keys():
                        continue
                    a, b = fmembers[base], fmembers[candidate]
                    values = _difference(dict(a, facades=1), dict(b, facades=1))
                    if values['delta_f1'] is not None:
                        deltas.append(values['delta_f1'])
                    per_facade.append(dict(header, region=current['region'],
                                           **(dict(zip(CLASS_FIELDS, key)) if is_class else {}),
                                           building_id=fkey[-1], **values))
                current.update(improved_f1_facades=sum(x > 1e-12 for x in deltas),
                               worsened_f1_facades=sum(x < -1e-12 for x in deltas),
                               tied_f1_facades=sum(abs(x) <= 1e-12 for x in deltas))
                comparisons.append(current)
    return comparisons, per_facade


def _evaluate_family(samples, channels, budgets):
    started = time.monotonic()
    family = '__'.join(channels)
    data = samples.rows(channels)
    roles = ('single',) if len(channels) == 1 else ('left', 'right', 'and', 'or')
    names = ({'single': channels[0]} if len(channels) == 1 else
             dict(left=channels[0], right=channels[1], and_='&'.join(channels), or_='|'.join(channels)))
    if len(channels) == 2:
        names.update({'and': names.pop('and_'), 'or': names.pop('or_')})
    rows, class_rows, audit = [], [], []
    for heldout in samples.buildings:
        testing = [r for r in data if r['building_id'] == heldout]
        left = [{k: v for k, v in r.items() if k != 'right'} for r in data]
        profiles = {'single' if len(channels) == 1 else 'left': calibration.prepare(left, heldout)}
        if len(channels) == 2:
            right = [{**{k: v for k, v in r.items() if k != 'right'}, 'left': r['right']} for r in data]
            profiles['right'] = calibration.prepare(right, heldout)
            profiles['pair'] = calibration.prepare(data, heldout)
        for point, role in itertools.product(budgets, roles):
            operation = role if role in ('and', 'or') else 'single'
            model = calibration.choose(profiles['pair' if role in ('and', 'or') else role], point, operation)
            choice = model['choice']
            current = dict(choice, family=family, point=point, role=role, predictor=names[role], heldout_building=heldout)
            evaluated = []
            if choice['status'] not in ('unsupported', 'budget_unattainable'):
                for row in testing:
                    prediction_row = dict(row, left=row['right']) if role == 'right' else row
                    prediction = calibration.predict(prediction_row, model)
                    values = evaluate_mask(prediction, row['labels'])
                    values['marked_fraction'] = (values['tp'] + values['fp']) / values['valid_pixels'] if values['valid_pixels'] else None
                    header = dict(family=family, comparison=family, role=role, predictor=names[role], point=point,
                                  mode=_mode(channels), target='step_event', split='val',
                                  building_id=heldout, heldout_building=heldout,
                                  sequence_id=row['sequence_id'], frame=row['frame'],
                                  training_exposure=report._exposure([c.split(':')[0] for c in channels], heldout, samples.origins),
                                  support_sha256=row['support_sha256'], calibration_status=choice['status'],
                                  calibration_feasible=choice['feasible'], achieved_train_fpr=choice.get('achieved_train_fpr'))
                    rows.append(dict(header, **values))
                    evaluated.append(values)
                    slices = evaluate_class_slices(prediction, row['reference_bits'], row['source_bits'],
                                                   np.ones(row['labels'].shape, bool), row['labels'], samples.categories)
                    class_rows.extend(dict(header, **s) for s in slices)
            fprs = [r['fpr'] for r in evaluated if r['fpr'] is not None]
            current.update(evaluated_pairs=len(evaluated), heldout_fpr=float(np.mean(fprs)) if fprs else None,
                           heldout_valid_pixels=sum(r['valid_pixels'] for r in evaluated))
            audit.append(current)
    tables, facades = _aggregate(rows)
    classes, class_facades = _aggregate(class_rows, True)
    comparisons, comparison_facades = _comparisons(rows, class_rows, channels, budgets)
    input_coverage = samples.coverage(channels)
    expected = sum(len(s['frames']) - 1 for s in samples.manifest['sequences'] if s['split'] == 'val')
    coverage = []
    for point, role in itertools.product(budgets, roles):
        selected = [r for r in rows if r['role'] == role and r['point'] == point]
        folds = [r for r in audit if r['role'] == role and r['point'] == point]
        coverage.append(dict(family=family, role=role, predictor=names[role], point=point,
                             expected_pairs=expected,
                             input_completed_pairs=sum(r['completed_common_pairs'] for r in input_coverage),
                             input_evaluable_pairs=len(data), evaluated_pairs=len(selected),
                             expected_facades=len(samples.buildings), evaluated_facades=len({r['building_id'] for r in selected}),
                             feasible_folds=sum(r['feasible'] is True for r in folds),
                             unattainable_folds=sum(r['status'] == 'budget_unattainable' for r in folds),
                             outside_budget_fixed_folds=sum(r['status'] == 'fixed_reference_outside_budget' for r in folds),
                             unsupported_folds=sum(r['status'] == 'unsupported' for r in folds),
                             valid_pixels=sum(r['valid_pixels'] for r in selected)))
    return dict(channels=list(channels), family=family, tables=tables, per_pair=rows, per_facade=facades,
                per_class=classes, per_class_facade=class_facades, comparisons=comparisons,
                comparison_per_facade=comparison_facades, calibration_audit=audit, coverage=coverage,
                input_coverage=[dict(r, family=family) for r in input_coverage],
                cpu_elapsed_seconds=time.monotonic() - started)


def _source_code():
    directory = Path(__file__).parent
    files = ('night_calibration.py', 'night_calibration_data.py', 'night_pair_calibration.py',
             'night_common.py', 'night_report.py', 'real_final_fpr.py', 'real_gt_metrics.py',
             'real_gt_slices.py', 'robustness_common.py', 'robustness_prepare.py', 'real_temporal_night.py')
    return {name: sha256(directory / name) for name in files}


def _git_state():
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        try:
            return subprocess.run(['git', *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return dict(commit=git('rev-parse', 'HEAD'), dirty_diff_sha256=digest(git('diff', 'HEAD') or ''),
                note='Relevant tracked and untracked executable files are also hashed separately.')


def _check_output(out, samples):
    protected = [samples.root, samples.prepared]
    protected += [Path(samples.manifest[k]).expanduser().resolve() for k in ('source_run', 'snapshot') if samples.manifest.get(k)]
    for path in protected:
        if out == path or out in path.parents or path in out.parents:
            raise ValueError('Analysis output overlaps protected input data; choose a new sibling run directory')


def _write_reports(out, receipts):
    fields = ('tables', 'per_pair', 'per_facade', 'per_class', 'per_class_facade', 'comparisons',
              'comparison_per_facade', 'calibration_audit', 'coverage', 'input_coverage')
    # Two passes allow heterogeneous class/overall and scalar/native audit fields
    # without retaining all per-pair records from all method families in RAM.
    columns = {key: [] for key in fields}
    for receipt in receipts:
        data = read_json(out / receipt['path'])
        for key in fields:
            columns[key] = list(dict.fromkeys([*columns[key], *(k for row in data[key] for k in row)]))
    handles, writers = {}, {}
    tables, comparisons, timings = [], [], []
    try:
        for key in fields:
            handles[key] = (out / f'{key}.csv').open('w', newline='', encoding='utf-8')
            writers[key] = csv.DictWriter(handles[key], columns[key] or ['status'])
            writers[key].writeheader()
        for receipt in receipts:
            data = read_json(out / receipt['path'])
            for key in fields:
                for row in data[key]:
                    writers[key].writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                                          for k, v in row.items()})
            tables.extend(data['tables'])
            comparisons.extend(r for r in data['comparisons'] if r['region'] == 'overall')
            timings.append(dict(family=data['family'], cpu_elapsed_seconds=data['cpu_elapsed_seconds']))
    finally:
        for stream in handles.values():
            stream.close()
    report._write_csv(out / 'cpu_timing.csv', timings)
    standalone = [r for r in tables if r['role'] == 'single']
    cols = ('predictor', 'point', 'precision', 'recall', 'f1', 'iou', 'fpr', 'facades', 'evaluated_pairs')
    comparison_cols = ('baseline', 'candidate', 'point', 'baseline_f1', 'candidate_f1', 'delta_f1',
                       'baseline_fpr', 'candidate_fpr', 'facades', 'evaluated_pairs',
                       'improved_f1_facades', 'worsened_f1_facades')
    content = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>VAL LOFO — итоговый FPR</title>',
               '<style>body{font:14px system-ui;margin:2rem}table{border-collapse:collapse}td,th{padding:.35rem;border:1px solid #ccc}p{max-width:1100px}summary{cursor:pointer}</style>',
               '<h1>Реальный VAL: калибровка итогового FPR</h1>',
               '<p>Метрики в таблицах — доли (0.05 = 5%); CSV также в долях. Каждый фасад оценивается с порогами остальных фасадов.</p>']
    content += ['<p>' + html.escape(warning) + '</p>' for warning in WARNINGS]
    # Use a plain table here: the earlier report formatter scales only a subset
    # of prefixed metrics and would mix fractions and percentages.
    def table(items, names):
        return '<table><tr>' + ''.join('<th>' + html.escape(n) + '</th>' for n in names) + '</tr>' + ''.join(
            '<tr>' + ''.join('<td>' + html.escape('—' if row.get(n) is None else
                f'{row[n]:.4f}' if isinstance(row[n], float) else str(row[n])) + '</td>' for n in names) + '</tr>' for row in items) + '</table>'
    content.append('<h2>Одиночные методы</h2>' + table(standalone, cols))
    for kind, title in (('previous_history', 'PREVIOUS → HISTORY'), ('fusion', 'AND/OR против каждого компонента')):
        current = [r for r in comparisons if r['comparison_kind'] == kind]
        content.append('<details open><summary>' + title + '</summary>' + table(current, comparison_cols) + '</details>')
    content.append('<p>Достижимость и пороги: calibration_audit.csv. Покрытие: coverage.csv и input_coverage.csv. '
                   'Классы: per_class.csv; классовые сопоставления: comparisons.csv. Время исходного инференса: timing.csv; CPU-анализ: cpu_timing.csv.</p></html>')
    (out / 'report.html').write_text('\n'.join(content), encoding='utf-8')
    lines = ['Реальный VAL, LOFO, сохранённые карты; без повторного инференса.', *WARNINGS, '',
             'Одиночные методы: целевой FPR; F1; фактический FPR; фасады; пары.']
    for row in standalone:
        pct = lambda v: 'нет' if v is None else f'{100*v:.2f}%'
        lines.append(f"{row['predictor']} @ {pct(row['point'])}: F1={pct(row['f1'])}; FPR={pct(row['fpr'])}; {row['facades']} фасадов, {row['evaluated_pairs']} пар")
    lines += ['', 'Полные попарные результаты: comparisons.csv; проверять обе базы, фактический FPR и покрытие.']
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(run_path, out_path, budgets=(.01, .05), methods=None, resume=False):
    started = time.monotonic()
    if (not budgets or len(set(budgets)) != len(budgets)
            or any(not np.isfinite(b) or not 0 < b < 1 for b in budgets)):
        raise ValueError('Budgets must be distinct finite values between zero and one')
    samples = Samples(run_path, methods=methods)
    if len(samples.buildings) < 2:
        raise ValueError('LOFO requires at least two distinct VAL facades')
    out = Path(out_path).expanduser().resolve()
    _check_output(out, samples)
    families = [(channel,) for channel in samples.channels] + list(itertools.combinations(samples.channels, 2))
    contract = dict(schema_version=1, kind='night_real_val_lofo', source=samples.provenance,
                    budgets=list(budgets), channels=list(samples.channels), code_sha256=_source_code(),
                    protocol='negative_only_equal_facade_final_FPR_strict_rank_v1',
                    calibration_scope='real_VAL_leave_one_facade_out', no_model_inference=True)
    contract_hash = digest(contract)
    manifest_path = out / 'run.json'
    if out.exists() and any(out.iterdir()):
        if not resume or not manifest_path.is_file():
            raise FileExistsError('Use a new output directory, or --resume for the same analysis')
        saved = read_json(manifest_path)
        if saved.get('contract_sha256') != contract_hash or saved.get('contract') != contract:
            raise ValueError('Inputs, code or analysis settings changed; use a new output directory')
    else:
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest_path, dict(run_id=out.name, created_at=datetime.now(timezone.utc).isoformat(),
                                      contract=contract, contract_sha256=contract_hash, git=_git_state(),
                                      environment=dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform())))
    receipts, reused = [], 0
    for index, channels in enumerate(families, 1):
        key = digest(channels)[:24]
        path = out / 'families' / f'{key}.json'
        marker = out / 'families' / f'{key}.done.json'
        if marker.is_file():
            receipt = read_json(marker)
            if (receipt.get('contract_sha256') != contract_hash or receipt.get('channels') != list(channels)
                    or receipt.get('path') != str(path.relative_to(out)) or not path.is_file()
                    or receipt.get('sha256') != sha256(path)):
                raise ValueError(f'Changed or incompatible completed family: {key}')
            reused += 1
            print(f'[{index}/{len(families)}] {" + ".join(channels)} — готовый CPU-результат', flush=True)
        else:
            print(f'[{index}/{len(families)}] {" + ".join(channels)} — калибровка VAL', flush=True)
            result = _evaluate_family(samples, channels, budgets)
            atomic_json(path, result)
            receipt = dict(path=str(path.relative_to(out)), sha256=sha256(path), channels=list(channels),
                           contract_sha256=contract_hash)
            atomic_json(marker, receipt)
        receipts.append(receipt)
    samples.validate_snapshot()
    _write_reports(out, receipts)
    report._write_csv(out / 'timing.csv', samples.timings)
    outputs = {path.name: sha256(path) for path in out.iterdir() if path.suffix in ('.csv', '.html', '.txt')}
    provenance = dict(kind='night_real_val_lofo_completed', status='completed', contract_sha256=contract_hash,
                      source=samples.provenance, families=receipts, outputs=outputs, warnings=WARNINGS,
                      no_model_inference=True, metrics_csv_units='fractions',
                      history_update_thresholds=samples.run.get('contract', {}).get('history_thresholds', {}),
                      elapsed_seconds_this_call=time.monotonic() - started, reused_families=reused,
                      finished_at=datetime.now(timezone.utc).isoformat())
    atomic_json(out / 'provenance.json', provenance)
    print(f'Готово: {out / "report.html"}; семейств: {len(families)}; переиспользовано: {reused}', flush=True)
    return dict(status='completed', out=str(out), families=len(families), reused_families=reused)
