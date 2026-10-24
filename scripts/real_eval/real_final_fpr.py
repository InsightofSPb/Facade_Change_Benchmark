#!/usr/bin/env python3
"""CPU LOFO real-VAL comparison with a budget on the FINAL scalar/AND/OR FPR."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

if __package__:
    from . import real_gt_calibration as calibration
    from . import real_gt_metrics as metrics
    from .real_gt_slices import SliceCollector
else:
    import real_gt_calibration as calibration
    import real_gt_metrics as metrics
    from real_gt_slices import SliceCollector


GOALS = (.01, .05)
OPERATIONS = ('native', 'scalar', 'and', 'or')


def _key(row):
    return row['sequence_id'], row['frame']


def final_fpr_profiles(native_rows, scalar_rows, heldout):
    """Negative-only profiles; equal facade, then equal negative-bearing frames.

    Both lists must already have identical keys, labels and pixel support.
    The gate is strict score > threshold. Equal scores are never split.
    """
    if len(native_rows) != len(scalar_rows):
        raise ValueError('Native/scalar rows must share exactly the same support')
    buildings = defaultdict(list)
    seen = set()
    for left, right in zip(native_rows, scalar_rows):
        key = _key(left)
        if key in seen or key != _key(right) or left['building_id'] != right['building_id']:
            raise ValueError('Duplicate or inconsistent pair/building identity')
        seen.add(key)
        if left['building_id'] == heldout:
            continue  # Neither held-out scores nor labels influence a threshold.
        labels, other_labels = np.asarray(left['labels']), np.asarray(right['labels'])
        native, scores = np.asarray(left['native']), np.asarray(right['scores'])
        if (labels.ndim != 1 or labels.dtype != np.uint8 or not np.isin(labels, (0, 1)).all()
                or not np.array_equal(labels, other_labels) or native.dtype != bool
                or native.shape != labels.shape or scores.shape != labels.shape
                or scores.dtype != np.float32 or not np.isfinite(scores).all()
                or not np.array_equal(left['pixels'], right['pixels'])):
            raise ValueError('Invalid or unmatched known-GT/native/scalar arrays')
        if np.any(labels == 0):
            buildings[left['building_id']].append((left, right))
    train_pairs = [pair for members in buildings.values() for pair in members]
    audit = dict(
        train_buildings=sorted(buildings),
        train_pair_keys=[list(_key(left)) for left, _ in train_pairs],
        train_negative_pixels=sum(int(np.count_nonzero(left['labels'] == 0)) for left, _ in train_pairs),
        train_support_sha256={f'{left["sequence_id"]}:{left["frame"]}': left.get('support_sha256')
                              for left, _ in train_pairs},
        weighting='equal facade, then equal frames containing known GT negatives',
        threshold_operator='>', calibration_split='val',
    )
    if not train_pairs:
        return dict(audit=audit, values=None, native_fpr=None)
    values, weights, selected = [], [], []
    for members in buildings.values():
        for left, right in members:
            negative = left['labels'] == 0
            count = int(np.count_nonzero(negative))
            values.append(right['scores'][negative])
            selected.append(left['native'][negative])
            weights.append(np.full(count, 1 / (len(buildings) * len(members) * count), np.float64))
    values, weights, selected = np.concatenate(values), np.concatenate(weights), np.concatenate(selected)
    order = np.argsort(values, kind='stable')
    values, weights, selected = values[order], weights[order], selected[order]
    ends = np.r_[np.flatnonzero(values[1:] != values[:-1]), len(values) - 1]
    total = float(weights.sum())
    native_weight = weights * selected
    native_total = float(native_weight.sum())
    tail_all = np.maximum(0., total - np.cumsum(weights)[ends]) / total
    tail_native = np.maximum(0., native_total - np.cumsum(native_weight)[ends]) / total
    native_fpr = float(np.clip(native_total / total, 0, 1))
    return dict(audit=audit, values=values[ends], native_fpr=native_fpr,
                scalar=np.clip(tail_all, 0, 1), and_=np.clip(tail_native, 0, native_fpr),
                or_=np.clip(native_fpr + tail_all - tail_native, native_fpr, 1))


def choose_final_threshold(profile, heldout, budget, operation):
    if operation not in OPERATIONS or not 0 <= budget < 1:
        raise ValueError('Expected native/scalar/and/or and a final FPR budget in [0,1)')
    record = dict(heldout_building=heldout, goal=float(budget), operation=operation,
                  threshold=None, gate_mode=None, achieved_train_fpr=None,
                  minimum_train_fpr=None, maximum_train_fpr=None, feasible=None,
                  native_train_fpr=profile['native_fpr'], **profile['audit'])
    if profile['values'] is None:
        return dict(record, status='unsupported', reason='no_other_facade_training_negatives')
    native = profile['native_fpr']
    minimum = native if operation in ('native', 'or') else 0.
    maximum = native if operation in ('native', 'and') else 1.
    record.update(minimum_train_fpr=minimum, maximum_train_fpr=maximum,
                  empty_gate_train_fpr=minimum, all_gate_train_fpr=maximum)
    if operation == 'native':
        feasible = native <= budget + 1e-12
        return dict(record, feasible=feasible, achieved_train_fpr=native,
                    status='fixed_reference_within_budget' if feasible else 'fixed_reference_outside_budget',
                    reason='native_binary_mask_unchanged')
    if minimum > budget + 1e-12:
        return dict(record, feasible=False, status='budget_unattainable', reason='native_OR_floor_exceeds_budget')
    if maximum <= budget + 1e-12:
        # A finite minimum-negative threshold would wrongly drop lower-scoring
        # positives. Explicit all-gate includes those without JSON infinity.
        return dict(record, feasible=True, status='calibrated', gate_mode='all', achieved_train_fpr=maximum)
    fpr = profile[{'scalar': 'scalar', 'and': 'and_', 'or': 'or_'}[operation]]
    index = int(np.flatnonzero(fpr <= budget + 1e-12)[0])
    return dict(record, feasible=True, status='calibrated', gate_mode='threshold',
                threshold=float(profile['values'][index]), achieved_train_fpr=float(fpr[index]))


def predict(native, scores, choice):
    native, scores = np.asarray(native), np.asarray(scores)
    if native.dtype != bool or native.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError('Invalid prediction arrays')
    if choice['operation'] == 'native':
        return native.copy()
    if choice['status'] != 'calibrated':
        raise ValueError('An infeasible/unsupported budget must not create a prediction')
    if choice['gate_mode'] == 'all':
        gate = np.ones(scores.shape, bool)
    elif choice['gate_mode'] == 'empty':
        gate = np.zeros(scores.shape, bool)
    elif choice['gate_mode'] == 'threshold':
        gate = scores > np.float32(choice['threshold'])
    else:
        raise ValueError('Unknown gate mode')
    return {'scalar': lambda: gate, 'and': lambda: native & gate, 'or': lambda: native | gate}[choice['operation']]()


def _primary(rows):
    return [row for row in metrics.aggregate_metrics(rows)[2] if row['aggregation'] == 'frame_then_facade_macro']


def matched_comparisons(per_pair):
    """Recompute both sides on candidate-supported folds, never compare totals."""
    families = defaultdict(list)
    for row in per_pair:
        families[row['family'], row['goal']].append(row)
    result = []
    for (family, goal), members in families.items():
        by_operation = {op: {_key(row): row for row in members if row['operation'] == op and row['valid_pixels']}
                        for op in OPERATIONS}
        for operation in ('scalar', 'and', 'or'):
            for baseline in (('native',) if operation == 'scalar' else ('native', 'scalar')):
                keys = sorted(by_operation[operation].keys() & by_operation[baseline].keys())
                if not keys:
                    continue
                before = [by_operation[baseline][key] for key in keys]
                after = [by_operation[operation][key] for key in keys]
                if any(a['support_sha256'] != b['support_sha256'] for a, b in zip(before, after)):
                    raise ValueError('Comparison supports differ')
                old, new = ({row['subset']: row for row in _primary(rows)} for rows in (before, after))
                for subset in sorted(old.keys() & new.keys()):
                    a, b = old[subset], new[subset]
                    result.append(dict(family=family, goal=goal, mode=a['mode'], subset=subset,
                                       baseline=baseline, candidate=operation,
                                       baseline_predictor=before[0]['predictor'], candidate_predictor=after[0]['predictor'],
                                       facades=a['units'], evaluated_pairs=a['evaluated_pairs'],
                                       full_family_pairs=len(by_operation['native']),
                                       heldout_buildings=sorted({row['building_id'] for row in before}),
                                       native_reference_outside_training_budget=any(
                                           row['native_reference_feasible'] is False for row in before),
                                       **{f'baseline_{m}': a[m] for m in metrics.METRICS},
                                       **{f'candidate_{m}': b[m] for m in metrics.METRICS},
                                       **{f'delta_{m}': b[m] - a[m] if a[m] is not None and b[m] is not None else None
                                          for m in metrics.METRICS}))
    return result


def _predictor(native_group, scalar_group, operation, goal):
    names = {'native': ':'.join(native_group), 'scalar': ':'.join(scalar_group),
             'and': ':'.join(native_group) + '&' + ':'.join(scalar_group),
             'or': ':'.join(native_group) + '|' + ':'.join(scalar_group)}
    return names[operation] + f'@final_fpr{goal:g}' + (':fixed' if operation == 'native' else '')


def _matched_class_comparisons(out):
    """Class deltas use controls emitted only on that candidate's feasible folds."""
    with (out / 'class_macro.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    groups = defaultdict(dict)
    for row in rows:
        if not row['family'].endswith(('/matched_and', '/matched_or')):
            continue
        key = tuple(row[field] for field in ('family', 'goal', 'region', 'class_id', 'class_name', 'direction'))
        role = 'native' if row['predictor'].endswith(':fixed') else 'candidate' if any(char in row['predictor'] for char in '&|') else 'scalar'
        groups[key][role] = row
    comparisons = []
    for key, members in groups.items():
        candidate = members['candidate']
        for baseline in ('native', 'scalar'):
            before = members[baseline]
            if any(before[field] != candidate[field] for field in
                   ('facades', 'positive_facades', 'evaluated_pairs', 'positive_pairs', 'valid_pixels', 'positive_pixels', 'negative_pixels')):
                raise ValueError('Class comparison coverage differs')
            values = {}
            for field in (*metrics.METRICS, 'marked_fraction'):
                old = float(before[field]) if before[field] else None
                new = float(candidate[field]) if candidate[field] else None
                values.update({f'baseline_{field}': old, f'candidate_{field}': new,
                               f'delta_{field}': new - old if old is not None and new is not None else None})
            comparisons.append(dict(zip(('family', 'goal', 'region', 'class_id', 'class_name', 'direction'), key),
                                    mode=candidate['mode'], baseline=baseline, candidate=key[0].rsplit('_', 1)[-1],
                                    baseline_predictor=before['predictor'], candidate_predictor=candidate['predictor'],
                                    **{field: int(candidate[field]) for field in
                                       ('facades', 'positive_facades', 'evaluated_pairs', 'positive_pairs', 'valid_pixels')}, **values))
    metrics.write_csv(out / 'class_comparisons.csv', comparisons)


def run(metrics_run, out_path, budgets=GOALS, native_methods=('anychange', 'geoscd'),
        scalar_methods=None, prediction_hook=None, class_slices=True):
    budgets = tuple(dict.fromkeys(float(value) for value in budgets))
    if not budgets or any(not np.isfinite(value) or not 0 <= value < 1 for value in budgets):
        raise ValueError('Budgets must be finite fractions in [0,1)')
    root, out = (Path(value).expanduser().resolve() for value in (metrics_run, out_path))
    if out.exists() or out == root or root in out.parents or out in root.parents:
        raise FileExistsError('Нужен новый отдельный каталог отчёта')
    source = calibration.Samples(root)  # Authenticates completed REAL VAL only.
    for value in (source.provenance['source_run'], source.provenance.get('snapshot', str(root))):
        protected = Path(value).resolve()
        if out == protected or protected in out.parents or out in protected.parents:
            raise FileExistsError('Нельзя писать внутрь исходных запусков')
    native_groups = [group for group in sorted(source.entries) if group[0] in metrics.NATIVE and group[0] in native_methods]
    scalar_groups = [group for group in sorted(source.entries) if group[0] not in metrics.NATIVE
                     and (scalar_methods is None or group[0] in scalar_methods)]
    families = [(first, second) for first in native_groups for second in scalar_groups
                if first[2] == second[2] or first[2] == 'previous' and second[2] == 'history']
    if not families:
        raise ValueError('Нет совместимых native/scalar real VAL карт для выбранных методов')
    out.mkdir(parents=True)
    config = dict(schema_version=1, metrics_run=str(root), budgets=list(budgets), native_methods=list(native_methods),
                  scalar_methods=list(scalar_methods) if scalar_methods is not None else None,
                  class_slices=class_slices, scope='exploratory real VAL LOFO final-output FPR budget',
                  no_model_inference=True, history_memory_unchanged=True)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    def state(status, **extra):
        metrics.write_json(out / 'run.json', dict(config, status=status, config_sha256=config_hash,
                                                timestamp_utc=datetime.now(timezone.utc).isoformat(), **extra))
    state('running')
    collector = None
    per_pair, audit, coverage, recipes = [], [], [], []
    try:
        collector = SliceCollector(root, source.provenance['categories']) if class_slices else None
        for number, (native_group, scalar_group) in enumerate(families, 1):
            family = 'final_fpr/' + ':'.join(native_group) + '+' + ':'.join(scalar_group)
            first, second = calibration.match_samples(source.load(native_group), source.load(scalar_group))
            buildings = sorted({row['building_id'] for row in first if len(row['labels'])})
            print(f'Итоговый FPR [{number}/{len(families)}]: {family}; пар {len(first)}, фасадов {len(buildings)}', flush=True)
            coverage.append(dict(family=family, mode=native_group[2],
                                 native_input_pairs=len(source.entries[native_group]), scalar_input_pairs=len(source.entries[scalar_group]),
                                 matched_pairs=len(first), annotated_pairs=sum(bool(len(row['labels'])) for row in first),
                                 buildings=buildings, missing_policy='missing maps stay missing; no zero imputation'))
            for heldout in buildings:
                profile = final_fpr_profiles(first, second, heldout)
                held = [(a, b) for a, b in zip(first, second) if a['building_id'] == heldout]
                print(f'  LOFO {heldout}; обучающих фасадов {len(profile["audit"]["train_buildings"])}', flush=True)
                for goal in budgets:
                    choices = {op: choose_final_threshold(profile, heldout, goal, op) for op in OPERATIONS}
                    for operation, choice in choices.items():
                        start = len(per_pair)
                        if operation == 'native' or choice['status'] == 'calibrated':
                            predictor = _predictor(native_group, scalar_group, operation, goal)
                            for left, right in held:
                                prediction = predict(left['native'], right['scores'], choice)
                                counts = metrics.evaluate_mask(prediction, left['labels'])
                                counts['ignored_pixels'] = int(np.prod(left['shape'])) - len(left['labels'])
                                row = dict(cohort=family, family=family, mode=native_group[2], predictor=predictor,
                                           operation=operation, goal=goal, heldout_building=heldout,
                                           building_id=heldout, sequence_id=left['sequence_id'], frame=left['frame'],
                                           support_sha256=left['support_sha256'], status='evaluated' if len(left['labels']) else 'no_annotation_support',
                                           train_budget_feasible=choice['feasible'], native_reference_feasible=choices['native']['feasible'],
                                           achieved_train_fpr=choice['achieved_train_fpr'], gate_mode=choice['gate_mode'], threshold=choice['threshold'],
                                           source_unit_ids=[left['unit_id'], right['unit_id']], **counts)
                                per_pair.append(row)
                                recipes.append(dict(family=family, goal=goal, operation=operation, predictor=predictor,
                                                    heldout_building=heldout, sequence_id=left['sequence_id'], frame=left['frame'],
                                                    known_support_sha256=left['support_sha256'], gt_artifact=left['gt_artifact'],
                                                    gate_mode=choice['gate_mode'], threshold=choice['threshold'], operator='>',
                                                    components=[{k: item[k] for k in ('unit_id', 'map_path', 'artifact_sha256')}
                                                                for item in (left, right)]))
                                values = dict(family=family, mode=native_group[2], predictor=predictor, goal=goal,
                                              heldout_building=heldout, sample=left, prediction=prediction)
                                if collector is not None:
                                    collector.record(**values)
                                    if operation in ('and', 'or'):
                                        # Candidate-supported fold controls prevent a
                                        # partially feasible OR from silently changing
                                        # the class cohort relative to its components.
                                        matched_family = family + '/matched_' + operation
                                        collector.record(**dict(values, family=matched_family))
                                        for baseline in ('native', 'scalar'):
                                            collector.record(**dict(values, family=matched_family,
                                                predictor=_predictor(native_group, scalar_group, baseline, goal),
                                                prediction=predict(left['native'], right['scores'], choices[baseline])))
                                if prediction_hook is not None:
                                    prediction_hook(**values)
                        evaluated = per_pair[start:]
                        held_fpr = [row['fpr'] for row in evaluated if row['fpr'] is not None]
                        audit.append(dict(choice, family=family, mode=native_group[2],
                                          actual_heldout_fpr=float(np.mean(held_fpr)) if held_fpr else None,
                                          heldout_fpr_within_budget=(float(np.mean(held_fpr)) <= goal + 1e-12) if held_fpr else None,
                                          evaluated_pairs=sum(bool(row['valid_pixels']) for row in evaluated),
                                          heldout_pairs_with_negatives=len(held_fpr)))
                        if choice['status'] in ('budget_unattainable', 'unsupported'):
                            print(f'    {operation} / {goal:.0%}: {choice["status"]}; {choice["reason"]}', flush=True)
        sequences, facades, aggregate = metrics.aggregate_metrics(per_pair)
        comparisons = matched_comparisons(per_pair)
        for name, rows in (('per_pair', per_pair), ('per_sequence', sequences), ('per_facade', facades),
                           ('aggregate', aggregate), ('calibration_audit', audit), ('coverage', coverage), ('comparisons', comparisons)):
            metrics.write_csv(out / (name + '.csv'), rows)
        metrics.write_json(out / 'thresholds.json', audit)
        metrics.write_json(out / 'prediction_recipes.json', recipes)
        if collector is not None:
            collector.finalize(out)
            _matched_class_comparisons(out)
        _write_summary(out, config, aggregate, audit, comparisons, coverage)
        script_hashes = {str(Path(module.__file__).resolve()): metrics.sha256(Path(module.__file__))
                         for module in (sys.modules[__name__], calibration, metrics)}
        script_hashes[str(Path(sys.modules[SliceCollector.__module__].__file__).resolve())] = metrics.sha256(
            Path(sys.modules[SliceCollector.__module__].__file__))
        metrics.write_json(out / 'provenance.json', dict(config, config_sha256=config_hash,
                           input_sha256=source.hashes, script_sha256=script_hashes,
                           target_definition=source.provenance.get('target_definition'), categories=source.provenance.get('categories'),
                           boundary_tolerance=source.provenance.get('boundary_tolerance'),
                           checkpoint_selection_not_cross_validated=True, threshold_operator='>',
                           no_class_specific_thresholds=True,
                           class_comparison_cohort='family/matched_and and family/matched_or: candidate and both controls on candidate-feasible folds only',
                           output_sha256={path.name: metrics.sha256(path) for path in sorted(out.iterdir()) if path.name != 'run.json'}))
        state('completed', families=len(families), evaluated_predictions=len(per_pair))
    except Exception as exc:
        state('failed', error=f'{type(exc).__name__}: {exc}')
        raise
    print(f'Готово: {out / "summary.txt"}', flush=True)
    return dict(status='completed', per_pair=per_pair, calibration_audit=audit, comparisons=comparisons, coverage=coverage)


def _write_summary(out, config, aggregate, audit, comparisons, coverage):
    percent = lambda value: '—' if value is None else f'{100 * value:.2f}'
    primary = {(r['cohort'], r['predictor'], r['subset']): r for r in aggregate if r['aggregation'] == 'frame_then_facade_macro'}
    notes = [
        'Разведочный real VAL LOFO: общий бюджет ИТОГОВОГО FPR на калибровочных фасадах.',
        'Фактический FPR исключённого фасада может отличаться. Это не сравнение при точно равном FPR и не независимый TEST.',
        'Порог каждого SCALAR/AND/OR выбирается только по отрицательным пикселям других фасадов, без оптимизации по F1/Recall.',
        'Равный вес фасадов, затем кадров с отрицательными пикселями. score > threshold; одинаковые значения не разделяются.',
        'Native GeoSCD/AnyChange неизменны; fixed_reference_outside_budget нельзя представлять как метод, уложившийся в бюджет.',
        'OR имеет нижний предел native FPR. Недостижимый бюджет помечается budget_unattainable, метрики такого fold не выдумываются.',
        'AND может совпасть с native, когда весь native укладывается в бюджет. all-gate означает полный пропуск scalar-ветви.',
        'У каждой native+scalar family общие пары и точные пиксели для калибровки и оценки. Отказы моделей не заменяются нулями.',
        'Сравнения с обоими компонентами пересчитаны на одних и тех же доступных fold/парах; partial OR coverage явно сохраняется.',
        'P/R/F1/IoU: положительные кадры внутри фасада, затем фасады. FPR: все кадры с известными отрицательными пикселями.',
        'Классовые срезы используют те же замороженные пороги; это локализация в GT-областях класса, не предсказание класса.',
        'Для сравнения классов используйте class_comparisons.csv или family/matched_and и family/matched_or в class_macro.csv: одинаковые fold кандидата и обоих компонентов.',
        'Обычные family в class_macro.csv показывают доступное покрытие каждого метода; OR там может иметь меньше фасадов и не сравнивается напрямую с полной native/scalar строкой.',
        'Веса моделей и память HISTORY не пересчитываются. LOFO относится к итоговому порогу; checkpoint ранее мог выбираться по VAL.',
        'CSV: доли; таблицы: проценты. TP/FP/FN/TN — пиксельные наблюдения, не независимые объекты.',
    ]
    lines = notes + ['', 'ИТОГОВЫЕ МЕТРИКИ (покрытие может различаться при недостижимом OR)',
                     'family | predictor | P | R | F1 | IoU | held-out FPR | пары +/все | фасады']
    for row in aggregate:
        if row['aggregation'] != 'frame_then_facade_macro' or row['subset'] != 'all':
            continue
        positive = primary.get((row['cohort'], row['predictor'], 'positive'), {})
        lines.append(' | '.join([row['cohort'], row['predictor'], *[percent(positive.get(m)) for m in ('precision', 'recall', 'f1', 'iou')],
                                 percent(row['fpr']), f'{positive.get("evaluated_pairs", 0)}/{row["evaluated_pairs"]}', str(row['units'])]))
    lines += ['', 'ДОСТИЖИМОСТЬ ПО FOLD (подробности и фактический heldout FPR: calibration_audit.csv)',
              'family | budget | operation | статусы | train FPR min–max | heldout FPR min–max']
    grouped = defaultdict(list)
    for row in audit:
        grouped[row['family'], row['goal'], row['operation']].append(row)
    for (family, goal, operation), rows in grouped.items():
        statuses = {status: sum(row['status'] == status for row in rows) for status in sorted({row['status'] for row in rows})}
        def span(key):
            values = [row[key] for row in rows if row[key] is not None]
            return percent(min(values)) + '–' + percent(max(values)) if values else '—'
        lines.append(' | '.join([family, percent(goal), operation, json.dumps(statuses), span('achieved_train_fpr'), span('actual_heldout_fpr')]))
    lines += ['', 'СОЧЕТАНИЯ ПРОТИВ ОБОИХ КОМПОНЕНТОВ НА ОБЩИХ FOLD',
              'family | budget | база → кандидат | F1 | ΔF1 п.п. | IoU | FPR | общих пар/полных | native вне бюджета']
    for row in comparisons:
        if row['subset'] != 'positive' or row['candidate'] == 'scalar':
            continue
        all_row = next(r for r in comparisons if r['family'] == row['family'] and r['goal'] == row['goal']
                       and r['baseline'] == row['baseline'] and r['candidate'] == row['candidate'] and r['subset'] == 'all')
        lines.append(' | '.join([row['family'], percent(row['goal']), row['baseline'] + ' → ' + row['candidate'],
                                 percent(row['baseline_f1']) + '→' + percent(row['candidate_f1']), percent(row['delta_f1']),
                                 percent(row['baseline_iou']) + '→' + percent(row['candidate_iou']),
                                 percent(all_row['baseline_fpr']) + '→' + percent(all_row['candidate_fpr']),
                                 f'{all_row["evaluated_pairs"]}/{row["full_family_pairs"]}', str(row['native_reference_outside_training_budget'])]))
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    metrics.write_json(out / 'summary.json', dict(config, status='completed', notes=notes, aggregate=aggregate,
                                                calibration_audit=audit, comparisons=comparisons, coverage=coverage))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics-run', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--budgets', type=float, nargs='+', default=list(GOALS), help='Доли: 0.01 0.05; можно добавить 0.10')
    parser.add_argument('--native-methods', nargs='+', choices=sorted(metrics.NATIVE), default=['anychange', 'geoscd'])
    parser.add_argument('--scalar-methods', nargs='+', help='Не задано: все сохранённые scalar методы и режимы')
    parser.add_argument('--no-class-slices', action='store_true')
    args = parser.parse_args()
    run(args.metrics_run, args.out, args.budgets, args.native_methods, args.scalar_methods, class_slices=not args.no_class_slices)


if __name__ == '__main__':
    main()
