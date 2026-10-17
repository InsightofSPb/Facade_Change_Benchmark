#!/usr/bin/env python3
"""Разведочная LOFO-калибровка на real VAL по сохранённым картам и GT; без inference."""
import argparse
import hashlib
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

if __package__:
    from . import real_gt_metrics as metrics
else:
    import real_gt_metrics as metrics

GOALS = (.01, .05)


def _training(rows, heldout):
    selected = [row for row in rows if row['building_id'] != heldout and np.any(np.asarray(row['labels']) == 0)]
    for row in selected:
        if np.asarray(row['scores']).shape != np.asarray(row['labels']).shape or not np.isfinite(row['scores']).all() or not np.isin(row['labels'], [0, 1]).all():
            raise ValueError('Калибровка принимает только конечные оценки известных GT-пикселей')
    return selected


def _profile(training_rows, heldout_building):
    rows = _training(training_rows, heldout_building)
    by_building = defaultdict(list)
    for row in rows:
        by_building[row['building_id']].append(row)
    audit = dict(train_buildings=sorted(by_building), train_pair_keys=[[row['sequence_id'], row['frame']] for row in rows],
                 train_valid_pixels=sum(len(row['labels']) for row in rows),
                 train_pair_valid_pixels={f"{row['sequence_id']}:{row['frame']}": len(row['labels']) for row in rows},
                 train_negative_pixels=sum(int((row['labels'] == 0).sum()) for row in rows),
                 train_support_sha256={f"{row['sequence_id']}:{row['frame']}": row.get('support_sha256') for row in rows})
    if not rows:
        return None, None, audit
    values, weights = [], []
    for members in by_building.values():
        for row in members:
            negative = np.asarray(row['scores'])[np.asarray(row['labels']) == 0]
            values.append(negative)
            weights.append(np.full(len(negative), 1 / (len(by_building) * len(members) * len(negative)), np.float64))
    values, weights = np.concatenate(values), np.concatenate(weights)
    order = np.argsort(values, kind='stable')
    values, weights = values[order], weights[order]
    ends = np.r_[np.flatnonzero(values[1:] != values[:-1]), len(values) - 1]
    cumulative = np.cumsum(weights)
    fpr = np.clip((cumulative[-1] - cumulative[ends]) / cumulative[-1], 0, 1)
    return values[ends], fpr, audit


def _choice(profile, heldout_building, budget):
    values, fpr, audit = profile
    if not 0 <= budget < 1:
        raise ValueError('Целевой FPR должен принадлежать [0, 1)')
    if values is None:
        return dict(status='unsupported', reason='no_training_negative_pixels', threshold=None, achieved_fpr=None,
                    heldout_building=heldout_building, goal=budget, **audit)
    index = int(np.flatnonzero(fpr <= budget + 1e-12)[0])
    return dict(status='calibrated', threshold=float(values[index]), achieved_fpr=float(fpr[index]),
                heldout_building=heldout_building, goal=budget, operator='>',
                weighting='equal facade, then equal frames with known negative pixels', **audit)


def calibrate_threshold(training_rows, heldout_building, budget):
    return _choice(_profile(training_rows, heldout_building), heldout_building, budget)


def _precision_recall(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels)
    if scores.ndim != 1 or labels.shape != scores.shape or not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError('AP принимает одномерные конечные оценки и известные GT 0/1')
    positives = int((labels == 1).sum())
    if not positives:
        return None
    order = np.argsort(-scores, kind='stable')
    ordered_scores, ordered_labels = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]), len(scores) - 1]
    tp = np.cumsum(ordered_labels == 1)[ends]
    recall, precision = tp / positives, tp / (ends + 1)
    return recall, precision


def average_precision(scores, labels):
    curve = _precision_recall(scores, labels)
    if curve is None:
        return None
    recall, precision = curve
    return float(np.sum(np.diff(np.r_[0., recall]) * precision))


def match_samples(left, right):
    """The same pair keys and exact pixel positions for training AND evaluation."""
    matched_left, matched_right = [], []
    right_by_key = {(row['sequence_id'], row['frame']): row for row in right}
    for old in left:
        key = old['sequence_id'], old['frame']
        if key not in right_by_key:
            continue
        new = right_by_key[key]
        if old['building_id'] != new['building_id'] or old['shape'] != new['shape']:
            raise ValueError('Несовместимые фасады/решётки в сравнении')
        pixels, old_indices, new_indices = np.intersect1d(old['pixels'], new['pixels'], assume_unique=True, return_indices=True)
        if not np.array_equal(old['labels'][old_indices], new['labels'][new_indices]):
            raise ValueError('Разметка на общей поддержке сравниваемых карт различается')
        for source, indices, target in ((old, old_indices, matched_left), (new, new_indices, matched_right)):
            item = {**source, 'pixels': pixels, 'labels': source['labels'][indices], 'scores': source['scores'][indices]}
            if source.get('native') is not None:
                item['native'] = source['native'][indices]
            item['support_sha256'] = hashlib.sha256(pixels.tobytes() + item['labels'].tobytes()).hexdigest()
            target.append(item)
    return matched_left, matched_right


class Samples:
    """At most three groups of compact known-GT vectors; full maps are temporary."""
    def __init__(self, root):
        self.root = root
        self.summary = metrics.read_json(root / 'summary.json')
        self.provenance = metrics.read_json(root / 'provenance.json')
        self.hashes = {str(root / name): metrics.sha256(root / name) for name in ('summary.json', 'provenance.json', 'per_pair.csv')}
        unit_records = {}
        for value, expected in self.provenance['source_metadata_sha256'].items():
            path = Path(value)
            if metrics.sha256(path) != expected:
                raise ValueError(f'Изменились исходные метаданные: {path}')
            self.hashes[str(path)] = expected
            if path.name == '2026-10-17_source_units.json':
                unit_records.update({row['unit_id']: row for row in metrics.read_json(path)})
            elif path.parent.name == 'units':
                row = metrics.read_json(path)
                if 'unit_id' in row:
                    unit_records[row['unit_id']] = row
        self.artifacts = self.provenance['input_artifact_sha256']
        paths_by_id = defaultdict(list)
        for path in self.artifacts:
            paths_by_id[Path(path).stem].append(Path(path))
        self.gt_records = {row['artifact_path']: row for row in self.summary['gt_records']}
        self.entries = defaultdict(dict)
        self.cache = OrderedDict()
        self.gt_cache = OrderedDict()
        for row in self.summary['per_pair']:
            if row['cohort'] != 'available' or len(row['source_unit_ids']) != 1 or '&' in row['predictor'] or '|' in row['predictor']:
                continue
            unit_id = row['source_unit_ids'][0]
            record = unit_records[unit_id]
            if (record.get('domain'), record.get('split'), record.get('status')) != ('real', 'val', 'completed'):
                raise ValueError('Калибровка допускает только готовые реальные VAL-карты')
            if any(record.get(key) != row.get(key) for key in ('sequence_id', 'frame', 'building_id', 'mode', 'target')):
                raise ValueError('Метаданные карты отличаются от вычисленной GT-пары')
            paths = paths_by_id[unit_id]
            if len(paths) != 1 or self.artifacts[str(paths[0])] != record['artifact_sha256']:
                raise ValueError('Неоднозначная или неподтверждённая исходная карта')
            group = (record['method'], record['variant'], record['mode'])
            key = (row['sequence_id'], row['frame'])
            entry = dict(row=row, record=record, map_path=paths[0])
            if key in self.entries[group] and self.entries[group][key]['record'] != record:
                raise ValueError('Дублирующиеся исходные карты')
            self.entries[group][key] = entry

    def gt(self, relative):
        if relative in self.gt_cache:
            self.gt_cache.move_to_end(relative)
            return self.gt_cache[relative]
        path = (self.root / relative).resolve()
        expected = self.gt_records[relative]['artifact_sha256']
        if self.root not in path.parents or self.provenance['output_gt_sha256'].get(relative) != expected or metrics.sha256(path) != expected:
            raise ValueError('Не совпадает SHA-256 вычисленной GT-маски')
        self.hashes[str(path)] = expected
        with np.load(path, allow_pickle=False) as archive:
            labels = archive['labels'].copy()
        if labels.dtype != np.uint8 or labels.ndim != 2 or not np.isin(labels, [0, 1, 255]).all():
            raise ValueError('Неверный формат GT')
        self.gt_cache[relative] = labels
        while len(self.gt_cache) > 16:
            self.gt_cache.popitem(last=False)
        return labels

    def load(self, group):
        if group in self.cache:
            self.cache.move_to_end(group)
            return self.cache[group]
        print('Читаем сохранённые оценки: ' + ':'.join(group), flush=True)
        rows = []
        for key, entry in sorted(self.entries[group].items()):
            path, record, row = entry['map_path'], entry['record'], entry['row']
            if metrics.sha256(path) != record['artifact_sha256']:
                raise ValueError(f'Изменилась сохранённая карта: {path}')
            self.hashes[str(path)] = record['artifact_sha256']
            labels = self.gt(row['gt_artifact'])
            with np.load(path, allow_pickle=False) as archive:
                scores, support = archive['scores'], archive['support']
                native = archive['native_prediction'] if group[0] in metrics.NATIVE else None
            if (scores.dtype != np.float32 or support.dtype != bool or scores.shape != labels.shape or support.shape != labels.shape
                    or not np.isfinite(scores[support]).all() or not np.isnan(scores[~support]).all()
                    or (native is not None and (native.dtype != bool or native.shape != labels.shape))):
                raise ValueError('Некорректные оценки/поддержка исходной карты')
            pixels = np.flatnonzero((labels != 255) & support).astype(np.int32)
            known_labels = labels.ravel()[pixels]
            rows.append(dict(sequence_id=key[0], frame=key[1], building_id=row['building_id'], shape=list(labels.shape),
                             gt_artifact=row['gt_artifact'],
                             pixels=pixels, labels=known_labels, scores=scores.ravel()[pixels].copy(),
                             native=native.ravel()[pixels].copy() if native is not None else None,
                             unit_id=record['unit_id'], artifact_sha256=record['artifact_sha256'],
                             map_path=str(path),
                             support_sha256=hashlib.sha256(pixels.tobytes() + known_labels.tobytes()).hexdigest()))
        self.cache[group] = rows
        while len(self.cache) > 3:
            self.cache.popitem(last=False)
        return rows


def _group_name(group):
    return ':'.join(group)


def _ranking(group, rows):
    records, curves = [], []
    by_building = defaultdict(list)
    grid = np.linspace(0, 1, 101)
    for row in rows:
        curve = _precision_recall(row['scores'], row['labels'])
        ap = average_precision(row['scores'], row['labels'])
        prevalence = float(np.mean(row['labels'] == 1)) if len(row['labels']) else None
        records.append(dict(level='frame', group=_group_name(group), mode=group[2], building_id=row['building_id'],
                            sequence_id=row['sequence_id'], frame=row['frame'], average_precision=ap,
                            constant_score_ap=prevalence, valid_pixels=len(row['labels']), positive_pixels=int((row['labels'] == 1).sum())))
        if curve is not None:
            recall, precision = curve
            envelope = np.maximum.accumulate(precision[::-1])[::-1]
            by_building[row['building_id']].append((ap, prevalence, envelope[np.searchsorted(recall, grid, side='left')]))
    for building, members in sorted(by_building.items()):
        records.append(dict(level='facade', group=_group_name(group), mode=group[2], building_id=building,
                            positive_pairs=len(members), average_precision=float(np.mean([item[0] for item in members])),
                            constant_score_ap=float(np.mean([item[1] for item in members]))))
        precision = np.mean([item[2] for item in members], axis=0)
        curves.extend(dict(group=_group_name(group), mode=group[2], building_id=building, recall=float(recall), precision=float(value),
                           diagnostic='mean per-frame precision envelope on held-out labelled observations; no threshold selection')
                      for recall, value in zip(grid, precision))
    facades = [row for row in records if row['level'] == 'facade']
    records.append(dict(level='frame_then_facade_macro', group=_group_name(group), mode=group[2], facades=len(facades),
                        average_precision=float(np.mean([row['average_precision'] for row in facades])) if facades else None,
                        constant_score_ap=float(np.mean([row['constant_score_ap'] for row in facades])) if facades else None,
                        positive_pairs=sum(row['positive_pairs'] for row in facades)))
    return records, curves


def run(metrics_run, out_path, prediction_hook=None):
    root, out = (Path(value).expanduser().resolve() for value in (metrics_run, out_path))
    if out.exists() or out == root or root in out.parents or out in root.parents:
        raise FileExistsError('Нужен новый отдельный каталог результатов калибровки')
    source = Samples(root)
    for value in (source.provenance['source_run'], source.provenance.get('snapshot', str(root))):
        protected = Path(value).resolve()
        if out == protected or protected in out.parents or out in protected.parents:
            raise FileExistsError('Нельзя писать внутрь исходного запуска или снимка')
    out.mkdir(parents=True)
    metrics.write_json(out / 'run.json', dict(status='running', metrics_run=str(root), no_model_inference=True))
    groups = sorted(source.entries)
    scalars = [group for group in groups if group[0] not in metrics.NATIVE]
    native = [group for group in groups if group[0] in metrics.NATIVE]
    families = [dict(name='standalone/' + _group_name(group), kind='standalone', groups=[group], mode=group[2]) for group in groups]
    for first in native:
        for second in scalars:
            if second[2] == first[2] or first[2] == 'previous' and second[2] == 'history':
                families.append(dict(name='fusion/' + _group_name(first) + '+' + _group_name(second),
                                     kind='fusion', groups=[first, second], mode=first[2]))
    for first in scalars:
        second = (first[0], first[1], 'history')
        if first[2] == 'previous' and second in groups:
            families.append(dict(name='matched_history/' + first[0] + ':' + first[1],
                                 kind='matched_history', groups=[first, second], mode='history'))
    per_pair, thresholds, audit, ranking, pr_curves, coverage, recipes = [], [], [], [], [], [], []
    fit_cache = {}
    for group in scalars:
        values, curves = _ranking(group, source.load(group))
        ranking.extend(values)
        pr_curves.extend(curves)

    def fit(rows, group, heldout, family):
        training = [row for row in rows if row['building_id'] != heldout]
        signature = (group, heldout, tuple((row['unit_id'], row['artifact_sha256'], row['support_sha256']) for row in training))
        if signature not in fit_cache:
            profile = _profile(training, heldout)
            fit_cache[signature] = {goal: _choice(profile, heldout, goal) for goal in GOALS}
        values = {goal: dict(item, family=family['name'], family_kind=family['kind'], group=_group_name(group),
                             training_scope='common pair keys and pixels of this family') for goal, item in fit_cache[signature].items()}
        thresholds.extend(values.values())
        audit.extend(values.values())
        return values

    def component(row, choice=None):
        return dict(unit_id=row['unit_id'], map_path=row['map_path'], map_sha256=row['artifact_sha256'],
                    native=choice is None, threshold=None if choice is None else choice['threshold'])

    def emit(family, heldout, goal, predictor, row, prediction, choice=None, parts=None, operation='single', support_rows=None):
        counts = metrics.evaluate_mask(np.asarray(prediction, dtype=bool), row['labels'])
        counts['ignored_pixels'] = int(np.prod(row['shape'])) - len(row['labels'])
        name = predictor + (f'@fpr{goal:g}' if goal is not None else '')
        parts = [component(row, choice)] if parts is None else parts
        recipe_id = f'prediction-{len(recipes):07d}'
        record = dict(cohort=family['name'], family=family['name'], family_kind=family['kind'], mode=family['mode'],
                      predictor=name, component=predictor, goal=goal, heldout_building=heldout,
                      building_id=row['building_id'], sequence_id=row['sequence_id'], frame=row['frame'],
                      support_sha256=row['support_sha256'], source_unit_id=parts[0]['unit_id'] if len(parts) == 1 else None,
                      source_unit_ids=[part['unit_id'] for part in parts], recipe_id=recipe_id,
                      threshold=None if choice is None else choice['threshold'],
                      status='evaluated' if counts['valid_pixels'] else 'no_annotation_support', **counts)
        per_pair.append(record)
        recipes.append(dict(recipe_id=recipe_id, family=family['name'], family_kind=family['kind'], mode=family['mode'],
                            predictor=name, goal=goal, heldout_building=heldout, building_id=row['building_id'],
                            sequence_id=row['sequence_id'], frame=row['frame'], gt_artifact=row['gt_artifact'],
                            operation=operation, components=parts, known_support_sha256=row['support_sha256'],
                            support_components=[{key: value for key, value in component(item).items() if key not in {'native', 'threshold'}}
                                                for item in (support_rows or [row])]))
        if prediction_hook is not None:
            prediction_hook(family=family['name'], mode=family['mode'], predictor=name, goal=goal,
                            heldout_building=heldout, sample=row, prediction=np.asarray(prediction, dtype=bool))

    for number, family in enumerate(families, 1):
        first = source.load(family['groups'][0])
        if len(family['groups']) == 2:
            first, second = match_samples(first, source.load(family['groups'][1]))
        else:
            second = None
        by_key = {(row['sequence_id'], row['frame']): row for row in second} if second is not None else {}
        buildings = sorted({row['building_id'] for row in first if len(row['labels'])})
        print(f"Сравнение {number}/{len(families)}: {family['name']}; пар {len(first)}, фасадов с GT {len(buildings)}.", flush=True)
        coverage.append(dict(family=family['name'], family_kind=family['kind'], mode=family['mode'],
                             input_pairs=[len(source.entries[group]) for group in family['groups']], matched_pairs=len(first),
                             annotated_pairs=sum(bool(len(row['labels'])) for row in first), buildings=buildings,
                             missing_policy='No missing map is replaced with a negative prediction'))
        for heldout in buildings:
            print(f'  Оцениваем фасад {heldout}; пороги по другим фасадам этого сравнения.', flush=True)
            held_rows = [row for row in first if row['building_id'] == heldout]
            if family['kind'] == 'standalone' and family['groups'][0][0] in metrics.NATIVE:
                choices = {}
            else:
                choices = fit(second if family['kind'] == 'fusion' else first,
                              family['groups'][1] if family['kind'] == 'fusion' else family['groups'][0], heldout, family)
            history_choices = fit(second, family['groups'][1], heldout, family) if family['kind'] == 'matched_history' else None
            goals = GOALS if choices else (None,)
            for goal in goals:
                chosen = choices.get(goal)
                other_choice = history_choices.get(goal) if history_choices else None
                supported = chosen is None or chosen['status'] == 'calibrated'
                supported &= other_choice is None or other_choice['status'] == 'calibrated'
                if not supported:
                    audit.append(dict(family=family['name'], heldout_building=heldout, goal=goal, status='skipped_evaluation',
                                      reason='No calibration negatives in other facades; this family has no paired numeric comparison'))
                    continue
                for row in held_rows:
                    key = row['sequence_id'], row['frame']
                    other = by_key.get(key)
                    support_rows = [row] if other is None else [row, other]
                    for name, mask in [('always_positive', np.ones(len(row['labels']), bool)), ('always_negative', np.zeros(len(row['labels']), bool))]:
                        emit(family, heldout, goal, name, row, mask, parts=[], operation=name, support_rows=support_rows)
                    if family['kind'] == 'standalone':
                        prediction = row['native'] if goal is None else row['scores'] > np.float32(chosen['threshold'])
                        emit(family, heldout, goal, _group_name(family['groups'][0]), row, prediction, chosen)
                    elif family['kind'] == 'fusion':
                        base = row['native']
                        gate = other['scores'] > np.float32(chosen['threshold'])
                        native_name, scalar_name = map(_group_name, family['groups'])
                        emit(family, heldout, goal, native_name, row, base, support_rows=support_rows)
                        emit(family, heldout, goal, scalar_name, other, gate, chosen, support_rows=support_rows)
                        for symbol, operation, mask in [('&', 'and', base & gate), ('|', 'or', base | gate)]:
                            emit(family, heldout, goal, native_name + symbol + scalar_name, row, mask, chosen,
                                 parts=[component(row), component(other, chosen)], operation=operation, support_rows=support_rows)
                    else:
                        emit(family, heldout, goal, _group_name(family['groups'][0]), row,
                             row['scores'] > np.float32(chosen['threshold']), chosen, support_rows=support_rows)
                        emit(family, heldout, goal, _group_name(family['groups'][1]), other,
                             other['scores'] > np.float32(other_choice['threshold']), other_choice, support_rows=support_rows)
    per_sequence, per_facade, aggregate = metrics.aggregate_metrics(per_pair)
    primary = {(row['cohort'], row['predictor'], row['subset']): row for row in aggregate if row['aggregation'] == 'frame_then_facade_macro'}
    comparisons = []
    for family in families:
        if family['kind'] == 'standalone':
            continue
        for goal in GOALS:
            suffix = f'@fpr{goal:g}'
            base_name = _group_name(family['groups'][0]) + suffix
            alternatives = [_group_name(family['groups'][1]) + suffix]
            if family['kind'] == 'fusion':
                alternatives = [_group_name(family['groups'][1]) + suffix] + [
                    _group_name(family['groups'][0]) + operation + _group_name(family['groups'][1]) + suffix for operation in ('&', '|')]
            for alternative in alternatives:
                for subset in ('positive', 'all', 'no_event'):
                    before = primary.get((family['name'], base_name, subset))
                    after = primary.get((family['name'], alternative, subset))
                    if before is None or after is None:
                        continue
                    comparisons.append(dict(family=family['name'], mode=family['mode'], family_kind=family['kind'], goal=goal,
                                            baseline=base_name, comparison=alternative, subset=subset,
                                            evaluated_pairs=before['evaluated_pairs'], facades=before['units'],
                                            **{f'baseline_{key}': before[key] for key in metrics.METRICS},
                                            **{f'comparison_{key}': after[key] for key in metrics.METRICS},
                                            **{f'delta_{key}': after[key] - before[key] if before[key] is not None and after[key] is not None else None for key in metrics.METRICS}))
    return finish(source, out, per_pair, per_sequence, per_facade, aggregate, thresholds, audit, ranking, pr_curves, coverage, comparisons, recipes)


def finish(source, out, per_pair, per_sequence, per_facade, aggregate, thresholds, audit, ranking, pr_curves, coverage, comparisons, recipes):
    for name, rows in [('per_pair', per_pair), ('per_sequence', per_sequence), ('per_facade', per_facade), ('aggregate', aggregate),
                       ('calibration_audit', audit), ('ranking', ranking), ('pr_curves', pr_curves), ('coverage', coverage), ('comparisons', comparisons)]:
        metrics.write_csv(out / (name + '.csv'), rows)
    metrics.write_json(out / 'thresholds.json', thresholds)
    metrics.write_json(out / 'prediction_recipes.json', recipes)
    notes = ['Разведочная калибровка порогов на real VAL: leave-one-building-out (LOFO). Не независимый TEST.',
             'Для оцениваемого фасада порог выбирается только по другим фасадам; равный вес фасадов, затем кадров с известными отрицательными GT-пикселями.',
             'Порог scalar-ветви: score > threshold, цели FPR 1% и 5%; одинаковые оценки не разделяются. Это не ограничение FPR сочетания OR.',
             'Каждое сочетание имеет собственную общую выборку пар/пикселей и повторно вычисленные native/scalar-базы; редкий GeoSCD не сокращает выборку AnyChange.',
             'PREVIOUS/HISTORY: и калибровка, и оценка используют одинаковые пары/пиксели двух режимов. При отсутствии отрицательных обучающих пикселей fold пропускается явно.',
             'AnyChange/GeoSCD: исходные бинарные маски неизменны. Модели, COCO-растеризация, RGB и память истории не пересчитываются.',
             'Память HISTORY сохраняет прежний synthetic-VAL порог обновления: изменён только итоговый порог детекции.',
             'Ранее выбор checkpoint мог использовать VAL. LOFO относится только к порогу, а не ко всему обучению/выбору модели.',
             'GT и неизвестные 255 сохраняются. Нет дополнительных отрицательных пикселей из неразмеченного фона.',
             'Основные P/R/F1/IoU: положительные GT-кадры → среднее внутри фасада → среднее фасадов. FPR: все известные отрицательные пиксели с тем же усреднением.',
             'Контроли always_positive/always_negative рассчитаны для каждой сравниваемой поддержки. AP точный по блокам равных оценок, без GT+ не определён.',
             'PR-кривые — только диагностика: 101 точка среднего precision envelope положительных кадров фасада; порог по ним не выбирается.',
             'Проценты в таблицах; CSV сохраняют доли. Разности — процентные пункты. Покрытие и пропуски: coverage.csv/calibration_audit.csv.']
    primary = {(row['cohort'], row['predictor'], row['subset']): row for row in aggregate if row['aggregation'] == 'frame_then_facade_macro'}
    percent = lambda value: '—' if value is None else f'{100 * value:.2f}'
    escaped = lambda value: str(value).replace('|', '\\|')
    lines = notes.copy()
    for mode in metrics.MODES:
        lines += ['', f'## {mode.upper()}: отдельные методы и точные контроли', '',
                  '| Метод / порог | P | R | F1 | IoU | FPR все | Пар +/все | Фасадов |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in aggregate:
            if row['mode'] != mode or row['aggregation'] != 'frame_then_facade_macro' or row['subset'] != 'all' or not row['cohort'].startswith('standalone/'):
                continue
            positive = primary.get((row['cohort'], row['predictor'], 'positive'), {})
            label = row['cohort'].split('/', 1)[1] + ' / ' + row['predictor']
            cells = [escaped(label), *[percent(positive.get(key)) for key in ('precision', 'recall', 'f1', 'iou')],
                     percent(row['fpr']), f"{positive.get('evaluated_pairs', 0)}/{row['evaluated_pairs']}", str(row['units'])]
            lines.append('| ' + ' | '.join(cells) + ' |')
        lines += ['', f'### {mode.upper()}: добавка к AnyChange/GeoSCD на одинаковых парах и пикселях', '',
                  '| Сочетание | База F1 → сочетание | ΔF1 | P | R | IoU | FPR базы → сочетания | Пар +/все |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in comparisons:
            if row['mode'] != mode or row['family_kind'] != 'fusion' or row['subset'] != 'positive' or not any(symbol in row['comparison'] for symbol in '&|'):
                continue
            all_frames = next(item for item in comparisons if item['family'] == row['family'] and item['goal'] == row['goal'] and item['comparison'] == row['comparison'] and item['subset'] == 'all')
            cells = [escaped(row['comparison']), percent(row['baseline_f1']) + '→' + percent(row['comparison_f1']), percent(row['delta_f1']),
                     *[percent(row['comparison_' + key]) for key in ('precision', 'recall', 'iou')],
                     percent(all_frames['baseline_fpr']) + '→' + percent(all_frames['comparison_fpr']),
                     f"{row['evaluated_pairs']}/{all_frames['evaluated_pairs']}"]
            lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '## PREVIOUS → HISTORY: общие обучающие и оцениваемые пары/пиксели', '',
              '| Метод / цель | F1 до→после | ΔF1 | R до→после | FPR до→после | Пар +/все |', '|---|---:|---:|---:|---:|---:|']
    for row in comparisons:
        if row['family_kind'] != 'matched_history' or row['subset'] != 'positive':
            continue
        all_frames = next(item for item in comparisons if item['family'] == row['family'] and item['goal'] == row['goal'] and item['subset'] == 'all')
        cells = [escaped(row['comparison']), percent(row['baseline_f1']) + '→' + percent(row['comparison_f1']), percent(row['delta_f1']),
                 percent(row['baseline_recall']) + '→' + percent(row['comparison_recall']),
                 percent(all_frames['baseline_fpr']) + '→' + percent(all_frames['comparison_fpr']),
                 f"{row['evaluated_pairs']}/{all_frames['evaluated_pairs']}"]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '## AP без выбора порога', '', '| Метод | AP | AP постоянного score (доля GT+) | Положительных пар | Фасадов |', '|---|---:|---:|---:|---:|']
    for row in ranking:
        if row['level'] == 'frame_then_facade_macro':
            lines.append('| ' + ' | '.join([escaped(row['group']), percent(row['average_precision']), percent(row['constant_score_ap']), str(row['positive_pairs']), str(row['facades'])]) + ' |')
    unsupported = [row for row in audit if row['status'] in {'unsupported', 'skipped_evaluation'}]
    lines += ['', f'Неподдержанных записей/fold-проверок: {len(unsupported)}; детали в calibration_audit.csv.']
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (out / 'tables.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    summary = dict(status='completed', scope='exploratory real VAL leave-one-building-out threshold calibration',
                   metrics_run=str(source.root), notes=notes, groups=[list(group) for group in source.entries],
                   per_pair=per_pair, aggregate=aggregate, thresholds=thresholds, calibration_audit=audit,
                   ranking=ranking, coverage=coverage, comparisons=comparisons)
    metrics.write_json(out / 'summary.json', summary)
    metrics.write_json(out / 'provenance.json', dict(metrics_run=str(source.root), input_sha256=source.hashes,
                       script_sha256=metrics.sha256(Path(__file__)), metrics_helper_sha256=metrics.sha256(Path(metrics.__file__)),
                       target_definition=source.provenance.get('target_definition'), categories=source.provenance.get('categories'),
                       boundary_tolerance=source.provenance.get('boundary_tolerance'), goals=list(GOALS),
                       no_model_inference=True, unchanged_history_memory_update_policy=True,
                       checkpoint_selection_not_cross_validated=True, threshold_operator='>',
                       prediction_recipes_sha256=metrics.sha256(out / 'prediction_recipes.json'),
                       curve_max_points_per_facade_group=101))
    metrics.write_json(out / 'run.json', dict(status='completed', metrics_run=str(source.root), no_model_inference=True))
    print(f'Калибровка завершена: {out / "summary.txt"}.', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics-run', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if __package__:
        from .real_gt_slices import SliceCollector
    else:
        from real_gt_slices import SliceCollector
    source = metrics.read_json(Path(args.metrics_run).expanduser() / 'provenance.json')
    collector = SliceCollector(args.metrics_run, source['categories'])
    run(args.metrics_run, args.out, prediction_hook=collector.record)
    collector.finalize(args.out)


if __name__ == '__main__':
    main()
