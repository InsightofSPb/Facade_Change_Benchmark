"""Final cache-only diagnosis of a completed directed-delta run; no fit or calibration."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import html
import importlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
from PIL import Image

continuous = importlib.import_module('scripts.real_eval.2026-10-19_continuous_semantic_updates')
pilot = continuous.pilot
SCHEMA = '2026-10-20-semantic-error-audit-v1'
SELECTED = (('marata_44', 2016, 2019), ('marata_44', 2019, 2022),
            ('manezhnyi_7', 2021, 2022), ('Voskova_10', 2019, 2021))
VARIANTS = tuple(v for v in continuous.VARIANTS if v != 'lpips_only')
PARTS = ('wrong_sign', 'zero_delta', 'right_sign_below_threshold', 'detected')
RATES = tuple(x + '_fraction' for x in PARTS) + ('sign_ceiling', 'recall', 'fpr')
COUNTS = ('gt_events', *PARTS, 'fp', 'negative_events')
KEYS = ('split', 'variant', 'class_name', 'direction')
IDENTITY = ('split', 'building_id', 'sequence_id', 'frame', 'fold',
            'variant', 'target', 'class_name', 'direction')
NOTES = (
    'Последняя диагностика: модели и пороги взяты из завершённого directed_delta; '
    'обучение, инференс изображений и подбор порогов не выполняются.',
    'GT означает изменение состава аннотаций. Нулевой битсет — неизвестно, а не здоровый фасад. '
    'Исчезновение класса не доказывает ремонт.',
    'wrong_sign: направленная разность < 0; zero_delta: ровно 0; '
    'right_sign_below_threshold: разность > 0, но score <= прежнего порога; detected: правильное событие найдено.',
    'sign_ceiling — доля GT-событий с правильным знаком. Это ограничение полноты текущего правила выбора '
    'направления, а не достижимая рабочая точка с приемлемым FPR и не потолок других архитектур.',
    'Доли и Recall: сначала среднее положительных пар фасада, затем равный вес фасадов с GT. '
    'Нулевой GT даёт N/A. FPR усредняется отдельно по парам с отрицательными событиями. '
    'Суммы пикселей диагностические; направленные события внутри пары усреднены micro.',
    'TRAIN fusion-OOF исключает фасад только из подгонки объединения и порога. '
    'DINO/MSDZip могли видеть его при обучении. Четыре временных TRAIN-фасада не описывают весь TRAIN DINO.',
    'Четыре иллюстрации выбраны после просмотра ошибок. VAL/TEST уже исследовались; '
    'это объяснение ошибок, не независимая проверка качества.',
    'Разложение не устанавливает причину ошибки: освещение, геометрия, семантика и разметка требуют '
    'содержательной проверки. Правильный знак при малой разности ещё не означает полезный сигнал.',
)


def identity(row):
    return tuple(str(row[k]) for k in IDENTITY)


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def decompose(truth, directed_delta, prediction, known):
    """Partition true class/direction events, never credit a hit of another class."""
    valid = np.broadcast_to(known, truth.shape)
    positive = truth & valid
    if np.any(prediction & (directed_delta <= 0) & valid):
        raise ValueError('Prediction violates the frozen direction gate')
    masks = (positive & (directed_delta < 0), positive & (directed_delta == 0),
             positive & (directed_delta > 0) & ~prediction, positive & prediction)
    counts = dict(zip(PARTS, (int(m.sum()) for m in masks)))
    n = int(positive.sum())
    if sum(counts.values()) != n:
        raise ValueError('GT event partition is incomplete')
    negative = valid & ~truth
    fp, negatives = int((prediction & negative).sum()), int(negative.sum())
    return dict(gt_events=n, **counts, fp=fp, negative_events=negatives,
                **{k + '_fraction': v / n if n else None for k, v in counts.items()},
                sign_ceiling=(counts['right_sign_below_threshold'] + counts['detected']) / n if n else None,
                recall=counts['detected'] / n if n else None, fpr=fp / negatives if negatives else None)


def aggregate(rows):
    """Equal pairs within a facade, then equal facades; never divide pooled counts."""
    facades, summaries = [], []
    for key, members in sorted(pilot.groups_from(rows, (*KEYS, 'building_id')).items()):
        result = dict(zip((*KEYS, 'building_id'), key), pairs=len(members),
                      positive_pairs=sum(r['gt_events'] > 0 for r in members),
                      **{k: sum(r[k] for r in members) for k in COUNTS})
        for name in RATES:
            values = [r[name] for r in members if r[name] is not None]
            result[name] = float(np.mean(values)) if values else None
            result[name + '_pairs'] = len(values)
        facades.append(result)
    for key, members in sorted(pilot.groups_from(facades, KEYS).items()):
        result = dict(zip(KEYS, key), facades=len(members), pairs=sum(r['pairs'] for r in members),
                      positive_pairs=sum(r['positive_pairs'] for r in members),
                      **{k: sum(r[k] for r in members) for k in COUNTS})
        for name in RATES:
            values = [r[name] for r in members if r[name] is not None]
            result[name] = float(np.mean(values)) if values else None
            result[name + '_facades'] = len(values)
        summaries.append(result)
    return summaries, facades


def gt_transitions(pair, names):
    known = pair['labels'] != 255
    old, new = (pair[k]['packed'][known] for k in ('reference', 'source'))
    codes, counts = np.unique(np.column_stack((old, new)), axis=0, return_counts=True)
    label = lambda bits: '|'.join(name for c, name in enumerate(names) if bits & (1 << c))
    return [dict(before_bits=int(a), after_bits=int(b), before_classes=label(a), after_classes=label(b),
                 changed=bool(a != b), pixels=int(n)) for (a, b), n in zip(codes, counts)]


def measurements(pair, events, scores, threshold, categories):
    """Replay existing count metrics without recomputing expensive AP ranks."""
    prediction = (scores > threshold) & pair['support']
    measured = [dict(target='binary_change', class_name='__all__', direction='change',
                     **continuous.metrics.evaluate_mask(prediction, pair['labels']))]
    predicted = None
    if events is not None:
        predicted = (events > threshold) & pair['support'][None, None]
        if not np.array_equal(predicted.any(axis=(0, 1)), prediction):
            raise ValueError('Binary union mismatch')
        labels = np.where((pair['labels'] != 255)[None, None], continuous.event_truth(pair), 255).astype(np.uint8)
        measured.append(dict(target='directed_events', class_name='__all__', direction='gain_loss',
                             **continuous.metrics.evaluate_mask(predicted, labels)))
        for d, direction in enumerate(('gain', 'loss')):
            for c, category in enumerate(categories):
                measured.append(dict(target='directed_class', class_name=category['name'], direction=direction,
                                     **continuous.metrics.evaluate_mask(predicted[d, c], labels[d, c])))
    return measured, prediction, predicted


def load_frozen(root, source, pairs, config, note):
    if config['categories'] != source.categories or config['category_to_channel'] != source.manifest['category_to_channel']:
        raise ValueError('Frozen ontology/channel mapping changed')
    subset = {s: dict(facades=sorted({p['context']['building_id'] for p in pairs if p['context']['split'] == s}),
                      pairs=sum(p['context']['split'] == s for p in pairs)) for s in ('train', 'val', 'test')}
    if config['temporal_subset'] != subset or config['semantic_model'] != source.model or config['method_contracts'] != source.contract['models']:
        raise ValueError('Frozen data subset or model identity changed')
    index = pilot.read_json(note(root / '2026-10-19_artifacts.json'))
    buildings = subset['train']['facades']
    expected = {'all_train': None, **{'holdout-' + pilot.digest(b)[:12]: b for b in buildings}}
    bundles = {}
    for item in index['models']:
        fold = item['fold']
        path = note(pilot.safe_path(root, item['path']))
        if pilot.sha256(path) != item['sha256']:
            raise ValueError('Frozen model checksum mismatch')
        bundle = pilot.read_json(path)
        if fold in bundles or fold not in expected or bundle['excluded_building'] != expected[fold]:
            raise ValueError('Unexpected/duplicate frozen fold')
        training = sorted(set(buildings) - {expected[fold]})
        if bundle['feature_design'] != 'directed_delta' or bundle['training_buildings'] != training:
            raise ValueError('Frozen fusion training exposure mismatch')
        for name in VARIANTS:
            if name != 'semantic_delta':
                model = bundle['models'][name]
                if (model['feature_design'] != 'directed_delta' or model['coefficients'][0] <= 0
                        or model['features'] != list(continuous.DELTA_FEATURES[:continuous.DELTA_FEATURE_COUNTS[name]])):
                    raise ValueError('Frozen feature/slope contract mismatch')
        for name in continuous.VARIANTS:
            choice = bundle['thresholds'][name]
            if (choice['train_buildings'] != training or choice['fitting_excluded_building'] != expected[fold]
                    or choice['status'] != 'calibrated' or choice['threshold'] < 0 or np.isnan(choice['threshold'])):
                raise ValueError('Frozen threshold contract mismatch')
        bundles[fold] = bundle
    if set(bundles) != set(expected):
        raise ValueError('Incomplete frozen models')
    return bundles, index


def heatmap(values, support, signed=False):
    """Fixed scales: probability black->yellow [0,1], delta blue->white->red [-1,1]."""
    x = np.clip(values, -1 if signed else 0, 1)
    if signed:
        image = np.stack((1 - np.maximum(-x, 0), 1 - np.abs(x), 1 - np.maximum(x, 0)), axis=-1)
    else:
        image = np.stack((x, x, np.zeros_like(x)), axis=-1)
    image = (image * 255).astype(np.uint8)
    image[~support] = [85, 85, 85]
    return image


def mask_panel(mask, support, known=None):
    image = np.zeros((*mask.shape, 3), np.uint8)
    image[mask & support] = [255, 190, 20]
    if known is not None:
        image[support & ~known] = [125, 125, 125]
    image[~support] = [45, 45, 45]
    return image


def error_panel(rgb, truth, delta, predicted, known):
    image = (rgb.astype(np.float32) * .3).astype(np.uint8)
    for mask, color in ((truth & (delta < 0), [210, 50, 220]), (truth & (delta == 0), [40, 170, 240]),
                        (truth & (delta > 0) & ~predicted, [255, 165, 0]),
                        (truth & predicted, [30, 220, 70]), (~truth & predicted, [240, 30, 30])):
        image[mask & known] = color
    image[~known] = [85, 85, 85]
    return image


def gallery(out, pair, categories, predictions, context, transitions):
    stem = pilot.digest(context)[:20]
    names = [c['name'] for c in categories]
    palette = pilot.semantic_report._palette(categories)
    ref, src = pair['reference'], pair['source']
    geometry, known = ref['support'] & src['support'], pair['labels'] != 255
    truth = continuous.event_truth(pair)
    delta = src['probabilities'] - ref['probabilities']
    predicted = predictions['semantic_lpips']
    cards = []

    def panel(label, rgb):
        relative = f'gallery/{stem}-{len(cards):03}.png'
        path = out / relative
        path.parent.mkdir(exist_ok=True)
        # Preserve native grid; the browser scales only the display. Open a card for native pixels.
        Image.fromarray(rgb).save(path)
        cards.append(f'<figure><a href="{relative}"><img loading="lazy" src="{relative}"></a>'
                     f'<figcaption>{html.escape(label)}</figcaption></figure>')

    panel('RGB раньше', ref['rgb'])
    panel('RGB позже', src['rgb'])
    for date, frame in (('раньше', ref), ('позже', src)):
        for field, label in (('packed', 'GT'), ('semantic', 'DINO ≥ 0.5')):
            masks = pilot.unpack(frame[field], len(names)) & frame['support']
            rgb = pilot.semantic_report._overlay(frame['rgb'], masks, names, palette)
            rgb[~frame['support']] = 45
            if field == 'packed':
                rgb[frame['support'] & (frame['packed'] == 0)] = 125
            panel(label + ' ' + date + ': поддержка этой даты', rgb)
    panel('Общая геометрическая поддержка', mask_panel(geometry, np.ones_like(geometry)))
    panel('Поддержка геометрии + LPIPS + MSDZip', mask_panel(pair['support'], geometry))
    panel('Область оценки: известный GT обеих дат', mask_panel(known, pair['support']))
    panel('Бинарный GT изменения аннотаций', mask_panel(pair['labels'] == 1, pair['support'], known))
    main = ''.join(cards)
    details = []
    for c, name in enumerate(names):
        start = len(cards)
        for date, frame in (('раньше', ref), ('позже', src)):
            channel = pilot.unpack(frame['packed'], len(names))[c]
            panel('GT ' + date, mask_panel(channel, frame['support'], frame['packed'] != 0))
            panel('DINO ≥ 0.5 ' + date,
                  mask_panel(pilot.unpack(frame['semantic'], len(names))[c], frame['support']))
            panel('p ' + date + ': чёрный 0 → жёлтый 1', heatmap(frame['probabilities'][c], frame['support']))
        panel('Δp: синий −1 → белый 0 → красный +1', heatmap(delta[c], geometry, True))
        for d, direction in enumerate(('gain', 'loss')):
            panel(direction + ': ошибки semantic_lpips', error_panel(src['rgb'], truth[d, c],
                  delta[c] if d == 0 else -delta[c], predicted[d, c], known))
        details.append('<details><summary>' + html.escape(name) + '</summary><div class="grid">'
                       + ''.join(cards[start:]) + '</div></details>')
    arrays = dict(labels=pair['labels'], common_support=pair['support'], geometry_support=geometry)
    for prefix, frame in (('before', ref), ('after', src)):
        for key in ('rgb', 'packed', 'support', 'semantic', 'probabilities'):
            arrays[prefix + '__' + key] = frame[key]
    for name, values in pair['maps'].items():
        arrays[name + '__scores'], arrays[name + '__support'] = values['scores'], values['support']
    for name, masks in predictions.items():
        arrays[name + '__gain'], arrays[name + '__loss'] = pilot.pack(masks[0]), pilot.pack(masks[1])
    relative = 'selected/' + stem + '.npz'
    checksum = pilot.atomic_npz(out / relative, arrays)
    metadata = dict(context, artifact_path=relative, sha256=checksum,
                    channel_names=names, selection='fixed posthoc failure review',
                    probabilities='float32 independent sigmoid; not physical-event probabilities')
    title = f"{context['building_id']}: {context['reference_year']} → {context['source_year']}"
    columns = ('before_classes', 'after_classes', 'changed', 'pixels')
    article = ('<article><h2>' + html.escape(title) + '</h2><div class="grid">' + main + '</div>'
               + pilot.semantic_report._table(transitions, columns) + ''.join(details) + '</article>')
    return article, metadata


def report(out, summary, articles, categories):
    primary = [r for r in summary if r['class_name'] == '__all__']
    columns = ('split', 'variant', 'wrong_sign_fraction', 'zero_delta_fraction',
               'right_sign_below_threshold_fraction', 'recall', 'sign_ceiling', 'fpr',
               'sign_ceiling_facades', 'positive_pairs')
    rows = [{k: f'{r[k]:.4f}' if isinstance(r[k], float) else r[k] for k in columns} for r in primary]
    palette = pilot.semantic_report._palette(categories)
    legend = ' '.join('<span style="display:inline-block;margin:5px"><i style="display:inline-block;width:14px;'
                      'height:14px;background:rgb(' + ','.join(str(int(x)) for x in palette[c['name']])
                      + ')"></i> ' + html.escape(c['name']) + '</span>' for c in categories)
    content = ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Последняя диагностика семантики</title>'
               '<style>body{font:15px sans-serif;margin:24px}table{border-collapse:collapse;font-size:12px}'
               'td,th{border:1px solid #ccc;padding:5px}.grid{display:flex;flex-wrap:wrap}'
               'figure{width:290px;margin:8px}img{width:100%}figcaption{font-size:13px}'
               'summary{cursor:pointer;padding:10px;background:#eee;margin-top:6px}</style>'
               '<h1>Замороженные модели: направление и порог</h1><ul>'
               + ''.join('<li>' + html.escape(n) + '</li>' for n in NOTES) + '</ul>'
               '<p>Доли в таблице: 0–1. Sign ceiling включает найденные события и пропуски с правильным знаком. '
               'Для FPR знаменатель — отрицательные class/direction-события, а не бинарные пиксели изменения.</p>'
               + '<table><tr>' + ''.join('<th>' + html.escape(k) + '</th>' for k in columns) + '</tr>'
               + ''.join('<tr>' + ''.join('<td>' + html.escape(str(r[k]) if r[k] is not None else 'N/A')
                         + '</td>' for k in columns) + '</tr>' for r in rows) + '</table>'
               + '<h2>Как читать изображения</h2><p>GT и DINO показаны в поддержке каждой даты, без обрезки по GT другой даты. '
               'В GT серый — неизвестно, тёмно-серый — вне поддержки. В поклассовых масках чёрный — '
               'отсутствие этого класса среди известных аннотаций, жёлтый — наличие. Наложения смешивают цвета '
               'пересечений; поклассовые панели и NPZ сохраняют все метки. Вероятности показаны на фиксированной шкале 0–1.</p>'
               '<p>Карты ошибок semantic_lpips: пурпурный — неверный знак, голубой — нулевая разность, '
               'оранжевый — правильный знак ниже порога, зелёный — TP, красный — FP, серый — вне оценки. '
               'Переходы GT в таблицах посчитаны по точным битсетам в общей известной области. '
               'Нажатие на изображение открывает исходную решётку; разделы классов раскрываются.</p>'
               + '<p>Цвета общих семантических наложений: ' + legend + '</p>'
               + ''.join(articles) + '</html>')
    (out / 'report.html').write_text(content, encoding='utf-8')


def run(evaluation_run, out_path, selected=SELECTED):
    root, out = Path(evaluation_run).expanduser().resolve(), Path(out_path).expanduser().resolve()
    upstream = pilot.read_json(root / 'run.json')
    semantic = Path(upstream['semantic_run']).expanduser().resolve()
    change = Path(upstream['change_run']).expanduser().resolve()
    semantic_state = pilot.read_json(semantic / 'run.json')
    protected = (root, semantic, change, Path(semantic_state['config']['prepared']).expanduser().resolve(),
                 Path(semantic_state['cache_root']).expanduser().resolve())
    if any(out == p or out in p.parents or p in out.parents for p in protected):
        raise ValueError('Output must be separate from all input/cache directories')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(kind='semantic_error_audit', schema=SCHEMA, status='running', evaluation_run=str(root),
                 started_utc=datetime.now(timezone.utc).isoformat(), limitations=list(NOTES))
    pilot.atomic_json(out / 'run.json', state)
    evidence, source = {}, None

    def note(path):
        evidence[str(path)] = pilot.sha256(path)
        return path

    try:
        note(root / 'run.json')
        if (upstream['status'] != 'completed' or upstream['schema'] != continuous.DELTA_SCHEMA
                or not upstream.get('parity_passed')):
            raise ValueError('Requires a completed directed_delta run with passed parity')
        frozen = pilot.read_json(note(root / 'resolved_config.json'))
        repo = Path(__file__).resolve().parents[2]
        expected_code = (continuous, pilot, continuous.calibration, continuous.metrics, pilot.semantic_geometry,
                         pilot.night_common, pilot.night_report, pilot.semantic_report)
        code = {str(Path(m.__file__).resolve().relative_to(repo)): pilot.sha256(m.__file__) for m in expected_code}
        if frozen['code_sha256'] != code or frozen['threshold_operator'] != '>':
            raise ValueError('Frozen evaluator source/threshold operator changed')
        source = pilot.Inputs(semantic, change, retain_probabilities=True)
        pairs = continuous.audit_inputs(source, out)
        saved_inputs = pilot.read_json(note(root / '2026-10-19_input_hashes.json'))
        if source.evidence != saved_inputs:
            raise ValueError('Inputs differ from the completed frozen run')
        bundles, index = load_frozen(root, source, pairs, frozen, note)
        chosen = {}
        for selector in selected:
            matches = [p for p in pairs if (p['context']['building_id'], p['context']['reference_year'],
                       p['context']['source_year']) == tuple(selector)]
            if len(matches) != 1:
                raise ValueError(f'Selected gallery pair missing or ambiguous: {selector}')
            chosen[(matches[0]['context']['sequence_id'], matches[0]['context']['frame'])] = selector
        saved_rows = read_csv(note(root / 'per_pair.csv'))
        expected_rows = {identity(r): r for r in saved_rows}
        if len(saved_rows) != len(expected_rows):
            raise ValueError('Duplicate frozen metric rows')
        saved_maps = {(m['fold'], m['sequence_id'], m['frame']): m for m in index['maps']}
        if len(saved_maps) != len(index['maps']):
            raise ValueError('Duplicate frozen map artifacts')
        git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, capture_output=True, text=True, check=False)
        dirty = subprocess.run(['git', 'diff', '--binary', 'HEAD'], cwd=repo, capture_output=True, text=True, check=False)
        code[str(Path(__file__).resolve().relative_to(repo))] = pilot.sha256(__file__)
        config = dict(schema=SCHEMA, evaluation_run=str(root), frozen_config=frozen, selected_pairs=list(selected),
                      variants=list(VARIANTS), threshold_operator='>', refit=False, inference=False,
                      primary='sign_ceiling: positive-pair mean within facade, then equal facades',
                      git_commit=git.stdout.strip(), tracked_diff_sha256=pilot.digest(dirty.stdout), code_sha256=code,
                      software=dict(python=platform.python_version(), numpy=np.__version__), limitations=list(NOTES))
        config['config_hash'] = pilot.digest(config)
        pilot.atomic_json(out / 'resolved_config.json', config)
        state['config_hash'] = config['config_hash']
        for item in index['models']:
            destination = pilot.safe_path(out, item['path'])
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(root / item['path'], destination)
        rows, replay, articles, selected_index, transitions = [], [], [], [], []
        seen, seen_maps = set(), set()
        contexts = 0
        for pair in pairs:
            original = pair['context']
            folds = ['all_train']
            if original['split'] == 'train':
                folds.append('holdout-' + pilot.digest(original['building_id'])[:12])
            truth, known = continuous.event_truth(pair), pair['labels'] != 255
            delta = pair['source']['probabilities'] - pair['reference']['probabilities']
            signed = np.stack((delta, -delta))
            for fold in folds:
                contexts += 1
                split = ('train_fit' if fold == 'all_train' else 'train_fusion_oof') if original['split'] == 'train' else original['split']
                context = dict(original, split=split, source_split=original['split'], fold=fold)
                print(f"Audit frozen: {contexts} {split} {original['building_id']} frame={original['frame']}", flush=True)
                bundle = bundles[fold]
                old_maps = None
                if split != 'train_fit':
                    map_key = (fold, original['sequence_id'], original['frame'])
                    item = saved_maps[map_key]
                    path = note(pilot.safe_path(root, item['artifact_path']))
                    if pilot.sha256(path) != item['artifact_sha256']:
                        raise ValueError('Frozen map checksum mismatch')
                    with np.load(path, allow_pickle=False) as z:
                        old_maps = dict(z)
                    for k in ('labels', 'support'):
                        if not np.array_equal(pair[k], old_maps[k]):
                            raise ValueError('Frozen map support/GT mismatch')
                    seen_maps.add(map_key)
                predictions = {}
                for name in continuous.VARIANTS:
                    events, scores = continuous.score_pair(pair, name, bundle['models'].get(name))
                    threshold = bundle['thresholds'][name]['threshold']
                    measured, mask, predicted = measurements(pair, events, scores, threshold, source.categories)
                    for measured_row in measured:
                        key = identity(dict(context, variant=name, **measured_row))
                        expected = expected_rows[key]
                        for count in pilot.semantic_report.COUNTS:
                            if measured_row[count] != int(expected[count]):
                                raise ValueError(f'Frozen metric replay mismatch: {key}, {count}')
                        seen.add(key)
                    if old_maps is not None:
                        if not np.array_equal(mask, old_maps[name + '__prediction']):
                            raise ValueError('Frozen binary mask replay mismatch')
                        if predicted is not None:
                            for d, direction in enumerate(('gain', 'loss')):
                                if not np.array_equal(pilot.pack(predicted[d]), old_maps[name + '__' + direction]):
                                    raise ValueError('Frozen event mask replay mismatch')
                    replay.append(dict(context, variant=name, metric_rows=len(measured), counts_match=True,
                                       saved_masks_checked=old_maps is not None, passed=True))
                    if predicted is None:
                        continue
                    predictions[name] = predicted
                    header = dict(context, variant=name, threshold=threshold)
                    rows.append(dict(header, class_name='__all__', direction='gain_loss',
                                     **decompose(truth, signed, predicted, known)))
                    for d, direction in enumerate(('gain', 'loss')):
                        for c, category in enumerate(source.categories):
                            rows.append(dict(header, class_name=category['name'], direction=direction,
                                             **decompose(truth[d, c], signed[d, c], predicted[d, c], known)))
                if fold == 'all_train' and (original['sequence_id'], original['frame']) in chosen:
                    transition_rows = gt_transitions(pair, source.names)
                    transitions.extend(dict(context, **r) for r in transition_rows)
                    article, entry = gallery(out, pair, source.categories, predictions, context, transition_rows)
                    articles.append(article)
                    selected_index.append(entry)
        if seen != set(expected_rows) or seen_maps != set(saved_maps) or contexts != upstream['parity_pairs']:
            raise ValueError('Frozen replay cohort incomplete')
        summaries, facades = aggregate(rows)
        report(out, summaries, articles, source.categories)
        for name, values in (('per_pair', rows), ('per_facade', facades), ('summary', summaries),
                             ('replay_checks', replay), ('gt_transitions', transitions)):
            pilot.semantic_report._write_csv(out / (name + '.csv'), values)
        pilot.atomic_json(out / 'selected_pairs.json', selected_index)
        state.update(status='completed', pairs=len(pairs), evaluation_contexts=contexts,
                     metric_rows_replayed=len(seen), gallery_pairs=len(articles), result_rows=len(rows), replay_passed=True)
    except KeyboardInterrupt:
        state.update(status='interrupted', error='KeyboardInterrupt')
    except Exception as error:
        state.update(status='blocked', error=f'{type(error).__name__}: {error}')
    finally:
        if source is not None:
            evidence.update(source.evidence)
        pilot.atomic_json(out / 'input_hashes.json', evidence)
        artifacts = [dict(path=str(p.relative_to(out)), sha256=pilot.sha256(p), bytes=p.stat().st_size)
                     for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('run.json', 'artifacts.json')]
        pilot.atomic_json(out / 'artifacts.json', artifacts)
        state['elapsed_seconds'] = time.monotonic() - started
        pilot.atomic_json(out / 'run.json', state)
    print(json.dumps(state, ensure_ascii=False), flush=True)
    print(f'Results: {out}', flush=True)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-run', required=True, help='Completed directed_delta evaluation with models and maps')
    parser.add_argument('--out', required=True, help='New directory; never overwritten')
    args = parser.parse_args(argv)
    state = run(args.evaluation_run, args.out)
    return 0 if state['status'] == 'completed' else 130 if state['status'] == 'interrupted' else 2


if __name__ == '__main__':
    raise SystemExit(main())
