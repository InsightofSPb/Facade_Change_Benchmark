"""CPU-only metrics and gallery for frozen semantic/change route experiments.

Zero annotation bitsets mean unknown. Temporal targets are changes of annotation
membership, not independently reviewed change ground truth. ROI selection never
shrinks the full evaluation support passed by the orchestrator.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from . import real_gt_metrics as binary
    from .real_gt_slices import evaluate_class_slices
    from .robustness_common import atomic_json
except ImportError:
    import real_gt_metrics as binary
    from real_gt_slices import evaluate_class_slices
    from robustness_common import atomic_json

COUNTS = ('tp', 'fp', 'fn', 'tn', 'valid_pixels', 'ignored_pixels', 'positive_pixels', 'negative_pixels')
METRICS = ('precision', 'recall', 'f1', 'iou', 'fpr', 'marked_fraction')
GROUP = ('run_id', 'config_revision', 'kind', 'split', 'method', 'segmenter', 'variant', 'vocabulary', 'route',
         'mode', 'point', 'target', 'class_name', 'direction')


def _ontology(categories, category_to_channel):
    """Return categories in declared bit order, independent of COCO list order."""
    if not isinstance(categories, list) or not 1 <= len(categories) <= 16:
        raise ValueError('Expected 1–16 explicitly named categories')
    if any(not isinstance(c, dict) or 'id' not in c or not isinstance(c.get('name'), str)
           or not c['name'] for c in categories):
        raise ValueError('Each category needs an id and a nonempty name')
    by_id = {c['id']: c for c in categories}
    if len(by_id) != len(categories) or len({c['name'] for c in categories}) != len(categories):
        raise ValueError('Category IDs and names must be unique')
    if not isinstance(category_to_channel, list) or len(category_to_channel) != len(categories):
        raise ValueError('Explicit category_to_channel is required')
    by_channel = {}
    seen = set()
    for item in category_to_channel:
        category_id, channel = item.get('category_id'), item.get('channel')
        if category_id not in by_id or category_id in seen or type(channel) is not int or channel in by_channel:
            raise ValueError('Invalid or duplicate category/channel mapping')
        seen.add(category_id)
        by_channel[channel] = by_id[category_id]
    if set(by_channel) != set(range(len(categories))):
        raise ValueError('Packed annotation channels must be contiguous from zero')
    return [by_channel[channel] for channel in range(len(categories))]


def _grid(packed, support, category_count):
    if (not isinstance(packed, np.ndarray) or packed.dtype != np.uint16 or packed.ndim != 2
            or not isinstance(support, np.ndarray) or support.dtype != bool or support.shape != packed.shape):
        raise ValueError('Expected uint16 H,W annotation bitset and bool H,W support')
    if np.any(packed.astype(np.uint32) >= 1 << category_count):
        raise ValueError('Annotation bits exceed the declared ontology')


def _score_grid(scores, shape):
    if scores is None:
        return
    if (not isinstance(scores, np.ndarray) or scores.shape != shape
            or not np.issubdtype(scores.dtype, np.number) or np.isinf(scores).any()):
        raise ValueError('Score shape/type mismatch or infinite score')
    # NaN outside the selected inference ROI is allowed. It never removes GT
    # from evaluation: the boolean prediction determines all confusion counts.


def _evaluate(prediction, labels):
    row = binary.evaluate_mask(prediction, labels)
    row['marked_fraction'] = ((row['tp'] + row['fp']) / row['valid_pixels'] if row['valid_pixels'] else None)
    row['status'] = 'evaluated' if row['valid_pixels'] else 'no_evaluable_pixels'
    return row


def segmentation_metrics(scores, predicted, class_names, packed, support, categories, category_to_channel):
    """Per-GT-class counts on known pixels; missing model classes stay N/A.

    Channels are matched by exact semantic name, never by channel index. Extra
    model channels (including legacy BACKGROUND) are not ground-truth classes.
    All annotated memberships survive overlaps. Scores are kept separate from
    the boolean decision and are not assumed to be calibrated probabilities.
    """
    ordered = _ontology(categories, category_to_channel)
    _grid(packed, support, len(ordered))
    if (not isinstance(class_names, (list, tuple)) or len(set(class_names)) != len(class_names)
            or any(not isinstance(name, str) or not name for name in class_names)):
        raise ValueError('Model class names must be nonempty and unique')
    if (not isinstance(predicted, np.ndarray) or predicted.dtype != bool
            or predicted.shape != (len(class_names), *packed.shape)):
        raise ValueError('Predicted channels must be bool C,H,W and match class_names')
    _score_grid(scores, predicted.shape)
    known = support & (packed != 0)
    rows = []
    for channel, category in enumerate(ordered):
        header = dict(kind='segmentation', target='annotated_semantic_membership',
                      class_id=category['id'], class_name=category['name'], channel=channel,
                      direction='state')
        if category['name'] not in class_names:
            rows.append(dict(header, status='unsupported_class',
                             **dict.fromkeys((*COUNTS, *METRICS), None)))
            continue
        labels = np.full(packed.shape, 255, np.uint8)
        labels[known] = ((packed[known] & np.uint16(1 << channel)) != 0).astype(np.uint8)
        rows.append(dict(header, **_evaluate(predicted[class_names.index(category['name'])], labels)))
    return rows


def change_metrics(prediction, scores, reference_bits, source_bits, support, categories, category_to_channel):
    """Overall detection and class-conditioned change/gain/loss localisation.

    Class slices describe localisation on annotated class regions. They do not
    claim a binary detector predicted class labels. Pixels missed by an ROI are
    still false negatives because support is the full pair evaluation support.
    """
    ordered = _ontology(categories, category_to_channel)
    _grid(reference_bits, support, len(ordered))
    _grid(source_bits, support, len(ordered))
    if (not isinstance(prediction, np.ndarray) or prediction.dtype != bool or prediction.shape != support.shape):
        raise ValueError('Change prediction must be bool H,W on the full pair grid')
    _score_grid(scores, support.shape)
    labels = binary.derive_labels(reference_bits, source_bits, support, len(ordered))
    rows = [dict(kind='change', target='annotation_membership_change', class_id=None,
                 class_name='__all_changes__', direction='change', **_evaluate(prediction, labels))]
    for row in evaluate_class_slices(prediction, reference_bits, source_bits, support, labels, ordered):
        row['ignored_pixels'] = int(support.size - row['valid_pixels'])
        row['status'] = 'evaluated' if row['valid_pixels'] else 'no_evaluable_pixels'
        rows.append(dict(kind='change', target='class_conditioned_localisation', **row))
    return rows


def _mean(values):
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return sum(values) / len(values) if values else None


def aggregate_metrics(rows):
    """Equal observations within facade, then equal physical-facade weights.

    Counts remain summed observations and are not unique physical pixels.
    Unsupported/failed observations remain explicit in coverage; they never
    become perfect predictions or zero-valued metrics. Splits are never pooled.
    """
    groups = defaultdict(list)
    for row in rows:
        if row.get('split') not in ('val', 'test', 'train') or not row.get('building_id'):
            raise ValueError('Each metric row requires a split and physical building_id')
        key = tuple(row.get(name, '') for name in GROUP)
        groups[key].append(row)
    per_facade, summaries = [], []
    for key, members in sorted(groups.items(), key=lambda item: str(item[0])):
        header = dict(zip(GROUP, key))
        buildings = defaultdict(list)
        for row in members:
            buildings[row['building_id']].append(row)
        facade_rows = []
        for building, observations in sorted(buildings.items()):
            evaluated = [row for row in observations if row.get('status') == 'evaluated'
                         and row.get('valid_pixels', 0)]
            statuses = sorted({row.get('status', 'unknown') for row in observations})
            current = dict(header, building_id=building, aggregation='observation_mean_within_facade',
                           status='evaluated' if evaluated else statuses[0] if len(statuses) == 1 else 'unavailable',
                           statuses=statuses, observations=len(observations), evaluated_observations=len(evaluated),
                           **{name: sum(row[name] for row in evaluated) if evaluated else None for name in COUNTS},
                           **{name: _mean(row.get(name) for row in evaluated) for name in METRICS})
            facade_rows.append(current)
            per_facade.append(current)
        evaluated = [row for row in facade_rows if row['evaluated_observations']]
        statuses = sorted({row.get('status', 'unknown') for row in members})
        summaries.append(dict(header, aggregation='frame_then_facade_macro',
                              status='evaluated' if evaluated else statuses[0] if len(statuses) == 1 else 'unavailable',
                              statuses=statuses, facades=len(facade_rows), evaluated_facades=len(evaluated),
                              observations=len(members), evaluated_observations=sum(r['evaluated_observations'] for r in evaluated),
                              **{name: sum(row[name] for row in evaluated) if evaluated else None for name in COUNTS},
                              **{name: _mean(row.get(name) for row in evaluated) for name in METRICS},
                              **{name + '_facades': sum(row.get(name) is not None for row in evaluated) for name in METRICS}))
    return {'per_facade': per_facade, 'summary': summaries, 'class_macro': _class_macro(summaries)}


def _class_macro(summaries):
    groups = defaultdict(list)
    keys = tuple(key for key in GROUP if key != 'class_name')
    for row in summaries:
        if row.get('kind') == 'segmentation':
            groups[tuple(row.get(key, '') for key in keys)].append(row)
    result = []
    for key, members in sorted(groups.items(), key=lambda item: str(item[0])):
        for scope in ('common_without_advertisements', 'all_supported'):
            requested = [row for row in members if scope == 'all_supported' or row['class_name'] != 'ADVERTISEMENTS']
            available = [row for row in requested if row['status'] == 'evaluated']
            result.append(dict(zip(keys, key), class_scope=scope, aggregation='class_mean_of_facade_macros',
                               status='evaluated' if available else 'unavailable',
                               classes=len(available), requested_classes=len(requested),
                               class_names=[row['class_name'] for row in available],
                               **{name: _mean(row.get(name) for row in available) for name in METRICS}))
    return result


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ['status']
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def _table(rows, columns):
    parts = ['<table><thead><tr>' + ''.join('<th>' + html.escape(key) + '</th>' for key in columns) + '</tr></thead><tbody>']
    for row in rows:
        values = []
        for key in columns:
            value = row.get(key)
            if value is None:
                text = 'N/A'
            elif key in METRICS:
                text = f'{100 * value:.2f}%'
            else:
                text = str(value)
            values.append('<td>' + html.escape(text) + '</td>')
        parts.append('<tr>' + ''.join(values) + '</tr>')
    return ''.join(parts) + '</tbody></table>'


def _palette(categories):
    result = {}
    for category in categories:
        color = category.get('color', '')
        try:
            result[category['name']] = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], np.float32)
            if len(color) != 7 or color[0] != '#':
                raise ValueError()
        except (ValueError, IndexError):
            digest = hashlib.sha256(category['name'].encode()).digest()
            result[category['name']] = np.array([64 + value % 192 for value in digest[:3]], np.float32)
    return result


def _overlay(rgb, masks, names, palette):
    if masks.dtype != bool or masks.ndim != 3 or masks.shape[1:] != rgb.shape[:2] or len(names) != len(masks):
        raise ValueError('Invalid gallery semantic mask grid')
    paint, count = np.zeros_like(rgb, dtype=np.float32), np.zeros(rgb.shape[:2], np.float32)
    for name, mask in zip(names, masks):
        color = palette.get(name)
        if color is None:
            color = _palette([{'name': name}])[name]
        paint += mask[..., None] * color
        count += mask
    marked = count > 0
    output = rgb.copy()
    mixed = rgb.astype(np.float32) * .45 + (paint / np.maximum(count[..., None], 1)) * .55
    output[marked] = np.clip(mixed[marked], 0, 255).astype(np.uint8)
    return output


def _thumb(out, array):
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError('Gallery RGB must be uint8 H,W,3')
    key = hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()[:24]
    relative = Path('assets') / (key + '.jpg')
    path = out / relative
    if not path.is_file():
        image = Image.fromarray(array)
        image.thumbnail((320, 320))
        image.save(path, 'JPEG', quality=88)
    return '<img loading="lazy" src="' + relative.as_posix() + '" alt="prediction comparison">'


def _sam_instances(path, metadata, class_names, source, palette):
    """Read only raw instance masks/indices/scores, not large probability maps."""
    fields = ('instance_masks', 'instance_scores', 'instance_class_indices', 'instance_prompt_indices')
    with np.load(path, allow_pickle=False) as archive:
        if 'extra__instance_masks' not in archive:
            return None, []
        arrays = {name: archive['extra__' + name].copy() for name in fields}
    masks = arrays['instance_masks']
    if masks.dtype != bool or masks.ndim != 3 or masks.shape[1:] != source.shape[:2]:
        raise ValueError('Raw SAM3 masks do not match the native RGB grid')
    count = len(masks)
    if any(arrays[name].shape != (count,) for name in fields[1:]):
        raise ValueError('Raw SAM3 scores/class/prompt indices disagree with masks')
    prompts = metadata.get('prompt_texts', [])
    names = class_names or metadata.get('class_names', [])
    if not np.isfinite(arrays['instance_scores']).all():
        raise ValueError('Raw SAM3 instance scores are not finite')
    areas = masks.sum(axis=(1, 2), dtype=np.int64)
    fractions = areas / (source.shape[0] * source.shape[1])
    instances = []
    for index in range(count):
        class_index, prompt_index = int(arrays['instance_class_indices'][index]), int(arrays['instance_prompt_indices'][index])
        instances.append({'instance': index, 'class_index': class_index, 'prompt_index': prompt_index,
                          'class_name': names[class_index] if 0 <= class_index < len(names) else 'unknown',
                          'prompt': prompts[prompt_index] if 0 <= prompt_index < len(prompts) else 'unknown',
                          'instance_score': float(arrays['instance_scores'][index]),
                          'area_pixels': int(areas[index]), 'area_fraction': float(fractions[index])})
    largest = sorted(range(count), key=lambda i: (-int(areas[i]), i))[:3]
    images = []
    for index in largest:
        row = instances[index]
        caption = (f"Исходная маска SAM3 #{index}: {row['class_name']}; запрос «{row['prompt']}»; "
                   f"оценка {row['instance_score']:.3f}; площадь {row['area_fraction']:.1%}")
        images.append((caption, _overlay(source, masks[index:index + 1], [row['class_name']], palette)))
    return {'instance_count': count, 'max_area_fraction': float(fractions.max()) if count else 0.,
            'large_mask_threshold': .5, 'large_mask_count': int((fractions >= .5).sum()),
            'interpretation': 'area flag only; a large mask is not automatically an error',
            'instances': instances}, images


def render_report(out, manifest, records, prepared_path=None):
    """Write CSV/JSON/HTML without inference; return paths and row counts.

    Records: kind/split/building_id/sequence_id/frame/reference_frame/method/
    segmenter/variant/route/mode; metrics is a list returned above. A record may
    have `cache_artifact` (absolute or relative to out.parent) containing
    predictions[C,H,W] or prediction[H,W]. `class_names` is a record list or a
    Unicode NPZ array. Optional source/reference RGB, packed/source_bits/
    reference_bits and support may be in the same archive. Alternatively pass
    prepared_path to load these exclusively from the prepared frame manifest.
    Failed records with status/error remain in coverage, never zero metrics.
    """
    out = Path(out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'assets').mkdir(exist_ok=True)
    records = list(records)
    rows, coverage, timing = [], [], []
    for record in records:
        context = {key: record.get(key, '') for key in (*GROUP, 'building_id', 'sequence_id', 'frame', 'reference_frame')}
        for row in record.get('metrics', record.get('rows', [])):
            rows.append(dict(context, **row))
        coverage.append({**context, 'status': record.get('status', 'completed'),
                         'cache_artifact': record.get('cache_artifact', ''), 'error': record.get('error', '')})
        timing.append({**{key: record.get(key, '') for key in ('kind', 'split', 'method', 'segmenter', 'variant', 'route', 'mode')},
                       'unit_id': record.get('unit_id', ''), 'cached': record.get('cached', record.get('reused', False)),
                       'elapsed_s': record.get('elapsed_seconds', record.get('elapsed_s', record.get('duration_s'))),
                       'init_seconds': record.get('init_seconds'),
                       'status': record.get('status', 'completed')})
    aggregates = aggregate_metrics(rows)
    for name, value in (('per_unit', rows), ('per_facade', aggregates['per_facade']),
                        ('summary', aggregates['summary']), ('class_macro', aggregates['class_macro']),
                        ('coverage', coverage), ('timing', timing)):
        _write_csv(out / (name + '.csv'), value)
    atomic_json(out / 'metrics.json', aggregates)
    ordered = _ontology(manifest['categories'], manifest['category_to_channel'])
    palette = _palette(ordered)
    by_sequence = {s['sequence_id']: s for s in manifest.get('sequences', [])}
    frames = {}
    errors, gallery, sam_diagnostics, raw_seen = [], [], [], {}

    def prepared_frame(sequence_id, index):
        key = (sequence_id, int(index))
        if key not in frames:
            try:
                from .night_common import load_frame
            except ImportError:
                from scripts.real_eval.night_common import load_frame
            frames[key] = load_frame(Path(prepared_path), by_sequence[sequence_id]['frames'][int(index)])
        return frames[key]

    for record in records:
        artifact = record.get('cache_artifact')
        if not artifact:
            continue
        try:
            path = Path(artifact)
            if not path.is_absolute():
                path = out.parent / path
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
            source = arrays.get('source', arrays.get('rgb'))
            source_bits = arrays.get('source_bits', arrays.get('packed'))
            reference, reference_bits = arrays.get('reference'), arrays.get('reference_bits')
            support = arrays.get('support')
            if source is None and prepared_path is not None:
                value = prepared_frame(record['sequence_id'], record['frame'])
                source, source_bits = value['rgb'], value['packed']
                support = value['support'] if support is None else support
            if record.get('kind') == 'change' and reference is None and prepared_path is not None:
                value = prepared_frame(record['sequence_id'], record['reference_frame'])
                reference, reference_bits = value['rgb'], value['packed']
                support = value['support'] if support is None else support & value['support']
            if source is None:
                continue
            if support is None:
                support = np.ones(source.shape[:2], bool)
            images = []
            if reference is not None:
                images.append(('Ранняя дата', reference))
            images.append(('Поздняя дата' if reference is not None else 'RGB', source))
            if source_bits is not None:
                masks = np.stack([(source_bits & (1 << i)) != 0 for i in range(len(ordered))]) & support
                images.append(('Разметка состояния', _overlay(source, masks, [c['name'] for c in ordered], palette)))
            if record.get('kind') == 'change':
                if reference_bits is not None and source_bits is not None:
                    labels = binary.derive_labels(reference_bits, source_bits, support, len(ordered))
                    images.append(('Изменение аннотаций', _overlay(source, (labels == 1)[None], ['change'], {'change': np.array([255, 130, 0])})))
                prediction = arrays.get('prediction', arrays.get('native_prediction'))
                if prediction is not None:
                    images.append(('Предсказание изменений', _overlay(source, (prediction & support)[None], ['change'], {'change': np.array([0, 220, 255])})))
            elif 'predictions' in arrays or 'predicted' in arrays:
                predictions = arrays.get('predictions', arrays.get('predicted'))
                names = record.get('class_names', arrays.get('class_names'))
                if names is not None:
                    images.append(('Предсказание состояния', _overlay(source, predictions & support, list(names), palette)))
            raw_note = ''
            native_path = record.get('native_cache_artifact')
            if native_path:
                native_path = Path(native_path)
                if not native_path.is_absolute():
                    native_path = out.parent / native_path
                raw_key = str(native_path.resolve())
                if raw_key in raw_seen:
                    raw_note = '<p><a href="#' + raw_seen[raw_key] + '">Исходные маски SAM3 для этого изображения и словаря</a></p>'
                else:
                    diagnostic, raw_images = _sam_instances(native_path, record.get('native_method_metadata', {}),
                                                           record.get('class_names', []), source, palette)
                    if diagnostic is not None:
                        anchor = 'sam-' + hashlib.sha256(raw_key.encode()).hexdigest()[:16]
                        raw_seen[raw_key] = anchor
                        sam_diagnostics.append({**{key: record.get(key) for key in
                                                   ('split', 'building_id', 'sequence_id', 'frame', 'segmenter', 'variant')},
                                                'native_cache_artifact': raw_key, **diagnostic})
                        raw_note = (f'<p id="{anchor}">Исходные ответы SAM3: {diagnostic["instance_count"]} масок; '
                                    f'максимальная площадь {diagnostic["max_area_fraction"]:.1%}; '
                                    f'масок площадью ≥50% изображения: {diagnostic["large_mask_count"]}. '
                                    'Ниже — до трёх крупнейших масок до объединения. Большая площадь сама по себе не означает ошибку.</p>')
                        images.extend(raw_images)
            title = ' · '.join(str(record.get(key, '')) for key in ('split', 'building_id', 'frame', 'method', 'segmenter', 'variant', 'route'))
            image_html = ''.join('<figure>' + _thumb(out, array) + '<figcaption>' + html.escape(label) + '</figcaption></figure>' for label, array in images)
            gallery.append('<details class="card" data-split="' + html.escape(str(record.get('split', '')), quote=True)
                           + '"><summary>' + html.escape(title) + '</summary>' + raw_note + '<div class="images">' + image_html + '</div></details>')
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            errors.append({'unit_id': record.get('unit_id'), 'cache_artifact': artifact, 'error': str(error)})
    atomic_json(out / 'gallery_errors.json', errors)
    atomic_json(out / 'sam3_instances.json', sam_diagnostics)
    _write_csv(out / 'sam3_instances.csv', [{**{key: value for key, value in row.items() if key != 'instances'},
                                          **instance} for row in sam_diagnostics for instance in row['instances']])
    columns = ('split', 'method', 'segmenter', 'variant', 'route', 'mode', 'class_name', 'direction',
               'status', 'iou', 'f1', 'precision', 'recall', 'fpr', 'evaluated_facades', 'evaluated_observations')
    parts = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Семантика и изменения фасадов</title>',
             '<style>body{font:15px system-ui;margin:2rem;color:#222}table{border-collapse:collapse;font-size:12px}'
             'td,th{border:1px solid #ccc;padding:.3rem}th{background:#eee}summary{cursor:pointer;padding:.7rem}'
             '.images{display:flex;flex-wrap:wrap}figure{margin:.4rem}figcaption{font-size:12px}img{max-width:320px}'
             '.card{border:1px solid #ddd;margin:.5rem 0}.scroll{overflow:auto}p{max-width:1050px}</style>',
             '<h1>Семантика и временные изменения</h1><p>Метрики разделены по TRAIN, VAL и TEST. '
             'Агрегация: среднее наблюдений внутри физического фасада, затем равные веса фасадов. '
             'Неразмеченные пиксели исключены; вне выбранных ROI пропущенные изменения остаются FN. '
             'Временная цель — изменение членства в COCO-классах после совмещения, а не независимая временная разметка.</p>',
             '<p>N/A означает отсутствие класса, измерения либо определённого знаменателя. '
             'Классовые change/gain/loss — срезы локализации бинарного детектора, не его предсказания названий классов. '
             'Показаны выбранные кропы; независимость от старого обучения проверяется отдельно.</p>',
             '<p>CSV: ' + ' · '.join('<a href="' + name + '.csv">' + name + '</a>' for name in ('per_unit', 'per_facade', 'summary', 'class_macro', 'coverage', 'timing')) + '</p>',
             '<p>Полнота запрошенной очереди (ожидалось / завершено / ошибка / не начато): '
             '<a href="coverage_matrix.csv">CSV</a> · <a href="coverage_matrix.json">JSON</a>. '
             'Оценка оставшегося времени: <a href="estimate.json">estimate.json</a>.</p>',
             '<details><summary>Метрики по классам и вариантам</summary><div class="scroll">' + _table(aggregates['summary'], columns) + '</div></details>',
             '<details><summary>Покрытие и ошибки</summary><div class="scroll">' + _table(coverage, ('split', 'method', 'variant', 'route', 'status', 'error')) + '</div></details>',
             '<h2>Галерея</h2><p>Слои с пересечениями показаны смешением цветов. Серый/неокрашенный фон сам по себе не означает здоровый фасад.</p>',
             '<p>Исходные маски SAM3, запросы, оценки и площади: <a href="sam3_instances.json">JSON</a> · <a href="sam3_instances.csv">CSV</a>.</p>',
             '<p>' + ' '.join('<span style="border-left:12px solid rgb(' + ','.join(str(int(v)) for v in palette[c['name']]) + ');padding:0 .6rem">' + html.escape(c['name']) + '</span>' for c in ordered) + '</p>',
             '<label>Поднабор <select id="split"><option value="">Все отдельно</option><option>val</option><option>test</option><option>train</option></select></label>',
             ''.join(gallery), '<p>Ошибок визуализации: ' + str(len(errors)) + '; подробности в gallery_errors.json.</p>',
             '<script>document.getElementById("split").addEventListener("change",function(){document.querySelectorAll(".card").forEach(c=>c.hidden=!!this.value&&c.dataset.split!==this.value)});</script></html>']
    (out / 'report.html').write_text(''.join(parts), encoding='utf-8')
    return {'report': str(out / 'report.html'), 'metric_rows': len(rows), 'gallery_cards': len(gallery),
            'gallery_errors': len(errors), 'records': len(records), 'sam3_native_inputs': len(sam_diagnostics)}
