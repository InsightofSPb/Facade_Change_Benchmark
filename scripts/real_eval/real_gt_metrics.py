#!/usr/bin/env python3
"""Метрики сохранённых реальных VAL-карт по изменениям исходной COCO-разметки."""
import argparse
import base64
import csv
import hashlib
import html
import importlib
import io
import json
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

MODES = ('previous', 'earliest', 'history')
NATIVE = {'geoscd', 'anychange'}
GATES = {'ssim', 'dinov2', 'semantic_dino', 'msdzip_abs', 'msdzip_mod256'}
METRICS = ('precision', 'recall', 'f1', 'iou', 'fpr')
COUNTS = ('tp', 'fp', 'fn', 'tn', 'valid_pixels', 'ignored_pixels', 'positive_pixels', 'negative_pixels')


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ['status']
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                          for key, value in row.items()} for row in rows)


def _dilate(mask, radius):
    if not radius:
        return mask
    height, width = mask.shape
    padded = np.pad(mask, radius, constant_values=False)
    result = np.zeros_like(mask)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result |= padded[dy:dy + height, dx:dx + width]
    return result


def derive_labels(old_bits, new_bits, support, category_count, boundary_tolerance=0):
    """All-category bitset change; zero bitsets mean unknown, never healthy."""
    if (old_bits.dtype != np.uint16 or new_bits.dtype != np.uint16 or old_bits.shape != new_bits.shape
            or old_bits.shape != support.shape or support.dtype != bool or support.ndim != 2
            or not 1 <= category_count <= 16 or type(boundary_tolerance) is not int or boundary_tolerance < 0):
        raise ValueError('Некорректные битовые маски, поддержка или радиус допуска')
    if np.any(old_bits.astype(np.uint32) >= (1 << category_count)) or np.any(new_bits.astype(np.uint32) >= (1 << category_count)):
        raise ValueError('В маске есть неизвестный бит категории')
    known = support & (old_bits != 0) & (new_bits != 0)
    changed = old_bits != new_bits
    labels = np.full(support.shape, 255, np.uint8)
    labels[known] = changed[known].astype(np.uint8)
    if boundary_tolerance:
        robust = np.zeros_like(support)
        for channel in range(category_count):
            old, new = (old_bits & (1 << channel)) != 0, (new_bits & (1 << channel)) != 0
            robust |= (new & ~_dilate(old, boundary_tolerance)) | (old & ~_dilate(new, boundary_tolerance))
        labels[known & changed & ~robust] = 255
    return labels


def _metrics(counts):
    tp, fp, fn, tn = (counts[key] for key in ('tp', 'fp', 'fn', 'tn'))
    ratio = lambda a, b: a / b if b else None
    return dict(precision=ratio(tp, tp + fp) if tp + fp else (0. if fn else None),
                recall=ratio(tp, tp + fn), f1=ratio(2 * tp, 2 * tp + fp + fn),
                iou=ratio(tp, tp + fp + fn), fpr=ratio(fp, fp + tn))


def evaluate_mask(prediction, labels):
    if prediction.dtype != bool or prediction.shape != labels.shape or labels.dtype != np.uint8 or not np.isin(labels, [0, 1, 255]).all():
        raise ValueError('Некорректные предсказание или GT')
    positive, negative = labels == 1, labels == 0
    counts = dict(tp=int((prediction & positive).sum()), fp=int((prediction & negative).sum()),
                  fn=int((~prediction & positive).sum()), tn=int((~prediction & negative).sum()),
                  valid_pixels=int((labels != 255).sum()), ignored_pixels=int((labels == 255).sum()),
                  positive_pixels=int(positive.sum()), negative_pixels=int(negative.sum()))
    return {**counts, **_metrics(counts)}


def default_adapter(config, selected, contract, repo, boundary_tolerance, decoded_cache_bytes=0):
    """Reconstruct frozen chronology; no model import, new alignment or crop mining."""
    sys.path.insert(0, str(repo))
    inputs = importlib.import_module('facade_change.research.2026-10-14_temporal_inputs')
    from facade_change.segmentation.data import PreparedData
    import cv2
    fingerprints = {}
    parents, _, _, _ = inputs._real(config, fingerprints)
    frozen = contract['inputs']['files_sha256']
    for path, digest in fingerprints.items():
        if frozen.get(path) != digest:
            raise ValueError(f'Исходные данные/геометрия отличаются от замороженного запуска: {path}')
    module_key = 'research/2026-10-14_temporal_inputs.py'
    expected_code = contract.get('source_sha256', {}).get(module_key)
    if expected_code and sha256(repo / 'facade_change' / module_key) != expected_code:
        raise ValueError('Изменился код восстановления исходных историй')
    data = PreparedData(config['dataset'], splits=('val',))
    parent_index = {row['sequence_id']: row for row in parents if row['split'] == 'val'}
    sequences = {}
    for identifier in selected:
        match = re.fullmatch(r'(real-[a-f0-9]+)-x(\d+)-y(\d+)', identifier)
        if not match or match[1] not in parent_index or identifier not in contract['sequence_ids']:
            raise ValueError(f'Не найдена замороженная реальная история: {identifier}')
        parent = parent_index[match[1]]
        x, y = int(match[2]), int(match[3])
        height, width = parent['shape']
        tile = config.get('real_tile_size', 512)
        width, height = min(tile, width), min(tile, height)
        sequences[identifier] = {**parent, 'sequence_id': identifier, 'crop_xywh': [x, y, width, height], 'shape': [height, width]}

    if decoded_cache_bytes < 0:
        raise ValueError("decoded_cache_bytes must be nonnegative")
    decoded_cache, cached_bytes = OrderedDict(), 0

    def decoded(image_id):
        nonlocal cached_bytes
        if image_id in decoded_cache:
            decoded_cache.move_to_end(image_id)
            return decoded_cache[image_id]
        arrays = data.load_image(image_id)
        size = sum(array.nbytes for array in arrays)
        if 0 < size <= decoded_cache_bytes:
            while decoded_cache and cached_bytes + size > decoded_cache_bytes:
                _, previous = decoded_cache.popitem(last=False)
                cached_bytes -= sum(array.nbytes for array in previous)
            decoded_cache[image_id] = arrays
            cached_bytes += size
        return arrays

    def loader(sequence, index):
        frame = sequence['frames'][index]
        current = data.images.get(frame['image_id'])
        if current is None or any(current.get(key) != frame.get(key) for key in ('sha256', 'building_id', 'view_id', 'year', 'width', 'height')):
            raise ValueError('Исходный COCO-кадр не соответствует сохранённой хронологии')
        rgb, packed, opaque = decoded(frame['image_id'])
        x, y, width, height = sequence['crop_xywh']
        shift = np.array([[1., 0, -x], [0, 1., -y], [0, 0, 1.]])
        matrix = shift @ np.asarray(frame['source_to_reference'])
        warp = lambda array, transform, size, flags: cv2.warpPerspective(array, transform, size, flags=flags,
                                                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        result = dict(rgb=warp(rgb, matrix, (width, height), cv2.INTER_LINEAR),
                      support=warp(opaque.astype(np.float32), matrix, (width, height), cv2.INTER_LINEAR) >= 1 - 1e-6,
                      packed=warp(packed, matrix, (width, height), cv2.INTER_NEAREST))
        if boundary_tolerance:
            radius = boundary_tolerance
            halo_shift = np.array([[1., 0, radius], [0, 1., radius], [0, 0, 1.]])
            result['packed_halo'] = warp(packed, halo_shift @ matrix, (width + 2 * radius, height + 2 * radius), cv2.INTER_NEAREST)
        return result
    provenance = dict(categories=data.categories, category_to_channel=data.metadata['category_to_channel'],
                      prepared_data=data.metadata, reconstructed_input_sha256=fingerprints,
                      input_module_sha256=sha256(repo / 'facade_change' / module_key))
    return sequences, loader, provenance


def load_map(root, record, artifact_hashes):
    path = (root / record['artifact_path']).resolve()
    if root not in path.parents or sha256(path) != record['artifact_sha256']:
        raise ValueError(f'Неверный путь/SHA-256 сохранённой карты: {path}')
    artifact_hashes[str(path)] = record['artifact_sha256']
    names = ['scores', 'support', 'reference', 'source']
    if record['method'] in NATIVE:
        names.append('native_prediction')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in names}
    scores, support = arrays['scores'], arrays['support']
    if (scores.dtype != np.float32 or scores.ndim != 2 or support.dtype != bool or support.shape != scores.shape
            or not np.isfinite(scores[support]).all() or not np.isnan(scores[~support]).all()):
        raise ValueError('Некорректная сохранённая карта оценки/поддержки')
    for key in ('reference', 'source'):
        if arrays[key].dtype != np.uint8 or arrays[key].shape != (*support.shape, 3):
            raise ValueError('Некорректная RGB-решётка сохранённой карты')
    if 'native_prediction' in arrays and (arrays['native_prediction'].dtype != bool or arrays['native_prediction'].shape != support.shape):
        raise ValueError('Некорректная авторская бинарная маска')
    return arrays


def calibration(saved, group, budget):
    candidates = [row for row in saved.get('calibration', [])
                  if tuple(row.get(key) for key in ('method', 'variant', 'mode', 'target')) == group]
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError(f'Неоднозначная калибровка: {group}')
    row = candidates[0]
    if row.get('status') != 'calibrated' or tuple(row.get(key) for key in ('selection_split', 'selection_domain', 'selection_arm')) != ('val', 'synthetic', 'clean'):
        return None
    choices = [item for item in row['choices'] if abs(float(item['fpr_target']) - budget) < 1e-12]
    if len(choices) != 1 or not np.isfinite(choices[0]['threshold']):
        raise ValueError(f'Нет фиксированного порога {budget}: {group}')
    return {**choices[0], 'group': list(group), 'selection_split': 'val', 'selection_domain': 'synthetic',
            'selection_arm': 'clean', 'source_unit_ids': row.get('source_unit_ids', [])}


def _aggregate(rows, field):
    groups = defaultdict(list)
    for row in rows:
        if row['valid_pixels']:
            for subset in ('all', 'positive', 'no_event'):
                if subset == 'all' or (subset == 'positive') == (row['positive_pixels'] > 0):
                    groups[(row['cohort'], row['mode'], row['predictor'], subset, row[field])].append(row)
    result = []
    for (cohort, mode, predictor, subset, identity), members in sorted(groups.items()):
        counts = {key: sum(row[key] for row in members) for key in COUNTS}
        result.append(dict(cohort=cohort, mode=mode, predictor=predictor, subset=subset, **{field: identity},
                           evaluated_pairs=len(members), **counts, **_metrics(counts)))
    return result


def aggregate_metrics(rows):
    sequences, facades = _aggregate(rows, 'sequence_id'), _aggregate(rows, 'building_id')
    aggregates = []
    for level, source in (('facade_pooled_macro', facades), ('sequence_pooled_macro', sequences)):
        grouped = defaultdict(list)
        for row in source:
            grouped[(row['cohort'], row['mode'], row['predictor'], row['subset'])].append(row)
        for (cohort, mode, predictor, subset), members in sorted(grouped.items()):
            values = {key: [row[key] for row in members if row[key] is not None] for key in METRICS}
            aggregates.append(dict(aggregation=level, cohort=cohort, mode=mode, predictor=predictor, subset=subset,
                                   units=len(members), evaluated_pairs=sum(row['evaluated_pairs'] for row in members),
                                   metric_units={key: len(value) for key, value in values.items()},
                                   **{key: float(np.mean(value)) if value else None for key, value in values.items()},
                                   **{key: sum(row[key] for row in members) for key in COUNTS}))
    frame_groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if not row['valid_pixels']:
            continue
        for subset in ('all', 'positive', 'no_event'):
            if subset == 'all' or (subset == 'positive') == (row['positive_pixels'] > 0):
                frame_groups[(row['cohort'], row['mode'], row['predictor'], subset)][row['building_id']].append(row)
    for (cohort, mode, predictor, subset), buildings in sorted(frame_groups.items()):
        values = {key: [] for key in METRICS}
        members = [row for frames in buildings.values() for row in frames]
        for frames in buildings.values():
            for key in METRICS:
                finite = [row[key] for row in frames if row[key] is not None]
                if finite:
                    values[key].append(float(np.mean(finite)))
        aggregates.append(dict(aggregation='frame_then_facade_macro', cohort=cohort, mode=mode, predictor=predictor,
                               subset=subset, units=len(buildings), evaluated_pairs=len(members),
                               metric_units={key: len(value) for key, value in values.items()},
                               **{key: float(np.mean(value)) if value else None for key, value in values.items()},
                               **{key: sum(row[key] for row in members) for key in COUNTS}))
    return sequences, facades, aggregates


def _gt_figure(rgb, labels, caption):
    overlay = rgb.copy()
    overlay[labels == 255] = 128
    overlay[labels == 1] = (.4 * overlay[labels == 1] + .6 * np.array([255, 180, 0])).astype(np.uint8)
    picture = Image.fromarray(overlay)
    picture.thumbnail((384, 384))
    buffer = io.BytesIO()
    picture.save(buffer, format='PNG')
    return f'<figure><img src="data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"><figcaption>{html.escape(caption)}</figcaption></figure>'


PALETTE = [(230, 40, 55), (35, 180, 80), (50, 110, 240), (240, 180, 20), (180, 65, 220), (20, 190, 200),
           (245, 110, 40), (125, 200, 45), (240, 90, 170), (50, 140, 150), (120, 70, 200), (200, 140, 80),
           (100, 180, 245), (150, 160, 55), (210, 100, 100), (110, 110, 220)]


def _semantic_figure(rgb, bits, support, category_count, caption):
    colors, count = np.zeros(rgb.shape, np.float32), np.zeros(bits.shape, np.float32)
    for channel in range(category_count):
        mask = (bits & (1 << channel)) != 0
        colors[mask] += PALETTE[channel]
        count[mask] += 1
    valid = support & (count > 0)
    overlay = np.full_like(rgb, 128)
    overlay[valid] = (.4 * rgb[valid] + .6 * colors[valid] / count[valid, None]).astype(np.uint8)
    # Reuse the PNG wrapper with no additional change overlay.
    return _gt_figure(overlay, np.zeros(bits.shape, np.uint8), caption)


def run(run_path, snapshot_path, review_path, out_path, geoscd_run=None, repo_path=None,
        all_real_val=False, boundary_tolerance=0, input_adapter=None):
    root, snapshot, review, out = (Path(value).expanduser().resolve() for value in (run_path, snapshot_path, review_path, out_path))
    review = review / 'summary.json' if review.is_dir() else review
    if type(boundary_tolerance) is not int or boundary_tolerance not in (0, 2):
        raise ValueError('Допуск границ должен быть 0 или 2 пикселя')
    protected = [root, snapshot, review.parent] + ([Path(geoscd_run).expanduser().resolve()] if geoscd_run else [])
    if out.exists() or any(out == path or path in out.parents or out in path.parents for path in protected):
        raise FileExistsError('Нужен новый отдельный каталог результатов')
    units_path, quick_path = snapshot / '2026-10-17_source_units.json', snapshot / '2026-10-17_quick_summary.json'
    metadata_paths = [units_path, quick_path, review, root / 'resolved_config.json', root / 'experiment_contract.json']
    records, saved, selection, config, contract = map(read_json, metadata_paths)
    if Path(saved['source_run']).expanduser().resolve() != root:
        raise ValueError('Снимок отчёта относится к другому запуску')
    if (review.parent / 'provenance.json').is_file():
        review_provenance = read_json(review.parent / 'provenance.json')
        metadata_paths.append(review.parent / 'provenance.json')
        if review_provenance.get('source_run') and Path(review_provenance['source_run']).expanduser().resolve() != root:
            raise ValueError('Галерея относится к другому запуску')
        for path in (units_path, quick_path):
            expected = review_provenance.get('source_metadata_sha256', {}).get(str(path))
            if expected and expected != sha256(path):
                raise ValueError('Изменился снимок, использованный при выборе историй')
    selected = (sorted({row['sequence_id'] for row in records if (row.get('domain'), row.get('split'), row.get('method'), row.get('mode'), row.get('status')) == ('real', 'val', 'rgb_diff', 'previous', 'completed')})
                if all_real_val else selection['selected_sequences'])
    if not selected or len(selected) != len(set(selected)):
        raise ValueError('Нужна непустая выборка уникальных реальных VAL-историй')
    filtered = [row for row in records if row.get('domain') == 'real' and row.get('split') == 'val'
                and row.get('sequence_id') in selected and row.get('mode') in MODES and row.get('frame', 0) >= 1]
    overlay_roots, overlay_failures = {}, []
    if geoscd_run:
        extra = Path(geoscd_run).expanduser().resolve()
        overlay = read_json(extra / 'run.json')
        metadata_paths.append(extra / 'run.json')
        if (Path(overlay['source_run']).expanduser().resolve() != root
                or overlay['snapshot_source_units_sha256'] != sha256(units_path)
                or overlay['snapshot_quick_summary_sha256'] != sha256(quick_path)):
            raise ValueError('Повтор GeoSCD не соответствует исходному запуску/снимку')
        for path in sorted((extra / 'units').glob('*.json')):
            row = read_json(path)
            if (row.get('domain'), row.get('split'), row.get('method'), row.get('variant')) != ('real', 'val', 'geoscd', 'frozen') or row.get('mode') not in ('previous', 'earliest') or row.get('sequence_id') not in selected:
                continue
            metadata_paths.append(path)
            if row.get('status') != 'completed':
                overlay_failures.append(row)
                continue
            key = (row['sequence_id'], row['frame'], row['mode'])
            filtered = [old for old in filtered if not (old.get('method') == 'geoscd' and old.get('variant') == 'frozen'
                       and (old['sequence_id'], old['frame'], old['mode']) == key)]
            filtered.append(row)
            overlay_roots[row['unit_id']] = extra
    indexed, group_keys = {}, defaultdict(set)
    groups = set()
    for row in filtered:
        mode = row['mode']
        group = (row['method'], row['variant'], mode, 'earliest_state' if mode == 'earliest' else 'step_event')
        if row.get('target') != group[3]:
            raise ValueError('Режим и смысл цели сохранённой карты не согласованы')
        groups.add(group)
        if row.get('status') != 'completed':
            continue
        key = (row['sequence_id'], row['frame'], group)
        if key in indexed:
            raise ValueError('Дублирующиеся результаты метода для одной пары')
        indexed[key] = row
        group_keys[group].add(key[:2])
    # Missing methods remain explicit in coverage, even if no selected frame succeeded.
    for mode in ('previous', 'earliest'):
        groups.update((method, 'frozen', mode, 'earliest_state' if mode == 'earliest' else 'step_event') for method in NATIVE)
    for row in saved.get('calibration', []):
        if row.get('mode') in MODES:
            groups.add(tuple(row.get(key) for key in ('method', 'variant', 'mode', 'target')))
    packaged_repo = Path(__file__).resolve().parents[2]
    repo = (Path(repo_path).expanduser().resolve() if repo_path else
            packaged_repo if (packaged_repo / "facade_change" / "__init__.py").is_file() else root.parents[1])
    print(f'Восстанавливаем COCO-разметку для {len(selected)} реальных VAL-историй; модели не запускаются.', flush=True)
    sequences, loader, gt_provenance = (input_adapter(config, selected, contract) if input_adapter else
                                       default_adapter(config, selected, contract, repo, boundary_tolerance))
    if set(sequences) != set(selected):
        raise ValueError('Выборка GT отличается от выбранных историй')
    category_count = len(gt_provenance['categories'])
    expected_keys = {(identifier, frame) for identifier in selected for frame in range(1, len(sequences[identifier]['frames']))}
    for identifier in selected:
        sequence = sequences[identifier]
        baseline = {frame for seq, frame, group in indexed if seq == identifier and group == ('rgb_diff', 'frozen', 'previous', 'step_event')}
        if baseline != set(range(1, len(sequence['frames']))):
            raise ValueError(f'RGB_DIFF не содержит целую восстановленную историю: {identifier}')
    specs = {mode: {} for mode in MODES}
    thresholds = {}
    for group in sorted(groups):
        method, variant, mode, target = group
        if target != ('earliest_state' if mode == 'earliest' else 'step_event'):
            continue
        for budget in ((None,) if method in NATIVE else (.01, .05)):
            identifier = f'{method}:{variant}:' + ('native' if budget is None else f'fpr{budget:g}')
            choice = None if budget is None else calibration(saved, group, budget)
            calibrated = budget is None or choice is not None
            if choice:
                thresholds[mode + ':' + identifier] = choice
            specs[mode][identifier] = dict(parts=[(group, choice)], operation='standalone',
                                           available_keys=group_keys[group] & expected_keys if calibrated else set(),
                                           calibrated=calibrated, method=method, variant=variant)
    for mode in MODES:
        native_specs = [(name, spec) for name, spec in specs[mode].items() if spec['method'] in NATIVE]
        gate_specs = [(name, spec) for name, spec in specs[mode].items() if spec['method'] in GATES]
        if mode == 'previous':
            gate_specs += [(name + '@history', spec) for name, spec in specs['history'].items() if spec['method'] in GATES]
        for native_name, native in native_specs:
            for gate_name, gate in gate_specs:
                for operation, symbol in (('and', '&'), ('or', '|')):
                    specs[mode][native_name + symbol + gate_name] = dict(parts=native['parts'] + gate['parts'], operation=operation,
                        available_keys=native['available_keys'] & gate['available_keys'],
                        calibrated=native['calibrated'] and gate['calibrated'], method='fusion', variant='explicit_components')
    active = {mode: {name for name, spec in values.items() if spec['available_keys']} for mode, values in specs.items()}
    common_keys = {mode: set.intersection(*(specs[mode][name]['available_keys'] for name in names)) if names else set()
                   for mode, names in active.items()}
    coverage = [dict(mode=mode, predictor=name, expected_pairs=len(expected_keys), available_pairs=len(spec['available_keys']),
                     missing_pairs=[list(key) for key in sorted(expected_keys - spec['available_keys'])],
                     common_pairs=len(common_keys[mode]) if name in active[mode] else 0,
                     status='available' if spec['available_keys'] else 'missing_predictions' if spec['calibrated'] else 'missing_calibration')
                for mode in MODES for name, spec in sorted(specs[mode].items())]
    out.mkdir(parents=True)
    (out / 'gt').mkdir()
    write_json(out / 'run.json', dict(status='running', source_run=str(root), selected_sequences=selected))
    per_pair, gt_records, artifact_hashes, gallery = [], [], {}, []
    for number, identifier in enumerate(selected, 1):
        sequence = sequences[identifier]
        building = sequence['building_id']
        print(f'GT {number}/{len(selected)}: {building}, {identifier}; наблюдений {len(sequence["frames"])}.', flush=True)
        frames = [loader(sequence, index) for index in range(len(sequence['frames']))]
        previous_for_history = {}
        shared_standalones = {name for name in specs['previous'].keys() & specs['history'].keys()
                              if specs['previous'][name]['operation'] == specs['history'][name]['operation'] == 'standalone'}
        for mode in MODES:
            for index in range(1, len(frames)):
                reference_index = 0 if mode == 'earliest' else index - 1
                previous, current = frames[reference_index], frames[index]
                geometric = previous['support'] & current['support']
                if boundary_tolerance:
                    radius = boundary_tolerance
                    if 'packed_halo' not in previous or 'packed_halo' not in current:
                        raise ValueError('Допуск границ требует сохранения контекста за краями кропа')
                    labels = derive_labels(previous['packed_halo'], current['packed_halo'],
                                           np.pad(geometric, radius, constant_values=False), category_count, radius)[radius:-radius, radius:-radius]
                else:
                    labels = derive_labels(previous['packed'], current['packed'], geometric, category_count)
                gt_id = hashlib.sha256(json.dumps([identifier, mode, index]).encode()).hexdigest()[:24]
                gt_path = out / 'gt' / (gt_id + '.npz')
                np.savez_compressed(gt_path, labels=labels, support=geometric, reference_bits=previous['packed'], source_bits=current['packed'])
                gt_row = dict(sequence_id=identifier, building_id=building, mode=mode, frame=index,
                              reference_index=reference_index, current_index=index,
                              reference_image_id=sequence['frames'][reference_index].get('image_id'),
                              source_image_id=sequence['frames'][index].get('image_id'),
                              reference_year=sequence['frames'][reference_index].get('year'), source_year=sequence['frames'][index].get('year'),
                              artifact_path=str(gt_path.relative_to(out)), artifact_sha256=sha256(gt_path),
                              valid_pixels=int((labels != 255).sum()), positive_pixels=int((labels == 1).sum()))
                gt_records.append(gt_row)
                if mode != 'history':
                    caption = f'{building} · {mode} · наблюдения {reference_index + 1} → {index + 1}; GT+: {gt_row["positive_pixels"]}/{gt_row["valid_pixels"]}'
                    gallery.append('<section><h2>' + html.escape(caption) + '</h2><div class="pair">' +
                                   _semantic_figure(previous['rgb'], previous['packed'], previous['support'], category_count, 'Классы предыдущего/первого наблюдения') +
                                   _semantic_figure(current['rgb'], current['packed'], current['support'], category_count, 'Классы текущего наблюдения') +
                                   _gt_figure(current['rgb'], labels, 'Изменение набора классов') + '</div></section>')
                predictions, loaded = {}, {}
                for name in sorted(active[mode]):
                    spec = specs[mode][name]
                    if (identifier, index) not in spec['available_keys']:
                        continue
                    masks, supports, source_ids = [], [], []
                    for group, choice in spec['parts']:
                        if group not in loaded:
                            record = indexed[(identifier, index, group)]
                            if record['building_id'] != building:
                                raise ValueError('Предсказание относится к другому фасаду')
                            arrays = load_map(overlay_roots.get(record['unit_id'], root), record, artifact_hashes)
                            pair_reference = frames[0] if group[2] == 'earliest' else frames[index - 1]
                            expected_support = pair_reference['support'] & current['support']
                            if (not np.array_equal(arrays['reference'], pair_reference['rgb']) or not np.array_equal(arrays['source'], current['rgb'])
                                    or arrays['support'].shape != expected_support.shape
                                    or (np.any(arrays['support'] & ~expected_support) if group[2] == 'history' else not np.array_equal(arrays['support'], expected_support))):
                                raise ValueError(f'RGB/support карты не совпадают с восстановленной геометрией: {record["unit_id"]}')
                            loaded[group] = (record, arrays)
                        record, arrays = loaded[group]
                        masks.append(arrays['native_prediction'] if group[0] in NATIVE else arrays['scores'] > np.float32(choice['threshold']))
                        supports.append(arrays['support'])
                        source_ids.append(record['unit_id'])
                    support = np.logical_and.reduce(supports)
                    prediction = (np.logical_or.reduce(masks) if spec['operation'] == 'or' else np.logical_and.reduce(masks)) & support
                    predictions[name] = (prediction, support, source_ids)
                shared = np.logical_and.reduce([predictions[name][1] for name in sorted(active[mode])]) if (identifier, index) in common_keys[mode] else None
                for name, (prediction, support, source_ids) in predictions.items():
                    for cohort, evaluation_support in [('available', support)] + ([('common', shared)] if shared is not None else []):
                        evaluation_labels = labels.copy()
                        evaluation_labels[~evaluation_support] = 255
                        values = evaluate_mask(prediction, evaluation_labels)
                        per_pair.append(dict(cohort=cohort, mode=mode, target='earliest_state' if mode == 'earliest' else 'step_event',
                                             predictor=name, sequence_id=identifier, building_id=building, frame=index,
                                             gt_artifact=gt_row['artifact_path'], source_unit_ids=source_ids,
                                             status='evaluated' if values['valid_pixels'] else 'no_annotation_support', **values))
                    if name in shared_standalones and mode == 'previous':
                        previous_for_history[(index, name)] = (prediction, support, source_ids)
                    if mode == 'history' and (index, name) in previous_for_history:
                        old_prediction, old_support, old_ids = previous_for_history.pop((index, name))
                        matched_labels = labels.copy()
                        matched_labels[~(old_support & support)] = 255
                        for compared_mode, mask, ids in [('previous', old_prediction, old_ids), ('history', prediction, source_ids)]:
                            values = evaluate_mask(mask, matched_labels)
                            per_pair.append(dict(cohort='matched_history', mode=compared_mode, target='step_event',
                                                 predictor=name, sequence_id=identifier, building_id=building, frame=index,
                                                 gt_artifact=gt_row['artifact_path'], source_unit_ids=ids,
                                                 status='evaluated' if values['valid_pixels'] else 'no_annotation_support', **values))
                print(f'  {mode} {index}: GT {gt_row["positive_pixels"]}/{gt_row["valid_pixels"]}; готовых вариантов {len(predictions)}.', flush=True)
    per_sequence, per_facade, aggregate = aggregate_metrics(per_pair)
    matched = {(row['mode'], row['predictor'], row['subset']): row for row in aggregate
               if row['cohort'] == 'matched_history' and row['aggregation'] == 'frame_then_facade_macro'}
    history_comparison = []
    for (mode, name, subset), old in sorted(matched.items()):
        if mode != 'previous':
            continue
        new = matched[('history', name, subset)]
        values = {f'{side}_{metric}': row[metric] for side, row in [('previous', old), ('history', new)] for metric in METRICS}
        values.update({f'delta_{metric}': new[metric] - old[metric] if old[metric] is not None and new[metric] is not None else None for metric in METRICS})
        history_comparison.append(dict(predictor=name, subset=subset, evaluated_pairs=old['evaluated_pairs'],
                                       facades=old['units'], valid_pixels=old['valid_pixels'], **values))
    for row in coverage:
        matching = [item for item in per_pair if item['mode'] == row['mode'] and item['predictor'] == row['predictor']]
        row['evaluated_available_pairs'] = sum(item['cohort'] == 'available' and item['valid_pixels'] > 0 for item in matching)
        row['evaluated_common_pairs'] = sum(item['cohort'] == 'common' and item['valid_pixels'] > 0 for item in matching)
        row['no_annotation_support_pairs'] = sum(item['cohort'] == 'available' and item['valid_pixels'] == 0 for item in matching)
    for name, rows in [('per_pair', per_pair), ('per_sequence', per_sequence), ('per_facade', per_facade), ('aggregate', aggregate), ('coverage', coverage), ('history_comparison', history_comparison)]:
        write_csv(out / (name + '.csv'), rows)
    target_definition = 'All-category COCO bitset change between indexed observations on known annotated support at both dates; previous/history use previous accepted observation, earliest uses first observation. Repair/replacement counts as change.'
    notes = ['GT: изменение исходной COCO-разметки всех категорий; неизвестные/iscrowd исключены.',
             'Пороги заморожены на синтетическом VAL; 1%/5% не являются обещанным FPR на реальных данных.',
             'available: собственные доступные пары/пиксели; common: одинаковые пары и пересечение поддержки всех доступных вариантов внутри режима.',
             'Общие когорты разных режимов могут отличаться. Прямое сравнение previous/history выделено в matched_history: одинаковые пары и пиксели для каждого метода. Добавки @history сравниваются на общей когорте previous.',
             'Варианты без единой карты перечислены в coverage и не входят в common. Отказ не заменяется пустой маской.',
             'Основная frame_then_facade_macro: метрика каждого кадра, среднее внутри фасада, затем равный вес фасадов; subset=positive — основной срез P/R/F1/IoU.',
             'facade_pooled_macro: суммы TP/FP/FN/TN внутри фасада, затем равный вес фасадов. sequence_pooled_macro: суммы внутри кроп-истории, затем равный вес историй.',
             'subset=positive: только пары с GT+; no_event: пары с известными GT-пикселями и без GT+, FPR только в наблюдаемой области.',
             'Счётчики относятся к пиксельным наблюдениям; перекрывающиеся кропы/даты не являются независимой физической площадью.',
             'Нулевой знаменатель: null; при GT+ и пустом предсказании precision=0. Пары без известных пикселей исключены из средних.',
             'Семантическая разметка не устраняет ошибки совмещения/неполноту аннотаций. TEST и модели не используются.']
    summary = dict(status='completed', scope='annotation_derived_state_change', source_run=str(root), selected_sequences=selected,
                   gt_coverage={mode: dict(pairs=sum(row['mode'] == mode for row in gt_records),
                                annotated_pairs=sum(row['mode'] == mode and row['valid_pixels'] > 0 for row in gt_records),
                                positive_pairs=sum(row['mode'] == mode and row['positive_pixels'] > 0 for row in gt_records),
                                positive_facades=len({row['building_id'] for row in gt_records if row['mode'] == mode and row['positive_pixels'] > 0})) for mode in MODES},
                   boundary_tolerance=boundary_tolerance, thresholds=thresholds, common_cohorts={mode: dict(predictors=sorted(active[mode]), pairs=[list(key) for key in sorted(common_keys[mode])]) for mode in MODES},
                   per_pair=per_pair, aggregate=aggregate, coverage=coverage, gt_records=gt_records, history_comparison=history_comparison,
                   overlay_failures=overlay_failures, notes=notes)
    provenance = dict(source_run=str(root), snapshot=str(snapshot), source_metadata_sha256={str(path): sha256(path) for path in metadata_paths},
                      script_sha256=sha256(Path(__file__)), selected_sequences=selected, selection='all_real_val' if all_real_val else 'existing_review_selection',
                      target_definition=target_definition, boundary_tolerance=boundary_tolerance, source_gt=gt_provenance,
                      categories=gt_provenance['categories'], thresholds=thresholds, input_artifact_sha256=artifact_hashes,
                      output_gt_sha256={row['artifact_path']: row['artifact_sha256'] for row in gt_records}, no_model_inference=True)
    write_json(out / 'provenance.json', provenance)
    write_json(out / 'summary.json', summary)
    lines = notes + ['', 'Основные таблицы: общая когорта внутри режима; P/R/F1/IoU на положительных GT-парах. Все доли в %.']
    lines[0:0] = [f"GT {mode}: размечено пар {counts['annotated_pairs']}/{counts['pairs']}, с GT+ {counts['positive_pairs']}, фасадов с GT+ {counts['positive_facades']}."
                  for mode, counts in summary['gt_coverage'].items()]
    primary = {(row['mode'], row['predictor'], row['subset']): row for row in aggregate
               if row['aggregation'] == 'frame_then_facade_macro' and row['cohort'] == 'common'}
    percent = lambda value: '—' if value is None else f'{100 * value:.2f}'
    coverage_lookup = {(row['mode'], row['predictor']): row for row in coverage}
    for mode in MODES:
        lines += ['', f'## {mode.upper()}', '', '| Метод / сочетание | P | R | F1 | IoU | FPR все | FPR без события | Пар +/все | Фасадов | TP/FP/FN/TN | Маска, % GT-пикселей | Доступно / ожидается |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|']
        for name in sorted(specs[mode]):
            positive = primary.get((mode, name, 'positive'), {})
            all_frames = primary.get((mode, name, 'all'), {})
            no_event = primary.get((mode, name, 'no_event'), {})
            count = all_frames.get('valid_pixels', 0)
            marked = (all_frames.get('tp', 0) + all_frames.get('fp', 0)) / count if count else None
            cov = coverage_lookup[(mode, name)]
            cells = [name.replace('|', '\\|'), *[percent(positive.get(key)) for key in ('precision', 'recall', 'f1', 'iou')],
                     percent(all_frames.get('fpr')), percent(no_event.get('fpr')),
                     f"{positive.get('evaluated_pairs', 0)}/{all_frames.get('evaluated_pairs', 0)}", str(all_frames.get('units', 0)),
                     '/'.join(str(all_frames[key]) for key in ('tp', 'fp', 'fn', 'tn')) if all_frames else '—',
                     percent(marked), f"{cov['available_pairs']}/{cov['expected_pairs']}"]
            lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '## PREVIOUS → HISTORY: одинаковые пары и пиксели', '',
              '| Метод | Пар +/все | P до→после | R до→после | F1 до→после | IoU до→после | ΔF1, п.п. | FPR все до→после |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in history_comparison:
        if row['subset'] != 'positive':
            continue
        all_frames = next(item for item in history_comparison if item['predictor'] == row['predictor'] and item['subset'] == 'all')
        cells = [row['predictor'], f"{row['evaluated_pairs']}/{all_frames['evaluated_pairs']}",
                 *[percent(row['previous_' + metric]) + '→' + percent(row['history_' + metric]) for metric in ('precision', 'recall', 'f1', 'iou')],
                 percent(row['delta_f1']), percent(all_frames['previous_fpr']) + '→' + percent(all_frames['history_fpr'])]
        lines.append('| ' + ' | '.join(cells) + ' |')
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (out / 'tables.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    legend = ' · '.join(f'<span style="color:rgb{PALETTE[channel]}">{html.escape(str(category["id"]) + ": " + category["name"])}</span>' for channel, category in enumerate(gt_provenance['categories']))
    (out / 'gt_gallery.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>GT изменений COCO</title><style>body{font:15px system-ui;margin:24px}.pair{display:flex;flex-wrap:wrap}figure{max-width:384px;margin:10px}img{width:100%}</style><h1>GT: изменения исходной COCO-разметки</h1><p>Жёлтый в панели изменений — изменение набора классов; серый — неизвестная/исключённая область. Ремонт тоже является изменением. В масках классов пересечения цветов смешиваются.</p><p>' + legend + '</p><main>' + ''.join(gallery) + '</main></html>', encoding='utf-8')
    write_json(out / 'run.json', dict(status='completed', source_run=str(root), selected_sequences=selected, no_model_inference=True))
    print(f'Готово: {out / "summary.txt"}; все счётчики: {out / "per_pair.csv"}.', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--review', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--geoscd-run')
    parser.add_argument('--repo')
    parser.add_argument('--all-real-val', action='store_true')
    parser.add_argument('--boundary-tolerance', type=int, choices=(0, 2), default=0)
    args = parser.parse_args()
    run(args.run, args.snapshot, args.review, args.out, args.geoscd_run, args.repo,
        args.all_real_val, args.boundary_tolerance)


if __name__ == '__main__':
    main()
