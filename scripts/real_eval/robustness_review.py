"""Rebuild a GT review from immutable prepared crops, without models or alignment."""
from __future__ import annotations

import copy
import csv
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .robustness_common import atomic_json, digest, load_pair, load_prepared, safe_path, sha256
from .robustness_review_stats import pair_statistics, summarize


def _transition_masks(out, row, arrays, categories):
    labels = arrays['labels']
    known = arrays['support'] & (labels != 255)
    yy, xx = np.indices(labels.shape)
    hatch = (labels == 255) & ((xx + yy) % 12 < 4)
    prefix = digest(row['pair_id'])[:24]
    paths = {}
    for channel in [None, *range(len(categories))]:
        rgba = np.zeros((*labels.shape, 4), np.uint8)
        rgba[hatch] = [135, 135, 135, 175]
        if channel is None:
            rgba[known & (labels == 1)] = [255, 40, 40, 175]
            key = 'all'
        else:
            before = (arrays['reference_bits'] & (1 << channel)) != 0
            after = (arrays['source_bits'] & (1 << channel)) != 0
            rgba[known & ~before & after] = [25, 200, 95, 190]
            rgba[known & before & ~after] = [220, 45, 160, 190]
            key = str(channel)
        relative = f'masks/{prefix}-{key}.png'
        Image.fromarray(rgba).save(out / relative)
        paths[key] = relative
    return paths


def _csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _tables(out, rows, summary):
    pair_rows, class_rows = [], []
    for row in rows:
        if row.get('status') != 'ready':
            continue
        identity = {key: row.get(key) for key in
                    ('pair_id', 'building_id', 'sequence_id', 'frame', 'reference_year', 'source_year')}
        pair_rows.append({**identity, **{key: value for key, value in row['stats'].items() if key != 'classes'}})
        class_rows.extend({**identity, **entry} for entry in row['stats']['classes'])
    _csv(out / 'pair_statistics.csv', pair_rows, list(pair_rows[0]) if pair_rows else ['pair_id'])
    _csv(out / 'class_statistics.csv', class_rows, list(class_rows[0]) if class_rows else ['pair_id', 'class_id'])
    _csv(out / 'threshold_statistics.csv', summary['thresholds'], list(summary['thresholds'][0]))


def review(prepared_path, out_path):
    """Read cached NPZs once; export a new gallery bound to the original manifest."""
    root, manifest = load_prepared(prepared_path)
    out = Path(out_path).expanduser().resolve()
    protected = [root, *[Path(manifest[key]).expanduser().resolve() for key in ('source_run', 'snapshot')
                        if manifest.get(key)]]
    if out.exists() or any(out == path or path in out.parents or out in path.parents for path in protected):
        raise FileExistsError('Нужен новый отдельный каталог просмотра вне исходных запусков и подготовки')
    from .robustness_prepare import _validate_category_mapping
    from .robustness_review_gallery import render_gallery
    categories = manifest['categories']
    _validate_category_mapping(categories, manifest['category_to_channel'])
    manifest_sha = sha256(root / 'manifest.json')
    out.mkdir(parents=True)
    (out / 'masks').mkdir()
    state = dict(schema_version=1, kind='real_robustness_review', status='running',
                 prepared=str(root), manifest_sha256=manifest_sha, out=str(out),
                 started_utc=datetime.now(timezone.utc).isoformat(), completed_pairs=0,
                 gpu_started=False,
                 code_sha256={name: sha256(Path(__file__).with_name(name)) for name in
                              ('robustness_review.py', 'robustness_review_stats.py',
                               'robustness_review_gallery.py', 'robustness_common.py',
                               'robustness_prepare.py')})
    atomic_json(out / 'run.json', state)
    rows = []
    print(f'CPU: статистика и просмотр {len(manifest["pairs"])} готовых записей; модели не запускаются.', flush=True)
    try:
        for index, original in enumerate(manifest['pairs'], 1):
            row = copy.deepcopy(original)
            if row.get('status') == 'ready':
                arrays = load_pair(root, row)
                row['stats'] = stats = pair_statistics(arrays, categories)
                row.update(positive_pixels=stats['changed_pixels'], negative_pixels=stats['unchanged_pixels'],
                           ignored_pixels=stats['total_pixels'] - stats['known_pixels'],
                           changed_channels=[item['channel'] for item in stats['classes'] if item['changed_pixels']])
                previews = {}
                for key, relative in row['previews'].items():
                    path = safe_path(root, relative)
                    if not path.is_file():
                        raise FileNotFoundError(f'Нет подготовленного изображения: {path}')
                    previews[key] = Path(os.path.relpath(path, out)).as_posix()
                row['previews'] = previews
                row['transition_masks'] = _transition_masks(out, row, arrays, categories)
                state['completed_pairs'] += 1
            rows.append(row)
            if index % 10 == 0 or index == len(manifest['pairs']):
                print(f'[{index}/{len(manifest["pairs"])}] прочитано; пар с GT-статистикой: {state["completed_pairs"]}', flush=True)
                atomic_json(out / 'run.json', state)
        if sha256(root / 'manifest.json') != manifest_sha:
            raise ValueError('Исходный манифест изменился во время чтения')
        summary = summarize(rows, categories)
        payload = dict(schema_version=1, kind='real_robustness_review_statistics',
                       prepared=str(root), manifest_sha256=manifest_sha,
                       boundary_tolerance=manifest.get('boundary_tolerance'), categories=categories,
                       summary=summary, pairs=rows)
        atomic_json(out / 'statistics.json', payload)
        _tables(out, rows, summary)
        render_gallery(out, rows, categories, summary, manifest_sha)
        state.update(status='completed', finished_utc=datetime.now(timezone.utc).isoformat(), summary=summary)
        atomic_json(out / 'run.json', state)
        print('Порог GT | от известной области: пары/фасады | от всего кропа: пары/фасады', flush=True)
        global_counts = {(item['denominator'], item['threshold_percent']): item for item in summary['thresholds']
                         if item['channel'] is None}
        for threshold in (0, 1, 5, 10, 15, 20, 30, 50):
            known, crop = (global_counts[(kind, threshold)] for kind in ('known', 'crop'))
            print(f'{threshold:>3}% | {known["pairs"]}/{known["facades"]} | {crop["pairs"]}/{crop["facades"]}', flush=True)
        print(f'Галерея: {out / "gallery.html"}', flush=True)
        return payload
    except Exception as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                     finished_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(out / 'run.json', state)
        raise
