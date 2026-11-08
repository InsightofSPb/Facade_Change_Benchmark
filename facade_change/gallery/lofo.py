#!/usr/bin/env python3
"""Галерея старых методов + AnyChange из сохранённого real-VAL LOFO.

Требования: Python >= 3.9, numpy, Pillow. GPU не используется.
Никаких обучения, inference или новой калибровки. Читаются готовые карты,
сохранённые JSON-рецепты порогов и исходные подготовленные RGB/аннотации.
Каждая маска сверяется с сохранёнными TP/FP/FN/TN и support_sha256.

Пример из корня установленного checkout:
  conda run --no-capture-output -n scd_bench python -m facade_change workflow gallery lofo \
    --run runs/2026-10-18-night-run-001 \
    --calibration-run runs/2026-10-18-night-real-val-lofo-001 \
    --out runs/2026-10-20-old-methods-gallery-001

Выход: index.html, pairs/*.png, panels/*/*.png, per_pair_verification.csv,
provenance.json. Только VAL: именно для него сохранены старые LOFO-пороги.
AnyChange использует исходную native_prediction, а не новый порог.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import html
import json
import csv
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# The uploaded standalone gallery embedded identical copies of these functions.
# Reuse their owners and record their current source hashes in provenance.
from scripts.real_eval import night_pair_calibration, real_gt_metrics
from scripts.real_eval.night_pair_calibration import predict_choice
from scripts.real_eval.real_gt_metrics import derive_labels, evaluate_mask

REFERENCE_SOURCE_SHA256 = {'night_pair_calibration.py': '7bd0caaf6c87a1e2139193931131ff0b0dd0dc5b0279734d16a32b6bca31c442', 'real_gt_metrics.py': '1fb5ad5fd72632b70370593a768f742b484a0a9f9b38e0924c5baff8922bd5dc'}

SPECS = [
    ('anychange', 'AnyChange', ('anychange:previous',)),
    ('semantic', 'Semantic DINO', ('semantic_dino:previous',)),
    ('lpips_semantic', 'LPIPS AND Semantic DINO', ('lpips:previous', 'semantic_dino:previous')),
    ('semantic_history', 'Semantic DINO HISTORY', ('semantic_dino:history',)),
    ('lpips_semantic_history', 'LPIPS AND Semantic DINO HISTORY', ('lpips:previous', 'semantic_dino:history')),
]
TERMINAL = {'completed', 'completed_with_failures', 'budget_exhausted', 'interrupted', 'failed_validation'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def safe_path(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('Ожидался относительный путь артефакта')
    result = (root / relative).resolve()
    if root not in result.parents:
        raise ValueError('Путь выходит из исходного каталога: ' + relative)
    return result


class Inputs:
    def __init__(self, args):
        self.root = args.run.expanduser().resolve()
        self.cal = args.calibration_run.expanduser().resolve()
        self.hashes = {}
        self.run = read_json(self.root / 'run.json')
        self.prov = read_json(self.cal / 'provenance.json')
        self.cal_run = read_json(self.cal / 'run.json')
        if self.run.get('status') not in TERMINAL:
            raise ValueError('Исходный прогон не имеет конечного статуса')
        if self.prov.get('kind') != 'night_real_val_lofo_completed' or self.prov.get('status') != 'completed':
            raise ValueError('--calibration-run должен указывать на законченный старый night real VAL LOFO')
        self.source = self.prov['source']
        contract = self.cal_run['contract']
        if (contract.get('kind') != 'night_real_val_lofo'
                or contract.get('protocol') != 'negative_only_equal_facade_final_FPR_strict_rank_v1'
                or contract.get('source') != self.source
                or digest(contract) != self.cal_run['contract_sha256']
                or self.prov['contract_sha256'] != self.cal_run['contract_sha256']):
            raise ValueError('Контракт LOFO не совпадает с сохранённым provenance')
        self.check(self.root / 'run.json', self.source['run_sha256'])
        for path in (self.cal / 'run.json', self.cal / 'provenance.json'):
            self.check(path)
        prepared_path = args.prepared or Path(self.run['prepared_path'])
        self.prepared = prepared_path.expanduser().resolve()
        if self.prepared.is_file():
            self.prepared = self.prepared.parent
        manifest_path = self.prepared / 'manifest.json'
        self.check(manifest_path, self.run['manifest_sha256'])
        if self.source['manifest_sha256'] != self.run['manifest_sha256']:
            raise ValueError('Калибровка и inference относятся к разным манифестам')
        self.manifest = read_json(manifest_path)
        if self.manifest.get('kind') != 'real_temporal_prepared':
            raise ValueError('Ожидался манифест подготовленных реальных хронологий')
        self.sequences = {s['sequence_id']: s for s in self.manifest['sequences']}
        if len(self.sequences) != len(self.manifest['sequences']):
            raise ValueError('Повторяющийся sequence_id')
        buildings = {}
        for s in self.sequences.values():
            if s['building_id'] in buildings and buildings[s['building_id']] != s['split']:
                raise ValueError('Один фасад попал в разные split')
            buildings[s['building_id']] = s['split']
            years = [f['year'] for f in s['frames']]
            if years != sorted(set(years)) or [f['frame'] for f in s['frames']] != list(range(len(years))):
                raise ValueError('Нарушен порядок кадров')
        self.families = {}
        for key, title, channels in SPECS:
            hits = [r for r in self.prov['families'] if set(r['channels']) == set(channels)]
            if len(hits) != 1:
                raise ValueError('Нет однозначного сохранённого семейства: ' + title)
            receipt = hits[0]
            path = safe_path(self.cal, receipt['path'])
            self.check(path, receipt['sha256'])
            data = read_json(path)
            if data['channels'] != receipt['channels'] or data['family'] != '__'.join(data['channels']):
                raise ValueError('Изменились каналы сохранённого семейства')
            self.families[key] = data

    def check(self, path, expected=None):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError('Не найден сохранённый файл: ' + str(path))
        value = self.hashes.get(str(path))
        if value is None:
            value = sha256(path)
            self.hashes[str(path)] = value
        if expected is not None and value != expected:
            raise ValueError('Не совпал SHA-256: ' + str(path))
        return value

    @lru_cache(maxsize=4)
    def frame(self, sequence_id, frame):
        record = self.sequences[sequence_id]['frames'][frame]
        expected = record['artifact_sha256']
        if self.source['frame_sha256'].get(record['artifact_path']) != expected:
            raise ValueError('Кадр не входит в снимок исходных данных LOFO')
        path = safe_path(self.prepared, record['artifact_path'])
        self.check(path, expected)
        with np.load(path, allow_pickle=False) as z:
            a = {k: z[k].copy() for k in ('rgb', 'support', 'packed')}
        support = a['support']
        if (support.dtype != bool or support.ndim != 2
                or a['rgb'].dtype != np.uint8 or a['rgb'].shape != (*support.shape, 3)
                or a['packed'].dtype != np.uint16 or a['packed'].shape != support.shape):
            raise ValueError('Неверные dtype/решётка подготовленного кадра')
        return a

    @lru_cache(maxsize=8)
    def map(self, channel, sequence_id, frame):
        method, mode = channel.split(':')
        identifier = digest([method, mode, sequence_id, frame])[:32]
        relative = 'units/' + identifier + '.json'
        expected = self.source['unit_sha256'].get(relative)
        if not expected:
            raise ValueError('Нет исходной записи карты в provenance: ' + relative)
        path = safe_path(self.root, relative)
        self.check(path, expected)
        record = read_json(path)
        sequence = self.sequences[sequence_id]
        identity = dict(method=method, mode=mode, unit_id=identifier, sequence_id=sequence_id,
                        frame=frame, split='val', domain='real', status='completed',
                        building_id=sequence['building_id'])
        if any(record.get(k) != v for k, v in identity.items()):
            raise ValueError('Метаданные карты не совпали с выбранной VAL-парой')
        if self.source['artifact_sha256'].get(record['artifact_path']) != record['artifact_sha256']:
            raise ValueError('Карта не входит в снимок LOFO')
        path = safe_path(self.root, record['artifact_path'])
        self.check(path, record['artifact_sha256'])
        before, after = self.frame(sequence_id, frame - 1), self.frame(sequence_id, frame)
        pair_support = before['support'] & after['support']
        with np.load(path, allow_pickle=False) as z:
            scores, support = z['scores'].copy(), z['support'].copy()
            if (support.dtype != bool or support.shape != pair_support.shape
                    or np.any(support & ~pair_support)
                    or not np.array_equal(z['reference'], before['rgb'])
                    or not np.array_equal(z['source'], after['rgb'])
                    or scores.dtype != np.float32 or scores.shape != support.shape
                    or not np.isfinite(scores[support]).all() or not np.isnan(scores[~support]).all()):
                raise ValueError('Карта не соответствует исходным RGB/поддержке: ' + str(path))
            values = z['native_prediction'].copy() if method == 'anychange' else scores
        if values.shape != support.shape or (method == 'anychange' and values.dtype != bool):
            raise ValueError('Неверная native-маска AnyChange')
        return values, support

    def prediction(self, key, sequence_id, frame, point, labels):
        data = self.families[key]
        role = 'single' if len(data['channels']) == 1 else 'and'
        building = self.sequences[sequence_id]['building_id']
        choices = [r for r in data['calibration_audit'] if r['role'] == role
                   and abs(float(r['point']) - point) < 1e-12 and r['heldout_building'] == building]
        saved = [r for r in data['per_pair'] if r['role'] == role
                 and abs(float(r['point']) - point) < 1e-12
                 and r['sequence_id'] == sequence_id and r['frame'] == frame]
        if len(choices) != 1 or len(saved) != 1:
            raise ValueError(f'Нет однозначного сохранённого решения: {key}, {building}, frame={frame}')
        choice, saved = choices[0], saved[0]
        if (building in choice['train_buildings'] or choice['calibration_split'] != 'val'
                or choice['threshold_operator'] != '>'):
            raise ValueError('Нарушен сохранённый протокол LOFO')
        arrays = [self.map(channel, sequence_id, frame) for channel in data['channels']]
        valid = labels != 255
        for _, support in arrays:
            valid &= support
        pixels = np.flatnonzero(valid)
        support_hash = hashlib.sha256(np.asarray(labels.shape, dtype='<i8').tobytes()
                                      + np.asarray(pixels, dtype='<i8').tobytes()).hexdigest()
        if support_hash != saved['support_sha256']:
            raise ValueError('Оцениваемые пиксели отличаются от прежней таблицы')
        row = {name: values.ravel()[pixels]
               for name, (values, _) in zip(('left', 'right'), arrays)}
        prediction = predict_choice(row, choice)
        counts = evaluate_mask(prediction, labels.ravel()[pixels])
        for field in ('tp', 'fp', 'fn', 'tn', 'valid_pixels', 'positive_pixels', 'negative_pixels'):
            if counts[field] != saved[field]:
                raise ValueError(f'Маска отличается от сохранённого результата: {key}, {field}')
        full = np.zeros(labels.shape, bool)
        full.ravel()[pixels] = prediction
        return full, valid, dict(method=key, predictor=saved['predictor'], family=data['family'],
                                sequence_id=sequence_id, frame=frame, building_id=building,
                                point=point, support_sha256=support_hash, parity='passed', **counts)


def overlay(rgb, labels, valid, prediction=None):
    value = rgb.copy()
    # Unknown/out-of-evaluation pixels are pale gray, never false positives.
    gray = rgb.mean(axis=2, keepdims=True)
    value[~valid] = np.repeat((0.30 * gray + 0.70 * 238).astype(np.uint8), 3, axis=2)[~valid]
    if prediction is None:
        masks = [(valid & (labels == 1), (239, 170, 37))]
    else:
        masks = [(valid & prediction & (labels == 1), (40, 192, 94)),
                 (valid & prediction & (labels == 0), (230, 59, 64)),
                 (valid & ~prediction & (labels == 1), (49, 117, 232))]
    for mask, color in masks:
        value[mask] = np.rint(0.32 * rgb[mask] + 0.68 * np.asarray(color)).astype(np.uint8)
    return Image.fromarray(value)


def font(size, explicit=None):
    candidates = [explicit] if explicit else [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
        '/mnt/c/Windows/Fonts/arial.ttf', 'DejaVuSans.ttf']
    for name in candidates:
        try:
            return ImageFont.truetype(str(name), size)
        except OSError:
            pass
    raise ValueError('Нужен шрифт с кириллицей: передайте --font /path/font.ttf')


def fit_font(text, width, size, explicit=None):
    while size > 10:
        f = font(size, explicit)
        if f.getlength(text) <= width:
            return f
        size -= 1
    return font(size, explicit)


def compose(panels, title, point, tile, font_path=None):
    gap, margin, heading, caption, footer = 18, 24, 66, 54, 72
    width = 4 * tile + 3 * gap + 2 * margin
    height = heading + 2 * (caption + tile) + gap + footer
    canvas = Image.new('RGB', (width, height), '#ffffff')
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), title, fill='#20334d', font=fit_font(title, width - 2 * margin, 25, font_path))
    for i, (name, picture) in enumerate(panels):
        row, col = divmod(i, 4)
        x, y = margin + col * (tile + gap), heading + row * (tile + caption + gap)
        lines = name.split('\n')
        for j, line in enumerate(lines):
            f = fit_font(line, tile, 19, font_path)
            draw.text((x, y + j * 23), line, fill='#20334d', font=f)
        picture = picture.copy()
        picture.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        canvas.paste(picture, (x + (tile-picture.width)//2, y + caption + (tile-picture.height)//2))
    y = height - footer + 13
    lines = [f'VAL · LOFO · цель калибровки FPR {100*point:g}% · сохранённые решения; AnyChange — исходная маска.',
             'GT — изменения аннотаций. Зелёный: TP; красный: FP; синий: FN. Светло-серый: вне оценки.']
    for j, line in enumerate(lines):
        draw.text((margin, y + 24*j), line, fill='#5c6877', font=fit_font(line, width-2*margin, 17, font_path))
    return canvas


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--calibration-run', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, help='Если подготовленные кадры перенесены; хеши должны совпасть')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--point', type=float, default=.05, help='Сохранённая рабочая точка; по умолчанию 0.05')
    parser.add_argument('--facade', nargs='+', help='Точные building_id; по умолчанию все VAL-фасады')
    parser.add_argument('--tile', type=int, default=512)
    parser.add_argument('--font')
    args = parser.parse_args(argv)
    if not np.isfinite(args.point) or not 0 < args.point < 1 or args.tile < 128:
        parser.error('Нужны 0 < point < 1 и tile >= 128')
    source = Inputs(args)
    selected = [s for s in source.sequences.values() if s['split'] == 'val'
                and (args.facade is None or s['building_id'] in args.facade)]
    if not selected or (args.facade and set(args.facade) - {s['building_id'] for s in selected}):
        parser.error('Указанный фасад не найден в VAL данного прогона')
    out = args.out.expanduser().resolve()
    for protected in (source.root, source.prepared, source.cal):
        if out == protected or out in protected.parents or protected in out.parents:
            parser.error('--out должен быть отдельным каталогом вне исходных данных')
    if out.exists():
        parser.error('--out уже существует; укажите новый каталог')
    font(16, args.font)
    out.mkdir(parents=True)
    (out / 'pairs').mkdir()
    write_json(out / 'run.json', dict(status='running', no_model_inference=True, no_recalibration=True))
    rows, cards, recipes, support_notes = [], [], [], []
    for sequence in sorted(selected, key=lambda s: (s['building_id'], s['sequence_id'])):
        sid = sequence['sequence_id']
        for index in range(1, len(sequence['frames'])):
            before, after = source.frame(sid, index-1), source.frame(sid, index)
            labels = derive_labels(before['packed'], after['packed'], before['support'] & after['support'],
                                   len(source.manifest['categories']), source.manifest.get('boundary_tolerance', 0))
            old_year, new_year = sequence['frames'][index-1]['year'], sequence['frames'][index]['year']
            title = f"{sequence['building_id']} · {old_year} → {new_year}"
            slug = re.sub(r'[^A-Za-z0-9_-]', '_', sequence['building_id']) + f'_{old_year}_{new_year}_' + digest(sid)[:8]
            folder = out / 'panels' / slug
            folder.mkdir(parents=True)
            panels = [(f'Предыдущее наблюдение · {old_year}', Image.fromarray(before['rgb'])),
                      (f'Текущее наблюдение · {new_year}', Image.fromarray(after['rgb'])),
                      ('GT: изменения аннотаций', overlay(after['rgb'], labels, labels != 255))]
            panel_names = ['before', 'after', 'gt']
            supports = []
            for key, name, _ in SPECS:
                prediction, valid, record = source.prediction(key, sid, index, args.point, labels)
                rows.append(record)
                supports.append(valid)
                if key == 'lpips_semantic_history':
                    name = 'LPIPS AND\nSemantic DINO HISTORY'
                panels.append((name, overlay(after['rgb'], labels, valid, prediction)))
                panel_names.append(key)
            if any(not np.array_equal(supports[0], value) for value in supports[1:]):
                support_notes.append(dict(sequence_id=sid, frame=index, note='Method-specific saved evaluation supports differ; gray pixels show excluded areas.'))
            for name, (_, picture) in zip(panel_names, panels):
                picture.save(folder / (name + '.png'))
            picture = compose(panels, title, args.point, args.tile, args.font)
            target = out / 'pairs' / (slug + '.png')
            picture.save(target)
            cards.append((title, target.relative_to(out).as_posix()))
            print(f'Готово: {title}; совпали маски/счётчики 5 методов', flush=True)
    with (out / 'per_pair_verification.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    page = ['<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
            '<title>Старые методы + AnyChange</title><style>body{font:16px system-ui;margin:24px;color:#20334d;background:#f5f7fa}article{background:white;padding:16px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}p{max-width:1100px}</style>',
            '<h1>Старые методы + AnyChange</h1>',
            '<p>Те же сохранённые real-VAL LOFO-решения. Без новой калибровки и инференса. '
            'AnyChange — исходная бинарная маска. HISTORY — прежняя замороженная память.</p>',
            '<p>GT означает изменение аннотированных классов. Светло-серые пиксели не оценивались. '
            'Каждая маска проверена по поддержке и TP/FP/FN/TN прежнего результата. Нажмите картинку для полного размера.</p>']
    for title, path in cards:
        page.append(f'<article><h2>{html.escape(title)}</h2><a href="{path}"><img src="{path}" loading="lazy" alt="{html.escape(title)}"></a></article>')
    page.append('</html>')
    (out / 'index.html').write_text('\n'.join(page), encoding='utf-8')
    for path, expected in source.hashes.items():
        if sha256(path) != expected:
            raise ValueError('Исходный файл изменился во время сборки: ' + path)
    write_json(out / 'provenance.json', dict(status='completed', source_run=str(source.root),
        calibration_run=str(source.cal), prepared_path=str(source.prepared), point=args.point,
        split='val', source_files_sha256=source.hashes, script_sha256=sha256(__file__),
        reference_source_sha256=REFERENCE_SOURCE_SHA256,
        dependency_source_sha256={Path(module.__file__).name: sha256(module.__file__)
                                  for module in (night_pair_calibration, real_gt_metrics)},
        no_model_inference=True, no_recalibration=True, history_memory_unchanged=True,
        native_anychange_unchanged=True, panels=len(cards), verified_method_masks=len(rows),
        facades=sorted({s['building_id'] for s in selected}), support_notes=support_notes,
        prediction_scope='Saved method-specific known-GT support; no coloring outside evaluation',
        output_sha256={p.relative_to(out).as_posix(): sha256(p) for p in out.rglob('*') if p.is_file() and p.name != 'run.json'}))
    write_json(out / 'run.json', dict(status='completed', pairs=len(cards), verified_method_masks=len(rows),
                                    no_model_inference=True, no_recalibration=True))
    print(f"\nГалерея: {out / 'index.html'}\nPNG для слайдов: {out / 'pairs'}", flush=True)


def main(argv=None):
    try:
        _main(argv)
    except (ValueError, OSError, KeyError) as error:
        print('Ошибка: ' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
