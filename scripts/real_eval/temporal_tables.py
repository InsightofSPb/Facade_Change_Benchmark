#!/usr/bin/env python3
"""Три таблицы из quick_units.csv, без моделей и сторонних библиотек.

python temporal_tables.py INPUT.csv --out NEW_DIRECTORY
Метрики усредняются сначала внутри фасада, затем между фасадами.
CSV: доли 0..1; summary.txt: проценты. Исходный CSV не изменяется.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

KEY = ('domain', 'split', 'mode', 'target', 'method', 'variant', 'decision', 'fpr_target')
COUNTS = ('tp', 'fp', 'fn', 'tn', 'evaluated_pixel_count', 'ignored_pixel_count')
METRICS = ('precision', 'recall', 'f1', 'iou', 'h0_pixel_fpr')


def ratio(a, b):
    return a / b if b else None


def macro(rows, metric):
    buildings = defaultdict(list)
    for row in rows:
        if row.get(metric) is not None:
            buildings[row['building_id']].append(row[metric])
    return mean(mean(v) for v in buildings.values()) if buildings else None


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    raw = args.csv.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    required = {*KEY, *COUNTS, 'unit_id', 'building_id', 'sequence_id', 'frame', 'arm', 'hypothesis'}
    if not required <= set(reader.fieldnames or []):
        parser.error('Не хватает колонок: ' + ', '.join(sorted(required - set(reader.fieldnames or []))))
    rows = list(reader)
    groups, coverage, seen = defaultdict(list), defaultdict(list), set()
    for row in rows:
        if row['mode'] not in {'previous', 'earliest', 'history'}:
            raise ValueError('Неизвестный режим: ' + row['mode'])
        key = tuple(row[k] for k in KEY)
        identity = (*key, row['sequence_id'], row['frame'])
        if identity in seen:
            raise ValueError('Дубликат: ' + repr(identity))
        seen.add(identity)
        coverage[key].append(row)
        # Реальным маскам необходим явный признак проверенной временной разметки.
        verified = row['domain'] == 'synthetic' or row.get('label_verified', '').lower() == 'true'
        if not verified:
            continue
        if not row['building_id'] or row['hypothesis'] not in {'H0', 'H1'}:
            raise ValueError('Нет фасада или H0/H1: ' + row['unit_id'])
        row.update({name: int(row[name]) for name in COUNTS})
        tp, fp, fn, tn, evaluated, ignored = (row[k] for k in COUNTS)
        if min(tp, fp, fn, tn, evaluated, ignored) < 0 or tp + fp + fn + tn != evaluated:
            raise ValueError('Некорректные счётчики: ' + row['unit_id'])
        if row['hypothesis'] == 'H0' and tp + fn:
            raise ValueError('H0 содержит положительную разметку: ' + row['unit_id'])
        positive = row['hypothesis'] == 'H1' and tp + fn > 0
        metrics = dict(precision=(ratio(tp, tp + fp) or 0.) if positive else None,
                       recall=ratio(tp, tp + fn) if positive else None,
                       f1=ratio(2 * tp, 2 * tp + fp + fn) if positive else None,
                       iou=ratio(tp, tp + fp + fn) if positive else None,
                       h0_pixel_fpr=ratio(fp, fp + tn) if row['hypothesis'] == 'H0' else None)
        for name, value in metrics.items():
            saved = row.get(name, '')
            if saved != '' and (value is None or not math.isclose(float(saved), value, abs_tol=1e-10)):
                raise ValueError('Метрика расходится со счётчиками: ' + row['unit_id'] + '/' + name)
        row.update(metrics)
        row['marked_fraction'] = ratio(tp + fp, evaluated)
        row['background_fpr'] = ratio(fp, fp + tn)
        groups[key].append(row)

    tables, coverage_rows = defaultdict(list), []
    for key, values in sorted(coverage.items()):
        selected = groups[key]
        meta = dict(zip(KEY, key))
        positive = [r for r in selected if r['recall'] is not None]
        h0 = [r for r in selected if r['hypothesis'] == 'H0']
        stable = [r for r in selected if r['arm'] == 'event' and r['tp'] + r['fn'] == 0]
        result = dict(meta, n_facades=len({r['building_id'] for r in selected}),
                      n_rows=len(selected), n_sequences=len({r['sequence_id'] for r in selected}),
                      n_positive=len(positive), n_h0=len(h0), n_event_without_change=len(stable))
        result.update({m: macro(selected, m) for m in METRICS})
        result.update({c: sum(r[c] for r in selected) for c in COUNTS})
        result.update(marked_fraction=macro(selected, 'marked_fraction'),
                      event_background_fpr=macro(positive, 'background_fpr'),
                      event_without_change_fpr=macro(stable, 'background_fpr'))
        # Кадр 3 (индекс с нуля): сохраняющееся повреждение в пятикадровой синтетике.
        persistence = [r for r in stable if r['frame'] == '3'] if meta['domain'] == 'synthetic' and meta['mode'] != 'earliest' else []
        result['n_persistent_units'] = len(persistence)
        result['persistent_fpr'] = macro(persistence, 'background_fpr')
        coverage_rows.append(dict(meta, input_rows=len(values), quantified_rows=len(selected),
                                  skipped_unverified_rows=len(values) - len(selected)))
        if selected:
            tables[key[:3]].append(result)

    lines = ['Метрики в %; среднее по кадрам внутри фасада, затем по фасадам.',
             'P/R/F1/IoU: только кадры с положительной разметкой; H0 FPR: контрольные H0.',
             'EARLIEST оценивает состояние относительно первого кадра; PREVIOUS/HISTORY — новое событие.',
             'VAL с подобранным на нём порогом — предварительная оценка, не независимый TEST.',
             'Счётчики — суммы пиксельных наблюдений, не уникальных пикселей фасада.',
             'marked_fraction — доля отмеченных оцениваемых пикселей, не измерение времени эксперта.',
             'Покрытие относится только к входному CSV; полнота исходного запуска здесь не проверяется.', '']
    fmt = lambda value: '—' if value is None else f'{100 * value:.2f}'
    for key, values in sorted(tables.items()):
        lines += [' / '.join(key).upper(),
                  'method / variant / point | P | R | F1 | IoU | H0 FPR | facades | positive/all | TP/FP/FN/TN | marked % | no-event FPR (n) | persistent FPR (n)']
        for r in values:
            point = r['fpr_target'] or 'native'
            lines.append(f"{r['method']} / {r['variant']} / {point} | " +
                         ' | '.join(fmt(r[m]) for m in METRICS) +
                         f" | {r['n_facades']} | {r['n_positive']}/{r['n_rows']} | " +
                         '/'.join(str(r[c]) for c in ('tp', 'fp', 'fn', 'tn')) +
                         f" | {fmt(r['marked_fraction'])} | {fmt(r['event_without_change_fpr'])} ({r['n_event_without_change']}) | {fmt(r['persistent_fpr'])} ({r['n_persistent_units']})")
        lines.append('')
    if not any(r['domain'] == 'real' for r in rows):
        lines.append('Реальных данных во входном CSV нет. Метрики реальных фасадов не рассчитаны.')
    skipped = sum(r['skipped_unverified_rows'] for r in coverage_rows)
    if skipped:
        lines.append(f'Без подтверждённой временной разметки: {skipped} строк; см. coverage.csv.')
    args.out.mkdir(parents=True, exist_ok=False)
    for key, values in tables.items():
        name = '_'.join(re.sub(r'[^a-zA-Z0-9_-]', '_', part) for part in key)
        write_csv(args.out / (name + '.csv'), values)
    if coverage_rows:
        write_csv(args.out / 'coverage.csv', coverage_rows)
    text = '\n'.join(lines) + '\n'
    (args.out / 'summary.txt').write_text(text, encoding='utf-8')
    provenance = dict(source=str(args.csv.resolve()), source_sha256=hashlib.sha256(raw).hexdigest(),
                      input_rows=len(rows), unique_unit_ids=len({r['unit_id'] for r in rows}),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      aggregation='frame mean within building, then equal building mean')
    (args.out / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(text)
    print('Сохранено:', args.out.resolve())


if __name__ == '__main__':
    main()
