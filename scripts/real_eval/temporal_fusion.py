#!/usr/bin/env python3
"""Fixed AND/OR combinations of SAVED synthetic-VAL maps; NumPy only, no inference.

--units is the existing quick_units.csv; --run contains its maps/*.npz.
Thresholds are reused, never optimized. Default auxiliary H0 budget: 0.01.
--budgets 0.01 0.05 includes both saved operating points.
"""
import argparse
import csv
import hashlib
import io
import json
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import numpy as np

BASES = ('geoscd', 'anychange')
AUX = ('msdzip_abs', 'msdzip_mod256', 'dinov2', 'ssim')
COUNT_FIELDS = ('tp', 'fp', 'fn', 'tn', 'evaluated_pixel_count')
MATCH_FIELDS = ('building_id', 'sequence_id', 'frame', 'arm', 'hypothesis', 'target', 'domain', 'split')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def counts(prediction, labels):
    positive, negative = labels == 1, labels == 0
    values = (np.count_nonzero(prediction & positive), np.count_nonzero(prediction & negative),
              np.count_nonzero(~prediction & positive), np.count_nonzero(~prediction & negative))
    return dict(zip(COUNT_FIELDS, (*map(int, values), int(sum(values)))))


def compare_arrays(left, right):
    for name in ('labels', 'support'):
        if not np.array_equal(left[name], right[name]):
            raise ValueError(f'Fusion refused: different {name}')


def prfi(row):
    tp, fp, fn = (row[k] for k in ('tp', 'fp', 'fn'))
    divide = lambda a, b: a / b if b else 0.
    return dict(precision=divide(tp, tp + fp), recall=divide(tp, tp + fn),
                f1=divide(2 * tp, 2 * tp + fp + fn), iou=divide(tp, tp + fp + fn))


def macro(rows, function):
    buildings = defaultdict(list)
    for row in rows:
        value = function(row)
        if value is not None:
            buildings[row['building_id']].append(value)
    return mean(mean(v) for v in buildings.values()) if buildings else None


def pooled_facade_metrics(rows):
    buildings = defaultdict(list)
    for row in rows:
        buildings[row['building_id']].append(row)
    values = [prfi({k: sum(r[k] for r in rs) for k in COUNT_FIELDS}) for rs in buildings.values()]
    return {k: mean(r[k] for r in values) if values else None for k in ('precision', 'recall', 'f1', 'iou')}


def aggregate(rows):
    positive = [r for r in rows if r['hypothesis'] == 'H1' and r['tp'] + r['fn'] > 0]
    h0 = [r for r in rows if r['hypothesis'] == 'H0']
    stable = [r for r in rows if r['arm'] == 'event' and r['tp'] + r['fn'] == 0]
    fpr = lambda r: r['fp'] / (r['fp'] + r['tn']) if r['fp'] + r['tn'] else None
    result = {k: macro(positive, lambda r, k=k: prfi(r)[k]) for k in ('precision', 'recall', 'f1', 'iou')}
    result.update({'all_' + k: v for k, v in pooled_facade_metrics(rows).items()})
    result.update({'event_' + k: v for k, v in pooled_facade_metrics([r for r in rows if r['arm'] == 'event']).items()})
    persistence = [r for r in stable if int(r['frame']) == 3 and r['target'] == 'step_event']
    result.update(h0_fpr=macro(h0, fpr), no_event_fpr=macro(stable, fpr),
                  persistent_fpr=macro(persistence, fpr), n_persistent=len(persistence),
                  marked_fraction=macro(rows, lambda r: (r['tp'] + r['fp']) / r['evaluated_pixel_count']
                                        if r['evaluated_pixel_count'] else None),
                  n_facades=len({r['building_id'] for r in rows}), n_units=len(rows),
                  n_positive=len(positive), n_h0=len(h0), n_no_event=len(stable))
    result.update({k: sum(r[k] for r in rows) for k in COUNT_FIELDS})
    return result


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run(args):
    root, out = Path(args.run).resolve(), Path(args.out).resolve()
    if out == root or root in out.parents:
        raise ValueError('Output must be outside the source run')
    if out.exists():
        raise FileExistsError(f'Use a new output directory: {out}')
    raw = Path(args.units).read_bytes()
    source = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    groups, coverage = defaultdict(dict), []
    for row in source:
        if (row['domain'], row['split']) != ('synthetic', 'val'):
            continue
        if row['variant'] != ('current_train' if row['method'].startswith('msdzip_') else 'frozen'):
            continue
        native = row['decision'] == 'author_native'
        budget = None if native else float(row['fpr_target'])
        if budget is not None and budget not in args.budgets:
            continue
        expected_target = 'earliest_state' if row['mode'] == 'earliest' else 'step_event'
        if row['mode'] not in {'previous', 'earliest', 'history'} or row['target'] != expected_target:
            raise ValueError('Unexpected mode/target')
        group = row['method'], row['mode'], budget
        key = row['sequence_id'], int(row['frame'])
        if key in groups[group]:
            raise ValueError('Duplicate unit in ' + repr(group))
        if not row['building_id'] or row['hypothesis'] not in {'H0', 'H1'}:
            raise ValueError('Missing facade or target hypothesis')
        groups[group][key] = row
    if not groups:
        raise ValueError('No matching synthetic VAL rows')
    reference = next((g for g in groups if g[1] == 'previous'), None)
    if reference is None:
        raise ValueError('No PREVIOUS group for coverage reference')
    expected = set().union(*(set(records) for g, records in groups.items() if g[1] == 'previous'))
    # All 4 evaluated frames per clean/event history; frame zero is initialization.
    by_sequence = defaultdict(set)
    for sequence, frame in expected:
        by_sequence[sequence].add(frame)
    if any(frames != {1, 2, 3, 4} for frames in by_sequence.values()):
        raise ValueError('Expected complete five-frame synthetic histories (evaluated indices 1..4)')
    complete = {g for g, records in groups.items() if set(records) == expected}
    for g in groups:
        if g not in complete:
            coverage.append(dict(item=str(g), status='incomplete_skipped', n_units=len(groups[g])))

    # Explicit, fixed design: baselines plus native base AND/OR each auxiliary.
    specs = []
    for g in sorted(complete, key=str):
        specs.append(dict(name=g[0], mode=g[1], aux_budget=g[2], operation='single', left=g, right=None))
    for mode in ('previous', 'earliest', 'history_addon'):
        for base in BASES:
            left = (base, 'previous' if mode == 'history_addon' else mode, None)
            for aux in AUX:
                if mode == 'history_addon' and aux == 'ssim':
                    continue  # No saved SSIM history in the current run.
                for budget in args.budgets:
                    right = (aux, 'history' if mode == 'history_addon' else mode, budget)
                    if left not in complete or right not in complete:
                        coverage.append(dict(item=f'{left} + {right}', status='missing_or_incomplete_skipped', n_units=0))
                        continue
                    for operation in ('AND', 'OR'):
                        specs.append(dict(name=f'{base} {operation} {aux}', mode=mode, aux_budget=budget,
                                          operation=operation, left=left, right=right))
    cache, verified, all_rows = OrderedDict(), {}, []

    def load(row):
        path = (root / row['artifact_path']).resolve()
        if root not in path.parents:
            raise ValueError('Artifact path escapes source run')
        stamp = (path.stat().st_size, path.stat().st_mtime_ns, row['artifact_sha256'])
        if path not in verified:
            if sha(path) != row['artifact_sha256']:
                raise ValueError('Artifact hash mismatch: ' + str(path))
            verified[path] = stamp
        elif verified[path] != stamp:
            raise ValueError('Source artifact changed: ' + str(path))
        if path in cache:
            cache.move_to_end(path)
            arrays = cache[path]
        else:
            with np.load(path, allow_pickle=False) as archive:
                names = ['scores', 'labels', 'support']
                if row['decision'] == 'author_native':
                    names.append('native_prediction')
                arrays = {name: archive[name] for name in names}
            score, labels, support = (arrays[k] for k in ('scores', 'labels', 'support'))
            if score.ndim != 2 or score.dtype != np.float32 or labels.dtype != np.uint8 or support.dtype != bool:
                raise ValueError('Invalid array dtypes or dimensions')
            if score.shape != labels.shape or score.shape != support.shape or not np.isin(labels, [0, 1, 255]).all():
                raise ValueError('Invalid label grid')
            if not np.isfinite(score[support]).all() or np.any((labels != 255) & ~support):
                raise ValueError('Nonfinite score or label outside support')
            if 'native_prediction' in arrays and (arrays['native_prediction'].dtype != bool or arrays['native_prediction'].shape != score.shape):
                raise ValueError('Invalid native mask')
            cache[path] = arrays
            if len(cache) > 16:
                cache.popitem(last=False)
        pred = (arrays['native_prediction'] if row['decision'] == 'author_native'
                else arrays['scores'] > np.float32(row['threshold'])) & arrays['support']
        actual = counts(pred, arrays['labels'])
        if any(actual[k] != int(row[k]) for k in COUNT_FIELDS):
            raise ValueError('Saved CSV counts disagree with map/threshold: ' + row['unit_id'])
        if row['hypothesis'] == 'H0' and actual['tp'] + actual['fn']:
            raise ValueError('Positive ground truth in H0')
        return arrays, pred

    # One frame at a time; no RGB or history-memory arrays are decompressed.
    for index, key in enumerate(sorted(expected), 1):
        loaded = {g: load(groups[g][key]) for g in complete}
        for spec in specs:
            left_row = groups[spec['left']][key]
            arrays, pred = loaded[spec['left']]
            right_row = None
            if spec['right'] is not None:
                right_row = groups[spec['right']][key]
                if any(left_row[k] != right_row[k] for k in MATCH_FIELDS):
                    raise ValueError('Fusion refused: different frame/target metadata')
                right_arrays, right_pred = loaded[spec['right']]
                compare_arrays(arrays, right_arrays)
                pred = pred & right_pred if spec['operation'] == 'AND' else pred | right_pred
            meta = {k: left_row[k] for k in MATCH_FIELDS}
            all_rows.append(dict(name=spec['name'], mode=spec['mode'], aux_budget=spec['aux_budget'],
                                 operation=spec['operation'], **meta,
                                 left_unit_id=left_row['unit_id'], right_unit_id=right_row['unit_id'] if right_row else '',
                                 left_threshold=left_row['threshold'], right_threshold=right_row['threshold'] if right_row else '',
                                 **counts(pred, arrays['labels'])))
        if index == 1 or index % 8 == 0 or index == len(expected):
            print(f'Saved maps: {index}/{len(expected)} frames; no model inference', flush=True)

    grouped = defaultdict(list)
    for row in all_rows:
        grouped[row['name'], row['mode'], row['aux_budget'], row['operation']].append(row)
    summary = [dict(name=k[0], mode=k[1], aux_budget=k[2], operation=k[3], **aggregate(rs))
               for k, rs in sorted(grouped.items(), key=lambda item: str(item[0]))]
    lines = ['Только SYNTHETIC VAL. Готовые пороги; без подбора комбинаций по качеству.',
             'aux_budget относится к отдельной вспомогательной ветви; FPR комбинации измеряется заново.',
             'P/R/F1/IoU: положительные кадры, среднее внутри фасада, затем между фасадами.',
             'Pseq/Rseq/F1seq/IoUseq: все шаги EVENT-ветки, суммы счётчиков внутри фасада, затем среднее фасадов.',
             'all_* в CSV: аналогично, но с включением CLEAN-ветки. Пустые знаменатели дают 0.',
             'history_addon = штатная PREVIOUS-маска базы + HISTORY-оценка вспомогательной ветви.',
             'Истории EARLIEST и step_event не смешиваются; реальные данные и TEST не оцениваются.',
             'Это предварительный VAL-анализ, не независимая оценка обобщения. Все числа далее в %.']
    percent = lambda v: '-' if v is None else f'{100*v:.2f}'
    metrics = ('precision', 'recall', 'f1', 'iou', 'event_precision', 'event_recall', 'event_f1', 'event_iou',
               'h0_fpr', 'no_event_fpr', 'persistent_fpr', 'marked_fraction')
    for mode in ('previous', 'earliest', 'history', 'history_addon'):
        selected = [r for r in summary if r['mode'] == mode]
        if not selected:
            continue
        lines += ['', mode.upper(), 'method | aux point | P | R | F1 | IoU | Pseq | Rseq | F1seq | IoUseq | H0 FPR | no-event FPR | persistent FPR | marked % | facades/units']
        for r in selected:
            point = 'native' if r['aux_budget'] is None else f"{100*r['aux_budget']:g}%"
            lines.append(f"{r['name']} | {point} | " + ' | '.join(percent(r[m]) for m in metrics) +
                         f" | {r['n_facades']}/{r['n_units']}")
    lines += ['', f'Skipped/incomplete combinations or groups: {len(coverage)}; see coverage.csv.']
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / 'metrics.csv', summary)
    write_csv(out / 'units.csv', all_rows)
    write_csv(out / 'coverage.csv', coverage)
    text = '\n'.join(lines) + '\n'
    (out / 'summary.txt').write_text(text, encoding='utf-8')
    provenance = dict(source_run=str(root), input_csv=str(Path(args.units).resolve()),
                      input_csv_sha256=hashlib.sha256(raw).hexdigest(), script_sha256=sha(__file__),
                      created_utc=datetime.now(timezone.utc).isoformat(), budgets=args.budgets,
                      threshold_policy='reuse saved component thresholds; no H1 tuning or composite calibration',
                      scope='synthetic VAL only; complete matching groups from input CSV',
                      expected_keys=sorted(expected), combinations=specs,
                      verified_artifacts={str(p.relative_to(root)): v[2] for p, v in verified.items()})
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(text, flush=True)
    print('Saved:', out)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--units', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--budgets', type=float, nargs='+', default=[.01])
    args = parser.parse_args(argv)
    if len(set(args.budgets)) != len(args.budgets) or any(x not in (.01, .05) for x in args.budgets):
        parser.error('Use unique saved budgets: 0.01 and/or 0.05')
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
