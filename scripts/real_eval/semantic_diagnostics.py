"""Read saved failures/timings: python -m scripts.real_eval.semantic_diagnostics --out RUN/smoke."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import sys

IDENTITY = ('config_revision', 'kind', 'split', 'sequence_id', 'frame', 'reference_frame',
            'method', 'segmenter', 'variant', 'route', 'mode')


def _json(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default


def _identity(row):
    return tuple('' if row.get(k) is None else str(row.get(k, '')) for k in IDENTITY)


def _duration(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def diagnose(out):
    """Count only the latest coverage snapshot; old unit files supply links only."""
    out = Path(out).expanduser().resolve()
    run = _json(out / 'run.json', {})
    if not run.get('config_revision') or not (out / 'coverage.csv').is_file():
        raise ValueError(f'Нет сохранённого run.json/coverage.csv: {out}; дождитесь первого отчёта')
    revision, warnings = run['config_revision'], []
    with (out / 'coverage.csv').open(newline='', encoding='utf-8') as stream:
        all_rows = list(csv.DictReader(stream))
    rows = [r for r in all_rows if r.get('config_revision') == revision]
    failed = [r for r in rows if r.get('status') == 'failed']
    if len(rows) != len(all_rows):
        warnings.append(f'coverage.csv: исключено строк другой revision: {len(all_rows)-len(rows)}')
    if run.get('records') != len(rows) or run.get('failed') != len(failed):
        warnings.append('run.json и coverage.csv расходятся; повторите диагностику после записи отчёта')
    wanted = {(_identity(r), r.get('error', '')) for r in failed}
    links = {}
    for path in sorted((out / 'units').glob('*.json')):
        try:
            unit = _json(path, {})
        except (OSError, json.JSONDecodeError):
            continue
        key = (_identity(unit), unit.get('error', ''))
        if unit.get('status') == 'failed' and key in wanted:
            links.setdefault(key, str(path))
    grouped, breakdown = defaultdict(list), Counter()
    for row in failed:
        grouped[row.get('error', '')].append(row)
        breakdown[tuple(row.get(k) or '-' for k in ('segmenter', 'method', 'route'))] += 1
    errors = [dict(error=error, count=len(members), example=members[0],
                   unit_file=links.get((_identity(members[0]), error)))
              for error, members in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))]
    timing, sources, seen, inventory = defaultdict(list), Counter(), set(), 0
    for row in _json(out / 'timing_observations.json', []) or []:
        if row.get('status') != 'completed':
            continue
        if row.get('inventory'):
            inventory += 1
            continue
        sources[row.get('source', 'unknown')] += 1
        if row.get('source') != 'measured' or row.get('cache_key') in seen:
            continue
        if row.get('cache_key'):
            seen.add(row['cache_key'])
        timing[tuple(row.get(k) for k in ('stage', 'method', 'segmenter', 'variant'))].append(row)
    timings = []
    for key, group in sorted(timing.items(), key=lambda item: str(item[0])):
        durations = [r['elapsed_seconds'] for r in group if _duration(r.get('elapsed_seconds'))]
        initializations = [r['init_seconds'] for r in group if _duration(r.get('init_seconds'))]
        timings.append(dict(group=key, samples=len(group), timed_samples=len(durations),
                            inference_seconds=sum(durations), init_seconds=sum(initializations),
                            init_samples=len(initializations), median_seconds=statistics.median(durations) if durations else None,
                            timing_only_samples=sum(bool(r.get('timing_only')) for r in group)))
    return dict(out=str(out), status=run.get('status'), config_revision=revision,
                records_source='coverage.csv', record_rows=len(rows), failed_rows=len(failed),
                failed_report=run.get('failed'), failure_breakdown=breakdown, errors=errors,
                timing=timings, observation_sources=dict(sources), inventory_observations=inventory,
                estimate=_json(out / 'estimate.json', {}) or {}, warnings=warnings)


def format_summary(summary, *, max_errors=8, max_error_chars=1400):
    lines = [f'OUT: {summary["out"]}', f'STATUS: {summary["status"]}', f'CONFIG_REVISION: {summary["config_revision"]}',
             f'ROWS_SOURCE: {summary["records_source"]}; RECORD_ROWS: {summary["record_rows"]}',
             f'FAILED_REPORT: {summary["failed_report"]}; FAILED_ROWS: {summary["failed_rows"]}']
    if summary['failed_rows']:
        lines += ['Это failed evaluation units, не число независимых падений: одна причина может повториться в зависимых маршрутах.',
                  'FAILURES (segmenter | method | route | count):']
        lines += ['  ' + ' | '.join(key) + f' | {value}' for key, value in sorted(summary['failure_breakdown'].items())]
    for number, item in enumerate(summary['errors'][:max_errors], 1):
        error, example = item['error'] or '(empty error)', item['example']
        if len(error) > max_error_chars:
            head = min(200, max_error_chars // 3)
            error = error[:head] + '\n… [середина сокращена; полный текст в unit JSON/coverage.csv] …\n' + error[-(max_error_chars-head):]
        lines += [f'ERROR {number}: units={item["count"]}', error,
                  '  EXAMPLE: ' + ' | '.join(f'{k}={example.get(k)}' for k in ('split', 'segmenter', 'method', 'variant', 'route', 'sequence_id', 'frame', 'reference_frame')),
                  '  UNIT: ' + (item['unit_file'] or f'{summary["out"]}/coverage.csv (unit JSON не найден)')]
    if len(summary['errors']) > max_errors:
        lines.append(f'Ещё групп ошибок: {len(summary["errors"])-max_errors}; увеличьте --max-errors.')
    estimate = summary['estimate']
    wall = estimate.get('wall_seconds')
    lines.append(f'WALL_MINUTES: {wall/60:.2f} (до генерации последнего отчёта; финальный render/закрытие workers не включены)' if _duration(wall) else 'WALL_MINUTES: N/A')
    lines += [f'OBSERVATION_SOURCES: {summary["observation_sources"]}; inventory excluded={summary["inventory_observations"]}',
              'MEASURED SUCCESSFUL NATIVE CALLS (stage | method | segmenter | variant | n | call s | median s | init s | timing probes):']
    for row in summary['timing']:
        median = f'{row["median_seconds"]:.3f}' if row['median_seconds'] is not None else 'N/A'
        lines.append('  ' + ' | '.join(str(value or '-') for value in row['group'])
                     + f' | {row["timed_samples"]}/{row["samples"]} | {row["inference_seconds"]:.3f} | {median}'
                     + f' | {row["init_seconds"]:.3f} ({row["init_samples"]}) | {row["timing_only_samples"]}')
    lines += ['call — сохранённое время вызова, не чистое GPU-время; init показан отдельно и не прибавляется к call.',
              'Замеры cache/imported не суммируются; неуспешные вызовы, CPU-отчёты и чтение кэша в суммы не входят.',
              f'ESTIMATE_STATUS: {estimate.get("status", "unavailable")}; remaining_seconds={estimate.get("total_lower_seconds")} .. {estimate.get("total_upper_seconds")}']
    if summary['failed_rows'] or summary['failed_report']:
        lines.append('ETA при ошибках неполон: успешные замеры не подтверждают время неисправных моделей/маршрутов.')
    lines.extend('WARNING: ' + warning for warning in summary['warnings'])
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-errors', type=int, default=8)
    parser.add_argument('--max-error-chars', type=int, default=1400)
    args = parser.parse_args(argv)
    if args.max_errors < 1 or args.max_error_chars < 200:
        parser.error('--max-errors must be >=1 and --max-error-chars >=200')
    try:
        summary = diagnose(args.out)
    except (OSError, ValueError) as error:
        print(f'DIAGNOSTICS_ERROR: {error}', file=sys.stderr)
        return 2
    print(format_summary(summary, max_errors=args.max_errors, max_error_chars=args.max_error_chars))
    return 0


if __name__ == '__main__':
    sys.exit(main())
