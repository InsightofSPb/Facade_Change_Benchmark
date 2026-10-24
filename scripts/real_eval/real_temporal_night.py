#!/usr/bin/env python3
"""Audit -> selected real histories -> timed pilot -> resumable frozen inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def _launch_detached(args, argv):
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    log = Path(str(out) + '.log')
    if out.exists() and not args.resume:
        raise FileExistsError('Output exists; use --resume or a new output directory')
    command = [sys.executable, '-u', '-s', str(Path(__file__).resolve()),
               *(value for value in argv if value != '--detach')]
    with log.open('a', encoding='utf-8') as stream:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True,
                                   cwd=str(Path(__file__).resolve().parents[2]))
    print(json.dumps(dict(status='launched', pid=process.pid, out=str(out), log=str(log),
                          note='Продолжение: та же команда с --resume. Компьютер должен оставаться включённым.'),
                     ensure_ascii=False, indent=2))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description='Готовые модели: реальные хронологии TRAIN/VAL/TEST')
    commands = parser.add_subparsers(dest='command', required=True)
    audit = commands.add_parser('audit', help='CPU: доступные фасады, годы и сохранённые совмещения')
    audit.add_argument('--run', required=True)
    audit.add_argument('--snapshot', required=True)
    audit.add_argument('--dataset', help='По умолчанию dataset исходного запуска')
    audit.add_argument('--alignment-runs', nargs='+', help='Явный список сохранённых alignment; иначе обнаружение')
    audit.add_argument('--alignment-session', help='Завершённая сессия dataset_statistics; только её выбранные пары')
    audit.add_argument('--runs-root', help='Корень runs для обнаружения существующих совмещений')
    audit.add_argument('--splits', nargs='+', choices=('train', 'val', 'test'), default=['test', 'val', 'train'])
    audit.add_argument('--out', required=True)
    prepare = commands.add_parser('prepare', help='CPU: отбор по размеченным изменениям, все годы и галерея')
    prepare.add_argument('--audit', required=True)
    prepare.add_argument('--out', required=True)
    prepare.add_argument('--max-facades', type=int, default=12)
    prepare.add_argument('--crops-per-facade', type=int, choices=(1, 2), default=1)
    prepare.add_argument('--tile-size', type=int, default=512)
    prepare.add_argument('--stride', type=int, default=256)
    prepare.add_argument('--min-support', type=float, default=.8)
    prepare.add_argument('--min-known-pixels', type=int, default=1024)
    prepare.add_argument('--buildings', nargs='+', help='Необязательно: явно ограничить фасады по audit')
    for name in ('pilot', 'run'):
        command = commands.add_parser(name, help=('Реальный замер на малой выборке' if name == 'pilot'
                                                  else 'PREVIOUS → EARLIEST → HISTORY, без обучения'))
        command.add_argument('--prepared', required=True)
        command.add_argument('--out', required=True)
        command.add_argument('--methods', nargs='+', help='По умолчанию девять существующих методов')
        command.add_argument('--history-methods', nargs='+', help='DINO, semantic DINO, MSDZip ABS/MOD256')
        command.add_argument('--modes', nargs='+', choices=('previous', 'earliest', 'history'),
                             default=['previous', 'earliest', 'history'])
        command.add_argument('--device', default=None)
        command.add_argument('--geoscd-run', help='Дополнительный кэш готовых пар GeoSCD')
        command.add_argument('--reuse-run', help='Переиспользовать проверенные парные карты отдельного pilot')
        command.add_argument('--resume', action='store_true')
        command.add_argument('--retry-failed', action='store_true')
        command.add_argument('--max-hours', type=float, help='Мягкий лимит: остановка между единицами работы')
        command.add_argument('--detach', action='store_true', help='Фоновый запуск, журнал OUT.log')
        if name == 'pilot':
            command.add_argument('--pilot-pairs', type=int, default=2)
            command.add_argument('--pilot-history-frames', type=int, default=3)
        else:
            command.add_argument('--no-report', action='store_true', help='Не собирать CPU-отчёт после очереди')
    report = commands.add_parser('report', help='Пересобрать таблицы и сочетания из готовых карт, без GPU')
    report.add_argument('--run', required=True)
    report.add_argument('--out', help='По умолчанию RUN/reports')
    report.add_argument('--budgets', type=float, nargs='+', default=[.01, .05])
    calibrate = commands.add_parser('calibrate', help='CPU: реальный VAL LOFO, все попарные AND/OR с итоговым FPR')
    calibrate.add_argument('--run', required=True, help='Завершённый ночной запуск с units и maps')
    calibrate.add_argument('--out', required=True, help='Новый каталог CPU-анализа рядом с исходным запуском')
    calibrate.add_argument('--budgets', type=float, nargs='+', default=[.01, .05])
    calibrate.add_argument('--methods', nargs='+', help='По умолчанию все методы PREVIOUS и доступные HISTORY')
    calibrate.add_argument('--resume', action='store_true', help='Продолжить с проверкой входов, кода и готовых семейств')
    status = commands.add_parser('status', help='Короткая сводка прогресса без запуска моделей')
    status.add_argument('--run', required=True)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    if args.command in ('pilot', 'run') and args.detach:
        _launch_detached(args, argv)
        return
    if args.command == 'audit':
        from scripts.real_eval.night_prepare import audit
        result = audit(args.run, args.snapshot, args.out, alignment_runs=args.alignment_runs,
                       runs_root=args.runs_root, splits=tuple(args.splits), dataset=args.dataset,
                       alignment_session=args.alignment_session)
        print(f'Аудит: {Path(args.out).resolve() / "audit.json"}', flush=True)
    elif args.command == 'prepare':
        from scripts.real_eval.night_prepare import prepare
        result = prepare(args.audit, args.out, max_facades=args.max_facades,
                         crops_per_facade=args.crops_per_facade, tile_size=args.tile_size,
                         stride=args.stride, min_support=args.min_support,
                         min_known_pixels=args.min_known_pixels, buildings=args.buildings)
        print(f'Галерея: {Path(args.out).resolve() / "gallery.html"}', flush=True)
    elif args.command in ('run', 'pilot'):
        from scripts.real_eval.night_run import run
        extra = ({'pilot_pairs': args.pilot_pairs, 'pilot_history_frames': args.pilot_history_frames}
                 if args.command == 'pilot' else {})
        result = run(args.prepared, args.out, methods=args.methods, history_methods=args.history_methods,
                     modes=tuple(args.modes), resume=args.resume, retry_failed=args.retry_failed,
                     device=args.device, max_hours=args.max_hours, pilot=args.command == 'pilot',
                     geoscd_run=args.geoscd_run, reuse_run=args.reuse_run, **extra)
        if args.command == 'run' and not args.no_report:
            from scripts.real_eval.night_report import report
            report(args.out)
        print(json.dumps({'out': str(Path(args.out).resolve()), 'status': result.get('status')},
                         ensure_ascii=False, indent=2))
    elif args.command == 'report':
        from scripts.real_eval.night_report import report
        report(args.run, args.out, budgets=tuple(args.budgets))
    elif args.command == 'calibrate':
        from scripts.real_eval.night_calibration import run
        result = run(args.run, args.out, budgets=tuple(args.budgets), methods=args.methods, resume=args.resume)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        from scripts.real_eval.robustness_common import read_json
        root = Path(args.run).expanduser().resolve()
        state = read_json(root / 'run.json')
        print(f"Статус: {state.get('status')}")
        path = root / 'summary.json'
        if path.is_file() and state.get('status') != 'running':
            summary = read_json(path)
            print(json.dumps({k: v for k, v in summary.items() if k != 'results'},
                             ensure_ascii=False, indent=2))
        elif (root / 'progress.json').is_file():
            progress = read_json(root / 'progress.json')
            latest = progress.pop('latest', {})
            print(json.dumps(progress, ensure_ascii=False))
            print('Последняя единица:', ' / '.join(str(latest.get(k, '')) for k in
                  ('method', 'mode', 'sequence_id', 'frame', 'status')))
        if (root / 'timing.json').is_file():
            timing = read_json(root / 'timing.json')
            print('Оценка полного объёма до переиспользования кэша (не гарантия):')
            for row in timing.get('methods', []):
                lo, hi = row.get('projected_seconds_low'), row.get('projected_seconds_high')
                estimate = 'не измерено' if lo is None or hi is None else f'{lo/60:.1f}–{hi/60:.1f} мин'
                print(f"  {row['method']} / {row['mode']}: {estimate}")
            print('Всего, ч:', timing.get('estimated_hours_low'), '—', timing.get('estimated_hours_high'))
            if timing.get('unmeasured'):
                print('Нет замера:', ', '.join(timing['unmeasured']))


if __name__ == '__main__':
    main()
