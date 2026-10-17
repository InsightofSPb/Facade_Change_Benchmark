#!/usr/bin/env python3
"""CPU review -> explicit selected GPU run -> CPU frozen-threshold report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Реальные VAL-пары: просмотр перед тестом устойчивости')
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='CPU: все кандидаты, GT и варианты освещения; без моделей')
    prepare.add_argument('--run', required=True, help='Исходный temporal-experiment')
    prepare.add_argument('--snapshot', required=True, help='Сохранённый temporal VAL snapshot')
    prepare.add_argument('--out', required=True, help='Новый каталог галереи')
    prepare.add_argument('--repo', help='Checkout с исходным неизменённым кодом чтения данных')
    prepare.add_argument('--boundary-tolerance', type=int, choices=(0, 2), default=0)
    execute = sub.add_parser('run', help='Прогон только по сохранённому выбору из галереи')
    execute.add_argument('--prepared', required=True)
    execute.add_argument('--selection', required=True, help='reviewed_selection.json, скачанный из галереи')
    execute.add_argument('--out', required=True)
    execute.add_argument('--methods', nargs='+', help='По умолчанию все 8 методов, включая GeoSCD и ABS')
    execute.add_argument('--geoscd-run', help='Дополнительный кэш исходных GeoSCD-пар')
    execute.add_argument('--device', help='По умолчанию устройство исходного запуска')
    execute.add_argument('--resume', action='store_true')
    execute.add_argument('--retry-failed', action='store_true')
    report = sub.add_parser('report', help='CPU: метрики и комбинации; без подбора новых порогов')
    report.add_argument('--run', required=True, help='Выход robustness run')
    report.add_argument('--out', required=True, help='Новый каталог отчёта')
    report.add_argument('--calibration-run', help='Необязательно: готовая real LOFO-калибровка исходных карт')
    report.add_argument('--budgets', nargs='+', type=float, default=[.01, .05])
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    if args.command == 'prepare':
        from scripts.real_eval.robustness_prepare import prepare
        result = prepare(args.run, args.snapshot, args.out, repo_path=args.repo or repo,
                         boundary_tolerance=args.boundary_tolerance)
        print(json.dumps({'gallery': str(Path(args.out).resolve() / 'gallery.html'),
                          'candidate_pairs': len(result['pairs']), 'gpu_started': False}, ensure_ascii=False))
    elif args.command == 'run':
        from scripts.real_eval.robustness_run import run
        result = run(args.prepared, args.selection, args.out, methods=args.methods,
                     geoscd_run=args.geoscd_run, resume=args.resume,
                     retry_failed=args.retry_failed, device=args.device)
        print(json.dumps({'out': str(Path(args.out).resolve()), 'status': result.get('status')}, ensure_ascii=False))
    else:
        from scripts.real_eval.robustness_report import report
        report(args.run, args.out, calibration_run=args.calibration_run, budgets=tuple(args.budgets))


if __name__ == '__main__':
    main()
