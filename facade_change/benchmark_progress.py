"""Progress and readable intermediate tables for the controlled benchmark."""
from __future__ import annotations

from contextlib import contextmanager
import math
import sys


def _number(value):
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _percent(value):
    number = _number(value)
    return "—" if number is None else f"{100 * number:.2f}%"


def _decimal(value, precision=4):
    number = _number(value)
    return "—" if number is None else f"{number:.{precision}f}"


def _method_table(method_summaries):
    columns = (
        ("Метод", "method", str),
        ("F1", "f1", _percent),
        ("IoU", "iou", _percent),
        ("Precision", "precision", _percent),
        ("Recall", "recall", _percent),
        ("H0 FPR", "h0_pixel_fpr", _percent),
        ("Сравнимо", "comparable_fraction", _percent),
        ("Видно правки", "retained_visible_edit_fraction", _percent),
        ("Время, с", "scoring_seconds", lambda value: _decimal(value, 1)),
        ("Порог", "threshold", _decimal),
        ("Пар", "case_count", lambda value: "—" if value is None else str(int(value))),
    )
    headers = [column[0] for column in columns]
    rows = []
    for method, summary in method_summaries.items():
        values = dict(summary, method=method)
        rows.append([formatter(values.get(key)) for _, key, formatter in columns])
    widths = [max([len(header)] + [len(row[index]) for row in rows])
              for index, header in enumerate(headers)]

    def render(row):
        return "  ".join(value.ljust(width) if index == 0 else value.rjust(width)
                         for index, (value, width) in enumerate(zip(row, widths)))

    lines = [render(headers), "  ".join("-" * width for width in widths)]
    lines.extend(render(row) for row in rows)
    if not rows:
        lines.append("Оценок пока нет.")
    return "\n".join(lines)


def format_crop_table(base, method_summaries, calibrated, completed_bases=None):
    """Format one crop across all its selected cases, without selecting a threshold."""
    part = str(base.get("split", "unknown")).upper()
    title = f"Кроп {base.get('base_id', '—')} | {part} | фасад {base.get('building_id', '—')}"
    if completed_bases is not None:
        title += f" | завершено кропов: {completed_bases}"
    if calibrated:
        status = f"{part}: пороги зафиксированы по всей выбранной validation."
    else:
        status = (f"{part}: предварительные пороги только по уже обработанной validation; "
                  "итоговое сравнение после её завершения.")
    return "\n".join((title, status, _method_table(method_summaries),
                      "Сравнимо / Видно правки — доли поддержки оценки / сохранённой видимой правки. "
                      "Время — только вычисление карт."))


def format_cumulative_test_table(method_summaries, completed_bases):
    """Format the cumulative held-out test summaries supplied by the evaluator."""
    return "\n".join((
        f"Накопленный TEST | завершено тестовых кропов: {completed_bases}",
        "TEST: пороги зафиксированы по всей выбранной validation.",
        _method_table(method_summaries),
        "Метрики агрегированы по фасадам; это промежуточный результат выбранного набора.",
    ))


def format_case_table(cases, method_case_metrics):
    """Show every selected state/condition without retuning thresholds per case."""
    methods = list(method_case_metrics)
    headers = ["Состояние", "Условие", "Гипотеза"] + methods
    rows = []
    for case in cases:
        hypothesis = case.get("hypothesis")
        if hypothesis is None:
            hypothesis = "H1" if case.get("state") in {"crack", "paint_patch"} else "H0"
        metric = "f1" if hypothesis == "H1" else "h0_pixel_fpr"
        rows.append([
            str(case.get("state", "—")), str(case.get("scenario_id", "—")), hypothesis,
        ] + [_percent(method_case_metrics[method].get(case["case_id"], {}).get(metric))
             for method in methods])
    widths = [max([len(header)] + [len(row[index]) for row in rows])
              for index, header in enumerate(headers)]

    def render(row):
        return "  ".join(value.ljust(width) if index < 3 else value.rjust(width)
                         for index, (value, width) in enumerate(zip(row, widths)))

    lines = ["Каждая выбранная пара: H1 — F1 (выше лучше); H0 — FPR (ниже лучше).",
             render(headers), "  ".join("-" * width for width in widths)]
    lines.extend(render(row) for row in rows)
    if not rows:
        lines.append("Оценок пока нет.")
    return "\n".join(lines)


def _load_tqdm():
    try:
        from tqdm import tqdm
        return tqdm
    except ImportError:
        return None


class _BenchmarkProgress:
    def __init__(self, total_jobs, total_bases):
        self.total_jobs = total_jobs
        self.total_bases = total_bases
        self.completed_jobs = 0
        self.base_jobs = 0
        self.completed_base_jobs = 0
        self.base_bar = None
        self._tqdm = _load_tqdm()
        self.overall_bar = None if self._tqdm is None else self._tqdm(
            total=total_jobs, desc="Весь benchmark", unit="оценка", position=0,
            dynamic_ncols=True, mininterval=0.5,
        )
        if self.overall_bar is None:
            self.write(f"Benchmark: {total_bases} кропов; {total_jobs} оценок; tqdm недоступен.")

    def start_base(self, base_index, base_jobs, description):
        self.close_base()
        self.base_jobs = base_jobs
        self.completed_base_jobs = 0
        label = f"Кроп {base_index}/{self.total_bases}: {description}"
        if self._tqdm is not None:
            self.base_bar = self._tqdm(
                total=base_jobs, desc=label, unit="оценка", position=1, leave=False,
                dynamic_ncols=True, mininterval=0.5,
            )
        else:
            self.write(f"{label}; {base_jobs} оценок")

    def job_started(self, method=None, case_label=None):
        postfix = " / ".join(str(value) for value in (method, case_label) if value is not None)
        if self.base_bar is not None and postfix:
            self.base_bar.set_postfix_str(postfix, refresh=True)

    def job_finished(self, method=None, case_id=None):
        self.completed_jobs += 1
        self.completed_base_jobs += 1
        postfix = " / ".join(str(value) for value in (method, case_id) if value is not None)
        if self.base_bar is not None:
            if postfix:
                self.base_bar.set_postfix_str(postfix, refresh=False)
            self.base_bar.update(1)
        if self.overall_bar is not None:
            self.overall_bar.update(1)
        elif self.completed_base_jobs == self.base_jobs:
            self.write(f"Прогресс: {self.completed_jobs}/{self.total_jobs} оценок")

    def write(self, message):
        if self._tqdm is None:
            print(message, flush=True)
        else:
            self._tqdm.write(str(message), file=sys.stdout)

    def close_base(self):
        if self.base_bar is not None:
            self.base_bar.close()
            self.base_bar = None

    def close(self):
        self.close_base()
        if self.overall_bar is not None:
            self.overall_bar.close()


@contextmanager
def progress_bars(total_jobs, total_bases):
    """Count each completed case/method evaluation in overall and crop bars."""
    progress = _BenchmarkProgress(total_jobs, total_bases)
    try:
        yield progress
    finally:
        progress.close()
