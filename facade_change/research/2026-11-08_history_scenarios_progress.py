"""TTY progress and bounded-interval plain logs for the history experiment."""
from __future__ import annotations

from contextlib import contextmanager
import sys
import threading
import time


class Progress:
    def __init__(self, *, stage, total=None, model=None, epoch=None, ledger=None,
                 interval=30.0, stream=None):
        self.stream = stream or sys.stderr
        self.stage, self.total, self.model, self.epoch = stage, total, model, epoch
        self.ledger, self.interval = ledger, float(interval)
        self.count, self.fields = 0, {}
        self.started = time.monotonic()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._bar = None

    def _line(self, status="running"):
        elapsed = max(time.monotonic() - self.started, 1e-9)
        rate = self.count / elapsed
        eta = (self.total - self.count) / rate if self.total is not None and rate else None
        values = dict(stage=self.stage, model=self.model, epoch=self.epoch,
                      processed=self.count, total=self.total, status=status,
                      seconds=round(elapsed, 1), rate=round(rate, 3),
                      eta_seconds=round(eta, 1) if eta is not None else None)
        values.update(self.fields)
        print("history-progress " + " ".join(f"{k}={v}" for k, v in values.items()
                                             if v is not None), file=self.stream, flush=True)

    def _heartbeat(self):
        while not self._stop.wait(self.interval):
            with self._lock:
                if self._bar is None:
                    self._line()
                else:
                    self._bar.refresh()

    def __enter__(self):
        if self.ledger:
            self.ledger.event("stage_started", stage=self.stage, model=self.model, epoch=self.epoch,
                              total=self.total)
        if self.stream.isatty():
            from tqdm import tqdm
            description = "/".join(str(x) for x in (self.stage, self.model, self.epoch) if x is not None)
            self._bar = tqdm(total=self.total, desc=description, file=self.stream, mininterval=.5)
        else:
            self._line("started")
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()
        return self

    def update(self, amount=1, **fields):
        with self._lock:
            self.count += amount
            self.fields.update(fields)
            if self._bar is not None:
                self._bar.update(amount)
                if fields:
                    self._bar.set_postfix(self.fields, refresh=False)

    def __exit__(self, kind, value, traceback):
        self._stop.set()
        self._thread.join(timeout=1)
        status = "failed" if kind else "completed"
        if self._bar is not None:
            self._bar.close()
        else:
            self._line(status)
        if self.ledger:
            self.ledger.event("stage_" + status, stage=self.stage, model=self.model, epoch=self.epoch,
                              processed=self.count, total=self.total,
                              error=str(value) if kind else None)
        return False


def progress(iterable, *, total=None, **kwargs):
    if total is None and hasattr(iterable, "__len__"):
        total = len(iterable)
    with Progress(total=total, **kwargs) as tracker:
        for value in iterable:
            yield value
            tracker.update()


@contextmanager
def stage(name, **kwargs):
    with Progress(stage=name, **kwargs) as tracker:
        yield tracker
