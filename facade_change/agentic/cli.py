"""Local pair analysis, causal retrieval and bounded Qwen tools (2026-11-08)."""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

from ..io import read_json, write_json
from .config import ROOT, defaults, import_method_options, load_config, write_new_config


HEARTBEAT_SECONDS = 15.0


class TerminalProgress:
    """A compact stderr view of existing events; it never controls execution."""
    def __init__(self, quiet=False, stream=None, heartbeat_interval=None):
        self.quiet, self.stream = quiet, stream if stream is not None else sys.stderr
        self.interval = HEARTBEAT_SECONDS if heartbeat_interval is None else heartbeat_interval
        self.started = time.monotonic()
        self.case = self.mode = self.task = None
        self.stage = self.pending_model = None
        self.stage_started = self.last_output = self.started
        self._lock, self._stop = threading.RLock(), threading.Event()
        self._thread = None

    @staticmethod
    def brief(value, limit=350):
        if isinstance(value, dict):
            value = ": ".join(str(value[k]) for k in ("code", "message", "error") if value.get(k))
        elif isinstance(value, (list, tuple)):
            value = "; ".join(TerminalProgress.brief(v, 150) for v in value[:3])
        text = re.sub(r"[\x00-\x1f\x7f]", " ", str(value or ""))
        return text if len(text) <= limit else text[:limit] + "…"

    def _write(self, message):
        context = " / ".join(str(v) for v in (self.case, self.mode) if v) or self.task or "agentic"
        try:
            print(f"[agentic +{time.monotonic()-self.started:.1f}s] {context}: {message}",
                  file=self.stream, flush=True)
            self.last_output = time.monotonic()
        except (OSError, ValueError):
            self._stop.set()  # A closed diagnostic stream cannot fail inference.

    def _begin(self, stage):
        self.stage, self.stage_started = stage, time.monotonic()
        self._write(stage)

    def _heartbeat(self):
        while not self._stop.wait(self.interval):
            with self._lock:
                if self.stage and time.monotonic() - self.last_output >= self.interval:
                    self._write(f"ожидание: {self.stage}; прошло {time.monotonic()-self.stage_started:.1f}s")

    @staticmethod
    def _details(values):
        parts = []
        for name in ("decision", "reason", "error", "limitations"):
            if values.get(name):
                parts.append(f"{name}={TerminalProgress.brief(values[name])}")
        duration = values.get("elapsed_seconds")
        if isinstance(duration, (int, float)):
            parts.append(f"{duration:.1f}s")
        return "; ".join(parts)

    def __call__(self, event, **values):
        if self.quiet or self._stop.is_set():
            return
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._heartbeat, name="agentic-progress", daemon=True)
                self._thread.start()
            if values.get("case_id"):
                self.case = values["case_id"]
            if values.get("mode"):
                self.mode = values["mode"]
            if event == "command_started":
                self.task = values["task"]
                self._begin(f"начало; результаты: {values.get('out')}")
            elif event == "history_started":
                self._begin("импорт истории и проверка сохранённых данных")
            elif event == "case_started":
                self.mode = None
                self._begin(f"подготовка случая {values.get('index', '?')}/{values.get('total', '?')}")
            elif event in {"case_prepared", "case_failed"}:
                self.stage = None
                self._write(f"подготовка: {values.get('status', 'error' if event == 'case_failed' else 'completed')}; "
                            + self._details(values))
            elif event == "mode_started":
                self._begin("начало режима")
            elif event in {"agent_model_requested", "pool_model_requested"}:
                role, op = values.get("role", "qwen"), values.get("op", "generate")
                if event == "pool_model_requested" and self.pending_model == (role, op, "agent"):
                    return  # Same call is also represented by the reasoning trace.
                self.pending_model = (role, op, "agent" if event.startswith("agent_") else "pool")
                self._begin(f"модель {role}: запрос {op}")
            elif event in {"agent_model_returned", "pool_model_returned"}:
                role, op = values.get("role", "qwen"), values.get("op", "generate")
                if event == "pool_model_returned" and self.pending_model == (role, op, "agent"):
                    return
                self.stage = self.pending_model = None
                usage = values.get("usage") or {}
                tokens = " ".join(f"{key}={usage[key]}" for key in ("input_tokens", "output_tokens")
                                  if usage.get(key) is not None)
                if usage.get("incomplete") or usage.get("unobserved_worker_cost"):
                    tokens = "токены: учёт неполный"
                self._write(f"модель {role}: {values.get('status', 'unknown')} {tokens}; " + self._details(values))
            elif event == "agent_tool_requested":
                self._begin(f"инструмент {values.get('name')}: вызов")
            elif event == "agent_tool_returned":
                self.stage = None
                self._write(f"инструмент {values.get('name')}: {values.get('status')}; " + self._details(values))
            elif event == "agent_validation_error":
                self._write("ошибка валидации: " + self.brief(values.get("error")))
            elif event == "resource_missing":
                self._write(f"ресурс отсутствует: {values.get('method')}/{values.get('field')}: "
                            + self.brief(values.get("path")))
            elif event in {"mode_completed", "mode_failed"}:
                self.stage = self.pending_model = None
                self._write(f"результат: {values.get('status', 'error')}; " + self._details(values))
                self.mode = None
            elif event == "report_started":
                self.case = self.mode = None
                self._begin("формирование HTML-отчёта")
            elif event in {"completed", "completed_with_errors", "failed", "interrupted"}:
                self.stage = self.pending_model = None
                self.case = self.mode = None
                summary = values.get("summary") or {}
                counts = summary.get("outcome_counts")
                if counts is not None:
                    self._write("итог: " + " ".join(f"{key}={counts.get(key, 0)}"
                                for key in ("completed", "abstained", "error"))
                                + f" ({summary.get('count_unit', 'результаты')}); " + self._details(summary))
                    self._write("HTML: " + str(summary.get("report_path") or "не создаётся для этой команды"))
                elif values.get("error"):
                    self._write(f"{event}: " + self.brief(values["error"]))

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.)


def parser():
    cli = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    tasks = cli.add_subparsers(dest="task", required=True)
    configure = tasks.add_parser("configure", help="Write a new explicit local configuration")
    configure.add_argument("--out", required=True)
    configure.add_argument("--methods-config")
    configure.add_argument("--methods", nargs="+")
    configure.add_argument("--semantic-run")
    configure.add_argument("--fusion-run")
    configure.add_argument("--fusion-variant", default="stack--previous")
    configure.add_argument("--gpu-mode", choices=("sequential", "resident"), default="sequential")
    configure.add_argument("--retrieval-encoder", choices=("jina", "dino"))
    configure.add_argument("--alignment-python")
    configure.add_argument("--detector-python")
    for name in ("run", "history", "check", "index", "search", "knowledge-import", "smoke"):
        task = tasks.add_parser(name)
        task.add_argument("--config")
        if name in {"run", "history", "smoke", "check"}:
            task.add_argument("--quiet", action="store_true", help="Disable terminal progress; final JSON stays on stdout")
        if name != "search":
            task.add_argument("--out", required=True)
        if name == "run":
            task.add_argument("--cases", help="JSON document with an explicit cases list")
            task.add_argument("--manifest")
            task.add_argument("--reference-id")
            task.add_argument("--source-id")
            task.add_argument("--reference")
            task.add_argument("--source")
            task.add_argument("--reference-date")
            task.add_argument("--source-date")
            task.add_argument("--building-id")
            task.add_argument("--view-id")
            task.add_argument("--split", choices=("train", "val", "test", "unassigned"), default="unassigned")
            task.add_argument("--region-id")
            task.add_argument("--bbox", nargs=4, type=int)
            task.add_argument("--alignment-run")
            task.add_argument("--modes", nargs="+", choices=("pair", "fixed_context", "agent", "none"))
        elif name == "history":
            task.add_argument("--histories", required=True)
            task.add_argument("--track-id", required=True)
            task.add_argument("--min-dates", type=int, default=3)
            task.add_argument("--available-at", required=True,
                              help="ISO date, or explicit retrospective assumption observation_date")
            task.add_argument("--features", help="Existing frozen DINO history features, no extraction")
        elif name == "check":
            task.add_argument("--roles", nargs="+", choices=("qwen", "sam3", "jina", "dino"), default=["qwen", "sam3"])
            task.add_argument("--load-model", action="store_true")
        elif name == "index":
            task.add_argument("--encoder", choices=("jina", "dino"), default="jina")
            task.add_argument("--observation-id", action="append")
            task.add_argument("--cutoff", required=True)
        elif name == "search":
            task.add_argument("--query", required=True)
            task.add_argument("--cutoff", required=True)
            task.add_argument("--limit", type=int, default=5)
            task.add_argument("--encoder", choices=("lexical", "jina"), default="lexical")
            task.add_argument("--out", help="Required for a model-backed query and its provenance")
            task.add_argument("--exclude-building", action="append", default=[])
        elif name == "knowledge-import":
            task.add_argument("--fragments", required=True, help="Sourced documents JSON, never an invented bibliography")
        elif name == "smoke":
            task.add_argument("--kind", choices=("cpu", "models", "handoff", "recovery", "context"), default="cpu")
    report = tasks.add_parser("report")
    report.add_argument("--run", required=True)
    report.add_argument("--out")
    comparison = tasks.add_parser("compare")
    comparison.add_argument("--runs", nargs="+", required=True)
    comparison.add_argument("--cases", required=True, help="Frozen input_manifest.json from the analysis run")
    comparison.add_argument("--labels", help="Explicit offline temporal-region labels; never forwarded to inference")
    comparison.add_argument("--out", required=True)
    return cli


def _pairs(arguments):
    if arguments.cases:
        if arguments.reference or arguments.source or arguments.manifest:
            raise ValueError("Use either --cases, --manifest or two raw image paths")
        document = read_json(arguments.cases)
        pairs = document if isinstance(document, list) else document.get("cases")
        if not isinstance(pairs, list):
            raise ValueError("--cases requires a list or {'cases': [...]} document")
        return pairs
    if arguments.manifest:
        if not arguments.reference_id or not arguments.source_id:
            raise ValueError("--manifest requires --reference-id and --source-id")
        pair = {"manifest": arguments.manifest, "reference_id": arguments.reference_id,
                "source_id": arguments.source_id}
    else:
        required = {key: getattr(arguments, key) for key in (
            "reference", "source", "reference_date", "source_date", "building_id", "view_id")}
        missing = ["--" + key.replace("_", "-") for key, value in required.items() if not value]
        if missing:
            raise ValueError("New pair requires explicit inputs: " + ", ".join(missing))
        pair = {"building_id": arguments.building_id, "view_id": arguments.view_id, "split": arguments.split}
        for role in ("reference", "source"):
            date = getattr(arguments, role + "_date")
            pair[role] = {"path": str(Path(getattr(arguments, role)).expanduser().resolve()),
                          "date": date, "date_status": "confirmed", "available_at": date}
    for key, value in (("region_id", arguments.region_id), ("bbox_xyxy", arguments.bbox),
                       ("alignment_run", arguments.alignment_run)):
        if value is not None:
            pair[key] = value
    return [pair]


def check(config, out, roles, load_model=False, progress=None):
    from .models import ModelPool
    from .runner import new_run, complete_run
    check_started = time.monotonic()
    journal = new_run(out, {"configuration": config, "roles": roles, "load_model": load_model},
                      "agentic_preflight", progress=progress)
    pool = ModelPool(config["models"], journal.out / "worker_logs", gpu_mode="sequential",
                     timeout=config["worker_timeout"])
    observed_pool = hasattr(pool, "progress")
    if observed_pool:
        pool.progress = journal.event
    try:
        results = {}
        for role in roles:
            started = time.monotonic()
            if not observed_pool:
                journal.event("pool_model_requested", role=role, op="check")
            results[role] = pool.call(role, {"op": "check", "load_model": load_model})
            if not observed_pool:
                journal.event("pool_model_returned", role=role, op="check", status=results[role].get("status"),
                              error=results[role].get("error"), usage=results[role].get("usage", {}),
                              elapsed_seconds=time.monotonic()-started)
        missing = []
        for method in config["methods"]:
            for key, value in config["method_options"].get(method, {}).items():
                if key in {"worker_python", "checkpoint_path", "training_run", "source_root", "dino_root", "dino_checkpoint"}:
                    if not Path(value).expanduser().exists():
                        missing.append({"method": method, "field": key, "path": value})
        status = "completed" if all(row.get("status") == "ok" for row in results.values()) and not missing else "failed"
        summary = {"status": status, "roles": results, "missing_method_resources": missing,
                   "model_execution": load_model,
                   "outcome_counts": {"completed": sum(row.get("status") == "ok" for row in results.values()),
                                      "abstained": 0, "error": sum(row.get("status") != "ok" for row in results.values()) + len(missing)},
                   "count_unit": "проверки ресурсов", "report_path": None,
                   "elapsed_seconds": time.monotonic()-check_started,
                   "scope": "Imports/cache availability; not real facade accuracy or full context validation"}
        for item in missing:
            journal.event("resource_missing", **item)
        write_json(journal.out / "summary.json", summary)
        complete_run(journal, status, summary=summary)
        return summary
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        pool.close()
        journal.close()


def _analysis_task(arguments, config, progress):
    # Legacy history verification writes progress to Python stdout. Redirect
    # only the parent operation, leaving the final JSON and subprocess pipes alone.
    sink = open(os.devnull, "w") if arguments.quiet else contextlib.nullcontext(sys.stderr)
    with sink as stream, contextlib.redirect_stdout(stream):
        if arguments.task == "run":
            from .runner import run_cases
            if arguments.modes:
                if "none" in arguments.modes and len(arguments.modes) != 1:
                    raise ValueError("--modes none cannot be combined with other modes")
                config["modes"] = [] if arguments.modes == ["none"] else arguments.modes
            return run_cases(config, _pairs(arguments), arguments.out, progress=progress)
        if arguments.task == "smoke":
            from .acceptance import run_smoke
            return run_smoke(config, arguments.out, arguments.kind, progress=progress)
        if arguments.task == "check":
            return check(config, arguments.out, arguments.roles, arguments.load_model, progress=progress)
        from .pipeline import pairs_from_history
        from .retrieval import SearchIndex
        from .store import RegionStore
        from .runner import run_cases
        progress("history_started")
        with RegionStore(config["store_root"]) as store:
            store.import_histories(arguments.histories, available_at=arguments.available_at)
            if arguments.features:
                SearchIndex(store).import_history_features(arguments.features)
        pairs = pairs_from_history(arguments.histories, arguments.track_id, arguments.min_dates,
                                   available_at=arguments.available_at)
        return run_cases(config, pairs, arguments.out, progress=progress)


def main(argv=None):
    arguments = parser().parse_args(argv)
    progress = None
    try:
        if arguments.task in {"run", "history", "smoke", "check"}:
            progress = TerminalProgress(quiet=arguments.quiet)
            progress("command_started", task=arguments.task, out=arguments.out)
        if arguments.task == "configure":
            config = import_method_options(defaults(), arguments.methods_config, arguments.methods, arguments.semantic_run)
            config["gpu_mode"] = arguments.gpu_mode
            config["retrieval_encoder"] = arguments.retrieval_encoder
            if arguments.detector_python:
                for method in config["methods"]:
                    config["method_options"].setdefault(method, {})["worker_python"] = arguments.detector_python
            alignment_python = arguments.alignment_python or arguments.detector_python or next(
                (options["worker_python"] for options in config["method_options"].values() if options.get("worker_python")), None)
            if alignment_python:
                config["alignment"]["worker_python"] = alignment_python
            if "dinov2" in config["method_options"]:
                dino = config["method_options"]["dinov2"]
                config["models"]["dino"] = {"python": dino.get("worker_python", sys.executable),
                    **{key: dino[key] for key in ("source_root", "checkpoint_path", "device", "trust_checkpoint") if key in dino}}
            if arguments.fusion_run:
                config["fusion"] = {"run": str(Path(arguments.fusion_run).expanduser().resolve()),
                                    "variant": arguments.fusion_variant, "threshold": "f1"}
            print(write_new_config(arguments.out, config))
            return 0
        if arguments.task == "report":
            from .report import render_report
            print(render_report(arguments.run, arguments.out))
            return 0
        if arguments.task == "compare":
            from .evaluation import compare_runs
            result = compare_runs(arguments.runs, arguments.out, arguments.cases, arguments.labels)
        else:
            config = load_config(arguments.config)
            if progress is not None:
                result = _analysis_task(arguments, config, progress)
            elif arguments.task == "search":
                from .retrieval import KnowledgeBase, SearchIndex
                from .store import RegionStore
                with RegionStore(config["store_root"]) as store:
                    if arguments.encoder == "jina":
                        if not arguments.out:
                            raise ValueError("Jina query requires a new --out for cost and provenance")
                        return _joint_search(config, arguments, store)
                    result = {"observations": SearchIndex(store).text(arguments.query, arguments.cutoff, limit=arguments.limit,
                                                                                exclude_buildings=arguments.exclude_building),
                        "knowledge": KnowledgeBase(config["knowledge_root"]).search(arguments.query, arguments.cutoff, limit=arguments.limit)}
            elif arguments.task in {"index", "knowledge-import"}:
                result = _index_or_import(config, arguments)
            else:
                raise ValueError("Unknown task")
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if isinstance(result, dict) and (result.get("status") in {"failed", "error"} or result.get("errors")):
            return 1
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"agentic: {exc}", file=sys.stderr)
        return 2
    finally:
        if progress is not None:
            progress.close()


def _index_or_import(config, arguments):
    from .models import ModelPool
    from .retrieval import KnowledgeBase, SearchIndex
    from .runner import new_run, complete_run
    from .store import RegionStore, causally_available
    journal = new_run(arguments.out, {"configuration": config, "task": arguments.task}, "agentic_index")
    try:
        if arguments.task == "knowledge-import":
            document = read_json(arguments.fragments)
            documents = document if isinstance(document, list) else document.get("documents", [document])
            knowledge = KnowledgeBase(config["knowledge_root"])
            result = {"documents": [knowledge.import_document(row) for row in documents]}
        else:
            results = []
            with RegionStore(config["store_root"]) as store:
                index = SearchIndex(store)
                wanted = set(arguments.observation_id or ())
                observations = [row for row in store.list_observations()
                    if (not wanted or row["observation_id"] in wanted) and causally_available(row, arguments.cutoff)]
                if not observations:
                    raise ValueError("No causally available observations selected for embedding")
                if wanted - {row["observation_id"] for row in observations}:
                    raise ValueError("Requested observations missing or unavailable at cutoff")
                pool = ModelPool(config["models"], journal.out / "worker_logs", gpu_mode="sequential",
                                 timeout=config["worker_timeout"], cache_dir=Path(config["cache_root"]) / "models")
                try:
                    for row in observations:
                        payload = {"op": "embed", "image_paths": [row["crop_path"]], "geometry": row["geometry"]}
                        if arguments.encoder == "dino":
                            payload["support_paths"] = [row["support_path"]]
                        response = pool.call(arguments.encoder, payload)
                        if response.get("status") != "ok":
                            raise RuntimeError(f"Embedding unavailable, no substitute vector generated: {response.get('error')}")
                        embedded = index.add_embedding(row["observation_id"], response["vectors"][0], response["space"])
                        results.append({"observation_id": row["observation_id"], "embedding": embedded,
                                        "usage": response.get("usage"), "cache_hit": response.get("cache_hit")})
                finally:
                    pool.close()
                result = {"observations": len(results), "results": results, "corpus_version": index.corpus_version}
        write_json(journal.out / "summary.json", result)
        complete_run(journal, "completed")
        return result
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()


def _joint_search(config, arguments, store):
    from .models import ModelPool
    from .retrieval import SearchIndex
    from .runner import new_run, complete_run
    journal = new_run(arguments.out, {"configuration": config, "query": arguments.query,
        "cutoff": arguments.cutoff, "exclude_buildings": arguments.exclude_building}, "agentic_joint_search")
    pool = ModelPool(config["models"], journal.out / "worker_logs", gpu_mode="sequential",
                     timeout=config["worker_timeout"], cache_dir=Path(config["cache_root"]) / "models")
    try:
        response = pool.call("jina", {"op": "embed", "texts": [arguments.query], "task": "retrieval.query"})
        if response.get("status") != "ok":
            raise RuntimeError(f"Jina unavailable; lexical search remains separate: {response.get('error')}")
        result = SearchIndex(store).similar(response["vectors"][0], response["space"], arguments.cutoff,
            allowed_splits=("train",), exclude_buildings=arguments.exclude_building, limit=arguments.limit)
        result["embedding_usage"] = response.get("usage", {})
        result["encoder_provenance"] = response.get("provenance", {})
        write_json(journal.out / "summary.json", result)
        complete_run(journal, "completed")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        pool.close()
        journal.close()
