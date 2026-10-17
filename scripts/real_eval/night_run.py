"""Inference-only temporal queue on prepared real RGB histories.

PREVIOUS and EARLIEST precede causal HISTORY. Annotation arrays never enter a
worker or memory; they remain in the preparation artifacts for the reporter.
"""
from __future__ import annotations

from collections import defaultdict
import importlib
import time
from pathlib import Path

import numpy as np

from . import night_common
from .real_gt_metrics import calibration
from .robustness_common import atomic_json, atomic_npz, digest, read_json, safe_path, sha256
from .robustness_run import (ALLOWED_METHODS, NATIVE_METHODS,
                             _method_contract, _source_rows, _verify_metadata,
                             model_identity, payload_sha256, validate_output,
                             _load_output, _event, _snapshot_gpu)
from .geoscd_real_retry import RECOVERABLE_SCORE_ERROR, WorkerProgress, make_geoscd_pool


HISTORY_METHODS = ("dinov2", "semantic_dino", "msdzip_abs", "msdzip_mod256")
DEFAULT_METHODS = night_common.DEFAULT_METHODS
MODE_ORDER = ("previous", "earliest", "history")


def _variant(method):
    return "current_train" if method.startswith("msdzip_") else "frozen"


def _threshold(manifest, method):
    value = calibration(manifest["source_summary"],
                        (method, _variant(method), "previous", "step_event"), .01)
    if value is None:
        raise ValueError(f"{method}: нет фиксированного synthetic VAL H0 1% порога обновления памяти")
    return float(value["threshold"])


def _jobs(manifest, methods, history_methods, modes, pilot=False,
          pilot_pairs=2, pilot_history_frames=3):
    sequences = manifest["sequences"]
    pair_keys = [(s["sequence_id"], f) for s in sequences for f in range(1, len(s["frames"]))]
    if not pair_keys:
        raise ValueError("Нет временных переходов в подготовленных последовательностях")
    if pilot:
        # First and late transitions cover different temporal distances. The
        # same representatives are used by every method, without using GT.
        positions = np.linspace(0, len(pair_keys)-1, min(pilot_pairs, len(pair_keys))).astype(int)
        pair_keys = [pair_keys[int(p)] for p in positions]
        history_sequence = max(sequences, key=lambda s: len(s["frames"]))
        history_keys = [(history_sequence["sequence_id"], f)
                        for f in range(1, min(len(history_sequence["frames"]), pilot_history_frames+1))]
    else:
        history_keys = pair_keys
    return [(method, mode, sequence, frame)
            for mode in MODE_ORDER if mode in modes
            for method in (history_methods if mode == "history" else methods)
            for sequence, frame in (history_keys if mode == "history" else pair_keys)]


def _caches(manifest, geoscd_run, reuse_run):
    source = Path(manifest["source_run"]).expanduser().resolve()
    candidates = [(source, row, "reused_source") for row in _source_rows(manifest)]
    for optional, origin in ((geoscd_run, "reused_geoscd"), (reuse_run, "reused_pilot")):
        if optional is None:
            continue
        root = Path(optional).expanduser().resolve()
        run = read_json(root / "run.json")
        contract = run.get("contract", run)
        if origin == "reused_geoscd" and Path(contract.get("source_run", "")).resolve() != source:
            raise ValueError("GeoSCD-кэш относится к другому исходному запуску")
        candidates.extend((root, read_json(p), origin) for p in sorted((root / "units").glob("*.json")))
    result = defaultdict(list)
    for root, row, origin in candidates:
        method, mode = row.get("method"), row.get("mode")
        if (row.get("status") != "completed" or row.get("domain") != "real"
                or mode not in ("previous", "earliest") or method not in ALLOWED_METHODS
                or row.get("variant", "frozen") != _variant(method)
                or (origin == "reused_geoscd" and method != "geoscd")):
            continue
        result[(method, mode, row["sequence_id"], int(row["frame"]))].append((root, row, origin))
    return result


def _input(root, sequence, frame, mode):
    previous = night_common.load_frame(root, sequence["frames"][frame-1])
    current = night_common.load_frame(root, sequence["frames"][frame])
    reference = (night_common.load_frame(root, sequence["frames"][0])
                 if mode == "earliest" else previous)
    return {"reference": reference["rgb"], "source": current["rgb"],
            "support": reference["support"] & current["support"]}, previous, current


def _saved_output(out, row, inputs, method, mode, previous_hash, threshold):
    path = safe_path(out, row["artifact_path"])
    if sha256(path) != row["artifact_sha256"]:
        raise ValueError(f"Изменилась сохранённая карта: {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    for name in ("reference", "source"):
        if not np.array_equal(arrays[name], inputs[name]):
            raise ValueError("Изменился RGB сохранённой единицы")
    if mode != "history":
        if payload_sha256(arrays) != payload_sha256(inputs):
            raise ValueError("Изменилась поддержка сохранённой пары")
        validate_output(arrays, inputs["support"], method)
    else:
        if (not np.array_equal(arrays["geometric_support"], inputs["support"])
                or np.any(arrays["support"] & ~inputs["support"])
                or row.get("previous_history_artifact") != previous_hash
                or row.get("memory_update_threshold") != threshold):
            raise ValueError("Нарушена причинная цепочка/поддержка/порог HISTORY")
        validate_output(arrays, arrays["support"], method)
    return arrays


def _projection(manifest, rows, methods, history_methods, modes):
    """Heuristic runtime range, explicitly not a statistical confidence interval."""
    entries, low_total, high_total = [], 0., 0.
    missing = []
    for mode in MODE_ORDER:
        if mode not in modes:
            continue
        for method in history_methods if mode == "history" else methods:
            measured = [r for r in rows if r["method"] == method and r["mode"] == mode
                        and r["status"] == "completed" and r.get("origin") == "inference"]
            times = [v for r in measured for v in r.get("scorer_seconds", []) if v > 0]
            n = sum(len(s["frames"])-1 for s in manifest["sequences"])
            # A local K cap does not cap the number of full-image references:
            # sparse visibility may retain an exemplar from every past frame.
            upper_calls = (sum((len(s["frames"])-1)*len(s["frames"])/2 for s in manifest["sequences"])
                           if mode == "history" else n)
            if not times:
                missing.append(f"{method}/{mode}")
                entries.append(dict(method=method, mode=mode, units=n, status="unmeasured"))
                continue
            init = max((r.get("initialization_seconds", 0.) for r in measured), default=0.)
            low = init + n*min(times)
            high = init + upper_calls*max(times)*1.5
            low_total += low
            high_total += high
            entries.append(dict(method=method, mode=mode, units=n, measured_units=len(measured),
                                measured_calls=len(times), initialization_seconds=init,
                                score_seconds_min=min(times), score_seconds_max=max(times),
                                projected_seconds_low=low, projected_seconds_high=high,
                                status="heuristic"))
    return {"scope": "Heuristic full selected workload, before reuse; not an ETA guarantee or confidence interval",
            "history_upper_bound": "all prior full frames; local exemplar cap does not bound global calls",
            "unmeasured": missing, "methods": entries,
            "estimated_hours_low": low_total/3600 if not missing else None,
            "estimated_hours_high": high_total/3600 if not missing else None,
            "measured_methods_hours_low": low_total/3600,
            "measured_methods_hours_high": high_total/3600}


def run(prepared_path, out_path, methods=None, history_methods=None,
        modes=MODE_ORDER, resume=False, retry_failed=False, device=None,
        max_hours=None, pilot=False, geoscd_run=None, pool_factory=None,
        pilot_pairs=2, pilot_history_frames=3, reuse_run=None):
    call_started = time.monotonic()
    prepared, manifest = night_common.load_prepared(prepared_path)
    prepared, out = Path(prepared).resolve(), Path(out_path).expanduser().resolve()
    methods = tuple(DEFAULT_METHODS if methods is None else methods)
    history_methods = tuple([m for m in HISTORY_METHODS if m in methods]
                            if history_methods is None else history_methods)
    if len(set(modes)) != len(modes) or set(modes)-set(MODE_ORDER):
        raise ValueError("Неверные или повторяющиеся режимы")
    modes = tuple(m for m in MODE_ORDER if m in modes)
    if (not methods or len(set(methods)) != len(methods) or set(methods)-ALLOWED_METHODS
            or len(set(history_methods)) != len(history_methods) or set(history_methods)-set(HISTORY_METHODS)
            or set(history_methods)-set(methods) or not modes):
        raise ValueError("Неверные методы/режимы; HISTORY поддерживает только четыре проверенных scorer")
    if retry_failed and not resume:
        raise ValueError("--retry-failed требует --resume")
    if (max_hours is not None and (not np.isfinite(max_hours) or max_hours < 0)
            or pilot_pairs < 1 or pilot_history_frames < 1):
        raise ValueError("Неверный лимит времени/размер pilot")
    protected = [prepared, Path(manifest["source_run"]).resolve()]
    protected += [Path(p).resolve() for p in (manifest.get("snapshot"), geoscd_run, reuse_run) if p]
    if any(out == p or out in p.parents or p in out.parents for p in protected):
        raise ValueError("Нужен отдельный выходной каталог")
    if out.exists() and not resume:
        raise FileExistsError(f"Каталог существует; продолжение с --resume: {out}")
    if resume and not (out / "run.json").is_file():
        raise ValueError("Нет run.json для продолжения")
    # Verify all prepared bytes and all weights before activating any worker.
    for number, sequence in enumerate(manifest["sequences"], 1):
        print(f"Проверка кропов {number}/{len(manifest['sequences'])}: {sequence['sequence_id']}", flush=True)
        shape = None
        for frame in sequence["frames"]:
            arrays = night_common.load_frame(prepared, frame)
            if shape is not None and arrays["support"].shape != shape:
                raise ValueError(f"Хронология не на общей решётке: {sequence['sequence_id']}")
            shape = arrays["support"].shape
    for filename, expected in manifest.get("metadata_sha256", {}).items():
        if sha256(Path(filename)) != expected:
            raise ValueError(f"Изменились исходные метаданные: {filename}")
    contracts, preflight_failures, thresholds = {}, {}, {}
    for method in methods:
        print(f"Проверка сохранённых настроек и весов: {method}", flush=True)
        try:
            contracts[method] = _method_contract(method, manifest, Path(manifest["source_run"]), device)
        except (ValueError, OSError, KeyError) as error:
            preflight_failures[method] = f"{type(error).__name__}: {error}"
            print(f"{method}: отказ до запуска worker: {error}", flush=True)
    if "history" in modes:
        for method in history_methods:
            if method in contracts:
                try:
                    thresholds[method] = _threshold(manifest, method)
                except (ValueError, KeyError) as error:
                    preflight_failures["history/"+method] = f"{type(error).__name__}: {error}"
    jobs = _jobs(manifest, methods, history_methods, modes, pilot, pilot_pairs, pilot_history_frames)
    paths = [Path(__file__), Path(night_common.__file__), Path(__file__).with_name("robustness_run.py"),
             Path(__file__).with_name("geoscd_real_retry.py"),
             Path(__file__).parents[2]/"facade_change/research/2026-10-14_temporal_history.py"]
    contract = {"schema_version": 1, "prepared_path": str(prepared / "manifest.json"),
                "manifest_sha256": sha256(prepared / "manifest.json"), "source_run": manifest["source_run"],
                "methods": list(methods), "history_methods": list(history_methods), "modes": list(modes),
                "models": contracts, "training_exposure": night_common.training_exposure(manifest, contracts),
                "history_thresholds": thresholds, "preflight_failures": preflight_failures,
                "pilot": bool(pilot), "pilot_pairs": pilot_pairs, "pilot_history_frames": pilot_history_frames,
                "jobs": [list(j) for j in jobs], "code_sha256": {str(p): sha256(p) for p in paths}}
    if resume and read_json(out / "run.json").get("contract") != contract:
        raise ValueError("Изменился контракт продолжения: данные, модели, код или очередь")
    # A pilot is a timing measurement: cached maps would erase the actual
    # initialization/inference costs and yield an unusable optimistic estimate.
    cached = {} if pilot else _caches(manifest, geoscd_run, reuse_run)
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Этот выходной каталог уже занят") from error
        return _process(prepared, manifest, out, contract, cached, retry_failed, pool_factory, max_hours, call_started)


def _process(prepared, manifest, out, contract, cached, retry_failed, pool_factory, max_hours, started):
    for name in ("maps", "units", "logs", "method_provenance"):
        (out/name).mkdir(exist_ok=True)
    previous_run = read_json(out/"run.json") if (out/"run.json").is_file() else {}
    run_record = {"schema_version": 1, "kind": "real_temporal_run", "contract": contract,
                  "prepared_path": contract["prepared_path"], "manifest_sha256": contract["manifest_sha256"],
                  "methods": contract["methods"], "history_methods": contract["history_methods"],
                  "modes": contract["modes"], "config_hash": digest(contract),
                  "started_at": previous_run.get("started_at", time.time()), "status": "running",
                  "budget_scope": "wall time of this call, including preflight; checked between units",
                  "scope": "GT-enriched real chronological crops; fixed models and synthetic VAL thresholds"}
    atomic_json(out/"run.json", run_record)
    _event(out, "resumed" if previous_run else "started", config_hash=digest(contract))
    jobs = contract["jobs"]
    sequences = {s["sequence_id"]: s for s in manifest["sequences"]}
    TemporalMemory = importlib.import_module("facade_change.research.2026-10-14_temporal_history").TemporalMemory
    rows, method_failures, blocked_sequences = [], {}, set()
    for key, error in contract["preflight_failures"].items():
        groups = ([tuple(key.split("/"))] if "/" in key
                  else [(mode, key) for mode in contract["modes"]])
        for group in groups:
            method_failures[group] = error
        _event(out, "preflight_method_failed", method=key, error=error)
    status, fatal, pool, active_key, active_metadata = "completed", None, None, None, None
    memory, history_key, previous_hash = None, None, None

    def save(row, arrays=None):
        if arrays is not None:
            path = out/"maps"/f"{row['unit_id']}.npz"
            row.update(artifact_path=str(path.relative_to(out)), artifact_sha256=atomic_npz(path, arrays))
        atomic_json(out/"units"/f"{row['unit_id']}.json", row)
        _event(out, "unit_"+row["status"], unit_id=row["unit_id"], method=row["method"],
               mode=row["mode"], sequence_id=row["sequence_id"], frame=row["frame"])
        rows.append(row)
        atomic_json(out/"progress.json", {"processed": len(rows), "total": len(jobs),
                    "completed": sum(r["status"] == "completed" for r in rows),
                    "failed": sum(r["status"] == "failed" for r in rows), "latest": row})

    try:
        print(f"Очередь: {len(sequences)} историй; {len(jobs)} единиц; PREVIOUS → EARLIEST → HISTORY.", flush=True)
        for method, mode, sequence_id, frame in jobs:
            group = (mode, method)
            if group in method_failures or (method, sequence_id) in blocked_sequences and mode == "history":
                continue
            sequence = sequences[sequence_id]
            inputs, previous, current = _input(prepared, sequence, frame, mode)
            threshold = contract["history_thresholds"].get(method)
            if mode == "history" and history_key != (method, sequence_id):
                history_key, previous_hash = (method, sequence_id), None
                memory = TemporalMemory(max_exemplars=3)
                first = night_common.load_frame(prepared, sequence["frames"][0])
                memory.score_then_update(first["rgb"], first["support"], lambda *args: None, threshold)
            identifier = night_common.unit_id(method, mode, sequence_id, frame)
            row = {"unit_id": identifier, "method": method, "variant": _variant(method),
                   "domain": "real", "split": sequence["split"], "building_id": sequence["building_id"],
                   "sequence_id": sequence_id, "frame": frame, "mode": mode,
                   "target": "earliest_state" if mode == "earliest" else "step_event",
                   "reference_frame": 0 if mode == "earliest" else frame-1,
                   "reference_year": sequence["frames"][0 if mode == "earliest" else frame-1].get("year"),
                   "source_year": sequence["frames"][frame].get("year"),
                   "input_sha256": payload_sha256(inputs),
                   "output_kind": "native_mask" if method in NATIVE_METHODS else "score"}
            if mode == "history":
                row.update(previous_history_artifact=previous_hash, memory_update_threshold=threshold)
            marker = out/"units"/f"{identifier}.json"
            saved = read_json(marker) if marker.is_file() else None
            label = f"[{len(rows)+1}/{len(jobs)}] {method} {mode} {sequence_id} кадр {frame}"
            if saved:
                if any(saved.get(k) != v for k, v in row.items()):
                    raise ValueError(f"Изменилась сохранённая единица/цепочка: {identifier}")
                if saved["status"] == "completed":
                    arrays = _saved_output(out, saved, inputs, method, mode, previous_hash, threshold)
                    if mode == "history":
                        memory = TemporalMemory.from_state_dict({k[8:]: v for k, v in arrays.items() if k.startswith("memory__")})
                        if int(memory.state_dict()["next_frame"]) != frame+1:
                            raise ValueError("Память не соответствует номеру сохранённого кадра")
                        previous_hash = saved["artifact_sha256"]
                    rows.append(saved)
                    print(label+" — уже готово", flush=True)
                    continue
                if saved["status"] == "failed" and not retry_failed:
                    rows.append(saved)
                    if mode == "history":
                        blocked_sequences.add((method, sequence_id))
                    continue
            if max_hours is not None and time.monotonic()-started >= max_hours*3600:
                status = "budget_exhausted"
                break
            if active_key != group:
                if pool is not None:
                    pool.release()
                pool, active_metadata, active_key = None, None, group
            print(label+" — обработка", flush=True)
            unit_start = time.monotonic()
            model = contract["models"][method]
            expected, options = model["expected_metadata"], model["options"]
            reused = False
            candidates = list(cached.get((method, mode, sequence_id, frame), []))
            if mode == "earliest" and frame == 1 and not contract["pilot"]:
                # The first transition is identical in both pair modes. Keep
                # the EARLIEST target/record, but reuse an already verified
                # PREVIOUS prediction rather than running the model twice.
                previous_id = night_common.unit_id(method, "previous", sequence_id, frame)
                previous_marker = out/"units"/f"{previous_id}.json"
                if previous_marker.is_file():
                    previous_record = read_json(previous_marker)
                    if (previous_record.get("status") == "completed"
                            and previous_record.get("method") == method
                            and previous_record.get("mode") == "previous"
                            and previous_record.get("sequence_id") == sequence_id
                            and previous_record.get("frame") == 1):
                        candidates.insert(0, (out, previous_record, "reused_previous_first_pair"))
            for cache_root, cache_row, origin in candidates:
                try:
                    if cache_row["building_id"] != sequence["building_id"] or cache_row["split"] != sequence["split"]:
                        raise ValueError("Кэш относится к другому фасаду/split")
                    _verify_metadata(expected, cache_row.get("method_metadata", {}), method)
                    arrays = _load_output(cache_root, cache_row, inputs, method)
                except (ValueError, OSError, KeyError) as error:
                    _event(out, "cache_rejected", unit_id=identifier, error=str(error), cache=str(cache_root))
                    continue
                output = {k: arrays[k] for k in ("scores", "native_prediction", "raw_scores") if k in arrays}
                row.update(status="completed", origin=origin, method_metadata=cache_row.get("method_metadata", {}),
                           source_result_unit_id=cache_row["unit_id"], source_result_sha256=cache_row["artifact_sha256"],
                           elapsed_seconds=time.monotonic()-unit_start, initialization_seconds=0., scorer_seconds=[])
                save(row, {**inputs, **output})
                reused = True
                print(label+" — сохранённая карта переиспользована", flush=True)
                break
            if reused:
                continue
            phase, init_seconds, score_times = "initialization", 0., []
            try:
                if pool is None:
                    factory = pool_factory
                    if factory is None:
                        from facade_change.methods.remote import RemotePool, _json_value
                        factory = make_geoscd_pool(RemotePool, _json_value)
                    pool = factory(log_dir=out/"logs")
                    _snapshot_gpu(out)
                if active_metadata is None:
                    init_start = time.monotonic()
                    with WorkerProgress(pool, label+" — инициализация"):
                        active_metadata = pool.activate(method, options)
                    init_seconds = time.monotonic()-init_start
                    _verify_metadata(expected, active_metadata, method)
                    provenance = out/"method_provenance"/f"{method}.json"
                    identity = model_identity(active_metadata)
                    if provenance.is_file() and read_json(provenance) != identity:
                        raise ValueError("Изменилась модель при продолжении")
                    atomic_json(provenance, identity)
                    _event(out, "method_initialized", method=method, mode=mode, seconds=init_seconds)
                phase = "score"

                def score(reference, source, support):
                    call_start = time.monotonic()
                    with WorkerProgress(pool, label+" — расчёт"):
                        inferred = pool.score(method, options, reference, source, support)
                    score_times.append(time.monotonic()-call_start)
                    _verify_metadata(active_metadata, inferred.get("metadata", active_metadata), method)
                    validate_output(inferred, support, method)
                    return inferred

                if mode == "history":
                    result = memory.score_then_update(current["rgb"], current["support"],
                        lambda a, b, s: score(a, b, s)["scores"], threshold)
                    support = inputs["support"] & result["support"]
                    output = {"scores": np.where(support, result["scores"], np.nan).astype(np.float32),
                              "support": support, "geometric_support": inputs["support"],
                              "reference_count": result["reference_count"],
                              "history_reference_frame": result["reference_frame"],
                              "accepted_change": result["accepted_change"],
                              **{"memory__"+k: v for k, v in memory.state_dict().items()}}
                    validate_output(output, support, method)
                else:
                    inferred = score(inputs["reference"], inputs["source"], inputs["support"])
                    output = {k: inferred[k] for k in ("scores", "native_prediction", "raw_scores") if k in inferred}
                row.update(status="completed", origin="inference", method_metadata=active_metadata,
                           elapsed_seconds=time.monotonic()-unit_start,
                           initialization_seconds=init_seconds, scorer_seconds=score_times,
                           comparison_count=len(score_times))
                save(row, {**inputs, **output})
                if mode == "history":
                    previous_hash = row["artifact_sha256"]
                print(label+f" — готово, {row['elapsed_seconds']:.1f} с", flush=True)
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                row.update(status="failed", error=message, failure_phase=phase, origin="inference",
                           elapsed_seconds=time.monotonic()-unit_start, initialization_seconds=init_seconds,
                           scorer_seconds=score_times, method_metadata=active_metadata or expected,
                           stderr_log=str(getattr(pool, "stderr_path", "")))
                save(row)
                print(label+" — отказ: "+message, flush=True)
                if mode == "history":
                    blocked_sequences.add((method, sequence_id))
                recoverable = method == "geoscd" and phase == "score" and str(error).startswith(
                    "Inference worker failed during score: "+RECOVERABLE_SCORE_ERROR+"; stderr log: ")
                if not recoverable:
                    method_failures[group] = message
                if pool is not None and not getattr(pool, "last_worker_retained", False):
                    pool.release()
                    pool, active_metadata = None, None
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as error:
        status, fatal = "failed_validation", f"{type(error).__name__}: {error}"
    finally:
        if pool is not None:
            pool.release()
        failed = sum(r["status"] == "failed" for r in rows)
        if status == "completed" and (failed or method_failures):
            status = "completed_with_failures"
        summary = {"schema_version": 1, "status": status, "total": len(jobs),
                   "completed": sum(r["status"] == "completed" for r in rows), "failed": failed,
                   "unprocessed": len(jobs)-len(rows), "results": rows, "error": fatal,
                   "method_failures": {"/".join(k): v for k, v in method_failures.items()},
                   "elapsed_seconds_this_call": time.monotonic()-started}
        atomic_json(out/"summary.json", summary)
        atomic_json(out/"progress.json", {"status": status, "processed": len(rows), "total": len(jobs),
                    "completed": summary["completed"], "failed": failed,
                    "unprocessed": summary["unprocessed"], "latest": rows[-1] if rows else None})
        timing = _projection(manifest, rows, contract["methods"], contract["history_methods"], contract["modes"])
        atomic_json(out/"timing.json", timing)
        if contract["pilot"]:
            print("Pilot: метод/режим | вызов scorer, с | инициализация, с | полный объём, ч (оценка)", flush=True)
            for estimate in timing["methods"]:
                label = f"{estimate['method']}/{estimate['mode']}"
                if estimate["status"] == "unmeasured":
                    print(label+" | нет измерения", flush=True)
                else:
                    print(label+f" | {estimate['score_seconds_min']:.2f}–{estimate['score_seconds_max']:.2f}"
                          f" | {estimate['initialization_seconds']:.1f}"
                          f" | {estimate['projected_seconds_low']/3600:.2f}–{estimate['projected_seconds_high']/3600:.2f}", flush=True)
            if not timing["unmeasured"]:
                print(f"Оценка всей очереди до переиспользования: {timing['estimated_hours_low']:.2f}–"
                      f"{timing['estimated_hours_high']:.2f} ч; это ориентир, не гарантия.", flush=True)
        run_record.update(status=status, finished_at=time.time())
        atomic_json(out/"run.json", run_record)
        _event(out, status, completed=summary["completed"], failed=failed, unprocessed=summary["unprocessed"])
    if fatal:
        raise ValueError(fatal)
    return summary
