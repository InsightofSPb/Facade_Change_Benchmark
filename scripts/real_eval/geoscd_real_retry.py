#!/usr/bin/env python3
"""Отдельный возобновляемый GeoSCD на сохранённых RGB-парах выбранных реальных историй."""
import argparse
import copy
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np


PREVIOUS_RUNNER_SHA256 = "c1a30439d1f2daaf90e0fb1d0cf6cecf52c4ded5811c112effe7c1ba8e80ba08"
MIGRATED_RUNNER_SHA256 = "f58ebf10f7cd726c918cd387dfab33dd4a9629b99dd77218623f3d2262cfcfca"
KNOWN_PREVIOUS_RUNNER_SHA256 = {PREVIOUS_RUNNER_SHA256, MIGRATED_RUNNER_SHA256}
RECOVERABLE_SCORE_ERROR = "ValueError: No finite valid SAM correspondence similarities"


def make_geoscd_pool(remote_pool_class, json_value):
    """Keep the loaded GeoSCD only after its one documented, cleaned-up pair failure."""
    class GeoSCDPool(remote_pool_class):
        def _request(self, operation, **payload):
            self.last_worker_retained = False
            if self.process is None:
                raise self._error("is not active")
            self._request_id += 1
            request_id = self._request_id
            message = {"id": request_id, "operation": operation, **payload}
            try:
                self.process.stdin.write(json.dumps(message, default=json_value, allow_nan=False) + "\n")
                self.process.stdin.flush()
            except (OSError, BrokenPipeError) as exc:
                self._dispose()
                raise self._error("closed its input unexpectedly") from exc
            try:
                line = self._responses.get(timeout=self._active_timeout)
            except queue.Empty as exc:
                self._dispose()
                raise self._error(f"timed out after {self._active_timeout:g}s during {operation}") from exc
            if line is None:
                self._dispose()
                raise self._error(f"exited without a response during {operation}")
            try:
                response = json.loads(line)
                if not isinstance(response, dict) or response.get("id") != request_id:
                    raise ValueError("response id does not match request")
            except (json.JSONDecodeError, ValueError) as exc:
                self._dispose()
                raise self._error(f"returned invalid JSON protocol during {operation}") from exc
            if not response.get("ok"):
                self.last_worker_retained = (operation == "score" and response.get("error") == RECOVERABLE_SCORE_ERROR
                                             and self.process.poll() is None)
                if not self.last_worker_retained:
                    self._dispose()
                raise self._error(f"failed during {operation}: {response.get('error', 'unknown error')}")
            return response
    return GeoSCDPool


class WorkerProgress:
    """Tail the worker's existing log and report liveness without touching its protocol."""
    def __init__(self, pool, label, heartbeat_seconds=15, poll_seconds=.2):
        self.pool, self.label = pool, label
        self.heartbeat_seconds, self.poll_seconds = heartbeat_seconds, poll_seconds
        self.stop = threading.Event()
        self.thread = None
        if not hasattr(pool, "_retry_log_offsets"):
            pool._retry_log_offsets = {}
        self.offsets = pool._retry_log_offsets

    def _drain(self, path):
        if not path:
            return
        path = Path(path)
        try:
            with path.open("rb") as stream:
                stream.seek(self.offsets.get(str(path), 0))
                chunk = stream.read(65536)
                self.offsets[str(path)] = stream.tell()
            if chunk:
                sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                sys.stdout.flush()
        except OSError:
            pass

    def _watch(self):
        started, last_heartbeat, path = time.monotonic(), time.monotonic(), None
        while True:
            current_path = getattr(self.pool, "stderr_path", None)
            if current_path != path:
                self._drain(path)
                path = current_path
            self._drain(path)
            now = time.monotonic()
            if now - last_heartbeat >= self.heartbeat_seconds:
                process = getattr(self.pool, "process", None)
                pid = getattr(process, "pid", "—")
                print(f"{self.label}: выполняется {int(now - started)} с; worker PID {pid}.", flush=True)
                last_heartbeat = now
            if self.stop.wait(self.poll_seconds):
                self._drain(getattr(self.pool, "stderr_path", None))
                return

    def __enter__(self):
        self.thread = threading.Thread(target=self._watch, name="geoscd-progress", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def payload_sha256(arrays):
    value = hashlib.sha256()
    for name in ("reference", "source", "support"):
        array = np.ascontiguousarray(arrays[name])
        value.update(json.dumps([name, list(array.shape), array.dtype.str]).encode())
        value.update(array.tobytes())
    return value.hexdigest()


def load_arrays(root, record, prediction=False):
    path = (root / record["artifact_path"]).resolve()
    if root not in path.parents or sha256(path) != record["artifact_sha256"]:
        raise ValueError(f"Неверный путь или SHA-256 карты: {path}")
    names = ["reference", "source", "support"] + (["scores", "native_prediction"] if prediction else [])
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in names}
    support = arrays["support"]
    if support.dtype != bool or support.ndim != 2 or not support.size:
        raise ValueError("Некорректная геометрическая поддержка")
    for name in ("reference", "source"):
        if arrays[name].dtype != np.uint8 or arrays[name].shape != (*support.shape, 3):
            raise ValueError(f"Некорректный исходный RGB: {name}")
    if prediction:
        validate_prediction(arrays, support)
    return arrays


def validate_prediction(arrays, support):
    scores, native = arrays["scores"], arrays["native_prediction"]
    if (scores.dtype != np.float32 or scores.shape != support.shape or not np.isfinite(scores[support]).all()
            or not np.isnan(scores[~support]).all() or native.dtype != bool or native.shape != support.shape):
        raise ValueError("GeoSCD не вернул корректные исходные оценки и авторскую бинарную маску")


def stable_metadata(metadata):
    result = copy.deepcopy(metadata)
    result.pop("worker_stderr_log", None)
    result.pop("gpu_preflight", None)
    if "worker_environment" in result:
        result["worker_environment"].pop("git_commit", None)
        result["worker_environment"].pop("git_dirty", None)
    return result


def run(run_path, review_path, out_path, repo_path=None, modes=("previous", "earliest"),
        resume=False, retry_failed=False, sequence_ids=None, pool_factory=None, snapshot_path=None,
        all_real_val=False, upgrade_runner=False):
    if upgrade_runner and not resume:
        raise ValueError("--upgrade-runner требует --resume")
    root, review, out = (Path(value).expanduser().resolve() for value in (run_path, review_path, out_path))
    review = review / "summary.json" if review.is_dir() else review
    review_data = read_json(review)
    review_provenance = review.parent / "provenance.json"
    provenance = read_json(review_provenance) if review_provenance.is_file() else {}
    if provenance.get("source_run") and Path(provenance["source_run"]).expanduser().resolve() != root:
        raise ValueError("Отчёт относится к другому исходному запуску")
    if snapshot_path is None:
        snapshot_path = provenance.get("snapshot")
    if snapshot_path is None:
        raise ValueError("Нужен --snapshot: каталог сохранённого быстрого отчёта")
    snapshot = Path(snapshot_path).expanduser().resolve()
    snapshot_units = snapshot / "2026-10-17_source_units.json"
    snapshot_summary = snapshot / "2026-10-17_quick_summary.json"
    if Path(read_json(snapshot_summary)["source_run"]).expanduser().resolve() != root:
        raise ValueError("Снимок быстрого отчёта относится к другому запуску")
    snapshot_hashes = {str(path): sha256(path) for path in (snapshot_units, snapshot_summary)}
    for path, digest in provenance.get("source_metadata_sha256", {}).items():
        if path in snapshot_hashes and snapshot_hashes[path] != digest:
            raise ValueError(f"Изменился снимок отчёта: {path}")
    if all_real_val and sequence_ids is not None:
        raise ValueError("--all-real-val и --sequence-ids нельзя совмещать")
    if all_real_val:
        selected = sorted({row["sequence_id"] for row in read_json(snapshot_units)
                           if row.get("domain") == "real" and row.get("split") == "val"
                           and row.get("method") == "rgb_diff" and row.get("mode") == "previous"
                           and row.get("status") == "completed"})
    else:
        selected = list(sequence_ids if sequence_ids is not None else review_data["selected_sequences"])
    modes = tuple(modes)
    if (not selected or len(selected) != len(set(selected)) or any(not isinstance(s, str) for s in selected)
            or not modes or len(modes) != len(set(modes)) or set(modes) - {"previous", "earliest"}):
        raise ValueError("Нужны непустые уникальные последовательности и режимы previous/earliest")
    config_path = root / "resolved_config.json"
    config = read_json(config_path)
    options = copy.deepcopy(config["method_options"]["geoscd"])
    options.setdefault("device", config.get("device", "cpu"))
    packaged_repo = Path(__file__).resolve().parents[2]
    repo = (Path(repo_path).expanduser().resolve() if repo_path else
            packaged_repo if (packaged_repo / "facade_change" / "__init__.py").is_file() else root.parents[1])
    baseline, cached, record_hashes = {}, {}, {}
    for path in sorted((root / "units").glob("*.json")):
        row = read_json(path)
        if (row.get("domain") != "real" or row.get("split") != "val" or row.get("sequence_id") not in selected
                or row.get("status") != "completed" or row.get("variant", "frozen") != "frozen"):
            continue
        if row.get("method") not in {"rgb_diff", "geoscd"} or row.get("mode") not in set(modes) | {"previous"}:
            continue
        key = (row["sequence_id"], row["mode"], int(row["frame"]))
        target = baseline if row["method"] == "rgb_diff" else cached
        if key in target:
            raise ValueError(f"Неоднозначные исходные записи: {key}")
        target[key] = row
        record_hashes[row["unit_id"]] = sha256(path)
    pairs = []
    for sequence in selected:
        frames = sorted(key[2] for key in baseline if key[:2] == (sequence, "previous"))
        if not frames or frames != list(range(1, frames[-1] + 1)):
            raise ValueError(f"Нет целой последовательности RGB_DIFF previous: {sequence}")
        buildings = {baseline[(sequence, "previous", frame)].get("building_id") for frame in frames}
        if len(buildings) != 1 or None in buildings:
            raise ValueError(f"Неоднозначный фасад: {sequence}")
        for mode in modes:
            if sorted(key[2] for key in baseline if key[:2] == (sequence, mode)) != frames:
                raise ValueError(f"Неполная исходная последовательность RGB_DIFF {mode}: {sequence}")
            pairs.extend((sequence, mode, frame) for frame in frames)
    contract = dict(source_run=str(root), snapshot=str(snapshot), review=str(review), repo=str(repo),
                    snapshot_source_units_sha256=snapshot_hashes[str(snapshot_units)],
                    snapshot_quick_summary_sha256=snapshot_hashes[str(snapshot_summary)],
                    review_sha256=sha256(review), resolved_config_sha256=sha256(config_path),
                    script_sha256=sha256(Path(__file__)),
                    repository_code_sha256={name: sha256(repo / name) for name in (
                        "facade_change/geoscd_full.py", "facade_change/methods/geoscd.py",
                        "facade_change/methods/remote.py") if (repo / name).is_file()},
                    selection_policy="all_real_val" if all_real_val else "review_or_explicit_sequences",
                    method_options=options, selected_sequences=selected, modes=list(modes),
                    input_records=[dict(unit_id=baseline[key]["unit_id"],
                                        record_sha256=record_hashes[baseline[key]["unit_id"]],
                                        artifact_sha256=baseline[key]["artifact_sha256"]) for key in pairs])
    for protected in (root, review.parent, snapshot):
        if out == protected or protected in out.parents or out in protected.parents:
            raise ValueError("Выходной каталог должен быть отдельным от исходного запуска и отчётов")
    if out.exists() and not resume:
        raise FileExistsError(f"Каталог уже существует; для продолжения нужен --resume: {out}")
    if resume and not (out / "run.json").is_file():
        raise ValueError("Нет run.json для продолжения")
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Этот каталог уже обрабатывается другим процессом") from exc
        if resume:
            previous_run = read_json(out / "run.json")
            old_contract = previous_run.get("contract", {})
            if old_contract != contract:
                same_inputs = {k: v for k, v in old_contract.items() if k != "script_sha256"} == {
                    k: v for k, v in contract.items() if k != "script_sha256"}
                if not (upgrade_runner and same_inputs and old_contract.get("script_sha256") in KNOWN_PREVIOUS_RUNNER_SHA256):
                    raise ValueError("Контракт продолжения изменился. --upgrade-runner разрешает только известную прежнюю версию скрипта при неизменных данных и настройках")
                old_script_sha256 = old_contract["script_sha256"]
                upgrade_path = out / "runner_upgrades" / f"{old_script_sha256}-{contract['script_sha256']}.json"
                if upgrade_path.exists():
                    upgrade = read_json(upgrade_path)
                    if upgrade.get("new_contract") != contract or upgrade.get("previous_run", {}).get("contract") != old_contract:
                        raise ValueError("Не совпадает сохранённое подтверждение обновления скрипта")
                else:
                    atomic_json(upgrade_path, dict(from_script_sha256=old_script_sha256,
                                to_script_sha256=contract["script_sha256"], upgraded_at=time.time(),
                                previous_run=previous_run, new_contract=contract,
                                reason="Known runner upgrade: recoverable GeoSCD error handling, live progress, or repository-relative script migration"))
                print(f"Обновление скрипта зафиксировано: {upgrade_path}. Готовые карты и попытки сохранены.", flush=True)
        return process_pairs(root, out, repo, pairs, baseline, cached, record_hashes, contract,
                             options, retry_failed, pool_factory)


def process_pairs(root, out, repo, pairs, baseline, cached, record_hashes, contract, options,
                  retry_failed, pool_factory):
    for folder in ("maps", "units", "attempts", "logs"):
        (out / folder).mkdir(exist_ok=True)
    run_record = {**contract, "contract": contract, "status": "running", "started_at": time.time(),
                  "runner_upgrades": [str(path.relative_to(out)) for path in sorted((out / "runner_upgrades").glob("*.json"))],
                  "scope": "GeoSCD on cached real VAL RGB pairs; no training, GT, or accuracy evaluation"}
    atomic_json(out / "run.json", run_record)
    counts, results, pool = Counter(), [], None
    final_status, fatal_error = "completed", None
    gpu_checked = False
    current = None
    original_metadata = [row.get("method_metadata", {}) for row in cached.values()]
    original_provenance = root / "method_provenance" / "full_geoscd.json"
    if original_provenance.is_file():
        original_metadata.append(read_json(original_provenance))

    def save_attempt(record, arrays=None):
        identifier = record["unit_id"]
        if arrays is not None:
            path = out / "maps" / f"{identifier}.npz"
            with path.with_suffix(".npz.tmp").open("wb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(path.with_suffix(".npz.tmp"), path)
            record.update(artifact_path=str(path.relative_to(out)), artifact_sha256=sha256(path))
        attempt = len(list((out / "attempts").glob(f"{identifier}-*.json"))) + 1
        record["attempt"] = attempt
        atomic_json(out / "attempts" / f"{identifier}-{attempt:04d}.json", record)
        atomic_json(out / "units" / f"{identifier}.json", record)
        results.append(record)

    try:
        print(f"GeoSCD: выбрано историй {len(contract['selected_sequences'])}, пар {len(pairs)}.", flush=True)
        for index, key in enumerate(pairs, 1):
            source = baseline[key]
            inputs = load_arrays(root, source)
            identifier = hashlib.sha256(json.dumps([*key, "geoscd", "frozen"]).encode()).hexdigest()[:32]
            current = dict(unit_id=identifier, method="geoscd", variant="frozen", variant_id="geoscd",
                           mode=key[1], target="step_event" if key[1] == "previous" else "earliest_state",
                           domain="real", split="val", sequence_id=key[0], frame=key[2],
                           building_id=source["building_id"], output_kind="native_mask", label_verified=False,
                           source_input_unit_id=source["unit_id"],
                           source_input_artifact_sha256=source["artifact_sha256"],
                           source_input_record_sha256=record_hashes[source["unit_id"]],
                           source_input_sha256=payload_sha256(inputs))
            prefix = f"[{index}/{len(pairs)}] {key[0]} {key[1]} наблюдение {key[2]}"
            marker = out / "units" / f"{identifier}.json"
            attempts = sorted((out / "attempts").glob(f"{identifier}-*.json"))
            previous = read_json(attempts[-1]) if attempts else (read_json(marker) if marker.is_file() else None)
            if previous:
                if any(previous.get(name) != value for name, value in current.items()):
                    raise ValueError(f"Изменилась сохранённая пара: {identifier}")
                if previous["status"] == "completed":
                    completed_arrays = load_arrays(out, previous, prediction=True)
                    if payload_sha256(completed_arrays) != current["source_input_sha256"]:
                        raise ValueError("RGB/support сохранённого результата отличаются от исходной пары")
                    if not marker.is_file() or read_json(marker) != previous:
                        atomic_json(marker, previous)
                    counts["skipped_completed"] += 1
                    results.append(previous)
                    print(prefix + " — уже готово", flush=True)
                    current = None
                    continue
                if previous["status"] == "failed" and not retry_failed:
                    counts["skipped_failed"] += 1
                    results.append(previous)
                    print(prefix + " — сохранённый отказ; повтор только с --retry-failed", flush=True)
                    current = None
                    continue
            print(prefix + " — обработка", flush=True)
            started = time.monotonic()
            if key in cached:
                existing = cached[key]
                output = load_arrays(root, existing, prediction=True)
                if existing.get("building_id") != current["building_id"] or payload_sha256(output) != current["source_input_sha256"]:
                    raise ValueError("Существующий GeoSCD не соответствует исходной RGB-паре")
                saved_options = existing.get("method_metadata", {}).get("configured_options")
                if saved_options is not None and saved_options != options:
                    raise ValueError("Настройки существующего GeoSCD отличаются от исходной конфигурации")
                current.update(status="completed", origin="reused_source", source_geoscd_unit_id=existing["unit_id"],
                               source_geoscd_artifact_sha256=existing["artifact_sha256"],
                               method_metadata=existing.get("method_metadata", {}))
                save_attempt(current, output)
                counts["reused"] += 1
            else:
                current["origin"] = "retry_inference"
                phase = "initialization"
                try:
                    if pool is None:
                        if pool_factory is None:
                            if not (repo / "facade_change" / "methods" / "remote.py").is_file():
                                raise ValueError(f"Не найден исходный RemotePool; укажите --repo: {repo}")
                            sys.path.insert(0, str(repo))
                            from facade_change.methods.remote import RemotePool, _json_value
                            pool_factory = make_geoscd_pool(RemotePool, _json_value)
                        pool = pool_factory(log_dir=out / "logs")
                    if not gpu_checked:
                        try:
                            gpu = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
                            snapshot_text = f"returncode={gpu.returncode}\n{gpu.stdout}\n{gpu.stderr}"
                        except (OSError, subprocess.TimeoutExpired) as exc:
                            snapshot_text = f"nvidia-smi: {exc}\n"
                        gpu_path = out / "logs" / f"gpu-{time.time_ns()}.txt"
                        gpu_path.write_text(snapshot_text, encoding="utf-8")
                        print(f"Состояние GPU сохранено: {gpu_path}. Другие процессы не изменяются.", flush=True)
                        for line in [line.strip() for line in snapshot_text.splitlines() if "MiB" in line][:3]:
                            print(line, flush=True)
                        gpu_checked = True
                    with WorkerProgress(pool, prefix + " — инициализация GeoSCD"):
                        metadata = pool.activate("geoscd", options)
                    for original in original_metadata:
                        for name in ("source", "checkpoints", "full_detector", "output_kind"):
                            if name in original and original[name] != metadata.get(name):
                                raise ValueError(f"GeoSCD отличается от исходного запуска: {name}")
                    stable = stable_metadata(metadata)
                    provenance_path = out / "model_provenance.json"
                    if provenance_path.is_file() and read_json(provenance_path) != stable:
                        raise ValueError("Изменились веса, код или окружение GeoSCD при продолжении")
                    if not provenance_path.is_file():
                        atomic_json(provenance_path, stable)
                    phase = "score"
                    with WorkerProgress(pool, prefix + " — расчёт GeoSCD"):
                        inferred = pool.score("geoscd", options, inputs["reference"], inputs["source"], inputs["support"])
                    output = {**inputs, "scores": inferred["scores"], "native_prediction": inferred["native_prediction"]}
                    validate_prediction(output, inputs["support"])
                    current.update(status="completed", method_metadata=inferred.get("metadata", metadata),
                                   elapsed_seconds=time.monotonic() - started)
                    save_attempt(current, output)
                    counts["inferred"] += 1
                except Exception as exc:
                    current.update(status="failed", error=f"{type(exc).__name__}: {exc}", failure_phase=phase,
                                   worker_retained=bool(getattr(pool, "last_worker_retained", False)),
                                   stderr_log=str(getattr(pool, "stderr_path", "")), elapsed_seconds=time.monotonic() - started)
                    save_attempt(current)
                    counts["failed"] += 1
                    print(prefix + f" — отказ: {exc}", flush=True)
                    known_pair_failure = str(exc).startswith(
                        "Inference worker failed during score: " + RECOVERABLE_SCORE_ERROR + "; stderr log: ")
                    if phase != "score" or not known_pair_failure:
                        final_status, fatal_error = f"failed_{phase}", current["error"]
                        current = None
                        break
                    print("Карта этой пары не создана; отказ сохранён. " + (
                        "GeoSCD остаётся загруженным для следующей пары." if current["worker_retained"]
                        else "Следующая пара будет обработана отдельно."), flush=True)
            if current["status"] == "completed":
                marked = int(np.count_nonzero(output["native_prediction"] & output["support"]))
                supported = int(np.count_nonzero(output["support"]))
                print(prefix + f" — готово ({current['origin']}), {time.monotonic() - started:.1f} с; "
                      f"маска: {marked}/{supported} пикселей поддержки.", flush=True)
            current = None
            atomic_json(out / "progress.json", dict(processed=len(results), total=len(pairs), counts=dict(counts)))
    except KeyboardInterrupt:
        final_status = "interrupted"
        print("Прервано пользователем; готовые пары сохранены. Продолжение: --resume.", flush=True)
    except Exception as exc:
        final_status, fatal_error = "failed_validation", f"{type(exc).__name__}: {exc}"
    finally:
        if pool is not None:
            pool.release()
        completed = sum(row.get("status") == "completed" for row in results)
        failures = sum(row.get("status") == "failed" for row in results)
        if final_status == "completed" and failures:
            final_status = "completed_with_failures"
        summary = dict(status=final_status, selected_sequences=contract["selected_sequences"],
                       selected=len(pairs), completed=completed, failed=failures,
                       reused=counts["reused"], inferred=counts["inferred"],
                       skipped_completed=counts["skipped_completed"], skipped_failed=counts["skipped_failed"],
                       unprocessed=len(pairs) - len(results), error=fatal_error, results=results)
        atomic_json(out / "summary.json", summary)
        run_record.update(status=final_status, finished_at=time.time(), error=fatal_error,
                          completed=completed, failed=failures)
        atomic_json(out / "run.json", run_record)
        print(f"GeoSCD: готово {completed}/{len(pairs)}, отказов {failures}; статус {final_status}. Отчёт: {out / 'summary.json'}", flush=True)
    if fatal_error:
        raise RuntimeError(fatal_error)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Исходный временной запуск")
    parser.add_argument("--review", required=True, help="summary.json галереи или его каталог")
    parser.add_argument("--snapshot", help="Каталог сохранённого быстрого отчёта")
    parser.add_argument("--out", required=True, help="Новый каталог результатов GeoSCD")
    parser.add_argument("--repo", help="Каталог Facade_Change_Benchmark")
    parser.add_argument("--modes", nargs="+", choices=("previous", "earliest", "both"), default=["both"])
    parser.add_argument("--sequence-ids", nargs="+")
    parser.add_argument("--all-real-val", action="store_true", help="Все реальные VAL-истории из снимка отчёта")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--upgrade-runner", action="store_true", help="Разрешить проверенное обновление прежнего скрипта при --resume")
    args = parser.parse_args()
    if "both" in args.modes and args.modes != ["both"]:
        parser.error("both используется отдельно")
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed требует --resume")
    if args.upgrade_runner and not args.resume:
        parser.error("--upgrade-runner требует --resume")
    modes = ("previous", "earliest") if args.modes == ["both"] else tuple(args.modes)
    result = run(args.run, args.review, args.out, args.repo, modes, args.resume, args.retry_failed,
                 args.sequence_ids, snapshot_path=args.snapshot, all_real_val=args.all_real_val,
                 upgrade_runner=args.upgrade_runner)
    return 130 if result["status"] == "interrupted" else 0


if __name__ == "__main__":
    raise SystemExit(main())
