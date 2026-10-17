"""Resume-safe inference on visually reviewed, fixed real robustness pairs.

Only RGB and geometric support enter the worker. Evaluation annotations and
selection metadata stay in the parent process. No training or downloads occur.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .robustness_common import atomic_json, atomic_npz, digest, load_pair, read_json, safe_path, sha256
from .geoscd_real_retry import RECOVERABLE_SCORE_ERROR, WorkerProgress, make_geoscd_pool, stable_metadata


DEFAULT_METHODS = ("rgb_diff", "ssim", "lpips", "dinov2", "semantic_dino", "anychange", "geoscd", "msdzip_abs")
ALLOWED_METHODS = frozenset((*DEFAULT_METHODS, "msdzip_mod256"))
TASKS = (("H1", "original"), ("H1", "exposure"), ("H1", "shadow"), ("H0", "exposure"), ("H0", "shadow"))
NATIVE_METHODS = frozenset({"geoscd", "anychange"})
_RUNTIME_METADATA = {"worker_stderr_log", "gpu_preflight", "last_padding", "configured_options",
                     "temporal_origin_audit", "device", "worker_python"}
_INFERENCE_FILES = {"method_worker.py", "io.py", "geoscd_full.py", "2026-10-04_msdzip_h0.py",
                    "hypothesis_benchmark.py", "preparation.py"}


def _inference_source(name):
    return name in _INFERENCE_FILES or name.startswith(("methods/", "segmentation/"))


def payload_sha256(arrays):
    result = hashlib.sha256()
    for name in ("reference", "source", "support"):
        value = np.ascontiguousarray(arrays[name])
        result.update(json.dumps([name, list(value.shape), value.dtype.str]).encode())
        result.update(value.tobytes())
    return result.hexdigest()


def model_identity(metadata):
    """Exclude only runtime/input-dependent fields; retain weights and settings."""
    result = stable_metadata(metadata)
    for key in _RUNTIME_METADATA:
        result.pop(key, None)
    environment = result.get("worker_environment", {})
    if isinstance(environment.get("source_sha256"), dict):
        environment["source_sha256"] = {name: value for name, value in environment["source_sha256"].items()
                                       if _inference_source(name)}
    return result


def _verify_metadata(expected, actual, method):
    expected, actual = model_identity(expected), model_identity(actual)
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"{method}: изменилось происхождение модели/настройки: {key}")


def _file_hashes(metadata):
    """Collect recorded file checksums without searching for alternate weights."""
    found = {}
    if isinstance(metadata, dict):
        if isinstance(metadata.get("path"), str) and isinstance(metadata.get("sha256"), str):
            found[metadata["path"]] = metadata["sha256"]
        if metadata.get("checkpoint_path") and metadata.get("checkpoint_sha256"):
            found[metadata["checkpoint_path"]] = metadata["checkpoint_sha256"]
        for value in metadata.values():
            for path, checksum in _file_hashes(value).items():
                if path in found and found[path] != checksum:
                    raise ValueError(f"Противоречивые хеши исходной модели: {path}")
                found[path] = checksum
    elif isinstance(metadata, list):
        for value in metadata:
            found.update(_file_hashes(value))
    return found


def _source_rows(manifest):
    rows = manifest.get("source_records", [])
    if isinstance(rows, dict):
        rows = list(rows.values())
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("source_records должен содержать сохранённые метаданные единиц")
    return rows


def _method_contract(method, manifest, source, device):
    config = manifest["source_config"]
    if not isinstance(config, dict):
        raise ValueError("source_config должен быть сохранённым объектом конфигурации")
    options = copy.deepcopy(config.get("method_options", {}).get(method, {}))
    rows = [r for r in _source_rows(manifest) if r.get("method") == method
            and r.get("status") == "completed" and r.get("variant", "frozen") ==
            ("current_train" if method.startswith("msdzip_") else "frozen")]
    provenance_path = source / "method_provenance" / f"full_{method}.json"
    provenance = read_json(provenance_path) if provenance_path.is_file() else None
    if provenance is None:
        provenance = next((r["method_metadata"] for r in rows if r.get("method_metadata")), {})
    configured = [r["method_metadata"]["configured_options"] for r in rows
                  if r.get("method_metadata", {}).get("configured_options") is not None]
    if provenance.get("configured_options") is not None:
        configured.append(provenance["configured_options"])
    if configured:
        if any(value != configured[0] for value in configured):
            raise ValueError(f"{method}: исходные единицы имеют разные настройки")
        for key, value in options.items():
            if key in configured[0] and configured[0][key] != value:
                raise ValueError(f"{method}: configured_options не совпадает с resolved_config: {key}")
        options.update(copy.deepcopy(configured[0]))
    if method not in {"rgb_diff", "ssim"}:
        options.setdefault("device", config.get("device", "cpu"))
    if method.startswith("msdzip_"):
        checkpoint = provenance.get("checkpoint_path") or options.get("checkpoint_path")
        checkpoint_hash = provenance.get("checkpoint_sha256")
        if not checkpoint or not checkpoint_hash:
            raise ValueError(f"{method}: нет записанного checkpoint_path/checkpoint_sha256; обучение не запускается")
        if options.get("checkpoint_path") and str(Path(options["checkpoint_path"]).expanduser().resolve()) != str(Path(checkpoint).expanduser().resolve()):
            raise ValueError(f"{method}: checkpoint противоречит исходной конфигурации")
        options["checkpoint_path"] = str(Path(checkpoint).expanduser().resolve())
        options.setdefault("worker_python", config.get("training_python") or provenance.get("worker_python")
                           or provenance.get("worker_environment", {}).get("executable"))
        if options.get("worker_python") is None:
            options.pop("worker_python", None)
        if provenance.get("checkpoint_dataset_verified"):
            fingerprint = provenance.get("input_sha256")
            if fingerprint:
                options.setdefault("dataset_fingerprint", fingerprint)
    if device is not None and method not in {"rgb_diff", "ssim"}:
        options["device"] = device
    # Do not enable downloads/training through an inherited convenience flag.
    if any(options.get(key) for key in ("download", "download_weights", "download-weights", "train", "auto_train")):
        raise ValueError(f"{method}: автоматическое обучение/скачивание запрещено")
    if method not in {"rgb_diff", "ssim"} and not provenance:
        raise ValueError(f"{method}: нет исходного происхождения модели")
    artifacts = _file_hashes(provenance)
    for path, expected in artifacts.items():
        actual_path = Path(path).expanduser().resolve()
        if not actual_path.is_file() or sha256(actual_path) != expected:
            raise ValueError(f"{method}: исходный файл модели отсутствует или изменился: {actual_path}")
    return {"options": options, "expected_metadata": provenance, "model_files_sha256": artifacts,
            "provenance_file": str(provenance_path) if provenance_path.is_file() else None,
            "provenance_sha256": sha256(provenance_path) if provenance_path.is_file() else None}


def task_arrays(pair, hypothesis, condition):
    h0 = hypothesis == "H0"
    support = pair["support"].copy()
    labels = np.where(support, 0, 255).astype(np.uint8) if h0 else pair["labels"].copy()
    return {"reference": pair["source" if h0 else "reference"].copy(),
            "source": pair["source" if condition == "original" else condition].copy(),
            "support": support, "labels": labels,
            "reference_bits": pair["source_bits" if h0 else "reference_bits"].copy(),
            "source_bits": pair["source_bits"].copy()}


def validate_output(output, support, method):
    scores = output.get("scores")
    if (not isinstance(scores, np.ndarray) or scores.dtype != np.float32 or scores.shape != support.shape
            or not np.isfinite(scores[support]).all() or not np.isnan(scores[~support]).all()
            or np.any(scores[support] < 0) or np.any(scores[support] > 1)):
        raise ValueError(f"{method}: неверная сетка/dtype/диапазон/поддержка карты scores")
    if method in NATIVE_METHODS and "native_prediction" not in output:
        raise ValueError(f"{method}: отсутствует авторская бинарная маска")
    for key in ("native_prediction", "raw_scores"):
        if key in output:
            value = output[key]
            dtype = np.bool_ if key == "native_prediction" else np.float32
            if value.dtype != dtype or value.shape != support.shape:
                raise ValueError(f"{method}: некорректный {key}")
            if key == "raw_scores" and not np.isfinite(value[support]).all():
                raise ValueError(f"{method}: нечисловой raw_scores на поддержке")
    if method in NATIVE_METHODS and not np.array_equal(scores[support], output["native_prediction"][support].astype(np.float32)):
        raise ValueError(f"{method}: scores расходится с авторской бинарной маской")


def _load_output(root, record, inputs, method):
    path = safe_path(root, record["artifact_path"])
    if sha256(path) != record["artifact_sha256"]:
        raise ValueError(f"Изменился сохранённый результат: {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if payload_sha256(arrays) != payload_sha256(inputs):
        raise ValueError(f"Сохранённый результат относится к другой RGB/support паре: {path}")
    validate_output(arrays, inputs["support"], method)
    return arrays


def _cache_records(manifest, geoscd_run, source):
    result = {}
    rows = [(source, row, "reused_source") for row in _source_rows(manifest)]
    overlay_contract = None
    if geoscd_run is not None:
        overlay = Path(geoscd_run).expanduser().resolve()
        saved_run = read_json(overlay / "run.json")
        overlay_contract = saved_run.get("contract", saved_run)
        if Path(overlay_contract.get("source_run", "")).expanduser().resolve() != source:
            raise ValueError("GeoSCD overlay относится к другому исходному запуску")
        rows.extend((overlay, read_json(path), "reused_geoscd") for path in sorted((overlay / "units").glob("*.json")))
    for root, row, origin in rows:
        if (row.get("status") != "completed" or row.get("domain") != "real" or row.get("split") != "val"
                or row.get("mode") != "previous" or row.get("method") not in ALLOWED_METHODS):
            continue
        method = row["method"]
        if row.get("variant", "frozen") != ("current_train" if method.startswith("msdzip_") else "frozen"):
            continue
        if origin == "reused_geoscd" and method != "geoscd":
            continue
        key = (method, row["sequence_id"], int(row["frame"]))
        if key in result and origin == "reused_source":
            raise ValueError(f"Неоднозначный исходный кэш: {key}")
        if key not in result:
            result[key] = (root, row, origin)
    return result, overlay_contract


def _event(out, event, **fields):
    with (out / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"timestamp": time.time(), "event_type": event, **fields}, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _snapshot_gpu(out):
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
        value = f"returncode={result.returncode}\n{result.stdout}\n{result.stderr}"
    except (OSError, subprocess.TimeoutExpired) as error:
        value = f"nvidia-smi: {error}\n"
    path = out / "logs" / f"gpu-{time.time_ns()}.txt"
    path.write_text(value, encoding="utf-8")
    print(f"Состояние GPU: {path}. Другие процессы не изменяются.", flush=True)


def run(prepared_path, selection_path, out_path, methods=None, geoscd_run=None,
        resume=False, retry_failed=False, device=None, pool_factory=None):
    prepared = Path(prepared_path).expanduser().resolve()
    manifest_path = prepared / "manifest.json" if prepared.is_dir() else prepared
    prepared = manifest_path.parent
    selection_path, out = (Path(value).expanduser().resolve() for value in (selection_path, out_path))
    manifest, selection = read_json(manifest_path), read_json(selection_path)
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "real_robustness_prepared":
        raise ValueError("Нужен подготовленный manifest real_robustness_prepared schema 1")
    if (selection.get("schema_version") != 1 or selection.get("reviewed") is not True
            or selection.get("manifest_sha256") != sha256(manifest_path)):
        raise ValueError("Нужен визуально проверенный selection с точным SHA-256 манифеста")
    ids = selection.get("pair_ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("selection.pair_ids должен быть непустым списком уникальных идентификаторов")
    pairs = {row["pair_id"]: row for row in manifest["pairs"]}
    if len(pairs) != len(manifest["pairs"]) or any(key not in pairs or pairs[key].get("status", "ready") != "ready" for key in ids):
        raise ValueError("Выбор содержит неизвестные/непригодные пары или манифест имеет дубликаты")
    methods = tuple(DEFAULT_METHODS if methods is None else methods)
    if not methods or len(set(methods)) != len(methods) or set(methods) - ALLOWED_METHODS:
        raise ValueError("Нужен непустой список уникальных поддерживаемых методов")
    if retry_failed and not resume:
        raise ValueError("--retry-failed требует --resume")
    source = Path(manifest["source_run"]).expanduser().resolve()
    repo = Path(manifest["repo_path"]).expanduser().resolve()
    protected = [prepared, source, Path(manifest["snapshot"]).expanduser().resolve()]
    if geoscd_run:
        protected.append(Path(geoscd_run).expanduser().resolve())
    if any(out == p or out in p.parents or p in out.parents for p in protected):
        raise ValueError("Выходной каталог должен быть отдельным от входных запусков")
    if out.exists() and not resume:
        raise FileExistsError(f"Каталог существует; продолжение только с --resume: {out}")
    if resume and not (out / "run.json").is_file():
        raise ValueError("Нет run.json для продолжения")
    for filename, expected in manifest.get("metadata_sha256", {}).items():
        path = Path(filename).expanduser().resolve()
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Изменились исходные метаданные: {filename}")
    # Verify every selected artifact before any model/GPU work.
    for key in ids:
        if not load_pair(prepared, pairs[key])["support"].any():
            raise ValueError(f"Нет геометрической поддержки выбранной пары: {key}")
    model_contracts = {}
    for method in methods:
        print(f"Проверка записанных настроек и хешей весов: {method}", flush=True)
        model_contracts[method] = _method_contract(method, manifest, source, device)
    cached, overlay_contract = _cache_records(manifest, geoscd_run, source)
    relevant_paths = [Path(__file__), Path(__file__).with_name("robustness_common.py"),
                      Path(__file__).with_name("geoscd_real_retry.py")]
    relevant_paths += [path for path in (repo / "facade_change").rglob("*.py")
                       if _inference_source(path.relative_to(repo / "facade_change").as_posix())]
    relevant_paths.append(repo / "third_party" / "2026-10-04_msdzip_compress_model.py")
    selected_cache = {}
    for key in ids:
        pair = pairs[key]
        original_inputs = task_arrays(load_pair(prepared, pair), "H1", "original")
        for method in methods:
            cache_key = (method, pair["sequence_id"], int(pair["frame"]))
            if cache_key not in cached:
                continue
            cache_root, row, origin = cached[cache_key]
            if row.get("building_id") != pair["building_id"]:
                raise ValueError("Исходный кэш относится к другому фасаду")
            _verify_metadata(model_contracts[method]["expected_metadata"], row.get("method_metadata", {}), method)
            _load_output(cache_root, row, original_inputs, method)
            selected_cache[digest(list(cache_key))] = {"root": str(cache_root), "origin": origin,
                                                     "record_sha256": digest(row), "artifact_sha256": row["artifact_sha256"]}
    contract = {"schema_version": 1, "prepared_path": str(manifest_path), "selection_path": str(selection_path),
                "manifest_sha256": sha256(manifest_path), "selection_sha256": sha256(selection_path),
                "pair_ids": ids, "methods": list(methods), "tasks": [list(x) for x in TASKS],
                "models": model_contracts, "geoscd_run": str(Path(geoscd_run).expanduser().resolve()) if geoscd_run else None,
                "geoscd_contract": overlay_contract,
                "cached_originals": selected_cache,
                "code_sha256": {str(path): sha256(path) for path in relevant_paths if path.is_file()}}
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Этот каталог уже обрабатывается другим процессом") from error
        if resume and read_json(out / "run.json").get("contract") != contract:
            raise ValueError("Изменился контракт продолжения: данные/выбор/методы/веса/код")
        return _process(prepared, manifest, pairs, out, contract, cached, retry_failed, pool_factory)


def _process(prepared, manifest, pairs, out, contract, cached, retry_failed, pool_factory):
    for name in ("maps", "units", "attempts", "logs", "method_provenance"):
        (out / name).mkdir(exist_ok=True)
    previous = read_json(out / "run.json") if (out / "run.json").is_file() else {}
    run_record = {"schema_version": 1, "kind": "real_robustness_run", "contract": contract,
                  "prepared_path": contract["prepared_path"], "selection_path": contract["selection_path"],
                  "manifest_sha256": contract["manifest_sha256"], "selection_sha256": contract["selection_sha256"],
                  "methods": contract["methods"], "started_at": previous.get("started_at", time.time()),
                  "status": "running", "scope": "Reviewed real VAL pairs; fixed weights; no threshold fitting; PREVIOUS only"}
    atomic_json(out / "run.json", run_record)
    _event(out, "resumed" if previous else "started", contract_sha256=digest(contract))
    results, method_failures, pool = [], {}, None
    status, fatal = "completed", None
    total = len(contract["pair_ids"]) * len(contract["methods"]) * len(TASKS)

    def save(record, arrays=None):
        identifier = record["unit_id"]
        if arrays is not None:
            path = out / "maps" / f"{identifier}.npz"
            atomic_npz(path, arrays)
            record.update(artifact_path=str(path.relative_to(out)), artifact_sha256=sha256(path))
        attempt = len(list((out / "attempts").glob(f"{identifier}-*.json"))) + 1
        record["attempt"] = attempt
        atomic_json(out / "attempts" / f"{identifier}-{attempt:04d}.json", record)
        atomic_json(out / "units" / f"{identifier}.json", record)
        _event(out, "unit_" + record["status"], unit_id=identifier, method=record["method"],
               pair_id=record["pair_id"], hypothesis=record["hypothesis"], condition=record["condition"], attempt=attempt)
        results.append(record)
        atomic_json(out / "progress.json", {"processed": len(results), "total": total,
                    "completed": sum(r["status"] == "completed" for r in results),
                    "failed": sum(r["status"] == "failed" for r in results)})

    try:
        print(f"Устойчивость: пар {len(contract['pair_ids'])}, методов {len(contract['methods'])}, задач {total}; "
              "4 новых сравнения + исходная пара на каждый метод/пару.", flush=True)
        for method in contract["methods"]:
            model = contract["models"][method]
            options = model["options"]
            expected = model["expected_metadata"]
            active_metadata, method_failed = None, False
            try:
                for key in contract["pair_ids"]:
                    if method_failed:
                        break
                    pair = pairs[key]
                    original = load_pair(prepared, pair)
                    for hypothesis, condition in TASKS:
                        inputs = task_arrays(original, hypothesis, condition)
                        identifier = digest([key, method, hypothesis, condition])[:32]
                        current = {"unit_id": identifier, "pair_id": key, "method": method,
                                   "variant": "current_train" if method.startswith("msdzip_") else "frozen",
                                   "variant_id": method, "domain": "real", "split": "val", "mode": "previous",
                                   "target": "step_event", "hypothesis": hypothesis, "condition": condition,
                                   "sequence_id": pair["sequence_id"], "frame": int(pair["frame"]),
                                   "building_id": pair["building_id"], "source_artifact_sha256": pair["artifact_sha256"],
                                   "source_input_unit_id": pair.get("source_unit_id"), "input_sha256": payload_sha256(inputs),
                                   "output_kind": "native_mask" if method in NATIVE_METHODS else "score"}
                        marker = out / "units" / f"{identifier}.json"
                        attempts = sorted((out / "attempts").glob(f"{identifier}-*.json"))
                        saved = read_json(attempts[-1]) if attempts else (read_json(marker) if marker.is_file() else None)
                        label = f"[{len(results)+1}/{total}] {method} {key} {hypothesis}/{condition}"
                        if saved:
                            if any(saved.get(k) != v for k, v in current.items()):
                                raise ValueError(f"Изменилась сохранённая единица: {identifier}")
                            if saved["status"] == "completed":
                                arrays = _load_output(out, saved, inputs, method)
                                for name in ("labels", "reference_bits", "source_bits"):
                                    if not np.array_equal(arrays[name], inputs[name]):
                                        raise ValueError(f"Изменилась оценочная разметка сохранённой единицы: {name}")
                                if not marker.is_file() or read_json(marker) != saved:
                                    atomic_json(marker, saved)
                                results.append(saved)
                                print(label + " — уже готово", flush=True)
                                continue
                            if saved["status"] == "failed" and not retry_failed:
                                results.append(saved)
                                print(label + " — сохранённый отказ; повтор с --retry-failed", flush=True)
                                continue
                        print(label + " — обработка", flush=True)
                        started = time.monotonic()
                        cache_key = (method, pair["sequence_id"], int(pair["frame"]))
                        if hypothesis == "H1" and condition == "original" and cache_key in cached:
                            root, source_record, origin = cached[cache_key]
                            _verify_metadata(expected, source_record.get("method_metadata", {}), method)
                            reused = _load_output(root, source_record, inputs, method)
                            # Source labels were intentionally unknown; use the prepared GT only here.
                            output = {name: value for name, value in reused.items()
                                      if name in {"scores", "raw_scores", "native_prediction", "semantic_delta",
                                                  "uncertainty_reference", "uncertainty_source", "probabilities_reference", "probabilities_source"}}
                            current.update(status="completed", origin=origin, source_result_unit_id=source_record["unit_id"],
                                           source_result_sha256=source_record["artifact_sha256"],
                                           method_metadata=source_record.get("method_metadata", {}), elapsed_seconds=time.monotonic()-started)
                            save(current, {**inputs, **output})
                            print(label + " — исходный результат переиспользован", flush=True)
                            continue
                        phase = "initialization"
                        try:
                            if pool is None:
                                factory = pool_factory
                                if factory is None:
                                    from facade_change.methods.remote import RemotePool, _json_value
                                    factory = make_geoscd_pool(RemotePool, _json_value)
                                pool = factory(log_dir=out / "logs")
                                _snapshot_gpu(out)
                            if active_metadata is None:
                                with WorkerProgress(pool, label + " — инициализация"):
                                    active_metadata = pool.activate(method, options)
                                _verify_metadata(expected, active_metadata, method)
                                model_path = out / "method_provenance" / f"{method}.json"
                                identity = model_identity(active_metadata)
                                if model_path.is_file() and read_json(model_path) != identity:
                                    raise ValueError(f"{method}: модель или окружение изменились при продолжении")
                                if not model_path.is_file():
                                    atomic_json(model_path, identity)
                                _event(out, "method_initialized", method=method)
                            phase = "score"
                            with WorkerProgress(pool, label + " — расчёт"):
                                inferred = pool.score(method, options, inputs["reference"], inputs["source"], inputs["support"])
                            output = {name: value for name, value in inferred.items()
                                      if name in {"scores", "native_prediction", "raw_scores"}}
                            output.update(inferred.get("auxiliary_arrays", {}))
                            validate_output(output, inputs["support"], method)
                            metadata = inferred.get("metadata", active_metadata)
                            _verify_metadata(active_metadata, metadata, method)
                            current.update(status="completed", origin="inference", method_metadata=metadata,
                                           elapsed_seconds=time.monotonic()-started)
                            save(current, {**inputs, **output})
                            print(label + f" — готово, {current['elapsed_seconds']:.1f} с", flush=True)
                        except Exception as error:
                            message = f"{type(error).__name__}: {error}"
                            current.update(status="failed", origin="inference", error=message, failure_phase=phase,
                                           method_metadata=active_metadata or expected,
                                           stderr_log=str(getattr(pool, "stderr_path", "")),
                                           elapsed_seconds=time.monotonic()-started,
                                           worker_retained=bool(getattr(pool, "last_worker_retained", False)))
                            save(current)
                            print(label + f" — отказ: {message}", flush=True)
                            recoverable = (method == "geoscd" and phase == "score" and str(error).startswith(
                                "Inference worker failed during score: " + RECOVERABLE_SCORE_ERROR + "; stderr log: "))
                            if not recoverable:
                                method_failed = True
                                method_failures[method] = message
                                _event(out, "method_failed", method=method, phase=phase, error=message)
                                break
                            if not getattr(pool, "last_worker_retained", False):
                                active_metadata = None
            finally:
                if pool is not None:
                    pool.release()
                    pool = None
    except KeyboardInterrupt:
        status = "interrupted"
        print("Остановлено пользователем. Готовые карты сохранены; продолжение с --resume.", flush=True)
    except Exception as error:
        status, fatal = "failed_validation", f"{type(error).__name__}: {error}"
    finally:
        if pool is not None:
            pool.release()
        completed = sum(row["status"] == "completed" for row in results)
        failed = sum(row["status"] == "failed" for row in results)
        if status == "completed" and (failed or method_failures):
            status = "completed_with_failures"
        summary = {"schema_version": 1, "status": status, "total": total, "completed": completed,
                   "failed": failed, "unprocessed": total-len(results), "method_failures": method_failures,
                   "error": fatal, "results": results, "pair_ids": contract["pair_ids"], "methods": contract["methods"]}
        atomic_json(out / "summary.json", summary)
        run_record.update(status=status, finished_at=time.time(), completed=completed, failed=failed,
                          unprocessed=total-len(results), method_failures=method_failures, error=fatal)
        atomic_json(out / "run.json", run_record)
        _event(out, "finished", status=status, completed=completed, failed=failed, unprocessed=total-len(results))
        print(f"Устойчивость: готово {completed}/{total}, отказов {failed}, не обработано {total-len(results)}. {out / 'summary.json'}", flush=True)
    if fatal:
        raise RuntimeError(fatal)
    return summary
