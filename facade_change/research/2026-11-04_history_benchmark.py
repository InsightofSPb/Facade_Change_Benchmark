"""Frozen, resumable PREVIOUS/EARLIEST/HISTORY on canonical spatial tracks."""
from __future__ import annotations

from collections import defaultdict
import csv
from functools import lru_cache
import hashlib
import html
import importlib
from pathlib import Path
import time

import numpy as np

from ..annotation_targets import _thumbnail
from ..benchmark_config import load_method_options
from ..crop_benchmark import _checkpoint_exposure, _gpu_report, _identity, _pin_model_files, _validate_arrays
from ..history_features import digest, progress
from ..io import environment, read_json, sha256, write_json
from ..methods.registry import EXTERNAL_METHODS
from ..methods.remote import RemotePool, RemoteScorer
from ..scorers import make_scorer


data_module = importlib.import_module("facade_change.research.2026-11-04_history_benchmark_data")
metrics = importlib.import_module("facade_change.research.2026-11-04_history_benchmark_metrics")
TemporalMemory = importlib.import_module("facade_change.research.2026-10-14_temporal_history").TemporalMemory
METHODS = ("rgb_diff", "ssim", "lpips", "dinov2", "semantic_dino", "anychange", "msdzip_abs")
MODES = ("previous", "earliest", "history")


class Paused(Exception):
    """A soft time budget elapsed between atomic units."""


def read_unit(root, identifier):
    path = root / "units" / (identifier + ".json")
    if not path.exists():
        return None
    record = read_json(path)
    artifact = data_module._inside(root, record["artifact_path"])
    if record["unit_id"] != identifier or sha256(artifact) != record["artifact_sha256"]:
        raise ValueError(f"Corrupt committed unit: {path}")
    with np.load(artifact, allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    return record, arrays


def scorer_options(method, options, args, data):
    settings = {"device": "cuda:0", "trust_checkpoint": False, **options.get(method, {})}
    if args.device is not None:
        settings["device"] = args.device
    if args.trust_checkpoint:
        settings["trust_checkpoint"] = True
    if method == "semantic_dino":
        settings.pop("trust_checkpoint", None)
        settings["expected_categories"] = data.categories
        settings["current_manifest"] = data.index["annotation_manifest"]
    if method == "msdzip_abs":
        settings.pop("dataset_fingerprint", None)
        settings["evaluation_split"] = data.split
    remote = method in EXTERNAL_METHODS or (method == "msdzip_abs" and bool(settings.get("worker_python")))
    if method not in EXTERNAL_METHODS:
        allowed = {"device", "trust_checkpoint", "checkpoint_path", "compression_tile_size",
                   "compression_stride", "zstd_level", "lzma_preset", "evaluation_split"}
        if remote:
            allowed.update(("worker_python", "worker_timeout"))
        settings = {k: v for k, v in settings.items() if k in allowed}
    return settings, remote


class PairCache:
    """A score is reusable only for identical ordered RGB, support, code and model."""

    def __init__(self, store, scorer, method, signature, deadline, reuse=None):
        self.store, self.scorer, self.method = store, scorer, method
        self.signature, self.deadline, self.reuse = signature, deadline, reuse
        self.identity = _identity(scorer.metadata)
        self.native = scorer.metadata.get("output_kind") == "native_mask"
        self.mode = "previous"
        self.calls = defaultdict(lambda: dict(requested=0, computed=0, reused=0, scoring_seconds=0.0))

    def check_time(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise Paused("Time budget reached; resume the same output to continue")

    def score(self, reference, current, support):
        self.check_time()
        hashes = [hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
                  for a in (reference, current, support)]
        key = digest([self.signature, reference.shape, hashes])
        identifier = "pair-" + key
        stats = self.calls[self.mode]
        stats["requested"] += 1
        cached = self.store.read_unit(identifier)
        if not cached and self.reuse:
            cached = read_unit(self.reuse, identifier)
            if cached:
                old, arrays = cached
                cached = (self.store.save_unit(dict(unit_id=identifier, kind="pair_map", method=self.method,
                    signature=self.signature, mode=self.mode, input_sha256=key, scoring_seconds=0.0,
                    reused_from=str(self.reuse), original_scoring_seconds=old["scoring_seconds"]), arrays), arrays)
        if cached:
            record, arrays = cached
            if record.get("input_sha256") != key or record.get("signature") != self.signature:
                raise ValueError("Cached pair input/model contract differs")
            stats["reused"] += 1
        else:
            started = time.perf_counter()
            scores = self.scorer(reference.copy(), current.copy(), support.copy())
            seconds = time.perf_counter() - started
            if _identity(self.scorer.metadata) != self.identity:
                raise ValueError("Method identity changed during inference")
            arrays = {"scores": np.asarray(scores)}
            for name in ("raw_scores", "native_prediction"):
                value = getattr(self.scorer, name, None)
                if value is not None:
                    arrays[name] = np.asarray(value)
            # Check the adapter's complete output, but do not duplicate large feature tensors.
            _validate_arrays({**arrays, **getattr(self.scorer, "auxiliary_arrays", {})},
                             support, self.native, self.scorer.metadata)
            arrays["support"] = support.copy()
            record = self.store.save_unit(dict(unit_id=identifier, kind="pair_map", method=self.method,
                mode=self.mode, signature=self.signature, input_sha256=key, scoring_seconds=seconds), arrays)
            stats["computed"] += 1
            stats["scoring_seconds"] += seconds
        if not np.array_equal(arrays["support"], support):
            raise ValueError("Cached pair support differs")
        _validate_arrays({k: v for k, v in arrays.items() if k != "support"},
                         support, self.native, self.scorer.metadata)
        return record, arrays

    def __call__(self, reference, current, support):
        record, arrays = self.score(reference, current, support)
        return arrays["scores"], {k: record[k] for k in ("unit_id", "artifact_sha256")}


def prediction_reader(root, index):
    @lru_cache(maxsize=24)
    def arrays(identifier):
        result = read_unit(root, identifier)
        if result is None:
            raise ValueError(f"Missing committed prediction: {identifier}")
        return result

    def prediction(row):
        expected = index[row["event_id"]]
        record, values = arrays(expected["unit_id"])
        if record["artifact_sha256"] != expected["artifact_sha256"]:
            raise ValueError("Map index and committed artifact disagree")
        return values["scores"], values["support"]
    return prediction


def histograms(store, data, method, mode, index, native, splits=data_module.SPLITS):
    result = {}
    prediction = prediction_reader(store.out, index)
    for split in splits:
        samples = [s for s in data.samples if s["split"] == split]
        if not samples:
            continue
        identifier = f"histogram-{method}-{mode}-{split}"
        signature = digest([index, data.fp.files[str(data.root / "index.json")], native])
        cached = store.read_unit(identifier)
        if cached:
            record, arrays = cached
            if record["signature"] != signature:
                raise ValueError("Histogram input contract changed")
            hist = dict(zip(arrays["buildings"].tolist(), arrays["histograms"]))
            coverage = record["coverage"]
        else:
            hist, coverage = metrics.collect(data, samples, prediction, native)
            store.save_unit(dict(unit_id=identifier, kind="histogram", method=method, mode=mode,
                                 split=split, signature=signature, coverage=coverage),
                            dict(buildings=np.array(list(hist)), histograms=np.stack(list(hist.values()))))
        result[split] = (hist, coverage)
    return result


def threshold(histograms_by_split, pilot, native):
    if pilot or native:
        return dict(threshold=0.5, selection_split=None,
                    criterion="diagnostic pilot only; not production calibration" if pilot and not native
                    else "author binary mask; no output calibration", prediction_rule="score > threshold")
    return metrics.calibrate(histograms_by_split["val"][0])


def history(store, data, method, cache, reset_threshold):
    index = {}
    events = {(r["track_id"], r["frame"]): r for r in data.samples}
    for track in progress(list(data.tracks.values()), f"{method}: history tracks"):
        memory = TemporalMemory(max_exemplars=3)
        chain = digest([cache.signature, reset_threshold, track["track_id"], "memory-capacity-3"])
        for frame, observation in enumerate(track["observations"]):
            cache.check_time()
            chain = digest([chain, observation["rgb_sha256"], observation["support_sha256"]])
            identifier = "history-" + chain
            cached = store.read_unit(identifier)
            if cached:
                record, arrays = cached
                if record["input_sha256"] != chain:
                    raise ValueError("History state chain differs")
                memory = TemporalMemory.from_state_dict({k[8:]: v for k, v in arrays.items()
                                                        if k.startswith("memory__")})
            else:
                rgb, support = data.frame(observation["observation_id"])
                started = time.perf_counter()
                result = memory.score_then_update(rgb, support, cache, reset_threshold)
                arrays = {k: result[k] for k in ("support", "accepted_change", "reference_count", "reference_frame")}
                arrays["scores"] = result["scores"].astype(np.float32)
                arrays.update({"memory__" + k: v for k, v in memory.state_dict().items()})
                record = store.save_unit(dict(unit_id=identifier, kind="history_state", method=method,
                    mode="history", track_id=track["track_id"], frame=frame, input_sha256=chain,
                    memory_threshold=reset_threshold, comparisons=result["comparisons"],
                    update_seconds=time.perf_counter() - started,
                    accepted_pixels=int(result["accepted_change"].sum())), arrays)
            if int(memory.state_dict()["next_frame"]) != frame + 1:
                raise ValueError("History checkpoint has the wrong frame counter")
            # Artifact hash binds every preceding state, including reset decisions.
            chain = digest([chain, record["artifact_sha256"]])
            if frame >= 2:
                index[events[track["track_id"], frame]["event_id"]] = {
                    k: record[k] for k in ("unit_id", "artifact_sha256")}
    return index


def write_tables(out, results):
    rows = []
    for row in results:
        for metric in row["metrics"]["macro"]:
            rows.append({k: row[k] for k in ("method", "mode", "split")} | metric)
    if rows:
        with (out / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def fusion_report(store, data, maps, thresholds):
    """Three predeclared AND filters, after merging overlaps, without new inference."""
    results = []
    for extra in (("semantic_dino",), ("msdzip_abs",), ("semantic_dino", "msdzip_abs")):
        members = ("anychange", *extra)
        if not set(members) <= set(maps):
            continue
        name = "_and_".join(members)
        for mode in MODES:
            readers = [prediction_reader(store.out, maps[m][mode]) for m in members]
            definition = [dict(native=m == "anychange", threshold=thresholds[m][mode]["threshold"])
                          for m in members]

            def prediction(row):
                values = [reader(row) for reader in readers]
                return np.stack([value[0] for value in values]), np.logical_and.reduce([value[1] for value in values])

            for split in data_module.SPLITS:
                samples = [row for row in data.samples if row["split"] == split]
                if not samples:
                    continue
                hist, coverage = metrics.collect(data, samples, prediction, native=True, fusion=definition)
                results.append(dict(method=name, mode=mode, split=split, coverage=coverage,
                    formula="pixelwise AND after per-method overlap merge; component thresholds already frozen",
                    components=dict(zip(members, definition)),
                    metrics=metrics.report(data, hist, .5, native=True)))
    return results


def timing_report(store, data, methods, calls, startup):
    records = store.records()
    rows = []
    for method in methods:
        measured = [r for r in records if r.get("method") == method and r.get("kind") == "pair_map"
                    and r.get("scoring_seconds", 0) > 0 and not r.get("reused_from")]
        seconds = [r["scoring_seconds"] for r in measured]
        for mode in MODES:
            mode_seconds = [r["scoring_seconds"] for r in measured if r.get("mode") == mode]
            values = mode_seconds or seconds
            mean = float(np.mean(values)) if values else None
            full = data.full_counts["all"]
            low = full["history_calls_min"] if mode == "history" else full["candidates"]
            high = full["history_calls_max"] if mode == "history" else low
            rows.append(dict(method=method, mode=mode, uncached_calls_measured=len(mode_seconds),
                timing_basis="this mode" if mode_seconds else "same method, other modes" if seconds else "unmeasured",
                mean_uncached_seconds=mean, cold_start_seconds=startup.get(method),
                current_invocation=calls.get(method, {}).get(mode, {}),
                full_calls_low=low, full_calls_high=high,
                projected_seconds_low=None if mean is None else mean * low,
                projected_seconds_high=None if mean is None else mean * high))
    measured_rows = [r for r in rows if r["mean_uncached_seconds"] is not None]
    projected_bytes = 0.0
    full = data.full_counts["all"]
    for method in methods:
        for kind, count in (("pair_map", 2 * full["candidates"] + full["history_calls_max"]),
                            ("history_state", full["observations"])):
            sizes = [(store.out / r["artifact_path"]).stat().st_size for r in records
                     if r.get("method") == method and r.get("kind") == kind]
            if sizes:
                projected_bytes += float(np.mean(sizes)) * count
    result = dict(methods=rows, unmeasured=sorted(set(methods) - {r["method"] for r in measured_rows}),
        estimated_hours_low=sum(r["projected_seconds_low"] for r in measured_rows) / 3600,
        estimated_hours_high=sum(r["projected_seconds_high"] for r in measured_rows) / 3600,
        artifact_bytes=sum((store.out / r["artifact_path"]).stat().st_size for r in records),
        projected_full_artifact_bytes_without_cache_savings=projected_bytes,
        disk_estimate_scope="pilot compression ratio extrapolation; RGB memory included; not a strict bound",
        scope="measured models only; no cache savings; includes worker exchange and warmup; "
              "excludes model loading, result persistence and CPU metrics")
    write_json(store.out / "timing.json", result)
    return result


def gallery(root, data, maps, thresholds):
    samples, seen = [], set()
    for row in sorted(data.samples, key=lambda r: (r["split"], r["building_id"], r["event_id"])):
        if row["split"] in ("train", "val") and (row["split"], row["building_id"]) not in seen:
            samples.append(row)
            seen.add((row["split"], row["building_id"]))
        if len(samples) == 6:
            break
    parts = ['<!doctype html><meta charset="utf-8"><title>History pilot</title>',
             '<style>body{font-family:sans-serif}td,th{padding:6px;border:1px solid #ccc}'
             'table{border-collapse:collapse}</style>',
             '<h1>Общие целевые шаги</h1><p>TRAIN/VAL, выбор по ID здания. GT: зелёный — изменение, '
             'чёрный — известная стабильность, серый — неизвестно. '
             'Предсказание красное; вне общей геометрии серое.</p>']
    for row in samples:
        track = data.tracks[row["track_id"]]
        frames = track["observations"]
        parts.append('<h2>' + html.escape(f'{row["split"]} {row["building_id"]}: '
                                         f'{row["previous_year"]} → {row["current_year"]}') + '</h2>')
        parts.append('<p>EARLIEST / PREVIOUS / CURRENT</p>')
        for frame in (0, row["frame"] - 1, row["frame"]):
            parts.append(_thumbnail(data.frame(frames[frame]["observation_id"])[0]))
        target = data.target(row["event_id"])
        gt = np.full((224, 224, 3), 120, np.uint8)
        gt[target["target"] == 0], gt[target["target"] == 1] = 0, [0, 230, 80]
        parts.append('<p>Разметка последнего шага:</p>' + _thumbnail(gt))
        parts.append('<table><tr><th>Метод</th><th>PREVIOUS</th><th>EARLIEST</th><th>HISTORY</th></tr>')
        current = data.frame(frames[row["frame"]]["observation_id"])[0]
        for method, modes in maps.items():
            parts.append('<tr><th>' + html.escape(method) + '</th>')
            for mode in MODES:
                scores, _ = prediction_reader(root, modes[mode])(row)
                mask = scores > thresholds[method][mode]["threshold"]
                rgb = current.copy()
                rgb[mask] = (rgb[mask].astype(float) * .35 + np.array([255, 0, 0]) * .65).astype(np.uint8)
                rgb[~target["support"]] = 120
                parts.append('<td>' + _thumbnail(rgb) + '</td>')
            parts.append('</tr>')
        parts.append('</table>')
    (root / "gallery.html").write_text('\n'.join(parts), encoding="utf-8")


def run(args):
    pilot = args.command == "pilot"
    methods = args.methods or list(METHODS)
    if (not methods or len(methods) != len(set(methods)) or not set(methods) <= set(METHODS)
            or tuple(args.modes) != MODES or args.history_methods is not None):
        raise ValueError("Canonical protocol runs all three modes for each selected method; GeoSCD/MOD256 excluded")
    if args.pilot_tracks < 1 or (args.max_hours is not None and args.max_hours <= 0):
        raise ValueError("pilot-tracks and max-hours must be positive")
    if args.geoscd_run or getattr(args, "no_report", False):
        raise ValueError("Canonical runs require the common report and do not use GeoSCD caches")
    data = data_module.Dataset(args.prepared, pilot, args.pilot_tracks)
    options = load_method_options(args.methods_config) if args.methods_config else {}
    if args.methods_config:
        data.fp.add(Path(args.methods_config).expanduser().resolve())
    annotation_manifest = Path(data.index["annotation_manifest"])
    data.fp.add(annotation_manifest, data.index["input_sha256"][str(annotation_manifest)])
    code_root = Path(__file__).resolve().parents[1]
    code = {str(path.relative_to(code_root)): sha256(path) for path in sorted(code_root.rglob("*.py"))}
    config = dict(prepared=str(data.root), pilot=pilot, pilot_tracks=args.pilot_tracks, methods=methods,
                  modes=list(MODES), device=args.device, trust_checkpoint=args.trust_checkpoint,
                  options=options, reuse_run=str(Path(args.reuse_run).resolve()) if args.reuse_run else None)
    out = Path(args.out).expanduser().resolve()
    if out == data.root or data.root in out.parents or out in data.root.parents:
        raise ValueError("Run output must be separate from preparation")
    if (out / "run.json").exists() and read_json(out / "run.json").get("status") == "completed":
        raise ValueError("Run already completed; use its report or choose a new output")
    reuse = Path(args.reuse_run).expanduser().resolve() if args.reuse_run else None
    if reuse:
        record = read_json(reuse / "run.json")
        if record.get("kind") != "canonical_history_benchmark" or record.get("status") != "completed":
            raise ValueError("--reuse-run requires a completed canonical-history pilot/run")
    store = data_module.Store(out, config, resume=args.resume)
    store.record["kind"] = "canonical_history_benchmark"
    write_json(out / "run.json", store.record)
    pool = RemotePool(log_dir=out / "worker_logs")
    deadline = None if args.max_hours is None else time.monotonic() + args.max_hours * 3600
    maps, thresholds, results, calls, startup, exposure = {}, {}, [], {}, {}, {}
    try:
        env = environment()
        env = {k: v for k, v in env.items() if k not in ("git_commit", "git_dirty")}
        store.contract(dict(protocol=data_module.PROTOCOL, inputs=data.fp.files.copy(), code=code,
                            environment=env, selection=data.selection))
        write_json(out / "selection.json", data.selection)
        for method in methods:
            settings, remote = scorer_options(method, options, args, data)
            store.event("gpu_preflight", method=method, report=_gpu_report(settings.get("device")))
            started = time.perf_counter()
            scorer = RemoteScorer(method, settings, pool) if remote else make_scorer(method, **settings)
            try:
                if isinstance(scorer, RemoteScorer):
                    scorer.activate()
                startup[method] = time.perf_counter() - started
                identity = _identity(scorer.metadata)
                path = out / "method_contracts" / (method + ".json")
                if path.exists() and read_json(path) != identity:
                    raise ValueError("Model/worker identity changed since the interrupted run")
                path.parent.mkdir(exist_ok=True)
                write_json(path, identity)
                _pin_model_files(identity, data.fp)
                exposure[method] = _checkpoint_exposure(scorer.metadata)
                store.event("method_activated", method=method, cold_start_seconds=startup[method],
                            checkpoint_exposure=exposure[method])
                signature = digest([code, method, settings, identity])
                cache = PairCache(store, scorer, method, signature, deadline, reuse)
                maps[method], thresholds[method] = {}, {}
                for mode in MODES:
                    cache.mode = mode
                    if mode == "history":
                        reset = thresholds[method]["previous"]["threshold"]
                        index = history(store, data, method, cache, reset)
                    else:
                        index = {}
                        for row in progress(data.samples, f"{method}: {mode}"):
                            frames = data.tracks[row["track_id"]]["observations"]
                            previous = row["frame"] - 1 if mode == "previous" else 0
                            a, sa = data.frame(frames[previous]["observation_id"])
                            b, sb = data.frame(frames[row["frame"]]["observation_id"])
                            record, _ = cache.score(a, b, sa & sb)
                            index[row["event_id"]] = {k: record[k] for k in ("unit_id", "artifact_sha256")}
                    maps[method][mode] = index
                    hist = histograms(store, data, method, mode, index, cache.native, splits=("val",))
                    selected = threshold(hist, pilot, cache.native)
                    if mode == "history":
                        selected.update(memory_threshold=reset,
                            memory_rule="PREVIOUS threshold fixed before history; heuristic, no separate search")
                    thresholds[method][mode] = selected
                    # Do not even collect TEST target histograms before this decision is frozen.
                    hist.update(histograms(store, data, method, mode, index, cache.native, splits=("train", "test")))
                    for split, (values, coverage) in hist.items():
                        results.append(dict(method=method, mode=mode, split=split, coverage=coverage,
                            metrics=metrics.report(data, values, selected["threshold"], cache.native)))
                    calls[method] = dict(cache.calls)
                    write_json(out / "thresholds.json", thresholds)
                    write_json(out / "maps_index.json", maps)
                    write_json(out / "progress.json", dict(method=method, mode=mode, completed=True,
                                                            selected_candidates=len(data.samples)))
                    from ..method_worker import cuda_peak
                    peak = scorer.metadata.get("cuda_peak") or cuda_peak(settings.get("device"))
                    store.event("mode_completed", method=method, mode=mode, threshold=selected, cuda_peak=peak)
                    usage_path = out / "resources.json"
                    usage = read_json(usage_path) if usage_path.exists() else {}
                    usage[method] = dict(cuda_peak=peak, cold_start_seconds=startup[method])
                    write_json(usage_path, usage)
                timing_report(store, data, methods, calls, startup)
            finally:
                close = getattr(scorer, "close", None)
                if close:
                    close()
        data.fp.verify()
        if code != {str(p.relative_to(code_root)): sha256(p) for p in sorted(code_root.rglob("*.py"))}:
            raise ValueError("Benchmark source changed during the run")
        fused = fusion_report(store, data, maps, thresholds)
        summary = dict(kind="canonical_history_benchmark", status="completed", protocol=data_module.PROTOCOL,
            pilot=pilot, methods=methods, modes=list(MODES), counts=data.selection["counts"],
            calibration="diagnostic only; fixed 0.5" if pilot else "full VAL; freeze before TEST evaluation",
            checkpoint_exposure=exposure, results=results, fusion_results=fused,
            target_scope=data.index["target_scope"], independent_event_labels=False)
        write_json(out / "summary.json", summary)
        write_tables(out, results + fused)
        gallery(out, data, maps, thresholds)
        timing_report(store, data, methods, calls, startup)
        store.finish("completed", summary_sha256=sha256(out / "summary.json"),
                     maps_index_sha256=sha256(out / "maps_index.json"),
                     thresholds_sha256=sha256(out / "thresholds.json"), pilot=pilot,
                     counts=data.selection["counts"], methods=methods)
        return store.record
    except BaseException as exc:
        status = ("paused" if isinstance(exc, Paused) else
                  "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
        store.finish(status, error=str(exc))
        if isinstance(exc, Paused):
            print(str(exc), flush=True)
            return store.record
        raise
    finally:
        pool.release()
        store.close()


def report(run_path, out=None):
    """Recreate human-readable exports without model initialization or inference."""
    root = Path(run_path).expanduser().resolve()
    state = read_json(root / "run.json")
    if state.get("status") != "completed" or state.get("summary_sha256") != sha256(root / "summary.json"):
        raise ValueError("Report requires a completed, intact benchmark")
    summary = read_json(root / "summary.json")
    destination = Path(out).expanduser().resolve() if out else root / "reports"
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "summary.json", summary)
    write_tables(destination, summary["results"] + summary.get("fusion_results", []))
    print(f"CPU report: {destination}", flush=True)
