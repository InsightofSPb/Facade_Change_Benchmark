"""Cached DINO history scores, reviewed-event supervision and temporal models.

No image encoder, alignment, model download or automatic event labelling occurs.
Use the workflow's history-scores, history-models and history-models benchmark.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import html
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np
from PIL import Image

from ..history_features import Fingerprints, _inside, _l2, _npz, digest
from ..io import read_json, sha256, write_json


NETWORK_MODULE = "facade_change.research.2026-10-25_temporal_networks"
MODELS = ("pair", "pooled", "gru", "transformer", "pooled-gru", "pooled-transformer")
SCORES = ("previous", "earliest", "history_mean")
LABEL_FIELDS = ("event_id", "prefix_sha256", "track_id", "building_id", "view_id", "split",
                "previous_year", "current_year", "history_start_year", "review_status",
                "comparable", "label", "event_type", "review_source", "notes")
RunStore = importlib.import_module("facade_change.2026-10-14_temporal_experiment").RunStore


class HistoryRun(RunStore):
    def event(self, kind, **values):
        if not hasattr(self, "unique_id"):
            self.unique_id = self.record.get("history_run_id") or "history-" + uuid.uuid4().hex
        self.record.update(run_id=self.unique_id, history_run_id=self.unique_id,
                           kind="supervised_history_experiment")
        return super().event(kind, run_id=self.unique_id,
                             event_id=f"{self.unique_id}:{self.sequence + 1}", **values)


@dataclass
class HistoryData:
    index_path: Path
    index: dict
    history_root: Path
    tracks: dict
    samples: list
    fingerprints: Fingerprints
    coverage: dict


def load_histories(index_path, splits=("train", "val"), min_years=3, history_root=None):
    """Validate the immutable feature/geometry snapshots; load selected splits only."""
    path = Path(index_path).expanduser().resolve()
    if path.is_dir():
        path = path / "index.json"
    fp = Fingerprints()
    index_sha = fp.add(path)
    index = read_json(path)
    if index.get("kind") != "frozen_history_features" or index.get("schema_version") != 1:
        raise ValueError("Expected history-features index.json schema 1")
    if min_years < 2 or not splits or set(splits) - {"train", "val", "test"}:
        raise ValueError("Use min_years >= 2 and explicit train/val/test splits")
    report = _inside(path.parent, index["report"])
    fp.add(report / "index.json", index_sha)
    record = read_json(report / "run.json")
    fp.add(report / "run.json")
    if record.get("status") != "completed" or record.get("index_sha256") != index_sha:
        raise ValueError("Feature index lacks a matching completed extraction")
    root = Path(history_root or index["history_root"]).expanduser().resolve()
    manifest = _inside(root, index["history_report"]) / "manifest.json"
    fp.add(manifest, index["history_manifest_sha256"])
    document = read_json(manifest)
    history_record = read_json(manifest.parent / "run.json")
    fp.add(manifest.parent / "run.json")
    if (not history_record.get("status", "").startswith("completed")
            or history_record.get("manifest_sha256") != index["history_manifest_sha256"]):
        raise ValueError("History snapshot lacks a matching completed preparation")
    if document.get("kind") != "canonical_crop_histories":
        raise ValueError("Unexpected history manifest kind")
    tracks, buildings, views, source_splits, observations = {}, {}, {}, {}, set()
    for track in document["tracks"]:
        key, building, split = track["track_id"], track["building_id"], track["split"]
        if key in tracks or not building or split not in {"train", "val", "test"}:
            raise ValueError("Invalid/duplicate history identity or split")
        if buildings.setdefault(building, split) != split:
            raise ValueError("Building leakage across splits")
        if views.setdefault(track["view_id"], building) != building:
            raise ValueError("View leakage across buildings")
        for observation in track["observations"]:
            if observation["observation_id"] in observations:
                raise ValueError("Duplicate observation identity")
            observations.add(observation["observation_id"])
            # The crop digest also detects repeated data even in older manifests.
            for hash_key in ("image_sha256", "rgb_sha256"):
                value = observation.get(hash_key)
                if value and source_splits.setdefault(value, split) != split:
                    raise ValueError("Image/crop hash leakage across splits")
        tracks[key] = track
    samples, seen, counts = [], set(), Counter()
    dimension = index["encoder"]["dimensions"]
    for entry in index["sequences"]:
        key = entry["track_id"]
        track = tracks.get(key)
        if key in seen or track is None or any(entry[k] != track[k] for k in
                ("building_id", "view_id", "split", "bbox_xywh", "distinct_years")):
            raise ValueError("Sequence metadata disagrees with its history snapshot")
        seen.add(key)
        if entry["split"] not in splits:
            continue
        counts["selected_tracks"] += 1
        sequence = _inside(path.parent, entry["path"])
        fp.add(sequence, entry["sha256"])
        with np.load(sequence, allow_pickle=False) as data:
            years, z = data["years"].copy(), data["z_years"].copy()
            if (years.dtype.kind not in "iu" or years.ndim != 1 or len(years) < 1
                    or np.any(np.diff(years) <= 0) or len(years) != track["distinct_years"]
                    or z.shape != (len(years), dimension) or z.dtype != np.float32
                    or not np.isfinite(z).all()
                    or not np.allclose(np.linalg.norm(z, axis=1), 1, atol=1e-4)):
                raise ValueError(f"Invalid normalized sequence or chronology: {key}")
            observed_ids = data["observation_ids"].tolist()
            if (observed_ids != entry["observation_ids"]
                    or observed_ids != [row["observation_id"] for row in track["observations"]]
                    or years.tolist() != sorted({row["year"] for row in track["observations"]})
                    or not np.array_equal(data["delta_years"], np.diff(years, prepend=years[0]))
                    or not np.array_equal(data["elapsed_years"], years - years[0])):
                raise ValueError("Sequence observation/time identity mismatch")
        counts["all_pair_prefixes"] += len(years) - 1
        for t in range(1, len(years)):
            if t + 1 < min_years:
                counts["short_prefixes"] += 1
                continue
            prefix = [row for row in track["observations"] if row["year"] <= int(years[t])]
            fingerprint = digest({"encoder": index["encoder"], "track_id": key,
                                  "observations": prefix, "years": years[:t+1].tolist(),
                                  "z_sha256": hashlib.sha256(z[:t+1].tobytes()).hexdigest()})
            event_id = digest([key, int(years[t-1]), int(years[t])])
            samples.append({"event_id": event_id, "prefix_sha256": fingerprint,
                            "track_id": key, "building_id": entry["building_id"],
                            "view_id": entry["view_id"], "split": entry["split"],
                            "previous_year": int(years[t-1]), "current_year": int(years[t]),
                            "years": years[:t+1].copy(), "past": z[:t].copy(),
                            "time": np.column_stack((np.diff(years[:t], prepend=years[0]),
                                                      years[:t] - years[0])).astype(np.float32),
                            "current": z[t].copy(), "pair_gap": int(years[t] - years[t-1])})
    counts["eligible_prefixes"] = len(samples)
    coverage = {**dict(counts), "min_prefix_years": min_years, "splits": list(splits),
                "buildings": len({s["building_id"] for s in samples}),
                "by_split": dict(Counter(s["split"] for s in samples)),
                "scope": "conditional on exported geometry and frozen-feature availability",
                "upstream_history_manifest_sha256": index["history_manifest_sha256"]}
    if not samples:
        raise ValueError("No eligible prefixes in selected splits; check history length and splits")
    return HistoryData(path, index, root, tracks, samples, fp, coverage)


def cosine_scores(sample):
    current, past = sample["current"], sample["past"]
    references = (past[-1], past[0], _l2(past.mean(0)))
    return {name: float(np.clip(1 - np.dot(current, ref), 0, 2))
            for name, ref in zip(SCORES, references)}


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_labels(data, path, min_years=3):
    """Labels apply to fixed year-pooled prefixes; unknown and occluded stay unlabelled."""
    data.fingerprints.add(path)
    lookup = {row["event_id"]: row for row in data.samples}
    accepted, seen, reasons = [], set(), Counter()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(LABEL_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("Use the review.csv schema exported by history-scores")
        for label in reader:
            track = data.tracks.get(label["track_id"])
            if track is None or label["split"] != track["split"]:
                raise ValueError("Label track or split disagrees with the history manifest")
            # Held-out labels are never interpreted by a TRAIN/VAL run.
            if label["split"] not in data.coverage["splits"]:
                continue
            key = label["event_id"]
            if key in seen:
                raise ValueError(f"Duplicate event label: {key}")
            seen.add(key)
            row = lookup.get(key)
            if row is None:
                raise ValueError(f"Unknown/ineligible labelled event: {key}; use the same prefix policy")
            for field in ("prefix_sha256", "track_id", "building_id", "view_id", "split",
                          "previous_year", "current_year"):
                if label[field] != str(row[field]):
                    raise ValueError(f"Stale or mismatched event label ({field}): {key}")
            status = label["review_status"].strip()
            if status not in {"", "unreviewed", "accepted", "excluded"}:
                raise ValueError(f"Invalid review_status: {status}")
            if status != "accepted":
                reasons[status or "unreviewed"] += 1
                continue
            if label["comparable"] != "yes" or label["label"] not in {"0", "1"}:
                raise ValueError("Accepted labels need comparable=yes and binary label=0/1")
            if not label["review_source"].strip():
                raise ValueError("Accepted event labels need reviewer/evidence in review_source")
            kind = label["event_type"].strip()
            if kind in {"occluded", "uncertain", "not-comparable"}:
                raise ValueError("Occluded/uncertain regions must be excluded, not binary labels")
            if ((kind == "unchanged" and label["label"] != "0")
                    or (kind in {"repair", "crack", "degradation", "appearance", "disappearance"}
                        and label["label"] != "1")):
                raise ValueError("Event type contradicts the binary event label")
            sample = dict(row)
            # A reviewed, already-known repair can begin a new local episode.
            # This is explicit input metadata, never inferred from the target label.
            start_year = int(label["history_start_year"] or row["years"][0])
            if start_year not in row["years"][:-1]:
                raise ValueError("history_start_year must be an observed year before the target")
            offset = int(np.flatnonzero(row["years"] == start_year)[0])
            sample["years"] = row["years"][offset:]
            sample["past"] = row["past"][offset:]
            if len(sample["years"]) < min_years:
                reasons["short_after_episode_reset"] += 1
                continue
            years = sample["years"][:-1]
            sample["time"] = np.column_stack((np.diff(years, prepend=years[0]), years-years[0])).astype(np.float32)
            sample.update(label=int(label["label"]), event_type=kind or "unspecified",
                          history_start_year=start_year, review_source=label["review_source"])
            accepted.append(sample)
    coverage = {"accepted": len(accepted), "excluded_or_unreviewed": dict(reasons),
                "missing_rows": len(set(lookup) - seen),
                "by_split": dict(Counter(s["split"] for s in accepted)),
                "buildings": len({s["building_id"] for s in accepted})}
    repairs = {}
    for sample in accepted:
        if sample["event_type"] == "repair":
            repairs.setdefault(sample["track_id"], []).append(sample["current_year"])
    for sample in accepted:
        past_repairs = [year for year in repairs.get(sample["track_id"], [])
                        if year <= sample["previous_year"]]
        if past_repairs and sample["history_start_year"] < max(past_repairs):
            raise ValueError(f"Known past repair requires history_start_year >= {max(past_repairs)} "
                             f"for event {sample['event_id']}")
    return accepted, coverage


def building_weights(samples):
    counts = Counter(row["building_id"] for row in samples)
    return np.array([1.0 / counts[row["building_id"]] for row in samples])


def average_precision(labels, scores, weights=None):
    """Non-interpolated AP, grouping ties exactly as sklearn average_precision_score."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    weights = np.ones(len(labels)) if weights is None else np.asarray(weights, dtype=float)
    if not len(labels) or not np.isfinite(scores).all():
        raise ValueError("AP requires nonempty finite scores")
    active = weights > 0
    labels, scores, weights = labels[active], scores[active], weights[active]
    positive = np.sum(labels * weights)
    if positive == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum((labels * weights)[order])
    total = np.cumsum(weights[order])
    ends = np.r_[np.flatnonzero(np.diff(scores[order]) != 0), len(scores)-1]
    recalls = tp[ends] / positive
    return float(np.sum(np.diff(recalls, prepend=0) * tp[ends] / total[ends]))


def metric_values(samples, scores, threshold, weights=None):
    labels = np.array([row["label"] for row in samples])
    scores = np.asarray(scores)
    weights = building_weights(samples) if weights is None else weights
    pred = scores >= threshold
    positive, negative = weights * labels, weights * (1-labels)
    return {"building_weighted_ap": average_precision(labels, scores, weights),
            "micro_ap_diagnostic": average_precision(labels, scores),
            "recall": float(np.sum(positive * pred) / positive.sum()) if positive.sum() else None,
            "actual_fpr": float(np.sum(negative * pred) / negative.sum()) if negative.sum() else None,
            "threshold": float(threshold), "events": len(samples),
            "positive_events": int(labels.sum()), "buildings": len({s["building_id"] for s in samples})}


def select_threshold(samples, scores, budget):
    """Lowest observed threshold within the building-weighted VAL FPR budget."""
    labels = np.array([s["label"] for s in samples])
    weights, scores = building_weights(samples), np.asarray(scores, dtype=float)
    if set(labels.tolist()) != {0, 1} or not 0 <= budget < 1 or not np.isfinite(scores).all():
        raise ValueError("Calibration requires both classes, finite scores and 0 <= FPR < 1")
    order = np.argsort(-scores, kind="stable")
    negative = weights * (1-labels)
    cumulative = np.cumsum(negative[order]) / negative.sum()
    ends = np.r_[np.flatnonzero(np.diff(scores[order]) != 0), len(scores)-1]
    eligible = ends[cumulative[ends] <= budget + 1e-12]
    return float(scores[order][eligible[-1]]) if len(eligible) else float(np.nextafter(scores.max(), np.inf))


def report_metrics(samples, scores, threshold, bootstrap=0, seed=42):
    result = metric_values(samples, scores, threshold)
    labels = np.array([s["label"] for s in samples])
    scores = np.asarray(scores)
    result["event_types"] = {}
    for kind in sorted({s["event_type"] for s in samples if s["label"] == 1}):
        selected = [i for i, s in enumerate(samples) if s["label"] == 0 or s["event_type"] == kind]
        subset = [samples[i] for i in selected]
        result["event_types"][kind] = metric_values(subset, scores[selected], threshold)
    groups = sorted({s["building_id"] for s in samples})
    per_building = []
    for group in groups:
        ids = [i for i, s in enumerate(samples) if s["building_id"] == group]
        ap = average_precision(labels[ids], scores[ids])
        if ap is not None:
            per_building.append(ap)
    result["macro_ap_positive_buildings"] = float(np.mean(per_building)) if per_building else None
    result["macro_ap_eligible_buildings"] = len(per_building)
    result["bootstrap"] = {"requested": bootstrap, "unit": "building_id", "intervals": {}}
    if bootstrap and len(groups) >= 2:
        rng, estimates = np.random.default_rng(seed), []
        base = building_weights(samples)
        for _ in range(bootstrap):
            frequencies = Counter(rng.choice(groups, len(groups), replace=True))
            weights = base * np.array([frequencies[s["building_id"]] for s in samples])
            estimates.append(metric_values(samples, scores, threshold, weights))
        for name in ("building_weighted_ap", "recall", "actual_fpr"):
            values = [r[name] for r in estimates if r[name] is not None]
            result["bootstrap"]["intervals"][name] = {
                "valid_replicates": len(values),
                "percentile_95": np.quantile(values, [.025, .975]).tolist() if values else None}
    return result


def paired_ap_differences(samples, columns, bootstrap, seed):
    """Paired facade resampling; fixes the same validation-trained models each time."""
    comparisons = []
    for name in columns:
        if name not in SCORES:
            reference = "pair_" + name.rsplit("_", 1)[1]
            if reference in columns and name != reference:
                comparisons.append((name, reference))
        if name.endswith("_gaps") and name[:-5] + "_none" in columns:
            comparisons.append((name, name[:-5] + "_none"))
    labels, base = [s["label"] for s in samples], building_weights(samples)
    groups = sorted({s["building_id"] for s in samples})
    estimates = {name: average_precision(labels, values, base) for name, values in columns.items()}
    replicates = {pair: [] for pair in comparisons}
    rng = np.random.default_rng(seed)
    if len(groups) >= 2:
        for _ in range(bootstrap):
            frequencies = Counter(rng.choice(groups, len(groups), replace=True))
            weights = base * np.array([frequencies[s["building_id"]] for s in samples])
            aps = {name: average_precision(labels, values, weights) for name, values in columns.items()}
            for pair in comparisons:
                if all(aps[name] is not None for name in pair):
                    replicates[pair].append(aps[pair[0]] - aps[pair[1]])
    return {f"{a}_minus_{b}": {
        "difference": estimates[a] - estimates[b] if estimates[a] is not None and estimates[b] is not None else None,
        "percentile_95": np.quantile(replicates[(a, b)], [.025, .975]).tolist() if replicates[(a, b)] else None,
        "valid_replicates": len(replicates[(a, b)]), "unit": "building_id"} for a, b in comparisons}


def render_gallery(data, rows, path, limit=60, ranking="disagreement"):
    """A diagnostic sample, with target-bounded chronology and all same-year images."""
    if ranking == "disagreement":
        priority = lambda r: (-np.ptp([r[name] for name in SCORES]), r["event_id"])
    else:
        priority = lambda r: (-max(r[name] for name in SCORES), r["event_id"])
    selected = sorted(rows, key=priority)[:limit]
    sections = []
    for row in selected:
        frames = []
        for observation in data.tracks[row["track_id"]]["observations"]:
            if not int(row.get("history_start_year") or -9999) <= observation["year"] <= row["current_year"]:
                continue
            source = _inside(data.history_root, observation["rgb"])
            if sha256(source) != observation["rgb_sha256"]:
                raise ValueError(f"Gallery crop hash mismatch: {source}")
            with Image.open(source) as original:
                thumbnail = original.convert("RGB")
                thumbnail.thumbnail((150, 150))
                stream = io.BytesIO()
                thumbnail.save(stream, "JPEG", quality=85)
            encoded = base64.b64encode(stream.getvalue()).decode()
            frames.append(f'<figure><img loading="lazy" src="data:image/jpeg;base64,{encoded}">'
                          f'<figcaption>{observation["year"]}</figcaption></figure>')
        values = " · ".join(f"{name}: {row[name]:.4f}" for name in SCORES)
        model_values = " · ".join(f"{html.escape(k)}: {v:.4f}" for k, v in row.items()
                                 if k.startswith("probability_"))
        sections.append(f'<section><h2>{html.escape(row["building_id"])} · {row["previous_year"]} → '
                        f'{row["current_year"]} · {html.escape(row["split"])}</h2>'
                        f'<code>{row["event_id"]}</code><div class="frames">{"".join(frames)}</div>'
                        f'<p>{values}</p><p>{model_values}</p></section>')
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Facade history review</title>'
            '<style>body{font:16px system-ui;margin:32px;background:#f2f4f7;color:#17202a}'
            'section{background:white;padding:20px;margin:20px 0;border-radius:12px}'
            '.frames{display:flex;flex-wrap:wrap;gap:12px}figure{margin:12px 0}h2{font-size:18px}'
            'code{overflow-wrap:anywhere}img{max-width:150px}figcaption{text-align:center}</style>'
            f'<h1>Facade history review</h1><p>{len(selected)} of {len(rows)} events, ranked by '
            f'{html.escape(ranking)}. This is a selected diagnostic gallery, not a representative '
            'quality estimate. Scores indicate appearance differences, not confirmed damage. '
            'Review the latest transition, including visibility, alignment and every same-year alternative. '
            'Fill review.csv; use excluded for ambiguous or incomparable cases.</p>'
            + "".join(sections) + '</html>')
    Path(path).write_text(page, encoding="utf-8")


def finish_run(store, data, summary):
    data.fingerprints.verify()
    write_json(store.out / "summary.json", summary)
    artifacts = {p.relative_to(store.out).as_posix(): sha256(p)
                 for p in sorted(store.out.rglob("*"))
                 if p.is_file() and p.name not in {"run.json", "events.jsonl", ".run.lock"}}
    store.finish("completed", summary=summary, artifact_sha256=artifacts)


def score_histories(args):
    data = load_histories(args.features, args.splits, args.min_years, args.history_root)
    config = vars(args).copy()
    store = HistoryRun(Path(args.out), config)
    try:
        store.contract({"inputs": data.fingerprints.files, "module_sha256": sha256(__file__)})
        rows = [{**{k: s[k] for k in LABEL_FIELDS if k in s}, **cosine_scores(s),
                 "history_start_year": int(s["years"][0]), "review_status": "unreviewed"}
                for s in data.samples]
        write_csv(store.out / "scores.csv", rows, (*LABEL_FIELDS, *SCORES))
        write_csv(store.out / "review.csv", rows, LABEL_FIELDS)
        render_gallery(data, rows, store.out / "gallery.html", args.gallery_limit)
        render_gallery(data, rows, store.out / "largest_changes.html", args.gallery_limit, "largest score")
        finish_run(store, data, {"coverage": data.coverage, "scores": list(SCORES),
                                "event_labels_created": False})
        return {"out": str(store.out), "events": len(rows)}
    except BaseException as exc:
        store.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        store.close()


def torch_runtime(device, seed):
    import torch
    if device != "cpu":
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise ValueError(f"Requested device unavailable: {device}")
        try:
            status = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv"],
                                    capture_output=True, text=True, timeout=10, check=False)
            print(status.stdout or status.stderr, flush=True)
        except FileNotFoundError:
            print("nvidia-smi unavailable; checking free memory with Torch", flush=True)
        free, total = torch.cuda.mem_get_info(device)
        print(f"{device}: {free / 2**30:.2f} / {total / 2**30:.2f} GiB free", flush=True)
        if free < 256 * 2**20:
            raise RuntimeError("Less than 256 MiB GPU memory available; use CPU or retry later")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def atomic_checkpoint(torch, path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def predict(model, samples, device, batch_size, shuffle_past=False, seed=42):
    import torch
    networks = importlib.import_module(NETWORK_MODULE)
    probabilities, embeddings = [], []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(samples), batch_size):
            batch = networks.batch_samples(samples[offset:offset+batch_size], device, shuffle_past, seed)
            logits, history = model(**batch)
            probabilities.extend(logits.sigmoid().cpu().tolist())
            embeddings.extend(history.cpu().numpy())
    return np.asarray(probabilities), np.asarray(embeddings)


def train_one(store, train, val, spec, args, torch):
    networks = importlib.import_module(NETWORK_MODULE)
    key = f"{spec['model']}_{spec['time_mode']}"
    folder = store.out / key
    folder.mkdir(exist_ok=True)
    model = networks.HistoryClassifier(**spec).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    last, best = folder / "last.pth", folder / "best.pth"
    start, best_ap, stale, best_epoch, best_state = 0, -1.0, 0, 0, None
    if last.exists():
        checkpoint = torch.load(last, map_location="cpu", weights_only=True)
        if checkpoint["spec"] != spec:
            raise ValueError("Checkpoint specification differs from the requested model")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, best_ap, stale = checkpoint["epoch"], checkpoint["best_ap"], checkpoint["stale"]
        best_epoch, best_state = checkpoint["best_epoch"], checkpoint["best_state"]
        torch.set_rng_state(checkpoint["rng"])
        if args.device != "cpu":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    weights = building_weights(train)
    weights /= weights.mean()
    for epoch in range(start, args.epochs):
        if stale >= args.patience:
            break
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(len(train))
        loss_sum = 0.0
        for offset in range(0, len(train), args.batch_size):
            ids = order[offset:offset+args.batch_size]
            rows = [train[i] for i in ids]
            batch = networks.batch_samples(rows, args.device)
            targets = torch.tensor([row["label"] for row in rows], dtype=torch.float32, device=args.device)
            logits, _ = model(**batch)
            losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            loss = (losses * torch.tensor(weights[ids], dtype=torch.float32, device=args.device)).mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
        scores, _ = predict(model, val, args.device, args.batch_size)
        ap = average_precision([s["label"] for s in val], scores, building_weights(val))
        improved = ap > best_ap
        if improved:
            best_ap, best_epoch, stale = ap, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        payload = {"spec": spec, "epoch": epoch+1, "state_dict": model.state_dict(),
                   "optimizer": optimizer.state_dict(), "best_ap": best_ap, "best_epoch": best_epoch,
                   "best_state": best_state, "stale": stale, "rng": torch.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if args.device != "cpu" else []}
        atomic_checkpoint(torch, last, payload)
        store.event("epoch_completed", model=key, epoch=epoch+1, train_loss=loss_sum/len(train),
                    val_building_weighted_ap=ap, best_epoch=best_epoch, checkpoint_sha256=sha256(last))
        print(f"{key}: epoch {epoch+1}/{args.epochs}, loss={loss_sum/len(train):.5f}, "
              f"VAL AP={ap:.5f}, best epoch={best_epoch}", flush=True)
    if best_state is None:
        raise ValueError("No trained checkpoint was produced")
    model.load_state_dict(best_state, strict=True)
    atomic_checkpoint(torch, best, {"spec": spec, "state_dict": best_state, "epoch": best_epoch})
    scores, embeddings = predict(model, val, args.device, args.batch_size)
    threshold = select_threshold(val, scores, args.fpr)
    _npz(folder / "val_predictions.npz", {"event_ids": np.array([s["event_id"] for s in val]),
                                          "probabilities": scores, "history_embeddings": embeddings})
    metrics = report_metrics(val, scores, threshold)
    info = {"spec": spec, "checkpoint": best.relative_to(store.out).as_posix(),
            "checkpoint_sha256": sha256(best), "best_epoch": best_epoch,
            "parameters": sum(p.numel() for p in model.parameters()),
            "threshold": threshold, "val": metrics,
            "embedding_scope": "observations strictly before the current target"}
    write_json(folder / "result.json", info)
    store.event("model_completed", model=key, checkpoint_sha256=info["checkpoint_sha256"])
    return info


def train_models(args):
    data = load_histories(args.features, ("train", "val"), args.min_years, args.history_root)
    samples, coverage = read_labels(data, args.labels, args.min_years)
    train = [s for s in samples if s["split"] == "train"]
    val = [s for s in samples if s["split"] == "val"]
    for split, rows in (("train", train), ("val", val)):
        if {s["label"] for s in rows} != {0, 1}:
            raise ValueError(f"{split.upper()} needs reviewed positive and negative events. "
                             "Fill review.csv from history-scores; scores/semantic masks are not event GT.")
    config = {k: v for k, v in vars(args).items() if k != "resume"}
    existing = Path(args.out) / "run.json"
    if args.resume and existing.exists() and read_json(existing).get("status") == "completed":
        raise ValueError("Training is already completed; use benchmark history-models or a new output directory")
    store = HistoryRun(Path(args.out), config, resume=args.resume)
    try:
        import torch
        networks = importlib.import_module(NETWORK_MODULE)
        contract = {"inputs": data.fingerprints.files, "module_sha256": sha256(__file__),
                    "networks_sha256": sha256(networks.__file__), "torch": str(torch.__version__),
                    "numpy": np.__version__, "min_years": args.min_years,
                    "primary_metric": "building_weighted_ap", "test_used": False,
                    "train_event_ids": [s["event_id"] for s in train],
                    "val_event_ids": [s["event_id"] for s in val]}
        store.contract(contract)
        torch_runtime(args.device, args.seed)
        store.event("device_ready", device=args.device, cuda_runtime=torch.version.cuda,
                    gpu_name=torch.cuda.get_device_name(args.device) if args.device != "cpu" else None)
        write_json(store.out / "coverage.json", {"features": data.coverage, "labels": coverage})
        baselines = {}
        val_scores = [cosine_scores(row) for row in val]
        for name in SCORES:
            values = [row[name] for row in val_scores]
            threshold = select_threshold(val, values, args.fpr)
            baselines[name] = {"threshold": threshold, "val": report_metrics(val, values, threshold)}
        models = {}
        for model in args.models:
            for time_mode in args.time_modes:
                torch.manual_seed(args.seed)
                spec = {"dimension": data.index["encoder"]["dimensions"], "model": model,
                        "width": args.width, "dropout": args.dropout, "time_mode": time_mode}
                key = f"{model}_{time_mode}"
                models[key] = train_one(store, train, val, spec, args, torch)
        bundle = {"schema_version": 1, "models": models, "baselines": baselines,
                  "networks_sha256": sha256(networks.__file__),
                  "encoder": data.index["encoder"], "min_years": args.min_years,
                  "fpr_budget": args.fpr, "seed": args.seed,
                  "building_splits": {t["building_id"]: t["split"] for t in data.tracks.values()},
                  "feature_index_sha256": sha256(data.index_path),
                  "labels_sha256": sha256(args.labels), "test_used": False}
        write_json(store.out / "models.json", bundle)
        finish_run(store, data, {"models": list(models), "labels": coverage,
                                "selection": "best building-weighted validation AP; threshold from VAL only",
                                "fpr_budget": args.fpr, "test_used": False})
        return {"out": str(store.out), "models": list(models)}
    except BaseException as exc:
        store.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        store.close()


def evaluate_models(args):
    run = Path(args.training_run).expanduser().resolve()
    record, bundle = read_json(run / "run.json"), read_json(run / "models.json")
    if record.get("status") != "completed" or record.get("artifact_sha256", {}).get("models.json") != sha256(run / "models.json"):
        raise ValueError("Training run is incomplete or models.json is corrupt")
    data = load_histories(args.features, args.splits, bundle["min_years"], args.history_root)
    if data.index["encoder"] != bundle["encoder"]:
        raise ValueError("Feature encoder/preprocessing differs from training")
    for track in data.tracks.values():
        old = bundle["building_splits"].get(track["building_id"])
        if old is not None and old != track["split"]:
            raise ValueError("A training building changed splits")
    if args.labels:
        samples, coverage = read_labels(data, args.labels, bundle["min_years"])
    else:
        samples, coverage = data.samples, {"status": "unlabelled inference; metrics not computed"}
    data.fingerprints.add(run / "run.json")
    data.fingerprints.add(run / "models.json")
    for info in bundle["models"].values():
        data.fingerprints.add(_inside(run, info["checkpoint"]), info["checkpoint_sha256"])
    if not samples:
        raise ValueError("No eligible events in selected evaluation splits")
    store = HistoryRun(Path(args.out), vars(args).copy())
    try:
        import torch
        networks = importlib.import_module(NETWORK_MODULE)
        if sha256(networks.__file__) != bundle["networks_sha256"]:
            raise ValueError("Temporal network implementation changed since training")
        store.contract({"inputs": data.fingerprints.files, "module_sha256": sha256(__file__),
                        "networks_sha256": sha256(networks.__file__), "thresholds_frozen": True})
        torch_runtime(args.device, bundle["seed"])
        store.event("device_ready", device=args.device, cuda_runtime=torch.version.cuda,
                    gpu_name=torch.cuda.get_device_name(args.device) if args.device != "cpu" else None)
        rows = [{**{k: s[k] for k in LABEL_FIELDS if k in s}, **cosine_scores(s)} for s in samples]
        columns = {name: np.array([r[name] for r in rows]) for name in SCORES}
        thresholds = {name: info["threshold"] for name, info in bundle["baselines"].items()}
        shuffled = {}
        for name, info in bundle["models"].items():
            checkpoint = _inside(run, info["checkpoint"])
            data.fingerprints.add(checkpoint, info["checkpoint_sha256"])
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if payload["spec"] != info["spec"]:
                raise ValueError("Saved model specification mismatch")
            model = networks.HistoryClassifier(**info["spec"]).to(args.device)
            model.load_state_dict(payload["state_dict"], strict=True)
            values, embeddings = predict(model, samples, args.device, args.batch_size)
            columns[name], thresholds[name] = values, info["threshold"]
            for row, value in zip(rows, values):
                row[f"probability_{name}"] = float(value)
            _npz(store.out / f"{name}_embeddings.npz", {
                "event_ids": np.array([s["event_id"] for s in samples]),
                "history_through_year": np.array([s["previous_year"] for s in samples]),
                "history_start_year": np.array([int(s["years"][0]) for s in samples]),
                "history_embeddings": embeddings, "probabilities": values})
            if info["spec"]["model"] in {"gru", "transformer"}:
                selected = [i for i, s in enumerate(samples) if len(s["past"]) >= 3]
                if selected:
                    values, _ = predict(model, [samples[i] for i in selected], args.device,
                                        args.batch_size, True, bundle["seed"])
                    shuffled[name] = (selected, values)
            del model
        metrics, perturbations, differences = {}, {}, {}
        for split in args.splits if args.labels else []:
            ids = [i for i, s in enumerate(samples) if s["split"] == split]
            if not ids:
                continue
            subset = [samples[i] for i in ids]
            metrics[split] = {name: report_metrics(subset, values[ids], thresholds[name], args.bootstrap,
                                                   bundle["seed"]) for name, values in columns.items()}
            differences[split] = paired_ap_differences(
                subset, {name: values[ids] for name, values in columns.items()}, args.bootstrap, bundle["seed"])
            perturbations[split] = {}
            for name, (selected, values) in shuffled.items():
                positions = [j for j, i in enumerate(selected) if samples[i]["split"] == split]
                if not positions:
                    continue
                original_ids = [selected[j] for j in positions]
                matched = [samples[i] for i in original_ids]
                perturbations[split][name] = {
                    "normal": report_metrics(matched, columns[name][original_ids], thresholds[name]),
                    "shuffled": report_metrics(matched, values[positions], thresholds[name]),
                    "scope": "inference perturbation; earlier appearances permuted, latest pair and date slots fixed"}
        write_csv(store.out / "predictions.csv", rows, (*LABEL_FIELDS, *SCORES,
                    *(f"probability_{name}" for name in bundle["models"])))
        write_json(store.out / "metrics.json", {"splits": metrics, "past_shuffle": perturbations,
                                               "paired_ap_differences": differences,
                                               "primary_metric": "building_weighted_ap",
                                               "threshold_origin": "training-run validation only"})
        render_gallery(data, rows, store.out / "gallery.html", args.gallery_limit)
        finish_run(store, data, {"feature_coverage": data.coverage, "label_coverage": coverage,
                                "splits": args.splits, "models": list(bundle["models"]),
                                "metrics": metrics, "thresholds_frozen": True})
        return {"out": str(store.out), "splits": args.splits, "events": len(samples)}
    except BaseException as exc:
        store.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        store.close()


def parser():
    cli = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    actions = cli.add_subparsers(dest="action", required=True)
    for name in ("score", "train", "evaluate"):
        command = actions.add_parser(name, allow_abbrev=False)
        command.add_argument("--features", required=True, help="features directory or index.json")
        command.add_argument("--history-root", help="Relocated RGB history root; original snapshot hashes still checked")
        command.add_argument("--out", required=True, help="New output directory (train alone supports --resume)")
        if name != "evaluate":
            command.add_argument("--min-years", type=int, default=3, help="Years available at target time, not full-track length")
        if name != "train":
            command.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val"])
            command.add_argument("--gallery-limit", type=int, default=60)
        if name != "score":
            command.add_argument("--labels", required=name == "train",
                                 help="Reviewed event CSV; omit during evaluate for unlabelled inference only")
            command.add_argument("--device", default="cpu")
            command.add_argument("--batch-size", type=int, default=64)
        if name == "train":
            command.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS[:4]))
            command.add_argument("--time-modes", nargs="+", choices=("gaps", "none"), default=["gaps", "none"])
            command.add_argument("--width", type=int, default=128)
            command.add_argument("--dropout", type=float, default=.1)
            command.add_argument("--epochs", type=int, default=100)
            command.add_argument("--patience", type=int, default=15)
            command.add_argument("--learning-rate", type=float, default=1e-3)
            command.add_argument("--weight-decay", type=float, default=1e-4)
            command.add_argument("--fpr", type=float, default=.05)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--resume", action="store_true")
        if name == "evaluate":
            command.add_argument("--training-run", required=True)
            command.add_argument("--bootstrap", type=int, default=500)
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    for name in ("min_years", "batch_size", "epochs", "patience"):
        if hasattr(args, name) and getattr(args, name) < (2 if name == "min_years" else 1):
            cli.error(f"--{name.replace('_', '-')} is too small")
    for name in ("gallery_limit", "bootstrap"):
        if hasattr(args, name) and getattr(args, name) < 0:
            cli.error(f"--{name.replace('_', '-')} must be nonnegative")
    if args.action == "train":
        if (args.width < 4 or args.width % 4 or not 0 <= args.dropout < 1
                or not 0 <= args.fpr < 1 or args.learning_rate <= 0 or args.weight_decay < 0
                or len(set(args.models)) != len(args.models) or len(set(args.time_modes)) != len(args.time_modes)):
            cli.error("Invalid model width, dropout, optimizer, FPR or duplicate model settings")
    try:
        result = {"score": score_histories, "train": train_models, "evaluate": evaluate_models}[args.action](args)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"history models failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
