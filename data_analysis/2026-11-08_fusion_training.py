"""Read-only semantic reuse and bounded-memory fitting on all TRAIN pixels."""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from facade_change.io import read_json, sha256

DATE = "2026-11-08"
benchmark = importlib.import_module("facade_change.research.2026-11-04_history_benchmark")


class SemanticCache:
    """A completed fusion run supplies predictions, never an inference fallback."""

    def __init__(self, path, source):
        self.out = Path(path).expanduser().resolve()
        record = read_json(self.out / "run.json")
        request = read_json(self.out / "request.json")
        if (record.get("kind") != "canonical_history_fusion"
                or record.get("status") != "completed"
                or request.get("source_sha256") != source.hashes):
            raise ValueError("Semantic cache must be a completed fusion of exactly this source run")
        self.hashes = {str(self.out / name): sha256(self.out / name)
                       for name in ("run.json", "request.json")}
        model = self.out / "2026-11-06_semantic_model.json"
        if model.is_file():
            fusion = importlib.import_module("data_analysis.2026-11-06_history_fusion")
            fusion.verify_semantic_identity(source.contract, read_json(model))
            self.hashes[str(model)] = sha256(model)
        elif not request.get("semantic_cache"):
            raise ValueError("Semantic cache lacks its model provenance")
        self.parent = (SemanticCache(request["semantic_cache"], source)
                       if request.get("semantic_cache") else None)
        if self.parent:
            self.hashes.update(self.parent.hashes)

    def read_unit(self, identifier):
        if not identifier.startswith("2026-11-06_semantic-"):
            raise ValueError("This read-only cache supplies semantic observations only")
        result = benchmark.read_unit(self.out, identifier)
        if result is None and self.parent:
            return self.parent.read_unit(identifier)
        if result is None:
            raise ValueError(f"Missing semantic cache entry {identifier}; no inference was started")
        return result

    def verify(self):
        for path, expected in self.hashes.items():
            if sha256(path) != expected:
                raise ValueError(f"Semantic cache changed: {path}")


def all_pixels(source, store, context):
    """One on-disk float32 matrix; rows are unique within a view/transition."""
    identifier = DATE + "_train-all-pixels"
    cached = store.read_unit(identifier)
    if cached:
        record = cached[0]
        path = store.out / record["pixels_path"]
        if sha256(path) != record["pixels_sha256"]:
            raise ValueError("All-pixel TRAIN cache is corrupt")
    else:
        width = len(source.columns) + 2 * len(source.data.categories) + 2
        path = store.out / (identifier + ".f32")
        temporary = path.with_suffix(".partial")
        buildings, ranges, count = [], [], 0
        with temporary.open("wb") as stream:
            for building in sorted(b for b, split in source.assignments.items() if split == "train"):
                start = count
                samples = [s for s in source.data.samples if s["building_id"] == building]
                for _, _, scores, truth, geometric, _ in source.blocks(samples, context):
                    valid = geometric & (truth[0] != 255)
                    rows = np.empty((int(valid.sum()), width), np.float32)
                    rows[:, :-2] = scores[:, valid].T
                    rows[:, -2] = truth[0, valid]
                    rows[:, -1] = len(buildings)
                    rows.tofile(stream)
                    count += len(rows)
                if count > start:
                    buildings.append(building)
                    ranges.append([start, count])
                print(f"TRAIN all pixels: {building}, {count - start}", flush=True)
        if not count:
            temporary.unlink()
            raise ValueError("No known TRAIN pixels")
        temporary.replace(path)
        record = store.save_unit(dict(unit_id=identifier, kind="all_training_pixels", split="train",
            pixels_path=path.name, pixels_sha256=sha256(path), shape=[count, width],
            buildings=buildings, ranges=ranges,
            sampling="all known TRAIN pixels after spatial overlap merge"), {})
    matrix = np.memmap(path, mode="r", dtype=np.float32, shape=tuple(record["shape"]))
    if set(np.unique(matrix[:, -2])) != {0, 1}:
        raise ValueError("TRAIN needs both changed and stable annotated pixels")
    return dict(x=matrix[:, :-2], y=matrix[:, -2], buildings=matrix[:, -1],
                ranges=record["ranges"], pixels_sha256=record["pixels_sha256"])


def fit_streaming(sample, normalizer, variant, groups, fusion, l2, chunk_size=8192):
    """Same convex objective as fit_logistic, without a full interaction matrix."""
    from scipy.optimize import minimize
    from scipy.special import expit

    width = len(normalizer["low"])
    ranges = sample["ranges"]

    def chunks():
        for start, stop in ranges:
            weight = 1.0 / ((stop - start) * len(ranges))
            for left in range(start, stop, chunk_size):
                right = min(left + chunk_size, stop)
                raw = sample["x"][left:right]
                x = fusion.design(fusion.normalize(raw[:, :width], normalizer),
                                  raw[:, width:], variant, groups).astype(np.float64)
                yield x, sample["y"][left:right], weight

    center = second = None
    prevalence = 0.0
    for x, y, weight in chunks():
        if center is None:
            center, second = np.zeros(x.shape[1]), np.zeros(x.shape[1])
        center += weight * x.sum(axis=0)
        second += weight * np.square(x).sum(axis=0)
        prevalence += weight * y.sum(dtype=np.float64)
    scale = np.maximum(np.sqrt(np.maximum(second - center ** 2, 0)), .05)
    start = np.zeros(len(center) + 1)
    start[0] = np.log(prevalence / (1 - prevalence))
    calls = 0

    def objective(parameters):
        nonlocal calls
        loss = .5 * l2 * np.dot(parameters[1:], parameters[1:])
        gradient = np.r_[0., l2 * parameters[1:]]
        for x, y, weight in chunks():
            z = (x - center) / scale
            logits = parameters[0] + z @ parameters[1:]
            error = weight * (expit(logits) - y)
            loss += weight * np.sum(np.logaddexp(0, logits) - y * logits)
            gradient[0] += error.sum()
            gradient[1:] += z.T @ error
        calls += 1
        if calls == 1 or calls % 10 == 0:
            print(f"Fit {variant['id']}: objective pass {calls}, loss={loss:.7f}", flush=True)
        return float(loss), gradient

    fitted = minimize(objective, start, method="L-BFGS-B", jac=True,
                      options=dict(maxiter=300, ftol=1e-10, gtol=1e-6))
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise ValueError(f"Fusion optimization failed: {fitted.message}")
    return dict(center=center, scale=scale, parameters=fitted.x,
                iterations=np.array(fitted.nit), objective=np.array(fitted.fun))
