#!/usr/bin/env python3
"""Class-conditioned localisation from saved real temporal GT and predictions.

This module never infers classes from a binary change detector. Each class is an
evaluation slice defined by its source annotations. No sky is inferred from
unannotated pixels. It is a companion to real_gt_calibration.py.
"""
import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np


METRICS = ("precision", "recall", "f1", "iou", "fpr", "marked_fraction")
COUNTS = ("tp", "fp", "fn", "tn", "valid_pixels", "positive_pixels", "negative_pixels")
GROUP = ("family", "mode", "predictor", "goal", "region", "class_id", "class_name", "direction")
PAIR_FIELDS = ("family", "mode", "predictor", "goal", "heldout_building", "building_id", "sequence_id", "frame",
               "region", "class_id", "class_name", "channel", "direction") + COUNTS + METRICS


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(counts):
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    return dict(precision=ratio(tp, tp + fp) if tp + fp else (0.0 if fn else None),
                recall=ratio(tp, tp + fn), f1=ratio(2 * tp, 2 * tp + fp + fn),
                iou=ratio(tp, tp + fp + fn), fpr=ratio(fp, fp + tn),
                marked_fraction=ratio(tp + fp, tp + fp + fn + tn))


def _evaluate(prediction, positive, negative):
    counts = dict(tp=int(np.count_nonzero(prediction & positive)),
                  fp=int(np.count_nonzero(prediction & negative)),
                  fn=int(np.count_nonzero(~prediction & positive)),
                  tn=int(np.count_nonzero(~prediction & negative)),
                  valid_pixels=int(np.count_nonzero(positive | negative)),
                  positive_pixels=int(np.count_nonzero(positive)),
                  negative_pixels=int(np.count_nonzero(negative)))
    return {**counts, **_metrics(counts)}


def _categories(categories):
    if not isinstance(categories, list) or not 1 <= len(categories) <= 16:
        raise ValueError("Expected 1–16 categories in packed-bit channel order")
    result = []
    for channel, category in enumerate(categories):
        if not isinstance(category, dict) or "id" not in category or not category.get("name"):
            raise ValueError("Each category needs its source id and name")
        result.append((channel, category["id"], str(category["name"])))
    if len({str(category_id) for _, category_id, _ in result}) != len(result):
        raise ValueError("Duplicate category ids")
    return result


def evaluate_class_slices(prediction, reference_bits, source_bits, support, labels,
                          categories, region_mask=None):
    """Return all categories × change/gain/loss, preserving ignored GT pixels.

    For class c, its ROI is old[c] OR new[c]. Positives are membership changes;
    negatives are globally stable, known pixels within that ROI. Changes of
    other classes and, for directional slices, the opposite direction are
    ignored. Thus the tables describe localisation conditioned on annotated
    class, not class predictions from the model. Classes can overlap.
    """
    arrays = (prediction, reference_bits, source_bits, support, labels)
    if (not all(isinstance(array, np.ndarray) for array in arrays)
            or not all(array.shape == prediction.shape for array in arrays)
            or prediction.ndim not in (1, 2) or prediction.dtype != bool
            or support.dtype != bool or reference_bits.dtype != np.uint16
            or source_bits.dtype != np.uint16 or labels.dtype != np.uint8
            or not np.isin(labels, (0, 1, 255)).all()):
        raise ValueError("Invalid prediction, class-bit, support or label arrays")
    parsed = _categories(categories)
    if (np.any(reference_bits.astype(np.uint32) >= (1 << len(parsed)))
            or np.any(source_bits.astype(np.uint32) >= (1 << len(parsed)))):
        raise ValueError("Bits exceed the declared category channels")
    allowed = support.copy()
    if region_mask is not None:
        if not isinstance(region_mask, np.ndarray) or region_mask.dtype != bool or region_mask.shape != prediction.shape:
            raise ValueError("Region mask must be boolean and match the prediction")
        allowed &= region_mask
    known = allowed & (labels != 255)
    changed = reference_bits != source_bits
    if np.any(known & ((labels == 1) != changed)):
        raise ValueError("Known binary labels disagree with class membership changes")
    rows = []
    for channel, category_id, name in parsed:
        old = (reference_bits & np.uint16(1 << channel)) != 0
        new = (source_bits & np.uint16(1 << channel)) != 0
        if not np.any(known & (old | new)):
            empty_counts = dict.fromkeys(COUNTS, 0)
            rows.extend(dict(class_id=category_id, class_name=name, channel=channel,
                             direction=direction, **empty_counts, **_metrics(empty_counts))
                        for direction in ("change", "gain", "loss"))
            continue
        negative = known & (labels == 0) & (old | new)
        for direction, membership_change in (("change", old ^ new),
                                              ("gain", ~old & new),
                                              ("loss", old & ~new)):
            positive = known & (labels == 1) & membership_change
            rows.append(dict(class_id=category_id, class_name=name, channel=channel,
                             direction=direction, **_evaluate(prediction, positive, negative)))
    return rows


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class _FacadeAccumulator:
    """Retain only compact sums, never millions of per-pair dictionaries."""
    def __init__(self):
        self.keys = {}
        self.facades = {}

    def add(self, row):
        key = tuple(row[field] for field in GROUP)
        self.keys.setdefault(key, None)
        if not row["valid_pixels"]:
            return
        facade_key = key + (row["building_id"],)
        # Counts, evaluated/positive pair counts, metric sums, metric counts.
        state = self.facades.get(facade_key)
        if state is None:
            state = np.zeros(len(COUNTS) + 2 + 2 * len(METRICS), np.float64)
            self.facades[facade_key] = state
        state[:len(COUNTS)] += [row[field] for field in COUNTS]
        state[len(COUNTS)] += 1
        state[len(COUNTS) + 1] += row["positive_pixels"] > 0
        values = _metrics(row)
        offset = len(COUNTS) + 2
        for index, metric in enumerate(METRICS):
            value = values[metric]
            if metric in ("precision", "recall", "f1", "iou") and not row["positive_pixels"]:
                continue
            if value is not None:
                state[offset + index] += value
                state[offset + len(METRICS) + index] += 1

    def finish(self):
        facades = []
        macro_state = {key: np.zeros(len(COUNTS) + 4 + 2 * len(METRICS), np.float64) for key in self.keys}
        offset = len(COUNTS) + 2
        macro_offset = len(COUNTS) + 4
        for key, state in self.facades.items():
            values = {metric: float(state[offset + i] / state[offset + len(METRICS) + i])
                      if state[offset + len(METRICS) + i] else None for i, metric in enumerate(METRICS)}
            facades.append(dict(zip(GROUP + ("building_id",), key), aggregation="frame_then_facade",
                                evaluated_pairs=int(state[len(COUNTS)]), positive_pairs=int(state[len(COUNTS) + 1]),
                                **{field: int(state[i]) for i, field in enumerate(COUNTS)}, **values,
                                **{metric + "_frames": int(state[offset + len(METRICS) + i]) for i, metric in enumerate(METRICS)}))
            total = macro_state[key[:-1]]
            total[:len(COUNTS) + 2] += state[:len(COUNTS) + 2]
            total[len(COUNTS) + 2] += 1
            total[len(COUNTS) + 3] += state[len(COUNTS) + 1] > 0
            for i, metric in enumerate(METRICS):
                if values[metric] is not None:
                    total[macro_offset + i] += values[metric]
                    total[macro_offset + len(METRICS) + i] += 1
        macro = []
        for key, state in macro_state.items():
            macro.append(dict(zip(GROUP, key), aggregation="frame_then_facade_macro",
                              status="evaluated" if state[len(COUNTS) + 2] else "no_evaluable_pixels",
                              facades=int(state[len(COUNTS) + 2]), positive_facades=int(state[len(COUNTS) + 3]),
                              evaluated_pairs=int(state[len(COUNTS)]), positive_pairs=int(state[len(COUNTS) + 1]),
                              **{field: int(state[i]) for i, field in enumerate(COUNTS)},
                              **{metric: float(state[macro_offset + i] / state[macro_offset + len(METRICS) + i])
                                 if state[macro_offset + len(METRICS) + i] else None for i, metric in enumerate(METRICS)},
                              **{metric + "_facades": int(state[macro_offset + len(METRICS) + i]) for i, metric in enumerate(METRICS)}))
        return facades, macro


def aggregate_slices(rows):
    """Frame means within facade, then facade means; counts stay summed counts."""
    accumulator = _FacadeAccumulator()
    for row in rows:
        accumulator.add(row)
    return accumulator.finish()


class SliceCollector:
    """Collect held-out predictions from the calibration engine without inference.

    Optional region_manifest schema:
    {"schema_version": 1, "entries": [
      {"sequence_id": "...", "mode": "previous", "frame": 1,
       "path": "masks/pair.npz", "sha256": "..."}]}

    Each NPZ must contain boolean `sky` and `known` on the aligned pair grid.
    `sky` must represent sky in either observation; `known` explicitly marks
    where that region classification is available. Slices never extend GT or
    call unannotated pixels stable. With no manifest, no sky table is produced.
    """
    def __init__(self, metrics_run, categories, region_manifest=None):
        self.root = Path(metrics_run).resolve()
        self.categories = categories
        _categories(categories)
        self.class_accumulator = _FacadeAccumulator()
        self.region_accumulator = _FacadeAccumulator()
        self.pair_counts = {"class_per_pair.csv": 0, "region_per_pair.csv": 0}
        self.spools = {name: tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="") for name in self.pair_counts}
        self.writers = {name: csv.DictWriter(stream, fieldnames=PAIR_FIELDS) for name, stream in self.spools.items()}
        for writer in self.writers.values():
            writer.writeheader()
        self.cache = {}
        self.hashes = {}
        self.region_cache = {}
        self.region_entries = {}
        self.region_coverage = {}
        provenance_path = self.root / "provenance.json"
        self.expected_hashes = (json.loads(provenance_path.read_text(encoding="utf-8"))
                                .get("output_gt_sha256", {}) if provenance_path.exists() else {})
        self.region_manifest = Path(region_manifest).resolve() if region_manifest else None
        if self.region_manifest:
            manifest = json.loads(self.region_manifest.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != 1 or not isinstance(manifest.get("entries"), list):
                raise ValueError("Invalid explicit region manifest")
            self.hashes[str(self.region_manifest)] = _sha256(self.region_manifest)
            for entry in manifest["entries"]:
                key = (str(entry["sequence_id"]), str(entry["mode"]), int(entry["frame"]))
                if key in self.region_entries:
                    raise ValueError("Duplicate pair in explicit region manifest")
                self.region_entries[key] = entry

    def _gt(self, sample):
        relative = str(sample["gt_artifact"])
        if relative not in self.cache:
            path = (self.root / relative).resolve()
            if self.root not in path.parents:
                raise ValueError("GT artifact is outside its metrics run")
            digest = _sha256(path)
            expected = self.expected_hashes.get(relative)
            if expected and digest != expected:
                raise ValueError("Saved GT artifact SHA-256 mismatch")
            self.hashes[str(path)] = digest
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in ("reference_bits", "source_bits", "support", "labels")}
            shape = arrays["labels"].shape
            if (len(shape) != 2 or any(array.shape != shape for array in arrays.values())
                    or arrays["support"].dtype != bool or arrays["labels"].dtype != np.uint8
                    or arrays["reference_bits"].dtype != np.uint16 or arrays["source_bits"].dtype != np.uint16):
                raise ValueError("Invalid saved GT slice grids")
            self.cache[relative] = arrays
        return self.cache[relative]

    def _regions(self, sample, mode, pixels, shape):
        key = (str(sample["sequence_id"]), str(mode), int(sample["frame"]))
        regions = {"annotated_support": np.ones(len(pixels), dtype=bool)}
        if not self.region_manifest:
            return regions
        entry = self.region_entries.get(key)
        if entry is None:
            self.region_coverage[key] = "missing_explicit_sky_mask"
            return regions
        if key not in self.region_cache:
            path = (self.region_manifest.parent / entry["path"]).resolve()
            if self.region_manifest.parent not in path.parents:
                raise ValueError("Region mask is outside its manifest directory")
            digest = _sha256(path)
            if digest != entry.get("sha256"):
                raise ValueError("Explicit region mask SHA-256 mismatch")
            with np.load(path, allow_pickle=False) as archive:
                sky, known = archive["sky"].copy(), archive["known"].copy()
            if sky.shape != shape or known.shape != shape or sky.dtype != bool or known.dtype != bool or np.any(sky & ~known):
                raise ValueError("Explicit sky/known masks do not match the aligned pair")
            self.hashes[str(path)] = digest
            self.region_cache[key] = (sky, known)
        sky, known = (array.ravel()[pixels] for array in self.region_cache[key])
        self.region_coverage[key] = "available"
        regions.update(with_sky=known, without_sky=known & ~sky, sky_only=known & sky)
        return regions

    def record(self, *, family, mode, predictor, goal, heldout_building, sample, prediction):
        arrays = self._gt(sample)
        shape = arrays["labels"].shape
        if tuple(sample["shape"]) != shape:
            raise ValueError("Calibration sample shape differs from GT")
        pixels = np.asarray(sample["pixels"] if "pixels" in sample else np.arange(np.prod(shape)), dtype=np.int64)
        if (pixels.ndim != 1 or np.any(pixels < 0) or np.any(pixels >= np.prod(shape))
                or len(np.unique(pixels)) != len(pixels)):
            raise ValueError("Invalid or duplicate calibration pixel indices")
        prediction = np.asarray(prediction)
        labels = np.asarray(sample["labels"])
        if prediction.dtype != bool or prediction.shape != pixels.shape or labels.dtype != np.uint8 or labels.shape != pixels.shape:
            raise ValueError("Flattened predictions and labels must match calibration pixels")
        saved_labels = arrays["labels"].ravel()[pixels]
        if not np.array_equal(labels, saved_labels):
            raise ValueError("Calibration labels differ from the saved GT artifact")
        reference, source, support = (arrays[name].ravel()[pixels] for name in ("reference_bits", "source_bits", "support"))
        goal_value = json.dumps(goal, sort_keys=True, ensure_ascii=False) if isinstance(goal, (dict, list)) else str(goal)
        metadata = dict(family=str(family), mode=str(mode), predictor=str(predictor), goal=goal_value,
                        heldout_building=str(heldout_building), building_id=str(sample["building_id"]),
                        sequence_id=str(sample["sequence_id"]), frame=int(sample["frame"]))
        for region, region_mask in self._regions(sample, mode, pixels, shape).items():
            for result in evaluate_class_slices(prediction, reference, source, support, labels, self.categories, region_mask):
                row = {**metadata, "region": region, **result}
                self.class_accumulator.add(row)
                if row["valid_pixels"]:
                    self.writers["class_per_pair.csv"].writerow(row)
                    self.pair_counts["class_per_pair.csv"] += 1
            eligible = support & region_mask
            row = {**metadata, "region": region, "class_id": "all",
                   "class_name": "all_annotation_classes", "direction": "change",
                   **_evaluate(prediction, eligible & (labels == 1), eligible & (labels == 0))}
            self.region_accumulator.add(row)
            if row["valid_pixels"]:
                self.writers["region_per_pair.csv"].writerow(row)
                self.pair_counts["region_per_pair.csv"] += 1

    def finalize(self, out_path):
        out = Path(out_path)
        out.mkdir(parents=True, exist_ok=True)
        class_facades, class_macro = self.class_accumulator.finish()
        region_facades, region_macro = self.region_accumulator.finish()
        outputs = {"class_per_facade.csv": class_facades, "class_macro.csv": class_macro,
                   "region_per_facade.csv": region_facades, "region_macro.csv": region_macro}
        for name, stream in self.spools.items():
            stream.flush()
            stream.seek(0)
            with (out / name).open("w", encoding="utf-8", newline="") as destination:
                shutil.copyfileobj(stream, destination, length=1024 * 1024)
            stream.close()
        for name, rows in outputs.items():
            _write_csv(out / name, rows)
        protocol = dict(schema_version=1, categories=self.categories,
                        metric_scope="class-conditioned binary change localisation, not class prediction",
                        class_roi="union of class membership in reference and source",
                        class_positives="membership XOR; gain/loss isolate its direction",
                        class_negatives="globally stable known pixels inside class ROI",
                        ignored="unknown GT, true changes of other classes, opposite direction for gain/loss",
                        class_overlap="classes may overlap; do not sum class counts",
                        aggregation="frame_then_facade_macro",
                        metric_frame_policy="precision/recall/f1/iou only positive frames; fpr all frames with known negatives; marked_fraction all valid frames",
                        counts_policy="summed pixel observations; macro metrics cannot be recomputed from summed counts",
                        storage_policy="zero-valid per-pair rows omitted; all class/direction keys retained in macro with null metrics if absent; per-pair CSV streamed",
                        metric_units="fractions [0,1]", no_model_inference=True,
                        sky_status="explicit_masks_provided" if self.region_manifest else "unavailable_no_explicit_sky_masks",
                        sky_scope="region slices retain existing known GT; unknown pixels are never promoted to negatives",
                        region_coverage=[dict(sequence_id=key[0], mode=key[1], frame=key[2], status=value)
                                         for key, value in self.region_coverage.items()],
                        input_sha256=self.hashes,
                        output_rows={**self.pair_counts, **{name: len(rows) for name, rows in outputs.items()}},
                        output_bytes={name: (out / name).stat().st_size for name in (*self.pair_counts, *outputs)})
        (out / "slice_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return protocol
