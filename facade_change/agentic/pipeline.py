"""GT-free pair integration on the existing native-RGB geometry and scorers.

The canonical region grid is pinned to one reference image and a fixed box.
This module does not prepare benchmark targets, train, or select thresholds.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from ..geometry import Canvas, expanded_canvas, warp_pair
from ..history_features import Fingerprints, _history_document, _inside, digest
from ..io import finish_record, load_rgb, read_json, run_record, sha256, write_json
from ..methods.remote import RemotePool


PAIR_FIELDS = {"case_id", "building_id", "view_id", "split", "reference", "source",
               "region_id", "bbox_xyxy", "alignment_run", "manifest", "reference_id",
               "source_id", "anchor_sha256", "prepared_provenance"}
IMAGE_FIELDS = {"path", "date", "date_status", "available_at", "image_id", "sha256",
                "crop_path", "support_path", "crop_sha256", "support_sha256",
                "source_to_anchor", "native_shape", "width", "height"}
ALLOWED_METHODS = {"rgb_diff", "ssim", "lpips", "dinov2", "semantic_dino", "anychange",
                   "zstd_abs", "zstd_mod256", "lzma_abs", "lzma_mod256", "msdzip_abs",
                   "msdzip_mod256", "jpegls_abs", "jpegls_mod256", "h264_rgb",
                   "arib_bps_abs", "arib_bps_mod256", "bcm_net_rgb"}


def _date(value):
    from .store import temporal_value
    return temporal_value(value)


def _absolute(value, parent=None):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(parent or Path.cwd()) / path).resolve()


def _checked_image(row, parent=None):
    if not isinstance(row, dict) or set(row) - IMAGE_FIELDS:
        raise ValueError("Observation accepts only RGB/date/geometry fields; GT is not an input")
    item = dict(row)
    path = _absolute(item["path"], parent)
    actual = sha256(path)
    if item.get("sha256") not in (None, actual):
        raise ValueError(f"Source image hash changed: {path}")
    rgb, _ = load_rgb(path)
    if item.get("native_shape") not in (None, list(rgb.shape[:2])):
        raise ValueError("Image native shape differs from the saved manifest")
    item.update(path=str(path), sha256=actual, native_shape=list(rgb.shape[:2]))
    item.setdefault("image_id", actual[:24])
    item.setdefault("date", None)
    item.setdefault("date_status", "unknown" if item["date"] is None else "confirmed")
    item.setdefault("available_at", item["date"])
    _date(item["date"])
    _date(item["available_at"])
    if item["date_status"] not in {"confirmed", "inferred", "unknown"}:
        raise ValueError("Invalid date_status")
    for key in ("crop_path", "support_path"):
        if key in item:
            item[key] = str(_absolute(item[key], parent))
    return item


def resolve_pair_spec(spec):
    """Project a reviewed manifest onto an explicit inference allowlist.

    A raw pair is strict: unknown fields, including labels, fail. A supplied
    dataset manifest may contain annotations; only the named image metadata
    fields below are read into this new object.
    """
    from .store import reject_forbidden
    reject_forbidden(spec)
    if not isinstance(spec, dict) or set(spec) - PAIR_FIELDS:
        raise ValueError("Unexpected pair fields; benchmark archives cannot be agent input")
    result = dict(spec)
    if "prepared_provenance" in result:
        provenance = result["prepared_provenance"]
        if (not isinstance(provenance, dict) or set(provenance) -
                {"manifest", "manifest_sha256", "track_id", "projection"}):
            raise ValueError("Prepared provenance accepts only named source identities, not benchmark contents")
        if provenance.get("manifest") and provenance.get("manifest_sha256") != sha256(_absolute(provenance["manifest"])):
            raise ValueError("Prepared source manifest hash changed")
    parent = None
    if "manifest" in result:
        if "reference" in result or "source" in result:
            raise ValueError("Choose manifest IDs or raw images, not both")
        path = _absolute(result.pop("manifest"))
        document = read_json(path)
        rows = document.get("images")
        if not isinstance(rows, list):
            raise ValueError("Prepared image manifest must contain images")
        lookup = {str(row["image_id"]): row for row in rows}
        if len(lookup) != len(rows):
            raise ValueError("Duplicate manifest image IDs")
        chosen = []
        for role in ("reference", "source"):
            identifier = str(result.pop(role + "_id"))
            if identifier not in lookup:
                raise ValueError(f"Unknown manifest image ID: {identifier}")
            row = lookup[identifier]
            if row.get("image_status") != "ready" or row.get("metadata_status") != "reviewed":
                raise ValueError("Pair requires ready images and reviewed spatial identities")
            chosen.append(row)
            result[role] = dict(path=row["image_path"], image_id=identifier,
                sha256=row["sha256"], date=row.get("year"), date_status="confirmed",
                available_at=row.get("year"), native_shape=[row["height"], row["width"]])
        for key in ("building_id", "view_id"):
            values = [r.get(key) for r in chosen]
            if not values[0] or values[0] != values[1] or result.get(key, values[0]) != values[0]:
                raise ValueError(f"Manifest pair has inconsistent {key}")
            result[key] = values[0]
        parts = {r["split"] for r in chosen if r.get("split")}
        if len(parts) > 1 or (parts and result.get("split", next(iter(parts))) not in parts):
            raise ValueError("Manifest pair split mismatch")
        result.setdefault("split", next(iter(parts)) if parts else "unassigned")
        parent = path.parent
        result["prepared_provenance"] = {"manifest": str(path), "manifest_sha256": sha256(path),
                                          "projection": "image metadata allowlist; annotations excluded"}
    for key in ("building_id", "view_id"):
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise ValueError(f"Explicit reviewed {key} required; it will not be inferred from a filename")
    result.setdefault("split", "unassigned")
    if result["split"] not in {"train", "val", "test", "unassigned"}:
        raise ValueError("Invalid split")
    for role in ("reference", "source"):
        result[role] = _checked_image(result[role], parent)
    if result["reference"]["sha256"] == result["source"]["sha256"]:
        raise ValueError("A temporal pair requires two different image contents")
    dates = [_date(result[r]["date"]) for r in ("reference", "source")]
    if all(d is not None for d in dates) and dates[0] >= dates[1]:
        raise ValueError("Reference date must precede source date")
    if "case_id" in result and (not isinstance(result["case_id"], str)
            or not result["case_id"] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in result["case_id"])):
        raise ValueError("case_id must be a safe identifier")
    return result


sanitize_pair_spec = resolve_pair_spec


def pairs_from_history(histories, track_id, min_dates=3, available_at="observation_date"):
    """Reuse canonical RGB histories with the same availability policy as import.

    ``observation_date`` explicitly assumes retrospective availability; an ISO
    date pins archive availability, and None leaves availability unknown.
    """
    if available_at != "observation_date":
        _date(available_at)
    root = _absolute(histories)
    if root.is_file():
        if root.name != "manifest.json":
            raise ValueError("Use a canonical history manifest.json or its directory")
        root = root.parent
    fp = Fingerprints()
    document, manifest_sha = _history_document(root, fp)
    tracks = [t for t in document["tracks"] if t["track_id"] == track_id]
    if len(tracks) != 1:
        raise ValueError("Select one existing track_id explicitly; no GT-based track selection")
    track = tracks[0]
    observations = track["observations"]
    years = [row["year"] for row in observations]
    if len(set(years)) < min_dates or len(set(years)) != len(years):
        raise ValueError(f"Need >= {min_dates} distinct dates and exactly one reviewed observation per date")
    x, y, w, h = track["bbox_xywh"]
    def frame(row):
        rgb, support = _inside(root, row["rgb"]), _inside(root, row["support"])
        fp.add(rgb, row["rgb_sha256"])
        fp.add(support, row["support_sha256"])
        original = _absolute(row["original_image_path"])
        fp.add(original, row["image_sha256"])
        return dict(path=str(original), sha256=row["image_sha256"], image_id=row["image_id"],
                    date=row["year"], date_status="confirmed",
                    available_at=str(row["year"]) if available_at == "observation_date" else available_at,
                    crop_path=str(rgb), crop_sha256=row["rgb_sha256"], support_path=str(support),
                    support_sha256=row["support_sha256"], source_to_anchor=row["alignment"]["matrix"])
    result = []
    for previous, current in zip(observations, observations[1:]):
        result.append(dict(building_id=track["building_id"], view_id=track["view_id"], split=track["split"],
            region_id=track_id, anchor_sha256=track["anchor_sha256"], bbox_xyxy=[x, y, x+w, y+h],
            reference=frame(previous), source=frame(current), prepared_provenance={
                "manifest": str(root / "manifest.json"), "manifest_sha256": manifest_sha,
                "track_id": track_id, "projection": "canonical RGB/support only; no targets"}))
    fp.verify()
    return result


def _matrix(value):
    result = np.asarray(value, np.float64)
    if result.shape != (3, 3) or not np.isfinite(result).all() or np.linalg.matrix_rank(result) != 3:
        raise ValueError("A nonsingular finite native homography is required")
    return result


def _support(path, shape):
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.shape != shape or array.dtype != np.uint8 or not np.isin(array, (0, 255)).all():
        raise ValueError("Geometric support must be a same-grid binary PNG")
    return array == 255


def _validate_prepared(pair, bbox, anchor):
    provenance = pair.get("prepared_provenance", {})
    if not provenance.get("manifest") or not provenance.get("track_id"):
        raise ValueError("Prepared crops require a completed canonical history track")
    path = _absolute(provenance["manifest"])
    fp = Fingerprints()
    document, manifest_hash = _history_document(path.parent, fp)
    if manifest_hash != provenance.get("manifest_sha256"):
        raise ValueError("Prepared history manifest changed")
    tracks = [t for t in document["tracks"] if t["track_id"] == provenance["track_id"]]
    if len(tracks) != 1:
        raise ValueError("Prepared track is missing or duplicated")
    track = tracks[0]
    x, y, w, h = track["bbox_xywh"]
    if (track["anchor_sha256"] != anchor or [x, y, x+w, y+h] != bbox
            or any(track[k] != pair[k] for k in ("building_id", "view_id", "split"))):
        raise ValueError("Prepared crop identity/coordinate frame differs")
    for role in ("reference", "source"):
        selected = pair[role]
        rows = [r for r in track["observations"] if r["image_sha256"] == selected["sha256"]]
        if len(rows) != 1 or _date(rows[0]["year"]) != _date(selected["date"]):
            raise ValueError("Prepared observation identity/date differs")
        row = rows[0]
        for field, name in (("crop", "rgb"), ("support", "support")):
            expected = _inside(path.parent, row[name])
            if (_absolute(selected[field + "_path"]) != expected
                    or selected[field + "_sha256"] != row[name + "_sha256"]):
                raise ValueError("Prepared RGB/support reference differs from its frozen manifest")
            fp.add(expected, row[name + "_sha256"])
        if not np.array_equal(_matrix(selected["source_to_anchor"]), _matrix(row["alignment"]["matrix"])):
            raise ValueError("Prepared source-to-anchor homography differs")
    fp.verify()


def _align_local(request, out):
    """Owned short-lived process: existing matchers/MAGSAC, no model retained."""
    from ..alignment import SIFTMatcher, LoFTRMatcher, align
    from ..pipeline import alignment_quality, save_diagnostics
    config = dict(request["options"])
    allowed = {"method", "checkpoint", "device", "max_side", "ransac_threshold", "confidence", "seed",
               "max_canvas_pixels", "max_canvas_side", "trust_checkpoint", "min_inliers",
               "min_inlier_ratio", "min_hull_fraction", "min_overlap_fraction"}
    if set(config) - allowed:
        raise ValueError("Unexpected alignment options; downloads are not enabled")
    record = run_record("agentic_pair_alignment", request)
    write_json(out / "run.json", record)
    started = time.monotonic()
    try:
        rows = [request[role] for role in ("reference", "source")]
        arrays = []
        for row in rows:
            if sha256(row["path"]) != row["sha256"]:
                raise ValueError("Alignment source changed")
            arrays.extend(load_rgb(row["path"]))
        method = config.get("method", "sift")
        if method not in {"sift", "loftr", "cascade"}:
            raise ValueError("Unsupported existing alignment method")
        attempts, chosen = [], None
        for name in (("sift", "loftr") if method == "cascade" else (method,)):
            try:
                matcher = SIFTMatcher() if name == "sift" else LoFTRMatcher(config.get("checkpoint"),
                    config.get("device", "cpu"), config.get("confidence", .4),
                    trust_checkpoint=config.get("trust_checkpoint", False), download_weights=False)
                value = align(*arrays, matcher, config.get("max_side", 1024),
                              config.get("ransac_threshold", 3.), config.get("seed", 42))
                canvas = expanded_canvas(arrays[0].shape, arrays[2].shape, value.source_to_reference,
                    config.get("max_canvas_pixels", 50_000_000), config.get("max_canvas_side", 16000))
                warped = warp_pair(*arrays, canvas)
                quality = alignment_quality(value, warped, config.get("min_inliers", 30),
                    config.get("min_inlier_ratio", .2), config.get("min_hull_fraction", .1),
                    config.get("min_overlap_fraction", .2))
                attempts.append({"method": name, "status": "completed", "quality": quality})
                chosen = value, canvas, warped, quality, name
                if quality["passed"] or method != "cascade":
                    break
            except Exception as exc:
                attempts.append({"method": name, "status": "error", "error": str(exc)})
                if method != "cascade":
                    raise
        if chosen is None:
            raise ValueError("No matcher produced comparable geometry")
        value, canvas, warped, quality, name = chosen
        record.update(observations={"reference": rows[0], "source": rows[1]}, attempts=attempts,
                      quality_gate=quality, selected_method=name, matcher=value.matches.metadata)
        value.diagnostics["quality_gate"] = quality
        write_json(out / "geometry.json", canvas.as_dict())
        if warped["overlap"].any():
            save_diagnostics(out, warped, value, canvas)
        record["elapsed_seconds"] = time.monotonic() - started
        finish_record(out, record, "completed_needs_review" if quality["passed"] else "completed_rejected")
    except BaseException as exc:
        finish_record(out, record, "failed", str(exc))
        raise


def _alignment(pair, config, out):
    if pair.get("alignment_run"):
        root = _absolute(pair["alignment_run"])
    else:
        root = out / "alignment"
        root.mkdir()
        options = dict(config.get("alignment", {}))
        python = options.pop("worker_python", sys.executable)
        timeout = float(options.pop("timeout_seconds", 600))
        if not Path(python).is_absolute() or not Path(python).is_file() or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Alignment worker requires an existing absolute interpreter and positive timeout")
        request = {"reference": pair["reference"], "source": pair["source"], "options": options}
        write_json(root / "request.json", request)
        env = os.environ.copy()
        repo = Path(__file__).resolve().parents[2]
        env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        with (root / "worker.log").open("w", encoding="utf-8") as stream:
            process = subprocess.run([str(python), "-s", "-B", "-m", "facade_change.agentic.pipeline",
                "align-worker", str(root / "request.json")], cwd=repo, env=env, stdout=stream,
                stderr=subprocess.STDOUT, timeout=timeout, check=False)
        if process.returncode:
            raise ValueError(f"Alignment failed; inspect {root / 'worker.log'}")
    record = read_json(root / "run.json")
    if record.get("kind") not in {"pair_alignment", "agentic_pair_alignment"}:
        raise ValueError("Saved alignment has an incompatible run kind")
    if not record.get("status", "").startswith("completed"):
        raise ValueError("Saved alignment did not complete")
    for role in ("reference", "source"):
        row = record.get("observations", {}).get(role, {})
        if row.get("sha256") != pair[role]["sha256"]:
            raise ValueError("Saved alignment belongs to different ordered images")
    path = root / "geometry.json"
    if record.get("artifact_sha256", {}).get("geometry.json") != sha256(path):
        raise ValueError("Saved alignment geometry hash differs")
    geometry = read_json(path)
    return _matrix(geometry["source_to_reference"]), record.get("quality_gate", {"passed": False}), {
        "alignment_run": str(root), "alignment_run_sha256": sha256(root / "run.json"),
        "geometry_sha256": sha256(path), "selected_method": record.get("selected_method")}


def _artifact(path, role, media_type="application/octet-stream"):
    return dict(path=str(Path(path).resolve()), sha256=sha256(path), role=role, media_type=media_type)


def _model_signature(method, options):
    """Pin configured weight/source files before looking up a cached result."""
    paths = {}
    resolved = dict(options)
    if method == "semantic_dino" and options.get("training_run"):
        training = _absolute(options["training_run"])
        config = read_json(training / "config.json")
        for key, stored in (("source_root", "source_root"), ("backbone_checkpoint", "checkpoint")):
            if key not in resolved and config.get(stored):
                resolved[key] = config[stored]
    for key, value in resolved.items():
        if not isinstance(value, str) or key not in {"source_root", "checkpoint_path", "dino_root",
                "dino_checkpoint", "backbone_checkpoint", "calibration_checkpoint", "training_run",
                "head_checkpoint", "sam_checkpoint", "model_config", "current_manifest"}:
            continue
        path = _absolute(value)
        if not path.exists():
            raise ValueError(f"Missing configured {key}: {path}")
        selected = [path] if path.is_file() else (sorted(path.rglob("*.py")) if key != "training_run"
            else [path / name for name in ("run.json", "config.json", "summary.json", "best.pth")])
        for item in selected:
            paths[str(item)] = sha256(item)
    package = Path(__file__).resolve().parents[1]
    code = {str(p.relative_to(package)): sha256(p) for p in sorted(package.rglob("*.py"))
            if "agentic" not in p.parts}
    executable = _absolute(options.get("worker_python", sys.executable))
    if not executable.is_file():
        raise ValueError(f"Missing method worker interpreter: {executable}")
    command = ("import importlib.metadata as m,json; names="
               "['numpy','Pillow','torch','torchvision','opencv-python-headless','opencv-python','kornia','lpips','zstandard']; "
               "installed={d.metadata['Name'].lower():d.version for d in m.distributions()}; "
               "print(json.dumps({n:installed.get(n.lower()) for n in names},sort_keys=True))")
    check = subprocess.run([str(executable), "-s", "-B", "-c", command], capture_output=True,
                           text=True, check=True, timeout=30)
    versions = json.loads(check.stdout)
    return {"method": method, "options": options, "configured_files": paths,
            "code": code, "adapter": sha256(__file__), "worker_python": str(executable),
            "worker_python_sha256": sha256(executable), "packages": versions}


def _method_maps(config, pair, out, reference, source, support, geometry, pool_factory):
    from ..crop_benchmark import _validate_arrays
    store_module = importlib.import_module("facade_change.2026-10-14_temporal_experiment")
    methods = config.get("methods", ["ssim", "lzma_abs"])
    if not isinstance(methods, list) or len(set(methods)) != len(methods) or not set(methods) <= ALLOWED_METHODS:
        raise ValueError("Select distinct existing methods; GeoSCD/RSCD are outside this integration")
    if methods and set(methods) <= {"rgb_diff"}:
        raise ValueError("RGB difference cannot be the only change detector")
    root = _absolute(config.get("cache_root", out / "method_cache"))
    cache = store_module.RunStore(root, {"kind": "agentic_method_cache", "schema_version": 1}, resume=root.exists())
    pool = pool_factory(log_dir=out / "worker_logs", timeout=config.get("method_timeout_seconds", 3600))
    results = []
    arrays_by_method = {}
    try:
        for method in methods:
            item = dict(method=method, status="pending", cache_hit=False, model_calls=0, model_load_attempts=0)
            started = time.monotonic()
            try:
                options = dict(config.get("method_options", {}).get(method, {}))
                if method == "semantic_dino":
                    options["export_probabilities"] = True
                signature = _model_signature(method, options)
                image_hashes = [digest({"shape": list(a.shape), "dtype": str(a.dtype),
                    "sha256": __import__("hashlib").sha256(a.tobytes()).hexdigest()})
                    for a in (reference, source, support)]
                coordinate_contract = {k: geometry[k] for k in ("anchor_sha256", "bbox_xyxy", "shape",
                    "reference_to_crop", "source_to_crop")}
                key = digest([signature, image_hashes, coordinate_contract])
                cached = cache.read_unit("method-" + key)
                if cached:
                    record, arrays = cached
                    metadata = record["metadata"]
                    item["cache_hit"] = True
                else:
                    item.update(model_calls=1, model_load_attempts=1)
                    value = pool.score(method, options, reference, source, support)
                    metadata = value["metadata"]
                    arrays = {k: value[k] for k in ("scores", "raw_scores", "native_prediction") if k in value}
                    arrays.update(value.get("auxiliary_arrays", {}))
                    _validate_arrays(arrays, support, metadata.get("output_kind") == "native_mask", metadata)
                    record = cache.save_unit(dict(unit_id="method-" + key, kind="agentic_method",
                        signature=signature, metadata=metadata, input_sha256=image_hashes,
                        scoring_seconds=time.monotonic() - started), arrays)
                _validate_arrays(arrays, support, metadata.get("output_kind") == "native_mask", metadata)
                folder = out / "maps" / method
                folder.mkdir(parents=True)
                paths = {}
                for name, array in arrays.items():
                    if name == "native_prediction":
                        array = array & support
                        path = folder / (name + ".png")
                        Image.fromarray(array.astype(np.uint8) * 255).save(path)
                    else:
                        path = folder / (name + ".npy")
                        np.save(path, array, allow_pickle=False)
                    paths[name + "_path"] = str(path)
                values = arrays["scores"][support]
                overlay = source.copy()
                weight = np.nan_to_num(arrays["scores"], nan=0.)[..., None] * .65
                overlay = np.rint(overlay * (1-weight) + np.array([255., 30., 20.]) * weight).astype(np.uint8)
                overlay[~support] = 40
                overlay_path = folder / "score_preview.png"
                Image.fromarray(overlay).save(overlay_path)
                paths["score_preview_path"] = str(overlay_path)
                threshold = .5 if metadata.get("output_kind") == "native_mask" else config.get("thresholds", {}).get(method)
                if threshold is not None:
                    if isinstance(threshold, bool) or not np.isfinite(threshold):
                        raise ValueError("Method threshold must be a finite explicit number")
                    prediction = (arrays["scores"] > threshold) & support
                    path = folder / "prediction.png"
                    Image.fromarray(prediction.astype(np.uint8) * 255).save(path)
                    paths["prediction_path"] = str(path)
                item.update(status="completed", **paths, metadata=metadata, threshold=threshold,
                    threshold_origin="author native binary decision" if metadata.get("output_kind") == "native_mask"
                    else config.get("threshold_origin", "explicit diagnostic setting; not calibrated"),
                    score_type="binary_native_mask" if metadata.get("output_kind") == "native_mask" else "method_score",
                    scale="[0,1]; not a calibrated probability", direction="ordered reference -> source; higher is more change",
                    summary={"mean": float(values.mean()), "p95": float(np.quantile(values, .95)),
                             "support_pixels": int(support.sum()), "positive_pixels":
                                 int((values > threshold).sum()) if threshold is not None else None},
                    cache_key=key, source_unit_id=record["unit_id"], source_artifact_sha256=record["artifact_sha256"],
                    model_loads=0 if item["cache_hit"] else 1,
                    original_scoring_seconds=record["scoring_seconds"])
                arrays_by_method[method] = arrays
            except Exception as exc:
                item.update(status="error", error=f"{type(exc).__name__}: {exc}")
            finally:
                pool.release()
            item["elapsed_seconds"] = time.monotonic() - started
            results.append(item)
        cache.finish("completed", methods=[{k: r[k] for k in ("method", "status")} for r in results])
    finally:
        pool.close()
        cache.close()
    return results, arrays_by_method


def _frozen_fusion(spec, arrays, maps, support, out):
    """Apply the repository's saved TRAIN fit and VAL threshold, never refit."""
    fusion = importlib.import_module("data_analysis.2026-11-06_history_fusion")
    root = _absolute(spec["run"])
    record = read_json(root / "run.json")
    if record.get("kind") != "canonical_history_fusion" or record.get("status") != "completed":
        raise ValueError("Fusion needs a completed canonical_history_fusion run")
    prefix = fusion.PREFIX
    summary_path = root / (prefix + "summary.json")
    if record.get("summary_sha256") != sha256(summary_path):
        raise ValueError("Fusion summary SHA differs")
    summary = read_json(summary_path)
    contract_path = root / "experiment_contract.json"
    contract = read_json(contract_path)
    if contract.get("fit_split") != "train" or contract.get("threshold_split") != "val":
        raise ValueError("Fusion must preserve TRAIN fit / VAL threshold provenance")
    for filename, expected in contract.get("code", {}).items():
        if filename == Path(fusion.__file__).name and sha256(fusion.__file__) != expected:
            raise ValueError("Frozen fusion implementation differs from the selected run")
    identifier = spec.get("variant", "stack--previous")
    variants = [v for v in contract["variants"] if v["id"] == identifier]
    if len(variants) != 1 or variants[0]["mode"] != "previous":
        raise ValueError("Pair integration needs a previous-mode variant; history/earliest inputs cannot be fabricated")
    variant = variants[0]
    columns = [(mode, method) for mode in fusion.MODES for method in fusion.METHODS]
    required = [columns[i][1] for i in variant["columns"]]
    if any(columns[i][0] != "previous" for i in variant["columns"]) or any(m not in arrays for m in required):
        raise ValueError("Missing frozen fusion component(s) or temporal modes")
    source_root = _absolute(summary["source"])
    source_hashes = summary["source_sha256"]
    threshold_source = source_root / "thresholds.json"
    if source_hashes.get(str(threshold_source)) != sha256(threshold_source):
        raise ValueError("Frozen component thresholds changed")
    source_thresholds = read_json(threshold_source)
    # Existing trained recipe must use the same feature/method contracts.
    actual = {row["method"]: row["metadata"] for row in maps if row["status"] == "completed"}
    from ..crop_benchmark import _identity
    def identity(value):
        stable = _identity(value)
        # Deployment/evaluation metadata is recorded separately; all model,
        # external-source, ontology and preprocessing identities remain exact.
        for key in ("worker_environment", "worker_python", "checkpoint_origin_audit", "evaluation_split_audit"):
            stable.pop(key, None)
        return stable
    method_contracts = {}
    for method in required:
        path = source_root / "method_contracts" / (method + ".json")
        expected = read_json(path)
        if identity(expected) != identity(actual[method]):
            raise ValueError(f"Frozen fusion component identity differs: {method}")
        old_env, new_env = expected.get("worker_environment", {}), actual[method].get("worker_environment", {})
        old_sources, new_sources = old_env.get("source_sha256", {}), new_env.get("source_sha256", {})
        for name, old_hash in old_sources.items():
            if name.startswith(("methods/", "segmentation/")) or name in {"2026-10-04_msdzip_h0.py", "scorers.py"}:
                if new_sources.get(name) != old_hash:
                    raise ValueError(f"Frozen fusion detector source changed: {method}/{name}")
        for name in ("numpy", "Pillow", "torch", "torchvision", "kornia", "lpips"):
            if name in old_env.get("packages", {}) and old_env["packages"][name] != new_env.get("packages", {}).get(name):
                raise ValueError(f"Frozen fusion detector package differs: {method}/{name}")
        method_contracts[str(path)] = sha256(path)
    normalizer_path = root / (prefix + "normalizer.json")
    normalizer = {k: np.asarray(v, np.float64) for k, v in read_json(normalizer_path).items()}
    if set(normalizer) != {"low", "high"} or any(v.shape != (len(columns),) for v in normalizer.values()):
        raise ValueError("Frozen normalizer column order/length mismatch")
    models, model_paths = {}, {}
    if variant["kind"].startswith("stack"):
        saved = fusion.benchmark.read_unit(root, prefix + "model-" + identifier)
        if saved is None or saved[0]["variant"] != variant or saved[0].get("split") != "train":
            raise ValueError("Missing or incompatible TRAIN fusion model")
        models[identifier] = saved[1]
        model_paths[str(root / saved[0]["artifact_path"])] = saved[0]["artifact_sha256"]
    calibration_kind = spec.get("threshold", "f1")
    if calibration_kind not in {"f1", "fpr05"}:
        raise ValueError("Choose an existing f1 or fpr05 threshold")
    threshold_path = root / (prefix + ("thresholds.json" if calibration_kind == "f1" else "fpr05_thresholds.json"))
    choice = read_json(threshold_path)[identifier]
    threshold = choice["threshold"]
    if threshold is None or not np.isfinite(threshold):
        raise ValueError("The selected recipe has no usable frozen threshold")
    count = int(support.sum())
    raw = np.zeros((count, len(columns)), np.float32)
    for i in variant["columns"]:
        raw[:, i] = arrays[columns[i][1]]["scores"][support]
    context, groups = np.empty((count, 0), np.float32), []
    if variant["kind"] in {"stack_classes", "stack_groups"}:
        semantic = arrays.get("semantic_dino", {})
        if not {"probabilities_reference", "probabilities_source"} <= semantic.keys():
            raise ValueError("Frozen semantic fusion requires actual endpoint probabilities")
        metadata = actual["semantic_dino"]
        expected = read_json(source_root / "method_contracts/semantic_dino.json")
        fusion.verify_semantic_identity(expected, metadata)
        probabilities = np.maximum(semantic["probabilities_reference"], semantic["probabilities_source"])
        context = probabilities[:, support].T
        if variant.get("context_offset", 0) != 0 or variant.get("context_size") != context.shape[1]:
            raise ValueError("Semantic context channels differ from the frozen recipe")
        if variant["kind"] == "stack_groups":
            source_run_path = source_root / "run.json"
            if source_hashes.get(str(source_run_path)) != sha256(source_run_path):
                raise ValueError("Frozen source benchmark run changed")
            prepared = _absolute(read_json(source_run_path)["config"]["prepared"])
            index_path = prepared / "index.json"
            if sha256(index_path) != contract.get("prepared_sha256"):
                raise ValueError("Frozen category/group metadata changed")
            # Only taxonomy metadata is selected; targets/annotation arrays are
            # never opened or exposed by the online integration.
            index = read_json(index_path)
            category_index = {category["name"]: i for i, category in enumerate(index["categories"])}
            groups = [[category_index[name] for name in group["categories"]] for group in index["groups"]]
            if spec.get("groups", groups) != groups:
                raise ValueError("Configured semantic groups differ from the frozen recipe")
            method_contracts[str(index_path)] = sha256(index_path)
    thresholds = np.array([source_thresholds[m][mode]["threshold"] for mode, m in columns])
    values = fusion.predict(raw, context, variant, normalizer, groups, models, thresholds)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Frozen fusion produced invalid values")
    scores = np.full(support.shape, np.nan, np.float32)
    scores[support] = values
    mask = (scores > threshold) & support
    scores_path, mask_path = out / "fusion_scores.npy", out / "fusion_prediction.png"
    np.save(scores_path, scores, allow_pickle=False)
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
    paths = [root / "run.json", summary_path, contract_path, normalizer_path, threshold_path, threshold_source]
    return dict(status="completed", decision="changed" if mask.any() else "no_change",
        mask_path=str(mask_path), scores_path=str(scores_path), threshold=float(threshold),
        scope="region contains at least one positive supported pixel; not an independently calibrated region classifier",
        variant=variant, threshold_selection=choice, positive_pixels=int(mask.sum()), input_methods=required,
        threshold_origin={"selection": calibration_kind, "selection_split": "val",
                          "path": str(threshold_path), "sha256": sha256(threshold_path)},
        provenance={"run": str(root), "fit_split": "train", "threshold_split": "val",
            "model_artifacts": model_paths,
            "artifacts": {**{str(p): sha256(p) for p in paths}, **method_contracts, **model_paths},
            "application_scope": "one non-overlapping supplied region; raw-score spatial merge is required for whole-facade evaluation"})


def _fusion_evidence(baseline, case, observations, support, source_rgb, store, out):
    """Expose the saved fusion as its own prediction, retaining component evidence."""
    inputs = [item for item in case["maps"] if item["method"] in baseline["input_methods"]]
    source_ids = [item["evidence_id"] for item in inputs]
    source = {**baseline["provenance"], "input_methods": [
        {"method": item["method"], "evidence_id": item["evidence_id"],
         "scores_path": item["scores_path"], "scores_sha256": sha256(item["scores_path"]),
         "source_artifact_sha256": item["source_artifact_sha256"]} for item in inputs]}
    scores = np.load(baseline["scores_path"], allow_pickle=False)
    weight = np.nan_to_num(scores, nan=0.)[..., None] * .65
    preview = np.rint(source_rgb * (1-weight) + np.array([255., 30., 20.]) * weight).astype(np.uint8)
    preview[~support] = 40
    preview_path = out / "fusion_score_preview.png"
    Image.fromarray(preview).save(preview_path)
    record = store.add_evidence(dict(kind="prediction", region_id=case["region_id"],
        observation_ids=[row["observation_id"] for row in observations],
        available_at=observations[1]["available_at"], source_evidence_ids=source_ids,
        coordinate_system="region_crop_pixels", model="frozen_canonical_history_fusion",
        artifacts=[_artifact(baseline["scores_path"], "scores_path", "application/octet-stream"),
                   _artifact(baseline["mask_path"], "prediction_path", "image/png"),
                   _artifact(case["support_path"], "support_path", "image/png"),
                   _artifact(preview_path, "score_preview", "image/png")],
        summary={"method": "fusion", "status": "completed", "variant": baseline["variant"],
                 "threshold": baseline["threshold"], "threshold_selection": baseline["threshold_selection"],
                 "threshold_origin": baseline["threshold_origin"], "positive_pixels": baseline["positive_pixels"],
                 "support_pixels": int(support.sum()), "scope": baseline["scope"],
                 "score_type": "frozen_fusion_score", "scale": "[0,1]; not a calibrated region probability",
                 "preview_interpretation": "red overlay shows fusion score; grey is outside geometric overlap"},
        parameters={"method": "fusion", "output_kind": "frozen_fusion_score", "units": "score",
                    "direction": "higher is more change", "geometry": {
                        "crop_shape": list(support.shape), "bbox_xyxy": case["geometry"]["bbox_xyxy"],
                        "anchor_sha256": case["geometry"]["anchor_sha256"]}},
        source=source, provenance=baseline["provenance"]))
    baseline["evidence_id"] = record["evidence_id"]
    case["method_evidence"].append(record)


def prepare_case(config, pair_spec, out, store, pool_factory=RemotePool):
    """Register a pair and compute independent existing maps on one region grid."""
    pair = resolve_pair_spec(pair_spec)
    out = _absolute(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "case.json").exists():
        raise FileExistsError("Case already exists; choose a new run directory")
    write_json(out / "input_manifest.json", pair)
    case = {k: pair[k] for k in ("building_id", "view_id", "split")}
    case.update(case_id=pair.get("case_id", digest(pair)[:24]), comparability="unknown", limitations=[],
                reference_date=pair["reference"]["date"], source_date=pair["source"]["date"],
                cutoff=pair["source"]["date"], maps=[], method_evidence=[], images=[])
    started = time.monotonic()
    geometry_started = started
    geometry_seconds = None
    reference, source = pair["reference"], pair["source"]
    anchor = pair.get("anchor_sha256", reference["sha256"])
    bbox = pair.get("bbox_xyxy", [0, 0, reference["native_shape"][1], reference["native_shape"][0]])
    if (not isinstance(bbox, list) or len(bbox) != 4 or any(type(v) is not int for v in bbox)
            or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
        raise ValueError("Region box must be a nonempty integer xyxy box on the native anchor grid")
    width, height = bbox[2]-bbox[0], bbox[3]-bbox[1]
    if width * height > config.get("max_region_pixels", 4_000_000):
        raise ValueError("Region exceeds configured pixel budget; select an explicit crop, no silent resize")
    if pair.get("region_id"):
        try:
            pinned = store.get_region(pair["region_id"])
        except KeyError:
            pinned = None
        if pinned is not None:
            anchor = pinned["anchor_sha256"]
            if "bbox_xyxy" not in pair:
                bbox = pinned["bbox_xyxy"]
                width, height = bbox[2]-bbox[0], bbox[3]-bbox[1]
    region = store.register_region(case["building_id"], case["view_id"], anchor, bbox,
                                   split=case["split"], region_id=pair.get("region_id"))
    case["region_id"] = region["region_id"]
    translation = np.array([[1, 0, -bbox[0]], [0, 1, -bbox[1]], [0, 0, 1]], float)
    prepared = all("crop_path" in row and "support_path" in row and "source_to_anchor" in row
                   for row in (reference, source))
    quality = {"passed": False}
    alignment_provenance = {}
    try:
        if prepared:
            _validate_prepared(pair, bbox, anchor)
            matrices = [_matrix(row["source_to_anchor"]) for row in (reference, source)]
            quality = {"passed": True, "interpretation": "saved canonical geometry; physical visibility unverified"}
            alignment_provenance = pair["prepared_provenance"]
        else:
            if reference["sha256"] != anchor:
                raise ValueError("Reference does not match pinned region anchor; explicit saved canonical correspondence required")
            matrix, quality, alignment_provenance = _alignment(pair, config, out)
            matrices = [np.eye(3), matrix]
        rgb_arrays, supports, observations = [], [], []
        for role, row, matrix in zip(("reference", "source"), (reference, source), matrices):
            folder = out / role
            folder.mkdir()
            transform = translation @ matrix
            if prepared:
                for key in ("crop", "support"):
                    if row.get(key + "_sha256") != sha256(row[key + "_path"]):
                        raise ValueError("Prepared RGB/support hash differs")
                rgb, opaque = load_rgb(row["crop_path"])
                support = _support(row["support_path"], (height, width)) & opaque
                if rgb.shape[:2] != (height, width):
                    raise ValueError("Prepared crop grid differs from the stable region")
            else:
                native, opaque = load_rgb(row["path"])
                canvas = Canvas(width, height, translation, matrix)
                # Both arguments are this original; source output is one native->crop warp.
                warped = warp_pair(native, opaque, native, opaque, canvas)
                rgb, support = warped["source"], warped["source_support"]
            crop_path, support_path = folder / "rgb.png", folder / "support.png"
            Image.fromarray(rgb).save(crop_path)
            Image.fromarray(support.astype(np.uint8) * 255).save(support_path)
            geometry = {"source_to_anchor": matrix.tolist(), "source_to_crop": transform.tolist(),
                "crop_to_source": np.linalg.inv(transform).tolist(), "crop_to_anchor": np.linalg.inv(translation).tolist(),
                "native_shape": row["native_shape"], "crop_shape": [height, width], "bbox_xyxy": bbox,
                "coordinate_system": "native anchor pixel centres; x=column y=row; region crop origin at box top-left"}
            observation = store.add_observation(dict(region_id=region["region_id"], date=row["date"],
                date_status=row["date_status"], available_at=row["available_at"], image_id=str(row["image_id"]),
                original_path=row["path"], original_sha256=row["sha256"], crop_path=str(crop_path),
                support_path=str(support_path), geometry=geometry, visibility="unknown",
                correspondence_status="aligned" if quality["passed"] else "alignment_rejected",
                provenance={"alignment": alignment_provenance, "quality_gate": quality,
                            "region_selection": "explicit fixed canonical box; no future support intersection"}))
            image_evidence = store.add_evidence(dict(kind="observation", region_id=region["region_id"],
                observation_ids=[observation["observation_id"]], available_at=row["available_at"],
                artifacts=[_artifact(observation["crop_path"], "rgb_crop", "image/png"),
                           _artifact(observation["support_path"], "geometric_support", "image/png")],
                summary={"role": role, "date": row["date"], "physical_visibility": "unknown"},
                coordinate_system="region_crop_pixels"))
            case[role + "_observation_id"] = observation["observation_id"]
            case["images"].append(dict(role=role, path=observation["crop_path"],
                observation_id=observation["observation_id"], evidence_id=image_evidence["evidence_id"]))
            observations.append(observation)
            rgb_arrays.append(rgb)
            supports.append(support)
        support = supports[0] & supports[1]
        support_path = out / "support.png"
        Image.fromarray(support.astype(np.uint8) * 255).save(support_path)
        case["support_path"] = str(support_path)
        case["geometry"] = {"anchor_sha256": anchor, "bbox_xyxy": bbox, "shape": [height, width],
            "reference_to_crop": observations[0]["geometry"]["source_to_crop"],
            "source_to_crop": observations[1]["geometry"]["source_to_crop"],
            "quality_gate": quality, "alignment": alignment_provenance}
        geometry_seconds = time.monotonic() - geometry_started
        case["coverage"] = {"overlap_pixels": int(support.sum()), "region_pixels": int(support.size),
            "reference_only_pixels": int((supports[0] & ~supports[1]).sum()),
            "newly_observed_pixels": int((supports[1] & ~supports[0]).sum()),
            "newly_observed_interpretation": "first observation, not automatically change"}
        case["limitations"].append("Geometric support does not establish visibility or absence of occlusion/damage")
        if not quality["passed"]:
            case["comparability"] = "alignment_rejected"
        elif not support.any():
            case["comparability"] = "no_comparable_support"
        elif reference["date"] is None or source["date"] is None or any(
                row["date_status"] == "unknown" for row in (reference, source)):
            case["comparability"] = "unknown_chronology"
        else:
            case["comparability"] = "geometrically_comparable_visibility_unknown"
        if any(row["date_status"] == "inferred" for row in (reference, source)):
            case["limitations"].append("Observation chronology includes inferred dates, not confirmed acquisition dates")
        if case["comparability"] != "geometrically_comparable_visibility_unknown":
            case["baseline"] = dict(status="unavailable", decision="not_comparable",
                                      reason=case["comparability"])
        else:
            case["maps"], arrays = _method_maps(config, pair, out, *rgb_arrays, support, case["geometry"], pool_factory)
            for item in case["maps"]:
                item["support_path"] = str(support_path)
                if item["status"] != "completed":
                    continue
                artifacts = [_artifact(value, "score_preview" if key == "score_preview_path" else key,
                                       "image/png" if Path(value).suffix == ".png" else "application/octet-stream") for key, value in item.items()
                             if key.endswith("_path") and isinstance(value, str) and Path(value).is_file()]
                evidence = store.add_evidence(dict(kind="prediction", region_id=case["region_id"],
                    observation_ids=[o["observation_id"] for o in observations], available_at=source["available_at"],
                    artifacts=artifacts, summary={k: item[k] for k in ("method", "status", "summary", "score_type", "scale", "direction", "threshold")},
                    parameters={"method": item["method"], "output_kind": item["metadata"].get("output_kind"),
                                "units": item["metadata"].get("native_units", item["metadata"].get("raw_units", "method_score")),
                                "direction": item["direction"], "semantic_mapping": item["metadata"].get("category_to_channel")},
                    provenance={"method_metadata": item["metadata"], "cache_key": item["cache_key"],
                                "source_artifact_sha256": item["source_artifact_sha256"]},
                    coordinate_system="region_crop_pixels"))
                item["evidence_id"] = evidence["evidence_id"]
                case["method_evidence"].append(evidence)
            case["baseline"] = dict(status="unavailable", decision="insufficient_data",
                                      reason="No frozen fusion selected; independent method maps retained")
            if config.get("fusion"):
                try:
                    case["baseline"] = _frozen_fusion(config["fusion"], arrays, case["maps"], support, out)
                    _fusion_evidence(case["baseline"], case, observations, support, rgb_arrays[1], store, out)
                except Exception as exc:
                    case["baseline"] = dict(status="error", decision="insufficient_data", error=f"{type(exc).__name__}: {exc}")
            elif config.get("diagnostic_baseline"):
                method = config["diagnostic_baseline"]
                item = next((m for m in case["maps"] if m["method"] == method and m["status"] == "completed"), None)
                if item and item.get("prediction_path"):
                    case["baseline"] = dict(status="diagnostic_only", method=method,
                        decision="changed" if item["summary"]["positive_pixels"] else "no_change",
                        mask_path=item["prediction_path"], threshold=item["threshold"],
                        scope="Explicit uncalibrated diagnostic; not the project's selected fusion")
        case["status"] = ("completed_with_errors" if case["baseline"]["status"] == "error"
                          or any(m["status"] == "error" for m in case["maps"]) else "completed")
    except Exception as exc:
        case.update(status="error", comparability="unresolved_correspondence",
                    baseline=dict(status="error", decision="not_comparable", error=f"{type(exc).__name__}: {exc}"))
        case["limitations"].append(str(exc))
    case["elapsed_seconds"] = time.monotonic() - started
    if case["baseline"].get("mask_path"):
        case["baseline"].update(mask_task="temporal_change", support_path=case["support_path"],
                                coordinate_system="region_crop_pixels")
    case["baseline_costs"] = dict(seconds=case["elapsed_seconds"],
        method_calls=sum(m.get("model_calls", 0) for m in case["maps"]),
        cache_hits=sum(bool(m.get("cache_hit")) for m in case["maps"]),
        model_loads=sum(m.get("model_loads", 0) for m in case["maps"]),
        images=2 * sum(m.get("status") == "completed" and not m.get("cache_hit") for m in case["maps"]))
    case["baseline_costs"].update(elapsed_seconds=case["elapsed_seconds"],
        model_calls=case["baseline_costs"]["method_calls"],
        model_load_attempts=sum(m.get("model_load_attempts", 0) for m in case["maps"]),
        geometry_seconds=geometry_seconds,
        scope="pair registration, geometry, independent methods, cache I/O and fusion; excludes VLM/retrieval/SAM3")
    case["preparation_costs"] = {**case["baseline_costs"],
        "shared": True, "scope": "shared once per case: registration, alignment/crop I/O, all configured method maps, caches and fusion; incremental VLM mode costs are separate"}
    write_json(out / "case.json", case)
    return case


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["align-worker"])
    parser.add_argument("request")
    args = parser.parse_args()
    _align_local(read_json(args.request), Path(args.request).resolve().parent)
