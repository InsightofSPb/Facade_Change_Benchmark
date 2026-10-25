"""Canonical crop tracks and cached frozen DINOv2 features.

Run ``python -m facade_change.history_features --help``. This preparation step
never estimates geometry, trains a model, invents event labels, or downloads
weights. Pixel coordinates are native to the earliest observation in each view.
The existing temporal RunStore owns the append-only attempt/event records.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import expanded_canvas, image_edges, transform_points
from .io import load_rgb, read_json, sha256, sync_directory, write_json


SCHEMA = 1
ROOT = Path(__file__).resolve().parents[1]
SPLITS = {"train", "val", "test"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _inside(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Artifact paths must be relative")
    path = (Path(root) / relative).resolve()
    if Path(root).resolve() not in path.parents:
        raise ValueError("Artifact path escapes its root")
    return path


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def _npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def progress(items, description):
    """tqdm when installed; useful bounded console progress without it."""
    items = list(items)
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    if tqdm is not None and sys.stderr.isatty():
        yield from tqdm(items, desc=description, unit="item")
        return
    start = last = time.monotonic()
    print(f"{description}: 0/{len(items)}", flush=True)
    for i, item in enumerate(items, 1):
        yield item
        now = time.monotonic()
        if i == len(items) or now - last >= 5:
            print(f"{description}: {i}/{len(items)} ({now-start:.1f}s)", flush=True)
            last = now


class Fingerprints:
    def __init__(self):
        self.files = {}

    def add(self, path, expected=None):
        path = Path(path).expanduser().resolve()
        value = sha256(path)
        if expected is not None and value != expected:
            raise ValueError(f"Hash mismatch: {path}")
        old = self.files.setdefault(str(path), value)
        if old != value:
            raise ValueError(f"Input changed during preparation: {path}")
        return value

    def verify(self):
        for path, expected in progress(sorted(self.files.items()), "Verify input fingerprints"):
            if sha256(path) != expected:
                raise ValueError(f"Input changed during preparation: {path}")


@contextmanager
def cache_lock(root):
    import fcntl
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".history-features.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is using {root}") from exc
        yield root


def _attempt(root, stage, config):
    # Adapt the existing journal instead of introducing a competing ledger.
    module = importlib.import_module("facade_change.2026-10-14_temporal_experiment")
    index = 1
    while (root / "reports" / f"{stage}-{index}").exists():
        index += 1
    class HistoryRunStore(module.RunStore):
        def event(self, kind, **values):
            # Simple folder names can repeat under smoke/full output roots.
            # Namespace event IDs without changing the project's ledger format.
            if not hasattr(self, "history_run_id"):
                self.history_run_id = f"{stage}-{uuid.uuid4().hex}"
            self.record["run_id"] = self.history_run_id
            return super().event(kind, run_id=self.history_run_id,
                                 event_id=f"{self.history_run_id}:{self.sequence + 1}", **values)
    store = HistoryRunStore(root / "reports" / f"{stage}-{index}", config)
    store.record["kind"] = "canonical_crop_histories" if stage == "prepare" else "frozen_history_features"
    _write(store.out / "run.json", store.record)
    return store


def _fail(store, exc):
    store.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                 error=f"{type(exc).__name__}: {exc}")


def _audit_images(images):
    buildings, views, hashes, identities, identifiers = {}, {}, {}, {}, set()
    for row in images:
        identifier = str(row["image_id"])
        if identifier in identifiers:
            raise ValueError(f"Duplicate image ID: {identifier}")
        identifiers.add(identifier)
        if row.get("metadata_status") != "reviewed" or not row.get("building_id") or not row.get("view_id"):
            raise ValueError(f"Temporal image lacks reviewed identity: {identifier}")
        if row.get("split") not in SPLITS or type(row.get("year")) is not int:
            raise ValueError(f"Temporal image lacks a known year/split: {identifier}")
        for field in ("width", "height"):
            if type(row.get(field)) is not int or row[field] <= 0:
                raise ValueError(f"Invalid image {field}: {identifier}")
        if not isinstance(row.get("sha256"), str) or len(row["sha256"]) != 64:
            raise ValueError(f"Missing image SHA-256: {identifier}")
        building, split = row["building_id"], row["split"]
        if buildings.setdefault(building, split) != split:
            raise ValueError(f"Building leakage across splits: {building}")
        if views.setdefault(row["view_id"], building) != building:
            raise ValueError(f"View assigned to different buildings: {row['view_id']}")
        if hashes.setdefault(row["sha256"], split) != split:
            raise ValueError("Identical image bytes occur in different splits")
        identity = (building, row["view_id"], row["year"])
        if identities.setdefault(row["sha256"], identity) != identity:
            raise ValueError("Identical image bytes have conflicting identity/year metadata")
    return {"status": "passed", "building_assignments": buildings,
            "scope": "reviewed temporal images in the input session; other semantic TRAIN images not audited"}


def read_session(session, fingerprints, quality_report=None):
    from .alignment_session import load_session
    loaded = load_session(session, fingerprints.files, quality_report=quality_report)
    inputs = read_json(loaded.path / "inputs.json")
    used = {str(row["image_id"]) for row in inputs["images"]}
    lookup = {str(row["image_id"]): dict(row) for row in loaded.manifest["images"]
              if str(row["image_id"]) in used}
    audit = _audit_images(list(lookup.values()))
    pairs = {(str(pair["reference_id"]), str(pair["source_id"])): pair for pair in loaded.pairs}
    if len(pairs) != len(loaded.pairs):
        raise ValueError("Duplicate observation pair in the alignment session")
    return loaded, lookup, pairs, audit


def checked_alignment(session, pair, fingerprints):
    """Use the project's hash-bound selection and geometry adapters."""
    from .saved_geometry import frame_geometry
    method = pair.get("selected_method")
    if method is None:
        return None, "unresolved_session_selection"
    if method == "geoscd":
        return None, "selected_dense_geometry_requires_separate_atlas_adapter"
    match = session.selected[pair["pair_id"]]
    frame = frame_geometry(match, fingerprints.files)
    selection, record = match["selection"], match["record"]
    return {"matrix": frame["source_to_reference"], "pair_id": pair["pair_id"], "method": method,
            "alignment_run": frame["alignment_run"],
            "alignment_run_sha256": selection["alignment_run_sha256"],
            "geometry_sha256": frame["geometry_sha256"], "review": selection["review"],
            "selection_status": selection["selection_status"], "quality_gate": record.get("quality_gate"),
            "session_completion": session.completion, "quality_report": session.quality_report}, None


def _previous_anchors(root):
    path = root / "manifest.json"
    if not path.exists():
        return {}
    document = read_json(path)
    if document.get("kind") != "canonical_crop_histories":
        raise ValueError("Output contains an unrelated manifest")
    return {(row["building_id"], row["view_id"]): row["anchor"]["sha256"] for row in document["views"]}


def plan_histories(session, root, fingerprints, limit_views=0, quality_report=None):
    session, images, pairs, audit = read_session(session, fingerprints, quality_report)
    pinned = _previous_anchors(root)
    groups = defaultdict(list)
    for row in images.values():
        groups[(row["building_id"], row["view_id"])].append(row)
    selected = sorted(groups)
    if limit_views:
        selected = selected[:limit_views]
    views = []
    for key in progress(selected, "Audit direct-to-anchor geometry"):
        observations = sorted(groups[key], key=lambda row: (row["year"], str(row["image_id"])))
        first_year = observations[0]["year"]
        earliest = [row for row in observations if row["year"] == first_year]
        anchor = next((row for row in earliest if row["sha256"] == pinned.get(key)), earliest[0])
        frames = []
        for row in observations:
            entry = {"image": row}
            if str(row["image_id"]) == str(anchor["image_id"]):
                entry.update(status="registered", alignment={"matrix": np.eye(3).tolist(),
                             "method": "identity", "selection_status": "anchor"})
            elif row["year"] == first_year:
                # The candidate alignment manifest has no within-year edges.
                entry.update(status="unregistered", reason="same_year_anchor_alternative_without_geometry")
            else:
                pair = pairs.get((str(anchor["image_id"]), str(row["image_id"])))
                if pair is None:
                    entry.update(status="unregistered", reason="missing_direct_anchor_pair")
                else:
                    try:
                        alignment, reason = checked_alignment(session, pair, fingerprints)
                        entry.update(status="registered" if alignment else "unregistered")
                        if alignment:
                            entry["alignment"] = alignment
                        else:
                            entry["reason"] = reason
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        entry.update(status="error", reason=f"{type(exc).__name__}: {exc}")
            frames.append(entry)
        views.append({"building_id": key[0], "view_id": key[1], "split": anchor["split"],
                      "anchor": anchor, "frames": frames,
                      "eligible_years": sorted({row["year"] for row in observations})})
    return views, audit, {"candidate_views": len(groups), "selected_views": len(views),
                          "candidate_images": len(images), "limited": bool(limit_views),
                          "alignment_session": str(session.path), "alignment_session_completion": session.completion,
                          "quality_report": session.quality_report}


def _frame_signature(anchor, frame, options, rendering):
    return {"schema": SCHEMA, "anchor_sha256": anchor["sha256"],
            "anchor_shape": [anchor["height"], anchor["width"]],
            "image_sha256": frame["image"]["sha256"],
            "image_shape": [frame["image"]["height"], frame["image"]["width"]],
            "source_to_anchor": frame["alignment"]["matrix"],
            "options": options, "rendering": rendering}


def _checked_frame_cache(root, signature):
    folder = root / "observations" / digest(signature)
    marker = folder / "observation.json"
    if not folder.exists():
        return None
    if not marker.is_file():
        raise ValueError(f"Incomplete committed observation cache: {folder}")
    record = read_json(marker)
    payload = {k: v for k, v in record.items() if k != "payload_sha256"}
    if (record.get("signature") != signature or record.get("cache_key") != digest(signature)
            or record.get("payload_sha256") != digest(payload)):
        raise ValueError("Observation cache signature mismatch")
    for crop in record["crops"]:
        for name in ("rgb", "support"):
            if sha256(_inside(folder, crop[name])) != crop[name + "_sha256"]:
                raise ValueError(f"Corrupt observation cache: {folder}/{crop[name]}")
    return record


def render_observation(root, anchor, frame, options, rendering, fingerprints):
    """Render once from original RGB; stable grid origin is anchor (0, 0)."""
    import cv2
    signature = _frame_signature(anchor, frame, options, rendering)
    image = frame["image"]
    fingerprints.add(image["image_path"], image["sha256"])
    cached = _checked_frame_cache(root, signature)
    if cached is not None:
        return cached, True
    rgb, opaque = load_rgb(image["image_path"])
    if rgb.shape[:2] != (image["height"], image["width"]):
        raise ValueError("Decoded native dimensions disagree with metadata")
    fingerprints.add(image["image_path"], image["sha256"])
    matrix = np.asarray(frame["alignment"]["matrix"], dtype=np.float64)
    canvas = expanded_canvas((anchor["height"], anchor["width"]), rgb.shape[:2], matrix,
                             max_pixels=options["max_canvas_pixels"], max_side=options["max_canvas_side"])
    transform = canvas.reference_to_canvas @ matrix
    size = (canvas.width, canvas.height)
    warped = cv2.warpPerspective(rgb, transform, size, flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    coverage = cv2.warpPerspective(opaque.astype(np.float32), transform, size,
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    support = coverage >= 1 - 1e-6
    low = np.rint(-canvas.reference_to_canvas[:2, 2]).astype(int)
    footprint = transform_points(image_edges(rgb.shape[:2]), matrix)
    bounds_low = np.floor(footprint.min(axis=0) + .5 + 1e-9).astype(int)
    bounds_high = np.ceil(footprint.max(axis=0) + .5 - 1e-9).astype(int)
    tile, stride = options["tile_size"], options["stride"]
    start = np.floor((bounds_low - tile + 1) / stride).astype(int) * stride
    end = np.ceil(bounds_high / stride).astype(int) * stride
    key = digest(signature)
    destination = root / "observations" / key
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = root / ".incomplete"
    temp_parent.mkdir(exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="observation-", dir=temp_parent))
    crops, last = [], time.monotonic()
    try:
        for y in range(int(start[1]), int(end[1]), stride):
            for x in range(int(start[0]), int(end[0]), stride):
                left, top = x - int(low[0]), y - int(low[1])
                xa, ya = max(0, left), max(0, top)
                xb, yb = min(canvas.width, left + tile), min(canvas.height, top + tile)
                if xa >= xb or ya >= yb:
                    continue
                piece_support = support[ya:yb, xa:xb]
                fraction = float(piece_support.sum()) / (tile * tile)
                if fraction + 1e-12 < options["min_valid_fraction"]:
                    continue
                pixels = np.zeros((tile, tile, 3), dtype=np.uint8)
                mask = np.zeros((tile, tile), dtype=bool)
                dx, dy = xa - left, ya - top
                pixels[dy:dy+yb-ya, dx:dx+xb-xa] = warped[ya:yb, xa:xb]
                mask[dy:dy+yb-ya, dx:dx+xb-xa] = piece_support
                stem = f"x{x:+d}_y{y:+d}"
                rgb_name, mask_name = stem + ".png", stem + "_support.png"
                Image.fromarray(pixels).save(temporary / rgb_name)
                Image.fromarray(mask.astype(np.uint8) * 255).save(temporary / mask_name)
                crops.append({"bbox_xywh": [x, y, tile, tile], "valid_fraction": fraction,
                              "rgb": rgb_name, "support": mask_name,
                              "rgb_sha256": sha256(temporary / rgb_name),
                              "support_sha256": sha256(temporary / mask_name)})
                if time.monotonic() - last >= 5:
                    print(f"  {image['year']} image {image['image_id']}: {len(crops)} crops", flush=True)
                    last = time.monotonic()
        fingerprints.add(image["image_path"], image["sha256"])
        record = {"schema_version": SCHEMA, "signature": signature, "cache_key": key,
                  "source_to_anchor": matrix.tolist(), "native_footprint": footprint.tolist(),
                  "canvas_origin_anchor_xy": low.tolist(), "crops": crops}
        record["payload_sha256"] = digest(record)
        _write(temporary / "observation.json", record)
        os.replace(temporary, destination)
        sync_directory(destination.parent)
    except BaseException:
        # Only this attempt's uncommitted temporary pixels are removed.
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return record, False


def assemble_tracks(views):
    tracks = {}
    for view in views:
        anchor = view["anchor"]
        for frame in view["frames"]:
            if frame.get("status") != "exported":
                continue
            image, cached = frame["image"], frame["cache"]
            folder = Path("observations") / cached["cache_key"]
            for crop in cached["crops"]:
                identity = {"building_id": view["building_id"], "view_id": view["view_id"],
                            "anchor_sha256": anchor["sha256"], "bbox_xywh": crop["bbox_xywh"]}
                track_id = digest(identity)[:24]
                track = tracks.setdefault(track_id, {"track_id": track_id, **identity,
                            "split": view["split"], "observations": [], "label_status": "unreviewed"})
                observation_id = digest([track_id, image["sha256"]])[:24]
                if any(row["observation_id"] == observation_id for row in track["observations"]):
                    continue
                track["observations"].append({"observation_id": observation_id,
                    "image_id": image["image_id"], "image_sha256": image["sha256"],
                    "original_image_path": image["image_path"], "year": image["year"],
                    "rgb": (folder / crop["rgb"]).as_posix(),
                    "support": (folder / crop["support"]).as_posix(),
                    "rgb_sha256": crop["rgb_sha256"], "support_sha256": crop["support_sha256"],
                    "valid_fraction": crop["valid_fraction"], "alignment": frame["alignment"]})
    result = []
    for track in tracks.values():
        track["observations"].sort(key=lambda row: (row["year"], str(row["image_id"])))
        groups = defaultdict(list)
        for row in track["observations"]:
            groups[row["year"]].append(row["observation_id"])
        years = sorted(groups)
        track.update(year_groups=[{"year": y, "observations": groups[y]} for y in years],
                     distinct_years=len(years),
                     delta_years=[0] + [b-a for a, b in zip(years, years[1:])],
                     elapsed_years=[y-years[0] for y in years])
        result.append(track)
    return sorted(result, key=lambda row: (row["building_id"], row["view_id"], row["bbox_xywh"]))


def history_summary(views, tracks, inventory):
    frames = [frame for view in views for frame in view["frames"]]
    summary = {**inventory, "kind": "canonical_crop_histories", "tracks": len(tracks),
               "observations": sum(len(t["observations"]) for t in tracks),
               "frame_statuses": dict(Counter(f["status"] for f in frames)),
               "exclusion_reasons": dict(Counter(f["reason"] for f in frames if f.get("reason"))),
               "reused_frames": sum(f.get("reused", False) for f in frames),
               "computed_frames": sum(f.get("status") == "exported" and not f.get("reused") for f in frames),
               "same_year_alternative_groups": sum(len(g["observations"]) > 1 for t in tracks for g in t["year_groups"]),
               "single_year_tracks_preserved": sum(t["distinct_years"] == 1 for t in tracks),
               "event_labels_created": False, "interpretation": "geometric crop tracks; review alignment and physical visibility"}
    for minimum in (2, 3, 4):
        subset = [t for t in tracks if t["distinct_years"] >= minimum]
        summary[f"histories_ge_{minimum}_years"] = {
            "tracks": len(subset), "buildings": len({t["building_id"] for t in subset}),
            "views": len({(t["building_id"], t["view_id"]) for t in subset}),
            "observations": sum(len(t["observations"]) for t in subset)}
    return summary


def _gallery(path, tracks, summary, root, limit=60):
    groups = defaultdict(list)
    for track in tracks:
        if track["distinct_years"] >= 2:
            groups[track["building_id"]].append(track)
    for group in groups.values():
        group.sort(key=lambda t: (-t["distinct_years"], t["track_id"]))
    selected = []
    while groups and len(selected) < limit:
        for building in sorted(list(groups)):
            selected.append(groups[building].pop(0))
            if not groups[building]:
                del groups[building]
            if len(selected) >= limit:
                break
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8">',
             '<title>Facade crop histories</title><style>body{font:15px system-ui;margin:24px}',
             '.row{display:flex;gap:8px;overflow:auto}figure{margin:0;min-width:150px}',
             'img{width:150px;height:150px;object-fit:contain}section{margin:28px 0}',
             'figcaption{font-size:12px;max-width:150px}pre{white-space:pre-wrap}</style>',
             '<h1>Canonical crop histories</h1><p>Original RGB; no augmentation. ',
             'Support is geometric availability, not an occlusion mask. Same-year images are alternatives.</p>',
             '<pre>' + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre>']
    for track in selected:
        title = f"{track['building_id']} / {track['view_id']} / {track['bbox_xywh']} / {track['split']}"
        parts.append('<section><h2>' + html.escape(title) + '</h2><div class="row">')
        for row in track["observations"]:
            src = os.path.relpath(_inside(root, row["rgb"]), path.parent)
            mask = os.path.relpath(_inside(root, row["support"]), path.parent)
            caption = f"{row['year']} · image {row['image_id']} · support {row['valid_fraction']:.1%}"
            parts.append('<figure><a href="' + html.escape(src, quote=True) + '"><img loading="lazy" src="'
                         + html.escape(src, quote=True) + '"></a><figcaption>' + html.escape(caption)
                         + '<br><a href="' + html.escape(mask, quote=True) + '">support</a></figcaption></figure>')
        parts.append('</div></section>')
    parts.append('</html>')
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(''.join(parts), encoding="utf-8")
    os.replace(temporary, path)


def prepare_histories(session, out, tile_size=224, stride=112, min_valid_fraction=.8,
                      max_canvas_pixels=50_000_000, max_canvas_side=16000,
                      limit_views=0, plan_only=False, quality_report=None):
    if type(tile_size) is not int or tile_size < 14 or tile_size % 14:
        raise ValueError("tile_size must be a positive multiple of DINOv2's 14-pixel patch size")
    if type(stride) is not int or not 0 < stride <= tile_size or not 0 < min_valid_fraction <= 1:
        raise ValueError("Require 0 < stride <= tile_size and 0 < min_valid_fraction <= 1")
    if limit_views < 0 or max_canvas_pixels < 1 or max_canvas_side < 1:
        raise ValueError("Invalid limit or canvas bound")
    options = {"tile_size": tile_size, "stride": stride, "min_valid_fraction": min_valid_fraction,
               "max_canvas_pixels": max_canvas_pixels, "max_canvas_side": max_canvas_side,
               "grid": "native anchor (0,0); fixed integer multiples of stride, including negative coordinates",
               "selection": "saved session selection; direct earliest-anchor homographies only"}
    with cache_lock(out) as root:
        fingerprints = Fingerprints()
        config = {"session": str(Path(session).expanduser().resolve()), "options": options,
                  "limit_views": limit_views, "plan_only": plan_only, "module_sha256": sha256(__file__),
                  "quality_report": str(Path(quality_report).expanduser().resolve()) if quality_report else None}
        store = _attempt(root, "prepare", config)
        try:
            views, audit, inventory = plan_histories(session, root, fingerprints, limit_views, quality_report)
            _write(store.out / "plan.json", {"views": views, "inventory": inventory, "audit": audit})
            if plan_only:
                summary = {**inventory, "registered_frames": sum(f["status"] == "registered" for v in views for f in v["frames"]),
                           "plan_only": True, "report": str(store.out)}
                _write(store.out / "summary.json", summary)
                fingerprints.verify()
                store.finish("planned", summary=summary)
                return summary
            import cv2
            import PIL
            rendering = {"cv2": cv2.__version__, "numpy": np.__version__, "Pillow": PIL.__version__,
                         "module_sha256": sha256(__file__),
                         "geometry_sha256": sha256(Path(__file__).with_name("geometry.py")),
                         "io_sha256": sha256(Path(__file__).with_name("io.py"))}
            jobs = [(view, frame) for view in views for frame in view["frames"] if frame["status"] == "registered"]
            for view, frame in progress(jobs, "Export canonical observations"):
                try:
                    cached, reused = render_observation(root, view["anchor"], frame, options, rendering, fingerprints)
                    frame.update(status="exported", cache=cached, reused=reused)
                    store.event("observation_reused" if reused else "observation_exported",
                                cache_key=cached["cache_key"], image_id=frame["image"]["image_id"],
                                crops=len(cached["crops"]))
                except (OSError, ValueError, KeyError, RuntimeError) as exc:
                    frame.update(status="error", reason=f"{type(exc).__name__}: {exc}")
                    store.event("observation_failed", image_id=frame["image"]["image_id"], reason=frame["reason"])
            tracks = assemble_tracks(views)
            summary = history_summary(views, tracks, inventory)
            compact_views = [{**view, "frames": [{k: v for k, v in frame.items() if k != "cache"}
                                                for frame in view["frames"]]} for view in views]
            document = {"schema_version": SCHEMA, "kind": "canonical_crop_histories",
                        "options": options, "views": compact_views, "tracks": tracks,
                        "leakage_check": audit, "input_sha256": fingerprints.files,
                        "report": store.out.relative_to(root).as_posix(),
                        "alignment_diagnostics_root": str(Path(inventory["alignment_session"]) / "quality"),
                        "quality_report": inventory["quality_report"],
                        "alignment_session_completion": inventory["alignment_session_completion"],
                        "label_scope": "unlabelled real RGB; existing annotations/reviews must be joined separately"}
            _write(store.out / "manifest.json", document)
            _write(store.out / "summary.json", summary)
            _write(store.out / "leakage_check.json", audit)
            _gallery(store.out / "gallery.html", tracks, summary, root)
            fingerprints.verify()
            status = "completed_with_issues" if any(f["status"] != "exported" for v in views for f in v["frames"]) else "completed_needs_review"
            store.finish(status, manifest_sha256=sha256(store.out / "manifest.json"), summary=summary)
            # Public indexes are replaceable pointers; immutable reports and pixels remain.
            _write(root / "summary.json", summary)
            _gallery(root / "gallery.html", tracks, summary, root)
            _write(root / "manifest.json", document)
            return {**summary, "manifest": str(root / "manifest.json"), "gallery": str(root / "gallery.html")}
        except BaseException as exc:
            _fail(store, exc)
            raise
        finally:
            store.close()


NORMALIZATION = {"rgb_scale": 255.0, "mean": [.485, .456, .406], "std": [.229, .224, .225]}


def read_dino_config(path=None, worker_python=None, trust_checkpoint=False):
    if path is None:
        local = ROOT / "configs" / "methods.local.json"
        path = local if local.exists() else ROOT / "configs" / "methods.example.json"
    path = Path(path).expanduser().resolve()
    options = read_json(path)["methods"]["dinov2"]
    result = {"config_path": str(path), "config_sha256": sha256(path),
              "trust_checkpoint": bool(trust_checkpoint)}
    for name in ("source_root", "checkpoint_path"):
        value = Path(options[name]).expanduser()
        result[name] = str((value if value.is_absolute() else path.parent / value).resolve())
    # Resolving a venv interpreter symlink changes its environment. Preserve it.
    python = Path(worker_python or options.get("worker_python") or sys.executable).expanduser()
    if not python.is_absolute():
        python = (Path.cwd() if worker_python else path.parent) / python
    result["worker_python"] = os.path.abspath(python)
    if not Path(result["worker_python"]).is_file():
        raise FileNotFoundError(f"DINOv2 worker Python does not exist: {result['worker_python']}")
    if not (Path(result["source_root"]) / "hubconf.py").is_file():
        raise FileNotFoundError(f"Local DINOv2 hubconf.py is missing: {result['source_root']}")
    if not Path(result["checkpoint_path"]).is_file():
        raise FileNotFoundError(f"Local DINOv2 checkpoint is missing: {result['checkpoint_path']}")
    return result


def encoder_contract(settings, device, save_patch_tokens, fingerprints):
    import torch
    from .methods.dinov2 import _source_provenance
    checkpoint_sha = fingerprints.add(settings["checkpoint_path"])
    fingerprints.add(settings["config_path"], settings["config_sha256"])
    sources = _source_provenance(settings["source_root"])
    contract = {"schema_version": SCHEMA, "backbone": "dinov2_vits14", "dimensions": 384,
                "checkpoint_sha256": checkpoint_sha,
                "python_tree_sha256": sources["python_tree_sha256"],
                "module_sha256": sha256(__file__),
                "loader_sha256": sha256(Path(__file__).parent / "methods" / "dinov2.py"),
                "normalization": NORMALIZATION, "resize": False, "augmentation": False,
                "invalid_pixels": "zero after normalization (ImageNet mean before normalization)",
                "pooling": "mean of fully supported x_norm_patchtokens, then L2 normalize",
                "cls": "L2-normalized x_norm_clstoken, diagnostic only",
                "patch_support": "all 196 pixels of each 14x14 patch are supported",
                "attention_caveat": "valid tokens can attend to invalid tokens; support is not an attention mask",
                "precision": "float32 inference and embeddings; optional float16 patch-token storage",
                "save_patch_tokens": bool(save_patch_tokens), "weights_frozen": True, "downloads": False,
                "device": str(device), "torch": torch.__version__, "numpy": np.__version__,
                "cuda_runtime": torch.version.cuda, "tf32": False}
    return contract


def _l2(values):
    values = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norm <= 1e-12):
        raise ValueError("Non-finite or zero feature vector")
    return (values / norm).astype(np.float32)


def supported_patches(mask, patch=14):
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.ndim != 2 or any(n % patch for n in mask.shape):
        raise ValueError("Expected a boolean support grid with dimensions divisible by 14")
    h, w = mask.shape
    return mask.reshape(h//patch, patch, w//patch, patch).all(axis=(1, 3)).reshape(-1)


class DinoEncoder:
    """Thin adapter around the project's offline, strict checkpoint loader."""
    def __init__(self, settings, device, contract):
        import torch
        from .methods.dinov2 import _freeze, _load_dino
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("This initial adapter supports cpu or cuda devices")
        hardware = {"device": str(self.device)}
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable in the configured DINOv2 environment")
            try:
                check = subprocess.run(["nvidia-smi"], text=True, capture_output=True, timeout=15)
                hardware["nvidia_smi"] = check.stdout or check.stderr
                print(hardware["nvidia_smi"], flush=True)
            except (OSError, subprocess.TimeoutExpired) as exc:
                hardware["nvidia_smi"] = f"unavailable: {exc}"
            free, total = torch.cuda.mem_get_info(self.device)
            hardware.update(free_bytes=free, total_bytes=total,
                            device_name=torch.cuda.get_device_name(self.device))
            print(f"DINOv2 GPU memory: {free / 2**30:.2f}/{total / 2**30:.2f} GiB free", flush=True)
            # Conservative preflight; actual batch peak is reported at completion.
            if free < 512 * 2**20:
                raise RuntimeError("Less than 512 MiB GPU memory is free; use --device cpu or retry later")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats(self.device)
        model, provenance = _load_dino(settings["source_root"], settings["checkpoint_path"],
                                       settings["trust_checkpoint"])
        if (provenance["checkpoint"]["sha256"] != contract["checkpoint_sha256"]
                or provenance["sources"]["python_tree_sha256"] != contract["python_tree_sha256"]):
            raise ValueError("DINOv2 checkpoint/source changed after preflight")
        self.model = _freeze(model, self.device)
        self.provenance = {**provenance, "hardware": hardware}

    def encode(self, rgbs, masks, save_patch_tokens=False):
        import torch
        rgb = np.stack(rgbs)
        masks = np.stack(masks)
        if rgb.dtype != np.uint8 or rgb.shape[:3] != masks.shape or rgb.shape[-1] != 3:
            raise ValueError("Invalid batched RGB/support shapes")
        valid = np.stack([supported_patches(mask) for mask in masks])
        if not valid.any(axis=1).all():
            raise ValueError("A crop has no fully supported 14x14 patch")
        with torch.inference_mode():
            pixels = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(self.device, torch.float32) / 255
            mean = pixels.new_tensor(NORMALIZATION["mean"]).view(1, 3, 1, 1)
            std = pixels.new_tensor(NORMALIZATION["std"]).view(1, 3, 1, 1)
            normalized = (pixels - mean) / std
            mask_tensor = torch.from_numpy(masks).to(self.device).unsqueeze(1)
            normalized = normalized.masked_fill(~mask_tensor, 0)
            features = self.model.forward_features(normalized)
            tokens = features["x_norm_patchtokens"].float().cpu().numpy()
            cls = features["x_norm_clstoken"].float().cpu().numpy()
        if tokens.shape != (len(rgbs), valid.shape[1], 384) or cls.shape != (len(rgbs), 384):
            raise ValueError("Unexpected original DINOv2 ViT-S/14 feature shape")
        if not np.isfinite(tokens).all():
            raise ValueError("Non-finite DINOv2 patch tokens")
        result = []
        for token, cls_token, patch_mask in zip(tokens, cls, valid):
            arrays = {"z": _l2(token[patch_mask].mean(axis=0)), "cls": _l2(cls_token),
                      "patch_support": patch_mask}
            if save_patch_tokens:
                arrays["patch_tokens"] = token.astype(np.float16)
                if not np.isfinite(arrays["patch_tokens"]).all():
                    raise ValueError("DINOv2 tokens overflow float16 storage")
            result.append(arrays)
        return result

    def close(self):
        self.model = None
        if self.device.type == "cuda":
            import torch
            self.provenance["hardware"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device)


def _history_document(root, fingerprints):
    path = root / "manifest.json"
    manifest_sha = fingerprints.add(path)
    document = read_json(path)
    if document.get("kind") != "canonical_crop_histories" or document.get("schema_version") != SCHEMA:
        raise ValueError("--histories must contain a canonical crop-history manifest")
    report = _inside(root, document["report"])
    fingerprints.add(report / "manifest.json", manifest_sha)
    fingerprints.add(report / "run.json")
    record = read_json(report / "run.json")
    if not record.get("status", "").startswith("completed") or record.get("manifest_sha256") != manifest_sha:
        raise ValueError("History manifest lacks a matching completed preparation record")
    buildings, views, tracks, observations = {}, {}, set(), set()
    for track in document["tracks"]:
        if track["track_id"] in tracks:
            raise ValueError("Duplicate track ID")
        tracks.add(track["track_id"])
        if track["split"] not in SPLITS or buildings.setdefault(track["building_id"], track["split"]) != track["split"]:
            raise ValueError("Building leakage in histories")
        if views.setdefault(track["view_id"], track["building_id"]) != track["building_id"]:
            raise ValueError("View leakage in histories")
        years = []
        for row in track["observations"]:
            if row["observation_id"] in observations or type(row["year"]) is not int:
                raise ValueError("Invalid/duplicate history observation")
            observations.add(row["observation_id"])
            years.append(row["year"])
            _inside(root, row["rgb"])
            _inside(root, row["support"])
        if years != sorted(years) or len(set(years)) != track["distinct_years"]:
            raise ValueError("History chronology/length mismatch")
    return document, manifest_sha


def _feature_signature(row, contract):
    return {"encoder_sha256": digest(contract), "rgb_sha256": row["rgb_sha256"],
            "support_sha256": row["support_sha256"]}


def _feature_cache(root, signature):
    folder = root / "observations" / digest(signature)
    if not folder.exists():
        return None
    record = read_json(folder / "feature.json")
    payload = {k: v for k, v in record.items() if k != "payload_sha256"}
    relative = (Path("observations") / digest(signature) / "features.npz").as_posix()
    if (record.get("signature") != signature or record.get("payload_sha256") != digest(payload)
            or record.get("feature_id") != digest(signature) or record.get("path") != relative
            or sha256(folder / "features.npz") != record.get("sha256")):
        raise ValueError(f"Corrupt frozen-feature cache: {folder}")
    with np.load(folder / "features.npz", allow_pickle=False) as arrays:
        if arrays["z"].shape != (384,) or not np.isfinite(arrays["z"]).all():
            raise ValueError("Invalid cached DINOv2 embedding")
    return record


def _save_feature(root, signature, arrays):
    key = digest(signature)
    destination = root / "observations" / key
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = root / ".incomplete"
    temporary_root.mkdir(exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="feature-", dir=temporary_root))
    try:
        _npz(temporary / "features.npz", arrays)
        record = {"schema_version": SCHEMA, "signature": signature, "feature_id": key,
                  "path": (Path("observations") / key / "features.npz").as_posix(),
                  "sha256": sha256(temporary / "features.npz"),
                  "supported_patches": int(arrays["patch_support"].sum())}
        record["payload_sha256"] = digest(record)
        _write(temporary / "feature.json", record)
        os.replace(temporary, destination)
        sync_directory(destination.parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return record


def _read_crop(root, row, size, fingerprints):
    rgb_path, mask_path = (_inside(root, row[k]) for k in ("rgb", "support"))
    fingerprints.add(rgb_path, row["rgb_sha256"])
    fingerprints.add(mask_path, row["support_sha256"])
    rgb, opaque = load_rgb(rgb_path)
    with Image.open(mask_path) as image:
        mask = np.asarray(image)
    if (rgb.shape != (size, size, 3) or mask.dtype != np.uint8 or mask.shape != (size, size)
            or not np.isin(mask, [0, 255]).all() or not opaque.all()):
        raise ValueError("Non-native crop or invalid binary support mask")
    mask = mask == 255
    if abs(float(mask.mean()) - row["valid_fraction"]) > 1e-9:
        raise ValueError("Crop support disagrees with its manifest")
    if not supported_patches(mask).any():
        raise ValueError("No fully supported DINOv2 patch in crop")
    return rgb, mask


def _sequence_arrays(track, feature_by_observation, feature_root):
    rows = track["observations"]
    vectors = []
    for row in rows:
        path = _inside(feature_root, feature_by_observation[row["observation_id"]]["path"])
        with np.load(path, allow_pickle=False) as data:
            vectors.append(data["z"])
    years = np.asarray([row["year"] for row in rows], dtype=np.int32)
    unique_years = np.unique(years)
    z = np.stack(vectors).astype(np.float32)
    pooled = np.stack([_l2(z[years == year].mean(axis=0)) for year in unique_years])
    # Same-year alternatives remain recoverable; they are not zero-gap time steps.
    return {"z_observations": z, "observation_years": years,
            "observation_ids": np.asarray([row["observation_id"] for row in rows]),
            "z_years": pooled, "years": unique_years,
            "year_counts": np.asarray([(years == y).sum() for y in unique_years], dtype=np.int32),
            "delta_years": np.diff(unique_years, prepend=unique_years[0]),
            "elapsed_years": unique_years - unique_years[0]}


def extract_features(histories, out, settings, device="cpu", batch_size=32,
                     min_years=2, save_patch_tokens=False):
    """Freeze each observation independently; no temporal training or event GT."""
    if batch_size < 1 or min_years < 1:
        raise ValueError("batch_size and min_years must be positive")
    history_root = Path(histories).expanduser().resolve()
    if Path(out).expanduser().resolve() == history_root:
        raise ValueError("Use a separate features directory, e.g. histories/features")
    with cache_lock(out) as root:
        fingerprints = Fingerprints()
        config = {"histories": str(history_root), "settings": settings, "device": device,
                  "batch_size": batch_size, "min_years": min_years,
                  "save_patch_tokens": save_patch_tokens, "module_sha256": sha256(__file__)}
        store = _attempt(root, "embed", config)
        encoder = None
        try:
            document, history_sha = _history_document(history_root, fingerprints)
            tracks = [t for t in document["tracks"] if t["distinct_years"] >= min_years]
            if not tracks:
                raise ValueError(f"No crop histories with at least {min_years} distinct years")
            contract = encoder_contract(settings, device, save_patch_tokens, fingerprints)
            store.contract({"history_manifest_sha256": history_sha, "encoder": contract})
            _write(store.out / "encoder.json", contract)
            jobs, by_observation = {}, {}
            for track in tracks:
                for row in track["observations"]:
                    signature = _feature_signature(row, contract)
                    key = digest(signature)
                    jobs.setdefault(key, {"row": row, "signature": signature, "observation_ids": []})["observation_ids"].append(row["observation_id"])
            pending, reused = [], 0
            for job in progress(jobs.values(), "Verify feature cache"):
                row = job["row"]
                for name in ("rgb", "support"):
                    fingerprints.add(_inside(history_root, row[name]), row[name + "_sha256"])
                cached = _feature_cache(root, job["signature"])
                if cached is None:
                    pending.append(job)
                else:
                    reused += 1
                    for observation_id in job["observation_ids"]:
                        by_observation[observation_id] = cached
                    store.event("feature_reused", feature_id=cached["feature_id"],
                                observation_ids=job["observation_ids"])
            if pending:
                print(f"DINOv2: {reused} cached, {len(pending)} pending; loading local frozen model", flush=True)
                encoder = DinoEncoder(settings, device, contract)
                _write(store.out / "model.json", encoder.provenance)
                store.event("model_loaded", checkpoint_sha256=contract["checkpoint_sha256"])
            for offset in progress(range(0, len(pending), batch_size), "DINOv2 batches"):
                batch = pending[offset:offset + batch_size]
                decoded = [_read_crop(history_root, job["row"], document["options"]["tile_size"], fingerprints)
                           for job in batch]
                arrays = encoder.encode([x[0] for x in decoded], [x[1] for x in decoded], save_patch_tokens)
                if len(arrays) != len(batch):
                    raise ValueError("Encoder returned a different batch size")
                for job, feature in zip(batch, arrays):
                    for name in ("rgb", "support"):
                        fingerprints.add(_inside(history_root, job["row"][name]), job["row"][name + "_sha256"])
                    record = _save_feature(root, job["signature"], feature)
                    for observation_id in job["observation_ids"]:
                        by_observation[observation_id] = record
                    store.event("feature_computed", feature_id=record["feature_id"],
                                observation_ids=job["observation_ids"], artifact_sha256=record["sha256"])
            sequence_index = []
            for track in progress(tracks, "Pack year-aware sequences"):
                path = store.out / "sequences" / (track["track_id"] + ".npz")
                _npz(path, _sequence_arrays(track, by_observation, root))
                sequence_index.append({**{k: track[k] for k in ("track_id", "building_id", "view_id", "split", "bbox_xywh", "distinct_years")},
                                       "path": path.relative_to(root).as_posix(), "sha256": sha256(path),
                                       "observation_ids": [row["observation_id"] for row in track["observations"]]})
            fingerprints.verify()
            from .methods.dinov2 import _source_provenance
            if _source_provenance(settings["source_root"])["python_tree_sha256"] != contract["python_tree_sha256"]:
                raise ValueError("DINOv2 source changed during extraction")
            if encoder:
                encoder.close()
                _write(store.out / "model.json", encoder.provenance)
                encoder = None
            summary = {"kind": "frozen_history_features", "histories": len(tracks),
                       "buildings": len({t["building_id"] for t in tracks}),
                       "views": len({(t["building_id"], t["view_id"]) for t in tracks}),
                       "observations": len(by_observation), "unique_crop_inputs": len(jobs),
                       "reused_features": reused, "computed_features": len(pending),
                       "feature_dimension": 384, "min_distinct_years": min_years,
                       "history_length_counts": dict(Counter(t["distinct_years"] for t in tracks)),
                       "trained_temporal_embedding": False, "event_labels_created": False}
            index = {"schema_version": SCHEMA, "kind": "frozen_history_features", "encoder": contract,
                     "history_manifest_sha256": history_sha, "history_root": str(history_root),
                     "history_report": document["report"], "sequences": sequence_index,
                     "features": by_observation, "input_sha256": fingerprints.files,
                     "report": store.out.relative_to(root).as_posix(),
                     "time_policy": "one L2-normalized mean per observed year; original alternatives retained; no interpolation",
                     "learning_scope": "frozen per-date z only; no learned temporal h and no event labels"}
            _write(store.out / "index.json", index)
            _write(store.out / "summary.json", summary)
            store.finish("completed", index_sha256=sha256(store.out / "index.json"), summary=summary)
            _write(root / "summary.json", summary)
            _write(root / "index.json", index)
            return {**summary, "index": str(root / "index.json")}
        except BaseException as exc:
            _fail(store, exc)
            raise
        finally:
            if encoder:
                encoder.close()
            store.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Reuse direct saved alignments to build canonical RGB crop histories")
    prepare.add_argument("--alignment-session", "--session", dest="session", required=True, type=Path)
    prepare.add_argument("--quality-report", type=Path, help="Exact completed quality rerun for a stopped/failed parent session")
    prepare.add_argument("--out", type=Path, default=ROOT / "data" / "histories")
    prepare.add_argument("--tile-size", type=int, default=224)
    prepare.add_argument("--stride", type=int, default=112)
    prepare.add_argument("--min-valid-fraction", type=float, default=.8)
    prepare.add_argument("--max-canvas-pixels", type=int, default=50_000_000)
    prepare.add_argument("--max-canvas-side", type=int, default=16000)
    prepare.add_argument("--limit-views", type=int, default=0, help="Smoke-test subset; use a separate output root")
    prepare.add_argument("--plan-only", action="store_true")
    embed = commands.add_parser("embed", help="Cache frozen offline DINOv2 features and pack per-year sequences")
    embed.add_argument("--histories", type=Path, default=ROOT / "data" / "histories")
    embed.add_argument("--out", type=Path)
    embed.add_argument("--methods-config", type=Path)
    embed.add_argument("--worker-python", help="Override the DINOv2 interpreter; preserve venv symlink")
    embed.add_argument("--device", default="cuda:0")
    embed.add_argument("--batch-size", type=int, default=32)
    embed.add_argument("--min-years", type=int, default=2)
    embed.add_argument("--save-patch-tokens", action="store_true")
    embed.add_argument("--trust-checkpoint", action="store_true")
    embed.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return result


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(arguments)
    try:
        if args.command == "prepare":
            summary = prepare_histories(**{k: v for k, v in vars(args).items() if k != "command"})
        else:
            settings = read_dino_config(args.methods_config, args.worker_python, args.trust_checkpoint)
            if not args.worker:
                # Resolve inputs before changing cwd, so an invocation outside ROOT
                # cannot point the worker at another config or output directory.
                command = [settings["worker_python"], "-s", "-u", "-m", "facade_change.history_features", "embed",
                           "--histories", str(args.histories.expanduser().resolve()),
                           "--out", str((args.out or args.histories / "features").expanduser().resolve()),
                           "--methods-config", settings["config_path"], "--worker-python", settings["worker_python"],
                           "--device", args.device, "--batch-size", str(args.batch_size),
                           "--min-years", str(args.min_years), "--worker"]
                if args.trust_checkpoint:
                    command.append("--trust-checkpoint")
                if args.save_patch_tokens:
                    command.append("--save-patch-tokens")
                print(f"DINOv2 worker: {settings['worker_python']}", flush=True)
                return subprocess.run(command, cwd=ROOT).returncode
            summary = extract_features(args.histories, args.out or args.histories / "features", settings,
                                       args.device, args.batch_size, args.min_years, args.save_patch_tokens)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0
    except KeyboardInterrupt:
        print("Interrupted; committed observations/features are retained for the next invocation.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        print(f"history_features failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
