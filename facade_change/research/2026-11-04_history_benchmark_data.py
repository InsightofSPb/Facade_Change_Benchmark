"""Canonical RGB histories and selected-observation annotation targets, no encoder."""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import importlib
from pathlib import Path

import numpy as np
from PIL import Image

from ..annotation_targets import load_annotation_source, project_masks, resolve_path, target_difference
from ..history_features import Fingerprints, _history_document, _inside, digest, progress
from ..io import environment, load_rgb, read_json, sha256, write_json


PROTOCOL = "canonical-history-val-calibration-v1"
SPLITS = ("train", "val", "test")
Store = importlib.import_module("facade_change.2026-10-14_temporal_experiment").RunStore


def source_annotations(document, fingerprints, override=None):
    if override:
        path = Path(override).expanduser().resolve()
    else:
        diagnostics = document.get("alignment_diagnostics_root")
        if not diagnostics:
            raise ValueError("Missing annotation source; supply --annotation-manifest")
        inputs = Path(diagnostics).parent / "inputs.json"
        expected = document.get("input_sha256", {}).get(str(inputs.resolve()))
        if not expected:
            raise ValueError("Missing saved session input hash; supply --annotation-manifest")
        fingerprints.add(inputs, expected)
        record = read_json(inputs)
        path = resolve_path(record["manifest"], inputs.parent)
        fingerprints.add(path, record["manifest_sha256"])
    return load_annotation_source(path, fingerprints)


def select_tracks(document):
    """One deterministic observation per year; no GT or method output is inspected."""
    tracks, image_splits = [], {}
    for original in document["tracks"]:
        groups = defaultdict(list)
        for row in original["observations"]:
            if image_splits.setdefault(row["image_sha256"], original["split"]) != original["split"]:
                raise ValueError("Source-image SHA appears in multiple splits")
            if not np.isfinite(row["valid_fraction"]) or not 0 <= row["valid_fraction"] <= 1:
                raise ValueError("Invalid geometric coverage in observation")
            groups[row["year"]].append(row)
        if len(groups) < 3:
            continue
        observations = [min(groups[year], key=lambda r: (-r["valid_fraction"], str(r["image_id"]),
                                                        r["observation_id"])) for year in sorted(groups)]
        tracks.append({**original, "observations": observations})
    return sorted(tracks, key=lambda t: (t["split"], t["building_id"], t["track_id"]))


def counts(tracks):
    result = {}
    for split in ("all", *SPLITS):
        selected = [t for t in tracks if split == "all" or t["split"] == split]
        lengths = [len(t["observations"]) for t in selected]
        result[split] = dict(tracks=len(selected), observations=sum(lengths),
            candidates=sum(k - 2 for k in lengths), warmup_pairs=len(selected),
            history_calls_min=sum(k - 1 for k in lengths),
            history_calls_max=sum(k * (k - 1) // 2 for k in lengths),
            buildings=len({t["building_id"] for t in selected}))
    return result


def support_image(root, row, shape):
    with Image.open(_inside(root, row["support"])) as image:
        values = np.asarray(image)
    if values.dtype != np.uint8 or values.shape != shape or not np.isin(values, (0, 255)).all():
        raise ValueError("Saved support must be a native-grid uint8 0/255 image")
    return values == 255


def prepare(args):
    path = Path(args.histories).expanduser().resolve()
    root = path if path.is_dir() else path.parent
    if path.is_file() and path.name != "manifest.json":
        raise ValueError("Use the canonical manifest.json or its directory")
    fp = Fingerprints()
    document, manifest_hash = _history_document(root, fp)
    if args.expected_sha256 and manifest_hash != args.expected_sha256:
        raise ValueError("Selected history manifest SHA-256 differs from --expected-sha256")
    options = document["options"]
    if (options["tile_size"] != 224 or options["stride"] != 192
            or options.get("min_valid_fraction") != .8):
        raise ValueError("This benchmark requires 224-pixel crops, stride 192 and minimum coverage 0.8")
    audit = document.get("leakage_check", {})
    if audit.get("status") != "passed":
        raise ValueError("Canonical histories need the saved reviewed-identity leakage check")
    assignments = audit.get("building_assignments", {})
    if any(assignments.get(t["building_id"]) != t["split"] for t in document["tracks"]):
        raise ValueError("History split differs from the reviewed-identity audit")
    if args.boundary_tolerance < 0:
        raise ValueError("Boundary tolerance must be nonnegative")
    summary_path = _inside(root, document["report"]) / "summary.json"
    fp.add(summary_path)
    preparation = read_json(summary_path)
    if preparation.get("limited"):
        raise ValueError("History export used limit_views; prepare a complete export first")
    tracks = select_tracks(document)
    if not tracks or set(t["split"] for t in tracks) != set(SPLITS):
        raise ValueError("Need at least one >=3-year history in each of TRAIN/VAL/TEST")
    summary = counts(tracks)
    if args.expected_candidates is not None and summary["all"]["candidates"] != args.expected_candidates:
        raise ValueError(f"Expected {args.expected_candidates} candidates, found {summary['all']['candidates']}")
    source = source_annotations(document, fp, args.annotation_manifest)
    dimensions = {frame["image"]["sha256"]: (frame["image"]["height"], frame["image"]["width"])
                  for view in document["views"] for frame in view["frames"]}
    categories = source["categories"]
    groups_path = Path(args.class_groups).expanduser().resolve()
    fp.add(groups_path)
    groups = read_json(groups_path)["groups"]
    names = {c["name"] for c in categories}
    if (len({g["name"] for g in groups}) != len(groups)
            or any(not g["categories"] or not set(g["categories"]) <= names for g in groups)):
        raise ValueError("Reporting groups do not match annotation categories")
    jobs = defaultdict(list)
    for track in tracks:
        if track["bbox_xywh"][2:] != [224, 224]:
            raise ValueError("Track geometry does not match the declared 224-pixel grid")
        for row in track["observations"]:
            for field in ("rgb", "support"):
                fp.add(_inside(root, row[field]), row[field + "_sha256"])
            jobs[row["image_sha256"]].append((track, row))
    config = {k: v for k, v in vars(args).items() if k != "resume"}
    out = Path(args.out).expanduser().resolve()
    if root == out or root in out.parents or out in root.parents:
        raise ValueError("Prepared output must be separate from the RGB history directory")
    if (out / "run.json").exists() and read_json(out / "run.json")["status"] == "completed":
        raise ValueError("Preparation already completed; use it directly or choose a new --out")
    store = Store(out, config, resume=args.resume)
    store.record["kind"] = "canonical_history_prepared"
    write_json(out / "run.json", store.record)
    projected, events, status = {}, [], Counter()
    try:
        from ..segmentation.data import rasterize_annotations
        import facade_change.annotation_targets as annotation_core
        env = {k: v for k, v in environment().items() if k not in ("git_commit", "git_dirty")}
        env["pycocotools"] = importlib.metadata.version("pycocotools")
        store.contract(dict(protocol=PROTOCOL, inputs=fp.files, source=sha256(__file__),
            environment=env,
            annotation_core=sha256(annotation_core.__file__), categories=categories, groups=groups,
            selection="maximum valid_fraction, then string image_id, then observation_id per track/year",
            target="selected previous/current observations; annotation differences, not reviewed events"))
        for image_hash, items in progress(sorted(jobs.items()), "Project selected observation masks"):
            image = source["images"].get(image_hash)
            packed = None
            if image is not None:
                coco_image = source["coco_images"].get(str(image["image_id"]))
                if coco_image is None or any(coco_image[k] != image[k] for k in ("height", "width")):
                    raise ValueError("COCO/annotation manifest image dimensions disagree")
                if dimensions.get(image_hash) != (image["height"], image["width"]):
                    raise ValueError("Annotation/history source image dimensions disagree")
            for track, row in items:
                if image is not None and (any(image.get(k) != track[k] for k in
                                             ("building_id", "view_id", "split")) or image.get("year") != row["year"]):
                    raise ValueError("Annotation/history identity, split or year mismatch")
                identifier = "observation-" + row["observation_id"]
                cached = store.read_unit(identifier)
                if cached:
                    projected[row["observation_id"]] = cached[0]
                    continue
                support = support_image(root, row, (224, 224))
                if image is None:
                    labels = np.zeros(support.shape, np.uint16)
                    reason = "missing_annotation_image"
                else:
                    if packed is None:
                        packed = rasterize_annotations(source["annotations"].get(str(image["image_id"]), []),
                            image["height"], image["width"], source["channel_map"])
                    labels = project_masks(packed, row["alignment"]["matrix"], track["bbox_xywh"], support)
                    reason = "projected" if labels.any() else "unannotated_crop"
                projected[row["observation_id"]] = store.save_unit(
                    dict(unit_id=identifier, observation_id=row["observation_id"], reason=reason),
                    dict(packed=labels, support=support))
        for track in progress(tracks, "Build matched target steps"):
            observations = []
            for row in track["observations"]:
                observations.append(store.read_unit("observation-" + row["observation_id"])[1])
            for index in range(2, len(observations)):
                previous, current = observations[index - 1:index + 1]
                target, layers, support, _ = target_difference(previous, current, range(len(categories)),
                                                               args.boundary_tolerance)
                # Causal, method-independent evaluation support, shared by all three modes.
                common = support & observations[0]["support"]
                target[~common], layers[:, ~common] = 255, 255
                event_id = digest([track["track_id"], index, track["observations"][index]["observation_id"]])[:24]
                identifier = "target-" + event_id
                cached = store.read_unit(identifier)
                record = cached[0] if cached else store.save_unit(dict(unit_id=identifier, event_id=event_id),
                    dict(target=target, class_targets=layers, support=common,
                         before=previous["packed"], after=current["packed"]))
                known = int((target != 255).sum())
                events.append(dict(event_id=event_id, track_id=track["track_id"], frame=index,
                    **{k: track[k] for k in ("building_id", "view_id", "split", "bbox_xywh")},
                    previous_year=track["observations"][index - 1]["year"],
                    current_year=track["observations"][index]["year"], known_pixels=known,
                    positive_pixels=int((target == 1).sum()), negative_pixels=int((target == 0).sum()),
                    target_path=record["artifact_path"], target_sha256=record["artifact_sha256"]))
                status["evaluable" if known else "no_common_known_pixels"] += 1
        fp.verify()
        index = dict(schema_version=1, kind="canonical_history_prepared", protocol=PROTOCOL,
            history_root=str(root), history_manifest_sha256=manifest_hash, tracks=tracks, events=events,
            categories=categories, groups=groups, observations=projected, full_counts=summary,
            split=dict(mode="reviewed", development_only=False, building_assignments=assignments,
                       source="saved canonical history leakage check; selected reviewed temporal images"),
            annotation_manifest=str(source["manifest_path"]), input_sha256=fp.files,
            source_exclusions=preparation.get("exclusion_reasons", {}), options=options,
            evaluation_support="first & previous & current geometry; known annotation pixels only",
            target_scope="class membership differences of selected images; annotation-derived")
        write_json(out / "index.json", index)
        store.finish("completed", index_sha256=sha256(out / "index.json"), counts=summary,
                     coverage=dict(status), rgb_recomputed=False, encoder_run=False)
        print(f"Prepared {len(events)} candidates; coverage={dict(status)}", flush=True)
        return index
    except BaseException as exc:
        store.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        store.close()


class Dataset:
    def __init__(self, prepared, pilot=False, pilot_tracks=3):
        self.root = Path(prepared).expanduser().resolve()
        self.index = read_json(self.root / "index.json")
        record = read_json(self.root / "run.json")
        if (self.index.get("kind") != "canonical_history_prepared" or record.get("status") != "completed"
                or record.get("index_sha256") != sha256(self.root / "index.json")):
            raise ValueError("Expected a completed, hash-bound canonical-history preparation")
        self.fp = Fingerprints()
        self.fp.add(self.root / "index.json")
        self.fp.add(self.root / "run.json")
        self.history_root = Path(self.index["history_root"])
        self.categories, self.groups = self.index["categories"], self.index["groups"]
        tracks = self.index["tracks"]
        self.full_counts = self.index["full_counts"]
        if pilot:
            chosen = []
            for split in ("train", "val"):
                candidates = sorted((t for t in tracks if t["split"] == split),
                                    key=lambda t: (len(t["observations"]), t["building_id"], t["track_id"]))
                order = []
                while candidates:
                    order.append(candidates.pop(0 if len(order) % 2 == 0 else -1))
                buildings = set()
                for track in order:
                    if track["building_id"] not in buildings:
                        chosen.append(track)
                        buildings.add(track["building_id"])
                        if len(buildings) == pilot_tracks:
                            break
                ids = {t["track_id"] for t in chosen}
                selected = [t for t in chosen if t["split"] == split]
                for track in order:
                    if len(selected) >= pilot_tracks:
                        break
                    if track["track_id"] not in ids:
                        chosen.append(track)
                        selected.append(track)
            tracks = chosen
        self.tracks = {t["track_id"]: t for t in tracks}
        self.samples = [s for s in self.index["events"] if s["track_id"] in self.tracks]
        self.events = {s["event_id"]: s for s in self.samples}
        self.observations = {r["observation_id"]: r for t in tracks for r in t["observations"]}
        self.split = self.index["split"]
        self.selection = dict(pilot=pilot, tracks=sorted(self.tracks), counts=counts(tracks))
        for row in self.observations.values():
            for field in ("rgb", "support"):
                self.fp.add(_inside(self.history_root, row[field]), row[field + "_sha256"])
        for row in self.samples:
            self.fp.add(_inside(self.root, row["target_path"]), row["target_sha256"])

    @lru_cache(maxsize=24)
    def frame(self, observation_id):
        row = self.observations[observation_id]
        rgb, alpha = load_rgb(_inside(self.history_root, row["rgb"]))
        if rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8:
            raise ValueError("Invalid native RGB crop")
        support = support_image(self.history_root, row, rgb.shape[:2])
        if np.any(support & ~alpha):
            raise ValueError("Saved crop support extends into transparent pixels")
        return rgb, support

    @lru_cache(maxsize=12)
    def target(self, event_id):
        row = self.events[event_id]
        with np.load(_inside(self.root, row["target_path"]), allow_pickle=False) as arrays:
            return {k: arrays[k].copy() for k in arrays.files}
