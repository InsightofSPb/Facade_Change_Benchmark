"""Project existing COCO masks onto frozen crop histories; retain unknown pixels.

Derived targets describe annotation differences on common annotated support.
They are not reviewed temporal events or labels for all unannotated crop pixels.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import html
import importlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from ..history_features import _inside, _npz, progress
from ..io import read_json, sha256, write_json
from ..segmentation.data import rasterize_annotations
from scripts.real_eval.real_gt_metrics import derive_labels


history = importlib.import_module("facade_change.research.2026-10-25_history_models")
EXTRA_FIELDS = ("label_kind", "target_scope", "known_pixels", "changed_pixels", "stable_pixels",
                "unknown_pixels", "known_fraction", "boundary_ambiguous_pixels", "reason",
                "uncomparable_target_pixels",
                "changed_categories", "target_path", "target_sha256", "annotation_run")
SCOPE = "selected target-class differences on common annotated support; outside unknown"
PALETTE = np.array([[240, 96, 84], [67, 169, 145], [98, 135, 220], [233, 177, 69],
                    [166, 110, 204], [80, 187, 204], [228, 119, 178], [169, 171, 68],
                    [210, 136, 71], [107, 155, 109], [142, 147, 163]], dtype=np.uint8)


def resolve_path(value, parent):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else parent / path).resolve()


def annotation_source(data, override=None):
    """Follow the exact saved session to its hash-bound COCO manifest, not latest files."""
    fp = data.fingerprints
    history_path = _inside(data.history_root, data.index["history_report"]) / "manifest.json"
    document = read_json(history_path)
    if override:
        manifest_path = Path(override).expanduser().resolve()
        fp.add(manifest_path)
    else:
        diagnostics = document.get("alignment_diagnostics_root")
        if not diagnostics:
            raise ValueError("History snapshot lacks the annotation source link; pass --annotation-manifest")
        inputs_path = Path(diagnostics).parent / "inputs.json"
        expected = document.get("input_sha256", {}).get(str(inputs_path.resolve()))
        if not expected:
            raise ValueError("History snapshot lacks the session input hash; pass --annotation-manifest")
        fp.add(inputs_path, expected)
        inputs = read_json(inputs_path)
        manifest_path = resolve_path(inputs["manifest"], inputs_path.parent)
        fp.add(manifest_path, inputs["manifest_sha256"])
    manifest = read_json(manifest_path)
    source = manifest.get("source", {})
    if not source.get("coco_json") or not source.get("coco_sha256"):
        raise ValueError("Annotation manifest needs source.coco_json and source.coco_sha256; "
                         "use the reviewed temporal/semantic manifest, not a bare COCO file")
    coco_path = resolve_path(source["coco_json"], manifest_path.parent)
    fp.add(coco_path, source["coco_sha256"])
    coco = read_json(coco_path)
    categories = sorted(coco["categories"], key=lambda c: c["id"])
    if (not 1 <= len(categories) <= 16 or len({c["id"] for c in categories}) != len(categories)
            or len({c["name"] for c in categories}) != len(categories)
            or any(type(c["id"]) is not int or c["id"] < 0 for c in categories)):
        raise ValueError("Expected 1–16 uniquely named categories with nonnegative integer IDs")
    if {(c["id"], c["name"]) for c in manifest["categories"]} != {(c["id"], c["name"]) for c in categories}:
        raise ValueError("COCO categories differ from the annotation manifest")
    channel_map = {c["id"]: i for i, c in enumerate(categories)}
    coco_images = {str(row["id"]): row for row in coco["images"]}
    if len(coco_images) != len(coco["images"]):
        raise ValueError("Duplicate COCO image ID")
    by_hash, annotations, ids = {}, defaultdict(list), set()
    for row in manifest["images"]:
        if not row.get("sha256"):
            continue
        if row["sha256"] in by_hash:
            raise ValueError("Ambiguous duplicate image SHA in annotation manifest")
        by_hash[row["sha256"]] = row
    for annotation in coco.get("annotations", []):
        if annotation["id"] in ids:
            raise ValueError("Duplicate COCO annotation ID")
        ids.add(annotation["id"])
        if (str(annotation["image_id"]) not in coco_images
                or annotation["category_id"] not in channel_map):
            raise ValueError("Annotation references an unknown image or category")
        annotations[str(annotation["image_id"])].append(annotation)
    return {"manifest_path": manifest_path, "coco_path": coco_path, "categories": categories,
            "channel_map": channel_map, "images": by_hash, "coco_images": coco_images,
            "annotations": annotations, "history_document": document}


def project_masks(packed, matrix, bbox, support):
    """Use the RGB crop's source-to-anchor H and rectangle; discrete nearest sampling."""
    import cv2
    x, y, width, height = bbox
    matrix = np.asarray(matrix, dtype=np.float64)
    if (packed.dtype != np.uint16 or packed.ndim != 2 or matrix.shape != (3, 3)
            or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12
            or support.dtype != bool or support.shape != (height, width)):
        raise ValueError("Invalid annotation raster, geometry, rectangle or crop support")
    shift = np.array([[1., 0, -x], [0, 1., -y], [0, 0, 1.]])
    result = cv2.warpPerspective(packed, shift @ matrix, (width, height), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    result[~support] = 0
    return result


def year_consensus(observations):
    """All alternatives must annotate and agree; disagreement is unknown, never a new date."""
    first = observations[0]
    support = np.logical_and.reduce([row["support"] for row in observations])
    known = support.copy()
    for row in observations:
        known &= (row["packed"] != 0) & (row["packed"] == first["packed"])
    packed = np.where(known, first["packed"], 0).astype(np.uint16)
    return {"packed": packed, "support": support,
            "possible_targets": np.bitwise_or.reduce([row["packed"] for row in observations])}


def target_difference(before, after, channels, radius):
    """Adapt the existing annotation-change evaluator to selected target channels."""
    support = before["support"] & after["support"]
    old, new = before["packed"], after["packed"]
    layers = []
    for channel in channels:
        # Bit 0 retains known annotation support even when this target is absent.
        # Bit 1 represents the selected target. Original bitsets are never collapsed.
        a = (old != 0).astype(np.uint16) | (((old & (1 << channel)) != 0).astype(np.uint16) << 1)
        b = (new != 0).astype(np.uint16) | (((new & (1 << channel)) != 0).astype(np.uint16) << 1)
        layers.append(derive_labels(a, b, support, 2, radius))
    stack = np.stack(layers)
    target = np.full(support.shape, 255, dtype=np.uint8)
    target[np.all(stack == 0, axis=0)] = 0
    target[np.any(stack == 1, axis=0)] = 1
    known = support & (old != 0) & (new != 0)
    selected_bits = sum(1 << channel for channel in channels)
    raw_changed = ((old & selected_bits) != (new & selected_bits)) & known
    ambiguous = raw_changed & (target == 255)
    return target, stack, support, ambiguous


def event_target(before, after, channels, radius=1, min_known_pixels=64,
                 min_known_fraction=.001, min_changed_pixels=8):
    target, layers, support, ambiguous = target_difference(before, after, channels, radius)
    known = int((target != 255).sum())
    changed = int((target == 1).sum())
    geometric = int(support.sum())
    fraction = known / geometric if geometric else 0.0
    selected_bits = sum(1 << channel for channel in channels)
    possible = before.get("possible_targets", before["packed"]) | after.get("possible_targets", after["packed"])
    uncomparable = int(((target == 255) & ((possible & selected_bits) != 0)).sum())
    result = {"label": "", "review_status": "unreviewed", "comparable": "unknown",
              "known_pixels": known, "changed_pixels": changed,
              "stable_pixels": int((target == 0).sum()), "unknown_pixels": int((target == 255).sum()),
              "known_fraction": fraction, "boundary_ambiguous_pixels": int(ambiguous.sum()),
              "uncomparable_target_pixels": uncomparable,
              "target_scope": SCOPE, "label_kind": "annotation_derived",
              "event_type": "", "reason": ""}
    if not known:
        result["reason"] = "no_common_annotation_support"
    elif known < min_known_pixels or fraction < min_known_fraction:
        result["reason"] = "insufficient_common_annotation_support"
    elif changed >= min_changed_pixels:
        result.update(label=1, review_status="derived", comparable="annotation_support_only",
                      event_type="annotation_change", reason="target_difference_on_annotated_support")
    elif changed or ambiguous.any():
        result["reason"] = "tiny_or_boundary_ambiguous_difference"
    elif uncomparable:
        result["reason"] = "target_annotations_without_comparable_support"
    else:
        result.update(label=0, review_status="derived", comparable="annotation_support_only",
                      event_type="annotation_stable", reason="stable_on_annotated_support_only")
    return result, target, layers


def _thumbnail(array):
    image = Image.fromarray(array)
    image.thumbnail((128, 128))
    stream = io.BytesIO()
    image.save(stream, "PNG")
    return '<img loading="lazy" src="data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode() + '">'


def mask_rgb(packed, categories):
    colors = np.full((*packed.shape, 3), 115, dtype=np.uint8)
    # Blend overlaps for display only; stored bitsets preserve every label.
    total, count = np.zeros_like(colors, dtype=np.float32), np.zeros(packed.shape, dtype=np.float32)
    for i in range(len(categories)):
        mask = (packed & (1 << i)) != 0
        total[mask] += PALETTE[i % len(PALETTE)]
        count[mask] += 1
    colors[count > 0] = (total[count > 0] / count[count > 0, None]).astype(np.uint8)
    return colors


def render_audit(data, source, rows, store, observations, limit):
    # Deterministic coverage across buildings and positive/stable/unknown states.
    groups = defaultdict(list)
    for row in rows:
        groups[(row["reason"], row["building_id"])].append(row)
    selected = []
    while groups and len(selected) < limit:
        for key in sorted(list(groups)):
            selected.append(groups[key].pop(0))
            if not groups[key]:
                del groups[key]
            if len(selected) >= limit:
                break
    parts = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Маски историй</title>',
             '<style>body{font:16px system-ui;margin:24px;background:#f2f4f7}section{background:white;'
             'padding:18px;margin:18px 0}.frames{display:flex;flex-wrap:wrap;gap:12px}'
             'figure{margin:8px}figcaption{max-width:260px}code{overflow-wrap:anywhere}</style>',
             '<h1>Перенос целевых масок на истории</h1><p>RGB сохранён целиком. Серое на масках — '
             'неизвестно, а не здоровый фасад. На карте разницы: красное — различие целевых меток, '
             'зелёное — совпадение на размеченной области, серое — нет основания для сравнения. '
             'Автоматические метки не подтверждают физическое событие.</p>',
             f'<p>Показано {len(selected)} из {len(rows)} событий с распределением по зданиям и статусам.</p>']
    legend = " · ".join(f'<span style="color:rgb{tuple(int(v) for v in PALETTE[i % len(PALETTE)])}">'
                        f'{html.escape(c["name"])}</span>' for i, c in enumerate(source["categories"]))
    parts.append('<p>' + legend + '</p>')
    for row in selected:
        parts.append(f'<section><h2>{html.escape(row["building_id"])}: {row["previous_year"]} → '
                     f'{row["current_year"]}</h2><code>{row["event_id"]}</code><p>'
                     f'{html.escape(row["reason"])} · label={row["label"]} · '
                     f'известно {row["known_fraction"]:.1%}</p><div class="frames">')
        for observation in data.tracks[row["track_id"]]["observations"]:
            if observation["year"] > row["current_year"]:
                continue
            rgb_path = _inside(data.history_root, observation["rgb"])
            data.fingerprints.add(rgb_path, observation["rgb_sha256"])
            with Image.open(rgb_path) as image:
                rgb = np.asarray(image.convert("RGB"))
            with np.load(store.out / observations[observation["observation_id"]]["artifact_path"], allow_pickle=False) as arrays:
                colored = mask_rgb(arrays["packed"], source["categories"])
            parts.append('<figure>' + _thumbnail(rgb) + _thumbnail(colored)
                         + f'<figcaption>{observation["year"]}: RGB / целевые маски</figcaption></figure>')
        with np.load(store.out / row["target_path"], allow_pickle=False) as arrays:
            target = arrays["target"]
        color = np.full((*target.shape, 3), 115, dtype=np.uint8)
        color[target == 1], color[target == 0] = [228, 82, 77], [75, 165, 130]
        parts.append('<figure>' + _thumbnail(color) + '<figcaption>Разница последних двух лет</figcaption></figure>')
        parts.append('</div></section>')
    parts.append('</html>')
    (store.out / "gallery.html").write_text("".join(parts), encoding="utf-8")


def export_targets(args):
    data = history.load_histories(args.features, args.splits, args.min_years, args.history_root)
    source = annotation_source(data, args.annotation_manifest)
    categories = source["categories"]
    names = {c["name"] for c in categories}
    if args.categories and (set(args.categories) - names or len(set(args.categories)) != len(args.categories)):
        raise ValueError(f"Unknown/duplicate target categories; available: {sorted(names)}")
    channels = [i for i, c in enumerate(categories) if not args.categories or c["name"] in args.categories]
    used_tracks = {s["track_id"] for s in data.samples}
    dimensions = {frame["image"]["sha256"]: (frame["image"]["height"], frame["image"]["width"])
                  for view in source["history_document"]["views"] for frame in view["frames"]}
    jobs = defaultdict(list)
    for key in sorted(used_tracks):
        track = data.tracks[key]
        for observation in track["observations"]:
            jobs[observation["image_sha256"]].append((track, observation))
    # Check identities before any rasterization; no holdout pixels are decoded.
    for image_hash, items in jobs.items():
        image = source["images"].get(image_hash)
        if image is None:
            continue
        coco_image = source["coco_images"].get(str(image["image_id"]))
        if coco_image is None or any(coco_image[k] != image[k] for k in ("height", "width")):
            raise ValueError("COCO image dimensions disagree with annotation manifest")
        if dimensions.get(image_hash) != (image["height"], image["width"]):
            raise ValueError("Annotation and history source image dimensions disagree")
        for track, observation in items:
            if any(image.get(k) != track[k] for k in ("building_id", "view_id", "split")) or image.get("year") != observation["year"]:
                raise ValueError("Annotation/history building, view, split or year mismatch")
    out = Path(args.out).expanduser().resolve()
    if args.resume and (out / "run.json").exists() and read_json(out / "run.json").get("status") == "completed":
        raise ValueError("Annotation export already completed; use a new output directory")
    store = history.HistoryRun(out, {k: v for k, v in vars(args).items() if k != "resume"}, resume=args.resume)
    observations, status_counts = {}, Counter()
    try:
        import cv2
        raster_module = importlib.import_module("facade_change.segmentation.data")
        evaluator = importlib.import_module("scripts.real_eval.real_gt_metrics")
        contract = {"inputs": dict(data.fingerprints.files), "module_sha256": sha256(__file__),
                    "rasterizer_sha256": sha256(raster_module.__file__),
                    "evaluator_sha256": sha256(evaluator.__file__), "cv2": cv2.__version__,
                    "categories": categories, "target_channels": channels, "target_scope": SCOPE,
                    "category_to_channel": [{"category_id": c["id"], "channel": i} for i, c in enumerate(categories)],
                    "unknown_policy": "unannotated, unsupported, crowd and conflicting alternatives remain unknown",
                    "negative_policy": "same current multilabel contract: absent class is negative only inside annotated union",
                    "event_negative_policy": "no known change, no ambiguous difference, no annotated target outside comparable support",
                    "rgb_recomputed": False, "dino_recomputed": False, "reviewed_event_gt_created": False}
        store.contract(contract)
        for image_hash, items in progress(sorted(jobs.items()), "Project source annotations"):
            image = source["images"].get(image_hash)
            if image is not None:
                data.fingerprints.add(items[0][1]["original_image_path"], image_hash)
            packed = None
            for track, observation in items:
                unit_id = "annotation-" + observation["observation_id"]
                support_path = _inside(data.history_root, observation["support"])
                data.fingerprints.add(support_path, observation["support_sha256"])
                cached = store.read_unit(unit_id)
                if cached is not None:
                    record, arrays = cached
                    if arrays["packed"].dtype != np.uint16 or arrays["support"].dtype != bool:
                        raise ValueError("Invalid cached projected annotations")
                    observations[observation["observation_id"]] = record
                    status_counts["reused_observations"] += 1
                    continue
                with Image.open(support_path) as support_image:
                    raw = np.asarray(support_image)
                width, height = track["bbox_xywh"][2:]
                if raw.dtype != np.uint8 or raw.shape != (height, width) or not np.isin(raw, [0, 255]).all():
                    raise ValueError("Invalid saved crop support")
                support = raw == 255
                if image is None:
                    labels = np.zeros(support.shape, dtype=np.uint16)
                    state = "missing_annotation_image"
                else:
                    if packed is None:
                        packed = rasterize_annotations(source["annotations"].get(str(image["image_id"]), []),
                                                       image["height"], image["width"], source["channel_map"])
                        status_counts["rasterized_images"] += 1
                    labels = project_masks(packed, observation["alignment"]["matrix"], track["bbox_xywh"], support)
                    state = "projected" if labels.any() else "unannotated_crop"
                record = {"unit_id": unit_id, "observation_id": observation["observation_id"],
                          "track_id": track["track_id"], "image_sha256": image_hash,
                          "source_annotation_image_id": image["image_id"] if image else None,
                          "status": state, "known_pixels": int((labels != 0).sum())}
                saved = store.save_unit(record, {"packed": labels, "support": support})
                observations[observation["observation_id"]] = saved
                status_counts["computed_observations"] += 1
        rows, class_counts = [], Counter()
        by_track = defaultdict(list)
        for sample in data.samples:
            by_track[sample["track_id"]].append(sample)
        for key in progress(sorted(by_track), "Derive target-class differences"):
            years = defaultdict(list)
            for observation in data.tracks[key]["observations"]:
                with np.load(store.out / observations[observation["observation_id"]]["artifact_path"], allow_pickle=False) as arrays:
                    years[observation["year"]].append({k: arrays[k].copy() for k in ("packed", "support")})
            consensus = {year: year_consensus(arrays) for year, arrays in years.items()}
            for sample in by_track[key]:
                result, target, layers = event_target(consensus[sample["previous_year"]], consensus[sample["current_year"]],
                    channels, args.boundary_tolerance, args.min_known_pixels, args.min_known_fraction, args.min_changed_pixels)
                changed_names = [categories[channel]["name"] for channel, layer in zip(channels, layers) if np.any(layer == 1)]
                class_counts.update(changed_names)
                target_path = store.out / "targets" / f"{sample['event_id']}.npz"
                _npz(target_path, {"target": target, "class_targets": layers,
                                    "category_ids": np.array([categories[i]["id"] for i in channels])})
                rows.append({**{k: sample[k] for k in history.LABEL_FIELDS if k in sample}, **result,
                             "history_start_year": int(sample["years"][0]),
                             "review_source": f"annotation-derived:{store.unique_id}:{sample['event_id']}",
                             "changed_categories": "|".join(changed_names),
                             "target_path": target_path.relative_to(store.out).as_posix(),
                             "target_sha256": sha256(target_path), "annotation_run": str(store.out),
                             "notes": "Derived annotation difference, not reviewed event truth; RGB occluders retained"})
        history.write_csv(store.out / "labels.csv", rows, (*history.LABEL_FIELDS, *EXTRA_FIELDS))
        write_json(store.out / "index.json", {"schema_version": 1, "kind": "history_annotation_targets",
                    "feature_index_sha256": sha256(data.index_path), "contract": contract,
                    "observations": observations, "events": rows, "input_sha256": data.fingerprints.files})
        render_audit(data, source, rows, store, observations, args.gallery_limit)
        summary = {"kind": "history_annotation_targets", "scope": SCOPE, "features": data.coverage,
                   "events": len(rows), "derived_positive": sum(r["label"] == 1 for r in rows),
                   "derived_stable": sum(r["label"] == 0 for r in rows),
                   "unknown": sum(r["label"] == "" for r in rows),
                   "reasons": dict(Counter(r["reason"] for r in rows)), "changes_by_category": dict(class_counts),
                   "observations": dict(Counter(r["status"] for r in observations.values())),
                   "processing": dict(status_counts), "reviewed_event_gt_created": False,
                   "unknown_pixels_are_negative": False, "occluder_categories_added": False,
                   "by_split": {split: dict(Counter(str(r["label"]) if r["label"] != "" else "unknown"
                                                   for r in rows if r["split"] == split)) for split in args.splits}}
        history.finish_run(store, data, summary)
        return {**summary, "out": str(store.out)}
    except BaseException as exc:
        store.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        store.close()


def parser():
    cli = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    cli.add_argument("--features", required=True)
    cli.add_argument("--history-root")
    cli.add_argument("--annotation-manifest", help="Override with a reviewed manifest linked to exact COCO bytes")
    cli.add_argument("--out", required=True)
    cli.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val"])
    cli.add_argument("--min-years", type=int, default=3)
    cli.add_argument("--categories", nargs="+", help="Exact COCO target names; default: every annotated category")
    cli.add_argument("--boundary-tolerance", type=int, default=1, help="Canonical pixels; ambiguous differences are ignored")
    cli.add_argument("--min-known-pixels", type=int, default=64)
    cli.add_argument("--min-known-fraction", type=float, default=.001)
    cli.add_argument("--min-changed-pixels", type=int, default=8)
    cli.add_argument("--gallery-limit", type=int, default=60)
    cli.add_argument("--resume", action="store_true")
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if (args.min_years < 2 or args.boundary_tolerance < 0 or args.min_known_pixels < 1
            or not 0 <= args.min_known_fraction <= 1 or args.min_changed_pixels < 1
            or args.gallery_limit < 0):
        cli.error("Invalid history length, tolerance or support thresholds")
    try:
        print(json.dumps(export_targets(args), ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"history targets failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
