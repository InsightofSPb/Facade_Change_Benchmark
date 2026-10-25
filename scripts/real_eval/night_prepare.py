"""CPU audit and GT-enriched real chronologies for frozen-model evaluation.

Audit opens metadata only. Preparation explicitly rasterizes annotations in the
requested splits, including TEST, for selection/evaluation; scorers receive only
RGB and geometric support. No alignment or model inference is performed here.
"""
from __future__ import annotations

import base64
import csv
import html
import importlib
import io
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from facade_change.derived import _starts
from facade_change.geometry import expanded_canvas
from facade_change.segmentation.data import PreparedData
from .real_gt_metrics import derive_labels
from .robustness_common import atomic_json, atomic_npz, digest, read_json, sha256
from .robustness_prepare import COLORS, _validate_category_mapping


def _csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _new_output(value, inputs):
    out = Path(value).expanduser().resolve()
    if out.exists() or any(out == parent or parent in out.parents or out in parent.parents
                           for parent in inputs):
        raise FileExistsError("Нужен новый отдельный каталог вне исходных запусков")
    return out


def _context(run_path, snapshot_path):
    run, snapshot = (Path(value).expanduser().resolve() for value in (run_path, snapshot_path))
    paths = [run / "resolved_config.json", run / "experiment_contract.json",
             snapshot / "2026-10-17_source_units.json", snapshot / "2026-10-17_quick_summary.json"]
    hashes = {str(path): sha256(path) for path in paths}
    config, contract, records, summary = (read_json(path) for path in paths)
    if contract.get("config") != config:
        raise ValueError("Исходная конфигурация отличается от замороженного контракта")
    if Path(summary["source_run"]).expanduser().resolve() != run:
        raise ValueError("Снимок относится к другому исходному запуску")
    if not isinstance(records, list):
        raise ValueError("Ожидался список единиц исходного снимка")
    return dict(source_run=str(run), snapshot=str(snapshot), source_config=config,
                source_contract=contract, source_records=records, source_summary=summary,
                metadata_sha256=hashes)


def _discover(root):
    """Inspect only run.json at one/two levels; never recurse into image trees."""
    if root is None:
        return [], []
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Не найден каталог запусков: {root}")
    accepted, excluded = [], []
    for path in sorted(set(root.glob("*/run.json")) | set(root.glob("*/*/run.json"))):
        try:
            row = read_json(path)
        except (ValueError, OSError) as error:
            excluded.append(dict(path=str(path.parent), reason="unreadable_run_metadata", error=str(error)))
            continue
        if row.get("kind") not in {"pair_alignment", "alignment_and_data_batch"}:
            continue
        status = str(row.get("status", ""))
        if status.startswith("completed") or status in {"failed", "interrupted"}:
            accepted.append(str(path.parent))
        else:
            excluded.append(dict(path=str(path.parent), reason="alignment_still_running", status=status))
    return accepted, excluded


def _parents(images, pairs, alignments, fingerprints, splits, strict_alignment_paths=()):
    """Same first-anchor validation as temporal_inputs._real, with explicit splits."""
    inputs = importlib.import_module("facade_change.research.2026-10-14_temporal_inputs")
    keys = {(str(row["reference_id"]), str(row["source_id"])) for row in pairs}
    groups, excluded = defaultdict(list), []
    for row in images:
        if row.get("split") not in splits:
            continue
        if (row.get("metadata_status") != "reviewed" or not row.get("building_id")
                or not row.get("view_id") or row.get("year") is None
                or row.get("temporal_eligible") is False):
            excluded.append(dict(image_id=row["image_id"], reason="ineligible_temporal_metadata"))
            continue
        groups[(row["split"], row["building_id"], row["view_id"])].append(row)
    sequences, coverage = [], []
    for key, observations in sorted(groups.items()):
        observations.sort(key=lambda row: (int(row["year"]), str(row["image_id"])))
        report = dict(split=key[0], building_id=key[1], view_id=key[2],
                      eligible_observations=len(observations), years=[row["year"] for row in observations], pairs=[])
        coverage.append(report)
        if len(observations) < 2:
            report["status"] = "fewer_than_two_observations"
            continue
        if len({int(row["year"]) for row in observations}) != len(observations):
            report["status"] = "ambiguous_same_year_observations"
            continue
        reference = observations[0]
        frames = [{**reference, "source_to_reference": np.eye(3).tolist(), "frame": 0}]
        for source in observations[1:]:
            pair = dict(reference_id=reference["image_id"], source_id=source["image_id"],
                        reference_year=reference["year"], source_year=source["year"])
            report["pairs"].append(pair)
            if (str(reference["image_id"]), str(source["image_id"])) not in keys:
                pair["status"] = "missing_prepared_first_anchor_pair"
                continue
            matches = alignments.get((reference["sha256"], source["sha256"]), [])
            pair["attempts"] = [dict(path=str(match["path"]), status=match["record"].get("status"),
                                      method=match["method"]) for match in matches]
            compatible = []
            for match, attempt in zip(matches, pair["attempts"]):
                mismatch = None
                for role, current in (("reference", reference), ("source", source)):
                    previous = match["record"]["observations"][role]
                    for field in ("split", "building_id", "view_id", "year", "width", "height", "metadata_status"):
                        if previous.get(field) != current.get(field):
                            mismatch = f"Alignment/current dataset {role}.{field} mismatch: {match['path']}"
                            break
                    if mismatch:
                        break
                if mismatch:
                    if match.get("selection") or any(match["path"] == root or root in match["path"].parents for root in strict_alignment_paths):
                        raise ValueError(mismatch)
                    attempt.update(reason="incompatible_current_metadata", error=mismatch)
                else:
                    compatible.append(match)
            accepted = [match for match in compatible if
                        str(match["record"].get("status", "")).startswith("completed")
                        and (match.get("selection") or match["record"].get("quality_gate", {}).get("passed") is True)]
            if not accepted:
                pair["status"] = "rejected_or_failed_alignment" if matches else "missing_alignment"
                continue
            chosen, frame_info = None, None
            for candidate in accepted:
                try:
                    from facade_change.saved_geometry import frame_geometry
                    candidate_info = frame_geometry(candidate, fingerprints)
                except (ValueError, KeyError, TypeError, OSError) as error:
                    if candidate.get("selection") or any(candidate["path"] == root or root in candidate["path"].parents for root in strict_alignment_paths):
                        raise
                    next(attempt for attempt in pair["attempts"] if attempt["path"] == str(candidate["path"])).update(
                        reason="invalid_discovered_geometry", error=f"{type(error).__name__}: {error}")
                    continue
                chosen, frame_info = candidate, candidate_info
                break
            if chosen is None:
                pair["status"] = "invalid_alignment_geometry"
                continue
            frames.append({**source, **frame_info, "frame": len(frames)})
            pair.update(status="accepted", selected_alignment=str(chosen["path"]))
        report["usable_observations"] = len(frames)
        if len(frames) < 2:
            report["status"] = "no_accepted_source"
            continue
        missing_rgb = [str(frame["image_path"]) for frame in frames if not Path(frame["image_path"]).is_file()]
        if missing_rgb:
            report.update(status="missing_rgb_files", missing_rgb_files=missing_rgb)
            continue
        report["status"] = "eligible_for_geometric_crops"
        # Preserve the original temporal module's JSON serialization exactly.
        sequence_id = "real-" + inputs.digest([key, [frame["sha256"] for frame in frames]])[:24]
        missing = [row["year"] for row in observations if row["image_id"] not in {f["image_id"] for f in frames}]
        sequences.append(dict(sequence_id=sequence_id, domain="real", split=key[0], building_id=key[1],
                              view_id=key[2], arm=None, frames=frames,
                              shape=[reference["height"], reference["width"]],
                              missing_dates=len(missing), missing_years=missing,
                              label_scope="changes of annotated class sets; GT-enriched selection"))
    return sequences, coverage, excluded


def audit(run_path, snapshot_path, out_path, alignment_runs=None, runs_root=None,
          splits=("train", "val", "test"), dataset=None, alignment_session=None, quality_report=None):
    """Freeze metadata/geometry and expose missing coverage without decoding RGB/GT."""
    splits = tuple(splits)
    if not splits or len(set(splits)) != len(splits) or not set(splits) <= {"train", "val", "test"}:
        raise ValueError("Нужны различные split из train, val, test")
    if quality_report is not None and alignment_session is None:
        raise ValueError("--quality-report requires --alignment-session")
    context = _context(run_path, snapshot_path)
    out = _new_output(out_path, [Path(context[key]) for key in ("source_run", "snapshot")])
    inputs = importlib.import_module("facade_change.research.2026-10-14_temporal_inputs")
    fingerprints = dict(context["metadata_sha256"])
    dataset_path = Path(dataset or context["source_config"]["dataset"]).expanduser().resolve()
    dataset_path = dataset_path / "dataset.json" if dataset_path.is_dir() else dataset_path
    images, pairs, split = inputs._dataset(dataset_path, fingerprints)
    if alignment_session is not None:
        if alignment_runs is not None or runs_root is not None:
            raise ValueError("--alignment-session cannot be mixed with alignment discovery or --alignment-runs")
        from facade_change.alignment_session import resolve_session
        alignment_runs = [str(resolve_session(alignment_session))]
    discovered, excluded_runs = _discover(runs_root)
    configured = context["source_config"].get("alignment_runs", []) if alignment_runs is None else alignment_runs
    configured = list(dict.fromkeys(str(Path(path).expanduser().resolve()) for path in configured))
    paths = list(dict.fromkeys([*configured, *discovered]))
    alignments, attempts, loaded = defaultdict(list), [], []
    for path in paths:
        try:
            found, run_attempts = inputs._alignments([path], fingerprints, quality_report=quality_report)
        except (ValueError, KeyError, TypeError, OSError) as error:
            if path in configured:
                raise
            excluded_runs.append(dict(path=path, reason="invalid_discovered_alignment", error=f"{type(error).__name__}: {error}"))
            continue
        loaded.append(path)
        attempts.extend(run_attempts)
        for key, matches in found.items():
            present = {match["path"] for match in alignments[key]}
            alignments[key].extend(match for match in matches if match["path"] not in present)
    sequences, views, excluded_images = _parents(images, pairs, alignments, fingerprints, splits,
                                                  tuple(Path(path) for path in configured))
    inventory = []
    by_building = defaultdict(list)
    for row in images:
        if row.get("split") in splits and row.get("building_id"):
            by_building[(row["split"], row["building_id"])].append(row)
    for (part, building), rows in sorted(by_building.items()):
        ready = [s for s in sequences if s["building_id"] == building]
        inventory.append(dict(split=part, building_id=building, images=len(rows),
                              years=";".join(map(str, sorted({r['year'] for r in rows if r.get('year') is not None}))),
                              views=len({r.get("view_id") for r in rows if r.get("view_id")}),
                              accepted_views=len(ready), accepted_pairs=sum(len(s["frames"])-1 for s in ready)))
    pair_status_counts = dict(Counter(pair["status"] for view in views for pair in view["pairs"]))
    coverage = dict(facades=len(inventory), eligible_facades=len({s["building_id"] for s in sequences}),
                    history_eligible_facades=len({s["building_id"] for s in sequences if len(s["frames"]) >= 3}),
                    accepted_views=len(sequences), accepted_pairs=sum(len(s["frames"])-1 for s in sequences),
                    first_anchor_pairs_needed=sum(len(view["pairs"]) for view in views),
                    pair_status_counts=pair_status_counts,
                    by_split={part: dict(facades=sum(row["split"] == part for row in inventory),
                                        eligible_facades=len({s["building_id"] for s in sequences if s["split"] == part}),
                                        history_eligible_facades=len({s["building_id"] for s in sequences
                                                                      if s["split"] == part and len(s["frames"]) >= 3}),
                                        accepted_pairs=sum(len(s["frames"])-1 for s in sequences if s["split"] == part))
                              for part in splits}, views=views, alignment_attempts=attempts,
                    excluded_alignment_runs=excluded_runs, excluded_images=excluded_images)
    document = dict(schema_version=1, kind="real_temporal_audit", **context,
                    dataset=str(dataset_path), requested_splits=list(splits), alignment_runs=loaded,
                    dataset_split=split, sequences=sequences, inventory=inventory, coverage=coverage,
                    rgb_loaded=False, annotations_rasterized=False)
    document["metadata_sha256"] = fingerprints
    document["audit_sha256"] = digest(document)
    out.mkdir(parents=True)
    atomic_json(out / "audit.json", document)
    atomic_json(out / "run.json", dict(schema_version=1, kind="real_temporal_audit", status="completed",
                                      audit_sha256=sha256(out / "audit.json"), coverage=coverage))
    _csv(out / "facades.csv", inventory)
    _csv(out / "alignment_coverage.csv", [dict(split=view["split"], building_id=view["building_id"],
                                               view_id=view["view_id"], **{k: v for k, v in pair.items() if k != "attempts"})
                                          for view in views for pair in view["pairs"]])
    print(f"Аудит: фасадов {coverage['facades']}; с пригодными историями {coverage['eligible_facades']}; "
          f"видов {len(sequences)}; последовательных сравнений {coverage['accepted_pairs']}.", flush=True)
    for part, counts in coverage["by_split"].items():
        print(f"  {part}: всего {counts['facades']} фасадов → {counts['eligible_facades']} с ≥2 датами "
              f"→ {counts['history_eligible_facades']} с ≥3 датами; {counts['accepted_pairs']} пар", flush=True)
    print(f"First-anchor: требуется {coverage['first_anchor_pairs_needed']} пар; "
          + "; ".join(f"{status}: {count}" for status, count in sorted(pair_status_counts.items())), flush=True)
    blockers = Counter(row["status"] for row in views if row["status"] != "eligible_for_geometric_crops")
    if blockers or excluded_runs:
        print("Блокеры видов: " + str(dict(blockers)) + f"; исключённых alignment-запусков {len(excluded_runs)}", flush=True)
    print(f"Метаданные: {out / 'audit.json'}; покрытие: {out / 'alignment_coverage.csv'}", flush=True)
    return document


def _verify_hashes(hashes):
    for path, expected in tqdm(hashes.items(), desc="Verify real history inputs", unit="file", dynamic_ncols=True):
        if sha256(path) != expected:
            raise ValueError(f"Изменился вход после аудита: {path}")


def _warp(data, sequence, frame, crop=None, include_rgb=False):
    import cv2
    current = data.images.get(frame["image_id"])
    if current is None or any(current.get(key) != frame.get(key)
                              for key in ("sha256", "building_id", "view_id", "year", "width", "height")):
        raise ValueError("COCO-кадр не соответствует замороженной хронологии")
    rgb, packed, opaque = data.load_image(frame["image_id"])
    height, width = sequence["shape"]
    x, y, width, height = crop if crop is not None else (0, 0, width, height)
    from facade_change.saved_geometry import FrameSampler
    sampler = FrameSampler(frame)
    box = (x, y, width, height)
    support = sampler.support(opaque, box)
    bits = sampler.warp(packed, box, nearest=True)
    bits[~support] = 0
    result = dict(support=support, packed=bits)
    if include_rgb:
        result["rgb"] = sampler.warp(rgb, box)
    return result


def _rank_view(data, sequence, tile_size, stride, min_support, min_known_pixels):
    height, width = sequence["shape"]
    changed_union = np.zeros((height, width), bool)
    known_union = np.zeros_like(changed_union)
    common_support = np.ones_like(changed_union)
    temporal_counts = np.zeros((height, width), np.uint32)
    previous = None
    for frame in sequence["frames"]:
        current = _warp(data, sequence, frame)
        common_support &= current["support"]
        if previous is not None:
            labels = derive_labels(previous["packed"], current["packed"],
                                   previous["support"] & current["support"], len(data.categories))
            changed_union |= labels == 1
            known_union |= labels != 255
            temporal_counts += labels == 1
        previous = current
    # Ranking excludes pixels never jointly supported by the complete chronology.
    changed_union &= common_support
    known_union &= common_support
    temporal_counts[~common_support] = 0
    candidates = []
    crop_h, crop_w = min(tile_size, height), min(tile_size, width)
    for y in _starts(height, tile_size, stride):
        for x in _starts(width, tile_size, stride):
            area = np.s_[y:y+crop_h, x:x+crop_w]
            support_fraction = float(common_support[area].mean())
            known, changed = int(known_union[area].sum()), int(changed_union[area].sum())
            reason = ("low_common_support" if support_fraction < min_support else
                      "too_few_known_pixels" if known < min_known_pixels else
                      "no_annotated_change" if changed == 0 else None)
            candidates.append(dict(sequence_id=sequence["sequence_id"] + f"-x{x}-y{y}",
                                   parent_sequence_id=sequence["sequence_id"], building_id=sequence["building_id"],
                                   view_id=sequence["view_id"], split=sequence["split"], crop_xywh=[x, y, crop_w, crop_h],
                                   frames=len(sequence["frames"]), previous_pairs=len(sequence["frames"])-1,
                                   common_support_fraction=support_fraction, known_union_pixels=known,
                                   changed_union_pixels=changed,
                                   temporal_changed_pixel_observations=int(temporal_counts[area].sum()),
                                   changed_known_percent=100 * changed / known if known else None,
                                   changed_crop_percent=100 * changed / (crop_w * crop_h),
                                   status="eligible" if reason is None else "excluded", exclusion_reason=reason,
                                   selected=False))
    view_stats = dict(parent_sequence_id=sequence["sequence_id"], building_id=sequence["building_id"],
                      view_id=sequence["view_id"], split=sequence["split"],
                      changed_union_pixels=int(changed_union.sum()), known_union_pixels=int(known_union.sum()),
                      temporal_changed_pixel_observations=int(temporal_counts.sum()),
                      candidate_crops=len(candidates), eligible_crops=sum(c["status"] == "eligible" for c in candidates))
    return candidates, view_stats


def _iou(a, b):
    x, y, w, h = a
    xx, yy, ww, hh = b
    overlap = max(0, min(x+w, xx+ww)-max(x, xx)) * max(0, min(y+h, yy+hh)-max(y, yy))
    return overlap / (w*h + ww*hh - overlap)


def _select(candidates, views, max_facades, crops_per_facade, buildings):
    eligible = [row for row in candidates if row["status"] == "eligible"]
    allowed = None if buildings is None else set(buildings)
    if allowed is not None:
        unknown = allowed - {row["building_id"] for row in candidates}
        if unknown:
            raise ValueError("Не найдены запрошенные фасады: " + ", ".join(sorted(unknown)))
        eligible = [row for row in eligible if row["building_id"] in allowed]
    facade_rows = []
    for building in sorted({row["building_id"] for row in eligible}):
        relevant = [row for row in views if row["building_id"] == building and row["eligible_crops"]]
        best_view = max(relevant, key=lambda row: (row["changed_union_pixels"], row["known_union_pixels"]))
        facade_rows.append(dict(building_id=building, split=best_view["split"],
                                ranking_changed_union_pixels=best_view["changed_union_pixels"],
                                ranking_view_id=best_view["view_id"],
                                known_union_pixels=best_view["known_union_pixels"],
                                eligible_crops=sum(row["building_id"] == building for row in eligible), selected=False))
    facade_rows.sort(key=lambda row: (-row["ranking_changed_union_pixels"], row["building_id"]))
    buckets = {part: [row for row in facade_rows if row["split"] == part] for part in ("test", "val", "train")}
    selection_order = []
    while any(buckets.values()) and len(selection_order) < max_facades:
        for part in ("test", "val", "train"):
            if buckets[part] and len(selection_order) < max_facades:
                selection_order.append(buckets[part].pop(0)["building_id"])
    chosen_buildings = {building: index + 1 for index, building in enumerate(selection_order)}
    selected = []
    for rank, facade in enumerate(facade_rows, 1):
        facade["rank"] = rank
        facade["selection_order"] = chosen_buildings.get(facade["building_id"])
        if facade["building_id"] not in chosen_buildings:
            continue
        facade["selected"] = True
        chosen = []
        ordered = sorted((row for row in eligible if row["building_id"] == facade["building_id"]),
                         key=lambda row: (-row["changed_union_pixels"], -row["known_union_pixels"], row["sequence_id"]))
        for row in ordered:
            if any(row["parent_sequence_id"] == prior["parent_sequence_id"] and
                   _iou(row["crop_xywh"], prior["crop_xywh"]) > .25 for prior in chosen):
                continue
            row["selected"] = True
            chosen.append(row)
            if len(chosen) >= crops_per_facade:
                break
        selected.extend(chosen)
    selected.sort(key=lambda row: (chosen_buildings[row["building_id"]], -row["changed_union_pixels"], row["sequence_id"]))
    return selected, facade_rows


def _image_uri(array, jpeg=False):
    stream = io.BytesIO()
    image = Image.fromarray(array)
    image.thumbnail((384, 384))
    image.save(stream, format="JPEG" if jpeg else "PNG", **({"quality": 88} if jpeg else {}))
    return "data:image/" + ("jpeg" if jpeg else "png") + ";base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def _semantic(rgb, packed, support, category_count):
    colors = np.zeros(rgb.shape, np.float32)
    counts = np.zeros(support.shape, np.float32)
    for channel in range(category_count):
        mask = (packed & (1 << channel)) != 0
        colors[mask] += COLORS[channel]
        counts += mask
    known = support & (counts > 0)
    result = (rgb.astype(float) * .35).astype(np.uint8)
    result[known] = (rgb[known] * .45 + (colors[known] / counts[known, None]) * .55).astype(np.uint8)
    return result


def _gallery(out, manifest, panels):
    e = html.escape
    legend = " ".join(f'<span style="border-left:15px solid rgb{COLORS[i]};padding-left:5px">{e(str(cat["name"]))}</span>'
                      for i, cat in enumerate(manifest["categories"]))
    cards = []
    for sequence in manifest["sequences"]:
        frames = panels[sequence["sequence_id"]]
        cells = "".join(f'<article><h4>{e(str(frame["year"]))} · {frame["frame"]}</h4>'
                        f'<img src="{frame["rgb"]}" alt="Исходный RGB">'
                        f'<img src="{frame["semantic"]}" alt="Аннотированные классы">'
                        + (f'<img src="{frame["change"]}" alt="Изменение классов относительно предыдущего">' if frame.get("change") else '<p>Первое наблюдение</p>')
                        + '</article>' for frame in frames)
        stats = sequence["selection_statistics"]
        cards.append(f'<section><h2>{e(sequence["building_id"])} · {e(sequence["split"].upper())}</h2>'
                     f'<p>{e(sequence["sequence_id"])}; кроп {e(str(sequence["crop_xywh"]))}; '
                     f'изменение: {stats["changed_union_pixels"]} пикселей, '
                     f'{stats["changed_known_percent"]:.2f}% известной области. '
                     f'Пропущенные даты: {e(str(sequence.get("missing_years", [])))}.</p>'
                     f'<div class="frames">{cells}</div></section>')
    document = '''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Реальные хронологии: просмотр</title>
<style>body{font:16px system-ui;margin:24px;background:#fafafa;color:#17212b}section{background:white;padding:16px;margin:20px 0;border:1px solid #ccd2d7}
.frames{display:flex;gap:12px;overflow-x:auto}article{flex:0 0 300px}img{width:100%;display:block;margin:6px 0}.legend{display:flex;gap:12px;flex-wrap:wrap}h2{font-size:22px}p{max-width:1100px}</style>
<h1>Реальные хронологии до запуска моделей</h1>
<p>Отбор по площади изменения аннотированных классов на всей доступной истории. Это обогащённая изменениями диагностическая выборка;
выбранный TEST не является репрезентативной оценкой полного TEST. Снимки исходные, без искусственных аугментаций.</p>
<p>В каждом столбце: исходный RGB, разметка классов, изменение набора классов относительно предыдущей доступной даты.
Красный — изменение; затемнение — неизвестная область. Неизвестная разметка не означает здоровый фасад.
Проверьте геометрию и соответствие аннотаций изображениям; сдвиг границ может давать ложное изменение GT.
EARLIEST будет оцениваться относительно первого кадра, HISTORY — относительно предыдущего события после причинного прохода всей последовательности.</p>
<div class="legend">''' + legend + '</div>' + ''.join(cards) + '</html>'
    (out / "gallery.html").write_text(document, encoding="utf-8")


def prepare(audit_path, out_path, max_facades=12, crops_per_facade=1, tile_size=512, stride=256,
            min_support=.8, min_known_pixels=1024, buildings=None):
    """Select 1–2 distinct crops per highly changing façade and save full histories."""
    path = Path(audit_path).expanduser().resolve()
    path = path / "audit.json" if path.is_dir() else path
    document = read_json(path)
    if document.get("schema_version") != 1 or document.get("kind") != "real_temporal_audit":
        raise ValueError("Ожидался real_temporal_audit schema 1")
    if digest({key: value for key, value in document.items() if key != "audit_sha256"}) != document.get("audit_sha256"):
        raise ValueError("Нарушена целостность audit.json")
    for value, name, minimum in ((max_facades, "max_facades", 1), (tile_size, "tile_size", 8),
                                 (stride, "stride", 1), (min_known_pixels, "min_known_pixels", 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} должен быть целым >= {minimum}")
    if type(crops_per_facade) is not int or crops_per_facade not in (1, 2):
        raise ValueError("crops_per_facade должен быть 1 или 2")
    if stride > tile_size or isinstance(min_support, bool) or not 0 < min_support <= 1:
        raise ValueError("Требуется stride <= tile_size и 0 < min_support <= 1")
    out = _new_output(out_path, [path.parent, Path(document["source_run"]), Path(document["snapshot"])])
    _verify_hashes(document["metadata_sha256"])
    data = PreparedData(document["dataset"], splits=tuple(document["requested_splits"]), allow_test=True)
    _validate_category_mapping(data.categories, data.metadata["category_to_channel"])
    source_hashes = dict(document["metadata_sha256"])
    source_hashes.update({value: data.metadata["sha256"][key] for key, value in data.metadata["paths"].items()})
    candidates, view_rows, panels = [], [], {}
    parents = document["sequences"]
    if not parents:
        raise ValueError("В аудите нет пригодных совмещённых историй; проверьте alignment_coverage.csv")
    out.mkdir(parents=True)
    state = dict(schema_version=1, kind="real_temporal_preparation", status="preparing",
                 audit_path=str(path), audited_views=len(parents), processed_views=0)
    atomic_json(out / "run.json", state)
    try:
        for index, parent in enumerate(tqdm(parents, desc="Inspect real histories", unit="view", dynamic_ncols=True), 1):
            tqdm.write(f"CPU [{index}/{len(parents)}] {parent['building_id']} · {parent['split']}: "
                       f"разметка {len(parent['frames'])} дат")
            rows, stats = _rank_view(data, parent, tile_size, stride, min_support, min_known_pixels)
            candidates.extend(rows)
            view_rows.append(stats)
            for frame in parent["frames"]:
                source_hashes[frame["image_path"]] = frame["sha256"]
            state["processed_views"] = index
            atomic_json(out / "run.json", state)
        selected, facade_rows = _select(candidates, view_rows, max_facades, crops_per_facade, buildings)
        _csv(out / "candidate_crops.csv", [dict(row, crop_xywh=";".join(map(str, row["crop_xywh"]))) for row in candidates])
        _csv(out / "facades.csv", facade_rows)
        _csv(out / "views.csv", view_rows)
        if not selected:
            raise ValueError("Нет кропов с достаточной поддержкой, известной разметкой и изменениями")
        parent_index = {row["sequence_id"]: row for row in parents}
        sequences = []
        for index, row in enumerate(tqdm(selected, desc="Export real histories", unit="history", dynamic_ncols=True), 1):
            parent = parent_index[row["parent_sequence_id"]]
            tqdm.write(f"CPU сохраняем [{index}/{len(selected)}] {row['building_id']} · {row['sequence_id']}")
            frames, previews, previous = [], [], None
            for frame_number, frame in enumerate(parent["frames"]):
                arrays = _warp(data, parent, frame, row["crop_xywh"], include_rgb=True)
                relative = f"frames/{row['sequence_id']}/{frame_number:03d}.npz"
                frame_hash = atomic_npz(out / relative, arrays)
                frames.append(dict(frame=frame_number, year=frame["year"], image_id=frame["image_id"],
                                   image_sha256=frame["sha256"], artifact_path=relative, artifact_sha256=frame_hash))
                preview = dict(frame=frame_number, year=frame["year"], rgb=_image_uri(arrays["rgb"], jpeg=True),
                               semantic=_image_uri(_semantic(arrays["rgb"], arrays["packed"], arrays["support"], len(data.categories))))
                if previous is not None:
                    labels = derive_labels(previous["packed"], arrays["packed"],
                                           previous["support"] & arrays["support"], len(data.categories))
                    overlay = arrays["rgb"].copy()
                    overlay[labels == 255] = (overlay[labels == 255] * .25).astype(np.uint8)
                    overlay[labels == 1] = (overlay[labels == 1] * .35 + np.array([255, 40, 40]) * .65).astype(np.uint8)
                    preview["change"] = _image_uri(overlay)
                previews.append(preview)
                previous = arrays
            panels[row["sequence_id"]] = previews
            sequences.append(dict(sequence_id=row["sequence_id"], parent_sequence_id=parent["sequence_id"],
                                  domain="real", building_id=row["building_id"], split=row["split"], view_id=row["view_id"],
                                  crop_xywh=row["crop_xywh"], shape=[row["crop_xywh"][3], row["crop_xywh"][2]],
                                  frames=frames, common_support_fraction=row["common_support_fraction"],
                                  missing_dates=parent["missing_dates"], missing_years=parent["missing_years"],
                                  selection_statistics={key: row[key] for key in ("changed_union_pixels", "known_union_pixels",
                                      "temporal_changed_pixel_observations", "changed_known_percent", "changed_crop_percent")}))
        _verify_hashes(source_hashes)
        coverage = dict(facades=len({s["building_id"] for s in sequences}), sequences=len(sequences),
                        frames=sum(len(s["frames"]) for s in sequences),
                        previous_pairs=sum(len(s["frames"])-1 for s in sequences),
                        earliest_pairs=sum(len(s["frames"])-1 for s in sequences),
                        by_split={part: dict(facades=len({s["building_id"] for s in sequences if s["split"] == part}),
                                            sequences=sum(s["split"] == part for s in sequences),
                                            previous_pairs=sum(len(s["frames"])-1 for s in sequences if s["split"] == part))
                                  for part in document["requested_splits"]},
                        candidate_crops=len(candidates), eligible_crops=sum(row["status"] == "eligible" for row in candidates),
                        audited_facades=document["coverage"]["facades"],
                        aligned_facades=document["coverage"]["eligible_facades"])
        manifest = dict(schema_version=1, kind="real_temporal_prepared",
                        **{key: document[key] for key in ("source_run", "snapshot", "source_config", "source_contract",
                                                       "source_records", "source_summary", "dataset", "requested_splits")},
                        categories=data.categories, category_to_channel=data.metadata["category_to_channel"],
                        prepared_data=data.metadata, audit_path=str(path), audit_file_sha256=sha256(path),
                        metadata_sha256=source_hashes, sequences=sequences, coverage=coverage,
                        selection_policy=dict(kind="GT-enriched diagnostic subset", max_facades=max_facades,
                            crops_per_facade=crops_per_facade, tile_size=tile_size, stride=stride,
                            min_support=min_support, min_known_pixels=min_known_pixels,
                            facade_rank="descending maximum per-view spatial union of adjacent annotated changes on common geometric support",
                            split_selection="round-robin TEST then VAL then TRAIN, descending facade rank within each split; exhausted splits skipped",
                            crop_rank="descending spatial union of adjacent annotated changes; ties by known pixels then sequence ID",
                            same_view_max_crop_iou=.25, requested_buildings=sorted(buildings) if buildings else None,
                            absolute_area_unit="pixels on each reference view; not physical facade area",
                            test_scope="GT-selected exploratory subset; no tuning permitted on TEST",
                            gt_scope="all annotated class-set changes; bits0 unknown; no semantic predictions used",
                            augmentation="none; original observed RGB", temporal_policy="all accepted dates in causal order; gaps explicit"))
        manifest["manifest_sha256"] = digest(manifest)
        atomic_json(out / "manifest.json", manifest)
        _gallery(out, manifest, panels)
        state.update(status="completed", coverage=coverage, manifest_sha256=sha256(out / "manifest.json"))
        atomic_json(out / "run.json", state)
        print(f"Подготовлено: {coverage['facades']} фасадов, {len(sequences)} кроп-историй, "
              f"{coverage['previous_pairs']} PREVIOUS + {coverage['earliest_pairs']} EARLIEST на метод.", flush=True)
        print(f"Галерея: {out / 'gallery.html'}", flush=True)
        return manifest
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error=f"{type(error).__name__}: {error}")
        atomic_json(out / "run.json", state)
        raise
