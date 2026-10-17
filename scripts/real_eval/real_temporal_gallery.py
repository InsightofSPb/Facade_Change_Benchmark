#!/usr/bin/env python3
"""Read-only real-VAL gallery from authenticated maps and frozen VAL thresholds."""
import argparse
import base64
import csv
import hashlib
import html
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

GROUPS = {
    "rgb_diff": ("rgb_diff", "frozen", "previous"),
    "geoscd": ("geoscd", "frozen", "previous"),
    "anychange": ("anychange", "frozen", "previous"),
    "ssim": ("ssim", "frozen", "previous"),
    "msdzip_abs": ("msdzip_abs", "current_train", "previous"),
    "msdzip_mod256": ("msdzip_mod256", "current_train", "previous"),
    "msdzip_mod256_history": ("msdzip_mod256", "current_train", "history"),
}
REQUIRED = tuple(key for key in GROUPS if key != "msdzip_mod256_history")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["sequence_id"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def threshold(summary, name, budget, optional=False):
    group = GROUPS[name]
    rows = [row for row in summary.get("calibration", []) if
            tuple(row.get(key) for key in ("method", "variant", "mode")) == group
            and row.get("target") == "step_event"]
    if not rows and optional:
        return None
    if len(rows) != 1:
        raise ValueError(f"Need exactly one saved calibration for {name}")
    row = rows[0]
    if (row.get("status") != "calibrated" or row.get("selection_split") != "val"
            or row.get("selection_domain") != "synthetic" or row.get("selection_arm") != "clean"):
        raise ValueError(f"Calibration for {name} is not frozen synthetic VAL clean-arm calibration")
    choices = [choice for choice in row["choices"] if abs(float(choice["fpr_target"]) - budget) < 1e-12]
    if len(choices) != 1 or not np.isfinite(choices[0]["threshold"]) or choices[0]["threshold"] < 0:
        raise ValueError(f"Missing or invalid saved {budget:.0%} threshold for {name}")
    return {**choices[0], "method": group[0], "variant": group[1], "mode": group[2],
            "target": "step_event", "source_unit_ids": row.get("source_unit_ids", []),
            "selection_split": "val", "selection_domain": "synthetic", "selection_arm": "clean"}


def crop_order(identifier):
    match = re.search(r"-x(\d+)-y(\d+)$", identifier)
    return (int(match[2]), int(match[1]), identifier) if match else (0, 0, identifier)


def load_map(root, record, native, hashes):
    path = (root / record["artifact_path"]).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"Missing or unsafe map path: {path}")
    actual = sha256(path)
    if actual != record["artifact_sha256"]:
        raise ValueError(f"Map hash mismatch: {path}")
    hashes[str(path)] = actual
    names = ["scores", "support", "reference", "source"] + (["native_prediction"] if native else [])
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in names}
    scores, support = arrays["scores"], arrays["support"]
    if (scores.dtype != np.float32 or scores.ndim != 2 or support.dtype != bool
            or support.shape != scores.shape or not np.isfinite(scores[support]).all()):
        raise ValueError(f"Invalid native score/support grid: {record['unit_id']}")
    for name in ("reference", "source"):
        if arrays[name].dtype != np.uint8 or arrays[name].shape != (*scores.shape, 3):
            raise ValueError(f"Invalid {name} RGB: {record['unit_id']}")
    if native and (arrays["native_prediction"].dtype != bool or arrays["native_prediction"].shape != scores.shape):
        raise ValueError(f"Invalid author-native prediction: {record['unit_id']}")
    return arrays


def figure(out, relative, array, caption):
    picture = Image.fromarray(array)
    picture.save(out / relative)
    picture.thumbnail((512, 512))
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (f'<figure><a href="{relative}" target="_blank"><img src="data:image/png;base64,{encoded}" '
            f'alt="{html.escape(caption)}"></a><figcaption>{html.escape(caption)}</figcaption></figure>')


def merge_geoscd_overlay(root, snapshot, records, geoscd_run):
    """Read a separate retry; never alter the original run or its snapshot."""
    if geoscd_run is None:
        return records, {}, None
    extra = Path(geoscd_run).expanduser().resolve()
    contract_path = extra / "run.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if Path(contract["source_run"]).expanduser().resolve() != root:
        raise ValueError("GeoSCD retry belongs to another source run")
    for name, key in (("2026-10-17_source_units.json", "snapshot_source_units_sha256"),
                      ("2026-10-17_quick_summary.json", "snapshot_quick_summary_sha256")):
        if contract[key] != sha256(snapshot / name):
            raise ValueError("GeoSCD retry belongs to another frozen snapshot")
    baseline = {(row["sequence_id"], row["frame"]): row for row in records
                if (row.get("domain"), row.get("split"), row.get("method"), row.get("variant"),
                    row.get("mode"), row.get("status")) == ("real", "val", "rgb_diff", "frozen", "previous", "completed")}
    replacements, roots, marker_hashes = {}, {}, {}
    failures = []
    for path in sorted((extra / "units").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if (row.get("domain"), row.get("split"), row.get("method"), row.get("variant"), row.get("mode")) != ("real", "val", "geoscd", "frozen", "previous"):
            continue
        key = row["sequence_id"], row["frame"]
        if (key not in baseline or row["building_id"] != baseline[key]["building_id"]
                or row.get("target") != "step_event"):
            raise ValueError("GeoSCD retry pair identity differs from real VAL baseline")
        marker_hashes[str(path)] = sha256(path)
        if row.get("status") != "completed":
            failures.append({"sequence_id": key[0], "frame": key[1], "status": row.get("status"), "error": row.get("error")})
            continue
        if key in replacements or row["unit_id"] in roots:
            raise ValueError("Duplicate completed GeoSCD retry pair")
        replacements[key] = row
        roots[row["unit_id"]] = extra
    merged = [row for row in records if not
              ((row.get("method"), row.get("variant"), row.get("mode"), row.get("domain"), row.get("split"))
               == ("geoscd", "frozen", "previous", "real", "val")
               and (row["sequence_id"], row["frame"]) in replacements)]
    merged.extend(replacements.values())
    info = {"path": str(extra), "run_json_sha256": sha256(contract_path),
            "unit_marker_sha256": marker_hashes, "completed_previous_pairs": len(replacements),
            "unsuccessful_previous_pairs": failures}
    return merged, roots, info


def run(run_path, snapshot_path, out_path, max_facades=3, crops_per_facade=1, sequence_ids=None, geoscd_run=None):
    root, snapshot, out = (Path(value).expanduser().resolve() for value in (run_path, snapshot_path, out_path))
    if out.exists() or out == root or root in out.parents or out == snapshot or snapshot in out.parents:
        raise FileExistsError("Use a NEW output directory outside the source run and snapshot")
    if geoscd_run is not None:
        extra = Path(geoscd_run).expanduser().resolve()
        if out == extra or extra in out.parents:
            raise FileExistsError("Use a NEW output directory outside the GeoSCD retry run")
    if max_facades < 1 or crops_per_facade < 1:
        raise ValueError("Selection limits must be positive")
    paths = [snapshot / "2026-10-17_source_units.json", snapshot / "2026-10-17_quick_summary.json"]
    records, saved = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if Path(saved["source_run"]).expanduser().resolve() != root:
        raise ValueError("Snapshot belongs to a different source run")
    baseline_certified = saved.get("method_status_snapshot", {}).get("rgb_diff", {}).get("status") == "completed"
    domain_split_counts = dict(Counter(f"{record.get('domain', 'unknown')}/{record.get('split', 'unknown')}"
                                      for record in records))
    thresholds = {name: threshold(saved, name, budget, optional=True) for name, budget in
                  (("ssim", .01), ("msdzip_abs", .05), ("msdzip_mod256", .05))}
    thresholds["msdzip_mod256_history"] = threshold(saved, "msdzip_mod256_history", .05, optional=True)
    records, overlay_roots, overlay_info = merge_geoscd_overlay(root, snapshot, records, geoscd_run)
    indexed, identities = defaultdict(lambda: defaultdict(dict)), {}
    for record in records:
        if record.get("domain") != "real" or record.get("split") != "val":
            continue
        identifier, building = record["sequence_id"], record["building_id"]
        if identifier in identities and identities[identifier] != building:
            raise ValueError("A sequence changes physical building identity")
        identities[identifier] = building
        group = tuple(record.get(key) for key in ("method", "variant", "mode"))
        if record.get("status") != "completed" or record.get("target") != "step_event" or record["frame"] < 1:
            continue
        for name, expected in GROUPS.items():
            if group == expected:
                if record["frame"] in indexed[identifier][name]:
                    raise ValueError("Duplicate completed method/sequence/frame unit")
                indexed[identifier][name][record["frame"]] = record
    coverage, eligible = [], []
    for identifier in sorted(identities, key=lambda value: (identities[value], crop_order(value))):
        baseline = set(indexed[identifier]["rgb_diff"])
        expected = set(range(1, max(baseline) + 1)) if baseline else set()
        missing = [name for name in REQUIRED if not expected or set(indexed[identifier][name]) != expected]
        complete = baseline_certified and bool(expected) and baseline == expected
        method_coverage = {}
        for name in GROUPS:
            present = set(indexed[identifier][name]) & expected
            method_coverage[name] = {"completed_frames": len(present), "expected_frames": len(expected),
                                     "missing_frames": sorted(expected - present),
                                     "status": "complete" if expected and present == expected else "partial" if present else "missing"}
        history = (set(indexed[identifier]["msdzip_mod256_history"]) == expected and bool(expected)
                   and thresholds["msdzip_mod256_history"] is not None)
        coverage.append({"sequence_id": identifier, "building_id": identities[identifier],
                         "frame_count": len(expected), "missing_required": missing,
                         "method_coverage": method_coverage,
                         "baseline_status": "complete" if complete else "method_not_completed" if not baseline_certified else "missing_or_noncontiguous",
                         "eligible": complete, "history_status": "complete" if history else "missing_or_incomplete",
                         "selected": False})
        if complete:
            eligible.append(identifier)
    print("Доступность результатов real VAL по кадрам базовой хронологии: " + "; ".join(
        f"{name} {sum(row['method_coverage'][name]['completed_frames'] for row in coverage)}/"
        f"{sum(row['method_coverage'][name]['expected_frames'] for row in coverage)}"
        for name in GROUPS), flush=True)
    if sequence_ids is not None:
        if not sequence_ids or len(sequence_ids) != len(set(sequence_ids)) or set(sequence_ids) - set(eligible):
            raise ValueError("Explicit sequence_ids must be distinct eligible whole real VAL histories")
        selected = [identifier for identifier in eligible if identifier in set(sequence_ids)]
    else:
        counts, selected = defaultdict(int), []
        for identifier in eligible:
            building = identities[identifier]
            if building not in counts and len(counts) >= max_facades:
                continue
            if counts[building] < crops_per_facade:
                selected.append(identifier)
                counts[building] += 1
    diagnostic = None
    if not selected:
        diagnostic = ("В снимке результатов нет полных непрерывных реальных историй VAL по RGB-DIFF. "
                      "Проверьте домены и разбиения в сводке, а также статус RGB-DIFF и coverage.csv. "
                      "Отсутствие результатов не означает отсутствие изменений.")
    out.mkdir(parents=True)
    (out / "images").mkdir()
    for row in coverage:
        row["selected"] = row["sequence_id"] in selected
    artifact_hashes, results, reviews = {}, [], []
    with (out / "gallery.html").open("w", encoding="utf-8") as page:
        page.write('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Изменения фасадов: реальные снимки</title>'
                   '<style>body{font:15px system-ui;background:#f3f4f6;margin:24px;color:#18212b}section{background:white;padding:18px;margin:24px 0}'
                   '.panels{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;max-width:280px}img{width:100%}figcaption{padding:6px}h3{margin-top:24px}</style>'
                   '<h1>Изменения фасадов: просмотр реальных снимков</h1><p>Только реальные наблюдения VAL. В этих сохранённых результатах реальные временные маски не подтверждены; метрики качества здесь не рассчитываются. '
                   'Хронология определяется полными сериями RGB-DIFF. Для каждой пары показаны имеющиеся результаты остальных методов. Отсутствующий результат не считается отрицательным предсказанием. Ограниченная выборка не представляет все фасады.</p>'
                   '<p>Пороги взяты из калибровки на синтетическом VAL. Значения 1%/5% — ограничения при калибровке, а не измеренная частота ложных срабатываний на реальных фотографиях. '
                   'Пурпурный цвет — предсказанная маска; серый — нет геометрической поддержки. Пересечение (∩) оставляет только совпавшие предсказания двух методов. '
                   'Доли показывают площадь маски относительно геометрической поддержки, а не точность. Нажмите на изображение, чтобы открыть PNG в исходном разрешении. '
                   'Номера наблюдений отражают хронологический порядок, а не календарные даты. Заполните review_template.csv; эти оценки не являются попиксельной временной разметкой.</p>')
        page.write('<p>Фиксированные пороги: ' + '; '.join(
            f"{html.escape(name)}: оценка &gt; {item['threshold']:.8g}, ограничение H0 на синтетическом VAL {item['fpr_target']:.0%}"
            for name, item in thresholds.items() if item is not None) + '</p>')
        if diagnostic:
            page.write(f"<section><h2>Нет доступной базовой истории</h2><p>{html.escape(diagnostic)}</p>"
                       f"<p>Записей по домену/разбиению: {html.escape(json.dumps(domain_split_counts, ensure_ascii=False))}</p></section>")
        for number, identifier in enumerate(selected):
            print(f"История {number + 1}/{len(selected)}: {identities[identifier]} · {identifier}", flush=True)
            selection_row = next(row for row in coverage if row["sequence_id"] == identifier)
            for frame in sorted(indexed[identifier]["rgb_diff"]):
                maps = {name: load_map(overlay_roots.get(indexed[identifier][name][frame]["unit_id"], root), indexed[identifier][name][frame], name in {"geoscd", "anychange"}, artifact_hashes)
                        for name in REQUIRED if frame in indexed[identifier][name]}
                base = maps["rgb_diff"]
                for name, arrays in maps.items():
                    if any(not np.array_equal(arrays[key], base[key]) for key in ("support", "reference", "source")):
                        raise ValueError(f"Pair RGB/support mismatch: {identifier} frame {frame} {name}")
                method_status = {name: "available" if name in maps else "missing_result"
                                 for name in GROUPS if name != "rgb_diff"}
                for name in ("ssim", "msdzip_abs", "msdzip_mod256"):
                    if name in maps and thresholds[name] is None:
                        method_status[name] = "missing_calibration"
                history_status = "missing_or_incomplete"
                names = list(maps)
                if frame in indexed[identifier]["msdzip_mod256_history"]:
                    history = load_map(root, indexed[identifier]["msdzip_mod256_history"][frame], False, artifact_hashes)
                    names.append("msdzip_mod256_history")
                    if not all(np.array_equal(history[key], base[key]) for key in ("support", "source")):
                        history_status = "not_comparable: history support or current RGB differs"
                    elif thresholds["msdzip_mod256_history"] is None:
                        history_status = "missing_calibration"
                    else:
                        maps["msdzip_mod256_history"] = history
                        history_status = "complete"
                method_status["msdzip_mod256_history"] = "available" if history_status == "complete" else history_status
                support = base["support"]
                masks = {name: arrays["native_prediction"] & support if name in {"geoscd", "anychange"}
                         else (arrays["scores"] > np.float32(thresholds[name]["threshold"])) & support
                         for name, arrays in maps.items() if name != "rgb_diff" and method_status[name] == "available"}
                for native in ("geoscd", "anychange"):
                    for gate in ("ssim", "msdzip_abs", "msdzip_mod256", "msdzip_mod256_history"):
                        if native in masks and gate in masks:
                            masks[native + "_and_" + gate] = masks[native] & masks[gate]
                prefix = f"images/s{number:02d}_f{frame:03d}"
                row = {"building_id": identities[identifier], "sequence_id": identifier, "frame": frame,
                       "history_status": history_status, "method_status": method_status, "image_paths": {}, "mask_paths": {}, "mask_fractions": {},
                       "support_fraction": float(support.mean()),
                       "source_unit_ids": [indexed[identifier][name][frame]["unit_id"] for name in names]}
                history_label = {
                    'complete': 'доступна',
                    'missing_or_incomplete': 'нет результата для этой пары',
                    'missing_calibration': 'нет сохранённого калибровочного порога',
                    'not_comparable: history support or current RGB differs': 'несопоставима: различаются геометрическая поддержка или текущий RGB',
                }[history_status]
                page.write(f'<section><h2>{html.escape(identities[identifier])} · {html.escape(identifier)}</h2>'
                           f'<p>Наблюдение {frame} → {frame + 1}. Геометрическая поддержка: {support.mean():.1%}. '
                           f'История для этой пары: {html.escape(history_label)}. '
                           f'Готовых кадров истории во всей серии: {selection_row["method_coverage"]["msdzip_mod256_history"]["completed_frames"]}/{selection_row["frame_count"]}.</p><div class="panels">')
                for key in ("reference", "source"):
                    relative = prefix + "_" + key + ".png"
                    row["image_paths"][key] = relative
                    page.write(figure(out, relative, base[key], "Предыдущий снимок RGB" if key == "reference" else "Текущий снимок RGB"))
                page.write('</div>')
                panels = {}
                for name, mask in masks.items():
                    relative = prefix + "_" + name + ".png"
                    overlay = base["source"].copy()
                    overlay[mask] = (.45 * overlay[mask] + .55 * np.array([255, 0, 255])).astype(np.uint8)
                    overlay[~support] = 128
                    row["image_paths"][name] = relative
                    row["mask_paths"][name] = prefix + "_" + name + "_mask.png"
                    binary = np.where(support, mask.astype(np.uint8) * 255, 128).astype(np.uint8)
                    Image.fromarray(binary).save(out / row["mask_paths"][name])
                    fraction = float(mask.sum() / support.sum()) if support.any() else None
                    row["mask_fractions"][name] = fraction
                    area = f"{fraction:.1%}" if fraction is not None else "нет поддержки"
                    panels[name] = figure(out, relative, overlay, f"{name.replace('_and_', ' ∩ ')}: {area} геометрической поддержки")
                unavailable = {"missing_result": "нет результата для этой пары",
                               "missing_or_incomplete": "нет результата для этой пары",
                               "missing_calibration": "нет сохранённого калибровочного порога",
                               "not_comparable: history support or current RGB differs": "несопоставимая поддержка или текущий RGB"}
                for native in ("geoscd", "anychange"):
                    page.write(f'<h3>{native}: исходная маска и пересечения</h3><div class="panels">' +
                               ''.join(value for name, value in panels.items() if name == native or name.startswith(native + "_and_")) + '</div>')
                    if native not in masks:
                        page.write(f'<p>{native}: нет результата для этой пары; маска и пересечения не построены.</p>')
                missing_text = ''.join(f'<p>{html.escape(name)}: {unavailable[method_status[name]]}; отсутствующая карта не считается отрицательным результатом.</p>'
                                       for name in thresholds if method_status[name] != "available")
                page.write(missing_text + '<details><summary>Отдельные маски SSIM/MSDZip</summary><div class="panels">' +
                           ''.join(value for name, value in panels.items() if name in thresholds) + '</div></details></section>')
                results.append(row)
                reviews.append({"sequence_id": identifier, "building_id": identities[identifier], "frame": frame,
                                "alignment_acceptable_uncertain_reject": "", "visible_change_yes_no_uncertain": "",
                                "occlusion_yes_no": "", "comment": ""})
        page.write('</html>')
    summary = {"status": "completed" if selected else "diagnostic_no_real_val_baseline",
               "diagnostic": diagnostic, "geoscd_overlay": overlay_info, "domain_split_counts": domain_split_counts, "baseline_method_completed": baseline_certified,
               "scope": "qualitative real VAL review; no temporal GT or numerical accuracy", "selected_sequences": selected,
               "thresholds": thresholds, "results": results, "coverage": coverage}
    provenance = {"geoscd_overlay": overlay_info, "source_run": str(root), "snapshot": str(snapshot), "source_metadata_sha256": {str(path): sha256(path) for path in paths},
                  "selection_policy": "whole contiguous completed RGB-DIFF real VAL histories; other methods available per frame; building lexicographic then crop y/x; no prediction-based selection",
                  "domain_split_counts": domain_split_counts, "baseline_method_completed": baseline_certified,
                  "selection_options": {"max_facades": max_facades, "crops_per_facade": crops_per_facade, "sequence_ids": sequence_ids},
                  "selected_sequences": selected, "thresholds": thresholds, "artifact_sha256": artifact_hashes,
                  "source_unit_ids": sorted({value for row in results for value in row["source_unit_ids"]}),
                  "review_limit": "CSV judgement fields are not dense temporal ground truth; no accuracy computed"}
    write_csv(out / "coverage.csv", coverage)
    write_csv(out / "review_template.csv", reviews)
    write_json(out / "summary.json", summary)
    write_json(out / "provenance.json", provenance)
    if diagnostic:
        print(diagnostic, flush=True)
    print(f"Сохранено реальных пар: {len(results)}; полных историй: {len(selected)}. Галерея: {out / 'gallery.html'}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-facades", type=int, default=3)
    parser.add_argument("--crops-per-facade", type=int, default=1)
    parser.add_argument("--sequence-ids", nargs="+")
    parser.add_argument("--geoscd-run", help="Separate GeoSCD retry output; original run stays unchanged")
    args = parser.parse_args()
    run(args.run, args.snapshot, args.out, args.max_facades, args.crops_per_facade, args.sequence_ids, args.geoscd_run)
