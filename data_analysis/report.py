"""Russian reports and local PNG galleries for cached facade diagnostics.

Reports only describe frozen decisions. Gallery examples may use evaluation
labels to expose failures; this selection never changes predictions or metrics.
"""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .core import COMPRESSION, FUSION, SPLITS

METRICS = ("f1", "precision", "recall", "h0_pixel_fpr")


def percent(value):
    """Display a rate as a percentage; undefined metrics remain undefined."""
    return "н/д" if value is None else f"{100 * value:.2f}%"


def _pp(value):
    return "н/д" if value is None else f"{100 * value:.2f} п.п."


def _write_csv(path, rows):
    """Keep all diagnostic fields, including optional raw-score statistics."""
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        if not fields:
            stream.write("")
            return
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def _method_rows(summary, dimension=None):
    """Flatten facade macros without pooling pixels across buildings."""
    result = []
    for method, slices in summary["methods"].items():
        groups = ([{"split": part, **slices["by_split"][part]} for part in SPLITS
                   if part in slices["by_split"]] if dimension is None
                  else slices.get("by_" + dimension, []))
        for group in groups:
            identity = {"split": group["split"], "method": method}
            if dimension is not None:
                identity[dimension] = group[dimension]
                result.append({**identity, "level": "split", "building_id": "",
                               "case_count": group["case_count"],
                               "building_count": group["building_count"], **group["means"]})
            for building in group["buildings"]:
                result.append({**identity, "level": "building",
                               "building_id": building["building_id"],
                               "case_count": building["case_count"], "building_count": 1,
                               **building["means"]})
    return result


def _index_methods(summary, split, dimension=None, value=None, building=None):
    result = {}
    for method, slices in summary["methods"].items():
        if dimension is None:
            group = slices["by_split"].get(split)
        else:
            group = next((g for g in slices.get("by_" + dimension, [])
                          if g["split"] == split and g[dimension] == value), None)
        if group is None:
            continue
        if building is not None:
            group = next((b for b in group["buildings"] if b["building_id"] == building), None)
        if group is not None:
            result[method] = group["means"]
    return result


def _pair_value(means, *names):
    for name in names:
        if name in means:
            return means[name]
    return None


def _pair_lines(summary, split, dimension=None, value=None, building=None):
    """Percent of removed baseline TP is distinct from lost Recall in p.p."""
    result = []
    for baseline, slices in summary.get("pairs", {}).items():
        if dimension is None:
            group = slices["by_split"].get(split)
        else:
            group = next((g for g in slices.get("by_" + dimension, [])
                          if g["split"] == split and g[dimension] == value), None)
        if group is None:
            continue
        if building is not None:
            group = next((b for b in group["buildings"] if b["building_id"] == building), None)
        if group is None:
            continue
        means = group["means"]
        loss = _pair_value(means, "and_lost_recall", "lost_recall")
        rejected = _pair_value(means, "rejected_baseline_tp_fraction", "and_rejected_baseline_tp_fraction",
                               "removed_baseline_tp_fraction", "rejected_tp_fraction")
        added = _pair_value(means, "added_recall")
        fpr_removed = _pair_value(means, "and_removed_h0_fpr", "removed_h0_fpr")
        result.append(f"{baseline}: потеря Recall при пересечении {_pp(loss)}; "
                      f"отброшено верных находок исходного метода {percent(rejected)}; "
                      f"потенциальный добавочный Recall MSDZip {_pp(added)}; "
                      f"снижение H0 FPR {_pp(fpr_removed)}")
    return result


def _method_lines(means):
    return [method + ": " + " / ".join(percent(values.get(key)) for key in METRICS)
            for method, values in means.items()]


def _geo_loss_line(means):
    if "geoscd" not in means or FUSION not in means:
        return []
    geo, fusion = means["geoscd"], means[FUSION]
    original, retained = geo.get("recall"), fusion.get("recall")
    loss = original - retained if original is not None and retained is not None else None
    delta = {key: fusion.get(key) - geo.get(key)
             if fusion.get(key) is not None and geo.get(key) is not None else None
             for key in METRICS}
    return ["AND − GeoSCD, п.п. (F1 / Precision / Recall / H0 FPR): " + " / ".join(
                "н/д" if delta[key] is None else f"{100 * delta[key]:+.2f}" for key in METRICS),
            f"GeoSCD Recall {percent(original)} → AND {percent(retained)}; потеря {_pp(loss)}."]


def summary_text(summary):
    """All requested methods, each split/facade, and class/condition diagnostics."""
    lines = [summary.get("scope", "EXPLORATORY cached-map diagnostics"),
             "Сохранённые карты; исходные пороги и маски зафиксированы. Повторного inference и подбора порогов нет.",
             "Агрегация: среднее случаев внутри фасада, затем равные веса фасадов; пиксели разных фасадов не объединяются.",
             f"Случаев: {summary['analyzed_case_count']}; фасады: {summary['buildings_by_split']}.",
             "Приоритет разбора — Recall: что MSDZip отвергает среди верных находок каждого исходного метода.",
             "Потеря Recall измеряется в процентных пунктах от всех GT-положительных пикселей случая; "
             "доля отброшенных верных находок — в процентах от TP исходного метода. Это разные знаменатели.",
             "Для H0 и случаев без видимых положительных пикселей F1/Precision/Recall не определены; H0 оценивается по FPR."]
    for part in SPLITS:
        if not summary.get("cases_by_split", {}).get(part, 0):
            continue
        means = _index_methods(summary, part)
        lines += ["", f"{part.upper()} — среднее по фасадам: F1 / Precision / Recall / H0 FPR",
                  *_method_lines(means), *_geo_loss_line(means),
                  "Диагностика пересечения с MSDZip (кандидаты для анализа, без выбора новых правил):",
                  *_pair_lines(summary, part)]
        buildings = sorted({b["building_id"] for slices in summary["methods"].values()
                            for b in slices["by_split"][part]["buildings"]})
        for building in buildings:
            means = _index_methods(summary, part, building=building)
            lines += ["", f"{part.upper()} | фасад {building}: F1 / Precision / Recall / H0 FPR",
                      *_method_lines(means), *_geo_loss_line(means),
                      *_pair_lines(summary, part, building=building)]
        states = sorted({g["state"] for slices in summary["methods"].values()
                         for g in slices.get("by_state", []) if g["split"] == part})
        for state in states:
            means = _index_methods(summary, part, "state", state)
            lines += ["", f"{part.upper()} | класс состояния {state}: F1 / Precision / Recall / H0 FPR",
                      *_method_lines(means), *_geo_loss_line(means),
                      *_pair_lines(summary, part, "state", state)]
            for building in buildings:
                means = _index_methods(summary, part, "state", state, building)
                if not means:
                    continue
                lines += [f"  {building}: " + "; ".join(
                    method + " Recall=" + percent(values.get("recall"))
                    + ", H0 FPR=" + percent(values.get("h0_pixel_fpr"))
                    for method, values in means.items()),
                    *("  " + line for line in _pair_lines(summary, part, "state", state, building))]
        nuisances = sorted({g["nuisance_kind"] for slices in summary["methods"].values()
                            for g in slices.get("by_nuisance_kind", []) if g["split"] == part})
        for nuisance in nuisances:
            means = _index_methods(summary, part, "nuisance_kind", nuisance)
            lines += ["", f"{part.upper()} | условие {nuisance}: F1 / Precision / Recall / H0 FPR",
                      *_method_lines(means), *_geo_loss_line(means),
                      *_pair_lines(summary, part, "nuisance_kind", nuisance)]
    lines += ["", "Галерея выбирает диагностические примеры по GT после оценки: сначала худшая потеря "
              "Recall GeoSCD при AND на каждом фасаде, затем оставшиеся случаи по той же потере. "
              "При равенстве — H1 перед H0, затем case_id. "
              "Это отбор ошибок для осмотра, а не обучение или подбор правила; численные таблицы содержат все случаи.",
              "Низкий MSDZip score показывает непосредственную причину отбрасывания порогом, "
              "но сам по себе не доказывает физическую или модельную причину ошибки. "
              "Нормированный score и сырая оценка bpb — разные величины; сопоставляйте столбцы score_diagnostics.csv.",
              "TEST и прежние результаты уже просмотрены; новые правила по этому разбору требуют отдельной проверки.",
              "Процедурные синтетические изменения; это не оценка реальных повреждений. Нет доверительных интервалов."]
    return "\n".join(lines) + "\n"


def _overlay(rgb, mask, color, alpha=.72):
    result = np.asarray(rgb, dtype=np.uint8).copy()
    result[mask] = ((1 - alpha) * result[mask].astype(np.float32)
                    + alpha * np.asarray(color, dtype=np.float32)).astype(np.uint8)
    return result


def _prediction_panel(rgb, prediction, labels):
    panel = _overlay(rgb, prediction & (labels == 1), (35, 210, 80))
    panel = _overlay(panel, prediction & (labels == 0), (235, 35, 45))
    panel = _overlay(panel, ~prediction & (labels == 1), (255, 205, 35))
    panel[labels == 255] = 128
    return panel


def _loss_panel(entry):
    labels = entry["labels"]
    geo = entry["predictions"]["geoscd"]
    msd = entry["predictions"][COMPRESSION]
    panel = _overlay(entry["source"], geo & msd & (labels == 1), (35, 210, 80))
    panel = _overlay(panel, geo & ~msd & (labels == 1), (255, 140, 0))
    panel = _overlay(panel, geo & ~msd & (labels == 0), (25, 140, 255))
    panel = _overlay(panel, ~geo & (labels == 1), (235, 35, 45))
    panel[labels == 255] = 128
    return panel


def _score_panel(score, labels):
    """Shared [0, 1] blue/yellow/red scale, not per-image min/max."""
    values = np.clip(np.nan_to_num(score, nan=0.), 0., 1.)
    # Color anchors at score 0, .5, 1, linearly interpolated.
    anchors = np.asarray(((30, 65, 160), (245, 215, 90), (180, 25, 45)), dtype=np.float32)
    left = np.where((values < .5)[..., None], anchors[0], anchors[1])
    right = np.where((values < .5)[..., None], anchors[1], anchors[2])
    fraction = np.where(values < .5, values * 2, (values - .5) * 2)[..., None]
    panel = (left + (right - left) * fraction).astype(np.uint8)
    panel[(labels == 255) | ~np.isfinite(score)] = 128
    return panel


def _entry_priority(entry):
    labels = entry["labels"]
    geo = entry["predictions"]["geoscd"]
    msd = entry["predictions"][COMPRESSION]
    positives = int(np.count_nonzero(labels == 1))
    lost = int(np.count_nonzero(geo & ~msd & (labels == 1)))
    eligible = entry["case"].get("hypothesis") == "H1"
    return lost / positives if positives and eligible else 0.


def _gallery_selection(entries, limit):
    ordered = sorted(entries, key=lambda e: (-_entry_priority(e),
                     e["case"].get("hypothesis") != "H1", str(e["case"]["case_id"])))
    selected, seen_cases = [], set()
    # Highest-loss representative per facade, in deterministic split/facade order.
    representatives = {}
    for entry in ordered:
        case = entry["case"]
        representatives.setdefault((case["split"], case["building_id"]), entry)
    for key in sorted(representatives, key=lambda k: (SPLITS.index(k[0]), k[1])):
        if len(selected) >= limit:
            break
        entry = representatives[key]
        selected.append(entry)
        seen_cases.add(entry["case"]["case_id"])
    for entry in ordered:
        if len(selected) >= limit:
            break
        if entry["case"]["case_id"] not in seen_cases:
            selected.append(entry)
            seen_cases.add(entry["case"]["case_id"])
    return selected


def _entry_details(entry, threshold, features=()):
    labels = entry["labels"]
    geo = entry["predictions"]["geoscd"]
    msd = entry["predictions"][COMPRESSION]
    positives = labels == 1
    kept, lost = geo & msd & positives, geo & ~msd & positives
    npositive = int(np.count_nonzero(positives))
    ntp = int(np.count_nonzero(geo & positives))
    nlost = int(np.count_nonzero(lost))
    eligible = entry["case"].get("hypothesis") == "H1"
    lines = [f"GT-положительных пикселей {npositive}; GeoSCD TP {ntp}; "
             f"AND сохраняет {int(np.count_nonzero(kept))}; MSDZip отбрасывает {nlost}.",
             "Потеря Recall: " + _pp(nlost / npositive if npositive and eligible else None)
             + "; доля отброшенных GeoSCD TP: " + percent(nlost / ntp if ntp and eligible else None) + "."]
    score = entry.get("scores", {}).get(COMPRESSION)
    raw = entry.get("raw_scores", {}).get(COMPRESSION)
    for region, mask in (("Отброшенные GeoSCD TP", lost), ("Сохранённые GeoSCD TP", kept)):
        parts = []
        for key, array in (("score", score), ("bpb", raw)):
            if array is None:
                continue
            values = np.asarray(array)[mask]
            values = values[np.isfinite(values)]
            if values.size:
                q = np.quantile(values, (.1, .5, .9))
                parts.append(f"{key} p10/медиана/p90 = {q[0]:.4g}/{q[1]:.4g}/{q[2]:.4g}")
        lines.append(region + ": " + ("; ".join(parts) if parts else "нет определённых значений") + ".")
    for row in features:
        if row["group"] not in {"rejected_tp", "kept_tp"}:
            continue
        title = "Отброшенные GeoSCD TP" if row["group"] == "rejected_tp" else "Сохранённые GeoSCD TP"
        n = row["pixel_count"]
        boundary_fraction = row["positive_boundary_pixel_count"] / n if n else None
        rgb = row.get("rgb_abs_meanchannel_mean")
        gap = row.get("threshold_gap_p50")
        lines.append(title + ": на границе GT " + percent(boundary_fraction)
                     + "; средний |RGB B − RGB A| по каналам "
                     + ("н/д" if rgb is None else f"{rgb:.4g} из 255")
                     + "; медиана (порог − score) " + ("н/д" if gap is None else f"{gap:+.4g}") + ".")
    lines.append(f"MSDZip сохраняет пиксель, только если score > {threshold:.8g}; "
                 "равенство порогу означает отбрасывание. GT используется только для диагностики.")
    return lines


def write_gallery(out, entries, summary, gallery_limit, feature_rows=()):
    out = Path(out)
    image_dir = out / "gallery"
    image_dir.mkdir(exist_ok=True)
    selected = _gallery_selection(entries, gallery_limit)
    features_by_case = {}
    for row in feature_rows:
        features_by_case.setdefault(row["case_id"], []).append(row)
    threshold = summary["thresholds"][COMPRESSION]
    if isinstance(threshold, dict):
        threshold = threshold["threshold"]
    pages = ["<!doctype html><html lang='ru'><head><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>Recall: сохранённые карты фасадов</title><style>",
             "body{font:16px system-ui;margin:24px;background:#f2f4f7;color:#20242b;max-width:1500px}"
             "section{background:white;padding:20px;margin:24px 0;border-radius:8px}"
             "pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}"
             ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}"
             "figure{margin:0}img{width:100%;max-width:512px;image-rendering:auto}"
             "figcaption{font-size:14px;margin-top:5px}a{color:#1759ad}"
             "</style></head><body><h1>Recall: что отбрасывает MSDZip</h1>",
             "<p>VAL и TEST, исходные пороги. Галерея выбрана по GT для диагностики: на каждом фасаде "
             "первым показан случай с наибольшей потерей Recall GeoSCD при AND, затем остальные по той же потере "
             "(при равенстве — H1 перед H0, затем по case_id). Это не подбор правила; численные отчёты включают все случаи.</p>",
             f"<p>Показано {len(selected)} из {len(entries)} случаев. "
             "Исходный метод: зелёный — TP, красный — FP, жёлтый — FN. "
             "Карта отбрасывания: оранжевый — утраченные GeoSCD TP; синий — удалённые GeoSCD FP; "
             "зелёный — сохранённые TP; красный — прежние пропуски GeoSCD. Серый — исключённая область.</p>",
             "<p>MSDZip score: общая шкала [0, 1], синий 0 → жёлтый 0.5 → красный 1; "
             f"порог {threshold:.8g}. Значения не растягиваются отдельно по каждой картинке.</p>",
             "<p>Граница GT — положительный пиксель, соседний с известным отрицательным пикселем "
             "в окрестности 8 соседей. RGB-разность вычислена на исходных uint8 RGB; "
             "положительное (порог − score) означает score ниже порога. "
             "Эти признаки описывают связь с отбрасыванием, но не доказывают его причинный механизм.</p>",
             "<p><a href='summary.txt'>Текстовый отчёт</a> · <a href='facades.csv'>Фасады</a> · "
             "<a href='classes.csv'>Классы</a> · <a href='conditions.csv'>Условия</a> · "
             "<a href='recall_cases.csv'>Recall по случаям</a> · "
             "<a href='score_diagnostics.csv'>Диагностика score и bpb</a></p>"]
    for number, entry in enumerate(selected):
        case, labels = entry["case"], entry["labels"]
        caption = " | ".join(str(case.get(key, "")) for key in
                              ("split", "building_id", "state", "nuisance_kind", "scenario_id", "hypothesis", "case_id"))
        gt = np.repeat((labels == 1)[..., None], 3, axis=2).astype(np.uint8) * 255
        gt[labels == 255] = 128
        panels = [("Reference RGB", entry["reference"]), ("Source RGB", entry["source"]), ("GT (evaluation only)", gt)]
        for method in summary["methods"]:
            panels.append((method, _prediction_panel(entry["source"], entry["predictions"][method], labels)))
        panels.append(("Что отброшено: GeoSCD ∩ ¬MSDZip", _loss_panel(entry)))
        msd_scores = entry.get("scores", {}).get(COMPRESSION)
        if msd_scores is not None:
            panels.append((f"MSDZip score [0, 1]; порог {threshold:.8g}", _score_panel(msd_scores, labels)))
            decision = np.repeat((entry["predictions"][COMPRESSION].astype(np.uint8) * 255)[..., None], 3, axis=2)
            decision[labels == 255] = 128
            panels.append(("MSDZip порог: белый — сохраняет, чёрный — отбрасывает", decision))
        pages += ["<section><h2>" + html.escape(caption) + "</h2><p>"
                  + "<br>".join(html.escape(line) for line in _entry_details(
                      entry, threshold, features_by_case.get(case["case_id"], ())))
                  + "</p><div class='grid'>"]
        for panel_number, (title, array) in enumerate(panels):
            filename = f"case-{number:03d}-panel-{panel_number:02d}.png"
            Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB").save(image_dir / filename)
            pages.append(f"<figure><a href='gallery/{filename}'><img loading='lazy' src='gallery/{filename}' "
                         "alt='" + html.escape(title, quote=True) + "'></a><figcaption>"
                         + html.escape(title) + "</figcaption></figure>")
        pages.append("</div></section>")
    pages.append("</body></html>")
    (out / "gallery.html").write_text("\n".join(pages), encoding="utf-8")
    return [entry["case"]["case_id"] for entry in selected]


def write_reports(out, summary, metric_rows, pair_rows, feature_rows, entries, gallery_limit):
    """Write the complete unified analysis; return its printable Russian text."""
    out = Path(out)
    if type(gallery_limit) is not int or gallery_limit < 0:
        raise ValueError("Gallery limit must be a nonnegative integer")
    entries, feature_rows = list(entries), list(feature_rows)
    _write_csv(out / "facades.csv", _method_rows(summary))
    _write_csv(out / "classes.csv", _method_rows(summary, "state"))
    conditions = []
    for dimension in ("nuisance_kind", "scenario_id"):
        for row in _method_rows(summary, dimension):
            value = row.pop(dimension)
            conditions.append({"dimension": dimension, "condition": value, **row})
    _write_csv(out / "conditions.csv", conditions)
    _write_csv(out / "case_metrics.csv", metric_rows)
    _write_csv(out / "recall_cases.csv", pair_rows)
    _write_csv(out / "score_diagnostics.csv", feature_rows)
    text = summary_text(summary)
    (out / "summary.txt").write_text(text, encoding="utf-8")
    gallery_case_ids = write_gallery(out, entries, summary, gallery_limit, feature_rows)
    summary["gallery_selection"] = {"case_ids": gallery_case_ids, "uses_evaluation_labels": True,
                                    "rule": "one maximum GeoSCD AND lost Recall case per split/facade; then remaining cases by maximum lost Recall; ties prefer H1 then case_id",
                                    "used_for_prediction_or_threshold_tuning": False}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return text
