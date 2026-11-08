"""CPU-only, immutable preparation and browser review of real VAL robustness pairs."""
from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from facade_change.augmentations import apply_nuisance
from . import real_gt_metrics
from .robustness_common import atomic_json, atomic_npz, digest, read_json, safe_path, sha256

CONDITION_SPECS = [
    {"id": "exposure", "kind": "exposure", "gain": .75},
    {"id": "shadow", "kind": "shadow", "template": "diagonal", "strength": .35, "edge_width": .035},
]
COLORS = [(230, 70, 70), (55, 150, 225), (240, 180, 40), (110, 190, 70),
          (180, 90, 210), (40, 190, 180), (230, 100, 170), (160, 130, 80),
          (100, 100, 230), (210, 160, 210), (180, 210, 70), (80, 180, 130),
          (250, 135, 45), (125, 90, 180), (170, 80, 105), (65, 130, 150)]


def _validate_category_mapping(categories, mapping):
    """Accept PreparedData's record list and legacy dictionaries without reordering channels."""
    message = "Порядок COCO-категорий не совпадает с каналами битовой маски"
    if isinstance(mapping, dict):
        entries = list(mapping.items())
    elif isinstance(mapping, list):
        if any(not isinstance(row, dict) or set(row) != {"category_id", "channel"} for row in mapping):
            raise ValueError("Некорректная запись category_to_channel")
        entries = [(row["category_id"], row["channel"]) for row in mapping]
    else:
        raise ValueError("category_to_channel должен быть списком записей или словарём")
    normalized = {}
    for category_id, channel in entries:
        if (type(category_id) not in (int, str) or type(channel) is not int
                or str(category_id) in normalized):
            raise ValueError("Неоднозначное соответствие COCO-категорий и каналов")
        normalized[str(category_id)] = channel
    expected = {str(category["id"]): index for index, category in enumerate(categories)}
    if len(expected) != len(categories) or normalized != expected:
        raise ValueError(message)


def _validate_frame(frame):
    rgb, support, packed = (frame[key] for key in ("rgb", "support", "packed"))
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
            or support.dtype != np.bool_ or support.shape != rgb.shape[:2]
            or packed.dtype != np.uint16 or packed.shape != support.shape):
        raise ValueError("Некорректная RGB-решётка, поддержка или COCO-битовая маска")


def _saved_rgb(run, record):
    path = safe_path(run, record["artifact_path"])
    if sha256(path) != record["artifact_sha256"]:
        raise ValueError("Изменился SHA-256 исходной RGB-DIFF карты: " + record["unit_id"])
    # Never deserialize score arrays, original placeholder labels or memory objects.
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in ("reference", "source", "support")}


def _overlap(first, second):
    if (first.get("view_id"), first.get("reference_year"), first.get("source_year")) != (
            second.get("view_id"), second.get("reference_year"), second.get("source_year")):
        return 0.
    a, b = first.get("crop_xywh"), second.get("crop_xywh")
    if not a or not b:
        return float(first["sequence_id"] == second["sequence_id"])
    area = max(0, min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])) * max(0, min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    return area / max(1, min(a[2]*a[3], b[2]*b[3]))


def recommend(pairs, limit=12, per_facade=2):
    """Deterministic GT-only enrichment with facade balance; never reads model scores."""
    groups = defaultdict(list)
    for row in pairs:
        if row["status"] == "ready" and row["positive_pixels"] > 0:
            groups[row["building_id"]].append(row)
    chosen, seen_channels = [], set()
    for _ in range(per_facade):
        for building in sorted(groups):
            candidates = [row for row in groups[building] if row not in chosen]
            if not candidates or len(chosen) >= limit:
                continue
            same = [row for row in chosen if row["building_id"] == building]
            def rank(row):
                years = (row.get("reference_year"), row.get("source_year"))
                new_date = all(years != (old.get("reference_year"), old.get("source_year")) for old in same)
                return (-len(set(row["changed_channels"]) - seen_channels), -int(new_date),
                        max((_overlap(row, old) for old in same), default=0.),
                        -row["positive_pixels"], row["pair_id"])
            best = min(candidates, key=rank)
            chosen.append(best)
            seen_channels.update(best["changed_channels"])
    return {row["pair_id"] for row in chosen}


def _overlay(rgb, labels):
    result = rgb.astype(np.float32).copy()
    result[labels == 1] = result[labels == 1] * .35 + np.array([255, 40, 40]) * .65
    unknown = labels == 255
    yy, xx = np.indices(labels.shape)
    hatch = unknown & (((xx + yy) % 12) < 4)
    result[hatch] = result[hatch] * .3 + np.array([135, 135, 135]) * .7
    return np.rint(np.clip(result, 0, 255)).astype(np.uint8)


def _semantic(rgb, bits, support, count):
    colors = np.zeros_like(rgb, np.float32)
    memberships = np.zeros(bits.shape, np.float32)
    for channel in range(count):
        selected = support & ((bits & (1 << channel)) != 0)
        colors[selected] += COLORS[channel]
        memberships[selected] += 1
    selected = memberships > 0
    output = rgb.astype(np.float32).copy()
    output[selected] = .45 * output[selected] + .55 * colors[selected] / memberships[selected, None]
    return np.rint(np.clip(output, 0, 255)).astype(np.uint8)


def _previews(out, row, arrays, category_count):
    images = {"reference": arrays["reference"], "source": arrays["source"],
              "gt": _overlay(arrays["source"], arrays["labels"]),
              "reference_classes": _semantic(arrays["reference"], arrays["reference_bits"], arrays["support"], category_count),
              "source_classes": _semantic(arrays["source"], arrays["source_bits"], arrays["support"], category_count),
              "exposure": arrays["exposure"], "shadow": arrays["shadow"]}
    paths = {}
    for name, image in images.items():
        relative = f"images/{row['pair_id']}-{name}.png"
        Image.fromarray(image).save(out / relative)
        paths[name] = relative
    return paths


def _gallery(out, manifest, manifest_sha):
    escaped = lambda value: html.escape(str(value), quote=True)
    captions = {"reference": "Раньше — RGB", "source": "Позже — RGB", "gt": "GT: изменение набора классов",
                "reference_classes": "Классы раньше", "source_classes": "Классы позже",
                "exposure": "Позже + экспозиция ×0,75", "shadow": "Позже + мягкая тень"}
    cards = []
    for row in manifest["pairs"]:
        valid = row["status"] == "ready"
        state = "positive" if row.get("positive_pixels", 0) else "no_event" if row.get("negative_pixels", 0) else "unknown"
        if not valid:
            state = "excluded"
        figures = "".join(f'<figure><a href="{escaped(path)}" target="_blank"><img loading="lazy" src="{escaped(path)}" alt="{escaped(captions[name])}"></a><figcaption>{escaped(captions[name])}</figcaption></figure>'
                          for name, path in row.get("previews", {}).items())
        years = f'{row.get("reference_year", "?")} → {row.get("source_year", "?")}'
        disabled = "" if valid else " disabled"
        reason = f'<p class="error">Исключено: {escaped(row.get("exclusion_reason"))}</p>' if not valid else ""
        change_names = [str(manifest["categories"][channel]["name"]) for channel in row.get("changed_channels", [])]
        cards.append(f'<section class="pair" data-building="{escaped(row["building_id"])}" data-state="{state}">'
                     f'<h2><label><input type="checkbox" class="choose" value="{escaped(row["pair_id"])}"{disabled}> '
                     f'{escaped(row["building_id"])} · {escaped(years)}</label></h2>'
                     f'<p>{escaped(row["sequence_id"])} · наблюдение {escaped(row["frame"])} '
                     f'{"· рекомендовано" if row.get("recommended") else ""}</p>'
                     f'<p>Изменение GT: {row.get("positive_pixels", 0)} px; без изменения: {row.get("negative_pixels", 0)} px; '
                     f'не оценивается: {row.get("ignored_pixels", 0)} px. Классы переходов: {escaped(", ".join(change_names) or "—")}.</p>'
                     f'{reason}<div class="panels">{figures}</div></section>')
    options = ''.join(f'<option value="{escaped(value)}">{escaped(value)}</option>'
                      for value in sorted({row["building_id"] for row in manifest["pairs"]}))
    legend = ' '.join(f'<span><i style="background:rgb{COLORS[i]}"></i>{escaped(cat["name"])}</span>'
                      for i, cat in enumerate(manifest["categories"]))
    payload = {"manifest_sha256": manifest_sha, "recommended": [row["pair_id"] for row in manifest["pairs"] if row.get("recommended")],
               "pairs": {row["pair_id"]: {"building_id": row["building_id"], "ready": row["status"] == "ready"} for row in manifest["pairs"]}}
    encoded = json.dumps(payload, ensure_ascii=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    page = '''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Проверка реальных пар перед тестом устойчивости</title><style>
body{font:15px system-ui;margin:22px;background:#f1f3f5;color:#18212b}header,.pair{background:white;padding:18px;margin-bottom:20px;border-radius:8px}
.toolbar{position:sticky;top:0;background:#e9eef5;padding:12px;z-index:5;box-shadow:0 2px 5px #bbb}button,select{font:inherit;margin:4px;padding:7px}h2{font-size:19px}
.panels{display:flex;gap:12px;flex-wrap:wrap}figure{margin:0;width:290px}img{width:100%;height:auto}figcaption{padding:5px}input{width:20px;height:20px}
.legend span{display:inline-block;margin:5px}.legend i{display:inline-block;width:14px;height:14px;margin-right:5px}.error{color:#b00020}.pair[hidden]{display:none}
</style><header><h1>Реальные VAL-пары: просмотр до запуска GPU</h1>
<p>Все доступные пары RGB-DIFF / PREVIOUS из фиксированного снимка; TEST не используется. Галерея не выполняет inference.
Выберите пары вручную или начните с рекомендации. Выбор по GT обогащён изменениями: это диагностический эксперимент, не репрезентативная оценка всей выборки.</p>
<p><b>Красный:</b> изменение аннотированного набора классов. <b>Серая штриховка:</b> неизвестная разметка, вне геометрической поддержки или допуск границы.
Отсутствие аннотации не считается здоровым фоном. Изменение разметки после совмещения требует визуальной проверки; убедитесь, что это не ошибка геометрии.</p>
<p>Накладываемые возмущения: линейная экспозиция ×0,75 и фиксированная мягкая диагональная тень (ослабление до 35%). GT и координаты неизменны.
H0: поздний снимок против его возмущения. Реальная пара: ранний снимок против возмущённого позднего. Нулевой GT допустим; полностью неизвестный GT не даёт реальных P/R/F1.</p>
<p>Откройте изображения кликом для исходного размера. В семантических панелях пересекающиеся классы смешиваются по цвету; метки сохраняются независимо.</p><div class="legend">''' + legend + '''</div></header>
<div class="toolbar"><button id="recommended">Выбрать рекомендованные (до 12)</button><button id="clear">Снять выбор</button>
<select id="building"><option value="">Все фасады</option>''' + options + '''</select>
<select id="state"><option value="">Все пары</option><option value="positive">Есть изменение GT</option><option value="no_event">Нет изменения GT</option><option value="unknown">GT неизвестен</option><option value="excluded">Исключённые</option></select>
<button id="export">Скачать reviewed_selection.json</button><p id="count" aria-live="polite"></p></div>''' + ''.join(cards) + '''
<script id="review-data" type="application/json">''' + encoded + '''</script><script>
const data=JSON.parse(document.getElementById('review-data').textContent);
const boxes=Array.from(document.querySelectorAll('.choose'));
function chosen(){return boxes.filter(b=>b.checked&&!b.disabled).map(b=>b.value).sort();}
function update(){const ids=chosen();const n=ids.length;const facades=new Set(ids.map(id=>data.pairs[id].building_id)).size;
 document.getElementById('count').textContent=`Выбрано ${n} пар / ${facades} фасадов. ${4*n} возмущённых сравнений на метод + ${n} исходных (кэш при совпадении); всего ${5*n} задач на метод. Фильтр не снимает скрытый выбор.`;}
boxes.forEach(b=>b.addEventListener('change',update));
document.getElementById('recommended').onclick=()=>{const ids=new Set(data.recommended);boxes.forEach(b=>b.checked=!b.disabled&&ids.has(b.value));update();};
document.getElementById('clear').onclick=()=>{boxes.forEach(b=>b.checked=false);update();};
function filter(){const b=document.getElementById('building').value,s=document.getElementById('state').value;
 document.querySelectorAll('.pair').forEach(e=>e.hidden=Boolean((b&&e.dataset.building!==b)||(s&&e.dataset.state!==s)));}
document.getElementById('building').onchange=filter;document.getElementById('state').onchange=filter;
document.getElementById('export').onclick=()=>{const ids=chosen();if(!ids.length){alert('Выберите хотя бы одну пару после просмотра.');return;}
 const selection={schema_version:1,kind:'real_robustness_selection',manifest_sha256:data.manifest_sha256,pair_ids:ids,reviewed:true};
 const url=URL.createObjectURL(new Blob([JSON.stringify(selection,null,2)+'\\n'],{type:'application/json'}));
 const a=document.createElement('a');a.href=url;a.download='reviewed_selection.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};update();
</script></html>'''
    (out / "gallery.html").write_text(page, encoding="utf-8")


def prepare(run_path, snapshot_path, out_path, repo_path=None, boundary_tolerance=0, input_adapter=None):
    """Freeze ALL real VAL previous pairs, reconstruct annotation GT, and export review."""
    root, snapshot, out = [Path(value).expanduser().resolve() for value in (run_path, snapshot_path, out_path)]
    repo = Path(repo_path).expanduser().resolve() if repo_path else Path(__file__).resolve().parents[2]
    if type(boundary_tolerance) is not int or boundary_tolerance not in (0, 2):
        raise ValueError("Допуск границ должен быть 0 или 2 пикселя")
    if out.exists() or any(out == p or p in out.parents or out in p.parents for p in (root, snapshot)):
        raise FileExistsError("Нужен новый отдельный каталог подготовки вне исходного запуска и снимка")
    paths = [snapshot / "2026-10-17_source_units.json", snapshot / "2026-10-17_quick_summary.json",
             root / "resolved_config.json", root / "experiment_contract.json"]
    metadata = {str(path): sha256(path) for path in paths}
    records, summary, config, contract = map(read_json, paths)
    if Path(summary["source_run"]).expanduser().resolve() != root:
        raise ValueError("Снимок относится к другому исходному запуску")
    if contract.get("config") != config:
        raise ValueError("Конфигурация отличается от замороженного контракта")
    baseline = [row for row in records if (row.get("domain"), row.get("split"), row.get("method"), row.get("mode"))
                == ("real", "val", "rgb_diff", "previous") and row.get("frame", 0) >= 1]
    indexed = {}
    for row in baseline:
        key = row["sequence_id"], row["frame"]
        if key in indexed or row.get("target") != "step_event" or row.get("variant") != "frozen":
            raise ValueError("Дублирующаяся или некорректная RGB-DIFF единица")
        if row["sequence_id"] not in contract.get("sequence_ids", []):
            raise ValueError("История отсутствует в замороженном контракте")
        if not row.get("building_id"):
            raise ValueError("Для реальной пары не указан фасад")
        indexed[key] = row
    ready = [row for row in baseline if row.get("status") == "completed"]
    selected = sorted({row["sequence_id"] for row in ready})
    if not selected:
        raise ValueError("Нет завершённых реальных VAL-пар RGB-DIFF / PREVIOUS")
    print(f"CPU: восстанавливаем {len(ready)} пар из {len(selected)} реальных VAL-историй; все кандидаты попадут в галерею.", flush=True)
    sequences, loader, provenance = (input_adapter(config, selected, contract) if input_adapter else
                                    real_gt_metrics.default_adapter(config, selected, contract, repo, boundary_tolerance,
                                                                    decoded_cache_bytes=512 * 1024 * 1024))
    if set(sequences) != set(selected):
        raise ValueError("Адаптер восстановил другую выборку историй")
    categories = provenance["categories"]
    if not 1 <= len(categories) <= 16:
        raise ValueError("Нужно от 1 до 16 категорий COCO")
    category_to_channel = provenance.get("category_to_channel", {str(cat["id"]): i for i, cat in enumerate(categories)})
    _validate_category_mapping(categories, category_to_channel)
    out.mkdir(parents=True)
    for directory in ("pairs", "images"):
        (out / directory).mkdir()
    state = {"schema_version": 1, "kind": "real_robustness_preparation", "status": "preparing",
             "source_run": str(root), "snapshot": str(snapshot), "completed_pairs": 0,
             "processed_pairs": 0, "total_pairs": len(baseline)}
    atomic_json(out / "run.json", state)
    pairs = []
    try:
        ordered = sorted(baseline, key=lambda row: (row["building_id"], row["sequence_id"], row["frame"]))
        current_id, cached = None, {}
        for number, source_row in enumerate(tqdm(ordered, desc="Prepare real robustness", unit="pair", dynamic_ncols=True), 1):
            state["processed_pairs"] = number
            identifier, frame = source_row["sequence_id"], source_row["frame"]
            row = {"pair_id": "pair-" + digest([identifier, frame])[:24], "sequence_id": identifier,
                   "frame": frame, "building_id": source_row["building_id"], "source_unit_id": source_row["unit_id"],
                   "recommended": False, "status": "excluded", "positive_pixels": 0, "negative_pixels": 0,
                   "ignored_pixels": 0, "changed_channels": []}
            if source_row.get("status") != "completed":
                row["exclusion_reason"] = "Исходная RGB-DIFF единица не завершена: " + str(source_row.get("status"))
                pairs.append(row)
                atomic_json(out / "run.json", state)
                tqdm.write(f"[{number}/{len(ordered)}] {identifier} frame {frame}: исключено — {row['exclusion_reason']}")
                continue
            sequence = sequences[identifier]
            if sequence.get("split") != "val" or sequence.get("domain") != "real" or sequence.get("building_id") != row["building_id"]:
                raise ValueError("История не является реальной VAL-историей указанного фасада")
            if not 1 <= frame < len(sequence["frames"]):
                raise ValueError("Номер кадра выходит за пределы восстановленной истории")
            if current_id != identifier:
                current_id, cached = identifier, {}
            for index in (frame-1, frame):
                if index not in cached:
                    cached[index] = loader(sequence, index)
                    _validate_frame(cached[index])
            previous, current = cached[frame-1], cached[frame]
            cached = {frame: current}
            support = previous["support"] & current["support"]
            saved = _saved_rgb(root, source_row)
            if any(not np.array_equal(saved[name], expected) for name, expected in (
                    ("reference", previous["rgb"]), ("source", current["rgb"]), ("support", support))):
                raise ValueError("Восстановленные RGB/support не совпадают с исходной картой: " + source_row["unit_id"])
            if boundary_tolerance:
                radius = boundary_tolerance
                if "packed_halo" not in previous or "packed_halo" not in current:
                    raise ValueError("Допуск границ требует COCO-контекста за краями кропа")
                labels = real_gt_metrics.derive_labels(previous["packed_halo"], current["packed_halo"],
                    np.pad(support, radius), len(categories), radius)[radius:-radius, radius:-radius].copy()
            else:
                labels = real_gt_metrics.derive_labels(previous["packed"], current["packed"], support, len(categories))
            row.update(view_id=sequence.get("view_id"), crop_xywh=sequence.get("crop_xywh"),
                       reference_year=sequence["frames"][frame-1].get("year"), source_year=sequence["frames"][frame].get("year"),
                       reference_image_id=sequence["frames"][frame-1].get("image_id"), source_image_id=sequence["frames"][frame].get("image_id"),
                       positive_pixels=int((labels == 1).sum()), negative_pixels=int((labels == 0).sum()),
                       ignored_pixels=int((labels == 255).sum()), geometric_pixels=int(support.sum()))
            row["years"] = [row["reference_year"], row["source_year"]]
            changes = previous["packed"] ^ current["packed"]
            row["changed_channels"] = [i for i in range(len(categories)) if np.any((labels == 1) & ((changes & (1 << i)) != 0))]
            if not support.any() or min(support.shape) < 8:
                row["exclusion_reason"] = "Нет общей поддержки или кроп меньше 8 пикселей"
                pairs.append(row)
                atomic_json(out / "run.json", state)
                tqdm.write(f"[{number}/{len(ordered)}] {identifier} frame {frame}: исключено — {row['exclusion_reason']}")
                continue
            arrays = {"reference": previous["rgb"], "source": current["rgb"], "support": support, "labels": labels,
                      "reference_bits": previous["packed"], "source_bits": current["packed"]}
            parameters = {}
            for spec in CONDITION_SPECS:
                result = apply_nuisance(current["rgb"], support, spec)
                if not np.array_equal(result["support"], support):
                    raise ValueError("Фотометрическое возмущение изменило поддержку")
                arrays[spec["id"]] = result["rgb"]
                parameters[spec["id"]] = result["parameters"]
            relative = "pairs/" + row["pair_id"] + ".npz"
            atomic_npz(out / relative, arrays)
            row.update(status="ready", artifact_path=relative, artifact_sha256=sha256(out / relative), nuisance_parameters=parameters)
            row["previews"] = _previews(out, row, arrays, len(categories))
            pairs.append(row)
            state["completed_pairs"] += 1
            atomic_json(out / "run.json", state)
            tqdm.write(f"[{number}/{len(ordered)}] {row['building_id']} {identifier} frame {frame}: GT+ {row['positive_pixels']}, GT− {row['negative_pixels']}; сохранено.")
        # A concurrently changed snapshot must never produce a reviewable manifest.
        if any(sha256(path) != expected for path, expected in metadata.items()):
            raise ValueError("Исходные метаданные изменились во время подготовки")
        recommended = recommend(pairs)
        for row in pairs:
            row["recommended"] = row["pair_id"] in recommended
        manifest = {"schema_version": 1, "kind": "real_robustness_prepared", "source_run": str(root), "snapshot": str(snapshot),
                    "repo_path": str(repo), "metadata_sha256": metadata, "source_config": config,
                    "source_contract_sha256": metadata[str(paths[3])], "source_contract": contract,
                    "source_summary": summary, "source_records": records, "categories": categories,
                    "category_to_channel": category_to_channel, "conditions": [spec["id"] for spec in CONDITION_SPECS],
                    "condition_specs": CONDITION_SPECS, "boundary_tolerance": boundary_tolerance,
                    "gt_provenance": provenance, "pairs": pairs,
                    "scope": "real VAL PREVIOUS annotation-set transitions; photometric nuisances; GT-enriched review, no TEST",
                    "recommendation_policy": "up to 12 positives; up to 2/facade; round-robin facades; changed-class/date diversity, then overlap and positive area; no method scores",
                    "preparation_code_sha256": sha256(Path(__file__)),
                    "summary": {"pairs": len(pairs), "ready_pairs": sum(row["status"] == "ready" for row in pairs),
                                "facades": len({row["building_id"] for row in pairs}), "recommended_pairs": len(recommended)}}
        pending_manifest = out / ".manifest.pending.json"
        atomic_json(pending_manifest, manifest)
        manifest_sha = sha256(pending_manifest)
        _gallery(out, manifest, manifest_sha)
        pending_manifest.replace(out / "manifest.json")
        state.update(status="prepared_awaiting_review", manifest_sha256=manifest_sha, **manifest["summary"])
        atomic_json(out / "run.json", state)
        print(f"Готово: {manifest['summary']['ready_pairs']} пар / {manifest['summary']['facades']} фасадов. Галерея: {out / 'gallery.html'}", flush=True)
        return manifest
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(out / "run.json", state)
        raise
