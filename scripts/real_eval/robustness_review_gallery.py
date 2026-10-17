"""Static Russian review UI for GT-area filtering; never starts inference."""
from __future__ import annotations

import html
import json
from pathlib import Path
from urllib.parse import urlsplit


def _escape(value):
    return html.escape(str(value), quote=True)


def _url(value):
    if (not isinstance(value, str) or not value or value.startswith(("/", "\\"))
            or "\\" in value or urlsplit(value).scheme or urlsplit(value).netloc):
        raise ValueError("Gallery images must use relative local paths")
    return _escape(value)


def _number(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def _figure(path, caption):
    return (f'<figure><a href="{_url(path)}" target="_blank" rel="noopener">'
            f'<img loading="lazy" src="{_url(path)}" alt="{_escape(caption)}"></a>'
            f'<figcaption>{_escape(caption)}</figcaption></figure>')


def render_gallery(out: Path, rows: list, categories: list, summary: dict, manifest_sha256: str) -> Path:
    """Write gallery.html. All percentages and masks are supplied by CPU preparation."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64
            or any(c not in "0123456789abcdef" for c in manifest_sha256)):
        raise ValueError("Expected SHA-256 of the unchanged prepared manifest")
    ids = [row["pair_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Gallery pair IDs must be unique")
    payload_rows, cards = [], []
    captions = {"reference_classes": "Классы раньше", "source_classes": "Классы позже",
                "exposure": "Позже + экспозиция ×0,75", "shadow": "Позже + мягкая тень"}
    for index, row in enumerate(rows):
        stats = row.get("stats", row.get("statistics"))
        ready = row.get("status") == "ready"
        if ready and not isinstance(stats, dict):
            raise ValueError("Ready gallery rows require GT statistics")
        previews = row.get("previews", {})
        masks = row.get("transition_masks", {})
        for path in [*previews.values(), *masks.values()]:
            _url(path)
        if ready and (not {"reference", "source"} <= previews.keys()
                      or not {"all", *map(str, range(len(categories)))} <= masks.keys()):
            raise ValueError("Ready gallery rows require before/source previews and all class transition masks")
        item = {key: row.get(key) for key in ("pair_id", "building_id", "sequence_id", "frame", "view_id",
                                             "reference_year", "source_year", "crop_xywh", "changed_channels")}
        item.update(status=row.get("status"), stats=stats, transition_masks=masks)
        payload_rows.append(item)
        disabled = "" if ready else " disabled"
        title = f'{row.get("building_id", "—")} · {row.get("reference_year", "?")} → {row.get("source_year", "?")}'
        content = [f'<article class="pair" data-index="{index}" id="pair-{index}">',
                   f'<h2><label><input class="choose" type="checkbox" value="{_escape(row["pair_id"])}"{disabled}> {_escape(title)}</label></h2>',
                   f'<p class="identity">{_escape(row.get("sequence_id", "—"))} · наблюдение {_escape(row.get("frame", "—"))}</p>',
                   '<p class="pair-metrics" aria-live="polite"></p>']
        if ready:
            content += ['<div class="panels">', _figure(previews["reference"], "Раньше — исходный RGB"),
                        _figure(previews["source"], "Позже — исходный RGB"),
                        '<figure><div class="overlay">',
                        f'<img loading="lazy" src="{_url(previews["source"])}" alt="Поздний RGB под маской перехода">',
                        f'<img class="transition" loading="lazy" src="{_url(masks["all"])}" alt="Маска изменения GT">',
                        '</div><figcaption><span class="transition-caption">GT: все классы</span> · ',
                        f'<a class="mask-link" href="{_url(masks["all"])}" target="_blank" rel="noopener">маска отдельно</a>',
                        '</figcaption></figure></div>', '<details class="class-details"><summary>Числа по классам: появление и исчезновение</summary>',
                        '<div class="table-scroll"><table><thead><tr><th>Класс</th><th>Область класса, px</th><th>Появилось, px</th>',
                        '<th>Исчезло, px</th><th>Стабильно, px</th><th>Изменение / область класса, %</th>',
                        '<th>Изменение / кроп, %</th><th>Область класса / кроп, %</th></tr></thead><tbody>']
            for entry in stats["classes"]:
                values = [entry["class_name"], entry["roi_pixels"], entry["gain_pixels"], entry["loss_pixels"],
                          entry["stable_pixels"], _number(entry["change_known_percent"]),
                          _number(entry["change_crop_percent"]), _number(entry["coverage_percent"])]
                content.append(f'<tr data-channel="{int(entry["channel"])}">' + ''.join(f'<td>{_escape(v)}</td>' for v in values) + '</tr>')
            content += ['</tbody></table></div></details>', '<details><summary>Классы и варианты освещения</summary><div class="extra-panels">']
            content.extend(_figure(previews[key], caption) for key, caption in captions.items() if key in previews)
            content += ['</div></details>']
        else:
            content.append(f'<p class="error">Исключено: {_escape(row.get("exclusion_reason", "нет подготовленной пары"))}</p>')
        content.append('</article>')
        cards.append(''.join(content))
    buildings = sorted({str(row.get("building_id", "—")) for row in rows})
    building_options = ''.join(f'<option value="{_escape(b)}">{_escape(b)}</option>' for b in buildings)
    class_options = ''.join(f'<option value="{i}">{_escape(c["name"])}</option>' for i, c in enumerate(categories))
    payload = {"manifest_sha256": manifest_sha256, "rows": payload_rows,
               "categories": categories, "summary": summary}
    encoded = (json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
               .replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
               .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))
    page = _TOP.replace("@BUILDINGS@", building_options).replace("@CLASSES@", class_options)
    page = (page.replace("@READY@", _escape(summary.get("ready_pairs", 0)))
            .replace("@FACADES@", _escape(summary.get("facades", 0)))
            .replace("@UNKNOWN@", _escape(summary.get("unknown_pairs", 0))))
    page += '<main id="cards">' + ''.join(cards) + '</main>'
    page += '<script id="review-data" type="application/json">' + encoded + '</script><script>' + _SCRIPT + '</script></body></html>'
    target = out / "gallery.html"
    target.write_text(page, encoding="utf-8")
    return target


_TOP = '''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Реальные изменения: статистика и выбор пар</title><style>
*{box-sizing:border-box}body{font:15px system-ui,sans-serif;margin:0;background:#f1f4f7;color:#172533;padding:22px;line-height:1.45}
header,.pair,.statistics{background:#fff;padding:20px;border-radius:10px;margin:0 0 18px;border:1px solid #dce4ed}h1{font-size:26px;margin:0 0 12px}h2{font-size:19px;margin:0 0 10px}
p{margin:9px 0}.muted,.identity{color:#526171}.identity{overflow-wrap:anywhere}.badges{display:flex;gap:12px;flex-wrap:wrap}.badge{background:#eef3fa;padding:9px 12px;border-radius:6px}
.toolbar{position:sticky;top:0;z-index:10;padding:14px;background:#e8eef7;border:1px solid #ccd7e5;border-radius:9px;box-shadow:0 3px 8px #0c22351a;margin-bottom:18px}.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:end}.controls label{display:flex;flex-direction:column;gap:4px}.controls input,.controls select{max-width:100%;min-width:110px}.controls input{width:120px}
button,input,select{font:inherit;padding:7px;border:1px solid #a8b7ca;border-radius:5px;background:white;color:inherit}button{cursor:pointer}button:hover{background:#dce8f7}.primary{background:#173e67;color:white;border-color:#173e67}.primary:hover{background:#275b91}button:disabled{cursor:default;opacity:.5}
.actions{display:flex;flex-wrap:wrap;gap:7px;margin-top:12px}.quick{display:flex;gap:5px;flex-wrap:wrap;align-items:center;margin-top:10px}.quick button{padding:4px 10px}.pair[hidden]{display:none}.choose{width:20px;height:20px;vertical-align:middle}.pair-metrics{font-weight:600;background:#f0f5fa;padding:9px;border-radius:6px}
.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.extra-panels{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}figure{margin:0;min-width:0}img{display:block;width:100%;height:auto;border-radius:4px}figcaption{padding:6px 0;color:#415365;font-size:13px}.overlay{position:relative}.overlay .transition{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.hide-masks .transition{visibility:hidden}
details{margin-top:12px}summary{cursor:pointer;padding:7px 0;font-weight:600}.table-scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px 9px;border:1px solid #d9e2ed;text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}th{background:#eaf0f7}tr.selected-class{background:#fff0c7}.legend{display:flex;flex-wrap:wrap;gap:12px}.legend span:before{content:'';display:inline-block;width:12px;height:12px;vertical-align:middle;margin-right:5px;background:var(--color)}.error{color:#9e1c24}#count{margin:10px 0 0;font-weight:600}#empty{padding:15px;background:#fff6da;border-radius:6px;margin-bottom:18px}
@media(max-width:850px){body{padding:10px}.toolbar{position:static}.panels{grid-template-columns:1fr}.extra-panels{grid-template-columns:repeat(2,minmax(0,1fr))}.controls label{max-width:100%}h1{font-size:22px}header,.pair,.statistics{padding:14px}}
</style></head><body>
<header><h1>Реальные изменения: сначала статистика, затем выбор</h1>
<div class="badges"><span class="badge">Подготовлено пар: <b>@READY@</b></span><span class="badge">Фасадов: <b>@FACADES@</b></span><span class="badge">Пар без известного GT: <b>@UNKNOWN@</b></span></div>
<p>Просмотр готовых VAL-кропов PREVIOUS. Фильтры используют GT, а не предсказания моделей. Выбор по изменениям — диагностическая выборка, не оценка всей коллекции. GPU здесь не запускается.</p>
<details><summary>Как считаются проценты и что означает маска</summary>
<p><b>Все классы:</b> изменение — смена аннотированного набора классов. Знаменатель «известная область» — пиксели, где разметка известна в обе даты и геометрия допустима.</p>
<p><b>Один класс:</b> изменение — появление или исчезновение именно этого класса. Его известная область — объединение присутствия класса в обе даты, ограниченное общим известным GT. Изменения другого класса в том же месте сюда не входят.</p>
<p>Знаменатель «весь кроп» включает все его пиксели; числитель учитывает только размеченные изменения. При неполной разметке это не полная доля истинных изменений кропа. Покрытие GT показывается отдельно. Неизвестный пиксель не считается здоровым. Для класса покрытие — его известная область / весь кроп. Проверяйте совмещение: разность аннотаций не заменяет независимую экспертную разметку изменений.</p>
<p>Экспозиция ×0,75 и мягкая тень сохранены заранее. Фильтры и просмотр не меняют RGB, GT, веса или пороги моделей.</p></details></header>
<section class="statistics"><h2>Сколько пар остаётся при разных порогах</h2><p id="stats-context" class="muted"></p>
<div class="table-scroll"><table id="threshold-table"><thead><tr><th>Минимум изменения, %</th><th>Все классы: пары</th><th>Все классы: фасады</th><th id="selected-heading">Выбранный класс: пары</th><th>Выбранный класс: фасады</th></tr></thead><tbody id="threshold-body"></tbody></table></div>
<p class="muted">Порог включительный: ≥. Таблица учитывает выбранный фасад, знаменатель и минимальное покрытие; текущий порог изменения и режим показа карточек к ней не применяются.</p></section>
<section class="toolbar" aria-label="Фильтры и выбор"><div class="controls">
<label>Фасад<select id="building"><option value="">Все фасады</option>@BUILDINGS@</select></label>
<label>Класс<select id="class-filter"><option value="all">Все классы</option>@CLASSES@</select></label>
<label>Знаменатель<select id="denominator"><option value="known">Известная область</option><option value="crop">Весь кроп</option></select></label>
<label>Минимум изменения, %<input id="min-change" type="number" min="0" max="100" step="any" value="0"></label>
<label><span id="coverage-label">Покрытие GT / кроп, % ≥</span><input id="min-coverage" type="number" min="0" max="100" step="any" value="0"></label>
<label>Показать<select id="state"><option value="ready">Пары с известным GT</option><option value="positive">Есть изменение</option><option value="no_event">Без изменения</option><option value="unknown">Нет известной области GT/класса</option><option value="excluded">Исключённые пары</option></select></label>
<label>Порядок<select id="sort"><option value="change_desc">Больше изменений сначала</option><option value="change_asc">Меньше изменений сначала</option><option value="facade">Фасад и дата</option></select></label></div>
<div class="quick">Минимум изменения: <button type="button" data-threshold="0">0%</button><button type="button" data-threshold="5">5%</button><button type="button" data-threshold="10">10%</button><button type="button" data-threshold="15">15%</button><button type="button" data-threshold="30">30%</button><button type="button" data-threshold="50">50%</button></div>
<p id="denominator-note" class="muted"></p><p id="filter-error" class="error" hidden>Введите проценты от 0 до 100.</p>
<div class="legend"><span style="--color:#ff2828">Все классы: изменение</span><span style="--color:#19c85f">Класс: появился</span><span style="--color:#dc2da0">Класс: исчез</span><span style="--color:#999">Штриховка: неизвестно</span><label><input id="show-masks" type="checkbox" checked> Показывать маски</label></div>
<div class="actions"><button id="recommended">Выбрать до 12 из видимых</button><button id="clear">Снять весь выбор</button><button id="export" class="primary">Скачать reviewed_selection.json</button></div>
<p class="muted">Рекомендация заменяет выбор: только видимые пары с изменениями, максимум две на фасад, с разнообразием классов и дат. Фильтрация сохраняет уже выбранные скрытые пары.</p><p id="count" aria-live="polite"></p></section>
<p id="empty" hidden>Под текущие фильтры пары не попали. Уменьшите порог или измените класс/покрытие.</p>
'''


_SCRIPT = r'''
'use strict';
const data=JSON.parse(document.getElementById('review-data').textContent);
const rows=data.rows;
const elements=Object.fromEntries(['building','class-filter','denominator','min-change','min-coverage','state','sort','recommended','clear','export','count','threshold-body','stats-context','selected-heading','coverage-label','denominator-note','filter-error','empty','show-masks','cards'].map(id=>[id,document.getElementById(id)]));
const cards=Array.from(document.querySelectorAll('.pair'));
const boxes=cards.map(card=>card.querySelector('.choose'));
const format=value=>value===null||!Number.isFinite(value)?'—':value.toLocaleString('ru-RU',{maximumFractionDigits:2});
function filters(){const minChange=Number(elements['min-change'].value),minCoverage=Number(elements['min-coverage'].value);return {
 building:elements.building.value,channel:elements['class-filter'].value==='all'?null:Number(elements['class-filter'].value),
 denominator:elements.denominator.value,min_change_percent:minChange,min_coverage_percent:minCoverage,state:elements.state.value,sort:elements.sort.value,
 valid:Number.isFinite(minChange)&&Number.isFinite(minCoverage)&&minChange>=0&&minChange<=100&&minCoverage>=0&&minCoverage<=100};}
function measurement(row,channel,denominator){
 if(row.status!=='ready'||!row.stats)return {changed:0,known:0,total:0,percent:null,coverage:null};
 const s=row.stats,part=channel===null?s:s.classes.find(c=>c.channel===channel);
 if(!part)return {changed:0,known:0,total:s.total_pixels,percent:null,coverage:null};
 const known=channel===null?s.known_pixels:part.roi_pixels;
 return {changed:part.changed_pixels,known,total:s.total_pixels,
 percent:known>0?100*part.changed_pixels/(denominator==='known'?known:s.total_pixels):null,
 coverage:s.total_pixels>0?100*known/s.total_pixels:null};
}
function passes(row,f){
 if(!f.valid||(f.building&&String(row.building_id)!==f.building))return false;
 if(f.state==='excluded')return row.status!=='ready';
 if(row.status!=='ready')return false;
 const m=measurement(row,f.channel,f.denominator);
 if(f.state==='unknown')return m.known===0;
 if(m.known===0||m.percent===null||m.percent+1e-10<f.min_change_percent||m.coverage+1e-10<f.min_coverage_percent)return false;
 if(f.state==='positive'&&m.changed===0)return false;
 if(f.state==='no_event'&&m.changed!==0)return false;
 return true;
}
function chosen(){return boxes.filter(box=>box.checked&&!box.disabled).map(box=>box.value).sort();}
function updateCount(){
 const ids=chosen(),selected=new Set(ids),visible=cards.filter(card=>!card.hidden).length;
 const facades=new Set(rows.filter(row=>selected.has(row.pair_id)).map(row=>row.building_id)).size;
 const visibleSelected=cards.filter((card,i)=>!card.hidden&&boxes[i].checked).length;
 elements.count.textContent=`Видно ${visible} пар. Выбрано ${ids.length} пар / ${facades} фасадов; из них скрыто фильтром ${ids.length-visibleSelected}. ${4*ids.length} возмущённых сравнений на метод + ${ids.length} исходных (кэш при совпадении).`;
 elements.empty.hidden=visible>0;
}
function thresholdCounts(channel,threshold,f){
 const selected=rows.filter(row=>{if(row.status!=='ready'||(f.building&&String(row.building_id)!==f.building)||!f.valid)return false;
 const m=measurement(row,channel,f.denominator);return m.known>0&&m.percent+1e-10>=threshold&&m.coverage+1e-10>=f.min_coverage_percent;});
 return [selected.length,new Set(selected.map(row=>row.building_id)).size];
}
function updateThresholds(f){
 elements['threshold-body'].replaceChildren();
 for(const threshold of [0,1,5,10,15,20,30,50]){
  const global=thresholdCounts(null,threshold,f),specific=thresholdCounts(f.channel,threshold,f);
  const tr=document.createElement('tr');
  for(const value of [threshold,...global,...specific]){const td=document.createElement('td');td.textContent=String(value);tr.appendChild(td);}
  elements['threshold-body'].appendChild(tr);
 }
 const name=f.channel===null?'Все классы':data.categories[f.channel].name;
 elements['selected-heading'].textContent=`${name}: пары`;
 elements['stats-context'].textContent=`Фасад: ${f.building||'все'}. Знаменатель: ${f.denominator==='known'?'известная область':'весь кроп'}. Покрытие ≥ ${format(f.min_coverage_percent)}%. Выбранный класс: ${name}.`;
}
function compareRows(a,b,f){
 if(f.sort!=='facade'){
  const x=measurement(a,f.channel,f.denominator).percent,y=measurement(b,f.channel,f.denominator).percent;
  if(x===null&&y!==null)return 1;if(y===null&&x!==null)return -1;
  if(x!==y)return f.sort==='change_asc'?x-y:y-x;
 }
 const year=value=>value===null||value===''||!Number.isFinite(Number(value))?Infinity:Number(value);
 return String(a.building_id).localeCompare(String(b.building_id),'ru')||year(a.reference_year)-year(b.reference_year)||year(a.source_year)-year(b.source_year)||Number(a.frame)-Number(b.frame)||a.pair_id.localeCompare(b.pair_id);
}
function update(){
 const f=filters();
 elements['filter-error'].hidden=f.valid;elements.recommended.disabled=!f.valid;
 elements['coverage-label'].textContent=f.channel===null?'Покрытие GT / кроп, % ≥':'Область класса / кроп, % ≥';
 const part=f.channel===null?'изменившихся пикселей / известные пиксели обеих дат':'появление + исчезновение класса / известное объединение этого класса';
 elements['denominator-note'].textContent=f.denominator==='known'?`Изменение, % = 100 × (${part}).`:'Изменение, % = 100 × пиксели с размеченным изменением / все пиксели кропа. Покрытие GT проверяется отдельно.';
 if(f.state==='unknown'||f.state==='excluded')elements['denominator-note'].textContent+=' В этом режиме числовые пороги не применяются; рекомендация не выбирает эти пары.';
 if(f.state==='unknown')elements['denominator-note'].textContent+=' Для отдельного класса это может означать его отсутствие в известной разметке либо отсутствие известного GT.';
 cards.forEach((card,i)=>{
  const row=rows[i];card.hidden=!passes(row,f);
  const m=measurement(row,f.channel,f.denominator),metric=card.querySelector('.pair-metrics');
  const knownPercent=m.known>0?100*m.changed/m.known:null,cropPercent=m.known>0?100*m.changed/m.total:null;
  metric.textContent=row.status!=='ready'?'Подготовленная пара недоступна':`Изменение: ${format(knownPercent)}% известной области · ${format(cropPercent)}% всего кропа; ${m.changed} px. Известная область: ${m.known} / ${m.total} px (${format(m.coverage)}% кропа).`;
  if(row.status==='ready'){
   const key=f.channel===null?'all':String(f.channel),mask=row.transition_masks[key];
   card.querySelector('.transition').src=mask;card.querySelector('.mask-link').href=mask;
   card.querySelector('.transition-caption').textContent=f.channel===null?'GT: изменение набора классов':`GT: ${data.categories[f.channel].name} — появление / исчезновение`;
   for(const tr of card.querySelectorAll('tr[data-channel]'))tr.classList.toggle('selected-class',f.channel!==null&&Number(tr.dataset.channel)===f.channel);
  }
 });
 const order=rows.map((row,i)=>({row,index:i})).sort((a,b)=>compareRows(a.row,b.row,f));
 for(const item of order)elements.cards.appendChild(cards[item.index]);
 updateThresholds(f);updateCount();
}
function overlap(a,b){
 if(a.view_id!==b.view_id||a.reference_year!==b.reference_year||a.source_year!==b.source_year)return 0;
 const x=a.crop_xywh,y=b.crop_xywh;if(!x||!y)return Number(a.sequence_id===b.sequence_id);
 const area=Math.max(0,Math.min(x[0]+x[2],y[0]+y[2])-Math.max(x[0],y[0]))*Math.max(0,Math.min(x[1]+x[3],y[1]+y[3])-Math.max(x[1],y[1]));
 return area/Math.max(1,Math.min(x[2]*x[3],y[2]*y[3]));
}
function recommendVisible(){
 const f=filters();if(!f.valid)return;
 const eligible=rows.filter((row,i)=>!cards[i].hidden&&row.status==='ready'&&measurement(row,f.channel,f.denominator).changed>0&&measurement(row,f.channel,f.denominator).known>0);
 const buildings=Array.from(new Set(eligible.map(row=>String(row.building_id)))).sort((a,b)=>a.localeCompare(b,'ru'));
 const selected=[],seen=new Set();
 for(let round=0;round<2;round++)for(const building of buildings){
  if(selected.length>=12)break;
  const same=selected.filter(row=>String(row.building_id)===building);
  const candidates=eligible.filter(row=>String(row.building_id)===building&&!selected.includes(row));
  const rank=row=>{
   const channels=(row.stats.classes||[]).filter(c=>c.changed_pixels>0).map(c=>c.channel);
   const fresh=channels.filter(c=>!seen.has(c)).length;
   const newDate=same.every(old=>old.reference_year!==row.reference_year||old.source_year!==row.source_year);
   return [-fresh,-Number(newDate),Math.max(0,...same.map(old=>overlap(row,old))),-measurement(row,f.channel,f.denominator).changed];
  };
  candidates.sort((a,b)=>{const x=rank(a),y=rank(b);for(let j=0;j<x.length;j++)if(x[j]!==y[j])return x[j]-y[j];return a.pair_id.localeCompare(b.pair_id);});
  if(candidates.length){selected.push(candidates[0]);for(const entry of candidates[0].stats.classes)if(entry.changed_pixels>0)seen.add(entry.channel);}
 }
 const ids=new Set(selected.map(row=>row.pair_id));boxes.forEach(box=>box.checked=!box.disabled&&ids.has(box.value));updateCount();
}
for(const id of ['building','class-filter','denominator','state','sort'])elements[id].onchange=update;
for(const id of ['min-change','min-coverage']){elements[id].oninput=update;elements[id].onchange=update;}
for(const button of document.querySelectorAll('[data-threshold]'))button.onclick=()=>{elements['min-change'].value=button.dataset.threshold;update();};
boxes.forEach(box=>box.addEventListener('change',updateCount));
elements.recommended.onclick=recommendVisible;
elements.clear.onclick=()=>{boxes.forEach(box=>box.checked=false);updateCount();};
elements['show-masks'].onchange=()=>document.body.classList.toggle('hide-masks',!elements['show-masks'].checked);
elements.export.onclick=()=>{
 const ids=chosen();if(!ids.length){alert('Выберите хотя бы одну пару после просмотра.');return;}
 const f=filters();delete f.valid;
 const selection={schema_version:1,kind:'real_robustness_selection',manifest_sha256:data.manifest_sha256,reviewed:true,pair_ids:ids,review_filters:f};
 const blob=new Blob([JSON.stringify(selection,null,2)+'\n'],{type:'application/json'});
 const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download='reviewed_selection.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
update();
'''
