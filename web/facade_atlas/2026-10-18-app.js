import {decodeMask,compareMasks,cropPlacement,cropSliderCut,pointToCrop,escapeHtml as esc,safeWebUrl} from './2026-10-18-logic.js';
import {buildReport} from './2026-10-18-reporter.js';

const $ = id => document.getElementById(id);
const state = {data:null,locations:new Map(),facade:null,before:0,after:1,view:'full',layer:'none',slider:50,selectedClasses:new Set(),selectedEvent:null,report:null,renderToken:0,masks:null};
const maskCache = new Map();
const fallbackNames = {Ulitsa_Mira_15:'Улица Мира, 15'};
const number = n => new Intl.NumberFormat('ru-RU').format(n);
const nameOf = f => state.locations.get(f.id)?.label || fallbackNames[f.id] || f.id.replaceAll('_',' ');
const categoryOf = id => state.data.categories.find(c=>c.id===Number(id));
const eventKey = event => event.id;
let toastTimer;
function toast(message) { $('toast').textContent=message; $('toast').hidden=false; clearTimeout(toastTimer); toastTimer=setTimeout(()=>$('toast').hidden=true,5500); }
async function fetchJson(url) { const r=await fetch(url); if(!r.ok) throw new Error(`Не удалось загрузить данные (${r.status})`); return r.json(); }
function setPressed(el,active){el.classList.toggle('active',active);el.setAttribute('aria-pressed',String(active));}
function loadNotes(){try{const value=JSON.parse(localStorage.getItem('facade-atlas-notes-v1')||'{}');return value&&typeof value==='object'&&!Array.isArray(value)?value:{};}catch{return {};}}
function notesForCurrent(){const a=loadNotes()[state.facade.id];return Array.isArray(a)?a:[];}
function fullDate(value){const d=new Date(value);return Number.isNaN(d.getTime())?'Дата не указана':d.toLocaleDateString('ru-RU');}

async function init(){
  const results=await Promise.allSettled([fetchJson('data/2026-10-18-facades.json'),fetchJson('data/2026-10-18-locations.json')]);
  if(results[0].status!=='fulfilled') throw results[0].reason;
  state.data=results[0].value;
  if(!state.data.facades?.length) throw new Error('В наборе нет доступных наблюдений');
  if(results[1].status==='fulfilled') for(const l of [...(results[1].value.locations||[]),...(results[1].value.unresolved||[])]) if(l.id) state.locations.set(l.id,l);
  state.selectedClasses=new Set(state.data.categories.map(c=>c.id));
  $('about-counts').textContent=`${state.data.facades.length} выбранных историй, ${state.data.facades.reduce((n,f)=>n+f.frames.length,0)} фотографий и ${state.data.categories.length} классов GT-разметки`;
  $('facade-count').textContent=state.data.facades.length;
  $('class-filters').innerHTML=state.data.categories.map(c=>`<label class="class-chip"><input type="checkbox" data-category="${c.id}" checked><i style="background:${esc(c.color)}"></i>${esc(c.label)}</label>`).join('');
  $('journal-category').innerHTML='<option value="all">Все классы</option>'+state.data.categories.map(c=>`<option value="${c.id}">${esc(c.label)}</option>`).join('');
  bindEvents(); map.init();
  const hash=decodeURIComponent(location.hash.slice(1));
  const initial=state.data.facades.find(f=>f.id===hash)||state.data.facades.find(f=>f.id==='Kamennoostrovskii_61')||state.data.facades[0];
  await selectFacade(initial.id);
  $('loading').hidden=true;$('workspace-content').hidden=false;$('app').setAttribute('aria-busy','false');fitViewer();
}

function renderList(){
  const query=$('search').value.trim().toLocaleLowerCase('ru');const sort=$('sort').value;
  const facades=state.data.facades.filter(f=>`${nameOf(f)} ${f.id}`.toLocaleLowerCase('ru').includes(query));
  facades.sort((a,b)=>sort==='history'?b.frames.length-a.frames.length:sort==='latest'?b.frames.at(-1).year-a.frames.at(-1).year:nameOf(a).localeCompare(nameOf(b),'ru'));
  $('list-summary').textContent=`${facades.length} из ${state.data.facades.length} объектов`;
  $('facade-list').innerHTML=facades.length?facades.map(f=>`<button class="facade-item ${f.id===state.facade?.id?'selected':''}" data-facade="${esc(f.id)}" aria-pressed="${f.id===state.facade?.id}"><img src="${esc(f.frames.at(-1).original_url)}" alt="" loading="lazy"><span class="item-info"><strong>${esc(nameOf(f))}</strong><small>${f.frames[0].year}—${f.frames.at(-1).year} · ${f.frames.length} фото${hasLocation(f)?'':' · адрес уточняется'}</small></span><span class="item-dot" title="Наблюдения доступны"></span></button>`).join(''):'<p class="empty">По этому запросу фасадов нет.</p>';
}
function hasLocation(f){const l=state.locations.get(f.id);return Number.isFinite(l?.lat)&&Number.isFinite(l?.lon);}
async function selectFacade(id){
  const f=state.data.facades.find(f=>f.id===id);if(!f)return;
  state.facade=f;state.before=Math.max(0,f.frames.length-2);state.after=f.frames.length-1;state.selectedEvent=null;state.masks=null;
  history.replaceState(null,'',`#${encodeURIComponent(id)}`);
  $('facade-title').textContent=nameOf(f);
  $('facade-number').textContent=`Объект ${String(state.data.facades.indexOf(f)+1).padStart(2,'0')}`;
  const loc=state.locations.get(id),url=safeWebUrl(loc?.source_url);
  $('facade-subtitle').innerHTML=`Доступный вид фасада · GT только в выделенной области ${url?` · <a href="${esc(url)}" target="_blank" rel="noopener noreferrer" title="${esc(loc.verification_note||'Координаты здания')}">Адрес на карте ↗</a>`:' · Координаты требуют уточнения'}`;
  $('stat-years').textContent=`${f.frames[0].year}—${f.frames.at(-1).year}`;$('stat-photos').textContent=f.frames.length;
  const options=f.frames.map((r,i)=>`<option value="${i}">${r.year}</option>`).join('');$('year-before').innerHTML=options;$('year-after').innerHTML=options;
  $('missing-years').textContent=f.missing_years.length?`Не вошли в подготовленную историю: ${f.missing_years.join(', ')}. Их состояние здесь не оценивается.`:'';
  renderList();renderNotes();map.render();await renderPair();
}
async function renderPair(){
  const token=++state.renderToken; const f=state.facade,a=f.frames[state.before],b=f.frames[state.after];
  $('year-before').value=String(state.before);$('year-after').value=String(state.after);
  $('before-badge').textContent=a.year;$('after-badge').textContent=b.year;
  $('image-before').src=state.view==='crop'?a.crop_url:a.aligned_url;$('image-after').src=state.view==='crop'?b.crop_url:b.aligned_url;
  $('image-before').alt=`${nameOf(f)}, ${a.year}, ${state.view==='crop'?'участок с разметкой':'совмещённый кадр'}`;
  $('image-after').alt=`${nameOf(f)}, ${b.year}, ${state.view==='crop'?'участок с разметкой':'совмещённый кадр'}`;
  setPressed($('view-full'),state.view==='full');setPressed($('view-crop'),state.view==='crop');
  $('pair-prev').disabled=state.after<=1;$('pair-next').disabled=state.after>=f.frames.length-1;
  renderTimeline();renderJournal();positionLayers();fitViewer();renderSlider();
  for(const id of ['mask-before','mask-after','difference-layer']) $(id).getContext('2d').clearRect(0,0,$(id).width,$(id).height);
  state.masks=null; renderViewerNote();
  try{const masks=await Promise.all([readMask(a.mask_url),readMask(b.mask_url)]);if(token!==state.renderToken)return;state.masks=masks;renderOverlays();}
  catch(e){if(token!==state.renderToken)return;$('viewer-note').textContent='Маски не загрузились. Фотографии доступны; попробуйте выбрать пару ещё раз.';toast(e.message);}
}
function renderTimeline(){
  $('timeline').innerHTML=state.facade.frames.map((r,i)=>`<button class="timeline-item ${i===state.after?'selected':''} ${i===state.before?'base':''}" data-year-index="${i}" aria-label="Сравнить ${r.year} с базовым годом" aria-pressed="${i===state.after}"><img src="${esc(r.crop_url)}" alt="" loading="lazy"><span>${r.year}</span></button>`).join('');
}
function positionLayers(){
  const p=cropPlacement(state.facade,state.view);
  for(const id of ['mask-before','mask-after','difference-layer','crop-boundary']){const e=$(id);e.style.left=`${p.x/p.width*100}%`;e.style.top=`${p.y/p.height*100}%`;e.style.width=`${p.w/p.width*100}%`;e.style.height=`${p.h/p.height*100}%`;}
  $('crop-boundary').hidden=state.view==='crop';
  const event=state.selectedEvent;
  $('event-boundary').hidden=!event;
  if(event){const [x,y,w,h]=event.bbox;Object.assign($('event-boundary').style,{left:`${(p.x+x)/p.width*100}%`,top:`${(p.y+y)/p.height*100}%`,width:`${w/p.width*100}%`,height:`${h/p.height*100}%`});}
}
function fitViewer(){if(!state.facade)return;const p=cropPlacement(state.facade,state.view);const maxHeight=window.innerWidth>=1700?660:window.innerWidth<=600?480:window.innerWidth<=1250?520:570;const maxWidth=$('viewer-scroll').clientWidth;if(!maxWidth)return;const width=Math.min(maxWidth,maxHeight*p.width/p.height);$('image-stage').style.width=`${width}px`;$('image-stage').style.height=`${width*p.height/p.width}px`;}
function renderSlider(){
  $('compare-slider').value=String(state.slider);$('image-after').style.clipPath=`inset(0 0 0 ${state.slider}%)`;$('slider-line').style.left=`${state.slider}%`;
  if(!state.facade)return;const cut=cropSliderCut(state.slider,cropPlacement(state.facade,state.view));
  $('mask-before').style.clipPath=`inset(0 ${100-cut}% 0 0)`;$('mask-after').style.clipPath=`inset(0 0 0 ${cut}%)`;
}
async function readMask(url){
  if(maskCache.has(url))return maskCache.get(url);
  const promise=new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>{try{const canvas=document.createElement('canvas');canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.drawImage(image,0,0);resolve(decodeMask(ctx.getImageData(0,0,canvas.width,canvas.height).data,canvas.width,canvas.height));}catch(e){reject(e);}};image.onerror=()=>reject(new Error('Не удалось прочитать маску наблюдения'));image.src=url;});
  maskCache.set(url,promise);promise.catch(()=>maskCache.delete(url));return promise;
}
function paintGT(canvas,mask){
  canvas.width=mask.width;canvas.height=mask.height;const ctx=canvas.getContext('2d');const image=ctx.createImageData(mask.width,mask.height);
  const cats=state.data.categories.filter(c=>state.selectedClasses.has(c.id)).map(c=>({...c,rgb:c.color.match(/[a-f0-9]{2}/gi).map(v=>parseInt(v,16))}));
  for(let i=0;i<mask.packed.length;i++){if(!mask.support[i]||!mask.packed[i])continue;let r=0,g=0,b=0,n=0;for(const c of cats){if(mask.packed[i]&(1<<c.channel)){r+=c.rgb[0];g+=c.rgb[1];b+=c.rgb[2];n++;}}if(n){image.data[i*4]=r/n;image.data[i*4+1]=g/n;image.data[i*4+2]=b/n;image.data[i*4+3]=115;}}
  ctx.putImageData(image,0,0);
}
function renderOverlays(){
  const isGT=state.layer==='gt',isChange=state.layer==='change';$('mask-before').hidden=!isGT;$('mask-after').hidden=!isGT;$('difference-layer').hidden=!isChange;$('class-filters').hidden=state.layer==='none';
  document.querySelectorAll('[data-layer]').forEach(b=>setPressed(b,b.dataset.layer===state.layer));
  if(state.masks){
    if(isGT){paintGT($('mask-before'),state.masks[0]);paintGT($('mask-after'),state.masks[1]);}
    if(isChange){const bits=state.data.categories.filter(c=>state.selectedClasses.has(c.id)).reduce((v,c)=>v|(1<<c.channel),0);const result=compareMasks(...state.masks,bits);const canvas=$('difference-layer');canvas.width=state.masks[0].width;canvas.height=state.masks[0].height;const ctx=canvas.getContext('2d');const image=ctx.createImageData(canvas.width,canvas.height);const palette=[[0,0,0,0],[60,192,142,155],[243,171,73,155],[173,135,231,165]];for(let i=0;i<result.values.length;i++){image.data.set(palette[result.values[i]],i*4);}ctx.putImageData(image,0,0);}
  }
  renderViewerNote();renderSlider();
}
function renderViewerNote(){
  const scope=state.view==='full'?'Совмещённые фотографии. Рамка показывает участок с GT; вне него разметки нет.':'Участок 512 × 512 px в общих координатах. Незаполненная маска не означает отсутствие дефекта.';
  const text=state.layer==='gt'?' Выбранные классы показаны цветом; при пересечении меток цвета смешиваются. Нажмите точку, чтобы прочитать все её метки.':state.layer==='change'?' Зелёный — добавление метки, охра — исчезновение, сиреневый — оба изменения. Только совместно размеченная область выбранных лет.':'';
  $('viewer-note').textContent=scope+text;
}
function journalRows(){
  const scope=$('journal-scope').value,cat=$('journal-category').value;
  const pairs=state.facade.pairs.filter(p=>scope==='history'||(p.from_index===state.before&&p.to_index===state.after));
  return pairs.flatMap(p=>p.events.filter(e=>cat==='all'||e.category_id===Number(cat)).map(e=>({...e,pair:p}))).sort((a,b)=>b.pair.to_year-a.pair.to_year||b.area_pixels-a.area_pixels);
}
function renderJournal(){
  const rows=journalRows();$('journal-count').textContent=rows.length;
  if(!rows.length){const pair=state.facade.pairs.find(p=>p.from_index===state.before&&p.to_index===state.after);$('journal').innerHTML=`<p class="empty">${$('journal-scope').value==='pair'&&!pair?'Журнал подготовлен для соседних наблюдений. Выберите соседние годы или «Вся история». Слой изменений доступен и для этой пары.':'Кандидатов от 64 пикселей по выбранным условиям нет. Это не является оценкой исправности фасада.'}</p>`;return;}
  $('journal').innerHTML=rows.map(e=>{const c=categoryOf(e.category_id);return `<button class="journal-entry ${state.selectedEvent?.id===e.id?'selected':''}" data-event="${esc(e.id)}"><div class="event-meta"><span>${e.pair.from_year} → ${e.pair.to_year}</span><span>КАНДИДАТ</span></div><div class="event-title"><i style="background:${esc(c.color)}"></i>${esc(c.label)}</div><div class="event-description">${e.direction==='added'?'Метка добавлена в разметке':'Метка исчезла из разметки'}</div><div class="event-location">${number(e.area_pixels)} px · участок [${e.bbox[0]}, ${e.bbox[1]}] <span>↗</span></div></button>`;}).join('');
}
async function selectEvent(id){const event=state.facade.pairs.flatMap(pair=>pair.events.map(e=>({...e,pair}))).find(e=>eventKey(e)===id);if(!event)return;state.before=event.pair.from_index;state.after=event.pair.to_index;state.selectedEvent=event;state.view='crop';state.layer='change';state.selectedClasses.add(event.category_id);const checkbox=document.querySelector(`[data-category="${event.category_id}"]`);if(checkbox)checkbox.checked=true;await renderPair();}
function renderNotes(){const notes=notesForCurrent();$('saved-notes').innerHTML=notes.map(n=>`<div class="saved-note"><strong>${esc(n.author)}</strong> <small>· ${esc(fullDate(n.date))} · ${esc(n.years?.join(' → ')||'')}</small><p>${esc(n.text)}</p></div>`).join('');}
function inspectPoint(event){
  if(!state.masks)return;const rect=$('image-stage').getBoundingClientRect();const point=pointToCrop((event.clientX-rect.left)/rect.width,(event.clientY-rect.top)/rect.height,cropPlacement(state.facade,state.view));
  if(!point){toast('Эта область находится вне доступной GT-разметки.');return;}
  const [x,y]=point;const descriptions=state.masks.map((m,i)=>{const at=y*m.width+x;const names=m.support[at]&&m.packed[at]?state.data.categories.filter(c=>m.packed[at]&(1<<c.channel)).map(c=>c.label).join(', '):'нет известной разметки';return `${state.facade.frames[i===0?state.before:state.after].year}: ${names}`;});toast(descriptions.join(' · '));
}

async function openReport(){
  const f=state.facade;const notes=notesForCurrent();const report=await buildReport({facade:f,categories:state.data.categories,location:state.locations.get(f.id)||null,reviewNotes:notes});
  state.report={...report,object_id:f.id,object_label:nameOf(f),coordinate_frame:'prepared_crop_reference',crop_xywh:f.crop_xywh,review_notes:notes};
  $('report-title').textContent=`Ведомость · ${nameOf(f)}`;
  $('report-content').innerHTML=`<div class="report-intro"><p><strong>${esc(report.summary)}</strong></p><p>${esc(report.scope_note)}</p><p>Источник: GT-разметка. Доступные годы: ${f.frames.map(r=>r.year).join(', ')}. Координаты участков ниже заданы внутри области 512 × 512 px.</p></div><div class="report-priority"><strong>Демонстрационный приоритет: ${esc(report.priority.label)}</strong><br>${esc(report.priority.reason)}<br>Люди и имущество: не оценено. Исторический материал: не оценено.</div><table class="report-table"><thead><tr><th>Годы</th><th>Наблюдение</th><th>Расположение</th><th>Фотоматериалы</th></tr></thead><tbody>${report.observations.map(o=>`<tr><td>${o.from_year} → ${o.to_year}</td><td>${esc(o.label)}<br><small>${number(o.area_pixels)} px · требует проверки</small></td><td>x=${o.bbox[0]}, y=${o.bbox[1]}<br>w=${o.bbox[2]}, h=${o.bbox[3]}</td><td><div class="report-photos"><a href="${esc(o.photo_before)}" target="_blank" rel="noopener noreferrer"><img src="${esc(o.photo_before)}" alt="${o.from_year}"></a><a href="${esc(o.photo_after)}" target="_blank" rel="noopener noreferrer"><img src="${esc(o.photo_after)}" alt="${o.to_year}"></a></div></td></tr>`).join('')||'<tr><td colspan="4">Нет включённых кандидатов изменений.</td></tr>'}</tbody></table><div class="report-notes"><h3>Комментарии</h3>${notes.length?notes.map(n=>`<p><strong>${esc(n.author)}</strong> · ${esc(fullDate(n.date))}<br>${esc(n.text)}</p>`).join(''):'<p>Профессиональная оценка не внесена.</p>'}<h3>Основания и ограничения</h3><p>${report.standards.map(s=>`${esc(s.name)}, п. ${esc(s.clause)}`).join('; ')} — ориентиры структуры материалов.</p>${report.limitations.map(l=>`<p>• ${esc(l)}</p>`).join('')}</div>`;
  $('report-dialog').showModal();
}
function downloadReport(){if(!state.report)return;const blob=new Blob([JSON.stringify(state.report,null,2)],{type:'application/json;charset=utf-8'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`2026-10-18-${state.facade.id}-observations.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);}
function showOriginals(){const f=state.facade;$('original-content').innerHTML=[state.before,state.after].map(i=>`<figure><figcaption>${f.frames[i].year} · исходный кадр, без совмещения</figcaption><img src="${esc(f.frames[i].original_url)}" alt="${esc(nameOf(f))}, ${f.frames[i].year}"></figure>`).join('');$('original-dialog').showModal();}

function bindEvents(){
  $('search').addEventListener('input',renderList);$('sort').addEventListener('change',renderList);
  $('facade-list').addEventListener('click',e=>{const b=e.target.closest('[data-facade]');if(b)selectFacade(b.dataset.facade);});
  for(const id of ['year-before','year-after'])$(id).addEventListener('change',()=>{state.before=Number($('year-before').value);state.after=Number($('year-after').value);if(state.before===state.after){if(id==='year-before')state.after=state.before===state.facade.frames.length-1?state.before-1:state.before+1;else state.before=state.after===0?1:state.after-1;}if(state.before>state.after)[state.before,state.after]=[state.after,state.before];state.selectedEvent=null;renderPair();});
  $('timeline').addEventListener('click',e=>{const b=e.target.closest('[data-year-index]');if(!b)return;const i=Number(b.dataset.yearIndex);if(i===state.before){toast('Этот год выбран базовым. Базовый год можно изменить в списке над фотографией.');return;}state.after=i;if(state.before>state.after)[state.before,state.after]=[state.after,state.before];state.selectedEvent=null;renderPair();});
  $('pair-prev').addEventListener('click',()=>{state.after=Math.max(1,state.after-1);state.before=state.after-1;state.selectedEvent=null;renderPair();});
  $('pair-next').addEventListener('click',()=>{state.after=Math.min(state.facade.frames.length-1,state.after+1);state.before=state.after-1;state.selectedEvent=null;renderPair();});
  $('view-full').addEventListener('click',()=>{state.view='full';renderPair();});$('view-crop').addEventListener('click',()=>{state.view='crop';renderPair();});
  $('compare-slider').addEventListener('input',e=>{state.slider=Number(e.target.value);renderSlider();});
  const handle=$('slider-line').firstElementChild;let sliding=false;
  handle.style.pointerEvents='auto';handle.style.cursor='ew-resize';handle.style.touchAction='none';
  handle.addEventListener('pointerdown',e=>{sliding=true;handle.setPointerCapture(e.pointerId);e.preventDefault();e.stopPropagation();});
  handle.addEventListener('pointermove',e=>{if(!sliding)return;const r=$('image-stage').getBoundingClientRect();state.slider=Math.max(0,Math.min(100,(e.clientX-r.left)/r.width*100));renderSlider();});
  handle.addEventListener('pointerup',()=>{sliding=false;});handle.addEventListener('pointercancel',()=>{sliding=false;});handle.addEventListener('click',e=>e.stopPropagation());
  $('reset-view').addEventListener('click',()=>{state.slider=50;state.selectedEvent=null;positionLayers();renderJournal();renderSlider();});
  document.querySelectorAll('[data-layer]').forEach(b=>b.addEventListener('click',()=>{state.layer=b.dataset.layer;renderOverlays();}));
  $('class-filters').addEventListener('change',e=>{if(!e.target.dataset.category)return;const id=Number(e.target.dataset.category);e.target.checked?state.selectedClasses.add(id):state.selectedClasses.delete(id);renderOverlays();});
  $('journal-scope').addEventListener('change',renderJournal);$('journal-category').addEventListener('change',renderJournal);
  $('journal').addEventListener('click',e=>{const b=e.target.closest('[data-event]');if(b)selectEvent(b.dataset.event);});
  $('image-stage').addEventListener('click',inspectPoint);
  $('original-button').addEventListener('click',showOriginals);
  for(const id of ['about-button','data-button'])$(id).addEventListener('click',()=>$('about-dialog').showModal());
  document.querySelectorAll('[data-close]').forEach(b=>b.addEventListener('click',()=>b.closest('dialog').close()));
  document.querySelectorAll('dialog').forEach(d=>d.addEventListener('click',e=>{if(e.target===d){const r=d.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)d.close();}}));
  $('report-button').addEventListener('click',()=>openReport().catch(e=>toast(`Не удалось сформировать отчёт: ${e.message}`)));
  $('report-download').addEventListener('click',downloadReport);$('report-print').addEventListener('click',()=>window.print());
  $('note-form').addEventListener('submit',e=>{e.preventDefault();const author=$('note-author').value.trim(),text=$('note-text').value.trim();if(!author||!text)return;const notes=loadNotes();const current=Array.isArray(notes[state.facade.id])?notes[state.facade.id]:[];notes[state.facade.id]=[...current,{author,text,date:new Date().toISOString(),years:[state.facade.frames[state.before].year,state.facade.frames[state.after].year],source:'user_note'}];try{localStorage.setItem('facade-atlas-notes-v1',JSON.stringify(notes));$('note-text').value='';renderNotes();toast('Комментарий сохранён в этом браузере.');}catch{toast('Браузер не разрешил сохранение. Скопируйте комментарий перед закрытием страницы.');}});
  new ResizeObserver(()=>{fitViewer();map.render();}).observe($('workspace'));
  window.addEventListener('hashchange',()=>{const id=decodeURIComponent(location.hash.slice(1));if(id!==state.facade?.id)selectFacade(id);});
}

const map={zoom:12,center:{lat:59.946,lon:30.325},tiles:new Map(),loaded:false,drag:null,
  project(lat,lon,z=this.zoom){const size=256*2**z,s=Math.sin(lat*Math.PI/180);return {x:(lon+180)/360*size,y:(.5-Math.log((1+s)/(1-s))/(4*Math.PI))*size};},
  unproject(x,y){const size=256*2**this.zoom;return {lon:x/size*360-180,lat:Math.atan(Math.sinh(Math.PI*(1-2*y/size)))*180/Math.PI};},
  entries(){return state.data.facades.filter(hasLocation).map(f=>({f,loc:state.locations.get(f.id)}));},
  init(){
    this.fit();$('map-plus').addEventListener('click',()=>{this.zoom=Math.min(17,this.zoom+1);this.render();});$('map-minus').addEventListener('click',()=>{this.zoom=Math.max(9,this.zoom-1);this.render();});$('map-fit').addEventListener('click',()=>{this.fit();this.render();});
    $('map-markers').addEventListener('click',e=>{const b=e.target.closest('[data-map-id]');if(b)selectFacade(b.dataset.mapId);});
    $('map').addEventListener('pointerdown',e=>{if(e.target.closest('button,a'))return;this.drag={x:e.clientX,y:e.clientY,c:this.project(this.center.lat,this.center.lon)};$('map').setPointerCapture(e.pointerId);$('map').classList.add('dragging');});
    $('map').addEventListener('pointermove',e=>{if(!this.drag)return;this.center=this.unproject(this.drag.c.x-(e.clientX-this.drag.x),this.drag.c.y-(e.clientY-this.drag.y));this.render();});
    const end=()=>{this.drag=null;$('map').classList.remove('dragging');};$('map').addEventListener('pointerup',end);$('map').addEventListener('pointercancel',end);
    this.render();
  },
  fit(){const entries=this.entries();if(!entries.length)return;const xs=entries.map(e=>e.loc.lon),ys=entries.map(e=>e.loc.lat);this.center={lat:(Math.min(...ys)+Math.max(...ys))/2,lon:(Math.min(...xs)+Math.max(...xs))/2};const w=$('map').clientWidth-46,h=$('map').clientHeight-54;this.zoom=9;for(let z=9;z<=16;z++){const a=this.project(Math.min(...ys),Math.min(...xs),z),b=this.project(Math.max(...ys),Math.max(...xs),z);if(Math.abs(a.x-b.x)<=w&&Math.abs(a.y-b.y)<=h)this.zoom=z;else break;}},
  render(){
    if(!state.data)return;const w=$('map').clientWidth,h=$('map').clientHeight;if(!w||!h)return;const c=this.project(this.center.lat,this.center.lon),left=c.x-w/2,top=c.y-h/2;const wanted=new Set();
    for(let x=Math.floor(left/256);x<=Math.floor((left+w)/256);x++)for(let y=Math.floor(top/256);y<=Math.floor((top+h)/256);y++){if(x<0||y<0||x>=2**this.zoom||y>=2**this.zoom)continue;const key=`${this.zoom}/${x}/${y}`;wanted.add(key);let image=this.tiles.get(key);if(!image){image=new Image();image.alt='';image.className='map-tile';image.draggable=false;image.onload=()=>{this.loaded=true;this.status();};image.onerror=()=>{image.style.visibility='hidden';this.status();};image.src=`https://tile.openstreetmap.org/${key}.png`;this.tiles.set(key,image);$('map-tiles').append(image);}image.style.left=`${x*256-left}px`;image.style.top=`${y*256-top}px`;}
    for(const [key,image] of this.tiles)if(!wanted.has(key)){image.remove();this.tiles.delete(key);}
    $('map-markers').innerHTML=this.entries().map(({f,loc})=>{const p=this.project(loc.lat,loc.lon);return `<button class="map-marker ${f.id===state.facade?.id?'selected':''}" data-map-id="${esc(f.id)}" style="left:${p.x-left}px;top:${p.y-top}px" title="${esc(nameOf(f))} · ${f.frames.length} фото" aria-label="${esc(nameOf(f))}, ${f.frames.length} фотографий"><span>${f.frames.length}</span></button>`;}).join('');this.status();
  },
  status(){const n=this.entries().length;$('map-status').textContent=this.loaded?`${n} зданий на карте · число фото`:`${n} зданий · подложка требует сети`;}
};

init().catch(e=>{console.error(e);$('app').setAttribute('aria-busy','false');$('loading').innerHTML=`<div class="error-box"><h2>Не удалось открыть данные</h2><p>${esc(e.message)}</p><p>Обновите страницу. Для локального запуска откройте сайт через HTTP-сервер, а не двойным щелчком по HTML.</p><button class="primary-button" id="retry">Повторить</button></div>`;$('retry').addEventListener('click',()=>location.reload());});
