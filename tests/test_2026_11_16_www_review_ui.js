// CPU DOM contract: proposals stay private and native coordinates are only scaled for display.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const elements = [], images = [], blobs = [], strokes = [];
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.value = ''; this.checked = false; this.attributes = {}; elements.push(this); }
  set value(v) { this.currentValue = String(v); }
  get value() { return this.currentValue; }
  append(e) { this.children.push(e); }
  set textContent(v) { this.text = v; if (this.tag === 'textarea') this.value = v; }
  get textContent() { return this.text; }
  setAttribute(k, v) { this.attributes[k] = v; }
  getAttribute(k) { return this.attributes[k]; }
  getContext() { return {clearRect(){}, drawImage(){}, fillText(){}, strokeRect(...args){strokes.push(args);}}; }
  getBoundingClientRect() { return this.rect; }
  setPointerCapture(id) { this.pointer = id; }
  releasePointerCapture() { this.pointer = null; }
  remove() { this.removed = true; }
  click() {}
}
class Image {
  constructor() { this.complete = false; this.naturalWidth = 0; images.push(this); }
  set src(v) { this.path = v; }
  getAttribute(k) { return k === 'src' ? this.path : null; }
}
const source = {evidence_id:'archive-image', source_page_url:'https://example.org/photo', observation_date:{start:'2010',end:'2010',precision:'year'},
  image:{src:'archive.png',sha256:'a'.repeat(64),shape:[2000,4000]}, retrieval_eligible:true};
const item = {question_id:'question',building_id:'building',region_id:'region',question:'Question',revision:0,previous_status:'candidate',
  frame:{shape:[16,16]},images:[{date:'2015',observation_id:'local',src:'local.png'}],proposed_boxes:[],positive_annotation_facts:[],
  archive_evidence:[source],label:{assessment:'confirmed_stability',events:[],reference_answer:'Human draft'}};
const payload = {candidate_sha256:'c'.repeat(64),archive_corpus_sha256:'d'.repeat(64),identity_source:null,items:[item]};
const ids = Object.fromEntries(['data','rows','reviewer','message','export','export-correspondence'].map(id=>[id,new Element(id)]));
ids.data.textContent = JSON.stringify(payload);
const document = {getElementById:id=>ids[id],createElement:tag=>new Element(tag)};
const context = vm.createContext({document,Image,Blob:class {constructor(parts){blobs.push(parts.join(''));}},
  URL:{createObjectURL:()=> 'blob:fixture',revokeObjectURL(){}},console});
const html = fs.readFileSync('facade_change/agentic/2026-11-15_www_review.html','utf8');
const script = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
vm.runInContext(script,context);
const editors = vm.runInContext('editors',context), editor = editors[0];
assert.equal(editor.include.checked,false);
assert.throws(()=>vm.runInContext('decisions("fixture-reviewer")',context),/выберите/);
editor.include.checked=true; editor.status.value='approved'; editor.note.value='Explicit human fixture review';
const before = JSON.parse(JSON.stringify(vm.runInContext('decisions("fixture-reviewer")',context)));
assert.deepEqual(before.decisions[0].label.events,[]);
const proposal = JSON.parse(editor.correspondence.value)[0];
proposal.bbox_xyxy=[40,20,100,90]; editor.correspondence.value=JSON.stringify([proposal]);
const archiveDetails = elements.find(e=>e.tag==='details'&&e.ontoggle);
archiveDetails.open=true; archiveDetails.ontoggle();
for (const image of images) { image.complete=true;image.naturalWidth=4000;image.onload(); }
editor.correspondence.oninput();
const projected = strokes.at(-1), expected=[6.4,3.2,9.6,11.2];
assert(expected.every((x,i)=>Math.abs(projected[i]-x)<1e-9));
ids.reviewer.value='fixture-reviewer'; ids['export-correspondence'].onclick();
const saved=JSON.parse(blobs.at(-1));
assert.equal(saved.temporal_gold,false); assert.equal(saved.access,'private_evaluation');
assert.deepEqual(saved.cases[0].proposals[0].bbox_xyxy,[40,20,100,90]);
assert.deepEqual(JSON.parse(JSON.stringify(vm.runInContext('decisions("fixture-reviewer")',context))),before);
const exports=blobs.length; proposal.bbox_xyxy=[0,0,4001,10]; editor.correspondence.value=JSON.stringify([proposal]);
ids['export-correspondence'].onclick(); assert.equal(blobs.length,exports);
assert.match(ids.message.textContent,/Неверная native/);
editor.label.value=JSON.stringify({...item.label,archive_correspondence:[]});
assert.throws(()=>vm.runInContext('decisions("fixture-reviewer")',context),/отдельно/);
console.log('Review UI: explicit inclusion, separate correspondence export, native-to-preview display and bounds passed.');
ids.data.textContent=JSON.stringify({...payload,items:[{...item,task:'earliest_evidence',label:{assessment:'earliest_presence',events:[]}}]});
const scoped=vm.createContext({document,Image,Blob:class{},URL:{createObjectURL:()=> 'blob:fixture',revokeObjectURL(){}}});
vm.runInContext(script,scoped);
const earliest=vm.runInContext('editors[0]',scoped);
earliest.include.checked=true;earliest.status.value='approved';earliest.note.value='Explicit fixture review';
assert.throws(()=>vm.runInContext('decisions("fixture-reviewer")',scoped),/Уточните/);
earliest.scope.value='supplied_observations';
assert.throws(()=>vm.runInContext('decisions("fixture-reviewer")',scoped),/Уточните/);
earliest.wording.value='Earliest evidence among the supplied observations?';
const chosen=vm.runInContext('decisions("fixture-reviewer")',scoped);
assert.equal(chosen.decisions[0].label.question_scope,'supplied_observations');
assert.equal(chosen.decisions[0].reviewed_question,earliest.wording.value);
assert.equal(item.question,'Question');
earliest.item.label.question_scope='supplied_observations';
earliest.item.question=earliest.wording.value;
earliest.scope.value='frozen_collection';
assert.throws(()=>vm.runInContext('decisions("fixture-reviewer")',scoped),/Уточните/);
console.log('Earliest scope: explicit human choice and revised wording required; original question unchanged.');

// Dragging uses the displayed canvas rectangle, but stores native crop pixels.
const drawItem={...item,frame:{shape:[80,160]},label:{...item.label,assessment:'confirmed_change'}};
ids.data.textContent=JSON.stringify({...payload,items:[drawItem]});
const drawing=vm.createContext({document,Image,Blob:class{},URL:{createObjectURL:()=>'',revokeObjectURL(){}}});
const firstElement=elements.length,firstImage=images.length;
vm.runInContext(script,drawing);
const drawEditor=vm.runInContext('editors[0]',drawing);
const canvas=elements.slice(firstElement).find(e=>e.tag==='canvas');
canvas.rect={left:100,top:200,width:640,height:320};
const localImage=images[firstImage];
const pointer=(x,y,id=1)=>({clientX:x,clientY:y,pointerId:id,button:0,preventDefault(){}});
canvas.onpointerdown(pointer(436,368));
assert.equal(canvas.pointer,undefined); // An unloaded image cannot be annotated.
localImage.complete=true;localImage.naturalWidth=160;localImage.naturalHeight=80;
canvas.onpointerdown(pointer(436,368));
canvas.onpointermove(pointer(116,208));
canvas.onpointerup(pointer(116,208));
assert.equal(drawEditor.boxes.length,1);
const box=drawEditor.boxes[0];
const coordinates=()=>Array.from(box.coords,v=>Number(v.value));
assert.deepEqual(coordinates(),[4,2,84,42]);
assert.equal(box.confirm.checked,false);
assert.equal(drawEditor.include.checked,false);
assert.equal(drawEditor.wording.value,'Question');
drawEditor.include.checked=true;drawEditor.status.value='approved';drawEditor.note.value='Human fixture';
assert.throws(()=>vm.runInContext('decisions("reviewer")',drawing),/проверенные/);
box.confirm.checked=true;
assert.throws(()=>vm.runInContext('decisions("reviewer")',drawing),/event_id/);
box.event.value='fixture-physical-event';
const approved=vm.runInContext('decisions("reviewer")',drawing).decisions[0];
assert.deepEqual(Array.from(approved.label.events[0].bbox_xyxy),[4,2,84,42]);
assert.equal(approved.label.events[0].coordinate_system,'region_crop_pixels');
assert.equal(approved.label.events[0].region_id,'region');
assert.equal(approved.reviewed_question,'Question');
const redraw=box.row.children.find(e=>e.textContent==='Перерисовать мышью');
redraw.onclick();canvas.onpointerdown(pointer(120,220));
canvas.onpointermove(pointer(240,280));canvas.onpointercancel(pointer(240,280));
assert.deepEqual(coordinates(),[4,2,84,42]);
assert.equal(box.confirm.checked,true); // Cancel leaves the existing decision intact.
canvas.rect={left:100,top:200,width:80,height:40};
canvas.onpointerdown(pointer(110,205));canvas.onpointermove(pointer(500,500));
canvas.onpointerup(pointer(500,500));
assert.deepEqual(coordinates(),[20,10,160,80]);
assert.equal(drawEditor.boxes.length,1);
assert.equal(box.confirm.checked,false); // Editing never retains old confirmation.
assert.throws(()=>vm.runInContext('decisions("reviewer")',drawing),/проверенные/);
canvas.onpointerdown(pointer(110,205));canvas.onpointerup(pointer(110,205));
assert.equal(drawEditor.boxes.length,1); // Clicks do not create zero-area events.
canvas.onpointerdown(pointer(110,205));canvas.onlostpointercapture(pointer(120,210));
canvas.onpointerup(pointer(120,210));
assert.equal(drawEditor.boxes.length,1);
localImage.naturalWidth=161;canvas.onpointerdown(pointer(110,205));
assert.match(ids.message.textContent,/Размер фото/);
localImage.naturalWidth=160;
box.confirm.checked=true;box.coords[0].value='';
assert.throws(()=>vm.runInContext('decisions("reviewer")',drawing),/недопустимая/);
box.row.children.find(e=>e.textContent==='Удалить рамку').onclick();
assert.equal(drawEditor.boxes.length,0);
assert.throws(()=>vm.runInContext('decisions("reviewer")',drawing),/проверенные/);
console.log('Mouse rectangles: scaling, reverse drag, clipping, cancellation, redraw, deletion and explicit approval passed.');
