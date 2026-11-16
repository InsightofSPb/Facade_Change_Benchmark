// CPU DOM contract: proposals stay private and native coordinates are only scaled for display.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const elements = [], images = [], blobs = [], strokes = [];
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.value = ''; this.checked = false; this.attributes = {}; elements.push(this); }
  append(e) { this.children.push(e); }
  set textContent(v) { this.text = v; if (this.tag === 'textarea') this.value = v; }
  get textContent() { return this.text; }
  setAttribute(k, v) { this.attributes[k] = v; }
  getAttribute(k) { return this.attributes[k]; }
  getContext() { return {clearRect(){}, drawImage(){}, fillText(){}, strokeRect(...args){strokes.push(args);}}; }
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
