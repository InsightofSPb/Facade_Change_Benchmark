// CPU DOM check: explicit human judgments only; no factual inference from a class.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const blobs=[];
class Element {
  constructor(tag){this.tag=tag;this.value='';this.children=[];}
  append(e){this.children.push(e);}
  set textContent(v){this.text=v;if(this.tag==='textarea')this.value=v;}
  get textContent(){return this.text;}
  setAttribute(){}
  click(){}
}
const ids=Object.fromEntries(['data','rows','reviewer','message','export'].map(id=>[id,new Element(id)]));
const options={target_event_match:['','matches','different_event','unresolved','not_applicable'],claim_support:['','supported','contradicted','unresolved'],citation_support:['','supports','contradicts','unresolved','no_citations']};
ids.data.textContent=JSON.stringify({review_id:'frozen-review-digest',judgment_options:options,cases:[
 {blind_id:'anonymous-one',question:'Fixture question',execution_status:'completed',decision:'change',answer:{answer_text:'Fixture description'},evidence:[],citation_ids:[]},
 {blind_id:'anonymous-two',question:'Fixture question',execution_status:'error',error:'Fixture worker error',evidence:[],citation_ids:[]}]});
const context=vm.createContext({document:{getElementById:id=>ids[id],createElement:tag=>new Element(tag)},
 Blob:class{constructor(parts){blobs.push(parts.join(''));}},URL:{createObjectURL:()=> 'blob:fixture',revokeObjectURL(){}}});
const template=fs.readFileSync('facade_change/agentic/2026-11-16_www_factual_review.html','utf8');
vm.runInContext(template.match(/<script>\s*([\s\S]*?)<\/script>/)[1],context);
const editors=vm.runInContext('editors',context);
assert(editors.every(e=>Object.values(e.fields).every(f=>f.value==='')));
ids.reviewer.value='explicit-fixture-reviewer';ids.export.onclick();
assert.equal(blobs.length,0);
editors[0].fields.target_event_match.value='different_event';
editors[0].notes.value='The class alone does not identify the target event.';
ids.export.onclick();
const result=JSON.parse(blobs[0]);
assert.equal(result.source_review_id,'frozen-review-digest');
assert.equal(result.access,'private_evaluation');
assert.equal(result.cases.length,1);
assert.equal(result.cases[0].target_event_match,'different_event');
assert.equal(result.cases[0].claim_support,'');
assert.equal(result.cases[0].citation_support,'');
assert(!('condition' in result.cases[0]));
assert(editors[1].fields.claim_support.value==='');
console.log('Factual UI: blank judgments stay unreviewed; explicit event judgment exports separately with review binding.');
