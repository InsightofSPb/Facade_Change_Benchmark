// Review contracts: no automatic approvals, editable drafts, native drawing, exact cohort.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const nodes=[],images=[],ids={};
class Element {
 constructor(tag){this.tag=tag;this.children=[];this.value='';this.dataset={};nodes.push(this)}
 set id(v){ids[v]=this;this.identifier=v}get id(){return this.identifier}
 appendChild(n){this.children.push(n)}replaceChildren(){this.children=[]}
 getContext(){return {clearRect(){},drawImage(){},strokeRect(){},fillRect(){},fillText(){}}}
 getBoundingClientRect(){return {left:10,top:20,width:448,height:448}}
 setPointerCapture(id){this.pointer=id}releasePointerCapture(){this.pointer=null}click(){this.onclick?.()}
 showModal(){}close(){}
}
class Image {constructor(){this.naturalWidth=224;this.naturalHeight=224;images.push(this)}}
for(const id of ['review-data','reviewer','message','progress','navigation','card','filter','search']){const e=new Element(id);e.id=id}
ids.filter.value='all';
const original=process.argv[2]?JSON.parse(fs.readFileSync(process.argv[2],'utf8')):{
 candidate_sha256:'c'.repeat(64),proposal_sha256:'d'.repeat(64),media:{rgb:'data:image/png;base64,AA=='},archive:{},cases:[
 {number:1,question_id:'change',building_id:'building',region_id:'region',shape:[224,224],task:'temporal_change',revision:0,status:'candidate',confidence:'high',recommendation:'review',archive_ids:[],archive_note:'',related_numbers:[],observations:[{id:'old',date:'2013',media:'rgb',classes:[]}],reviewed_question:'Что изменилось?',label:{assessment:'confirmed_change',observability:'visible',reference_answer:'Changed coating',limitations:[],events:[{event_id:'event',region_id:'region',coordinate_system:'region_crop_pixels',bbox_xyxy:[1,2,30,40]}]}},
 {number:2,question_id:'earliest',building_id:'building',region_id:'region2',shape:[224,224],task:'earliest_evidence',revision:0,status:'candidate',confidence:'high',recommendation:'review',archive_ids:[],archive_note:'',related_numbers:[],observations:[{id:'early',date:'2010',media:'rgb',classes:[]}],reviewed_question:'Самое раннее из показанных локальных наблюдений?',label:{assessment:'earliest_presence',observability:'visible',reference_answer:'2010',limitations:[],events:[],present_by:'2010',earliest_observation_id:'early',onset:'unknown',question_scope:'supplied_observations'}}]};
ids['review-data'].textContent=JSON.stringify(original);
const context=vm.createContext({document:{getElementById:id=>ids[id],createElement:tag=>new Element(tag)},Image,localStorage:{setItem(){},getItem(){return null}},__REVIEW_TEST__:true,console});
const html=fs.readFileSync('scripts/2026-11-16-assistant-review.html','utf8');const script=html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];vm.runInContext(script,context);
const run=s=>vm.runInContext(s,context);
assert(run('state.every(s=>s.status==="candidate")'));
assert.throws(()=>run('decisions("fixture")'),/хотя бы/);
run('data.cases.forEach((c,i)=>validateRow(c,state[i]))');
if(process.argv[2]){
 run('data.cases.forEach((c,i)=>{current=i;render()})');
 assert(run('state.every(s=>s.status==="candidate")'));
 console.log('Real payload: '+original.cases.length+' cards validate/render; approvals remain zero.');
}else{
 run('render()');
 let approve=nodes.find(n=>n.textContent==='Подтвердить ответ и выделенные области');approve.click();assert.equal(run('state[0].status'),'approved');
 assert.equal(run('decisions("fixture").decisions.length'),1);
 let answer=nodes.find(n=>n.tag==='textarea'&&n.value==='Changed coating');answer.value='Edited';answer.oninput();assert.equal(run('state[0].status'),'candidate');
 run('drawMode=true');let canvas=nodes.find(n=>n.tag==='canvas');const pointer=(x,y)=>({clientX:x,clientY:y,button:0,pointerId:1});
 canvas.onpointerdown(pointer(90,100));assert.equal(canvas.pointer,undefined);
 images[0].onload();canvas.onpointerdown(pointer(90,100));canvas.onpointermove(pointer(30,40));canvas.onpointerup(pointer(30,40));
 assert.deepEqual(Array.from(run('state[0].label.events[0].bbox_xyxy')),[10,10,40,40]);assert.equal(run('state[0].status'),'candidate');
 const before=JSON.stringify(run('state[0].label.events'));canvas.onpointerdown(pointer(30,40));canvas.onpointermove(pointer(400,400));canvas.onpointercancel();assert.equal(JSON.stringify(run('state[0].label.events')),before);
 run('current=1;render()');run('state[1].label.present_by="2009"');assert.throws(()=>run('validateRow(data.cases[1],state[1])'),/наблюдение/);run('state[1].label.present_by="2010"');
 assert.throws(()=>run('restore({...snapshot(),candidate_sha256:"wrong"})'),/другому пакету/);
 run('restore(snapshot())');assert(run('state.every(s=>s.status==="candidate")'));
 console.log('Assistant review: explicit approval, edit invalidation, drawing/cancel, earliest precision and restore binding passed.');
}
