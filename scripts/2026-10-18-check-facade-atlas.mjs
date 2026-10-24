// Optional developer checks: node >=18. No Node packages required.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';
import assert from 'node:assert/strict';

const repo=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const web=path.join(repo,'web/facade_atlas');
const args=process.argv.slice(2);
if(args.length && (args.length!==2 || args[0]!=='--site')) throw Error('Usage: node scripts/2026-10-18-check-facade-atlas.mjs [--site GENERATED_DIRECTORY]');
const read=name=>fs.readFileSync(path.join(web,name),'utf8');
const loadModule=async name=>import('data:text/javascript;base64,'+Buffer.from(read(name)).toString('base64'));
for(const name of fs.readdirSync(web).filter(n=>n.endsWith('.js'))){
  const r=spawnSync(process.execPath,['--input-type=module','--check'],{input:read(name),encoding:'utf8'});
  assert.equal(r.status,0,`${name}: ${r.stderr}`);
}
const {decodeMask,compareMasks,cropPlacement,cropSliderCut,pointToCrop,safeWebUrl}=await loadModule('2026-10-18-logic.js');
const {buildReport}=await loadModule('2026-10-18-reporter.js');

const decoded=decodeMask(new Uint8Array([255,7,255,255,0,0,0,255]),2,1);
assert.deepEqual([...decoded.packed],[2047,0]);assert.deepEqual([...decoded.support],[1,0]);
const a={width:4,height:1,packed:new Uint16Array([0,3,3,1]),support:new Uint8Array([1,1,1,0])};
const b={width:4,height:1,packed:new Uint16Array([1,5,3,2]),support:new Uint8Array([1,1,1,1])};
assert.deepEqual([...compareMasks(a,b).values],[0,3,0,0]);assert.equal(compareMasks(a,b).valid,2);
assert.deepEqual([...compareMasks(a,b,4).values],[0,1,0,0]);assert.deepEqual([...compareMasks(a,b,2).values],[0,2,0,0]);assert.equal(compareMasks(a,b,0).changed,0);
assert.equal(safeWebUrl('javascript:alert(1)'),null);
assert.equal(safeWebUrl('https://example.com/'),'https://example.com/');
const evidence={facade:{frames:[],pairs:[]},categories:[],location:null};
const empty=await buildReport(evidence);assert.equal(empty.observations.length,0);assert.equal(empty.assessment,null);assert.equal(empty.people_property_risk,null);assert.equal(empty.heritage_risk,null);
const replacement={provider:'test',value:7};assert.equal(await buildReport(evidence,async()=>replacement),replacement);
await assert.rejects(()=>buildReport(evidence,async()=>{throw Error('provider_failed');}),/provider_failed/);

const html=read('index.html'),app=read('2026-10-18-app.js');
const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);assert.equal(ids.length,new Set(ids).size,'Duplicate DOM IDs');
for(const match of app.matchAll(/\$\('([^']+)'\)/g))assert.ok(ids.includes(match[1])||match[1]==='retry',`Missing DOM element: ${match[1]}`);
let counts={};
if(args.length){
  const site=path.resolve(args[1]);
  const data=JSON.parse(fs.readFileSync(path.join(site,'data/2026-10-18-facades.json')));
  assert.equal(new Set(data.facades.map(f=>f.id)).size,data.facades.length,'Duplicate building IDs');
  let frames=0,pairs=0,events=0;
  for(const f of data.facades){
    assert.ok(f.frames.length>=2);assert.deepEqual(f.frames.map(r=>r.year),f.frames.map(r=>r.year).sort((a,b)=>a-b));
    assert.equal(new Set(f.frames.map(r=>r.year)).size,f.frames.length);
    const [cx,cy,cw,ch]=f.crop_xywh;assert.ok(cx>=0&&cy>=0&&cx+cw<=f.reference_width&&cy+ch<=f.reference_height);
    for(const r of f.frames){frames++;for(const key of ['original_url','aligned_url','crop_url','mask_url'])assert.ok(fs.existsSync(path.join(site,r[key])),r[key]);}
    for(const p of f.pairs){pairs++;assert.equal(p.to_index,p.from_index+1);assert.equal(f.frames[p.from_index].year,p.from_year);assert.equal(f.frames[p.to_index].year,p.to_year);for(const e of p.events){events++;const [x,y,w,h]=e.bbox;assert.ok(x>=0&&y>=0&&w>0&&h>0&&x+w<=cw&&y+h<=ch);assert.ok(e.area_pixels>=data.contract.event_candidates.minimum_component_pixels);}}
    for(const mode of ['full','crop']){const p=cropPlacement(f,mode);assert.deepEqual(pointToCrop((p.x+.5)/p.width,(p.y+.5)/p.height,p),[0,0]);assert.ok(Math.abs(cropSliderCut(100*(p.x+p.w)/p.width,p)-100)<1e-9);}
    const report=await buildReport({facade:f,categories:data.categories,location:null});assert.equal(report.observations.length,f.pairs.reduce((n,p)=>n+p.events.length,0));assert.equal(report.people_property_risk,null);assert.equal(report.heritage_risk,null);
  }
  counts={facades:data.facades.length,frames,pairs,events};
}
console.log(JSON.stringify({status:'passed',...counts,browser_ui:'not_executed'},null,2));
