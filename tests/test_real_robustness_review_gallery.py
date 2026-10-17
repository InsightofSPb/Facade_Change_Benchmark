"""Review UI checks: escaping, threshold semantics, visible recommendation, export."""
import copy
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from scripts.real_eval.robustness_review_gallery import render_gallery


def fixtures():
    categories = [{"id": 3, "name": "Трещина</script>"}, {"id": 8, "name": "Стена"}]
    rows = []
    for identifier, building, changed, known, roi, class_changed in (
            ("a1", "A<script>alert(1)</script>", 60, 100, 20, 10),
            ("b1", "B", 20, 100, 40, 20),
            ("a2", "A<script>alert(1)</script>", 10, 100, 0, 0),
            ("unknown", "B", 0, 0, 0, 0)):
        classes = [{"channel": 0, "class_id": 3, "class_name": categories[0]["name"], "roi_pixels": roi,
                    "changed_pixels": class_changed, "gain_pixels": class_changed, "loss_pixels": 0,
                    "stable_pixels": roi-class_changed, "change_known_percent": 100*class_changed/roi if roi else None,
                    "change_crop_percent": class_changed/2, "coverage_percent": roi/2},
                   {"channel": 1, "class_id": 8, "class_name": "Стена", "roi_pixels": known,
                    "changed_pixels": changed, "gain_pixels": 0, "loss_pixels": changed,
                    "stable_pixels": known-changed, "change_known_percent": changed if known else None,
                    "change_crop_percent": changed/2, "coverage_percent": known/2}]
        rows.append({"pair_id": identifier, "building_id": building, "status": "ready", "sequence_id": identifier,
                     "frame": 1, "view_id": building, "reference_year": 2020, "source_year": 2022,
                     "crop_xywh": [0, 0, 20, 10], "stats": {"total_pixels": 200, "support_pixels": 180,
                     "known_pixels": known, "changed_pixels": changed, "unchanged_pixels": known-changed,
                     "change_known_percent": changed if known else None, "change_crop_percent": changed/2,
                     "coverage_percent": known/2, "coverage_support_percent": 100*known/180, "classes": classes},
                     "previews": {"reference": "../prepared/old.png", "source": "../prepared/new.png", "shadow": "../prepared/shadow.png"},
                     "transition_masks": {"all": f"masks/{identifier}-all.png", "0": f"masks/{identifier}-0.png", "1": f"masks/{identifier}-1.png"}})
    rows.append({"pair_id": "excluded", "building_id": "B", "status": "excluded", "sequence_id": "excluded",
                 "exclusion_reason": "<img src=x onerror=alert(1)>", "frame": 1})
    return rows, categories, {"ready_pairs": 4, "facades": 2, "unknown_pairs": 1, "thresholds": []}


NODE_HARNESS = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
(async()=>{
const page=fs.readFileSync(0,'utf8');
const scripts=[...page.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)].map(x=>x[1]);
assert.equal(scripts.length,2);
const data=JSON.parse(scripts[0]);
const node=()=>({textContent:'',value:'',hidden:false,disabled:false,checked:false,children:[],dataset:{},
 appendChild(child){this.children=this.children.filter(x=>x!==child);this.children.push(child);return child;},
 replaceChildren(...children){this.children=children;},
 addEventListener(event,fn){this[event]=fn;},classList:{toggle(){}}});
const fields=['building','class-filter','denominator','min-change','min-coverage','state','sort','recommended','clear','export','count','threshold-body','stats-context','selected-heading','coverage-label','denominator-note','filter-error','empty','show-masks','cards'];
const els=Object.fromEntries(fields.map(name=>[name,node()]));
Object.assign(els['class-filter'],{value:'all'});els.denominator.value='known';els['min-change'].value='0';els['min-coverage'].value='0';els.state.value='ready';els.sort.value='change_desc';els['show-masks'].checked=true;
els['review-data']={textContent:scripts[0]};
const cards=data.rows.map((row,index)=>{const card=node();card.dataset.index=String(index);
 const box=node();box.value=row.pair_id;box.disabled=row.status!=='ready';
 card.parts={'.choose':box,'.pair-metrics':node(),'.transition':node(),'.mask-link':node(),'.transition-caption':node()};
 card.classRows=(row.stats?.classes||[]).map(c=>({...node(),dataset:{channel:String(c.channel)}}));
 card.querySelector=s=>card.parts[s];card.querySelectorAll=s=>s==='tr[data-channel]'?card.classRows:[];return card;});
const quick=[0,5,10,15,30,50].map(x=>({...node(),dataset:{threshold:String(x)}}));
const alerts=[],links=[],blobs=[];
const document={body:node(),getElementById:id=>els[id],querySelectorAll:s=>s==='.pair'?cards:s==='[data-threshold]'?quick:[],
 createElement:tag=>{const n=node();if(tag==='a')n.click=()=>links.push(n);return n;}};
const context={document,Blob,URL:{createObjectURL:blob=>{blobs.push(blob);return 'blob:test';},revokeObjectURL(){}},alert:x=>alerts.push(x),setTimeout:fn=>fn(),console};
vm.runInNewContext(scripts[1],context,{timeout:1000});
const visible=()=>cards.filter(c=>!c.hidden).map(c=>c.parts['.choose'].value).sort();
const selected=()=>cards.filter(c=>c.parts['.choose'].checked).map(c=>c.parts['.choose'].value).sort();
const change=(id,value)=>{els[id].value=value;(els[id].oninput||els[id].onchange)();};
assert.deepStrictEqual(visible(),['a1','a2','b1']);assert.deepStrictEqual(selected(),[]);
assert.ok(cards[0].parts['.pair-metrics'].textContent.includes('60% известной области · 30% всего кропа'));
assert.equal(els['threshold-body'].children.length,8);assert.equal(els['threshold-body'].children[0].children[1].textContent,'3');
assert.equal(cards[4].parts['.choose'].disabled,true);
els.export.onclick();assert.equal(alerts.length,1);assert.equal(links.length,0);
quick.find(b=>b.dataset.threshold==='15').onclick();assert.deepStrictEqual(visible(),['a1','b1']);
change('min-change','30');assert.deepStrictEqual(visible(),['a1']);
change('class-filter','0');assert.deepStrictEqual(visible(),['a1','b1']);
assert.ok(cards[0].parts['.transition'].src.endsWith('a1-0.png'));
assert.ok(els['coverage-label'].textContent.includes('класса'));
change('min-change','50');assert.deepStrictEqual(visible(),['a1','b1'],'inclusive class ROI cutoff');
change('min-coverage','15');assert.deepStrictEqual(visible(),['b1']);
els.recommended.onclick();assert.deepStrictEqual(selected(),['b1'],'recommendation must respect coverage/class filters');
change('denominator','crop');assert.deepStrictEqual(visible(),[]);
change('min-change','10');assert.deepStrictEqual(visible(),['b1'],'crop denominator and inclusive cutoff');
change('building','A<script>alert(1)</script>');assert.deepStrictEqual(visible(),[]);assert.deepStrictEqual(selected(),['b1']);
assert.ok(els.count.textContent.includes('скрыто фильтром 1'));
els.export.onclick();assert.equal(links.length,1);assert.equal(links[0].download,'reviewed_selection.json');
const selection=JSON.parse(await blobs[0].text());assert.deepStrictEqual(selection.pair_ids,['b1']);assert.equal(selection.manifest_sha256,data.manifest_sha256);assert.equal(selection.reviewed,true);assert.equal(selection.review_filters.denominator,'crop');
els.recommended.onclick();assert.deepStrictEqual(selected(),[],'empty visible recommendation must not select hidden rows');
change('building','');change('state','unknown');assert.deepStrictEqual(visible(),['a2','unknown']);
els.recommended.onclick();assert.deepStrictEqual(selected(),[]);
change('state','excluded');assert.deepStrictEqual(visible(),['excluded']);
change('state','ready');change('class-filter','all');change('min-change','0');change('min-coverage','0');change('denominator','known');
els.recommended.onclick();assert.deepStrictEqual(selected(),['a1','a2','b1']);
const counts={};for(const row of data.rows.filter(r=>selected().includes(r.pair_id)))counts[row.building_id]=(counts[row.building_id]||0)+1;
assert.ok(Object.values(counts).every(n=>n<=2));assert.ok(selected().length<=12);
change('min-change','101');assert.deepStrictEqual(visible(),[]);assert.equal(els['filter-error'].hidden,false);assert.equal(els.recommended.disabled,true);
els.clear.onclick();assert.deepStrictEqual(selected(),[]);assert.equal(alerts.length,1);
console.log(JSON.stringify({ok:true,filter_checks:14,initial_selection:0,export_bound:true,recommendation_visible_only:true}));
})().catch(e=>{console.error(e);process.exit(1)});
"""


class ReviewGalleryTests(unittest.TestCase):
    def test_escaped_page_no_script_injection_and_no_pair_preselection(self):
        rows, categories, summary = fixtures()
        with tempfile.TemporaryDirectory() as directory:
            path = render_gallery(Path(directory), rows, categories, summary, "a"*64)
            text = path.read_text(encoding="utf-8")
        self.assertNotIn("A<script>alert(1)</script>", text)
        self.assertNotIn("<img src=x onerror=alert(1)>", text)
        self.assertIn("A&lt;script&gt;alert(1)&lt;/script&gt;", text)
        scripts = re.findall(r"<script(?:\s[^>]*)?>([\s\S]*?)</script>", text)
        self.assertEqual(len(scripts), 2)
        self.assertEqual(json.loads(scripts[0])["rows"][0]["building_id"], rows[0]["building_id"])
        for tag in re.findall(r"<input\s[^>]*>", text):
            if 'class="choose"' in tag:
                self.assertNotIn("checked", tag)
        self.assertLess(text.index('class="statistics"'), text.index('id="cards"'))
        self.assertIn('src="../prepared/new.png"', text)

    def test_requires_transition_masks_and_relative_image_urls(self):
        rows, categories, summary = fixtures()
        with tempfile.TemporaryDirectory() as directory:
            broken = copy.deepcopy(rows)
            del broken[0]["transition_masks"]["1"]
            with self.assertRaisesRegex(ValueError, "transition masks"):
                render_gallery(Path(directory), broken, categories, summary, "a"*64)
            broken = copy.deepcopy(rows)
            broken[0]["previews"]["source"] = "javascript:alert(1)"
            with self.assertRaisesRegex(ValueError, "relative local paths"):
                render_gallery(Path(directory), broken, categories, summary, "a"*64)

    @unittest.skipUnless(shutil.which("node"), "Node is needed for browser-script behavior checks")
    def test_filters_visible_recommendation_and_manifest_bound_export(self):
        rows, categories, summary = fixtures()
        with tempfile.TemporaryDirectory() as directory:
            path = render_gallery(Path(directory), rows, categories, summary, "b"*64)
            process = subprocess.run([shutil.which("node"), "-e", NODE_HARNESS],
                                     input=path.read_text(encoding="utf-8"), text=True,
                                     capture_output=True, timeout=10)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertTrue(json.loads(process.stdout)["export_bound"])


if __name__ == "__main__":
    unittest.main()
