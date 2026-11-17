"""Reduced report: independent control/stack evidence and exact saved-array reads."""
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr
import importlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from facade_change.io import sha256, write_json

report = importlib.import_module("data_analysis.2026-11-08_history_scenarios_report")
previous = importlib.import_module("tests.test_2026-11-08_history_scenarios_report")
storage = importlib.import_module("facade_change.research.2026-11-08_history_scenarios_storage")


class ReducedFixture:
    def __init__(self, root):
        self.root = root
        self.metadata, self.overall, self.grouped, self.predictions = {}, {}, defaultdict(list), {}
        self.rows = []

    def add(self, method, control, val_tp=6, test_tp=6, *, task="direct", parent=None, head=None, test_fp=None):
        meta = dict(method=method, id=method, task=task, control=control, family="convgru", size="large",
                    scenario="C" if method.startswith("C-") else "B" if method.startswith("B-") else "main",
                    target_type="annotation-derived progression proxy" if task == "progression" else "annotation-derived latest change",
                    status="completed", checkpoint=dict(sha256="weights:" + method))
        if parent:
            meta.update(parent_method=parent, fusion_kind=head)
        self.metadata[method] = meta
        for split, tp in (("val", val_tp), ("test", test_tp)):
            fp = 1 if split == "val" else (10 - tp if test_fp is None else test_fp)
            histogram = np.zeros((2, report.metrics.BINS), np.int64)
            histogram[0, 200], histogram[0, 1800] = 90 - fp, fp
            histogram[1, 200], histogram[1, 1800] = 10 - tp, tp
            counts = report.metrics.metrics(histogram, .5)
            row = dict(method=method, building_id=split + "-building", split=split, val_fpr_budget=.05,
                       slice_type="overall", slice="ANY_CHANGE", **counts)
            self.rows.append(row)
            self.overall[method, split, .05] = dict(method=method, **report.fixed_macro([row]))
            target = np.zeros((10, 10), np.uint8)
            target[0] = 1
            name = method + "-" + split + ".npz"
            np.savez(root := self.root / name, target=target, support=np.ones((10, 10), bool),
                     forecast_support=np.ones((10, 10), bool))
            self.grouped[method, split].append(dict(event_id=split + "-event", building_id=split + "-building",
                split=split, method=method, path=name, sha256=sha256(root), view_id="v", bbox_xywh=[0,0,10,10],
                previous_year=2000, current_year=2001, quantizer_step=.1))
        return meta

    def tables(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return report.reduced_tables(self.root, self.metadata, self.overall, self.rows, self.predictions,
                                         self.grouped, dict(bootstrap_repeats=32, quantizer_step=.1))


class ReducedReportTests(unittest.TestCase):
    def test_residual_selected_on_val_and_history_win_does_not_mean_stack_win(self):
        with tempfile.TemporaryDirectory() as temporary:
            f = ReducedFixture(Path(temporary))
            base = "direct-convgru-large-history"
            control = "direct-convgru-large-pair-only"
            f.add(base, "history", val_tp=6, test_tp=9)
            f.add(control, "pair-only", val_tp=4, test_tp=3)
            f.add(base + "-residual", "history", val_tp=9, test_tp=6)
            f.add(control + "-residual", "pair-only", val_tp=4, test_tp=4)
            f.add("stack", "stack", val_tp=8, test_tp=8)
            main, details, comparisons, selected = f.tables()
            self.assertEqual(selected[base], base + "-residual")
            row = next(row for row in main if row["base_method"] == base)
            self.assertEqual(row["control"], control + "-residual")
            self.assertGreater(row["delta_f1"], 0)
            self.assertLess(row["delta_f1_stack"], 0)
            self.assertEqual(row["history_conclusion"], "PRACTICALLY_USEFUL")
            self.assertEqual(row["stack_practical_status"], "NO_GAIN")
            self.assertEqual(row["status"], "NO_GAIN")
            self.assertEqual(row["primary_checkpoint_id"], "weights:" + base)
            self.assertTrue(any(row["relation"] == "working_stack_common_support" for row in comparisons))

    def test_fusion_requires_matching_head_and_B_regularized_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            f = ReducedFixture(Path(temporary))
            f.add("B-history", "history", val_tp=5, test_tp=7)
            f.add("B-regularized", "regularized", val_tp=5, test_tp=6)
            f.add("B-history+score-cnn", "history", val_tp=9, test_tp=8, parent="B-history", head="score-cnn")
            f.add("B-regularized+score-mlp", "regularized", val_tp=7, test_tp=2, parent="B-regularized", head="score-mlp")
            f.add("stack", "stack", test_tp=3)
            main, _, _, selected = f.tables()
            row = next(row for row in main if row["base_method"] == "B-history")
            self.assertEqual(row["control"], "B-regularized+score-cnn")
            self.assertEqual(row["history_conclusion"], "INCOMPLETE_CONTROL")
            self.assertEqual(row["status"], "INCOMPLETE_CONTROL")
            self.assertIsNone(row["delta_f1"])

    def test_primary_checkpoint_nll_and_same_input_attention_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            f = ReducedFixture(Path(temporary))
            base, control = "h0-convgru-large-history", "h0-convgru-large-last-only"
            for method, kind, tp in ((base, "history", 6), (control, "last-only", 5)):
                f.add(method, kind, task="h0", test_tp=tp)
                f.add(method + "+cross-attention", kind, task="h0", val_tp=9, test_tp=tp + 1,
                      parent=method, head="cross-attention")
                f.add(method + "+feature-mlp", kind, task="h0", val_tp=7, test_tp=tp,
                      parent=method, head="feature-mlp")
            f.add("h0-copy-last", "copy-last", task="h0")
            f.add("stack", "stack", test_tp=3)
            for method, nll in ((base,8.),(control,6.),("h0-copy-last",7.),(base+"-prediction-best",.01)):
                f.predictions[method,"test"] = dict(U_stable=nll)
            main, _, comparisons, _ = f.tables()
            row = next(row for row in main if row["base_method"] == base)
            self.assertEqual(row["prediction_nll"],8.)
            self.assertFalse(row["prediction_improved"])
            self.assertEqual(row["prediction_checkpoint_id"],"weights:"+base)
            self.assertEqual(row["feature_mlp_control"],base+"+feature-mlp")
            self.assertIsNotNone(row["delta_f1_feature_mlp"])
            self.assertTrue(any(row["relation"]=="attention_vs_same_input_feature_mlp" for row in comparisons))
            self.assertNotEqual(row["status"],"PRACTICALLY_USEFUL")

    def test_progression_has_own_latest_control_and_no_stack_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            f = ReducedFixture(Path(temporary))
            f.add("C-history","history",task="progression",test_tp=7)
            f.add("C-latest","latest",task="progression",test_tp=5)
            f.add("stack","stack",test_tp=8)
            main, _, comparisons, _ = f.tables()
            row=next(row for row in main if row["model"]=="C-history")
            self.assertEqual(row["control"],"C-latest")
            self.assertEqual(row["stack_practical_status"],"N/A")
            self.assertIsNone(row["delta_f1_stack"])
            self.assertFalse(any(row["method"]=="C-history" and row["control"]=="stack" for row in comparisons))

    def test_schema2_matches_legacy_metrics_without_reading_forecast_tensors(self):
        data=previous.Fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            legacy=previous.maps(root,data)
            store=storage.SharedMapStore(root / "schema2")
            modern=[]
            for record in legacy:
                row=dict(record,frame=2,track_id="track")
                with np.load(root / record["path"],allow_pickle=False) as archive:
                    arrays={key:archive[key].copy() for key in archive.files}
                arrays["forecast"]=np.full((128,16,16),7.,np.float32)
                arrays["scale"]=np.full((128,16,16),.3,np.float32)
                score,support=arrays.pop("score"),arrays.pop("support")
                modern.append(store.save(row,record["method"],record,dict(score=score,support=support,extras=arrays)))
            oldrows=[row for row in legacy if row["method"]=="history"]
            newrows=[row for row in modern if row["method"]=="history"]
            olddata=report.SavedMapDataset(root,oldrows,data.categories,data.groups)
            newdata=report.SavedMapDataset(root / "schema2",newrows,data.categories,data.groups)
            _,oldpredict=report._map_predictions(root,oldrows)
            _,newpredict=report._map_predictions(root / "schema2",newrows)
            forbidden=[]
            original=storage.ArrayMap.__getitem__
            def access(mapping,key):
                if key in ("forecast","scale"):
                    forbidden.append(key)
                    raise AssertionError("report metrics must not unpack forecast tensors")
                return original(mapping,key)
            with mock.patch.object(storage.ArrayMap,"__getitem__",access):
                a,_=report.collect_results(olddata,[s for s in olddata.samples if s["split"]=="test"],oldpredict)
                b,_=report.collect_results(newdata,[s for s in newdata.samples if s["split"]=="test"],newpredict)
            self.assertEqual(forbidden,[])
            for building in a:
                np.testing.assert_array_equal(a[building],b[building])

    def test_cached_stack_inputs_never_replace_primary_or_residual_score(self):
        data=previous.Fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            row=next(row for row in data.samples if row["split"]=="test" and "positive" in row["event_id"])
            arrays=data.target(row["event_id"])
            np.savez(root / "map.npz", **arrays, score=np.full((4,4),.9,np.float32), stack_raw=np.zeros((1,4,4),np.float32))
            contract=dict(kind="synthetic_identity",channels=1)
            record=dict(row,method="direct-convgru-large-history",path="map.npz",stack_contract=contract)
            _,predict=report._map_predictions(root,[record])
            hist,_=report.collect_results(data,[row],predict)
            self.assertEqual(report.metrics.metrics(hist[row["building_id"]][0],.5)["tp"],4)
            self.assertEqual(predict(row)["extras"],{})
            record["method"] += "-residual"
            _,predict=report._map_predictions(root,[record])
            self.assertEqual(predict(row)["extras"],{})

    def test_reduced_full_report_streams_every_event_class_and_can_rebuild(self):
        import csv
        data=previous.Fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            manifest=previous.maps(root,data)
            names={"history":"direct-convgru-large-history","pair":"direct-convgru-large-pair-only"}
            for row in manifest:
                row["method"]=names[row["method"]]
                row.update(size="large",scenario="main",checkpoint_sha256="primary:"+row["method"])
            records=[dict(id=name,task="direct",control="history" if old=="history" else "pair-only",family="convgru",
                          size="large",scenario="main",status="completed",checkpoint=dict(sha256="primary:"+name))
                     for old,name in names.items()]
            config=dict(matrix_protocol="reduced-v2",bootstrap_repeats=8,gallery_limit=0,
                        matrix=dict(variants=records,fusion_variants=[],planned_fit_counts=dict(primary=2,total=2)))
            with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                first=report.build_report(data,manifest,root,config,records)
                second=report.report_only(root)
            self.assertEqual(first,second)
            self.assertEqual(first["fit_counts"]["actual_completed"],2)
            self.assertEqual(first["per_event_metrics"]["event_method_pairs"],len(manifest))
            self.assertEqual(first["per_event_metrics"]["rows"],len(manifest)*len(report.metrics.slices(data)))
            with (root / (report.PREFIX+"per_event.csv")).open(encoding="utf-8-sig") as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),first["per_event_metrics"]["rows"])
            self.assertEqual({row["split"] for row in rows},{"train","val","test"})
            self.assertTrue(all(row["metric_checkpoint_id"].startswith("primary:") for row in rows))
            with (root / (report.PREFIX+"prediction_metrics.csv")).open(encoding="utf-8-sig") as stream:
                prediction_rows=list(csv.DictReader(stream))
            self.assertTrue(prediction_rows)
            for row in prediction_rows:
                self.assertEqual(row["metric_checkpoint_id"],row["prediction_checkpoint_id"])
                self.assertEqual(row["producer_id"],row["prediction_producer_id"])
                self.assertTrue(row["producer_id"].startswith("primary:"))
                self.assertEqual(float(row["quantizer_step"]),.1)
            for name in ("metrics.csv","per_building.csv","comparisons.csv"):
                with (root / (report.PREFIX+name)).open(encoding="utf-8-sig") as stream:
                    header=next(csv.reader(stream))
                self.assertIn("metric_checkpoint_id",header)
                self.assertIn("producer_id",header)

    def test_fixed_references_have_contract_identity_without_invented_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            path=root / (report.PREFIX+"h0-copy-last.json")
            write_json(path,dict(scale=[.2],distribution="logistic",quantizer_step=.1,fit_split="train"))
            metadata={"h0-copy-last":dict(method="h0-copy-last",control="copy-last",task="h0"),
                      "stack":dict(method="stack",control="stack",stack_contract=dict(kind="synthetic_identity",channels=1))}
            report._annotate_reference_producers(root,metadata)
            copy=report._checkpoint_fields(metadata["h0-copy-last"],metadata)
            self.assertIsNone(copy["metric_checkpoint_id"])
            self.assertEqual(copy["producer_kind"],"fixed_reference")
            self.assertEqual(copy["reference_contract_sha256"],sha256(path))
            self.assertIn(sha256(path),copy["producer_id"])
            stack=report._checkpoint_fields(metadata["stack"],metadata)
            self.assertIsNone(stack["metric_checkpoint_id"])
            self.assertTrue(stack["producer_id"].startswith("reference:stack:"))

    def test_histogram_summary_never_adapts_unrequested_test_data(self):
        fixture=previous.Fixture()
        class MetadataOnly:
            categories=fixture.categories
            groups=fixture.groups
            @property
            def evaluation_data(self):
                raise AssertionError("Unrequested TEST access")
        histogram=np.zeros((len(report.metrics.slices(fixture)),2,report.metrics.BINS),np.int64)
        histogram[:,0,100]=8
        histogram[:,1,1800]=2
        aggregate,rows=report.summarize_histograms(MetadataOnly(),{"val-building":histogram},.5,split="val")
        self.assertEqual(aggregate[0]["f1"],1.)
        self.assertTrue(rows)

    def test_any_saved_training_event_can_be_visualized_without_inference(self):
        data=previous.Fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            manifest=previous.maps(root,data)
            records=[dict(id="history",task="direct",control="history",family="convgru",size="medium",status="completed"),
                     dict(id="pair",task="direct",control="pair-only",family="convgru",size="medium",status="completed")]
            with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                report.build_report(data,manifest,root,dict(bootstrap_repeats=8,gallery_limit=0),records)
            scientific={row["path"]:sha256(root / row["path"]) for row in manifest}
            event=next(row["event_id"] for row in manifest if row["split"]=="train")
            with mock.patch.object(report,"evaluate_predictor",side_effect=AssertionError("No model inference")),redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                result=report.render_event(root,event,methods=["history"])
            self.assertEqual(result["event_id"],event)
            self.assertFalse(result["inference_performed"])
            self.assertTrue(Path(result["path"]).exists())
            self.assertEqual(scientific,{row["path"]:sha256(root / row["path"]) for row in manifest})
            with self.assertRaisesRegex(ValueError,"Unknown saved event"):
                report.render_event(root,"not-present")


if __name__=="__main__":
    unittest.main()
