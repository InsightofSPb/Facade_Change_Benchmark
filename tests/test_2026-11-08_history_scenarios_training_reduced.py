"""Reduced-v2 fit counts, primary-F1 patience and resumable retained winners."""
import importlib
import tempfile
from pathlib import Path
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,"Optional Torch unavailable")
class ReducedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.m=importlib.import_module("facade_change.research.2026-11-08_history_scenarios_training")
        cls.data=importlib.import_module("facade_change.research.2026-11-08_history_scenarios_data")

    def cfg(self,**kwargs):
        return dict(matrix_protocol="reduced-v2",smoke=True,epochs=kwargs.get("epochs",3),patience=kwargs.get("patience",2),
            effective_batch=2,microbatch=1,amp=False,device="cpu",fusion_epochs=1,seed=13)

    def test_exact_18_48_8_and_no_hidden_grid(self):
        config={"matrix_protocol":"reduced-v2"}
        matrix=self.m.matrix_contract(config)
        self.assertEqual(matrix["planned_fit_counts"],dict(primary=18,score_fusion=48,feature_fusion=8,total=74))
        self.assertEqual(len(matrix["variants"]),18)
        self.assertEqual(len(matrix["fusion_variants"]),56)
        self.assertEqual({r["size"] for r in matrix["variants"] if r["scenario"]=="main"},{"large"})
        self.assertEqual({r["id"] for r in matrix["variants"] if r["scenario"]=="B"},{"B-history","B-regularized"})
        self.assertFalse(any(r.get("ablation") or r["control"] in ("mean","repeat-history") for r in matrix["variants"]))
        heads=matrix["fusion_variants"]
        self.assertEqual(sum(r["fit_kind"]=="score-fusion" for r in heads),48)
        self.assertEqual(sum(r["fit_kind"]=="feature-fusion" for r in heads),8)
        self.assertFalse(any("residual" in r["parent_method"] or r["scenario"]=="C" for r in heads))
        self.assertEqual(self.m.options(config)["fusion_epochs"],5)
        for change in ({"epochs":61},{"fusion_epochs":11},{"fusion_epochs":2.5},{"seed":29}):
            with self.assertRaises(ValueError):self.m.options({**config,**change})

    def test_secondary_prediction_improvement_does_not_extend_f1_patience(self):
        dataset=self.data.SyntheticData();config=self.cfg(epochs=6,patience=1)
        spec=next(x for x in self.m.resolve_matrix(config) if x["task"]=="actual-next")
        values=[dict(f1=.2,iou=.1,recall=.5,fpr=.03,threshold=.5,prediction_score=5.),
                dict(f1=.2,iou=.11,recall=.6,fpr=.03,threshold=.5,prediction_score=4.)]
        with tempfile.TemporaryDirectory() as out, mock.patch.object(self.m,"_validation",side_effect=values):
            record=self.m.train_variant(spec,dataset,out,config)
        self.assertEqual(record["completed_epochs"],2)
        self.assertEqual(record["stop_reason"],"early_stopping")
        self.assertEqual(record["checkpoint"]["epoch"],1)
        self.assertEqual(record["prediction_checkpoint"]["epoch"],2)
        self.assertEqual(record["checkpoint"]["metrics"]["prediction_score"],5.)

    def test_resume_restores_optimizer_scheduler_rng_and_uses_retained_weights(self):
        dataset=self.data.SyntheticData();config=self.cfg(epochs=3,patience=4)
        spec=self.m.resolve_matrix(config)[0]
        class Interrupt:
            def event(self,kind,**values):
                if kind=="history_scenarios_epoch" and values["epoch"]==1:raise KeyboardInterrupt()
                return "synthetic-event"
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            reference=self.m.train_variant(spec,dataset,root/"one",config)
            with self.assertRaises(KeyboardInterrupt):self.m.train_variant(spec,dataset,root/"two",config,ledger=Interrupt())
            resumed=self.m.train_variant(spec,dataset,root/"two",config,resume=True)
            self.assertEqual(reference["optimizer_steps"],resumed["optimizer_steps"])
            self.assertEqual(reference["checkpoint"]["epoch"],resumed["checkpoint"]["epoch"])
            a=self.m._load_torch(reference["checkpoint"]["path"])["model"]
            b=self.m._load_torch(resumed["checkpoint"]["path"])["model"]
            for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
            with mock.patch.object(self.m,"build_model",side_effect=AssertionError("Repeated fit")):
                self.assertEqual(self.m.train_variant(spec,dataset,root/"two",config,resume=True),resumed)
            self.assertLessEqual(len(resumed["retained"]),5)

    def test_completed_resume_preserves_full_records_coverage_and_checks_fusion_weights(self):
        from contextlib import redirect_stdout,redirect_stderr
        import io
        dataset=self.data.SyntheticData();config=self.cfg(epochs=1)
        config["fpr_budgets"]=[.05]
        primary=self.m.resolve_matrix(config)[0]
        head=next(s for s in self.m.resolve_fusion_matrix(config)
                  if s["parent_method"]==primary["id"] and s["fusion_kind"]=="logreg")
        store_type=importlib.import_module("facade_change.2026-10-14_temporal_experiment").RunStore
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(self.m,"resolve_matrix",return_value=[primary]), \
             mock.patch.object(self.m,"resolve_fusion_matrix",return_value=[head]), \
             redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            out=Path(directory)/"run";store=store_type(out,config)
            try:
                records=self.m.train_all(dataset,out,config,ledger=store)
                maps=self.m.export_all(dataset,records,out,config,ledger=store)
                self.assertEqual(len(records),2)
                self.assertEqual(self.m.read_json(out/"2026-11-08_progression_review_manifest.json")["test"]["status"],"ELIGIBLE")
                protected=lambda:{str(p.relative_to(out)):(self.m.sha256(p),p.stat().st_mtime_ns)
                    for p in out.rglob("*") if p.is_file() and p.name not in ("events.jsonl","run.json",".run.lock")}
                before=protected()
                with mock.patch.object(self.m,"build_model",side_effect=AssertionError("Repeated model construction")), \
                     mock.patch.object(self.m,"prediction",side_effect=AssertionError("Repeated inference")), \
                     mock.patch.object(self.m,"_progression",side_effect=AssertionError("Reinspected coverage")), \
                     mock.patch.object(dataset,"targets",side_effect=AssertionError("Read GT during completed resume")):
                    resumed=self.m.train_all(dataset,out,config,ledger=store,resume=True)
                    self.assertEqual(resumed,records)
                    self.assertEqual(self.m.export_all(dataset,resumed,out,config,ledger=store,resume=True),maps)
                self.assertEqual(protected(),before)
                self.assertEqual(len(self.m.read_json(out/"2026-11-08_training_records.json")),2)
                fusion=next(r for r in records if r.get("fit_kind")=="score-fusion")
                weights=Path(fusion["checkpoint"]["path"])
                payload=bytearray(weights.read_bytes());payload[-1]^=1;weights.write_bytes(payload)
                with self.assertRaisesRegex(ValueError,"checkpoint hash mismatch"):
                    self.m.export_all(dataset,records,out,config,ledger=store,resume=True)
                with self.assertRaisesRegex(ValueError,"checkpoint hash mismatch"):
                    self.m.train_all(dataset,out,config,ledger=store,resume=True)
            finally:store.close()

    def test_semantic_val_winner_and_no_test_label_access_before_fit(self):
        dataset=self.data.SyntheticData();config=self.cfg(epochs=1)
        spec=next(x for x in self.m.resolve_matrix(config) if x["id"]=="A-learned")
        original=dataset.targets
        def guarded(row):
            if row["split"]=="test":raise AssertionError("TEST target used in primary fit")
            return original(row)
        with tempfile.TemporaryDirectory() as out, mock.patch.object(dataset,"targets",side_effect=guarded):
            record=self.m.train_variant(spec,dataset,out,config)
        self.assertIsNotNone(record["semantic_checkpoint"])
        self.assertIsNotNone(record["checkpoint"]["metrics"]["semantic_miou"])


if __name__=="__main__":unittest.main()
