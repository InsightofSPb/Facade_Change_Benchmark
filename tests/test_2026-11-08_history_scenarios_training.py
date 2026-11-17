"""Scientific matrix, real epoch resume and bounded TRAIN-only fusion contracts."""
import importlib
import tempfile
from pathlib import Path
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional CPU Torch is not installed")
class TrainingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.train = importlib.import_module("facade_change.research.2026-11-08_history_scenarios_training")
        cls.data = importlib.import_module("facade_change.research.2026-11-08_history_scenarios_data")

    def config(self):
        return dict(seed=13, epochs=3, patience=4, effective_batch=2, smoke=True,
                    device="cpu", fusion_epochs=1, amp=False)

    def test_full_matrix_has_both_scales_tasks_controls_and_one_seed(self):
        matrix = self.train.resolve_matrix(dict(seed=13))
        core = [x for x in matrix if x["scenario"] == "main"]
        self.assertEqual(len(core), 60)
        self.assertEqual({x["size"] for x in core}, {"medium", "large"})
        for family in ("convgru", "transformer"):
            for size in ("medium", "large"):
                for task in ("direct", "actual-next", "h0"):
                    controls = {x["control"] for x in core if x["family"] == family and
                                x["size"] == size and x["task"] == task}
                    self.assertEqual(controls, {"history", "mean", "repeat-history",
                                                "pair-only" if task == "direct" else "last-only"})
        with self.assertRaises(ValueError):
            self.train.resolve_matrix(dict(seed=29))

    def test_epoch_resume_restores_optimizer_scheduler_rng_and_exact_parameters(self):
        data = self.data.SyntheticData()
        cfg = self.config()
        spec = self.train.resolve_matrix(cfg)[0]
        class StopAfterEpoch:
            def event(self, kind, **values):
                if kind == "history_scenarios_epoch" and values["epoch"] == 1:
                    raise KeyboardInterrupt()
                return "synthetic-event"
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            complete = self.train.train_variant(spec, data, root / "complete", cfg)
            with self.assertRaises(KeyboardInterrupt):
                self.train.train_variant(spec, data, root / "resume", cfg, ledger=StopAfterEpoch())
            resumed = self.train.train_variant(spec, data, root / "resume", cfg, resume=True)
            self.assertEqual(complete["optimizer_steps"], resumed["optimizer_steps"])
            self.assertEqual(complete["selected_epoch"], resumed["selected_epoch"])
            a = self.train._load_torch(complete["checkpoint"]["path"])
            b = self.train._load_torch(resumed["checkpoint"]["path"])
            self.assertEqual(a["scheduler"], b["scheduler"])
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
            with mock.patch.object(self.train, "build_model", side_effect=AssertionError("Retrained completed model")):
                self.assertEqual(self.train.train_variant(spec, data, root / "resume", cfg, resume=True), resumed)
            changed = {**cfg, "lr": .05}
            with self.assertRaises(ValueError):
                self.train.train_variant(spec, data, root / "resume", changed, resume=True)

    def test_forecast_fit_includes_first_train_transition_and_never_test(self):
        data = self.data.SyntheticData()
        seen = []
        original = data.split_samples
        def split(name, predictive=False):
            seen.append((name, predictive))
            self.assertNotEqual(name, "test")
            return original(name, predictive)
        data.split_samples = split
        cfg = {**self.config(), "epochs": 1}
        spec = next(x for x in self.train.resolve_matrix(cfg) if x["task"] == "actual-next")
        with tempfile.TemporaryDirectory() as out:
            record = self.train.train_variant(spec, data, out, cfg)
        self.assertEqual(record["status"], "completed")
        self.assertIn(("train", True), seen)
        self.assertIsNotNone(record["prediction_checkpoint"])

    def test_normalizer_tamper_rejects_completed_model_resume(self):
        import json
        data=self.data.SyntheticData()
        cfg={**self.config(),"epochs":1}
        spec=self.train.resolve_matrix(cfg)[0]
        with tempfile.TemporaryDirectory() as out:
            self.train.train_variant(spec,data,out,cfg)
            path=Path(out)/"2026-11-08_feature_normalization.json"
            state=json.loads(path.read_text())
            state["mean"][0]+=1.
            path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError,"normalization artifact hash"):
                self.train.train_variant(spec,data,out,cfg,resume=True)

    def test_b_inference_reads_only_pair_even_when_older_features_are_deleted(self):
        data=self.data.SyntheticData()
        norm=self.data.fit_normalizer(data)
        row=max(data.samples,key=lambda r:r["frame"])
        spec=next(s for s in self.train.resolve_matrix(self.config()) if s["id"]=="B-history")
        model=self.train.build_model(spec,data,self.config()).eval()
        with torch.inference_mode():
            reference,_=self.train._forward(model,data,row,spec,norm,"cpu",pair_only=True)
        observations=data.tracks[row["track_id"]]["observations"]
        for observation in observations[:row["frame"]-1]:
            data._features.pop(observation["observation_id"])
        with mock.patch.object(data,"inputs",side_effect=AssertionError("Read full history")), \
             mock.patch.object(data,"targets",side_effect=AssertionError("Read GT")), \
             mock.patch.object(data,"annotation_history",side_effect=AssertionError("Read GT")), \
             torch.inference_mode():
            actual,inputs=self.train._forward(model,data,row,spec,norm,"cpu",pair_only=True)
        self.assertEqual(inputs["features"].shape[1],2)
        torch.testing.assert_close(reference["logits"],actual["logits"],rtol=0,atol=0)

    def test_model_insufficient_data_is_isolated_but_contract_error_is_fatal(self):
        data = self.data.SyntheticData()
        cfg = self.config()
        specs = self.train.resolve_matrix(cfg)[:2]
        report = importlib.import_module("data_analysis.2026-11-08_history_scenarios_report")
        completed = dict(id=specs[1]["id"], spec=specs[1], status="completed")
        with tempfile.TemporaryDirectory() as out, \
             mock.patch.object(self.train, "resolve_matrix", return_value=specs), \
             mock.patch.object(self.train, "train_variant", side_effect=[report.InsufficientData("No VAL positives"), completed]):
            records = self.train.train_all(data, out, cfg)
        self.assertEqual([r["status"] for r in records], ["INSUFFICIENT_DATA", "completed"])
        with tempfile.TemporaryDirectory() as out, \
             mock.patch.object(self.train, "resolve_matrix", return_value=specs), \
             mock.patch.object(self.train, "train_variant", side_effect=ValueError("split leakage")):
            with self.assertRaisesRegex(ValueError, "split leakage"):
                self.train.train_all(data, out, cfg)

    def test_resume_after_early_stop_epoch_does_not_add_optimizer_steps(self):
        data = self.data.SyntheticData()
        cfg = {**self.config(), "epochs": 5, "patience": 1}
        spec = self.train.resolve_matrix(cfg)[0]
        evaluator = lambda *a, **k: dict(recall=.1, fpr=.01, threshold=.5)
        class StopAfterSecond:
            def event(self, kind, **values):
                if kind == "history_scenarios_epoch" and values["epoch"] == 2:
                    raise KeyboardInterrupt()
                return "test-event"
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(KeyboardInterrupt):
                self.train.train_variant(spec,data,root,cfg,evaluator=evaluator,ledger=StopAfterSecond())
            with mock.patch.object(self.train,"_loss",side_effect=AssertionError("Extra resumed epoch")):
                record=self.train.train_variant(spec,data,root,cfg,evaluator=evaluator,resume=True)
        self.assertEqual(record["completed_epochs"],2)
        self.assertEqual(record["stop_reason"],"early_stopping")

    def test_native_anychange_stack_overlap_uses_or_before_nonlinearity(self):
        import numpy as np
        from types import SimpleNamespace
        report = importlib.import_module("data_analysis.2026-11-08_history_scenarios_report")
        contract = dict(method_names=["continuous", "anychange"],
            normalizer=dict(low=[0., 0.], high=[1., 1.]), parameters=[-1., 0., 2.],
            center=[0., 0.], scale=[1., 1.])
        rows = [dict(event_id=str(i), building_id="b", view_id="v", previous_year=2000,
                     current_year=2001, bbox_xywh=[0, 0, 2, 2], known_pixels=4) for i in range(2)]
        target = np.array([[0, 1], [0, 1]], np.uint8)
        arrays = dict(target=target, class_targets=target[None], support=np.ones((2, 2), bool),
                      before=np.ones((2, 2), np.uint16), after=np.ones((2, 2), np.uint16))
        data = SimpleNamespace(samples=rows, categories=[dict(id=1, name="CRACK")], groups=[],
                               target=lambda identifier: arrays, stack_contract=contract)
        def predict(row):
            raw = np.zeros((2, 2, 2), np.float32)
            raw[1] = int(row["event_id"])
            return dict(score=np.zeros((2, 2), np.float32), support=arrays["support"],
                        extras=dict(stack_raw=raw), stack_contract=self.train._stack_contract(data))
        histograms, _ = report.collect_results(data, rows, predict)
        values = report.metrics.metrics(histograms["b"][0], .5)
        self.assertEqual((values["tp"], values["fp"]), (2, 2))
        # Averaging native 0/1 first would give probability .5 and zero detections.

    def test_same_checkpoint_intervention_preserves_last_pair_and_common_support(self):
        data = self.data.SyntheticData()
        normalizer = self.data.fit_normalizer(data)
        row = max(data.samples, key=lambda r: r["frame"])
        spec = self.train.resolve_matrix(self.config())[0]
        a = self.train._inputs(data, row, spec, normalizer, "cpu")
        b = self.train._inputs(data, row, spec, normalizer, "cpu", "repeat")
        torch.testing.assert_close(a["features"][:, -2:], b["features"][:, -2:], rtol=0, atol=0)
        torch.testing.assert_close(a["support"], b["support"], rtol=0, atol=0)
        torch.testing.assert_close(a["years"], b["years"], rtol=0, atol=0)

    def test_six_tasks_export_freeze_before_test_resume_and_report_without_models(self):
        data = self.data.SyntheticData()
        cfg = {**self.config(), "epochs": 1, "bootstrap_repetitions": 8,
               "fpr_budgets": [.05, .10, .15]}
        specs = self.train.resolve_matrix(cfg)
        chosen = [next(s for s in specs if s["task"] == task and s["control"] == control)
                  for task, control in (("direct", "history"), ("direct", "pair-only"),
                      ("actual-next", "history"), ("actual-next", "last-only"),
                      ("h0", "history"), ("h0", "last-only"))]
        chosen += [s for s in specs if s["id"] in ("A-learned", "A-independent", "B-history", "C-history")]
        store_class = importlib.import_module("facade_change.2026-10-14_temporal_experiment").RunStore
        report = importlib.import_module("data_analysis.2026-11-08_history_scenarios_report")
        with tempfile.TemporaryDirectory() as parent:
            out = Path(parent) / "run"
            store = store_class(out, cfg)
            try:
                records = [self.train.train_variant(s, data, out, cfg, ledger=store) for s in chosen]
                original_inputs = data.inputs
                def checked_inputs(row, predictive=False):
                    if row["split"] == "test":
                        self.assertTrue((out / "2026-11-08_frozen_val_decisions.json").exists())
                    return original_inputs(row, predictive)
                data.inputs = checked_inputs
                maps = self.train.export_all(data, records, out, cfg, ledger=store)
                self.assertTrue(any(r["method"].endswith("prediction-best") for r in maps))
                self.assertTrue(any(r["method"] == "C-history" for r in maps))
                for split in ("train", "val", "test"):
                    saved = next(r for r in maps if r["split"] == split and
                                 r["method"] == "actual-next-convgru-smoke-history")
                    self.assertIn("G", self.train._map_arrays(out, saved))
                summary = report.build_report(data, maps, out, cfg, records)
                with mock.patch.object(self.train, "prediction", side_effect=AssertionError("Repeated inference")), \
                     mock.patch.object(self.train, "build_model", side_effect=AssertionError("Unexpected model")):
                    self.assertEqual(self.train.export_all(data, records, out, cfg, ledger=store, resume=True), maps)
                    rebuilt = report.report_only(out)
                self.assertEqual(summary, rebuilt)
            finally:
                store.close()

    def test_fusion_uses_identical_capacity_and_falls_back_on_missing_forecast(self):
        import numpy as np
        model = self.train.StackFusion()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 3)
        result = self.train._fused(dict(score=np.array([[.9, .9]]), support=np.ones((1,2),bool),
            extras=dict(forecast_support=np.array([[True,False]]))), np.array([[.2,.2]]),
            dict(weight=[1., 1.], bias=0.))
        self.assertAlmostEqual(float(result["score"][0,1]), .2, places=6)
        self.assertGreater(float(result["score"][0,0]), .2)


if __name__ == "__main__":
    unittest.main()
