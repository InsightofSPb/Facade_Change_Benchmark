"""CPU contracts, including a complete synthetic canonical run, not real results."""
from argparse import Namespace
import contextlib
import importlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from facade_change.io import read_json, sha256, write_json

fusion = importlib.import_module("data_analysis.2026-11-06_history_fusion")
memory = importlib.import_module("data_analysis.2026-11-08_variability_memory")
previous_tests = importlib.import_module("tests.test_2026-11-06_history_fusion")
original = previous_tests.original


class StreamingTests(unittest.TestCase):
    def test_progress_resume_accepts_only_the_pinned_code_transition(self):
        migration = read_json(Path(memory.__file__).with_name("2026-11-08_memory_progress_compatibility.json"))
        runner = Path(memory.__file__).name
        current = dict(protocol="canonical-variability-memory-v1", config=dict(epochs=10),
                       code={runner: sha256(memory.__file__), "model.py": "unchanged"})
        self.assertEqual(current["code"][runner], migration["after_sha256"])
        legacy = {**current, "code": {**current["code"], runner: migration["before_sha256"]}}
        with tempfile.TemporaryDirectory() as directory:
            store = fusion.data_module.Store(Path(directory) / "resume", current["config"])
            try:
                store.contract(legacy)
                before = (store.out / "experiment_contract.json").read_bytes()
                memory.pin_contract(store, current)
                self.assertEqual(before, (store.out / "experiment_contract.json").read_bytes())
                events = (store.out / "events.jsonl").read_text()
                self.assertIn("progress_only_code_update", events)
                for changed in (
                        {**current, "config": dict(epochs=30)},
                        {**current, "code": {**current["code"], "model.py": "changed"}},
                        {**current, "code": {**current["code"], runner: "future-non-progress-edit"}}):
                    with self.assertRaisesRegex(ValueError, "changed"):
                        memory.pin_contract(store, changed)
            finally:
                store.close()

    def test_streamed_objective_matches_dense_semantic_interactions(self):
        rng = np.random.default_rng(7)
        raw = rng.random((173, 4)).astype(np.float32)
        labels = (raw[:, 0] + raw[:, 2] > .9).astype(np.uint8)
        sample = dict(x=raw, y=labels, ranges=[[0, 53], [53, 173]])
        normalizer = dict(low=np.zeros(2), high=np.ones(2))
        item = dict(id="classes", kind="stack_classes", columns=[0, 1])
        weights = np.r_[np.full(53, 1 / 53), np.full(120, 1 / 120)]
        dense = fusion.fit_logistic(fusion.design(fusion.normalize(raw[:, :2], normalizer),
                                                 raw[:, 2:], item, []), labels, weights, .01)
        streamed = fusion.training.fit_streaming(sample, normalizer, item, [], fusion, .01, chunk_size=19)
        np.testing.assert_allclose(streamed["parameters"], dense["parameters"], atol=3e-6)
        self.assertAlmostEqual(float(streamed["objective"]), float(dense["objective"]), places=9)

    def test_all_pixels_reads_train_only_and_detects_corruption(self):
        source = Namespace(assignments={"a": "train", "b": "val"}, columns=[("previous", "x")],
            data=Namespace(categories=[dict(name="x")], samples=[dict(building_id=b) for b in "ab"]))
        def blocks(samples, context):
            self.assertEqual([s["building_id"] for s in samples], ["a"])
            truth = np.arange(40).reshape(1, 4, 10) % 2
            truth[0, 0, 0] = 255
            yield ("a",), (0, 0), np.zeros((3, 4, 10)), truth, np.ones((4, 10), bool), None
        source.blocks = blocks
        with tempfile.TemporaryDirectory() as directory:
            store = fusion.data_module.Store(Path(directory) / "out", {})
            try:
                sample = fusion.training_sample(source, store, 0, 13)
                self.assertEqual(len(sample["y"]), 39)
                self.assertIsInstance(sample["x"], np.memmap)
                self.assertEqual(sample["ranges"], [[0, 39]])
                with mock.patch.object(source, "blocks", side_effect=AssertionError("Repeated pixel extraction")):
                    self.assertEqual(len(fusion.training_sample(source, store, 0, 13)["y"]), 39)
                with next(store.out.glob("*.f32")).open("r+b") as stream:
                    stream.write(b"bad!")
                with self.assertRaisesRegex(ValueError, "corrupt"):
                    fusion.training_sample(source, store, 0, 13)
            finally:
                store.close()


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is optional for CPU-only analysis environments")
class MemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.module = importlib.import_module("facade_change.research.2026-11-08_variability_memory_model")

    def inputs(self):
        return (torch.randn(4, 8, 4, 4), torch.ones((4, 1, 4, 4), dtype=torch.bool),
                torch.tensor([2001., 2005., 2010., 2020.]), torch.rand(7, 16, 16))

    def test_current_never_enters_memory_and_return_transition_is_not_vetoed(self):
        model = self.module.VariabilityMemory(channels=8, hidden=4, mode="convgru")
        x, support, years, scores = self.inputs()
        x[-1] = x[0]  # A -> ... -> B -> A, a familiar state but a new transition.
        with mock.patch.object(model, "past_memory", wraps=model.past_memory) as call:
            correction = model(x, support, years, scores)
        self.assertEqual(len(call.call_args.args[0]), 3)
        torch.testing.assert_close(call.call_args.args[2], years[:-1])
        torch.testing.assert_close(correction, torch.zeros_like(correction))
        # At initialization, a positive last-pair logit survives even if A is in memory.
        self.assertTrue(torch.all(torch.sigmoid(2 + correction) > .5))
        model.decoder[-1].weight.data.fill_(.1)
        x.requires_grad_()
        projected = model.project(torch.nn.functional.normalize(x, dim=1))
        state, _ = model.past_memory(projected[:-1], support[:-1], years[:-1], years[-1])
        state.sum().backward()
        torch.testing.assert_close(x.grad[-1], torch.zeros_like(x.grad[-1]))

    def test_capacity_control_ignores_older_content_and_support(self):
        model = self.module.VariabilityMemory(channels=8, hidden=4, mode="convgru-last")
        model.decoder[-1].weight.data.fill_(.1)
        x, support, years, scores = self.inputs()
        before = model(x, support, years, scores)
        x[:2] = 100 * torch.randn_like(x[:2])
        support[:2] = False
        torch.testing.assert_close(before, model(x, support, years, scores))
        history = self.module.VariabilityMemory(channels=8, hidden=4, mode="convgru")
        self.assertEqual(sum(p.numel() for p in history.parameters()), sum(p.numel() for p in model.parameters()))

    def test_invalid_features_are_masked_and_dates_are_checked(self):
        for mode in self.module.MODES:
            model = self.module.VariabilityMemory(channels=8, hidden=4, mode=mode)
            model.decoder[-1].weight.data.fill_(.1)
            x, support, years, scores = self.inputs()
            support[:, :, :2] = False
            a = model(x, support, years, scores)
            x[:, :, :2] = float("nan")
            torch.testing.assert_close(a, model(x, support, years, scores))
            with self.assertRaisesRegex(ValueError, "chronological"):
                model(x, support, years.flip(0), scores)

    def test_fixed_positive_cohort_includes_zero_recall(self):
        rows = [dict(f1=0., iou=0., precision=None, recall=0., fpr=0., ap_binned=.1, positive_pixels=10),
                dict(f1=None, iou=None, precision=None, recall=None, fpr=0., ap_binned=None, positive_pixels=0)]
        result = memory.fixed_macro(rows)
        self.assertEqual(result["f1_buildings"], 1)
        self.assertEqual(result["fpr_buildings"], 2)

    def test_signed_corrections_merge_before_sigmoid_and_native_masks_use_or(self):
        class Data:
            categories = [dict(name="CRACK")]
            groups = []
            def target(self, event_id):
                return dict(target=np.zeros((2, 2), np.uint8),
                            class_targets=np.zeros((1, 2, 2), np.uint8),
                            before=np.ones((2, 2), np.uint16), after=np.ones((2, 2), np.uint16),
                            support=np.ones((2, 2), bool))
        samples = [dict(event_id=str(i), building_id="b", view_id="v", previous_year=1,
                        current_year=2, bbox_xywh=[0, 0, 2, 2]) for i in range(2)]
        values = [np.stack([np.full((2, 2), n) for n in (.1, 0, -2)]),
                  np.stack([np.full((2, 2), n) for n in (.9, 1, 4)])]
        prediction = lambda row: (values[int(row["event_id"])], np.ones((2, 2), bool))
        block = next(metrics for metrics in fusion.metrics.merged_blocks(
            Data(), samples, prediction, [False, True, False], block_size=2, logit_channels=[2]))
        np.testing.assert_allclose(block[2][:, 0, 0], [.5, 1, 1])
        with self.assertRaisesRegex(ValueError, "scores"):
            list(fusion.metrics.merged_blocks(Data(), samples, prediction, [False, True, False], block_size=2))

    def test_full_cached_fusion_refit_and_memory_pipeline(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            fusion.data_module.prepare(original.fixture(root))
            with mock.patch.object(fusion.benchmark, "RemoteScorer", previous_tests.SemanticFake), mock.patch.object(
                    fusion.benchmark, "make_scorer", previous_tests.SemanticFake):
                fusion.benchmark.run(original.arguments(root, methods=list(fusion.METHODS)))
            args = Namespace(run=str(root / "full"), out=str(root / "fusion"), sample_per_building=200,
                             seed=13, l2=.01, semantic_device="cpu", all_pairs=False, resume=False)
            with mock.patch.object(fusion, "RemoteScorer", previous_tests.SemanticFake):
                fusion.run(args)
            args.semantic_cache = args.out
            args.out = str(root / "all")
            args.sample_per_building = 0
            with mock.patch.object(fusion, "RemoteScorer", side_effect=AssertionError("Inference repeated")):
                refit = fusion.run(args)
            self.assertEqual(refit["context"]["newly_computed"], 0)
            self.assertGreater(refit["baseline_parity_rows"], 0)

            def fake_features(source, store, device):
                with mock.patch.object(source.data, "target", side_effect=AssertionError("GT entered feature encoder")):
                    for identifier in source.data.observations:
                        rgb, support = source.data.frame(identifier)
                        unit = memory.PREFIX + "features-" + identifier
                        if not store.read_unit(unit):
                            arrays = dict(patch_tokens=np.full((256, 384), rgb.mean() / 255, np.float16),
                                          patch_support=np.ones(256, bool))
                            store.save_unit(dict(unit_id=unit, kind="fake_features"), arrays)
                return dict(observations=len(source.data.observations), procedural_test_double=True)

            args = Namespace(run=args.run, fusion=str(root / "fusion"), out=str(root / "memory"),
                             device="cpu", epochs=10, hidden=4, batch_size=2, lr=.001,
                             seeds=[13], history_shift_tokens=[1], resume=False)
            with mock.patch.object(memory, "prepare_features", side_effect=fake_features), mock.patch.object(
                    fusion, "RemoteScorer", side_effect=AssertionError("Detector inference repeated")):
                result = memory.run(args)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(result["checkpoints"]), 4)
            self.assertEqual({r["split"] for r in result["results"]}, {"train", "val", "test"})
            self.assertEqual({r["method"] for r in result["results"]}, {*fusion.METHODS, "stack", *memory.MODES})
            self.assertTrue((root / "memory" / (memory.PREFIX + "history_shift.csv")).exists())
            status = read_json(root / "memory/run.json")
            status["status"] = "interrupted"
            write_json(root / "memory/run.json", status)
            contract_path = root / "memory/experiment_contract.json"
            legacy = read_json(contract_path)
            migration = read_json(Path(memory.__file__).with_name("2026-11-08_memory_progress_compatibility.json"))
            legacy["code"][Path(memory.__file__).name] = migration["before_sha256"]
            write_json(contract_path, legacy)
            args.resume = True
            with mock.patch.object(memory, "prepare_features", side_effect=fake_features):
                resumed = memory.run(args)
            self.assertEqual(result["results"], resumed["results"])
            self.assertEqual(read_json(contract_path), legacy)


if __name__ == "__main__":
    unittest.main()
