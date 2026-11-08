"""CPU contracts for fusion; procedural examples are not facade results."""
from argparse import Namespace
import contextlib
import importlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from facade_change.io import read_json, sha256, write_json

fusion = importlib.import_module("data_analysis.2026-11-06_history_fusion")
original = importlib.import_module("tests.test_2026-11-04_history_benchmark")
benchmark, metrics = fusion.benchmark, fusion.metrics


class SemanticFake(original.FakeScorer):
    def __init__(self, method, *args, **kwargs):
        super().__init__(method, *args, **kwargs)
        self.metadata.update(
            categories=[dict(id=3, name="ORNAMENT_INTACT"), dict(id=9, name="CRACK")],
            category_to_channel=[dict(category_id=3, channel=0), dict(category_id=9, channel=1)],
            head_checkpoint=dict(sha256="fake-head"),
            backbone_provenance=dict(checkpoint=dict(sha256="fake-backbone"),
                                     sources=dict(python_tree_sha256="fake-source")))

    def __call__(self, a, b, support):
        result = super().__call__(a, b, support)
        if self.method == "semantic_dino":
            def probability(rgb):
                p = np.stack((np.full(support.shape, .8), rgb.mean(axis=2) / 255)).astype(np.float32)
                p[:, ~support] = np.nan
                return p
            self.auxiliary_arrays = dict(probabilities_reference=probability(a),
                                         probabilities_source=probability(b))
        return result


class FusionTests(unittest.TestCase):
    def test_semantic_self_pair_encodes_one_observation_once(self):
        semantic = importlib.import_module("facade_change.methods.2026-10-14_semantic_dino")
        scorer = semantic.SemanticDINOScorer.__new__(semantic.SemanticDINOScorer)
        scorer.export_probabilities = True
        scorer._probabilities = mock.Mock(return_value=np.full((2, 3, 4), .3, np.float32))
        rgb, support = np.zeros((3, 4, 3), np.uint8), np.ones((3, 4), bool)
        stub = SimpleNamespace(bernoulli_entropy=lambda p: np.zeros(p.shape[1:], np.float32))
        with mock.patch.dict("sys.modules", {"facade_change.segmentation.evaluate": stub}):
            score = scorer(rgb, rgb.copy(), support)
        self.assertEqual(scorer._probabilities.call_count, 1)
        np.testing.assert_array_equal(score, 0)
        np.testing.assert_allclose(scorer.auxiliary_arrays["probabilities_source"], .3)

    def test_earliest_semantics_uses_its_own_endpoints(self):
        item = dict(kind="stack_classes", columns=[0], context_offset=1, context_size=1)
        a = fusion.design(np.ones((2, 1)), np.array([[.1, .8], [.9, .8]]), item, [])
        np.testing.assert_array_equal(a[0], a[1])

    def test_semantic_interactions_can_select_complementary_experts(self):
        rng = np.random.default_rng(13)
        scores = rng.integers(0, 2, size=(1000, 2)).astype(float)
        context = rng.integers(0, 2, size=(1000, 1)).astype(float)
        target = np.where(context[:, 0] > 0, scores[:, 0], scores[:, 1])
        variant = dict(columns=[0, 1], kind="stack_classes", id="gated")
        design = fusion.design(scores, context, variant, [])
        model = fusion.fit_logistic(design[:600], target[:600], np.ones(600), .001)
        normalizer = dict(low=np.zeros(2), high=np.ones(2))
        output = fusion.predict(scores[600:], context[600:], variant, normalizer, [],
                                {"gated": model}, np.full(2, .5))
        self.assertGreater(((output > .5) == target[600:]).mean(), .98)
        global_variant = dict(columns=[0, 1], kind="stack", id="global")
        global_model = fusion.fit_logistic(scores[:600], target[:600], np.ones(600), .001)
        global_output = fusion.predict(scores[600:], context[600:], global_variant,
            normalizer, [], {"global": global_model}, np.full(2, .5))
        self.assertLess(((global_output > .5) == target[600:]).mean(), .85)

    def test_or_and_mean_and_constant_normalization(self):
        raw = np.array([[.9, .1], [.2, .8], [.6, .7], [.1, .2]])
        normalizer = dict(low=np.zeros(2), high=np.ones(2))
        for kind, expected in (
                ("and", [0, 0, 1, 0]), ("or", [1, 1, 1, 0]),
                ("mean", [.5, .5, .65, .15]), ("mean_scaled", [.5, .5, .65, .15])):
            item = dict(kind=kind, columns=[0, 1])
            output = fusion.predict(raw, np.empty((4, 0)), item, normalizer, [], {}, np.full(2, .5))
            np.testing.assert_allclose(output, expected)
        np.testing.assert_array_equal(
            fusion.normalize(raw, dict(low=np.zeros(2), high=np.zeros(2))), np.zeros_like(raw))

    def test_train_reservoir_never_reads_other_splits(self):
        source = Namespace(assignments={"a": "train", "b": "val", "c": "test"},
                           data=Namespace(samples=[dict(building_id=b) for b in "abc"]))
        def blocks(samples, store):
            self.assertEqual([s["building_id"] for s in samples], ["a"])
            yield ("a",), (0, 0), np.arange(40).reshape(1, 4, 10) / 40, (
                np.arange(40).reshape(1, 4, 10) % 2), np.ones((4, 10), bool), None
        source.blocks = blocks
        with tempfile.TemporaryDirectory() as root:
            store = fusion.data_module.Store(Path(root) / "sample", {})
            try:
                sample = fusion.training_sample(source, store, 10, 13)
                self.assertEqual(set(sample["buildings"]), {"a"})
                self.assertEqual(len(sample["y"]), 10)
                resumed = fusion.training_sample(source, store, 10, 13)
                np.testing.assert_array_equal(sample["x"], resumed["x"])
            finally:
                store.close()

    def test_native_overlap_is_or_before_cross_method_and(self):
        class Data:
            categories = [dict(id=1, name="CRACK")]
            groups = []
            def target(self, event):
                return dict(target=np.zeros((2, 2), np.uint8),
                            class_targets=np.zeros((1, 2, 2), np.uint8),
                            before=np.ones((2, 2), np.uint16), after=np.ones((2, 2), np.uint16),
                            support=np.ones((2, 2), bool))
        rows = [dict(event_id=str(i), building_id="b", view_id="v", previous_year=1,
                     current_year=2, bbox_xywh=[0, 0, 2, 2], known_pixels=4) for i in range(2)]
        def prediction(row):
            i = int(row["event_id"])
            return np.stack((np.full((2, 2), i), np.full((2, 2), .9 - .8 * i))), np.ones((2, 2), bool)
        block = next(metrics.merged_blocks(Data(), rows, prediction, [True, False], 2))
        np.testing.assert_allclose(block[2][0], 1)
        np.testing.assert_allclose(block[2][1], .5)

    def test_full_pipeline_parity_predicted_context_resume_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                benchmark.data_module.prepare(original.fixture(root))
                with mock.patch.object(benchmark, "RemoteScorer", SemanticFake), mock.patch.object(
                        benchmark, "make_scorer", SemanticFake):
                    benchmark.run(original.arguments(root, methods=list(fusion.METHODS)))
                args = Namespace(run=str(root / "full"), out=str(root / "fusion"),
                    sample_per_building=200, seed=13, l2=.01,
                    semantic_device="cpu", all_pairs=False, resume=False)
                source = fusion.Source(args.run)
                store = fusion.data_module.Store(Path(args.out), dict(preflight=True))
                try:
                    with mock.patch.object(fusion, "RemoteScorer", SemanticFake), mock.patch.object(
                            source.data, "target", side_effect=AssertionError("GT entered semantic inference")):
                        context = fusion.semantic_context(source, store, "cpu")
                    self.assertEqual(context["events"], len(source.data.samples))
                finally:
                    store.close()
                # Use a new output after this separate inference-contract check.
                args.out = str(root / "complete")
                with mock.patch.object(fusion, "RemoteScorer", SemanticFake):
                    summary = fusion.run(args)
                self.assertEqual(summary["variants"], 105)
                self.assertEqual(summary["rows"], 105 * 3 * 4)
                self.assertEqual(summary["baseline_parity_rows"], 10 * 3 * 3 * 4)
                self.assertTrue(all(r["fpr"] <= .05 + 1e-12 for r in summary["fpr05"]
                                    if r["split"] == "val" and r["calibrated"]))
                self.assertEqual({r["split"] for r in summary["results"]}, set(fusion.SPLITS))
                self.assertEqual({r["method"] for r in summary["results"] if r["mode"] == "previous+history"},
                                 {"stack", "stack_groups", "stack_classes"})
                self.assertTrue((Path(args.out) / (fusion.PREFIX + "report.txt")).is_file())
                before = {p.name: sha256(p) for p in (Path(args.out) / "maps").glob("*.npz")}
                record = read_json(Path(args.out) / "run.json")
                record["status"] = "interrupted"
                write_json(Path(args.out) / "run.json", record)
                args.resume = True
                with mock.patch.object(fusion, "RemoteScorer", side_effect=AssertionError("Inference repeated")):
                    resumed = fusion.run(args)
                after = {p.name: sha256(p) for p in (Path(args.out) / "maps").glob("*.npz")}
                self.assertEqual(before, after)
                self.assertEqual(summary["results"], resumed["results"])
                self.assertEqual(len(fusion.variants(source, all_pairs=True)), 321)
                path = root / "full/thresholds.json"
                path.write_text(path.read_text() + "\n")
                with self.assertRaisesRegex(ValueError, "hash"):
                    fusion.Source(args.run)


if __name__ == "__main__":
    unittest.main()
