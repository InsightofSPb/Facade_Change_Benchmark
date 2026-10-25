"""Classwise fractions, masked loss, building aggregation and real CPU workflows."""
import contextlib
import csv
import importlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from facade_change.io import read_json, sha256, write_json

reg = importlib.import_module("facade_change.research.2026-10-25_history_regression")
hm = reg.history
target_tests = importlib.import_module("test_2026-10-25_history_targets")

try:
    import torch
except ImportError:
    torch = None


def row(building, values, valid=None):
    return {"building_id": building, "fractions": np.asarray(values, dtype=np.float32),
            "valid": np.asarray(valid if valid is not None else [True]*len(values), dtype=bool)}


class FractionMathTests(unittest.TestCase):
    def test_unknown_is_excluded_from_denominator_and_never_filled_with_zero(self):
        layers = np.array([[[1, 0], [255, 255]], [[255, 255], [255, 255]],
                           [[0, 0], [0, 255]]], dtype=np.uint8)
        result = reg.fractions_from_maps(layers, 1, 0)
        np.testing.assert_allclose(result["fractions"], [.5, np.nan, 0], equal_nan=True)
        self.assertEqual(result["known_pixels"].tolist(), [2, 0, 3])
        self.assertEqual(result["valid"].tolist(), [True, False, True])

    def test_insufficient_class_support_is_unknown(self):
        result = reg.fractions_from_maps(np.array([[[1, 0], [255, 255]]], np.uint8), 1, .6)
        self.assertFalse(result["valid"][0])
        self.assertTrue(np.isnan(result["fractions"][0]))

    def test_stable_fraction_has_conditional_scope_even_if_other_pixels_unknown(self):
        layer = np.zeros((1, 14, 14), np.uint8)
        layer[:, :, :4] = 255
        result = reg.fractions_from_maps(layer)
        self.assertEqual(result["fractions"][0], 0)
        self.assertEqual(result["known_pixels"][0], 140)
        self.assertIn("outside unknown", reg.SCOPE)

    def test_invalid_mask_codes_rejected(self):
        with self.assertRaisesRegex(ValueError, "0/1/255"):
            reg.fractions_from_maps(np.full((2, 3, 3), 2, np.uint8))

    def test_duplicate_crops_do_not_increase_building_weight(self):
        samples = [row("a", [0, np.nan], [True, False]), row("b", [1, 1])]
        predictions = np.zeros((2, 2))
        original = reg.report_metrics(samples, predictions)
        duplicate = reg.report_metrics([samples[0]]*9 + [samples[1]], np.zeros((10, 2)))
        self.assertAlmostEqual(original["macro_building_mae"], .75)
        self.assertAlmostEqual(duplicate["macro_building_mae"], .75)
        weights = reg.loss_weights(samples)
        np.testing.assert_allclose(weights.sum(0)/len(samples), [.5, .5])

    def test_paired_building_bootstrap_uses_identical_draws(self):
        samples = [row("a", [0, .2]), row("b", [1, .7])]
        p = np.array([[.1, .3], [.8, .6]])
        result = reg.paired_intervals(samples, {"pair_gaps": p, "gru_gaps": p}, 20, 42)
        delta = result["comparisons"]["gru_gaps_minus_pair_gaps"]
        self.assertEqual(delta["difference"], 0)
        self.assertEqual(delta["percentile_95"], [0, 0])

    @unittest.skipIf(torch is None, "Torch required")
    def test_unknown_outputs_have_zero_loss_gradient(self):
        rows = [row("a", [0, np.nan], [True, False])]
        logits = torch.zeros((1, 2), requires_grad=True)
        value = reg.loss(logits, rows, reg.loss_weights(rows), torch)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertGreater(logits.grad[0, 0].item(), 0)
        self.assertEqual(logits.grad[0, 1].item(), 0)

    @unittest.skipIf(torch is None, "Torch required")
    def test_single_class_preserves_batch_and_output_axes(self):
        model = reg.build_model({"dimension": 8, "model": "pair", "width": 8,
                                 "dropout": 0., "time_mode": "none", "outputs": 1})
        batch = {"past": torch.zeros((2, 2, 8)), "time_features": torch.zeros((2, 2, 2)),
                 "lengths": torch.tensor([2, 2]), "current": torch.zeros((2, 8)), "pair_gap": torch.ones(2)}
        self.assertEqual(model(**batch)[0].shape, (2, 1))

    @unittest.skipIf(torch is None, "Torch required")
    def test_fraction_head_learns_distinct_continuous_targets(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        model = reg.build_model({"dimension": 8, "model": "gru", "width": 8,
                                 "dropout": 0., "time_mode": "gaps", "outputs": 2})
        batch = {"past": torch.randn((2, 2, 8)), "time_features": torch.zeros((2, 2, 2)),
                 "lengths": torch.tensor([2, 2]), "current": torch.randn((2, 8)), "pair_gap": torch.ones(2)}
        rows = [row("a", [.2, .8]), row("b", [.7, .1])]
        optimizer = torch.optim.Adam(model.parameters(), lr=.03)
        for _ in range(100):
            optimizer.zero_grad()
            value = reg.loss(model(**batch)[0], rows, reg.loss_weights(rows), torch)
            value.backward()
            optimizer.step()
        self.assertLess(float(value.detach()), .001)


@unittest.skipUnless(importlib.util.find_spec("pycocotools") and torch is not None, "COCO and Torch required")
class FractionPipelineTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.fixture = target_tests.make_fixture(self.root)
        torch.set_num_threads(1)
        self.silence = contextlib.redirect_stdout(io.StringIO())
        self.silence.__enter__()
        self.addCleanup(self.silence.__exit__, None, None, None)
        self.annotations = self.root / "annotations"
        target_tests.targets.export_targets(target_tests.targets.parser().parse_args([
            "--features", str(self.fixture.features), "--out", str(self.annotations),
            "--gallery-limit", "0", "--splits", "train", "val", "test"]))

    def prepare(self, *extra):
        return reg.prepare(reg.parser().parse_args([
            "--features", str(self.fixture.features), "--annotation-run", str(self.annotations),
            "--out", str(self.root / "fractions"), *extra]))

    def train_args(self, name="trained", *extra):
        return hm.parser().parse_args(["train", "--features", str(self.fixture.features),
            "--class-targets", str(self.root / "fractions/index.json"), "--out", str(self.root / name),
            "--epochs", "2", "--width", "8", "--batch-size", "4", *extra])

    def test_preparation_never_decodes_rgb_or_uses_binary_labels(self):
        # All binary rows are unreviewed. Class maps remain valid and are the sole target source.
        path = self.annotations / "index.json"
        index = read_json(path)
        for event in index["events"]:
            event.update(label="", review_status="unreviewed")
        write_json(path, index)
        record = read_json(self.annotations / "run.json")
        record["artifact_sha256"]["index.json"] = sha256(path)
        write_json(self.annotations / "run.json", record)
        for path in self.fixture.root.glob("*.png"):
            path.unlink()
        result = self.prepare()
        self.assertGreater(result["by_split"]["train"]["classes"][1]["nonzero"], 0)
        self.assertGreater(result["by_split"]["train"]["classes"][1]["zero"], 0)
        with mock.patch.object(hm, "read_labels", side_effect=AssertionError("binary labels must not be read")):
            hm.train_models(self.train_args("trained", "--models", "gru", "--time-modes", "gaps"))

    def test_full_train_test_inference_and_facade_metrics(self):
        self.prepare("--splits", "train", "val", "test")
        hm.train_models(self.train_args())
        bundle = read_json(self.root / "trained/models.json")
        self.assertEqual(len(bundle["models"]), 8)
        self.assertEqual(bundle["objective"], "class_fractions")
        self.assertNotIn("threshold", bundle["models"]["gru_gaps"])
        self.assertEqual(bundle["baselines"]["train_mean"]["constant"][0], 0.)
        argv = ["evaluate", "--features", str(self.fixture.features),
                "--training-run", str(self.root / "trained"), "--splits", "test",
                "--class-targets", str(self.root / "fractions"), "--bootstrap", "10",
                "--out", str(self.root / "evaluated")]
        hm.evaluate_models(hm.parser().parse_args(argv))
        metrics = read_json(self.root / "evaluated/metrics.json")
        self.assertEqual(set(metrics["splits"]), {"test"})
        self.assertFalse(metrics["independent_temporal_event_test"])
        self.assertIn("gru_gaps_minus_pair_gaps", metrics["paired_mae_differences"]["test"]["comparisons"])
        self.assertIn("gru_gaps", metrics["past_shuffle"]["test"])
        with np.load(self.root / "evaluated/gru_gaps_embeddings.npz") as archive:
            self.assertEqual(archive["predicted_fractions"].shape, (8, 2))
            self.assertEqual(archive["history_embeddings"].shape, (8, 8))
        with (self.root / "evaluated/predictions.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 16)
        argv.remove("--class-targets")
        argv.remove(str(self.root / "fractions"))
        argv[argv.index(str(self.root / "evaluated"))] = str(self.root / "unlabelled")
        hm.evaluate_models(hm.parser().parse_args(argv))
        self.assertEqual(read_json(self.root / "unlabelled/metrics.json")["splits"], {})

    def test_train_does_not_open_test_arrays(self):
        self.prepare("--splits", "train", "val", "test")
        (self.root / "fractions/test.npz").unlink()
        for path in self.fixture.embed.glob("sequences/test*.npz"):
            path.unlink()
        hm.train_models(self.train_args("trained", "--models", "pair", "--time-modes", "none"))

    def test_unavailable_targets_fail_before_gpu_or_output(self):
        self.prepare("--min-known-pixels", "1000")
        with mock.patch.object(hm, "torch_runtime") as runtime, self.assertRaisesRegex(ValueError, "TRAIN has no usable"):
            hm.train_models(self.train_args())
        runtime.assert_not_called()
        self.assertFalse((self.root / "trained").exists())

    def test_corrupted_target_artifact_is_rejected(self):
        self.prepare()
        with (self.root / "fractions/train.npz").open("ab") as stream:
            stream.write(b"modified")
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            hm.train_models(self.train_args())

    def test_resume_matches_uninterrupted_regression_weights(self):
        self.prepare()
        extra = ("--models", "gru", "--time-modes", "gaps")
        hm.train_models(self.train_args("continuous", *extra))
        original = hm.HistoryRun.event
        def interrupt(store, kind, **values):
            result = original(store, kind, **values)
            if kind == "epoch_completed":
                raise KeyboardInterrupt()
            return result
        with mock.patch.object(hm.HistoryRun, "event", interrupt), self.assertRaises(KeyboardInterrupt):
            hm.train_models(self.train_args("interrupted", *extra))
        hm.train_models(self.train_args("interrupted", *extra, "--resume"))
        a = torch.load(self.root / "continuous/gru_gaps/best.pth", weights_only=True)
        b = torch.load(self.root / "interrupted/gru_gaps/best.pth", weights_only=True)
        for name, value in a["state_dict"].items():
            torch.testing.assert_close(value, b["state_dict"][name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
