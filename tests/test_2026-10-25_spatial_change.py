"""Spatial scientific contracts and a procedural CPU workflow; no facade claims."""
import contextlib
import copy
import hashlib
import importlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from facade_change import workflow
from facade_change.io import read_json, sha256, write_json
from facade_change.segmentation.metrics import confusion_counts


spatial = importlib.import_module("facade_change.research.2026-10-25_spatial_change")
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from facade_change.segmentation.model import masked_multilabel_loss
    networks = importlib.import_module(spatial.NETWORK)
data_module = spatial.data_module
metrics = spatial.metrics
history = spatial.history
old_fixture = importlib.import_module("test_2026-10-25_history_models")


def fixture(root):
    base = old_fixture.FeatureFixture(root)
    entries = {}
    for track, sequence in zip(base.tracks, base.sequences):
        with np.load(base.features / sequence["path"], allow_pickle=False) as arrays:
            z = arrays["z_observations"].copy()
        for index, observation in enumerate(track["observations"]):
            path = base.root / (observation["observation_id"] + "-support.png")
            Image.fromarray(np.full((14, 14), 255, np.uint8)).save(path)
            observation.update(support=path.name, support_sha256=sha256(path))
            path = base.features / (observation["observation_id"] + "-patches.npz")
            np.savez(path, patch_tokens=z[index][None].astype(np.float16),
                     patch_support=np.ones(1, bool), z=z[index])
            entries[observation["observation_id"]] = {"path": path.relative_to(base.features).as_posix(),
                                                       "sha256": sha256(path)}
    base.save()
    base.index["encoder"]["save_patch_tokens"] = True
    for track in base.tracks:
        for observation in track["observations"]:
            entries[observation["observation_id"]]["signature"] = {
                "encoder_sha256": history.digest(base.index["encoder"]),
                "rgb_sha256": observation["rgb_sha256"], "support_sha256": observation["support_sha256"]}
    base.index["features"] = entries
    publish_features(base)
    data = history.load_histories(base.features, ("train", "val", "test"))
    target_root = root / "targets"
    (target_root / "targets").mkdir(parents=True)
    categories = [{"id": 4, "name": "ORNAMENT"}, {"id": 9, "name": "CRACK"}]
    events = []
    for sample in data.samples:
        target = np.zeros((2, 14, 14), np.uint8)
        if sample["track_id"].endswith("_1") and sample["current_year"] == 2018:
            target[1] = 1
        target[0, :2] = 255
        target[1, -2:] = 255
        path = target_root / "targets" / (sample["event_id"] + ".npz")
        np.savez(path, class_targets=target, category_ids=[4, 9], target=metrics.binary_target(target))
        row = {k: sample[k] for k in history.LABEL_FIELDS if k in sample}
        row.update(target_path=path.relative_to(target_root).as_posix(), target_sha256=sha256(path),
                   history_start_year=int(sample["years"][0]))
        events.append(row)
    index = {"kind": "history_annotation_targets", "schema_version": 1,
             "feature_index_sha256": sha256(base.features / "index.json"),
             "input_sha256": {str(base.features / "index.json"): sha256(base.features / "index.json")},
             "contract": {"categories": categories, "target_channels": [0, 1],
                          "target_scope": "selected target-class differences on common annotated support"},
             "events": events}
    write_json(target_root / "index.json", index)
    write_json(target_root / "run.json", {"status": "completed"})
    return base, target_root


def publish_features(base):
    write_json(base.features / "index.json", base.index)
    write_json(base.embed / "index.json", base.index)
    write_json(base.embed / "run.json", {"status": "completed",
                                        "index_sha256": sha256(base.features / "index.json")})


@unittest.skipUnless(HAS_TORCH, "optional torch dependency is unavailable")
class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(16)
        self.x = torch.randn(4, 8, 3, 3)
        self.mask = torch.ones(4, 1, 3, 3, dtype=torch.bool)
        self.years = torch.tensor([2005, 2010, 2012, 2026])

    def test_dense_shape_and_current_excluded_from_memory(self):
        model = networks.SpatialChangeModel(8, 3, "convgru", hidden=4)
        seen = []
        hook = model.memory.register_forward_hook(lambda module, inputs, output: seen.append(inputs[0].clone()))
        output = model(self.x, self.mask, self.years, (42, 42))
        changed = self.x.clone()
        changed[-1] += 500
        model(changed, self.mask, self.years, (42, 42))
        hook.remove()
        self.assertEqual(output.shape, (1, 3, 42, 42))
        self.assertEqual(len(seen), 6)
        for first, second in zip(seen[:3], seen[3:]):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        model2 = networks.SpatialChangeModel(8, 3, "convgru-last", hidden=4)
        self.assertEqual(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model2.parameters()))

    def test_pair_and_capacity_control_ignore_earlier_pixels(self):
        for mode in ("pair", "convgru-last"):
            model = networks.SpatialChangeModel(8, 2, mode, hidden=4).eval()
            before = model(self.x, self.mask, self.years, (42, 42))
            changed = self.x.clone()
            changed[:-2] += 1000
            after = model(changed, self.mask, self.years, (42, 42))
            torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_invalid_values_and_missing_context_do_not_change_predictions(self):
        model = networks.SpatialChangeModel(8, 2, "convgru", hidden=4, extra_channels=20).eval()
        self.mask[0] = False
        extra = torch.randn(4, 20, 3, 3)
        available = torch.zeros_like(extra, dtype=torch.bool)
        before = model(self.x, self.mask, self.years, (42, 42), extra, available)
        changed = self.x.clone()
        changed[0] = 10000
        after = model(changed, self.mask, self.years, (42, 42), extra * 10000, available)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        available[:, 0] = True
        self.assertFalse(torch.equal(before, model(self.x, self.mask, self.years, (42, 42), extra, available)))

    def test_masked_multilabel_gradients_and_legacy_parity(self):
        logits = torch.randn(1, 2, 3, 3, requires_grad=True)
        target = torch.zeros_like(logits)
        valid = torch.ones_like(logits, dtype=torch.bool)
        valid[:, 0, 0] = False
        target[:, 1] = 1
        loss = masked_multilabel_loss(logits, target, valid)
        loss.backward()
        self.assertTrue((logits.grad[:, 0, 0] == 0).all())
        full = torch.ones(1, 3, 3, dtype=torch.bool)
        torch.testing.assert_close(masked_multilabel_loss(logits, target, full),
                                   masked_multilabel_loss(logits, target, full[:, None].expand_as(logits)))
        zero = masked_multilabel_loss(logits, target, torch.zeros_like(valid))
        self.assertEqual(float(zero), 0.)

    def test_small_spatial_head_can_overfit_dense_fixture(self):
        model = networks.SpatialChangeModel(8, 2, "pair", hidden=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=.03)
        target = torch.zeros(1, 2, 42, 42)
        target[:, 1, :, 21:] = 1
        valid = torch.ones_like(target, dtype=torch.bool)
        losses = []
        for _ in range(60):
            optimizer.zero_grad()
            loss = masked_multilabel_loss(model(self.x, self.mask, self.years, (42, 42)), target, valid)
            losses.append(float(loss.detach()))
            loss.backward()
            optimizer.step()
        self.assertLess(losses[-1], losses[0] * .25)


class MetricTests(unittest.TestCase):
    def test_channel_validity_and_unknown_binary_union(self):
        pred = np.ones((2, 2, 2), bool)
        truth = np.zeros_like(pred)
        valid = np.ones_like(pred)
        valid[0] = False
        counts = confusion_counts(pred, truth, valid)
        np.testing.assert_array_equal(counts, [[0, 0, 0], [0, 4, 0]])
        labels = np.array([[[0, 1, 255]], [[255, 255, 0]]], np.uint8)
        np.testing.assert_array_equal(metrics.binary_target(labels), [[255, 1, 255]])

    def test_overlaps_count_once_and_native_union_preserved(self):
        data = SimpleNamespace(categories=[{"id": 1, "name": "crack"}])
        rows = [{"event_id": str(i), "building_id": "b", "view_id": "v",
                 "previous_year": 1, "current_year": 2, "bbox_xywh": [i, 0, 2, 2]} for i in range(2)]
        def predict(row):
            return (np.full((1, 2, 2), int(row["event_id"]), np.float32),
                    np.ones((2, 2), bool), np.ones((1, 2, 2), np.uint8))
        hist, coverage = metrics.collect_histograms(data, rows, predict, True, True, block_size=2)
        self.assertEqual(coverage["unique_geometric_pixels"], 6)
        self.assertEqual(hist["b"][-1].sum(), 6)
        result = metrics.report(hist, data.categories, [.5, .5], True, True)
        self.assertAlmostEqual(result["binary"]["primary"]["recall"], 4 / 6)
        self.assertIsNone(result["binary"]["primary"]["auprc_binned"])

    def test_conflicting_overlap_fails(self):
        data = SimpleNamespace(categories=[{"id": 1, "name": "crack"}])
        rows = [{"event_id": str(i), "building_id": "b", "view_id": "v",
                 "previous_year": 1, "current_year": 2, "bbox_xywh": [0, 0, 2, 2]} for i in range(2)]
        def predict(row):
            return np.ones((1, 2, 2)), np.ones((2, 2), bool), np.full((1, 2, 2), int(row["event_id"]), np.uint8)
        with self.assertRaisesRegex(ValueError, "contradictory"):
            metrics.collect_histograms(data, rows, predict)

    def test_histogram_ap_and_equal_building_calibration(self):
        hist = np.zeros((2, 2, metrics.BINS), np.int64)
        hist[:, 0, 10] = 100
        hist[:, 1, 900] = 1
        self.assertEqual(metrics.average_precision(hist[0]), 1.)
        thresholds, _ = metrics.calibrate({"a": hist, "b": hist * 1000}, 2)
        report = metrics.report({"a": hist, "b": hist * 1000}, [{"id": 1, "name": "crack"}], thresholds)
        self.assertEqual(report["classes"]["primary"]["f1"], 1.)
        self.assertEqual(report["classes"]["per_class"][0]["auprc_binned"], 1.)


@unittest.skipUnless(HAS_TORCH, "optional torch dependency is unavailable")
class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base, self.targets = fixture(self.root)
        torch.set_num_threads(1)
        self.common = ["--features", str(self.base.features), "--targets", str(self.targets)]

    def invoke(self, arguments):
        with contextlib.redirect_stdout(io.StringIO()):
            return spatial.main(arguments)

    def test_data_prefixes_target_mapping_and_no_test_payload_reads(self):
        data = data_module.SpatialData(self.base.features, self.targets)
        self.assertEqual([c["id"] for c in data.categories], [4, 9])
        self.assertFalse(any(s["split"] == "test" for s in data.samples))
        for sample in data.samples:
            self.assertEqual(sample["observations"][-1]["year"], sample["current_year"])
            self.assertTrue(all(o["year"] < sample["current_year"] for o in sample["observations"][:-1]))
        self.assertFalse(any("test_" in p for p in data.fp.files))

    def test_rebind_targets_to_patch_cache_without_changing_geometry(self):
        original = copy.deepcopy(self.base.index)
        original["encoder"]["save_patch_tokens"] = False
        path = self.root / "old-feature-index.json"
        write_json(path, original)
        index = read_json(self.targets / "index.json")
        index["feature_index_sha256"] = sha256(path)
        index["input_sha256"] = {str(path): sha256(path)}
        samples = history.load_histories(self.base.features, ("train", "val", "test")).samples
        for row, sample in zip(index["events"], samples):
            prefix = [o for o in self.base.tracks[[t["track_id"] for t in self.base.tracks].index(sample["track_id"])]["observations"]
                      if o["year"] <= sample["current_year"]]
            z = np.concatenate((sample["past"], sample["current"][None]))
            row["prefix_sha256"] = history.digest({"encoder": original["encoder"], "track_id": sample["track_id"],
                "observations": prefix, "years": sample["years"].tolist(), "z_sha256": hashlib.sha256(z.tobytes()).hexdigest()})
        write_json(self.targets / "index.json", index)
        data = data_module.SpatialData(self.base.features, self.targets)
        self.assertTrue(data.samples)
        original["history_manifest_sha256"] = "changed"
        write_json(path, original)
        with self.assertRaises(ValueError):
            data_module.SpatialData(self.base.features, self.targets)

    def test_twenty_context_channels_are_separate_and_missing_is_not_zero_prediction(self):
        aux = self.root / "aux"
        aux.mkdir()
        observation = self.base.tracks[0]["observations"][0]
        path = aux / "values.npz"
        np.savez(path, values=np.zeros((20, 1, 1), np.float32), available=np.ones((20, 1, 1), bool))
        index = {"schema_version": 1, "kind": "spatial_auxiliary_maps",
                 "history_manifest_sha256": self.base.index["history_manifest_sha256"],
                 "source_kind": "prediction", "source": {"id": "procedural_fixture", "revision": "v1"},
                 "channels": [{"id": f"context_{i}", "name": f"Context {i}", "kind": "context"} for i in range(20)],
                 "observations": {observation["observation_id"]: {
                     "path": path.name, "sha256": sha256(path), "grid": "patch",
                     "bbox_xywh": self.base.tracks[0]["bbox_xywh"],
                     **{k: observation[k] for k in ("rgb_sha256", "support_sha256")}}}}
        write_json(aux / "index.json", index)
        data = data_module.SpatialData(self.base.features, self.targets, auxiliary=aux / "index.json")
        item = data.load(data.samples[0])
        self.assertEqual(item["extra"].shape[1], 20)
        self.assertEqual(item["target"].shape[0], 2)
        self.assertTrue(item["available"][0].all())
        self.assertFalse(item["available"][1].any())

    def test_train_evaluate_saved_gallery_and_frozen_test_thresholds(self):
        training = self.root / "training"
        self.invoke(["train", *self.common, "--out", str(training), "--epochs", "2", "--hidden", "4", "--batch-size", "4"])
        models = read_json(training / "models.json")
        self.assertEqual(len(models["models"]), 4)
        evaluation = self.root / "evaluation"
        self.invoke(["evaluate", *self.common, "--training-run", str(training), "--out", str(evaluation),
                     "--baselines", "dino-difference", "rgb_diff", "--bootstrap", "10", "--gallery-limit", "2"])
        result = read_json(evaluation / "metrics-val.json")
        self.assertEqual(len(result["methods"]), 6)
        self.assertTrue((evaluation / "gallery.html").is_file())
        self.assertIn("CRACK", (evaluation / "gallery.html").read_text())
        for method in result["methods"].values():
            self.assertEqual(method["coverage"]["crop_events"], 8)
        self.invoke(["gallery", "--evaluation-run", str(evaluation), "--out", str(self.root / "gallery"),
                     "--gallery-limit", "1"])
        self.invoke(["evaluate", *self.common, "--training-run", str(training), "--out", str(self.root / "test"),
                     "--splits", "test", "--baselines", "dino-difference", "rgb_diff",
                     "--calibration-run", str(evaluation), "--bootstrap", "0", "--gallery-limit", "0"])
        test = read_json(self.root / "test/metrics-test.json")
        for name in result["methods"]:
            self.assertEqual(test["methods"][name]["thresholds"], result["methods"][name]["thresholds"])
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.invoke(["train", *self.common, "--out", str(training), "--resume"])

    def test_test_baseline_cannot_calibrate_itself(self):
        with self.assertRaisesRegex(ValueError, "VAL"):
            self.invoke(["evaluate", *self.common, "--splits", "test", "--baselines", "dino-difference",
                         "--out", str(self.root / "bad")])

    def test_anychange_adapter_keeps_native_mask_and_receives_no_labels(self):
        class FakeAnyChange:
            def __init__(self, name, options, pool):
                self.pool = pool
                self.native_prediction = None
            def activate(self):
                return {"method": "anychange", "output_kind": "native_mask", "revision": "fixture"}
            def __call__(self, reference, source, support):
                self.native_prediction = np.zeros(support.shape, bool)
                self.native_prediction[:, :7] = True
                return np.where(support, self.native_prediction, np.nan).astype(np.float32)
            def close(self):
                self.pool.close()
        out = self.root / "anychange"
        with mock.patch("facade_change.methods.remote.RemoteScorer", FakeAnyChange):
            self.invoke(["evaluate", *self.common, "--out", str(out), "--baselines", "anychange",
                         "--gallery-limit", "1", "--bootstrap", "0"])
        report = read_json(out / "metrics-val.json")["methods"]["anychange"]
        self.assertNotIn("classes", report)
        self.assertIsNone(report["binary"]["primary"]["auprc_binned"])
        self.assertEqual(report["thresholds"], [.5, .5, .5])

    def test_reordered_target_channels_fail_before_training(self):
        index = read_json(self.targets / "index.json")
        path = self.targets / index["events"][0]["target_path"]
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        arrays["category_ids"] = arrays["category_ids"][::-1]
        np.savez(path, **arrays)
        index["events"][0]["target_sha256"] = sha256(path)
        write_json(self.targets / "index.json", index)
        with mock.patch.object(history, "torch_runtime", side_effect=AssertionError("GPU must not be allocated")):
            with self.assertRaisesRegex(ValueError, "channel order"):
                self.invoke(["train", *self.common, "--out", str(self.root / "bad")])

    def test_interrupted_training_resumes(self):
        out = self.root / "resume"
        arguments = ["train", *self.common, "--out", str(out), "--models", "pair", "--epochs", "2", "--hidden", "4"]
        save = history.atomic_checkpoint
        calls = []
        def interrupt(torch_module, path, payload):
            save(torch_module, path, payload)
            if path.name == "last.pth":
                calls.append(1)
                if len(calls) == 1:
                    raise KeyboardInterrupt()
        with mock.patch.object(history, "atomic_checkpoint", interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.invoke(arguments)
        self.assertEqual(read_json(out / "run.json")["status"], "interrupted")
        self.invoke([*arguments, "--resume"])
        self.assertEqual(read_json(out / "run.json")["status"], "completed")
        self.assertEqual(torch.load(out / "pair/last.pth", weights_only=True)["epoch"], 2)

    def test_workflow_routes(self):
        for stage, prefix in (("train", "train"), ("benchmark", "evaluate"), ("gallery", "gallery")):
            command = workflow.COMMANDS[stage]["spatial-change"]
            self.assertEqual(command.prefix, (prefix,))
            self.assertEqual(command.module, spatial.__name__)


if __name__ == "__main__":
    unittest.main()
