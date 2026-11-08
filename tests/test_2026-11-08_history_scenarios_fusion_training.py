"""Actual short fusion fits on cached synthetic tensors; never primary training."""
from contextlib import redirect_stderr, redirect_stdout
import importlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:
    torch = None


class MemoryStore:
    def __init__(self, arrays):
        self.arrays = arrays
        self.reads = []

    def read(self, record, fields=None):
        if fields is None:
            raise AssertionError("Fusion must read explicitly selected fields")
        self.reads.append((record["event_id"], tuple(fields)))
        return {key: self.arrays[record["event_id"]][key] for key in fields}


@unittest.skipIf(torch is None, "PyTorch optional")
class CachedFusionTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.training = importlib.import_module("facade_change.research.2026-11-08_history_scenarios_training")
        cls.fusion = importlib.import_module("facade_change.research.2026-11-08_history_scenarios_fusion")
        cls.data = importlib.import_module("facade_change.research.2026-11-08_history_scenarios_data")

    def fixture(self):
        dataset = self.data.SyntheticData()
        normalizer = self.data.fit_normalizer(dataset)
        arrays, records = {}, []
        for row in dataset.samples:
            if row["split"] == "test":
                continue
            target = dataset.targets(row)
            current = dataset.feature_target(row)
            normalized = normalizer.apply(current["features"], current["support"])
            score, _ = dataset.scores(row)
            raw, _ = dataset.stack_components(row)
            support = dataset.evaluation_support(row)
            # A pre-existing synthetic frozen forecast; no predictor is built.
            forecast = normalized + np.float32(.15)
            arrays[row["event_id"]] = dict(score=score, support=support, stack_raw=raw,
                forecast=forecast, target_features=normalized, forecast_support=support,
                stack_score=score, target=np.where(support, target["target"], 255).astype(np.uint8))
            records.append(dict(event_id=row["event_id"], split=row["split"],
                artifact_sha256="synthetic-" + row["event_id"], stack_contract=dataset.stack_contract))
        config = dict(matrix_protocol="reduced-v2", device="cpu", smoke=True,
            epochs=1, patience=1, effective_batch=8, fusion_epochs=1, amp=False)
        return dataset, MemoryStore(arrays), records, config

    def spec(self, config, kind):
        return next(row for row in self.training.resolve_fusion_matrix(config)
                    if row["fusion_kind"] == kind and row["parent_method"] == "h0-convgru-smoke-history")

    def test_all_five_heads_fit_cached_train_validate_without_base_networks_or_test(self):
        for kind in self.fusion.KINDS:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                dataset, store, records, config = self.fixture()
                original = dataset.split_samples
                visited = []

                def split(name, *args, **kwargs):
                    visited.append(name)
                    self.assertNotEqual(name, "test", "Scientific TEST accessed during fusion fit")
                    return original(name, *args, **kwargs)

                torch.manual_seed(13)
                initial = self.fusion.build_head(kind, channels=dataset.channels).state_dict()
                with mock.patch.object(dataset, "split_samples", side_effect=split), \
                     mock.patch.object(self.training, "build_model", side_effect=AssertionError("base reconstruction")), \
                     mock.patch.object(self.training, "_forward", side_effect=AssertionError("base inference")), \
                     redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    record = self.training._train_reduced_fusion(self.spec(config, kind), dataset, store,
                        records, directory, config)
                self.assertEqual(record["status"], "completed")
                self.assertEqual(record["completed_epochs"], 1)
                self.assertGreater(record["optimizer_steps"], 0)
                self.assertEqual(record["fit_split"], "train")
                self.assertIn("train", visited)
                self.assertTrue(any("val" in event for event, _ in store.reads))
                restored = self.training._load_head(record, "cpu")
                self.assertTrue(any(not torch.equal(initial[key], restored.state_dict()[key]) for key in initial))
                self.assertTrue(all("test" not in event for event, _ in store.reads))
                if kind in self.fusion.SCORE_KINDS:
                    self.assertTrue(all(set(fields) == {"score", "support", "stack_raw"} for _, fields in store.reads))
                else:
                    self.assertTrue(any("forecast" in fields and "target_features" in fields for _, fields in store.reads))
                # A completed compatible resume performs no new optimizer fit.
                with mock.patch.object(torch.optim, "AdamW", side_effect=AssertionError("retrained fusion")), \
                     mock.patch.object(torch.optim, "Adam", side_effect=AssertionError("retrained fusion")), \
                     redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    resumed = self.training._train_reduced_fusion(self.spec(config, kind), dataset, store,
                        records, directory, config, resume=True)
                self.assertEqual(resumed["fusion_checkpoint_id"], record["fusion_checkpoint_id"])

    def test_primary_fusion_checkpoint_selected_by_f1_not_recall_or_iou(self):
        dataset, store, records, config = self.fixture()
        config["fusion_epochs"] = 3
        validation = [dict(f1=.2, iou=.9, recall=.2, fpr=.02, threshold=.75),
                      dict(f1=.8, iou=.4, recall=.5, fpr=.02, threshold=.5),
                      dict(f1=.4, iou=.7, recall=.99, fpr=.02, threshold=.25)]
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(self.training._report(), "evaluate_predictor", side_effect=validation), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            record = self.training._train_reduced_fusion(self.spec(config, "logreg"), dataset, store,
                records, directory, config)
        self.assertEqual(record["selected_epoch"], 2)
        self.assertEqual(record["checkpoint"]["metrics"]["f1"], .8)
        self.assertEqual(record["winners"]["iou"]["epoch"], 1)
        self.assertEqual(record["winners"]["recall"]["epoch"], 3)

    def test_feature_unknown_final_event_still_flushes_previous_gradients(self):
        dataset, store, records, config = self.fixture()
        final = dataset.split_samples("train")[-1]
        store.arrays[final["event_id"]]["target"] = np.full_like(store.arrays[final["event_id"]]["target"], 255)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            record = self.training._train_reduced_fusion(self.spec(config, "feature-mlp"), dataset, store,
                records, directory, config)
        self.assertEqual(record["optimizer_steps"], 1)

    def test_canonical_nonlinear_outputs_agree_on_overlap_and_saved_evaluation(self):
        dataset, store, records, config = self.fixture()
        original = dataset.split_samples("val")[0]
        overlap = dict(original, event_id=original["event_id"] + "-overlap", bbox_xywh=[4, 0, 8, 8])
        dataset.samples.append(overlap)
        dataset.events[overlap["event_id"]] = overlap
        store.arrays[overlap["event_id"]] = dict(store.arrays[original["event_id"]])
        records.append(dict(next(r for r in records if r["event_id"] == original["event_id"]),
            event_id=overlap["event_id"], artifact_sha256="synthetic-overlap"))
        lookup = {record["event_id"]: record for record in records}
        for kind in self.fusion.SCORE_KINDS:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                head = self.fusion.build_head(kind)
                with self.training._score_outputs(head, dataset, store, lookup, "val", "cpu", directory) as predict:
                    saved = {row["event_id"]: predict(row) for row in dataset.split_samples("val")}
                    np.testing.assert_array_equal(saved[original["event_id"]]["score"][:, 4:8],
                        saved[overlap["event_id"]]["score"][:, :4])
                    online = self.training._report().evaluate_predictor(dataset, predict, split="val",
                        spec=self.spec(config, kind), budget=.05)
                self.assertFalse(list(Path(directory).iterdir()), "Temporary score blocks were not retired")
                replay = self.training._report().evaluate_predictor(dataset,
                    lambda row: saved[row["event_id"]], split="val", spec=self.spec(config, kind), budget=.05)
                for metric in ("threshold", "f1", "iou", "recall", "fpr"):
                    self.assertEqual(online[metric], replay[metric])


if __name__ == "__main__":
    unittest.main()
