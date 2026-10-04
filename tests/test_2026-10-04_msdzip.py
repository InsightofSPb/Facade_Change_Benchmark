import importlib
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from facade_change.io import sha256


msdzip = importlib.import_module("facade_change.2026-10-04_msdzip_h0")


class ResidualContextTests(unittest.TestCase):
    def test_official_predictor_is_exact_upstream_source(self):
        self.assertEqual(sha256(msdzip.SOURCE_PATH), msdzip.SOURCE_SHA256)
        self.assertIn("class MixedModel", msdzip.SOURCE_PATH.read_text())

    def test_abs_mod256_signed_subtraction_and_no_current_target_in_context(self):
        reference = np.full((1, 3, 3), 100, np.uint8)
        source = reference.copy()
        source[0, 0] = [110, 90, 100]
        self.assertEqual(msdzip.residual_bytes(reference, source, "abs")[0, 0].tolist(), [10, 10, 0])
        residual = msdzip.residual_bytes(reference, source, "mod256")
        self.assertEqual(residual[0, 0].tolist(), [10, 246, 0])
        support = np.ones((1, 3), bool)
        contexts, labels = msdzip._windows(residual, support, [0, 1, 2], 2)
        self.assertEqual(contexts.tolist(), [[0, 0], [0, 10], [10, 246]])
        mutated = residual.copy().reshape(-1)
        mutated[2] = 77
        after, new_labels = msdzip._windows(mutated.reshape(residual.shape), support, [2], 2)
        np.testing.assert_array_equal(contexts[2], after[0])
        self.assertNotEqual(labels[2], new_labels[0])

    def test_context_does_not_cross_unsupported_pixels_and_lanes_are_native(self):
        residual = np.arange(12, dtype=np.uint8).reshape(1, 4, 3)
        support = np.array([[True, False, True, True]])
        positions = np.array([0, 1, 2, 6, 7, 8, 9])
        contexts, targets = msdzip._windows(residual, support, positions, 4)
        self.assertEqual(contexts[3].tolist(), [0, 0, 0, 0])
        self.assertEqual(contexts[4].tolist(), [0, 0, 0, 6])
        visited = []
        for _, _, indices in msdzip._batches(contexts, targets, positions, 4):
            for lane, index in enumerate(indices):
                if index >= 0:
                    self.assertEqual(positions[index] % 4, lane)
                    visited.append(int(index))
        self.assertEqual(sorted(visited), list(range(len(positions))))


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("cv2"), "PyTorch/OpenCV required for original predictor smoke")
class OriginalMSDZipTests(unittest.TestCase):
    def setUp(self):
        import torch
        from test_hypothesis_dataset import HypothesisDatasetTests
        self.torch = torch
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.fixture = HypothesisDatasetTests("test_h0_uses_reference_and_sham_is_identical_with_matching_nuisance")
        self.fixture.setUp()
        self.root = self.fixture.root
        self.dataset, _, self.index = self.fixture._export("hypotheses")

    def tearDown(self):
        self.fixture.tearDown()
        self.torch.set_num_threads(self.old_threads)

    def _train(self, name="fit"):
        with patch.object(msdzip, "load_rgb", wraps=msdzip.load_rgb) as reader:
            result = msdzip.train_msdzip_h0(
                self.dataset, self.root / name, epochs=1, max_train_bytes=32, max_val_bytes=16,
                model_batch_size=4, timesteps=2, hidden_dim=4, ffn_dim=8, vocab_dim=2,
                max_bases_per_split=1, window_groups=2,
            )
        return result, reader.call_args_list

    def test_training_reads_only_unchanged_train_val_and_same_targets_for_representations(self):
        summary, reads = self._train()
        permitted = set()
        h1_or_test = set()
        for case in self.index["cases"]:
            if case["split"] in {"train", "val"} and case["state"] == "unchanged":
                permitted.update({self.dataset / case["reference_rgb"], self.dataset / case["source_rgb"]})
            elif case["state"] in {"crack", "paint_patch"} or case["split"] == "test":
                h1_or_test.add(self.dataset / case["source_rgb"])
        actual = {call.args[0] for call in reads}
        self.assertTrue(actual <= permitted)
        self.assertFalse(actual & h1_or_test)
        self.assertEqual(summary["sampled_bytes"], {"train": 32, "val": 16})
        self.assertEqual(summary["buildings"], {"train": ["b_train"], "val": ["b_val"]})
        abs_metadata = summary["results"]["abs"]["metadata"]
        mod_metadata = summary["results"]["mod256"]["metadata"]
        self.assertEqual(abs_metadata["sampling_sha256"], mod_metadata["sampling_sha256"])
        self.assertEqual(abs_metadata["training_case_ids"], mod_metadata["training_case_ids"])
        self.assertTrue(np.isfinite(summary["results"]["abs"]["best_val_bpb"]))

    def test_frozen_native_scores_ignore_earlier_cases_and_unsupported_rgb(self):
        summary, _ = self._train()
        checkpoint = self.root / "fit" / summary["results"]["abs"]["checkpoint_path"]
        scorer = msdzip.MSDZipScorer(checkpoint, dataset_fingerprint=summary["input_sha256"], representation="abs")
        before = {key: value.clone() for key, value in scorer.model.state_dict().items()}
        reference = np.arange(45, dtype=np.uint8).reshape(3, 5, 3)
        source = reference + np.uint8(1)
        support = np.ones((3, 5), bool)
        support[:, 2] = False
        first = scorer(reference, source, support)
        raw = scorer.raw_scores.copy()
        self.assertEqual(first.shape, support.shape)
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.isnan(first[~support]).all())
        self.assertTrue(np.isfinite(raw[support]).all())
        np.testing.assert_allclose(first[support], 1 - np.exp(-raw[support] / 8), atol=1e-7)
        scorer(reference, 255 - source, support)
        np.testing.assert_array_equal(scorer(reference, source, support), first)
        changed = source.copy()
        changed[~support] = 255
        np.testing.assert_array_equal(scorer(reference, changed, support), first)
        self.assertEqual(scorer.model.last, [])
        for key, tensor in scorer.model.state_dict().items():
            self.assertTrue(self.torch.equal(before[key], tensor))
        self.assertFalse(any(parameter.requires_grad for parameter in scorer.model.parameters()))

    def test_checkpoint_representation_and_dataset_mismatch_fail(self):
        summary, _ = self._train()
        checkpoint = self.root / "fit" / summary["results"]["abs"]["checkpoint_path"]
        with self.assertRaisesRegex(ValueError, "representation differs"):
            msdzip.MSDZipScorer(checkpoint, representation="mod256")
        fingerprint = {**summary["input_sha256"], "split": "changed"}
        with self.assertRaisesRegex(ValueError, "dataset/split fingerprint"):
            msdzip.MSDZipScorer(checkpoint, dataset_fingerprint=fingerprint)
        data = self.torch.load(checkpoint, weights_only=True)
        data["metadata"]["buildings"]["train"] = ["b_test"]
        bad = self.root / "bad.pt"
        self.torch.save(data, bad)
        with self.assertRaisesRegex(ValueError, "training buildings"):
            msdzip.MSDZipScorer(bad)

    def test_original_shape_constraints_and_finite_logits(self):
        for args in ((4, 3, 6, 8, 2), (4, 2, 3, 8, 2)):
            with self.assertRaises(ValueError):
                msdzip._model_args(*args)
        model = msdzip._new_model(msdzip._model_args(4, 2, 4, 8, 2), self.torch.device("cpu"))
        logits = msdzip._predict(model, self.torch.zeros((4, 2), dtype=self.torch.long))
        self.assertEqual(tuple(logits.shape), (4, 256))
        self.assertTrue(self.torch.isfinite(logits).all())

    def test_vectorized_groups_preserve_original_lane_logits_and_gradients(self):
        model = msdzip._new_model(msdzip._model_args(4, 2, 4, 8, 2), self.torch.device("cpu"))
        contexts = self.torch.arange(16, dtype=self.torch.long).reshape(2, 4, 2)
        serial = self.torch.stack([msdzip._predict(model, x) for x in contexts])
        serial.square().sum().backward()
        serial_grad = {key: parameter.grad.clone() for key, parameter in model.named_parameters() if parameter.grad is not None}
        model.zero_grad(set_to_none=True)
        grouped = msdzip._predict_groups(model, contexts)
        grouped.square().sum().backward()
        self.torch.testing.assert_close(grouped, serial, rtol=1e-5, atol=1e-6)
        for key, parameter in model.named_parameters():
            if key in serial_grad:
                self.torch.testing.assert_close(parameter.grad, serial_grad[key], rtol=1e-4, atol=1e-5)
        self.assertEqual(model.last, [])


if __name__ == "__main__":
    unittest.main()
