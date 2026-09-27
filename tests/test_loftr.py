"""Loader safety and coarse-mask regression tests; no weights or GPU required."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from facade_change.alignment import load_loftr_state, loftr_inputs, resolve_loftr_checkpoint


class LoFTRTests(unittest.TestCase):
    def test_modern_loader_never_silently_falls_back_to_pickle(self):
        calls = []

        def load(path, map_location=None, weights_only=False):
            calls.append((map_location, weights_only))
            raise ValueError("unsafe serialized object")

        torch = SimpleNamespace(load=load)
        with self.assertRaisesRegex(ValueError, "unsafe serialized"):
            load_loftr_state(torch, "weights.ckpt", trust_checkpoint=True)
        self.assertEqual(calls, [("cpu", True)])

    def test_legacy_loader_requires_explicit_trust_and_records_it(self):
        tensor = object()
        calls = []

        def load(path, map_location=None):
            calls.append((path, map_location))
            return {"state_dict": {"weight": tensor}}

        torch = SimpleNamespace(load=load, is_tensor=lambda value: value is tensor)
        with self.assertRaisesRegex(RuntimeError, "--trust-checkpoint"):
            load_loftr_state(torch, "weights.ckpt")
        self.assertEqual(calls, [])
        with self.assertWarns(RuntimeWarning):
            state, loader = load_loftr_state(torch, "weights.ckpt", trust_checkpoint=True)
        self.assertEqual(state, {"weight": tensor})
        self.assertEqual(loader, "legacy_pickle_explicitly_trusted")
        self.assertEqual(calls, [("weights.ckpt", "cpu")])

    def test_modern_loader_validates_tensor_state_dictionary(self):
        def load(path, map_location=None, weights_only=False):
            self.assertTrue(weights_only)
            return {"state_dict": {"weight": "wrong"}}

        torch = SimpleNamespace(load=load, is_tensor=lambda value: False)
        with self.assertRaisesRegex(ValueError, "string-to-tensor"):
            load_loftr_state(torch, "weights.ckpt")

    def test_cache_lookup_and_download_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "checkpoints"
            download = Mock()
            torch = SimpleNamespace(hub=SimpleNamespace(
                get_dir=lambda: directory, download_url_to_file=download))
            with self.assertRaises(FileNotFoundError):
                resolve_loftr_checkpoint(torch)
            download.assert_not_called()
            cache.mkdir()
            checkpoint = cache / "loftr_outdoor.ckpt"
            checkpoint.write_bytes(b"fixture")
            path, origin = resolve_loftr_checkpoint(torch, "auto")
            self.assertEqual(path, checkpoint.resolve())
            self.assertEqual(origin, "torch_cache")
            with self.assertRaises(FileNotFoundError):
                resolve_loftr_checkpoint(torch, cache / "missing.ckpt", download_weights=True)
            download.assert_not_called()

    def test_unequal_shapes_keep_native_top_left_and_exclude_padding(self):
        gray = np.arange(17 * 25, dtype=np.uint16).reshape(17, 25).astype(np.uint8)
        support = np.ones(gray.shape, dtype=bool)
        support[4, 10] = False
        image, coarse = loftr_inputs(gray, support)
        self.assertEqual(image.shape, (24, 32))
        self.assertEqual(coarse.shape, (3, 4))
        np.testing.assert_array_equal(image[:17, :25], gray)
        np.testing.assert_array_equal(coarse, [
            [True, False, True, False],
            [True, True, True, False],
            [False, False, False, False],
        ])
        other, other_mask = loftr_inputs(np.zeros((32, 16), np.uint8), np.ones((32, 16), bool))
        self.assertEqual(other.shape, (32, 16))
        self.assertEqual(other_mask.shape, (4, 2))
        self.assertTrue(other_mask.all())


if __name__ == "__main__":
    unittest.main()
