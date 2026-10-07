import importlib.util
import os
import random
import sys
import tempfile
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from facade_change.methods.arib_compat import (
    canonical_normalized_rgb, install_posterior_rgb_compatibility,
)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required")
class PosteriorCompatibilityTests(unittest.TestCase):
    def test_canonicalization_recovers_encoder_context_and_wrapper_is_reversible(self):
        import torch

        values = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 16, 16).expand(1, 3, 16, 16)
        expected = (values >> 4 << 4).float() / 255.0
        decoded = torch.zeros_like(expected)
        for shift in (7, 6, 5, 4):
            decoded += ((values >> shift) & 1).float() * (1 << shift) / 255.0
        self.assertTrue(torch.any(decoded != expected))
        torch.testing.assert_close(canonical_normalized_rgb(decoded), expected, rtol=0, atol=0)
        calls = []

        def original(image, *args):
            calls.append((image, args))
            return "original-result"

        posterior = SimpleNamespace(_compress_qz=original)
        model = SimpleNamespace(sig=SimpleNamespace(lvae=posterior))
        handle = install_posterior_rgb_compatibility(model)
        self.assertIs(install_posterior_rgb_compatibility(model), handle)
        self.assertEqual(posterior._compress_qz(decoded, "ids", "mups", "inputs", "coder"), "original-result")
        torch.testing.assert_close(calls[0][0], expected, rtol=0, atol=0)
        self.assertEqual(calls[0][1], ("ids", "mups", "inputs", "coder"))
        self.assertFalse(handle.metadata["source_files_modified"])
        metadata = handle.metadata
        metadata["source_files_modified"] = True
        self.assertFalse(handle.metadata["source_files_modified"])
        handle.close()
        handle.close()
        self.assertIs(posterior._compress_qz, original)
        another = install_posterior_rgb_compatibility(model)
        self.assertIsNot(another, handle)
        another.close()

    def test_wrong_domains_are_rejected_and_cleanup_preserves_later_changes(self):
        import torch

        for image in (torch.zeros((1, 3, 4, 4), dtype=torch.uint8),
                      torch.zeros((1, 1, 4, 4)), torch.full((1, 3, 4, 4), float("nan")),
                      torch.full((1, 3, 4, 4), 1.1)):
            with self.assertRaises(ValueError):
                canonical_normalized_rgb(image)
        model = SimpleNamespace(sig=SimpleNamespace(lvae=SimpleNamespace(_compress_qz=lambda *args: None)))
        handle = install_posterior_rgb_compatibility(model)
        replacement = lambda *args: None
        model.sig.lvae._compress_qz = replacement
        with self.assertRaisesRegex(RuntimeError, "changed after"):
            handle.close()
        self.assertIs(model.sig.lvae._compress_qz, replacement)


@unittest.skipUnless(os.environ.get("FACADE_ARIB_SOURCE") and importlib.util.find_spec("torch"),
                     "Set FACADE_ARIB_SOURCE to the original compiled author source for the CPU roundtrip smoke")
class OriginalArIBRoundtripTests(unittest.TestCase):
    def test_original_sparse_failure_then_exact_fixed_zero_random_and_sparse(self):
        import torch

        source = Path(os.environ["FACADE_ARIB_SOURCE"]).resolve()
        if not (source / "src/utils/coder/mixcoder.so").is_file():
            self.skipTest("Compile the original author mixcoder first")
        from facade_change.io import read_json, sha256
        manifest = read_json(Path(__file__).resolve().parents[1] / "third_party/arib_bps_provenance.json")
        for relative, digest in manifest["sha256"].items():
            self.assertEqual(sha256(source / relative), digest, f"Original source mismatch: {relative}")
        sys.path.insert(0, str(source / "src"))
        from modules.arib_bps import ARIB_BPS
        from config.imagenet32_config import CFG

        previous_threads, previous_mkldnn = torch.get_num_threads(), torch.backends.mkldnn.enabled
        torch.set_num_threads(1)
        torch.backends.mkldnn.enabled = False
        torch.manual_seed(42)
        model = ARIB_BPS(**asdict(CFG), dropout=0).eval().requires_grad_(False)
        sparse = np.zeros((32, 32, 3), np.uint8)
        sparse[8:16, 8:16] = [255, 80, 20]
        handle = None
        try:
            with tempfile.TemporaryDirectory(prefix="arib-compat-smoke-") as temporary, torch.inference_mode():
                stream, restored = Path(temporary) / "tile.arib", Path(temporary) / "restored.png"

                def encode(image):
                    random.seed(42)
                    torch.manual_seed(42)
                    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0)
                    model.compress_to_file(tensor, str(stream))
                    return stream.read_bytes()

                def decode():
                    model.decompress_from_file(str(stream), str(restored), "cpu")
                    with Image.open(restored) as image:
                        return np.asarray(image).copy()

                original_stream = encode(sparse)
                original_decoded = decode()
                self.assertGreater(np.count_nonzero(original_decoded != sparse), 0)
                handle = install_posterior_rgb_compatibility(model)
                np.testing.assert_array_equal(decode(), sparse)
                self.assertEqual(encode(sparse), original_stream)
                np.testing.assert_array_equal(decode(), sparse)
                controls = {
                    "zero": np.zeros((32, 32, 3), np.uint8),
                    "random": np.random.default_rng(42).integers(0, 256, (32, 32, 3), dtype=np.uint8),
                }
                for name, image in controls.items():
                    started = time.perf_counter()
                    encoded = encode(image)
                    np.testing.assert_array_equal(decode(), image)
                    print(f"ArIB compatibility {name}: {len(encoded)} actual bytes, "
                          f"{time.perf_counter()-started:.3f} CPU seconds, exact RGB roundtrip")
        finally:
            if handle:
                handle.close()
            torch.set_num_threads(previous_threads)
            torch.backends.mkldnn.enabled = previous_mkldnn


if __name__ == "__main__":
    unittest.main()
