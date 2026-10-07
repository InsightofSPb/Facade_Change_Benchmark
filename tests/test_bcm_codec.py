"""BCM container, conditional cost, rejection and optional author roundtrip."""
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from facade_change.methods.bcm_net import (BCMCodec, load_author_model, pack_channel,
                                          split_vvc_access_units, unpack_channel)
from facade_change.methods.lossless import TileCodecScorer


def nal(kind, payload=b"x", prefix=b"\0\0\0\1"):
    return prefix + bytes((0, kind * 8 + 1)) + payload


class BCMCodecTests(unittest.TestCase):
    def test_actual_access_units_cover_every_byte_and_reject_missing_framing(self):
        first = nal(20) + nal(15) + nal(16) + nal(7)
        second = nal(20, prefix=b"\0\0\1") + nal(17) + nal(1) + nal(21)
        self.assertEqual(split_vvc_access_units(first + second), (first, second))
        for invalid in (nal(7) + nal(1), first, first + second + nal(20), nal(20) + second):
            with self.subTest(stream=invalid), self.assertRaises(RuntimeError):
                split_vvc_access_units(invalid)

    def test_original_container_roundtrip_and_reference_plus_b_cost(self):
        stream = nal(20) + nal(7) + nal(20) + nal(1)
        corrections = [[b"a", b"bb", b"ccc", b"dddd"], [b"e", b"ff", b"ggg", b"hhhh"]]
        data = pack_channel(stream, corrections, 32, 32)
        self.assertEqual(unpack_channel(data), (stream, corrections))
        self.assertEqual(len(data), 10 + len(stream) + 2 * 16 + 20)
        first, second = split_vvc_access_units(stream)
        reference = 6 + len(first) + 16 + sum(map(len, corrections[0]))
        target = len(second) + 20 + sum(map(len, corrections[1]))
        self.assertEqual(reference + target, len(data))
        for invalid in (data[:-1], data + b"extra"):
            with self.assertRaises(ValueError):
                unpack_channel(invalid)

    def test_theoretical_rgb_pair_uses_a_and_cannot_claim_full_bitstream(self):
        seen = []
        def theoretical(a, b):
            seen.append((a.copy(), b.copy()))
            return 2.
        codec = SimpleNamespace(metadata={"cost_mode": "theoretical"}, theoretical_bpb=theoretical)
        scorer = TileCodecScorer()
        scorer._configure_tiles("bcm_net_rgb", codec, 32, 32, "rgb_pair")
        a = np.full((32, 32, 3), 10, dtype=np.uint8)
        b = np.full_like(a, 30)
        scorer(a, b, np.ones((32, 32), dtype=bool))
        np.testing.assert_array_equal(seen[0][0], a)
        np.testing.assert_array_equal(seen[0][1], b)
        self.assertFalse(scorer.metadata["last_codec_stats"]["bitstream_measured"])
        self.assertFalse(scorer.metadata["last_codec_stats"]["all_tiles_roundtrip_verified"])

    def test_missing_original_source_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
                "facade_change.methods.bcm_net.importlib.util.spec_from_file_location",
                side_effect=AssertionError("imported")):
            with self.assertRaisesRegex(ValueError, "pinned original"):
                load_author_model(tmp)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required")
    def test_decoder_error_rejected_even_if_uint8_cast_could_hide_it(self):
        import torch
        class BadNetwork:
            def compress(self, *args, **kwargs):
                return [b"x"] * 4
            def decompress(self, strings, base, *args, **kwargs):
                return torch.zeros_like(base) + 256
        codec = BCMCodec.__new__(BCMCodec)
        codec.network, codec.device = BadNetwork(), "cpu"
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        stream = nal(20) + nal(7) + nal(20) + nal(1)
        first, second = split_vvc_access_units(stream)
        part = {"b_base_bytes": len(second), "reference_base_bytes": len(first), "stream_bytes": len(stream)}
        with patch.object(codec, "_base_channels", return_value=(image, image, [stream]*3, [part]*3)), patch.object(
                codec, "_decode_base", return_value=np.zeros((2, 32, 32), dtype=np.uint8)):
            with self.assertRaisesRegex(RuntimeError, "exact RGB roundtrip"):
                codec.encode(image, image)


@unittest.skipUnless(os.environ.get("FACADE_BCM_SOURCE") and os.environ.get("FACADE_BCM_VTM_ENCODER")
                     and os.environ.get("FACADE_BCM_VTM_DECODER") and os.environ.get("FACADE_BCM_VTM_CONFIG"),
                     "Set BCM source and VTM encoder/decoder/config env paths for full author smoke")
class OriginalBCMCodecTests(unittest.TestCase):
    def test_actual_vtm_and_original_neural_coder_restore_native_rgb(self):
        codec = BCMCodec(os.environ["FACADE_BCM_SOURCE"], device="cpu",
            vtm_encoder=os.environ["FACADE_BCM_VTM_ENCODER"],
            vtm_decoder=os.environ["FACADE_BCM_VTM_DECODER"],
            vtm_config=os.environ["FACADE_BCM_VTM_CONFIG"])
        y, x = np.indices((32, 32))
        a = np.stack(((x * 3 + y * 2) % 256, (x + y) % 256, (x * 2 + y * 4) % 256), axis=-1).astype(np.uint8)
        b = a.copy()
        b[8:16, 8:16] = (b[8:16, 8:16].astype(np.uint16) + 7).astype(np.uint8)
        try:
            reference_hash = None
            for target in (a, b, np.zeros_like(a)):
                size, stats = codec.encode(a, target)
                self.assertTrue(stats["roundtrip_verified"])
                self.assertEqual(size + stats["reference_bytes"], stats["stream_bytes"])
                self.assertGreater(stats["b_base_bytes"], 0)
                if reference_hash is None:
                    reference_hash = stats["reference_channel_sha256"]
                self.assertEqual(reference_hash, stats["reference_channel_sha256"],
                                 "Reference bytes must not carry free information fromB")
            self.assertGreater(codec.theoretical_bpb(a, b), 0)
            for reference, target in ((np.full_like(a, 255), np.zeros_like(a)),
                                      (np.zeros_like(a), np.full_like(a, 255))):
                _, stats = codec.encode(reference, target)
                self.assertTrue(stats["roundtrip_verified"])
        finally:
            codec.close()


if __name__ == "__main__":
    unittest.main()
