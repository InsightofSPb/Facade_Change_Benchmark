"""CPU round trips and accounting for the small paid H0/H1 codec."""
from __future__ import annotations

import hashlib
import unittest
import zlib

import numpy as np

from facade_change.research import paid_codec as codec


class PaidCodecTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.reference = rng.integers(0, 256, size=(24, 29, 3), dtype=np.uint8)
        self.support = np.ones(self.reference.shape[:2], dtype=bool)
        self.support[2:5, 3:7] = False
        self.candidate = np.zeros(self.support.shape, dtype=bool)
        self.candidate[7:19, 8:23] = True

    def _roundtrip(self, target, candidate=None):
        stream, accounting = codec.encode(self.reference, target, self.support, candidate)
        decoded = codec.decode(self.reference, self.support, stream)
        np.testing.assert_array_equal(decoded[self.support], target[self.support])
        np.testing.assert_array_equal(decoded[~self.support], self.reference[~self.support])
        self.assertEqual(accounting["total_bits"], 8 * len(stream))
        self.assertEqual(sum(accounting["partitions"].values()), 8 * len(stream))
        self.assertEqual(set(accounting["partitions"]), set(codec.PARTITIONS))
        self.assertEqual(accounting["compressor"]["name"], "stdlib_zlib")
        return stream, accounting

    def _replace_sections(self, stream, *, nuisance=None, bitmap=None, color=None, residual=None):
        """Re-seal deliberately malformed sections to exercise decoder guards."""
        values = list(codec.HEADER.unpack(stream[:codec.HEADER.size]))
        sizes = values[6:10]
        offset, sections = codec.HEADER.size, []
        for size in sizes:
            sections.append(stream[offset:offset + size])
            offset += size
        replacements = (nuisance, bitmap, color, residual)
        sections = [original if replacement is None else replacement
                    for original, replacement in zip(sections, replacements)]
        values[6:10] = [len(section) for section in sections]
        payload = b"".join(sections)
        values[-1] = hashlib.sha256(payload).digest()
        return codec.HEADER.pack(*values) + payload

    def test_identity_both_arms_and_empty_update_are_paid(self):
        self._roundtrip(self.reference)
        _, paid = self._roundtrip(self.reference, np.zeros(self.support.shape, dtype=bool))
        self.assertGreater(paid["partitions"]["C_bits"], 0)
        self.assertEqual(paid["partitions"]["V_bits"], 32)
        self.assertEqual(paid["arm"], "H1")

    def test_mod256_wraparound_nonrectangular_candidate(self):
        target = self.reference.copy()
        candidate = self.candidate.copy()
        candidate[10:14, 12:16] = False
        target[candidate] = (target[candidate].astype(np.uint16) + (250, 17, 129)) % 256
        self._roundtrip(target)
        self._roundtrip(target, candidate)

    def test_fill_shift_and_different_unsupported_target(self):
        target = self.reference.copy()
        target[self.candidate] = (200, 40, 80)
        target[~self.support] = (4, 6, 9)
        self._roundtrip(target, self.candidate)
        _, accounting = self._roundtrip(target)
        self.assertEqual(accounting["support_contract"], codec.SUPPORT_CONTRACT)

    def test_colour_mode_selection_and_unsupported_target_invariance(self):
        rng = np.random.default_rng(31)
        reference = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        support = np.ones((64, 64), dtype=bool)
        support[:4, :4] = False
        candidate = np.zeros_like(support)
        candidate[16:50, 9:57] = True
        for expected in ("fill", "modular_shift"):
            target = reference.copy()
            target[candidate] = ((23, 34, 51) if expected == "fill" else
                                 (target[candidate].astype(np.uint16) + (23, 34, 51)) % 256)
            stream, accounting = codec.encode(reference, target, support, candidate)
            self.assertEqual(accounting["update_mode"], expected)
            changed = target.copy()
            changed[~support] ^= 255
            self.assertEqual(codec.encode(reference, changed, support, candidate)[0], stream)
            np.testing.assert_array_equal(codec.decode(reference, support, stream)[support], target[support])

    def test_shared_reference_and_support_mismatch_fail(self):
        stream, _ = self._roundtrip(self.reference)
        wrong_reference = self.reference.copy()
        wrong_reference[0, 0, 0] ^= 1
        wrong_support = self.support.copy()
        wrong_support[0, 0] = False
        for reference, support in ((wrong_reference, self.support), (self.reference, wrong_support)):
            with self.assertRaisesRegex(ValueError, "hash"):
                codec.decode(reference, support, stream)

    def test_tamper_truncation_and_trailing_bytes_fail(self):
        stream, _ = self._roundtrip(self.reference, self.candidate)
        changed = bytearray(stream)
        changed[-1] ^= 1
        for invalid in (bytes(changed), stream[:-1], stream[:10], stream + b"extra"):
            with self.assertRaises(ValueError):
                codec.decode(self.reference, self.support, invalid)

    def test_malformed_update_and_nuisance_fail_after_resealing(self):
        stream, _ = self._roundtrip(self.reference, self.candidate)
        for invalid in (self._replace_sections(stream, color=b"\x02\0\0\0"),
                        self._replace_sections(stream, nuisance=b"\x02")):
            with self.assertRaises(ValueError):
                codec.decode(self.reference, self.support, invalid)

    def test_compressed_section_strict_boundaries_and_sizes(self):
        stream, _ = self._roundtrip(self.reference, self.candidate)
        values = codec.HEADER.unpack(stream[:codec.HEADER.size])
        residual = stream[-values[9]:]
        for payload in (residual + b"extra", residual[:-1], zlib.compress(b"short"),
                        zlib.compress(bytes((int(self.support.sum()) * 3) + 1))):
            invalid = self._replace_sections(stream, residual=payload)
            with self.assertRaises(ValueError):
                codec.decode(self.reference, self.support, invalid)

    def test_mask_padding_and_outside_support_fail(self):
        # 24*29=696 is byte-aligned; change fixture width for a padding guard.
        reference = self.reference[:, :28].copy()[:23]
        support = self.support[:, :28].copy()[:23]
        candidate = np.zeros(support.shape, dtype=bool)
        stream, _ = codec.encode(reference, reference, support, candidate)
        packed = np.packbits(candidate.reshape(-1), bitorder="big")
        packed[-1] |= 1
        invalid = self._replace_sections(stream, bitmap=zlib.compress(packed.tobytes(), level=9))
        with self.assertRaisesRegex(ValueError, "padding"):
            codec.decode(reference, support, invalid)
        candidate[2, 3] = True
        with self.assertRaises(ValueError):
            codec.encode(reference, reference, support, candidate)
        packed = np.packbits(candidate.reshape(-1), bitorder="big")
        invalid = self._replace_sections(stream, bitmap=zlib.compress(packed.tobytes(), level=9))
        with self.assertRaisesRegex(ValueError, "support"):
            codec.decode(reference, support, invalid)

    def test_wrong_types_empty_support_and_shape_fail(self):
        for target, support, candidate in (
            (self.reference.astype(np.float32), self.support, None),
            (self.reference, self.support.astype(np.uint8), None),
            (self.reference, np.zeros_like(self.support), None),
            (self.reference, self.support, self.candidate.astype(np.uint8)),
            (self.reference[:2], self.support, None),
        ):
            with self.assertRaises(ValueError):
                codec.encode(self.reference, target, support, candidate)

    def test_both_arms_deterministic_actual_delta(self):
        target = self.reference.copy()
        target[self.candidate] = (32, 190, 70)
        first = codec.compare_hypotheses(self.reference, target, self.support, self.candidate)
        second = codec.compare_hypotheses(self.reference, target, self.support, self.candidate)
        self.assertEqual(first, second)
        self.assertEqual(first["delta_bits"], 8 * (len(first["h0_stream"]) - len(first["h1_stream"])))
        self.assertEqual(first["delta_bits_per_supported_pixel"], first["delta_bits"] / self.support.sum())
        for arm in ("h0", "h1"):
            np.testing.assert_array_equal(codec.decode(self.reference, self.support, first[arm + "_stream"])[self.support],
                                          target[self.support])
        with self.assertRaises(ValueError):
            codec.compare_hypotheses(self.reference, target, self.support, None)

    def test_affine_family_can_explain_photometric_nuisance(self):
        # Sufficient pixels make affine parameters worth their transmitted bytes.
        rng = np.random.default_rng(5)
        reference = rng.integers(0, 160, (80, 80, 3), dtype=np.uint8)
        target = np.clip((reference.astype(np.uint16) * 3 + 1) // 2 + (10, 20, 30), 0, 255).astype(np.uint8)
        support = np.ones(reference.shape[:2], dtype=bool)
        stream, accounting = codec.encode(reference, target, support)
        self.assertEqual(accounting["predictor_mode"], "affine")
        np.testing.assert_array_equal(codec.decode(reference, support, stream), target)


if __name__ == "__main__":
    unittest.main()
