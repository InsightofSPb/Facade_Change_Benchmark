"""H0 isolation and honest costs for the original neural image codec."""
import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from facade_change.codec_training import sample_h0_tiles, _tiles
from facade_change.methods.arib_bps import ArIBCodec, load_author_model
from facade_change.methods.lossless import TileCodecScorer
from facade_change.io import write_json


class ArIBPipelineTests(unittest.TestCase):
    def test_theoretical_cost_cannot_claim_coded_bytes_or_exact_restoration(self):
        codec = SimpleNamespace(metadata={"cost_mode": "theoretical"}, theoretical_bpb=lambda tile: 2.)
        scorer = TileCodecScorer()
        scorer._configure_tiles("arib_bps_mod256", codec, tile_size=32, stride=32)
        a = np.zeros((32, 32, 3), dtype=np.uint8)
        scorer(a, a, np.ones((32, 32), dtype=bool))
        stats = scorer.metadata["last_codec_stats"]
        self.assertFalse(stats["bitstream_measured"])
        self.assertFalse(stats["all_tiles_roundtrip_verified"])
        self.assertNotIn("charged_bytes", stats)
        self.assertIn("not a bitstream rate", scorer.metadata["raw_units"])

    def test_missing_original_sources_fail_before_any_model_import(self):
        with tempfile.TemporaryDirectory() as folder, patch(
                "facade_change.methods.arib_bps.importlib.import_module", side_effect=AssertionError("imported")):
            with self.assertRaisesRegex(ValueError, "pinned original"):
                load_author_model(folder)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required")
    def test_edited_training_config_is_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as folder, patch(
                "facade_change.methods.arib_bps.load_author_model", side_effect=AssertionError("loaded")):
            write_json(Path(folder) / "run.json", {"schema_version": 1, "kind": "arib_h0_training",
                "status": "completed_exploratory", "config": {"author_config": "changed"},
                "config_sha256": "old"})
            with self.assertRaisesRegex(ValueError, "configuration hash changed"):
                ArIBCodec("unused", folder, "abs")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required")
    def test_author_roundtrip_mismatch_cannot_yield_a_score(self):
        class BadCoder:
            def compress_to_file(self, x, path):
                Path(path).write_bytes(b"stream")

            def decompress_from_file(self, source, destination, device):
                Image.fromarray(np.ones((32, 32, 3), dtype=np.uint8)).save(destination)

        codec = ArIBCodec.__new__(ArIBCodec)
        codec.model, codec.device, codec.seed = BadCoder(), "cpu", 42
        zeros = np.zeros((32, 32, 3), dtype=np.uint8)
        with self.assertRaisesRegex(RuntimeError, "exact RGB roundtrip"):
            codec.encode(zeros, zeros)

    def test_h0_sampling_ignores_test_sham_and_damage_and_zeroes_unsupported_bytes(self):
        bases = [{"base_id": part+"-base", "building_id": part+"-building", "split": part}
                 for part in ("train", "val", "test")]
        cases = []
        for base in bases:
            for state in ("unchanged", "self_paste", "crack", "paint_patch"):
                cases.append({**base, "case_id": base["base_id"]+"-"+state, "state": state,
                    "hypothesis": "H1" if state in {"crack", "paint_patch"} else "H0",
                    "sham_self_paste": state == "self_paste", "scenario_id": "shadow"})
        h0 = importlib.import_module("facade_change.2026-10-04_msdzip_h0")
        support = np.ones((48, 48), dtype=bool)
        support[:4] = False

        def rgb(*args):
            a = np.full((48, 48, 3), 255, dtype=np.uint8)
            return a, np.zeros_like(a), support

        with patch.object(h0, "_rgb_inputs", side_effect=rgb) as reader:
            plan = sample_h0_tiles(None, None, bases, cases, 32, {"train": 4, "val": 2}, 42, {})
            again = sample_h0_tiles(None, None, bases, list(reversed(cases)), 32,
                                    {"train": 4, "val": 2}, 42, {})
            arrays = _tiles(None, None, bases, cases, plan["train"], "mod256", 32, {})
        self.assertEqual(plan, again)
        self.assertEqual({part:len(rows) for part,rows in plan.items()}, {"train":4,"val":2})
        self.assertTrue(any(row["support_fraction"] < 1 for row in plan["train"]))
        for call in reader.call_args_list:
            case = call.args[3]
            self.assertEqual(case["state"], "unchanged")
            self.assertIn(case["split"], ("train", "val"))
        for tile, item in zip(arrays, plan["train"]):
            valid = np.zeros((32, 32), dtype=bool)
            crop = support[item["row"]:item["row"]+32, item["col"]:item["col"]+32]
            valid[:crop.shape[0], :crop.shape[1]] = crop
            self.assertTrue((tile[~valid] == 0).all())
            self.assertTrue((tile[valid] == 1).all())
        self.assertFalse(set(item["building_id"] for item in plan["train"]) &
                         set(item["building_id"] for item in plan["val"]))


if __name__ == "__main__":
    unittest.main()
