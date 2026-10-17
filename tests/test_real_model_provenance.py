"""Real temporal producer layout, stripped snapshots, and strict model identity."""
import contextlib
import copy
import importlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.real_eval import night_run, robustness_run
from scripts.real_eval.robustness_common import atomic_json, atomic_npz, read_json, sha256
from tests.test_real_temporal_run import Factory, fixture, metadata


class SourceModelProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.prepared, self.source = fixture(self.root)
        self.manifest = read_json(self.prepared / "manifest.json")

    def tearDown(self):
        self.temp.cleanup()

    def test_real_temporal_producer_checkpoint_is_found_without_snapshot_metadata(self):
        temporal = importlib.import_module("facade_change.2026-10-14_temporal_experiment")
        checkpoint = self.root / "abs.pt"
        checkpoint.write_bytes(b"recorded full-training checkpoint; never loaded")
        value = {**metadata("msdzip_abs"), "checkpoint_path": str(checkpoint),
                 "checkpoint_sha256": sha256(checkpoint), "checkpoint_dataset_verified": True,
                 "input_sha256": {"dataset": "original-training-fingerprint"},
                 "buildings": {"train": ["train-facade"]}}
        scorer = SimpleNamespace(activate=lambda: copy.deepcopy(value))
        store = SimpleNamespace(out=self.source, event=lambda *args, **kwargs: None)
        temporal._activate_verified(store, {"id": "msdzip_abs", "method": "msdzip_abs"},
                                    scorer, [], "inference")
        self.manifest["source_records"] = [
            {"method": "msdzip_abs", "status": "completed", "variant": "current_train"}]
        contract = robustness_run._method_contract("msdzip_abs", self.manifest, self.source, None)
        path = self.source / "method_provenance/inference_msdzip_abs.json"
        self.assertEqual(contract["provenance_file"], str(path))
        self.assertEqual(contract["provenance_sha256"], sha256(path))
        self.assertEqual(contract["options"]["checkpoint_path"], str(checkpoint))
        self.assertEqual(contract["options"]["dataset_fingerprint"], value["input_sha256"])
        checkpoint.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "отсутствует или изменился"):
            robustness_run._method_contract("msdzip_abs", self.manifest, self.source, None)

    def test_neural_methods_use_inference_layout_with_stripped_snapshot(self):
        for method in ("lpips", "dinov2", "semantic_dino", "anychange", "geoscd", "msdzip_mod256"):
            with self.subTest(method=method):
                value = metadata(method)
                if method.startswith("msdzip_"):
                    checkpoint = self.root / "mod256.pt"
                    checkpoint.write_bytes(b"full-mod256")
                    value.update(checkpoint_path=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
                path = self.source / "method_provenance" / f"inference_{method}.json"
                atomic_json(path, value)
                self.manifest["source_records"] = [{"method": method, "status": "completed",
                    "variant": "current_train" if method.startswith("msdzip_") else "frozen"}]
                contract = robustness_run._method_contract(method, self.manifest, self.source, None)
                self.assertEqual(contract["provenance_file"], str(path))
                self.assertEqual(contract["expected_metadata"], value)

    def test_conflicting_full_and_inference_metadata_are_rejected(self):
        path = self.source / "method_provenance/inference_dinov2.json"
        atomic_json(path, {**metadata("dinov2"), "fixture": "different-model"})
        with self.assertRaisesRegex(ValueError, "противоречивое происхождение"):
            robustness_run._method_contract("dinov2", self.manifest, self.source, None)

    def test_smoke_model_is_never_substituted_for_full_training(self):
        checkpoint = self.root / "smoke.pt"
        checkpoint.write_bytes(b"smoke-only")
        atomic_json(self.source / "method_provenance/smoke_msdzip_abs.json",
                    {**metadata("msdzip_abs"), "checkpoint_path": str(checkpoint),
                     "checkpoint_sha256": sha256(checkpoint)})
        with self.assertRaisesRegex(ValueError, "checkpoint_path/checkpoint_sha256"):
            robustness_run._method_contract("msdzip_abs", self.manifest, self.source, None)

    def test_pilot_works_with_actual_layout_and_no_embedded_model_metadata(self):
        for path in (self.source / "method_provenance").glob("full_*.json"):
            path.rename(path.with_name(path.name.replace("full_", "inference_", 1)))
        factory = Factory()
        with patch.object(night_run, "_snapshot_gpu"), contextlib.redirect_stdout(io.StringIO()):
            result = night_run.run(self.prepared, self.root / "pilot", methods=("rgb_diff", "dinov2", "geoscd"),
                                  pilot=True, pilot_pairs=2, pilot_history_frames=3, pool_factory=factory)
        self.assertEqual(result["status"], "completed")
        self.assertEqual((result["total"], result["completed"], result["unprocessed"]), (15, 15, 0))
        self.assertEqual(result["method_failures"], {})
        self.assertEqual(set(factory.activations), {"rgb_diff", "dinov2", "geoscd"})

    def test_full_queue_reuses_hash_bound_map_from_stripped_snapshot(self):
        sequence = self.manifest["sequences"][0]
        inputs, _, _ = night_run._input(self.prepared, sequence, 1, "previous")
        path = self.source / "maps/previous-ready.npz"
        atomic_npz(path, {**inputs, "scores": np.where(inputs["support"], .2, np.nan).astype(np.float32)})
        original = {"unit_id": "previous-ready", "method": "rgb_diff", "variant": "frozen",
                    "status": "completed", "domain": "real", "mode": "previous", "split": "test",
                    "sequence_id": sequence["sequence_id"], "building_id": sequence["building_id"],
                    "frame": 1, "artifact_path": "maps/previous-ready.npz", "artifact_sha256": sha256(path),
                    "method_metadata": metadata("rgb_diff")}
        marker = self.source / "units/previous-ready.json"
        atomic_json(marker, original)
        self.manifest["source_records"] = [{k: v for k, v in original.items() if k != "method_metadata"}]
        self.manifest["source_summary"]["source_json_sha256"] = {"units/previous-ready.json": sha256(marker)}
        atomic_json(self.prepared / "manifest.json", self.manifest)
        factory = Factory()
        with patch.object(night_run, "_snapshot_gpu"), contextlib.redirect_stdout(io.StringIO()):
            result = night_run.run(self.prepared, self.root / "full", methods=("rgb_diff",),
                                  modes=("previous",), pool_factory=factory)
        self.assertEqual(result["completed"], 3)
        self.assertEqual(result["results"][0]["origin"], "reused_source")
        self.assertEqual(len(factory.calls), 2)


if __name__ == "__main__":
    unittest.main()
