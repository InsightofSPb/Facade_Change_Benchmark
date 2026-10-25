"""Frozen semantic head and explicit auxiliary worker contracts; no downloads."""
import importlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from facade_change.io import sha256, write_json
from facade_change.method_worker import serve
from facade_change.methods.remote import RemotePool, RemoteScorer, validate_auxiliary_arrays


semantic = importlib.import_module("facade_change.methods.2026-10-14_semantic_dino")
HAS_TORCH = importlib.util.find_spec("torch") is not None


class SemanticMetadataTests(unittest.TestCase):
    def test_exact_id_channel_mapping_is_required(self):
        categories = [{"id": 2, "name": "A"}, {"id": 7, "name": "B"}]
        mapping = [{"category_id": 2, "channel": 0}, {"category_id": 7, "channel": 1}]
        config = {"categories": categories, "category_to_channel": mapping}
        self.assertEqual(semantic._category_mapping(config, categories)[1], mapping)
        with self.assertRaisesRegex(ValueError, "mapping"):
            semantic._category_mapping({**config, "category_to_channel": list(reversed(mapping))})
        with self.assertRaisesRegex(ValueError, "ontology"):
            semantic._category_mapping(config, [{"id": 2, "name": "B"}, {"id": 7, "name": "A"}])
        with self.assertRaisesRegex(ValueError, "unique"):
            semantic._category_mapping({**config, "categories": [categories[0], categories[0]]})

    def test_training_overlap_and_unknown_origins_are_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            training, current = root / "training.json", root / "current.json"
            write_json(training, {"images": [
                {"image_id": 1, "split": "train", "building_id": "B", "view_id": "front",
                 "sha256": "a", "metadata_status": "reviewed"},
                {"image_id": 2, "split": "train", "building_id": None, "sha256": "unknown"}]})
            write_json(current, {"images": [
                {"image_id": 99, "split": "test", "building_id": "b", "sha256": "new"},
                {"image_id": 88, "split": "val", "building_id": "C", "sha256": "a"}]})
            config = {"training_image_ids": [1, 2], "dataset_provenance": {
                "paths": {"manifest": str(training)}, "sha256": {"manifest": sha256(training)}}}
            audit = semantic._training_origin(config, current)
            self.assertTrue(audit["known_checkpoint_train_holdout_overlap"])
            self.assertEqual(audit["unverified_training_image_ids"], [2])
            self.assertFalse(audit["checkpoint_train_holdout_identity_disjointness_verified"])
            self.assertEqual(audit["train_overlap_by_current_split"]["test"][0]["matched_on"], ["building_id"])
            self.assertEqual(audit["train_overlap_by_current_split"]["val"][0]["matched_on"], ["sha256"])
            training.unlink()
            missing = semantic._training_origin(config, current)
            self.assertFalse(missing["training_manifest_verified"])
            self.assertFalse(missing["checkpoint_train_holdout_identity_disjointness_verified"])

    def test_dataset_json_is_not_accepted_as_current_identity_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dataset.json"
            write_json(path, {"kind": "semantic_crop_index", "manifest": "manifest.json"})
            with self.assertRaisesRegex(ValueError, "nonempty images list"):
                semantic._training_origin({}, path)


class AuxiliaryProtocolTests(unittest.TestCase):
    def setUp(self):
        self.support = np.ones((3, 4), bool)
        self.support[0, 0] = False
        self.metadata = {"category_to_channel": [{"category_id": 2, "channel": 0},
                                                 {"category_id": 7, "channel": 1}]}
        self.delta = np.full((2, 3, 4), -.2, np.float32)
        self.delta[:, ~self.support] = np.nan

    def test_numeric_names_dtype_shape_mapping_and_support_are_checked(self):
        validate_auxiliary_arrays({"semantic_delta": self.delta}, self.support, self.metadata)
        invalid = [({"not_allowed": self.delta}, self.metadata),
                   ({"semantic_delta": self.delta.astype(np.float64)}, self.metadata),
                   ({"semantic_delta": self.delta[:1]}, self.metadata),
                   ({"semantic_delta": np.nan_to_num(self.delta)}, self.metadata),
                   ({"semantic_delta": self.delta}, {})]
        for arrays, metadata in invalid:
            with self.subTest(arrays=list(arrays)), self.assertRaises(ValueError):
                validate_auxiliary_arrays(arrays, self.support, metadata)

    def test_worker_exports_explicit_float_arrays_without_object_payloads(self):
        outer = self

        class Scorer:
            metadata = outer.metadata

            def __call__(self, reference, source, support):
                self.auxiliary_arrays = {"semantic_delta": outer.delta.copy()}
                return np.abs(outer.delta).max(axis=0)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rgb = np.zeros((3, 4, 3), np.uint8)
            np.savez(root / "input.npz", reference=rgb, source=rgb, support=self.support)
            requests = [{"id": 1, "operation": "init", "method": "fake"},
                        {"id": 2, "operation": "score", "input_path": str(root / "input.npz"),
                         "output_path": str(root / "output.npz")},
                        {"id": 3, "operation": "close"}]
            output = io.StringIO()
            with patch("facade_change.methods.registry.make_method", return_value=Scorer()):
                serve(io.StringIO("\n".join(map(json.dumps, requests))), output)
            self.assertTrue(all(json.loads(row)["ok"] for row in output.getvalue().splitlines()))
            with np.load(root / "output.npz", allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), {"scores", "semantic_delta"})
                np.testing.assert_array_equal(archive["semantic_delta"], self.delta)

    def test_remote_process_roundtrip_keeps_class_channels_and_signed_values(self):
        script = '''
import numpy as np
import facade_change.methods.registry as registry
from facade_change.method_worker import main

class Scorer:
    metadata = {"category_to_channel": [{"category_id": 2, "channel": 0}, {"category_id": 7, "channel": 1}]}
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, reference, source, support):
        score = np.full(support.shape, .2, np.float32)
        score[~support] = np.nan
        self.auxiliary_arrays = {"semantic_delta": np.stack([score, -score])}
        return score
registry.make_method = Scorer
main()
'''
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "worker.py"
            path.write_text(script)
            popen = subprocess.Popen

            def launch(command, **kwargs):
                return popen([sys.executable, "-s", "-B", str(path)], **kwargs)

            rgb = np.zeros((3, 4, 3), np.uint8)
            with patch("facade_change.methods.remote.subprocess.Popen", side_effect=launch), \
                    RemotePool(log_dir=Path(folder) / "logs", timeout=20) as pool:
                scorer = RemoteScorer("semantic_dino", pool=pool)
                score = scorer(rgb, rgb, self.support)
                delta = scorer.auxiliary_arrays["semantic_delta"]
                self.assertEqual(delta.shape, (2, 3, 4))
                np.testing.assert_array_equal(delta[0], score)
                np.testing.assert_array_equal(delta[1], -score)


@unittest.skipUnless(HAS_TORCH, "CPU torch is required for actual semantic head fixture")
class FrozenSemanticDINOTests(unittest.TestCase):
    def setUp(self):
        import torch
        from facade_change.methods.dinov2 import _source_provenance
        from facade_change.segmentation.model import FrozenDinoSegmenter

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "hubconf.py").write_text("# fixture source fingerprint\n")
        self.weights = self.root / "backbone.pth"
        self.weights.write_bytes(b"fixture backbone fingerprint")

        class Backbone(torch.nn.Module):
            patch_size = 14
            embed_dim = 1

            def get_intermediate_layers(self, rgb, n, reshape):
                feature = torch.nn.functional.avg_pool2d(rgb[:, :1], 14)
                return tuple(feature for _ in range(n))

        self.model = FrozenDinoSegmenter(Backbone(), 2)
        with torch.no_grad():
            self.model.head[1].weight[0].fill_(.3)
            self.model.head[1].weight[1].fill_(-.2)
            self.model.head[1].bias.copy_(torch.tensor([1., -.4]))
        self.provenance = {"sources": _source_provenance(self.source),
                           "checkpoint": {"sha256": sha256(self.weights)}, "num_classes": 2}
        self.dataset = self.root / "dataset.json"
        self.manifest = self.root / "manifest.json"
        self.categories = [{"id": 2, "name": "DAMAGE"}, {"id": 7, "name": "REPAIR"}]
        write_json(self.dataset, {"crop_size": 28, "stride": 14})
        write_json(self.manifest, {"categories": self.categories, "images": [
            {"image_id": 1, "split": "train", "building_id": "B", "metadata_status": "reviewed", "sha256": "a"}]})
        self.config = {"source_root": str(self.source), "checkpoint": str(self.weights),
                       "categories": self.categories, "training_image_ids": [1],
                       "category_to_channel": [{"category_id": 2, "channel": 0}, {"category_id": 7, "channel": 1}],
                       "dataset_provenance": {"paths": {"dataset": str(self.dataset), "manifest": str(self.manifest)},
                                              "sha256": {"dataset": sha256(self.dataset), "manifest": sha256(self.manifest)}}}
        self.saved = {"kind": "frozen_dinov2_semantic_head", "schema_version": 1, "epoch": 2,
                      "config": self.config, "backbone_provenance": self.provenance,
                      "head_state_dict": self.model.head.state_dict()}
        self.write_run()

    def write_run(self):
        import torch

        write_json(self.root / "config.json", self.config)
        write_json(self.root / "summary.json", {"best_epoch": 2})
        torch.save(self.saved, self.root / "best.pth")
        write_json(self.root / "run.json", {"kind": "dino_semantic_training", "status": "completed_smoke",
                   "config": self.config, "model": self.provenance,
                   "artifact_sha256": {name: sha256(self.root / name) for name in ("config.json", "summary.json", "best.pth")}})

    def scorer(self, **options):
        with patch("facade_change.segmentation.model.load_model", return_value=(self.model, self.provenance)):
            return semantic.SemanticDINOScorer(training_run=self.root, **options)

    def test_real_torch_head_sigmoid_signs_native_grid_and_independent_uncertainty(self):
        scorer = self.scorer(export_probabilities=True, current_manifest=self.manifest, batch_size=2)
        self.assertEqual((scorer.crop_size, scorer.crop_stride), (28, 14))
        before = np.full((45, 53, 3), 60, np.uint8)
        after = np.full_like(before, 200)
        support = np.ones((45, 53), bool)
        support[:2] = False
        score = scorer(before, after, support)
        arrays = scorer.auxiliary_arrays
        delta = arrays["semantic_delta"]
        self.assertTrue(np.all(delta[0, support] > 0))
        self.assertTrue(np.all(delta[1, support] < 0))
        np.testing.assert_allclose(score, np.abs(delta).max(axis=0), equal_nan=True)
        np.testing.assert_allclose(delta, arrays["probabilities_source"] - arrays["probabilities_reference"], equal_nan=True)
        self.assertFalse(np.allclose(arrays["probabilities_source"][:, support].sum(axis=0), 1))
        self.assertFalse(scorer.metadata["uncertainty_gating"])
        self.assertTrue(all(not parameter.requires_grad for parameter in scorer.model.parameters()))
        validate_auxiliary_arrays(arrays, support, scorer.metadata)
        identity = scorer(before, before, support)
        np.testing.assert_array_equal(identity[support], 0.)
        self.assertTrue(np.all(scorer.auxiliary_arrays["uncertainty_source"][support] > 0))
        scorer.close()
        scorer.close()

    def test_probabilities_are_optional_and_small_rgb_uses_native_grid(self):
        scorer = self.scorer()
        rgb = np.zeros((9, 11, 3), np.uint8)
        support = np.ones((9, 11), bool)
        self.assertEqual(scorer(rgb, rgb, support).shape, (9, 11))
        self.assertNotIn("probabilities_reference", scorer.auxiliary_arrays)
        scorer.close()

    def test_ontology_and_source_fingerprint_fail_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, "ontology"):
            self.scorer(expected_categories=[{"id": 2, "name": "OTHER"}])
        (self.source / "hubconf.py").write_text("# changed source\n")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.scorer()

    def test_saved_head_channel_mismatch_is_rejected(self):
        self.provenance["num_classes"] = 3
        self.write_run()
        with self.assertRaisesRegex(ValueError, "channel count"):
            self.scorer()


if __name__ == "__main__":
    unittest.main()
