"""Configuration uses recorded assets and explicit ontology, without model imports."""
import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.real_eval.semantic_config import (
    CONTEXT_VERSION, LEGACY_CHECKPOINTS, _semantic_options, _vocabularies,
    add_configure_parser, configure,
)
from scripts.real_eval.robustness_common import atomic_json, sha256


class SemanticConfigTests(unittest.TestCase):
    def prepared(self, root):
        prepared = root / "prepared"
        prepared.mkdir()
        source = root / "source"
        source.mkdir()
        training = root / "existing-training-run"
        training.mkdir()
        atomic_json(root / "identities.json", {"images": []})
        atomic_json(root / "dataset.json", {"manifest": "identities.json"})
        manifest = {
            "schema_version": 1, "kind": "real_temporal_prepared",
            "source_run": str(source), "dataset": str(root / "dataset.json"),
            "source_config": {"method_options": {"semantic_dino": {
                "training_run": str(training), "worker_python": sys.executable}}},
            "categories": [{"id": 7, "name": "CRACK"}, {"id": 19, "name": "ADVERTISEMENTS"}],
            "sequences": [{"sequence_id": "one", "building_id": "A", "split": "val", "frames": [
                {"frame": 0, "year": 2000, "artifact_path": "first.npz", "artifact_sha256": "a" * 64},
                {"frame": 1, "year": 2002, "artifact_path": "second.npz", "artifact_sha256": "b" * 64},
            ]}],
        }
        atomic_json(prepared / "manifest.json", manifest)
        return prepared, manifest

    def arguments(self, root, prepared, *extra):
        parser = argparse.ArgumentParser()
        add_configure_parser(parser.add_subparsers(dest="command", required=True))
        return parser.parse_args([
            "configure", "--prepared", str(prepared), "--out", str(root / "config.json"),
            "--lposs-root", str(root / "missing-lposs"), "--lposs-python", sys.executable,
            "--sam3-python", sys.executable, *extra,
        ])

    def test_config_preserves_declared_assets_and_current_ontology(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prepared, manifest = self.prepared(root)
            args = self.arguments(root, prepared, "--reuse-run", str(root / "old-run"))
            with contextlib.redirect_stdout(io.StringIO()):
                result = configure(args)
            saved = json.loads((root / "config.json").read_text())
            self.assertEqual(result, saved)
            self.assertEqual(list(result["segmenters"]), [
                "dino_semantic", "maskclip_head", "maskclip_depth1", "maskclip_stock", "lposs_stock", "sam3"])
            dino = result["segmenters"]["dino_semantic"]
            self.assertEqual(dino["confidence"], .5)
            self.assertEqual(dino["training_run"], manifest["source_config"]["method_options"]["semantic_dino"]["training_run"])
            self.assertEqual(dino["current_manifest"], str(root / "identities.json"))
            self.assertEqual(dino["expected_categories"], manifest["categories"])
            self.assertEqual([entry["name"] for entry in result["vocabularies"]["domain"]], ["CRACK", "ADVERTISEMENTS"])
            self.assertEqual(result["vocabulary_provenance"]["name"], CONTEXT_VERSION)
            self.assertFalse(result["vocabulary_provenance"]["historical_context60_reproduced"])
            for name, relative in LEGACY_CHECKPOINTS.items():
                self.assertEqual(result["segmenters"][name]["checkpoint"], str(root / "missing-lposs" / relative))
                self.assertEqual(result["segmenters"][name]["variants"], ["fixed"])
                self.assertEqual(result["configure_diagnostics"][name]["status"], "missing_declared_assets")
            self.assertEqual(result["segmenters"]["sam3"]["variants"], ["domain", "expanded_flat", "expanded_layered"])
            self.assertTrue(Path(result["segmenters"]["sam3"]["checkpoint"]).is_absolute())
            self.assertEqual(result["source"]["prepared_manifest_sha256"], sha256(prepared / "manifest.json"))
            self.assertEqual(result["cache_root"], str(root / "semantic-cache-v1"))
            self.assertEqual(result["reuse_runs"], [str(root / "old-run")])
            with self.assertRaises(FileExistsError):
                configure(args)

    def test_source_records_recover_missing_options_and_conflicts_fail(self):
        manifest = {"source_run": "/nonexistent/source", "source_config": {"method_options": {
            "semantic_dino": {"training_run": None}}}, "source_records": {
                "old": {"method": "semantic_dino", "status": "failed", "method_metadata": {
                    "configured_options": {"training_run": "/wrong"}}},
                "complete": {"method": "semantic_dino", "status": "completed", "method_metadata": {
                    "configured_options": {"training_run": "/recorded/run"},
                    "worker_environment": {"executable": "/recorded/python"}}},
            }}
        self.assertEqual(_semantic_options(manifest), {"training_run": "/recorded/run", "worker_python": "/recorded/python"})
        manifest["source_config"]["method_options"]["semantic_dino"]["training_run"] = "/different/run"
        with self.assertRaisesRegex(ValueError, "training_run"):
            _semantic_options(manifest)

    def test_vocabulary_extension_keeps_shared_prompts_and_hashes_original(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "exact.json"
            atomic_json(path, {"name": "user-vocabulary", "concepts": [
                {"name": "crack", "prompts": ["a thin masonry crack"], "group": "damage"},
                {"name": "WINDOW", "prompts": ["window", "arched window"], "group": "structure"},
            ]})
            domain, expanded, origin = _vocabularies([{"id": 7, "name": "CRACK"}], path)
            self.assertEqual(domain, expanded[:1])
            self.assertEqual(domain, [{"name": "CRACK", "prompts": ["a thin masonry crack"], "group": "domain"}])
            self.assertEqual(expanded[1]["prompts"], ["window", "arched window"])
            self.assertEqual(origin["sha256"], sha256(path))
            self.assertEqual(origin["origin"], "user_file")
            self.assertIsNone(origin["historical_context60_reproduced"])
            atomic_json(path, {"classes": ["WINDOW", "window"]})
            with self.assertRaisesRegex(ValueError, "Повтор"):
                _vocabularies([{"name": "CRACK"}], path)
            atomic_json(path, ["CRACK"])
            with self.assertRaisesRegex(ValueError, "дополнительных"):
                _vocabularies([{"name": "CRACK"}], path)

    def test_current_names_cannot_collapse_to_equal_channels(self):
        with self.assertRaisesRegex(ValueError, "уникальными"):
            _vocabularies([{"name": "CRACK"}, {"name": "crack"}], None)


if __name__ == "__main__":
    unittest.main()
