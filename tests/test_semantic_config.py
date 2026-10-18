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
    CONTEXT_VERSION, LEGACY_CHECKPOINTS, STOCK_REQUIRED_FILES, _semantic_options, _vocabularies,
    add_configure_parser, configure, inspect_stock_root, repair_config, resolve_stock_root,
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

    def stock_checkout(self, root):
        for relative in (*STOCK_REQUIRED_FILES, "configs/stock_lposs.yaml"):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# local fixture, no imports or model execution\n")
        return root

    def test_stock_auto_selects_verified_sibling_and_keeps_trained_assets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prepared, _ = self.prepared(root)
            stock = self.stock_checkout(root / "LPOSS_pr58_cuda_20260820")
            args = self.arguments(root, prepared)
            with contextlib.redirect_stdout(io.StringIO()):
                result = configure(args)
            self.assertEqual(result["source"]["stock_root_resolution"]["status"], "existing_sibling_discovered")
            for name in ("maskclip_stock", "lposs_stock"):
                self.assertEqual(result["segmenters"][name]["source_root"], str(stock))
                self.assertEqual(result["configure_diagnostics"][name]["missing"], [])
            for name, relative in LEGACY_CHECKPOINTS.items():
                self.assertEqual(result["segmenters"][name]["source_root"], str(root / "missing-lposs"))
                self.assertEqual(result["segmenters"][name]["checkpoint"], str(root / "missing-lposs" / relative))

    def test_explicit_missing_stock_root_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.stock_checkout(root / "Compress_to_prevent")
            chosen, resolution = resolve_stock_root(root / "LPOSS", root / "explicit-missing")
            self.assertEqual(chosen, root / "explicit-missing")
            self.assertTrue(resolution["explicit"])
            self.assertTrue(resolution["candidates"][0]["missing"])

    def test_declared_valid_root_precedes_siblings_and_ambiguity_is_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            declared = self.stock_checkout(root / "LPOSS")
            self.stock_checkout(root / "LPOSS_pr58_cuda_20260820")
            self.stock_checkout(root / "Compress_to_prevent")
            chosen, resolution = resolve_stock_root(declared)
            self.assertEqual(chosen, declared)
            self.assertEqual(resolution["status"], "declared_root_present")
            (declared / "configs/stock_lposs.yaml").unlink()
            chosen, resolution = resolve_stock_root(declared)
            self.assertEqual(chosen, declared)
            self.assertEqual(resolution["status"], "ambiguous")
            self.assertIsNone(resolution["selected_root"])

    def test_discovery_requires_source_and_configuration_and_does_not_recurse(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidate = root / "Compress_to_prevent"
            (candidate / "configs").mkdir(parents=True)
            (candidate / "configs/stock_lposs.yaml").write_text("clip: {}")
            self.stock_checkout(root / "unrelated/nested/LPOSS_pr58_cuda_20260820")
            chosen, resolution = resolve_stock_root(root / "LPOSS")
            self.assertEqual(resolution["status"], "missing")
            self.assertEqual(chosen, root / "LPOSS")
            self.assertEqual(len(resolution["candidates"]), 3)
            missing = inspect_stock_root(candidate)["missing"]
            self.assertTrue(any(row["path"].endswith("stock_features.py") for row in missing))
            self.assertFalse(any(row["field"] == "model_config" for row in missing))

    def test_config_diagnostics_show_missing_stock_model_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prepared, _ = self.prepared(root)
            with contextlib.redirect_stdout(io.StringIO()):
                result = configure(self.arguments(root, prepared))
            missing = result["configure_diagnostics"]["maskclip_stock"]["missing"]
            self.assertIn({"field": "model_config", "path": str(root / "missing-lposs/configs/stock_lposs.yaml")}, missing)
            self.assertEqual(result["source"]["stock_root_resolution"]["status"], "missing")

    def test_repair_preserves_custom_settings_and_source_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prepared, _ = self.prepared(root)
            with contextlib.redirect_stdout(io.StringIO()):
                config = configure(self.arguments(root, prepared))
            config["segmenters"]["maskclip_stock"]["clip_checkpoint"] = "/custom/clip.bin"
            config["segmenters"]["maskclip_stock"]["confidence"] = .37
            config["roi"]["max_regions"] = 3
            config["change_methods"]["msdzip_abs"] = {"threshold": .123}
            config["vocabularies"]["domain"][0]["prompts"] = ["my exact prompt"]
            source = root / "config.json"
            atomic_json(source, config)
            before = source.read_bytes()
            stock = self.stock_checkout(root / "Compress_to_prevent")
            with contextlib.redirect_stdout(io.StringIO()):
                result = repair_config(source, root / "repaired.json")
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(result["segmenters"]["maskclip_stock"]["source_root"], str(stock))
            for name, spec in config["segmenters"].items():
                expected = dict(spec)
                if name in {"maskclip_stock", "lposs_stock"}:
                    expected["source_root"] = str(stock)
                self.assertEqual(result["segmenters"][name], expected)
            for key in ("roi", "change_methods", "vocabularies", "cache_root", "splits"):
                self.assertEqual(result[key], config[key])
            with self.assertRaises(FileExistsError):
                repair_config(source, source)
            with self.assertRaises(FileExistsError):
                repair_config(source, root / "repaired.json")

    def test_repair_preserves_explicit_and_custom_stock_roots(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prepared, _ = self.prepared(root)
            with contextlib.redirect_stdout(io.StringIO()):
                configure(self.arguments(root, prepared, "--stock-root", str(root / "explicit")))
            stock = self.stock_checkout(root / "Compress_to_prevent")
            with contextlib.redirect_stdout(io.StringIO()):
                retained = repair_config(root / "config.json", root / "kept.json")
                changed = repair_config(root / "config.json", root / "overridden.json", stock_root=stock)
            self.assertEqual(retained["segmenters"]["maskclip_stock"]["source_root"], str(root / "explicit"))
            self.assertEqual(changed["segmenters"]["maskclip_stock"]["source_root"], str(stock))
            self.assertTrue(changed["source"]["stock_root_resolution"]["explicit"])
            old = json.loads((root / "config.json").read_text())
            old["source"].pop("stock_root_resolution")
            atomic_json(root / "legacy-custom.json", old)
            with contextlib.redirect_stdout(io.StringIO()):
                preserved = repair_config(root / "legacy-custom.json", root / "legacy-kept.json")
            self.assertEqual(preserved["source"]["stock_root_repair"]["models"][0]["status"], "custom_root_preserved")

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
