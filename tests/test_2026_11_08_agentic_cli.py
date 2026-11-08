"""Integration contracts; the explicit CPU test model does not validate Qwen."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from facade_change.agentic.acceptance import fixture
from facade_change.agentic.cli import _index_or_import, main
from facade_change.agentic.config import defaults, load_config, write_new_config
from facade_change.agentic.runner import run_cases
from facade_change.agentic.store import RegionStore
from facade_change.io import read_json, write_json


class ContractPool:
    """Only injected by these tests, never a command-line model backend."""
    def __init__(self, models, *args, **kwargs):
        self.models = {"qwen": {"model": "explicit-test-double"}}

    def call(self, role, request):
        messages = request["messages"]
        evidence = list(dict.fromkeys(block["evidence_id"] for m in messages for block in m["content"]
                                     if block["type"] == "image"))
        value = {"type": "final", "decision": "change", "visible_difference": "Contract fixture only",
                 "evidence_ids": evidence, "semantic_interpretation": None, "localization": None,
                 "limitations": ["Injected model; no GPU or scientific quality claim"], "missing_information": []}
        return {"status": "ok", "text": json.dumps(value), "usage": {"model_calls": 1,
            "input_tokens": 40, "output_tokens": 30, "images": len(evidence)}}

    def close(self):
        pass


class ErrorPool(ContractPool):
    def call(self, role, request):
        return {"status": "error", "error": {"code": "missing_weights", "message": "Contract test"}, "usage": {}}


class ApplicationIntegration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = defaults()
        self.config.update(store_root=str(self.root / "store"), cache_root=str(self.root / "cache"),
                           knowledge_root=str(self.root / "knowledge"), methods=["ssim", "lzma_abs"])
        self.pairs = fixture(self.root / "fixture")

    def test_raw_pair_history_three_modes_and_saved_report(self):
        out = self.root / "run"
        result = run_cases(self.config, self.pairs, out, pool_factory=ContractPool)
        self.assertEqual(result["errors"], [])
        self.assertTrue((out / "report/report.html").is_file())
        with RegionStore(self.config["store_root"]) as store:
            self.assertEqual(len(store.list_regions()), 1)
            self.assertEqual(len(store.list_observations()), 3)
            claims = [r for r in store.list_evidence() if r["kind"] == "agent_hypothesis"]
            self.assertEqual(len(claims), 6)
        for case in (out / "cases").iterdir():
            for mode in ("pair", "fixed_context", "agent"):
                directory = case / "modes" / mode
                final = read_json(directory / "result.json")
                self.assertEqual(final["status"], "completed")
                self.assertTrue(final["evidence_ids"])
                self.assertNotIn('"labels"', (directory / "messages.json").read_text())
            pair_messages = (case / "modes/pair/messages.json").read_text()
            self.assertNotIn('"operation": "get_maps"', pair_messages)
        self.assertEqual(read_json(out / "run.json")["status"], "completed")

    def test_returned_model_error_fails_run_without_claiming_no_change(self):
        config = {**self.config, "modes": ["pair"]}
        out = self.root / "failed"
        result = run_cases(config, self.pairs[:1], out, pool_factory=ErrorPool)
        self.assertTrue(result["errors"])
        self.assertEqual(read_json(out / "run.json")["status"], "completed_with_errors")
        prediction = read_json(out / "cases/synthetic-2020/modes/pair/result.json")
        self.assertEqual(prediction["decision"], "insufficient_data")

    def test_new_pair_cli_requires_real_explicit_paths(self):
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            code = main(["run", "--out", str(self.root / "missing")])
        self.assertEqual(code, 2)
        self.assertIn("--reference", errors.getvalue())
        self.assertFalse((self.root / "missing").exists())

    def test_fixed_cohort_does_not_depend_on_preparation_success(self):
        config = {**self.config, "modes": []}
        good = self.root / "good"
        failed = self.root / "preparation-error"
        run_cases(config, self.pairs[:1], good)
        with patch("facade_change.agentic.pipeline.prepare_case", side_effect=RuntimeError("alignment unavailable")):
            result = run_cases(config, self.pairs[:1], failed)
        self.assertTrue(result["errors"])
        self.assertEqual(read_json(good / "input_manifest.json"), read_json(failed / "input_manifest.json"))
        from facade_change.agentic.evaluation import compare_runs
        comparison = compare_runs([good, failed], self.root / "comparison", good / "input_manifest.json")
        self.assertTrue(any(row["status"] == "error" for row in comparison["cases"]))

    def test_gt_case_is_rejected_before_creating_run(self):
        supplied = {**self.pairs[0], "labels": "private.png"}
        with self.assertRaises(ValueError):
            run_cases(self.config, [supplied], self.root / "forbidden", pool_factory=ContractPool)
        self.assertFalse((self.root / "forbidden").exists())

    def test_jina_index_uses_real_adapter_contract_and_space(self):
        config = {**self.config, "modes": []}
        run_cases(config, self.pairs[:1], self.root / "prepare")
        requests = []
        class EmbeddingPool:
            def __init__(self, *args, **kwargs):
                pass
            def call(self, role, payload):
                requests.append(payload)
                if "support_paths" in payload:
                    raise AssertionError("Jina does not take DINO's support API")
                return {"status": "ok", "vectors": [[1., 2.]], "space": {
                    "model": "test-only", "revision": "fake", "representation": "joint_image_text",
                    "preprocessing": {"test": True}}, "usage": {}}
            def close(self):
                pass
        args = SimpleNamespace(task="index", out=str(self.root / "index"), observation_id=None,
                               cutoff="2025", encoder="jina")
        with patch("facade_change.agentic.models.ModelPool", EmbeddingPool):
            result = _index_or_import(config, args)
        self.assertEqual(result["observations"], 2)
        self.assertIsInstance(result["corpus_version"], str)
        self.assertEqual(len(requests), 2)

    def test_configuration_preserves_venv_interpreter_symlink_and_offline_policy(self):
        import sys
        config = self.config
        config["models"]["qwen"]["python"] = sys.executable
        path = write_new_config(self.root / "config.json", config)
        self.assertEqual(load_config(path)["models"]["qwen"]["python"], sys.executable)
        config["models"]["qwen"]["local_files_only"] = False
        write_json(path, config)
        with self.assertRaises(ValueError):
            load_config(path)


if __name__ == "__main__":
    unittest.main()
