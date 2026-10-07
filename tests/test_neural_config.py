"""Completed H0 checkpoints can extend the frozen codec comparison safely."""
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from facade_change.io import read_json, sha256, write_json


class NeuralConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "runs/data"
        self.dataset.mkdir(parents=True)
        self.rscd = ("rscd_cmu", "rscd_diff_cmu", "rscd_pscd")
        split = {"mode": "reviewed", "development_only": False}
        write_json(self.dataset / "config.json", {"frozen": True})
        write_json(self.dataset / "split.json", split)
        write_json(self.dataset / "index.json", {"split": split, "cases": []})
        write_json(self.dataset / "summary.json", {"index_path": "index.json", "case_count": 0})
        write_json(self.dataset / "run.json", {"kind": "hypothesis_dataset", "status": "completed_needs_review",
            "config": {"input_sha256": {"config": sha256(self.dataset / "config.json")}},
            "artifact_sha256": {name: sha256(self.dataset / name)
                                for name in ("config.json", "split.json", "index.json", "summary.json")}})
        self.fingerprint = {key: sha256(self.dataset / file) for key, file in (
            ("parent_run", "run.json"), ("parent_summary", "summary.json"),
            ("parent_index", "index.json"), ("split", "split.json"), ("config", "config.json"))}
        (self.root / "configs").mkdir()
        write_json(self.root / "configs/benchmark.local.json", {
            "dataset_run": str(self.dataset), "out": str(self.root / "runs/old"),
            "device": "cpu", "trust_checkpoint": True,
            "method_options": {name: {"device": "cpu", "source_root": "/author/rscd"}
                               for name in self.rscd}})
        self.reuse = SimpleNamespace(summary={"selected_base_count": 10, "selected_case_count": 60},
            methods=("msdzip_mod256", "geoscd") + self.rscd, input_sha256=self.fingerprint.copy())
        spec = importlib.util.spec_from_file_location("configure_neural_codecs_test",
            Path(__file__).resolve().parents[1] / "scripts/configure_codecs.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.reuse_patch = patch.object(self.module, "ReuseResults", return_value=self.reuse)
        self.reuse_patch.start()
        self.addCleanup(self.reuse_patch.stop)

    def training(self, representations=("abs", "mod256")):
        run = self.root / "runs/training"
        run.mkdir()
        config = {"dataset_run": str(self.dataset), "input_sha256": self.fingerprint.copy(),
                  "representations": list(representations), "tile_size": 32,
                  "author_config": "imagenet32_config", "seed": 73}
        for representation in representations:
            (run / representation).mkdir()
            for component in ("sig", "ins"):
                (run / representation / (component + ".pth")).write_bytes(b"test checkpoint")
        write_json(run / "summary.json", {"status": "completed_exploratory",
            "representations": {representation: {} for representation in representations}})
        record = {"schema_version": 1, "kind": "arib_h0_training", "status": "completed_exploratory",
                  "config": config, "artifact_sha256": {
                      path.relative_to(run).as_posix(): sha256(path)
                      for path in run.rglob("*") if path.is_file()}}
        self.write_training_record(run, record)
        return run

    @staticmethod
    def write_training_record(run, record):
        record["config_sha256"] = hashlib.sha256(json.dumps(record["config"], sort_keys=True,
            ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        write_json(run / "run.json", record)

    def configure(self, **kwargs):
        return self.module.configure(reuse_run=self.root / "runs/old", repo_root=self.root, **kwargs)

    def test_both_completed_representations_use_active_python_and_actual_bits_by_default(self):
        run = self.training()
        result = self.configure(training_run=run)
        config = read_json(result["config"])
        self.assertEqual(config["methods"], list(self.rscd) + ["jpegls_mod256", "h264_rgb",
                                                            "arib_bps_abs", "arib_bps_mod256"])
        self.assertEqual(result["recompute_methods"], list(self.rscd))
        self.assertEqual(result["cached_methods"], ["msdzip_mod256", "geoscd"])
        for representation in ("abs", "mod256"):
            options = config["method_options"]["arib_bps_" + representation]
            self.assertEqual(options["worker_python"], str(Path(sys.executable).resolve()))
            self.assertEqual(options["source_root"], str(self.root / "third_party/arib_bps"))
            self.assertEqual(options["training_run"], str(run))
            self.assertEqual(options["device"], "cuda:0")
            self.assertEqual(options["cost_mode"], "bitstream")
            self.assertEqual(options["seed"], 73)
        self.assertEqual(config["device"], "cpu")
        self.assertTrue(config["trust_checkpoint"])
        self.assertEqual(config["selection_path"], str(self.root / "runs/old/selection.json"))

    def test_only_fitted_representation_is_added_and_cached_additions_are_recomputed(self):
        run = self.training(("mod256",))
        self.reuse.methods += ("h264_rgb", "arib_bps_mod256", "arib_bps_abs")
        result = self.configure(training_run=run, source_root=self.root / "author",
                                neural_device="cpu", cost_mode="theoretical")
        config = read_json(result["config"])
        self.assertNotIn("arib_bps_abs", config["methods"])
        self.assertEqual(result["recompute_methods"], list(self.rscd) + ["h264_rgb", "arib_bps_mod256"])
        self.assertIn("arib_bps_abs", result["cached_methods"])
        self.assertEqual(result["new_methods"], ["jpegls_mod256"])
        options = config["method_options"]["arib_bps_mod256"]
        self.assertEqual(options["source_root"], str(self.root / "author"))
        self.assertEqual(options["device"], "cpu")
        self.assertEqual(options["cost_mode"], "theoretical")

    def test_completed_run_schema_and_representations_are_required(self):
        run = self.training()
        original = read_json(run / "run.json")
        for updates in ({"status": "running"}, {"schema_version": 2}, {"kind": "other"}):
            with self.subTest(updates=updates):
                write_json(run / "run.json", {**original, **updates})
                with self.assertRaisesRegex(ValueError, "completed H0 training"):
                    self.configure(training_run=run)
        for representations in ([], ["abs", "abs"], ["unknown"]):
            with self.subTest(representations=representations):
                record = json.loads(json.dumps(original))
                record["config"]["representations"] = representations
                self.write_training_record(run, record)
                with self.assertRaisesRegex(ValueError, "representations"):
                    self.configure(training_run=run)
        self.assertFalse((self.root / "configs/codecs.local.json").exists())

    def test_tampered_training_configuration_is_rejected(self):
        run = self.training()
        record = read_json(run / "run.json")
        record["config"]["seed"] = 999
        write_json(run / "run.json", record)
        with self.assertRaisesRegex(ValueError, "configuration hash"):
            self.configure(training_run=run)

    def test_training_and_reused_comparison_must_share_dataset_fingerprint(self):
        run = self.training()
        self.reuse.input_sha256["split"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "different parent dataset fingerprint"):
            self.configure(training_run=run)

    def test_dataset_metadata_is_verified_against_training_without_rgb_reads(self):
        run = self.training()
        record = read_json(self.dataset / "run.json")
        record["new_metadata"] = True
        write_json(self.dataset / "run.json", record)
        with self.assertRaisesRegex(ValueError, "different parent dataset fingerprint"):
            self.configure(training_run=run)

    def test_missing_corrupt_or_escaping_checkpoints_are_rejected(self):
        run = self.training()
        path = run / "abs/sig.pth"
        original = path.read_bytes()
        for contents in (None, b"changed"):
            with self.subTest(contents=contents):
                if contents is None:
                    path.unlink()
                else:
                    path.write_bytes(contents)
                with self.assertRaisesRegex(ValueError, "artifact missing or hash changed"):
                    self.configure(training_run=run)
        path.unlink()
        outside = self.root / "outside.pth"
        outside.write_bytes(original)
        path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes its run"):
            self.configure(training_run=run)

    def test_summary_must_confirm_the_completed_representations(self):
        run = self.training()
        write_json(run / "summary.json", {"status": "running", "representations": {"abs": {}}})
        record = read_json(run / "run.json")
        record["artifact_sha256"]["summary.json"] = sha256(run / "summary.json")
        self.write_training_record(run, record)
        with self.assertRaisesRegex(ValueError, "summary disagrees"):
            self.configure(training_run=run)

    def test_output_config_and_input_runs_remain_immutable(self):
        run = self.training()
        existing = self.root / "runs/existing"
        existing.mkdir()
        with self.assertRaisesRegex(FileExistsError, "Output already exists"):
            self.configure(training_run=run, out=existing)
        for kwargs in ({"out": run / "nested"}, {"destination": run / "launch.json"},
                       {"destination": self.dataset / "launch.json"},
                       {"out": self.root / "runs/new", "destination": self.root / "runs/new/config.json"}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    self.configure(training_run=run, **kwargs)
        result = self.configure(training_run=run)
        before = Path(result["config"]).read_bytes()
        with self.assertRaises(FileExistsError):
            self.configure(training_run=run)
        self.assertEqual(Path(result["config"]).read_bytes(), before)
        self.assertFalse((self.root / "runs/2026-10-07-codecs-001").exists())


if __name__ == "__main__":
    unittest.main()
