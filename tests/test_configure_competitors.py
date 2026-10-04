import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("configure_competitors", ROOT / "scripts/configure_competitors.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ConfigureCompetitorsTests(unittest.TestCase):
    """Temporary local layouts; no model loads, pip installs or GPU allocation."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.repo = self.home / "Facade_Change_Benchmark"
        (self.repo / "configs").mkdir(parents=True)
        for name in ("methods.example.json", "benchmark.example.json"):
            shutil.copyfile(ROOT / "configs" / name, self.repo / "configs" / name)
        self.reuse = self.repo / "runs/2026-10-04-compression-quick-001"
        self.reuse.mkdir(parents=True)
        self._file(self.reuse / "selection.json", "{}")
        self.dataset = self.repo / "runs/2026-10-04-h0h1"
        self._file(self.dataset / "run.json", "{}")
        self.record = {"config": {"dataset_run": str(self.dataset)}}
        self.mock_reuse = patch.object(MODULE, "ReuseResults")
        self.loader = self.mock_reuse.start()
        self.loader.return_value.record = self.record
        template = MODULE._local_paths(MODULE._read(self.repo / "configs/methods.example.json"),
                                       self.repo, self.home)["methods"]
        for method, options in template.items():
            for key in ("worker_python", "checkpoint_path", "dino_checkpoint", "sam_checkpoint"):
                if key in options and not (method == "geoscd" and key == "checkpoint_path"):
                    self._file(Path(options[key]), "fixture")
                    if key == "worker_python":
                        Path(options[key]).chmod(0o755)
            for key in ("source_root", "dino_root", "py_utils_root"):
                if key in options:
                    Path(options[key]).mkdir(parents=True, exist_ok=True)
        source = self.home / "Semantic_Change_Detection_Comparison/benchmark/sources"
        for name in (
            "dinov2/hubconf.py", "dinov2/dinov2/models/vision_transformer.py",
            "rscd/src/robust_scene_change_detect/models/CD_model.py", "py_utils/src/py_utils/utils_torch.py",
            "anychange/torchange/models/segment_any_change/anychange.py",
        ):
            self._file(source / name)
        for name in ("src/pixel_match.py", "src/flow_cd.py", "src/segment_anything_model/build_sam.py"):
            self._file(self.home / "GeoSCD-facade" / name)
        self.vggt = self.home / "existing-weights/custom-vggt.pt"
        self._file(self.vggt, "existing VGGT")
        self._json(self.repo / "runs/2026-10-04-geoscd-full-smoke/run.json",
                   {"kind": "geoscd_full_batch", "config": {"checkpoint": str(self.vggt)}})

    def tearDown(self):
        self.mock_reuse.stop()
        self.temp.cleanup()

    def _file(self, path, content="source"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def _json(self, path, value):
        self._file(path, json.dumps(value))

    def _configure(self, **options):
        return MODULE.configure(self.repo, owner_home=self.home, **options)

    def test_detects_recorded_vggt_and_creates_absolute_configs_after_cache_verification(self):
        report = self._configure()
        self.loader.assert_called_once_with(self.reuse)
        self.assertEqual(report["vggt_checkpoint"], str(self.vggt))
        self.assertEqual(len(report["created"]), 2)
        methods = MODULE._read(self.repo / "configs/methods.local.json")["methods"]
        self.assertEqual(methods["geoscd"]["checkpoint_path"], str(self.vggt))
        self.assertTrue(all(Path(options["worker_python"]).is_absolute() for options in methods.values()))
        benchmark = MODULE._read(self.repo / "configs/benchmark.local.json")
        self.assertEqual(benchmark["selection_path"], str(self.reuse / "selection.json"))
        self.assertEqual(benchmark["dataset_run"], str(self.dataset))

    def test_invalid_or_incomplete_reused_run_writes_no_config(self):
        self.loader.side_effect = ValueError("Reuse requires the complete original run with saved maps")
        with self.assertRaisesRegex(ValueError, "complete original run"):
            self._configure()
        self.assertFalse(list((self.repo / "configs").glob("*.local.json")))

    def test_missing_model_source_fails_before_config_creation(self):
        source = self.home / "Semantic_Change_Detection_Comparison/benchmark/sources/anychange"
        shutil.rmtree(source)
        with self.assertRaisesRegex(FileNotFoundError, "anychange.source_root"):
            self._configure()
        self.assertFalse(list((self.repo / "configs").glob("*.local.json")))

    def test_existing_local_files_remain_byte_identical(self):
        self._configure()
        originals = {}
        for name in ("methods.local.json", "benchmark.local.json"):
            target = self.repo / "configs" / name
            # Deliberate noncanonical formatting must survive unchanged.
            target.write_text(target.read_text() + "\n\n")
            originals[name] = target.read_bytes()
        report = self._configure()
        self.assertEqual(len(report["preserved"]), 2)
        self.assertEqual(report["created"], [])
        for name in ("methods.local.json", "benchmark.local.json"):
            self.assertEqual((self.repo / "configs" / name).read_bytes(), originals[name])

    def test_existing_wrong_paths_fail_without_rewriting_user_config_or_creating_peer(self):
        self._configure()
        target = self.repo / "configs/methods.local.json"
        methods = MODULE._read(target)
        methods["methods"]["dinov2"]["worker_python"] = str(self.home / "missing-python")
        self._json(target, methods)
        original = target.read_bytes()
        (self.repo / "configs/benchmark.local.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "dinov2.worker_python"):
            self._configure()
        self.assertEqual(target.read_bytes(), original)
        self.assertFalse((self.repo / "configs/benchmark.local.json").exists())

    def test_malformed_existing_config_is_reported_and_preserved(self):
        target = self.repo / "configs/methods.local.json"
        self._file(target, "user-edited invalid JSON\n")
        original = target.read_bytes()
        with self.assertRaises(json.JSONDecodeError):
            self._configure()
        self.assertEqual(target.read_bytes(), original)
        self.assertFalse((self.repo / "configs/benchmark.local.json").exists())

    def test_dry_run_and_fallback_preserve_checkpoint_without_writing(self):
        self.vggt.unlink()
        fallback = self.home / "GeoSCD-facade/src/pretrained/model.pt"
        self._file(fallback, "old VGGT retained")
        report = self._configure(dry_run=True)
        self.assertEqual(report["vggt_checkpoint"], str(fallback))
        self.assertEqual(len(report["would_create"]), 2)
        self.assertFalse(list((self.repo / "configs").glob("*.local.json")))
        self.assertEqual(fallback.read_text(), "old VGGT retained")

    def test_unavailable_preferred_record_uses_another_existing_geoscd_record(self):
        self.vggt.unlink()
        latest = self.home / "existing-weights/latest-vggt.pt"
        self._file(latest, "latest")
        self._json(self.repo / "runs/2026-10-04-geoscd-full-other/run.json",
                   {"kind": "geoscd_full_batch", "config": {"checkpoint": str(latest)}})
        report = self._configure(dry_run=True)
        self.assertEqual(report["vggt_checkpoint"], str(latest))


if __name__ == "__main__":
    unittest.main()
