"""Regressions from the lposs report: optional Torch features are not imports."""
from contextlib import redirect_stdout
import io
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

AUDIT = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/audit_environment.py"))


def old_torch():
    def load(path, map_location=None):
        raise AssertionError("Audit must not load checkpoint weights")
    return SimpleNamespace(
        __version__="1.12.1", __file__="fixture/torch.py", load=load,
        version=SimpleNamespace(cuda="11.3"),
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
        from_numpy=lambda array: SimpleNamespace(numpy=lambda: array),
    )


class EnvironmentAuditTests(unittest.TestCase):
    def test_old_torch_retains_bridge_and_cuda_without_weights_only(self):
        scope = {}
        with patch.dict("sys.modules", {"torch": old_torch()}):
            exec(AUDIT["PROBES"]["torch"], scope)
        self.assertFalse(scope["info"]["weights_only"])
        self.assertFalse(scope["info"]["default_safe_checkpoint_loader_available"])
        self.assertTrue(scope["info"]["trusted_legacy_checkpoint_loader_available"])
        self.assertTrue(scope["info"]["numpy_bridge"])
        self.assertTrue(scope["info"]["cuda_available"])
        self.assertEqual(scope["info"]["device_count"], 1)

    def test_report_exposes_legacy_option_without_claiming_inference(self):
        scope = {}
        with patch.dict("sys.modules", {"torch": old_torch()}):
            exec(AUDIT["PROBES"]["torch"], scope)

        def fake_probe(code, timeout=45):
            details = scope["info"] if code == AUDIT["PROBES"]["torch"] else {}
            return {"status": "ok", "details": details, "stdout": "", "stderr": ""}

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "audit"
            with patch.dict(AUDIT["main"].__globals__, {
                "probe": fake_probe,
                "capture": lambda *args: {"returncode": 0, "stdout": "", "stderr": ""},
            }), patch("shutil.which", return_value=None), redirect_stdout(io.StringIO()):
                self.assertEqual(AUDIT["main"](["--out", str(out)]), 0)
            report = json.loads((out / "audit.json").read_text())
            ready = report["readiness"]
            self.assertTrue(ready["loftr_checkpoint_loader_supported"])
            self.assertFalse(ready["loftr_default_safe_checkpoint_loader_supported"])
            self.assertTrue(ready["loftr_trusted_legacy_checkpoint_loader_available"])
            self.assertTrue(ready["loftr_checkpoint_requires_explicit_trust"])
            self.assertEqual(ready["loftr_checkpoint_and_inference"], "not_tested")
            self.assertIn("--trust-checkpoint", (out / "summary.txt").read_text())

    def test_numpy_bridge_failure_does_not_hide_cuda_details(self):
        torch = old_torch()
        def broken_bridge(array):
            raise RuntimeError("fixture bridge failure")
        torch.from_numpy = broken_bridge
        scope = {}
        with patch.dict("sys.modules", {"torch": torch}):
            exec(AUDIT["PROBES"]["torch"], scope)
        self.assertFalse(scope["info"]["numpy_bridge"])
        self.assertIn("fixture bridge failure", scope["info"]["numpy_bridge_error"])
        self.assertTrue(scope["info"]["cuda_available"])


class CleanupAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "repository"
        (self.root / "runs").mkdir(parents=True)
        (self.root / "configs").mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def run_dir(self, name, **config):
        path = self.root / "runs" / name
        path.mkdir()
        self.json(path / "run.json", {"status": "completed_exploratory", "config": config})
        return path

    def json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def plan(self, keep, **kwargs):
        return AUDIT["cleanup_plan"](self.root, [keep], active_paths=kwargs.pop("active_paths", []), **kwargs)

    def decisions(self, report):
        return {Path(row["path"]).name: row["decision"] for row in report["runs"]}

    def test_transitive_metadata_closure_config_and_relative_paths(self):
        source = self.run_dir("source")
        dataset = self.run_dir("dataset")
        result = self.run_dir("result", dataset_run="../dataset")
        unused = self.run_dir("unused")
        checkpoint = self.run_dir("checkpoint")
        self.json(dataset / "manifest.json", {"source": {"crop_run": "../source"}})
        self.json(self.root / "configs" / "methods.local.json", {"model_run": "../runs/checkpoint"})
        self.json(self.root / "configs" / "methods.example.json", {"model_run": str(unused)})
        report = self.plan(result)
        self.assertEqual(self.decisions(report), {"source": "retained", "dataset": "retained",
                         "result": "retained", "unused": "review_candidate", "checkpoint": "retained"})
        self.assertIn(str(dataset / "manifest.json"), report["metadata_sha256"])
        self.assertTrue(report["read_only"])
        self.assertTrue(unused.exists())
        self.assertEqual(report["unresolved"], [])

    def test_active_incomplete_unknown_and_corrupt_runs_are_protected(self):
        final = self.run_dir("final")
        active = self.run_dir("active")
        data = self.run_dir("data")
        running = self.run_dir("running", dataset=str(data))
        self.json(running / "run.json", {"status": "running", "dataset": str(data)})
        unknown = self.root / "runs" / "unknown"
        unknown.mkdir()
        corrupt = self.run_dir("corrupt")
        (corrupt / "run.json").write_text("{broken")
        report = self.plan(final, active_paths=[active / "best.pth"])
        self.assertTrue(all(row["decision"] == "retained" for row in report["runs"]))
        self.assertTrue(report["unresolved"])

    def test_jsonl_indexes_and_artifact_key_references_preserve_archives(self):
        latest = self.run_dir("latest")
        source = self.run_dir("source")
        archive = self.root / "runs" / "preserved.tar.gz"
        archive.write_bytes(b"not read as an archive")
        disposable = self.root / "runs" / "old.tar.gz"
        disposable.write_bytes(b"not read")
        (latest / "index.jsonl").write_text(json.dumps({"parent": "../source"}) + "\n")
        self.json(source / "artifacts.json", {"../preserved.tar.gz": "a" * 64})
        report = self.plan(latest)
        self.assertEqual(self.decisions(report)[archive.name], "retained")
        self.assertEqual(self.decisions(report)[disposable.name], "review_candidate")

    def test_symlinks_are_not_traversed_and_internal_targets_are_retained(self):
        latest = self.run_dir("latest")
        source = self.run_dir("source")
        outside = Path(self.temp.name) / "private"
        self.json(outside / "run.json", {"unrelated": "secret"})
        (latest / "linked").symlink_to(outside, target_is_directory=True)
        (self.root / "runs" / "link-source").symlink_to(source, target_is_directory=True)
        self.json(latest / "links.json", {"checkpoint": "../link-source/model.pth"})
        original = Path.read_bytes
        def guarded(path):
            if path.is_relative_to(outside):
                raise AssertionError("External symlink contents must never be read")
            return original(path)
        with patch.object(Path, "read_bytes", guarded):
            report = self.plan(latest)
        self.assertEqual(self.decisions(report)["source"], "retained")
        self.assertEqual(self.decisions(report)["link-source"], "retained")
        self.assertNotIn(str(outside / "run.json"), report["metadata_sha256"])

    def test_external_repositories_are_inventory_only_and_known_dependencies_marked(self):
        external = Path(self.temp.name) / "DINO"
        external.mkdir()
        checkpoint = external / "weights.pth"
        checkpoint.write_bytes(b"never load weights")
        other = Path(self.temp.name) / "LPOSS"
        other.mkdir()
        latest = self.run_dir("latest", checkpoint=str(checkpoint))
        report = self.plan(latest, scan_roots=[Path(self.temp.name)])
        external_rows = {Path(row["path"]).name: row for row in report["external_inventory"]}
        self.assertEqual(external_rows["DINO"]["decision"], "referenced_keep")
        self.assertEqual(external_rows["LPOSS"]["decision"], "unverified_not_for_deletion")

    def test_missing_artifact_still_retains_its_existing_run(self):
        parent = self.run_dir("parent")
        latest = self.run_dir("latest", checkpoint=str(parent / "missing.pth"))
        report = self.plan(latest)
        self.assertEqual(self.decisions(report)["parent"], "retained")
        self.assertTrue(any(row.get("resolved") == str(parent / "missing.pth") for row in report["unresolved"]))

    def test_requires_existing_keep_inside_runs_and_reports_missing_reference(self):
        latest = self.run_dir("latest", dataset="../missing/run.json")
        report = self.plan(latest)
        self.assertTrue(any(row.get("value") == "../missing/run.json" for row in report["unresolved"]))
        with self.assertRaisesRegex(ValueError, "outside"):
            self.plan(Path(self.temp.name))
        with self.assertRaisesRegex(ValueError, "at least one"):
            AUDIT["cleanup_plan"](self.root, [], active_paths=[])

    def test_known_untracked_legacy_copies_and_bytecode_are_advisory(self):
        latest = self.run_dir("latest")
        old = self.root / "scripts" / "analyze_method_complementarity.py"
        old.parent.mkdir()
        old.write_text("# old copy")
        cache = self.root / "facade_change" / "__pycache__"
        cache.mkdir(parents=True)
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=1)):
            report = self.plan(latest)
        self.assertEqual({row["decision"] for row in report["local_candidates"]},
                         {"review_candidate", "regenerable_cache"})
        self.assertTrue(old.is_file())
        self.assertTrue(cache.is_dir())
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=0)):
            tracked_report = self.plan(latest)
        self.assertNotIn(str(old), [row["path"] for row in tracked_report["local_candidates"]])

    def test_cli_cleanup_does_not_probe_environment_or_load_models(self):
        latest = self.run_dir("latest")
        out = Path(self.temp.name) / "report"
        with patch.dict(AUDIT["main"].__globals__, {
                "ROOT": self.root,
                "probe": lambda *a, **kw: self.fail("No environment probes during cleanup audit"),
                "_active_run_paths": lambda runs: ([], True),
            }), redirect_stdout(io.StringIO()):
            self.assertEqual(AUDIT["main"](["--cleanup-plan", "--keep-run", str(latest), "--out", str(out)]), 0)
        self.assertTrue((out / "cleanup.json").is_file())
        self.assertTrue((out / "cleanup.csv").is_file())
        self.assertIn("nothing was moved or deleted", (out / "summary.txt").read_text())


if __name__ == "__main__":
    unittest.main()
