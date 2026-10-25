"""Regressions from the lposs report: optional Torch features are not imports."""
from contextlib import redirect_stdout
import io
import json
import runpy
import subprocess
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
    def test_cpu_suite_optional_skips_remain_visible_without_failing_audit(self):
        def fake_probe(code, timeout=45):
            if code == AUDIT["TEST_CODE"]:
                self.assertEqual(timeout, 900)
                return {"status": "ok", "stdout": "", "stderr": "optional codec skipped\n",
                        "details": {"successful": True, "tests_run": 20, "skipped": 3}}
            return {"status": "ok", "stdout": "", "stderr": "", "details": {}}

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "audit"
            with patch.dict(AUDIT["main"].__globals__, {
                "probe": fake_probe,
                "capture": lambda *args: {"returncode": 0, "stdout": "", "stderr": ""},
            }), patch("shutil.which", return_value=None), redirect_stdout(io.StringIO()):
                self.assertEqual(AUDIT["main"](["--run-tests", "--out", str(out)]), 0)
            report = json.loads((out / "audit.json").read_text())
            self.assertTrue(report["readiness"]["cpu_project_tests_passed"])
            self.assertEqual(report["tests"]["details"]["skipped"], 3)
            self.assertIn("optional codec skipped", (out / "tests.log").read_text())

    def test_cpu_suite_failures_empty_and_all_skipped_are_rejected(self):
        for details in ({"successful": False, "tests_run": 20, "skipped": 3},
                        {"successful": True, "tests_run": 0, "skipped": 0},
                        {"successful": True, "tests_run": 3, "skipped": 3}):
            with self.subTest(details=details), tempfile.TemporaryDirectory() as directory:
                def fake_probe(code, timeout=45):
                    if code == AUDIT["TEST_CODE"]:
                        self.assertEqual(timeout, 120)
                    return {"status": "ok", "stdout": "", "stderr": "",
                            "details": details if code == AUDIT["TEST_CODE"] else {}}
                out = Path(directory) / "audit"
                with patch.dict(AUDIT["main"].__globals__, {
                    "probe": fake_probe,
                    "capture": lambda *args: {"returncode": 0, "stdout": "", "stderr": ""},
                }), patch("shutil.which", return_value=None), redirect_stdout(io.StringIO()):
                    self.assertEqual(AUDIT["main"]([
                        "--run-tests", "--test-timeout", "120", "--out", str(out)]), 2)
                report = json.loads((out / "audit.json").read_text())
                self.assertFalse(report["readiness"]["cpu_project_tests_passed"])

    def test_timeout_preserves_partial_test_output(self):
        error = subprocess.TimeoutExpired(["python", "tests"], 1,
                                          output=b"partial stdout\n", stderr=b"last test\n")
        with patch("subprocess.run", side_effect=error):
            result = AUDIT["capture"](["python", "tests"], timeout=1)
        self.assertIsNone(result["returncode"])
        self.assertEqual(result["stdout"], "partial stdout\n")
        self.assertIn("last test\n", result["stderr"])
        self.assertIn("timed out", result["stderr"])

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

    def test_recorded_lineage_and_input_hash_references_preserve_archives(self):
        latest = self.run_dir("latest")
        source = self.run_dir("source")
        archive = self.root / "runs" / "preserved.tar.gz"
        archive.write_bytes(b"not read as an archive")
        disposable = self.root / "runs" / "old.tar.gz"
        disposable.write_bytes(b"not read")
        self.json(latest / "config.json", {"parent": "../source"})
        self.json(source / "provenance.json", {"input_sha256": {"../preserved.tar.gz": "a" * 64}})
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
        self.json(latest / "config.json", {"checkpoint": "../link-source/model.pth"})
        original = Path.open
        def guarded(path, *args, **kwargs):
            if path.resolve().is_relative_to(outside):
                raise AssertionError("External symlink contents must never be read")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
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

    def test_path_extraction_ignores_prose_aliases_and_coco_rle(self):
        latest = self.run_dir("latest")
        source = self.run_dir("source")
        self.json(latest / "manifest.json", {
            "file_name": "hash-prefix-facade.jpg", "resolved_file_name": "facade.jpg",
            "notes": "TRAIN/VAL use annotated support only", "statistic": "F1/IoU",
            "description": "/not a real path; see results.json",
            "segmentation": {"size": [512, 512], "counts": "a" * 500 + "/b" * 50},
            "counts": "x/y" * 500, "source_run": str(source),
        })
        report = self.plan(latest)
        self.assertEqual(report["unresolved_statistics"]["occurrences"], 0)
        self.assertEqual(self.decisions(report)["source"], "retained")
        self.assertTrue(report["dependency_scan_complete"])

    def test_repeated_crop_references_have_bounded_grouped_evidence(self):
        parent = self.run_dir("parent")
        (parent / "weights.pth").write_bytes(b"do not load")
        latest = self.run_dir("latest")
        external = Path(self.temp.name) / "images"
        external.mkdir()
        image = external / "facade.png"
        image.write_bytes(b"do not decode")
        count = 3000
        self.json(latest / "manifest.json", {"images": [
            {"source_run": str(parent), "image_path": str(image),
             "rgb": f"samples/missing-{i}.png", "file_name": "facade.png"}
            for i in range(count)]})
        report = self.plan(latest, scan_roots=[Path(self.temp.name)])
        self.assertLess(len(json.dumps(report).encode()), 30_000)
        self.assertEqual(self.decisions(report)["parent"], "retained")
        self.assertEqual(report["unresolved_statistics"]["occurrences"], count)
        self.assertEqual(len(report["unresolved"]), 1)
        self.assertEqual(report["unresolved"][0]["count"], count)
        self.assertEqual(len(report["unresolved"][0]["examples"]), 2)
        reference = next(row for row in report["runs"] if Path(row["path"]) == parent)
        self.assertEqual(reference["reference_statistics"]["occurrences"], count)
        self.assertEqual(len(reference["references"]), 1)
        inventory = next(row for row in report["external_inventory"] if Path(row["path"]) == external)
        self.assertEqual(inventory["decision"], "referenced_keep")
        self.assertEqual(inventory["reference_statistics"]["occurrences"], count)

    def test_previous_and_current_audit_outputs_never_become_dependencies(self):
        latest = self.run_dir("latest")
        unused = self.run_dir("unused")
        previous = self.root / "runs" / "previous-cleanup"
        self.json(previous / "cleanup.json", {"kind": "cleanup_audit", "runs": [{"path": str(unused)}]})
        current = self.root / "runs" / "current-cleanup"
        current.mkdir()
        original = Path.read_bytes
        def guarded(path):
            if path.name == "cleanup.json":
                raise AssertionError("Previous cleanup reports must not be parsed")
            return original(path)
        with patch.object(Path, "read_bytes", guarded):
            report = self.plan(latest, output_directory=current)
        self.assertEqual(self.decisions(report)["unused"], "review_candidate")
        self.assertEqual(self.decisions(report)["previous-cleanup"], "retained")
        self.assertEqual(self.decisions(report)["current-cleanup"], "retained")
        self.assertNotIn(str(previous / "cleanup.json"), report["metadata_sha256"])

    def test_artifact_index_is_not_parsed_even_when_larger_than_metadata_bound(self):
        latest = self.run_dir("latest")
        index = latest / "index.jsonl"
        index.write_text((json.dumps({"image_id": 1, "file_name": "photo.png"}) + "\n") * 100)
        with patch.dict(AUDIT["cleanup_plan"].__globals__, {"MAX_METADATA_BYTES": 256}):
            report = self.plan(latest)
        self.assertTrue(report["dependency_scan_complete"])
        self.assertNotIn(str(index), report["metadata_sha256"])
        self.assertEqual(report["artifact_payload_files_not_parsed"], 1)

    def test_parse_failure_blocks_deletion_recommendations_and_diagnostics_are_capped(self):
        latest = self.run_dir("latest")
        unused = self.run_dir("unused")
        self.json(latest / "large_config.json", {"notes": "x" * 500})
        with patch.dict(AUDIT["cleanup_plan"].__globals__, {"MAX_METADATA_BYTES": 256}):
            report = self.plan(latest)
        self.assertFalse(report["dependency_scan_complete"])
        self.assertEqual(self.decisions(report)["unused"], "unverified_scan_incomplete")
        evidence = AUDIT["_Evidence"](limit=2)
        for i in range(100):
            evidence.add(str(i), {"source": "file.json", "value": "x" * 10000})
        self.assertEqual(len(evidence.rows()), 2)
        self.assertEqual(evidence.statistics()["omitted_occurrences"], 98)
        self.assertLess(len(json.dumps(evidence.rows())), 3000)

    def test_environment_package_relative_hashes_do_not_become_run_artifacts(self):
        latest = self.run_dir("latest")
        package = self.root / "facade_change"
        (package / "methods").mkdir(parents=True)
        (package / "research").mkdir()
        for relative in ("alignment.py", "methods/dinov2.py", "research/paid.py"):
            (package / relative).write_text("# source fingerprint fixture")
        self.json(latest / "nested" / "run.json", {"status": "completed", "environment": {
            "source_sha256": {"alignment.py": "a" * 64, "methods/dinov2.py": "b" * 64,
                              "paid.py": "c" * 64, "retired_module.py": "d" * 64}}})
        report = self.plan(latest)
        self.assertEqual(report["unresolved_statistics"]["occurrences"], 0)
        self.assertEqual(report["missing_source_snapshot_statistics"]["occurrences"], 1)
        self.assertTrue(report["dependency_scan_complete"])
        self.assertEqual(report["missing_source_snapshots"][0]["value"], "retired_module.py")

    def test_qualified_source_hash_still_retains_cross_run_dependency(self):
        source = self.run_dir("source")
        latest = self.run_dir("latest")
        self.json(latest / "provenance.json", {"source_sha256": {"../source/missing.py": "a" * 64}})
        report = self.plan(latest)
        self.assertEqual(self.decisions(report)["source"], "retained")
        self.assertEqual(report["missing_source_snapshot_statistics"]["occurrences"], 0)
        self.assertTrue(report["unresolved"])

    def test_parse_error_survives_saturation_by_ordinary_reference_diagnostics(self):
        latest = self.run_dir("latest")
        for index in range(85):
            self.json(latest / f"reference-{index:03d}_config.json", {"weights": f"missing-{index}.pth"})
        broken = latest / "zzz-broken_config.json"
        broken.write_text("{invalid")
        report = self.plan(latest)
        self.assertEqual(report["unresolved_statistics"]["reported_groups"], 80)
        self.assertGreater(report["unresolved_statistics"]["omitted_occurrences"], 0)
        self.assertEqual(report["scan_error_statistics"]["occurrences"], 1)
        self.assertEqual(report["scan_errors"][0]["source"], str(broken))
        out = Path(self.temp.name) / "audit-report"
        out.mkdir()
        with redirect_stdout(io.StringIO()) as output:
            AUDIT["write_cleanup_report"](out, report)
        self.assertIn("zzz-broken_config.json", output.getvalue())
        self.assertIn("metadata_not_parsed:", output.getvalue())

    def test_console_reference_sample_is_bounded_and_labels_occurrences(self):
        latest = self.run_dir("latest")
        for index in range(8):
            self.json(latest / f"references-{index}_config.json", {"weights": ["missing.pth"] * (index + 1)})
        report = self.plan(latest)
        out = Path(self.temp.name) / "sampled-audit"
        out.mkdir()
        with redirect_stdout(io.StringIO()) as output:
            AUDIT["write_cleanup_report"](out, report)
        sample = [line for line in output.getvalue().splitlines() if line.startswith("  count=")]
        self.assertEqual(len(sample), 5)
        self.assertIn("count=8", sample[0])
        self.assertIn("key=weights", sample[0])
        self.assertIn("value=missing.pth", sample[0])
        self.assertIn("not unique missing files", output.getvalue())

    def test_annotation_payloads_not_opened_while_input_owners_are_retained(self):
        dataset = self.run_dir("dataset")
        prior_annotations = self.run_dir("prior-annotations")
        prior_checkpoint = self.run_dir("prior-checkpoint")
        unrelated = self.run_dir("unrelated")
        annotations = prior_annotations / "preprocessed" / "annotations.json"
        annotations.parent.mkdir()
        annotations.write_text("not parsed as JSON" * 100)
        checkpoint = prior_checkpoint / "best.pth"
        checkpoint.write_bytes(b"never load a checkpoint")
        self.json(dataset / "prepared" / "manifest.json", {"source": {"coco_json": str(annotations)}})
        model = self.run_dir("model", dataset_run=str(dataset), checkpoint=str(checkpoint))
        original = Path.open
        def guarded(path, *args, **kwargs):
            if path == annotations or path == checkpoint:
                raise AssertionError("Only paths, never annotation/checkpoint contents, may be inspected")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guarded), patch.dict(
                AUDIT["cleanup_plan"].__globals__, {"MAX_METADATA_BYTES": 1024}):
            report = self.plan(model)
        self.assertTrue(report["dependency_scan_complete"])
        self.assertEqual(self.decisions(report)["prior-annotations"], "retained")
        self.assertEqual(self.decisions(report)["prior-checkpoint"], "retained")
        self.assertEqual(self.decisions(report)["dataset"], "retained")
        self.assertEqual(self.decisions(report)["unrelated"], "review_candidate")
        self.assertEqual(report["scan_error_statistics"]["occurrences"], 0)
        self.assertNotIn(str(annotations), report["metadata_sha256"])
        self.assertEqual(report["artifact_payload_files_not_parsed"], 1)

    def test_output_hash_tables_and_pixel_results_are_not_lineage(self):
        latest = self.run_dir("latest")
        unrelated = self.run_dir("unrelated")
        self.json(latest / "run.json", {"status": "completed", "config": {},
                                        "artifact_sha256": {"old/missing.png": "a" * 64}})
        self.json(latest / "metrics.json", {"example_path": str(unrelated)})
        (latest / "index.jsonl").write_text("incomplete or malformed generated index")
        report = self.plan(latest)
        self.assertTrue(report["dependency_scan_complete"])
        self.assertEqual(report["unresolved_statistics"]["occurrences"], 0)
        self.assertEqual(self.decisions(report)["unrelated"], "review_candidate")
        self.assertEqual(report["artifact_payload_files_not_parsed"], 2)

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
