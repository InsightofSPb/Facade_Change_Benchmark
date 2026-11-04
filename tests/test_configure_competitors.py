import contextlib
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch


from facade_change.io import read_json, sha256, write_json
from scripts.real_eval.robustness_common import digest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("configure_competitors", ROOT / "scripts/configure_competitors.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

BUNDLE_SPEC = importlib.util.spec_from_file_location(
    "model_consolidation", ROOT / "scripts/2026-11-04-consolidate-models.py")
BUNDLE = importlib.util.module_from_spec(BUNDLE_SPEC)
BUNDLE_SPEC.loader.exec_module(BUNDLE)


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


class ModelConsolidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.models = self.root / "models"
        self.source_run = self.root / "runs/2026-10-18-night-run-001"
        self.output = self.root / "configs/2026-11-04-methods.local.json"
        self.audit = self.models / "2026-11-04-consolidation.json"
        self.training = self.root / "runs/semantic-training"
        self.missing = self.root / "old-machine"
        self.dino = self.file(self.models / "dinov2/dinov2_vits14_pretrain.pth", b"frozen dino")
        self.sam = self.file(self.models / "downloads/sam_vit_h_4b8939.pth", b"frozen sam")
        self.dino_source = self.models / "dinov2/source"
        self.file(self.dino_source / "hubconf.py", b"def model(): pass\n")
        self.file(self.dino_source / "dinov2/models/vision_transformer.py", b"class Model: pass\n")
        dino_sources = BUNDLE.dino_sources(self.dino_source)
        dino_sources["path"] = str(self.missing / "dino-source")
        self.anychange = self.root / "env/lib/python3.10/site-packages"
        subtree = Path("torchange/models/segment_any_change")
        for name, content in (("__init__.py", b""), ("anychange.py", b"class AnyChange: pass\n"),
                              ("segment_anything/model.py", b"class SAM: pass\n")):
            self.file(self.anychange / subtree / name, content)
        anychange_sources = BUNDLE.anychange_sources(self.anychange)
        anychange_sources["source_root"] = str(self.missing / "anychange")
        for entry in anychange_sources["files"]:
            entry["path"] = str(self.missing / "anychange" / entry["relative_path"])
        anychange_sources["files"].append(dict(
            path=str(self.missing / "anychange/LICENSE"), relative_path="LICENSE",
            sha256=hashlib.sha256(b"original license").hexdigest(), size_bytes=16))
        anychange_sources["manifest_sha256"] = digest([
            dict(path=entry["relative_path"], sha256=entry["sha256"])
            for entry in anychange_sources["files"]])
        self.geoscd = self.root / "sources/geoscd"
        self.file(self.geoscd / "src/model.py", b"class GeoSCD: pass\n")
        lpips_files = [self.entry(self.file(self.root / "env/lpips" / name, name.encode()))
                       for name in ("__init__.py", "lpips.py", "pretrained_networks.py")]
        backbone = self.file(self.root / "cache/alexnet.pth", b"alexnet weights")
        calibration = self.file(self.root / "env/lpips/weights/alex.pth", b"lpips calibration")
        head = self.file(self.training / "best.pth", b"semantic head with frozen provenance")
        write_json(self.training / "config.json", dict(checkpoint=str(self.missing / self.dino.name)))
        write_json(self.training / "run.json", dict(kind="dino_semantic_training", status="completed_exploratory"))
        write_json(self.training / "summary.json", dict(best_epoch=2))
        vggt = self.file(self.root / "cache/vggt_1b.pt", b"vggt weights")
        dino_entry = self.entry(self.dino, self.missing / "weights" / self.dino.name)
        sam_entry = self.entry(self.sam, self.missing / "weights" / self.sam.name)
        metadata = {
            "rgb_diff": {"method": "rgb_diff"},
            "ssim": {"method": "ssim"},
            "lpips": dict(weights=dict(backbone=self.entry(backbone), calibration=self.entry(calibration)),
                          package=dict(version="0.1.4", source_files=lpips_files)),
            "dinov2": dict(checkpoint=dino_entry, sources=dino_sources),
            "semantic_dino": dict(training_run=str(self.training), head_checkpoint=self.entry(head),
                                  backbone_provenance=dict(checkpoint=deepcopy(dino_entry),
                                                           sources=deepcopy(dino_sources))),
            "anychange": dict(checkpoint=deepcopy(sam_entry), source=anychange_sources),
            "geoscd": dict(checkpoints=dict(vggt=self.entry(vggt), sam=deepcopy(sam_entry)),
                           source=self.geo_sources(self.geoscd)),
        }
        for representation in ("abs", "mod256"):
            checkpoint = self.file(self.root / "runs/msdzip" / (representation + ".pt"),
                                   ("frozen msdzip " + representation).encode())
            metadata["msdzip_" + representation] = dict(
                checkpoint_path=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                representation=representation, protocol="frozen fixture", window_groups=4)
        self.contract = dict(methods=list(BUNDLE.DEFAULT_METHODS), models={})
        for method in BUNDLE.DEFAULT_METHODS:
            options = {} if method in {"rgb_diff", "ssim"} else {
                "worker_python": str(Path(sys.executable).resolve()), "device": "cuda:0"}
            if method == "semantic_dino":
                options["training_run"] = str(self.training)
                options["crop_size"] = 224
            if method == "geoscd":
                options.update(source_root=str(self.geoscd), points_per_side=32, iou_thresh=.65)
            if method.startswith("msdzip_"):
                options["dataset_fingerprint"] = {"frozen_training": "fixture"}
            self.contract["models"][method] = dict(
                options=options, expected_metadata=metadata[method],
                model_files_sha256=BUNDLE._file_hashes(metadata[method]))
        self.save_contract()
        self.originals = self.snapshot()

    def file(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def entry(self, path, recorded_path=None):
        return dict(path=str(recorded_path or path), sha256=sha256(path), size_bytes=path.stat().st_size)

    def geo_sources(self, path):
        path = Path(path)
        return dict(root=str(path), commit="fixture-pinned-commit",
                    source_sha256={"src/model.py": sha256(path / "src/model.py")})

    def save_contract(self, **overrides):
        self.source_run.mkdir(parents=True, exist_ok=True)
        record = dict(kind="real_temporal_run", status="completed", contract=self.contract,
                      config_hash=digest(self.contract))
        record.update(overrides)
        write_json(self.source_run / "run.json", record)

    def snapshot(self):
        return {path: sha256(path) for path in self.root.rglob("*") if path.is_file()}

    def consolidate(self, prepared=None, dino_search_root=None, output_stream=None):
        with patch.object(BUNDLE, "geoscd_sources", side_effect=self.geo_sources), \
                patch.object(BUNDLE.Path, "home", return_value=self.root), \
                contextlib.redirect_stdout(output_stream if output_stream is not None else io.StringIO()):
            return BUNDLE.consolidate(self.source_run if prepared is None else None,
                                      self.models, self.anychange, self.output, "cpu", prepared=prepared,
                                      dino_search_root=dino_search_root)

    def assert_preflight_failure(self, message):
        with patch.object(BUNDLE, "_copy_weight") as copy:
            with self.assertRaisesRegex(ValueError, message):
                self.consolidate()
        copy.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.audit.exists())

    def test_all_nine_relocate_deduplicate_preserve_provenance_and_repeat(self):
        existing_dino_stat = self.dino.stat()
        report = self.consolidate()
        config = read_json(self.output)["methods"]
        self.assertEqual(list(config), list(BUNDLE.DEFAULT_METHODS))
        self.assertEqual(config["dinov2"]["checkpoint_path"], config["semantic_dino"]["backbone_checkpoint"])
        self.assertEqual(config["anychange"]["checkpoint_path"], config["geoscd"]["sam_checkpoint"])
        self.assertEqual(config["semantic_dino"]["training_run"], str(self.training))
        self.assertEqual(config["semantic_dino"]["crop_size"], 224)
        self.assertEqual(config["semantic_dino"]["source_root"], str(self.dino_source))
        self.assertEqual(config["geoscd"]["iou_thresh"], .65)
        for method in ("msdzip_abs", "msdzip_mod256"):
            self.assertEqual(config[method]["worker_python"], str(Path(sys.executable).resolve()))
            self.assertEqual(config[method]["dataset_fingerprint"], {"frozen_training": "fixture"})
        for method in set(BUNDLE.DEFAULT_METHODS) - {"rgb_diff", "ssim"}:
            self.assertEqual(config[method]["device"], "cpu")
        destinations = {entry["path"] for entry in report["weight_relocations"].values()}
        self.assertEqual(len(destinations), 8)
        for entry in report["weight_relocations"].values():
            self.assertEqual(sha256(entry["path"]), entry["sha256"])
        audit = report["source_audits"]["anychange"]
        self.assertTrue(audit["python_files_match"])
        self.assertEqual(audit["packaging_differences"], ["LICENSE"])
        self.assertNotEqual(audit["original_manifest_sha256"], audit["actual_manifest_sha256"])
        for path, checksum in self.originals.items():
            self.assertEqual(sha256(path), checksum, str(path))
        self.assertEqual(self.dino.stat().st_ino, existing_dino_stat.st_ino)
        self.assertEqual(self.dino.stat().st_mtime_ns, existing_dino_stat.st_mtime_ns)
        before = self.snapshot()
        self.assertEqual(self.consolidate(), report)
        self.assertEqual(self.snapshot(), before)

    def test_bad_run_kind_status_or_contract_checksum_rejected_before_writes(self):
        for override in ({"kind": "other_run"}, {"status": "running"}, {"status": "failed"},
                         {"status": "completed_unverified"}, {"config_hash": "0" * 64}):
            with self.subTest(override=override):
                self.save_contract(**override)
                self.assert_preflight_failure("completed, hash-bound")

    def test_prepared_metadata_relocates_while_original_runtime_checks_stay_strict(self):
        from scripts.real_eval.robustness_run import _method_contract

        prepared = self.root / "prepared"
        prepared.mkdir()
        manifest = dict(schema_version=1, kind="real_temporal_prepared", source_run=str(self.source_run),
            source_config=dict(device="cuda:0", method_options={
                name: deepcopy(model["options"]) for name, model in self.contract["models"].items()}),
            source_records=[dict(method=name, status="completed",
                variant="current_train" if name.startswith("msdzip_") else "frozen",
                method_metadata=deepcopy(model["expected_metadata"]))
                for name, model in self.contract["models"].items()],
            categories=[dict(id=1, name="WALL")], sequences=[dict(sequence_id="history", building_id="b1",
                split="train", frames=[dict(frame=index, year=year, artifact_path=f"frames/{index}.npz",
                    artifact_sha256="0" * 64) for index, year in enumerate((2010, 2020))])])
        path = prepared / "manifest.json"
        write_json(path, manifest)
        original = path.read_bytes()
        source, contracts = BUNDLE._contracts(prepared=prepared)
        self.assertEqual(source, path)
        self.assertEqual(list(contracts), list(BUNDLE.DEFAULT_METHODS))
        self.assertEqual(contracts["semantic_dino"]["options"]["training_run"], str(self.training))
        with self.assertRaisesRegex(ValueError, "dinov2"):
            _method_contract("dinov2", manifest, self.source_run, None)
        report = self.consolidate(prepared=prepared)
        self.assertEqual(report["source_record"], str(path))
        self.assertEqual(report["source_record_sha256"], sha256(path))
        self.assertEqual(path.read_bytes(), original)
        self.assertTrue(self.output.is_file())

    def test_dino_search_selects_exact_relocated_tree_and_rejects_other_candidates(self):
        search_root = self.root / "other-projects"
        search_root.mkdir()
        matched = search_root / "b-relocated-dino"
        self.dino_source.rename(matched)
        different = search_root / "a-different-dino"
        shutil.copytree(matched, different)
        self.file(different / "dinov2/models/vision_transformer.py", b"class OtherModel: pass\n")
        unrelated = self.file(search_root / "unrelated/hubconf.py", b"not a DINO source tree\n")
        discovery = BUNDLE.subprocess.CompletedProcess([], 0, stdout="\n".join(map(str, (
            different / "hubconf.py", unrelated, matched / "hubconf.py"))) + "\n", stderr="")
        captured = io.StringIO()
        with patch.object(BUNDLE.subprocess, "run", return_value=discovery) as search:
            report = self.consolidate(dino_search_root=search_root, output_stream=captured)
        search.assert_called_once()
        command = search.call_args.args[0]
        self.assertEqual(command[0], "rg")
        self.assertIn(str(search_root), command)
        self.assertIn("--no-ignore", command)
        config = read_json(self.output)["methods"]
        expected = BUNDLE.dino_sources(matched)["python_tree_sha256"]
        for method in ("dinov2", "semantic_dino"):
            self.assertEqual(config[method]["source_root"], str(matched))
            self.assertEqual(report["source_audits"][method]["python_tree_sha256"], expected)
        self.assertIn(expected, captured.getvalue())
        self.assertIn("MATCH", captured.getvalue())

    def test_dino_search_hash_mismatch_reports_expected_and_actual_without_writes(self):
        search_root = self.root / "other-projects"
        search_root.mkdir()
        different = search_root / "different-dino"
        self.dino_source.rename(different)
        self.file(different / "dinov2/models/vision_transformer.py", b"class ChangedModel: pass\n")
        expected = self.contract["models"]["dinov2"]["expected_metadata"]["sources"]["python_tree_sha256"]
        actual = BUNDLE.dino_sources(different)["python_tree_sha256"]
        self.assertNotEqual(expected, actual)
        discovery = BUNDLE.subprocess.CompletedProcess(
            [], 0, stdout=str(different / "hubconf.py") + "\n", stderr="")
        before, captured = self.snapshot(), io.StringIO()
        with patch.object(BUNDLE.subprocess, "run", return_value=discovery), \
                patch.object(BUNDLE, "_copy_weight") as copy:
            with self.assertRaisesRegex(ValueError, "no DINO source tree matching"):
                self.consolidate(dino_search_root=search_root, output_stream=captured)
        copy.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.audit.exists())
        self.assertEqual(self.snapshot(), before)
        for text in (expected, actual, str(different), "EXPECTED", "DIFF"):
            self.assertIn(text, captured.getvalue())

    def test_missing_method_or_conflicting_recorded_hash_rejected(self):
        original = deepcopy(self.contract)
        del self.contract["models"]["msdzip_mod256"]
        self.save_contract()
        self.assert_preflight_failure("no frozen model contract for msdzip_mod256")
        self.contract = original
        model = self.contract["models"]["dinov2"]
        model["model_files_sha256"][model["expected_metadata"]["checkpoint"]["path"]] = "0" * 64
        self.save_contract()
        self.assert_preflight_failure("Conflicting model hashes")

    def test_anychange_changed_missing_or_extra_python_rejected_before_copy(self):
        path = self.anychange / "torchange/models/segment_any_change/anychange.py"
        original = path.read_bytes()
        extra = path.with_name("extra.py")
        for change in ("changed", "missing", "extra"):
            with self.subTest(change=change):
                path.write_bytes(original)
                extra.unlink(missing_ok=True)
                if change == "changed":
                    path.write_bytes(b"changed implementation")
                elif change == "missing":
                    path.unlink()
                else:
                    extra.write_bytes(b"additional implementation")
                self.assert_preflight_failure("AnyChange Python sources differ")

    def test_partial_and_wrong_hash_candidates_are_not_used(self):
        partial = self.sam.with_name(self.sam.name + ".part")
        self.sam.rename(partial)
        self.file(self.models / "wrong-copy" / self.sam.name, b"different weights")
        before = self.snapshot()
        self.assert_preflight_failure("No matching weights for SHA256")
        self.assertEqual(self.snapshot(), before)

    def test_existing_wrong_destination_is_never_replaced(self):
        target = self.file(self.models / self.sam.name, b"unrelated existing model")
        self.assert_preflight_failure("Existing destination differs")
        self.assertEqual(target.read_bytes(), b"unrelated existing model")

    def test_different_semantic_backbone_is_not_deduplicated(self):
        backbone = self.file(self.root / "cache/semantic-backbone.pth", b"different frozen backbone")
        model = self.contract["models"]["semantic_dino"]
        model["expected_metadata"]["backbone_provenance"]["checkpoint"] = self.entry(backbone)
        model["model_files_sha256"] = BUNDLE._file_hashes(model["expected_metadata"])
        self.save_contract()
        self.consolidate()
        config = read_json(self.output)["methods"]
        semantic = Path(config["semantic_dino"]["backbone_checkpoint"])
        self.assertNotEqual(str(semantic), config["dinov2"]["checkpoint_path"])
        self.assertEqual(semantic.read_bytes(), backbone.read_bytes())

    def test_shared_sam_consumers_with_different_hashes_fail_before_copy(self):
        alternate = self.file(self.models / "alternate" / self.sam.name, b"another frozen SAM")
        model = self.contract["models"]["geoscd"]
        model["expected_metadata"]["checkpoints"]["sam"] = self.entry(alternate)
        model["model_files_sha256"] = BUNDLE._file_hashes(model["expected_metadata"])
        self.save_contract()
        self.assert_preflight_failure("Consumers require different weights")

    def test_existing_config_and_audit_tampering_is_not_overwritten(self):
        self.consolidate()
        original_config, original_audit = self.output.read_bytes(), self.audit.read_bytes()
        for path, message in ((self.output, "Output config already exists"),
                              (self.audit, "Existing consolidation record differs")):
            with self.subTest(path=path):
                self.output.write_bytes(original_config)
                self.audit.write_bytes(original_audit)
                write_json(path, {"tampered": True})
                with patch.object(BUNDLE, "_copy_weight") as copy:
                    with self.assertRaisesRegex(ValueError, message):
                        self.consolidate()
                copy.assert_not_called()
                self.assertEqual(read_json(path), {"tampered": True})


if __name__ == "__main__":
    unittest.main()
