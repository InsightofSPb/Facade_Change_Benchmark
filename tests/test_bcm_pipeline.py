"""BCM orchestration preserves H0 isolation, exact references, and provenance."""
import hashlib
import importlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from facade_change.bcm_training import train_bcm_h0, validate_bcm_training_run
from facade_change.cli import parser
from facade_change.io import read_json, sha256, write_json
from facade_change.methods.registry import ALL_METHODS, make_method
from facade_change.method_checks import check_methods

FINGERPRINT = {key: "a"*64 for key in ("parent_run", "parent_summary", "parent_index", "split", "config")}


def _record(run, config):
    write_json(run / "run.json", {"schema_version": 1, "kind": "bcm_h0_training",
        "status": "completed_exploratory", "config": config,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False,
                                                   allow_nan=False).encode()).hexdigest(),
        "artifact_sha256": {path.relative_to(run).as_posix(): sha256(path)
                            for path in run.rglob("*") if path.is_file() and path.name != "run.json"}})


class BCMIntegrationTests(unittest.TestCase):
    def test_cli_uses_the_bounded_full_codec_pilot_defaults(self):
        args = vars(parser().parse_args(["bcm-train", "--dataset-run", "data", "--out", "pilot",
            "--vtm-encoder", "enc", "--vtm-decoder", "dec", "--vtm-config", "ra.cfg",
            "--init-checkpoint", "MRNet.pth"]))
        self.assertEqual((args["epochs"], args["max_train_patches"], args["max_val_patches"],
                          args["batch_size"], args["tile_size"], args["lr"], args["seed"]),
                         (3, 160, 16, 1, 32, 1e-4, 42))
        self.assertEqual(args["init_checkpoint"], "MRNet.pth")
        self.assertIn("bcm_net_rgb", ALL_METHODS)
        adapter = unittest.mock.Mock(return_value="adapter")
        with patch("facade_change.methods.registry.importlib.import_module",
                   return_value=SimpleNamespace(BCMScorer=adapter)):
            self.assertEqual(make_method("bcm_net_rgb", training_run="pilot"), "adapter")
        adapter.assert_called_once_with(method="bcm_net_rgb", training_run="pilot")

    def test_preflight_binds_dataset_and_forces_bitstreams_without_TEST_reads(self):
        h0 = importlib.import_module("facade_change.2026-10-04_msdzip_h0")
        calls = []

        class Pool:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        class Scorer:
            def __init__(self, method, options, pool):
                calls.append((method, options))
                self.metadata = {"last_codec_stats": {"preflight": True}}
                self.native_prediction = None

            def activate(self):
                pass

            def __call__(self, a, b, support):
                return np.mean(np.abs(b.astype(np.float32)-a), axis=2).astype(np.float32)

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root / "config.json"
            write_json(config, {"dataset_run": str(root / "data"), "out": str(root / "benchmark"),
                "method_options": {"bcm_net_rgb": {"training_run": "pilot", "cost_mode": "theoretical"}}})
            with patch.object(h0, "_parent", return_value=(None, None, None, None, FINGERPRINT)), \
                 patch.object(h0, "_rgb_inputs", side_effect=AssertionError("TEST/RGB must not be read")), \
                 patch("facade_change.method_checks.RemotePool", Pool), \
                 patch("facade_change.method_checks.RemoteScorer", Scorer):
                result = check_methods(config, root / "check", ["bcm_net_rgb"], bitstream_check=True)
            self.assertEqual(result["status"], "completed_needs_review")
            self.assertEqual(calls[0][0], "bcm_net_rgb")
            self.assertEqual(calls[0][1]["dataset_fingerprint"], FINGERPRINT)
            self.assertEqual(calls[0][1]["cost_mode"], "bitstream")
            self.assertEqual((calls[0][1]["tile_size"], calls[0][1]["stride"]), (32, 32))
            report = read_json(root / "check/report.json")["bcm_net_rgb"]
            self.assertEqual([row["control"] for row in report["controls"]],
                             ["unchanged", "paint_patch", "wraparound", "wraparound_reverse"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required")
class BCMTrainingTests(unittest.TestCase):
    def fixture(self, folder, nonfinite=False):
        import torch
        folder = Path(folder)
        dataset, source = folder / "data", folder / "source"
        dataset.mkdir()
        source.mkdir()
        bases = [{"base_id": key, "building_id": key, "split": part}
                 for key, part in (("train-a", "train"), ("train-b", "train"),
                                   ("val-a", "val"), ("val-b", "val"), ("test", "test"))]
        cases = [{**base, "case_id": base["base_id"]+"-"+state, "state": state,
                  "hypothesis": "H0" if state == "unchanged" else "H1",
                  "sham_self_paste": False, "scenario_id": "shadow"}
                 for base in bases for state in ("unchanged", "paint_patch")]
        parent = {"config": {"states": ["unchanged", "paint_patch"], "scenarios": [{"id": "shadow"}]}}
        h0 = importlib.import_module("facade_change.2026-10-04_msdzip_h0")
        reads, calls, models = [], [], []

        def rgb(root, parent, base, case, checked):
            reads.append(case)
            self.assertIn(case["split"], ("train", "val"))
            self.assertEqual(case["state"], "unchanged")
            size = 64 if case["base_id"] == "val-a" else 32
            value = {"train-a": 20, "train-b": 30, "val-a": 10, "val-b": 100}[case["base_id"]]
            a = np.full((size, size, 3), value, dtype=np.uint8)
            b = np.full_like(a, 250)
            checked[case["case_id"]] = "verified"
            return a, b, np.ones((size, size), dtype=bool)

        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.5))

        class Codec:
            def __init__(self, *args, **kwargs):
                calls.append(kwargs)
                self.network = Network().requires_grad_(False)
                self.metadata = {"source": "toy author", "cost_mode": kwargs["cost_mode"]}
                models.append(self.network)

            def base_reconstruct(self, a, b):
                # Deliberately distinguish lossy VTM A from the exact reference A.
                return np.full_like(a, 199), b-5, {"base_bytes": 17}

            def close(self):
                pass

        def nll(network, residues, base, reference=None):
            self.assertEqual(tuple(residues.shape[1:]), (1, 32, 32))
            self.assertTrue(torch.all(residues == 5))
            self.assertTrue(torch.all(base == 245))
            self.assertFalse(torch.any(reference == 199))
            if nonfinite:
                return network.weight * torch.full((len(base),), float("inf"))
            return (torch.nn.functional.softplus(network.weight)
                    + reference.mean(dim=(1, 2, 3))/255) * 1024

        patches = [patch("facade_change.methods.bcm_net.BCMCodec", Codec),
                   patch("facade_change.methods.bcm_training_core.bcm_nll", nll),
                   patch.object(h0, "_parent", return_value=(dataset, parent, {}, {}, FINGERPRINT.copy())),
                   patch.object(h0, "_rgb_inputs", side_effect=rgb),
                   patch("facade_change.bcm_training._select", return_value=(bases, cases))]
        for active in patches:
            active.start()
            self.addCleanup(active.stop)
        kwargs = {"dataset_run": dataset, "out": folder / "pilot", "source_root": source,
                  "vtm_encoder": folder / "enc", "vtm_decoder": folder / "dec",
                  "vtm_config": folder / "ra.cfg", "init_checkpoint": folder / "MRNet.pth",
                  "epochs": 2, "max_train_patches": 2, "max_val_patches": 5, "lr": .1, "seed": 73}
        return kwargs, reads, calls, models

    def test_training_uses_exact_A_signed_residuals_and_macro_validation(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            kwargs, reads, calls, models = self.fixture(folder)
            result = train_bcm_h0(**kwargs)
            run = Path(result["out"])
            _, record, summary = validate_bcm_training_run(run, FINGERPRINT)
            self.assertTrue(reads)
            self.assertEqual(result["selected_patches"], {"train": 2, "val": 5})
            self.assertEqual(calls[0]["checkpoint"], kwargs["init_checkpoint"])
            self.assertEqual((calls[0]["qp"], calls[0]["cost_mode"]), (37, "bitstream"))
            self.assertEqual(calls[0]["seed"], 73)
            self.assertLess(float(models[0].weight), .5)
            expected = float(torch.nn.functional.softplus(models[0].weight).detach()) + 55/255
            final = summary["history"][-1]["val_building_macro_residual_theoretical_bpb"]
            self.assertAlmostEqual(final, expected, places=6)
            self.assertGreater(abs(final - (expected - 27/255)), .1)
            sampling = read_json(run / "sampling.json")["partitions"]
            self.assertFalse(set(row["building_id"] for row in sampling["train"]) &
                             set(row["building_id"] for row in sampling["val"]))
            self.assertEqual(record["config"]["channels"], ["R", "G", "B"])
            for name, digest in record["artifact_sha256"].items():
                self.assertEqual(sha256(run / name), digest)
            weights = torch.load(run / "model.pth", map_location="cpu", weights_only=True)
            self.assertAlmostEqual(float(weights["weight"]), float(models[0].weight), places=6)

    def test_nonfinite_loss_records_failed_run_without_completed_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            kwargs, _, _, _ = self.fixture(folder, nonfinite=True)
            with self.assertRaisesRegex(RuntimeError, "objective became nonfinite"):
                train_bcm_h0(**kwargs)
            record = read_json(Path(kwargs["out"]) / "run.json")
            self.assertEqual(record["status"], "failed")
            self.assertFalse((Path(kwargs["out"]) / "model.pth").exists())
            with self.assertRaisesRegex(ValueError, "completed H0"):
                validate_bcm_training_run(kwargs["out"])


class BCMConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "configs").mkdir()
        self.dataset = self.root / "data"
        self.dataset.mkdir()
        self.fingerprint = FINGERPRINT.copy()
        self.rscd = ("rscd_cmu", "rscd_diff_cmu", "rscd_pscd")
        write_json(self.root / "configs/benchmark.local.json", {"dataset_run": str(self.dataset),
            "out": str(self.root / "old"), "device": "cpu",
            "method_options": {name: {"source_root": "/author/rscd"} for name in self.rscd}})
        self.reuse = SimpleNamespace(summary={"selected_base_count": 10, "selected_case_count": 60},
            methods=("rgb_diff",) + self.rscd + ("bcm_net_rgb",), input_sha256=self.fingerprint.copy())
        spec = importlib.util.spec_from_file_location("configure_bcm_test",
            Path(__file__).resolve().parents[1] / "scripts/configure_codecs.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        for active in (patch.object(self.module, "ReuseResults", return_value=self.reuse),
                       patch.object(self.module, "_dataset_fingerprint", return_value=self.fingerprint.copy())):
            active.start()
            self.addCleanup(active.stop)
        self.run = self.root / "training"
        self.run.mkdir()
        self.config = {"input_sha256": self.fingerprint.copy(), "tile_size": 32, "qp": 37,
                       "channels": ["R", "G", "B"], "seed": 73}
        for key in ("vtm_encoder", "vtm_decoder", "vtm_config", "vtm_scc_config"):
            path = self.root / key
            path.write_bytes(b"pinned local artifact")
            self.config[key] = str(path)
        for name in ("sampling.json", "history.json", "base_reconstruction.json"):
            write_json(self.run / name, {})
        (self.run / "model.pth").write_bytes(b"local checkpoint")
        write_json(self.run / "summary.json", {"status": "completed_exploratory",
            "checkpoint": "model.pth", "channels": ["R", "G", "B"]})
        _record(self.run, self.config)

    def configure(self, **kwargs):
        return self.module.configure(repo_root=self.root, reuse_run=self.root / "old",
                                     bcm_training_run=self.run, **kwargs)

    def test_full_BCM_config_preserves_cached_methods_and_records_actual_cost(self):
        result = self.configure()
        config = read_json(result["config"])
        options = config["method_options"]["bcm_net_rgb"]
        self.assertEqual(result["cached_methods"], ["rgb_diff"])
        self.assertEqual(result["recompute_methods"], list(self.rscd) + ["bcm_net_rgb"])
        self.assertEqual((options["cost_mode"], options["qp"], options["tile_size"], options["seed"]),
                         ("bitstream", 37, 32, 73))
        self.assertEqual(options["training_run"], str(self.run))
        self.assertEqual(options["vtm_encoder"], self.config["vtm_encoder"])
        self.assertEqual(options["source_root"], str(self.root / "third_party/bcm_net"))
        self.assertFalse((self.root / "runs/2026-10-07-codecs-001").exists())

    def test_training_fingerprint_corrupt_artifacts_and_changed_protocol_are_rejected(self):
        for config in ({**self.config, "input_sha256": {**self.fingerprint, "split": "b"*64}},
                       {**self.config, "qp": 0}, {**self.config, "channels": ["Y"]}):
            with self.subTest(config=config):
                _record(self.run, config)
                with self.assertRaises(ValueError):
                    self.configure()
        _record(self.run, self.config)
        (self.run / "model.pth").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact missing or hash changed"):
            self.configure()
        self.assertFalse((self.root / "configs/codecs.local.json").exists())

    def test_symlink_escape_missing_VTM_and_output_inside_training_are_rejected(self):
        original = (self.run / "model.pth").read_bytes()
        outside = self.root / "outside.pth"
        outside.write_bytes(original)
        (self.run / "model.pth").unlink()
        (self.run / "model.pth").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes its run"):
            self.configure()
        (self.run / "model.pth").unlink()
        (self.run / "model.pth").write_bytes(original)
        Path(self.config["vtm_encoder"]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "vtm_encoder missing"):
            self.configure()
        Path(self.config["vtm_encoder"]).write_bytes(b"pinned local artifact")
        with self.assertRaisesRegex(ValueError, "immutable input run"):
            self.configure(out=self.run / "nested")


if __name__ == "__main__":
    unittest.main()
