"""Resource profiling must be bounded, reproducible and isolated from TEST."""
import importlib
import json
from pathlib import Path
import random
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import torch

preflight = importlib.import_module("facade_change.research.2026-11-08_history_scenarios_preflight")
data = importlib.import_module("facade_change.research.2026-11-08_history_scenarios_data")


class ResourcePreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self):
        return dict(matrix_protocol="reduced-v2", device="cpu", smoke=True, seed=13,
                    epochs=2, patience=2, effective_batch=2, microbatch=1,
                    amp=False, profile_steps=2, profile_warmup_steps=1,
                    cpu_threads=1, fusion_epochs=1, fusion_lr=.01,
                    bootstrap_repeats=4, minimum_free_ram_gib=0,
                    minimum_free_vram_gib=0)

    def test_examples_are_deterministic_train_quantiles(self):
        dataset = data.SyntheticData()
        selected = preflight.deterministic_examples(dataset, predictive=True)
        self.assertEqual([row["role"] for row in selected], ["typical", "long", "maximum"])
        self.assertTrue(all(row["row"]["split"] == "train" for row in selected))
        self.assertEqual(selected[-1]["row"]["frame"], max(row["frame"] for row in dataset.split_samples("train", True)))
        self.assertEqual(selected, preflight.deterministic_examples(dataset, True))

    def test_validation_profile_preserves_complete_overlap_scenes(self):
        dataset=data.SyntheticData()
        initial=preflight.validation_examples(dataset)
        selected=initial[0]
        duplicate={**selected,"event_id":selected["event_id"]+"-overlap","bbox_xywh":[2,0,8,8]}
        dataset.samples.append(duplicate)
        rows=preflight.validation_examples(dataset)
        self.assertIn(duplicate["event_id"],{row["event_id"] for row in rows})
        self.assertTrue(all(row["split"]=="val" for row in rows))
        keys=("building_id","view_id","previous_year","current_year")
        scenes={tuple(row[k] for k in keys) for row in rows}
        self.assertLessEqual(len(scenes),2)
        expected={row["event_id"] for row in dataset.samples if tuple(row[k] for k in keys) in scenes}
        self.assertEqual({row["event_id"] for row in rows},expected)

    def test_rng_and_threads_restore_even_on_failure(self):
        random.seed(3)
        np.random.seed(3)
        torch.manual_seed(3)
        expected = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        threads = torch.get_num_threads()
        with self.assertRaises(RuntimeError):
            with preflight.isolated_rng(1):
                random.random()
                np.random.random()
                torch.rand(3)
                raise RuntimeError("profile interrupted")
        self.assertEqual(random.getstate(), expected[0])
        np.testing.assert_array_equal(np.random.get_state()[1], expected[1][1])
        torch.testing.assert_close(torch.get_rng_state(), expected[2], rtol=0, atol=0)
        self.assertEqual(torch.get_num_threads(), threads)

    def test_fingerprint_ignores_available_resources_but_not_science_config(self):
        dataset, config = data.SyntheticData(), self.config()
        hardware = preflight.hardware_snapshot()
        left = preflight.fingerprint(dataset, config, {"code": "hash"}, hardware)
        changed = {**hardware, "free_vram_bytes": 11, "gpu_processes": ["foreign"],
                   "ram": {**hardware["ram"], "MemAvailable": 12}}
        self.assertEqual(left, preflight.fingerprint(dataset, config, {"code": "hash"}, changed))
        right = preflight.fingerprint(dataset, {**config, "epochs": 3}, {"code": "hash"}, hardware)
        self.assertNotEqual(left["sha256"], right["sha256"])
        report = dict(protocol=preflight.PROTOCOL, fingerprint=left, status="MEASURED", ready_for_full=True)
        self.assertEqual(preflight.compatible(report, dataset, config, {"code": "hash"}, changed), (True, []))
        report["ready_for_full"] = False
        self.assertFalse(preflight.compatible(report, dataset, config, {"code": "hash"}, hardware)[0])

    def test_resources_fail_without_deleting_or_reducing_configuration(self):
        config = self.config()
        original = dict(config)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "existing.txt"
            path.write_text("keep")
            report = {"storage": {"recommended_free_output_bytes": 2**100}}
            with self.assertRaisesRegex(RuntimeError, "Resource preflight blocked"):
                preflight.check_resources(report, config, root)
            self.assertEqual(path.read_text(), "keep")
            self.assertEqual(config, original)
            checks = preflight.check_resources(report, config, root, raise_on_failure=False)
            self.assertFalse(checks["passed"])

    def test_fusion_peak_and_measured_host_ram_participate_in_resource_gate(self):
        hardware=dict(cuda_available=True,free_vram_bytes=1000,ram={"MemAvailable":100000})
        config={**self.config(),"device":"cuda:0","resource_safety_factor":1.25}
        report=dict(training_profiles=[dict(peak_allocated_vram_bytes=100,peak_reserved_vram_bytes=200)],
                    fusion_profiles=[dict(peak_allocated_vram_bytes=900,peak_reserved_vram_bytes=1000,
                                          peak_host_rss_bytes=10000)])
        with tempfile.TemporaryDirectory() as root:
            checked=preflight.check_resources(report,config,root,hardware,raise_on_failure=False)
            self.assertFalse(checked["passed"])
            self.assertEqual(checked["required_vram_bytes"],1250)
            self.assertEqual(checked["required_ram_bytes"],12500)

    def test_only_own_inactive_cuda_cache_is_counted_as_reusable(self):
        hardware=dict(cuda_available=True,free_vram_bytes=20,own_reserved_vram_bytes=200,
                      own_allocated_vram_bytes=50,ram={"MemAvailable":100000},gpu_processes=["foreign"])
        config={**self.config(),"device":"cuda:0","resource_safety_factor":1.25}
        report=dict(fusion_profiles=[dict(peak_reserved_vram_bytes=100)])
        with tempfile.TemporaryDirectory() as root:
            checked=preflight.check_resources(report,config,root,hardware,raise_on_failure=False)
            self.assertTrue(checked["passed"])
            self.assertEqual(checked["own_reusable_vram_bytes"],150)
            self.assertEqual(checked["available_vram_to_this_process_bytes"],170)
            self.assertEqual(checked["free_vram_bytes"],20)
            self.assertEqual(checked["gpu_processes"],["foreign"])

    def test_expiry_and_overlay_identity(self):
        dataset,config=data.SyntheticData(),{**self.config(),"preflight_max_age_hours":24}
        observations=[o["observation_id"] for t in dataset.tracks.values() if t["split"]=="train" for o in t["observations"]]
        test_observation=next(o["observation_id"] for t in dataset.tracks.values() if t["split"]=="test" for o in t["observations"])
        key="feature:"+observations[0]
        dataset.cache_index={key:dict(origin="run_overlay",artifact_root="/first",artifact_path="first.npy",unit_sha256="first-audit",artifact_sha256="same-scientific-bytes")}
        hardware=preflight.hardware_snapshot()
        original=preflight.fingerprint(dataset,config,{"code":"hash"},hardware)
        dataset.cache_index[key].update(artifact_root="/second",artifact_path="second.npy",unit_sha256="second-audit")
        dataset.cache_index["feature:"+test_observation]=dict(artifact_sha256="new-test-cache")
        self.assertEqual(original,preflight.fingerprint(dataset,config,{"code":"hash"},hardware))
        report=dict(protocol=preflight.PROTOCOL,fingerprint=original,status="MEASURED",ready_for_full=True,created_unix=time.time()-25*3600)
        valid,reasons=preflight.compatible(report,dataset,config,{"code":"hash"},hardware)
        self.assertFalse(valid)
        self.assertTrue(any("expired" in reason for reason in reasons))
        report["created_unix"]=time.time()
        self.assertTrue(preflight.compatible(report,dataset,config,{"code":"hash"},hardware)[0])
        dataset.cache_index[key]["artifact_sha256"]="changed-scientific-bytes"
        self.assertFalse(preflight.compatible(report,dataset,config,{"code":"hash"},hardware)[0])

    def test_profile_failure_persists_partial_report_and_reraises(self):
        dataset=data.SyntheticData()
        training=preflight._module("training")
        torch.manual_seed(117)
        before=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.object(training,"build_model",side_effect=RuntimeError("simulated OOM")):
                with self.assertRaisesRegex(RuntimeError,"simulated OOM"):
                    preflight.run_preflight(dataset,self.config(),root,source_snapshot={"code":"hash"})
            report=json.loads((Path(root)/"2026-11-08_resource_report.json").read_text())
            self.assertEqual(report["status"],"NOT_MEASURED")
            self.assertFalse(report["ready_for_full"])
            self.assertEqual(report["error"]["message"],"simulated OOM")
            self.assertEqual(report["fingerprint"]["identity"]["config"]["effective_batch"],2)
            self.assertFalse(any(Path(root).glob("*profile-temporary-*")))
            torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)

    def test_unavailable_report_does_not_invent_measurements(self):
        with tempfile.TemporaryDirectory() as root:
            result = preflight.unavailable_report(self.config(), root, source_snapshot={"code": "hash"}, reason="Real inputs absent")
            self.assertEqual(result["status"], "NOT_MEASURED")
            self.assertEqual(result["counts"], dict(primary=18, score_fusion=48, feature_fusion=8, short_fusion=56, total=74))
            self.assertFalse(result["ready_for_full"])
            self.assertEqual(result["training_profiles"], [])
            self.assertTrue((Path(root) / "2026-11-08_resource_report.txt").exists())
            saved = json.loads((Path(root) / "2026-11-08_resource_report.json").read_text())
            self.assertEqual(saved, result)

    def test_actual_profile_has_no_test_scientific_access_and_restores_rng(self):
        dataset = data.SyntheticData()
        def guard(method):
            def guarded(row, *args, **kwargs):
                self.assertNotEqual(row["split"], "test", "Scientific TEST access during profiling")
                return method(row, *args, **kwargs)
            return guarded
        for name in ("inputs", "pair_inputs", "targets", "feature_target", "semantic", "annotation_history", "scores", "evaluation_support"):
            setattr(dataset, name, guard(getattr(dataset, name)))
        torch.manual_seed(101)
        before = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as root:
            result = preflight.run_preflight(dataset, self.config(), root, source_snapshot={"code": "hash"}, progress=lambda *a, **k: None)
            torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
            self.assertEqual(result["status"], "SYNTHETIC_MEASURED")
            self.assertFalse(result["ready_for_full"])
            self.assertEqual(len({row["model_id"] for row in result["training_profiles"]}), 18)
            self.assertEqual(len(result["fusion_profiles"]), 5)
            self.assertEqual(sum(row["maps_to_fit_count"] for row in result["fusion_profiles"]), 56)
            self.assertEqual({row["split"] for row in result["training_profiles"]}, {"train"})
            self.assertTrue(all(len(row["seconds_per_optimizer_cycle"]) == 2 for row in result["training_profiles"]))
            self.assertTrue(all(row["effective_batch"] == 2 for row in result["training_profiles"]))
            self.assertEqual(result["storage"]["schema_version"], 2)
            self.assertTrue(any("report_only_full_entrypoint_seconds" in row for row in result["validation_profiles"]))
            self.assertGreater(result["measured_artifact_writes"]["materialized_bytes"],0)
            self.assertEqual(result["eligible_fit_plan"]["actual_primary_fits"],0)
            self.assertTrue(all(row["optimizer"]=="Adam" and row["effective_batch"]==2 for row in result["fusion_profiles"]))
            self.assertTrue(all(len(row["inference_seconds_per_unit"])==2 for row in result["fusion_profiles"]))
            self.assertTrue(all("VAL_inference_seconds" in row and "final_inference_seconds" in row for row in result["time_estimates"]["fusion"]))
            self.assertGreater(result["storage"]["permanent_uncompressed_upper_bytes"], 0)
            self.assertFalse(any(Path(root).glob("*profile-temporary-*")))
