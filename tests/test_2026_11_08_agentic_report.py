"""Saved-output tests: no real model or GPU quality claim (2026-11-08)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from facade_change.agentic.evaluation import compare_runs
from facade_change.agentic.report import render_report
from facade_change.io import sha256


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class SavedAgenticReports(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.manifest = {"availability_policy": {"history": "past_only", "similar_splits": ["train"]},
                         "cases": [{"case_id": "a", "building_id": "A", "view_id": "A-front", "split": "test"},
                                   {"case_id": "b", "building_id": "B", "view_id": "B-front", "split": "test"},
                                   {"case_id": "c", "building_id": "C", "view_id": "C-front", "split": "test"}]}
        save(self.root / "cases.json", self.manifest)
        save(self.run / "input_manifest.json", self.manifest)
        save(self.run / "config.json", {"modes": ["pair", "fixed_context", "agent"]})
        labels = {"label_scope": "region_temporal_change", "provenance": "test fixture, not real evidence",
                  "cases": [{"case_id": "a", "change": True}, {"case_id": "b", "change": True},
                            {"case_id": "c", "change": False}]}
        save(self.root / "labels.json", labels)
        for case in self.manifest["cases"]:
            save(self.run / "cases" / case["case_id"] / "case.json",
                 {**case, "baseline": {"decision": "no_change", "status": "completed"}, "images": [], "maps": []})

    def tearDown(self):
        self.temporary.cleanup()

    def result(self, case, mode, decision, status="completed", **values):
        root = self.run / "cases" / case / "modes" / mode
        save(root / "result.json", {"case_id": case, "decision": decision, "status": status, **values})
        save(root / "costs.json", {"seconds": 2, "model_calls": 1, "input_tokens": 10,
                                  "output_tokens": 5, "images": 2, "tool_calls": 0,
                                  "model_loads": 1, "cache_hits": 0, "peak_reserved_bytes": 20})
        return root

    def compare(self, output="comparison"):
        return compare_runs([self.run], self.root / output, self.root / "cases.json", self.root / "labels.json")

    def test_missing_and_errors_remain_in_facade_recall_and_coverage(self):
        self.result("a", "pair", "change")
        self.result("b", "pair", "insufficient_data", "error", error="GPU unavailable")
        self.result("c", "pair", "change")
        report = self.compare()
        summary = next(row for row in report["summary"] if row["mode"] == "pair")
        self.assertEqual(summary["cases"], 3)
        self.assertEqual(summary["recall_macro"], .5)
        self.assertEqual(summary["coverage_macro"], 2 / 3)
        self.assertEqual(summary["fpr_macro"], 1.)
        self.assertEqual(summary["unresolved_positive"], 1)
        self.assertEqual(summary["conditional_recall"], 1.)
        self.assertEqual(summary["seconds"], 6)
        self.assertEqual(len(report["cases"]), 12)
        missing = next(row for row in report["summary"] if row["mode"] == "agent")
        self.assertEqual(missing["status_counts"], {"missing": 3})
        self.assertEqual(missing["coverage_macro"], 0)
        self.assertEqual(missing["recall_macro"], 0)
        self.assertIsNone(missing["seconds"])
        self.assertTrue((self.root / "comparison" / "report.html").is_file())

    def test_text_localization_is_not_a_pixel_mask(self):
        self.result("a", "agent", "change", visible_difference="A crack",
                    localization={"coordinate_system": "region", "bbox_xyxy": [0, 0, 2, 2]})
        target = json.loads((self.root / "labels.json").read_text())
        target["cases"][0].update(mask_path="truth.npy", support_path="support.npy",
                                  mask_task="temporal_change", coordinate_system="region",
                                  boxes_xyxy=[[0, 0, 2, 2]])
        np.save(self.root / "truth.npy", np.ones((2, 2), bool))
        np.save(self.root / "support.npy", np.ones((2, 2), bool))
        save(self.root / "labels.json", target)
        report = self.compare()
        row = next(row for row in report["cases"] if row["case_id"] == "a" and row["mode"] == "agent")
        self.assertEqual(row["pixels"]["status"], "not_available")
        self.assertEqual(row["localization"]["best_box_iou"], 1.)
        self.assertFalse(list((self.root / "comparison").glob("*.npy")))

    def test_pixel_metrics_require_temporal_semantics_and_matching_support(self):
        base = self.run / "cases" / "a"
        np.save(base / "prediction.npy", np.ones((2, 2), bool))
        np.save(base / "support.npy", np.array([[1, 0], [0, 0]], bool))
        np.save(self.root / "truth.npy", np.array([[1, 0], [0, 0]], bool))
        np.save(self.root / "known.npy", np.ones((2, 2), bool))
        case = json.loads((base / "case.json").read_text())
        case["baseline"].update(mask_path="prediction.npy", support_path="support.npy",
                                mask_task="temporal_change", coordinate_system="anchor-a")
        save(base / "case.json", case)
        labels = json.loads((self.root / "labels.json").read_text())
        labels["cases"][0].update(mask_path="truth.npy", support_path="known.npy",
                                  mask_task="temporal_change", coordinate_system="anchor-a")
        save(self.root / "labels.json", labels)
        report = self.compare()
        row = next(row for row in report["cases"] if row["case_id"] == "a" and row["mode"] == "baseline")
        self.assertEqual(row["pixels"]["valid_pixels"], 1)
        self.assertEqual(row["pixels"]["iou"], 1.)
        case["baseline"]["mask_task"] = "single_image_semantic"
        save(base / "case.json", case)
        report = self.compare("semantic-mask-comparison")
        row = next(row for row in report["cases"] if row["case_id"] == "a" and row["mode"] == "baseline")
        self.assertEqual(row["pixels"]["status"], "not_evaluated")
        self.assertEqual(row["pixels"]["reason"], "mask_is_not_explicit_temporal_change")

    def test_cohort_and_access_policy_cannot_change_between_modes(self):
        changed = json.loads(json.dumps(self.manifest))
        changed["availability_policy"]["history"] = "all_dates"
        save(self.run / "input_manifest.json", changed)
        with self.assertRaisesRegex(ValueError, "availability policy"):
            self.compare()
        self.assertFalse((self.root / "comparison").exists())

    def test_single_frame_damage_is_not_temporal_gt(self):
        save(self.root / "labels.json", {"label_scope": "damage_segmentation", "cases": []})
        with self.assertRaisesRegex(ValueError, "single-image damage"):
            self.compare()

    def test_report_escapes_model_text_and_keeps_missing_cases(self):
        base = self.run / "cases" / "a"
        image = np.full((4, 4, 3), 100, np.uint8)
        Image.fromarray(image).save(base / 'image".png')
        np.save(base / "scores.npy", np.arange(16, dtype=np.float32).reshape(4, 4))
        np.save(base / "support.npy", np.ones((4, 4), bool))
        case = json.loads((base / "case.json").read_text())
        case.update(images=[{"role": "source", "path": 'image".png'}],
                    maps=[{"method": "test fixture", "status": "completed", "scores_path": "scores.npy",
                           "support_path": "support.npy", "metadata": {"units": "fixture scores"}}])
        save(base / "case.json", case)
        self.result("a", "pair", "change", visible_difference="<script>alert('x')</script>")
        (self.run / "cases" / "b" / "case.json").unlink()
        path = render_report(self.run)
        html = path.read_text()
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("Case missing", html)
        self.assertIn("per-map scale, not probability", html)
        self.assertTrue((path.parent / "report_sources.json").is_file())
        self.assertTrue(list((path.parent / "assets").glob("*.png")))
        with self.assertRaises(FileExistsError):
            render_report(self.run)

    def test_evidence_and_tool_failures_are_reported(self):
        mode = self.result("a", "agent", "change", evidence_ids=["known", "invented"])
        save(mode / "evidence.json", {"evidence": [{"evidence_id": "known"}]})
        (mode / "trace.jsonl").write_text('\n'.join(json.dumps(row) for row in [
            {"event_type": "tool_returned", "name": "history", "status": "completed"},
            {"event_type": "tool_returned", "name": "sam3", "status": "error"}]))
        report = self.compare()
        row = next(row for row in report["cases"] if row["case_id"] == "a" and row["mode"] == "agent")
        self.assertEqual(row["invalid_evidence_ids"], ["invented"])
        self.assertEqual(row["evidence_reference_valid_fraction"], .5)
        self.assertEqual(row["tool_success_fraction"], .5)

    def test_recorded_artifact_mutation_is_rejected_before_comparison(self):
        path = self.result("a", "pair", "change") / "result.json"
        relative = path.relative_to(self.run).as_posix()
        save(self.run / "run.json", {"status": "completed", "artifact_sha256": {relative: sha256(path)}})
        save(path, {"status": "completed", "decision": "no_change"})
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.compare()
        self.assertFalse((self.root / "comparison").exists())

    def test_classwise_masks_require_explicit_coordinates(self):
        base = self.run / "cases" / "a"
        np.save(base / "prediction.npy", np.array([[1, 0], [0, 0]], bool))
        np.save(base / "support.npy", np.ones((2, 2), bool))
        np.save(self.root / "truth.npy", np.array([[1, 0], [0, 0]], bool))
        np.save(self.root / "known.npy", np.ones((2, 2), bool))
        case = json.loads((base / "case.json").read_text())
        spec = dict(mask_path="prediction.npy", support_path="support.npy",
                    mask_task="temporal_change", coordinate_system="anchor-a")
        case["baseline"]["class_masks"] = {"crack": spec}
        save(base / "case.json", case)
        labels = json.loads((self.root / "labels.json").read_text())
        labels["cases"][0]["class_masks"] = {"crack": {**spec, "mask_path": "truth.npy", "support_path": "known.npy"}}
        save(self.root / "labels.json", labels)
        report = self.compare()
        self.assertEqual(report["classes"][0]["iou_macro"], 1.)
        labels["cases"][0]["class_masks"]["crack"]["coordinate_system"] = "different-anchor"
        save(self.root / "labels.json", labels)
        report = self.compare("wrong-frame")
        self.assertEqual(report["classes"], [])
        row = next(row for row in report["cases"] if row["case_id"] == "a" and row["mode"] == "baseline")
        self.assertEqual(row["classes"]["crack"]["reason"], "coordinate_system_mismatch")

    def test_runtime_elapsed_and_nested_worker_memory_are_measured(self):
        mode = self.result("a", "agent", "change")
        save(mode / "costs.json", {"elapsed_seconds": 9.5, "peak_reserved_bytes": 0,
             "calls": [{"kind": "qwen", "usage": {"attempts": [
                 {"usage": {"memory": {"peak_reserved_bytes": 120, "peak_allocated_bytes": 80}}},
                 {"usage": {"memory": {"peak_reserved_bytes": 90, "peak_allocated_bytes": 75}}}]}}]})
        case_path = self.run / "cases" / "a" / "case.json"
        case = json.loads(case_path.read_text())
        case["baseline_costs"] = {"elapsed_seconds": 3.25, "scope": "shared preparation and configured methods"}
        save(case_path, case)
        report = self.compare()
        row = next(row for row in report["cases"] if row["case_id"] == "a" and row["mode"] == "agent")
        self.assertEqual(row["seconds"], 9.5)
        self.assertEqual(row["peak_reserved_bytes"], 120)
        self.assertEqual(row["peak_allocated_bytes"], 80)
        base = next(row for row in report["cases"] if row["case_id"] == "a" and row["mode"] == "baseline")
        self.assertEqual(base["seconds"], 3.25)

    def test_cost_allocation_separates_preparation_increment_and_cached_source(self):
        case_path = self.run / "cases" / "a" / "case.json"
        case = json.loads(case_path.read_text())
        case["preparation_costs"] = {"seconds": 8, "geometry_seconds": 3,
                                      "scope": "shared geometry and detector maps"}
        case["baseline_costs"] = {"seconds": 8}
        save(case_path, case)
        for mode in ("pair", "fixed_context", "agent"):
            root = self.result("a", mode, "change")
            save(root / "costs.json", {"elapsed_seconds": 2, "model_calls": 0, "cache_hits": 1,
                 "calls": [{"usage": {"elapsed_seconds": .01},
                            "source_usage": {"elapsed_seconds": 40, "model_calls": 1}}]})
        save(self.run / "summary.json", {"elapsed_seconds": 15, "cost_regime": "fresh workers per mode"})
        report = self.compare()
        cases = {row["mode"]: row for row in report["cases"] if row["case_id"] == "a"}
        self.assertEqual(cases["baseline"]["seconds"], 8)
        self.assertIsNone(cases["baseline"]["incremental_seconds"])
        self.assertEqual(cases["pair"]["incremental_seconds"], 2)
        self.assertEqual(cases["pair"]["shared_preparation_seconds"], 8)
        self.assertEqual(cases["pair"]["pair_geometry_plus_increment_seconds"], 5)
        self.assertIsNone(cases["pair"]["allocated_with_shared_seconds"])
        for mode in ("fixed_context", "agent"):
            self.assertEqual(cases[mode]["allocated_with_shared_seconds"], 10)
            self.assertIsNone(cases[mode]["pair_geometry_plus_increment_seconds"])
            self.assertEqual(cases[mode]["cached_source_usage"][0]["elapsed_seconds"], 40)
            self.assertEqual(cases[mode]["model_calls"], 0)
        self.assertEqual(report["source_runs"][0]["actual_run_elapsed_seconds"], 15)
        case["preparation_costs"].pop("geometry_seconds")
        save(case_path, case)
        unknown = self.compare("unknown-geometry")
        pair = next(row for row in unknown["cases"] if row["case_id"] == "a" and row["mode"] == "pair")
        self.assertIsNone(pair["pair_geometry_plus_increment_seconds"])


if __name__ == "__main__":
    unittest.main()
