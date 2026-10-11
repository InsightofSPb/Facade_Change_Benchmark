"""Model-free cached analysis tests with independently sealed native-RGB inputs."""
from __future__ import annotations

import copy
import builtins
import hashlib
import csv
import io
from contextlib import redirect_stdout
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from facade_change.benchmark_results import ReuseResults
from facade_change.hypothesis_benchmark import _case_metrics, _building_macro
from facade_change.io import sha256


METHODS = ("msdzip_mod256", "anychange", "dinov2", "geoscd", "rgb_diff", "lpips")
COUNT_KEYS = (
    "tp_both", "tp_compression_only", "tp_baseline_only", "fn_both",
    "fp_both", "fp_compression_only", "fp_baseline_only", "tn_both",
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _seal(root, kind, status, config):
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    record = {"schema_version": 1, "kind": kind, "status": status,
              "config": config, "config_sha256": hashlib.sha256(payload).hexdigest(),
              "artifact_sha256": {p.relative_to(root).as_posix(): sha256(p)
                                  for p in sorted(root.rglob("*"))
                                  if p.is_file() and p != root / "run.json"}}
    _write(root / "run.json", record)
    return record


def _fingerprint(dataset):
    return {"parent_run": sha256(dataset / "run.json"),
            "parent_summary": sha256(dataset / "summary.json"),
            "parent_index": sha256(dataset / "index.json"),
            "split": sha256(dataset / "split.json"),
            "config": sha256(dataset / "config.json")}


def _snapshot(*roots):
    return {str(p): p.read_bytes() for root in roots for p in root.rglob("*") if p.is_file()}


def make_fixture(root):
    """Two disjoint facades, six cases each, six frozen scorer outputs."""
    dataset, benchmark = root / "dataset", root / "benchmark"
    dataset.mkdir()
    benchmark.mkdir()
    split = {"mode": "reviewed", "development_only": False,
             "building_assignments": {"val-facade": "val", "test-facade": "test"}}
    scenarios = [{"id": "clean", "kind": "exposure"}, {"id": "blur", "kind": "blur"}]
    dataset_config = {"states": ["unchanged", "crack", "paint_patch"], "scenarios": scenarios}
    _write(dataset / "config.json", dataset_config)
    _write(dataset / "split.json", split)
    bases, cases = [], []
    support = np.ones((4, 4), dtype=bool)
    reference = np.full((4, 4, 3), 80, dtype=np.uint8)
    for part in ("val", "test"):
        base = {"base_id": part + "-base", "building_id": part + "-facade",
                "view_id": part + "-view", "split": part, "dataset_crop_id": part + "-crop",
                "path": "bases/" + part + "-base"}
        bases.append(base)
        folder = dataset / base["path"]
        folder.mkdir(parents=True)
        Image.fromarray(reference).save(folder / "reference_rgb.png")
        Image.fromarray(support.astype(np.uint8) * 255).save(folder / "reference_support.png")
        for state in dataset_config["states"]:
            for scenario in scenarios:
                identifier = f"{base['base_id']}-{state}-{scenario['id']}"
                prefix = "cases/" + identifier
                folder = dataset / prefix
                folder.mkdir(parents=True)
                edit = np.zeros((4, 4), dtype=bool)
                if state != "unchanged":
                    edit.flat[:4] = True
                source = reference.copy()
                source[edit] = 180
                labels = edit.astype(np.uint8)
                Image.fromarray(source).save(folder / "source_rgb.png")
                Image.fromarray(labels).save(folder / "labels_reference.png")
                for name, values in (
                    ("source_support", support), ("true_visibility", support),
                    ("comparable", support), ("source_occlusion", np.zeros_like(support)),
                    ("true_edit_mask_reference", edit), ("visible_edit_mask_reference", edit),
                ):
                    Image.fromarray(values.astype(np.uint8) * 255).save(folder / (name + ".png"))
                identity = np.eye(3).tolist()
                geometry = {"source_to_reference": identity, "reference_to_source": identity}
                _write(folder / "geometry.json", geometry)
                paths = {"reference_rgb": base["path"] + "/reference_rgb.png",
                         "reference_support": base["path"] + "/reference_support.png"}
                paths.update({name: prefix + "/" + name + ".png" for name in (
                    "source_rgb", "source_support", "labels_reference", "comparable", "true_visibility",
                    "source_occlusion", "true_edit_mask_reference", "visible_edit_mask_reference")})
                paths["geometry"] = prefix + "/geometry.json"
                case = {"case_id": identifier, **{k: base[k] for k in (
                    "base_id", "building_id", "view_id", "split")},
                    "parent_dataset_crop_id": base["dataset_crop_id"], "state": state,
                    "scenario": copy.deepcopy(scenario), "scenario_id": scenario["id"],
                    "nuisance_kind": scenario["kind"], "hypothesis": "H0" if state == "unchanged" else "H1",
                    "sham_self_paste": False, "full_edit_pixel_count": int(edit.sum()),
                    "visible_edit_pixel_count": int(edit.sum()), "comparable_fraction": 1.,
                    "retained_visible_edit_fraction": 1., "exclude_from_visible_recall": False,
                    "reference_file_name": part + "-2000.png", "source_file_name": part + "-2001.png",
                    "reference_year": 2000, "source_year": 2001,
                    "source_image_sha256": sha256(folder / "source_rgb.png"),
                    **paths, **geometry,
                    "artifact_sha256": {key: sha256(dataset / value) for key, value in paths.items()}}
                cases.append(case)
    index = {"bases": bases, "cases": cases, "split": split}
    _write(dataset / "index.json", index)
    _write(dataset / "summary.json", {"index_path": "index.json", "case_count": len(cases)})
    parent_config = {**dataset_config, "input_sha256": {"config": sha256(dataset / "config.json")}}
    _seal(dataset, "hypothesis_dataset", "completed_needs_review", parent_config)
    fingerprint = _fingerprint(dataset)
    selection = {"schema_version": 1, "mode": "fixed", "bases": bases,
                 "case_ids": [case["case_id"] for case in cases], "input_sha256": fingerprint,
                 "selected_base_count": 2, "selected_case_count": len(cases)}
    _write(benchmark / "selection.json", selection)
    shutil.copyfile(dataset / "summary.json", benchmark / "parent_summary.json")
    shutil.copyfile(dataset / "split.json", benchmark / "split.json")
    metadata, thresholds, rows = {}, {}, []
    for method in METHODS:
        native = method in ("anychange", "geoscd")
        metadata[method] = {"method": method, "output_kind": "native_mask" if native else "score",
                            "raw_units": "theoretical bits per RGB byte" if method == "msdzip_mod256" else None}
        thresholds[method] = {"threshold": .5, "selection_split": None if native else "val",
                              "prediction_rule": "author native binary mask" if native else "score > threshold",
                              "criterion": "author inference; no threshold calibration" if native else "validation-only fixture"}
        for case in cases:
            labels = np.asarray(Image.open(dataset / case["labels_reference"]))
            prediction = np.zeros((4, 4), dtype=bool)
            if case["hypothesis"] == "H1":
                indices = {"msdzip_mod256": [1, 2, 5, 6], "anychange": [0, 4],
                           "dinov2": [0, 1, 4], "geoscd": [0, 1, 3, 5],
                           "rgb_diff": [0, 1, 2, 3, 4], "lpips": [0, 2, 6]}[method]
            else:
                indices = {"msdzip_mod256": [3, 6], "anychange": [6],
                           "dinov2": [0], "geoscd": [3],
                           "rgb_diff": [0, 4], "lpips": [2]}[method]
            prediction.flat[indices] = True
            scores = prediction.astype(np.float32) if native else np.where(prediction, .75, .25).astype(np.float32)
            score_path = f"scores/{method}/{case['case_id']}.npy"
            prediction_path = f"predictions/{method}/{case['case_id']}.png"
            (benchmark / score_path).parent.mkdir(parents=True, exist_ok=True)
            (benchmark / prediction_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(benchmark / score_path, scores, allow_pickle=False)
            Image.fromarray(prediction.astype(np.uint8) * 255).save(benchmark / prediction_path)
            row = {**case, "method": method, "threshold": .5, "scoring_seconds": .1,
                   "score_path": score_path, "score_sha256": sha256(benchmark / score_path),
                   "prediction_path": prediction_path, "prediction_sha256": sha256(benchmark / prediction_path),
                   **_case_metrics(case, scores, labels, np.float32(.5))}
            if native:
                row.update(native_prediction_path=prediction_path,
                           native_prediction_sha256=sha256(benchmark / prediction_path))
            if method == "msdzip_mod256":
                raw_path = f"native_bpb/{method}/{case['case_id']}.npy"
                (benchmark / raw_path).parent.mkdir(parents=True, exist_ok=True)
                np.save(benchmark / raw_path, scores * 4, allow_pickle=False)
                row.update(raw_score_path=raw_path, native_bpb_path=raw_path,
                           raw_score_sha256=sha256(benchmark / raw_path),
                           raw_score_units="theoretical bits per RGB byte")
            rows.append(row)
    _write(benchmark / "metrics.json", {"cases": rows})
    _write(benchmark / "threshold_selection.json", {"methods": thresholds})
    _write(benchmark / "summary.json", {"status": "completed_exploratory", "selected_base_count": 2,
          "selected_case_count": len(cases), "scored_case_method_count": len(rows),
          "thresholds": {method: .5 for method in METHODS}})
    config = {"dataset_run": str(dataset), "input_sha256": fingerprint,
              "methods": list(METHODS), "scorers": metadata}
    _seal(benchmark, "hypothesis_benchmark", "completed_exploratory", config)
    return dataset, benchmark



from data_analysis import core, recall, runner


def make_frozen_fixture(dataset, benchmark, destination):
    """An independent legacy five-method frozen run, without archive scripts."""
    destination.mkdir()
    reuse = ReuseResults(benchmark)
    cases = _read(dataset / "index.json")["cases"]
    methods = (*METHODS[:4], core.FUSION)
    rows = []
    for case in cases:
        labels = np.asarray(Image.open(dataset / case["labels_reference"]))
        predictions = {method: np.load(benchmark / f"scores/{method}/{case['case_id']}.npy",
                                      allow_pickle=False) > np.float32(.5)
                       for method in METHODS[:4]}
        predictions[core.FUSION] = predictions["geoscd"] & predictions[core.COMPRESSION]
        rows.extend(core.prediction_metrics(case, method, predictions[method], labels)
                    for method in methods)
    thresholds = dict.fromkeys(METHODS[:4], .5)
    config = {"source": reuse.provenance, "methods": list(methods),
              "fusion_method": core.FUSION, "rule": "AND",
              "requested_splits": ["val", "test"], "input_sha256": _fingerprint(dataset),
              "case_ids": [c["case_id"] for c in cases], "thresholds": thresholds}
    _write(destination / "frozen_contract.json", config)
    _write(destination / "selected_cases.json", {"cases": cases})
    _write(destination / "summary.json", {
        "status": "completed_exploratory", "thresholds": thresholds,
        "source": reuse.provenance, "analyzed_case_count": len(cases),
        "methods": {method: core.slices([r for r in rows if r["method"] == method],
                                       _building_macro) for method in methods}})
    _write_csv(destination / "metrics.csv", rows)
    _seal(destination, "frozen_geoscd_fusion", "completed_exploratory", config)
    return destination


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _reseal(root):
    record = _read(root / "run.json")
    return _seal(root, record["kind"], record["status"], record["config"])


class CachedAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset, self.benchmark = make_fixture(self.root)

    def run_analysis(self, out, fusion_run=None, gallery_limit=0):
        with redirect_stdout(io.StringIO()):
            return runner.run_analysis(self.benchmark, out, fusion_run=fusion_run,
                                       gallery_limit=gallery_limit)

    def case(self, hypothesis="H1", building="facade"):
        case = copy.deepcopy(_read(self.dataset / "index.json")["cases"][2])
        case.update(hypothesis=hypothesis, building_id=building)
        return case

    def test_all_methods_legacy_fusion_parity_and_immutable_inputs(self):
        fusion = make_frozen_fixture(self.dataset, self.benchmark, self.root / "frozen")
        before = _snapshot(self.dataset, self.benchmark, fusion)
        output = self.root / "analysis"
        seen = []
        original_load = ReuseResults.load
        original_inputs = runner._case_inputs
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if (name.split(".")[0] in {"torch", "cv2"}
                    or name == "facade_change.scorers"
                    or name.startswith("facade_change.methods.")):
                self.fail("Cached analysis attempted an inference dependency import: " + name)
            return original_import(name, *args, **kwargs)

        def guarded_inputs(root, parent, base, case, hashes):
            self.assertTrue((output / "analysis_contract.json").is_file())
            contract = _read(output / "analysis_contract.json")
            self.assertIn(case["case_id"], contract["case_ids"])
            self.assertEqual(contract["requested_splits"], ["val", "test"])
            return original_inputs(root, parent, base, case, hashes)

        def guarded_load(reuse, method, case_id):
            # Both splits are processed only after a durable frozen contract exists.
            self.assertTrue((output / "analysis_contract.json").is_file())
            contract = _read(output / "analysis_contract.json")
            self.assertEqual(contract["thresholds"], dict.fromkeys(METHODS, .5))
            self.assertEqual(contract["test_fusion_rules"], ["GeoSCD AND MSDZip only"])
            seen.append((method, case_id))
            return original_load(reuse, method, case_id)

        with (patch.object(ReuseResults, "load", guarded_load),
              patch.object(runner, "_case_inputs", guarded_inputs),
              patch.object(builtins, "__import__", guarded_import)):
            summary = self.run_analysis(output, fusion_run=fusion, gallery_limit=2)
        self.assertEqual(summary["status"], "completed_exploratory")
        self.assertEqual(summary["analyzed_case_count"], 12)
        self.assertEqual(summary["cases_by_split"], {"val": 6, "test": 6})
        self.assertEqual(summary["buildings_by_split"], {"val": 1, "test": 1})
        self.assertEqual(set(summary["methods"]), set(METHODS) | {core.FUSION})
        self.assertEqual(set(summary["pairs"]), set(core.BASELINES))
        self.assertEqual(len(summary["val_fusion_diagnostics"]), 10)
        for result in summary["val_fusion_diagnostics"].values():
            self.assertEqual(set(result["by_split"]), {"val"})
            self.assertTrue(all(g["split"] == "val" for g in result["by_state"]))
        for part in ("val", "test"):
            geo = summary["methods"]["geoscd"]["by_split"][part]["means"]
            fused = summary["methods"][core.FUSION]["by_split"][part]["means"]
            diagnostic = summary["pairs"]["geoscd"]["by_split"][part]["means"]
            self.assertAlmostEqual(geo["recall"], .75)
            self.assertAlmostEqual(fused["recall"], .25)
            self.assertAlmostEqual(fused["f1"], 1 / 3)
            self.assertAlmostEqual(fused["h0_pixel_fpr"], 1 / 16)
            self.assertEqual(diagnostic["lost_recall"], .5)
            self.assertAlmostEqual(diagnostic["rejected_baseline_tp_fraction"], 2 / 3)
        self.assertEqual(set(seen), {(m, c["case_id"]) for m in METHODS
                                    for c in _read(self.dataset / "index.json")["cases"]})
        self.assertEqual(_snapshot(self.dataset, self.benchmark, fusion), before)
        for name in ("summary.json", "summary.txt", "facades.csv", "classes.csv",
                     "conditions.csv", "case_metrics.csv", "recall_cases.csv",
                     "score_diagnostics.csv", "gallery.html", "analysis_contract.json",
                     "selected_cases.json", "input_artifacts.json", "run.json"):
            self.assertTrue((output / name).is_file(), name)
        self.assertEqual(len(_csv(output / "case_metrics.csv")), 12 * 7)
        self.assertEqual(len(_csv(output / "recall_cases.csv")), 12 * 5)
        self.assertEqual(len(_csv(output / "score_diagnostics.csv")), 12 * 10)
        self.assertEqual({r["building_id"] for r in _csv(output / "facades.csv")},
                         {"val-facade", "test-facade"})
        text = (output / "summary.txt").read_text()
        self.assertIn("val-facade", text)
        self.assertIn("test-facade", text)
        self.assertIn("crack", text)
        self.assertIn("blur", text)
        self.assertIn("GeoSCD", (output / "gallery.html").read_text())
        record = _read(output / "run.json")
        self.assertEqual(record["status"], "completed_exploratory")
        self.assertEqual(record["artifact_sha256"]["summary.json"], sha256(output / "summary.json"))

    def test_recall_denominators_and_ignored_pixels(self):
        labels = np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 255]], dtype=np.uint8)
        geo = np.array([[1, 1, 1, 1, 0, 0, 1, 1, 0, 1]], dtype=bool)
        msd = np.array([[1, 1, 0, 0, 0, 0, 1, 0, 1, 1]], dtype=bool)
        counts = core.pair_counts(msd, geo, labels)
        self.assertEqual(counts, {"tp_both": 2, "tp_compression_only": 0,
            "tp_baseline_only": 2, "fn_both": 2, "fp_both": 1,
            "fp_compression_only": 1, "fp_baseline_only": 1, "tn_both": 0})
        self.assertEqual(sum(counts.values()), 9)
        result = recall.diagnostic_row(self.case(), "geoscd", counts)
        self.assertAlmostEqual(result["baseline_recall"], 4 / 6)
        self.assertAlmostEqual(result["confirmed_recall"], 2 / 6)
        self.assertAlmostEqual(result["lost_recall"], 2 / 6)
        self.assertAlmostEqual(result["rejected_baseline_tp_fraction"], 2 / 4)
        self.assertAlmostEqual(result["removed_h1_fpr"], 1 / 3)
        h0 = recall.diagnostic_row(self.case("H0"), "geoscd", counts)
        self.assertIsNone(h0["lost_recall"])
        self.assertIsNone(h0["baseline_recall"])
        self.assertAlmostEqual(h0["removed_h0_fpr"], 1 / 3)
        with self.assertRaises(ValueError):
            core.pair_counts(msd.astype(np.uint8), geo, labels)
        with self.assertRaises(ValueError):
            recall.diagnostic_row(self.case(), "geoscd", {**counts, "tp_both": True})

    def test_facade_means_have_equal_weight_not_pooled_counts(self):
        high = dict.fromkeys(core.COUNT_KEYS, 0)
        high.update(tp_baseline_only=1, tn_both=1)
        low = dict.fromkeys(core.COUNT_KEYS, 0)
        low.update(tp_both=9, tn_both=9)
        rows = [recall.diagnostic_row(self.case(building="A"), "geoscd", high)
                for _ in range(3)]
        rows += [recall.diagnostic_row(self.case(building="B"), "geoscd", low)]
        result = recall.diagnostic_macro(rows)
        self.assertEqual(result["case_count"], 4)
        self.assertEqual(result["building_count"], 2)
        self.assertEqual(result["means"]["lost_recall"], .5)
        self.assertNotEqual(result["means"]["lost_recall"], .75)
        self.assertNotEqual(result["means"]["lost_recall"], 3 / 12)
        self.assertEqual(result["pixel_counts_diagnostic_only"]["tp_both"], 9)

    def test_boundaries_and_score_gaps_ignore_unknown_and_image_outside(self):
        labels = np.array([[1, 1, 255], [1, 0, 255]], dtype=np.uint8)
        expected = np.array([[1, 1, 0], [1, 0, 0]], dtype=bool)
        np.testing.assert_array_equal(recall.positive_boundary(labels), expected)
        np.testing.assert_array_equal(recall.positive_boundary(np.ones((2, 2), dtype=np.uint8)),
                                      np.zeros((2, 2), dtype=bool))
        np.testing.assert_array_equal(recall.positive_boundary(
            np.array([[1, 255], [255, 255]], dtype=np.uint8)), np.zeros((2, 2), dtype=bool))
        geo = labels != 0
        msd = np.zeros_like(geo)
        msd[0, 0] = True
        scores = np.full(labels.shape, .25, dtype=np.float32)
        scores[0, 0] = .75
        scores[labels == 255] = np.nan
        rgb_a = np.full((*labels.shape, 3), 255, dtype=np.uint8)
        rgb_b = np.zeros_like(rgb_a)
        rows = recall.feature_rows(self.case(), geo, msd, labels, scores, scores * 4,
                                   rgb_a, rgb_b, .5)
        groups = {row["group"]: row for row in rows}
        rejected = groups["rejected_tp"]
        self.assertEqual(rejected["pixel_count"], 2)
        self.assertEqual(rejected["positive_boundary_pixel_count"], 2)
        self.assertEqual(rejected["threshold_gap_mean"], .25)
        self.assertEqual(rejected["raw_bpb_mean"], 1.)
        self.assertEqual(rejected["rgb_abs_meanchannel_mean"], 255.)
        self.assertEqual(groups["kept_tp"]["threshold_gap_mean"], -.25)
        self.assertEqual(groups["rejected_tp_other"]["pixel_count"], 0)
        self.assertIsNone(groups["rejected_tp_other"]["msd_score_mean"])

    def test_original_threshold_is_strict_and_native_masks_preserved(self):
        relative = "scores/dinov2/test-base-crack-clean.npy"
        values = np.load(self.benchmark / relative, allow_pickle=False)
        values.flat[7] = np.float32(.5)
        np.save(self.benchmark / relative, values, allow_pickle=False)
        metrics = _read(self.benchmark / "metrics.json")
        for row in metrics["cases"]:
            if row["score_path"] == relative:
                row["score_sha256"] = sha256(self.benchmark / relative)
        _write(self.benchmark / "metrics.json", metrics)
        _reseal(self.benchmark)
        output = self.root / "strict"
        summary = self.run_analysis(output)
        row = next(row for row in _csv(output / "case_metrics.csv")
                   if row["case_id"] == "test-base-crack-clean" and row["method"] == "dinov2")
        self.assertEqual(int(row["fp"]), 1)
        self.assertEqual(summary["thresholds"], dict.fromkeys(METHODS, .5))
        relative = "predictions/geoscd/test-base-crack-clean.png"
        native = np.asarray(Image.open(self.benchmark / relative)).copy()
        native.flat[1] = 0
        Image.fromarray(native).save(self.benchmark / relative)
        metrics = _read(self.benchmark / "metrics.json")
        for row in metrics["cases"]:
            if row["prediction_path"] == relative:
                row["prediction_sha256"] = sha256(self.benchmark / relative)
                row["native_prediction_sha256"] = sha256(self.benchmark / relative)
        _write(self.benchmark / "metrics.json", metrics)
        _reseal(self.benchmark)
        with self.assertRaisesRegex(ValueError, "author.*mask|native"):
            self.run_analysis(self.root / "native-changed")

    def test_corrupted_cached_score_rejected_before_numerical_analysis(self):
        score = self.benchmark / "scores/lpips/test-base-crack-clean.npy"
        values = np.load(score, allow_pickle=False)
        values[0, 0] = .125
        np.save(score, values, allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "changed|SHA256|hash"):
            self.run_analysis(self.root / "corrupted")

    def test_frozen_source_and_artifact_tampering_rejected(self):
        fusion = make_frozen_fixture(self.dataset, self.benchmark, self.root / "frozen")
        summary_path = fusion / "summary.json"
        pristine = summary_path.read_bytes()
        summary_path.write_bytes(pristine + b" ")
        with self.assertRaisesRegex(ValueError, "artifact|SHA256"):
            self.run_analysis(self.root / "artifact-bad", fusion_run=fusion)
        summary_path.write_bytes(pristine)
        config = _read(fusion / "frozen_contract.json")
        original_config = copy.deepcopy(config)
        config["source"]["run_sha256"] = "0" * 64
        _write(fusion / "frozen_contract.json", config)
        _seal(fusion, "frozen_geoscd_fusion", "completed_exploratory", config)
        with self.assertRaisesRegex(ValueError, "another benchmark"):
            self.run_analysis(self.root / "source-bad", fusion_run=fusion)
        config = original_config
        config["thresholds"]["msdzip_mod256"] = .75
        _write(fusion / "frozen_contract.json", config)
        _seal(fusion, "frozen_geoscd_fusion", "completed_exploratory", config)
        with self.assertRaisesRegex(ValueError, "thresholds differ"):
            self.run_analysis(self.root / "threshold-bad", fusion_run=fusion)

    def test_resealed_fusion_count_mismatch_marks_output_failed(self):
        fusion = make_frozen_fixture(self.dataset, self.benchmark, self.root / "frozen")
        rows = _csv(fusion / "metrics.csv")
        row = next(row for row in rows if row["method"] == core.FUSION
                   and row["case_id"] == "test-base-crack-clean")
        row["tp"] = str(int(row["tp"]) + 1)
        _write_csv(fusion / "metrics.csv", rows)
        _reseal(fusion)
        output = self.root / "counts-bad"
        with self.assertRaisesRegex(ValueError, "Per-case metrics differ"):
            self.run_analysis(output, fusion_run=fusion)
        self.assertEqual(_read(output / "run.json")["status"], "failed")
        self.assertTrue((output / "analysis_contract.json").is_file())


if __name__ == "__main__":
    unittest.main()
