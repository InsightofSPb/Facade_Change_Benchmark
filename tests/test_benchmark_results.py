import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from facade_change.benchmark_results import ReuseResults
from facade_change.benchmark_subset import replay_selection, select_quick_subset
from facade_change.io import finish_record, read_json, sha256, write_json
from test_benchmark_subset import source_rows


def fixture_rows():
    bases, cases = source_rows(val=1, test=1, train=0, crops=1)
    for base in bases:
        base.update(path="bases/" + base["base_id"], dataset_crop_id=base["base_id"] + "/crop")
    for case in cases:
        base = next(row for row in bases if row["base_id"] == case["base_id"])
        case.update(hypothesis="H1" if case["state"] in {"crack", "paint_patch"} else "H0",
                    sham_self_paste=case["state"] == "self_paste",
                    parent_dataset_crop_id=base["dataset_crop_id"],
                    reference_rgb=base["path"] + "/reference_rgb.png",
                    reference_support=base["path"] + "/reference_support.png")
    return bases, cases


def save_selection(root):
    bases, cases = fixture_rows()
    selected, chosen, quick = select_quick_subset(bases, cases, max_bases=2)
    hashes = {key: hashlib.sha256(key.encode()).hexdigest()
              for key in ("parent_run", "parent_summary", "parent_index", "split", "config")}
    selection = {**quick, "quick_selection": copy.deepcopy(quick), "bases": selected,
                 "case_ids": [row["case_id"] for row in chosen], "input_sha256": hashes}
    write_json(root / "selection.json", selection)
    return bases, cases, selected, chosen, hashes


def save_run(root, raw=True):
    bases, cases, selected, chosen, hashes = save_selection(root)
    method = "msdzip_abs" if raw else "rgb_diff"
    choice = {"threshold": float(np.float32(.43)), "selection_split": "val",
              "prediction_rule": "score > threshold", "curve": [{"threshold": .43}]}
    config = {"methods": [method], "input_sha256": hashes,
              "scorers": {method: {"method": method, "checkpoint_sha256": "preserved"}}}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    record = {"kind": "hypothesis_benchmark", "config": config, "config_sha256": digest}
    rows = []
    for case in chosen:
        relative = f"scores/{method}/{case['case_id']}.npy"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        values = np.array([[.43, .5, .1], [.99, .2, np.nan]], dtype=np.float32)
        np.save(path, values, allow_pickle=False)
        row = {**case, "method": method, "score_path": relative, "score_sha256": sha256(path),
               "threshold": choice["threshold"], "scoring_seconds": 43.125,
               "evaluated_pixel_count": 5, "ignored_pixel_count": 1}
        if raw:
            relative_raw = f"native_bpb/{method}/{case['case_id']}.npy"
            raw_path = root / relative_raw
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(raw_path, values * np.float32(8), allow_pickle=False)
            row["native_bpb_path"] = relative_raw
        rows.append(row)
    write_json(root / "metrics.json", {"cases": rows})
    write_json(root / "threshold_selection.json", {"methods": {method: choice}})
    write_json(root / "summary.json", {"status": "completed_exploratory", "selected_base_count": 2,
               "selected_case_count": 12, "scored_case_method_count": 12, "thresholds": {method: choice["threshold"]}})
    finish_record(root, record, "completed_exploratory")
    return bases, cases, selected, chosen, hashes, method


class SelectionReplayTests(unittest.TestCase):
    def test_replay_keeps_exact_order_and_verified_parent_row_objects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bases, cases, selected, chosen, hashes = save_selection(root)
            actual_bases, actual_cases, metadata = replay_selection(list(reversed(bases)), list(reversed(cases)), hashes, root / "selection.json")
            self.assertEqual(actual_bases, selected)
            self.assertEqual(actual_cases, chosen)
            self.assertIs(actual_bases[0], selected[0])
            self.assertIs(actual_cases[0], chosen[0])
            self.assertEqual(metadata["replayed_selection_sha256"], sha256(root / "selection.json"))

    def test_replay_rejects_changed_inputs_identity_order_and_scenarios(self):
        for corruption in ("fingerprint", "base", "case", "scenario", "support", "hypothesis", "duplicate", "order", "missing"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                bases, cases, selected, chosen, hashes = save_selection(root)
                saved = read_json(root / "selection.json")
                if corruption == "fingerprint":
                    hashes["split"] = "0" * 64
                elif corruption == "base":
                    selected[0]["dataset_crop_id"] = "different"
                elif corruption == "case":
                    chosen[0]["view_id"] = "different"
                elif corruption == "scenario":
                    chosen[0]["scenario"] = {"id": "different", "kind": chosen[0]["nuisance_kind"]}
                elif corruption == "support":
                    chosen[0]["reference_support"] = "other/support.png"
                elif corruption == "hypothesis":
                    chosen[0]["hypothesis"] = "H1"
                elif corruption == "duplicate":
                    saved["case_ids"].append(saved["case_ids"][0])
                elif corruption == "order":
                    saved["bases"].reverse()
                else:
                    saved["case_ids"].pop()
                write_json(root / "selection.json", saved)
                with self.assertRaises(ValueError):
                    replay_selection(bases, cases, hashes, root / "selection.json")

    def test_replay_requires_full_fingerprint_and_rejects_hidden_building_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bases, cases, _, _, hashes = save_selection(root)
            hashes.pop("parent_index")
            with self.assertRaisesRegex(ValueError, "five"):
                replay_selection(bases, cases, hashes, root / "selection.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bases, cases, _, _, hashes = save_selection(root)
            bases.append({**bases[0], "base_id": "hidden", "split": "train"})
            with self.assertRaisesRegex(ValueError, "across splits"):
                replay_selection(bases, cases, hashes, root / "selection.json")


class ReuseResultsTests(unittest.TestCase):
    def test_merged_run_reuses_generic_raw_maps_with_original_units(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, bases, cases, hashes, method = save_run(root)
            metrics = read_json(root / "metrics.json")
            for row in metrics["cases"]:
                row["raw_score_path"] = row.pop("native_bpb_path")
                row["raw_score_units"] = "original perceptual distance"
            write_json(root / "metrics.json", metrics)
            finish_record(root, read_json(root / "run.json"), "completed_exploratory")
            reuse = ReuseResults(root)
            reuse.validate_selection(hashes, bases, cases)
            result = reuse.load(method, cases[0]["case_id"])
            self.assertIsNotNone(result["raw_scores"])
            self.assertEqual(result["raw_score_units"], "original perceptual distance")

    def test_author_native_masks_can_be_reused_without_validation_threshold(self):
        for forged in (False, True):
            with self.subTest(forged=forged), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _, _, bases, cases, hashes, method = save_run(root, raw=False)
                record = read_json(root / "run.json")
                record["config"]["scorers"][method]["output_kind"] = "score" if forged else "native_mask"
                record["config_sha256"] = hashlib.sha256(json.dumps(
                    record["config"], sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
                choice = {"threshold": .5, "grid_index": 50, "selection_split": None,
                          "prediction_rule": "author native binary mask",
                          "criterion": "author inference; no threshold calibration"}
                write_json(root / "threshold_selection.json", {"methods": {method: choice}})
                summary = read_json(root / "summary.json")
                summary["thresholds"][method] = .5
                write_json(root / "summary.json", summary)
                metrics = read_json(root / "metrics.json")
                for row in metrics["cases"]:
                    path = root / row["score_path"]
                    values = np.load(path, allow_pickle=False)
                    native = values > .5
                    scores = native.astype(np.float32)
                    scores[np.isnan(values)] = np.nan
                    np.save(path, scores, allow_pickle=False)
                    relative = f"native_predictions/{method}/{row['case_id']}.png"
                    image_path = root / relative
                    image_path.parent.mkdir(exist_ok=True, parents=True)
                    Image.fromarray(native.astype(np.uint8) * 255).save(image_path)
                    row.update(threshold=.5, score_sha256=sha256(path),
                               native_prediction_path=relative, native_prediction_sha256=sha256(image_path))
                write_json(root / "metrics.json", metrics)
                finish_record(root, record, "completed_exploratory")
                if forged:
                    with self.assertRaisesRegex(ValueError, "thresholds"):
                        ReuseResults(root)
                else:
                    reuse = ReuseResults(root)
                    reuse.validate_selection(hashes, bases, cases)
                    result = reuse.load(method, cases[0]["case_id"])
                    np.testing.assert_array_equal(result["native_prediction"], result["scores"] > .5)

    def test_reuse_loads_scores_without_models_preserves_thresholds_timing_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, bases, cases, hashes, method = save_run(root)
            with patch("torch.load", side_effect=AssertionError("Model loading forbidden")):
                reuse = ReuseResults(root)
                reuse.validate_selection(hashes, bases, cases)
                result = reuse.load(method, cases[0]["case_id"])
            self.assertEqual(reuse.methods, (method,))
            self.assertEqual(result["scoring_seconds"], 43.125)
            self.assertEqual(result["scores"].shape, (2, 3))
            np.testing.assert_array_equal(result["raw_scores"], result["scores"] * np.float32(8))
            self.assertEqual(result["row"]["threshold"], reuse.thresholds[method]["threshold"])
            metadata = reuse.metadata(method)
            metadata["checkpoint_sha256"] = "mutated"
            self.assertEqual(reuse.metadata(method)["checkpoint_sha256"], "preserved")
            self.assertEqual(reuse.provenance["run_sha256"], sha256(root / "run.json"))

    def test_old_rgb_scores_do_not_require_raw_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, bases, cases, hashes, method = save_run(root, raw=False)
            reuse = ReuseResults(root)
            reuse.validate_selection(hashes, bases, cases)
            self.assertIsNone(reuse.load(method, cases[0]["case_id"])["raw_scores"])

    def test_missing_maps_or_tampering_is_rejected_before_new_inference(self):
        for corruption in ("missing", "scores", "raw", "metrics", "selection", "thresholds", "config", "status"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _, _, _, cases, _, method = save_run(root)
                case_id = cases[0]["case_id"]
                if corruption == "missing":
                    (root / f"scores/{method}/{case_id}.npy").unlink()
                elif corruption in {"scores", "raw"}:
                    relative = "scores" if corruption == "scores" else "native_bpb"
                    (root / f"{relative}/{method}/{case_id}.npy").write_bytes(b"tampered")
                elif corruption in {"metrics", "selection"}:
                    (root / (corruption + ".json")).write_text("{}")
                elif corruption == "thresholds":
                    (root / "threshold_selection.json").write_text("{}")
                else:
                    record = read_json(root / "run.json")
                    if corruption == "status":
                        record["status"] = "failed"
                    else:
                        record["config"]["methods"] = ["other"]
                    write_json(root / "run.json", record)
                with self.assertRaises(ValueError):
                    ReuseResults(root)

    def test_reuse_requires_exact_selected_cases_and_original_parent_metadata(self):
        for corruption in ("subset", "superset", "identity", "coverage", "input"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                all_bases, all_cases, bases, cases, hashes, _ = save_run(root)
                reuse = ReuseResults(root)
                if corruption == "subset":
                    cases.pop()
                elif corruption == "superset":
                    cases.append(next(row for row in all_cases if row not in cases))
                elif corruption == "identity":
                    cases[0]["nuisance_kind"] = "contrast"
                elif corruption == "coverage":
                    cases[0]["visible_edit_pixel_count"] = 999
                else:
                    hashes["parent_run"] = "0" * 64
                with self.assertRaises(ValueError):
                    reuse.validate_selection(hashes, bases, cases)

    def test_loaded_artifacts_are_checked_again_after_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, cases, _, method = save_run(root)
            reuse = ReuseResults(root)
            (root / f"scores/{method}/{cases[0]['case_id']}.npy").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                reuse.load(method, cases[0]["case_id"])

    def test_path_escape_is_rejected_even_with_a_matching_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            save_run(root)
            outside = root.parent / "outside.npy"
            np.save(outside, np.zeros((2, 3), np.float32))
            metrics = read_json(root / "metrics.json")
            metrics["cases"][0].update(score_path="../outside.npy", score_sha256=sha256(outside))
            write_json(root / "metrics.json", metrics)
            record = read_json(root / "run.json")
            finish_record(root, record, "completed_exploratory")
            record = read_json(root / "run.json")
            record["artifact_sha256"]["../outside.npy"] = sha256(outside)
            write_json(root / "run.json", record)
            with self.assertRaisesRegex(ValueError, "escapes"):
                ReuseResults(root)


if __name__ == "__main__":
    unittest.main()
