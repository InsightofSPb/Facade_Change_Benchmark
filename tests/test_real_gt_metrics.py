"""Real temporal GT semantics and authenticated saved-map evaluation fixtures."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


from scripts.real_eval import real_gt_metrics as metrics

SCRIPT = Path(metrics.__file__)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(root):
    return {str(path.relative_to(root)): sha(path) for path in root.rglob("*") if path.is_file()}


def saved_map_fixture(directory):
    root, snapshot, review = (directory / name for name in ("run", "snapshot", "review"))
    for folder in (root / "maps", root / "units", snapshot, review):
        folder.mkdir(parents=True)
    frames, sequences, records, calibration, payloads = {}, {}, [], [], {}
    for number, building in enumerate(("A", "B")):
        sequence_id = f"real-{building.lower()}-x0-y0"
        sequences[sequence_id] = dict(sequence_id=sequence_id, building_id=building,
                                     frames=[dict(image_id=f"{building}{i}", year=2000+i) for i in range(3)])
        for index in range(3):
            packed = np.ones((3, 4), np.uint16)
            if index == 1 or (index == 2 and building == "B"):
                packed[0, 0] = 2
                packed[0, 1] = 3 if building == "A" else 1
                if building == "B":
                    packed[1, 0] = 2
            packed[2, 2:] = 0
            support = np.ones(packed.shape, bool)
            support[2, 3] = False
            frames[(sequence_id, index)] = dict(rgb=np.full((3, 4, 3), 10+40*number+10*index, np.uint8),
                                                support=support, packed=packed)
    methods = ("rgb_diff", "ssim", "dinov2", "msdzip_abs", "msdzip_mod256", "geoscd", "anychange")
    for method in methods:
        variant = "current_train" if method.startswith("msdzip_") else "frozen"
        modes = ("previous", "earliest", "history") if method in {"ssim", "dinov2", "msdzip_abs", "msdzip_mod256"} else ("previous", "earliest")
        for mode in modes:
            target = "earliest_state" if mode == "earliest" else "step_event"
            if method not in {"geoscd", "anychange"}:
                calibration.append(dict(method=method, variant=variant, mode=mode, target=target,
                                        status="calibrated", selection_split="val", selection_domain="synthetic",
                                        selection_arm="clean", choices=[dict(fpr_target=.01, threshold=.5),
                                                                      dict(fpr_target=.05, threshold=.8)]))
            for sequence_id, sequence in sequences.items():
                building = sequence["building_id"]
                for frame in (1, 2):
                    old = frames[(sequence_id, 0 if mode == "earliest" else frame - 1)]
                    new = frames[(sequence_id, frame)]
                    prediction = np.zeros((3, 4), bool)
                    prediction[0, 0] = True
                    if method != "geoscd":
                        prediction[0, 2] = True
                        if building == "A":
                            prediction[0, 1] = True
                    scores = np.where(prediction, .9, .1).astype(np.float32)
                    scores[~new["support"]] = np.nan
                    arrays = dict(reference=old["rgb"], source=new["rgb"], support=new["support"], scores=scores,
                                  labels=np.array([{"never": "use placeholder labels"}], object),
                                  memory__poison=np.array([{"never": "load model memory"}], object))
                    if method in {"geoscd", "anychange"}:
                        arrays["native_prediction"] = prediction
                        arrays["scores"][new["support"]] = 0  # Native mask remains authoritative.
                    identifier = f"{method}-{building}-{mode}-{frame}"
                    row = dict(unit_id=identifier, method=method, variant=variant, mode=mode, target=target,
                               domain="real", split="val", building_id=building, sequence_id=sequence_id,
                               frame=frame, status="completed", output_kind="native_mask" if method in {"geoscd", "anychange"} else "score")
                    if method == "geoscd" and building == "B" and mode == "previous" and frame == 2:
                        row.update(status="failed", error="No finite valid SAM correspondence similarities")
                    else:
                        path = root / "maps" / f"{identifier}.npz"
                        np.savez_compressed(path, **arrays)
                        row.update(artifact_path=str(path.relative_to(root)), artifact_sha256=sha(path))
                        payloads[identifier] = (row, arrays)
                    records.append(row)
                    (root / "units" / f"{identifier}.json").write_text(json.dumps(row))
    for domain, split in (("real", "test"), ("real", "train"), ("synthetic", "val")):
        records.append(dict(records[0], unit_id=f"FORBIDDEN-{domain}-{split}", domain=domain, split=split,
                            sequence_id=f"FORBIDDEN-{domain}-{split}", artifact_path="missing.npz"))
    (root / "resolved_config.json").write_text(json.dumps({}))
    (root / "experiment_contract.json").write_text(json.dumps({"inputs": {}}))
    (snapshot / "2026-10-17_source_units.json").write_text(json.dumps(records))
    (snapshot / "2026-10-17_quick_summary.json").write_text(json.dumps(dict(source_run=str(root), calibration=calibration)))
    (review / "summary.json").write_text(json.dumps(dict(selected_sequences=list(sequences))))
    (review / "provenance.json").write_text(json.dumps(dict(source_run=str(root), snapshot=str(snapshot))))

    def adapter(config, selected, contract):
        assert set(selected) == set(sequences)
        def loader(sequence, index):
            return frames[(sequence["sequence_id"], index)]
        return sequences, loader, dict(categories=[{"id": 3, "name": "surface"}, {"id": 8, "name": "damage"}])

    return root, snapshot, review, adapter, frames, payloads


class RealGTLabelTests(unittest.TestCase):
    def test_full_semantic_bitsets_and_unknown_pixels(self):
        old = np.array([[0, 1, 2, 3], [1, 1, 3, 2]], np.uint16)
        new = np.array([[1, 0, 2, 3], [2, 3, 1, 1]], np.uint16)
        support = np.ones(old.shape, bool)
        support[1, 3] = False
        labels = metrics.derive_labels(old, new, support, 2)
        np.testing.assert_array_equal(labels, [[255, 255, 0, 0], [1, 1, 1, 255]])
        counted = metrics.evaluate_mask(np.ones(old.shape, bool), labels)
        self.assertEqual(tuple(counted[key] for key in ("tp", "fp", "fn", "tn")), (3, 2, 0, 0))
        self.assertEqual((counted["valid_pixels"], counted["ignored_pixels"]), (5, 3))

    def test_repair_and_persistence_have_distinct_previous_and_earliest_targets(self):
        first = np.ones((3, 4), np.uint16)
        changed = np.full(first.shape, 2, np.uint16)
        repaired = first.copy()
        support = np.ones(first.shape, bool)
        previous_repair = metrics.derive_labels(changed, repaired, support, 2)
        history_repair = metrics.derive_labels(changed, repaired, support, 2)
        earliest_repair = metrics.derive_labels(first, repaired, support, 2)
        np.testing.assert_array_equal(previous_repair, np.ones(first.shape, np.uint8))
        np.testing.assert_array_equal(history_repair, previous_repair)
        np.testing.assert_array_equal(earliest_repair, np.zeros(first.shape, np.uint8))
        np.testing.assert_array_equal(metrics.derive_labels(changed, changed, support, 2), np.zeros(first.shape, np.uint8))
        np.testing.assert_array_equal(metrics.derive_labels(first, changed, support, 2), np.ones(first.shape, np.uint8))

    def test_tolerance_ignores_boundary_jitter_without_erasing_novel_thin_change(self):
        old = np.ones((7, 9), np.uint16)
        new = old.copy()
        old[2:5, 2:5] = 2
        new[2:5, 3:6] = 2
        new[0, 8] = 4
        support = np.ones(old.shape, bool)
        exact = metrics.derive_labels(old, new, support, 3)
        tolerant = metrics.derive_labels(old, new, support, 3, boundary_tolerance=1)
        self.assertEqual(int((exact == 1).sum()), 7)
        np.testing.assert_array_equal(tolerant[2:5, 2], [255, 255, 255])
        np.testing.assert_array_equal(tolerant[2:5, 5], [255, 255, 255])
        self.assertEqual(int(tolerant[0, 8]), 1)
        self.assertEqual(int((tolerant == 1).sum()), 1)
        self.assertTrue(np.all(tolerant[old == new] == 0))

    def test_existing_coco_rasterizer_clears_all_classes_under_crowd(self):
        repository = str(Path(__file__).resolve().parents[1])
        if repository not in sys.path:
            sys.path.insert(0, repository)
        from facade_change.segmentation.data import PreparedData
        from pycocotools import mask as mask_utils
        # Exercise actual COCO RLE decoding and repository multi-label composition.
        raster = PreparedData.__new__(PreparedData)
        raster.images, raster.metadata = {1: {"split": "val"}}, {}
        raster.category_to_channel = {3: 0, 8: 1}
        masks = {
            "surface": np.array([[1, 1, 1, 0], [0, 0, 0, 0]], bool),
            "damage": np.array([[0, 1, 1, 0], [0, 0, 0, 0]], bool),
            "crowd": np.array([[0, 1, 0, 0], [0, 0, 0, 0]], bool),
        }
        raster.annotations_by_image = {1: [dict(id=i, image_id=1, category_id=category, iscrowd=crowd,
                                               segmentation=mask_utils.encode(np.asfortranarray(masks[key].astype(np.uint8))))
                                           for i, (key, category, crowd) in enumerate(
                                               (("surface", 3, 0), ("damage", 8, 0), ("crowd", 8, 1)), 1)]}
        packed = raster._rasterize(1, 2, 4)
        np.testing.assert_array_equal(packed, [[1, 0, 3, 0], [0, 0, 0, 0]])
        labels = metrics.derive_labels(packed, np.ones(packed.shape, np.uint16), np.ones(packed.shape, bool), 2)
        self.assertEqual(labels[0, 1], 255, "Crowd is unknown, never a negative or a change")
        self.assertEqual(labels[1, 0], 255, "Unannotated pixels remain unknown")
        self.assertEqual(labels[0, 2], 1, "Removing one overlapping class is a semantic change")

    def test_default_prepared_coco_adapter_reuses_frozen_geometry_and_excludes_test(self):
        repository = Path(__file__).resolve().parents[1]
        if str(repository) not in sys.path:
            sys.path.insert(0, str(repository))
        source = repository / "tests/test_2026-10-14_temporal_inputs.py"
        source_spec = importlib.util.spec_from_file_location("real_gt_input_fixture", source)
        source_module = importlib.util.module_from_spec(source_spec)
        source_spec.loader.exec_module(source_module)
        case = source_module.TemporalInputsTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        from pycocotools import mask as mask_utils
        categories = [{"id": 1, "name": "surface"}, {"id": 2, "name": "damage"}]
        coco_path = case.dataset / "coco.json"
        coco = json.loads(coco_path.read_text())
        coco["categories"] = categories
        annotation_id = 0
        for position, image_id in enumerate((6, 7)):
            damage = np.zeros((32, 40), bool)
            if position:
                damage[5:10, 5:10] = True
            crowd = np.zeros(damage.shape, bool)
            crowd[1:3, 1:3] = True
            for mask, category, iscrowd in ((~damage, 1, 0), (damage, 2, 0), (crowd, 2, 1)):
                if not mask.any():
                    continue
                encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                encoded["counts"] = encoded["counts"].decode("ascii")
                annotation_id += 1
                coco["annotations"].append(dict(id=annotation_id, image_id=image_id, category_id=category,
                                                 segmentation=encoded, iscrowd=iscrowd))
        coco_path.write_text(json.dumps(coco))
        for filename in ("dataset.json", "manifest.json"):
            path = case.dataset / filename
            value = json.loads(path.read_text())
            value["categories"] = categories
            if filename == "dataset.json":
                value["source_coco_sha256"] = sha(coco_path)
            path.write_text(json.dumps(value))
        config = case.real_config()
        sequences, provenance, _ = source_module.inputs.load_sequences(config)
        selected = next(row for row in sequences if row["split"] == "val" and row["crop_xywh"][:2] == [0, 0])
        contract = dict(inputs={"files_sha256": provenance["files_sha256"]},
                        sequence_ids=[row["sequence_id"] for row in sequences])
        before = hashes(case.root)
        metadata, loader, actual = metrics.default_adapter(config, [selected["sequence_id"]], contract, repository, 0)
        first, second = (loader(metadata[selected["sequence_id"]], index) for index in (0, 1))
        labels = metrics.derive_labels(first["packed"], second["packed"], first["support"] & second["support"], 2)
        self.assertEqual(int((labels == 1).sum()), 25)
        self.assertTrue(np.all(labels[1:3, 1:3] == 255))
        np.testing.assert_array_equal(second["rgb"], source_module.inputs.load_frame(selected, 1)["rgb"])
        self.assertEqual(actual["prepared_data"]["selected_splits"], ["val"])
        self.assertFalse(actual["prepared_data"]["test_rgb_loaded"])
        self.assertFalse(actual["prepared_data"]["test_annotations_rasterized"])
        from facade_change.segmentation.data import PreparedData
        original_load = PreparedData.load_image
        with patch.object(PreparedData, "load_image", autospec=True, side_effect=original_load) as load:
            cached_sequences, cached_loader, _ = metrics.default_adapter(
                config, [selected["sequence_id"]], contract, repository, 0, decoded_cache_bytes=1024 * 1024)
            cached_first = cached_loader(cached_sequences[selected["sequence_id"]], 0)
            cached_again = cached_loader(cached_sequences[selected["sequence_id"]], 0)
            self.assertEqual(load.call_count, 1)
            for key in ("rgb", "support", "packed"):
                np.testing.assert_array_equal(cached_first[key], first[key])
                np.testing.assert_array_equal(cached_again[key], first[key])
        with patch.object(PreparedData, "load_image", autospec=True, side_effect=original_load) as load:
            small_sequences, small_loader, _ = metrics.default_adapter(
                config, [selected["sequence_id"]], contract, repository, 0, decoded_cache_bytes=1)
            small_loader(small_sequences[selected["sequence_id"]], 0)
            small_loader(small_sequences[selected["sequence_id"]], 0)
            self.assertEqual(load.call_count, 2, "Oversized frames must not be cached")
        self.assertEqual(hashes(case.root), before)
        geometry = Path(config["alignment_runs"][-1]) / "geometry.json"
        geometry.write_bytes(geometry.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            metrics.default_adapter(config, [selected["sequence_id"]], contract, repository, 0)


class RealGTSavedMapTests(unittest.TestCase):
    def test_cached_maps_targets_fusions_macros_and_read_only_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, snapshot, review, adapter, frames, payloads = saved_map_fixture(base)
            before = {path: hashes(path) for path in (root, snapshot, review)}
            frame_before = copy.deepcopy(frames)
            out = base / "metrics"
            with contextlib.redirect_stdout(io.StringIO()), patch("subprocess.Popen", side_effect=AssertionError("No inference allowed")):
                summary = metrics.run(root, snapshot, review, out, input_adapter=adapter)
            pairs = summary["per_pair"]
            def pair(predictor, building="A", frame=1, mode="previous", cohort="available"):
                return next(row for row in pairs if (row["predictor"], row["building_id"], row["frame"], row["mode"], row["cohort"]) ==
                            (predictor, building, frame, mode, cohort))
            ssim, geo = "ssim:frozen:fpr0.01", "geoscd:frozen:native"
            self.assertEqual(summary["thresholds"]["previous:" + ssim]["threshold"], .5)
            self.assertEqual(summary["thresholds"]["previous:ssim:frozen:fpr0.05"]["threshold"], .8)
            self.assertEqual(tuple(pair(ssim)[key] for key in ("tp", "fp", "fn", "tn")), (2, 1, 0, 7))
            self.assertEqual(tuple(pair(geo)[key] for key in ("tp", "fp", "fn", "tn")), (1, 0, 1, 8))
            self.assertEqual(tuple(pair(geo + "&" + ssim)[key] for key in ("tp", "fp", "fn", "tn")), (1, 0, 1, 8))
            self.assertEqual(tuple(pair(geo + "|" + ssim)[key] for key in ("tp", "fp", "fn", "tn")), (2, 1, 0, 7))
            self.assertEqual(pair(geo + "&" + ssim + "@history")["tp"], 1)
            for mode, expected_a, expected_b in (("previous", 2, 0), ("history", 2, 0), ("earliest", 0, 2)):
                self.assertEqual(pair(ssim, frame=2, mode=mode)["positive_pixels"], expected_a)
                self.assertEqual(pair(ssim, "B", frame=2, mode=mode)["positive_pixels"], expected_b)
            self.assertFalse(any(row["predictor"] == geo and row["mode"] == "previous" and row["building_id"] == "B" and row["frame"] == 2 for row in pairs))
            coverage = next(row for row in summary["coverage"] if row["mode"] == "previous" and row["predictor"] == geo)
            self.assertEqual((coverage["available_pairs"], coverage["expected_pairs"]), (3, 4))
            self.assertEqual(coverage["missing_pairs"], [["real-b-x0-y0", 2]])
            aggregate = {row["aggregation"]: row for row in summary["aggregate"] if
                         (row["cohort"], row["mode"], row["predictor"], row["subset"]) == ("available", "previous", ssim, "all")}
            self.assertAlmostEqual(aggregate["facade_pooled_macro"]["precision"], 11 / 24)
            self.assertAlmostEqual(aggregate["facade_pooled_macro"]["f1"], 17 / 30)
            self.assertAlmostEqual(aggregate["frame_then_facade_macro"]["f1"], .525)
            self.assertEqual(aggregate["frame_then_facade_macro"]["units"], 2)
            comparison = next(row for row in summary["history_comparison"] if row["predictor"] == ssim and row["subset"] == "all")
            self.assertEqual((comparison["evaluated_pairs"], comparison["valid_pixels"], comparison["delta_f1"]), (4, 40, 0))
            self.assertTrue(all(row["building_id"] in {"A", "B"} for row in pairs))
            self.assertIn("PREVIOUS", (out / "tables.md").read_text())
            self.assertIn("EARLIEST", (out / "tables.md").read_text())
            self.assertIn("HISTORY", (out / "tables.md").read_text())
            self.assertTrue(json.loads((out / "provenance.json").read_text())["no_model_inference"])
            self.assertEqual({path: hashes(path) for path in before}, before)
            for key in frames:
                for field in frames[key]:
                    np.testing.assert_array_equal(frames[key][field], frame_before[key][field])
            with self.assertRaises(FileExistsError):
                metrics.run(root, snapshot, review, out, input_adapter=adapter)

    def test_geoscd_overlay_authentication_and_restored_rgb_match(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, snapshot, review, adapter, frames, payloads = saved_map_fixture(base)
            overlay = base / "geo_retry"
            (overlay / "units").mkdir(parents=True)
            (overlay / "maps").mkdir()
            manifest = dict(source_run=str(root), snapshot_source_units_sha256=sha(snapshot / "2026-10-17_source_units.json"),
                            snapshot_quick_summary_sha256=sha(snapshot / "2026-10-17_quick_summary.json"))
            (overlay / "run.json").write_text(json.dumps(manifest))
            record, arrays = copy.deepcopy(payloads["geoscd-B-earliest-2"])
            record.update(unit_id="retry-geoscd-B-previous-2", mode="previous", target="step_event")
            arrays["reference"] = frames[("real-b-x0-y0", 1)]["rgb"].copy()
            path = overlay / "maps" / "retry.npz"
            np.savez_compressed(path, **arrays)
            record.update(artifact_path="maps/retry.npz", artifact_sha256=sha(path))
            marker = overlay / "units" / "retry.json"
            marker.write_text(json.dumps(record))
            before = {item: hashes(item) for item in (root, snapshot, review, overlay)}
            with contextlib.redirect_stdout(io.StringIO()), patch("subprocess.Popen", side_effect=AssertionError("No inference allowed")):
                summary = metrics.run(root, snapshot, review, base / "metrics", geoscd_run=overlay, input_adapter=adapter)
            coverage = next(row for row in summary["coverage"] if row["mode"] == "previous" and row["predictor"] == "geoscd:frozen:native")
            self.assertEqual((coverage["available_pairs"], coverage["common_pairs"]), (4, 4))
            completed = next(row for row in summary["per_pair"] if (row["predictor"], row["building_id"], row["mode"], row["frame"], row["cohort"]) ==
                             ("geoscd:frozen:native", "B", "previous", 2, "available"))
            self.assertEqual((completed["tp"], completed["fp"], completed["fn"], completed["tn"]), (0, 1, 0, 9))
            self.assertEqual(completed["source_unit_ids"], [record["unit_id"]])
            self.assertEqual({item: hashes(item) for item in before}, before)
            # Authentic cached bytes alone are insufficient: RGB must match GT geometry.
            arrays["reference"][0, 0, 0] += 1
            np.savez_compressed(path, **arrays)
            record["artifact_sha256"] = sha(path)
            marker.write_text(json.dumps(record))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "RGB/support"):
                metrics.run(root, snapshot, review, base / "bad_rgb", geoscd_run=overlay, input_adapter=adapter)
            # A changed map without a matching digest must also be rejected.
            path.write_bytes(path.read_bytes() + b"tampered")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                metrics.run(root, snapshot, review, base / "bad_hash", geoscd_run=overlay, input_adapter=adapter)


if __name__ == "__main__":
    unittest.main()
