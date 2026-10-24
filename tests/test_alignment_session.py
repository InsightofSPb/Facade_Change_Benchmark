"""Completed incremental sessions -> native crops, histories, benchmark, atlas."""
import contextlib
import csv
import importlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from facade_change import cli
from facade_change.alignment_session import load_session
from facade_change.crop_dataset import prepare_crop_dataset
from facade_change.geoscd import _save_pair, dense_warp
from facade_change.io import finish_record, load_rgb, read_json, run_record, sha256, write_json
from facade_change.saved_geometry import FrameSampler
from scripts.real_eval.night_prepare import audit, prepare
from test_real_temporal_prepare import fixture


def session_fixture(root, dense=False):
    data = fixture(root)
    manifest_path = data["dataset"] / "manifest.json"
    manifest = read_json(manifest_path)
    for row in manifest["images"]:
        row.update(image_status="ready", gold_member=True)
    for category, color in zip(manifest["categories"], ("#aaaaaa", "#ff0000")):
        category["color"] = color
    coco_path = data["dataset"] / "coco.json"
    coco = read_json(coco_path)
    coco["categories"] = manifest["categories"]
    write_json(coco_path, coco)
    config_path = data["dataset"] / "dataset.json"
    config = read_json(config_path)
    config.update(categories=manifest["categories"], source_coco_sha256=sha256(coco_path))
    write_json(config_path, config)
    images = {str(row["image_id"]): row for row in manifest["images"]}
    for row in manifest["pairs"]:
        row.update(reference_year=images[str(row["reference_id"])]["year"],
                   source_year=images[str(row["source_id"])]["year"])
    write_json(manifest_path, manifest)
    split = read_json(data["dataset"] / "split.json")
    split.update(mode="reviewed", development_only=False)
    write_json(data["dataset"] / "split.json", split)
    session = root / "aligned/sessions/report-1"
    session.mkdir(parents=True)
    rows, reviews = [], []
    for index, (pair, value) in enumerate(zip(manifest["pairs"], data["runs"])):
        path = Path(value)
        reference, source = (images[str(pair[k])] for k in ("reference_id", "source_id"))
        method = "geoscd" if dense and index % 2 == 1 else "sift"
        state = "computed_needs_review" if method == "geoscd" else "passed"
        decision = "accept" if method == "geoscd" else ""
        record = run_record("geoscd_geometry_pair" if method == "geoscd" else "pair_alignment",
                            dict(method=method, manifest_sha256=sha256(manifest_path),
                                 reference_id=str(reference["image_id"]), source_id=str(source["image_id"])))
        record["observations"] = dict(reference=reference, source=source)
        if method == "geoscd":
            yy, xx = np.mgrid[:518, :518]
            grid = np.stack((xx, yy), axis=-1).astype(np.float32)
            prediction = {"reference_to_source": grid, "source_to_reference": grid,
                          "reference_projected_z": np.ones((518, 518)),
                          "source_projected_z": np.ones((518, 518)),
                          "reference_occlusion": np.ones((518, 518), bool),
                          "source_occlusion": np.ones((518, 518), bool)}
            _save_pair(path, reference, source, (*load_rgb(reference["image_path"]),
                       *load_rgb(source["image_path"])), prediction, 0.)
        else:
            record["quality_gate"] = {"passed": True}
            write_json(path / "geometry.json", {"reference_to_canvas": np.eye(3).tolist(),
                       "source_to_reference": np.eye(3).tolist(), "width": 24, "height": 16})
        finish_record(path, record, "completed_needs_review" if method == "geoscd" else "completed")
        rows.append({**pair, "selected_method": method,
                     "selection_status": "reviewed" if decision else "provisional_gate_pass",
                     "methods": {method: {"state": state, "review": decision, "path": str(path), "reused": True}}})
        reviews.append(dict(pair_id=pair["pair_id"], method=method,
                            reference_sha256=reference["sha256"], source_sha256=source["sha256"],
                            alignment_run_sha256=sha256(path / "run.json"), state=state, decision=decision, notes=""))
    write_json(session / "inputs.json", {"manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
                                         "images": manifest["images"], "preparation": {}})
    write_json(session / "pairs.json", rows)
    with (session / "review.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reviews[0]))
        writer.writeheader()
        writer.writerows(reviews)
    write_json(session / "summary.json", {"pairs": len(rows), "methods": {"sift": {}, "geoscd": {}}, "plan_only": False})
    record = run_record("incremental_alignment_preparation", {"manifest_sha256": sha256(manifest_path), "plan_only": False})
    finish_record(session, record, "completed_needs_review")
    return data, session


class AlignmentSessionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data, self.session = session_fixture(self.root, dense=True)
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def rehash(self):
        finish_record(self.session, read_json(self.session / "run.json"), "completed_needs_review")

    def test_mixed_session_to_crops_synthetic_benchmark_without_rematching(self):
        before = {p: sha256(p) for p in self.root.rglob("*") if p.is_file()}
        out = self.root / "crops"
        with patch("facade_change.pipeline.SIFTMatcher", side_effect=AssertionError("rematching")), \
                patch("facade_change.geoscd.OfficialGeometry", side_effect=AssertionError("model loaded")):
            self.assertEqual(cli.main(["workflow", "prepare", "crops", "--alignment-session", str(self.root / "aligned"),
                                       "--tile-size", "16", "--stride", "16", "--out", str(out)]), 0)
        summary = read_json(out / "summary.json")
        self.assertEqual(summary["pair_method_counts"], {"sift": 4, "geoscd": 4})
        self.assertEqual(summary["derivative_failures"], 0)
        index = read_json(out / summary["index_path"])
        for crop in index["crops"]:
            x, y, x1, y1 = crop["bbox_canvas_xyxy"]
            original, _ = load_rgb(crop["original_source_path"])
            actual, _ = load_rgb(out / crop["source_rgb"])
            np.testing.assert_array_equal(actual, original[y:y1, x:x1])
            if crop["alignment_method"] == "geoscd":
                self.assertIsNone(crop["source_to_crop"])
                self.assertEqual(crop["alignment_selection"]["review"], "accept")
        config = self.root / "controls.json"
        write_json(config, {"selection": {"splits": ["train", "val", "test"], "crops_per_building": 1},
                            "states": ["unchanged", "crack"], "scenarios": [{"id": "clean", "kind": "identity"}]})
        self.assertEqual(cli.main(["workflow", "prepare", "synthetic", "--crop-run", str(out),
                                   "--config", str(config), "--out", str(self.root / "synthetic")]), 0)
        self.assertEqual(cli.main(["workflow", "benchmark", "synthetic", "--dataset-run", str(self.root / "synthetic"),
                                   "--methods", "rgb_diff", "--out", str(self.root / "benchmark")]), 0)
        self.assertTrue((self.root / "benchmark/gallery.html").is_file())
        self.assertEqual({p: sha256(p) for p in before}, before)

    def test_real_histories_preserve_native_arrays_and_work_with_atlas_and_runner(self):
        write_json(self.data["snapshot"] / "2026-10-17_quick_summary.json", {
            "source_run": str(self.data["run"]), "calibration": [{"method": "dinov2", "variant": "frozen",
                "mode": "previous", "target": "step_event", "status": "calibrated",
                "selection_split": "val", "selection_domain": "synthetic", "selection_arm": "clean",
                "choices": [{"fpr_target": .01, "threshold": 1.}]}]})
        baseline = audit(self.data["run"], self.data["snapshot"], self.root / "audit", alignment_session=self.session)
        self.assertEqual(baseline["coverage"]["accepted_pairs"], 8)
        for sequence in baseline["sequences"]:
            self.assertEqual([f["year"] for f in sequence["frames"]], [2000, 2001, 2002])
            self.assertIn("dense_geometry", sequence["frames"][2])
            self.assertNotIn("source_to_reference", sequence["frames"][2])
        prepared = self.root / "real_histories"
        document = prepare(self.root / "audit", prepared, max_facades=3, tile_size=16, stride=16, min_known_pixels=4)
        self.assertEqual(document["coverage"]["frames"], 9)
        for sequence in document["sequences"]:
            x, y, w, h = sequence["crop_xywh"]
            for frame in sequence["frames"]:
                original = next(r for r in self.data["images"] if r["image_id"] == frame["image_id"])
                with np.load(prepared / frame["artifact_path"], allow_pickle=False) as arrays:
                    np.testing.assert_array_equal(arrays["rgb"], load_rgb(original["image_path"])[0][y:y+h, x:x+w])
        # Existing frozen real runner; only the model capability is substituted.
        from scripts.real_eval import night_run
        from test_real_temporal_run import Factory, metadata
        provenance = self.data["run"] / "method_provenance"
        provenance.mkdir()
        for method in ("rgb_diff", "dinov2"):
            write_json(provenance / f"full_{method}.json", metadata(method))
        factory = Factory()
        with patch.object(night_run, "_snapshot_gpu"):
            result = night_run.run(prepared, self.root / "real_benchmark", methods=("rgb_diff", "dinov2"),
                                    pool_factory=factory)
        self.assertEqual(result["status"], "completed", result.get("method_failures"))
        self.assertEqual({r["mode"] for r in result["results"]}, {"previous", "earliest", "history"})
        from facade_change.gallery import history_bundle, atlas, atlas_export
        bundle = self.root / "histories.tar.gz"
        history_bundle.collect(prepared, bundle, include_originals=True)
        atlas.extract_bundle(bundle, self.root / "unpacked")
        atlas_export.export(self.root / "unpacked", self.root / "atlas", min_pixels=1)
        self.assertTrue((self.root / "atlas/data/2026-10-18-facades.json").is_file())

    def test_session_rejections_are_authoritative_and_rankings_do_not_change_selection(self):
        rows = read_json(self.session / "pairs.json")
        rows[0].update(selected_method=None, selection_status="unresolved")
        write_json(self.session / "pairs.json", rows)
        quality = self.session / "quality/report-1"
        quality.mkdir(parents=True)
        qrecord = run_record("alignment_quality_diagnostics", {
            "session_sha256": {name: sha256(self.session / name) for name in ("pairs.json", "inputs.json")}})
        for name in ("protocol", "quality_summary", "quality_cycles", "quality_drift"):
            write_json(quality / (name + ".json"), {})
        write_json(quality / "quality_ranking.json", [{"pair_id": row["pair_id"], "advisory_order": ["loftr"]} for row in rows])
        finish_record(quality, qrecord, "completed_exploratory")
        summary = read_json(self.session / "summary.json")
        summary.update(quality_report=str(quality), quality_run_sha256=sha256(quality / "run.json"))
        write_json(self.session / "summary.json", summary)
        self.rehash()
        summary = prepare_crop_dataset(out=self.root / "crops", alignment_session=self.session, tile_size=16, stride=16)
        self.assertEqual(summary["exported_pair_count"], 7)
        self.assertEqual(summary["pair_method_counts"], {"sift": 3, "geoscd": 4})
        excluded = read_json(self.root / "crops" / summary["index_path"])["excluded_pairs"]
        self.assertEqual(excluded[0]["reason"], "unresolved_session_selection")
        with self.assertRaisesRegex(ValueError, "authoritative"):
            prepare_crop_dataset(out=self.root / "other", alignment_session=self.session, methods=["sift"])
        (quality / "quality_ranking.json").write_text("[]")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_session(self.session)

    def test_unlabelled_temporal_loader_uses_session_geometry_without_semantic_gt(self):
        inputs = importlib.import_module("facade_change.research.2026-10-14_temporal_inputs")
        config = {"dataset": str(self.data["dataset"]), "alignment_runs": [str(self.session)],
                  "real_tile_size": 16, "real_stride": 16}
        with patch("facade_change.segmentation.data.PreparedData._rasterize", side_effect=AssertionError("GT read")):
            sequences, provenance, _ = inputs.load_sequences(config)
            self.assertTrue(sequences)
            self.assertFalse(provenance["semantic_labels_loaded"])
            for sequence in sequences:
                self.assertIn(sequence["split"], ("train", "val"))
                for index in range(len(sequence["frames"])):
                    frame = inputs.load_frame(sequence, index)
                    self.assertTrue(np.all(frame["event_labels"] == 255))

    def test_missing_first_anchor_pair_is_not_replaced_by_a_chain(self):
        rows = read_json(self.session / "pairs.json")
        # Keep the direct 2001->2000 edge, but explicitly exclude 2002->2000.
        rows[1].update(selected_method=None, selection_status="unresolved")
        write_json(self.session / "pairs.json", rows)
        self.rehash()
        report = audit(self.data["run"], self.data["snapshot"], self.root / "audit", alignment_session=self.session)
        sequence = next(row for row in report["sequences"] if row["building_id"] == "train-wide")
        self.assertEqual([frame["year"] for frame in sequence["frames"]], [2000, 2001])
        self.assertEqual(sequence["missing_years"], [2002])

    def test_split_leakage_and_metadata_changes_are_rejected(self):
        inputs = read_json(self.session / "inputs.json")
        inputs["images"][0]["split"] = "test"
        write_json(self.session / "inputs.json", inputs)
        self.rehash()
        with self.assertRaisesRegex(ValueError, "identity differs"):
            load_session(self.session)

    def test_running_sessions_and_ambiguous_cache_roots_are_refused(self):
        record = read_json(self.session / "run.json")
        record["status"] = "running"
        write_json(self.session / "run.json", record)
        with self.assertRaisesRegex(ValueError, "not completed"):
            load_session(self.session)
        second = self.session.parent / "report-2"
        second.mkdir()
        write_json(second / "run.json", record)
        with self.assertRaisesRegex(ValueError, "found 2"):
            load_session(self.root / "aligned")

    def test_hash_bound_selection_and_dense_review_cannot_be_bypassed(self):
        pair = Path(self.data["runs"][1]) / "run.json"
        pair.write_bytes(pair.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_session(self.session)
        pair.write_bytes(pair.read_bytes()[:-1])
        rows = read_json(self.session / "pairs.json")
        rows[1]["methods"]["geoscd"]["review"] = ""
        rows[1]["selection_status"] = "provisional_gate_pass"
        write_json(self.session / "pairs.json", rows)
        self.rehash()
        with self.assertRaisesRegex(ValueError, "hash-bound review"):
            load_session(self.session)

    def test_cache_reuse_accepts_old_ids_paths_but_not_changed_split_or_geometry(self):
        pair = Path(self.data["runs"][0]) / "run.json"
        record = read_json(pair)
        for row in record["observations"].values():
            row["image_id"] = "old-" + str(row["image_id"])
            row["image_path"] = "/old/location.png"
        write_json(pair, record)
        with (self.session / "review.csv").open(newline="") as stream:
            reviews = list(csv.DictReader(stream))
        reviews[0]["alignment_run_sha256"] = sha256(pair)
        with (self.session / "review.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(reviews[0]))
            writer.writeheader()
            writer.writerows(reviews)
        self.rehash()
        summary = prepare_crop_dataset(out=self.root / "crops", alignment_session=self.session, tile_size=16, stride=16)
        self.assertEqual(summary["derivative_failures"], 0)
        geometry = pair.parent / "geometry.json"
        geometry.write_bytes(geometry.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_session(self.session)

    def test_dense_sampler_matches_existing_geometry_on_translation_alpha_and_black(self):
        session = load_session(self.session)
        from facade_change.saved_geometry import frame_geometry
        frame = frame_geometry(session.selected["1-3"], {})
        info = frame["dense_geometry"]
        yy, xx = np.mgrid[:518, :518]
        coordinates = np.stack((xx, yy), axis=-1).astype(np.float32) + [518/24, 0]
        valid = np.ones((518, 518), bool)
        field = self.root / "translated.npz"
        np.savez_compressed(field, reference_to_source=coordinates, reference_projection_valid=valid)
        sampler = FrameSampler({"dense_geometry": {**info, "path": str(field), "sha256": sha256(field)}})
        rgb = np.zeros((16, 24, 3), np.uint8)
        rgb[..., 0] = np.arange(24)
        opaque = np.ones((16, 24), bool)
        opaque[4, 5] = False
        expected = dense_warp(rgb, np.ones_like(opaque), rgb, opaque, coordinates, valid)
        np.testing.assert_array_equal(sampler.warp(rgb, (0, 0, 24, 16)), expected["source"])
        np.testing.assert_array_equal(sampler.support(opaque, (0, 0, 24, 16)), expected["source_support"])
        self.assertFalse(sampler.support(opaque, (0, 0, 24, 16))[:, -1].any())


if __name__ == "__main__":
    unittest.main()
