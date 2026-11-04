"""Canonical history integration; procedural pixels are not real-model evidence."""
from argparse import Namespace
import contextlib
import importlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from facade_change import workflow
from facade_change.io import read_json, sha256, write_json


runner = importlib.import_module("facade_change.research.2026-11-04_history_benchmark")
data_module, metrics = runner.data_module, runner.metrics


def fixture(root):
    histories = root / "histories"
    report = histories / "reports/prepare-1"
    report.mkdir(parents=True)
    categories = [{"id": 3, "name": "ORNAMENT_INTACT"}, {"id": 9, "name": "CRACK"}]
    images, annotations, manifest_images, tracks = [], [], [], []
    for part_index, split in enumerate(("train", "val", "test")):
        for building in range(2):
            track = dict(track_id=f"{split}_{building}_track", building_id=f"{split}_{building}",
                         view_id=f"{split}_{building}_front", split=split, bbox_xywh=[0, 0, 224, 224],
                         distinct_years=3 + building * 2, observations=[])
            for frame in range(track["distinct_years"]):
                identifier = len(images) + 1
                path = histories / f"rgb-{identifier}.png"
                rgb = np.full((224, 224, 3), 30 + part_index, np.uint8)
                if frame == 2:
                    rgb[30:160, 30:160] = 230
                rgb[0, 0] = [part_index, building, frame]
                Image.fromarray(rgb).save(path)
                support = histories / f"support-{identifier}.png"
                geometric = np.full((224, 224), 255, np.uint8)
                if frame == 0:
                    geometric[:4] = 0
                Image.fromarray(geometric).save(support)
                images.append(dict(id=identifier, file_name=path.name, width=224, height=224))
                image = dict(image_id=identifier, image_path=str(path), sha256=sha256(path), width=224,
                             height=224, year=2010 + frame * 3, metadata_status="reviewed",
                             **{k: track[k] for k in ("building_id", "view_id", "split")})
                manifest_images.append(image)
                annotations.append(dict(id=len(annotations) + 1, image_id=identifier, category_id=3,
                                        segmentation=[[10, 10, 210, 10, 210, 210, 10, 210]]))
                if frame == 2:
                    annotations.append(dict(id=len(annotations) + 1, image_id=identifier, category_id=9,
                                            segmentation=[[30, 30, 160, 30, 160, 160, 30, 160]]))
                track["observations"].append(dict(observation_id=f"obs-{identifier}", image_id=identifier,
                    year=image["year"], image_sha256=image["sha256"], original_image_path=str(path),
                    rgb=path.name, rgb_sha256=sha256(path), support=support.name,
                    support_sha256=sha256(support), valid_fraction=float((geometric > 0).mean()),
                    alignment={"matrix": np.eye(3).tolist()}))
            tracks.append(track)
    coco = root / "coco.json"
    write_json(coco, dict(images=images, categories=categories, annotations=annotations))
    manifest = root / "annotations.json"
    write_json(manifest, dict(source=dict(coco_json=str(coco), coco_sha256=sha256(coco)),
                             images=manifest_images, categories=categories))
    document = dict(kind="canonical_crop_histories", schema_version=1, tracks=tracks, report="reports/prepare-1",
                    options=dict(tile_size=224, stride=192, min_valid_fraction=.8),
                    views=[dict(frames=[dict(image=row) for row in manifest_images])],
                    leakage_check=dict(status="passed", building_assignments={t["building_id"]: t["split"] for t in tracks}))
    write_json(histories / "manifest.json", document)
    write_json(report / "manifest.json", document)
    write_json(report / "summary.json", dict(limited=False, exclusion_reasons={}))
    write_json(report / "run.json", dict(status="completed", manifest_sha256=sha256(histories / "manifest.json")))
    groups = root / "groups.json"
    write_json(groups, dict(groups=[dict(name="structural_damage", categories=["CRACK"])]))
    return Namespace(histories=str(histories), annotation_manifest=str(manifest), class_groups=str(groups),
                     expected_sha256=sha256(histories / "manifest.json"), expected_candidates=12,
                     boundary_tolerance=0, out=str(root / "prepared"), resume=False)


class FakeScorer:
    """Exercises seven adapter routes; never represents their real weights/quality."""
    fail_at = None
    calls = 0

    def __init__(self, method, *args, **kwargs):
        self.method = method
        self.metadata = dict(method=method, output_kind="native_mask" if method == "anychange" else "score",
                             settings={"procedural_test_double": True})
        self.native_prediction = self.raw_scores = None
        self.auxiliary_arrays = {}

    def activate(self):
        return self.metadata

    def __call__(self, a, b, support):
        FakeScorer.calls += 1
        if FakeScorer.calls == FakeScorer.fail_at:
            raise RuntimeError("injected interruption")
        score = np.abs(a.astype(np.float32) - b).mean(axis=2) / 255
        if self.method == "anychange":
            self.native_prediction = (score > .2) & support
            score = self.native_prediction.astype(np.float32)
        score[~support] = np.nan
        return score

    def close(self):
        pass


def arguments(root, pilot=False, methods=None):
    return Namespace(command="pilot" if pilot else "run", prepared=str(root / "prepared"),
        out=str(root / ("pilot" if pilot else "full")), methods=methods or ["rgb_diff", "ssim"],
        modes=list(runner.MODES), history_methods=None, device="cpu", trust_checkpoint=False,
        methods_config=None, pilot_tracks=1, max_hours=None, geoscd_run=None, no_report=False,
        reuse_run=None, resume=False)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prepare_args = fixture(self.root)
        with contextlib.redirect_stdout(io.StringIO()):
            data_module.prepare(self.prepare_args)
        FakeScorer.calls, FakeScorer.fail_at = 0, None

    def run_quiet(self, args, fake=False):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.ExitStack() as stack:
            if fake:
                stack.enter_context(mock.patch.object(runner, "make_scorer", FakeScorer))
                stack.enter_context(mock.patch.object(runner, "RemoteScorer", FakeScorer))
            return runner.run(args)

    def test_actual_rgb_ssim_full_pipeline_and_cpu_report(self):
        state = self.run_quiet(arguments(self.root))
        self.assertEqual(state["status"], "completed")
        out = self.root / "full"
        summary = read_json(out / "summary.json")
        self.assertEqual(len(summary["results"]), 2 * 3 * 3)
        thresholds = read_json(out / "thresholds.json")
        for method in thresholds.values():
            self.assertEqual(method["previous"]["selection_split"], "val")
            self.assertEqual(method["history"]["memory_threshold"], method["previous"]["threshold"])
        self.assertTrue((out / "gallery.html").exists())
        with mock.patch.object(runner, "make_scorer", side_effect=AssertionError("report must not infer")):
            runner.report(out)
        self.assertTrue((out / "reports/metrics.csv").exists())
        data = data_module.Dataset(self.root / "prepared")
        sample = data.samples[0]
        target = data.target(sample["event_id"])
        self.assertTrue((target["target"][:10] == 255).all())
        self.assertFalse(target["support"][:4].any())

    def test_seven_routes_native_history_and_pilot_is_not_calibration(self):
        args = arguments(self.root, pilot=True, methods=list(runner.METHODS))
        self.run_quiet(args, fake=True)
        out = self.root / "pilot"
        summary = read_json(out / "summary.json")
        self.assertEqual(len(summary["results"]), 7 * 3 * 2)
        self.assertEqual(len(summary["fusion_results"]), 3 * 3 * 2)
        self.assertEqual(summary["counts"]["test"]["candidates"], 0)
        for modes in read_json(out / "thresholds.json").values():
            for threshold in modes.values():
                self.assertIsNone(threshold["selection_split"])
        maps = read_json(out / "maps_index.json")
        for record in maps["anychange"]["history"].values():
            _, arrays = runner.read_unit(out, record["unit_id"])
            self.assertTrue(np.isin(arrays["scores"][arrays["support"]], [0, 1]).all())

    def test_interruption_resume_and_cross_run_pair_reuse(self):
        args = arguments(self.root, methods=["rgb_diff"])
        FakeScorer.fail_at = 7
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.run_quiet(args, fake=True)
        self.assertEqual(read_json(self.root / "full/run.json")["status"], "failed")
        saved = {p.name: sha256(p) for p in (self.root / "full/units").glob("*.json")}
        args.resume, FakeScorer.fail_at = True, None
        self.run_quiet(args, fake=True)
        self.assertTrue(all(sha256(self.root / "full/units" / name) == value for name, value in saved.items()))
        args.out, args.resume, args.reuse_run = str(self.root / "reuse"), False, str(self.root / "full")
        before = FakeScorer.calls
        self.run_quiet(args, fake=True)
        self.assertEqual(before, FakeScorer.calls)
        self.assertEqual(read_json(self.root / "full/summary.json")["results"],
                         read_json(self.root / "reuse/summary.json")["results"])

    def test_modified_inputs_rejected_before_inference(self):
        index = read_json(self.root / "prepared/index.json")
        row = index["tracks"][0]["observations"][0]
        (self.root / "histories" / row["rgb"]).write_bytes(b"corruption")
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            self.run_quiet(arguments(self.root))

    def test_annotation_projection_rejects_mismatched_source_grid(self):
        manifest = read_json(self.prepare_args.annotation_manifest)
        coco_path = Path(manifest["source"]["coco_json"])
        coco = read_json(coco_path)
        manifest["images"][0]["width"] = coco["images"][0]["width"] = 223
        write_json(coco_path, coco)
        manifest["source"]["coco_sha256"] = sha256(coco_path)
        write_json(self.prepare_args.annotation_manifest, manifest)
        self.prepare_args.out = str(self.root / "wrong-grid")
        with self.assertRaisesRegex(ValueError, "source image dimensions"):
            data_module.prepare(self.prepare_args)

    def test_history_state_resume_restores_completed_prefix(self):
        args = arguments(self.root, methods=["rgb_diff"])
        original = runner.TemporalMemory.score_then_update
        calls = []

        def interrupt(memory, *values):
            calls.append(int(memory.state_dict()["next_frame"]))
            if len(calls) == 4:
                raise RuntimeError("stop inside history")
            return original(memory, *values)

        with mock.patch.object(runner.TemporalMemory, "score_then_update", interrupt):
            with self.assertRaisesRegex(RuntimeError, "inside history"):
                self.run_quiet(args, fake=True)
        preserved = {p.name: sha256(p) for p in (self.root / "full/units").glob("history-*.json")}
        self.assertEqual(len(preserved), 3)
        args.resume = True
        remaining = []

        def record(memory, *values):
            remaining.append(1)
            return original(memory, *values)

        with mock.patch.object(runner.TemporalMemory, "score_then_update", record):
            self.run_quiet(args, fake=True)
        self.assertEqual(len(remaining), 24 - 3)
        self.assertTrue(all(sha256(self.root / "full/units" / name) == value for name, value in preserved.items()))

    def test_full_seven_routes_freeze_val_before_other_split_metrics(self):
        collected = []
        collect, calibrate = metrics.collect, metrics.calibrate

        def tracked_collect(data, samples, *args, **kwargs):
            collected.append({s["split"] for s in samples})
            return collect(data, samples, *args, **kwargs)

        def tracked_calibrate(histograms):
            self.assertEqual(collected[-1], {"val"})
            return calibrate(histograms)

        with mock.patch.object(metrics, "collect", side_effect=tracked_collect), \
                mock.patch.object(metrics, "calibrate", side_effect=tracked_calibrate):
            self.run_quiet(arguments(self.root, methods=list(runner.METHODS)), fake=True)
        summary = read_json(self.root / "full/summary.json")
        self.assertEqual(len(summary["results"]), 63)
        self.assertEqual(len(summary["fusion_results"]), 27)
        baselines = {(r["mode"], r["split"]): r for r in summary["results"] if r["method"] == "anychange"}
        for result in summary["fusion_results"]:
            base = baselines[result["mode"], result["split"]]["metrics"]["macro"][0]
            filtered = result["metrics"]["macro"][0]
            self.assertLessEqual(filtered["fp"], base["fp"])
            self.assertEqual(filtered["known_pixels"], base["known_pixels"])

    def test_cli_routes_canonical_pilot(self):
        with mock.patch.object(runner, "run", return_value={"status": "completed"}) as run:
            workflow.main(["benchmark", "pilot", "--prepared", str(self.root / "prepared"),
                           "--out", str(self.root / "cli"), "--methods", "rgb_diff", "--device", "cpu"])
        self.assertEqual(run.call_args.args[0].command, "pilot")


class ContractTests(unittest.TestCase):
    def test_same_year_selection_and_candidate_count(self):
        base = dict(year=2000, image_id="b", image_sha256="x", valid_fraction=.9, observation_id="b")
        rows = [base, {**base, "image_id": "a", "observation_id": "a"},
                {**base, "image_id": "z", "observation_id": "z", "valid_fraction": .95},
                {**base, "year": 2003, "observation_id": "later"},
                {**base, "year": 2007, "observation_id": "last"}]
        tracks = data_module.select_tracks(dict(tracks=[dict(track_id="a", building_id="a", split="train",
                                                             observations=rows)]))
        self.assertEqual(tracks[0]["observations"][0]["observation_id"], "z")
        self.assertEqual(data_module.counts(tracks)["all"]["candidates"], 1)

    def test_pair_cache_is_support_sensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            store = data_module.Store(directory + "/run", {})
            self.addCleanup(store.close)
            scorer = FakeScorer("rgb_diff")
            cache = runner.PairCache(store, scorer, "rgb_diff", "fixture", None)
            rgb = np.zeros((8, 8, 3), np.uint8)
            support = np.ones((8, 8), bool)
            a, _ = cache.score(rgb, rgb, support)
            support[0] = False
            b, _ = cache.score(rgb, rgb, support)
            self.assertNotEqual(a["unit_id"], b["unit_id"])
            self.assertEqual(cache.calls["previous"]["computed"], 2)

    def test_history_reset_handles_state_return_and_future_independence(self):
        memory = runner.TemporalMemory(3)
        support = np.ones((4, 4), bool)
        scorer = FakeScorer("anychange")
        outputs = []
        for value in (0, 0, 255, 0):
            outputs.append(memory.score_then_update(np.full((4, 4, 3), value, np.uint8), support, scorer, .5))
        self.assertFalse(outputs[1]["accepted_change"].any())
        self.assertTrue(outputs[2]["accepted_change"].all())
        self.assertTrue(outputs[3]["accepted_change"].all())
        old = outputs[2]["scores"].copy()
        memory.score_then_update(np.full((4, 4, 3), 123, np.uint8), support, scorer, .5)
        np.testing.assert_array_equal(outputs[2]["scores"], old)

    def test_threshold_strict_equality_and_undefined_metrics(self):
        h = np.zeros((2, metrics.BINS), np.int64)
        h[0, 1024], h[1, 1025] = 10, 4
        row = metrics.metrics(h, .5)
        self.assertEqual((row["tp"], row["fp"], row["tn"]), (4, 0, 10))
        self.assertIsNone(metrics.metrics(np.zeros_like(h), .5)["f1"])
        with self.assertRaisesRegex(ValueError, "changed and stable"):
            metrics.calibrate({"a": np.zeros((1, 2, metrics.BINS), np.int64)})

    def test_overlap_counted_once_and_class_unknown_not_negative(self):
        arrays = dict(target=np.array([[1, 0], [255, 0]], np.uint8), support=np.ones((2, 2), bool),
                      class_targets=np.array([[[1, 0], [255, 0]]], np.uint8),
                      before=np.array([[1, 1], [0, 1]], np.uint16), after=np.ones((2, 2), np.uint16))
        data = Namespace(categories=[{"name": "CRACK"}], groups=[], target=lambda _: arrays)
        row = dict(building_id="b", view_id="v", previous_year=1, current_year=2,
                   bbox_xywh=[0, 0, 2, 2], event_id="a", known_pixels=3)
        hist, coverage = metrics.collect(data, [row, {**row, "event_id": "b"}],
                                         lambda _: (np.array([[1., 0.], [1., 0.]]), arrays["support"]))
        self.assertEqual(coverage["known_pixels"], 3)
        self.assertEqual(hist["b"][0].sum(), 3)
        self.assertEqual(hist["b"][1].sum(), 3)

    def test_fusion_happens_after_overlap_merging(self):
        arrays = dict(target=np.ones((1, 1), np.uint8), support=np.ones((1, 1), bool),
                      class_targets=np.ones((1, 1, 1), np.uint8), before=np.ones((1, 1), np.uint16),
                      after=np.ones((1, 1), np.uint16))
        data = Namespace(categories=[{"name": "CRACK"}], groups=[], target=lambda _: arrays)
        row = dict(building_id="b", view_id="v", previous_year=1, current_year=2,
                   bbox_xywh=[0, 0, 1, 1], event_id="a", known_pixels=1)
        scores = {"a": np.array([[[1.]], [[.4]]]), "b": np.array([[[0.]], [[.8]]])}
        hist, _ = metrics.collect(data, [row, {**row, "event_id": "b"}],
            lambda r: (scores[r["event_id"]], arrays["support"]), native=True,
            fusion=[dict(native=True, threshold=.5), dict(native=False, threshold=.5)])
        self.assertEqual(metrics.metrics(hist["b"][0], .5, native=True)["tp"], 1)

    def test_conflicting_overlaps_stay_unknown_after_third_crop(self):
        stable = dict(target=np.zeros((1, 1), np.uint8), support=np.ones((1, 1), bool),
                      class_targets=np.zeros((1, 1, 1), np.uint8), before=np.ones((1, 1), np.uint16),
                      after=np.ones((1, 1), np.uint16))
        changed = {**stable, "target": np.ones((1, 1), np.uint8),
                   "class_targets": np.ones((1, 1, 1), np.uint8)}
        data = Namespace(categories=[{"name": "CRACK"}], groups=[],
                         target=lambda event: changed if event == "b" else stable)
        row = dict(building_id="b", view_id="v", previous_year=1, current_year=2,
                   bbox_xywh=[0, 0, 1, 1], known_pixels=1)
        hist, coverage = metrics.collect(data, [{**row, "event_id": name} for name in "abc"],
                                         lambda _: (np.ones((1, 1)), stable["support"]))
        self.assertEqual(coverage["conflicting_overlap_pixels"], 1)
        self.assertEqual(coverage["known_pixels"], 0)
        self.assertEqual(hist["b"].sum(), 0)


if __name__ == "__main__":
    unittest.main()
