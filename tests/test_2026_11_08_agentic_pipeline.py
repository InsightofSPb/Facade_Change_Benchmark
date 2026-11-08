"""Contract and thin CPU integration checks; no real learned model claim."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from facade_change.agentic.pipeline import (_frozen_fusion, pairs_from_history, prepare_case, resolve_pair_spec)
from facade_change.agentic.store import RegionStore
from facade_change.history_features import digest
from facade_change.io import read_json, sha256, write_json
from facade_change.methods.registry import make_method


class ExistingCpuPool:
    """Exercise the real existing CPU algorithms without subprocess overhead."""
    calls = 0

    def __init__(self, **kwargs):
        pass

    def score(self, method, options, reference, source, support):
        type(self).calls += 1
        scorer = make_method(method, **options)
        result = {"scores": scorer(reference, source, support), "metadata": scorer.metadata}
        if scorer.raw_scores is not None:
            result["raw_scores"] = scorer.raw_scores
        return result

    def close(self):
        pass

    release = close


def saved_stack_fixture(root, methods, metadata):
    """A tiny saved TRAIN-model/VAL-threshold contract, not a learned quality claim."""
    fusion = importlib.import_module("data_analysis.2026-11-06_history_fusion")
    saved, source = root / "fusion", root / "source"
    saved.mkdir(); source.mkdir()
    (source / "method_contracts").mkdir()
    (saved / "maps").mkdir(); (saved / "units").mkdir()
    columns = len(fusion.METHODS) * len(fusion.MODES)
    item = dict(id="stack--previous", mode="previous", kind="stack", method="stack",
                columns=[fusion.METHODS.index(method) for method in methods], native=False,
                context_size=1, context_offset=0)
    write_json(saved / "experiment_contract.json", dict(fit_split="train", threshold_split="val",
        variants=[item], code={Path(fusion.__file__).name: sha256(fusion.__file__)}))
    write_json(source / "thresholds.json", {method: {mode: {"threshold": .5}
        for mode in fusion.MODES} for method in fusion.METHODS})
    summary = dict(source=str(source), source_sha256={str(source / "thresholds.json"): sha256(source / "thresholds.json")})
    write_json(saved / (fusion.PREFIX + "summary.json"), summary)
    write_json(saved / "run.json", dict(kind="canonical_history_fusion", status="completed",
        summary_sha256=sha256(saved / (fusion.PREFIX + "summary.json"))))
    write_json(saved / (fusion.PREFIX + "normalizer.json"), dict(low=[0]*columns, high=[1]*columns))
    write_json(saved / (fusion.PREFIX + "thresholds.json"), {item["id"]: {"threshold": .5, "selection_split": "val"}})
    identifier = fusion.PREFIX + "model-" + item["id"]
    model = dict(center=np.zeros(len(methods)), scale=np.ones(len(methods)),
                 parameters=np.r_[-1., np.ones(len(methods))])
    np.savez(saved / "maps/model.npz", **model)
    write_json(saved / "units" / (identifier + ".json"), dict(unit_id=identifier, variant=item,
        split="train", artifact_path="maps/model.npz", artifact_sha256=sha256(saved / "maps/model.npz")))
    for method in methods:
        write_json(source / "method_contracts" / (method + ".json"), metadata[method])
    return saved


class PairIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.a = np.random.default_rng(29).integers(10, 245, (192, 224, 3), dtype=np.uint8)
        self.b = self.a.copy()
        self.b[50:100, 70:115] = [250, 20, 35]
        self.c = self.b.copy()
        self.c[20:45, 140:165] = [25, 230, 60]
        for name, array in (("a", self.a), ("b", self.b), ("c", self.c)):
            Image.fromarray(array).save(self.root / (name + ".png"))
        self.pair = dict(building_id="synthetic-building", view_id="front", split="train",
            reference=dict(path=str(self.root / "a.png"), date=2020),
            source=dict(path=str(self.root / "b.png"), date=2021))
        self.config = dict(methods=["ssim", "lzma_abs"], thresholds={"ssim": .15},
            diagnostic_baseline="ssim", cache_root=str(self.root / "cache"),
            alignment={"min_inliers": 8, "min_hull_fraction": .05})
        ExistingCpuPool.calls = 0

    def tearDown(self):
        self.tmp.cleanup()

    def alignment(self, *args):
        return np.eye(3), {"passed": True}, {"fixture": "identity; code test only"}

    def test_real_sift_and_existing_workers_without_annotations(self):
        with RegionStore(self.root / "store") as store:
            case = prepare_case(self.config, self.pair, self.root / "case", store)
            self.assertEqual(case["status"], "completed", case)
            self.assertEqual([m["status"] for m in case["maps"]], ["completed", "completed"])
            self.assertEqual(case["baseline"]["status"], "diagnostic_only")
            self.assertEqual(case["baseline"]["decision"], "changed")
            self.assertEqual(len(store.list_observations()), 2)
            a = np.asarray(Image.open(case["images"][0]["path"]))
            b = np.asarray(Image.open(case["images"][1]["path"]))
            support = np.asarray(Image.open(case["support_path"])) > 0
            expected = make_method("ssim")(a, b, support)
            np.testing.assert_array_equal(expected, np.load(case["maps"][0]["scores_path"]))
            self.assertIn("native_units", case["maps"][1]["metadata"])
            self.assertTrue(Path(case["maps"][0]["score_preview_path"]).is_file())
            self.assertFalse((self.root / "annotations").exists())

    def test_raw_fields_fail_closed_and_manifest_projects_only_images(self):
        for extra in ({"gt": "hidden"}, {"reference": {**self.pair["reference"], "labels": [1]}}):
            with self.assertRaises(ValueError):
                resolve_pair_spec({**self.pair, **extra})
        rows = []
        for role in ("reference", "source"):
            item = self.pair[role]
            rows.append(dict(image_id=role, image_path=item["path"], sha256=sha256(item["path"]),
                height=192, width=224, building_id="synthetic-building", view_id="front", split="train",
                image_status="ready", metadata_status="reviewed", year=item["date"],
                annotations=[{"private": "never expose"}]))
        path = self.root / "manifest.json"
        write_json(path, {"images": rows, "annotations": [{"answer": "private"}], "targets": {"x": 1}})
        selected = resolve_pair_spec(dict(manifest=str(path), reference_id="reference", source_id="source"))
        text = json.dumps(selected)
        self.assertNotIn("private", text)
        self.assertNotIn('"annotations"', text)
        self.assertEqual(selected["reference"]["sha256"], rows[0]["sha256"])

    def test_stable_region_prefix_and_pair_cache_geometry_invalidation(self):
        with RegionStore(self.root / "store") as store, patch(
                "facade_change.agentic.pipeline._alignment", side_effect=self.alignment):
            first = prepare_case(self.config, self.pair, self.root / "first", store, ExistingCpuPool)
            self.assertEqual(first["status"], "completed", first.get("limitations"))
            second = prepare_case(self.config, self.pair, self.root / "second", store, ExistingCpuPool)
            self.assertEqual(second["status"], "completed", second.get("limitations"))
            self.assertTrue(all(row["cache_hit"] for row in second["maps"]))
            self.assertEqual(ExistingCpuPool.calls, 2)
            third_pair = {**self.pair, "region_id": first["region_id"],
                          "source": dict(path=str(self.root / "c.png"), date=2022)}
            third = prepare_case(self.config, third_pair, self.root / "third", store, ExistingCpuPool)
            self.assertEqual(first["region_id"], third["region_id"])
            past = store.history(first["region_id"], 2021)
            self.assertEqual(len(past["observations"]), 2)
            self.assertEqual(past["excluded"]["future"], 1)
            self.assertEqual(first["reference_observation_id"], third["reference_observation_id"])
        # A different explicit box is a different physical region/cache input.
        with RegionStore(self.root / "store") as store, patch(
                "facade_change.agentic.pipeline._alignment", side_effect=self.alignment):
            changed = prepare_case(self.config, {**self.pair, "bbox_xyxy": [0, 0, 112, 96]},
                                   self.root / "other_box", store, ExistingCpuPool)
            self.assertFalse(any(m["cache_hit"] for m in changed["maps"]))

    def test_unsupported_and_newly_observed_pixels_are_not_negative_evidence(self):
        matrix = np.array([[1., 0, 224], [0, 1, 0], [0, 0, 1]])
        with RegionStore(self.root / "store") as store, patch(
                "facade_change.agentic.pipeline._alignment", return_value=(matrix, {"passed": True}, {})):
            pair = {**self.pair, "bbox_xyxy": [224, 0, 448, 192]}
            case = prepare_case(self.config, pair, self.root / "case", store, ExistingCpuPool)
            self.assertEqual(case["comparability"], "no_comparable_support")
            self.assertEqual(case["baseline"]["decision"], "not_comparable")
            self.assertEqual(case["coverage"]["newly_observed_pixels"], 224*192)
            self.assertEqual(case["maps"], [])
            self.assertEqual(ExistingCpuPool.calls, 0)

    def test_bad_alignment_and_missing_model_remain_explicit_failures(self):
        with RegionStore(self.root / "store") as store, patch(
                "facade_change.agentic.pipeline._alignment", return_value=(np.eye(3), {"passed": False}, {})):
            case = prepare_case(self.config, self.pair, self.root / "rejected", store, ExistingCpuPool)
            self.assertEqual(case["comparability"], "alignment_rejected")
            self.assertEqual(case["maps"], [])
        with RegionStore(self.root / "new_store") as store, patch(
                "facade_change.agentic.pipeline._alignment", side_effect=self.alignment):
            config = {**self.config, "methods": ["anychange"], "method_options": {
                "anychange": {"checkpoint_path": str(self.root / "missing.pth")}}}
            case = prepare_case(config, self.pair, self.root / "missing", store, ExistingCpuPool)
            self.assertEqual(case["maps"][0]["status"], "error")
            self.assertNotIn("scores_path", case["maps"][0])
            self.assertEqual(case["baseline"]["decision"], "insufficient_data")

    def test_prepared_fusion_is_exposed_by_maps_and_fixed_context_but_not_pair(self):
        from facade_change.agentic.reasoning import run_reasoning
        from facade_change.agentic.retrieval import KnowledgeBase, SearchIndex
        from facade_change.agentic.tools import ToolService

        saved = saved_stack_fixture(self.root, ["ssim"], {"ssim": make_method("ssim").metadata})
        config = {**self.config, "methods": ["ssim"], "fusion": {"run": str(saved)}}
        class InspectPool:
            models = {"qwen": {"model": "explicit-test-double"}}
            def __init__(self):
                self.messages = []
            def call(self, role, request):
                self.messages.append(request["messages"])
                return {"status": "ok", "text": json.dumps({"type": "final", "decision": "insufficient_data",
                    "visible_difference": "Test contract only", "evidence_ids": [], "localization": None,
                    "limitations": [], "missing_information": []}), "usage": {"model_calls": 1}}

        with RegionStore(self.root / "store") as store, patch(
                "facade_change.agentic.pipeline._alignment", side_effect=self.alignment):
            case = prepare_case(config, self.pair, self.root / "case", store, ExistingCpuPool)
            self.assertEqual(case["status"], "completed", case)
            self.assertEqual(len(case["maps"]), 1)
            self.assertEqual(len(case["method_evidence"]), 2)
            pool = InspectPool()
            search, knowledge = SearchIndex(store), KnowledgeBase(self.root / "knowledge")
            service = ToolService(store, search, knowledge, pool, case, self.root / "tool")
            result = service.execute("get_maps", {"method": "fusion"})
            self.assertEqual(result["status"], "ok", result)
            self.assertEqual(len(result["evidence"]), 1)
            record = result["evidence"][0]
            self.assertEqual(record["evidence_id"], case["baseline"]["evidence_id"])
            self.assertEqual(record["coordinate_system"], "region_crop_pixels")
            self.assertEqual(record["summary"]["variant"]["id"], "stack--previous")
            self.assertEqual(record["summary"]["threshold_origin"]["selection_split"], "val")
            self.assertEqual(record["source_evidence_ids"], [case["maps"][0]["evidence_id"]])
            self.assertEqual(record["source"]["input_methods"][0]["method"], "ssim")
            self.assertEqual(record["source"]["model_artifacts"][str(saved / "maps/model.npz")],
                             sha256(saved / "maps/model.npz"))
            artifacts = {item["role"]: item for item in record["artifacts"]}
            self.assertEqual(set(artifacts), {"scores_path", "prediction_path", "support_path", "score_preview"})
            self.assertEqual(len(result["images"]), 1)
            np.testing.assert_array_equal(np.load(artifacts["scores_path"]["path"]),
                                          np.load(case["baseline"]["scores_path"]))
            for artifact in artifacts.values():
                self.assertEqual(artifact["sha256"], sha256(artifact["path"]))
            limited = service.execute("get_maps", {"limit": 1})
            self.assertEqual(limited["images"][0]["evidence_id"], record["evidence_id"])
            self.assertEqual(len(limited["evidence"]), 2)
            for mode in ("pair", "fixed_context"):
                output = self.root / mode
                service = ToolService(store, search, knowledge, pool, case, output)
                run_reasoning(mode, case, service, pool, output, budgets={"max_map_images": 1})
                messages = json.dumps(pool.messages[-1])
                if mode == "pair":
                    self.assertNotIn(record["evidence_id"], messages)
                    self.assertNotIn("stack--previous", messages)
                else:
                    self.assertIn(record["evidence_id"], messages)
                    self.assertIn("stack--previous", messages)
                    self.assertIn(artifacts["score_preview"]["path"], messages)
                    self.assertIn(case["maps"][0]["evidence_id"], messages)

    def test_prepared_three_date_history_reuses_immutable_observations_without_gt(self):
        history = self.root / "histories"
        history.mkdir()
        report = history / "reports" / "prepare-1"
        report.mkdir(parents=True)
        support = history / "support.png"
        Image.fromarray(np.full((192, 224), 255, np.uint8)).save(support)
        identity = dict(building_id="synthetic-building", view_id="front",
                        anchor_sha256=sha256(self.root / "a.png"), bbox_xywh=[0, 0, 224, 192])
        track_id = digest(identity)[:24]
        observations = []
        for year, name in ((2020, "a"), (2021, "b"), (2022, "c")):
            original = self.root / (name + ".png")
            rgb = history / (name + ".png")
            rgb.write_bytes(original.read_bytes())
            image_hash = sha256(original)
            observations.append(dict(observation_id=digest([track_id, image_hash])[:24], image_id=name,
                image_sha256=image_hash, original_image_path=str(original), year=year,
                rgb=rgb.name, support=support.name, rgb_sha256=sha256(rgb), support_sha256=sha256(support),
                alignment={"matrix": np.eye(3).tolist()}, valid_fraction=1.))
        document = dict(kind="canonical_crop_histories", schema_version=1, report="reports/prepare-1",
            tracks=[dict(track_id=track_id, **identity, split="train", distinct_years=3, observations=observations)])
        write_json(history / "manifest.json", document)
        write_json(report / "manifest.json", document)
        write_json(report / "run.json", dict(status="completed", manifest_sha256=sha256(history / "manifest.json")))
        for policy_number, available_at in enumerate(("observation_date", "2023-01-01", None)):
            with self.subTest(available_at=available_at), RegionStore(self.root / f"store-{policy_number}") as store:
                store.import_histories(history, available_at=available_at)
                pairs = pairs_from_history(history, track_id, available_at=available_at)
                self.assertEqual(pairs, pairs_from_history(history / "manifest.json", track_id,
                                                          available_at=available_at))
                self.assertEqual(len(pairs), 2)
                for number, pair in enumerate(pairs):
                    out = self.root / f"case-{policy_number}-{number}"
                    case = prepare_case(self.config, pair, out, store, ExistingCpuPool)
                    self.assertEqual(case["status"], "completed", case.get("limitations"))
                    self.assertEqual(case["region_id"], track_id)
                    self.assertFalse((out / "alignment").exists())
                stored = store.list_observations()
                self.assertEqual(len(stored), 3)
                if available_at != "observation_date":
                    self.assertEqual({row["available_at"] for row in stored}, {available_at})
                selected = store.history(track_id, 2021)
                self.assertEqual(len(selected["observations"]), 2 if available_at == "observation_date" else 0)
                selected = store.history(track_id, "2023-01-01")
                self.assertEqual(len(selected["observations"]), 0 if available_at is None else 3)


class FrozenFusionTests(unittest.TestCase):
    def test_actual_saved_stack_predict_and_hash_failures(self):
        fusion = importlib.import_module("data_analysis.2026-11-06_history_fusion")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = root / "out"
            out.mkdir()
            metadata_by_method = {method: {"method": method, "output_kind": "score"} for method in fusion.METHODS}
            saved = saved_stack_fixture(root, fusion.METHODS, metadata_by_method)
            maps, arrays = [], {}
            support = np.ones((2, 2), bool)
            for method in fusion.METHODS:
                maps.append(dict(method=method, status="completed", metadata=metadata_by_method[method]))
                arrays[method] = {"scores": np.full((2, 2), .3, np.float32)}
            result = _frozen_fusion({"run": str(saved)}, arrays, maps, support, out)
            self.assertEqual(result["decision"], "changed")
            from scipy.special import expit
            np.testing.assert_allclose(np.load(result["scores_path"]), expit(-1 + .3*len(fusion.METHODS)), rtol=1e-6)
            with self.assertRaises(ValueError):
                _frozen_fusion({"run": str(saved), "variant": "stack--history"}, arrays, maps, support, out)
            (saved / "maps/model.npz").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                _frozen_fusion({"run": str(saved)}, arrays, maps, support, out)


if __name__ == "__main__":
    unittest.main()
