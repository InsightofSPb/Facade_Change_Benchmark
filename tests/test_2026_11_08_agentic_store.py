"""Causal storage/retrieval contracts; no learned model is loaded."""
from __future__ import annotations

import copy
import contextlib
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from facade_change.agentic.retrieval import (KnowledgeBase, SearchIndex, dino_embedding_space,
                                             history_descriptor)
from facade_change.agentic.store import RegionStore, temporal_value
from facade_change.history_features import digest
from facade_change.io import read_json, sha256, write_json


SPACE = {"model": "contract-test-vectors", "revision": "fixture-v1",
         "representation": "visual", "preprocessing": {"fixture": True}}


def dino_contract_fixture():
    """Recorded extractor metadata only; fake hashes do not imply model inference."""
    from facade_change import history_features as hf
    return {"schema_version": 1, "backbone": "dinov2_vits14", "dimensions": 384,
        "checkpoint_sha256": "f" * 64, "python_tree_sha256": "e" * 64,
        "module_sha256": sha256(hf.__file__),
        "loader_sha256": sha256(Path(hf.__file__).parent / "methods" / "dinov2.py"),
        "normalization": copy.deepcopy(hf.NORMALIZATION), "resize": False, "augmentation": False,
        "invalid_pixels": "zero after normalization (ImageNet mean before normalization)",
        "pooling": "mean of fully supported x_norm_patchtokens, then L2 normalize",
        "patch_support": "all 196 pixels of each 14x14 patch are supported",
        "precision": "float32 inference and embeddings; optional float16 patch-token storage",
        "weights_frozen": True, "tf32": False, "save_patch_tokens": False, "device": "cpu",
        "fixture": "contract vectors only; no pretrained inference"}


class StoreFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = RegionStore(self.root / "store")
        self.serial = 0

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def region(self, name="facade", split="train"):
        return self.store.register_region(name, name + "-front", "a" * 64, [0, 0, 8, 8], split)

    def row(self, region, date="2020", available_at=None, description="crack"):
        self.serial += 1
        path = self.root / f"raw-{self.serial}.png"
        Image.fromarray(np.full((8, 8, 3), self.serial, np.uint8)).save(path)
        support = self.root / f"support-{self.serial}.npy"
        np.save(support, np.ones((8, 8), bool), allow_pickle=False)
        return {"region_id": region["region_id"], "date": date,
            "date_status": "confirmed" if date else "unknown",
            "available_at": date if available_at is None else available_at,
            "original_path": str(path), "crop_path": str(path), "support_path": str(support),
            "geometry": {"source_to_crop": np.eye(3).tolist(), "crop_shape": [8, 8],
                         "coordinate_system": "native_anchor_pixel_centres_xy"},
            "correspondence_status": "aligned", "visibility": "unknown",
            "description": description, "description_source": "test fixture, not model evidence"}

    def test_stable_region_uses_existing_track_identity_and_causal_prefix(self):
        region = self.region()
        expected = digest({"building_id": "facade", "view_id": "facade-front",
                           "anchor_sha256": "a" * 64, "bbox_xywh": [0, 0, 8, 8]})[:24]
        self.assertEqual(region["region_id"], expected)
        first = self.store.add_observation(self.row(region, "2019"))
        before = self.store.history(region["region_id"], "2020")["observations"]
        self.store.add_observation(self.row(region, "2025"))
        self.assertEqual(self.region()["region_id"], expected)
        self.assertEqual(self.store.history(region["region_id"], "2020")["observations"], before)
        self.assertEqual(first["observation_id"], before[0]["observation_id"])
        with self.assertRaisesRegex(ValueError, "Canonical anchor"):
            self.store.register_region("facade", "facade-front", "b"*64, [0, 0, 8, 8], "train")

    def test_dates_availability_and_current_pair_are_all_filtered(self):
        region = self.region()
        a = self.store.add_observation(self.row(region, "2019"))
        b = self.store.add_observation(self.row(region, "2020", "2023"))
        c = self.store.add_observation(self.row(region, None))
        current = self.store.add_observation(self.row(region, "2021"))
        history = self.store.history(region["region_id"], "2021", [current["observation_id"]])
        self.assertEqual([r["observation_id"] for r in history["observations"]], [a["observation_id"]])
        self.assertEqual(history["excluded"]["future"], 1)
        self.assertEqual(history["excluded"]["undated_or_unavailable"], 1)
        self.assertGreater(temporal_value("2020"), temporal_value("2020-06-01"))
        with self.assertRaises(ValueError):
            self.store.history(region["region_id"], None)

    def test_gt_fields_and_mixed_archives_never_enter_online_records(self):
        region = self.region()
        row = self.row(region)
        for payload in ({"packed": [1, 2]}, {"provenance": {"nested": {"gt_path": "hidden"}}}):
            with self.assertRaises(ValueError):
                self.store.add_observation({**row, **payload})
        archive = self.root / "mixed.npz"
        np.savez(archive, support=np.ones((8, 8), bool), packed=np.ones((8, 8), np.uint16))
        with self.assertRaises(Exception):
            self.store.add_observation({**row, "support_path": str(archive)})
        valid = self.store.add_observation(row)
        self.assertNotIn("packed", json.dumps(valid))

    def test_repeated_run_reuses_immutable_observation_and_rejects_changes(self):
        region = self.region()
        row = self.row(region)
        old = self.store.add_observation(row)
        duplicate = self.root / "second-run.png"
        shutil.copyfile(row["crop_path"], duplicate)
        changed_paths = {**row, "crop_path": str(duplicate), "provenance": {"run": "second"}}
        self.assertEqual(self.store.add_observation(changed_paths), old)
        changed_paths["geometry"] = {**row["geometry"], "coordinate_system": "explicit description of same anchor grid",
                                     "crop_to_source": np.eye(3).tolist()}
        self.assertEqual(self.store.add_observation(changed_paths), old)
        bad_geometry = copy.deepcopy(changed_paths)
        bad_geometry["geometry"]["source_to_crop"][0][2] = 1.
        with self.assertRaises(ValueError):
            self.store.add_observation(bad_geometry)
        with self.assertRaises(ValueError):
            self.store.add_observation({**row, "date": "2022"})
        self.store.close()
        self.store = RegionStore(self.root / "store")
        self.assertEqual(self.store.get_observation(old["observation_id"]), old)
        path = self.root / "store" / "observations" / (old["observation_id"] + ".json")
        content = json.loads(path.read_text())
        content["date"] = "2000"
        path.write_text(json.dumps(content))
        with self.assertRaisesRegex(ValueError, "changed"):
            self.store.get_observation(old["observation_id"])

    def test_split_identity_and_grid_checks(self):
        region = self.region()
        with self.assertRaisesRegex(ValueError, "Building leakage"):
            self.region(split="test")
        row = self.row(region)
        row["geometry"]["crop_shape"] = [9, 8]
        with self.assertRaisesRegex(ValueError, "grids"):
            self.store.add_observation(row)
        row["geometry"]["crop_shape"] = [8, 8]
        self.store.add_observation(row)
        other = self.region("other", "test")
        with self.assertRaisesRegex(ValueError, "across splits"):
            self.store.add_observation({**row, "region_id": other["region_id"]})

    def test_statements_are_typed_and_do_not_become_human_confirmation(self):
        region = self.region()
        obs = self.store.add_observation(self.row(region))
        base = {"region_id": region["region_id"], "observation_ids": [obs["observation_id"]],
                "kind": "agent_hypothesis", "summary": "Possible repair", "available_at": "2021"}
        first = self.store.add_evidence(base)
        self.assertEqual(self.store.add_evidence(base)["evidence_id"], first["evidence_id"])
        with self.assertRaisesRegex(ValueError, "author and source"):
            self.store.add_evidence({**base, "kind": "human_confirmation"})
        confirmed = self.store.add_evidence({**base, "kind": "human_confirmation", "author": "fixture-reviewer",
            "source": {"locator": "fixture-review"}, "source_evidence_ids": [first["evidence_id"]]})
        self.assertNotEqual(first["evidence_id"], confirmed["evidence_id"])
        self.assertEqual(self.store.get_evidence(first["evidence_id"])["kind"], "agent_hypothesis")

    def test_journal_ids_are_namespaced_and_survive_reopening(self):
        first = self.region()
        namespace = self.store.store_id
        with RegionStore(self.root / "another-store") as other:
            second = other.register_region("facade", "facade-front", "a"*64, [0, 0, 8, 8], "train")
            self.assertEqual(first["region_id"], second["region_id"])
            self.assertNotEqual(first["source_event_id"], second["source_event_id"])
        self.store.close()
        self.store = RegionStore(self.root / "store")
        self.assertEqual(self.store.store_id, namespace)
        self.assertEqual(self.store.get_region(first["region_id"])["source_event_id"], first["source_event_id"])
        events = [json.loads(x) for x in (self.root / "store/journal/events.jsonl").read_text().splitlines()]
        self.assertEqual({e["run_id"] for e in events}, {namespace})
        self.assertTrue(all(e["event_id"].startswith(namespace + ":") for e in events))

    def test_historical_statements_are_causal_typed_and_cannot_cite_current_pair(self):
        region = self.region()
        old = self.store.add_observation(self.row(region, "2018"))
        current = self.store.add_observation(self.row(region, "2020"))
        base = {"kind": "human_confirmation", "region_id": region["region_id"],
            "observation_ids": [old["observation_id"]], "available_at": "2019",
            "summary": "A sourced past confirmation", "author": "fixture reviewer", "source": {"locator": "fixture:1"}}
        confirmed = self.store.add_evidence(base)
        future = self.store.add_evidence({**base, "available_at": "2025", "summary": "Future information"})
        current_prediction = self.store.add_evidence({"kind": "prediction", "region_id": region["region_id"],
            "observation_ids": [current["observation_id"]], "available_at": "2020", "summary": "current-pair result"})
        indirect = self.store.add_evidence({"kind": "agent_hypothesis", "region_id": region["region_id"],
            "observation_ids": [old["observation_id"]], "available_at": "2020", "summary": "indirect current-pair claim",
            "source_evidence_ids": [current_prediction["evidence_id"]]})
        result = self.store.history(region["region_id"], "2020", [current["observation_id"]])
        self.assertEqual([e["evidence_id"] for e in result["evidence"]], [confirmed["evidence_id"]])
        self.assertEqual(result["evidence"][0]["kind"], "human_confirmation")
        self.assertNotIn(future["evidence_id"], json.dumps(result))
        self.assertNotIn(indirect["evidence_id"], json.dumps(result))

    def test_retrieval_filters_split_time_target_duplicates_and_invalidates_cache(self):
        target = self.store.add_observation(self.row(self.region("target", "test"), "2021"))
        train = self.store.add_observation(self.row(self.region("analogue"), "2020"))
        late = self.store.add_observation(self.row(self.region("late"), "2025"))
        test = self.store.add_observation(self.row(self.region("holdout", "test"), "2020"))
        index = SearchIndex(self.store)
        for item in (target, train, late, test):
            index.add_embedding(item["observation_id"], [1., 0.], SPACE)
        args = dict(cutoff="2021", exclude_buildings=["target"], exclude_hashes=[target["original_sha256"]])
        first = index.similar([1., 0.], SPACE, **args)
        self.assertEqual([m["observation"]["observation_id"] for m in first["matches"]], [train["observation_id"]])
        self.assertTrue(index.similar([1., 0.], SPACE, **args)["cache_hit"])
        extra = self.store.add_observation(self.row(self.region("other"), "2019"))
        index.add_embedding(extra["observation_id"], [0.5, 0.5], SPACE)
        later = index.similar([1., 0.], SPACE, **args)
        self.assertFalse(later["cache_hit"])
        self.assertNotEqual(later["cache_key"], first["cache_key"])
        changed_filter = index.similar([1., 0.], SPACE, **{**args, "cutoff": "2026"})
        self.assertNotEqual(changed_filter["cache_key"], later["cache_key"])
        self.assertEqual(len(index.text("crack", **args)["matches"]), 2)

    def test_future_corpus_additions_do_not_change_causal_search_packet(self):
        index = SearchIndex(self.store)
        row = self.store.add_observation(self.row(self.region("old"), "2019"))
        index.add_embedding(row["observation_id"], [1, 0], SPACE)
        before = index.similar([1, 0], SPACE, "2020")
        late = self.store.add_observation(self.row(self.region("future"), "2025"))
        index.add_embedding(late["observation_id"], [1, 0], SPACE)
        # An inaccessible future artifact cannot become a dependency of a past query.
        Path(late["crop_path"]).unlink()
        after = index.similar([1, 0], SPACE, "2020")
        self.assertEqual(before["matches"], after["matches"])
        self.assertEqual(before["cache_key"], after["cache_key"])
        self.assertEqual(before["corpus_version"], after["corpus_version"])

    def test_embedding_spaces_are_not_mixed_and_input_bytes_are_verified(self):
        obs = self.store.add_observation(self.row(self.region()))
        index = SearchIndex(self.store)
        index.add_embedding(obs["observation_id"], [1., 0.], SPACE)
        other = {**SPACE, "revision": "different-weights"}
        result = index.similar([1., 0.], other, "2021")
        self.assertEqual(result["matches"], [])
        with self.assertRaisesRegex(ValueError, "immutable"):
            index.add_embedding(obs["observation_id"], [0., 1.], SPACE)
        Image.fromarray(np.full((8, 8, 3), 222, np.uint8)).save(obs["crop_path"])
        with self.assertRaisesRegex(ValueError, "hash differs"):
            index.get_embedding(obs["observation_id"])

    def test_history_baseline_preserves_order_intervals_and_prefix(self):
        region = self.region()
        rows = [self.store.add_observation(self.row(region, year)) for year in ("2018", "2019", "2020")]
        entries = {row["observation_id"]: {"space": SPACE, "vector": vector}
                   for row, vector in zip(rows, ([1, 0], [0, 1], [1, 1]))}
        original = history_descriptor(rows, entries)
        shuffled = history_descriptor(rows[::-1], entries)
        self.assertEqual(original, shuffled)
        changed = copy.deepcopy(rows)
        changed[1]["date"] = "2018-12-30"
        different = history_descriptor(changed, entries)
        self.assertNotEqual(original["vector"], different["vector"])
        self.assertFalse(original["trained"])
        index = SearchIndex(self.store)
        for row in rows:
            index.add_embedding(row["observation_id"], entries[row["observation_id"]]["vector"], SPACE)
        analogue_region = self.region("analogue")
        for year, vector in zip(("2018", "2019", "2020"), ([1, 0], [0, 1], [1, 1])):
            row = self.store.add_observation(self.row(analogue_region, year))
            index.add_embedding(row["observation_id"], vector, SPACE)
        result = index.similar_history(region["region_id"], "2020")
        self.assertEqual(result["matches"][0]["region_id"], analogue_region["region_id"])
        self.assertAlmostEqual(result["matches"][0]["score"], 1., places=6)


class KnowledgeTests(unittest.TestCase):
    def test_taxonomy_is_sourced_and_textbook_future_claims_are_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            knowledge = KnowledgeBase(Path(temp))
            taxonomy = Path(__file__).resolve().parents[1] / "configs" / "semantic_class_groups.json"
            knowledge.seed_taxonomy(taxonomy, "2010")
            current = knowledge.search("CRACK", "2020")
            self.assertEqual(current["fragments"][0]["source"]["kind"], "project_taxonomy")
            self.assertTrue(knowledge.related("CRACK", "2020")["relations"])
            knowledge.import_document({"document_id": "provided-excerpt", "available_at": "2025",
                "source": {"kind": "user_document", "title": "Test fixture excerpt", "locator": "fixture:1"},
                "concepts": [], "relations": [], "fragments": [{"fragment_id": "text-1", "text": "CRACK future fixture text",
                    "concept_ids": ["CRACK"], "locator": "page:1"}]})
            before = knowledge.search("CRACK", "2020")
            self.assertEqual(before["cache_key"], current["cache_key"])
            self.assertEqual(before["fragments"], current["fragments"])
            after = knowledge.search("CRACK", "2025")
            self.assertTrue(any(x["document_id"] == "provided-excerpt" for x in after["fragments"]))
            with self.assertRaises(ValueError):
                knowledge.import_document({"source": {"title": "Unattributed"}, "available_at": "2020"})

    def test_knowledge_revision_is_immutable_and_cache_hashes_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            knowledge = KnowledgeBase(Path(temp))
            value = {"document_id": "d", "available_at": "2020",
                "source": {"kind": "user_document", "title": "Fixture", "locator": "fixture:1"},
                "concepts": [], "relations": [], "fragments": [{"fragment_id": "f", "text": "repair", "concept_ids": []}]}
            knowledge.import_document(value)
            query = knowledge.search("repair", "2020")
            self.assertTrue(knowledge.search("repair", "2020")["cache_hit"])
            with self.assertRaisesRegex(ValueError, "immutable"):
                knowledge.import_document({**value, "available_at": "2021"})
            path = Path(temp) / "query_cache" / (query["cache_key"] + ".json")
            corrupt = json.loads(path.read_text())
            corrupt["result"]["fragments"][0]["text"] = "injected"
            path.write_text(json.dumps(corrupt))
            with self.assertRaisesRegex(ValueError, "Corrupt"):
                knowledge.search("repair", "2020")


class CanonicalImportTests(unittest.TestCase):
    """Exercise current canonical preparation and feature artifacts, not hand-built lookalikes."""
    def setUp(self):
        from test_history_features import Fixture
        from facade_change import history_features as hf
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = Fixture(self.root)
        self.histories = self.root / "histories"
        self.silence = contextlib.redirect_stdout(io.StringIO())
        self.silence.__enter__()
        self.addCleanup(self.silence.__exit__, None, None, None)
        hf.prepare_histories(self.fixture.session, self.histories, tile_size=28,
                             stride=28, min_valid_fraction=1.)
        self.store = RegionStore(self.root / "store")
        self.addCleanup(self.store.close)

    def test_real_format_import_matches_track_ids_pixels_geometry_and_incremental_dates(self):
        from facade_change import history_features as hf
        imported = self.store.import_histories(self.histories, available_at="observation_date")
        manifest = read_json(self.histories / "manifest.json")
        self.assertEqual({r["region_id"] for r in self.store.list_regions()},
                         {t["track_id"] for t in manifest["tracks"]})
        self.assertEqual(imported["count"], sum(len(t["observations"]) for t in manifest["tracks"]))
        track = next(t for t in manifest["tracks"] if t["distinct_years"] == 3)
        prefix = self.store.history(track["track_id"], "2013")["observations"]
        for observation in track["observations"]:
            row = self.store.get_observation(observation["observation_id"])
            self.assertEqual(row["crop_sha256"], observation["rgb_sha256"])
            expected = np.asarray(observation["alignment"]["matrix"])
            np.testing.assert_allclose(np.asarray(row["geometry"]["crop_to_anchor"]) @
                                       np.asarray(row["geometry"]["source_to_crop"]), expected)
        self.fixture.add(2026, 0, 112, alternative=True)
        self.fixture.save()
        hf.prepare_histories(self.fixture.session, self.histories, tile_size=28,
                             stride=28, min_valid_fraction=1.)
        self.store.import_histories(self.histories, available_at="observation_date")
        self.assertEqual(prefix, self.store.history(track["track_id"], "2013")["observations"])
        self.assertEqual(len(self.store.history(track["track_id"], "2026")["observations"]), 4)

    def test_existing_feature_writer_cache_import_without_model_dependency(self):
        """Actual cache writer/reader and journal, deterministic contract vectors only."""
        from facade_change import history_features as hf
        self.store.import_histories(self.histories, available_at="observation_date")
        manifest = read_json(self.histories / "manifest.json")
        observation = manifest["tracks"][0]["observations"][0]
        features = self.histories / "features"
        features.mkdir()
        encoder = dino_contract_fixture()
        vector = hf._l2(np.arange(1, 385, dtype=np.float32))
        saved = hf._save_feature(features, hf._feature_signature(observation, encoder),
                                 {"z": vector, "patch_support": np.ones(4, dtype=bool)})
        run = hf._attempt(features, "embed", {"fixture": "existing feature cache import"})
        try:
            index = {"schema_version": 1, "kind": "frozen_history_features", "encoder": encoder,
                     "features": {observation["observation_id"]: saved},
                     "report": run.out.relative_to(features).as_posix()}
            write_json(run.out / "index.json", index)
            run.finish("completed", index_sha256=sha256(run.out / "index.json"))
            write_json(features / "index.json", index)
        finally:
            run.close()
        search = SearchIndex(self.store)
        self.assertEqual(search.import_history_features(features)["count"], 1)
        actual = search.get_embedding(observation["observation_id"])
        np.testing.assert_allclose(actual["vector"], vector, atol=1e-7)
        live_space = dino_embedding_space(encoder["checkpoint_sha256"], encoder["python_tree_sha256"])
        self.assertEqual(actual["space"], live_space)
        found = search.similar(vector, live_space, "2026", allowed_splits=("train", "val"))
        self.assertEqual(found["matches"][0]["observation_id"], observation["observation_id"])
        different_weights = dino_embedding_space("b" * 64, encoder["python_tree_sha256"])
        self.assertEqual(search.similar(vector, different_weights, "2026")["matches"], [])
        search.import_history_features(features)
        self.assertEqual(search.get_embedding(observation["observation_id"]), actual)
        source_evidence = [r for r in self.store.list_evidence() if r["kind"] == "measurement"]
        self.assertEqual(len(source_evidence), 1)
        self.assertEqual(source_evidence[0]["artifacts"][0]["sha256"], saved["sha256"])
        self.assertEqual(source_evidence[0]["provenance"]["encoder_contract"], encoder)
        (features / saved["path"]).write_bytes(b"corrupted cache")
        with self.assertRaisesRegex(ValueError, "Corrupt frozen-feature"):
            search.import_history_features(features)

    def test_dino_space_preserves_semantics_and_rejects_incomplete_preprocessing(self):
        contract = dino_contract_fixture()
        live = dino_embedding_space(contract["checkpoint_sha256"], contract["python_tree_sha256"])
        historical = lambda value: dino_embedding_space(value["checkpoint_sha256"],
            value["python_tree_sha256"], encoder_contract=value)
        self.assertEqual(historical(contract), live)
        diagnostics = {**contract, "device": "cuda:1", "torch": "other-runtime",
                       "save_patch_tokens": True, "checkpoint_path": "/another/local/path"}
        self.assertEqual(historical(diagnostics), live)
        for key in ("checkpoint_sha256", "python_tree_sha256", "module_sha256", "loader_sha256"):
            self.assertNotEqual(historical({**contract, key: "b" * 64}), live)
        for key in ("normalization", "patch_support", "pooling", "precision", "module_sha256"):
            missing = copy.deepcopy(contract)
            missing.pop(key)
            with self.assertRaises(ValueError):
                historical(missing)
        for key, value in (("resize", True), ("tf32", True), ("pooling", "CLS token")):
            with self.assertRaisesRegex(ValueError, "Unsupported or incomplete"):
                historical({**contract, key: value})
        with self.assertRaisesRegex(ValueError, "complete SHA-256"):
            dino_embedding_space("unversioned", contract["python_tree_sha256"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional CPU torch unavailable")
    def test_existing_feature_export_import_parity_and_cache_binding(self):
        import torch
        from test_history_features import TOY_HUB
        from facade_change import history_features as hf
        torch.set_num_threads(1)
        self.store.import_histories(self.histories, available_at="observation_date")
        source = self.root / "toy_dino"
        source.mkdir()
        (source / "hubconf.py").write_text(TOY_HUB)
        checkpoint = self.root / "toy.pth"
        torch.save({"weight": torch.arange(384, dtype=torch.float32)/384 + 1}, checkpoint)
        config = self.root / "methods.json"
        write_json(config, {"methods": {"dinov2": {"source_root": str(source), "checkpoint_path": str(checkpoint)}}})
        settings = hf.read_dino_config(config)
        features = self.histories / "features"
        with mock.patch("torch.hub.download_url_to_file", side_effect=AssertionError("No downloads")):
            hf.extract_features(self.histories, features, settings, device="cpu", batch_size=3, min_years=1)
        index = read_json(features / "index.json")
        search = SearchIndex(self.store)
        imported = search.import_history_features(features)
        self.assertEqual(imported["count"], len(index["features"]))
        observation_id, original = next(iter(index["features"].items()))
        with np.load(features / original["path"], allow_pickle=False) as arrays:
            np.testing.assert_allclose(search.get_embedding(observation_id)["vector"], arrays["z"], atol=1e-7)
        stable = search.get_embedding(observation_id)
        search.import_history_features(features)
        self.assertEqual(search.get_embedding(observation_id), stable)
        (features / original["path"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Corrupt frozen-feature"):
            search.import_history_features(features)


if __name__ == "__main__":
    unittest.main()
