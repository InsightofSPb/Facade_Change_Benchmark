"""Data-contract and causal preprocessing tests; fixtures are not scientific evidence."""
import copy
import importlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from facade_change.history_features import Fingerprints
from facade_change.io import sha256, write_json

data = importlib.import_module("facade_change.research.2026-11-08_history_scenarios_data")


class DataTests(unittest.TestCase):
    def setUp(self):
        self.dataset = data.SyntheticData(channels=6, grid=4)

    def test_shapes_spatial_support_years_and_separate_targets(self):
        row = self.dataset.samples[0]
        direct = self.dataset.inputs(row)
        forecast = self.dataset.inputs(row, predictive=True)
        self.assertEqual(direct["features"].shape, (3, 6, 4, 4))
        self.assertEqual(forecast["features"].shape, (2, 6, 4, 4))
        self.assertEqual(forecast["support"].shape, (2, 1, 4, 4))
        self.assertTrue(forecast["padding"].all())
        np.testing.assert_array_equal(forecast["years"], [2001, 2004])
        self.assertEqual(forecast["target_year"], 2011)
        self.assertEqual(forecast["time_precision"], "year")
        self.assertNotIn("target", forecast)
        targets = self.dataset.feature_targets(row)
        self.assertEqual(targets["target_features"].shape, (6, 4, 4))
        self.assertEqual(targets["change_target"].shape, (1, 4, 4))
        self.assertTrue(np.all(targets["stable"] <= targets["valid"]))

    def test_forecast_preprocessing_ignores_deleted_current_and_future_and_gt(self):
        row = self.dataset.samples[0]
        before = self.dataset.forecast_prefix(row["track_id"], row["current_year"])
        track = self.dataset.tracks[row["track_id"]]
        # Corrupt all unavailable data first. No target-side crop support can enter inputs.
        for observation in track["observations"][row["frame"]:]:
            identifier = observation["observation_id"]
            self.dataset._features[identifier] = (np.full((6, 4, 4), np.nan), np.zeros((1, 4, 4), bool))
            self.dataset._labels[identifier]["packed"].fill(65535)
        with mock.patch.object(self.dataset, "annotation", side_effect=AssertionError("GT used")), mock.patch.object(
                self.dataset, "feature_target", side_effect=AssertionError("Target used")), mock.patch.object(
                self.dataset, "scores", side_effect=AssertionError("Current detector used")), mock.patch.object(
                self.dataset, "semantic", side_effect=AssertionError("Current semantics used")):
            after = self.dataset.inputs(row, predictive=True)
            for key in ("features", "support", "years", "padding"):
                np.testing.assert_array_equal(before[key], after[key])
            track["observations"] = track["observations"][:row["frame"]]
            deleted = self.dataset.forecast_prefix(row["track_id"], row["current_year"])
            for key in ("features", "support", "years", "padding"):
                np.testing.assert_array_equal(before[key], deleted[key])

    def test_train_normalization_never_uses_val_test_or_labels(self):
        normalizer = data.fit_normalizer(self.dataset)
        for track in self.dataset.tracks.values():
            if track["split"] != "train":
                for observation in track["observations"]:
                    x, mask = self.dataset._features[observation["observation_id"]]
                    self.dataset._features[observation["observation_id"]] = (x * 10000 + 10000, mask)
        with mock.patch.object(self.dataset, "annotation", side_effect=AssertionError("GT used")):
            repeated = data.fit_normalizer(self.dataset)
        np.testing.assert_array_equal(normalizer.mean, repeated.mean)
        np.testing.assert_array_equal(normalizer.scale, repeated.scale)
        dense = np.concatenate([x[:, support[0]] for _, x, support in self.dataset.iter_train_features()], axis=1)
        np.testing.assert_allclose(normalizer.mean, dense.mean(1), atol=1e-6)
        np.testing.assert_allclose(normalizer.scale, dense.std(1), atol=1e-6)
        state = normalizer.state_dict()
        self.assertEqual(state["fit_split"], "train")
        reloaded = data.FrozenNormalizer.from_state_dict(state)
        row = self.dataset.inputs(self.dataset.samples[0])
        expected = normalizer.apply(row["features"], row["support"])
        np.testing.assert_array_equal(expected, reloaded.apply(row["features"], row["support"]))
        self.assertFalse(expected[:, :, 0, 0].any())

    def test_annotations_unknown_not_stable_and_first_train_transition_included(self):
        self.assertEqual(len(self.dataset.split_samples("train", True)), 6)
        self.assertEqual(len(self.dataset.split_samples("train")), 4)
        self.assertEqual(len(self.dataset.split_samples("test", True)), 4)
        for row in self.dataset.samples:
            target = self.dataset.targets(row)
            self.assertEqual(target["target"][0, 0], 255)
            ft = self.dataset.feature_targets(row)
            self.assertFalse(ft["stable"][0, 0, 0])
            self.assertFalse(ft["valid"][0, 0, 0])
        original = np.array([[255, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], np.uint8)
        pooled, known = data.pool_temporal_target(original, (2, 2))
        np.testing.assert_array_equal(pooled, [[[255, 1], [0, 0]]])
        self.assertFalse(known[0, 0, 0])

    def test_semantic_prediction_has_no_gt_accessor(self):
        row = self.dataset.samples[0]
        with mock.patch.object(self.dataset, "annotation", side_effect=AssertionError("GT used")), mock.patch.object(
                self.dataset, "targets", side_effect=AssertionError("GT used")):
            pred = self.dataset.semantic(row)
        self.assertEqual(pred["probabilities"].shape, (3, 11, 8, 8))
        self.assertFalse(pred["target_labels_used"])
        annotations = self.dataset.annotation_history(row)
        self.assertEqual(annotations["labels"].shape, (3, 11, 8, 8))
        self.assertFalse(annotations["known"][:, :, 0, 0].any())

    def test_existing_semantic_cache_nan_outside_support_is_masked(self):
        adapter = data.ScenarioData.__new__(data.ScenarioData)
        adapter.categories = [dict(id=7, name="CRACK")]
        adapter.tracks = {"track": dict(observations=[dict(observation_id="one", year=2001)])}
        adapter.frozen = SimpleNamespace(root=Path("/unused/cache"))
        adapter.saved = SimpleNamespace(pin=mock.Mock())
        mask = np.ones((224, 224), bool)
        mask[0] = False
        probabilities = np.full((1, 224, 224), .5, np.float16)
        probabilities[:, ~mask] = np.nan
        record = dict(kind="predicted_semantic_context", artifact_path="one.npz")
        with mock.patch.object(data.benchmark, "read_unit", return_value=(record,
                dict(probabilities=probabilities, support=mask))):
            result = adapter.semantic(dict(track_id="track", frame=0))
        self.assertTrue(np.isfinite(result["probabilities"]).all())
        self.assertFalse(result["probabilities"][:, :, 0].any())

    def test_split_leakage_in_building_view_or_image_is_rejected(self):
        for kind in ("building_id", "view_id", "image_sha256"):
            tracks = copy.deepcopy(list(self.dataset.tracks.values()))
            a, b = tracks[0], tracks[2]
            if kind == "image_sha256":
                b["observations"][0][kind] = a["observations"][0][kind]
            else:
                b[kind] = a[kind]
            with self.assertRaisesRegex(ValueError, "leakage"):
                data.validate_partitions(tracks)

    def test_ontology_preserves_ids_and_records_source_reordering(self):
        categories = [dict(id=9, name="CRACK"), dict(id=22, name="REPAIRS")]
        original = dict(category_to_channel=[dict(category_id=22, channel=0), dict(category_id=9, channel=1)])
        contract = data.ontology_contract(categories, original)
        self.assertEqual(contract["category_to_channel"][0]["category_id"], 9)
        self.assertEqual(contract["source_head_category_to_channel"], original["category_to_channel"])
        original["category_to_channel"][0]["channel"] = 1
        with self.assertRaisesRegex(ValueError, "bijection"):
            data.ontology_contract(categories, original)

    def test_cached_geometry_requires_known_past_anchor_and_fixed_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            track = copy.deepcopy(next(iter(self.dataset.tracks.values())))
            track["anchor_sha256"] = "first-anchor"
            for row in track["observations"]:
                row["alignment"] = {"matrix": np.eye(3).tolist()}
            document = dict(options=dict(
                grid="native anchor (0,0); fixed integer multiples of stride, including negative coordinates",
                selection="saved session selection; direct earliest-anchor homographies only"),
                views=[dict(building_id=track["building_id"], view_id=track["view_id"],
                            anchor=dict(sha256="first-anchor", year=2001))])
            path = root / "manifest.json"
            write_json(path, document)
            canonical = SimpleNamespace(history_root=root, index=dict(history_manifest_sha256=sha256(path)),
                                        fp=Fingerprints(), tracks={track["track_id"]: track})
            result = data.audit_causal_geometry(canonical)
            self.assertEqual(result["status"], "passed")
            self.assertIn("retrospectively", result["cohort_caveat"])
            document["views"][0]["anchor"]["year"] = 2021
            write_json(path, document)
            canonical.index["history_manifest_sha256"] = sha256(path)
            canonical.fp = Fingerprints()
            with self.assertRaisesRegex(ValueError, "future observation"):
                data.audit_causal_geometry(canonical)

    def test_synthetic_baseline_uses_exact_canonical_evaluator_support(self):
        canonical_metrics = data.memory.metrics
        rows = self.dataset.split_samples("test")
        def predict(row):
            score, support = self.dataset.scores(row)
            return score[None], support
        blocks = list(canonical_metrics.merged_blocks(self.dataset, rows, predict, [False]))
        self.assertTrue(blocks)
        # Existing evaluator produces building/view/date groups rather than crop statistics.
        self.assertEqual({b[0][0] for b in blocks}, {r["building_id"] for r in rows})

    def test_common_evaluation_support_is_explicit_and_shared(self):
        dataset = self.dataset
        row = dataset.samples[0]
        original = dataset.targets(row)
        common = dataset.evaluation_data.target(row["event_id"])
        self.assertTrue(original["support"][0, 1])
        self.assertFalse(common["support"][0, 1])
        self.assertEqual(common["target"][0, 1], 255)
        self.assertTrue(np.all(common["class_targets"][:, 0, 1] == 255))
        support = dataset.evaluation_support(row)
        np.testing.assert_array_equal(support, common["support"])
        with mock.patch.object(dataset, "targets", side_effect=AssertionError("GT used to choose geometric support")):
            np.testing.assert_array_equal(dataset.evaluation_support(row), support)
        expected = int(np.count_nonzero(common["target"] != 255))
        self.assertEqual(dataset.evaluation_data.events[row["event_id"]]["known_pixels"], expected)

    def test_missing_features_are_prepared_only_in_new_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old"
            old.mkdir()
            (old / "units").mkdir()
            source_root = root / "source"
            source_root.mkdir()
            write_json(source_root / "run.json", dict(kind="source_fixture"))
            adapter = data.ScenarioData.__new__(data.ScenarioData)
            adapter.overlay = None
            adapter.saved = SimpleNamespace(out=old, encoder_sha256="encoder-hash")
            adapter.canonical = SimpleNamespace(observations={"one": {}}, fp=Fingerprints(),
                frame=lambda identifier: (np.ones((224, 224, 3), np.uint8), np.ones((224, 224), bool)))
            adapter.encoder = dict(checkpoint_sha256="same", device="cpu", weights_frozen=True)
            adapter.source = SimpleNamespace(root=source_root, config=dict(options=dict(dinov2={})))
            fake = SimpleNamespace(encode=lambda rgb, support, save_patch_tokens: [dict(
                patch_tokens=np.ones((256, 384), np.float16), patch_support=np.ones(256, bool))], close=lambda: None)
            store = data.fusion.data_module.Store(root / "new", {})
            try:
                with mock.patch.object(data.memory.history_features, "encoder_contract", return_value=adapter.encoder), mock.patch.object(
                        data.memory.history_features, "DinoEncoder", return_value=fake) as encoder:
                    result = adapter.prepare_missing(store)
                    self.assertEqual(result["newly_computed"], 1)
                    self.assertEqual(encoder.call_count, 1)
                    self.assertEqual(adapter.feature("one")[0].shape, (384, 16, 16))
                    self.assertEqual(adapter.prepare_missing(store)["newly_computed"], 0)
                    self.assertEqual(encoder.call_count, 1)
                self.assertEqual(list((old / "units").iterdir()), [])
                self.assertTrue((store.out / "units/2026-11-08_causal-features-one.json").exists())
            finally:
                store.close()

    def test_split_scoped_evaluation_does_not_inspect_test_targets(self):
        original = self.dataset.targets
        seen = []
        def targets(row):
            seen.append(row["split"])
            self.assertEqual(row["split"], "val")
            return original(row)
        with mock.patch.object(self.dataset, "targets", side_effect=targets):
            scoped = self.dataset.evaluation_for(split="val")
            self.assertEqual({row["split"] for row in scoped.samples}, {"val"})
            self.assertEqual(len(seen), len(scoped.samples))
            self.assertIs(scoped, self.dataset.evaluation_for(split="val"))

    def test_preflight_repairs_only_train_val_and_keeps_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old"
            (old / "units").mkdir(parents=True)
            source = root / "source"
            source.mkdir()
            write_json(source / "run.json", dict(kind="source_fixture"))
            adapter = data.ScenarioData.__new__(data.ScenarioData)
            adapter.overlay = None
            adapter.saved = SimpleNamespace(out=old, encoder_sha256="encoder-hash")
            adapter.tracks = {split: dict(split=split, observations=[dict(observation_id=split)])
                              for split in ("train", "val", "test")}
            seen = []
            def frame(identifier):
                seen.append(identifier)
                return np.ones((224, 224, 3), np.uint8), np.ones((224, 224), bool)
            adapter.canonical = SimpleNamespace(observations={split: {} for split in adapter.tracks},
                                               fp=Fingerprints(), frame=frame)
            adapter.encoder = dict(checkpoint_sha256="same", device="cpu", weights_frozen=True)
            adapter.source = SimpleNamespace(root=source, config=dict(options=dict(dinov2={})))
            adapter.cache_index = {"feature:" + key: None for key in adapter.tracks}
            original_index = dict(adapter.cache_index)
            fake = SimpleNamespace(encode=lambda rgb, support, save_patch_tokens: [dict(
                patch_tokens=np.ones((256, 384), np.float16), patch_support=np.ones(256, bool))], close=lambda: None)
            store = data.fusion.data_module.Store(root / "new", {})
            try:
                with mock.patch.object(data.memory.history_features, "encoder_contract", return_value=adapter.encoder), mock.patch.object(
                        data.memory.history_features, "DinoEncoder", return_value=fake):
                    adapter.prepare_missing(store, splits=("train", "val"))
                    self.assertEqual(seen, ["train", "val"])
                    self.assertEqual(adapter.missing_features(splits=("train", "val")), [])
                    self.assertEqual(adapter.missing_features(), ["test"])
                    self.assertIsNotNone(adapter.cache_index["feature:train"])
                    self.assertIsNone(adapter.cache_index["feature:test"])
                    self.assertEqual(adapter.source_cache_index, original_index)
                    adapter.prepare_missing(store, splits=("test",))
                    self.assertEqual(adapter.missing_features(), [])
                    self.assertEqual(adapter.source_cache_index, original_index)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
