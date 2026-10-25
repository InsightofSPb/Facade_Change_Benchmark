"""Feature/label contracts, no-future models, metrics and a CPU training smoke test.

The vectors below are procedural fixtures, not real DINO or facade evidence.
"""
import contextlib
import csv
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


hm = importlib.import_module("facade_change.research.2026-10-25_history_models")


class FeatureFixture:
    def __init__(self, root):
        self.root = root / "histories"
        self.features = self.root / "features"
        self.prepare = self.root / "reports/prepare-1"
        self.embed = self.features / "reports/embed-1"
        self.prepare.mkdir(parents=True)
        (self.embed / "sequences").mkdir(parents=True)
        self.tracks, self.sequences = [], []
        rng = np.random.default_rng(72)
        years = np.array([2009, 2013, 2018, 2026], dtype=np.int32)
        for split in ("train", "val", "test"):
            for building in range(2):
                for label in (0, 1):
                    key = f"{split}_{building}_{label}"
                    observations = []
                    for year in years:
                        crop = self.root / f"{key}_{year}.png"
                        Image.fromarray(rng.integers(0, 256, (14, 14, 3), dtype=np.uint8)).save(crop)
                        observations.append({"observation_id": f"{key}_{year}", "year": int(year),
                                             "rgb": crop.name, "rgb_sha256": sha256(crop),
                                             "image_sha256": hm.digest([key, int(year)])})
                    track = {"track_id": key, "building_id": f"{split}_{building}",
                             "view_id": f"{split}_{building}_front", "split": split,
                             "bbox_xywh": [label*14, 0, 14, 14], "distinct_years": 4,
                             "observations": observations}
                    z = rng.normal(0, .01, (4, 8)).astype(np.float32)
                    z[:, 0] += 1
                    z[2:, 1] += label * 2
                    z = hm._l2(z)
                    path = self.embed / "sequences" / f"{key}.npz"
                    hm._npz(path, {"years": years, "z_years": z, "z_observations": z,
                                   "observation_ids": np.array([r["observation_id"] for r in observations]),
                                   "observation_years": years, "year_counts": np.ones(4, dtype=np.int32),
                                   "delta_years": np.diff(years, prepend=years[0]), "elapsed_years": years-years[0]})
                    self.tracks.append(track)
                    self.sequences.append({k: track[k] for k in
                                           ("track_id", "building_id", "view_id", "split", "bbox_xywh", "distinct_years")})
                    self.sequences[-1].update(path=path.relative_to(self.features).as_posix(),
                                              sha256=sha256(path),
                                              observation_ids=[r["observation_id"] for r in observations])
        self.save()

    def save(self):
        manifest = {"kind": "canonical_crop_histories", "schema_version": 1,
                    "tracks": self.tracks, "report": "reports/prepare-1"}
        write_json(self.prepare / "manifest.json", manifest)
        write_json(self.prepare / "run.json", {"status": "completed_needs_review",
                    "manifest_sha256": sha256(self.prepare / "manifest.json")})
        self.index = {"kind": "frozen_history_features", "schema_version": 1,
                      "report": "reports/embed-1", "history_report": "reports/prepare-1",
                      "history_root": str(self.root), "encoder": {"dimensions": 8, "fixture": True},
                      "history_manifest_sha256": sha256(self.prepare / "manifest.json"),
                      "sequences": self.sequences}
        write_json(self.features / "index.json", self.index)
        write_json(self.embed / "index.json", self.index)
        write_json(self.embed / "run.json", {"status": "completed", "index_sha256": sha256(self.features / "index.json")})

    def labels(self, path):
        data = hm.load_histories(self.features, ("train", "val", "test"))
        rows = []
        for sample in data.samples:
            row = {k: sample[k] for k in hm.LABEL_FIELDS if k in sample}
            label = int(sample["track_id"].endswith("_1"))
            row.update(label=label, review_status="accepted", comparable="yes",
                       event_type="crack" if label else "unchanged", review_source="procedural test fixture")
            rows.append(row)
        hm.write_csv(path, rows, hm.LABEL_FIELDS)
        return rows


class DataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = FeatureFixture(self.root)
        self.labels = self.root / "labels.csv"
        self.label_rows = self.fixture.labels(self.labels)

    def test_prefixes_have_no_current_or_future_in_history(self):
        data = hm.load_histories(self.fixture.features)
        self.assertEqual(len(data.samples), 16)
        first = data.samples[0]
        self.assertEqual(first["current_year"], 2018)
        self.assertEqual(first["past"].shape, (2, 8))
        np.testing.assert_array_equal(first["time"], [[0, 0], [4, 4]])
        self.assertEqual(first["pair_gap"], 5)
        self.assertEqual(data.coverage["short_prefixes"], 8)

    def test_future_vector_does_not_change_earlier_event_or_fingerprint(self):
        before = hm.load_histories(self.fixture.features).samples[0]
        entry = self.fixture.sequences[0]
        path = self.fixture.features / entry["path"]
        with np.load(path, allow_pickle=False) as stored:
            arrays = {k: stored[k].copy() for k in stored.files}
        arrays["z_years"][-1] = -arrays["z_years"][-1]
        hm._npz(path, arrays)
        entry["sha256"] = sha256(path)
        self.fixture.save()
        after = hm.load_histories(self.fixture.features).samples[0]
        self.assertEqual(before["prefix_sha256"], after["prefix_sha256"])
        self.assertEqual(hm.cosine_scores(before), hm.cosine_scores(after))

    def test_train_does_not_open_test_features_or_interpret_test_labels(self):
        for entry in self.fixture.sequences:
            if entry["split"] == "test":
                (self.fixture.features / entry["path"]).unlink()
        for row in self.label_rows:
            if row["split"] == "test":
                row["label"] = "deliberately-invalid-test-label"
        hm.write_csv(self.labels, self.label_rows, hm.LABEL_FIELDS)
        data = hm.load_histories(self.fixture.features)
        rows, _ = hm.read_labels(data, self.labels)
        self.assertEqual(len(rows), 16)

    def test_building_leakage_is_rejected(self):
        self.fixture.tracks[-1]["building_id"] = "train_0"
        self.fixture.save()
        with self.assertRaisesRegex(ValueError, "Building leakage"):
            hm.load_histories(self.fixture.features)

    def test_hash_corruption_is_rejected(self):
        (self.fixture.features / self.fixture.sequences[0]["path"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            hm.load_histories(self.fixture.features)

    def test_bad_chronology_is_rejected_even_with_new_hash(self):
        entry = self.fixture.sequences[0]
        path = self.fixture.features / entry["path"]
        with np.load(path, allow_pickle=False) as stored:
            arrays = {k: stored[k].copy() for k in stored.files}
        arrays["years"][1] = 2009
        hm._npz(path, arrays)
        entry["sha256"] = sha256(path)
        self.fixture.save()
        with self.assertRaisesRegex(ValueError, "chronology"):
            hm.load_histories(self.fixture.features)

    def test_stale_labels_and_ambiguous_binary_targets_rejected(self):
        data = hm.load_histories(self.fixture.features)
        self.label_rows[0]["prefix_sha256"] = "old"
        hm.write_csv(self.labels, self.label_rows, hm.LABEL_FIELDS)
        with self.assertRaisesRegex(ValueError, "Stale"):
            hm.read_labels(data, self.labels)
        self.fixture.labels(self.labels)
        with self.labels.open() as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["comparable"] = "no"
        hm.write_csv(self.labels, rows, hm.LABEL_FIELDS)
        with self.assertRaisesRegex(ValueError, "comparable=yes"):
            hm.read_labels(hm.load_histories(self.fixture.features), self.labels)

    def test_unreviewed_and_excluded_are_never_negative_labels(self):
        self.label_rows[0].update(review_status="unreviewed", label="")
        self.label_rows[1].update(review_status="excluded", comparable="no", label="")
        hm.write_csv(self.labels, self.label_rows, hm.LABEL_FIELDS)
        rows, coverage = hm.read_labels(hm.load_histories(self.fixture.features), self.labels)
        self.assertEqual(len(rows), 14)
        self.assertEqual(coverage["excluded_or_unreviewed"], {"unreviewed": 1, "excluded": 1})

    def test_explicit_episode_reset_rechecks_prefix_eligibility(self):
        self.label_rows[0]["history_start_year"] = 2013
        hm.write_csv(self.labels, self.label_rows, hm.LABEL_FIELDS)
        rows, coverage = hm.read_labels(hm.load_histories(self.fixture.features), self.labels)
        self.assertEqual(len(rows), 15)
        self.assertEqual(coverage["excluded_or_unreviewed"]["short_after_episode_reset"], 1)

    def test_known_repair_requires_a_new_episode_for_later_events(self):
        self.label_rows[2]["event_type"] = "repair"
        hm.write_csv(self.labels, self.label_rows, hm.LABEL_FIELDS)
        with self.assertRaisesRegex(ValueError, "Known past repair"):
            hm.read_labels(hm.load_histories(self.fixture.features), self.labels)
        self.label_rows[3]["history_start_year"] = 2018
        hm.write_csv(self.labels, self.label_rows, hm.LABEL_FIELDS)
        _, coverage = hm.read_labels(hm.load_histories(self.fixture.features), self.labels)
        self.assertEqual(coverage["excluded_or_unreviewed"]["short_after_episode_reset"], 1)

    def test_cosine_values_on_known_vectors(self):
        sample = {"past": np.array([[1., 0.], [0., 1.]], dtype=np.float32),
                  "current": np.array([0., 1.], dtype=np.float32)}
        scores = hm.cosine_scores(sample)
        self.assertEqual(scores["previous"], 0)
        self.assertEqual(scores["earliest"], 1)
        self.assertAlmostEqual(scores["history_mean"], 1-1/np.sqrt(2), places=6)

    def test_score_workflow_creates_blank_review_and_target_bounded_gallery(self):
        out = self.root / "score"
        with contextlib.redirect_stdout(io.StringIO()):
            code = workflow.main(["benchmark", "history-scores", "--features", str(self.fixture.features),
                                  "--out", str(out), "--gallery-limit", "3"])
        self.assertEqual(code, 0)
        with (out / "review.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(all(row["label"] == "" and row["review_status"] == "unreviewed" for row in rows))
        self.assertEqual(read_json(out / "run.json")["status"], "completed")
        self.assertIn("data:image/jpeg;base64", (out / "gallery.html").read_text())
        with self.assertRaises(FileExistsError):
            hm.score_histories(hm.parser().parse_args(["score", "--features", str(self.fixture.features), "--out", str(out)]))


class MetricsTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "Optional sklearn AP parity check")
    def test_ap_ties_weights_and_zero_weight_bootstrap_match_sklearn(self):
        from sklearn.metrics import average_precision_score
        labels, scores, weights = [0, 1, 0, 1, 1], [.9, .8, .8, .2, .1], [0, .5, 1, 2, .2]
        self.assertAlmostEqual(hm.average_precision(labels, scores, weights),
                               average_precision_score(labels, scores, sample_weight=weights))
        self.assertIsNone(hm.average_precision([0, 0], [1, 2]))

    def test_threshold_ties_cannot_exceed_validation_budget(self):
        samples = [{"label": y, "building_id": str(i//2)} for i, y in enumerate([0, 1, 0, 1])]
        scores = np.array([.9, .9, .2, .95])
        threshold = hm.select_threshold(samples, scores, 0)
        values = hm.metric_values(samples, scores, threshold)
        self.assertEqual(threshold, .95)
        self.assertEqual(values["actual_fpr"], 0)
        self.assertEqual(values["recall"], .5)

    def test_buildings_have_equal_weight_and_bootstrap_counts_groups(self):
        samples = [{"label": int(i % 2 == 0), "building_id": "large", "event_type": "crack"}
                   for i in range(20)]
        samples += [{"label": 1, "building_id": "small", "event_type": "repair"}]
        weights = hm.building_weights(samples)
        self.assertAlmostEqual(weights[:20].sum(), weights[-1])
        metrics = hm.report_metrics(samples, np.linspace(0, 1, 21), .5, bootstrap=20)
        self.assertEqual(metrics["bootstrap"]["unit"], "building_id")
        interval = metrics["bootstrap"]["intervals"]["building_weighted_ap"]["percentile_95"]
        self.assertTrue(np.isfinite(interval).all())


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch required for temporal training")
class NetworkTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.networks = importlib.import_module(hm.NETWORK_MODULE)
        rng = np.random.default_rng(4)
        self.samples = [{"event_id": str(i), "past": rng.normal(size=(n, 8)).astype(np.float32),
                         "time": np.array([[0, 0], [4, 4], [5, 9]])[:n].astype(np.float32),
                         "current": rng.normal(size=8).astype(np.float32), "pair_gap": 8}
                        for i, n in enumerate((2, 3))]

    def test_all_models_ignore_padding_and_history_ignores_current(self):
        for architecture in hm.MODELS:
            model = self.networks.HistoryClassifier(8, architecture, width=8, dropout=0).eval()
            batch = self.networks.batch_samples(self.samples, "cpu")
            logits, history = model(**batch)
            batch["past"][0, 2] = 999
            batch["time_features"][0, 2] = 999
            padded_logits, _ = model(**batch)
            torch.testing.assert_close(logits, padded_logits)
            batch["current"] = batch["current"] * -99
            _, changed_history = model(**batch)
            torch.testing.assert_close(history, changed_history)

    def test_no_time_ablation_is_invariant_to_calendar_gaps(self):
        for architecture in ("gru", "transformer", "pair", "pooled"):
            model = self.networks.HistoryClassifier(8, architecture, 8, 0, "none").eval()
            batch = self.networks.batch_samples(self.samples, "cpu")
            before = model(**batch)[0]
            batch["time_features"] += 500
            batch["pair_gap"] += 500
            torch.testing.assert_close(before, model(**batch)[0])

    def test_gap_conditioning_can_affect_output(self):
        model = self.networks.HistoryClassifier(8, "gru", 8, 0, "gaps").eval()
        batch = self.networks.batch_samples(self.samples, "cpu")
        before = model(**batch)[0]
        batch["time_features"] += 10
        self.assertFalse(torch.allclose(before, model(**batch)[0]))

    def test_shuffle_keeps_latest_pair_and_date_slots(self):
        original = self.networks.batch_samples(self.samples, "cpu")
        shuffled = self.networks.batch_samples(self.samples, "cpu", True)
        torch.testing.assert_close(original["past"][1, -1], shuffled["past"][1, -1])
        torch.testing.assert_close(original["current"], shuffled["current"])
        torch.testing.assert_close(original["time_features"], shuffled["time_features"])
        self.assertFalse(torch.equal(original["past"][1, :-1], shuffled["past"][1, :-1]))
        torch.testing.assert_close(original["past"][0], shuffled["past"][0])

    def test_model_really_learns_a_known_target(self):
        model = self.networks.HistoryClassifier(8, "gru", 8, 0)
        optimizer = torch.optim.Adam(model.parameters(), lr=.03)
        batch = self.networks.batch_samples(self.samples, "cpu")
        target = torch.tensor([0., 1.])
        for _ in range(60):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(**batch)[0], target)
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), .03)


@unittest.skipIf(torch is None, "PyTorch required for temporal training")
class TrainingTests(unittest.TestCase):
    def setUp(self):
        DataTests.setUp(self)
        torch.set_num_threads(1)
        self.silence = contextlib.redirect_stdout(io.StringIO())
        self.silence.__enter__()
        self.addCleanup(self.silence.__exit__, None, None, None)

    def arguments(self, out, *extra):
        return hm.parser().parse_args(["train", "--features", str(self.fixture.features),
              "--labels", str(self.labels), "--out", str(out), "--epochs", "2", "--width", "8",
              "--batch-size", "4", "--patience", "3", *extra])

    def test_full_training_evaluation_and_unlabelled_embeddings(self):
        run = self.root / "trained"
        hm.train_models(self.arguments(run))
        bundle = read_json(run / "models.json")
        self.assertEqual(len(bundle["models"]), 8)
        self.assertFalse(bundle["test_used"])
        out = self.root / "test-evaluation"
        argv = ["evaluate", "--features", str(self.fixture.features), "--labels", str(self.labels),
                "--training-run", str(run), "--out", str(out), "--splits", "test", "--bootstrap", "10"]
        hm.evaluate_models(hm.parser().parse_args(argv))
        report = read_json(out / "metrics.json")
        self.assertEqual(set(report["splits"]), {"test"})
        for name, info in bundle["models"].items():
            self.assertEqual(report["splits"]["test"][name]["threshold"], info["threshold"])
        with np.load(out / "gru_gaps_embeddings.npz", allow_pickle=False) as embeddings:
            self.assertEqual(embeddings["history_embeddings"].shape, (8, 8))
        self.assertIn("gru_gaps", report["past_shuffle"]["test"])
        argv.remove("--labels")
        argv.remove(str(self.labels))
        argv[argv.index(str(out))] = str(self.root / "unlabelled")
        hm.evaluate_models(hm.parser().parse_args(argv))
        self.assertEqual(read_json(self.root / "unlabelled/metrics.json")["splits"], {})

    def test_resume_matches_uninterrupted_weights(self):
        continuous, interrupted = self.root / "continuous", self.root / "interrupted"
        extra = ("--models", "gru", "--time-modes", "gaps")
        hm.train_models(self.arguments(continuous, *extra))
        original = hm.HistoryRun.event
        def interrupt(store, kind, **values):
            result = original(store, kind, **values)
            if kind == "epoch_completed":
                raise KeyboardInterrupt()
            return result
        with mock.patch.object(hm.HistoryRun, "event", interrupt), self.assertRaises(KeyboardInterrupt):
            hm.train_models(self.arguments(interrupted, *extra))
        self.assertEqual(read_json(interrupted / "run.json")["status"], "interrupted")
        hm.train_models(self.arguments(interrupted, *extra, "--resume"))
        a = torch.load(continuous / "gru_gaps/best.pth", weights_only=True)
        b = torch.load(interrupted / "gru_gaps/best.pth", weights_only=True)
        for name, value in a["state_dict"].items():
            torch.testing.assert_close(value, b["state_dict"][name], rtol=0, atol=0)

    def test_missing_review_labels_fail_before_gpu_or_output(self):
        for row in self.label_rows:
            row.update(review_status="unreviewed", label="")
        hm.write_csv(self.labels, self.label_rows, hm.LABEL_FIELDS)
        out = self.root / "not-created"
        with mock.patch.object(hm, "torch_runtime") as runtime, self.assertRaisesRegex(ValueError, "reviewed positive"):
            hm.train_models(self.arguments(out))
        runtime.assert_not_called()
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
