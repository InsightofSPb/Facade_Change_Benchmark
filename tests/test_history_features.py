"""CPU contracts: known geometry, incremental histories and an offline toy hub.

The toy model tests the DINOv2 adapter's plumbing, not pretrained DINO quality.
No real facade images, model downloads or training are used.
"""
import contextlib
import csv
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from facade_change import history_features as hf
from facade_change.io import finish_record, read_json, run_record, sha256, write_json


class Fixture:
    def __init__(self, root):
        self.root = root
        self.session = root / "aligned/sessions/report-1"
        self.session.mkdir(parents=True)
        (root / "aligned/.preparation.lock").touch()
        self.images, self.pairs = [], []
        self.world = np.random.default_rng(7).integers(0, 256, (56, 112, 3), dtype=np.uint8)
        self.add(2009, 28, 84)
        self.add(2013, 14, 98)
        self.add(2020, 0, 112)
        self.save()

    def add(self, year, left, right, alternative=False):
        identifier = str(len(self.images) + 1)
        pixels = self.world[:, left:right].copy()
        if alternative:
            pixels = (pixels.astype(np.int16) + 5).clip(0, 255).astype(np.uint8)
        path = self.root / (identifier + ".png")
        Image.fromarray(pixels).save(path)
        row = {"image_id": identifier, "image_path": str(path), "sha256": sha256(path),
               "width": pixels.shape[1], "height": pixels.shape[0], "year": year,
               "building_id": "building", "view_id": "building_front", "split": "train",
               "metadata_status": "reviewed"}
        self.images.append(row)
        if len(self.images) > 1 and year > self.images[0]["year"]:
            anchor = self.images[0]
            folder = self.root / ("alignment-" + identifier)
            folder.mkdir()
            matrix = np.eye(3)
            matrix[0, 2] = left - 28
            write_json(folder / "geometry.json", {"source_to_reference": matrix.tolist()})
            record = {"kind": "pair_alignment", "status": "completed_needs_review",
                      "config": {"method": "loftr"}, "observations": {"reference": anchor, "source": row},
                      "quality_gate": {"passed": True},
                      "artifact_sha256": {"geometry.json": sha256(folder / "geometry.json")}}
            write_json(folder / "run.json", record)
            pair = {"pair_id": "1-" + identifier, "building_id": row["building_id"],
                    "view_id": row["view_id"], "split": "train", "reference_id": "1", "source_id": identifier,
                    "reference_year": anchor["year"], "source_year": year,
                    "selected_method": "loftr", "selection_status": "provisional_gate_pass",
                    "methods": {"loftr": {"state": "passed", "path": str(folder), "review": ""}}}
            self.pairs.append(pair)
        return row

    def save(self):
        manifest = self.root / "temporal_manifest.json"
        write_json(manifest, {"images": self.images, "pairs": self.pairs})
        write_json(self.session / "inputs.json", {"images": self.images, "manifest": str(manifest),
                                                   "manifest_sha256": sha256(manifest)})
        write_json(self.session / "pairs.json", self.pairs)
        fields = ["pair_id", "method", "reference_sha256", "source_sha256", "alignment_run_sha256",
                  "state", "decision", "notes"]
        with (self.session / "review.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            images = {x["image_id"]: x for x in self.images}
            for pair in self.pairs:
                for method, entry in pair["methods"].items():
                    writer.writerow({"pair_id": pair["pair_id"], "method": method,
                                     "reference_sha256": images[pair["reference_id"]]["sha256"],
                                     "source_sha256": images[pair["source_id"]]["sha256"],
                                     "alignment_run_sha256": sha256(Path(entry["path"]) / "run.json"),
                                     "state": entry["state"], "decision": entry["review"]})
        write_json(self.session / "summary.json", {"pairs": len(self.pairs), "plan_only": False})
        record = run_record("incremental_alignment_preparation", {"manifest_sha256": sha256(manifest), "plan_only": False})
        finish_record(self.session, record, "completed_needs_review")


class HistoryFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = Fixture(self.root)
        self.out = self.root / "histories"
        self.silence = contextlib.redirect_stdout(io.StringIO())
        self.silence.__enter__()
        self.addCleanup(self.silence.__exit__, None, None, None)

    def prepare(self, **kwargs):
        return hf.prepare_histories(self.fixture.session, self.out, tile_size=28, stride=14,
                                    min_valid_fraction=1., **kwargs)

    def test_known_shift_preserves_pixels_and_later_only_regions(self):
        summary = self.prepare()
        document = read_json(self.out / "manifest.json")
        self.assertEqual(summary["computed_frames"], 3)
        self.assertGreater(summary["histories_ge_3_years"]["tracks"], 0)
        self.assertTrue(any(t["bbox_xywh"][0] < 0 for t in document["tracks"]))
        self.assertTrue(any(t["distinct_years"] == 1 for t in document["tracks"]))
        for track in document["tracks"]:
            x, y, w, h = track["bbox_xywh"]
            expected = self.fixture.world[y:y+h, x+28:x+28+w]
            for observation in track["observations"]:
                with Image.open(self.out / observation["rgb"]) as image:
                    np.testing.assert_array_equal(np.asarray(image), expected)
                with Image.open(self.out / observation["support"]) as image:
                    self.assertTrue((np.asarray(image) == 255).all())
            self.assertEqual(track["delta_years"], [0] + list(np.diff([g["year"] for g in track["year_groups"]])))
        report = self.out / document["report"]
        record = read_json(report / "run.json")
        self.assertTrue(record["status"].startswith("completed"))
        self.assertEqual(record["manifest_sha256"], sha256(self.out / "manifest.json"))
        events = [json.loads(line) for line in (report / "events.jsonl").read_text().splitlines()]
        self.assertEqual([e["sequence"] for e in events], list(range(1, len(events)+1)))
        self.assertEqual({e["run_id"] for e in events}, {record["run_id"]})
        self.assertNotEqual(record["run_id"], report.name)
        self.assertTrue((report / "leakage_check.json").is_file())
        self.assertTrue((self.out / "gallery.html").is_file())

    def test_incremental_new_year_reuses_old_pixels_and_track_ids(self):
        self.prepare()
        old = read_json(self.out / "manifest.json")
        paths = {self.out / row["rgb"] for t in old["tracks"] for row in t["observations"]}
        before = {path: (path.stat().st_mtime_ns, sha256(path)) for path in paths}
        self.fixture.add(2026, 0, 112, alternative=True)
        self.fixture.save()
        summary = self.prepare()
        new = read_json(self.out / "manifest.json")
        self.assertEqual(summary["reused_frames"], 3)
        self.assertEqual(summary["computed_frames"], 1)
        self.assertEqual({t["track_id"] for t in old["tracks"]}, {t["track_id"] for t in new["tracks"]})
        self.assertEqual(before, {path: (path.stat().st_mtime_ns, sha256(path)) for path in paths})
        self.assertEqual(read_json(self.out / "reports/prepare-1/manifest.json"), old)

    def test_same_year_alternatives_are_preserved_not_extra_time_steps(self):
        self.fixture.add(2020, 0, 112, alternative=True)
        self.fixture.add(2009, 28, 84, alternative=True)
        self.fixture.save()
        summary = self.prepare()
        document = read_json(self.out / "manifest.json")
        track = next(t for t in document["tracks"] if t["distinct_years"] == 3)
        self.assertEqual(len(track["observations"]), 4)
        self.assertEqual(track["delta_years"], [0, 4, 7])
        self.assertEqual(len(track["year_groups"][-1]["observations"]), 2)
        self.assertEqual(summary["exclusion_reasons"]["same_year_anchor_alternative_without_geometry"], 1)

    def test_new_earlier_year_requires_new_anchor_geometry(self):
        self.prepare()
        old = read_json(self.out / "manifest.json")
        new_anchor = self.fixture.add(2005, 28, 84, alternative=True)
        self.fixture.save()
        summary = self.prepare()
        new = read_json(self.out / "manifest.json")
        self.assertEqual(new["views"][0]["anchor"]["sha256"], new_anchor["sha256"])
        self.assertEqual(summary["reused_frames"], 0)
        self.assertEqual(summary["exclusion_reasons"]["missing_direct_anchor_pair"], 3)
        self.assertFalse({t["track_id"] for t in old["tracks"]} & {t["track_id"] for t in new["tracks"]})

    def test_missing_direct_alignment_does_not_use_a_chain(self):
        self.fixture.pairs = self.fixture.pairs[:1]
        self.fixture.save()
        summary = self.prepare()
        self.assertEqual(summary["exclusion_reasons"]["missing_direct_anchor_pair"], 1)
        self.assertEqual(summary["histories_ge_3_years"]["tracks"], 0)

    def test_geometry_tampering_stops_before_export(self):
        geometry = self.root / "alignment-2/geometry.json"
        geometry.write_text('{"source_to_reference": [[1,0,99],[0,1,0],[0,0,1]]}')
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.prepare()
        self.assertFalse((self.out / "manifest.json").exists())

    def test_review_edits_require_applied_selection(self):
        review = self.fixture.session / "review.csv"
        with review.open() as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["decision"] = "reject"
        with review.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.prepare()
        self.assertFalse((self.out / "manifest.json").exists())

    def test_failed_parent_requires_matching_quality_snapshot_and_keeps_parent_unchanged(self):
        from test_alignment_session import quality_fixture
        quality = quality_fixture(self.fixture.session)
        path = self.fixture.session / "run.json"
        record = read_json(path)
        record["status"] = "failed"
        record.pop("artifact_sha256", None)
        write_json(path, record)
        original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "not completed"):
            self.prepare()
        result = self.prepare(quality_report=quality)
        self.assertEqual(result["computed_frames"], 3)
        self.assertEqual(path.read_bytes(), original)
        document = read_json(self.out / "manifest.json")
        self.assertEqual(document["alignment_session_completion"]["basis"], "explicit_quality_snapshot")
        self.assertEqual(document["quality_report"], str(quality))

    def test_building_split_leakage_stops_before_export(self):
        self.fixture.images[1]["split"] = "val"
        self.fixture.save()
        with self.assertRaisesRegex(ValueError, "[Ss]plit leakage"):
            self.prepare()
        self.assertFalse((self.out / "manifest.json").exists())
        self.assertEqual(read_json(self.out / "reports/prepare-1/run.json")["status"], "failed")

    def test_conflicting_duplicate_metadata_rejected(self):
        self.fixture.images.append({**self.fixture.images[0], "image_id": "999", "year": 2026})
        self.fixture.save()
        with self.assertRaisesRegex(ValueError, "conflicting identity/year"):
            self.prepare()

    def test_interruption_reuses_committed_observation_on_retry(self):
        original = hf.render_observation
        calls = []
        def interrupt(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise KeyboardInterrupt()
            return original(*args, **kwargs)
        with mock.patch.object(hf, "render_observation", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.prepare()
        self.assertEqual(read_json(self.out / "reports/prepare-1/run.json")["status"], "interrupted")
        self.assertFalse((self.out / "manifest.json").exists())
        result = self.prepare()
        self.assertEqual(result["reused_frames"], 1)
        self.assertEqual(result["computed_frames"], 2)

    def test_corrupt_committed_pixels_are_not_reused_or_overwritten(self):
        self.prepare()
        document = read_json(self.out / "manifest.json")
        path = self.out / document["tracks"][0]["observations"][0]["rgb"]
        path.write_bytes(b"corrupt")
        result = self.prepare()
        self.assertGreater(result["frame_statuses"].get("error", 0), 0)
        self.assertEqual(path.read_bytes(), b"corrupt")

    def test_plan_does_not_export_or_replace_existing_manifest(self):
        self.prepare()
        before = sha256(self.out / "manifest.json")
        result = self.prepare(plan_only=True)
        self.assertTrue(result["plan_only"])
        self.assertEqual(sha256(self.out / "manifest.json"), before)
        self.assertEqual(read_json(self.out / "reports/prepare-2/run.json")["status"], "planned")

    def test_publication_failure_preserves_previous_authoritative_manifest(self):
        self.prepare()
        previous = sha256(self.out / "manifest.json")
        gallery = hf._gallery
        def fail_on_public_gallery(path, *args, **kwargs):
            if path == self.out / "gallery.html":
                raise OSError("simulated publication failure")
            return gallery(path, *args, **kwargs)
        with mock.patch.object(hf, "_gallery", side_effect=fail_on_public_gallery):
            with self.assertRaisesRegex(OSError, "publication failure"):
                self.prepare()
        self.assertEqual(sha256(self.out / "manifest.json"), previous)
        # The previous index remains usable by the embedding stage.
        hf._history_document(self.out, hf.Fingerprints())

    def test_black_rgb_and_alpha_have_distinct_support(self):
        rgb = np.zeros((28, 28, 4), dtype=np.uint8)
        rgb[:, :, 3] = 255
        rgb[:14, :14, 3] = 0
        path = self.root / "rgba.png"
        Image.fromarray(rgb).save(path)
        row = {**self.fixture.images[0], "image_path": str(path), "sha256": sha256(path), "width": 28, "height": 28}
        frame = {"image": row, "alignment": {"matrix": np.eye(3).tolist()}}
        options = {"tile_size": 28, "stride": 28, "min_valid_fraction": .7,
                   "max_canvas_pixels": 10000, "max_canvas_side": 100}
        self.out.mkdir()
        record, _ = hf.render_observation(self.out, row, frame, options, {}, hf.Fingerprints())
        self.assertEqual(len(record["crops"]), 1)
        self.assertEqual(record["crops"][0]["valid_fraction"], .75)
        mask = np.zeros((28, 28), dtype=bool)
        mask[14:, :] = True
        mask[:14, 14:] = True
        np.testing.assert_array_equal(hf.supported_patches(mask), [False, True, True, True])

    def test_cli_plan(self):
        result = subprocess.run([sys.executable, "-s", "-m", "facade_change", "workflow", "prepare", "crop-histories",
                                 "--alignment-session", str(self.fixture.session), "--out", str(self.out), "--plan-only"],
                                cwd=hf.ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"registered_frames": 3', result.stdout)


TOY_HUB = '''import torch
from torch import nn
from torch.nn import functional as F
class Toy(nn.Module):
    patch_size = 14
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(384))
    def forward_features(self, x):
        assert not self.training
        assert not any(p.requires_grad for p in self.parameters())
        assert not torch.is_grad_enabled()
        tokens = F.avg_pool2d(x, 14, 14).flatten(2).transpose(1, 2).repeat(1, 1, 128) + self.weight
        return {"x_norm_patchtokens": tokens, "x_norm_clstoken": tokens.mean(dim=1)}
def dinov2_vits14(pretrained=False):
    assert pretrained is False
    return Toy()
'''


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional CPU torch is unavailable")
class FrozenFeatureTest(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = Fixture(self.root)
        self.fixture.add(2020, 0, 112, alternative=True)
        self.fixture.save()
        self.histories = self.root / "histories"
        self.features = self.histories / "features"
        self.silence = contextlib.redirect_stdout(io.StringIO())
        self.silence.__enter__()
        self.addCleanup(self.silence.__exit__, None, None, None)
        hf.prepare_histories(self.fixture.session, self.histories, tile_size=28, stride=28, min_valid_fraction=1.)
        source = self.root / "toy_dino"
        source.mkdir()
        (source / "hubconf.py").write_text(TOY_HUB)
        checkpoint = self.root / "toy.pth"
        torch.save({"weight": torch.arange(384, dtype=torch.float32) / 384 + 1}, checkpoint)
        self.config = self.root / "methods.json"
        write_json(self.config, {"methods": {"dinov2": {"worker_python": sys.executable,
                                                       "source_root": str(source), "checkpoint_path": str(checkpoint)}}})
        self.settings = hf.read_dino_config(self.config)

    def extract(self, **kwargs):
        return hf.extract_features(self.histories, self.features, self.settings, device="cpu", batch_size=3, **kwargs)

    def test_offline_pipeline_time_arrays_and_exact_cache_reuse(self):
        with mock.patch("torch.hub.download_url_to_file", side_effect=AssertionError("Download forbidden")):
            result = self.extract(save_patch_tokens=True)
        self.assertGreater(result["computed_features"], 0)
        index = read_json(self.features / "index.json")
        sequence = next(x for x in index["sequences"] if x["distinct_years"] == 3)
        with np.load(self.features / sequence["path"], allow_pickle=False) as arrays:
            self.assertEqual(arrays["z_observations"].shape, (4, 384))
            self.assertEqual(arrays["z_years"].shape, (3, 384))
            np.testing.assert_array_equal(arrays["years"], [2009, 2013, 2020])
            np.testing.assert_array_equal(arrays["delta_years"], [0, 4, 7])
            np.testing.assert_array_equal(arrays["elapsed_years"], [0, 4, 11])
            np.testing.assert_array_equal(arrays["year_counts"], [1, 1, 2])
            np.testing.assert_allclose(np.linalg.norm(arrays["z_years"], axis=1), 1, atol=1e-6)
        artifacts = {self.features / row["path"] for row in index["features"].values()}
        before = {p: p.stat().st_mtime_ns for p in artifacts}
        with mock.patch.object(hf, "DinoEncoder", side_effect=AssertionError("All features should be cached")):
            again = self.extract(save_patch_tokens=True)
        self.assertEqual(again["computed_features"], 0)
        self.assertEqual(again["reused_features"], result["computed_features"])
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in artifacts})
        self.assertFalse(result["trained_temporal_embedding"])
        self.assertFalse(result["event_labels_created"])

    def test_masked_pooling_uses_only_complete_patches(self):
        contract = hf.encoder_contract(self.settings, "cpu", True, hf.Fingerprints())
        encoder = hf.DinoEncoder(self.settings, "cpu", contract)
        self.addCleanup(encoder.close)
        image = np.zeros((28, 28, 3), dtype=np.uint8)
        image[:, 14:, :] = [100, 140, 200]
        mask = np.ones((28, 28), dtype=bool)
        mask[:14, 14:] = False
        feature = encoder.encode([image], [mask], True)[0]
        self.assertEqual(feature["z"].shape, (384,))
        self.assertEqual(feature["patch_tokens"].shape, (4, 384))
        np.testing.assert_array_equal(feature["patch_support"], [True, False, True, True])
        normalized = (image.astype(np.float32)/255 - np.asarray(hf.NORMALIZATION["mean"])) / hf.NORMALIZATION["std"]
        pixels_mean = normalized[mask].mean(axis=0)
        expected = np.tile(pixels_mean, 128) + np.arange(384)/384 + 1
        np.testing.assert_allclose(feature["z"], hf._l2(expected), atol=1e-6)

    def test_checkpoint_change_creates_new_features_without_overwriting_old(self):
        import torch
        first = self.extract()
        index = read_json(self.features / "index.json")
        old = {row["path"]: sha256(self.features / row["path"]) for row in index["features"].values()}
        torch.save({"weight": torch.arange(384, dtype=torch.float32) / 384 + 2}, self.settings["checkpoint_path"])
        second = self.extract()
        self.assertEqual(second["reused_features"], 0)
        self.assertEqual(second["computed_features"], first["computed_features"])
        self.assertEqual(old, {path: sha256(self.features / path) for path in old})

    def test_bad_crop_hash_fails_before_model_allocation_and_preserves_index(self):
        self.extract()
        before = sha256(self.features / "index.json")
        document = read_json(self.histories / "manifest.json")
        row = next(t for t in document["tracks"] if t["distinct_years"] >= 2)["observations"][0]
        (self.histories / row["rgb"]).write_bytes(b"changed")
        with mock.patch.object(hf, "DinoEncoder", side_effect=AssertionError("Must fail in preflight")):
            with self.assertRaisesRegex(ValueError, "Hash mismatch"):
                self.extract()
        self.assertEqual(sha256(self.features / "index.json"), before)
        self.assertEqual(read_json(self.features / "reports/embed-2/run.json")["status"], "failed")

    def test_feature_corruption_is_not_overwritten(self):
        self.extract()
        index = read_json(self.features / "index.json")
        path = self.features / next(iter(index["features"].values()))["path"]
        path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Corrupt frozen-feature"):
            self.extract()
        self.assertEqual(path.read_bytes(), b"corrupt")

    def test_relative_worker_preserves_venv_symlink(self):
        python = self.root / "venv/bin/python"
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        config = read_json(self.config)
        config["methods"]["dinov2"]["worker_python"] = "venv/bin/python"
        write_json(self.config, config)
        settings = hf.read_dino_config(self.config)
        self.assertEqual(settings["worker_python"], str(python))
        self.assertNotEqual(settings["worker_python"], str(python.resolve()))

    def test_parent_resolves_relative_inputs_before_worker_changes_directory(self):
        with mock.patch("os.getcwd", return_value=str(self.root)):
            result = hf.main(["embed", "--histories", "histories", "--methods-config", "methods.json",
                              "--device", "cpu", "--batch-size", "3"])
        self.assertEqual(result, 0)
        self.assertTrue((self.features / "index.json").is_file())

    def test_interrupted_embeddings_resume_committed_units(self):
        original = hf.DinoEncoder.encode
        calls = []
        def interrupt(encoder, *args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise KeyboardInterrupt()
            return original(encoder, *args, **kwargs)
        with mock.patch.object(hf.DinoEncoder, "encode", interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.extract()
        self.assertFalse((self.features / "index.json").exists())
        self.assertEqual(read_json(self.features / "reports/embed-1/run.json")["status"], "interrupted")
        result = self.extract()
        self.assertEqual(result["reused_features"], 3)
        self.assertGreater(result["computed_features"], 0)

    def test_worker_cli_uses_configured_interpreter(self):
        result = subprocess.run([sys.executable, "-s", "-m", "facade_change", "workflow", "prepare", "history-features",
                                 "--histories", str(self.histories), "--methods-config", str(self.config),
                                 "--device", "cpu", "--batch-size", "3"],
                                cwd=hf.ROOT, text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DINOv2 worker: " + sys.executable, result.stdout)
        self.assertTrue((self.features / "index.json").is_file())


if __name__ == "__main__":
    unittest.main()
