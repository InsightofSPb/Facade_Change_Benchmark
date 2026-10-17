"""Small authenticated real-VAL gallery fixtures; no model or dataset inference."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


from scripts.real_eval import real_temporal_gallery as gallery

SCRIPT = Path(gallery.__file__)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(root):
    return {str(path.relative_to(root)): sha(path) for path in root.rglob("*") if path.is_file()}


def fixture(directory):
    root, snapshot = directory / "run", directory / "snapshot"
    (root / "maps").mkdir(parents=True)
    snapshot.mkdir()
    records = []
    for building in ("A", "B", "C"):
        for name, (method, variant, mode) in gallery.GROUPS.items():
            if name == "msdzip_mod256_history" and building != "A":
                continue
            for frame in range(1, 4):
                if building == "C" and name == "geoscd" and frame == 2:
                    continue
                support = np.ones((3, 4), bool)
                support[-1, -1] = False
                scores = np.array([[.2, .5, .8, .9], [.1, .3, .7, .6], [.5, .4, .9, np.nan]], np.float32)
                native = name in {"geoscd", "anychange"}
                if native:
                    scores[support] = 0  # Native prediction must override these scores.
                reference = (np.arange(36).reshape(3, 4, 3) + 20 * (frame - 1)).astype(np.uint8)
                source = (np.arange(36).reshape(3, 4, 3) + 20 * frame).astype(np.uint8)
                arrays = dict(scores=scores, support=support, reference=reference, source=source,
                              labels=np.array([{"never": "read unverified labels"}], object),
                              memory__poison=np.array([{"never": "read model memory"}], object))
                if native:
                    arrays["native_prediction"] = np.array([[True, False, True, False],
                                                           [False, True, False, True], [True, False, True, True]])
                identifier = f"{building}-{name}-{frame}"
                path = root / "maps" / f"{identifier}.npz"
                np.savez_compressed(path, **arrays)
                records.append(dict(unit_id=identifier, method=method, variant=variant, mode=mode,
                                    target="step_event", domain="real", split="val", building_id=building,
                                    sequence_id=f"{building}-view-x0-y0", frame=frame, status="completed",
                                    artifact_path=str(path.relative_to(root)), artifact_sha256=sha(path)))
    # These records must be filtered before artifact loading or gallery selection.
    for domain, split in (("real", "test"), ("real", "train"), ("synthetic", "val")):
        records.append(dict(records[0], unit_id=f"FORBIDDEN-{domain}-{split}",
                            domain=domain, split=split, building_id="FORBIDDEN",
                            sequence_id=f"FORBIDDEN-{domain}-{split}", artifact_path="missing.npz"))
    calibration = []
    for name in ("ssim", "msdzip_abs", "msdzip_mod256", "msdzip_mod256_history"):
        method, variant, mode = gallery.GROUPS[name]
        calibration.append(dict(method=method, variant=variant, mode=mode, target="step_event", status="calibrated",
                                selection_split="val", selection_domain="synthetic", selection_arm="clean",
                                choices=[dict(fpr_target=.01, threshold=.5),
                                         dict(fpr_target=.05, threshold=.75 if mode == "history" else .4)]))
    saved = dict(source_run=str(root.resolve()), calibration=calibration,
                 method_status_snapshot={"rgb_diff": {"status": "completed"}})
    (snapshot / "2026-10-17_source_units.json").write_text(json.dumps(records))
    (snapshot / "2026-10-17_quick_summary.json").write_text(json.dumps(saved))
    return root, snapshot, records


class RealTemporalGalleryTests(unittest.TestCase):
    def test_whole_histories_frozen_masks_and_readonly_sources(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            root, snapshot, _ = fixture(directory)
            before = hashes(root), hashes(snapshot)
            out = directory / "gallery"
            with contextlib.redirect_stdout(io.StringIO()):
                result = gallery.run(root, snapshot, out)
            self.assertEqual(result["selected_sequences"], ["A-view-x0-y0", "B-view-x0-y0", "C-view-x0-y0"])
            self.assertEqual(len(result["results"]), 9)
            for building in ("A", "B", "C"):
                self.assertEqual([row["frame"] for row in result["results"] if row["building_id"] == building], [1, 2, 3])
            coverage = {row["building_id"]: row for row in result["coverage"]}
            self.assertEqual(set(coverage), {"A", "B", "C"})
            self.assertTrue(coverage["C"]["selected"])
            self.assertIn("geoscd", coverage["C"]["missing_required"])
            self.assertEqual(coverage["C"]["method_coverage"]["geoscd"],
                             dict(completed_frames=2, expected_frames=3, missing_frames=[2], status="partial"))
            self.assertEqual(coverage["B"]["history_status"], "missing_or_incomplete")
            self.assertEqual(result["thresholds"]["ssim"]["threshold"], .5)
            self.assertEqual(result["thresholds"]["msdzip_abs"]["threshold"], .4)
            self.assertEqual(result["thresholds"]["msdzip_mod256_history"]["threshold"], .75)
            first = result["results"][0]
            with np.load(root / "maps/A-ssim-1.npz", allow_pickle=False) as archive:
                scores, support, source = (archive[key] for key in ("scores", "support", "source"))
            with np.load(root / "maps/A-geoscd-1.npz", allow_pickle=False) as archive:
                native = archive["native_prediction"] & support
            expected = dict(ssim=(scores > np.float32(.5)) & support,
                            msdzip_abs=(scores > np.float32(.4)) & support,
                            msdzip_mod256_history=(scores > np.float32(.75)) & support,
                            geoscd=native, geoscd_and_ssim=native & (scores > np.float32(.5)))
            for method, prediction in expected.items():
                with Image.open(out / first["mask_paths"][method]) as picture:
                    mask = np.array(picture)
                np.testing.assert_array_equal(mask, np.where(support, prediction.astype(np.uint8) * 255, 128))
            with Image.open(out / first["image_paths"]["ssim"]) as picture:
                overlay = np.array(picture)
            expected_overlay = source.copy()
            expected_overlay[expected["ssim"]] = (.45 * source[expected["ssim"]] + .55 * np.array([255, 0, 255])).astype(np.uint8)
            expected_overlay[~support] = 128
            np.testing.assert_array_equal(overlay, expected_overlay)
            for row in result["results"]:
                if row["building_id"] == "B":
                    self.assertNotIn("msdzip_mod256_history", row["mask_paths"])
                    self.assertEqual(row["history_status"], "missing_or_incomplete")
                if row["building_id"] == "C" and row["frame"] == 2:
                    self.assertEqual(row["method_status"]["geoscd"], "missing_result")
                    self.assertNotIn("geoscd", row["mask_paths"])
                    self.assertFalse(any(key.startswith("geoscd_and_") for key in row["mask_paths"]))
                    self.assertIn("anychange_and_ssim", row["mask_paths"])
            self.assertNotIn("FORBIDDEN", (out / "gallery.html").read_text())
            self.assertEqual((hashes(root), hashes(snapshot)), before)
            output_before = hashes(out)
            with self.assertRaises(FileExistsError):
                gallery.run(root, snapshot, out)
            self.assertEqual(hashes(out), output_before)
            with self.assertRaises(FileExistsError):
                gallery.run(root, snapshot, root / "gallery")

    def test_incompatible_optional_history_is_disclosed_without_dropping_pairs(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            root, snapshot, records = fixture(directory)
            record = next(row for row in records if row["unit_id"] == "A-msdzip_mod256_history-2")
            path = root / record["artifact_path"]
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in ("scores", "support", "reference", "source")}
            arrays["support"][0, 0] = False
            np.savez_compressed(path, **arrays)
            record["artifact_sha256"] = sha(path)
            (snapshot / "2026-10-17_source_units.json").write_text(json.dumps(records))
            with contextlib.redirect_stdout(io.StringIO()):
                result = gallery.run(root, snapshot, directory / "gallery")
            self.assertEqual(len(result["results"]), 9)
            changed = next(row for row in result["results"] if row["building_id"] == "A" and row["frame"] == 2)
            self.assertTrue(changed["history_status"].startswith("not_comparable"))
            self.assertNotIn("msdzip_mod256_history", changed["mask_paths"])
            self.assertNotIn("geoscd_and_msdzip_mod256_history", changed["mask_paths"])
            self.assertIn("geoscd_and_msdzip_mod256", changed["mask_paths"])

    def test_absent_geoscd_keeps_all_frames_and_available_native_fusions(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            root, snapshot, records = fixture(directory)
            records = [row for row in records if not (row["domain"] == "real" and row["split"] == "val"
                       and row["method"] == "geoscd")]
            (snapshot / "2026-10-17_source_units.json").write_text(json.dumps(records))
            before = hashes(root), hashes(snapshot)
            out = directory / "gallery"
            with contextlib.redirect_stdout(io.StringIO()):
                result = gallery.run(root, snapshot, out)
            self.assertEqual(len(result["results"]), 9)
            self.assertTrue(all(row["selected"] for row in result["coverage"]))
            for coverage in result["coverage"]:
                self.assertEqual(coverage["method_coverage"]["geoscd"],
                                 dict(completed_frames=0, expected_frames=3, missing_frames=[1, 2, 3], status="missing"))
            for row in result["results"]:
                self.assertEqual(row["method_status"]["geoscd"], "missing_result")
                self.assertFalse(any(key.startswith("geoscd") for key in row["mask_paths"]))
                self.assertFalse(any(key.startswith("geoscd") for key in row["image_paths"]))
                self.assertIn("anychange_and_ssim", row["mask_paths"])
                self.assertIn("anychange_and_msdzip_abs", row["mask_paths"])
            self.assertFalse(list((out / "images").glob("*geoscd*")), "Missing GeoSCD must not be represented by zero masks")
            self.assertEqual((hashes(root), hashes(snapshot)), before)

    def test_partial_history_retains_its_available_frames(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            root, snapshot, records = fixture(directory)
            records = [row for row in records if row["unit_id"] != "A-msdzip_mod256_history-2"]
            (snapshot / "2026-10-17_source_units.json").write_text(json.dumps(records))
            with contextlib.redirect_stdout(io.StringIO()):
                result = gallery.run(root, snapshot, directory / "gallery")
            a_rows = [row for row in result["results"] if row["building_id"] == "A"]
            self.assertEqual([row["frame"] for row in a_rows], [1, 2, 3])
            a_coverage = next(row for row in result["coverage"] if row["building_id"] == "A")
            self.assertEqual(a_coverage["method_coverage"]["msdzip_mod256_history"],
                             dict(completed_frames=2, expected_frames=3, missing_frames=[2], status="partial"))
            for row in a_rows:
                if row["frame"] == 2:
                    self.assertEqual(row["method_status"]["msdzip_mod256_history"], "missing_or_incomplete")
                    self.assertNotIn("msdzip_mod256_history", row["mask_paths"])
                    self.assertNotIn("geoscd_and_msdzip_mod256_history", row["mask_paths"])
                else:
                    self.assertEqual(row["history_status"], "complete")
                    self.assertIn("msdzip_mod256_history", row["mask_paths"])
                    self.assertIn("geoscd_and_msdzip_mod256_history", row["mask_paths"])

    def test_no_real_val_or_no_contiguous_baseline_writes_diagnostic_without_fallback(self):
        for case in ("no_real_val", "broken_baseline"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as name:
                directory = Path(name)
                root, snapshot, records = fixture(directory)
                if case == "no_real_val":
                    records = [row for row in records if (row["domain"], row["split"]) != ("real", "val")]
                else:
                    records = [row for row in records if not (row["domain"] == "real" and row["split"] == "val"
                               and row["method"] == "rgb_diff" and row["frame"] == 2)]
                (snapshot / "2026-10-17_source_units.json").write_text(json.dumps(records))
                before = hashes(root), hashes(snapshot)
                out = directory / "diagnostic"
                with contextlib.redirect_stdout(io.StringIO()):
                    result = gallery.run(root, snapshot, out)
                self.assertEqual(result["selected_sequences"], [])
                self.assertEqual(result["results"], [])
                self.assertEqual(result["status"], "diagnostic_no_real_val_baseline")
                self.assertTrue(result["diagnostic"])
                self.assertEqual(result["domain_split_counts"].get("real/test"), 1)
                self.assertEqual(result["domain_split_counts"].get("real/train"), 1)
                self.assertEqual(result["domain_split_counts"].get("synthetic/val"), 1)
                if case == "no_real_val":
                    self.assertEqual(result["domain_split_counts"].get("real/val", 0), 0)
                for filename in ("gallery.html", "summary.json", "coverage.csv", "provenance.json"):
                    self.assertTrue((out / filename).is_file())
                self.assertFalse(list(out.rglob("*.png")))
                self.assertEqual((hashes(root), hashes(snapshot)), before)

    def test_hash_mismatch_and_incompatible_primary_support_are_rejected(self):
        for change in ("hash", "support"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as name:
                directory = Path(name)
                root, snapshot, records = fixture(directory)
                record = next(row for row in records if row["unit_id"] == "A-msdzip_abs-1")
                path = root / record["artifact_path"]
                if change == "hash":
                    path.write_bytes(path.read_bytes() + b"unauthenticated modification")
                    message = "hash mismatch"
                else:
                    with np.load(path, allow_pickle=False) as archive:
                        arrays = {key: archive[key] for key in ("scores", "support", "reference", "source")}
                    arrays["support"][0, 0] = False
                    np.savez_compressed(path, **arrays)
                    record["artifact_sha256"] = sha(path)
                    (snapshot / "2026-10-17_source_units.json").write_text(json.dumps(records))
                    message = "RGB/support mismatch"
                with self.assertRaisesRegex(ValueError, message):
                    gallery.run(root, snapshot, directory / "gallery")


class GeoRetryOverlayTests(unittest.TestCase):
    def test_extra_geo_pair_is_verified_and_rendered_without_changing_sources(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            root, snapshot, records = fixture(directory)
            extra = directory / "retry"
            (extra / "units").mkdir(parents=True)
            (extra / "maps").mkdir()
            original = next(row for row in records if row["unit_id"] == "A-geoscd-2")
            map_path = extra / "maps/retry-C-2.npz"
            map_path.write_bytes((root / original["artifact_path"]).read_bytes())
            row = dict(original, unit_id="retry-C-2", sequence_id="C-view-x0-y0", building_id="C",
                       artifact_path="maps/retry-C-2.npz", artifact_sha256=sha(map_path))
            (extra / "units/retry-C-2.json").write_text(json.dumps(row))
            contract = {"source_run": str(root),
                        "snapshot_source_units_sha256": sha(snapshot / "2026-10-17_source_units.json"),
                        "snapshot_quick_summary_sha256": sha(snapshot / "2026-10-17_quick_summary.json")}
            (extra / "run.json").write_text(json.dumps(contract))
            before = hashes(root), hashes(snapshot), hashes(extra)
            with contextlib.redirect_stdout(io.StringIO()):
                summary = gallery.run(root, snapshot, directory / "gallery", geoscd_run=extra)
            target = next(row for row in summary["results"] if row["building_id"] == "C" and row["frame"] == 2)
            self.assertIn("geoscd_and_msdzip_mod256", target["mask_paths"])
            self.assertIn("retry-C-2", target["source_unit_ids"])
            self.assertEqual(summary["geoscd_overlay"]["completed_previous_pairs"], 1)
            self.assertEqual((hashes(root), hashes(snapshot), hashes(extra)), before)
            with self.assertRaises(FileExistsError):
                gallery.run(root, snapshot, extra / "gallery", geoscd_run=extra)
            contract["snapshot_source_units_sha256"] = "not-the-source"
            (extra / "run.json").write_text(json.dumps(contract))
            with self.assertRaisesRegex(ValueError, "another frozen snapshot"):
                gallery.run(root, snapshot, directory / "wrong", geoscd_run=extra)


if __name__ == "__main__":
    unittest.main()
