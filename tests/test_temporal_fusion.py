"""Saved-map fusion checks: fixed decisions, matching targets, and facade metrics."""
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


from scripts.real_eval import temporal_fusion as fusion

SCRIPT = Path(fusion.__file__)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(root):
    return {str(path.relative_to(root)): file_hash(path)
            for path in root.rglob("*") if path.is_file()}


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture(directory):
    root = directory / "run"
    (root / "maps").mkdir(parents=True)
    rows = []
    for method in ("geoscd", "anychange", "msdzip_abs", "msdzip_mod256", "dinov2", "ssim"):
        native = method in ("geoscd", "anychange")
        modes = ["previous", "earliest"]
        if method in ("msdzip_abs", "msdzip_mod256", "dinov2"):
            modes.append("history")
        for mode in modes:
            for arm in ("clean", "event"):
                for frame in range(1, 5):
                    positive = arm == "event" and (frame != 3 or mode == "earliest")
                    labels = np.array([[1, 1, 0, 0, 255] if positive else [0, 0, 0, 0, 255]], np.uint8)
                    support = np.array([[True, True, True, True, False]])
                    # Base scores intentionally contradict the saved native mask.
                    scores = np.array([[0., 0., 0., 0., np.nan] if native
                                       else [.9, .9, .5, .9, np.nan]], np.float32)
                    arrays = dict(scores=scores, labels=labels, support=support,
                                  reference=np.array([{"do not": "load RGB"}], object),
                                  memory__poison=np.array([{"do not": "load memory"}], object))
                    if native:
                        arrays["native_prediction"] = np.array([[True, False, True, False, True]])
                    identifier = f"{method}-{mode}-{arm}-{frame}"
                    path = root / "maps" / f"{identifier}.npz"
                    np.savez_compressed(path, **arrays)
                    for budget in ([None] if native else [.01, .05]):
                        threshold = "" if native else (.5 if budget == .01 else .4)
                        prediction = (arrays["native_prediction"] if native else scores > np.float32(threshold)) & support
                        row = dict(unit_id=identifier, method=method,
                                   variant="current_train" if method.startswith("msdzip_") else "frozen",
                                   mode=mode, target="earliest_state" if mode == "earliest" else "step_event",
                                   domain="synthetic", split="val", building_id="facade", sequence_id=arm,
                                   frame=frame, arm=arm, hypothesis="H0" if arm == "clean" else "H1",
                                   decision="author_native" if native else "val_calibrated",
                                   fpr_target="" if native else budget, threshold=threshold,
                                   artifact_path=str(path.relative_to(root)), artifact_sha256=file_hash(path))
                        positive_mask, negative_mask = labels == 1, labels == 0
                        row.update(tp=int((prediction & positive_mask).sum()),
                                   fp=int((prediction & negative_mask).sum()),
                                   fn=int((~prediction & positive_mask).sum()),
                                   tn=int((~prediction & negative_mask).sum()),
                                   evaluated_pixel_count=int((labels != 255).sum()))
                        rows.append(row)
    units = directory / "quick_units.csv"
    save_csv(units, rows)
    return root, units, rows


class TemporalFusionTests(unittest.TestCase):
    def test_counts_ignore_unlabelled_pixels_and_aggregate_all_sequence_frames(self):
        counted = fusion.counts(np.array([[True, False, True, False, True]]),
                                np.array([[1, 1, 0, 0, 255]], np.uint8))
        self.assertEqual(counted, dict(tp=1, fp=1, fn=1, tn=1, evaluated_pixel_count=4))
        rows = []
        for building, arm, frame, values in (
                ("A", "event", 2, (1, 0, 1, 2)),
                ("A", "event", 3, (0, 6, 0, 2)),
                ("A", "clean", 3, (0, 2, 0, 2)),
                ("B", "event", 2, (3, 1, 0, 0)),
                ("B", "event", 3, (0, 0, 0, 4)),
                ("B", "clean", 3, (0, 0, 0, 4))):
            rows.append(dict(building_id=building, arm=arm, frame=frame, target="step_event",
                             hypothesis="H0" if arm == "clean" else "H1",
                             **dict(zip(("tp", "fp", "fn", "tn"), values)),
                             evaluated_pixel_count=sum(values)))
        result = fusion.aggregate(rows)
        self.assertEqual((result["n_facades"], result["n_units"]), (2, 6))
        self.assertAlmostEqual(result["precision"], (1 + 3 / 4) / 2)
        self.assertAlmostEqual(result["all_precision"], (1 / 9 + 3 / 4) / 2)
        self.assertAlmostEqual(result["all_f1"], (2 / 11 + 6 / 7) / 2)
        self.assertAlmostEqual(result["all_recall"], (.5 + 1.) / 2)
        self.assertAlmostEqual(result["event_precision"], (1 / 7 + 3 / 4) / 2)
        self.assertAlmostEqual(result["event_f1"], (2 / 9 + 6 / 7) / 2)
        self.assertAlmostEqual(result["h0_fpr"], .25)
        self.assertAlmostEqual(result["no_event_fpr"], .375)
        self.assertAlmostEqual(result["persistent_fpr"], .375)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (4, 9, 1))
        self.assertNotAlmostEqual(result["all_precision"], 4 / 13,
                                  msg="Each facade must have equal weight, not each pixel")

    def test_fixed_fusion_native_masks_thresholds_readonly_inputs_and_output_guard(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            root, units, _ = fixture(directory)
            before, csv_before = snapshot(root), file_hash(units)
            args = SimpleNamespace(run=root, units=units, out=directory / "fusion", budgets=[.01, .05])
            with contextlib.redirect_stdout(io.StringIO()):
                summary = fusion.run(args)
            with (args.out / "units.csv").open(encoding="utf-8-sig", newline="") as stream:
                output = list(csv.DictReader(stream))
            by_key = {(row["name"], row["mode"], row["aux_budget"], row["sequence_id"], row["frame"]): row
                      for row in output}
            for base in ("geoscd", "anychange"):
                for aux in ("msdzip_abs", "msdzip_mod256", "dinov2", "ssim"):
                    for mode in ("previous", "earliest", *([] if aux == "ssim" else ["history_addon"])):
                        for operation, expected in (("AND", (1, 0, 1, 2)), ("OR", (2, 2, 0, 0))):
                            row = by_key[(f"{base} {operation} {aux}", mode, "0.01", "event", "1")]
                            self.assertEqual(tuple(int(row[k]) for k in ("tp", "fp", "fn", "tn")), expected)
                            self.assertEqual(float(row["right_threshold"]), .5)
                            if mode == "history_addon":
                                self.assertIn("-previous-", row["left_unit_id"])
                                self.assertIn("-history-", row["right_unit_id"])
                            elif mode == "earliest":
                                self.assertEqual(row["target"], "earliest_state")
            lower = by_key[("geoscd AND ssim", "previous", "0.05", "event", "1")]
            self.assertEqual(tuple(int(lower[k]) for k in ("tp", "fp", "fn", "tn")), (1, 1, 1, 1))
            self.assertEqual(float(lower["right_threshold"]), .4)
            native = by_key[("geoscd", "previous", "", "event", "1")]
            self.assertEqual((int(native["tp"]), int(native["fp"])), (1, 1))
            self.assertFalse(any(row["mode"] == "history_addon" and "ssim" in row["name"] for row in output))
            self.assertTrue(summary)
            self.assertEqual(snapshot(root), before)
            self.assertEqual(file_hash(units), csv_before)
            out_before = snapshot(args.out)
            with self.assertRaises(FileExistsError):
                fusion.run(args)
            self.assertEqual(snapshot(args.out), out_before)

    def test_rejects_changed_hash_and_mismatched_labels_or_support(self):
        for change in ("hash", "labels", "support"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as name:
                directory = Path(name)
                root, units, rows = fixture(directory)
                identifier = "msdzip_abs-history-event-1"
                path = root / "maps" / f"{identifier}.npz"
                if change == "hash":
                    path.write_bytes(path.read_bytes() + b"changed artifact")
                    expected = "hash mismatch"
                else:
                    with np.load(path, allow_pickle=False) as archive:
                        arrays = {key: archive[key] for key in ("scores", "labels", "support")}
                    if change == "labels":
                        arrays["labels"][0, 0] = 0
                    else:
                        arrays["support"][0, -1] = True
                        arrays["scores"][0, -1] = 0
                    np.savez_compressed(path, **arrays)
                    for row in rows:
                        if row["unit_id"] == identifier:
                            row["artifact_sha256"] = file_hash(path)
                            pred = (arrays["scores"] > np.float32(row["threshold"])) & arrays["support"]
                            row.update(fusion.counts(pred, arrays["labels"]))
                    save_csv(units, rows)
                    expected = "different " + change
                args = SimpleNamespace(run=root, units=units, out=directory / "fusion", budgets=[.01])
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, expected):
                    fusion.run(args)
                self.assertFalse(args.out.exists(), "A rejected analysis should not publish partial tables")

    def test_default_cli_budget_is_one_percent_and_source_output_is_forbidden(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            root, units, _ = fixture(directory)
            forbidden = SimpleNamespace(run=root, units=units, out=root / "fusion", budgets=[.01])
            with self.assertRaisesRegex(ValueError, "outside"):
                fusion.run(forbidden)
            out = directory / "cli-report"
            result = subprocess.run([sys.executable, str(SCRIPT), "--run", str(root), "--units", str(units),
                                     "--out", str(out)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            provenance = json.loads((out / "provenance.json").read_text())
            self.assertEqual(provenance["budgets"], [.01])
            with (out / "metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
                points = {row["aux_budget"] for row in csv.DictReader(stream)}
            self.assertEqual(points, {"", "0.01"})

    def test_coverage_requires_all_four_transition_indices_and_skips_partial_groups(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            root, units, rows = fixture(directory)
            # The number of rows is unchanged, but an invalid frame cannot certify coverage.
            damaged = [dict(row) for row in rows]
            for row in damaged:
                if row["method"] == "geoscd" and row["mode"] == "previous" and row["frame"] == 4:
                    row["frame"] = 5
            save_csv(units, damaged)
            args = SimpleNamespace(run=root, units=units, out=directory / "rejected", budgets=[.01])
            with self.assertRaisesRegex(ValueError, "indices 1..4"):
                fusion.run(args)
            self.assertFalse(args.out.exists())
            partial = [row for row in rows if not (row["method"] == "msdzip_abs"
                       and row["mode"] == "history" and row["arm"] == "event" and row["frame"] == 4)]
            save_csv(units, partial)
            args.out = directory / "partial"
            with contextlib.redirect_stdout(io.StringIO()):
                summary = fusion.run(args)
            self.assertFalse(any("msdzip_abs" in row["name"] and row["mode"] in {"history", "history_addon"}
                                 for row in summary))
            with (args.out / "coverage.csv").open(encoding="utf-8-sig", newline="") as stream:
                coverage = list(csv.DictReader(stream))
            self.assertTrue(any(row["status"] == "incomplete_skipped" and "msdzip_abs" in row["item"]
                                for row in coverage))


if __name__ == "__main__":
    unittest.main()
