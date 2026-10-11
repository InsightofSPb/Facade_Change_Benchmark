"""CPU research contracts against an independently sealed native-RGB dataset."""
from __future__ import annotations

import builtins
import ast
import copy
import csv
import hashlib
import importlib
import importlib.util
import io
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
from PIL import Image

from facade_change.io import sha256


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _seal(root, kind, status, config):
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    record = {"schema_version": 1, "kind": kind, "status": status,
              "config": config, "config_sha256": hashlib.sha256(payload).hexdigest(),
              "artifact_sha256": {p.relative_to(root).as_posix(): sha256(p)
                                  for p in sorted(root.rglob("*"))
                                  if p.is_file() and p != root / "run.json"}}
    _write(root / "run.json", record)
    return record


def _fingerprint(dataset):
    return {"parent_run": sha256(dataset / "run.json"),
            "parent_summary": sha256(dataset / "summary.json"),
            "parent_index": sha256(dataset / "index.json"),
            "split": sha256(dataset / "split.json"),
            "config": sha256(dataset / "config.json")}


def _snapshot(*roots):
    return {str(p): p.read_bytes() for root in roots for p in root.rglob("*") if p.is_file()}


def make_dataset(root):
    """TRAIN/VAL/TEST: two buildings and two native-RGB views per building."""
    dataset = root / "dataset"
    dataset.mkdir()
    split = {"mode": "reviewed", "development_only": False,
             "building_assignments": {f"{part}-facade-{building}": part
                                      for part in ("train", "val", "test")
                                      for building in range(2)}}
    scenarios = [{"id": "clean", "kind": "exposure"}, {"id": "blur", "kind": "blur"}]
    dataset_config = {"states": ["unchanged", "crack", "paint_patch"], "scenarios": scenarios}
    _write(dataset / "config.json", dataset_config)
    _write(dataset / "split.json", split)
    bases, cases = [], []
    support = np.ones((4, 4), dtype=bool)
    reference = np.full((4, 4, 3), 80, dtype=np.uint8)
    for part, building, view in itertools.product(("train", "val", "test"), range(2), range(2)):
        base_id = f"{part}-b{building}-v{view}"
        base = {"base_id": base_id, "building_id": f"{part}-facade-{building}",
                "view_id": f"{part}-view-{building}-{view}", "split": part,
                "dataset_crop_id": base_id + "-crop", "path": "bases/" + base_id}
        bases.append(base)
        folder = dataset / base["path"]
        folder.mkdir(parents=True)
        Image.fromarray(reference).save(folder / "reference_rgb.png")
        Image.fromarray(support.astype(np.uint8) * 255).save(folder / "reference_support.png")
        for state in dataset_config["states"]:
            for scenario in scenarios:
                identifier = f"{base['base_id']}-{state}-{scenario['id']}"
                prefix = "cases/" + identifier
                folder = dataset / prefix
                folder.mkdir(parents=True)
                edit = np.zeros((4, 4), dtype=bool)
                if state != "unchanged":
                    edit.flat[:4] = True
                source = reference.copy()
                source[edit] = 180
                labels = edit.astype(np.uint8)
                Image.fromarray(source).save(folder / "source_rgb.png")
                Image.fromarray(labels).save(folder / "labels_reference.png")
                for name, values in (
                    ("source_support", support), ("true_visibility", support),
                    ("comparable", support), ("source_occlusion", np.zeros_like(support)),
                    ("true_edit_mask_reference", edit), ("visible_edit_mask_reference", edit),
                ):
                    Image.fromarray(values.astype(np.uint8) * 255).save(folder / (name + ".png"))
                identity = np.eye(3).tolist()
                geometry = {"source_to_reference": identity, "reference_to_source": identity}
                _write(folder / "geometry.json", geometry)
                paths = {"reference_rgb": base["path"] + "/reference_rgb.png",
                         "reference_support": base["path"] + "/reference_support.png"}
                paths.update({name: prefix + "/" + name + ".png" for name in (
                    "source_rgb", "source_support", "labels_reference", "comparable", "true_visibility",
                    "source_occlusion", "true_edit_mask_reference", "visible_edit_mask_reference")})
                paths["geometry"] = prefix + "/geometry.json"
                case = {"case_id": identifier, **{k: base[k] for k in (
                    "base_id", "building_id", "view_id", "split")},
                    "parent_dataset_crop_id": base["dataset_crop_id"], "state": state,
                    "scenario": copy.deepcopy(scenario), "scenario_id": scenario["id"],
                    "nuisance_kind": scenario["kind"], "hypothesis": "H0" if state == "unchanged" else "H1",
                    "sham_self_paste": False, "full_edit_pixel_count": int(edit.sum()),
                    "visible_edit_pixel_count": int(edit.sum()), "comparable_fraction": 1.,
                    "retained_visible_edit_fraction": 1., "exclude_from_visible_recall": False,
                    "reference_file_name": part + "-2000.png", "source_file_name": part + "-2001.png",
                    "reference_year": 2000, "source_year": 2001,
                    "source_image_sha256": sha256(folder / "source_rgb.png"),
                    **paths, **geometry,
                    "artifact_sha256": {key: sha256(dataset / value) for key, value in paths.items()}}
                cases.append(case)
    index = {"bases": bases, "cases": cases, "split": split}
    _write(dataset / "index.json", index)
    _write(dataset / "summary.json", {"index_path": "index.json", "case_count": len(cases)})
    parent_config = {**dataset_config, "input_sha256": {"config": sha256(dataset / "config.json")}}
    _seal(dataset, "hypothesis_dataset", "completed_needs_review", parent_config)
    return dataset


def _csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class ResearchSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = make_dataset(self.root)
        self.common = importlib.import_module("facade_change.research.common")

    def select(self, **kwargs):
        return self.common.parent_selection(self.dataset, **kwargs)

    def rewrite_index(self, index):
        _write(self.dataset / "index.json", index)
        record = _read(self.dataset / "run.json")
        _seal(self.dataset, record["kind"], record["status"], record["config"])

    def test_selection_is_train_val_only_paired_and_deterministic_without_rgb_reads(self):
        before = _snapshot(self.dataset)
        original_open = builtins.open
        original_image_open = Image.open
        seen = []

        def guarded_open(path, *args, **kwargs):
            if isinstance(path, (str, Path)) and str(path).endswith(".png"):
                seen.append(str(path))
                self.assertNotIn("/test-", str(path))
            return original_open(path, *args, **kwargs)

        def guarded_image_open(path, *args, **kwargs):
            if isinstance(path, (str, Path)):
                self.assertNotIn("/test-", str(path))
            return original_image_open(path, *args, **kwargs)

        with patch.object(builtins, "open", guarded_open), patch.object(Image, "open", guarded_image_open):
            selected = self.select(max_buildings_per_split=1, max_bases_per_building=1,
                                   scenarios_per_base=2, seed=42)
            repeated = self.select(max_buildings_per_split=1, max_bases_per_building=1,
                                   scenarios_per_base=2, seed=42)
        bases, cases = selected["bases"], selected["cases"]
        self.assertEqual({b["split"] for b in bases}, {"train", "val"})
        self.assertEqual(len(bases), 2)
        self.assertEqual({c["split"] for c in cases}, {"train", "val"})
        self.assertEqual(len(cases), 12)
        self.assertEqual([c["case_id"] for c in cases], [c["case_id"] for c in repeated["cases"]])
        ownership = {}
        for base in bases:
            self.assertEqual(ownership.setdefault(base["building_id"], base["split"]), base["split"])
            rows = [c for c in cases if c["base_id"] == base["base_id"]]
            self.assertEqual({(c["state"], c["scenario_id"]) for c in rows},
                             set(itertools.product(("unchanged", "crack", "paint_patch"), ("clean", "blur"))))
            for row in rows:
                for key in ("building_id", "view_id", "split"):
                    self.assertEqual(row[key], base[key])
                self.assertEqual(row["parent_dataset_crop_id"], base["dataset_crop_id"])
        self.assertEqual(_snapshot(self.dataset), before)

    def test_split_leakage_rejected_even_with_resealed_metadata(self):
        index = _read(self.dataset / "index.json")
        changed = next(b for b in index["bases"] if b["split"] == "val")
        changed["building_id"] = "train-facade-0"
        for case in index["cases"]:
            if case["base_id"] == changed["base_id"]:
                case["building_id"] = changed["building_id"]
        self.rewrite_index(index)
        with self.assertRaisesRegex(ValueError, "split|building|facade|здани|leak"):
            self.select()

    def test_parent_index_hash_tampering_is_rejected(self):
        path = self.dataset / "index.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "SHA256|hash|changed"):
            self.select()

    def test_case_identity_and_state_metadata_are_strict(self):
        index = _read(self.dataset / "index.json")
        case = next(c for c in index["cases"] if c["split"] == "train")
        case["view_id"] = "different-view"
        self.rewrite_index(index)
        with self.assertRaisesRegex(ValueError, "identity|base|view|metadata"):
            self.select()

    def test_safe_identifiers_and_scenario_parameters_fail_before_output(self):
        pristine = _read(self.dataset / "index.json")
        for invalid in ("../escaped", "nested/case", 17, ["case"]):
            with self.subTest(identifier=invalid):
                index = copy.deepcopy(pristine)
                index["cases"][0]["case_id"] = invalid
                self.rewrite_index(index)
                with self.assertRaises(ValueError):
                    self.select()
        for mutate in (lambda c: c.update(nuisance_kind="undeclared-kind"),
                       lambda c: c["scenario"].update(kind="different-parameters")):
            index = copy.deepcopy(pristine)
            mutate(index["cases"][0])
            self.rewrite_index(index)
            with self.assertRaisesRegex(ValueError, "scenario|nuisance|protocol"):
                self.select()
        self.assertFalse((self.root / "escaped").exists())
        self.assertEqual({p.name for p in self.root.iterdir()}, {"dataset"})


class CalibrationAndSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = make_dataset(self.root)
        self.common = importlib.import_module("facade_change.research.common")

    def item(self, building, hypothesis, values, labels=None, split="val"):
        values = np.asarray(values, np.float32).reshape(1, -1)
        if labels is None:
            labels = np.full(values.shape, hypothesis == "H1", dtype=np.uint8)
        else:
            labels = np.asarray(labels, np.uint8).reshape(values.shape)
        case = {"building_id": building, "split": split, "hypothesis": hypothesis,
                "state": "crack" if hypothesis == "H1" else "unchanged",
                "sham_self_paste": False}
        return {"case": case, "scores": values, "labels": labels, "method": "rgb_paid"}

    def test_recall_optimization_respects_facade_equal_h0_fpr(self):
        # At t=.1, A contributes FPR1 and B contributes FPR0: macro=.5,
        # whereas pixel pooling would give .9 and incorrectly reject it at .6.
        items = [self.item("A", "H0", [.8] * 9), self.item("B", "H0", [.1]),
                 self.item("A", "H1", [.5] * 9), self.item("B", "H1", [.5])]
        result = self.common.calibrate_recall(items, (.6,))
        choice = result["choices"][0]
        self.assertEqual(choice["threshold"], float(np.float32(.1)))
        self.assertEqual(choice["h0_pixel_fpr"], .5)
        self.assertEqual(choice["recall"], 1.)
        self.assertEqual(choice["selection_split"], "val")
        self.assertLessEqual(choice["h0_pixel_fpr"], choice["fpr_target"])
        self.assertGreater(9 / 10, choice["fpr_target"])
        # Recall itself must also weight A and B equally, not their pixels/cases.
        items = [self.item("A", "H0", [.5]), self.item("B", "H0", [.5]),
                 self.item("A", "H1", [.7] * 9), self.item("B", "H1", [.3])]
        choice = self.common.calibrate_recall(items, (0.,))["choices"][0]
        self.assertEqual(choice["threshold"], .5)
        self.assertEqual(choice["h0_pixel_fpr"], 0.)
        self.assertEqual(choice["recall"], .5)
        self.assertNotEqual(choice["recall"], 9 / 10)

    def test_strict_float32_threshold_seen_value_ties_and_ignored_pixels(self):
        threshold = np.float32(.4)
        higher = np.nextafter(threshold, np.float32(np.inf))
        items = [self.item("A", "H0", [.2]),
                 self.item("A", "H1", [higher, threshold, .99], [1, 0, 255])]
        result = self.common.calibrate_recall(items, (0.,))
        choice = result["choices"][0]
        # Both .2 and .4 have Recall1/FPR0; the documented highest tie wins.
        self.assertEqual(choice["threshold"], float(threshold))
        self.assertEqual(choice["recall"], 1.)
        self.assertEqual(choice["h0_pixel_fpr"], 0.)
        self.assertNotIn(float(np.float32(.99)), [p["threshold"] for p in result["curve"]])
        row = self.common.metric_row(items[1]["case"], items[1]["scores"], items[1]["labels"],
                                     choice["threshold"], "rgb_paid")
        self.assertEqual((row["tp"], row["fp"], row["fn"], row["tn"], row["ignored_pixel_count"]),
                         (1, 0, 0, 1, 1))

    def test_train_test_items_do_not_enter_validation_choice(self):
        val = [self.item("A", "H0", [.2]), self.item("A", "H1", [.8])]
        expected = self.common.calibrate_recall(val, (.01, .05))
        foreign = [self.item("TRAIN", "H0", [1000.], split="train"),
                   self.item("TEST", "H1", [-1000.], split="test")]
        self.assertEqual(self.common.calibrate_recall(val + foreign, (.01, .05)), expected)
        foreign[1]["scores"][0, 0] = 1e20
        foreign[1]["labels"][0, 0] = 0
        self.assertEqual(self.common.calibrate_recall(val + foreign, (.01, .05)), expected)
        for targets in ((), (True,), (float("nan"),), (-.01,), (1.01,)):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                self.common.calibrate_recall(val, targets)

    def source_fixture(self):
        source = self.root / "sealed-benchmark"
        source.mkdir()
        weights = self.root / "unexpected-model-location"
        weights.mkdir()
        msd = weights / "arbitrary-name.bin"
        dino = weights / "another-arbitrary-name.bin"
        msd.write_bytes(b"recorded MSDZip bytes")
        dino.write_bytes(b"recorded DINO bytes")
        dino_root = self.root / "unusual-dino-source"
        dino_root.mkdir()
        config = {"dataset_run": str(self.dataset), "input_sha256": _fingerprint(self.dataset),
                  "scorers": {"msdzip_mod256": {"checkpoint_path": str(msd), "checkpoint_sha256": sha256(msd)},
                              "dinov2": {"sources": {"path": str(dino_root)},
                                         "checkpoint": {"path": str(dino), "sha256": sha256(dino)}}}}
        _write(source / "summary.json", {"status": "completed_exploratory"})
        _seal(source, "hypothesis_benchmark", "completed_exploratory", config)
        return source, msd, dino, dino_root

    def test_source_resolver_uses_exact_recorded_paths_without_opening_models_or_test(self):
        cli = importlib.import_module("facade_change.research.__main__")
        source, msd, dino, dino_root = self.source_fixture()
        before = _snapshot(self.dataset, source, msd.parent, dino_root)
        original_open = builtins.open

        def guarded_open(path, *args, **kwargs):
            if isinstance(path, (str, Path)):
                self.assertNotIn(Path(path), {msd, dino})
                self.assertFalse(str(path).endswith(".png"))
            return original_open(path, *args, **kwargs)

        with patch.object(builtins, "open", guarded_open):
            resolved = cli.resolve_source(source)
            paid_resolved = cli.resolve_source(source, need_models=False)
        self.assertEqual(resolved, {"dataset_run": str(self.dataset),
            "msdzip_checkpoint": str(msd), "dino_root": str(dino_root), "dino_checkpoint": str(dino),
            "expected_msdzip_sha256": sha256(msd), "expected_dino_sha256": sha256(dino)})
        self.assertEqual(paid_resolved, {"dataset_run": str(self.dataset)})
        self.assertEqual(_snapshot(self.dataset, source, msd.parent, dino_root), before)
        record = _read(source / "run.json")
        record["config"]["scorers"]["msdzip_mod256"].pop("checkpoint_sha256")
        _seal(source, record["kind"], record["status"], record["config"])
        with self.assertRaisesRegex(ValueError, "lacks|paths|hashes"):
            cli.resolve_source(source)

    def test_source_resolver_rejects_changed_summary_and_parent(self):
        cli = importlib.import_module("facade_change.research.__main__")
        source, *_ = self.source_fixture()
        summary = source / "summary.json"
        pristine = summary.read_bytes()
        summary.write_bytes(pristine + b" ")
        with self.assertRaisesRegex(ValueError, "summary hash"):
            cli.resolve_source(source)
        summary.write_bytes(pristine)
        index = self.dataset / "index.json"
        index.write_bytes(index.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "SHA256|hash"):
            cli.resolve_source(source)


class PaidExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = make_dataset(self.root)
        self.paid = importlib.import_module("facade_change.research.paid")

    def run_paid(self, output):
        with redirect_stdout(io.StringIO()):
            return self.paid.run_paid(self.dataset, output, max_buildings_per_split=1,
                                      max_bases_per_building=1, scenarios_per_base=2,
                                      seed=42, tile_size=3, stride=2,
                                      fpr_targets=(.01, .05))

    def test_paid_end_to_end_never_reads_test_and_seals_artifacts(self):
        before = _snapshot(self.dataset)
        output = self.root / "paid"
        original_open = builtins.open

        def guarded_open(path, *args, **kwargs):
            if isinstance(path, (str, Path)) and str(path).endswith(".png"):
                self.assertNotIn("/test-", str(path))
                self.assertTrue((output / "experiment_contract.json").is_file())
            return original_open(path, *args, **kwargs)

        with patch.object(builtins, "open", guarded_open):
            self.run_paid(output)
        self.assertEqual(_snapshot(self.dataset), before)
        record = _read(output / "run.json")
        self.assertEqual(record["status"], "completed_exploratory")
        cases = _read(output / "selected_cases.json")["cases"]
        self.assertEqual(len(cases), 12)
        self.assertEqual({c["split"] for c in cases}, {"train", "val"})
        self.assertEqual(record["config"]["case_ids"], [c["case_id"] for c in cases])
        self.assertEqual(record["config"]["requested_splits"], ["train", "val"])
        for case in cases:
            for method in ("rgb_paid", "rgb_diff"):
                relative = f"scores/{method}/{case['case_id']}.npy"
                values = np.load(output / relative, allow_pickle=False)
                self.assertEqual(values.shape, (4, 4))
                self.assertEqual(values.dtype, np.float32)
                self.assertTrue(np.isfinite(values).all())
                self.assertEqual(record["artifact_sha256"][relative], sha256(output / relative))
            for suffix in ("h0", "rgb_paid", "oracle_paid"):
                path = output / "streams" / f"{case['case_id']}.{suffix}.bin"
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)
                self.assertEqual(record["artifact_sha256"][path.relative_to(output).as_posix()], sha256(path))
        events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        self.assertEqual([e["sequence"] for e in events], list(range(1, len(events) + 1)))
        self.assertEqual(len({e["event_id"] for e in events}), len(events))
        self.assertEqual(events[0]["event_type"], "started")
        self.assertEqual(events[-1]["event_type"], "completed")
        self.assertTrue(all(e["config_sha256"] == record["config_sha256"] for e in events))
        self.assertEqual(events[-1]["summary_sha256"], sha256(output / "summary.json"))
        tile_rows = _csv(output / "tile_costs.csv")
        self.assertEqual(len(tile_rows), len(cases) * 4 * 3)
        by_id = {c["case_id"]: c for c in cases}
        next_offsets = {}
        codec = importlib.import_module("facade_change.research.paid_codec")
        for row in tile_rows:
            case = by_id[row["case_id"]]
            key = row["case_id"], row["method"]
            offset, size = int(row["stream_offset"]), int(row["stream_bytes"])
            self.assertEqual(offset, next_offsets.get(key, 0))
            next_offsets[key] = offset + size
            bundle = output / "streams" / f"{case['case_id']}.{row['method']}.bin"
            stream = bundle.read_bytes()[offset:offset + size]
            self.assertEqual(hashlib.sha256(stream).hexdigest(), row["stream_sha256"])
            self.assertEqual(int(row["total_bits"]), 8 * len(stream))
            partitions = ast.literal_eval(row["partitions"])
            self.assertEqual(sum(partitions.values()), 8 * len(stream))
            self.assertGreater(partitions["header_bits"], 0)
            self.assertGreater(partitions["nuisance_bits"], 0)
            self.assertEqual(partitions["C_bits"] > 0, row["method"] != "h0")
            self.assertEqual(partitions["V_bits"] > 0, row["method"] != "h0")
            y, x = int(row["row"]), int(row["col"])
            a = np.asarray(Image.open(self.dataset / case["reference_rgb"]))[y:y + 3, x:x + 3]
            b = np.asarray(Image.open(self.dataset / case["source_rgb"]))[y:y + 3, x:x + 3]
            support = np.asarray(Image.open(self.dataset / case["reference_support"]))[y:y + 3, x:x + 3] == 255
            # Decode each framed tile using only shared A/support plus charged stream.
            np.testing.assert_array_equal(codec.decode(a, support, stream)[support], b[support])
        for (case_id, method), end in next_offsets.items():
            self.assertEqual(end, (output / "streams" / f"{case_id}.{method}.bin").stat().st_size)
        metrics = _csv(output / "case_metrics.csv")
        self.assertEqual({r["method"] for r in metrics}, {"rgb_paid", "rgb_diff"})
        self.assertNotIn("oracle_paid", {r["method"] for r in metrics})
        self.assertEqual({r["split"] for r in metrics}, {"train", "val"})
        sealed = _snapshot(output)
        with self.assertRaisesRegex((ValueError, FileExistsError), "exist|fresh|directory"):
            self.run_paid(output)
        self.assertEqual(_snapshot(output), sealed)

    def test_counterfactual_gt_does_not_change_nonoracle_streams_or_scores(self):
        first = self.root / "first"
        self.run_paid(first)
        index = _read(self.dataset / "index.json")
        for case in index["cases"]:
            if case["split"] not in {"train", "val"} or case["hypothesis"] != "H1":
                continue
            edit = np.zeros((4, 4), dtype=bool)
            edit.flat[-4:] = True
            for key in ("labels_reference", "true_edit_mask_reference", "visible_edit_mask_reference"):
                values = edit.astype(np.uint8) * (1 if key == "labels_reference" else 255)
                path = self.dataset / case[key]
                Image.fromarray(values).save(path)
                case["artifact_sha256"][key] = sha256(path)
        _write(self.dataset / "index.json", index)
        record = _read(self.dataset / "run.json")
        _seal(self.dataset, record["kind"], record["status"], record["config"])
        second = self.root / "second"
        self.run_paid(second)
        selected = _read(first / "selected_cases.json")["cases"]
        self.assertEqual([c["case_id"] for c in selected],
                         [c["case_id"] for c in _read(second / "selected_cases.json")["cases"]])
        oracle_changed = False
        for case in selected:
            for method in ("rgb_paid", "rgb_diff"):
                relative = f"scores/{method}/{case['case_id']}.npy"
                self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes())
            for suffix in ("h0", "rgb_paid"):
                relative = f"streams/{case['case_id']}.{suffix}.bin"
                self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes())
            relative = f"streams/{case['case_id']}.oracle_paid.bin"
            oracle_changed |= (first / relative).read_bytes() != (second / relative).read_bytes()
        self.assertTrue(oracle_changed, "The GT-only diagnostic must reflect its changed paid mask")


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional CPU PyTorch is unavailable")
class ContextExperimentTests(unittest.TestCase):
    def test_real_frozen_msdzip_and_head_training_reference_provider_fixture(self):
        import torch
        module = importlib.import_module("facade_change.2026-10-04_msdzip_h0")
        training = importlib.import_module("facade_change.research.context_training")
        context = importlib.import_module("facade_change.research.context")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = make_dataset(root)
            original_threads = torch.get_num_threads()
            self.addCleanup(torch.set_num_threads, original_threads)
            torch.set_num_threads(1)
            with redirect_stdout(io.StringIO()):
                fit = module.train_msdzip_h0(
                    dataset, root / "original-fit", representations=["mod256"], epochs=1,
                    max_train_bytes=24, max_val_bytes=12, model_batch_size=4,
                    timesteps=2, hidden_dim=4, ffn_dim=8, vocab_dim=2,
                    max_bases_per_split=1, window_groups=2, seed=42, device="cpu")
            checkpoint = root / "original-fit" / fit["results"]["mod256"]["checkpoint_path"]
            dino_root = root / "fake-dino-source"
            dino_root.mkdir()
            dino_checkpoint = root / "fake-dino.pt"
            dino_checkpoint.write_bytes(b"test-only reference encoder provider")
            before = _snapshot(dataset, checkpoint.parent, dino_root)
            dino_bytes = dino_checkpoint.read_bytes()
            encoder = torch.nn.Linear(1, 1)
            encoder.eval()
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)
            encoder_before = {key: value.detach().clone() for key, value in encoder.state_dict().items()}
            provider_calls = []

            def provider(reference, support, model, device="cpu"):
                self.assertIs(model, encoder)
                self.assertEqual(reference.dtype, np.uint8)
                self.assertEqual(support.dtype, bool)
                provider_calls.append(reference.copy())
                value = float(reference[support].mean()) / 255
                features = (torch.arange(8, dtype=torch.float32, device=device) / 7 + value).reshape(8, 1, 1)
                return features, {"native_shape": list(support.shape), "patch_grid": [1, 1],
                                  "context": "test-only frozen reference provider", "target_features": False}

            output = root / "context"
            original_open = builtins.open

            def guarded_open(path, *args, **kwargs):
                if isinstance(path, (str, Path)) and str(path).endswith(".png"):
                    self.assertNotIn("/test-", str(path))
                    self.assertTrue((output / "experiment_contract.json").is_file())
                return original_open(path, *args, **kwargs)

            with (patch.object(context, "load_reference_encoder", return_value=(encoder, {"fixture": True})),
                  patch.object(context, "extract_reference", provider),
                  patch.object(builtins, "open", guarded_open),
                  redirect_stdout(io.StringIO())):
                summary = training.run_context(
                    dataset, output, checkpoint, dino_root, dino_checkpoint,
                    epochs=2, max_train_bytes=24, max_val_bytes=12, batch_size=12,
                    width=4, tokens=2, seed=42, device="cpu", max_buildings_per_split=1,
                    max_bases_per_building=1, scenarios_per_base=2)
            self.assertEqual(summary["status"], "completed_exploratory")
            self.assertEqual(summary["selected_bytes"], {"train": 24, "val": 12})
            self.assertEqual(summary["case_count"], 12)
            self.assertTrue(summary["original_weights_unchanged"])
            self.assertEqual(set(summary["methods"]), {"baseline", "spatial", "spectral"})
            parameters = summary["matched_trainable_parameters"]
            self.assertEqual(parameters["spatial"], parameters["spectral"])
            self.assertGreater(parameters["spatial"], 0)
            self.assertEqual(len(summary["history"]), 4)
            self.assertEqual(len(provider_calls), 2, "One frozen reference feature extraction per selected base")
            cases = _read(output / "selected_cases.json")["cases"]
            plans = _read(output / "sampling.json")
            by_id = {c["case_id"]: c for c in cases}
            for split, samples in plans.items():
                self.assertIn(split, {"train", "val"})
                self.assertEqual(sum(len(s["target_byte_offsets"]) for s in samples),
                                 summary["selected_bytes"][split])
                for sample in samples:
                    case = by_id[sample["case_id"]]
                    self.assertEqual(case["split"], split)
                    self.assertEqual(case["state"], "unchanged")
                    self.assertEqual(case["hypothesis"], "H0")
            scorer = module.MSDZipScorer(checkpoint, device="cpu", representation="mod256",
                                         dataset_fingerprint=fit["input_sha256"])
            for case in cases:
                reference = np.asarray(Image.open(dataset / case["reference_rgb"]))
                source = np.asarray(Image.open(dataset / case["source_rgb"]))
                support = np.asarray(Image.open(dataset / case["reference_support"])) == 255
                expected = scorer(reference, source, support)
                baseline = np.load(output / "scores" / "baseline" / f"{case['case_id']}.npy", allow_pickle=False)
                raw = np.load(output / "raw_bpb" / "baseline" / f"{case['case_id']}.npy", allow_pickle=False)
                np.testing.assert_array_equal(baseline, expected)
                np.testing.assert_array_equal(raw, scorer.raw_scores)
            for mode in ("spatial", "spectral"):
                saved = torch.load(output / f"{mode}.pt", map_location="cpu", weights_only=True)
                metadata = saved["metadata"]
                rows = [r for r in summary["history"] if r["mode"] == mode]
                best = min(rows, key=lambda r: r["val_bpb"])
                self.assertEqual(metadata["best_epoch"], best["epoch"])
                self.assertEqual(metadata["val_bpb"], best["val_bpb"])
                self.assertEqual(metadata["sampling_sha256"], sha256(output / "sampling.json"))
                self.assertTrue(saved["state_dict"]["logit_head.weight"].abs().sum() > 0,
                                "The actual correction head must undergo optimizer updates")
            for key, value in encoder_before.items():
                self.assertTrue(torch.equal(value, encoder.state_dict()[key]))
            self.assertEqual(_snapshot(dataset, checkpoint.parent, dino_root), before)
            self.assertEqual(dino_checkpoint.read_bytes(), dino_bytes)
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertEqual(sum(e["event_type"] == "epoch_completed" for e in events), 4)
            self.assertEqual(sum(e["event_type"] == "head_frozen" for e in events), 2)
            self.assertEqual(sum(e["event_type"] == "case_scored" for e in events), 36)
            self.assertTrue(all(e.get("cost_mode") == "model_bits" for e in events
                                if e["event_type"] == "case_scored"))
            record = _read(output / "run.json")
            self.assertEqual(record["status"], "completed_exploratory")
            self.assertEqual(events[-1]["event_type"], "completed")
            self.assertEqual(record["artifact_sha256"]["summary.json"], sha256(output / "summary.json"))


if __name__ == "__main__":
    unittest.main()
