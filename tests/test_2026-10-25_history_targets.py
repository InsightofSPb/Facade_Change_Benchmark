"""Mask transfer, partial annotation semantics and explicit weak supervision."""
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

from facade_change import crop_targets, workflow
from facade_change.crop_dataset import prepare_crop_dataset
from facade_change.hypothesis_dataset import prepare_hypothesis_dataset
from facade_change.io import read_json, sha256, write_json
from facade_change.segmentation.data import rasterize_annotations


targets = importlib.import_module("facade_change.research.2026-10-25_history_targets")
models = importlib.import_module("facade_change.research.2026-10-25_history_models")
features = importlib.import_module("test_2026-10-25_history_models")


def make_fixture(root):
    fixture = features.FeatureFixture(root)
    categories = [{"id": 4, "name": "ORNAMENT_INTACT"}, {"id": 9, "name": "CRACK"}]
    images, manifest_images, annotations = [], [], []
    for track, entry in zip(fixture.tracks, fixture.sequences):
        track["bbox_xywh"] = entry["bbox_xywh"] = [0, 0, 14, 14]
        for observation in track["observations"]:
            path = fixture.root / observation["rgb"]
            support = fixture.root / f"{observation['observation_id']}_support.png"
            Image.fromarray(np.full((14, 14), 255, dtype=np.uint8)).save(support)
            observation.update(image_sha256=sha256(path), original_image_path=str(path),
                               support=support.name, support_sha256=sha256(support),
                               alignment={"matrix": np.eye(3).tolist()})
            identifier = len(images) + 100  # Deliberately different from the history's observation IDs.
            image = {"id": identifier, "height": 14, "width": 14, "file_name": path.name}
            images.append(image)
            manifest_images.append({"image_id": identifier, "height": 14, "width": 14,
                                    "image_path": str(path), "sha256": sha256(path), "year": observation["year"],
                                    **{k: track[k] for k in ("split", "building_id", "view_id")}})
            annotations.append({"id": len(annotations)+1, "image_id": identifier, "category_id": 4,
                                "segmentation": [[0, 0, 14, 0, 14, 14, 0, 14]]})
            if track["track_id"].endswith("_1") and observation["year"] >= 2018:
                annotations.append({"id": len(annotations)+1, "image_id": identifier, "category_id": 9,
                                    "segmentation": [[0, 0, 7, 0, 7, 14, 0, 14]]})
    coco = root / "coco.json"
    write_json(coco, {"images": images, "categories": categories, "annotations": annotations})
    manifest = root / "annotation_manifest.json"
    write_json(manifest, {"source": {"coco_json": str(coco), "coco_sha256": sha256(coco)},
                          "images": manifest_images, "categories": categories})
    inputs = root / "session/inputs.json"
    inputs.parent.mkdir()
    write_json(inputs, {"manifest": str(manifest), "manifest_sha256": sha256(manifest)})
    fixture.save()
    history_path = fixture.prepare / "manifest.json"
    document = read_json(history_path)
    document.update(views=[{"frames": [{"image": row} for row in manifest_images]}],
                    alignment_diagnostics_root=str(inputs.parent / "quality"),
                    input_sha256={str(inputs): sha256(inputs)})
    write_json(history_path, document)
    write_json(fixture.prepare / "run.json", {"status": "completed", "manifest_sha256": sha256(history_path)})
    fixture.index["history_manifest_sha256"] = sha256(history_path)
    write_json(fixture.features / "index.json", fixture.index)
    write_json(fixture.embed / "index.json", fixture.index)
    write_json(fixture.embed / "run.json", {"status": "completed", "index_sha256": sha256(fixture.features / "index.json")})
    return fixture


class MaskSemanticsTests(unittest.TestCase):
    def state(self, values):
        array = np.asarray(values, dtype=np.uint16)
        return {"packed": array, "support": np.ones(array.shape, dtype=bool)}

    def test_blank_masks_or_disappearing_annotation_are_unknown(self):
        empty, crack = self.state(np.zeros((14, 14))), self.state(np.ones((14, 14)))
        for before, after in ((empty, empty), (crack, empty), (empty, crack)):
            row, mask, _ = targets.event_target(before, after, [0])
            self.assertEqual(row["label"], "")
            self.assertEqual(row["reason"], "no_common_annotation_support")
            self.assertTrue((mask == 255).all())

    def test_changes_and_stability_are_restricted_to_common_annotated_pixels(self):
        old, new = np.ones((14, 14), np.uint16), np.ones((14, 14), np.uint16)
        old[:, :2] = new[:, :2] = 0
        new[:, 5:10] |= 2
        row, mask, _ = targets.event_target(self.state(old), self.state(new), [0, 1])
        self.assertEqual(row["label"], 1)
        self.assertTrue((mask[:, :2] == 255).all())
        self.assertEqual(row["comparable"], "annotation_support_only")
        self.assertNotEqual(row["review_status"], "accepted")
        stable, _, _ = targets.event_target(self.state(old), self.state(old), [0, 1])
        self.assertEqual(stable["label"], 0)
        self.assertGreater(stable["unknown_pixels"], 0)

    def test_non_target_class_changes_do_not_become_target_events(self):
        old, new = self.state(np.ones((14, 14))), self.state(np.full((14, 14), 5))
        row, _, _ = targets.event_target(old, new, [1])
        self.assertEqual(row["label"], 0)
        all_classes, _, _ = targets.event_target(old, new, [0, 1, 2])
        self.assertEqual(all_classes["label"], 1)

    def test_boundary_tolerance_abstains_instead_of_assigning_negative(self):
        old, new = np.ones((14, 14), np.uint16), np.ones((14, 14), np.uint16)
        old[:, 3] |= 2
        new[:, 4] |= 2
        row, _, _ = targets.event_target(self.state(old), self.state(new), [1], radius=1)
        self.assertEqual(row["label"], "")
        self.assertEqual(row["reason"], "tiny_or_boundary_ambiguous_difference")
        self.assertGreater(row["boundary_ambiguous_pixels"], 0)

    def test_same_year_conflicts_are_unknown(self):
        consensus = targets.year_consensus([self.state(np.ones((14, 14))), self.state(np.full((14, 14), 2))])
        self.assertTrue((consensus["packed"] == 0).all())

    def test_partly_occluded_target_does_not_turn_stable_remainder_into_negative(self):
        old, new = np.ones((14, 14), np.uint16), np.ones((14, 14), np.uint16)
        new[:, :4] = 0
        row, _, _ = targets.event_target(self.state(old), self.state(new), [0])
        self.assertEqual(row["label"], "")
        self.assertEqual(row["reason"], "target_annotations_without_comparable_support")
        self.assertGreater(row["known_pixels"], 64)
        self.assertEqual(row["uncomparable_target_pixels"], 56)

    def test_partial_same_year_conflict_cannot_produce_negative(self):
        old, alternative = np.ones((14, 14), np.uint16), np.ones((14, 14), np.uint16)
        alternative[:, :4] = 2
        consensus = targets.year_consensus([self.state(old), self.state(alternative)])
        row, _, _ = targets.event_target(consensus, consensus, [0, 1])
        self.assertEqual(row["label"], "")
        self.assertEqual(row["reason"], "target_annotations_without_comparable_support")

    def test_geometry_maps_source_to_exact_crop_and_preserves_overlap(self):
        packed = np.zeros((14, 28), np.uint16)
        packed[:, 14:] = 3
        support = np.ones((14, 14), bool)
        support[:, 0] = False
        matrix = np.array([[1, 0, -7], [0, 1, 0], [0, 0, 1]])
        mapped = targets.project_masks(packed, matrix, [7, 0, 14, 14], support)
        self.assertTrue((mapped[:, 1:] == 3).all())
        self.assertTrue((mapped[:, 0] == 0).all())

    def test_existing_all_class_evaluator_parity(self):
        rng = np.random.default_rng(20)
        a, b = (self.state(rng.integers(0, 8, (14, 14), dtype=np.uint16)) for _ in range(2))
        for radius in (0, 1):
            actual = targets.target_difference(a, b, [0, 1, 2], radius)[0]
            expected = targets.derive_labels(a["packed"], b["packed"], a["support"], 3, radius)
            np.testing.assert_array_equal(actual, expected)


@unittest.skipUnless(importlib.util.find_spec("pycocotools"), "COCO rasterizer required")
class CropTargetPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        from test_alignment_session import session_fixture
        self.input, session = session_fixture(self.root, dense=True)
        self.crops = self.root / "crops"
        prepare_crop_dataset(alignment_session=session, out=self.crops,
                             tile_size=16, stride=8, min_valid_fraction=.8)
        self.summary = read_json(self.crops / "summary.json")
        self.index_path = self.crops / self.summary["index_path"]
        self.index = read_json(self.index_path)
        manifest = read_json(self.input["dataset"] / "manifest.json")
        coco = self.input["dataset"] / "coco.json"
        manifest["source"] = {"coco_json": str(coco), "coco_sha256": sha256(coco)}
        self.annotation_manifest = self.root / "annotation_manifest.json"
        write_json(self.annotation_manifest, manifest)

    def args(self, *extra):
        return targets.parser().parse_args(["--crops", str(self.crops),
            "--annotation-manifest", str(self.annotation_manifest),
            "--out", str(self.root / "crop-targets"), *extra])

    def export(self, *extra):
        return crop_targets.export_crop_targets(self.args(*extra))

    def repin(self, *relative):
        parent = read_json(self.crops / "run.json")
        for name in relative:
            parent["artifact_sha256"][name] = sha256(self.crops / name)
        write_json(self.crops / "run.json", parent)

    def test_shared_functions_and_both_public_commands_preserve_history_arguments(self):
        self.assertIs(crop_targets.event_target, targets.event_target)
        self.assertIs(crop_targets.project_masks, targets.project_masks)
        legacy = targets.parser().parse_args(["--features", "features/index.json", "--out", "run"])
        self.assertFalse(hasattr(legacy, "crops"))
        self.assertEqual(legacy.min_years, 3)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(workflow.main(["prepare", "targets", *[
                "--crops", str(self.crops), "--annotation-manifest", str(self.annotation_manifest),
                "--out", str(self.root / "cli-targets"), "--gallery-limit", "0"]]), 0)
        self.assertEqual(read_json(self.root / "cli-targets/summary.json")["kind"], "crop_annotation_targets")

    def test_mixed_geometry_masks_match_exact_saved_rgb_crop_grid(self):
        before = {p: sha256(p) for p in self.crops.rglob("*") if p.is_file()}
        summary = self.export()
        self.assertGreater(summary["derived_positive"], 0)
        self.assertGreater(summary["derived_stable"], 0)
        self.assertFalse(summary["reviewed_event_gt_created"])
        out = self.root / "crop-targets"
        exported = read_json(out / "index.json")
        source = targets.load_annotation_source(self.annotation_manifest, crop_targets.Fingerprints())
        self.assertEqual(summary["events"], sum(r["split"] in {"train", "val"} for r in self.index["crops"]))
        saw_dense = saw_offset = saw_unknown = False
        for event in exported["events"]:
            row = next(r for r in self.index["crops"] if r["dataset_crop_id"] == event["dataset_crop_id"])
            for role in ("reference", "source"):
                observation_id = crop_targets.digest([event["track_id"], role])
                record = exported["observations"][observation_id]
                with np.load(out / record["artifact_path"], allow_pickle=False) as arrays:
                    projected, support = arrays["packed"], arrays["support"]
                    image = source["images"][row["source_image_sha256"][role]]
                    packed = rasterize_annotations(source["annotations"][str(image["image_id"])],
                                                   image["height"], image["width"], source["channel_map"])
                    if role == "source" and row.get("dense_geometry"):
                        from facade_change.saved_geometry import FrameSampler
                        x0, y0, x1, y1 = row["bbox_canvas_xyxy"]
                        expected = FrameSampler({"dense_geometry": row["dense_geometry"]}).warp(
                            packed, (x0, y0, x1-x0, y1-y0), nearest=True)
                        expected[~support] = 0
                        saw_dense = True
                    else:
                        expected = targets.project_masks(packed, row[role + "_to_crop"], [0, 0, 16, 16], support)
                    saw_offset |= row["bbox_canvas_xyxy"][0] != 0
                    np.testing.assert_array_equal(projected, expected)
            with np.load(out / event["target_path"], allow_pickle=False) as arrays:
                self.assertTrue(np.isin(arrays["target"], [0, 1, 255]).all())
                saw_unknown |= bool((arrays["target"] == 255).any())
        self.assertTrue(saw_dense and saw_offset)
        self.assertTrue(saw_unknown)
        for path, expected in before.items():
            self.assertEqual(sha256(path), expected)
        self.assertIn("парные кропы", (out / "gallery.html").read_text())

    def test_default_annotation_source_follows_crop_parent_not_latest(self):
        parent = read_json(self.crops / "run.json")
        parent["config"]["manifest_path"] = str(self.annotation_manifest)
        parent["config"]["input_sha256"]["manifest"] = sha256(self.annotation_manifest)
        write_json(self.crops / "run.json", parent)
        args = self.args("--gallery-limit", "0")
        args.annotation_manifest = None
        self.assertGreater(crop_targets.export_crop_targets(args)["events"], 0)

    def test_crop_masks_need_no_feature_cache_or_rgb_decoding_and_do_not_read_test(self):
        for image in self.input["images"]:
            if image["split"] == "test":
                Path(image["image_path"]).unlink()
        original_open = Image.open
        def only_support(path, *args, **kwargs):
            if "support" not in Path(path).name and Path(path).name != "geometric_overlap.png":
                raise AssertionError("RGB decoding is not needed for mask export")
            return original_open(path, *args, **kwargs)
        with mock.patch.object(Image, "open", side_effect=only_support):
            result = self.export("--gallery-limit", "0")
        self.assertEqual(set(result["by_split"]), {"train", "val"})

    def test_unknown_annotations_do_not_become_negative_crops(self):
        coco = self.input["dataset"] / "coco.json"
        content = read_json(coco)
        content["annotations"] = []
        write_json(coco, content)
        manifest = read_json(self.annotation_manifest)
        manifest["source"]["coco_sha256"] = sha256(coco)
        write_json(self.annotation_manifest, manifest)
        result = self.export("--gallery-limit", "0")
        self.assertEqual(result["unknown"], result["events"])
        self.assertEqual(result["derived_stable"], 0)

    def test_selected_category_changes_only_affect_requested_targets(self):
        result = self.export("--categories", "WALL", "--gallery-limit", "0")
        self.assertEqual(result["derived_positive"], 0)
        self.assertEqual(result["derived_stable"], result["events"])

    def test_source_hash_and_category_mapping_tampering_fail_before_output(self):
        coco = self.input["dataset"] / "coco.json"
        coco.write_bytes(coco.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            self.export()
        self.assertFalse((self.root / "crop-targets").exists())

    def test_crop_transform_tampering_is_rejected(self):
        row = self.index["crops"][0]
        row["source_to_crop"][0][2] += 2
        write_json(self.index_path, self.index)
        self.repin(self.summary["index_path"])
        with self.assertRaisesRegex(ValueError, "crop.json"):
            self.export()

    def test_building_leakage_is_rejected_before_annotation_decode(self):
        self.index["crops"][-1]["building_id"] = self.index["crops"][0]["building_id"]
        write_json(self.index_path, self.index)
        self.repin(self.summary["index_path"])
        with self.assertRaisesRegex(ValueError, "split leakage"):
            self.export()

    def test_resume_uses_committed_maps_and_completed_outputs_are_immutable(self):
        original = crop_targets.CropRun.save_unit
        def interrupted(store, record, arrays):
            result = original(store, record, arrays)
            if len(list((store.out / "units").glob("*.json"))) == 2:
                raise KeyboardInterrupt()
            return result
        with mock.patch.object(crop_targets.CropRun, "save_unit", interrupted), self.assertRaises(KeyboardInterrupt):
            self.export("--gallery-limit", "0")
        self.assertEqual(read_json(self.root / "crop-targets/run.json")["status"], "interrupted")
        result = self.export("--gallery-limit", "0", "--resume")
        self.assertEqual(result["processing"]["reused_observations"], 2)
        with self.assertRaisesRegex(ValueError, "already completed"):
            self.export("--gallery-limit", "0", "--resume")
        crop_parent_hash = sha256(self.crops / "run.json")
        with self.assertRaisesRegex(ValueError, "separate from the input"):
            self.export("--gallery-limit", "0", "--resume", "--out", str(self.crops))
        self.assertEqual(sha256(self.crops / "run.json"), crop_parent_hash)

    def test_full_synthetic_matrix_reuses_same_crops_without_real_annotation_labels(self):
        config = Path(__file__).resolve().parents[1] / "configs/2026-10-25_crop_controls.json"
        options = read_json(config)
        self.assertEqual(options["selection"]["splits"], ["train", "val"])
        self.assertEqual((len(options["states"]), len(options["scenarios"])), (4, 18))
        with mock.patch("facade_change.segmentation.data.rasterize_annotations", side_effect=AssertionError("synthetic GT is procedural")):
            result = prepare_hypothesis_dataset(self.crops, config, self.root / "synthetic")
        index = read_json(self.root / "synthetic/index.json")
        self.assertEqual(len(index["cases"]), 72 * len(index["bases"]))
        self.assertTrue(all(case["split"] in {"train", "val"} for case in index["cases"]))
        self.assertEqual(result["case_count"], len(index["cases"]))


@unittest.skipUnless(importlib.util.find_spec("pycocotools"), "COCO rasterizer required")
class TargetPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = make_fixture(self.root)
        self.silence = contextlib.redirect_stdout(io.StringIO())
        self.silence.__enter__()
        self.addCleanup(self.silence.__exit__, None, None, None)

    def args(self, *extra):
        return targets.parser().parse_args(["--features", str(self.fixture.features),
                                           "--out", str(self.root / "targets"), *extra])

    def test_polygons_rle_overlaps_and_crowd_keep_existing_contract(self):
        from pycocotools import mask as mask_utils
        mask = np.zeros((14, 14), dtype=np.uint8)
        mask[:, 6:] = 1
        rle = mask_utils.encode(np.asfortranarray(mask))
        annotations = [
            {"id": 1, "category_id": 4, "segmentation": [[0, 0, 10, 0, 10, 14, 0, 14]]},
            {"id": 2, "category_id": 9, "segmentation": rle},
            {"id": 3, "category_id": 4, "iscrowd": 1, "segmentation": [[7, 0, 8, 0, 8, 14, 7, 14]]},
        ]
        packed = rasterize_annotations(annotations, 14, 14, {4: 0, 9: 1})
        self.assertTrue((packed[:, 6] == 3).all())
        self.assertTrue((packed[:, 7] == 0).all())
        self.assertTrue((packed[:, 11] == 2).all())

    def test_export_reuses_features_preserves_rgb_and_requires_explicit_derived_training(self):
        files = [*self.fixture.root.glob("*.png"), *self.fixture.embed.glob("sequences/*.npz")]
        hashes = {p: sha256(p) for p in files}
        # No TEST source images are required by a TRAIN/VAL annotation export.
        for track in self.fixture.tracks:
            if track["split"] == "test":
                for row in track["observations"]:
                    Path(row["original_image_path"]).unlink()
        summary = targets.export_targets(self.args())
        self.assertGreater(summary["derived_positive"], 0)
        self.assertGreater(summary["derived_stable"], 0)
        self.assertFalse(summary["reviewed_event_gt_created"])
        labels = self.root / "targets/labels.csv"
        with labels.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(all(r["review_status"] == "derived" for r in rows))
        with self.assertRaisesRegex(ValueError, "allow-derived-labels"):
            models.read_labels(models.load_histories(self.fixture.features), labels)
        accepted, coverage = models.read_labels(models.load_histories(self.fixture.features), labels, allow_derived=True)
        self.assertEqual(coverage["label_kinds"], {"annotation_derived": len(accepted)})
        self.assertIn("not an independent", models.report_metrics(accepted, np.arange(len(accepted)), 5)["reference_scope"])
        for path, value in hashes.items():
            if path.exists():
                self.assertEqual(sha256(path), value)
        self.assertIn("неизвестно", (self.root / "targets/gallery.html").read_text())

    def test_derived_label_tampering_is_rejected(self):
        targets.export_targets(self.args("--gallery-limit", "0"))
        with (self.root / "targets/labels.csv").open() as stream:
            reader = csv.DictReader(stream)
            rows, fields = list(reader), reader.fieldnames
        rows[0]["label"] = str(1-int(rows[0]["label"]))
        path = self.root / "edited.csv"
        models.write_csv(path, rows, fields)
        with self.assertRaisesRegex(ValueError, "differs from its source"):
            models.read_labels(models.load_histories(self.fixture.features), path, allow_derived=True)

    def test_annotation_source_hash_mismatch_fails_before_export(self):
        with (self.root / "coco.json").open("a") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            targets.export_targets(self.args())
        self.assertFalse((self.root / "targets").exists())

    def test_resume_reuses_committed_annotation_units(self):
        original = models.HistoryRun.save_unit
        def interrupt(store, record, arrays):
            original(store, record, arrays)
            raise KeyboardInterrupt()
        with mock.patch.object(models.HistoryRun, "save_unit", interrupt), self.assertRaises(KeyboardInterrupt):
            targets.export_targets(self.args("--gallery-limit", "0"))
        self.assertEqual(read_json(self.root / "targets/run.json")["status"], "interrupted")
        summary = targets.export_targets(self.args("--gallery-limit", "0", "--resume"))
        self.assertEqual(summary["processing"]["reused_observations"], 1)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required for training smoke")
    def test_real_training_loop_accepts_derived_masks_without_fake_review(self):
        import torch
        torch.set_num_threads(1)
        targets.export_targets(self.args("--gallery-limit", "0"))
        args = models.parser().parse_args(["train", "--features", str(self.fixture.features),
                    "--labels", str(self.root / "targets/labels.csv"), "--allow-derived-labels",
                    "--models", "gru", "--time-modes", "gaps", "--width", "8", "--epochs", "1",
                    "--out", str(self.root / "trained")])
        models.train_models(args)
        bundle = read_json(self.root / "trained/models.json")
        self.assertIn("annotation_derived", bundle["training_reference"]["train"])
        self.assertIn("not an independent", bundle["models"]["gru_gaps"]["val"]["reference_scope"])
        models.evaluate_models(models.parser().parse_args([
            "evaluate", "--features", str(self.fixture.features),
            "--training-run", str(self.root / "trained"),
            "--labels", str(self.root / "targets/labels.csv"), "--allow-derived-labels",
            "--splits", "val", "--bootstrap", "0", "--gallery-limit", "2",
            "--out", str(self.root / "evaluated")]))
        with (self.root / "evaluated/predictions.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(all(r["review_status"] == "derived" for r in rows))
        self.assertTrue(all(r["label_kind"] == "annotation_derived" for r in rows))


if __name__ == "__main__":
    unittest.main()
