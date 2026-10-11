"""CPU training integration and native-pixel validation ownership contracts."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from facade_change.io import read_json, sha256, write_json


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_COCO = importlib.util.find_spec("pycocotools") is not None
if HAS_TORCH:
    import torch
    from torch import nn
    from facade_change.segmentation import data as segmentation_data
    from facade_change.segmentation.model import FrozenDinoSegmenter
    from facade_change.segmentation.prepared import PreparedAugmentations, prepare_augmentations
    from facade_change.segmentation.train import dino_paths, ownership_masks, train

    class TinyBackbone(nn.Module):
        patch_size = 14
        embed_dim = 6

        def __init__(self):
            super().__init__()
            self.project = nn.Conv2d(3, self.embed_dim, kernel_size=14, stride=14)

        def get_intermediate_layers(self, value, n, reshape):
            if not reshape:
                raise AssertionError("segmentation requires feature maps")
            feature = self.project(value)
            return tuple(feature * (index + 1) for index in range(n))


def make_grid(width, height, size, xs, ys):
    return [
        {"crop_id": f"{x}:{y}", "x": x, "y": y,
         "width": min(size, width - x), "height": min(size, height - y)}
        for y in ys for x in xs
    ]


@unittest.skipUnless(HAS_TORCH, "optional torch dependency is unavailable")
class OwnershipTests(unittest.TestCase):
    def test_edge_anchored_overlapping_grid_counts_each_native_pixel_once(self):
        width, height, size = 16, 17, 8
        # Last crops are edge-anchored and therefore overlap more than the stride.
        rows = make_grid(width, height, size, [0, 6, 8], [0, 6, 9])
        masks = ownership_masks(rows, width, height, size)
        coverage = np.zeros((height, width), dtype=np.uint8)
        for row in rows:
            x, y, w, h = (row[key] for key in ("x", "y", "width", "height"))
            coverage[y:y+h, x:x+w] += masks[row["crop_id"]][:h, :w]
        np.testing.assert_array_equal(coverage, np.ones_like(coverage))

    def test_padding_is_not_owned(self):
        rows = make_grid(5, 3, 8, [0], [0])
        mask = ownership_masks(rows, 5, 3, 8)["0:0"]
        self.assertEqual(mask.sum(), 15)
        self.assertTrue(mask[:3, :5].all())
        self.assertFalse(mask[3:].any())
        self.assertFalse(mask[:, 5:].any())

    def test_holes_missing_tiles_and_boundaries_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "uncovered gaps"):
            ownership_masks(make_grid(18, 8, 8, [0, 10], [0]), 18, 8, 8)
        rows = make_grid(14, 14, 8, [0, 6], [0, 6])
        with self.assertRaisesRegex(ValueError, "complete rectangular grid"):
            ownership_masks(rows[:-1], 14, 14, 8)
        with self.assertRaisesRegex(ValueError, "image boundaries"):
            ownership_masks(make_grid(14, 8, 8, [6], [0]), 14, 8, 8)

    def test_methods_config_paths_are_resolved_at_each_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_folder = root / "configs"
            config_folder.mkdir()
            source = root / "source"
            source.mkdir()
            (source / "hubconf.py").write_text("# local fixture")
            weights = root / "weights.pth"
            weights.write_bytes(b"fixture")
            write_json(root / "methods.json", {"methods": {"dinov2": {
                "source_root": "source", "checkpoint_path": "weights.pth"}}})
            write_json(config_folder / "benchmark.json", {"methods_config": "../methods.json"})
            actual_root, actual_weights, hashes = dino_paths(config_folder / "benchmark.json")
            self.assertEqual(actual_root, str(source))
            self.assertEqual(actual_weights, str(weights))
            self.assertEqual(len(hashes), 2)
            write_json(root / "recursive.json", {"methods_config": "recursive.json"})
            with self.assertRaisesRegex(ValueError, "Recursive"):
                dino_paths(root / "recursive.json")


@unittest.skipUnless(HAS_TORCH and HAS_COCO, "optional torch/pycocotools dependencies unavailable")
class TrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "prepared"
        self.dataset.mkdir()
        categories = [
            {"id": 11, "name": "DELAMINATION"},
            {"id": 3, "name": "CRACK"},
            {"id": 9, "name": "REPAIRS"},
        ]
        manifest_images, coco_images, annotations, crops = [], [], [], []
        for image_id, split in enumerate(("train", "train", "val", "test"), 1):
            path = self.dataset / f"image-{image_id}.png"
            rgb = np.zeros((32, 35, 3), dtype=np.uint8)
            rgb[..., 0] = np.arange(35, dtype=np.uint8)[None] * 5
            rgb[..., 1] = np.arange(32, dtype=np.uint8)[:, None] * 5
            rgb[..., 2] = 50 + image_id * 20  # Distinct image bytes across splits.
            Image.fromarray(rgb).save(path)
            image_hash = sha256(path)
            manifest_images.append({
                "image_id": image_id, "file_name": path.name, "image_path": str(path),
                "sha256": image_hash, "width": 35, "height": 32,
                "building_id": f"facade-{image_id}", "view_id": f"view-{image_id}",
                "metadata_status": "reviewed", "split": split,
            })
            coco_images.append({"id": image_id, "file_name": path.name, "width": 35, "height": 32})
            for row in make_grid(35, 32, 32, [0, 3], [0]):
                crops.append({**row, "crop_id": f"{image_id}:{row['crop_id']}",
                              "image_id": image_id, "image_path": str(path),
                              "image_sha256": image_hash, "building_id": f"facade-{image_id}",
                              "view_id": f"view-{image_id}", "split": split,
                              "pad_right": 0, "pad_bottom": 0})
            if split != "test":
                # The labels intentionally overlap: CRACK and REPAIRS must coexist.
                for category_id, polygon in (
                    (3, [2, 2, 20, 2, 20, 26, 2, 26]),
                    (9, [10, 5, 30, 5, 30, 30, 10, 30]),
                    (11, [0, 20, 8, 20, 8, 32, 0, 32]),
                ):
                    annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                        "category_id": category_id, "segmentation": [polygon]})
            else:
                annotations.append({"id": 999, "image_id": image_id,
                                    "category_id": 999, "segmentation": None})
        write_json(self.dataset / "coco.json", {
            "images": coco_images, "categories": categories, "annotations": annotations,
        })
        write_json(self.dataset / "manifest.json", {"images": manifest_images, "categories": categories})
        write_json(self.dataset / "split.json", {})
        (self.dataset / "crops.jsonl").write_text("".join(json.dumps(row) + "\n" for row in crops))
        write_json(self.dataset / "dataset.json", {
            "kind": "semantic_crop_index", "crop_size": 32, "source_coco": "coco.json",
            "source_coco_sha256": sha256(self.dataset / "coco.json"), "manifest": "manifest.json",
            "crops": "crops.jsonl", "split": "split.json", "categories": categories,
        })
        # The TEST image is genuinely unavailable: successful TRAIN/VAL cannot have read it.
        (self.dataset / "image-4.png").unlink()
        self.source = self.root / "dino_source"
        self.source.mkdir()
        (self.source / "hubconf.py").write_text("# The model loader is mocked in this CPU test.")
        self.weights = self.root / "weights.pth"
        torch.save({}, self.weights)

    def test_real_optimization_best_reload_reports_and_test_isolation(self):
        loaded_models, initial_heads, initial_backbones, read_paths = [], [], [], []
        # Unknown-address singleton photographs stay in TRAIN. Their metrics
        # must not be presented as independent, confirmed facade statistics.
        manifest = read_json(self.dataset / "manifest.json")
        manifest["images"][1].update(building_id=None, view_id=None,
                                     metadata_status="unknown_building",
                                     training_role="auxiliary_semantic_train")
        write_json(self.dataset / "manifest.json", manifest)
        rows = [json.loads(line) for line in (self.dataset / "crops.jsonl").read_text().splitlines()]
        for row in rows:
            if row["image_id"] == 2:
                row.update(building_id=None, view_id=None)
        (self.dataset / "crops.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

        def load_fixture(source, checkpoint, num_classes, device, trust):
            model = FrozenDinoSegmenter(TinyBackbone(), num_classes).to(device)
            loaded_models.append(model)
            initial_heads.append({key: value.clone() for key, value in model.head.state_dict().items()})
            initial_backbones.append({key: value.clone() for key, value in model.backbone.state_dict().items()})
            return model, {"fixture": "TinyBackbone CPU integration", "backbone_frozen": True}

        original_load_rgb = segmentation_data.load_rgb

        def track_rgb(path):
            read_paths.append(Path(path).name)
            self.assertNotEqual(Path(path).name, "image-4.png")
            return original_load_rgb(path)

        augmentations = self.root / "augmentations"
        with patch("facade_change.segmentation.data.load_rgb", side_effect=track_rgb):
            prepared_summary = prepare_augmentations(
                self.dataset, augmentations, cutmix_prob=1., cutout_prob=1.)
        self.assertEqual(prepared_summary["train_originals"], 4)
        self.assertEqual(prepared_summary["train_augmented"], 8)
        self.assertEqual(prepared_summary["val_originals"], 2)
        self.assertEqual(set(read_paths), {"image-1.png", "image-2.png", "image-3.png"})
        sample_reads = []
        original_load = PreparedAugmentations.load

        def track_sample(instance, record):
            sample_reads.append((record["image_id"], record["variant"]))
            return original_load(instance, record)

        out = self.root / "trained"
        # Preparation owns all augmentation and full photograph decoding. This
        # makes an accidental return to on-the-fly training fail immediately.
        with patch("facade_change.segmentation.train.load_model", side_effect=load_fixture), \
                patch.object(PreparedAugmentations, "load", new=track_sample), \
                patch.object(segmentation_data.PreparedData, "load_image",
                             side_effect=AssertionError("training decoded a full photograph")), \
                patch("facade_change.segmentation.data.augment_crop",
                      side_effect=AssertionError("training generated an augmentation")), \
                patch("facade_change.segmentation.prepared.augment_crop",
                      side_effect=AssertionError("training generated an augmentation")), \
                patch("facade_change.segmentation.prepared.cutmix_crop",
                      side_effect=AssertionError("training generated CutMix")), \
                patch("facade_change.segmentation.prepared.cutout_crop",
                      side_effect=AssertionError("training generated CutOut")):
            summary = train(self.dataset, out, source_root=self.source, checkpoint=self.weights,
                            augmentations=augmentations, device="cpu", epochs=2, batch_size=2,
                            learning_rate=.02)
        self.assertEqual(summary["status"], "completed_exploratory")
        self.assertFalse(summary["test_evaluated"])
        self.assertEqual({image_id for image_id, _ in sample_reads}, {1, 2, 3})
        self.assertEqual({variant for _, variant in sample_reads}, {0, 1, 2})
        self.assertFalse(any(image_id == 3 and variant != 0 for image_id, variant in sample_reads))
        config = read_json(out / "config.json")
        self.assertEqual(config["training_image_ids"], [1, 2])
        self.assertEqual(config["validation_image_ids"], [3])
        self.assertEqual(config["category_to_channel"], [
            {"category_id": 3, "channel": 0}, {"category_id": 9, "channel": 1},
            {"category_id": 11, "channel": 2},
        ])
        model = loaded_models[0]
        self.assertTrue(any(not torch.equal(initial_heads[0][key], value)
                            for key, value in model.head.state_dict().items()))
        self.assertTrue(all(torch.equal(initial_backbones[0][key], value)
                            for key, value in model.backbone.state_dict().items()))
        self.assertTrue(all(parameter.grad is None for parameter in model.backbone.parameters()))
        best = torch.load(out / "best.pth", map_location="cpu", weights_only=True)
        last = torch.load(out / "last.pth", map_location="cpu", weights_only=True)
        self.assertEqual(last["epoch"], 2)
        self.assertEqual(best["epoch"], summary["best_epoch"])
        self.assertEqual(best["val_miou"], summary["best_val_miou"])
        for key, value in model.head.state_dict().items():
            torch.testing.assert_close(value, best["head_state_dict"][key])
        metrics = read_json(out / "metrics.json")
        self.assertEqual(metrics["primary"]["miou"], best["val_miou"])
        self.assertEqual(metrics["primary"], read_json(out / "best-val.json")["primary"])
        self.assertEqual(len(read_json(out / "history.json")), 2)
        for split, image_ids in (("train", {1, 2}), ("val", {3})):
            for epoch in (1, 2):
                epoch_metrics = read_json(out / f"{split}-epoch-{epoch:03d}.json")
                self.assertEqual({row["image_id"] for row in epoch_metrics["image_records"]}, image_ids)
                for key in ("f1", "miou", "precision", "recall", "loss"):
                    value = epoch_metrics["primary"][key]
                    self.assertIsNotNone(value, (split, epoch, key))
                    self.assertTrue(np.isfinite(value), (split, epoch, key))
                self.assertEqual(epoch_metrics["primary"]["building_count"], 1)
                if split == "train":
                    self.assertEqual(epoch_metrics["auxiliary_image_average"]["image_count"], 1)
                    self.assertEqual(epoch_metrics["per_auxiliary_image"][0]["image_id"], 2)
                per_group = {row["name"]: row for row in epoch_metrics["per_group"]}
                self.assertEqual(set(per_group), {"structural_damage", "repairs"})
                for group in per_group.values():
                    self.assertTrue(all(key in group for key in ("f1", "miou", "precision", "recall", "loss")))
                for record in epoch_metrics["image_records"]:
                    # TRAIN diagnostics also score original native pixels once,
                    # never three cached variants or overlapping crop duplicates.
                    np.testing.assert_array_equal(
                        np.asarray(record["counts"])[:, [0, 2]].sum(axis=1),
                        [18 * 24, 20 * 25, 8 * 12],
                    )
                    self.assertIn("group_counts", record)
                    self.assertIn("loss_stats", record)
                    # Group union merges the overlapping CRACK/DELAMINATION
                    # positives before counting, instead of summing classes.
                    np.testing.assert_array_equal(
                        np.asarray(record["group_counts"])[:, [0, 2]].sum(axis=1), [492, 500])
                    self.assertEqual(record["loss_stats"]["valid_pixels"], 782)
        self.assertEqual([row["image_id"] for row in metrics["image_records"]], [3])
        np.testing.assert_array_equal(
            np.asarray(metrics["image_records"][0]["counts"])[:, [0, 2]].sum(axis=1),
            [18 * 24, 20 * 25, 8 * 12],
        )
        for name in ("gallery.html", "per_class.csv", "per_building.csv", "run.json"):
            self.assertTrue((out / name).is_file(), name)
        self.assertEqual(len(list((out / "previews").glob("*-class-*.jpg"))), 3)
        record = read_json(out / "run.json")
        self.assertEqual(record["status"], "completed_exploratory")
        self.assertEqual(record["artifact_sha256"]["best.pth"], sha256(out / "best.pth"))

    def test_prepare_only_cli_needs_no_model_and_legacy_crop_budget_is_rejected(self):
        out = self.root / "cli-augmentations"
        command = [sys.executable, "-m", "facade_change.segmentation", "--dataset", str(self.dataset)]
        result = subprocess.run(command + ["--prepare-only", "--out", str(out)],
                                text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = read_json(out / "summary.json")
        self.assertEqual(summary["train_originals"], 4)
        self.assertEqual(summary["train_augmented"], 8)
        self.assertEqual(summary["val_originals"], 2)
        self.assertFalse(summary["test_rgb_loaded"])
        self.assertFalse(summary["test_annotations_rasterized"])
        config = read_json(out / "config.json")
        self.assertEqual(config["policy"]["cutmix_prob"], .25)
        self.assertEqual(config["policy"]["cutout_prob"], .125)
        rejected = subprocess.run(command + ["--crops-per-image", "4"],
                                  text=True, capture_output=True, timeout=30)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("unrecognized arguments", rejected.stderr)
        self.assertIn("--crops-per-image", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
