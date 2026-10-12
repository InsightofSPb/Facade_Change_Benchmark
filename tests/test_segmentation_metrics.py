import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from facade_change.segmentation.metrics import (
    aggregate_metrics,
    confusion_counts,
    group_confusion_counts,
    loss_statistics,
    merge_loss_statistics,
    resolve_groups,
    write_gallery,
    write_full_image_panels,
    write_full_image_gallery,
    semantic_legend,
    write_reports,
)


class SegmentationMetricsTests(unittest.TestCase):
    def setUp(self):
        self.categories = [{"id": 8, "name": "CRACK"}, {"id": 2, "name": "REPAIRS"}]

    def test_overlapping_labels_and_ignored_pixels(self):
        target = np.array([[[1, 1], [0, 0]], [[1, 0], [1, 0]]], dtype=bool)
        prediction = np.array([[[1, 0], [1, 1]], [[1, 1], [1, 1]]], dtype=bool)
        valid = np.array([[1, 1], [0, 0]], dtype=bool)
        counts = confusion_counts(prediction, target, valid)
        np.testing.assert_array_equal(counts, [[1, 0, 1], [1, 1, 0]])
        self.assertEqual(counts.dtype, np.int64)

    def test_equal_facades_and_count_pooling_inside_facade(self):
        records = [
            {"image_id": 1, "building_id": "large", "counts": [[90, 0, 0], [0, 0, 0]]},
            {"image_id": 2, "building_id": "large", "counts": [[0, 0, 10], [0, 0, 0]]},
            {"image_id": 3, "building_id": "small", "counts": [[0, 0, 1], [0, 0, 0]]},
        ]
        result = aggregate_metrics(records, self.categories)
        self.assertAlmostEqual(result["primary"]["miou"], 0.45)
        self.assertAlmostEqual(result["pixel_pooled_secondary"]["miou"], 90 / 101)
        self.assertEqual(result["primary"]["building_count"], 2)
        self.assertEqual(result["per_building"][0]["image_count"], 2)
        self.assertEqual(result["per_class"][0]["category_id"], 8)
        self.assertEqual(result["per_class"][1]["eligible_buildings"]["iou"], 0)

    def test_absent_ground_truth_with_false_positive_is_zero_iou(self):
        result = aggregate_metrics([
            {"image_id": 1, "building_id": "facade", "counts": [[4, 0, 0], [0, 2, 0]]}
        ], self.categories)
        self.assertEqual(result["primary"]["miou"], 0.5)
        self.assertEqual(result["per_class"][1]["iou"], 0)
        self.assertIsNone(result["per_class"][1]["recall"])
        self.assertEqual(result["per_class"][1]["precision"], 0)

    def test_all_invalid_is_null_and_json_safe(self):
        labels = np.ones((2, 3, 3), dtype=bool)
        counts = confusion_counts(labels, labels, np.zeros((3, 3), dtype=bool))
        result = aggregate_metrics([
            {"image_id": 1, "building_id": "facade", "counts": counts.tolist()}
        ], self.categories)
        self.assertIsNone(result["primary"]["miou"])
        self.assertEqual(result["primary"]["eligible_buildings"]["miou"], 0)
        json.dumps(result, allow_nan=False)
        self.assertIsNone(aggregate_metrics([], self.categories)["primary"]["miou"])

    def test_native_images_must_not_be_counted_twice(self):
        record = {"image_id": 1, "building_id": "facade", "counts": [[1, 0, 0], [0, 0, 0]]}
        with self.assertRaisesRegex(ValueError, "duplicate native image"):
            aggregate_metrics([record, record], self.categories)
        with self.assertRaisesRegex(ValueError, "confirmed building_id"):
            aggregate_metrics([{**record, "building_id": None}], self.categories)

    def test_invalid_counts_and_unthresholded_scores_rejected(self):
        record = {"image_id": 1, "building_id": "facade", "counts": [[1.5, 0, 0], [0, 0, 0]]}
        with self.assertRaisesRegex(ValueError, "nonnegative integers"):
            aggregate_metrics([record], self.categories)
        with self.assertRaisesRegex(ValueError, "boolean"):
            confusion_counts(np.zeros((2, 3, 3)), np.zeros((2, 3, 3), bool), np.ones((3, 3), bool))

    def test_csv_reports_keep_undefined_values_empty(self):
        metrics = aggregate_metrics([
            {"image_id": 1, "building_id": "facade", "counts": [[1, 0, 1], [0, 0, 0]]}
        ], self.categories)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_reports(directory, metrics)
            with Path(paths["per_class"]).open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["category_id"], "8")
            self.assertEqual(rows[1]["iou"], "")
            self.assertEqual(rows[1]["iou_buildings"], "0")
            with Path(paths["per_building"]).open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["miou"], "0.5")

    def test_f1_is_computed_before_facade_averaging(self):
        records = [
            {"image_id": 1, "building_id": "a", "counts": [[1, 0, 9], [0, 0, 0]]},
            {"image_id": 2, "building_id": "b", "counts": [[1, 9, 0], [0, 0, 0]]},
        ]
        result = aggregate_metrics(records, self.categories)
        self.assertAlmostEqual(result["primary"]["f1"], 2 / 11)
        self.assertAlmostEqual(result["per_class"][0]["f1"], 2 / 11)
        self.assertNotAlmostEqual(result["primary"]["f1"], .55)

    def test_group_union_counts_overlapping_labels_once(self):
        groups = [{"name": "both", "channels": [0, 1], "category_ids": [8, 2],
                   "category_names": ["CRACK", "REPAIRS"]}]
        target = np.array([[[1, 1, 0]], [[1, 0, 1]]], bool)
        prediction = np.array([[[1, 0, 0]], [[1, 1, 0]]], bool)
        valid = np.ones((1, 3), bool)
        class_counts = confusion_counts(prediction, target, valid)
        group_counts = group_confusion_counts(prediction, target, valid, groups)
        np.testing.assert_array_equal(group_counts, [[2, 0, 1]])
        self.assertNotEqual(class_counts[:, 1].sum(), group_counts[:, 1].sum())
        result = aggregate_metrics([{"image_id": 1, "building_id": "f",
                                    "counts": class_counts, "group_counts": group_counts}],
                                   self.categories, groups)
        self.assertAlmostEqual(result["per_group"][0]["f1"], .8)
        self.assertAlmostEqual(result["per_group"][0]["miou"], 2 / 3)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_reports(directory, result)
            self.assertTrue(Path(paths["per_facade_group"]).is_file())
            with Path(paths["per_group"]).open(encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["f1"], "0.8")

    def test_loss_native_statistics_do_not_depend_on_tile_batching(self):
        logits = np.array([[[1., -2., 3., -.5]], [[-.25, .5, 1., -4.]]])
        probabilities = 1 / (1 + np.exp(-logits))
        target = np.array([[[1, 0, 1, 0]], [[0, 1, 0, 1]]], bool)
        valid = np.array([[1, 1, 1, 0]], bool)
        whole = loss_statistics(probabilities, target, valid, logits)
        parts = [loss_statistics(probabilities[:, :, i:i+2], target[:, :, i:i+2],
                                 valid[:, i:i+2], logits[:, :, i:i+2]) for i in (0, 2)]
        merged = merge_loss_statistics(parts)
        for key in whole:
            np.testing.assert_allclose(whole[key], merged[key])
        result = aggregate_metrics([{"image_id": 1, "building_id": "f",
                                    "counts": confusion_counts(probabilities >= .5, target, valid),
                                    "loss_stats": merged}], self.categories)
        bce = np.asarray(whole["bce_sum"]).sum() / (2 * valid.sum())
        dice = np.mean(1 - (2 * np.asarray(whole["intersection"]) + 1e-6) /
                       (np.asarray(whole["probability_sum"]) + np.asarray(whole["target_sum"]) + 1e-6))
        self.assertAlmostEqual(result["primary"]["loss"], .5 * bce + .5 * dice)
        self.assertIsNone(merge_loss_statistics([]))

    def test_loss_stability_absent_classes_and_group_restriction(self):
        logits = np.array([[[1000., -1000.]], [[1000., -1000.]]])
        target = np.array([[[1, 1]], [[0, 0]]], bool)
        probabilities = np.array([[[1., 0.]], [[1., 0.]]])
        valid = np.ones((1, 2), bool)
        groups = resolve_groups(self.categories)
        stats = loss_statistics(probabilities, target, valid, logits)
        result = aggregate_metrics([{"image_id": 1, "building_id": "f",
                                    "counts": confusion_counts(probabilities >= .5, target, valid),
                                    "group_counts": group_confusion_counts(probabilities >= .5, target, valid, groups),
                                    "loss_stats": stats}], self.categories, groups)
        # REPAIRS has no positives: its Dice loss is excluded, not assigned one.
        repair = next(row for row in result["per_group"] if row["name"] == "repairs")
        self.assertEqual(repair["dice"], 0.)
        self.assertEqual(repair["bce"], 500.)
        self.assertEqual(repair["loss"], 250.)
        json.dumps(result, allow_nan=False)

    def test_unknown_train_images_are_never_invented_facades(self):
        result = aggregate_metrics([
            {"image_id": 1, "building_id": "known", "counts": [[1, 0, 0], [0, 0, 0]]},
            {"image_id": 2, "building_id": None, "counts": [[0, 0, 100], [0, 0, 0]]},
        ], self.categories, allow_unknown_buildings=True)
        self.assertEqual(result["primary"]["building_count"], 1)
        self.assertEqual(result["primary"]["miou"], 1)
        self.assertEqual(result["auxiliary_image_average"]["image_count"], 1)
        self.assertEqual(result["auxiliary_image_average"]["miou"], 0)
        self.assertAlmostEqual(result["pixel_pooled_secondary"]["miou"], 1 / 101)

    def test_group_resolution_is_explicit_and_matches_channel_order(self):
        groups = resolve_groups(self.categories)
        self.assertEqual(groups[0]["channels"], [0])
        self.assertEqual(groups[1]["channels"], [1])
        self.assertEqual(resolve_groups([{"id": 123, "name": "FIXTURE"}])[0]["name"], "class:FIXTURE")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "groups.json"
            for members, message in [(["CRACK", "CRACK"], "duplicates"),
                                     (["TYPO"], "unknown categories")]:
                path.write_text(json.dumps({"schema_version": 1,
                                           "groups": [{"name": "g", "categories": members}]}))
                with self.assertRaisesRegex(ValueError, message):
                    resolve_groups(self.categories, path)

    def test_partial_loss_statistics_are_not_silently_averaged(self):
        row = {"image_id": 1, "building_id": "known", "counts": [[1, 0, 0], [0, 0, 0]]}
        stats = loss_statistics(np.zeros((2, 1, 1)), np.zeros((2, 1, 1), bool), np.ones((1, 1), bool))
        with self.assertRaisesRegex(ValueError, "every image or none"):
            aggregate_metrics([row, {**row, "image_id": 2, "loss_stats": stats}], self.categories)

    def test_gallery_preserves_multilabel_overlaps_and_marks_ignored_support(self):
        categories = [{"id": 8, "name": "CRACK", "color": "#ff0000"},
                      {"id": 2, "name": "REPAIRS", "color": "#00ff00"}]
        rgb = np.full((24, 24, 3), 120, dtype=np.uint8)
        labels = np.zeros((2, 24, 24), bool)
        labels[:, 4:20, 4:20] = True
        valid = np.ones((24, 24), bool)
        valid[:2] = False
        # Predictions outside support must not appear as evaluated classes.
        prediction = labels.copy()
        prediction[:, :2] = True
        original_prediction = prediction.copy()
        preview = {"rgb": rgb, "target": labels, "valid": valid, "prediction": prediction,
                   "image_id": 5, "building_id": "facade <test>", "crop_id": "fixed-01"}
        with tempfile.TemporaryDirectory() as directory:
            gallery = write_gallery(directory, [preview], categories, epoch=4, checkpoint_name="best.pth")
            page = gallery.read_text(encoding="utf-8")
            for text in ("Оригинальный RGB", "Разметка на RGB", "Прогноз на RGB",
                         "Предсказанные классы", "Серая клетка", "полосами", "best.pth",
                         "эпоха 4", "Трещина", "Ремонт", "facade &lt;test&gt;"):
                self.assertIn(text, page)
            self.assertNotIn("facade <test>", page)
            folder = Path(directory) / "previews"
            with Image.open(folder / "val-00-classes.png") as image:
                colors = np.asarray(image)
            region_colors = set(map(tuple, colors[4:20, 4:20].reshape(-1, 3)))
            self.assertEqual(region_colors, {(255, 0, 0), (0, 255, 0)})
            np.testing.assert_array_equal(colors[2, 0], [0, 0, 0])
            self.assertTrue(np.all(colors[:2, :, 0] == colors[:2, :, 1]))
            self.assertEqual(len(list(folder.glob("*-class-*.jpg"))), 2)
            for category in categories:
                with Image.open(folder / f"val-00-class-{category['id']}.jpg") as pair:
                    self.assertEqual(pair.size, (48, 24))
            np.testing.assert_array_equal(prediction, original_prediction)

    def test_gallery_rejects_inconsistent_channels_before_writing(self):
        preview = {"rgb": np.zeros((3, 3, 3), np.uint8), "target": np.zeros((1, 3, 3), bool),
                   "prediction": np.zeros((1, 3, 3), bool), "valid": np.ones((3, 3), bool)}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "preview"
            with self.assertRaisesRegex(ValueError, "one target/prediction channel"):
                write_gallery(out, [preview], self.categories)
            self.assertFalse(out.exists())


    def test_full_photo_panels_preserve_native_pixels_and_multilabel_bits(self):
        categories = [{"id": 8, "name": "CRACK", "color": "#ff0000"},
                      {"id": 2, "name": "REPAIRS", "color": "#00ff00"},
                      {"id": 42, "name": "ABSENT", "color": "#0000ff"}]
        # Cross the rendering block boundary. Pixel values and full dimensions
        # must survive export, including unknown GT and overlapping classes.
        rgb = (np.arange(270 * 25 * 3).reshape(270, 25, 3) % 256).astype(np.uint8)
        target = np.zeros((270, 25), np.uint16)
        target[3:267, 2:23] = 3
        prediction = target.copy()
        prediction[:2, :] = 2  # Full-photo inference is visible outside GT.
        valid = target != 0
        before = [array.copy() for array in (rgb, target, prediction, valid)]
        with tempfile.TemporaryDirectory() as directory:
            section = write_full_image_panels(
                directory, rgb=rgb, target=target, prediction=prediction, valid=valid,
                categories=categories, image_id="../../unsafe?<id>",
                building_id="facade <test>", split="test", source_name="source.jpg")
            self.assertEqual(len(section["panels"]), 5)
            folder = Path(directory) / Path(section["panels"][0]["src"]).parent
            self.assertTrue(folder.resolve().is_relative_to(Path(directory).resolve()))
            self.assertNotIn("unsafe", folder.name)
            for panel in section["panels"]:
                self.assertEqual(panel["src"], panel["href"])
                with Image.open(Path(directory) / panel["href"]) as image:
                    self.assertEqual(image.size, (25, 270))
            with Image.open(folder / "original.png") as image:
                np.testing.assert_array_equal(np.asarray(image), rgb)
            with Image.open(folder / "gt-masks.png") as image:
                rendered = np.asarray(image)
                region = rendered[3:267, 2:23]
                self.assertEqual(set(map(tuple, region.reshape(-1, 3))), {(255, 0, 0), (0, 255, 0)})
                self.assertTrue(np.all(rendered[:2, :, 0] == rendered[:2, :, 1]))
            with Image.open(folder / "gt-overlay.png") as image:
                np.testing.assert_array_equal(np.asarray(image)[:2], rgb[:2])
            with Image.open(folder / "predicted-masks.png") as image:
                np.testing.assert_array_equal(np.asarray(image)[:2], np.broadcast_to([0, 255, 0], (2, 25, 3)))
            with np.load(folder / "masks.npz", allow_pickle=False) as masks:
                np.testing.assert_array_equal(masks["target_bits"], target)
                np.testing.assert_array_equal(masks["prediction_bits"], prediction)
                np.testing.assert_array_equal(masks["valid"], valid)
                np.testing.assert_array_equal(masks["category_ids"], [8, 2, 42])
                self.assertEqual(masks["target_bits"].dtype, np.uint16)
            # Repeat identities must not silently overwrite a previous export.
            with self.assertRaises(FileExistsError):
                write_full_image_panels(directory, rgb=rgb, target=target, prediction=prediction,
                                        valid=valid, categories=categories,
                                        image_id="../../unsafe?<id>", split="test")
        for actual, expected in zip((rgb, target, prediction, valid), before):
            np.testing.assert_array_equal(actual, expected)

    def test_large_photo_previews_bound_browser_decode_but_link_native_pixels(self):
        rgb = np.full((37, 3201, 3), 123, np.uint8)
        bits = np.ones((37, 3201), np.uint16)
        with tempfile.TemporaryDirectory() as directory:
            section = write_full_image_panels(
                directory, rgb=rgb, target=bits, prediction=bits,
                valid=np.ones(bits.shape, bool), categories=self.categories,
                image_id=99, building_id="large", split="test")
            for panel in section["panels"]:
                self.assertNotEqual(panel["src"], panel["href"])
                self.assertTrue(panel["src"].endswith("-preview.jpg"))
                with Image.open(Path(directory) / panel["src"]) as preview:
                    self.assertEqual(preview.width, 1600)
                    self.assertLessEqual(max(preview.size), 1600)
                with Image.open(Path(directory) / panel["href"]) as native:
                    self.assertEqual(native.size, (3201, 37))
            with Image.open(Path(directory) / section["panels"][0]["href"]) as native:
                np.testing.assert_array_equal(np.asarray(native), rgb)
            page = write_full_image_gallery(directory, [section], self.categories,
                                            split="test").read_text(encoding="utf-8")
            for panel in section["panels"]:
                self.assertIn(panel["src"], page)
                self.assertIn(panel["href"], page)

    def test_full_photo_invalid_channel_bits_fail_before_writing(self):
        rgb = np.zeros((3, 3, 3), np.uint8)
        target = np.zeros((3, 3), np.uint16)
        prediction = target.copy()
        prediction[0, 0] = 4  # Only two category channels exist.
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "gallery"
            with self.assertRaisesRegex(ValueError, "outside the category channels"):
                write_full_image_panels(out, rgb=rgb, target=target, prediction=prediction,
                                        valid=np.ones((3, 3), bool), categories=self.categories,
                                        image_id=1)
            self.assertFalse(out.exists())
            with self.assertRaisesRegex(ValueError, "uint16 bit masks"):
                write_full_image_panels(out, rgb=rgb, target=target.astype(bool), prediction=target,
                                        valid=np.ones((3, 3), bool), categories=self.categories,
                                        image_id=1)
            self.assertFalse(out.exists())

    def test_full_photo_gallery_is_static_with_complete_legend(self):
        categories = [{"id": 8, "name": "CRACK", "color": "#ff0000"},
                      {"id": 2, "name": "REPAIRS", "color": "#00ff00"},
                      {"id": 42, "name": "ABSENT", "color": "#0000ff"}]
        with tempfile.TemporaryDirectory() as directory:
            section = write_full_image_panels(
                directory, rgb=np.zeros((10, 17, 3), np.uint8),
                target=np.zeros((10, 17), np.uint16), prediction=np.zeros((10, 17), np.uint16),
                valid=np.zeros((10, 17), bool), categories=categories,
                image_id=5, building_id="facade <unsafe>", split="test")
            page = write_full_image_gallery(directory, [section], categories, split="test",
                                            checkpoint_name="best.pth").read_text(encoding="utf-8")
            for text in ("facade &lt;unsafe&gt;", "ABSENT", "#0000ff", "best.pth", "GT MASKS",
                         "PREDICTED MASKS", "не оценён", "не здоровый фон", "masks.npz"):
                self.assertIn(text, page)
            for forbidden in ("<details", "<select", "<script", "facade <unsafe>"):
                self.assertNotIn(forbidden, page)
            for panel in section["panels"]:
                self.assertIn(panel["src"], page)
            self.assertEqual(len(semantic_legend(categories)), 3)
            self.assertEqual(semantic_legend(categories)[0]["color"], "#ff0000")

    def test_shared_gallery_escapes_labels_and_rejects_external_assets(self):
        from facade_change.visualization import write_static_gallery

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / 'frame & 1.png'
            Image.new("RGB", (4, 3)).save(asset)
            section = {"title": '<script>bad</script>', "caption": '<img src=x>',
                       "panels": [{"label": '<unsafe>', "src": asset.name}]}
            page = write_static_gallery(root, '<title>', [section]).read_text()
            self.assertIn('frame%20%26%201.png', page)
            self.assertIn('&lt;script&gt;bad&lt;/script&gt;', page)
            self.assertNotIn('<script>', page)
            for source in ('../outside.png', str(asset.resolve())):
                with self.assertRaisesRegex(ValueError, 'within the output'):
                    write_static_gallery(root, 'Bad source', [
                        {"title": 'external', "panels": [{"label": 'external', "src": source}]}])
            with self.assertRaisesRegex(ValueError, '#RRGGBB'):
                write_static_gallery(root, 'Bad color', [], legend=[
                    {"name": 'class', "color": '#fff; background:url(x)'}])


if __name__ == "__main__":
    unittest.main()
