"""Scientific contract tests for shared semantic-route reports (no model weights)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.real_eval.semantic_report import (aggregate_metrics, change_metrics,
                                               render_report, segmentation_metrics)


class SemanticReportTests(unittest.TestCase):
    def setUp(self):
        # Deliberately shuffled COCO IDs/list order: bit0=A, bit1=B, bit2=ADS.
        self.categories = [{'id': 19, 'name': 'B'}, {'id': 2, 'name': 'A'},
                           {'id': 71, 'name': 'ADVERTISEMENTS'}]
        self.mapping = [{'category_id': 19, 'channel': 1}, {'category_id': 71, 'channel': 2},
                        {'category_id': 2, 'channel': 0}]

    def semantic(self, predicted, packed, names=('B', 'A'), support=None):
        if support is None:
            support = np.ones(packed.shape, bool)
        return segmentation_metrics(predicted.astype(np.float32), predicted, list(names), packed, support,
                                    self.categories, self.mapping)

    def test_names_and_explicit_channels_preserve_overlap(self):
        packed = np.array([[1, 3, 2]], np.uint16)
        predicted = np.array([[[False, True, True]], [[True, True, False]]])
        rows = {row['class_name']: row for row in self.semantic(predicted, packed)}
        self.assertEqual(rows['A']['tp'], 2)
        self.assertEqual(rows['B']['tp'], 2)
        self.assertEqual(rows['A']['iou'], 1.)
        self.assertEqual(rows['B']['channel'], 1)
        self.assertEqual(rows['ADVERTISEMENTS']['status'], 'unsupported_class')
        self.assertIsNone(rows['ADVERTISEMENTS']['fn'])
        self.assertIsNone(rows['ADVERTISEMENTS']['iou'])

    def test_unknown_and_invalid_pixels_never_become_healthy(self):
        packed = np.array([[1, 0, 2, 1]], np.uint16)
        support = np.array([[True, True, True, False]])
        predicted = np.array([[[False, True, True, True]], [[True, True, False, True]]])
        rows = self.semantic(predicted, packed, support=support)
        self.assertEqual(rows[0]['valid_pixels'], 2)
        self.assertEqual(rows[0]['ignored_pixels'], 2)
        self.assertEqual(rows[0]['fp'], 0)
        self.assertEqual(rows[1]['fp'], 0)

    def test_change_missed_outside_roi_stays_false_negative(self):
        before = np.array([[1, 1, 1, 0]], np.uint16)
        after = np.array([[2, 2, 1, 2]], np.uint16)
        prediction = np.array([[True, False, False, True]])
        # Only first pixel was computed. NaN scores do not shrink evaluation.
        scores = np.array([[.9, np.nan, np.nan, np.nan]], np.float32)
        rows = change_metrics(prediction, scores, before, after, np.ones(before.shape, bool),
                              self.categories, self.mapping)
        overall = rows[0]
        self.assertEqual((overall['tp'], overall['fn'], overall['tn']), (1, 1, 1))
        self.assertEqual(overall['valid_pixels'], 3)
        self.assertEqual(overall['recall'], .5)
        b_gain = next(row for row in rows if row['class_name'] == 'B' and row['direction'] == 'gain')
        self.assertEqual((b_gain['tp'], b_gain['fn']), (1, 1))
        self.assertEqual(b_gain['target'], 'class_conditioned_localisation')

    def test_one_date_unknown_ignored(self):
        before, after = np.array([[1, 0]], np.uint16), np.array([[0, 2]], np.uint16)
        rows = change_metrics(np.ones(before.shape, bool), None, before, after, np.ones(before.shape, bool),
                              self.categories, self.mapping)
        self.assertEqual(rows[0]['status'], 'no_evaluable_pixels')
        self.assertIsNone(rows[0]['recall'])

    def test_bad_mapping_rejected(self):
        with self.assertRaises(ValueError):
            segmentation_metrics(None, np.zeros((1, 1, 1), bool), ['A'], np.ones((1, 1), np.uint16),
                                 np.ones((1, 1), bool), self.categories,
                                 [{'category_id': 19, 'channel': 0}, {'category_id': 2, 'channel': 0},
                                  {'category_id': 71, 'channel': 2}])

    def test_facades_equal_weight_and_splits_separate(self):
        packed = np.array([[1]], np.uint16)
        good = self.semantic(np.array([[[False]], [[True]]]), packed)[0]
        bad = self.semantic(np.array([[[False]], [[False]]]), packed)[0]
        rows = []
        for index in range(9):
            rows.append(dict(bad, method='m', route='full', split='val', building_id='many', frame=index))
        rows.append(dict(good, method='m', route='full', split='val', building_id='one', frame=0))
        rows.append(dict(good, method='m', route='full', split='test', building_id='heldout', frame=0))
        result = aggregate_metrics(rows)
        summary = {row['split']: row for row in result['summary']}
        self.assertEqual(summary['val']['iou'], .5)
        self.assertEqual(summary['val']['evaluated_facades'], 2)
        self.assertEqual(summary['test']['iou'], 1.)
        self.assertEqual(set(summary), {'val', 'test'})

    def test_unsupported_stays_unavailable_in_aggregation(self):
        row = self.semantic(np.zeros((2, 1, 1), bool), np.array([[4]], np.uint16))[-1]
        result = aggregate_metrics([dict(row, split='test', building_id='one', method='legacy')])
        summary = result['summary'][0]
        self.assertEqual(summary['status'], 'unsupported_class')
        self.assertIsNone(summary['iou'])
        self.assertIsNone(summary['tp'])
        self.assertEqual(summary['evaluated_facades'], 0)

    def test_raw_sam_masks_keep_prompts_and_area_before_fusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packed = np.array([[1, 1], [2, 2]], np.uint16)
            prediction = np.stack([(packed & 2) != 0, (packed & 1) != 0])
            result_path, native_path = root / 'result.npz', root / 'native.npz'
            np.savez_compressed(result_path, predictions=prediction, packed=packed,
                                source=np.full((2, 2, 3), 128, np.uint8), support=np.ones((2, 2), bool))
            masks = np.array([[[True, True], [True, True]], [[True, False], [False, False]]])
            np.savez_compressed(native_path, extra__instance_masks=masks,
                                extra__instance_scores=np.array([.9, .7], np.float32),
                                extra__instance_class_indices=np.array([1, 0], np.int32),
                                extra__instance_prompt_indices=np.array([1, 0], np.int32))
            context = dict(kind='segmentation', split='val', building_id='facade', sequence_id='seq',
                           frame=0, method=None, segmenter='sam3', route='baseline',
                           cache_artifact=str(result_path), native_cache_artifact=str(native_path),
                           native_method_metadata={'prompt_texts': ['window', 'wall surface']},
                           class_names=['B', 'A'], status='completed')
            result = render_report(root / 'report', {'categories': self.categories,
                                   'category_to_channel': self.mapping},
                                   [dict(context, variant='expanded_flat'), dict(context, variant='expanded_layered')])
            self.assertEqual(result['gallery_errors'], 0)
            self.assertEqual(result['sam3_native_inputs'], 1)
            diagnostic = json.loads((root / 'report/sam3_instances.json').read_text())[0]
            self.assertEqual(diagnostic['instance_count'], 2)
            self.assertEqual(diagnostic['max_area_fraction'], 1.)
            self.assertEqual(diagnostic['large_mask_count'], 1)
            self.assertEqual(diagnostic['instances'][0]['prompt'], 'wall surface')
            self.assertEqual(diagnostic['instances'][0]['class_name'], 'A')
            page = (root / 'report/report.html').read_text()
            self.assertEqual(page.count('Исходная маска SAM3 #0:'), 1)
            self.assertIn('href="#sam-', page)

    def test_report_without_gpu_has_rows_gallery_and_failure_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packed = np.array([[1, 3], [2, 0]], np.uint16)
            predicted = np.stack([(packed & 2) != 0, (packed & 1) != 0])
            artifact = root / 'prediction.npz'
            np.savez_compressed(artifact, predictions=predicted,
                                class_names=np.array(['B', 'A']), packed=packed,
                                source=np.full((2, 2, 3), 128, np.uint8), support=np.ones((2, 2), bool))
            context = dict(kind='segmentation', split='val', building_id='facade', sequence_id='seq',
                           frame=0, method='maskclip', variant='ours', route='full')
            records = [dict(context, cache_artifact=str(artifact), metrics=self.semantic(predicted, packed),
                            elapsed_s=1.25, status='completed'),
                       dict(context, method='missing_model', status='failed', error='<failed>')]
            result = render_report(root / 'report', {'categories': self.categories,
                                   'category_to_channel': self.mapping}, records)
            self.assertEqual(result['metric_rows'], 3)
            self.assertEqual(result['gallery_cards'], 1)
            self.assertEqual(result['gallery_errors'], 0)
            page = (root / 'report/report.html').read_text()
            self.assertIn('missing_model', page)
            self.assertIn('&lt;failed&gt;', page)
            self.assertIn('N/A', page)
            metrics = json.loads((root / 'report/metrics.json').read_text())
            self.assertEqual(len(metrics['summary']), 3)
            self.assertTrue(list((root / 'report/assets').glob('*.jpg')))


if __name__ == '__main__':
    unittest.main()
