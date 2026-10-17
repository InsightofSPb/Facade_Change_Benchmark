"""Area filters use explicit denominators, known annotations and facade counts."""
import copy
import unittest

import numpy as np

from scripts.real_eval.robustness_review_stats import pair_statistics, summarize, THRESHOLDS

CATEGORIES = [{"id": 3, "name": "wall"}, {"id": 9, "name": "crack"}]


def pair(old, new, support=None, labels=None):
    old, new = np.asarray(old, np.uint16), np.asarray(new, np.uint16)
    if old.ndim == 1:
        old, new = old[None, :], new[None, :]
    support = np.ones(old.shape, bool) if support is None else np.asarray(support, bool)
    if labels is None:
        labels = np.full(old.shape, 255, np.uint8)
        known = support & (old != 0) & (new != 0)
        labels[known] = (old != new)[known]
    else:
        labels = np.asarray(labels, np.uint8).reshape(old.shape)
    return dict(support=support, labels=labels, reference_bits=old, source_bits=new)


def counted(changed, known=100, total=100):
    old, new = np.zeros((1, total), np.uint16), np.zeros((1, total), np.uint16)
    old[0, :known] = 1
    new[0, :known] = 1
    new[0, :changed] = 3
    return pair(old, new)


class ReviewStatisticsTests(unittest.TestCase):
    def test_sparse_19_pixel_example_has_two_distinct_denominators(self):
        old, new = np.zeros((512, 512), np.uint16), np.zeros((512, 512), np.uint16)
        old.flat[:9102] = 1
        new.flat[:9102] = 1
        new.flat[:19] = 3
        arrays = pair(old, new)
        before = {name: value.copy() for name, value in arrays.items()}
        stats = pair_statistics(arrays, CATEGORIES)
        self.assertEqual((stats['changed_pixels'], stats['known_pixels'], stats['total_pixels']), (19, 9102, 262144))
        self.assertAlmostEqual(stats['change_known_percent'], 100 * 19 / 9102)
        self.assertAlmostEqual(stats['change_crop_percent'], 100 * 19 / 262144)
        self.assertAlmostEqual(stats['coverage_percent'], 100 * 9102 / 262144)
        self.assertEqual(stats['unchanged_pixels'], 9083)
        self.assertEqual(stats['classes'][0]['changed_pixels'], 0)
        self.assertEqual(stats['classes'][1]['change_known_percent'], 100)
        for name, value in before.items():
            np.testing.assert_array_equal(arrays[name], value)

    def test_class_membership_not_all_class_change_inside_class_roi(self):
        arrays = pair([1, 1, 3, 2, 1, 0], [1, 3, 2, 1, 3, 1], labels=[0, 1, 1, 1, 255, 255])
        stats = pair_statistics(arrays, CATEGORIES)
        self.assertEqual(stats['known_pixels'], 4)
        self.assertEqual(stats['changed_pixels'], 3)
        wall, crack = stats['classes']
        self.assertEqual((wall['roi_pixels'], wall['changed_pixels'], wall['gain_pixels'], wall['loss_pixels'], wall['stable_pixels']),
                         (4, 2, 1, 1, 2))
        self.assertEqual(wall['change_known_percent'], 50)
        self.assertAlmostEqual(wall['change_crop_percent'], 100 * 2 / 6)
        self.assertEqual((crack['roi_pixels'], crack['changed_pixels'], crack['stable_pixels']), (3, 2, 1))
        self.assertAlmostEqual(crack['change_known_percent'], 100 * 2 / 3)
        # Pixel 4 has known bitsets but is ignored by boundary tolerance.
        # Pixel 5 has no old annotation: neither enters any class statistics.
        self.assertEqual(sum(c['gain_pixels'] + c['loss_pixels'] for c in stats['classes']), 4)

    def test_unknown_and_absent_classes_have_undefined_known_percentage(self):
        arrays = pair(np.zeros((2, 3)), np.zeros((2, 3)), support=np.zeros((2, 3), bool))
        stats = pair_statistics(arrays, CATEGORIES)
        self.assertIsNone(stats['change_known_percent'])
        self.assertIsNone(stats['coverage_support_percent'])
        self.assertEqual(stats['change_crop_percent'], 0)
        self.assertEqual(stats['coverage_percent'], 0)
        self.assertTrue(all(row['change_known_percent'] is None for row in stats['classes']))
        self.assertTrue(all(row['change_crop_percent'] == 0 for row in stats['classes']))
        stable = pair_statistics(pair([1, 1], [1, 1]), CATEGORIES)
        self.assertEqual(stable['classes'][0]['change_known_percent'], 0)
        self.assertIsNone(stable['classes'][1]['change_known_percent'])

    def test_geometric_support_and_annotation_coverage_are_distinct(self):
        arrays = pair([1, 1, 0, 0], [3, 1, 0, 0], support=[[True, True, True, False]])
        stats = pair_statistics(arrays, CATEGORIES)
        self.assertEqual(stats['support_pixels'], 3)
        self.assertEqual(stats['known_pixels'], 2)
        self.assertEqual(stats['coverage_percent'], 50)
        self.assertAlmostEqual(stats['coverage_support_percent'], 100 * 2 / 3)

    def test_invalid_labels_mapping_and_bitsets_are_rejected(self):
        original = pair([1, 1], [1, 3])
        cases = []
        wrong = copy.deepcopy(original); wrong['labels'][0, 0] = 1; cases.append(wrong)
        wrong = copy.deepcopy(original); wrong['labels'][0, 1] = 0; cases.append(wrong)
        wrong = copy.deepcopy(original); wrong['source_bits'][0, 0] = 4; cases.append(wrong)
        wrong = copy.deepcopy(original); wrong['reference_bits'][0, 0] = 0; cases.append(wrong)
        wrong = copy.deepcopy(original); wrong['support'][0, 0] = False; cases.append(wrong)
        wrong = copy.deepcopy(original); wrong['labels'][0, 0] = 2; cases.append(wrong)
        wrong = copy.deepcopy(original); wrong['labels'] = wrong['labels'].astype(float); cases.append(wrong)
        for arrays in cases:
            with self.subTest(arrays=arrays), self.assertRaises(ValueError):
                pair_statistics(arrays, CATEGORIES)
        with self.assertRaises(ValueError):
            pair_statistics(original, [CATEGORIES[0], CATEGORIES[0]])

    def test_thresholds_include_cutoff_and_count_distinct_facades(self):
        rows = [dict(status='ready', building_id=building, stats=pair_statistics(counted(changed), CATEGORIES))
                for building, changed in [('A', 10), ('A', 15), ('B', 50)]]
        rows.append(dict(status='ready', building_id='C', stats=pair_statistics(counted(0, known=0), CATEGORIES)))
        rows.append(dict(status='excluded', building_id='D'))
        result = summarize(rows, CATEGORIES)
        self.assertEqual((result['ready_pairs'], result['facades'], result['unknown_pairs']), (4, 3, 1))
        indexed = {(row['channel'], row['denominator'], row['threshold_percent']): row for row in result['thresholds']}
        self.assertEqual(len(indexed), 3 * 2 * len(THRESHOLDS))
        self.assertEqual(indexed[None, 'known', 0]['pairs'], 3)  # Unknown C is not a 0% sample.
        self.assertEqual(indexed[None, 'known', 10]['pairs'], 3)
        self.assertEqual(indexed[None, 'known', 10]['facades'], 2)
        self.assertEqual(indexed[None, 'known', 15]['pairs'], 2)
        self.assertEqual(indexed[None, 'known', 30]['pairs'], 1)
        self.assertEqual(indexed[None, 'known', 50]['pairs'], 1)
        self.assertEqual(indexed[0, 'known', 0]['pairs'], 3)
        self.assertEqual(indexed[0, 'known', 1]['pairs'], 0)  # Wall is stable in all pairs.
        self.assertEqual(indexed[1, 'known', 50]['pairs'], 3)  # Crack's own ROI is entirely new.
        self.assertEqual(indexed[1, 'crop', 50]['pairs'], 1)

    def test_crop_threshold_excludes_small_coverage_and_absent_class(self):
        rows = [dict(status='ready', building_id='A', stats=pair_statistics(counted(5, known=10, total=100), CATEGORIES)),
                dict(status='ready', building_id='B', stats=pair_statistics(pair([1, 1], [1, 1]), CATEGORIES))]
        thresholds = {(r['channel'], r['denominator'], r['threshold_percent']): r for r in summarize(rows, CATEGORIES)['thresholds']}
        self.assertEqual(thresholds[None, 'known', 50]['pairs'], 1)
        self.assertEqual(thresholds[None, 'crop', 50]['pairs'], 0)
        self.assertEqual(thresholds[None, 'crop', 5]['pairs'], 1)
        self.assertEqual(thresholds[1, 'known', 0]['pairs'], 1)  # Absent crack in B is excluded.
        self.assertEqual(thresholds[1, 'crop', 0]['pairs'], 1)


if __name__ == '__main__':
    unittest.main()
