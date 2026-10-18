import unittest

from scripts.real_eval.semantic_timing import estimate


class SemanticTimingTests(unittest.TestCase):
    def group(self, **extra):
        return dict(stage='segment_full', segmenter='sam3', variant='ours', units=26, **extra)

    def observation(self, elapsed=2., init=10., **extra):
        return dict(stage='segment_full', segmenter='sam3', variant='ours',
                    elapsed_seconds=elapsed, init_seconds=init,
                    source='measured', status='completed', **extra)

    def test_init_counted_once_and_completed_cache_subtracted(self):
        result = estimate([self.group(cached_units=1)], [self.observation()])
        self.assertEqual(result['total_lower_seconds'], .8 * (25 * 2 + 10))
        self.assertEqual(result['total_upper_seconds'], 1.5 * (25 * 2 + 10))
        self.assertEqual(result['total_central_seconds'], 60)
        self.assertEqual(result['groups'][0]['model_initializations'], 1)

    def test_missing_route_never_reported_as_zero(self):
        unseen = dict(stage='change_roi', method='msdzip_abs', units=10)
        result = estimate([self.group(), unseen], [self.observation()])
        self.assertIsNone(result['total_lower_seconds'])
        self.assertIsNone(result['total_upper_seconds'])
        self.assertGreater(result['known_upper_seconds'], 0)
        self.assertEqual(result['unresolved_groups'], 1)
        self.assertIsNone(result['groups'][1]['upper_seconds'])

    def test_imported_results_are_not_measurements(self):
        observed = self.observation()
        observed.update(source='imported', elapsed_seconds=None, init_seconds=None)
        result = estimate([self.group()], [observed])
        self.assertIsNone(result['total_upper_seconds'])
        self.assertEqual(result['groups'][0]['timing_source'], 'unmeasured')

    def test_verified_cache_timing_fallback_and_deduplication(self):
        observation = self.observation(cache_key='same')
        observation['source'] = 'cache'
        result = estimate([self.group()], [observation, observation])
        self.assertEqual(result['groups'][0]['timing_source'], 'cache_fallback')
        self.assertEqual(result['groups'][0]['inference']['samples'], 1)
        measured = self.observation(elapsed=3., cache_key='same')
        result = estimate([self.group()], [observation, measured])
        self.assertEqual(result['groups'][0]['timing_source'], 'measured')
        self.assertEqual(result['groups'][0]['inference']['median_seconds'], 3.)

    def test_all_cached_legitimate_zero_compute_without_probe(self):
        result = estimate([self.group(cached_units=26)], [])
        self.assertEqual(result['total_upper_seconds'], 0.)
        self.assertEqual(result['unresolved_groups'], 0)
        self.assertEqual(result['groups'][0]['timing_probe_status'], 'not_required_all_cached')

    def test_no_roi_smoke_does_not_claim_future_roi_free(self):
        workload = dict(stage='segment_roi', segmenter='sam3', variant='ours', units=0, roi_upper_bound=84)
        coverage = [{**workload, 'status': 'no_roi'}]
        result = estimate([workload], [], coverage)
        self.assertIsNone(result['total_upper_seconds'])
        self.assertEqual(result['groups'][0]['timing_probe_status'], 'no_roi_observed')
        self.assertEqual(result['unresolved_maximum_units'], 84)

    def test_roi_central_and_upper_workloads_are_distinct(self):
        workload = dict(stage='segment_roi', segmenter='sam3', variant='ours', units=12,
                        roi_upper_bound=84, estimated_units=30, cached_units=2)
        observation = self.observation()
        observation['stage'] = 'segment_roi'
        result = estimate([workload], [observation])
        self.assertEqual(result['total_central_seconds'], 28 * 2 + 10)
        self.assertEqual(result['total_upper_seconds'], 1.5 * (82 * 2 + 10))

    def test_estimated_roi_count_below_nominal_keeps_range_coherent(self):
        workload = dict(stage='segment_roi', segmenter='sam3', variant='ours', units=84,
                        estimated_units=12)
        observation = self.observation()
        observation['stage'] = 'segment_roi'
        result = estimate([workload], [observation])
        self.assertLessEqual(result['total_lower_seconds'], result['total_central_seconds'])
        self.assertLessEqual(result['total_central_seconds'], result['total_upper_seconds'])

    def test_different_vocabulary_not_silent_timing_fallback(self):
        observation = self.observation()
        observation['variant'] = 'extended'
        result = estimate([self.group()], [observation])
        self.assertIsNone(result['total_upper_seconds'])

    def test_unknown_init_unresolved_unless_not_needed(self):
        observation = self.observation()
        del observation['init_seconds']
        result = estimate([self.group()], [observation])
        self.assertIsNone(result['total_upper_seconds'])
        self.assertIn('model_initialization_not_measured', result['groups'][0]['unresolved'])
        result = estimate([self.group(model_initializations=0)], [observation])
        self.assertEqual(result['total_central_seconds'], 52)

    def test_cd_init_fallback_across_roi_selectors_keeps_inference_separate(self):
        workload = dict(stage='change_roi', method='dinov2', segmenter='sam3', variant='expanded_flat',
                        units=10, model_initializations=1)
        full = dict(stage='change_full', method='dinov2', segmenter=None, variant=None,
                    elapsed_seconds=100., init_seconds=7., status='completed', source='measured')
        roi = dict(stage='change_roi', method='dinov2', segmenter='sam3', variant='expanded_flat',
                   elapsed_seconds=2., status='completed', source='measured', timing_only=True)
        result = estimate([workload], [full, roi])
        self.assertEqual(result['total_central_seconds'], 27.)
        group = result['groups'][0]
        self.assertEqual(group['initialization_source'], 'same_change_method_measured_fallback')
        self.assertEqual(group['inference']['median_seconds'], 2.)
        self.assertEqual(group['timing_probe_status'], 'timing_only_probe')
        self.assertEqual(group['task_samples'], 0)
        self.assertEqual(group['timing_only_samples'], 1)
        full['method'] = 'anychange'
        self.assertIsNone(estimate([workload], [full, roi])['total_upper_seconds'])

    def test_invalid_counts_and_failed_probe_rejected_or_unknown(self):
        with self.assertRaises(ValueError):
            estimate([self.group(cached_units=27)], [])
        with self.assertRaises(ValueError):
            estimate([self.group(estimated_units=float('nan'))], [])
        observation = self.observation()
        observation['status'] = 'failed'
        result = estimate([self.group()], [observation])
        self.assertIsNone(result['total_upper_seconds'])
        self.assertEqual(result['groups'][0]['timing_probe_status'], 'failed')


if __name__ == '__main__':
    unittest.main()
