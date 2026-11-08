"""Causal current-episode history invariants on exact, tiny RGB sequences."""
import importlib
import io
import unittest

import numpy as np


TemporalMemory = importlib.import_module(
    "facade_change.research.2026-10-14_temporal_history").TemporalMemory


def rgb(values):
    return np.repeat(np.array([values], np.uint8)[..., None], 3, axis=-1)


def difference(reference, current, support):
    result = np.abs(reference.astype(float) - current).mean(axis=2)
    result[~support] = np.nan
    return result


class TemporalHistoryTests(unittest.TestCase):
    def setUp(self):
        self.support = np.ones((1, 2), bool)

    def test_repair_and_repeated_damage_start_new_local_episodes(self):
        memory = TemporalMemory(3)
        outputs = [memory.score_then_update(rgb(values), self.support, difference, 50)
                   for values in ([0, 0], [100, 0], [100, 0], [0, 0], [100, 0])]
        self.assertTrue(np.isnan(outputs[0]["scores"]).all())
        np.testing.assert_array_equal(outputs[0]["initialized_support"], self.support)
        for frame in (1, 3, 4):
            np.testing.assert_array_equal(outputs[frame]["accepted_change"], [[True, False]])
            self.assertEqual(outputs[frame]["scores"][0, 0], 100)
            self.assertEqual(outputs[frame]["episode_start"][0, 0], frame)
        self.assertFalse(outputs[2]["accepted_change"].any())
        self.assertEqual(outputs[3]["reference_frame"][0, 0], 1)
        self.assertEqual(outputs[4]["reference_frame"][0, 0], 3)

    def test_unobserved_support_initializes_without_change(self):
        memory = TemporalMemory()
        first = memory.score_then_update(rgb([0, 0]), np.array([[True, False]]), difference, 50)
        second = memory.score_then_update(rgb([0, 255]), self.support, difference, 50)
        np.testing.assert_array_equal(first["support"], [[False, False]])
        np.testing.assert_array_equal(second["initialized_support"], [[False, True]])
        self.assertFalse(second["accepted_change"].any())
        self.assertTrue(np.isnan(second["scores"][0, 1]))
        third = memory.score_then_update(rgb([0, 255]), self.support, difference, 50)
        self.assertEqual(third["scores"][0, 1], 0)

    def test_current_observation_never_scores_itself(self):
        memory = TemporalMemory()
        seen = []

        def scorer(reference, current, support):
            seen.append((reference.copy(), current.copy()))
            return difference(reference, current, support)

        memory.score_then_update(rgb([1, 2]), self.support, scorer, 50)
        self.assertEqual(seen, [])
        result = memory.score_then_update(rgb([101, 102]), self.support, scorer, 50)
        self.assertEqual(len(seen), 1)
        np.testing.assert_array_equal(seen[0][0], rgb([1, 2]))
        np.testing.assert_array_equal(result["scores"], [[100, 100]])
        self.assertTrue(result["accepted_change"].all())

    def test_past_prefix_outputs_independent_of_future(self):
        histories = ([0, 10, 20, 250], [0, 10, 20, 30])
        prefixes = []
        for sequence in histories:
            memory = TemporalMemory()
            outputs = [memory.score_then_update(rgb([value, value]), self.support, difference, 50)
                       for value in sequence]
            prefixes.append(outputs[:3])
        for left, right in zip(*prefixes):
            for key in ("scores", "accepted_change", "reference_frame", "episode_start"):
                np.testing.assert_array_equal(left[key], right[key])

    def test_finite_minimum_and_auxiliary_winner_match(self):
        memory = TemporalMemory()
        memory.score_then_update(rgb([0, 40]), self.support, difference, 255)
        memory.score_then_update(rgb([30, 10]), self.support, difference, 255)

        def scorer(reference, current, support):
            return difference(reference, current, support), {"red": int(reference[0, 0, 0])}

        result = memory.score_then_update(rgb([0, 10]), self.support, scorer, 255)
        np.testing.assert_array_equal(result["scores"], [[0, 0]])
        np.testing.assert_array_equal(result["reference_frame"], [[0, 1]])
        np.testing.assert_array_equal(result["winning_comparison"], [[0, 1]])
        np.testing.assert_array_equal(result["reference_count"], [[2, 2]])
        self.assertEqual(result["comparisons"][1], {"reference_frame": 1, "metadata": {"red": 30}})

    def test_nonfinite_comparators_do_not_corrupt_memory(self):
        memory = TemporalMemory()
        memory.score_then_update(rgb([0, 0]), self.support, difference, 50)
        result = memory.score_then_update(
            rgb([100, 100]), self.support, lambda *args: np.array([[np.inf, np.nan]]), 50)
        self.assertFalse(result["support"].any())
        self.assertFalse(result["accepted_change"].any())
        result = memory.score_then_update(rgb([100, 100]), self.support, difference, 50)
        np.testing.assert_array_equal(result["scores"], [[100, 100]])
        np.testing.assert_array_equal(result["reference_frame"], [[0, 0]])

    def test_capacity_is_local_and_rgb_exemplars_are_actual_observations(self):
        memory = TemporalMemory(2)
        original = rgb([1, 2])
        memory.score_then_update(original, self.support, difference, 255)
        for value in (10, 20, 30):
            memory.score_then_update(rgb([value, 99]), np.array([[True, False]]), difference, 255)
        state = memory.state_dict()
        self.assertTrue((state["valid"].sum(axis=0) <= 2).all())
        np.testing.assert_array_equal(state["rgb"][0], original)
        np.testing.assert_array_equal(state["valid"][0], [[False, True]])
        np.testing.assert_array_equal(state["frame_ids"], [0, 2, 3])
        result = memory.score_then_update(rgb([30, 2]), self.support, difference, 255)
        np.testing.assert_array_equal(result["scores"], [[0, 0]])
        np.testing.assert_array_equal(result["reference_frame"], [[3, 0]])

    def test_pickle_free_restore_matches_uninterrupted_sequence(self):
        memory = TemporalMemory(2)
        memory.score_then_update(rgb([0, 0]), self.support, difference, 50)
        memory.score_then_update(rgb([100, 0]), self.support, difference, 50)
        stream = io.BytesIO()
        np.savez_compressed(stream, **memory.state_dict())
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as archive:
            restored = TemporalMemory.from_state_dict(dict(archive))
        for values in ([100, 10], [0, 0], [200, 0]):
            a = memory.score_then_update(rgb(values), self.support, difference, 50)
            b = restored.score_then_update(rgb(values), self.support, difference, 50)
            for key in a:
                if key != "comparisons":
                    np.testing.assert_array_equal(a[key], b[key])
        for key, value in memory.state_dict().items():
            np.testing.assert_array_equal(value, restored.state_dict()[key])

    def test_empty_state_and_empty_observations_round_trip(self):
        memory = TemporalMemory.from_state_dict(TemporalMemory().state_dict())
        result = memory.score_then_update(rgb([0, 0]), ~self.support, difference, 0)
        self.assertFalse(result["support"].any())
        self.assertEqual(result["frame_index"], 0)
        restored = TemporalMemory.from_state_dict(memory.state_dict())
        result = restored.score_then_update(rgb([50, 50]), self.support, difference, 0)
        self.assertEqual(result["frame_index"], 1)
        self.assertTrue(result["initialized_support"].all())
        self.assertFalse(result["accepted_change"].any())

    def test_threshold_is_strict_fixed_finite_and_nonnegative(self):
        for invalid in (np.inf, np.nan, -1, True, "1"):
            with self.assertRaises(ValueError):
                TemporalMemory().score_then_update(rgb([0, 0]), self.support, difference, invalid)
        memory = TemporalMemory()
        memory.score_then_update(rgb([0, 0]), self.support, difference, 50)
        result = memory.score_then_update(rgb([50, 51]), self.support, difference, 50)
        np.testing.assert_array_equal(result["accepted_change"], [[False, True]])
        with self.assertRaisesRegex(ValueError, "fixed"):
            memory.score_then_update(rgb([0, 0]), self.support, difference, 51)

    def test_callback_failure_leaves_memory_unchanged(self):
        memory = TemporalMemory()
        memory.score_then_update(rgb([0, 0]), self.support, difference, 50)
        before = memory.state_dict()
        with self.assertRaisesRegex(ValueError, "same-grid"):
            memory.score_then_update(rgb([100, 0]), self.support, lambda *args: np.zeros((2, 2)), 50)
        for key, value in before.items():
            np.testing.assert_array_equal(value, memory.state_dict()[key])

    def test_restore_rejects_stale_episode_and_future_exemplars(self):
        memory = TemporalMemory()
        memory.score_then_update(rgb([0, 0]), self.support, difference, 50)
        state = memory.state_dict()
        state["episode_start"][0, 0] = 1
        with self.assertRaisesRegex(ValueError, "predates"):
            TemporalMemory.from_state_dict(state)
        state = memory.state_dict()
        state["frame_ids"][0] = 1
        with self.assertRaisesRegex(ValueError, "past"):
            TemporalMemory.from_state_dict(state)


if __name__ == "__main__":
    unittest.main()
