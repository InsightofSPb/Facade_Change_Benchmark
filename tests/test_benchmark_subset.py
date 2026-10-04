import copy
import json
import unittest
from collections import Counter

from facade_change.benchmark_subset import QUICK_FAMILIES, QUICK_STATES, select_quick_subset


def source_rows(val=6, test=11, train=4, crops=2, families=None):
    scenarios = [("clean", "identity"), ("jpeg_40", "jpeg")]
    scenarios += [("shadow_" + template + "_" + strength, "shadow")
                  for template in ("band", "diagonal") for strength in ("15", "35", "55")]
    scenarios += [(family + "_" + variant, family)
                  for family in ("exposure", "contrast", "white_balance", "blur") for variant in ("low", "high")]
    scenarios += [("occlusion", "occlusion")]
    if families is not None:
        scenarios = [row for row in scenarios if row[1] in families]
    bases, cases = [], []
    for split, count in (("train", train), ("val", val), ("test", test)):
        for building in range(count):
            building_id = split + "_building_" + str(building)
            for crop in range(crops):
                base_id = building_id + "_crop_" + str(crop)
                base = {"base_id": base_id, "building_id": building_id, "split": split, "view_id": base_id + "_view"}
                bases.append(base)
                for state in (*QUICK_STATES, "self_paste"):
                    for scenario, family in scenarios:
                        spec = {"id": scenario, "kind": family}
                        if family == "shadow":
                            spec.update(template=scenario.split("_")[1], strength=int(scenario.split("_")[2]) / 100)
                        cases.append({**base, "case_id": base_id + "_" + state + "_" + scenario,
                                      "state": state, "scenario_id": scenario, "nuisance_kind": family,
                                      "scenario": spec,
                                      "visible_edit_pixel_count": 100 if state in {"crack", "paint_patch"} else 0})
    return bases, cases


class QuickSubsetTests(unittest.TestCase):
    def test_ten_distinct_facades_validation_first_and_six_shared_cases(self):
        bases, cases = source_rows()
        selected, rows, metadata = select_quick_subset(bases, cases)
        self.assertEqual(len(selected), 10)
        self.assertEqual(len({base["building_id"] for base in selected}), 10)
        self.assertEqual([base["split"] for base in selected], ["val"] * 3 + ["test"] * 7)
        self.assertEqual(len(rows), 60)
        self.assertEqual(metadata["selected_buildings_by_split"], {"val": 3, "test": 7})
        for base in selected:
            current = [row for row in rows if row["base_id"] == base["base_id"]]
            self.assertEqual(Counter(row["state"] for row in current), dict.fromkeys(QUICK_STATES, 2))
            by_state = [{row["scenario_id"] for row in current if row["state"] == state} for state in QUICK_STATES]
            self.assertTrue(all(value == by_state[0] for value in by_state))
            self.assertEqual(len({row["nuisance_kind"] for row in current}), 2)
        self.assertEqual({row["nuisance_kind"] for row in rows}, set(QUICK_FAMILIES))
        self.assertFalse(any(row["state"] == "self_paste" or row["nuisance_kind"] in {"identity", "jpeg"} for row in rows))
        self.assertEqual(json.loads(json.dumps(metadata)), metadata)

    def test_intensities_are_balanced_without_adding_pairs(self):
        for seed in (0, 1, 42, 43, 2026):
            with self.subTest(seed=seed):
                _, rows, metadata = select_quick_subset(*source_rows(), seed=seed)
                conditions = [row for row in rows if row["state"] == "unchanged"]
                shadow = [row for row in conditions if row["nuisance_kind"] == "shadow"]
                self.assertGreaterEqual(len(shadow), 3)
                self.assertEqual({row["scenario"]["strength"] for row in shadow}, {.15, .35, .55})
                for family in ("exposure", "contrast", "white_balance", "blur"):
                    current = [row for row in conditions if row["nuisance_kind"] == family]
                    self.assertGreaterEqual(len(current), 2)
                    self.assertEqual({row["scenario_id"].rsplit("_", 1)[1] for row in current}, {"low", "high"})
                self.assertEqual(len(rows), 60)
                self.assertIn("shadow strengths", metadata["variant_allocation_strategy"])

    def test_shadow_templates_cycle_and_missing_strength_uses_scenario_ids(self):
        _, rows, _ = select_quick_subset(*source_rows(val=8, test=20), max_bases=18)
        shadow = [row for row in rows if row["state"] == "unchanged" and row["nuisance_kind"] == "shadow"]
        for strength in (.15, .35, .55):
            self.assertEqual({row["scenario"]["template"] for row in shadow if row["scenario"]["strength"] == strength},
                             {"band", "diagonal"})
        bases, cases = source_rows(val=1, test=1, families={"shadow", "occlusion"})
        for row in cases:
            row["scenario"].pop("strength", None)
        _, rows, metadata = select_quick_subset(bases, cases)
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(group[0] == "scenario_id" for base in metadata["bases"] for group in base["selected_variant_groups"]))

    def test_seeded_selection_is_input_order_independent_and_preserves_inputs(self):
        bases, cases = source_rows()
        before = copy.deepcopy((bases, cases))
        first = select_quick_subset(bases, cases, seed=42)
        reordered = select_quick_subset(list(reversed(bases)), list(reversed(cases)), seed=42)
        self.assertEqual(first, reordered)
        self.assertEqual((bases, cases), before)
        self.assertNotEqual([row["case_id"] for row in first[1]],
                            [row["case_id"] for row in select_quick_subset(bases, cases, seed=43)[1]])

    def test_selection_ignores_gt_and_keeps_all_shadow_options_eligible(self):
        bases, cases = source_rows()
        first = select_quick_subset(bases, cases)
        for row in cases:
            row["visible_edit_pixel_count"] = 0
            row["full_edit_pixel_count"] = 999999
            row["labels_reference"] = "arbitrary-unreadable-path"
        second = select_quick_subset(bases, cases)
        self.assertEqual(first[2], second[2])
        self.assertEqual([row["case_id"] for row in first[1]], [row["case_id"] for row in second[1]])
        for base in second[2]["bases"]:
            self.assertEqual(len(base["eligible_scenario_ids_by_family"]["shadow"]), 6)

    def test_shortage_is_transparent_and_never_reuses_same_facade(self):
        bases, cases = source_rows(val=1, test=1, families={"shadow", "occlusion"})
        selected, rows, metadata = select_quick_subset(bases, cases, max_bases=10)
        self.assertEqual((len(selected), len(rows)), (2, 12))
        self.assertEqual(metadata["requested_max_bases"], 10)
        self.assertEqual(metadata["available_buildings_by_split"], {"val": 1, "test": 1})
        self.assertEqual(metadata["selected_buildings_by_split"], {"val": 1, "test": 1})

    def test_available_capacity_reallocates_without_train(self):
        for val, test, expected in ((1, 15, {"val": 1, "test": 9}),
                                    (15, 1, {"val": 9, "test": 1})):
            with self.subTest(val=val, test=test):
                selected, rows, metadata = select_quick_subset(*source_rows(val=val, test=test))
                self.assertEqual(metadata["selected_buildings_by_split"], expected)
                self.assertEqual(len(selected), 10)
                self.assertFalse(any(row["split"] == "train" for row in rows))

    def test_budget_and_required_partitions_and_families(self):
        for budget in (0, 1, True, 2.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                select_quick_subset(*source_rows(), max_bases=budget)
        for val, test in ((0, 1), (1, 0)):
            with self.subTest(val=val, test=test), self.assertRaises(ValueError):
                select_quick_subset(*source_rows(val=val, test=test))
        with self.assertRaisesRegex(ValueError, "two distinct"):
            select_quick_subset(*source_rows(families={"shadow"}))
        selected, rows, metadata = select_quick_subset(*source_rows(), max_bases=2)
        self.assertEqual(metadata["selected_buildings_by_split"], {"val": 1, "test": 1})
        self.assertEqual(len(rows), 12)

    def test_inconsistent_or_missing_state_variant_is_rejected(self):
        for corruption in ("missing", "duplicate", "family", "scenario", "identity"):
            bases, cases = source_rows(val=1, test=1, train=0, crops=1)
            row = next(row for row in cases if row["split"] == "val" and row["state"] == "crack"
                       and row["nuisance_kind"] == "shadow")
            if corruption == "missing":
                cases.remove(row)
            elif corruption == "duplicate":
                cases.append({**row, "case_id": row["case_id"] + "_duplicate"})
            elif corruption == "family":
                row["nuisance_kind"] = "exposure"
            elif corruption == "scenario":
                row["scenario"]["strength"] = .999
            else:
                row["building_id"] = "wrong_building"
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                select_quick_subset(bases, cases)

    def test_cross_split_building_leakage_and_unknown_base_are_rejected(self):
        bases, cases = source_rows(val=1, test=1, train=0, crops=1)
        bases[1]["building_id"] = bases[0]["building_id"]
        with self.assertRaisesRegex(ValueError, "across splits"):
            select_quick_subset(bases, cases)
        bases, cases = source_rows(val=1, test=1, train=0, crops=1)
        cases[0]["base_id"] = "missing_base"
        with self.assertRaisesRegex(ValueError, "known bases"):
            select_quick_subset(bases, cases)


if __name__ == "__main__":
    unittest.main()
