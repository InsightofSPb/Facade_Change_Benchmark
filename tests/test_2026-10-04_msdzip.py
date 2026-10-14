import contextlib
import copy
import importlib
import io
import json
import os
import sys
from collections import Counter
from pathlib import Path
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from facade_change.io import sha256


msdzip = importlib.import_module("facade_change.2026-10-04_msdzip_h0")


class ResidualContextTests(unittest.TestCase):
    def test_official_predictor_is_exact_upstream_source(self):
        self.assertEqual(sha256(msdzip.SOURCE_PATH), msdzip.SOURCE_SHA256)
        self.assertIn("class MixedModel", msdzip.SOURCE_PATH.read_text())

    def test_abs_mod256_signed_subtraction_and_no_current_target_in_context(self):
        reference = np.full((1, 3, 3), 100, np.uint8)
        source = reference.copy()
        source[0, 0] = [110, 90, 100]
        self.assertEqual(msdzip.residual_bytes(reference, source, "abs")[0, 0].tolist(), [10, 10, 0])
        residual = msdzip.residual_bytes(reference, source, "mod256")
        self.assertEqual(residual[0, 0].tolist(), [10, 246, 0])
        support = np.ones((1, 3), bool)
        contexts, labels = msdzip._windows(residual, support, [0, 1, 2], 2)
        self.assertEqual(contexts.tolist(), [[0, 0], [0, 10], [10, 246]])
        mutated = residual.copy().reshape(-1)
        mutated[2] = 77
        after, new_labels = msdzip._windows(mutated.reshape(residual.shape), support, [2], 2)
        np.testing.assert_array_equal(contexts[2], after[0])
        self.assertNotEqual(labels[2], new_labels[0])

    def test_context_does_not_cross_unsupported_pixels_and_lanes_are_native(self):
        residual = np.arange(12, dtype=np.uint8).reshape(1, 4, 3)
        support = np.array([[True, False, True, True]])
        positions = np.array([0, 1, 2, 6, 7, 8, 9])
        contexts, targets = msdzip._windows(residual, support, positions, 4)
        self.assertEqual(contexts[3].tolist(), [0, 0, 0, 0])
        self.assertEqual(contexts[4].tolist(), [0, 0, 0, 6])
        visited = []
        for _, _, indices in msdzip._batches(contexts, targets, positions, 4):
            for lane, index in enumerate(indices):
                if index >= 0:
                    self.assertEqual(positions[index] % 4, lane)
                    visited.append(int(index))
        self.assertEqual(sorted(visited), list(range(len(positions))))


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("cv2"), "PyTorch/OpenCV required for original predictor smoke")
class OriginalMSDZipTests(unittest.TestCase):
    def setUp(self):
        import torch
        from test_hypothesis_dataset import HypothesisDatasetTests
        self.torch = torch
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.fixture = HypothesisDatasetTests("test_h0_uses_reference_and_sham_is_identical_with_matching_nuisance")
        self.fixture.setUp()
        self.root = self.fixture.root
        self.dataset, _, self.index = self.fixture._export("hypotheses")

    def tearDown(self):
        self.fixture.tearDown()
        self.torch.set_num_threads(self.old_threads)

    def _train(self, name="fit"):
        with patch.object(msdzip, "load_rgb", wraps=msdzip.load_rgb) as reader:
            result = msdzip.train_msdzip_h0(
                self.dataset, self.root / name, epochs=1, max_train_bytes=32, max_val_bytes=16,
                model_batch_size=4, timesteps=2, hidden_dim=4, ffn_dim=8, vocab_dim=2,
                max_bases_per_split=1, window_groups=2,
            )
        return result, reader.call_args_list

    def test_training_reads_only_unchanged_train_val_and_same_targets_for_representations(self):
        summary, reads = self._train()
        permitted = set()
        h1_or_test = set()
        for case in self.index["cases"]:
            if case["split"] in {"train", "val"} and case["state"] == "unchanged":
                permitted.update({self.dataset / case["reference_rgb"], self.dataset / case["source_rgb"]})
            elif case["state"] in {"crack", "paint_patch"} or case["split"] == "test":
                h1_or_test.add(self.dataset / case["source_rgb"])
        actual = {call.args[0] for call in reads}
        self.assertTrue(actual <= permitted)
        self.assertFalse(actual & h1_or_test)
        self.assertEqual(summary["sampled_bytes"], {"train": 32, "val": 16})
        self.assertEqual(summary["buildings"], {"train": ["b_train"], "val": ["b_val"]})
        abs_metadata = summary["results"]["abs"]["metadata"]
        mod_metadata = summary["results"]["mod256"]["metadata"]
        self.assertEqual(abs_metadata["sampling_sha256"], mod_metadata["sampling_sha256"])
        self.assertEqual(abs_metadata["training_case_ids"], mod_metadata["training_case_ids"])
        self.assertTrue(np.isfinite(summary["results"]["abs"]["best_val_bpb"]))

    def test_frozen_native_scores_ignore_earlier_cases_and_unsupported_rgb(self):
        summary, _ = self._train()
        checkpoint = self.root / "fit" / summary["results"]["abs"]["checkpoint_path"]
        scorer = msdzip.MSDZipScorer(checkpoint, dataset_fingerprint=summary["input_sha256"], representation="abs")
        before = {key: value.clone() for key, value in scorer.model.state_dict().items()}
        reference = np.arange(45, dtype=np.uint8).reshape(3, 5, 3)
        source = reference + np.uint8(1)
        support = np.ones((3, 5), bool)
        support[:, 2] = False
        first = scorer(reference, source, support)
        raw = scorer.raw_scores.copy()
        self.assertEqual(first.shape, support.shape)
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.isnan(first[~support]).all())
        self.assertTrue(np.isfinite(raw[support]).all())
        np.testing.assert_allclose(first[support], 1 - np.exp(-raw[support] / 8), atol=1e-7)
        scorer(reference, 255 - source, support)
        np.testing.assert_array_equal(scorer(reference, source, support), first)
        changed = source.copy()
        changed[~support] = 255
        np.testing.assert_array_equal(scorer(reference, changed, support), first)
        self.assertEqual(scorer.model.last, [])
        for key, tensor in scorer.model.state_dict().items():
            self.assertTrue(self.torch.equal(before[key], tensor))
        self.assertFalse(any(parameter.requires_grad for parameter in scorer.model.parameters()))

    def test_inference_progress_is_throttled_flushed_and_preserves_exact_scores(self):
        summary, _ = self._train()
        checkpoint = self.root / "fit" / summary["results"]["abs"]["checkpoint_path"]
        scorer = msdzip.MSDZipScorer(checkpoint, representation="abs")
        reference = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        source = reference + np.uint8(3)
        support = np.ones((8, 8), bool)
        support[:, 2] = False
        with contextlib.redirect_stdout(io.StringIO()):
            expected = scorer(reference, source, support)
            expected_raw = scorer.raw_scores.copy()
        clock = [0.]

        def tick():
            clock[0] += 2.
            return clock[0]

        with patch.object(msdzip.time, "monotonic", side_effect=tick), \
                patch.object(msdzip, "print", create=True) as output:
            actual = scorer(reference, source, support)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(scorer.raw_scores, expected_raw)
        messages = [call.args[0] for call in output.call_args_list]
        self.assertIn(f"supported_bytes={int(support.sum()) * 3}", messages[0])
        self.assertIn("start:", messages[0])
        self.assertIn("complete:", messages[-1])
        self.assertIn(f"bytes={int(support.sum()) * 3}/{int(support.sum()) * 3}", messages[-1])
        updates = [float(message.rsplit("elapsed_s=", 1)[1]) for message in messages if "progress:" in message]
        self.assertGreater(len(updates), 1)
        self.assertGreaterEqual(updates[0], 5.)
        self.assertTrue(all(right - left >= 5. for left, right in zip(updates, updates[1:])))
        self.assertTrue(all(call.kwargs.get("flush") is True for call in output.call_args_list))

    def test_checkpoint_representation_and_dataset_mismatch_fail(self):
        summary, _ = self._train()
        checkpoint = self.root / "fit" / summary["results"]["abs"]["checkpoint_path"]
        with self.assertRaisesRegex(ValueError, "representation differs"):
            msdzip.MSDZipScorer(checkpoint, representation="mod256")
        fingerprint = {**summary["input_sha256"], "split": "changed"}
        with self.assertRaisesRegex(ValueError, "dataset/split fingerprint"):
            msdzip.MSDZipScorer(checkpoint, dataset_fingerprint=fingerprint)
        data = self.torch.load(checkpoint, weights_only=True)
        data["metadata"]["buildings"]["train"] = ["b_test"]
        bad = self.root / "bad.pt"
        self.torch.save(data, bad)
        with self.assertRaisesRegex(ValueError, "training buildings"):
            msdzip.MSDZipScorer(bad)

    def _resume_train(self, name, **kwargs):
        options = dict(representations=("abs",), epochs=2, max_train_bytes=32, max_val_bytes=16,
                       model_batch_size=4, timesteps=2, hidden_dim=4, ffn_dim=8, vocab_dim=2,
                       max_bases_per_split=1, window_groups=2, checkpoint_every_steps=1)
        options.update(kwargs)
        return msdzip.train_msdzip_h0(self.dataset, self.root / name, **options)

    def test_mid_epoch_resume_matches_uninterrupted_parameters_and_losses(self):
        full = self._resume_train("full")
        paused = self._resume_train("resumed", max_steps_per_call=1)
        self.assertEqual(paused["status"], "interrupted")
        saved = self.torch.load(self.root / "resumed/2026-10-14_msdzip_resume_abs.pt", weights_only=True)
        self.assertEqual(saved["training"]["global_step"], 1)
        self.assertEqual(saved["training"]["progress"]["step"], 1)
        self.assertIn("optimizer", saved)
        self.assertEqual(set(saved["rng"]), {"torch", "numpy", "python", "cuda"})
        resumed = self._resume_train("resumed", resume=True)
        full_checkpoint = self.torch.load(self.root / "full" / full["results"]["abs"]["checkpoint_path"], weights_only=True)
        resumed_checkpoint = self.torch.load(self.root / "resumed" / resumed["results"]["abs"]["checkpoint_path"], weights_only=True)
        self.assertEqual(full_checkpoint["metadata"], resumed_checkpoint["metadata"])
        for key, tensor in full_checkpoint["state_dict"].items():
            self.assertTrue(self.torch.equal(tensor, resumed_checkpoint["state_dict"][key]), key)
        for left, right in zip(full["results"]["abs"]["history"], resumed["results"]["abs"]["history"]):
            self.assertEqual(left["train_bpb"], right["train_bpb"])
            self.assertEqual(left["val_bpb"], right["val_bpb"])
        final_states = [self.torch.load(self.root / name / "2026-10-14_msdzip_resume_abs.pt", weights_only=True)
                        for name in ("full", "resumed")]
        for key, tensor in final_states[0]["state_dict"].items():
            self.assertTrue(self.torch.equal(tensor, final_states[1]["state_dict"][key]), key)
        for parameter, values in final_states[0]["optimizer"]["state"].items():
            for key, tensor in values.items():
                self.assertTrue(self.torch.equal(tensor, final_states[1]["optimizer"]["state"][parameter][key]))
        events = [json.loads(line) for line in (self.root / "resumed/events.jsonl").read_text().splitlines()]
        self.assertIn("retry", [event["event_type"] for event in events])
        self.assertIn("checkpoint_resume", [event["event_type"] for event in events])
        self.assertEqual([event["sequence"] for event in events], list(range(1, len(events) + 1)))

    def test_incomplete_event_tail_is_preserved_across_repeated_resumes(self):
        self._resume_train("event_recovery", max_steps_per_call=1)
        out = self.root / "event_recovery"
        path = out / "events.jsonl"
        for fragment in (b'{"event_type":"interrup', b'{"sequence":'):
            original = path.read_bytes()
            with path.open("ab") as stream:
                stream.write(fragment)
            result = self._resume_train("event_recovery", resume=True, max_steps_per_call=1)
            self.assertEqual(result["status"], "interrupted")
            self.assertTrue(path.read_bytes().startswith(original))
            self.assertIn(fragment, [item.read_bytes() for item in out.glob("2026-10-14_msdzip_event_tail_*.bin")])
        completed = self._resume_train("event_recovery", resume=True)
        self.assertEqual(completed["status"], "completed_exploratory")
        with patch.object(msdzip, "_new_model", side_effect=AssertionError("completed fit repeated")):
            self.assertEqual(self._resume_train("event_recovery", resume=True), completed)
        events = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(sum(event["event_type"] == "event_tail_recovered" for event in events), 2)
        self.assertEqual([event["sequence"] for event in events], list(range(1, len(events) + 1)))
        corrupted = path.read_bytes() + b"not-json\n"
        path.write_bytes(corrupted)
        with self.assertRaisesRegex(ValueError, "event log corruption"):
            self._resume_train("event_recovery", resume=True)
        self.assertEqual(path.read_bytes(), corrupted)

    def test_missing_event_newline_and_unrecorded_valid_retry_are_recoverable(self):
        self._resume_train("event_boundary", max_steps_per_call=1)
        out = self.root / "event_boundary"
        path = out / "events.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        appended = {**events[-1], "sequence": len(events) + 1, "event_type": "retry"}
        # Simulate a complete append that precedes run.json's next update and
        # whose final newline has not reached disk.
        with path.open("ab") as stream:
            stream.write(json.dumps(appended).encode())
        result = self._resume_train("event_boundary", resume=True)
        self.assertEqual(result["status"], "completed_exploratory")
        events = [json.loads(line) for line in path.read_text().splitlines()]
        restored = [event for event in events if event["event_type"] == "event_tail_recovered"]
        self.assertEqual(len(restored), 1)
        self.assertTrue(restored[0]["newline_restored"])
        self.assertEqual(self._resume_train("event_boundary", resume=True), result)

    def test_partial_run_reuses_completed_representation(self):
        self._resume_train("abs_steps")
        completed = self.torch.load(self.root / "abs_steps/2026-10-14_msdzip_resume_abs.pt", weights_only=True)
        paused = self._resume_train("two_representations", representations=("abs", "mod256"),
                                    max_steps_per_call=completed["training"]["global_step"] + 1)
        self.assertEqual(paused["status"], "interrupted")
        self.assertEqual(set(paused["results"]), {"abs"})
        original = msdzip._new_model
        with patch.object(msdzip, "_new_model", wraps=original) as models:
            resumed = self._resume_train("two_representations", representations=("abs", "mod256"), resume=True)
        self.assertEqual(models.call_count, 1)
        self.assertEqual(set(resumed["results"]), {"abs", "mod256"})
        self.assertEqual(resumed["results"]["abs"], paused["results"]["abs"])

    def test_resume_refuses_changed_config_input_or_completed_artifact(self):
        self._resume_train("resume_checks", max_steps_per_call=1)
        out = self.root / "resume_checks"
        before = sha256(out / "run.json")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self._resume_train("resume_checks", resume=True, lr=.01)
        self.assertEqual(sha256(out / "run.json"), before)
        with self.assertRaises(FileExistsError):
            self._resume_train("resume_checks")
        with patch.object(msdzip, "SOURCE_SHA256", "changed"):
            with self.assertRaisesRegex(ValueError, "source hash"):
                self._resume_train("resume_checks", resume=True)
        self.assertEqual(sha256(out / "run.json"), before)
        record = json.loads((out / "run.json").read_text())
        relative = next(name for name in record["config"]["selected_artifact_sha256"] if name.endswith(".png"))
        image = self.dataset / relative
        original = image.read_bytes()
        image.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "changed|disagrees"):
            self._resume_train("resume_checks", resume=True)
        self.assertEqual(sha256(out / "run.json"), before)
        image.write_bytes(original)
        resume_checkpoint = out / "2026-10-14_msdzip_resume_abs.pt"
        original_checkpoint = resume_checkpoint.read_bytes()
        resume_checkpoint.write_bytes(original_checkpoint + b"changed")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self._resume_train("resume_checks", resume=True)
        self.assertEqual(sha256(out / "run.json"), before)
        resume_checkpoint.write_bytes(original_checkpoint)
        result = self._resume_train("resume_checks", resume=True)
        with patch.object(msdzip, "_new_model", side_effect=AssertionError("retrained completed fit")):
            reused = self._resume_train("resume_checks", resume=True)
        self.assertEqual(result, reused)
        checkpoint = out / result["results"]["abs"]["checkpoint_path"]
        checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self._resume_train("resume_checks", resume=True)

    def test_keyboard_interrupt_replays_only_from_last_committed_step(self):
        original = msdzip._predict_groups
        calls = 0

        def interrupt(model, contexts):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt("simulated external stop")
            return original(model, contexts)

        with patch.object(msdzip, "_predict_groups", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self._resume_train("signal")
        saved = self.torch.load(self.root / "signal/2026-10-14_msdzip_resume_abs.pt", weights_only=True)
        self.assertEqual(saved["training"]["global_step"], 1)
        resumed = self._resume_train("signal", resume=True)
        full = self._resume_train("full_signal")
        for key in ("train_bpb", "val_bpb"):
            self.assertEqual([row[key] for row in resumed["results"]["abs"]["history"]],
                             [row[key] for row in full["results"]["abs"]["history"]])

    def test_history_clean_pairs_train_val_and_never_test_or_event_pixels(self):
        from test_hypothesis_dataset import HistoryDatasetTests
        from facade_change.io import finish_record, read_json, write_json
        fixture = HistoryDatasetTests("test_clean_controls_persistence_contexts_and_native_support")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        dataset, _, manifest = fixture.prepare()
        # Include a TEST sequence with deliberately missing RGB. Only its
        # metadata may be audited by training; a single open must fail this test.
        test_base = copy.deepcopy(manifest["bases"][0])
        test_base.update(base_id="test-base", building_id="building-5", view_id="test-view",
                         image_sha256="test-image", split="test")
        manifest["bases"].append(test_base)
        test_sequence = copy.deepcopy(manifest["sequences"][0])
        test_sequence.update(base_id="test-base", building_id="building-5", split="test", sequence_id="test-sequence")
        for frame in test_sequence["frames"]:
            frame["paths"]["rgb"] = "missing-test-rgb.png"
        manifest["sequences"].append(test_sequence)
        write_json(dataset / "manifest.json", manifest)
        summary = read_json(dataset / "summary.json")
        summary.update(selected_crops=len(manifest["bases"]), sequences=len(manifest["sequences"]))
        write_json(dataset / "summary.json", summary)
        parent = read_json(dataset / "run.json")
        finish_record(dataset, parent, "completed_exploratory")
        permitted = {dataset / frame["paths"]["rgb"] for sequence in manifest["sequences"]
                     if sequence["arm"] == "clean" and sequence["split"] in {"train", "val"}
                     for frame in sequence["frames"]}
        with patch.object(msdzip, "load_rgb", wraps=msdzip.load_rgb) as reader:
            result = msdzip.train_msdzip_h0(
                dataset, fixture.root / "fit-history", representations=("abs",), epochs=1,
                max_train_bytes=32, max_val_bytes=16, model_batch_size=4, timesteps=2,
                hidden_dim=4, ffn_dim=8, vocab_dim=2, window_groups=2)
        self.assertTrue({call.args[0] for call in reader.call_args_list} <= permitted)
        self.assertFalse(result["test_rgb_loaded"])
        self.assertEqual(result["training_dataset_kind"], "history_dataset")
        self.assertEqual(result["pair_policy"], "adjacent observed clean-arm frames")
        self.assertFalse(set(result["buildings"]["train"]) & set(result["buildings"]["val"]))
        scorer = msdzip.MSDZipScorer(fixture.root / "fit-history" / result["results"]["abs"]["checkpoint_path"],
                                   dataset_fingerprint=result["input_sha256"])
        self.assertTrue(scorer.metadata["checkpoint_dataset_verified"])
        manifest["bases"][-1]["view_id"] = manifest["bases"][0]["view_id"]
        write_json(dataset / "manifest.json", manifest)
        finish_record(dataset, parent, "completed_exploratory")
        with self.assertRaisesRegex(ValueError, "leakage"):
            msdzip._parent(dataset)

    def test_actual_temporal_runner_trains_scores_reports_and_resumes_both_representations(self):
        from facade_change.io import read_json
        spec = importlib.util.spec_from_file_location(
            "msdzip_temporal_fixture", Path(__file__).with_name("test_2026-10-14_temporal_inputs.py"))
        fixtures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixtures)
        fixture = fixtures.TemporalInputsTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        crops_path = fixture.dataset / "crops.jsonl"
        crops = [json.loads(line) for line in crops_path.read_text().splitlines()]
        for crop in crops:
            crop.update(width=8, height=8, pad_right=32, pad_bottom=32)
        crops_path.write_text("".join(json.dumps(crop) + "\n" for crop in crops))
        history = fixture.history()
        history_index = msdzip._parent(history)[2]
        bases = {base["base_id"]: base for base in history_index["bases"]}
        self.assertTrue(any(frame["nuisance"]["kind"] == "occlusion"
                            for sequence in history_index["sequences"] for frame in sequence["frames"]))
        for case in history_index["cases"]:
            self.assertEqual(case["reference_support"], bases[case["base_id"]]["support"])
            self.assertEqual(case["source_support"], bases[case["base_id"]]["support"])
        runner = importlib.import_module("facade_change.2026-10-14_temporal_experiment")
        environment = runner.environment()
        methods = ["msdzip_abs", "msdzip_mod256"]
        options = {"history_run": str(history), "device": "cpu", "methods": methods,
                   "history_methods": methods, "methods_config": "", "semantic_run": "",
                   "training_python": sys.executable,
                   "method_options": {name: {"worker_python": sys.executable} for name in methods},
                   "training": {"epochs": 1, "max_train_bytes": 32, "max_val_bytes": 16,
                                "model_batch_size": 4, "timesteps": 2, "hidden_dim": 4,
                                "ffn_dim": 8, "vocab_dim": 2, "window_groups": 2,
                                "checkpoint_every_steps": 2}}
        output = fixture.root / "temporal-msdzip"
        console = io.StringIO()
        with patch.dict(os.environ, {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}), \
                patch.object(runner, "environment", return_value=copy.deepcopy(environment)), \
                contextlib.redirect_stdout(console):
            result = runner.run_temporal_experiment(output, **options)
            self.assertEqual(result["status"], "completed_exploratory",
                             {"methods": result.get("methods"), "stdout": console.getvalue()[-6000:]})
            units = [read_json(path) for path in sorted((output / "units").glob("*.json"))]
            committed = {unit["unit_id"]: unit["artifact_sha256"] for unit in units}
            self.assertEqual(Counter(unit["method"] for unit in units), dict.fromkeys(methods, 52))
            resumed = runner.run_temporal_experiment(output, resume=True)
            self.assertEqual(resumed["status"], "completed_exploratory", resumed.get("methods"))
        after = [read_json(path) for path in sorted((output / "units").glob("*.json"))]
        self.assertEqual(committed, {unit["unit_id"]: unit["artifact_sha256"] for unit in after})
        for method in methods:
            representation = method.removeprefix("msdzip_")
            smoke = read_json(output / "smoke_training" / representation / "2026-10-04_msdzip_h0_summary.json")
            trained = read_json(output / "training" / representation / "2026-10-04_msdzip_h0_summary.json")
            self.assertEqual(smoke["model_args"]["batchsize"], 2)
            self.assertEqual(trained["model_args"]["batchsize"], 4)
            self.assertFalse(trained["test_rgb_loaded"])
            training_sha = trained["results"][representation]["checkpoint_sha256"]
            smoke_sha = smoke["results"][representation]["checkpoint_sha256"]
            self.assertNotEqual(training_sha, smoke_sha)
            smoke_marker = read_json(output / "smoke" / f"{method}.json")
            self.assertEqual(smoke_marker["method_metadata"]["checkpoint_sha256"], smoke_sha)
            for unit in units:
                if unit["method"] == method:
                    self.assertEqual(unit["variant"], "current_train")
                    self.assertEqual(unit["method_metadata"]["checkpoint_sha256"], training_sha)
                    self.assertTrue(unit["method_metadata"]["checkpoint_dataset_verified"])
                    self.assertEqual(unit["method_metadata"]["input_sha256"], trained["input_sha256"])
            events = [json.loads(line) for line in
                      (output / "smoke_training" / representation / "events.jsonl").read_text().splitlines()]
            self.assertIn("interrupted", [event["event_type"] for event in events])
            self.assertIn("checkpoint_resume", [event["event_type"] for event in events])
        for name in ("2026-10-14_temporal_thresholds.json", "2026-10-14_temporal_summary.json",
                     "2026-10-14_temporal_gallery.html"):
            self.assertTrue((output / name).is_file(), name)
        self.assertFalse(read_json(output / "experiment_contract.json")["inputs"]["test_rgb_loaded"])
        self.assertFalse(Path(fixture.images[-1]["image_path"]).exists())
        self.assertFalse(any(unit["split"] == "test" for unit in units))
        with np.load(output / units[-1]["artifact_path"], allow_pickle=False) as arrays:
            self.assertEqual(arrays["scores"].shape, (8, 8))

    def test_original_shape_constraints_and_finite_logits(self):
        for args in ((4, 3, 6, 8, 2), (4, 2, 3, 8, 2)):
            with self.assertRaises(ValueError):
                msdzip._model_args(*args)
        model = msdzip._new_model(msdzip._model_args(4, 2, 4, 8, 2), self.torch.device("cpu"))
        logits = msdzip._predict(model, self.torch.zeros((4, 2), dtype=self.torch.long))
        self.assertEqual(tuple(logits.shape), (4, 256))
        self.assertTrue(self.torch.isfinite(logits).all())

    def test_vectorized_groups_preserve_original_lane_logits_and_gradients(self):
        model = msdzip._new_model(msdzip._model_args(4, 2, 4, 8, 2), self.torch.device("cpu"))
        contexts = self.torch.arange(16, dtype=self.torch.long).reshape(2, 4, 2)
        serial = self.torch.stack([msdzip._predict(model, x) for x in contexts])
        serial.square().sum().backward()
        serial_grad = {key: parameter.grad.clone() for key, parameter in model.named_parameters() if parameter.grad is not None}
        model.zero_grad(set_to_none=True)
        grouped = msdzip._predict_groups(model, contexts)
        grouped.square().sum().backward()
        self.torch.testing.assert_close(grouped, serial, rtol=1e-5, atol=1e-6)
        for key, parameter in model.named_parameters():
            if key in serial_grad:
                self.torch.testing.assert_close(parameter.grad, serial_grad[key], rtol=1e-4, atol=1e-5)
        self.assertEqual(model.last, [])


if __name__ == "__main__":
    unittest.main()
