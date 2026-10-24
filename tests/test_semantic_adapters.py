"""Contracts that do not load pretrained models or require a GPU."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.real_eval import semantic_adapters as adapters


VOCAB = [
    {"name": "CRACK", "prompts": ["crack", "wall crack"], "group": "target"},
    {"name": "WINDOW", "prompts": ["window"], "group": "object"},
]


class MutableProcessor:
    """Mimics SAM3 in-place state/output storage; earlier arrays may be reused."""
    def __init__(self):
        self.reset_count = 0
        self.seen = []
        self.buffer = np.zeros((1, 1, 2, 3), np.float32)
        self.confidence = np.zeros(1, np.float32)
        self.boxes = np.zeros((1, 4), np.float32)

    def reset_all_prompts(self, state):
        self.reset_count += 1
        state.pop("prompt", None)
        for key in ("scores", "masks_logits", "boxes"):
            state.pop(key, None)

    def set_text_prompt(self, prompt, state):
        if "prompt" in state:
            raise AssertionError("prompts were not reset")
        self.seen.append(prompt)
        state["prompt"] = prompt
        values = {"crack": (.2, .8), "wall crack": (.9, .5), "window": (.7, .6)}
        probability, score = values[prompt]
        self.buffer.fill(probability)
        self.confidence.fill(score)
        self.boxes[:] = [0, 0, 3, 2]
        state.update(masks_logits=self.buffer, scores=self.confidence, boxes=self.boxes)
        return state


class SemanticAdaptersTests(unittest.TestCase):
    def test_module_has_no_torch_import(self):
        # Execute the module with a guard: model frameworks must remain lazy.
        original = __import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "sam3", "open_clip", "mmcv"}:
                raise AssertionError("Model framework imported at module load: " + name)
            return original(name, *args, **kwargs)
        spec = importlib.util.spec_from_file_location("lazy_semantic_adapters", adapters.__file__)
        with patch("builtins.__import__", side_effect=guarded):
            spec.loader.exec_module(importlib.util.module_from_spec(spec))

    def test_sam3_preserves_instances_before_state_reuse(self):
        processor = MutableProcessor()
        state = {"backbone_out": {"image_features": "retained"}}
        result = adapters.sam3_queries(processor, state, VOCAB, (2, 3))
        extra = result["extra_arrays"]
        np.testing.assert_allclose(extra["instance_probabilities"][:, 0, 0], [.2, .9, .7])
        np.testing.assert_allclose(extra["instance_scores"], [.8, .5, .6])
        np.testing.assert_allclose(result["scores"][:, 0, 0], [.45, .42])
        np.testing.assert_array_equal(extra["instance_class_indices"], [0, 0, 1])
        np.testing.assert_array_equal(extra["instance_prompt_indices"], [0, 1, 2])
        self.assertEqual(processor.reset_count, 4)
        self.assertNotIn("prompt", state)
        self.assertIn("backbone_out", state)
        self.assertFalse(extra["instance_masks"][0].any())
        self.assertTrue(extra["instance_masks"][1].all())

    def test_sam3_extra_queries_do_not_change_target_scores(self):
        narrow = adapters.sam3_queries(MutableProcessor(), {}, VOCAB[:1], (2, 3))
        extended = adapters.sam3_queries(MutableProcessor(), {}, VOCAB, (2, 3))
        np.testing.assert_array_equal(narrow["scores"][0], extended["scores"][0])

    def test_sam3_rejects_actual_logits_misnamed_probability(self):
        processor = MutableProcessor()
        native = processor.set_text_prompt
        def bad(prompt, state):
            result = native(prompt, state)
            result["masks_logits"].fill(-2)
            return result
        processor.set_text_prompt = bad
        with self.assertRaisesRegex(ValueError, "not probabilities"):
            adapters.sam3_queries(processor, {}, VOCAB, (2, 3))

    def test_sam3_no_instances_preserves_class_shape(self):
        class EmptyProcessor(MutableProcessor):
            def set_text_prompt(self, prompt, state):
                return {"scores": np.empty(0), "masks_logits": np.empty((0, 1, 2, 3)), "boxes": np.empty((0, 4))}
        result = adapters.sam3_queries(EmptyProcessor(), {}, VOCAB, (2, 3))
        self.assertEqual(result["scores"].shape, (2, 2, 3))
        self.assertEqual(result["extra_arrays"]["instance_masks"].shape, (0, 2, 3))
        self.assertEqual(float(result["scores"].sum()), 0)

    def test_native_probabilities_never_renormalized(self):
        source = np.array([[[.2]], [[.3]]], np.float32)
        output = adapters.validate_prediction({"scores": source.copy(), "class_names": ["a", "b"]}, (1, 1))
        np.testing.assert_array_equal(output["scores"], source)
        with self.assertRaises(ValueError):
            adapters.validate_prediction({"scores": source * 10, "class_names": ["a", "b"]}, (1, 1))

    def test_bfloat16_tensors_convert_to_float_before_numpy(self):
        class Tensor:
            converted = False
            def detach(self):
                return self
            def is_floating_point(self):
                return True
            def float(self):
                self.converted = True
                return self
            def cpu(self):
                return self
            def numpy(self):
                if not self.converted:
                    raise TypeError("BFloat16 has no native NumPy dtype")
                return np.asarray([.25], np.float32)
        np.testing.assert_array_equal(adapters._numpy(Tensor()), [.25])

    def test_remote_clip_model_configuration_is_rejected(self):
        self.assertEqual(adapters._local_clip_architecture("ViT-B-16"), "ViT-B-16")
        with self.assertRaisesRegex(ValueError, "remote model configs"):
            adapters._local_clip_architecture("hf-hub:laion/some-model")

    def test_stock_softmax_explicit_scale_and_stable_extremes(self):
        raw = np.array([[[.01]], [[.02]]], np.float32)
        low = adapters.softmax_scores(raw, 1)
        high = adapters.softmax_scores(raw, 100)
        self.assertGreater(high[1, 0, 0], low[1, 0, 0])
        np.testing.assert_allclose(high.sum(0), 1)
        np.testing.assert_allclose(adapters.softmax_scores(raw + 1000, 100).sum(0), 1)
        with self.assertRaises(ValueError):
            adapters.softmax_scores(raw, 0)

    def test_checkpoint_filter_only_drops_dormant_dino(self):
        state = {"base_model.clip_backbone.backbone.weight": 1,
                 "base_model.clip_backbone.decode_head.class_embeddings": 2,
                 "base_model.dino_encoder.weight": 3, "unexpected": 4}
        retained, omitted = adapters.semantic_checkpoint_state(state)
        self.assertEqual(omitted, ["base_model.dino_encoder.weight"])
        self.assertIn("unexpected", retained)  # strict loader must reject this.
        self.assertIn("base_model.clip_backbone.decode_head.class_embeddings", retained)
        with self.assertRaises(ValueError):
            adapters.semantic_checkpoint_state({"weight": 1})

    def test_preflight_hashes_without_loading_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sam3.pt"
            checkpoint.write_bytes(b"not a torch pickle; must not load")
            with patch.object(importlib.util, "find_spec", return_value=None):
                result = adapters.preflight({"kind": "sam3", "checkpoint": str(checkpoint)})
            self.assertTrue(result["ok"], result["errors"])
            self.assertEqual(len(result["files"]["checkpoint"]["sha256"]), 64)
            before = result["files"]["checkpoint"]["sha256"]
            checkpoint.write_bytes(b"changed checkpoint content")
            with patch.object(importlib.util, "find_spec", return_value=None):
                after = adapters.preflight({"kind": "sam3", "checkpoint": str(checkpoint)})
            self.assertNotEqual(before, after["files"]["checkpoint"]["sha256"])

    def test_sam3_only_detector_subtree_from_merged_checkpoint(self):
        state = {"detector.backbone.weight": 1, "tracker.weight": 2}
        self.assertEqual(adapters.sam3_image_state(state), {"backbone.weight": 1})
        self.assertEqual(adapters.sam3_image_state({"backbone.weight": 3}), {"backbone.weight": 3})
        with self.assertRaises(ValueError):
            adapters.sam3_image_state({})

    def test_sam3_omits_only_inactive_interactive_neck_and_keeps_strict_loading(self):
        from unittest.mock import Mock
        active = {'backbone.vision_backbone.convs.0.conv_1x1.weight': np.ones((2, 2)),
                  'segmentation_head.weight': np.ones((2, 2))}
        extra = 'backbone.vision_backbone.sam2_convs.0.conv_1x1.weight'
        merged = {'detector.' + key: value for key, value in active.items()}
        merged['detector.' + extra] = np.zeros((2, 2))
        merged['tracker.weight'] = np.zeros((1,))
        model = Mock()
        model.inst_interactive_predictor = None
        model.state_dict.return_value = active
        metadata = adapters.load_sam3_image_state(model, merged)
        model.load_state_dict.assert_called_once_with(active, strict=True)
        self.assertEqual(metadata['omitted_inactive_keys'], [extra])
        self.assertEqual(metadata['loaded_keys'], len(active))
        self.assertIn('detector.' + extra, merged)  # Checkpoint mapping is not mutated.

        # A broad sam2_convs-prefix filter would silently accept both cases.
        for key, value in ((extra, np.zeros((3, 3))),
                           ('backbone.vision_backbone.sam2_convs.unknown.weight', np.zeros((2, 2)))):
            model.reset_mock()
            model.load_state_dict.side_effect = RuntimeError('unexpected key')
            invalid = {**active, key: value}
            with self.assertRaisesRegex(RuntimeError, 'unexpected key'):
                adapters.load_sam3_image_state(model, invalid)
            model.load_state_dict.assert_called_once_with(invalid, strict=True)

    def test_sam3_active_interactive_branch_and_unknown_keys_are_not_filtered(self):
        from unittest.mock import Mock
        active = 'backbone.vision_backbone.sam2_convs.0.conv_1x1.weight'
        unknown = 'backbone.vision_backbone.sam2_convs.unknown.weight'
        model = Mock()
        model.inst_interactive_predictor = None
        model.state_dict.return_value = {active: 1}
        state = {active: 1, unknown: 2, 'unexpected_image_head.weight': 3}
        metadata = adapters.load_sam3_image_state(model, state)
        model.load_state_dict.assert_called_once_with(state, strict=True)
        self.assertEqual(metadata['omitted_inactive_keys'], [])

    def test_sam3_required_state_errors_propagate(self):
        from unittest.mock import Mock
        for state in ({'other.weight': 1}, {'required.weight': np.zeros((3,))}):
            model = Mock()
            model.state_dict.return_value = {'required.weight': np.zeros((2,))}
            model.load_state_dict.side_effect = RuntimeError('missing key or shape mismatch')
            with self.assertRaisesRegex(RuntimeError, 'missing key or shape mismatch'):
                adapters.load_sam3_image_state(model, state)
            model.load_state_dict.assert_called_once_with(state, strict=True)

    def test_legacy_torch_loader_requires_explicit_trust_and_never_hides_other_errors(self):
        from unittest.mock import Mock
        legacy = Mock()
        legacy.load.side_effect = TypeError("unexpected keyword argument 'weights_only'")
        with self.assertRaisesRegex(RuntimeError, 'trust_checkpoint:true'):
            adapters._load_tensors(legacy, 'local.pth')
        self.assertEqual(legacy.load.call_count, 1)
        legacy.reset_mock()
        legacy.load.side_effect = [TypeError("unexpected keyword argument 'weights_only'"), {'weight': 1}]
        self.assertEqual(adapters._load_tensors(legacy, 'local.pth', trust_checkpoint=True), {'weight': 1})
        self.assertEqual(legacy.load.call_args_list[-1].kwargs, {'map_location': 'cpu'})
        legacy.load.side_effect = TypeError('unrelated error')
        with self.assertRaisesRegex(TypeError, 'unrelated error'):
            adapters._load_tensors(legacy, 'local.pth', trust_checkpoint=True)

    def test_auto_rejects_missing_or_ambiguous_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "Downloads are disabled"):
                adapters._one_local("auto", "weights", [])
            (root / "a").write_text("a")
            (root / "b").write_text("b")
            with self.assertRaisesRegex(FileNotFoundError, "auto candidates"):
                adapters._one_local("auto", "weights", [root / "a", root / "b"])


if __name__ == "__main__":
    unittest.main()
