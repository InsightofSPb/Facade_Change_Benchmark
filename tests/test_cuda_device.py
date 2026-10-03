"""LoFTR resolves legacy CUDA device queries without requiring a GPU or weights."""
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from facade_change.alignment import LoFTRMatcher


class CUDADeviceTests(unittest.TestCase):
    def setUp(self):
        self.cuda = SimpleNamespace(is_available=Mock(return_value=True),
                                    current_device=Mock(return_value=3),
                                    mem_get_info=Mock(return_value=(8_000, 16_000)),
                                    get_device_name=Mock(return_value="Mock GPU"))
        torch = ModuleType("torch")
        torch.cuda, torch.__version__ = self.cuda, "legacy-test"
        kornia = ModuleType("kornia")
        kornia.__version__ = "test"
        feature = ModuleType("kornia.feature")
        self.model = Mock()
        self.model.to.return_value = self.model
        self.model.eval.return_value = self.model
        feature.LoFTR = Mock(return_value=self.model)
        kornia.feature = feature
        patches = (patch.dict(sys.modules, {"torch": torch, "kornia": kornia, "kornia.feature": feature}),
                   patch("facade_change.alignment.resolve_loftr_checkpoint",
                         return_value=(Path("mock.ckpt"), "torch_cache")),
                   patch("facade_change.alignment.load_loftr_state", return_value=({"weight": object()}, "mock")),
                   patch("facade_change.alignment.sha256", return_value="mock-checkpoint-sha256"))
        for mocked in patches:
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_bare_cuda_uses_current_nonzero_index_for_queries_and_model(self):
        matcher = LoFTRMatcher(device="cuda")
        self.cuda.current_device.assert_called_once_with()
        self.cuda.mem_get_info.assert_called_once_with("cuda:3")
        self.cuda.get_device_name.assert_called_once_with("cuda:3")
        self.model.to.assert_called_once_with("cuda:3")
        self.assertEqual(matcher.device, "cuda:3")
        self.assertEqual(matcher.metadata["device"], "cuda:3")

    def test_explicit_cuda_index_is_preserved_even_when_current_device_differs(self):
        matcher = LoFTRMatcher(device="cuda:1")
        self.cuda.current_device.assert_not_called()
        self.cuda.mem_get_info.assert_called_once_with("cuda:1")
        self.cuda.get_device_name.assert_called_once_with("cuda:1")
        self.model.to.assert_called_once_with("cuda:1")
        self.assertEqual(matcher.device, "cuda:1")
        self.assertEqual(matcher.metadata["device"], "cuda:1")

    def test_unavailable_cuda_fails_without_cpu_fallback(self):
        self.cuda.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError, "no silent CPU fallback"):
            LoFTRMatcher(device="cuda")
        self.cuda.current_device.assert_not_called()
        self.cuda.mem_get_info.assert_not_called()
        self.model.to.assert_not_called()


if __name__ == "__main__":
    unittest.main()
