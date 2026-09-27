"""Regressions from the lposs report: optional Torch features are not imports."""
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

AUDIT = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/audit_environment.py"))


def old_torch():
    def load(path, map_location=None):
        return None
    return SimpleNamespace(
        __version__="1.12.1", __file__="fixture/torch.py", load=load,
        version=SimpleNamespace(cuda="11.3"),
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
        from_numpy=lambda array: SimpleNamespace(numpy=lambda: array),
    )


class EnvironmentAuditTests(unittest.TestCase):
    def test_old_torch_retains_bridge_and_cuda_without_weights_only(self):
        scope = {}
        with patch.dict("sys.modules", {"torch": old_torch()}):
            exec(AUDIT["PROBES"]["torch"], scope)
        self.assertFalse(scope["info"]["weights_only"])
        self.assertTrue(scope["info"]["numpy_bridge"])
        self.assertTrue(scope["info"]["cuda_available"])
        self.assertEqual(scope["info"]["device_count"], 1)

    def test_numpy_bridge_failure_does_not_hide_cuda_details(self):
        torch = old_torch()
        def broken_bridge(array):
            raise RuntimeError("fixture bridge failure")
        torch.from_numpy = broken_bridge
        scope = {}
        with patch.dict("sys.modules", {"torch": torch}):
            exec(AUDIT["PROBES"]["torch"], scope)
        self.assertFalse(scope["info"]["numpy_bridge"])
        self.assertIn("fixture bridge failure", scope["info"]["numpy_bridge_error"])
        self.assertTrue(scope["info"]["cuda_available"])


if __name__ == "__main__":
    unittest.main()
