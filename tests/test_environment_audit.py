"""Regressions from the lposs report: optional Torch features are not imports."""
from contextlib import redirect_stdout
import io
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

AUDIT = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/audit_environment.py"))


def old_torch():
    def load(path, map_location=None):
        raise AssertionError("Audit must not load checkpoint weights")
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
        self.assertFalse(scope["info"]["default_safe_checkpoint_loader_available"])
        self.assertTrue(scope["info"]["trusted_legacy_checkpoint_loader_available"])
        self.assertTrue(scope["info"]["numpy_bridge"])
        self.assertTrue(scope["info"]["cuda_available"])
        self.assertEqual(scope["info"]["device_count"], 1)

    def test_report_exposes_legacy_option_without_claiming_inference(self):
        scope = {}
        with patch.dict("sys.modules", {"torch": old_torch()}):
            exec(AUDIT["PROBES"]["torch"], scope)

        def fake_probe(code, timeout=45):
            details = scope["info"] if code == AUDIT["PROBES"]["torch"] else {}
            return {"status": "ok", "details": details, "stdout": "", "stderr": ""}

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "audit"
            with patch.dict(AUDIT["main"].__globals__, {
                "probe": fake_probe,
                "capture": lambda *args: {"returncode": 0, "stdout": "", "stderr": ""},
            }), patch("shutil.which", return_value=None), redirect_stdout(io.StringIO()):
                self.assertEqual(AUDIT["main"](["--out", str(out)]), 0)
            report = json.loads((out / "audit.json").read_text())
            ready = report["readiness"]
            self.assertTrue(ready["loftr_checkpoint_loader_supported"])
            self.assertFalse(ready["loftr_default_safe_checkpoint_loader_supported"])
            self.assertTrue(ready["loftr_trusted_legacy_checkpoint_loader_available"])
            self.assertTrue(ready["loftr_checkpoint_requires_explicit_trust"])
            self.assertEqual(ready["loftr_checkpoint_and_inference"], "not_tested")
            self.assertIn("--trust-checkpoint", (out / "summary.txt").read_text())

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
