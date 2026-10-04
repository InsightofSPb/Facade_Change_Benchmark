"""Bootstrap regression: a broken SymPy dependency cannot block its own repair."""
import contextlib
import importlib
from importlib import metadata
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

# Keep project dependencies imported before patch.dict restores sys.modules.
import numpy
import PIL


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/setup_competitors.sh"
BOOTSTRAP = SCRIPT.read_text().split("<<'PY'\n", 1)[1].split("try:\n    lpips_version", 1)[0]


class SetupBootstrapTests(unittest.TestCase):
    def _run(self, available=False, torch_version="2.6.0+cu118", broken=False):
        state = {"available": available, "commands": [], "nms_called": False}
        torch = types.ModuleType("torch")
        torch.tensor = lambda values: values
        torchvision = types.ModuleType("torchvision")
        operations = types.ModuleType("torchvision.ops")

        def nms(*args):
            if not state["available"]:
                raise ImportError("SymPy cannot import missing mpmath")
            state["nms_called"] = True

        operations.batched_nms = nms
        versions = {"torch": torch_version, "torchvision": "0.21.0+cu118",
                    "numpy": "1.26.4", "Pillow": "11.1.0"}
        original_import = importlib.import_module

        def import_module(name, *args, **kwargs):
            if name != "mpmath":
                return original_import(name, *args, **kwargs)
            if broken:
                raise ImportError("Existing mpmath installation is broken")
            if not state["available"]:
                raise ModuleNotFoundError("No module named 'mpmath'", name="mpmath")
            return types.ModuleType("mpmath")

        def pip(command, **kwargs):
            state["commands"].append(command)
            self.assertTrue(kwargs["check"])
            self.assertEqual(command[-1], "mpmath==1.3.0")
            self.assertIn("--no-deps", command)
            state["available"] = True

        with patch.dict(sys.modules, {"torch": torch, "torchvision": torchvision, "torchvision.ops": operations}), \
                patch.object(metadata, "version", side_effect=versions.__getitem__), \
                patch.object(importlib, "import_module", side_effect=import_module), \
                patch("subprocess.run", side_effect=pip), contextlib.redirect_stdout(io.StringIO()):
            exec(compile(BOOTSTRAP, str(SCRIPT), "exec"), {})
        return state

    def test_missing_mpmath_is_repaired_before_the_core_import_check(self):
        state = self._run()
        self.assertTrue(state["nms_called"])
        self.assertEqual(len(state["commands"]), 1)
        self.assertFalse(any(name in state["commands"][0][-1] for name in ("torch", "numpy", "Pillow")))

    def test_working_dependency_is_reused_without_pip(self):
        state = self._run(available=True)
        self.assertTrue(state["nms_called"])
        self.assertEqual(state["commands"], [])

    def test_unsupported_core_and_broken_existing_dependency_remain_explicit(self):
        with self.assertRaisesRegex(SystemExit, "Torch >= 2.6"):
            self._run(torch_version="1.12.1")
        with self.assertRaisesRegex(ImportError, "installation is broken"):
            self._run(available=True, broken=True)
