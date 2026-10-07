"""Local launch configuration retains the saved comparison and its sources."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from facade_change.io import read_json, write_json


class CodecSetupTests(unittest.TestCase):
    def test_configuration_retains_cached_methods_and_recomputes_rscd_only(self):
        script = Path(__file__).resolve().parents[1] / "scripts/configure_codecs.py"
        spec = importlib.util.spec_from_file_location("configure_codecs_test", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rscd = ("rscd_cmu", "rscd_diff_cmu", "rscd_pscd")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            write_json(root / "configs/benchmark.local.json", {
                "dataset_run": str(root / "runs/data"), "out": str(root / "runs/old"),
                "device": "cpu", "trust_checkpoint": True,
                "method_options": {name: {"device": "cpu", "source_root": "/author/rscd"}
                                   for name in rscd}})
            reused = SimpleNamespace(summary={"selected_base_count": 10, "selected_case_count": 60},
                                     methods=("msdzip_mod256", "geoscd") + rscd)
            with patch.object(module, "ReuseResults", return_value=reused):
                result = module.configure(reuse_run=root / "runs/old", repo_root=root)
                config = read_json(result["config"])
                self.assertEqual(config["recompute_methods"], list(rscd))
                self.assertEqual(result["cached_methods"], ["msdzip_mod256", "geoscd"])
                self.assertEqual(result["new_methods"], ["jpegls_mod256", "h264_rgb"])
                self.assertEqual(config["selection_path"], str(root / "runs/old/selection.json"))
                self.assertEqual(config["method_options"]["h264_rgb"]["worker_python"],
                                 str(Path(sys.executable).resolve()))
                self.assertEqual(config["method_options"]["rscd_cmu"]["source_root"], "/author/rscd")
                self.assertEqual(config["device"], "cpu")
                self.assertTrue(config["trust_checkpoint"])
                with self.assertRaises(FileExistsError):
                    module.configure(reuse_run=root / "runs/old", repo_root=root)


if __name__ == "__main__":
    unittest.main()
