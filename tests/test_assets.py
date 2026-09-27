import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from facade_change.assets import inventory_assets
from facade_change.io import read_json, write_json


class AssetInventoryTests(unittest.TestCase):
    def test_duplicate_small_code_and_config_only(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repo = base / "source"
            repo.mkdir()
            for name in ("original.py", "copy.py"):
                (repo / name).write_text("raise RuntimeError('never execute')\n")
            for name in ("original.yaml", "copy.yaml"):
                (repo / name).write_text("seed: 42\n")
            excluded = {repo / "train.log", repo / "model.pt", repo / "huge.py"}
            for path in excluded:
                path.write_bytes(b"x" * (2 * 1024 * 1024 + 1) if path.name == "huge.py" else b"data")
            config = base / "config.json"
            write_json(config, {"asset_roots": [{"path": "source"}]})
            original_open = Path.open

            def protected_open(path, *args, **kwargs):
                if path in excluded:
                    raise AssertionError("Opened excluded checkpoint, log or large script")
                return original_open(path, *args, **kwargs)

            with patch.object(Path, "open", protected_open):
                report = inventory_assets(config, base / "out")
            self.assertEqual(report["summary"]["duplicate_group_count"], 2)
            self.assertEqual(report["summary"]["duplicate_file_count"], 4)
            self.assertEqual(report["summary"]["duplicate_extra_file_count"], 2)
            self.assertEqual(report["summary"]["script_config_hashed_count"], 4)
            for row in report["duplicate_candidates"]:
                self.assertEqual(len(row["sha256"]), 64)
                self.assertEqual(len(row["paths"]), 2)

    def test_metadata_only_scan_prunes_data_and_does_not_infer_training(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repo = base / "LPOSS"
            for folder in (repo / "weights", repo / "scripts", repo / "datasets_mine", repo / ".cache"):
                folder.mkdir(parents=True)
            checkpoint = repo / "weights" / "trained_baskov_2016.pt"
            checkpoint.write_bytes(b"not a pickle")
            (repo / "scripts" / "train.py").write_text("raise RuntimeError('do not import')")
            (repo / "datasets_mine" / "wrong.pth").write_bytes(b"skip")
            (repo / ".cache" / "wrong.pth").write_bytes(b"skip")
            (repo / "linked").symlink_to(repo / "datasets_mine", target_is_directory=True)
            config = base / "config.json"
            write_json(config, {"asset_roots": [{"name": "LPOSS", "path": "LPOSS"}]})
            original_open = Path.open

            def protected_open(path, *args, **kwargs):
                if path == checkpoint:
                    raise AssertionError("Inventory opened checkpoint content")
                return original_open(path, *args, **kwargs)

            with patch.object(Path, "open", protected_open):
                report = inventory_assets(config, base / "out")
            self.assertEqual(report["summary"]["asset_count"], 2)
            found = next(row for row in report["assets"] if row["candidate_role"] == "checkpoint")
            self.assertEqual(found["size_bytes"], len(b"not a pickle"))
            self.assertIsNone(found["training_groups"])
            self.assertIsNone(found["content_sha256"])
            self.assertEqual(report["roots"][0]["symlinks_skipped"], 1)
            self.assertEqual(read_json(base / "out" / "run.json")["status"], "completed_inventory")

    def test_explicit_training_evidence_missing_root_and_depth_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repo = base / "source"
            (repo / "deep").mkdir(parents=True)
            (repo / "model.ckpt").write_bytes(b"model")
            (repo / "deep" / "hidden.log").write_text("training")
            config = base / "config.json"
            write_json(config, {
                "asset_roots": [{"path": "source", "max_depth": 0}, {"path": "missing"}],
                "asset_metadata": [{"path": "source/model.ckpt", "training_groups": ["building_a"],
                                    "training_groups_source": "reviewed training manifest"}],
            })
            report = inventory_assets(config, base / "out")
            self.assertEqual(report["assets"][0]["training_groups"], ["building_a"])
            self.assertEqual(report["roots"][1]["status"], "missing")
            self.assertEqual(report["roots"][0]["depth_limited_directories"], 1)
            self.assertIn("training_log", report["summary"]["roles_not_found_in_scan"])
            with self.assertRaises(FileExistsError):
                inventory_assets(config, base / "out")

    def test_entry_limit_is_reported_as_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repo = base / "source"
            repo.mkdir()
            for name in ("a.py", "b.py", "c.py"):
                (repo / name).write_text("pass")
            config = base / "config.json"
            write_json(config, {"asset_roots": [{"path": "source", "max_entries": 1}]})
            report = inventory_assets(config, base / "out")
            self.assertEqual(report["roots"][0]["status"], "entry_limit_reached")
            self.assertEqual(report["summary"]["roots_not_scanned_completely"], 1)
            self.assertEqual(len(report["assets"]), 1)


if __name__ == "__main__":
    unittest.main()
