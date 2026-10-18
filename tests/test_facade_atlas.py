"""Focused archive, local-server and real-export contracts for the facade atlas."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import threading
import unittest
from unittest import mock
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("facade_atlas_launcher", ROOT / "scripts/2026-10-18-facade-atlas.py")
ATLAS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ATLAS)

try:
    import numpy as np
    from PIL import Image, features
    import scipy  # noqa: F401
    EXPORT_AVAILABLE = hasattr(Image, "Resampling") and features.check("webp")
except ImportError:
    EXPORT_AVAILABLE = False


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def make_tar(path, entries):
    with tarfile.open(path, "w:gz") as archive:
        for name, content, kind in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            if kind == tarfile.REGTYPE:
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
            else:
                member.linkname = "../../outside"
                archive.addfile(member)


def synthetic_bundle(root, corrupt_hash=False, omit_originals=False):
    """Two genuine packed frames with a known class-bit change, no model output."""
    source = root / "source-fixture"
    source.mkdir()
    (source / "frames").mkdir()
    (source / "originals").mkdir()
    matrix = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    audit_frames, prepared_frames, original_frames = [], [], []
    for index, year in enumerate((2013, 2025)):
        rgb = np.full((512, 512, 3), 80 + 20 * index, dtype=np.uint8)
        packed = np.ones((512, 512), dtype=np.uint16)
        if index:
            packed[:256] = 3
        support = np.ones((512, 512), dtype=bool)
        npz_path = source / "frames" / f"{index}.npz"
        np.savez_compressed(npz_path, rgb=rgb, packed=packed, support=support)
        image_path = source / "originals" / f"{index}.png"
        Image.fromarray(rgb).save(image_path)
        image_hash = ATLAS.file_sha256(image_path)
        prepared_frames.append({"frame": index, "year": year, "image_id": index,
                                "image_sha256": image_hash, "artifact_path": f"frames/{index}.npz",
                                "artifact_sha256": ATLAS.file_sha256(npz_path)})
        original_frames.append({"sequence_id": "fixture-crop", "parent_sequence_id": "fixture",
                                "frame": index, "image_id": index, "year": year,
                                "artifact_path": f"originals/{index}.png", "sha256": image_hash,
                                "width": 512, "height": 512, "source_to_reference": matrix,
                                "crop_xywh": [0, 0, 512, 512]})
        audit_frames.append({"image_id": index, "year": year, "sha256": image_hash,
                             "width": 512, "height": 512, "source_to_reference": matrix})
    audit = {"kind": "real_temporal_audit", "sequences": [
        {"sequence_id": "fixture", "building_id": "fixture-building", "view_id": "fixture-view",
         "split": "test", "shape": [512, 512], "frames": audit_frames}]}
    write_json(source / "source/audit.json", audit)
    manifest = {
        "kind": "real_temporal_prepared", "schema_version": 1,
        "categories": [{"id": 1, "name": "WATER_STAIN", "color": "#aabbcc"},
                       {"id": 2, "name": "MISSING_ELEMENT", "color": "#ddeeff"}],
        "category_to_channel": [{"category_id": 1, "channel": 0}, {"category_id": 2, "channel": 1}],
        "audit_file_sha256": ATLAS.file_sha256(source / "source/audit.json"),
        "sequences": [{"sequence_id": "fixture-crop", "parent_sequence_id": "fixture",
                       "building_id": "fixture-building", "view_id": "fixture-view", "split": "test",
                       "shape": [512, 512], "crop_xywh": [0, 0, 512, 512], "missing_years": [],
                       "frames": prepared_frames}],
    }
    write_json(source / "manifest.json", manifest)
    if not omit_originals:
        write_json(source / "2026-10-18-originals.json", {"frames": original_frames})
    files = [{"path": path.relative_to(source).as_posix(), "sha256": ATLAS.file_sha256(path),
              "bytes": path.stat().st_size} for path in sorted(source.rglob("*")) if path.is_file()]
    if corrupt_hash:
        files[0]["sha256"] = "0" * 64
    bundle = {"kind": "facade_prototype_input_bundle", "schema_version": 1,
              "prepared_manifest_sha256": ATLAS.file_sha256(source / "manifest.json"),
              "sequences": 1, "frames": 2, "files": files}
    write_json(source / "2026-10-18-bundle.json", bundle)
    archive_path = root / "fixture.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(source).as_posix(), recursive=False)
    return archive_path


class AtlasArchiveTests(unittest.TestCase):
    def test_regular_files_unpack_without_exposing_extra_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "good.tar.gz"
            make_tar(archive, [("frames/one.bin", b"123", tarfile.REGTYPE)])
            summary = ATLAS.extract_bundle(archive, root / "unpacked")
            self.assertEqual(summary, {"files": 1, "uncompressed_bytes": 3})
            self.assertEqual((root / "unpacked/frames/one.bin").read_bytes(), b"123")

    def test_unsafe_members_and_duplicates_rejected(self):
        cases = [
            [("../outside", b"bad", tarfile.REGTYPE)],
            [("/outside", b"bad", tarfile.REGTYPE)],
            [("a/../../outside", b"bad", tarfile.REGTYPE)],
            [("C:/outside", b"bad", tarfile.REGTYPE)],
            [("a\\outside", b"bad", tarfile.REGTYPE)],
            [("link", b"", tarfile.SYMTYPE)],
            [("link", b"", tarfile.LNKTYPE)],
            [("device", b"", tarfile.CHRTYPE)],
            [("same", b"one", tarfile.REGTYPE), ("same", b"two", tarfile.REGTYPE)],
        ]
        for entries in cases:
            with self.subTest(entries=entries), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                archive = root / "bad.tar.gz"
                make_tar(archive, entries)
                with self.assertRaises(ValueError):
                    ATLAS.extract_bundle(archive, root / "unpacked")
                self.assertFalse((root / "outside").exists())

    def test_archive_size_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "large.tar.gz"
            make_tar(archive, [("one", b"1234", tarfile.REGTYPE)])
            with mock.patch.object(ATLAS, "MAX_MEMBER_BYTES", 3), self.assertRaisesRegex(ValueError, "Слишком большой"):
                ATLAS.extract_bundle(archive, root / "unpacked")

    def test_argument_modes(self):
        parsed = ATLAS.parse_args(["--bundle", "x.tar.gz", "--out", "output", "--prepare-only"])
        self.assertTrue(parsed.prepare_only)
        self.assertEqual(parsed.port, 8765)
        parsed = ATLAS.parse_args(["--serve", "output", "--port", "9000"])
        self.assertEqual(parsed.port, 9000)
        for arguments in (["--bundle", "x"], ["--serve", "output", "--out", "new"],
                          ["--serve", "output", "--prepare-only"], ["--serve", "output", "--port", "0"],
                          ["--serve", "output", "--bundle", "x", "--out", "new"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                ATLAS.parse_args(arguments)

    def test_existing_empty_or_nonempty_output_is_never_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "exists.tar.gz"
            archive.write_bytes(b"irrelevant; output check precedes unpack")
            for contents in (False, True):
                output = root / ("full" if contents else "empty")
                output.mkdir()
                if contents:
                    (output / "keep.txt").write_text("unchanged")
                with self.assertRaisesRegex(ValueError, "уже существует"):
                    ATLAS.prepare_site(archive, output)
                self.assertEqual(sorted(path.name for path in output.iterdir()), ["keep.txt"] if contents else [])


@unittest.skipUnless(EXPORT_AVAILABLE, "numpy, scipy and Pillow with WebP are required for actual export")
class AtlasExportTests(unittest.TestCase):
    def test_prepare_only_through_real_exporter(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = synthetic_bundle(root)
            output = root / "site"
            with mock.patch.object(ATLAS, "serve_site") as server:
                result = ATLAS.main(["--bundle", str(archive), "--out", str(output), "--prepare-only"])
                server.assert_not_called()
            self.assertEqual(result, 0)
            record = ATLAS.read_json(output / "run.json")
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["input_archive_sha256"], ATLAS.file_sha256(archive))
            self.assertFalse(record["configuration"]["model_inference"])
            data = ATLAS.read_json(output / "data/2026-10-18-facades.json")
            self.assertEqual(len(data["facades"]), 1)
            pair = data["facades"][0]["pairs"][0]
            self.assertEqual((pair["valid_pixels"], pair["changed_pixels"]), (512 * 512, 256 * 512))
            self.assertEqual(pair["changed_fraction"], 0.5)
            self.assertEqual(len(pair["events"]), 1)
            self.assertEqual(pair["events"][0]["category_id"], 2)
            self.assertEqual(pair["events"][0]["direction"], "added")
            self.assertEqual(pair["events"][0]["bbox"], [0, 0, 512, 256])
            self.assertTrue((output / "data/2026-10-18-locations.json").is_file())
            self.assertFalse((output / "source/audit.json").exists())
            self.assertFalse(list(output.rglob("*.py")))
            self.assertFalse(list(root.glob(".site-input-*")))

    def test_bad_hash_leaves_failed_run_and_never_serves(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = synthetic_bundle(root, corrupt_hash=True)
            output = root / "site"
            with mock.patch.object(ATLAS, "serve_site") as server:
                result = ATLAS.main(["--bundle", str(archive), "--out", str(output)])
                server.assert_not_called()
            self.assertEqual(result, 1)
            self.assertEqual(ATLAS.read_json(output / "run.json")["status"], "failed")
            self.assertFalse((output / "index.html").exists())
            with self.assertRaisesRegex(ValueError, "уже существует"):
                ATLAS.prepare_site(archive, output)

    def test_missing_originals_gives_actionable_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = synthetic_bundle(root, omit_originals=True)
            with self.assertRaisesRegex(ValueError, "include-originals"):
                ATLAS.prepare_site(archive, root / "site")
            self.assertEqual(ATLAS.read_json(root / "site/run.json")["status"], "failed")


class AtlasServerTests(unittest.TestCase):
    def test_only_completed_generated_site_is_served_on_loopback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            site = root / "site"
            site.mkdir()
            (site / "index.html").write_text("<h1>Atlas fixture</h1>")
            write_json(site / "data/2026-10-18-facades.json", {"facades": []})
            write_json(site / "run.json", {"kind": "facade_atlas_local_run", "status": "running"})
            with self.assertRaisesRegex(ValueError, "завершённую"):
                ATLAS.make_server(site, 0)
            write_json(site / "run.json", {"kind": "facade_atlas_local_run", "status": "completed"})
            (site / "source.py").write_text("private source")
            (site / "assets").mkdir()
            outside = root / "outside.txt"
            outside.write_text("outside secret")
            symlink_available = True
            try:
                (site / "assets/leak.txt").symlink_to(outside)
            except OSError:
                symlink_available = False
            with ATLAS.make_server(site, 0) as server:
                self.assertEqual(server.server_address[0], "127.0.0.1")
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                base = f"http://127.0.0.1:{server.server_port}"
                try:
                    with urllib.request.urlopen(base + "/", timeout=5) as response:
                        self.assertIn(b"Atlas fixture", response.read())
                    with urllib.request.urlopen(base + "/data/2026-10-18-facades.json", timeout=5) as response:
                        self.assertEqual(json.load(response), {"facades": []})
                    denied = ["/source.py", "/run.json", "/data/", "/../outside.txt"]
                    if symlink_available:
                        denied.append("/assets/leak.txt")
                    for path in denied:
                        with self.subTest(path=path), self.assertRaises(urllib.error.HTTPError) as caught:
                            urllib.request.urlopen(base + path, timeout=5)
                        self.assertEqual(caught.exception.code, 404)
                finally:
                    server.shutdown()
                    thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
