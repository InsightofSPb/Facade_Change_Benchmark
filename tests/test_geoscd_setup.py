import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class GeoSCDSetupTests(unittest.TestCase):
    """Exercise the shell entry point without installing packages or fetching weights."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "benchmark"
        (self.repo / "scripts").mkdir(parents=True)
        (self.repo / "configs").mkdir()
        source = Path(__file__).resolve().parents[1]
        self.script = self.repo / "scripts/setup_geoscd.sh"
        shutil.copyfile(source / "scripts/setup_geoscd.sh", self.script)
        shutil.copyfile(source / "configs/2026-10-04_geoscd_geometry.requirements.txt",
                        self.repo / "configs/2026-10-04_geoscd_geometry.requirements.txt")
        self.geoscd = self.root / "GeoSCD"
        self.geoscd.mkdir()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "commands.jsonl"
        self.download_log = self.root / "downloads.jsonl"
        self._executable("git", """import sys
if sys.argv[1:][-2:] == ['rev-parse', 'HEAD']:
    print('dd31369654e96d6843cc4bbcebce854a8bc2159b')
""")
        self._executable("conda", """import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ['COMMAND_LOG']).open('a') as stream:
    stream.write(json.dumps({'args': args, 'PYTHONNOUSERSITE': os.environ.get('PYTHONNOUSERSITE')}) + '\\n')
if os.environ.get('PYTHONNOUSERSITE') != '1':
    sys.exit(91)
if args and args[0] == 'run':
    python = args[args.index('python'):]
    if python[1] != '-s':
        sys.exit(92)
    if python[2:] == ['-m', 'pip', 'check'] and os.environ.get('FAIL_PIP_CHECK') == '1':
        print('synthetic environment dependency conflict', file=sys.stderr)
        sys.exit(1)
    if python[2] == '-':
        result = subprocess.run([sys.executable, '-s', *python[2:]], input=sys.stdin.read(), text=True)
        sys.exit(result.returncode)
""")
        fake_modules = self.root / "fake_modules"
        (fake_modules / "torch").mkdir(parents=True)
        (fake_modules / "torch/__init__.py").write_text("""import json, os
from pathlib import Path
class Hub:
    @staticmethod
    def download_url_to_file(url, target, progress):
        with Path(os.environ['DOWNLOAD_LOG']).open('a') as stream:
            stream.write(json.dumps({'url': url, 'target': target}) + '\\n')
        Path(target).write_bytes(url.encode())
hub = Hub()
""")
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                        GEOSCD_ROOT=str(self.geoscd), COMMAND_LOG=str(self.log),
                        DOWNLOAD_LOG=str(self.download_log), PYTHONPATH=str(fake_modules))
        self.env.pop("PYTHONNOUSERSITE", None)

    def tearDown(self):
        self.temp.cleanup()

    def _executable(self, name, body):
        path = self.bin / name
        path.write_text(f"#!{sys.executable}\n" + body)
        path.chmod(0o755)

    def _run(self, *flags):
        return subprocess.run(["bash", str(self.script), *flags], env=self.env,
                              text=True, capture_output=True, timeout=20)

    def _commands(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def _downloads(self):
        return [json.loads(line) for line in self.download_log.read_text().splitlines()] if self.download_log.exists() else []

    def test_full_setup_isolates_every_python_and_downloads_both_matching_models(self):
        sam3 = self.repo / "models/sam3.pt"
        sam3.parent.mkdir()
        sam3.write_bytes(b"user SAM3 remains untouched")
        result = self._run("--full", "--download-weights")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self._commands()
        self.assertTrue(all(row["PYTHONNOUSERSITE"] == "1" for row in commands))
        self.assertIn(["env", "config", "vars", "set", "-n", "facade-geoscd", "PYTHONNOUSERSITE=1"],
                      [row["args"] for row in commands])
        python_commands = [row["args"][row["args"].index("python"):] for row in commands if "python" in row["args"]]
        self.assertTrue(all(command[1] == "-s" for command in python_commands))
        self.assertTrue(any("torch==2.6.0" in command for command in python_commands))
        self.assertTrue(any("git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf" in command
                            for command in python_commands))
        self.assertEqual([row["url"] for row in self._downloads()], [
            "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"])
        self.assertTrue((self.repo / "models/vggt_1b.pt").is_file())
        self.assertTrue((self.repo / "models/sam_vit_h_4b8939.pth").is_file())
        self.assertFalse(list(self.repo.glob("models/*.download")))
        self.assertEqual(sam3.read_bytes(), b"user SAM3 remains untouched")
        self.assertIn("conda deactivate; conda activate facade-geoscd", result.stdout)

    def test_existing_legacy_vggt_is_reused_without_copy_and_existing_sam_is_preserved(self):
        legacy = self.geoscd / "src/pretrained/model.pt"
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(b"old VGGT")
        sam = self.repo / "models/sam_vit_h_4b8939.pth"
        sam.parent.mkdir()
        sam.write_bytes(b"old SAM1")
        result = self._run("--download-weights", "--full")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._downloads(), [])
        self.assertFalse((self.repo / "models/vggt_1b.pt").exists())
        self.assertEqual(legacy.read_bytes(), b"old VGGT")
        self.assertEqual(sam.read_bytes(), b"old SAM1")
        self.assertIn(f"Reusing legacy VGGT checkpoint: {legacy}", result.stdout)

    def test_geometry_only_download_does_not_install_sam_or_fetch_its_weights(self):
        result = self._run("--download-weights")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self._downloads()), 1)
        self.assertIn("VGGT-1B", self._downloads()[0]["url"])
        self.assertFalse((self.repo / "models/sam_vit_h_4b8939.pth").exists())
        self.assertFalse(any("segment-anything" in " ".join(row["args"]) for row in self._commands()))

    def test_failed_environment_check_stops_before_any_weights_download(self):
        self.env["FAIL_PIP_CHECK"] = "1"
        result = self._run("--full", "--download-weights")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("synthetic environment dependency conflict", result.stderr)
        self.assertEqual(self._downloads(), [])
        self.assertFalse((self.repo / "models").exists())

    def test_full_without_download_requires_explicit_download_flag(self):
        result = self._run("--full")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._downloads(), [])
        self.assertIn("Weights were not downloaded", result.stdout)


if __name__ == "__main__":
    unittest.main()
