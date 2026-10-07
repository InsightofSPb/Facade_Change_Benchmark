"""Exercise setup isolation and failure boundaries without network/package writes."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


COMMIT = "bdc3c61d7cf3377dc72d411a32713f5f353d61e8"


class NeuralSetupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "benchmark"
        (self.repo / "scripts").mkdir(parents=True)
        (self.repo / "third_party").mkdir()
        self.script = self.repo / "scripts/setup_neural_codecs.sh"
        shutil.copyfile(Path(__file__).resolve().parents[1] / "scripts/setup_neural_codecs.sh", self.script)
        self.source = self.repo / "third_party/arib_bps"
        self.template = self.root / "author-template"
        (self.template / ".git").mkdir(parents=True)
        (self.template / ".git/head").write_text(COMMIT)
        (self.template / "src/modules").mkdir(parents=True)
        (self.template / "src/modules/__init__.py").write_text("")
        (self.template / "src/modules/arib_bps.py").write_text("class ARIB_BPS: pass\nclass SIG: pass\nclass INS: pass\n")
        (self.template / "src/utils/coder").mkdir(parents=True)
        (self.template / "src/utils/coder/python_interface.cpp").write_text('// unchanged author fixture\n')
        (self.template / "src/utils/coder/mix_coder.h").write_bytes(b'// unchanged author header\r\n')
        hashes = {str(path.relative_to(self.template)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in self.template.rglob("*") if path.is_file() and ".git" not in path.parts}
        self.manifest = {"repository": "https://github.com/ZZ022/ArIB-BPS", "commit": COMMIT, "sha256": hashes}
        (self.repo / "third_party/arib_bps_provenance.json").write_text(json.dumps(self.manifest))
        shutil.copytree(self.template, self.source)
        self.fake_modules = self.root / "fake-modules"
        self.fake_modules.mkdir()
        for name in ("torch", "torchvision", "numpy", "PIL", "pandas", "tqdm"):
            self._module(name)
        self._module("pybind11", 'def get_include(): return "/worker pybind includes"\n')
        self.state = self.root / "envs.json"
        self.state.write_text(json.dumps({"envs": ["/env/scd_bench", "/env/facade-codecs"]}))
        self.base_marker = self.root / "base-packages.txt"
        self.base_marker.write_bytes(b"base packages must remain unchanged")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "commands.jsonl"
        self._executable("conda", r'''import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ['COMMAND_LOG']).open('a') as stream:
    stream.write(json.dumps({'tool': 'conda', 'args': args, 'user_site': os.environ.get('PYTHONNOUSERSITE')}) + '\n')
state_path = Path(os.environ['ENV_STATE'])
if args == ['env', 'list', '--json']:
    print(state_path.read_text())
elif args[0] == 'create':
    state = json.loads(state_path.read_text())
    state['envs'].append('/env/' + args[args.index('-n')+1])
    state_path.write_text(json.dumps(state))
elif args[0] == 'run':
    python_args = args[args.index('python')+1:]
    if not python_args or python_args.pop(0) != '-s':
        raise SystemExit(91)
    if python_args[:2] == ['-m', 'pip']:
        assert args[args.index('-n')+1] == 'facade-codecs', args
        assert '--no-deps' in python_args, args
        for requirement in python_args[python_args.index('--no-deps')+1:]:
            name, version = requirement.split('==')
            extra = 'def get_include(): return "/worker pybind includes"\n' if name == 'pybind11' else ''
            (Path(os.environ['FAKE_MODULES']) / (name+'.py')).write_text('__version__ = '+repr(version)+'\n'+extra)
    elif python_args[0] == '-':
        script = sys.stdin.read()
        # The compiler is stubbed: its result cannot be loaded as a real extension.
        if python_args[1:] and python_args[1].endswith('.so'):
            assert Path(python_args[1]).read_bytes() == b'synthetic built coder'
        else:
            result = subprocess.run([sys.executable, '-s', *python_args], input=script, text=True)
            raise SystemExit(result.returncode)
    else:
        raise SystemExit(subprocess.run([sys.executable, '-s', *python_args]).returncode)
''')
        self._executable("git", r'''import json, os, shutil, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ['COMMAND_LOG']).open('a') as stream:
    stream.write(json.dumps({'tool': 'git', 'args': args}) + '\n')
operation_args = args[2:] if args[:1] == ['-c'] else args
if operation_args[0] == 'clone':
    if os.environ.get('FAIL_CLONE'):
        print('synthetic network unavailable', file=sys.stderr)
        raise SystemExit(17)
    shutil.copytree(os.environ['AUTHOR_TEMPLATE'], args[-1])
else:
    root = Path(args[1])
    operation = args[2:]
    if operation == ['rev-parse', '--show-toplevel']:
        print(root.resolve())
    elif operation == ['rev-parse', 'HEAD']:
        print((root/'.git/head').read_text())
    elif operation[:2] == ['checkout', '--detach']:
        if os.environ.get('GLOBAL_AUTOCRLF') and not (root/'.git/autocrlf').is_file():
            for path in (root/'src').rglob('*.py'):
                path.write_bytes(path.read_bytes().replace(b'\n', b'\r\n'))
        (root/'.git/head').write_text(operation[2])
    elif operation == ['config', '--local', 'core.autocrlf', 'false']:
        (root/'.git/autocrlf').write_text('false')
    elif operation == ['diff', '--quiet', 'HEAD', '--']:
        raise SystemExit(1 if os.environ.get('DIRTY_AUTHOR') else 0)
    else:
        raise SystemExit(93)
''')
        self._executable("g++", r'''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ['COMMAND_LOG']).open('a') as stream:
    stream.write(json.dumps({'tool': 'g++', 'args': args}) + '\n')
if os.environ.get('FAIL_BUILD'):
    print('synthetic compiler error', file=sys.stderr)
    raise SystemExit(23)
Path(args[args.index('-o')+1]).write_bytes(b'synthetic built coder')
''')
        self.env = dict(os.environ, PATH=str(self.bin)+os.pathsep+os.environ["PATH"],
                        PYTHONPATH=str(self.fake_modules), COMMAND_LOG=str(self.log),
                        ENV_STATE=str(self.state), AUTHOR_TEMPLATE=str(self.template),
                        FAKE_MODULES=str(self.fake_modules), CODECS_ENV="facade-codecs",
                        CODECS_BASE_ENV="scd_bench", ARIB_SOURCE_DIR=str(self.source),
                        CONDA_DEFAULT_ENV="scd_bench")
        self.env.pop("PYTHONNOUSERSITE", None)

    def tearDown(self):
        self.assertEqual(self.base_marker.read_bytes(), b"base packages must remain unchanged")
        self.temporary.cleanup()

    def _module(self, name, extra=""):
        (self.fake_modules / (name+".py")).write_text(
            "import os\n"
            f"if os.environ.get('BROKEN_MODULE') == {name!r}: raise RuntimeError('synthetic dependency failure')\n"
            "__version__ = 'existing-version'\n" + extra)

    def _executable(self, name, body):
        path = self.bin / name
        path.write_text(f"#!{sys.executable}\n"+body)
        path.chmod(0o755)

    def _run(self, *arguments):
        return subprocess.run(["bash", str(self.script), *arguments], env=self.env,
                              text=True, capture_output=True, timeout=20)

    def _commands(self, tool=None):
        rows = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        return [row for row in rows if tool is None or row["tool"] == tool]

    def test_help_has_no_side_effects_and_protected_environment_is_rejected(self):
        result = self._run("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("No pretrained weights", result.stdout)
        self.assertEqual(self._commands(), [])
        self.env["CODECS_ENV"] = "lposs"
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("protected environment lposs", result.stderr)
        self.assertEqual(self._commands(), [])

    def test_healthy_reruns_do_not_install_or_clone_and_preserve_author_sources(self):
        before = {name: (self.source/name).read_bytes() for name in self.manifest["sha256"]}
        for _ in range(2):
            result = self._run()
            self.assertEqual(result.returncode, 0, result.stderr)
        rows = self._commands()
        self.assertFalse(any(row["args"][0] in {"create", "clone"} for row in rows))
        self.assertFalse(any("pip" in row["args"] for row in rows))
        self.assertTrue(all(row["user_site"] == "1" for row in self._commands("conda")))
        compiler = self._commands("g++")
        self.assertEqual(len(compiler), 2)
        self.assertIn("-I/worker pybind includes", compiler[0]["args"])
        self.assertEqual(compiler[0]["args"][compiler[0]["args"].index("-include")+1], "cstdint")
        self.assertEqual(before, {name: (self.source/name).read_bytes() for name in before})
        self.assertEqual((self.source/"src/utils/coder/mixcoder.so").read_bytes(), b"synthetic built coder")
        self.assertFalse(list(self.source.rglob(".mixcoder-build.*")))

    def test_only_missing_environment_and_helpers_are_created_once(self):
        self.state.write_text(json.dumps({"envs": ["/env/scd_bench"]}))
        for name in ("pybind11", "tqdm"):
            (self.fake_modules/(name+".py")).unlink()
        shutil.rmtree(self.source)
        self.env["GLOBAL_AUTOCRLF"] = "1"
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        creations = [row["args"] for row in self._commands("conda") if row["args"][0] == "create"]
        self.assertEqual(creations, [["create", "-y", "-n", "facade-codecs", "--clone", "scd_bench"]])
        pip = [row["args"] for row in self._commands("conda") if "pip" in row["args"]]
        self.assertEqual(len(pip), 1)
        self.assertEqual(pip[0][-3:], ["--no-deps", "pybind11==2.13.6", "tqdm==4.67.1"])
        self.assertEqual(sum("clone" in row["args"] for row in self._commands("git")), 1)
        self.assertEqual((self.source/".git/head").read_text(), COMMIT)
        self.assertEqual((self.source/"src/utils/coder/mix_coder.h").read_bytes(),
                         b'// unchanged author header\r\n')

    def test_broken_existing_dependencies_are_not_replaced(self):
        for name in ("torch", "tqdm"):
            with self.subTest(name=name):
                self.log.unlink(missing_ok=True)
                self.env["BROKEN_MODULE"] = name
                result = self._run()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(name, result.stderr)
                self.assertIn("synthetic dependency failure", result.stderr)
                self.assertFalse(any("pip" in row["args"] for row in self._commands()))
                self.assertEqual(self._commands("git"), [])
                self.assertEqual(self._commands("g++"), [])

    def test_missing_clone_source_does_not_create_a_new_environment(self):
        self.state.write_text(json.dumps({"envs": ["/env/base"]}))
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Source conda environment 'scd_bench' does not exist", result.stderr)
        self.assertFalse(any(row["args"][0] == "create" or "pip" in row["args"]
                             for row in self._commands("conda")))
        self.assertEqual(self._commands("git"), [])

    def test_source_head_and_source_hash_failures_stop_before_compilation(self):
        (self.source/".git/head").write_text("wrong-commit")
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Existing checkout was not reset", result.stderr)
        self.assertEqual(self._commands("g++"), [])
        (self.source/".git/head").write_text(COMMIT)
        altered = self.source/"src/utils/coder/mix_coder.h"
        altered.write_bytes(b"user changes remain untouched")
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Original ArIB source differs: src/utils/coder/mix_coder.h", result.stderr)
        self.assertEqual(altered.read_bytes(), b"user changes remain untouched")
        self.assertEqual(self._commands("g++"), [])

    def test_network_failure_leaves_no_partial_author_checkout(self):
        shutil.rmtree(self.source)
        self.env["FAIL_CLONE"] = "1"
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Failed to download ArIB-BPS", result.stderr)
        self.assertIn("synthetic network unavailable", result.stderr)
        self.assertFalse(self.source.exists())
        self.assertFalse(list(self.source.parent.glob(".arib-clone.*")))
        self.assertEqual(self._commands("g++"), [])

    def test_build_failure_preserves_existing_binary_and_removes_temporary_output(self):
        old_binary = self.source/"src/utils/coder/mixcoder.so"
        old_binary.write_bytes(b"old working binary")
        self.env["FAIL_BUILD"] = "1"
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("existing mixcoder.so was preserved", result.stderr)
        self.assertIn("synthetic compiler error", result.stderr)
        self.assertEqual(old_binary.read_bytes(), b"old working binary")
        self.assertFalse(list(old_binary.parent.glob(".mixcoder-build.*")))


if __name__ == "__main__":
    unittest.main()
