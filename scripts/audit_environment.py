#!/usr/bin/env python3
"""Inspect the active interpreter without installing packages or loading weights.

The runner uses only the standard library. Imports run in separate processes so
a broken native extension does not prevent the rest of the report being saved.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "FACADE_AUDIT_JSON="
PROBES = {
    "numpy": """
import numpy as np
a = np.array([0, 255], dtype=np.uint8)
assert np.abs(a.astype(np.int16) - a[::-1].astype(np.int16)).tolist() == [255, 255]
info = {"version": np.__version__, "file": np.__file__}
""",
    "pillow": """
import PIL
from PIL import Image
import numpy as np
a = np.array([[[255, 0, 9]]], dtype=np.uint8)
assert np.array(Image.fromarray(a)).tolist() == a.tolist()
info = {"version": PIL.__version__, "file": PIL.__file__}
""",
    "opencv": """
import cv2
import numpy as np
assert hasattr(cv2, "USAC_MAGSAC"), "USAC_MAGSAC unavailable"
cv2.SIFT_create(nfeatures=100)
a = np.array([[[255, 0, 0]]], dtype=np.uint8)
assert cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).shape == (1, 1)
info = {"version": cv2.__version__, "file": cv2.__file__, "sift": True, "magsac": True}
""",
    "torch": """
import inspect
import torch
import numpy as np
a = np.zeros((2, 3), dtype=np.float32)
assert np.array_equal(torch.from_numpy(a).numpy(), a), "Torch/NumPy bridge failed"
assert "weights_only" in inspect.signature(torch.load).parameters, "torch.load lacks weights_only"
cuda = torch.cuda.is_available()
info = {"version": torch.__version__, "file": torch.__file__, "compiled_cuda": torch.version.cuda,
        "cuda_available": cuda, "device_count": torch.cuda.device_count() if cuda else 0,
        "numpy_bridge": True, "weights_only": True}
""",
    "kornia": """
import kornia
from kornia.feature import LoFTR
info = {"version": kornia.__version__, "file": kornia.__file__, "loftr_class_imported": True,
        "weights_loaded": False, "inference_tested": False}
""",
}
TEST_CODE = """
import unittest
suite = unittest.defaultTestLoader.discover("tests")
result = unittest.TextTestRunner(verbosity=2).run(suite)
info = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "successful": result.wasSuccessful()}
"""


def capture(command, timeout=45):
    started = time.monotonic()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    try:
        proc = subprocess.run(command, cwd=ROOT, env=env, text=True, errors="replace",
                              capture_output=True, timeout=timeout)
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr,
                "elapsed_seconds": round(time.monotonic() - started, 3)}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc),
                "elapsed_seconds": round(time.monotonic() - started, 3)}


def probe(code, timeout=45):
    command = "import json\n" + code + "\nprint(" + repr(MARKER) + " + json.dumps(info))\n"
    result = capture([sys.executable, "-B", "-c", command], timeout)
    lines = [line[len(MARKER):] for line in result["stdout"].splitlines() if line.startswith(MARKER)]
    if result["returncode"] == 0 and lines:
        result.update(status="ok", details=json.loads(lines[-1]))
    else:
        result["status"] = "failed"
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="New report directory; never overwritten")
    parser.add_argument("--run-tests", action="store_true", help="Run the small CPU suite with this interpreter")
    args = parser.parse_args(argv)
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    packages = {}
    for name in ("numpy", "Pillow", "opencv-python", "opencv-python-headless",
                 "opencv-contrib-python", "opencv-contrib-python-headless", "torch", "kornia"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    source_paths = sorted((ROOT / "facade_change").glob("*.py")) + sorted((ROOT / "tests").glob("*.py")) + [Path(__file__).resolve()]
    report = {"schema_version": 1, "run_id": out.name, "status": "running",
              "started_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
              "executable": sys.executable, "prefix": sys.prefix,
              "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"), "virtual_env": os.environ.get("VIRTUAL_ENV"),
              "python_supported": sys.version_info >= (3, 10), "installed_distributions": packages,
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
              "git": capture(["git", "rev-parse", "HEAD"], 5), "probes": {}}
    write_json(out / "audit.json", report)
    for name, code in PROBES.items():
        print(f"Checking {name} ...", flush=True)
        report["probes"][name] = probe(code)
        write_json(out / "audit.json", report)
    if shutil.which("nvidia-smi"):
        report["nvidia_smi"] = capture(["nvidia-smi"], 15)
    else:
        report["nvidia_smi"] = {"returncode": None, "stdout": "", "stderr": "nvidia-smi not found"}
    def good(name):
        return report["probes"][name]["status"] == "ok"
    manifest_ready = report["python_supported"] and good("numpy") and good("pillow")
    sift_ready = manifest_ready and good("opencv")
    tests = {"status": "not_requested"}
    if args.run_tests:
        if sift_ready:
            print("Running CPU project tests ...", flush=True)
            tests = probe(TEST_CODE, timeout=60)
            details = tests.get("details", {})
            if tests["status"] == "ok" and not (details.get("successful") and details.get("tests_run", 0) > 0 and details.get("skipped") == 0):
                tests["status"] = "failed_or_skipped"
            (out / "tests.log").write_text(tests["stdout"] + tests["stderr"], encoding="utf-8")
        else:
            tests = {"status": "not_run_missing_prerequisites"}
    report.update(tests=tests, readiness={
        "manifest_imports": manifest_ready, "sift_imports_and_features": sift_ready,
        "cpu_project_tests_passed": tests["status"] == "ok",
        "loftr_import_prerequisites": sift_ready and good("torch") and good("kornia"),
        "loftr_checkpoint_and_inference": "not_tested"})
    report.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(out / "audit.json", report)
    lines = ["Facade environment audit", f"Environment: {report['conda_environment'] or report['virtual_env'] or 'system/unknown'}",
             f"Python: {report['python']} ({report['executable']})", f"Python >=3.10: {report['python_supported']}",
             "Installed distributions: " + json.dumps(packages, ensure_ascii=False), ""]
    for name, result in report["probes"].items():
        lines.append(f"{name}: {result['status']}")
        if result.get("details"):
            lines.append("  " + json.dumps(result["details"], ensure_ascii=False))
        if result["stderr"].strip():
            lines.append("  " + "\n  ".join(result["stderr"].strip().splitlines()[-8:]))
    lines += ["", "Readiness: " + json.dumps(report["readiness"], ensure_ascii=False),
              "Tests: " + tests["status"] + " " + json.dumps(tests.get("details", {})),
              "LoFTR weights/inference and real facade data were not tested.",
              "No packages installed/updated; no weights downloaded; no GPU workloads started."]
    if args.run_tests and tests.get("status") not in {"ok", "not_run_missing_prerequisites"}:
        lines.append("Test failure details: tests.log")
    lines += ["", "GPU state:", report["nvidia_smi"]["stdout"] or report["nvidia_smi"]["stderr"]]
    summary = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Report: {out / 'summary.txt'}")
    return 0 if sift_ready and (not args.run_tests or tests["status"] == "ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
