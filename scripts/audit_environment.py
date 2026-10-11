#!/usr/bin/env python3
"""Inspect the active interpreter without installing packages or loading weights.

The runner uses only the standard library. Imports run in separate processes so
a broken native extension does not prevent the rest of the report being saved.
"""
from __future__ import annotations

import argparse
import csv
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
info = {"version": torch.__version__, "file": torch.__file__, "compiled_cuda": torch.version.cuda,
        "weights_only": "weights_only" in inspect.signature(torch.load).parameters,
        "cuda_kernel_execution": "not_tested"}
info["default_safe_checkpoint_loader_available"] = info["weights_only"]
info["trusted_legacy_checkpoint_loader_available"] = not info["weights_only"] and callable(torch.load)
try:
    a = np.zeros((2, 3), dtype=np.float32)
    info["numpy_bridge"] = bool(np.array_equal(torch.from_numpy(a).numpy(), a))
except Exception as exc:
    info.update(numpy_bridge=False, numpy_bridge_error=str(exc))
try:
    info["cuda_available"] = torch.cuda.is_available()
    info["device_count"] = torch.cuda.device_count() if info["cuda_available"] else 0
except Exception as exc:
    info.update(cuda_available=None, cuda_query_error=str(exc))
""",
    "kornia": """
import kornia
from kornia.feature import LoFTR
info = {"version": kornia.__version__, "file": kornia.__file__, "loftr_class_imported": True,
        "weights_loaded": False, "inference_tested": False}
""",
    "project_cpu": """
from facade_change import cli, data, geometry, pipeline
info = {"project_cpu_imports": True, "cli_file": cli.__file__}
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



# Cleanup is deliberately a read-only inventory. Metadata contains dependencies,
# not a proof that an unreferenced artifact has no scientific value.
PATH_SUFFIXES = {".json", ".jsonl", ".csv", ".npz", ".npy", ".pth", ".pt", ".ckpt",
                 ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".py", ".html", ".gz", ".zip", ".tar"}


def _path_strings(value, key=""):
    if isinstance(value, dict):
        for name, child in value.items():
            # Artifact/source hashes often use the file path as the dictionary key.
            if isinstance(name, str) and ("/" in name or Path(name).suffix.lower() in PATH_SUFFIXES):
                yield name, "dictionary_key"
            yield from _path_strings(child, str(name))
    elif isinstance(value, list):
        for child in value:
            yield from _path_strings(child, key)
    elif isinstance(value, str) and value and len(value) < 4096:
        if "://" in value or "\n" in value or value in {"auto", "none", "null"}:
            return
        if (value.startswith(("/", "./", "../", "~/")) or "/" in value
                or Path(value).suffix.lower() in PATH_SUFFIXES):
            yield value, key


def _lexical(path):
    """Make paths absolute without dereferencing symlinks."""
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _symlink_component(path):
    return next((p for p in (path, *path.parents) if p.is_symlink()), None)


def _metadata_files(directory):
    """Do not enter symlink directories or read symlink metadata files."""
    files, links, logical_bytes = [], [], 0
    def failed(error):
        raise error
    for base, dirs, names in os.walk(directory, followlinks=False, onerror=failed):
        for name in list(dirs):
            path = Path(base) / name
            if path.is_symlink():
                links.append(path)
                dirs.remove(name)
        dirs[:] = [name for name in dirs if name not in {".git", "__pycache__"}]
        for name in names:
            path = Path(base) / name
            if path.is_symlink():
                links.append(path)
                continue
            logical_bytes += path.stat().st_size
            if path.suffix.lower() in {".json", ".jsonl"}:
                files.append(path)
    return files, links, logical_bytes


def _active_run_paths(runs):
    """Best-effort Linux process inspection; no processes are changed."""
    result, available = set(), Path("/proc").is_dir()
    if available:
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit() or int(proc.name) == os.getpid():
                continue
            try:
                command = (proc / "cmdline").read_bytes().decode("utf-8", "replace").split("\0")
                cwd = Path(os.readlink(proc / "cwd"))
            except (OSError, PermissionError):
                continue
            for arg in [str(cwd), *command]:
                # Match full paths and cwd-relative CLI path arguments, including --out=...
                arg = arg.split("=", 1)[-1] if arg.startswith("--") else arg
                if not arg or "\n" in arg:
                    continue
                path = _lexical(cwd / arg)
                if path.is_relative_to(runs):
                    result.add(path)
    return sorted(result), available


def cleanup_plan(repository, keep_runs, scan_roots=(), active_paths=None):
    """Retain transitive dependencies of selected, configured and unfinished runs.

    All existing interpretations of a relative metadata path are kept when the
    origin is ambiguous. External paths are reported, never recursively scanned.
    There is intentionally no delete/apply counterpart to this advisory report.
    """
    repository = Path(repository).resolve()
    runs = repository / "runs"
    if not runs.is_dir() or runs.is_symlink():
        raise ValueError("Cleanup audit needs a real, non-symlink repository/runs directory")
    entries = {_lexical(p): {"path": str(_lexical(p)), "reasons": [], "references": [],
                            "logical_bytes": None, "run_status": None}
               for p in runs.iterdir() if not p.name.startswith(".")}
    if not keep_runs:
        raise ValueError("Provide at least one --keep-run; no implicit newest-run selection")

    def owner(path):
        try:
            parts = path.relative_to(runs).parts
            return runs / parts[0] if parts else None
        except ValueError:
            return None

    seeds, edges, unresolved, metadata_hashes = set(), {}, [], {}
    external_references, warnings = [], []

    def retain(path, reason):
        target = owner(path)
        if target in entries:
            seeds.add(target)
            if reason not in entries[target]["reasons"]:
                entries[target]["reasons"].append(reason)

    for raw in keep_runs:
        path = _lexical(repository / Path(raw).expanduser())
        if owner(path) not in entries or not path.exists():
            raise ValueError(f"--keep-run is missing or outside repository/runs: {raw}")
        retain(path, "explicit_keep")
    if active_paths is None:
        active_paths, available = _active_run_paths(runs)
    else:
        available = True
    if not available:
        warnings.append("Process inspection unavailable; unfinished runs remain protected")
    for path in active_paths:
        retain(_lexical(path), "active_process_reference")

    def record_reference(source, origin, raw, key):
        candidate = Path(raw).expanduser()
        bases = [origin.parent, owner(origin), repository, runs]
        candidates = {candidate} if candidate.is_absolute() else {
            base / candidate for base in bases if base is not None}
        paths = sorted({_lexical(p) for p in candidates})
        found = [p for p in paths if p.exists() or _symlink_component(p)]
        if not found:
            found = [p for p in paths if owner(p) in entries]
        if not found:
            unresolved.append({"source": str(origin), "value": raw, "key": key})
            return
        for path in found:
            target = owner(path)
            ref = {"source": str(origin), "value": raw, "resolved": str(path), "key": key}
            if target in entries:
                if not path.exists() and not _symlink_component(path):
                    unresolved.append({"source": str(origin), "value": raw, "key": key,
                                       "resolved": str(path), "error": "referenced artifact is missing; owning run retained"})
                edges.setdefault(source, set()).add(target)
                entries[target]["references"].append(ref)
                link = _symlink_component(path)
                if link:
                    retain(target, "symlink_reference_needs_review")
                    # Read the link destination only; never open or traverse it.
                    resolved = path.resolve(strict=False)
                    linked_owner = owner(resolved)
                    if linked_owner in entries:
                        edges.setdefault(source, set()).add(linked_owner)
                        retain(linked_owner, "symlink_target_needs_review")
                    else:
                        external_references.append({**ref, "resolved": str(resolved), "via_symlink": True})
            elif not path.is_relative_to(repository):
                external_references.append(ref)

    def inspect_metadata(source, path):
        try:
            # Large prediction files are not metadata; never consume unbounded RAM.
            if path.stat().st_size > 64 * 1024 * 1024:
                raise ValueError("metadata exceeds 64 MiB audit bound")
            payload = path.read_bytes()
            metadata_hashes[str(path)] = hashlib.sha256(payload).hexdigest()
            if path.suffix.lower() == ".jsonl":
                documents = (json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip())
            else:
                documents = [json.loads(payload)]
            for document in documents:
                for raw, key in _path_strings(document):
                    record_reference(source, path, raw, key)
        except (OSError, ValueError, UnicodeError) as exc:
            unresolved.append({"source": str(path), "error": str(exc)})
            if source in entries:
                retain(source, "unreadable_metadata")

    for path, entry in entries.items():
        if path.is_symlink():
            retain(path, "symlink_entry_needs_review")
            target = path.resolve(strict=False)
            retain(target, "symlink_target_needs_review")
            continue
        if not path.is_dir():
            entry["logical_bytes"] = path.stat().st_size
            if path.suffix.lower() in {".gz", ".zip", ".tar"}:
                entry["run_status"] = "archive_not_inspected"
            else:
                retain(path, "non_run_artifact_needs_review")
            continue
        try:
            files, links, logical_bytes = _metadata_files(path)
            entry["logical_bytes"] = logical_bytes
            for link in links:
                retain(path, "contains_symlink_needs_review")
                retain(link.resolve(strict=False), "symlink_target_needs_review")
            marker = path / "run.json"
            if not marker.is_file():
                marker = path / "summary.json"
            state = json.loads(marker.read_text()) if marker.is_file() and not marker.is_symlink() else {}
            status = state.get("status") if isinstance(state, dict) else None
            entry["run_status"] = status
            if not isinstance(status, str) or not status.startswith("completed"):
                retain(path, "unfinished_or_unknown_status")
            for metadata in files:
                inspect_metadata(path, metadata)
        except (OSError, ValueError, UnicodeError) as exc:
            retain(path, "unreadable_run")
            unresolved.append({"source": str(path), "error": str(exc)})
    configs = repository / "configs"
    if configs.is_dir() and not configs.is_symlink():
        config_files, links, _ = _metadata_files(configs)
        for path in config_files:
            # Example configs do not express active local dependencies.
            if ".example." not in path.name:
                inspect_metadata("configs", path)
        if links:
            warnings.append("Symlink config files were not opened; review their dependencies manually")
    for target in edges.get("configs", ()):
        retain(target, "repository_config_dependency")
    retained = set(seeds)
    pending = list(seeds)
    while pending:
        parent = pending.pop()
        for child in edges.get(parent, ()):
            if child != parent:
                reason = "dependency_of:" + parent.name
                if reason not in entries[child]["reasons"]:
                    entries[child]["reasons"].append(reason)
            if child not in retained:
                retained.add(child)
                pending.append(child)
    rows = []
    for path, entry in sorted(entries.items()):
        entry["decision"] = "retained" if path in retained else "review_candidate"
        entry["dependencies"] = sorted(str(p) for p in edges.get(path, ()) if p != path)
        rows.append(entry)
    external = []
    for raw in scan_roots:
        directory = _lexical(raw)
        if _symlink_component(directory) or not directory.is_dir():
            warnings.append(f"External inventory root unavailable or symlink; skipped: {directory}")
            continue
        for item in sorted(directory.iterdir()):
            if item.name.startswith(".") or item == repository:
                continue
            refs = [r for r in external_references
                    if Path(r["resolved"]) == item or Path(r["resolved"]).is_relative_to(item)]
            external.append({"path": str(item), "kind": "symlink" if item.is_symlink() else
                             "directory" if item.is_dir() else "file",
                             "decision": "referenced_keep" if refs else "unverified_not_for_deletion",
                             "references": refs})
    local_candidates = []
    for relative in ("scripts/analyze_method_complementarity.py", "scripts/evaluate_frozen_geoscd_fusion.py",
                     "tests/test_method_complementarity.py", "tests/test_frozen_geoscd_fusion.py"):
        item = repository / relative
        if not item.is_file() or item.is_symlink():
            continue
        try:
            tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", relative],
                                     cwd=repository, capture_output=True, timeout=5).returncode
        except (OSError, subprocess.TimeoutExpired):
            tracked = None
        if tracked == 1:
            local_candidates.append({"path": str(item), "decision": "review_candidate",
                                     "reason": "untracked_legacy_copy_superseded_by_data_analysis",
                                     "logical_bytes": item.stat().st_size})
    for folder in ("facade_change", "data_analysis", "scripts", "tests"):
        directory = repository / folder
        if not directory.is_dir() or directory.is_symlink():
            continue
        for base, dirs, _ in os.walk(directory, followlinks=False):
            dirs[:] = [name for name in dirs if not (Path(base) / name).is_symlink()]
            if "__pycache__" in dirs:
                local_candidates.append({"path": str(Path(base) / "__pycache__"),
                                         "decision": "regenerable_cache", "reason": "Python bytecode cache",
                                         "logical_bytes": None})
                dirs.remove("__pycache__")
    return {"schema_version": 1, "kind": "cleanup_audit", "status": "completed_advisory",
            "repository": str(repository), "created_utc": datetime.now(timezone.utc).isoformat(),
            "read_only": True, "process_inspection_available": available,
            "runs": rows, "local_candidates": local_candidates, "external_inventory": external,
            "external_references": external_references, "unresolved": unresolved,
            "metadata_sha256": metadata_hashes, "warnings": warnings,
            "scope": "Dependency closure from metadata/configs and best-effort live process paths. "
                     "Candidates require review; no assertion that unreferenced scientific evidence is disposable. "
                     "External sources, weights and scripts are never deletion candidates; symlinks are not traversed."}


def write_cleanup_report(out, report):
    write_json(out / "cleanup.json", report)
    with (out / "cleanup.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["decision", "path", "run_status", "logical_bytes", "reasons"])
        writer.writeheader()
        for row in report["runs"]:
            writer.writerow({key: "; ".join(row[key]) if key == "reasons" else row[key]
                             for key in writer.fieldnames})
        for row in report["local_candidates"]:
            writer.writerow({"decision": row["decision"], "path": row["path"], "run_status": "",
                             "logical_bytes": row["logical_bytes"], "reasons": row["reason"]})
    lines = ["Read-only cleanup dependency audit; nothing was moved or deleted."]
    for row in report["runs"]:
        lines.append(f"{row['decision']}: {row['path']} | {', '.join(row['reasons']) or 'no retained metadata reference'}")
    for row in report["local_candidates"]:
        lines.append(f"{row['decision']}: {row['path']} | {row['reason']}")
    candidates = [row for row in report["runs"] if row["decision"] == "review_candidate"]
    lines += [f"Review candidates: {len(candidates)}; logical bytes: "
              f"{sum(row['logical_bytes'] or 0 for row in candidates)}",
              f"Unresolved metadata references/errors: {len(report['unresolved'])}",
              f"External items: {len(report['external_inventory'])}; unverified items are not deletion candidates.",
              report["scope"], *report["warnings"], f"Report: {out / 'cleanup.json'}"]
    text = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(text, encoding="utf-8")
    print(text)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="New report directory; never overwritten")
    parser.add_argument("--run-tests", action="store_true", help="Run the small CPU suite with this interpreter")
    parser.add_argument("--cleanup-plan", action="store_true", help="Read-only run dependency/cleanup inventory; no environment probes")
    parser.add_argument("--keep-run", action="append", default=[], help="Retain this run and all dependencies; repeatable")
    parser.add_argument("--scan-root", action="append", default=[], help="Shallow external inventory, e.g. $HOME; never deletion candidates")
    args = parser.parse_args(argv)
    if not args.cleanup_plan and (args.keep_run or args.scan_root):
        parser.error("--keep-run/--scan-root require --cleanup-plan")
    if args.cleanup_plan and (args.run_tests or not args.keep_run):
        parser.error("--cleanup-plan needs --keep-run and cannot be combined with --run-tests")
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    if args.cleanup_plan:
        report = cleanup_plan(ROOT, args.keep_run, args.scan_root)
        write_cleanup_report(out, report)
        return 0
    packages = {}
    for name in ("numpy", "Pillow", "opencv-python", "opencv-python-headless",
                 "opencv-contrib-python", "opencv-contrib-python-headless", "torch", "kornia"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    source_paths = sorted((ROOT / "facade_change").glob("*.py")) + sorted((ROOT / "tests").glob("*.py")) + [Path(__file__).resolve()]
    report = {"schema_version": 2, "run_id": out.name, "status": "running",
              "started_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
              "executable": sys.executable, "prefix": sys.prefix,
              "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"), "virtual_env": os.environ.get("VIRTUAL_ENV"),
              "python_supported": sys.version_info >= (3, 9), "installed_distributions": packages,
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
    manifest_ready = report["python_supported"] and good("numpy") and good("pillow") and good("project_cpu")
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
    torch_details = report["probes"]["torch"].get("details", {})
    loftr_imports = sift_ready and good("torch") and good("kornia") and bool(torch_details.get("numpy_bridge"))
    report.update(tests=tests, readiness={
        "manifest_imports": manifest_ready, "sift_imports_and_features": sift_ready,
        "cpu_project_tests_passed": tests["status"] == "ok",
        "loftr_import_prerequisites": loftr_imports,
        "loftr_checkpoint_loader_supported": loftr_imports and bool(
            torch_details.get("weights_only") or torch_details.get("trusted_legacy_checkpoint_loader_available")),
        "loftr_default_safe_checkpoint_loader_supported": loftr_imports and bool(torch_details.get("weights_only")),
        "loftr_trusted_legacy_checkpoint_loader_available": loftr_imports and bool(
            torch_details.get("trusted_legacy_checkpoint_loader_available")),
        "loftr_checkpoint_requires_explicit_trust": loftr_imports and not bool(torch_details.get("weights_only")),
        "torch_cuda_available": torch_details.get("cuda_available"),
        "loftr_checkpoint_and_inference": "not_tested"})
    report.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(out / "audit.json", report)
    lines = ["Facade environment audit", f"Environment: {report['conda_environment'] or report['virtual_env'] or 'system/unknown'}",
             f"Python: {report['python']} ({report['executable']})", f"Python >=3.9: {report['python_supported']}",
             "Installed distributions: " + json.dumps(packages, ensure_ascii=False), ""]
    for name, result in report["probes"].items():
        lines.append(f"{name}: {result['status']}")
        if result.get("details"):
            lines.append("  " + json.dumps(result["details"], ensure_ascii=False))
        if result["stderr"].strip():
            lines.append("  " + "\n  ".join(result["stderr"].strip().splitlines()[-8:]))
    cv_version = report["probes"]["opencv"].get("details", {}).get("version")
    cv_packages = {name: version for name, version in packages.items() if name.startswith("opencv-") and version}
    if cv_version and any(not (version == cv_version or version.startswith(cv_version + ".")) for version in cv_packages.values()):
        lines.append("OpenCV metadata/runtime mismatch: " + json.dumps(cv_packages) + " vs imported cv2=" + cv_version
                     + ". Imports/tests determine functionality; do not reinstall automatically.")
    if good("torch") and not torch_details.get("weights_only"):
        lines.append("Torch lacks weights_only: default safe checkpoint loading is unavailable; "
                     "the adapter supports explicitly trusted weights with --trust-checkpoint. "
                     "No checkpoint was loaded by this audit.")
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
