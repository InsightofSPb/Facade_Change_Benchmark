#!/usr/bin/env python
"""Create local runner configs from existing sources, environments and completed runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from facade_change.benchmark_results import ReuseResults


def _read(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a JSON object: {path}")
    return value


def _absolute(value, directory):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (directory / path).resolve()


def _local_paths(value, repo_root, owner_home):
    if isinstance(value, dict):
        return {key: _local_paths(item, repo_root, owner_home) for key, item in value.items()}
    if isinstance(value, list):
        return [_local_paths(item, repo_root, owner_home) for item in value]
    if isinstance(value, str) and value.startswith("/home/sasha/"):
        suffix = Path(value).relative_to("/home/sasha")
        if suffix.parts[0] == "Facade_Change_Benchmark":
            return str(repo_root.joinpath(*suffix.parts[1:]))
        return str(owner_home / suffix)
    return value


def _vggt_checkpoint(repo_root, geoscd_root):
    preferred = repo_root / "runs/2026-10-04-geoscd-full-smoke/run.json"
    records = [preferred]
    other_records = list((repo_root / "runs").glob("*/run.json"))
    other_records.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    records.extend(path for path in other_records if path != preferred)
    attempted = []
    for path in records:
        if not path.is_file():
            continue
        try:
            record = _read(path)
        except (ValueError, OSError):
            continue
        if record.get("kind") not in {"geoscd_full_batch", "geoscd_full_pair"}:
            continue
        value = record.get("config", {}).get("checkpoint")
        if not isinstance(value, str) or not value:
            continue
        candidate = _absolute(value, path.parent)
        attempted.append(candidate)
        if candidate.is_file():
            return candidate, str(path)
    for candidate in (repo_root / "models/vggt_1b.pt", geoscd_root / "src/pretrained/model.pt"):
        attempted.append(candidate)
        if candidate.is_file():
            return candidate.resolve(), "existing checkpoint fallback"
    names = "\n".join(f"  {path}" for path in dict.fromkeys(attempted))
    raise FileNotFoundError("Local VGGT checkpoint not found. No weights were downloaded. Checked:\n" + names)


def _validate_method_paths(methods):
    source_files = {
        "dinov2": ["hubconf.py", "dinov2/models/vision_transformer.py"],
        "rscd_cmu": ["src/robust_scene_change_detect/models/CD_model.py"],
        "rscd_diff_cmu": ["src/robust_scene_change_detect/models/CD_model.py"],
        "rscd_pscd": ["src/robust_scene_change_detect/models/CD_model.py"],
        "anychange": ["torchange/models/segment_any_change/anychange.py"],
        "geoscd": ["src/pixel_match.py", "src/flow_cd.py", "src/segment_anything_model/build_sam.py"],
    }
    missing = []
    checked = set()
    file_keys = {"worker_python", "checkpoint_path", "dino_checkpoint", "sam_checkpoint",
                 "backbone_checkpoint", "calibration_checkpoint"}
    root_keys = {"source_root", "dino_root", "py_utils_root"}
    required = {"lpips": {"worker_python"},
                "dinov2": {"worker_python", "source_root", "checkpoint_path"},
                "anychange": {"worker_python", "source_root", "checkpoint_path"},
                "geoscd": {"worker_python", "source_root", "checkpoint_path", "sam_checkpoint"}}
    for method in ("rscd_cmu", "rscd_diff_cmu", "rscd_pscd"):
        required[method] = {"worker_python", "source_root", "checkpoint_path", "dino_root", "dino_checkpoint", "py_utils_root"}
    for method, options in methods.items():
        if not isinstance(options, dict):
            raise ValueError(f"Method options must be an object: {method}")
        for key in required.get(method, set()):
            if not options.get(key):
                missing.append(f"{method}.{key}: required path is absent")
        for key, value in options.items():
            if key not in file_keys | root_keys or not value:
                continue
            path = Path(value).expanduser().resolve()
            if key in root_keys:
                exists = path.is_dir()
            else:
                exists = path.is_file() and (key != "worker_python" or os.access(path, os.X_OK))
            if not exists:
                missing.append(f"{method}.{key}: {path}")
            checked.add(str(path))
        if options.get("source_root"):
            for relative in source_files.get(method, []):
                path = Path(options["source_root"]) / relative
                if not path.is_file():
                    missing.append(f"{method} source: {path}")
                checked.add(str(path))
        if options.get("dino_root"):
            path = Path(options["dino_root"]) / "hubconf.py"
            if not path.is_file():
                missing.append(f"{method} DINO source: {path}")
            checked.add(str(path))
        if options.get("py_utils_root"):
            path = Path(options["py_utils_root"]) / "src/py_utils/utils_torch.py"
            if not path.is_file():
                missing.append(f"{method} utility source: {path}")
            checked.add(str(path))
    if missing:
        raise FileNotFoundError("Missing local inference files; no configuration was written:\n  " + "\n  ".join(missing))
    return len(checked)


def configure(repo_root=REPO_ROOT, reuse_run=None, out=None, dry_run=False, *, owner_home=None):
    """Validate first, then exclusively create absent configs; preserve existing files."""
    repo_root = Path(repo_root).expanduser().resolve()
    owner_home = Path(owner_home).expanduser().resolve() if owner_home is not None else Path.home()
    reuse_run = _absolute(reuse_run or "runs/2026-10-04-compression-quick-001", repo_root)
    out = _absolute(out or "runs/2026-10-04-all-methods-001", repo_root)
    config_dir = repo_root / "configs"
    methods_target, benchmark_target = config_dir / "methods.local.json", config_dir / "benchmark.local.json"
    existing_benchmark = _read(benchmark_target) if benchmark_target.exists() else None
    active_methods_target = methods_target
    if existing_benchmark is not None:
        if not existing_benchmark.get("reuse_run") or not existing_benchmark.get("out"):
            raise ValueError(f"Existing benchmark config requires reuse_run and out: {benchmark_target}")
        reuse_run = _absolute(existing_benchmark["reuse_run"], config_dir)
        out = _absolute(existing_benchmark["out"], config_dir)
        if existing_benchmark.get("methods_config"):
            active_methods_target = _absolute(existing_benchmark["methods_config"], config_dir)
    if active_methods_target != methods_target and not active_methods_target.is_file():
        raise FileNotFoundError(f"Existing benchmark references a missing methods config: {active_methods_target}")
    # This checks completed status, metrics, selection, thresholds and every saved map/hash.
    # It never loads a model or checkpoint.
    reuse = ReuseResults(reuse_run)
    dataset_value = reuse.record.get("config", {}).get("dataset_run")
    if not isinstance(dataset_value, str) or not dataset_value:
        raise ValueError("Completed reuse run has no recorded dataset_run path")
    dataset_run = _absolute(dataset_value, reuse_run)
    if not (dataset_run / "run.json").is_file():
        raise FileNotFoundError(f"Recorded parent dataset is missing: {dataset_run}")

    existing_methods = active_methods_target.exists()
    methods = (_read(active_methods_target) if existing_methods else
               _local_paths(_read(config_dir / "methods.example.json"), repo_root, owner_home))
    if set(methods) != {"methods"} or not isinstance(methods["methods"], dict):
        raise ValueError("Methods template must contain one 'methods' object")
    if existing_methods:
        path_keys = {"worker_python", "source_root", "checkpoint_path", "dino_root", "dino_checkpoint",
                     "py_utils_root", "sam_checkpoint", "backbone_checkpoint", "calibration_checkpoint"}
        for options in methods["methods"].values():
            if not isinstance(options, dict):
                raise ValueError(f"Existing methods config has invalid options: {active_methods_target}")
            for key in path_keys & set(options):
                if options[key]:
                    options[key] = str(_absolute(options[key], active_methods_target.parent))
        geoscd = methods["methods"].get("geoscd", {})
        checkpoint = (Path(geoscd["checkpoint_path"]) if geoscd.get("checkpoint_path") else None)
        origin = str(active_methods_target)
    else:
        geoscd = methods["methods"]["geoscd"]
        checkpoint, origin = _vggt_checkpoint(repo_root, Path(geoscd["source_root"]))
        geoscd["checkpoint_path"] = str(checkpoint)
    checked = _validate_method_paths(methods["methods"])

    benchmark = existing_benchmark if existing_benchmark is not None else _read(config_dir / "benchmark.example.json")
    if existing_benchmark is None:
        benchmark.update(dataset_run=str(dataset_run), out=str(out), reuse_run=str(reuse_run),
                         selection_path=str(reuse_run / "selection.json"), methods_config=str(active_methods_target))
    else:
        if not benchmark.get("dataset_run") or _absolute(benchmark["dataset_run"], config_dir) != dataset_run:
            raise ValueError("Existing benchmark dataset_run disagrees with its completed reuse run")
        selection_path = _absolute(benchmark.get("selection_path", str(reuse_run / "selection.json")), config_dir)
        if (hashlib.sha256(selection_path.read_bytes()).digest()
                != hashlib.sha256((reuse_run / "selection.json").read_bytes()).digest()):
            raise ValueError("Existing benchmark selection differs from its completed reuse run")
    if not isinstance(benchmark.get("methods"), list) or set(benchmark["methods"]) - set(methods["methods"]):
        raise ValueError("Benchmark selects a method missing from the methods config")
    result = {"dry_run": bool(dry_run), "reuse_run": str(reuse_run), "dataset_run": str(dataset_run),
              "out": str(out), "vggt_checkpoint": str(checkpoint) if checkpoint else None, "vggt_source": origin,
              "checked_local_paths": checked, "created": [], "preserved": [], "would_create": []}
    for target, content in ((active_methods_target, methods), (benchmark_target, benchmark)):
        if target.exists():
            result["preserved"].append(str(target))
            continue
        if dry_run:
            result["would_create"].append(str(target))
            continue
        # Exclusive creation also protects user changes made between validation and writing.
        with target.open("x", encoding="utf-8") as stream:
            json.dump(content, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        result["created"].append(str(target))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Existing facade repository checkout")
    parser.add_argument("--reuse-run", default="runs/2026-10-04-compression-quick-001",
                        help="Complete previous run; relative to repo-root")
    parser.add_argument("--out", default="runs/2026-10-04-all-methods-001",
                        help="New benchmark output; relative to repo-root")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing configs")
    args = parser.parse_args(argv)
    try:
        result = configure(**vars(args))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["preserved"]:
        print("Existing local configs were preserved; command-line paths apply only to newly created configs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
