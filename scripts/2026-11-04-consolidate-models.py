#!/usr/bin/env python
"""Copy verified frozen inference weights to models; preserve source experiments."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from facade_change.io import read_json, sha256
from facade_change.methods.anychange import _source_provenance as anychange_sources
from facade_change.methods.dinov2 import _source_provenance as dino_sources
from facade_change.geoscd import source_provenance as geoscd_sources
from scripts.real_eval.night_common import DEFAULT_METHODS, load_prepared
from scripts.real_eval.robustness_common import digest
from scripts.real_eval.robustness_run import _file_hashes, _recorded_method_contract


def _contracts(source_run=None, prepared=None):
    if prepared is not None:
        root, manifest = load_prepared(prepared)
        source = Path(manifest["source_run"])
        return root / "manifest.json", {
            method: _recorded_method_contract(method, manifest, source, None) for method in DEFAULT_METHODS}
    path = Path(source_run).expanduser().resolve() / "run.json"
    record = read_json(path)
    contract = record.get("contract", {})
    if (record.get("kind") != "real_temporal_run"
            or record.get("status") not in {"completed", "completed_with_failures"}
            or record.get("config_hash") != digest(contract)):
        raise ValueError("--source-run must have a completed, hash-bound real_temporal_run contract")
    models = contract.get("models", {})
    for method in DEFAULT_METHODS:
        if (not isinstance(models.get(method), dict)
                or not isinstance(models[method].get("options"), dict)
                or not isinstance(models[method].get("expected_metadata"), dict)):
            raise ValueError(f"Source run has no frozen model contract for {method}")
        recorded = models[method].get("model_files_sha256", {})
        for filename, checksum in _file_hashes(models[method]["expected_metadata"]).items():
            if filename in recorded and recorded[filename] != checksum:
                raise ValueError(f"Conflicting model hashes: {method}: {filename}")
    return path, models


def _exclusive_json(path, value):
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Existing file differs; choose a new output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def _copy_weight(source, target, checksum):
    if target.exists():
        if not target.is_file() or sha256(target) != checksum:
            raise ValueError(f"Existing model differs; not overwriting: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix="2026-11-04-copy-", suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
            with source.open("rb") as reader:
                shutil.copyfileobj(reader, stream, length=1024 * 1024)
            stream.flush()
            os.fsync(stream.fileno())
        if sha256(temporary) != checksum:
            raise ValueError(f"Copied weights differ from recorded hash: {source}")
        # Atomic publication without replacing any existing model file.
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def consolidate(source_run, models, anychange_source, out_config, device="cuda:0", *, prepared=None,
                dino_search_root=None):
    source_record, contracts = _contracts(source_run, prepared)
    source_hash = sha256(source_record)
    provenance_files = {}
    for contract in contracts.values():
        for path, checksum in contract.get("provenance_files_sha256", {}).items():
            if path in provenance_files and provenance_files[path] != checksum:
                raise ValueError(f"Conflicting source provenance hashes: {path}")
            if sha256(path) != checksum:
                raise ValueError(f"Source provenance changed: {path}")
            provenance_files[path] = checksum
    root = Path(models).expanduser().resolve()
    output = Path(out_config).expanduser().resolve()
    report_path = root / "2026-11-04-consolidation.json"
    options = {method: deepcopy(contracts[method]["options"]) for method in DEFAULT_METHODS}
    metadata = {method: contracts[method]["expected_metadata"] for method in DEFAULT_METHODS}
    plans, relocations, source_audits, issues, hashes = {}, {}, {}, [], {}

    def checked_hash(path):
        info = path.stat()
        key = (str(path), info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if key not in hashes:
            hashes[key] = sha256(path)
        return hashes[key]

    def weight(method, key, entry, relative):
        try:
            original, expected = Path(entry["path"]).expanduser().resolve(), entry["sha256"]
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError("Missing recorded SHA-256")
            target = root / relative
            candidates = [target, original]
            if root.is_dir():
                candidates.extend(sorted(root.rglob(original.name)))
            if original.name == "sam_vit_h_4b8939.pth":
                candidates.append(Path.home() / "LPOSS/weights/sam" / original.name)
            if target.exists() and (not target.is_file() or checked_hash(target) != expected):
                raise ValueError(f"Existing destination differs: {target}")
            matched = next((path for path in dict.fromkeys(candidates)
                            if path.is_file() and not path.name.endswith(".part") and checked_hash(path) == expected), None)
            if matched is None:
                raise ValueError(f"No matching weights for SHA256 {expected}; checked: "
                                 + ", ".join(str(path) for path in dict.fromkeys(candidates)))
            if target in plans and plans[target]["sha256"] != expected:
                raise ValueError(f"Consumers require different weights at {target}")
            plans[target] = dict(source=matched, sha256=expected)
            relocations[str(original)] = dict(path=str(target), sha256=expected)
            options[method][key] = str(target)
            print(f"OK {method}.{key}: {matched} -> {target}", flush=True)
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(f"{method}.{key}: {error}")

    for role in ("backbone", "calibration"):
        weight("lpips", role + "_checkpoint", metadata["lpips"]["weights"][role],
               f"lpips/2026-11-04-{role}.pth")
    weight("dinov2", "checkpoint_path", metadata["dinov2"]["checkpoint"], "dinov2/dinov2_vits14_pretrain.pth")
    semantic = metadata["semantic_dino"]
    backbone = semantic["backbone_provenance"]["checkpoint"]
    same_dino = backbone["sha256"] == metadata["dinov2"]["checkpoint"]["sha256"]
    weight("semantic_dino", "backbone_checkpoint", backbone,
           "dinov2/dinov2_vits14_pretrain.pth" if same_dino else "semantic_dino/2026-11-04-backbone.pth")
    weight("semantic_dino", "head_checkpoint", semantic["head_checkpoint"], "semantic_dino/2026-11-04-best.pth")
    weight("anychange", "checkpoint_path", metadata["anychange"]["checkpoint"], "sam_vit_h_4b8939.pth")
    for role, key, destination in (("vggt", "checkpoint_path", "vggt_1b.pt"),
                                    ("sam", "sam_checkpoint", "sam_vit_h_4b8939.pth")):
        weight("geoscd", key, metadata["geoscd"]["checkpoints"][role], destination)
    for representation in ("abs", "mod256"):
        method = "msdzip_" + representation
        entry = dict(path=metadata[method]["checkpoint_path"], sha256=metadata[method]["checkpoint_sha256"])
        weight(method, "checkpoint_path", entry, f"msdzip/2026-11-04-{representation}.pt")

    discovered = []
    if dino_search_root is not None:
        search_root = Path(dino_search_root).expanduser().resolve()
        if not search_root.is_dir():
            raise ValueError(f"DINO source search directory missing: {search_root}")
        print(f"Searching DINO source trees in {search_root}", flush=True)
        scan = subprocess.run(["rg", "--files", "--hidden", "--no-ignore", "--follow", str(search_root),
                               "-g", "hubconf.py", "-g", "!**/.git/**"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if scan.stderr:
            print(scan.stderr, file=sys.stderr, end="", flush=True)
        if scan.returncode not in (0, 1, 2):
            raise ValueError(f"DINO source search failed with exit code {scan.returncode}")
        discovered = sorted({Path(line).resolve().parent for line in scan.stdout.splitlines()
                             if (Path(line).parent / "dinov2/models/vision_transformer.py").is_file()})
        print(f"Found {len(discovered)} DINO source candidates", flush=True)
    source_cache = {}
    for method, expected in (("dinov2", metadata["dinov2"]["sources"]),
                             ("semantic_dino", semantic["backbone_provenance"]["sources"])):
        print(f"EXPECTED {method}: {expected['python_tree_sha256']} "
              f"({expected.get('python_file_count', '?')} Python files)", flush=True)
        candidates = [root / "dinov2/source", Path(expected["path"]), *discovered]
        matched = None
        for candidate in dict.fromkeys(path.resolve() for path in candidates):
            if not candidate.is_dir():
                print(f"MISSING {method} source: {candidate}", flush=True)
                continue
            try:
                if candidate not in source_cache:
                    source_cache[candidate] = dino_sources(candidate)
                actual = source_cache[candidate]
            except (OSError, ValueError) as error:
                print(f"UNREADABLE {method} source: {candidate}: {error}", flush=True)
                continue
            same = actual["python_tree_sha256"] == expected["python_tree_sha256"]
            print(f"{'MATCH' if same else 'DIFFERENT'} {candidate}: "
                  f"{actual['python_tree_sha256']} ({actual['python_file_count']} Python files)", flush=True)
            if same:
                matched = candidate
                break
        if matched is None:
            issues.append(f"{method}: no DINO source tree matching the recorded Python hash")
        else:
            options[method]["source_root"] = str(matched)
            source_audits[method] = dict(path=str(matched), python_tree_sha256=expected["python_tree_sha256"])

    try:
        candidate = Path(anychange_source).expanduser().resolve()
        actual = anychange_sources(candidate)
        expected = metadata["anychange"]["source"]
        old_files = {row["relative_path"]: row["sha256"] for row in expected["files"]}
        new_files = {row["relative_path"]: row["sha256"] for row in actual["files"]}
        old_python = {name: checksum for name, checksum in old_files.items() if name.endswith(".py")}
        new_python = {name: checksum for name, checksum in new_files.items() if name.endswith(".py")}
        if not old_python or old_python != new_python:
            changed = sorted(name for name in old_python.keys() | new_python.keys()
                             if old_python.get(name) != new_python.get(name))
            raise ValueError("AnyChange Python sources differ: " + ", ".join(changed))
        packaging = sorted(name for name in old_files.keys() | new_files.keys()
                           if not name.endswith(".py") and old_files.get(name) != new_files.get(name))
        options["anychange"]["source_root"] = str(candidate)
        source_audits["anychange"] = dict(path=str(candidate), python_files_match=True,
            original_manifest_sha256=expected["manifest_sha256"], actual_manifest_sha256=actual["manifest_sha256"],
            packaging_differences=packaging)
        print(f"OK AnyChange: {len(old_python)} matching Python files; packaging differences: {packaging}", flush=True)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        issues.append(str(error))

    try:
        expected = metadata["geoscd"]["source"]
        actual = geoscd_sources(options["geoscd"]["source_root"])
        if actual["source_sha256"] != expected["source_sha256"]:
            raise ValueError("GeoSCD source files differ from the recorded experiment")
        source_audits["geoscd"] = actual
        for row in metadata["lpips"]["package"]["source_files"]:
            if checked_hash(Path(row["path"])) != row["sha256"]:
                raise ValueError(f"LPIPS package source differs: {row['path']}")
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        issues.append(str(error))

    for method, settings in options.items():
        worker = settings.get("worker_python")
        if worker and (not Path(worker).is_file() or not os.access(worker, os.X_OK)):
            issues.append(f"{method}: worker Python missing or not executable: {worker}")
        if method not in {"rgb_diff", "ssim"}:
            settings["device"] = device
        if any(settings.get(key) for key in ("download", "download_weights", "train", "auto_train")):
            issues.append(f"{method}: training/download convenience flags must be disabled")
    config = {"methods": options}
    if output.exists() and read_json(output) != config:
        issues.append(f"Output config already exists with different contents: {output}")
    if issues:
        raise ValueError("Consolidation stopped before copying or writing config:\n  " + "\n  ".join(issues))

    report = dict(schema_version=1, kind="frozen_model_consolidation", source_record=str(source_record),
        source_record_sha256=source_hash, config_path=str(output), config_hash=digest(config),
        methods=list(DEFAULT_METHODS), weight_relocations=relocations, source_audits=source_audits,
        original_contract_hashes={method: digest(contract) for method, contract in contracts.items()},
        provenance_files_sha256=provenance_files,
        source_files_removed=False, training_started=False,
        scope="Frozen inference weights; original semantic training records and worker environments remain required")
    if report_path.exists() and read_json(report_path) != report:
        raise ValueError(f"Existing consolidation record differs: {report_path}")
    for target, entry in plans.items():
        _copy_weight(entry["source"], target, entry["sha256"])
    if sha256(source_record) != source_hash:
        raise ValueError("Source experiment record changed during consolidation")
    for path, checksum in provenance_files.items():
        if sha256(path) != checksum:
            raise ValueError(f"Source provenance changed during consolidation: {path}")
    _exclusive_json(report_path, report)
    _exclusive_json(output, config)
    print(f"Saved config: {output}\nSaved audit: {report_path}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-run", help="Completed night run with frozen model contracts")
    source.add_argument("--prepared", help="Prepared histories used only to resolve original model provenance")
    parser.add_argument("--models", default="models")
    parser.add_argument("--anychange-source", required=True,
                        help="Root containing torchange/models/segment_any_change")
    parser.add_argument("--out-config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dino-search-root", help="Search local hubconf.py trees and accept only exact recorded source hashes")
    args = parser.parse_args()
    try:
        consolidate(args.source_run, args.models, args.anychange_source, args.out_config, args.device,
                    prepared=args.prepared, dino_search_root=args.dino_search_root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Model consolidation failed: {error}\n")


if __name__ == "__main__":
    main()
