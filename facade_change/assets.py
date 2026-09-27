"""Bounded inventory of explicitly named source repositories.

Config keys: ``asset_roots`` is a list of objects with ``name``, ``path``,
optional ``max_depth`` (default 4) and ``max_entries`` (default 20000).
``asset_metadata`` optionally declares ``path``, ``training_groups`` and
``training_groups_source`` for individual files. Paths resolve from the config.
Only small code/config files are hashed for exact duplicate candidates.
Checkpoints and logs are never opened. No asset is executed, decoded or copied.
"""
from __future__ import annotations

import hashlib
import os
from collections import Counter, defaultdict
from pathlib import Path

from .io import finish_record, new_directory, read_json, run_record, write_json


PRUNED_NAMES = {
    "__pycache__", "node_modules", "venv", "env", "site-packages", "cache",
    "caches", "data", "dataset", "datasets", "datasets_mine", "images",
    "imgs", "masks", "tiles", "crops", "previews", "gallery", "galleries",
}
EXPECTED_ROLES = ("code", "config", "checkpoint", "training_log")
MAX_HASH_FILE_BYTES = 2 * 1024 * 1024
MAX_HASH_TOTAL_BYTES = 64 * 1024 * 1024


def candidate_role(path: Path) -> str | None:
    name, suffix = path.name.lower(), path.suffix.lower()
    if suffix in {".pt", ".pth", ".ckpt", ".safetensors"} or name.endswith(".pth.tar"):
        return "checkpoint"
    if suffix in {".py", ".sh", ".ipynb"}:
        return "code"
    if suffix == ".log" or name.startswith("events.out.tfevents"):
        return "training_log"
    if suffix in {".csv", ".json", ".jsonl", ".txt"} and any(
        word in name for word in ("train", "history", "metrics", "loss", "log")
    ):
        return "training_log"
    if suffix in {".yaml", ".yml", ".toml", ".ini", ".cfg", ".json"}:
        return "config"
    if name.startswith("requirements") and suffix == ".txt":
        return "config"
    return None


def _path(value, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _scan(root: Path, max_depth: int, max_entries: int) -> tuple[list[Path], dict]:
    status = {"status": "scanned", "entries_seen": 0, "pruned_directories": 0,
              "depth_limited_directories": 0, "symlinks_skipped": 0, "errors": []}
    if root.is_symlink():
        status.update(status="symlink_skipped", symlinks_skipped=1)
        return [], status
    if not root.exists():
        status["status"] = "missing"
        return [], status
    if root.is_file():
        status["entries_seen"] = 1
        return [root], status
    found = []
    pending = [(root, 0)]
    while pending:
        directory, depth = pending.pop()
        try:
            with os.scandir(directory) as stream:
                for entry in stream:
                    status["entries_seen"] += 1
                    if status["entries_seen"] > max_entries:
                        status["status"] = "entry_limit_reached"
                        return sorted(found), status
                    if entry.is_symlink():
                        status["symlinks_skipped"] += 1
                        continue
                    if entry.name.startswith("."):
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        name = entry.name.lower()
                        if name in PRUNED_NAMES or name.startswith(("images_", "dataset_")):
                            status["pruned_directories"] += 1
                        elif depth >= max_depth:
                            status["depth_limited_directories"] += 1
                        else:
                            pending.append((Path(entry.path), depth + 1))
                    elif entry.is_file(follow_symlinks=False) and candidate_role(Path(entry.path)):
                        found.append(Path(entry.path))
        except OSError as exc:
            status["errors"].append({"path": str(directory), "error": str(exc)})
    if status["errors"]:
        status["status"] = "scanned_with_errors"
    return sorted(found), status


def inventory_assets(config_path: str | Path, out: str | Path) -> dict:
    """Record candidates and gaps within configured scan bounds, preserving originals."""
    config_path = Path(config_path).expanduser().resolve()
    config = read_json(config_path)
    roots = config.get("asset_roots", [])
    if not isinstance(roots, list):
        raise ValueError("asset_roots must be a list of {name, path} objects")
    resolved_roots = []
    for item in roots:
        if not isinstance(item, dict) or not item.get("path"):
            raise ValueError("Each asset_roots entry requires a path")
        root = _path(item["path"], config_path.parent).absolute()
        if root in {Path("/"), Path("/home"), Path("/mnt"), Path.home()}:
            raise ValueError(f"Use a specific source repository or checkpoint directory: {root}")
        max_depth, max_entries = item.get("max_depth", 4), item.get("max_entries", 20000)
        if (type(max_depth) is not int or not 0 <= max_depth <= 12
                or type(max_entries) is not int or not 1 <= max_entries <= 100000):
            raise ValueError("asset root max_depth must be 0..12 and max_entries 1..100000")
        resolved_roots.append({"name": item.get("name", root.name), "path": str(root),
                               "max_depth": max_depth, "max_entries": max_entries})
    declared = {}
    for item in config.get("asset_metadata", []):
        if not isinstance(item, dict) or not item.get("path"):
            raise ValueError("asset_metadata entries require path")
        groups = item.get("training_groups")
        if groups is not None and (not isinstance(groups, list) or not all(
            isinstance(group, str) and group.strip() for group in groups
        )):
            raise ValueError("training_groups must be an explicit list of nonempty group IDs")
        source = item.get("training_groups_source")
        if groups is not None and (not isinstance(source, str) or not source.strip()):
            raise ValueError("Declared training_groups require training_groups_source")
        declared[str(_path(item["path"], config_path.parent).absolute())] = item
    out = new_directory(out)
    record = run_record("asset_inventory", {"asset_roots": resolved_roots,
                                             "asset_metadata": list(declared.values()),
                                             "max_hash_file_bytes": MAX_HASH_FILE_BYTES,
                                             "max_hash_total_bytes": MAX_HASH_TOTAL_BYTES})
    write_json(out / "run.json", record)
    try:
        assets, root_reports = [], []
        seen = set()
        hashed_bytes = 0
        for item in resolved_roots:
            root = Path(item["path"])
            paths, status = _scan(root, item["max_depth"], item["max_entries"])
            root_reports.append({**item, **status})
            for path in paths:
                role = candidate_role(path)
                key = str(path.absolute())
                if key in seen or role is None:
                    continue
                try:
                    size = path.stat().st_size
                except OSError as exc:
                    status["errors"].append({"path": key, "error": str(exc)})
                    root_reports[-1]["status"] = "scanned_with_errors"
                    continue
                seen.add(key)
                metadata = declared.get(key, {})
                groups = metadata.get("training_groups")
                digest, hash_status = None, "role_excluded"
                if role in {"code", "config"}:
                    hash_status = "file_size_limit" if size > MAX_HASH_FILE_BYTES else "total_byte_limit"
                    if size <= MAX_HASH_FILE_BYTES and hashed_bytes + size <= MAX_HASH_TOTAL_BYTES:
                        try:
                            with path.open("rb") as stream:
                                content = stream.read(size + 1)
                            hashed_bytes += len(content)
                            hash_status = "file_changed_during_scan"
                            if len(content) == size:
                                digest = hashlib.sha256(content).hexdigest()
                                hash_status = "hashed"
                        except OSError as exc:
                            hash_status = "read_error"
                            status["errors"].append({"path": key, "error": str(exc)})
                            root_reports[-1]["status"] = "scanned_with_errors"
                assets.append({"path": key, "source_root": item["name"],
                               "size_bytes": size, "candidate_role": role,
                               "training_groups": groups,
                               "training_groups_status": "declared" if groups is not None else "unknown",
                               "training_groups_source": metadata.get("training_groups_source"),
                               "content_sha256": digest, "hash_status": hash_status,
                               "compatibility": "not_tested"})
        assets.sort(key=lambda item: item["path"])
        hashes = defaultdict(list)
        for row in assets:
            if row["content_sha256"] is not None:
                hashes[row["content_sha256"]].append(row)
        duplicates = [{"sha256": digest, "size_bytes": rows[0]["size_bytes"],
                       "paths": [row["path"] for row in rows]}
                      for digest, rows in sorted(hashes.items()) if len(rows) > 1]
        counts = Counter(item["candidate_role"] for item in assets)
        summary = {"asset_count": len(assets), "candidate_roles": dict(counts),
                   "roles_not_found_in_scan": [role for role in EXPECTED_ROLES if not counts[role]],
                   "roots_not_scanned_completely": sum(
                       row["status"] != "scanned" or bool(row["depth_limited_directories"])
                       for row in root_reports),
                   "training_groups_unknown": sum(row["training_groups"] is None for row in assets),
                   "duplicate_group_count": len(duplicates),
                   "duplicate_file_count": sum(len(row["paths"]) for row in duplicates),
                   "duplicate_extra_file_count": sum(len(row["paths"]) - 1 for row in duplicates),
                   "script_config_hashed_count": sum(row["hash_status"] == "hashed" for row in assets),
                   "hash_bytes_read": hashed_bytes}
        report = {"schema_version": 1, "roots": root_reports, "assets": assets, "summary": summary,
                  "duplicate_candidates": duplicates,
                  "scope": "Metadata and filename role candidates do not confirm compatibility. "
                           "Gaps refer only to the bounded scan, not to the whole computer. "
                           "Only code/config files up to 2 MiB are hashed, within a 64 MiB total budget. "
                           "No checkpoints or logs are opened; no source assets are executed, copied or modified.",
                  "pruned_directory_names": sorted(PRUNED_NAMES)}
        write_json(out / "assets.json", report)
        lines = [f"Asset candidates: {len(assets)}", f"Candidate roles: {dict(counts)}",
                 f"Exact duplicate candidate groups: {len(duplicates)} "
                 f"({summary['duplicate_file_count']} files)",
                 "Roles not found within scan: " + ", ".join(summary["roles_not_found_in_scan"]),
                 *[f"{row['name']}: {row['status']} ({row['path']})" for row in root_reports],
                 report["scope"]]
        (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        finish_record(out, record, "completed_inventory")
        return report
    except Exception as exc:
        finish_record(out, record, "failed", str(exc))
        raise
