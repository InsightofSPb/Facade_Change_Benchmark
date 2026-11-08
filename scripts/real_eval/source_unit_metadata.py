"""Recover omitted unit metadata only from the snapshot's verified JSON markers."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

from .robustness_common import safe_path


@lru_cache(maxsize=32768)
def _verified_marker(path, expected_sha256, file_signature):
    # Stat fields are part of the key: changed files must be read and verified
    # again. Repeated method preflights can share unchanged marker reads.
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"Изменился исходный marker относительно снимка: {path}")
    record = json.loads(payload)
    if not isinstance(record, dict):
        raise ValueError(f"Исходный marker должен быть объектом: {path}")
    return record


def source_rows(manifest):
    """Read snapshot rows; restore stripped provenance without inventing it.

    Older snapshots without per-marker hashes keep their original rows. A caller
    may use a separately verified inference provenance file, but must not treat
    those rows as having verified metadata for cache reuse.
    """
    rows = manifest.get("source_records", [])
    if isinstance(rows, dict):
        rows = list(rows.values())
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("source_records должен содержать сохранённые метаданные единиц")
    summary = manifest.get("source_summary", {})
    if not isinstance(summary, dict):
        raise ValueError("source_summary должен быть объектом")
    hashes = summary.get("source_json_sha256", {})
    if not isinstance(hashes, dict):
        raise ValueError("source_json_sha256 должен быть объектом")
    restored = []
    for row in rows:
        if "method_metadata" in row or not hashes:
            restored.append(row)
            continue
        identifier = row.get("unit_id")
        relative = f"units/{identifier}.json"
        expected = hashes.get(relative)
        if expected is None:
            restored.append(row)
            continue
        if (not isinstance(identifier, str) or not identifier
                or Path(identifier).name != identifier or identifier in {".", ".."}):
            raise ValueError("Некорректный unit_id исходного marker")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"Некорректный SHA-256 исходного marker: {relative}")
        source = manifest.get("source_run")
        if not isinstance(source, str) or not source:
            raise ValueError("Для проверки исходного marker требуется source_run")
        path = safe_path(source, relative)
        info = path.stat()
        signature = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        original = _verified_marker(str(path), expected, signature)
        projected = {key: value for key, value in original.items() if key != "method_metadata"}
        if projected != row:
            raise ValueError(f"Исходный marker расходится с записью снимка: {path}")
        if "method_metadata" not in original:
            restored.append(row)
            continue
        metadata = original["method_metadata"]
        if not isinstance(metadata, dict):
            raise ValueError(f"Некорректные метаданные модели в исходном marker: {path}")
        restored.append({**row, "method_metadata": metadata})
    return restored
