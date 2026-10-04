"""Small I/O and provenance boundary; no model imports or automatic downloads."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def new_directory(path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    return out


def load_rgb(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Decoded native grid, uint8 RGB, boolean fully opaque support.

    Do not apply EXIF rotation: COCO dimensions refer to the stored pixel grid.
    Do not silently convert palettes, grayscale or high-bit-depth data.
    """
    with Path(path).open("rb") as handle:
        header = handle.read(25)
    if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 25 and header[24] != 8:
        raise ValueError(f"Expected 8-bit PNG: {path}")
    with Image.open(path) as image:
        if image.format == "TIFF" and any(v != 8 for v in image.tag_v2.get(258, (8,))):
            raise ValueError(f"Expected 8-bit TIFF: {path}")
        if image.mode not in {"RGB", "RGBA"}:
            raise ValueError(f"Expected 8-bit RGB/RGBA, got {image.mode}: {path}")
        array = np.array(image)
    if array.dtype != np.uint8:
        raise ValueError(f"Expected uint8: {path}")
    support = array[..., 3] == 255 if array.shape[2] == 4 else np.ones(array.shape[:2], bool)
    return np.ascontiguousarray(array[..., :3]), support


def environment() -> dict:
    versions = {}
    for name in ("numpy", "Pillow", "opencv-python-headless", "opencv-python", "torch", "torchvision",
                 "kornia", "lpips", "safetensors", "einops", "scikit-image"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    package = Path(__file__).resolve().parent
    source = {p.relative_to(package).as_posix(): sha256(p) for p in sorted(package.rglob("*.py"))}
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "packages": versions, "source_sha256": source}
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "-C", str(package.parent), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
        result["git_dirty"] = bool(subprocess.check_output(
            ["git", "-C", str(package.parent), "status", "--porcelain"],
            stderr=subprocess.DEVNULL, text=True, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        result["git_commit"] = None
    return result


def run_record(kind: str, config: dict) -> dict:
    serialized = json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return {"schema_version": 1, "kind": kind, "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(), "config": config,
            "config_sha256": hashlib.sha256(serialized).hexdigest(), "environment": environment()}


def finish_record(out: Path, record: dict, status: str, error: str | None = None) -> None:
    record.update(status=status, finished_utc=datetime.now(timezone.utc).isoformat(),
                  run_id=out.name)
    if error:
        record["error"] = error
    record["artifact_sha256"] = {str(p.relative_to(out)): sha256(p)
                                 for p in sorted(out.rglob("*"))
                                 if p.is_file() and p != out / "run.json"}
    write_json(out / "run.json", record)
