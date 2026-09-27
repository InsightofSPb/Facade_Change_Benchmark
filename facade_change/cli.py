from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys

from .io import environment


def parser():
    p = argparse.ArgumentParser(description="Facade data inventory and native-RGB pair alignment")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Report installed packages and GPU availability")
    demo = sub.add_parser("demo", help="Run CPU SIFT on a procedural synthetic pair")
    demo.add_argument("--out", required=True)
    manifest = sub.add_parser("manifest", help="Resolve every COCO image; preserve unresolved rows")
    manifest.add_argument("--config", required=True)
    manifest.add_argument("--out", required=True)
    manifest.add_argument("--overrides", help="Reviewed metadata CSV")
    pair = sub.add_parser("align", help="Align exactly one temporal pair; output diagnostics")
    pair.add_argument("--manifest", required=True, dest="manifest_path")
    pair.add_argument("--reference-id", required=True)
    pair.add_argument("--source-id", required=True)
    pair.add_argument("--out", required=True)
    pair.add_argument("--method", choices=["sift", "loftr"], default="sift")
    pair.add_argument("--max-side", type=int, default=1024)
    pair.add_argument("--ransac-threshold", type=float, default=3.)
    pair.add_argument("--checkpoint")
    pair.add_argument("--device", default="cpu")
    pair.add_argument("--confidence", type=float, default=.4)
    pair.add_argument("--seed", type=int, default=42)
    pair.add_argument("--max-canvas-pixels", type=int, default=50_000_000)
    pair.add_argument("--max-canvas-side", type=int, default=16000)
    pair.add_argument("--allow-inferred-metadata", action="store_true")
    return p


def main(argv=None):
    args = vars(parser().parse_args(argv))
    command = args.pop("command")
    try:
        if command == "doctor":
            result = environment()
            result["importable"] = {m: importlib.util.find_spec(m) is not None for m in ("cv2", "torch", "kornia")}
            if shutil.which("nvidia-smi"):
                proc = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
                result["nvidia_smi"] = proc.stdout or proc.stderr
            else:
                result["nvidia_smi"] = "not installed"
        elif command == "manifest":
            from .data import build_manifest
            result = build_manifest(args["config"], args["out"], args["overrides"])["summary"]
        elif command == "demo":
            from .demo import run_demo
            result = run_demo(**args)
        else:
            from .pipeline import run_pair
            result = run_pair(**args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if command == "manifest" and any(k != "ready" for k in result["image_status"]):
            return 2  # Completed inventory with unresolved images; inspect the saved manifest.
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
