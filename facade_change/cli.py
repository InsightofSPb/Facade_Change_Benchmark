from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys

from .io import environment


def alignment_options(command):
    command.add_argument("--max-side", type=int, default=1024)
    command.add_argument("--ransac-threshold", type=float, default=3.)
    command.add_argument("--checkpoint", default="auto", help="Local checkpoint or auto: standard Torch cache")
    command.add_argument("--device", default="cpu")
    command.add_argument("--confidence", type=float, default=.4)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--max-canvas-pixels", type=int, default=50_000_000)
    command.add_argument("--max-canvas-side", type=int, default=16000)
    command.add_argument("--allow-inferred-metadata", action="store_true")
    command.add_argument("--trust-checkpoint", action="store_true",
                         help="Explicitly trust local/official checkpoint for legacy Torch pickle loading")
    command.add_argument("--download-weights", action="store_true",
                         help="Download the official LoFTR outdoor checkpoint only if absent")
    command.add_argument("--min-inliers", type=int, default=30)
    command.add_argument("--min-inlier-ratio", type=float, default=.2)
    command.add_argument("--min-hull-fraction", type=float, default=.1)
    command.add_argument("--min-overlap-fraction", type=float, default=.2)


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
    pair.add_argument("--method", choices=["sift", "loftr", "cascade"], default="sift")
    alignment_options(pair)
    prep = sub.add_parser("prepare", help="Prepare pairs/splits from an existing manifest; no image rescan")
    prep.add_argument("--manifest", required=True, dest="manifest_path")
    prep.add_argument("--out", required=True)
    prep.add_argument("--overrides")
    prep.add_argument("--previous-split", help="Previous reviewed split.json; preserve gold cohort and building assignments")
    prep.add_argument("--split-mode", choices=["dev", "reviewed"], default="dev")
    prep.add_argument("--pair-policy", choices=["adjacent", "first-anchor", "all"], default="first-anchor")
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--val-fraction", type=float, default=.10)
    prep.add_argument("--test-fraction", type=float, default=.20)
    prep.add_argument("--assets-config", help="Optional bounded code/config/log/weight inventory")
    batch = sub.add_parser("batch", help="Compare methods and optionally create native crops/controlled cases")
    batch.add_argument("--manifest", required=True, dest="manifest_path")
    batch.add_argument("--out", required=True)
    batch.add_argument("--methods", nargs="+", choices=["sift", "loftr", "cascade"], default=["sift", "loftr", "cascade"])
    batch.add_argument("--pair", action="append", dest="pairs", help="REFERENCE_ID:SOURCE_ID; repeat for explicit pairs")
    batch.add_argument("--limit", type=int, default=3, help="Pair limit; 0 means all selected pairs")
    batch.add_argument("--split", choices=["all", "dev", "train", "val", "test"], default="all")
    batch.add_argument("--crops", action="store_true")
    batch.add_argument("--tile-size", type=int, default=256)
    batch.add_argument("--stride", type=int, default=128)
    batch.add_argument("--min-valid-fraction", type=float, default=.8)
    batch.add_argument("--controls", type=int, default=0, help="Source crops per pair for small factorial/sham examples")
    batch.add_argument("--crop-method", choices=["sift", "loftr", "cascade"])
    alignment_options(batch)
    crop = sub.add_parser("crops", help="Warp crop windows directly from original RGB")
    crop.add_argument("--pair-run", required=True)
    crop.add_argument("--out", required=True)
    crop.add_argument("--tile-size", type=int, default=256)
    crop.add_argument("--stride", type=int, default=128)
    crop.add_argument("--min-valid-fraction", type=float, default=.8)
    crop.add_argument("--split", choices=["dev", "train", "val", "test"], default="dev")
    crop.add_argument("--group-id")
    controlled = sub.add_parser("controlled", help="Procedural state/nuisance factorial and sham controls")
    controlled.add_argument("--crops-path", required=True)
    controlled.add_argument("--out", required=True)
    controlled.add_argument("--seed", type=int, default=42)
    controlled.add_argument("--max-crops", type=int, default=3)
    assets = sub.add_parser("inventory", help="Bounded candidate inventory; no image or checkpoint loading")
    assets.add_argument("--config", required=True, dest="config_path")
    assets.add_argument("--out", required=True)
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
        elif command == "prepare":
            from .preparation import prepare_dataset
            result = prepare_dataset(**args)["summary"]
        elif command == "batch":
            from .batch import run_batch
            result = run_batch(**args)
        elif command == "crops":
            from .derived import build_crops
            result = build_crops(**args)
        elif command == "controlled":
            from .derived import controlled_examples
            result = controlled_examples(**args)
        elif command == "inventory":
            from .assets import inventory_assets
            result = inventory_assets(**args)["summary"]
        elif command == "align":
            from .pipeline import run_pair
            result = run_pair(**args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if command == "manifest" and any(k != "ready" for k in result["image_status"]):
            return 2  # Completed inventory with unresolved images; inspect the saved manifest.
        if command == "align" and not result["quality_gate"]["passed"]:
            return 2
        if command == "batch" and any(result[name] for name in ("failed_runs", "rejected_runs", "derivative_failures")):
            return 2
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
