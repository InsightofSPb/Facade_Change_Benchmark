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
    dataset_crops = sub.add_parser("crop-dataset", help="Export accepted existing batch geometry; preserve prepared splits")
    dataset_crops.add_argument("--batch-run", required=True)
    dataset_crops.add_argument("--out", required=True)
    dataset_crops.add_argument("--methods", nargs="+", choices=["loftr", "sift", "cascade"], default=["loftr", "sift"],
                               help="Priority order; at most one accepted alignment per pair")
    dataset_crops.add_argument("--tile-size", type=int, default=256)
    dataset_crops.add_argument("--stride", type=int, default=128)
    dataset_crops.add_argument("--min-valid-fraction", type=float, default=.8)
    dataset_crops.add_argument("--controls", type=int, default=0,
                               help="Total source crops for procedural controls across the dataset")
    dataset_crops.add_argument("--seed", type=int, default=42)
    hypotheses = sub.add_parser("h0h1", help="Deterministic state edits and observation nuisances from reviewed crops")
    hypotheses.add_argument("--crop-run", required=True)
    hypotheses.add_argument("--config", required=True, dest="config_path")
    hypotheses.add_argument("--out", required=True)
    trial = sub.add_parser("h0h1-benchmark", help="Change scorers on frozen procedural H0/H1 cases")
    trial.add_argument("--dataset-run", required=True)
    trial.add_argument("--out", required=True)
    trial.add_argument("--methods", nargs="+", choices=["rgb_diff", "ssim", "zstd_abs", "zstd_mod256", "lzma_abs",
                       "lzma_mod256", "msdzip_abs", "msdzip_mod256"], default=["rgb_diff", "ssim"])
    trial.add_argument("--max-bases-per-split", type=int, default=1,
                       help="Frozen SHA-ranked base crops per partition; 0 uses the full existing dataset")
    trial.add_argument("--compression-tile-size", type=int, default=32)
    trial.add_argument("--compression-stride", type=int, default=16)
    trial.add_argument("--zstd-level", type=int, default=3)
    trial.add_argument("--lzma-preset", type=int, default=3)
    trial.add_argument("--msdzip-abs-checkpoint")
    trial.add_argument("--msdzip-mod256-checkpoint")
    trial.add_argument("--device", default="cpu")
    trial.add_argument("--trust-checkpoint", action="store_true",
                       help="Explicit trust for legacy Torch without weights_only support")
    train = sub.add_parser("msdzip-train", help="Train original MSDZip on reviewed train H0 residuals only")
    train.add_argument("--dataset-run", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--representations", nargs="+", choices=["abs", "mod256"], default=["abs", "mod256"])
    train.add_argument("--device", default="cpu")
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--max-train-bytes", type=int, default=2_000_000)
    train.add_argument("--max-val-bytes", type=int, default=200_000)
    train.add_argument("--model-batch-size", type=int, default=32)
    train.add_argument("--window-groups", type=int, default=16,
                       help="Independent fixed-lane batches per update; recorded and reused during scoring")
    train.add_argument("--timesteps", type=int, default=16)
    train.add_argument("--hidden-dim", type=int, default=256)
    train.add_argument("--ffn-dim", type=int, default=4096)
    train.add_argument("--vocab-dim", type=int, default=16)
    train.add_argument("--lr", type=float, default=.001)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--max-bases-per-split", type=int, default=0)
    train.add_argument("--trust-checkpoint", action="store_true")
    geo = sub.add_parser("geoscd", help="GeoSCD geometry-only dense alignment trial; no SAM/change detector")
    geo.add_argument("--manifest", required=True, dest="manifest_path")
    geo.add_argument("--out", required=True)
    geo.add_argument("--geoscd-root", required=True, help="Clean pinned official GeoSCD checkout")
    geo.add_argument("--checkpoint", required=True, help="Existing local VGGT-1B model.pt; never downloaded here")
    geo.add_argument("--pair", action="append", dest="pairs", help="REFERENCE_ID:SOURCE_ID; repeat for explicit pairs")
    geo.add_argument("--limit", type=int, default=3, help="Pair limit; 0 means all selected pairs")
    geo.add_argument("--split", choices=["all", "dev", "train", "val", "test"], default="all")
    geo.add_argument("--device", default="cuda:0")
    geo.add_argument("--resolution", type=int, choices=[518], default=518,
                     help="Use the VGGT grid directly, avoiding extra intrinsic/depth resize")
    geo.add_argument("--seed", type=int, default=42)
    geo.add_argument("--comparison-run", help="Existing SIFT/LoFTR batch on the same manifest")
    full_geo = sub.add_parser("geoscd-full", help="Full official GeoSCD: VGGT plus SAM ViT-H change masks")
    full_geo.add_argument("--manifest", required=True, dest="manifest_path")
    full_geo.add_argument("--out", required=True)
    full_geo.add_argument("--geoscd-root", required=True, help="Clean pinned official GeoSCD checkout")
    full_geo.add_argument("--checkpoint", required=True, help="Existing local VGGT-1B checkpoint")
    full_geo.add_argument("--sam-checkpoint", required=True, help="Existing SAM1 ViT-H checkpoint; SAM3 is incompatible")
    full_geo.add_argument("--pair", action="append", dest="pairs", help="REFERENCE_ID:SOURCE_ID; repeat for explicit pairs")
    full_geo.add_argument("--limit", type=int, default=3, help="Pair limit; 0 means all selected pairs")
    full_geo.add_argument("--split", choices=["all", "dev", "train", "val", "test"], default="all")
    full_geo.add_argument("--device", default="cuda:0")
    full_geo.add_argument("--seed", type=int, default=42)
    full_geo.add_argument("--mode", choices=["initial", "occupy"], default="occupy")
    full_geo.add_argument("--points-per-side", type=int, default=32)
    full_geo.add_argument("--iou-thresh", type=float, default=.65)
    full_geo.add_argument("--sem-filter", type=float, default=None)
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
        elif command == "crop-dataset":
            from .crop_dataset import prepare_crop_dataset
            result = prepare_crop_dataset(**args)
        elif command == "h0h1":
            from .hypothesis_dataset import prepare_hypothesis_dataset
            result = prepare_hypothesis_dataset(**args)
        elif command == "h0h1-benchmark":
            from .hypothesis_benchmark import run_hypothesis_benchmark
            result = run_hypothesis_benchmark(**args)
        elif command == "msdzip-train":
            from importlib import import_module
            result = import_module(".2026-10-04_msdzip_h0", __package__).train_msdzip_h0(**args)
        elif command == "geoscd":
            from .geoscd import run_geoscd
            result = run_geoscd(**args)
        elif command == "geoscd-full":
            from .geoscd_full import run_geoscd_full
            result = run_geoscd_full(**args)
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
        if command in ("geoscd", "geoscd-full") and result["failed_runs"]:
            return 2
        if command == "crop-dataset" and result.get("derivative_failures", 0):
            return 2
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
