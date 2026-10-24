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
    p = argparse.ArgumentParser(description="Facade change research: prepare -> train -> benchmark -> gallery")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("workflow", help="Main workflow: prepare, train, benchmark, gallery")
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
    crop_input = dataset_crops.add_mutually_exclusive_group(required=True)
    crop_input.add_argument("--batch-run")
    crop_input.add_argument("--alignment-session", help="Completed dataset_statistics session (or an unambiguous cache root)")
    dataset_crops.add_argument("--quality-report", help="Explicit completed quality report for a separate evaluation or interrupted preparation")
    dataset_crops.add_argument("--split-file", help="Existing reviewed split.json if not beside the alignment manifest")
    dataset_crops.add_argument("--out", required=True)
    dataset_crops.add_argument("--methods", nargs="+", choices=["loftr", "sift", "cascade"],
                               help="Batch priority (default loftr sift); omit for authoritative session selections")
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
    history = sub.add_parser("prepare-history", help="CPU five-frame synthetic histories with matched no-event controls")
    history.add_argument("--dataset", required=True)
    history.add_argument("--out", required=True)
    history.add_argument("--seed", type=int, default=42)
    history.add_argument("--crops-per-building", type=int, default=1)
    history.add_argument("--max-buildings-per-split", type=int, default=0)
    history.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    history.add_argument("--final-event", choices=["mixed", "growth", "partial_repair", "delamination"], default="mixed")
    transitions = sub.add_parser("mine-transitions", help="Review candidates from aligned real COCO class transitions; not event ground truth")
    transitions.add_argument("--dataset", required=True)
    transitions.add_argument("--alignment-run", required=True)
    transitions.add_argument("--out", required=True)
    transitions.add_argument("--crop-mode", choices=["component", "windows"], default="component",
                             help="Full connected transition region plus context; windows restores legacy fixed-size selection")
    transitions.add_argument("--context-padding", type=int, default=100,
                             help="Context on each side of the transition bounding box, in native reference pixels")
    transitions.add_argument("--window-sizes", nargs="+", type=int, default=[50, 100],
                             help="Used only with --crop-mode windows")
    transitions.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    transitions.add_argument("--min-support-fraction", type=float, default=.5)
    transitions.add_argument("--min-transition-fraction", type=float, default=.02)
    transitions.add_argument("--min-component-pixels", type=int, default=25,
                             help="Minimum connected candidate area, including inside each exported window")
    transitions.add_argument("--boundary-tolerance-pixels", type=int, default=2,
                             help="Ignore same-class contour displacement within this native reference-grid radius")
    transitions.add_argument("--event-types", nargs="+", default=["replacement", "appearance", "disappearance"],
                             choices=["replacement", "appearance", "disappearance"])
    transitions.add_argument("--max-windows-per-pair", type=int, default=20)
    transitions.add_argument("--alignment-method", choices=["sift", "loftr", "cascade"])
    transitions.add_argument("--all-class-pairs", action=argparse.BooleanOptionalAction, default=True,
                             help="All directed COCO class replacements by default; --no-all-class-pairs restores the two original replacement pairs")
    review = sub.add_parser("review-transitions", help="Static landing page of all cached synthetic and real histories; no inference")
    review.add_argument("--candidate-runs", nargs="+", required=True,
                        help="Completed mine-transitions directories for TRAIN/VAL")
    review.add_argument("--out", required=True)
    review.add_argument("--synthetic-run", help="Completed prepare-history run from the same dataset")
    review.add_argument("--preview-max-side", type=int, default=0,
                        help="Display-only JPEG maximum side; 0 preserves native dimensions (default)")
    trial = sub.add_parser("h0h1-benchmark", help="Change scorers on frozen procedural H0/H1 cases")
    from .methods.registry import ALL_METHODS
    trial.add_argument("--config", dest="benchmark_config", help="Common benchmark JSON; explicit flags override it")
    trial.add_argument("--methods-config", help="Local source/weight/worker-Python options by method")
    trial.add_argument("--selection", dest="selection_path", help="Replay exact saved crop/case IDs and input hashes")
    trial.add_argument("--reuse-run", help="Complete previous result directory; retain its methods, maps and thresholds")
    trial.add_argument("--recompute-methods", nargs="+", choices=ALL_METHODS,
                       help="Explicitly replace these cached methods; all other cached maps stay frozen")
    trial.add_argument("--dataset-run")
    trial.add_argument("--out")
    trial.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=["rgb_diff", "ssim"])
    trial.add_argument("--max-bases-per-split", type=int, default=1,
                       help="Frozen SHA-ranked base crops per partition; 0 uses the full existing dataset")
    trial.add_argument("--quick-bases", type=int, default=0,
                       help="Up to this many distinct val/test buildings; 3 states x 2 shared nuisance scenarios per crop; overrides max-bases-per-split")
    trial.add_argument("--selection-seed", type=int, default=42,
                       help="Fixed seed for quick crop and nuisance selection before scoring")
    trial.add_argument("--compression-tile-size", type=int, default=32)
    trial.add_argument("--compression-stride", type=int, default=16)
    trial.add_argument("--zstd-level", type=int, default=3)
    trial.add_argument("--lzma-preset", type=int, default=3)
    trial.add_argument("--msdzip-abs-checkpoint")
    trial.add_argument("--msdzip-mod256-checkpoint")
    trial.add_argument("--device", default="cpu")
    trial.add_argument("--trust-checkpoint", action="store_true",
                       help="Explicit trust for legacy Torch without weights_only support")
    # Omitted CLI flags must not silently override JSON configuration. Defaults
    # remain in the runner signature, keeping existing direct calls unchanged.
    for action in trial._actions:
        if action.dest != "help":
            action.default = argparse.SUPPRESS
    method_check = sub.add_parser("methods-check", help="Small codec roundtrips or RSCD validation/identity controls")
    method_check.add_argument("--config", required=True, dest="benchmark_config")
    method_check.add_argument("--out", required=True)
    method_check.add_argument("--methods", required=True, nargs="+", choices=ALL_METHODS)
    method_check.add_argument("--max-val-bases", type=int, default=1)
    method_check.add_argument("--bitstream-check", action="store_true",
                              help="Require exact neural codec stream roundtrips; mismatch fails")
    codec_train = sub.add_parser("arib-train", help="Adapt original ArIB-BPS on H0 residual-image tiles")
    codec_train.add_argument("--dataset-run", required=True)
    codec_train.add_argument("--out", required=True)
    codec_train.add_argument("--source-root", default="third_party/arib_bps")
    codec_train.add_argument("--representations", nargs="+", choices=["abs", "mod256"], default=["abs", "mod256"])
    codec_train.add_argument("--author-config", choices=["imagenet32_config", "imagenet64_config", "imagenet64_small_config", "cifar_config"], default="imagenet32_config")
    codec_train.add_argument("--device", default="cpu")
    codec_train.add_argument("--epochs", type=int, default=3)
    codec_train.add_argument("--max-train-patches", type=int, default=160)
    codec_train.add_argument("--max-val-patches", type=int, default=16)
    codec_train.add_argument("--batch-size", type=int, default=1)
    codec_train.add_argument("--tile-size", type=int, default=32)
    codec_train.add_argument("--lr", type=float, default=2e-4)
    codec_train.add_argument("--seed", type=int, default=42)
    bcm_train = sub.add_parser("bcm-train", help="Adapt original BCM-Net with its VTM base on H0 RGB tiles")
    bcm_train.add_argument("--dataset-run", required=True)
    bcm_train.add_argument("--out", required=True)
    bcm_train.add_argument("--source-root", default="third_party/bcm_net")
    bcm_train.add_argument("--vtm-encoder", required=True)
    bcm_train.add_argument("--vtm-decoder", required=True)
    bcm_train.add_argument("--vtm-config", required=True)
    bcm_train.add_argument("--vtm-scc-config", help="Author class-SCC VTM config; otherwise derive sibling per-class/classSCC.cfg")
    bcm_train.add_argument("--init-checkpoint", help="Explicit local original medical checkpoint for domain fine-tuning; omit for fresh weights")
    bcm_train.add_argument("--trust-checkpoint", action="store_true",
                           help="Explicit trust for a local legacy Torch checkpoint that cannot use weights_only loading")
    bcm_train.add_argument("--device", default="cpu")
    bcm_train.add_argument("--epochs", type=int, default=3)
    bcm_train.add_argument("--max-train-patches", type=int, default=160)
    bcm_train.add_argument("--max-val-patches", type=int, default=16)
    bcm_train.add_argument("--batch-size", type=int, default=1)
    bcm_train.add_argument("--tile-size", type=int, choices=[32], default=32)
    bcm_train.add_argument("--lr", type=float, default=1e-4)
    bcm_train.add_argument("--seed", type=int, default=42)
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
    train.add_argument("--resume", action="store_true")
    train.add_argument("--checkpoint-every-steps", type=int, default=100)
    train.add_argument("--max-steps-per-call", type=int, default=0)
    temporal = sub.add_parser("temporal-experiment", help="Resumable TRAIN/VAL pair and state-history experiment")
    temporal.add_argument("--out", required=True)
    temporal.add_argument("--config", dest="config_path")
    temporal.add_argument("--dataset")
    temporal.add_argument("--history-run")
    temporal.add_argument("--alignment-runs", nargs="+")
    temporal.add_argument("--semantic-run")
    temporal.add_argument("--methods-config")
    temporal.add_argument("--methods", nargs="+")
    temporal.add_argument("--device")
    temporal.add_argument("--training-python")
    temporal.add_argument("--crops-per-building", type=int)
    temporal.add_argument("--max-buildings-per-split", type=int)
    temporal.add_argument("--resume", action="store_true")
    temporal.add_argument("--report-only", action="store_true")
    temporal.add_argument("--preflight-only", action="store_true")
    temporal.add_argument("--smoke-only", action="store_true")
    temporal.add_argument("--detach", action="store_true", help="Launch a new session; log to OUT.log")
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
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["workflow"]:
        from .workflow import main as workflow_main
        return workflow_main(argv[1:])
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
        elif command == "prepare-history":
            from .hypothesis_dataset import prepare_history_dataset
            result = prepare_history_dataset(**args)
        elif command == "mine-transitions":
            from .semantic_training import mine_transition_candidates
            result = mine_transition_candidates(**args)
        elif command == "review-transitions":
            from .semantic_training import prepare_transition_review
            result = prepare_transition_review(**args)
        elif command == "h0h1-benchmark":
            from .hypothesis_benchmark import run_hypothesis_benchmark
            from .benchmark_config import benchmark_arguments
            args = benchmark_arguments(args)
            result = run_hypothesis_benchmark(**args)
        elif command == "methods-check":
            from .method_checks import check_methods
            result = check_methods(**args)
        elif command == "msdzip-train":
            from importlib import import_module
            result = import_module(".2026-10-04_msdzip_h0", __package__).train_msdzip_h0(**args)
        elif command == "temporal-experiment":
            from importlib import import_module
            module = import_module(".2026-10-14_temporal_experiment", __package__)
            detached = args.pop("detach")
            result = module.launch_detached(argv) if detached else module.run_temporal_experiment(**args)
        elif command == "arib-train":
            from .codec_training import train_arib_h0
            result = train_arib_h0(**args)
        elif command == "bcm-train":
            from .bcm_training import train_bcm_h0
            result = train_bcm_h0(**args)
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
        if command == "h0h1-benchmark":
            from pathlib import Path
            output = Path(args["out"]).expanduser().resolve()
            print((output / "summary.txt").read_text(encoding="utf-8"))
            print(f"Полный отчёт: {output / 'summary.json'}\nГалерея: {output / 'gallery.html'}")
        else:
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
        if command == "temporal-experiment" and (result.get("status") == "completed_with_failures"
                or any(item.get("status") == "failed" for item in result.get("methods", {}).values())
                or result.get("missing_methods")):
            return 2
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
