"""Prepare fixed crops/augmentations or train the local DINOv2 semantic head."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset", help="Completed preparation directory or dataset.json; required for preparation/training")
    parser.add_argument("--prepare-only", action="store_true", help="Save every TRAIN crop plus two augmentations; VAL originals only; no model/GPU")
    parser.add_argument("--augmentations", help="Completed fixed augmentation directory used for training")
    parser.add_argument("--reuse-augmentations", help="During preparation, reuse unchanged assets from an earlier snapshot")
    parser.add_argument("--evaluate-run", help="Completed training directory: evaluate best checkpoint on fixed VAL observation conditions")
    parser.add_argument("--evaluation-mode", choices=("robustness", "originals"), default="robustness",
                        help="Evaluation only: VAL robustness (default), or unchanged originals with full-photo galleries")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"),
                        help="With --evaluation-mode originals: requested original splits; default val; TEST is explicit opt-in")
    parser.add_argument("--export-probabilities", action="store_true", help="With --evaluate-run, save class probabilities and annotated-support uncertainty diagnostics")
    parser.add_argument("--class-groups", help="Explicit semantic class-group JSON (default: repository semantic_class_groups.json)")
    parser.add_argument("--methods-config", help="Existing methods or benchmark JSON containing DINOv2 paths")
    parser.add_argument("--source-root", help="Local original DINOv2 repository; overrides methods config")
    parser.add_argument("--checkpoint", help="Local dinov2_vits14_pretrain.pth; overrides methods config")
    parser.add_argument("--trust-checkpoint", action="store_true", help="Allow legacy pickle for explicitly trusted weights")
    parser.add_argument("--out", help="New run directory; never overwritten")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-eval-fraction", type=float, default=0.2,
                        help="Full original TRAIN evaluation every ceil(epochs*fraction) epochs and at the end; default 0.2 (4,8,12,16,20 for 20 epochs). VAL and online TRAIN metrics run every epoch")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutmix-prob", type=float, help="Preparation only; default 0.25")
    parser.add_argument("--cutout-prob", type=float, help="Preparation only; default 0.125")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Short training check using an already complete augmentation snapshot")
    arguments = vars(parser.parse_args())
    if arguments["out"] is None:
        tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        suffix = (("-semantic-originals" if arguments["evaluation_mode"] == "originals" else "-semantic-robustness") if arguments["evaluate_run"] else "-semantic-augmentations"
                  if arguments["prepare_only"] else "-dino-seg" + ("-smoke" if arguments["smoke"] else ""))
        arguments["out"] = str(Path("runs") / (tag + suffix))
    try:
        prepare_only = arguments.pop("prepare_only")
        evaluation = arguments.pop("evaluate_run")
        evaluation_mode = arguments.pop("evaluation_mode")
        splits = arguments.pop("splits")
        export = arguments.pop("export_probabilities")
        reuse = arguments.pop("reuse_augmentations")
        cutmix, cutout = arguments.pop("cutmix_prob"), arguments.pop("cutout_prob")
        if evaluation:
            allowed = {"--evaluate-run", "--augmentations", "--device", "--batch-size",
                       "--out", "--smoke", "--no-amp", "--export-probabilities",
                       "--evaluation-mode", "--splits"}
            explicit = {value.split("=", 1)[0] for value in sys.argv[1:] if value.startswith("--")}
            rejected = sorted(explicit - allowed)
            if rejected:
                parser.error("--evaluate-run uses the saved training settings; incompatible options: "
                             + ", ".join(rejected))
            from .evaluate import evaluate_checkpoint
            options = {}
            if evaluation_mode != "robustness" or splits is not None:
                options = {"evaluation_mode": evaluation_mode, "splits": splits}
            result = evaluate_checkpoint(evaluation, arguments["out"],
                                         augmentations=arguments["augmentations"], device=arguments["device"],
                                         batch_size=arguments["batch_size"], export_probabilities=export,
                                         smoke=arguments["smoke"], no_amp=arguments["no_amp"], **options)
            print(f"\nEvaluation ({evaluation_mode}): {result['out']}", flush=True)
            return
        if splits is not None or any(value.split("=", 1)[0] == "--evaluation-mode" for value in sys.argv[1:]):
            parser.error("--evaluation-mode/--splits require --evaluate-run")
        if export:
            parser.error("--export-probabilities requires --evaluate-run")
        if not arguments["dataset"]:
            parser.error("--dataset is required for preparation and training")
        if prepare_only:
            if arguments["smoke"] or arguments["augmentations"]:
                parser.error("--prepare-only creates a complete snapshot; do not combine with --smoke/--augmentations")
            if any(value.split("=", 1)[0] == "--train-eval-fraction" for value in sys.argv[1:]):
                parser.error("--train-eval-fraction applies to training only")
            from .prepared import prepare_augmentations
            result = prepare_augmentations(arguments["dataset"], arguments["out"], seed=arguments["seed"],
                                           cutmix_prob=.25 if cutmix is None else cutmix,
                                           cutout_prob=.125 if cutout is None else cutout, reuse=reuse)
            print(f"\nГотовые кропы: {result['out']}; TRAIN оригиналы: {result['train_originals']}, "
                  f"аугментации: {result['train_augmented']}; VAL: {result['val_originals']}", flush=True)
            return
        if reuse is not None or cutmix is not None or cutout is not None:
            parser.error("Augmentation options belong to --prepare-only; training reads --augmentations unchanged")
        if not arguments["augmentations"]:
            parser.error("Training requires --augmentations from a completed --prepare-only run")
        from .train import train
        result = train(**arguments)
    except (ImportError, ModuleNotFoundError) as error:
        parser.exit(2, f"Missing training dependency: {error}\nUse the existing DINO environment and install pycocotools==2.0.11 if missing.\n")
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        parser.exit(2, f"Segmentation training stopped: {error}\n")
    print(f"\nГотово: {result['out']}\nЛучший VAL mIoU: {result['best_val_miou']:.2%}; эпоха {result['best_epoch']}", flush=True)


if __name__ == "__main__":
    main()
