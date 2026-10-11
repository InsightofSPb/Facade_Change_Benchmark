"""Train a local frozen-DINOv2 semantic baseline on a prepared dataset."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Completed preparation directory or dataset.json")
    parser.add_argument("--methods-config", help="Existing methods or benchmark JSON containing DINOv2 paths")
    parser.add_argument("--source-root", help="Local original DINOv2 repository; overrides methods config")
    parser.add_argument("--checkpoint", help="Local dinov2_vits14_pretrain.pth; overrides methods config")
    parser.add_argument("--trust-checkpoint", action="store_true", help="Allow legacy pickle for explicitly trusted weights")
    parser.add_argument("--out", help="New run directory; never overwritten")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crops-per-image", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutmix-prob", type=float, default=.5)
    parser.add_argument("--cutout-prob", type=float, default=.25)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="One epoch, two images/split, four VAL crops/image; not a quality result")
    arguments = vars(parser.parse_args())
    if arguments["out"] is None:
        tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        arguments["out"] = str(Path("runs") / (tag + "-dino-seg" + ("-smoke" if arguments["smoke"] else "")))
    try:
        from .train import train
        result = train(**arguments)
    except (ImportError, ModuleNotFoundError) as error:
        parser.exit(2, f"Missing training dependency: {error}\nUse the existing DINO environment and install pycocotools==2.0.11 if missing.\n")
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        parser.exit(2, f"Segmentation training stopped: {error}\n")
    print(f"\nГотово: {result['out']}\nЛучший VAL mIoU: {result['best_val_miou']:.2%}; эпоха {result['best_epoch']}", flush=True)


if __name__ == "__main__":
    main()
