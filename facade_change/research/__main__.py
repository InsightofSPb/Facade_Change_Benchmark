"""One local entrypoint for the two bounded research experiments."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ..io import read_json, sha256
from .common import digest, parent_selection, positive_integer


def resolve_source(source_run, need_models=True):
    """Read sealed benchmark metadata, never TEST score maps/RGB or weights."""
    root = Path(source_run).expanduser().resolve()
    record = read_json(root / "run.json")
    if record.get("kind") != "hypothesis_benchmark" or record.get("status") != "completed_exploratory":
        raise ValueError("source-run must be a completed hypothesis_benchmark")
    config = record["config"]
    if digest(config) != record.get("config_sha256"):
        raise ValueError("Source benchmark config hash disagrees")
    summary_path = root / "summary.json"
    if record.get("artifact_sha256", {}).get("summary.json") != sha256(summary_path):
        raise ValueError("Source benchmark summary hash disagrees")
    if read_json(summary_path).get("status") != "completed_exploratory":
        raise ValueError("Source benchmark summary is incomplete")
    selection = parent_selection(config["dataset_run"])
    if selection["fingerprints"] != config.get("input_sha256"):
        raise ValueError("Source benchmark and dataset provenance disagree")
    result = {"dataset_run": config["dataset_run"]}
    if need_models:
        scorers = config.get("scorers", {})
        msdzip = scorers.get("msdzip_mod256", {})
        dino = scorers.get("dinov2", {})
        # These fields are produced by the existing adapters, not guessed paths.
        result.update(msdzip_checkpoint=msdzip.get("checkpoint_path"),
                      dino_root=dino.get("sources", {}).get("path"),
                      dino_checkpoint=dino.get("checkpoint", {}).get("path"),
                      expected_msdzip_sha256=msdzip.get("checkpoint_sha256"),
                      expected_dino_sha256=dino.get("checkpoint", {}).get("sha256"))
        if not all(result.values()):
            raise ValueError("Source run lacks MSDZip/DINO local paths and hashes; pass --dataset-run plus explicit model paths")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bounded TRAIN/VAL paid H0/H1 and frozen reference-context feasibility")
    parser.add_argument("experiment", choices=("paid", "context", "both"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-run", help="Completed local comparison: resolve dataset and existing model paths from provenance")
    source.add_argument("--dataset-run", help="Completed procedural hypothesis dataset; context needs explicit model paths")
    parser.add_argument("--out", required=True, help="Fresh output directory; never overwrite or resume implicitly")
    parser.add_argument("--max-buildings-per-split", type=int, default=2)
    parser.add_argument("--max-bases-per-building", type=int, default=1)
    parser.add_argument("--scenarios-per-base", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fpr-targets", nargs="+", type=float, default=[.01, .05])
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--proposal-threshold", type=float, default=.08)
    parser.add_argument("--msdzip-checkpoint")
    parser.add_argument("--dino-root")
    parser.add_argument("--dino-checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-bytes", type=int, default=32768)
    parser.add_argument("--max-val-bytes", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--trust-checkpoint", action="store_true", help="Explicit trust for existing local legacy MSDZip/DINO pickle checkpoints")
    args = parser.parse_args(argv)
    try:
        models_needed = args.experiment != "paid"
        resolved = resolve_source(args.source_run, models_needed) if args.source_run else {"dataset_run": args.dataset_run}
        if models_needed:
            for key in ("msdzip_checkpoint", "dino_root", "dino_checkpoint"):
                value = getattr(args, key)
                if value:
                    resolved[key] = value
                    # Explicit alternate models have their own hashes in the run.
                    if key == "msdzip_checkpoint":
                        resolved.pop("expected_msdzip_sha256", None)
                    if key == "dino_checkpoint":
                        resolved.pop("expected_dino_sha256", None)
                if not resolved.get(key):
                    raise ValueError(f"Missing --{key.replace('_', '-')}")
        common = {key: getattr(args, key) for key in ("max_buildings_per_split", "max_bases_per_building",
                                                   "scenarios_per_base", "seed", "fpr_targets")}
        if args.source_run:
            source_root = Path(args.source_run).expanduser().resolve()
            source_record = read_json(source_root / "run.json")
            common["source_lineage"] = {"run_path": str(source_root), "run_sha256": sha256(source_root / "run.json"),
                                       "config_sha256": source_record["config_sha256"],
                                       "summary_sha256": sha256(source_root / "summary.json")}
        if not args.fpr_targets or any(not math.isfinite(x) or not 0 <= x <= 1 for x in args.fpr_targets):
            raise ValueError("--fpr-targets must be finite probabilities")
        positive_integer(args.tile_size, "tile_size")
        positive_integer(args.stride, "stride")
        if args.stride > args.tile_size or not math.isfinite(args.proposal_threshold) or not 0 <= args.proposal_threshold <= 1:
            raise ValueError("Invalid paid tile/stride or proposal threshold")
        if models_needed:
            for name in ("epochs", "max_train_bytes", "max_val_bytes", "batch_size", "width", "tokens"):
                positive_integer(getattr(args, name), name)
            if not math.isfinite(args.lr) or args.lr <= 0:
                raise ValueError("--lr must be finite and positive")
            for model, expected in (("msdzip", "expected_msdzip_sha256"), ("dino", "expected_dino_sha256")):
                if expected in resolved and sha256(resolved[model + "_checkpoint"]) != resolved[expected]:
                    raise ValueError(f"{model} checkpoint differs from the source run")
        out = Path(args.out).expanduser().resolve()
        if out.exists():
            raise FileExistsError(f"Output must be a fresh directory: {out}")
        if args.experiment == "both":
            # Validate both sets of inputs before the first experiment allocates work.
            parent_selection(resolved["dataset_run"], **{k: v for k, v in common.items() if k not in {"fpr_targets", "source_lineage"}})
            for key in ("msdzip_checkpoint", "dino_checkpoint"):
                if not Path(resolved[key]).is_file():
                    raise FileNotFoundError(resolved[key])
            if not (Path(resolved["dino_root"]) / "hubconf.py").is_file():
                raise FileNotFoundError("Local DINOv2 hubconf.py missing")
            out.mkdir(parents=True)
        results = {}
        if args.experiment in ("paid", "both"):
            from .paid import run_paid
            destination = out / "paid" if args.experiment == "both" else out
            run_paid(resolved["dataset_run"], destination, **common, tile_size=args.tile_size,
                     stride=args.stride, proposal_threshold=args.proposal_threshold)
            results["paid"] = str(destination)
        if models_needed:
            from .context_training import run_context
            destination = out / "context" if args.experiment == "both" else out
            run_context(out=destination, **resolved, **common,
                        **{key: getattr(args, key) for key in ("device", "epochs", "max_train_bytes", "max_val_bytes",
                                                               "batch_size", "lr", "width", "tokens", "trust_checkpoint")})
            results["context"] = str(destination)
        print(json.dumps({"status": "completed_exploratory", "runs": results}, ensure_ascii=False, indent=2))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"{type(exc).__name__}: {exc}")
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
