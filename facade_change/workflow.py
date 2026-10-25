"""Prepare data -> optionally train models -> benchmark -> build galleries.

Choose a stage and task; use --help on that task for its inputs and options.
Reuse saved data, checkpoints and maps to continue from the required stage.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import import_module
import subprocess
import sys


@dataclass(frozen=True)
class Command:
    module: str
    help: str
    prefix: tuple = ()
    required: tuple = ()
    forbidden: tuple = ()
    legacy_process: bool = False


CORE = "facade_change.cli"
SEGMENTATION = "facade_change.segmentation.__main__"
REAL = "scripts.real_eval.real_temporal_night"
ROBUSTNESS = "scripts.real_eval.real_robustness"

COMMANDS = {
    "prepare": {
        "dataset": Command("facade_change.dataset_setup", "COCO, reviewed identities and facade-disjoint splits"),
        "align": Command(CORE, "Align original RGB pairs; optionally export crops", ("batch",)),
        "crops": Command(CORE, "Export a crop dataset from saved alignment", ("crop-dataset",)),
        "crop-histories": Command("facade_change.history_features", "Canonical multi-year RGB crop tracks from saved alignment", ("prepare",)),
        "history-features": Command("facade_change.history_features", "Frozen DINOv2 features and year-aware crop sequences", ("embed",)),
        "history-targets": Command("facade_change.research.2026-10-25_history_targets",
                                   "Project target-class COCO masks into saved histories; retain unknown areas"),
        "temporal-embeddings": Command("facade_change.research.2026-10-25_history_models",
                                       "Apply trained history encoders to cached features", ("evaluate",)),
        "synthetic": Command(CORE, "Controlled H0/H1 cases from reviewed RGB crops", ("h0h1",)),
        "history": Command(CORE, "Five-frame synthetic histories and matched controls", ("prepare-history",)),
        "augmentations": Command(SEGMENTATION, "Fixed segmentation crops and augmentations", ("--prepare-only",),
                                 forbidden=("--evaluate-run",)),
        "transitions": Command(CORE, "Aligned real transition candidates for review", ("mine-transitions",)),
        "real-audit": Command(REAL, "Inventory existing real histories and alignment", ("audit",)),
        "real": Command(REAL, "Prepare reviewed real history crops from an audit", ("prepare",)),
        "robustness": Command(ROBUSTNESS, "Prepare real observation-variation controls", ("prepare",)),
    },
    "train": {
        "history-models": Command("facade_change.research.2026-10-25_history_models",
                                  "Pair/pooled classifiers, GRU and temporal Transformer", ("train",)),
        "segmentation": Command(SEGMENTATION, "Train the DINO semantic head on prepared augmentations",
                                forbidden=("--prepare-only", "--evaluate-run")),
        "msdzip": Command(CORE, "Train MSDZip on TRAIN H0 RGB residuals", ("msdzip-train",)),
        "arib": Command(CORE, "Adapt ArIB-BPS on TRAIN H0 residuals", ("arib-train",)),
        "bcm": Command(CORE, "Adapt BCM-Net on TRAIN H0 RGB tiles", ("bcm-train",)),
    },
    "benchmark": {
        "history-scores": Command("facade_change.research.2026-10-25_history_models",
                                  "Cached previous/earliest/history cosine scores and review table", ("score",)),
        "history-models": Command("facade_change.research.2026-10-25_history_models",
                                  "Frozen temporal models, validation thresholds and facade statistics", ("evaluate",)),
        "synthetic": Command(CORE, "Compare methods on fixed H0/H1 cases", ("h0h1-benchmark",)),
        "real": Command(REAL, "PREVIOUS / EARLIEST / HISTORY with frozen models", ("run",)),
        "pilot": Command(REAL, "Bounded real-pair and history pilot", ("pilot",)),
        "robustness": Command(ROBUSTNESS, "Score an explicitly reviewed robustness selection", ("run",)),
        "semantic": Command("scripts.real_eval.semantic_routes", "Run configured semantic routes", ("run",)),
        "segmentation": Command(SEGMENTATION, "Evaluate a saved segmentation checkpoint",
                                required=("--evaluate-run",), forbidden=("--prepare-only",)),
        "metrics": Command("scripts.real_eval.real_gt_metrics", "Evaluate saved maps against annotation transitions",
                           legacy_process=True),
        "calibration": Command("scripts.real_eval.real_gt_calibration", "Calibrate saved real-GT maps with explicit folds",
                               legacy_process=True),
        "real-lofo": Command(REAL, "Real VAL LOFO and pairwise combinations from saved maps", ("calibrate",)),
        "final-fpr": Command("scripts.real_eval.real_final_fpr", "Compare components and AND/OR at final FPR budgets",
                             legacy_process=True),
        "fusion": Command("scripts.real_eval.temporal_fusion", "Frozen component-threshold fusion on synthetic VAL"),
        "temporal-snapshot": Command("facade_change.research.temporal_quick_report",
                                     "CPU metrics and coverage from a snapshot of saved synthetic VAL units"),
        "semantic-updates": Command("scripts.real_eval.2026-10-19_semantic_updates", "Cache-only reconstructable C,V updates"),
        "continuous-updates": Command("scripts.real_eval.2026-10-19_continuous_semantic_updates",
                                      "Cache-only raw-pair or directed-delta event fusion"),
    },
    "gallery": {
        "sheets": Command("facade_change.gallery.report_sheets", "Compose pair and chronology sheets from saved report panels"),
        "lofo": Command("facade_change.gallery.lofo", "Render saved real-VAL LOFO decisions with mask/count verification"),
        "histories": Command(CORE, "Review saved synthetic and real transition histories", ("review-transitions",)),
        "real": Command(REAL, "Rebuild the real-history HTML report and tables", ("report",)),
        "real-crops": Command("scripts.real_eval.real_temporal_gallery", "Build the saved real-VAL crop gallery"),
        "robustness": Command(ROBUSTNESS, "Review prepared robustness crops before inference", ("review",)),
        "robustness-results": Command(ROBUSTNESS, "Report saved robustness results", ("report",)),
        "comparison": Command("data_analysis.__main__", "Offline gallery/recall analysis of the fixed six-method benchmark"),
        "errors": Command("scripts.real_eval.2026-10-20_semantic_error_audit", "Audit a frozen directed-delta result"),
        "bundle": Command("facade_change.gallery.history_bundle", "Package prepared real histories for the atlas"),
        "atlas": Command("facade_change.gallery.atlas", "Build or serve the local atlas from a saved bundle"),
    },
}

STAGES = {
    "prepare": "Data, splits, alignment, crops and synthetic controls",
    "train": "Optional training; skip for frozen or zero-shot models",
    "benchmark": "Inference, saved-map evaluation and calibration",
    "gallery": "Review, reports and the atlas from saved artifacts",
}


def parser():
    result = argparse.ArgumentParser(
        prog="facade workflow", description=__doc__, allow_abbrev=False,
        epilog="Use STAGE TASK --help for the existing task options. "
               "See docs/2026-10-24-main-workflow.md for inputs and outputs.")
    stages = result.add_subparsers(dest="stage", required=True)
    for stage, commands in COMMANDS.items():
        group = stages.add_parser(stage, help=STAGES[stage], description=STAGES[stage], allow_abbrev=False)
        tasks = group.add_subparsers(dest="task", required=True)
        for name, command in commands.items():
            tasks.add_parser(name, help=command.help, add_help=False, allow_abbrev=False)
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cli = parser()
    # Only consume the stage/task here: their owner parses every remaining flag,
    # including --help, repeated options and values with spaces, exactly once.
    selected = cli.parse_args(argv[:2])
    command = COMMANDS[selected.stage][selected.task]
    arguments = argv[2:]
    explicit = {arg.split("=", 1)[0] for arg in arguments if arg.startswith("--")}
    if not {"--help", "-h"}.intersection(arguments):
        missing = set(command.required) - explicit
        forbidden = set(command.forbidden) & explicit
        if missing:
            cli.error(f"{selected.stage} {selected.task} requires {', '.join(sorted(missing))}")
        if forbidden:
            cli.error(f"Options belong to a different stage: {', '.join(sorted(forbidden))}")
    forwarded = [*command.prefix, *arguments]
    if command.legacy_process:
        # These saved-evaluator sources are fingerprinted by frozen analyses.
        # Preserve their no-argument main() and ordinary process exit behavior.
        return subprocess.run([sys.executable, "-m", command.module, *forwarded], check=False).returncode
    result = import_module(command.module).main(forwarded)
    return 0 if result is None else result


if __name__ == "__main__":
    raise SystemExit(main())
