"""One entry point for offline benchmark comparison and recall diagnostics."""
from __future__ import annotations

import argparse
import importlib
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "history-fusion":
        return importlib.import_module("data_analysis.2026-11-06_history_fusion").main(argv[1:])
    if argv and argv[0] == "variability-memory":
        return importlib.import_module("data_analysis.2026-11-08_variability_memory").main(argv[1:])
    from .runner import run_analysis

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Complete original H0/H1 benchmark with saved maps")
    parser.add_argument("--out", required=True, help="New output directory")
    parser.add_argument("--fusion-run", help="Optional previous frozen GeoSCD AND MSDZip run to verify exact parity")
    parser.add_argument("--gallery-limit", type=int, default=60,
                        help="Maximum diagnostic gallery cases; all cases remain in numerical analysis")
    args = parser.parse_args(argv)
    try:
        run_analysis(args.run, args.out, args.fusion_run, args.gallery_limit)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Ошибка анализа: {exc}\n")


if __name__ == "__main__":
    main()
