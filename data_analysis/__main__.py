"""One entry point for offline benchmark comparison and recall diagnostics."""
from __future__ import annotations

import argparse

from .runner import run_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Complete original H0/H1 benchmark with saved maps")
    parser.add_argument("--out", required=True, help="New output directory")
    parser.add_argument("--fusion-run", help="Optional previous frozen GeoSCD AND MSDZip run to verify exact parity")
    parser.add_argument("--gallery-limit", type=int, default=60,
                        help="Maximum diagnostic gallery cases; all cases remain in numerical analysis")
    args = parser.parse_args()
    try:
        run_analysis(args.run, args.out, args.fusion_run, args.gallery_limit)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Ошибка анализа: {exc}\n")


if __name__ == "__main__":
    main()
