"""Prepare COCO with an 80/8/12 series-aware facade split using two input paths."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facade_change.data import DEFAULT_METADATA_RULES
from facade_change.dataset_setup import prepare_from_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coco_path", type=Path)
    parser.add_argument("image_root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--runs-root", type=Path, help="Where to find previous reviewed preparation runs")
    parser.add_argument("--previous-split", type=Path, help="Explicit override of automatic split discovery")
    parser.add_argument("--split-policy", choices=("series-v2", "legacy"), default="series-v2",
                        help="series-v2: 80/8/12 crop targets, medium-series holdouts; legacy: preserve the old policy")
    parser.add_argument("--rename-map", type=Path, action="append", dest="rename_maps")
    parser.add_argument("--metadata-rules", type=Path, default=DEFAULT_METADATA_RULES)
    parser.add_argument("--metadata-csv", type=Path, help="Reviewed table using current COCO image IDs")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--crop-stride", type=int, default=448)
    args = parser.parse_args()
    try:
        result = prepare_from_paths(**vars(args))
    except (OSError, ValueError) as exc:
        print(f"Подготовка не завершена: {exc}", file=sys.stderr)
        return 2
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())
