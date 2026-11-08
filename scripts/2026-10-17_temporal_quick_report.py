#!/usr/bin/env python3
"""Compatibility launcher; implementation lives in facade_change.research.temporal_quick_report."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facade_change.research.temporal_quick_report import main

if __name__ == "__main__":
    raise SystemExit(main())
