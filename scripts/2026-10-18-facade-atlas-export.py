"""Compatibility entry point for facade_change.gallery.atlas_export."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facade_change.gallery.atlas_export import main

if __name__ == "__main__":
    raise SystemExit(main())
