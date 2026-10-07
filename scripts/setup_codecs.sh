#!/usr/bin/env bash
# Verify classical codecs in the active environment; do not alter any packages.
set -euo pipefail
CODECS_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CODECS_REPO_ROOT"
if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  echo 'FFmpeg and ffprobe are required. Install once: sudo apt-get install ffmpeg' >&2
  exit 1
fi
python - <<'PYCODECS'
import sys
import numpy as np
from facade_change.methods.lossless import FFmpegCodec
rng = np.random.default_rng(42)
a = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
b = a.copy()
b[10:18, 10:18] = [255, 0, 1]
for kind in ('jpegls', 'h264'):
    codec = FFmpegCodec(kind)
    charged, stats = codec.encode(a, b)
    print(kind, codec.metadata['ffmpeg'], stats)
print('Classical codecs ready; active Python:', sys.executable)
PYCODECS
