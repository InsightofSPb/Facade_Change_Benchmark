#!/usr/bin/env bash
# Separate environment and external source checkout; never modify lposs.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
geoscd_root="${GEOSCD_ROOT:-$(dirname "$repo_root")/GeoSCD-facade}"
geoscd_commit="dd31369654e96d6843cc4bbcebce854a8bc2159b"
geoscd_env="facade-geoscd"
download_weights=false
if [[ "${1:-}" == "--download-weights" && $# == 1 ]]; then
    download_weights=true
elif [[ $# != 0 ]]; then
    echo 'Usage: bash scripts/setup_geoscd.sh [--download-weights]' >&2
    exit 2
fi
command -v conda >/dev/null || { echo 'conda must be available in this shell.' >&2; exit 1; }

if [[ ! -e "$geoscd_root" ]]; then
    git clone --no-checkout https://github.com/ZilingLiu/GeoSCD.git "$geoscd_root"
    git -C "$geoscd_root" checkout --detach "$geoscd_commit"
fi
[[ "$(git -C "$geoscd_root" rev-parse HEAD)" == "$geoscd_commit" ]] || {
    echo "Expected GeoSCD commit $geoscd_commit in $geoscd_root; use a separate clean checkout." >&2
    exit 1
}
git -C "$geoscd_root" diff --quiet HEAD -- || {
    echo "Tracked GeoSCD sources have changes: $geoscd_root" >&2
    exit 1
}

if ! conda run -n "$geoscd_env" python -c 'import sys; assert sys.version_info[:2] == (3, 10)' >/dev/null 2>&1; then
    conda create -y -n "$geoscd_env" python=3.10
fi
conda run --no-capture-output -n "$geoscd_env" python -m pip install \
    torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
conda run --no-capture-output -n "$geoscd_env" python -m pip install \
    -r "$repo_root/configs/2026-10-04_geoscd_geometry.requirements.txt"
conda run --no-capture-output -n "$geoscd_env" python -m pip install -e "$repo_root"
conda run --no-capture-output -n "$geoscd_env" python -m pip check

if "$download_weights"; then
    # This opt-in is the only download path; inference never fetches weights.
    conda run --no-capture-output -n "$geoscd_env" python - "$geoscd_root/src/pretrained/model.pt" <<'PY'
import sys
from pathlib import Path
import torch
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
if not path.is_file():
    temporary = path.with_suffix('.pt.download')
    torch.hub.download_url_to_file(
        'https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt',
        str(temporary), progress=True,
    )
    temporary.replace(path)
print(f'VGGT checkpoint: {path}')
PY
fi
echo "Ready: conda activate $geoscd_env"
echo "GeoSCD source: $geoscd_root"
echo "Checkpoint: $geoscd_root/src/pretrained/model.pt"
