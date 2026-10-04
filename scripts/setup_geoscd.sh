#!/usr/bin/env bash
# Separate environment and external source checkout; never modify lposs.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
geoscd_root="${GEOSCD_ROOT:-$(dirname "$repo_root")/GeoSCD-facade}"
geoscd_commit="dd31369654e96d6843cc4bbcebce854a8bc2159b"
sam_commit="dca509fe793f601edb92606367a655c15ac00fdf"
geoscd_env="facade-geoscd"
download_weights=false
full=false
for argument in "$@"; do
    case "$argument" in
        --download-weights) download_weights=true ;;
        --full) full=true ;;
        *) echo 'Usage: bash scripts/setup_geoscd.sh [--full] [--download-weights]' >&2; exit 2 ;;
    esac
done
# A user-site package can otherwise satisfy pip dependencies outside this environment.
# Apply this before the first Python process, including when rerunning a partial setup.
export PYTHONNOUSERSITE=1
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

if ! conda run -n "$geoscd_env" python -s -c 'import sys; assert sys.version_info[:2] == (3, 10)' >/dev/null 2>&1; then
    conda create -y -n "$geoscd_env" python=3.10
fi
conda env config vars set -n "$geoscd_env" PYTHONNOUSERSITE=1
conda run --no-capture-output -n "$geoscd_env" python -s -m pip install \
    torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
conda run --no-capture-output -n "$geoscd_env" python -s -m pip install \
    -r "$repo_root/configs/2026-10-04_geoscd_geometry.requirements.txt"
if "$full"; then
    # GeoSCD uses its bundled modified SAM; its predictor also imports this package.
    conda run --no-capture-output -n "$geoscd_env" python -s -m pip install \
        "git+https://github.com/facebookresearch/segment-anything.git@$sam_commit"
fi
conda run --no-capture-output -n "$geoscd_env" python -s -m pip install -e "$repo_root"
conda run --no-capture-output -n "$geoscd_env" python -s -m pip check

vggt_checkpoint="$repo_root/models/vggt_1b.pt"
legacy_vggt_checkpoint="$geoscd_root/src/pretrained/model.pt"
sam_checkpoint="$repo_root/models/sam_vit_h_4b8939.pth"
if [[ ! -f "$vggt_checkpoint" && -f "$legacy_vggt_checkpoint" ]]; then
    # Keep a previously downloaded large checkpoint in place instead of copying it.
    vggt_checkpoint="$legacy_vggt_checkpoint"
    echo "Reusing legacy VGGT checkpoint: $vggt_checkpoint"
fi

if "$download_weights"; then
    # This opt-in is the only download path; inference never fetches weights.
    conda run --no-capture-output -n "$geoscd_env" python -s - "$vggt_checkpoint" "$sam_checkpoint" "$full" <<'PY'
import sys
from pathlib import Path
import torch

checkpoints = [(Path(sys.argv[1]), 'https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt')]
if sys.argv[3] == 'true':
    checkpoints.append((Path(sys.argv[2]), 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'))
for path, url in checkpoints:
    if path.is_file():
        print(f'Reusing checkpoint: {path}')
        continue
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.download')
    torch.hub.download_url_to_file(
        url, str(temporary), progress=True,
    )
    temporary.replace(path)
    print(f'Downloaded checkpoint: {path}')
PY
fi
echo "Ready. Reactivate to apply persistent user-site isolation:"
echo "conda deactivate; conda activate $geoscd_env"
echo "GeoSCD source: $geoscd_root"
echo "VGGT checkpoint: $vggt_checkpoint"
if "$full"; then
    echo "SAM1 ViT-H checkpoint: $sam_checkpoint (sam3.pt is a different model and is not used)"
fi
if ! "$download_weights"; then
    echo 'Weights were not downloaded; pass --download-weights to fetch missing checkpoints.'
fi
