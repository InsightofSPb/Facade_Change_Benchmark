#!/usr/bin/env bash
# Add only missing inference dependencies to an existing benchmark environment.
set -euo pipefail

benchmark_env="scd_bench"
download_lpips_weights=false
while (($#)); do
    case "$1" in
        --env)
            (($# >= 2)) || { echo '--env requires an existing conda environment.' >&2; exit 2; }
            benchmark_env="$2"
            shift 2
            ;;
        --download-lpips-weights) download_lpips_weights=true; shift ;;
        --help|-h)
            echo 'Usage: bash scripts/setup_competitors.sh [--env scd_bench] [--download-lpips-weights]'
            echo 'Preserves installed Torch/torchvision/NumPy. Does not create environments or download DINO, RSCD, SAM or VGGT.'
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
command -v conda >/dev/null || { echo 'conda must be available in this shell.' >&2; exit 1; }
export PYTHONNOUSERSITE=1

conda run --no-capture-output -n "$benchmark_env" python -s - "$download_lpips_weights" <<'PY'
import hashlib
import importlib
from importlib import metadata
from pathlib import Path
import subprocess
import sys

if sys.version_info < (3, 10):
    raise SystemExit('Competitor workers require Python >= 3.10; use the existing scd_bench environment.')

# Read versions without importing torchvision: Torch Dynamo reaches SymPy,
# which cannot initialize when its mpmath dependency is absent.
preserved = {name: metadata.version(name) for name in ('torch', 'torchvision', 'numpy', 'Pillow')}
if tuple(int(part) for part in preserved['torch'].split('+')[0].split('.')[:2]) < (2, 6):
    raise SystemExit('RSCD safe checkpoint loading requires Torch >= 2.6; use scd_bench, not lposs.')

def install(requirement, no_deps=True):
    command = [sys.executable, '-s', '-m', 'pip', 'install']
    if no_deps:
        command.append('--no-deps')
    subprocess.run(command + [requirement], check=True)

try:
    importlib.import_module('mpmath')
except ModuleNotFoundError as exc:
    if exc.name != 'mpmath':
        raise
    install('mpmath==1.3.0')
    importlib.invalidate_caches()
    importlib.import_module('mpmath')

import torch
import torchvision
import numpy
import PIL
from torchvision.ops import batched_nms
batched_nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([1.]), torch.tensor([0]), 0.5)

try:
    lpips_version = metadata.version('lpips')
except metadata.PackageNotFoundError:
    lpips_version = None
if lpips_version not in (None, '0.1.4'):
    raise SystemExit(f'Existing LPIPS is {lpips_version}, expected 0.1.4. Select a compatible environment explicitly.')
if lpips_version is None:
    install('lpips==0.1.4')

# Do not reinstall working packages. requests has only HTTP dependencies and cannot pull Torch.
for module, requirement, no_deps in (
    ('scipy', 'scipy==1.14.1', True),
    ('tqdm', 'tqdm==4.67.1', True),
    ('requests', 'requests==2.32.3', False),
    ('safetensors', 'safetensors==0.5.3', True),
    ('einops', 'einops==0.8.1', True),
):
    try:
        importlib.import_module(module)
    except ModuleNotFoundError:
        install(requirement, no_deps=no_deps)
        importlib.invalidate_caches()
        importlib.import_module(module)

for module in ('skimage', 'lpips'):
    importlib.import_module(module)
after = {name: metadata.version(name) for name in preserved}
if after != preserved:
    raise RuntimeError(f'Core versions changed unexpectedly: before={preserved}, after={after}')
print('Preserved core versions:', preserved, flush=True)
print('CUDA available:', torch.cuda.is_available(), flush=True)

checkpoint = Path(torch.hub.get_dir()) / 'checkpoints/alexnet-owt-7be5be79.pth'
if sys.argv[1] == 'true' and not checkpoint.is_file():
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_name(checkpoint.name + '.download')
    if temporary.exists():
        raise SystemExit(f'An unfinished download already exists: {temporary}; keep it for inspection or move it before retrying.')
    try:
        torch.hub.download_url_to_file(
            'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',
            str(temporary), hash_prefix='7be5be79', progress=True,
        )
        temporary.replace(checkpoint)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
if checkpoint.is_file():
    digest = hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if not digest.hexdigest().startswith('7be5be79'):
        raise SystemExit(f'Unexpected AlexNet SHA-256: {checkpoint}; the existing file was preserved.')
    print('LPIPS backbone:', checkpoint, flush=True)
else:
    print('LPIPS AlexNet backbone is missing. Rerun setup with --download-lpips-weights to download it explicitly.', flush=True)
print('LPIPS calibration uses the official 0.1.4 package weights; model inference never downloads weights.', flush=True)
PY
