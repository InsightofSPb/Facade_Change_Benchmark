#!/usr/bin/env bash
# Keep author sources byte-identical; use a separate Python environment.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash scripts/setup_bcm_codec.sh [--sources-only] [--skip-vtm-build]

Clone and verify pinned BCM-Net and original VTM-15.0 sources. By default,
create/reuse the separate facade-bcm environment and build unmodified 8-bit VTM.
No checkpoints, datasets, training or benchmark are downloaded/launched.
--sources-only verifies sources without conda/package changes or compilation.
--skip-vtm-build prepares sources/environment without compiling VTM.

Environment variables:
  BCM_ENV          Separate conda environment (default: facade-bcm)
  BCM_SOURCE_DIR   Author checkout (default: third_party/bcm_net)
  BCM_VTM_DIR      VTM checkout (default: third_party/vtm)
  BCM_BUILD_JOBS   Positive build parallelism (default: 2)
  MAX_JOBS         torchac extension build parallelism (default: 2)
  BCM_VTM_REPOSITORY  Explicit VTM source URL; default is original HHI.
                     Allowed fallback: https://github.com/ffvvc/VVCSoftware_VTM.git

Requires Linux/WSL, git, python3; default setup also requires conda, cmake and g++.
Existing checkouts/environments are never reset or upgraded. No source patches.
HELP
}
BCM_SOURCES_ONLY=0
BCM_SKIP_BUILD=0
for argument in "$@"; do
    case "$argument" in
        --help|-h) usage; exit 0 ;;
        --sources-only) BCM_SOURCES_ONLY=1; BCM_SKIP_BUILD=1 ;;
        --skip-vtm-build) BCM_SKIP_BUILD=1 ;;
        *) usage >&2; exit 2 ;;
    esac
done
BCM_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
BCM_ENV="${BCM_ENV:-facade-bcm}"
BCM_SOURCE_DIR="${BCM_SOURCE_DIR:-$BCM_REPO_ROOT/third_party/bcm_net}"
BCM_VTM_DIR="${BCM_VTM_DIR:-$BCM_REPO_ROOT/third_party/vtm}"
BCM_BUILD_JOBS="${BCM_BUILD_JOBS:-2}"
MAX_JOBS="${MAX_JOBS:-2}"
BCM_VTM_REPOSITORY="${BCM_VTM_REPOSITORY:-https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git}"
BCM_COMMIT=6ba35a99228d03bd0831501d79c62715c06ded2e
BCM_VTM_COMMIT=2e935808159fefa3c85a599c3b79615cd9555429
BCM_TEMP=""
fail() { echo "BCM setup: $*" >&2; exit 1; }
cleanup() { [[ -z "$BCM_TEMP" ]] || rm -rf -- "$BCM_TEMP"; }
trap cleanup EXIT
[[ "$BCM_ENV" =~ ^[A-Za-z0-9_-]+$ ]] || fail 'BCM_ENV must be a conda environment name.'
case "$BCM_ENV" in
    base|scd_bench|lposs|facade-codecs|facade-geoscd) fail "Refusing package changes in $BCM_ENV." ;;
esac
[[ "$BCM_BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || fail 'BCM_BUILD_JOBS must be a positive integer.'
[[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]] || fail 'MAX_JOBS must be a positive integer.'
case "$BCM_VTM_REPOSITORY" in
    https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git|https://github.com/ffvvc/VVCSoftware_VTM.git) ;;
    *) fail 'VTM source must be the original HHI repository or the documented ffvvc mirror.' ;;
esac
for executable in git python3; do
    command -v "$executable" >/dev/null || fail "$executable is required."
done
export PYTHONNOUSERSITE=1
export MAX_JOBS
unset PYTHONPATH
BCM_MANIFEST="$BCM_REPO_ROOT/third_party/bcm_net_provenance.json"
[[ -f "$BCM_MANIFEST" ]] || fail 'Pinned BCM provenance manifest is missing.'

checkout_source() {
    local repository="$1" directory="$2" commit="$3" branch="${4:-}"
    if [[ ! -e "$directory" ]]; then
        mkdir -p "$(dirname "$directory")"
        BCM_TEMP="$(mktemp -d "$(dirname "$directory")/.bcm-source.XXXXXX")"
        if [[ -n "$branch" ]]; then
            git -c core.autocrlf=false clone --depth 1 --branch "$branch" --single-branch \
                "$repository" "$BCM_TEMP/source" || fail 'VTM download failed; set the documented mirror explicitly if HHI is unavailable.'
        else
            git -c core.autocrlf=false clone --no-checkout "$repository" "$BCM_TEMP/source" || fail 'BCM author download failed.'
            git -C "$BCM_TEMP/source" config --local core.autocrlf false
            git -C "$BCM_TEMP/source" checkout --detach "$commit" || fail 'Pinned BCM commit is unavailable.'
        fi
        [[ ! -e "$directory" ]] || fail "Directory appeared during setup: $directory."
        mv -T "$BCM_TEMP/source" "$directory"
        rm -rf -- "$BCM_TEMP"
        BCM_TEMP=""
    fi
    [[ -d "$directory" ]] || fail "Source path is not a directory: $directory."
    local actual_top expected_top actual_head
    actual_top="$(git -C "$directory" rev-parse --show-toplevel)" || fail 'Source must be an independent Git checkout.'
    expected_top="$(cd "$directory" && pwd -P)"
    [[ "$(cd "$actual_top" && pwd -P)" == "$expected_top" ]] || fail 'Source path belongs to another repository.'
    actual_head="$(git -C "$directory" rev-parse HEAD)"
    [[ "$actual_head" == "$commit" ]] || fail "Unexpected source commit $actual_head; expected $commit. Checkout was preserved."
    git -C "$directory" diff --quiet HEAD -- || fail 'Tracked author sources have changes; checkout was preserved.'
}
checkout_source https://github.com/LiuXiangrui/BCM-Net.git "$BCM_SOURCE_DIR" "$BCM_COMMIT"
checkout_source "$BCM_VTM_REPOSITORY" "$BCM_VTM_DIR" "$BCM_VTM_COMMIT" VTM-15.0
BCM_SOURCE_DIR="$(cd "$BCM_SOURCE_DIR" && pwd -P)"
BCM_VTM_DIR="$(cd "$BCM_VTM_DIR" && pwd -P)"
python3 -s - "$BCM_MANIFEST" "$BCM_SOURCE_DIR" "$BCM_VTM_DIR" <<'PYVERIFY'
import hashlib, json, pathlib, subprocess, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
for section, directory in ((manifest, sys.argv[2]), (manifest['vtm'], sys.argv[3])):
    root = pathlib.Path(directory).resolve()
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if head != section['commit']:
        raise SystemExit('Source commit does not match provenance')
    for relative, expected in section['sha256'].items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise SystemExit(f'Missing or unsafe source: {relative}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f'Author bytes changed: {relative}')
    print('Verified:', head, f"({len(section['sha256'])} SHA-256 checks)")
PYVERIFY
if [[ "$BCM_SOURCES_ONLY" == 1 ]]; then
    echo 'Source verification complete; no environment changes or compilation.'
    exit 0
fi
command -v conda >/dev/null || fail 'conda is required for the separate facade-bcm environment.'
command -v g++ >/dev/null || fail 'g++ is required for the author torchac extension.'
if [[ "$BCM_SKIP_BUILD" == 0 ]]; then
    command -v cmake >/dev/null || fail 'cmake is required to compile VTM.'
fi
BCM_ENV_JSON="$(conda env list --json)"
BCM_ENV_EXISTS="$(python3 -s - "$BCM_ENV" "$BCM_ENV_JSON" <<'PYENV'
import json, pathlib, sys
print(int(any(pathlib.Path(path).name == sys.argv[1] for path in json.loads(sys.argv[2])['envs'])))
PYENV
)"
if [[ "$BCM_ENV_EXISTS" == 0 ]]; then
    conda create -y -n "$BCM_ENV" python=3.10 numpy=1.26.4 pip ninja || fail 'Separate environment creation failed.'
fi
bcm_python() { conda run --no-capture-output -n "$BCM_ENV" python -s "$@"; }
BCM_MISSING="$(bcm_python - <<'PYDEPS'
import importlib.metadata, sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit('Existing BCM environment must use Python3.10; it was not upgraded')
required = {'torch': '1.13.1+cu117', 'numpy': '1.26.4', 'einops': '0.6.1', 'torchac': '0.9.3', 'tqdm': '4.65.0', 'Pillow': '11.3.0'}
for name, version in required.items():
    try:
        actual = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        print(name)
        continue
    if actual != version:
        raise SystemExit(f'Existing {name}={actual}, expected {version}; setup will not replace it')
PYDEPS
)" || fail 'Existing separate environment versions disagree; no packages were replaced.'
if [[ "$BCM_MISSING" == *torch* ]]; then
    # Only torch itself is needed by the author network; torchvision/audio are unused.
    if ! bcm_python -c 'import importlib.metadata; importlib.metadata.version("torch")' >/dev/null 2>&1; then
        bcm_python -m pip install 'torch==1.13.1+cu117' --extra-index-url https://download.pytorch.org/whl/cu117
    fi
fi
BCM_HELPERS=()
while IFS= read -r dependency; do
    case "$dependency" in
        numpy) BCM_HELPERS+=('numpy==1.26.4') ;;
        einops) BCM_HELPERS+=('einops==0.6.1') ;;
        torchac) BCM_HELPERS+=('torchac==0.9.3') ;;
        tqdm) BCM_HELPERS+=('tqdm==4.65.0') ;;
        Pillow) BCM_HELPERS+=('Pillow==11.3.0') ;;
        torch|'') ;;
        *) fail "Unexpected missing dependency: $dependency." ;;
    esac
done <<< "$BCM_MISSING"
if (( ${#BCM_HELPERS[@]} )); then
    bcm_python -m pip install --no-deps "${BCM_HELPERS[@]}"
fi
bcm_python - "$BCM_SOURCE_DIR" "$BCM_REPO_ROOT" <<'PYIMPORT'
import sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
import torch, numpy, einops, torchac, PIL
from Network import Network
import facade_change.cli
print('BCM Python:', sys.executable)
print('torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('Author Network, torchac and facade CLI imports passed; no weights loaded')
PYIMPORT
if [[ "$BCM_SKIP_BUILD" == 0 ]]; then
    for executable in cmake g++; do
        command -v "$executable" >/dev/null || fail "$executable is required to compile VTM."
    done
    # Default TypeDef.h is unchanged: 8/10-bit build, no 16-bit source patch.
    # GCC13 no longer supplies <cstdint> transitively. Include the standard
    # header through the compiler; keep every VTM source byte unchanged.
    BCM_VTM_CXX_FLAGS='-include cstdint -Wno-error=deprecated-declarations -Wno-error=address'
    BCM_GCC_MAJOR="$(g++ -dumpversion)"
    BCM_GCC_MAJOR="${BCM_GCC_MAJOR%%.*}"
    if [[ "$BCM_GCC_MAJOR" =~ ^[0-9]+$ ]] && (( BCM_GCC_MAJOR >= 12 )); then
        # GCC12 added a warning for original IntraSearch array comparison.
        BCM_VTM_CXX_FLAGS+=' -Wno-error=array-compare'
    fi
    cmake -S "$BCM_VTM_DIR" -B "$BCM_VTM_DIR/build" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="$BCM_VTM_CXX_FLAGS"
    cmake --build "$BCM_VTM_DIR/build" --parallel "$BCM_BUILD_JOBS" --target EncoderApp DecoderApp
fi
echo "BCM source: $BCM_SOURCE_DIR"
echo "VTM source/configs: $BCM_VTM_DIR"
echo "Worker Python: conda run -n $BCM_ENV python -s"
echo 'Find built EncoderApp*/DecoderApp* in third_party/vtm/bin; pass their absolute paths to the adapter.'
echo 'Use --AccessUnitDelimiter=1 for two-frame access-unit accounting (adapter sets it).'
echo 'Use --TemporalFilterFutureReference=0 for causal B|A (adapter sets it without editing RA configs).'
echo 'No pretrained checkpoints downloaded. Author folder:'
echo 'https://drive.google.com/drive/folders/1Ogi8ZKouMTsHS59nr_xdu1wQIEGhFbuB?usp=sharing'
