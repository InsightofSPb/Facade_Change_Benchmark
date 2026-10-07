#!/usr/bin/env bash
# Build the author's coder in a separate environment; never alter author sources.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash scripts/setup_neural_codecs.sh [--help]

Clone scd_bench into facade-codecs only when facade-codecs does not exist.
Verify inherited Python/torch/numpy/Pillow, then install only
missing pybind11==2.13.6 and tqdm==4.67.1 in the separate codec environment.
Download and verify the pinned ArIB-BPS author sources; compile its entropy coder.
No pretrained weights are downloaded and no training or benchmark is launched.

Optional environment variables:
  CODECS_ENV       Separate target environment name (default: facade-codecs)
  CODECS_BASE_ENV  Existing environment to clone (default: scd_bench)
  ARIB_SOURCE_DIR  Author checkout (default: third_party/arib_bps in this repo)

Existing environments/checkouts are inspected, never reset or upgraded.
Requires conda, git and g++; network access is needed only for missing sources
or the two missing helper packages. Run from a Linux/WSL shell.
HELP
}
for argument in "$@"; do
    case "$argument" in
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

CODECS_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
CODECS_ENV="${CODECS_ENV:-facade-codecs}"
CODECS_BASE_ENV="${CODECS_BASE_ENV:-scd_bench}"
ARIB_SOURCE_DIR="${ARIB_SOURCE_DIR:-$CODECS_REPO_ROOT/third_party/arib_bps}"
ARIB_COMMIT=bdc3c61d7cf3377dc72d411a32713f5f353d61e8
ARIB_REPOSITORY=https://github.com/ZZ022/ArIB-BPS

fail() { echo "Neural codec setup: $*" >&2; exit 1; }
[[ "$CODECS_ENV" =~ ^[A-Za-z0-9_-]+$ && "$CODECS_BASE_ENV" =~ ^[A-Za-z0-9_-]+$ ]] || \
    fail 'CODECS_ENV and CODECS_BASE_ENV must be environment names, not paths.'
[[ "$CODECS_ENV" != "$CODECS_BASE_ENV" ]] || fail 'Target environment must differ from the environment being cloned.'
case "$CODECS_ENV" in
    base|scd_bench|lposs|facade-geoscd) fail "Refusing package changes in protected environment $CODECS_ENV." ;;
esac
for executable in conda git g++; do
    command -v "$executable" >/dev/null || fail "$executable is required in this shell."
done
[[ -f "$CODECS_REPO_ROOT/third_party/arib_bps_provenance.json" ]] || fail 'Pinned third_party/arib_bps_provenance.json is missing.'
# Disable user-site packages for every worker Python, including a rerun.
export PYTHONNOUSERSITE=1
codec_python() { conda run --no-capture-output -n "$CODECS_ENV" python -s "$@"; }

if ! CODECS_ENV_JSON="$(conda env list --json)"; then
    fail 'Unable to list conda environments; no environment was modified.'
fi
if ! CODECS_ENV_STATE="$(conda run --no-capture-output -n base python -s -c '
import json, pathlib, sys
names = {pathlib.Path(path).name for path in json.loads(sys.argv[3])["envs"]}
names.add("base")  # The base prefix is commonly named miniconda3/anaconda3.
target, source = sys.argv[1:3]
if target in names:
    print("present")
elif source in names:
    print("missing")
else:
    raise SystemExit(f"Source conda environment {source!r} does not exist")
' "$CODECS_ENV" "$CODECS_BASE_ENV" "$CODECS_ENV_JSON")"; then
    fail 'Unable to resolve the target/source environment; inspect the conda error above.'
fi
case "$CODECS_ENV_STATE" in
    present) echo "Reusing separate environment: $CODECS_ENV" ;;
    missing)
        conda create -y -n "$CODECS_ENV" --clone "$CODECS_BASE_ENV" || \
            fail "Failed to clone $CODECS_BASE_ENV into $CODECS_ENV; the source environment was not changed."
        ;;
    *) fail "Unexpected conda environment state: $CODECS_ENV_STATE" ;;
esac

if ! codec_python - <<'PYREQUIRED'
import importlib, sys
if sys.version_info < (3, 9):
    raise SystemExit("Python >=3.9 is required; the existing environment was not upgraded")
for name in ("torch", "numpy", "PIL"):
    try:
        module = importlib.import_module(name)
    except Exception as error:
        raise SystemExit(f"Required inherited dependency {name!r} cannot import: {type(error).__name__}: {error}")
    print(f"{name}: {getattr(module, '__version__', 'unknown')}")
print("Codec worker Python:", sys.executable)
PYREQUIRED
then
    fail "Inherited dependencies are incomplete or broken in $CODECS_ENV. Repair that separate environment explicitly; setup will not replace torch or change $CODECS_BASE_ENV."
fi
if ! CODECS_MISSING_HELPERS="$(codec_python - <<'PYHELPERS'
import importlib, importlib.util
for name, requirement in (("pybind11", "pybind11==2.13.6"), ("tqdm", "tqdm==4.67.1")):
    if importlib.util.find_spec(name) is None:
        print(requirement)
    else:
        try:
            importlib.import_module(name)
        except Exception as error:
            raise SystemExit(f"Existing helper {name!r} cannot import: {type(error).__name__}: {error}; it was not replaced")
PYHELPERS
)"; then
    fail "Existing codec helpers are broken in $CODECS_ENV; no helper package was replaced."
fi
if [[ -n "$CODECS_MISSING_HELPERS" ]]; then
    mapfile -t CODECS_HELPER_PACKAGES <<< "$CODECS_MISSING_HELPERS"
    codec_python -m pip install --no-deps "${CODECS_HELPER_PACKAGES[@]}" || \
        fail "Unable to install missing helper packages into $CODECS_ENV. Check the pip/network error above; no base packages were changed."
fi
codec_python -c 'import pybind11, tqdm; print("pybind11:", pybind11.__version__, "tqdm:", tqdm.__version__)' || \
    fail "Helper import verification failed in $CODECS_ENV."

CODECS_CLONE_TEMP=""
CODECS_BUILD_TEMP=""
cleanup() {
    [[ -z "$CODECS_BUILD_TEMP" ]] || rm -f -- "$CODECS_BUILD_TEMP"
    [[ -z "$CODECS_CLONE_TEMP" ]] || rm -rf -- "$CODECS_CLONE_TEMP"
}
trap cleanup EXIT
if [[ ! -e "$ARIB_SOURCE_DIR" ]]; then
    mkdir -p "$(dirname "$ARIB_SOURCE_DIR")"
    CODECS_CLONE_TEMP="$(mktemp -d "$(dirname "$ARIB_SOURCE_DIR")/.arib-clone.XXXXXX")"
    git -c core.autocrlf=false clone --no-checkout "$ARIB_REPOSITORY.git" "$CODECS_CLONE_TEMP/source" || \
        fail 'Failed to download ArIB-BPS author sources; check the git/network error above.'
    # Override a global Windows/WSL newline setting before materializing blobs.
    git -C "$CODECS_CLONE_TEMP/source" config --local core.autocrlf false || \
        fail 'Unable to disable newline conversion in the new author checkout.'
    git -C "$CODECS_CLONE_TEMP/source" checkout --detach "$ARIB_COMMIT" || \
        fail "Downloaded ArIB-BPS does not provide pinned commit $ARIB_COMMIT."
    [[ ! -e "$ARIB_SOURCE_DIR" ]] || fail "Source directory appeared during setup: $ARIB_SOURCE_DIR."
    mv -T "$CODECS_CLONE_TEMP/source" "$ARIB_SOURCE_DIR"
fi
[[ -d "$ARIB_SOURCE_DIR" ]] || fail "Author source is not a directory: $ARIB_SOURCE_DIR."
ARIB_SOURCE_DIR="$(cd "$ARIB_SOURCE_DIR" && pwd -P)"
if ! CODECS_SOURCE_TOP="$(git -C "$ARIB_SOURCE_DIR" rev-parse --show-toplevel)"; then
    fail "Author source is not a Git checkout: $ARIB_SOURCE_DIR. Existing files were left untouched."
fi
[[ "$(cd "$CODECS_SOURCE_TOP" && pwd -P)" == "$ARIB_SOURCE_DIR" ]] || fail 'Author source must be its own checkout, not a directory in another repository.'
if ! CODECS_SOURCE_HEAD="$(git -C "$ARIB_SOURCE_DIR" rev-parse HEAD)"; then
    fail 'Cannot read the author source commit.'
fi
[[ "$CODECS_SOURCE_HEAD" == "$ARIB_COMMIT" ]] || fail "Expected ArIB-BPS commit $ARIB_COMMIT; found $CODECS_SOURCE_HEAD. Existing checkout was not reset."
git -C "$ARIB_SOURCE_DIR" diff --quiet HEAD -- || fail 'Tracked author sources have changes; existing checkout was not reset.'
if ! codec_python - "$CODECS_REPO_ROOT" "$ARIB_SOURCE_DIR" "$ARIB_COMMIT" "$ARIB_REPOSITORY" <<'PYSOURCES'
import hashlib, json, pathlib, sys
repo, root = map(pathlib.Path, sys.argv[1:3])
manifest = json.loads((repo / "third_party/arib_bps_provenance.json").read_text())
if manifest["commit"] != sys.argv[3] or manifest["repository"].rstrip("/") != sys.argv[4]:
    raise SystemExit("Author provenance manifest disagrees with the pinned repository/commit")
for relative, expected in manifest["sha256"].items():
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise SystemExit(f"Missing or unsafe author source: {relative}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"Original ArIB source differs: {relative}")
print("Verified original ArIB source:", manifest["commit"], f"({len(manifest['sha256'])} files)")
PYSOURCES
then
    fail 'Pinned author source verification failed; compilation was not started.'
fi
if ! CODECS_INCLUDE_PATHS="$(codec_python - <<'PYINCLUDES'
import pybind11, sysconfig
for path in dict.fromkeys((pybind11.get_include(), sysconfig.get_path("include"), sysconfig.get_path("platinclude"))):
    if path:
        print(path)
PYINCLUDES
)"; then
    fail 'Unable to obtain C++ include paths from the codec worker Python.'
fi
CODECS_INCLUDES=()
while IFS= read -r CODECS_INCLUDE_PATH; do
    [[ -z "$CODECS_INCLUDE_PATH" ]] || CODECS_INCLUDES+=("-I$CODECS_INCLUDE_PATH")
done <<< "$CODECS_INCLUDE_PATHS"
CODECS_CODER_DIR="$ARIB_SOURCE_DIR/src/utils/coder"
CODECS_BUILD_TEMP="$(mktemp "$CODECS_CODER_DIR/.mixcoder-build.XXXXXX.so")"
# The upstream header omits <cstdint>; force-include it without editing the file.
g++ -O3 -Wall -shared -std=c++11 -include cstdint -fPIC "${CODECS_INCLUDES[@]}" \
    "$CODECS_CODER_DIR/python_interface.cpp" -o "$CODECS_BUILD_TEMP" || \
    fail 'Author entropy coder build failed; the existing mixcoder.so was preserved. Inspect the compiler error above.'
if ! codec_python - "$CODECS_BUILD_TEMP" <<'PYCODER'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("mixcoder", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if not all(hasattr(module, name) for name in ("MixEncoder", "MixDecoder")):
    raise SystemExit("Built author coder does not expose MixEncoder/MixDecoder")
PYCODER
then
    fail 'Built author coder cannot import in the codec worker Python; the existing mixcoder.so was preserved.'
fi
mv -f "$CODECS_BUILD_TEMP" "$CODECS_CODER_DIR/mixcoder.so"
CODECS_BUILD_TEMP=""
if ! codec_python - "$ARIB_SOURCE_DIR/src" <<'PYMODEL'
import sys
sys.path.insert(0, sys.argv[1])
from modules.arib_bps import ARIB_BPS, SIG, INS
print("Author SIG/INS networks and entropy coder import successfully")
PYMODEL
then
    fail 'Author networks cannot import in the codec worker; inspect the dependency error above.'
fi
echo "Ready: $CODECS_ENV; original ArIB source: $ARIB_SOURCE_DIR"
echo "Worker command: conda run --no-capture-output -n $CODECS_ENV python -s"
echo 'Setup verifies imports/build only; lossless round trips are required after H0 training.'
