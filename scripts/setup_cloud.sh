#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m pip install -r requirements-cpu.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
