#!/usr/bin/env bash
set -euo pipefail
# Execute from any location; no developer-specific absolute paths.
cd "$(dirname "$0")/.."
if command -v conda >/dev/null 2>&1; then
  conda create -n med-vlm-repro python=3.11 -y
  echo "Run: conda activate med-vlm-repro"
  echo "Then install the CUDA-specific PyTorch build suitable for this machine"
  echo "Then run: python -m pip install -r requirements-inference.txt -r requirements-download.txt"
else
  echo "conda not found; create a Python 3.11 virtualenv manually." >&2
  exit 1
fi
