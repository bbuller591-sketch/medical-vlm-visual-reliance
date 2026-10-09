#!/usr/bin/env bash
# This script requires an authenticated gh CLI and a repo name available to its owner.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v gh >/dev/null || { echo 'Install GitHub CLI first'; exit 1; }
gh auth status >/dev/null 2>&1 || { echo 'Run gh auth login first'; exit 2; }
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then git init -b main; fi
git add README.md docs scripts tests configs results figures requirements-analysis.txt requirements-inference.txt .gitignore
if ! git diff --cached --quiet; then git commit -m 'Release medical VLM visual reliance reproducibility package'; fi
gh repo create medical-vlm-visual-reliance --public --source . --remote origin --push
