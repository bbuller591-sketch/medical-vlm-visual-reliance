#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v gh >/dev/null || { echo 'Install GitHub CLI first'; exit 1; }
gh auth status >/dev/null 2>&1 || { echo 'Run gh auth login first'; exit 2; }
gh repo create medical-vlm-visual-reliance --public --source . --remote origin --push
