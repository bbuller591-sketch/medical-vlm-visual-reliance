#!/usr/bin/env bash
# Rebuild the clickable, self-contained PDF report from the authoritative Markdown.
# Requires Pandoc >= 3.x and Tectonic (or paths through PANDOC_BIN, TECTONIC_BIN).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT="$(cd "$ROOT/.." && pwd)"
PANDOC_BIN="${PANDOC_BIN:-$(command -v pandoc || true)}"
TECTONIC_BIN="${TECTONIC_BIN:-$(command -v tectonic || true)}"
if [[ -z "$PANDOC_BIN" && -x "$PARENT/latex_conversion_tools/pandoc-3.12.1/bin/pandoc" ]]; then
 PANDOC_BIN="$PARENT/latex_conversion_tools/pandoc-3.12.1/bin/pandoc"
fi
if [[ -z "$TECTONIC_BIN" && -x "$PARENT/latex_conversion_tools/tectonic_musl/tectonic" ]]; then
 TECTONIC_BIN="$PARENT/latex_conversion_tools/tectonic_musl/tectonic"
fi
[[ -x "$PANDOC_BIN" ]] || { echo 'Pandoc executable unavailable; set PANDOC_BIN' >&2; exit 2; }
[[ -x "$TECTONIC_BIN" ]] || { echo 'Tectonic executable unavailable; set TECTONIC_BIN' >&2; exit 2; }
TEMP_MD="$(mktemp "$ROOT/.tex_source_XXXXXXXX.md")"
trap 'rm -f "$TEMP_MD"' EXIT
python - "$ROOT/MEDICAL_VLM_PROGRESS_REPORT.md" "$TEMP_MD" <<'PY'
from pathlib import Path
import sys
src=Path(sys.argv[1]).read_text(encoding='utf-8')
lines=src.splitlines()
assert lines[0].startswith('# Visual Reliance and Post-Training Gains in Medical VLMs')
Path(sys.argv[2]).write_text('\n'.join(['']+lines[1:])+'\n',encoding='utf-8')
PY
cd "$ROOT"
"$PANDOC_BIN" "$TEMP_MD" --from=markdown+tex_math_dollars+pipe_tables --to=latex --standalone --toc --toc-depth=3 --shift-heading-level-by=-1 \
 --metadata=title:'Visual Reliance and Post-Training Gains in Medical VLMs' \
 --metadata=author:'Research Progress Report' --metadata=date:'9 October 2026' \
 -V geometry:margin=0.80in -V fontsize:10pt -V colorlinks=true -V linkcolor=blue -V urlcolor=blue \
 -V toc-title='Contents' -V papersize=a4 -o MEDICAL_VLM_PROGRESS_REPORT.tex
# Keep the clickable table of contents on its own page instead of starting Part 0 at its foot.
python - <<'PYTHON_TEX'
from pathlib import Path
p=Path('MEDICAL_VLM_PROGRESS_REPORT.tex')
s=p.read_text()
assert s.count(r'\tableofcontents')==1
s=s.replace(r'\tableofcontents', r'\tableofcontents' + '\n\\clearpage', 1)
p.write_text(s)
PYTHON_TEX
"$TECTONIC_BIN" --keep-logs MEDICAL_VLM_PROGRESS_REPORT.tex
if command -v pdfinfo >/dev/null 2>&1; then pdfinfo MEDICAL_VLM_PROGRESS_REPORT.pdf | grep '^Pages:'; fi
echo "LaTeX: $ROOT/MEDICAL_VLM_PROGRESS_REPORT.tex"
echo "PDF: $ROOT/MEDICAL_VLM_PROGRESS_REPORT.pdf"
