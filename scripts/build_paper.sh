#!/usr/bin/env bash
set -euo pipefail

TEX_FILE="${1:-agentic_sleep_paper_draft.tex}"

if command -v latexmk >/dev/null 2>&1; then
  exec latexmk -pdf -interaction=nonstopmode "$TEX_FILE"
fi

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "pdflatex not found. Run ./scripts/check_latex_env.sh for install instructions." >&2
  exit 1
fi

pdflatex -interaction=nonstopmode -halt-on-error "$TEX_FILE"
pdflatex -interaction=nonstopmode -halt-on-error "$TEX_FILE"

