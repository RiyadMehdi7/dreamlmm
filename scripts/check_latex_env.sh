#!/usr/bin/env bash
set -euo pipefail

echo "Checking LaTeX toolchain..."

for cmd in pdflatex bibtex latexmk; do
  if command -v "$cmd" >/dev/null 2>&1; then
    echo "  [ok] $cmd -> $(command -v "$cmd")"
  else
    echo "  [missing] $cmd"
  fi
done

echo
echo "If missing on macOS (Homebrew):"
echo "  brew install --cask basictex"
echo "  sudo tlmgr update --self"
echo "  sudo tlmgr install latexmk collection-fontsrecommended collection-latexrecommended collection-latexextra"
echo
echo "Alternative (larger install):"
echo "  brew install --cask mactex-no-gui"

