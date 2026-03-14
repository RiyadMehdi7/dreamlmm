PYTHON ?= python3
PAPER_TEX := agentic_sleep_paper_draft.tex
PAPER_PDF := agentic_sleep_paper_draft.pdf

.PHONY: help run sweep paper paper-latexmk clean latex-check render-tables

help:
	@echo "Targets:"
	@echo "  run           Run single simulated experiment"
	@echo "  sweep         Run seed sweep and generate LaTeX result tables"
	@echo "  render-tables Render LaTeX fragment from saved JSON"
	@echo "  paper         Build paper PDF (pdflatex fallback)"
	@echo "  paper-latexmk Build paper PDF with latexmk (preferred)"
	@echo "  latex-check   Check local LaTeX toolchain availability"
	@echo "  clean         Remove common LaTeX artifacts"

run:
	$(PYTHON) -m agentic_sleep.cli --games 20 --turns 50

sweep:
	mkdir -p results generated
	$(PYTHON) -m agentic_sleep.cli sweep --games 20 --turns 50 --num-seeds 10 \
	  --output-json results/sweep.json \
	  --output-tex generated/llm_results_tables.tex

render-tables:
	$(PYTHON) -m agentic_sleep.cli render-tex \
	  --input-json results/sweep.json \
	  --output-tex generated/llm_results_tables.tex

paper:
	pdflatex -interaction=nonstopmode -halt-on-error $(PAPER_TEX)
	pdflatex -interaction=nonstopmode -halt-on-error $(PAPER_TEX)

paper-latexmk:
	latexmk -pdf -interaction=nonstopmode $(PAPER_TEX)

latex-check:
	./scripts/check_latex_env.sh

clean:
	rm -f *.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.out *.toc *.synctex.gz $(PAPER_PDF)

