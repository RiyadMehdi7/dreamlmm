# Agentic Sleep Prototype

A stdlib-only prototype of the "Agentic Sleep" framework described in the prompt:

- Entropy-guided trigger (`ERGO`-style delta entropy)
- 3-tier memory (STIM / MTEM / LTSM)
- Consolidation from episodic traces into heuristic rules + a lightweight knowledge graph
- Dream phase for failure-seeded synthetic rule hardening
- Control vs Sleep experiment harness on long-horizon logic-game-like tasks

## Run (single)

```bash
python -m agentic_sleep.cli --games 20 --turns 50
```

## Seed Sweep (for publishable stats)

```bash
python -m agentic_sleep.cli sweep --games 20 --turns 50 --num-seeds 10 \
  --output-json results/sweep.json \
  --output-tex results/sweep_tables.tex
```

## External Benchmarks (JSON/JSONL)

Run on a real/custom benchmark file instead of synthetic generated tasks:

```bash
python -m agentic_sleep.cli run --benchmark-file examples/benchmark_minimal.json
python -m agentic_sleep.cli sweep --benchmark-file examples/benchmark_minimal.json --num-seeds 5
```

Schema details: `BENCHMARK_FORMAT.md`

## Optional LLM Policy (OpenAI Responses API)

Set an API key, then run with `--policy openai`:

```bash
export OPENAI_API_KEY=...
python -m agentic_sleep.cli run --policy openai --model gpt-4.1-mini \
  --games 5 --turns 20
```

The memory/sleep pipeline remains unchanged; only the waking decision policy is replaced.

## Render LaTeX Tables from JSON

```bash
python -m agentic_sleep.cli render-tex \
  --input-json results/sweep.json \
  --output-tex generated/llm_results_tables.tex
```

## Paper Build (LaTeX)

```bash
make latex-check
make paper        # pdflatex fallback
make paper-latexmk
```

## Notes

- This is a research prototype / simulation scaffold, not a production LLM integration.
- It now supports optional LLM-backed waking policy calls (`--policy openai`) while keeping the same sleep/consolidation loop.
- The flow is LangGraph-inspired, with node-like functions and conditional routing.
