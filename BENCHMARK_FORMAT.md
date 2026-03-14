# Benchmark File Format

Use `--benchmark-file` to run the control-vs-sleep comparison on externally defined tasks instead of synthetic generated scenarios.

Supported formats:
- `.json`: list of game objects or `{ "games": [...] }`
- `.jsonl`: one turn object per line grouped by `(domain, game_id)`

## JSON Schema (Game-Oriented)

```json
{
  "games": [
    {
      "domain": "codenames_realish",
      "game_id": 1,
      "turns": [
        {
          "turn": 1,
          "tags": ["semantic:double_meaning", "team:misread_prone"],
          "candidates": ["safe_clue", "aggressive_multi", "double_meaning", "assassin_avoid", "teammate_model"],
          "best_action": "double_meaning",
          "difficulty": 0.72
        }
      ]
    }
  ]
}
```

Required per turn:
- `tags` (list of strings)
- `candidates` (list of strings)
- `best_action` (string, must be in `candidates`)

Optional per turn:
- `turn` (int; defaults to sequence index)
- `difficulty` (float; defaults to `0.7`)

## JSONL Schema (Turn-Oriented)

Each line is a JSON object with:
- `domain`
- `game_id`
- `turn`
- `tags`
- `candidates`
- `best_action`
- optional `difficulty`

## Run Examples

Single run:

```bash
python3 -m agentic_sleep.cli run \
  --benchmark-file examples/benchmark_minimal.json \
  --seed 7
```

Seed sweep:

```bash
python3 -m agentic_sleep.cli sweep \
  --benchmark-file examples/benchmark_minimal.json \
  --num-seeds 5 \
  --output-json results/benchmark_sweep.json \
  --output-tex generated/llm_results_tables.tex
```

## Converting Recorded Events into a Benchmark

If you collected `--record-events` traces from a real environment, use
`agentic_sleep.benchmarks.convert_trace_events_to_benchmark_games(...)` to convert
turn events into replayable benchmark games.

