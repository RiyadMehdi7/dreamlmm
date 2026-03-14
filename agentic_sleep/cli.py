from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

from .experiment import run_comparison_experiment, run_seed_sweep_experiment
from .paper_tables import render_latex_results_fragment
from .policy import PolicySpec


def _add_policy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy", choices=["sim", "openai"], default="sim", help="Decision policy backend")
    parser.add_argument("--model", default="gpt-4.1-mini", help="LLM model name (for --policy openai)")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument("--max-output-tokens", type=int, default=300, help="LLM max output tokens")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Env var holding provider API key")
    parser.add_argument("--base-url", default="https://api.openai.com/v1/responses", help="Responses API URL")
    parser.add_argument("--timeout-seconds", type=float, default=60.0, help="HTTP timeout for LLM calls")


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--games", type=int, default=20, help="Games per domain")
    parser.add_argument("--turns", type=int, default=50, help="Turns per game")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--benchmark-file", help="JSON/JSONL benchmark file (overrides synthetic scenario generation)")
    parser.add_argument("--record-events", action="store_true", help="Capture per-turn and sleep-cycle events")
    parser.add_argument("--output-json", help="Write full result JSON to path")
    parser.add_argument("--output-tex", help="Write auto-generated LaTeX result fragment to path")
    _add_policy_args(parser)


def _build_policy_spec(args: argparse.Namespace) -> PolicySpec | None:
    if args.policy == "sim":
        return None
    return PolicySpec(
        kind=args.policy,
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        timeout_seconds=args.timeout_seconds,
    )


def _write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _write_json(path: str | Path, obj: Any) -> None:
    payload = asdict(obj) if is_dataclass(obj) else obj
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def _parse_seeds(args: argparse.Namespace) -> list[int]:
    if getattr(args, "seeds", None):
        return [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    start = int(args.seed_start)
    count = int(args.num_seeds)
    return [start + i for i in range(count)]


def _print_and_save(result: Any, output_json: str | None, output_tex: str | None) -> None:
    payload = asdict(result) if is_dataclass(result) else result
    print(json.dumps(payload, indent=2, sort_keys=True))
    if output_json:
        _write_json(output_json, payload)
    if output_tex:
        _write_text(output_tex, render_latex_results_fragment(payload))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic Sleep research runner")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run a single control-vs-sleep experiment")
    _add_run_args(run_p)

    sweep_p = sub.add_parser("sweep", help="Run repeated-seed experiments with summary statistics")
    sweep_p.add_argument("--games", type=int, default=20, help="Games per domain")
    sweep_p.add_argument("--turns", type=int, default=50, help="Turns per game")
    sweep_p.add_argument("--benchmark-file", help="JSON/JSONL benchmark file (overrides synthetic scenario generation)")
    sweep_p.add_argument("--seeds", help="Comma-separated seed list (e.g., 7,8,9)")
    sweep_p.add_argument("--seed-start", type=int, default=7, help="Start seed if --seeds omitted")
    sweep_p.add_argument("--num-seeds", type=int, default=5, help="Count if --seeds omitted")
    sweep_p.add_argument("--record-events", action="store_true", help="Capture events for each run (large)")
    sweep_p.add_argument("--output-json", help="Write sweep JSON to path")
    sweep_p.add_argument("--output-tex", help="Write auto-generated LaTeX fragment to path")
    _add_policy_args(sweep_p)

    tex_p = sub.add_parser("render-tex", help="Render LaTeX table fragment from a saved JSON result")
    tex_p.add_argument("--input-json", required=True, help="Path to single-run or sweep JSON")
    tex_p.add_argument("--output-tex", required=True, help="Output .tex fragment path")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(argv if argv is not None else sys.argv[1:])

    # Backward-compatible mode:
    # `python -m agentic_sleep.cli --games 20 --turns 50`
    if not argv or argv[0].startswith("-"):
        parser = argparse.ArgumentParser(description="Agentic Sleep prototype experiment runner (single run)")
        _add_run_args(parser)
        args = parser.parse_args(argv)
        result = run_comparison_experiment(
            games=args.games,
            turns=args.turns,
            seed=args.seed,
            policy_spec=_build_policy_spec(args),
            record_events=args.record_events,
            benchmark_file=args.benchmark_file,
        )
        _print_and_save(result, args.output_json, args.output_tex)
        return

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        result = run_comparison_experiment(
            games=args.games,
            turns=args.turns,
            seed=args.seed,
            policy_spec=_build_policy_spec(args),
            record_events=args.record_events,
            benchmark_file=args.benchmark_file,
        )
        _print_and_save(result, args.output_json, args.output_tex)
        return

    if args.command == "sweep":
        seeds = _parse_seeds(args)
        result = run_seed_sweep_experiment(
            seeds=seeds,
            games=args.games,
            turns=args.turns,
            policy_spec=_build_policy_spec(args),
            record_events=args.record_events,
            benchmark_file=args.benchmark_file,
        )
        _print_and_save(result, args.output_json, args.output_tex)
        return

    if args.command == "render-tex":
        payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        _write_text(args.output_tex, render_latex_results_fragment(payload))
        print(f"Wrote {args.output_tex}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
