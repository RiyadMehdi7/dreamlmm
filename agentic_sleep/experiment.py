from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Dict, Iterable, List, Sequence

from .benchmarks import load_benchmark_file
from .engine import AgentConfig, AgenticSleepRunner
from .models import GameSummary, TurnTask
from .policy import PolicySpec, build_policy


@dataclass
class ExperimentResult:
    control: Dict[str, float]
    sleep: Dict[str, float]
    consolidation_efficiency: float
    heuristic_compression_ratio: float
    by_domain: Dict[str, Dict[str, Dict[str, float]]]
    metadata: Dict[str, object] = field(default_factory=dict)
    traces: Dict[str, List[dict]] | None = None


@dataclass
class MetricDistribution:
    mean: float
    std: float
    ci95_low: float
    ci95_high: float
    n: int


@dataclass
class SeedSweepResult:
    seeds: List[int]
    runs: List[ExperimentResult]
    summary: Dict[str, Dict[str, MetricDistribution]]
    paired_deltas: Dict[str, MetricDistribution]
    by_domain_summary: Dict[str, Dict[str, Dict[str, MetricDistribution]]]
    metadata: Dict[str, object] = field(default_factory=dict)


CODENAMES_ACTIONS = [
    "safe_clue",
    "aggressive_multi",
    "double_meaning",
    "assassin_avoid",
    "teammate_model",
]

MMG_ACTIONS = [
    "cross_check_alibi",
    "timeline_reconstruct",
    "motive_probe",
    "contradiction_test",
    "scene_reinspect",
]


def _best_action_for_tags(domain: str, tags: set[str]) -> str:
    if domain == "codenames":
        if "semantic:double_meaning" in tags:
            return "double_meaning"
        if "risk:assassin_near" in tags:
            return "assassin_avoid"
        if "team:misread_prone" in tags:
            return "teammate_model"
        if "board:clustered" in tags:
            return "aggressive_multi"
        return "safe_clue"

    if "evidence:timeline_gap" in tags:
        return "timeline_reconstruct"
    if "social:contradiction" in tags:
        return "contradiction_test"
    if "scene:uninspected" in tags:
        return "scene_reinspect"
    if "motive:unclear" in tags:
        return "motive_probe"
    return "cross_check_alibi"


def _sample_tags(rng: random.Random, domain: str) -> set[str]:
    if domain == "codenames":
        universe = [
            "semantic:double_meaning",
            "risk:assassin_near",
            "team:misread_prone",
            "board:clustered",
            "board:sparse",
            "semantic:literal_bias",
            "team:aligned",
        ]
        tags = set(rng.sample(universe, k=3))
        if rng.random() < 0.55:
            tags.add(rng.choice(universe[:4]))
        return tags

    universe = [
        "evidence:timeline_gap",
        "social:contradiction",
        "scene:uninspected",
        "motive:unclear",
        "witness:reliable",
        "social:deflection",
        "evidence:weak_chain",
    ]
    tags = set(rng.sample(universe, k=3))
    if rng.random() < 0.55:
        tags.add(rng.choice(universe[:4]))
    return tags


def generate_games(
    seed: int,
    games: int,
    turns_per_game: int,
    domains: Sequence[str] = ("codenames", "mmg"),
) -> List[List[TurnTask]]:
    rng = random.Random(seed)
    all_games: List[List[TurnTask]] = []
    game_id = 1
    for _ in range(games):
        for domain in domains:
            tasks: List[TurnTask] = []
            for turn in range(1, turns_per_game + 1):
                tags = _sample_tags(rng, domain)
                candidates = CODENAMES_ACTIONS if domain == "codenames" else MMG_ACTIONS
                best_action = _best_action_for_tags(domain, tags)
                difficulty = 0.55 + rng.random() * 0.35
                tasks.append(
                    TurnTask(
                        domain=domain,
                        game_id=game_id,
                        turn=turn,
                        tags=tags,
                        candidates=list(candidates),
                        best_action=best_action,
                        difficulty=difficulty,
                    )
                )
            all_games.append(tasks)
            game_id += 1
    return all_games


def _aggregate_games(games: Iterable[GameSummary]) -> Dict[str, float]:
    items = list(games)
    if not items:
        return {"games": 0.0, "win_rate": 0.0, "deductive_accuracy": 0.0}
    return {
        "games": float(len(items)),
        "win_rate": sum(1 for g in items if g.win) / len(items),
        "deductive_accuracy": mean(g.deductive_accuracy for g in items),
    }


def _aggregate_by_domain(games: Iterable[GameSummary]) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[GameSummary]] = {}
    for g in games:
        groups.setdefault(g.domain, []).append(g)
    return {domain: _aggregate_games(items) for domain, items in groups.items()}


def _metric_distribution(values: List[float], bounds: tuple[float, float] | None = None) -> MetricDistribution:
    n = len(values)
    if n == 0:
        return MetricDistribution(mean=0.0, std=0.0, ci95_low=0.0, ci95_high=0.0, n=0)
    mu = mean(values)
    sigma = stdev(values) if n > 1 else 0.0
    radius = (1.96 * sigma / math.sqrt(n)) if n > 1 else 0.0
    lo = mu - radius
    hi = mu + radius
    if bounds is not None:
        lo = max(bounds[0], lo)
        hi = min(bounds[1], hi)
    return MetricDistribution(mean=mu, std=sigma, ci95_low=lo, ci95_high=hi, n=n)


def run_comparison_experiment(
    games: int = 20,
    turns: int = 50,
    seed: int = 7,
    policy_spec: PolicySpec | None = None,
    record_events: bool = False,
    benchmark_file: str | None = None,
) -> ExperimentResult:
    benchmark_meta: Dict[str, object] = {}
    if benchmark_file:
        loaded = load_benchmark_file(benchmark_file)
        scenarios = loaded.scenarios
        benchmark_meta = loaded.metadata
    else:
        scenarios = generate_games(seed=seed, games=games, turns_per_game=turns)

    control_policy = build_policy(policy_spec)
    sleep_policy = build_policy(policy_spec)
    control_runner = AgenticSleepRunner(
        AgentConfig(mode="control"),
        seed=seed,
        policy=control_policy,
        record_events=record_events,
    )
    sleep_runner = AgenticSleepRunner(
        AgentConfig(mode="sleep", sleep_interval=10, entropy_threshold=0.16),
        seed=seed,
        policy=sleep_policy,
        record_events=record_events,
    )

    control_summaries: List[GameSummary] = []
    sleep_summaries: List[GameSummary] = []

    for tasks in scenarios:
        control_summaries.append(control_runner.play_game(tasks))
        sleep_summaries.append(sleep_runner.play_game(tasks))

    control_stats = _aggregate_games(control_summaries)
    sleep_stats = _aggregate_games(sleep_summaries)
    control_by_domain = _aggregate_by_domain(control_summaries)
    sleep_by_domain = _aggregate_by_domain(sleep_summaries)
    by_domain: Dict[str, Dict[str, Dict[str, float]]] = {}
    for domain in sorted(set(control_by_domain) | set(sleep_by_domain)):
        by_domain[domain] = {
            "control": control_by_domain.get(domain, {}),
            "sleep": sleep_by_domain.get(domain, {}),
        }

    b_control = max(1, control_runner.state.metrics.token_budget)
    b_sleep = max(1, sleep_runner.state.metrics.token_budget)
    s_control = control_stats["win_rate"]
    s_sleep = sleep_stats["win_rate"]
    consolidation_efficiency = (s_sleep - s_control) / max(1e-9, (b_sleep / b_control))
    hcr_values = sleep_runner.state.metrics.hcr_values or [1.0]
    mean_hcr = mean(hcr_values)

    control_stats.update(
        {
            "token_budget": float(b_control),
            "avg_entropy": control_runner.state.metrics.average_entropy,
            "sleep_cycles": float(control_runner.state.metrics.sleep_cycles),
            "consolidations": float(control_runner.state.metrics.consolidations),
        }
    )
    sleep_stats.update(
        {
            "token_budget": float(b_sleep),
            "avg_entropy": sleep_runner.state.metrics.average_entropy,
            "sleep_cycles": float(sleep_runner.state.metrics.sleep_cycles),
            "avg_hcr": mean_hcr,
            "consolidations": float(sleep_runner.state.metrics.consolidations),
        }
    )

    traces = None
    if record_events:
        traces = {"control": control_runner.events, "sleep": sleep_runner.events}

    metadata: Dict[str, object] = {
        "seed": seed,
        "games_per_domain": games,
        "turns_per_game": turns,
        "policy": {
            "kind": policy_spec.kind if policy_spec else "sim",
            "model": policy_spec.model if policy_spec else None,
            "temperature": policy_spec.temperature if policy_spec else None,
        },
    }
    if benchmark_meta:
        metadata["benchmark"] = benchmark_meta
        metadata["games_per_domain"] = None
        metadata["turns_per_game"] = None

    return ExperimentResult(
        control=control_stats,
        sleep=sleep_stats,
        consolidation_efficiency=consolidation_efficiency,
        heuristic_compression_ratio=mean_hcr,
        by_domain=by_domain,
        metadata=metadata,
        traces=traces,
    )


def run_seed_sweep_experiment(
    seeds: Sequence[int],
    games: int = 20,
    turns: int = 50,
    policy_spec: PolicySpec | None = None,
    record_events: bool = False,
    benchmark_file: str | None = None,
) -> SeedSweepResult:
    seed_list = [int(s) for s in seeds]
    runs: List[ExperimentResult] = []
    for seed in seed_list:
        runs.append(
            run_comparison_experiment(
                games=games,
                turns=turns,
                seed=seed,
                policy_spec=policy_spec,
                record_events=record_events,
                benchmark_file=benchmark_file,
            )
        )

    summary: Dict[str, Dict[str, MetricDistribution]] = {"control": {}, "sleep": {}}
    tracked_metrics = [
        "win_rate",
        "deductive_accuracy",
        "token_budget",
        "avg_entropy",
        "sleep_cycles",
        "consolidations",
    ]
    for side in ("control", "sleep"):
        for metric in tracked_metrics + (["avg_hcr"] if side == "sleep" else []):
            values = [float(getattr(run, side).get(metric, 0.0)) for run in runs]
            bounds = (0.0, 1.0) if metric in {"win_rate", "deductive_accuracy"} else None
            summary[side][metric] = _metric_distribution(values, bounds=bounds)

    top_level_metrics = {
        "consolidation_efficiency": [run.consolidation_efficiency for run in runs],
        "heuristic_compression_ratio": [run.heuristic_compression_ratio for run in runs],
    }
    summary["global"] = {k: _metric_distribution(v) for k, v in top_level_metrics.items()}

    paired_deltas = {
        "delta_win_rate": _metric_distribution([run.sleep["win_rate"] - run.control["win_rate"] for run in runs], bounds=(-1.0, 1.0)),
        "delta_deductive_accuracy": _metric_distribution(
            [run.sleep["deductive_accuracy"] - run.control["deductive_accuracy"] for run in runs]
            ,
            bounds=(-1.0, 1.0),
        ),
        "token_budget_ratio_sleep_over_control": _metric_distribution(
            [
                (run.sleep["token_budget"] / max(1.0, run.control["token_budget"]))
                for run in runs
            ]
        ),
    }

    domains = sorted({d for run in runs for d in run.by_domain.keys()})
    by_domain_summary: Dict[str, Dict[str, Dict[str, MetricDistribution]]] = {}
    for domain in domains:
        by_domain_summary[domain] = {"control": {}, "sleep": {}}
        for side in ("control", "sleep"):
            for metric in ("win_rate", "deductive_accuracy"):
                values = []
                for run in runs:
                    values.append(float(run.by_domain.get(domain, {}).get(side, {}).get(metric, 0.0)))
                by_domain_summary[domain][side][metric] = _metric_distribution(values, bounds=(0.0, 1.0))

    metadata = {
        "num_seeds": len(seed_list),
        "seeds": seed_list,
        "games_per_domain": games,
        "turns_per_game": turns,
        "policy": {
            "kind": policy_spec.kind if policy_spec else "sim",
            "model": policy_spec.model if policy_spec else None,
            "temperature": policy_spec.temperature if policy_spec else None,
        },
    }
    if benchmark_file:
        metadata["benchmark_file"] = benchmark_file
        metadata["games_per_domain"] = None
        metadata["turns_per_game"] = None

    return SeedSweepResult(
        seeds=seed_list,
        runs=runs,
        summary=summary,
        paired_deltas=paired_deltas,
        by_domain_summary=by_domain_summary,
        metadata=metadata,
    )
