from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .models import TurnTask


@dataclass
class LoadedBenchmark:
    scenarios: List[List[TurnTask]]
    metadata: Dict[str, object]


def load_benchmark_file(path: str | Path) -> LoadedBenchmark:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Benchmark file not found: {p}")

    if p.suffix.lower() == ".jsonl":
        scenarios = _load_jsonl_benchmark(p)
    else:
        scenarios = _load_json_benchmark(p)

    domains = sorted({t.domain for game in scenarios for t in game})
    turns_per_game = sorted({len(game) for game in scenarios})
    metadata = {
        "benchmark_file": str(p),
        "benchmark_name": p.stem,
        "games_total": len(scenarios),
        "domains": domains,
        "turns_per_game_unique": turns_per_game,
        "source_format": p.suffix.lower().lstrip("."),
    }
    return LoadedBenchmark(scenarios=scenarios, metadata=metadata)


def _load_json_benchmark(path: Path) -> List[List[TurnTask]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "games" in obj:
        games = obj["games"]
    else:
        games = obj
    if not isinstance(games, list):
        raise ValueError("Benchmark JSON must be a list of games or an object with a 'games' list")

    scenarios: List[List[TurnTask]] = []
    next_game_id = 1
    for g in games:
        if not isinstance(g, dict):
            raise ValueError("Each game entry must be an object")
        domain = str(g.get("domain", "custom"))
        game_id = int(g.get("game_id", next_game_id))
        turns = g.get("turns")
        if not isinstance(turns, list):
            raise ValueError("Each game object must contain a 'turns' list")
        scenarios.append(_parse_game_turns(turns, domain=domain, game_id=game_id))
        next_game_id = max(next_game_id, game_id + 1)
    return scenarios


def _load_jsonl_benchmark(path: Path) -> List[List[TurnTask]]:
    rows: List[Dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"Each JSONL line must be an object (line {line_no})")
        rows.append(obj)

    grouped: Dict[tuple[str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        domain = str(row.get("domain", "custom"))
        game_id = int(row.get("game_id", 0))
        grouped.setdefault((domain, game_id), []).append(row)

    scenarios: List[List[TurnTask]] = []
    for (domain, game_id), items in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        items.sort(key=lambda r: int(r.get("turn", 0)))
        scenarios.append(_parse_game_turns(items, domain=domain, game_id=game_id))
    return scenarios


def _parse_game_turns(turn_rows: Sequence[Dict[str, Any]], domain: str, game_id: int) -> List[TurnTask]:
    tasks: List[TurnTask] = []
    for idx, row in enumerate(turn_rows, start=1):
        candidates = row.get("candidates")
        tags = row.get("tags")
        best_action = row.get("best_action")
        if not isinstance(candidates, list) or not all(isinstance(c, str) for c in candidates):
            raise ValueError(f"Turn row missing valid 'candidates' list (domain={domain}, game_id={game_id}, turn={idx})")
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            raise ValueError(f"Turn row missing valid 'tags' list (domain={domain}, game_id={game_id}, turn={idx})")
        if not isinstance(best_action, str):
            raise ValueError(f"Turn row missing string 'best_action' (domain={domain}, game_id={game_id}, turn={idx})")
        if best_action not in candidates:
            raise ValueError(
                f"'best_action' must be one of candidates (domain={domain}, game_id={game_id}, turn={idx})"
            )
        turn = int(row.get("turn", idx))
        difficulty = float(row.get("difficulty", 0.7))
        tasks.append(
            TurnTask(
                domain=domain,
                game_id=game_id,
                turn=turn,
                tags=set(tags),
                candidates=list(candidates),
                best_action=best_action,
                difficulty=difficulty,
            )
        )
    return tasks


def convert_trace_events_to_benchmark_games(
    traces: Iterable[dict],
    default_candidates: Sequence[str],
    domain: str = "trace_replay",
) -> List[dict]:
    """
    Convert recorded `record-events` output (turn events) into benchmark game JSON entries.

    This is useful when you have a real agent/environment producing per-turn logs and want to
    replay those trajectories through the Agentic Sleep comparison harness.
    """
    grouped: Dict[int, List[dict]] = {}
    for event in traces:
        if not isinstance(event, dict) or event.get("type") != "turn":
            continue
        game_id = int(event.get("game_id", 0))
        grouped.setdefault(game_id, []).append(event)

    games: List[dict] = []
    for game_id, events in sorted(grouped.items()):
        turns = []
        for i, e in enumerate(sorted(events, key=lambda x: int(x.get("turn", 0))), start=1):
            action = str(e.get("best_action") or e.get("action") or default_candidates[0])
            turns.append(
                {
                    "turn": int(e.get("turn", i)),
                    "tags": list(e.get("tags", [])),
                    "candidates": list(default_candidates),
                    "best_action": action if action in default_candidates else default_candidates[0],
                    "difficulty": float(e.get("difficulty", 0.7)),
                }
            )
        games.append({"domain": str(events[0].get("domain", domain) if events else domain), "game_id": game_id, "turns": turns})
    return games

