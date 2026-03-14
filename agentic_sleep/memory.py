from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from .models import BiTemporalEdge, Episode, EpisodeStep, HeuristicRule


@dataclass
class ShortTermInteractionMemory:
    max_steps: int = 200
    steps: Deque[EpisodeStep] = field(default_factory=deque)

    def add(self, step: EpisodeStep) -> None:
        self.steps.append(step)
        while len(self.steps) > self.max_steps:
            self.steps.popleft()

    def clear(self) -> None:
        self.steps.clear()

    @property
    def token_estimate(self) -> int:
        return sum(step.token_estimate for step in self.steps)

    def snapshot(self) -> List[EpisodeStep]:
        return list(self.steps)


@dataclass
class MidTermEpisodicMemory:
    episodes: List[Episode] = field(default_factory=list)
    next_episode_id: int = 1

    def append_episode(self, steps: List[EpisodeStep]) -> Optional[Episode]:
        if not steps:
            return None
        episode = Episode(
            episode_id=self.next_episode_id,
            start_turn=steps[0].turn,
            end_turn=steps[-1].turn,
            steps=list(steps),
        )
        self.next_episode_id += 1
        self.episodes.append(episode)
        return episode


@dataclass
class LongTermSemanticMemory:
    heuristics: Dict[str, HeuristicRule] = field(default_factory=dict)
    edges: List[BiTemporalEdge] = field(default_factory=list)
    community_summaries: Dict[str, str] = field(default_factory=dict)

    def upsert_rule(self, action: str, tags: set[str]) -> HeuristicRule:
        key = self._rule_key(action, tags)
        if key not in self.heuristics:
            self.heuristics[key] = HeuristicRule(rule_id=key, action=action, tags=set(tags))
        return self.heuristics[key]

    def get_top_rules(self, limit: int = 5) -> List[HeuristicRule]:
        return sorted(self.heuristics.values(), key=lambda r: r.score, reverse=True)[:limit]

    def active_edges(self) -> List[BiTemporalEdge]:
        return [edge for edge in self.edges if edge.active]

    def add_or_update_edge(
        self,
        subject: str,
        predicate: str,
        obj: str,
        current_turn: int,
        episode_id: int,
    ) -> None:
        for edge in reversed(self.edges):
            if edge.subject == subject and edge.predicate == predicate and edge.active:
                if edge.object != obj:
                    edge.t_invalid = current_turn
                    edge.t_expired = current_turn
                else:
                    return
                break
        self.edges.append(
            BiTemporalEdge(
                subject=subject,
                predicate=predicate,
                object=obj,
                t_created=current_turn,
                t_expired=None,
                t_valid=current_turn,
                t_invalid=None,
                provenance_episode_id=episode_id,
            )
        )

    def _rule_key(self, action: str, tags: set[str]) -> str:
        return f"{action}::{'|'.join(sorted(tags))}"


@dataclass
class MemoryHierarchy:
    stim: ShortTermInteractionMemory = field(default_factory=ShortTermInteractionMemory)
    mtem: MidTermEpisodicMemory = field(default_factory=MidTermEpisodicMemory)
    ltsm: LongTermSemanticMemory = field(default_factory=LongTermSemanticMemory)

    def record_step(self, step: EpisodeStep) -> None:
        self.stim.add(step)

    def flush_stim_to_episode(self) -> Optional[Episode]:
        steps = self.stim.snapshot()
        episode = self.mtem.append_episode(steps)
        self.stim.clear()
        return episode

