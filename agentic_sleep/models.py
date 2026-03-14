from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


@dataclass
class EpisodeStep:
    turn: int
    domain: str
    thought: str
    action: str
    observation: str
    success: bool
    entropy: float
    tags: Set[str]

    @property
    def token_estimate(self) -> int:
        payload = f"{self.thought} {self.action} {self.observation} {' '.join(sorted(self.tags))}"
        return estimate_tokens(payload)


@dataclass
class Episode:
    episode_id: int
    start_turn: int
    end_turn: int
    steps: List[EpisodeStep] = field(default_factory=list)

    @property
    def token_estimate(self) -> int:
        return sum(step.token_estimate for step in self.steps)


@dataclass
class BiTemporalEdge:
    subject: str
    predicate: str
    object: str
    t_created: int
    t_expired: Optional[int]
    t_valid: int
    t_invalid: Optional[int]
    provenance_episode_id: int

    def key(self) -> tuple[str, str]:
        return (self.subject, self.predicate)

    @property
    def active(self) -> bool:
        return self.t_invalid is None


@dataclass
class HeuristicRule:
    rule_id: str
    action: str
    tags: Set[str]
    alpha: float = 1.0
    beta: float = 1.0
    support: int = 0
    synthetic_support: int = 0
    synthetic_successes: int = 0
    confidence: float = 0.5
    examples: List[str] = field(default_factory=list)

    @property
    def posterior_success(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def repetitiveness(self) -> float:
        if self.support <= 1:
            return 0.0
        unique_examples = len(set(self.examples))
        return 1.0 - (unique_examples / max(1, len(self.examples)))

    def record_outcome(self, success: bool, example: str) -> None:
        self.support += 1
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0
        if len(self.examples) < 25:
            self.examples.append(example)

    def record_synthetic(self, success: bool) -> None:
        self.synthetic_support += 1
        if success:
            self.synthetic_successes += 1

    @property
    def synthetic_success_rate(self) -> float:
        if self.synthetic_support == 0:
            return 0.5
        return self.synthetic_successes / self.synthetic_support

    @property
    def score(self) -> float:
        return self.confidence * self.posterior_success * (1.0 + min(self.support, 10) / 10.0)

    def description(self) -> str:
        tag_part = ", ".join(sorted(self.tags)) or "generic context"
        return (
            f"When observing [{tag_part}], prefer action '{self.action}' "
            f"(p={self.posterior_success:.2f}, conf={self.confidence:.2f}, n={self.support})."
        )


@dataclass
class ConsolidationReport:
    episode_id: int
    raw_tokens: int
    consolidated_tokens: int
    hcr: float
    promoted_rule_ids: List[str]
    pruned_rule_ids: List[str]
    community_summaries: Dict[str, str]


@dataclass
class DreamReport:
    seeds_tested: int
    rules_tested: int
    rules_refined: int
    guardian_rejections: int
    mean_confidence: float


@dataclass
class AgentMetrics:
    shannon_entropy: float = 0.0
    prev_entropy: float = 0.0
    token_budget: int = 0
    turns: int = 0
    sleep_cycles: int = 0
    consolidations: int = 0
    average_entropy: float = 0.0
    successes: int = 0
    failures: int = 0
    hcr_values: List[float] = field(default_factory=list)
    last_entropy_spike: float = 0.0

    def update_entropy(self, entropy: float) -> None:
        self.prev_entropy = self.shannon_entropy
        self.shannon_entropy = entropy
        self.last_entropy_spike = self.shannon_entropy - self.prev_entropy
        self.average_entropy = (
            (self.average_entropy * self.turns + entropy) / max(1, self.turns + 1)
        )

    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total else 0.0


@dataclass
class TurnTask:
    domain: str
    game_id: int
    turn: int
    tags: Set[str]
    candidates: List[str]
    best_action: str
    difficulty: float


@dataclass
class TurnResult:
    success: bool
    reward: float
    observation: str


@dataclass
class GameSummary:
    domain: str
    game_id: int
    turns: int
    successes: int
    win: bool
    deductive_accuracy: float

