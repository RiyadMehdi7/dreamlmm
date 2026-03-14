from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List

from .memory import LongTermSemanticMemory
from .models import DreamReport, EpisodeStep


@dataclass
class DreamConfig:
    synthetic_trials_per_rule: int = 4
    min_support_for_hardening: int = 2
    confidence_mix: float = 0.5


class DreamSimulator:
    def __init__(self, rng: random.Random, config: DreamConfig | None = None) -> None:
        self.rng = rng
        self.config = config or DreamConfig()

    def harden(
        self,
        ltsm: LongTermSemanticMemory,
        failure_seeds: Iterable[EpisodeStep],
    ) -> DreamReport:
        seeds = [s for s in failure_seeds if not s.success]
        seed_tags = sorted({tag for seed in seeds for tag in seed.tags})
        known_tags = {tag for rule in ltsm.heuristics.values() for tag in rule.tags}
        allowed_tags = known_tags | set(seed_tags)

        rules_refined = 0
        guardian_rejections = 0
        tested_rules = 0

        for rule in ltsm.get_top_rules(limit=12):
            if rule.support < self.config.min_support_for_hardening:
                continue
            tested_rules += 1
            synthetic_successes = 0
            for _ in range(self.config.synthetic_trials_per_rule):
                counterfactual_tag = self.rng.choice(seed_tags or list(rule.tags) or ["misc:unknown"])
                if counterfactual_tag not in allowed_tags:
                    guardian_rejections += 1
                    continue
                success = self._simulate_counterfactual(rule, counterfactual_tag)
                rule.record_synthetic(success)
                synthetic_successes += int(success)
            new_confidence = (
                (1.0 - self.config.confidence_mix) * rule.posterior_success
                + self.config.confidence_mix * rule.synthetic_success_rate
            )
            rule.confidence = max(0.05, min(0.99, new_confidence))
            rules_refined += 1

        mean_confidence = 0.0
        if ltsm.heuristics:
            mean_confidence = sum(r.confidence for r in ltsm.heuristics.values()) / len(ltsm.heuristics)
        return DreamReport(
            seeds_tested=len(seeds),
            rules_tested=tested_rules,
            rules_refined=rules_refined,
            guardian_rejections=guardian_rejections,
            mean_confidence=mean_confidence,
        )

    def _simulate_counterfactual(self, rule, counterfactual_tag: str) -> bool:
        # Synthetic robustness test:
        # reward rules that remain predictive under nearby tags but penalize brittle over-specialization.
        overlap_bonus = 1.0 if counterfactual_tag in rule.tags else 0.6
        support_bonus = min(0.2, rule.support / 25.0)
        posterior = rule.posterior_success
        p = max(0.05, min(0.95, overlap_bonus * 0.5 + support_bonus + posterior * 0.3))
        return self.rng.random() < p

