from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set

from .models import ConsolidationReport, Episode, HeuristicRule, estimate_tokens
from .memory import LongTermSemanticMemory


@dataclass
class ConsolidationConfig:
    success_threshold: float = 0.58
    repetitiveness_threshold: float = 0.85
    universal_rule_limit: int = 5


class ConsolidationEngine:
    def __init__(self, config: ConsolidationConfig | None = None) -> None:
        self.config = config or ConsolidationConfig()

    def consolidate(self, episode: Episode, ltsm: LongTermSemanticMemory) -> ConsolidationReport:
        if not episode.steps:
            return ConsolidationReport(
                episode_id=episode.episode_id,
                raw_tokens=0,
                consolidated_tokens=0,
                hcr=1.0,
                promoted_rule_ids=[],
                pruned_rule_ids=[],
                community_summaries={},
            )

        self._extract_graph_facts(episode, ltsm)
        updated_rule_ids = self._extract_procedures(episode, ltsm)
        promoted, pruned = self._select_signal_rules(updated_rule_ids, ltsm)
        community_summaries = self._summarize_communities(ltsm)

        universal_rules = ltsm.get_top_rules(self.config.universal_rule_limit)
        consolidated_payload = " ".join([rule.description() for rule in universal_rules]) + " " + " ".join(
            community_summaries.values()
        )
        raw_tokens = episode.token_estimate
        consolidated_tokens = estimate_tokens(consolidated_payload)
        hcr = (raw_tokens / max(1, consolidated_tokens)) if raw_tokens else 1.0

        return ConsolidationReport(
            episode_id=episode.episode_id,
            raw_tokens=raw_tokens,
            consolidated_tokens=consolidated_tokens,
            hcr=hcr,
            promoted_rule_ids=promoted,
            pruned_rule_ids=pruned,
            community_summaries=community_summaries,
        )

    def _extract_graph_facts(self, episode: Episode, ltsm: LongTermSemanticMemory) -> None:
        for step in episode.steps:
            for tag in step.tags:
                ltsm.add_or_update_edge(
                    subject=f"tag:{tag}",
                    predicate="supports_action" if step.success else "fails_with_action",
                    obj=f"action:{step.action}",
                    current_turn=step.turn,
                    episode_id=episode.episode_id,
                )

    def _extract_procedures(self, episode: Episode, ltsm: LongTermSemanticMemory) -> List[str]:
        updated_rule_ids: List[str] = []
        for step in episode.steps:
            for tag in step.tags:
                rule = ltsm.upsert_rule(step.action, {tag})
                rule.record_outcome(
                    step.success,
                    example=f"{step.domain}:{tag}:{'ok' if step.success else 'fail'}:t{step.turn}",
                )
                updated_rule_ids.append(rule.rule_id)
        return updated_rule_ids

    def _select_signal_rules(
        self, updated_rule_ids: Sequence[str], ltsm: LongTermSemanticMemory
    ) -> tuple[List[str], List[str]]:
        promoted: List[str] = []
        pruned: List[str] = []
        for rule_id in set(updated_rule_ids):
            rule = ltsm.heuristics[rule_id]
            is_signal = (
                rule.posterior_success > self.config.success_threshold
                and rule.repetitiveness <= self.config.repetitiveness_threshold
            )
            if is_signal:
                promoted.append(rule_id)
            elif rule.support >= 4 and rule.posterior_success <= 0.45:
                pruned.append(rule_id)
        for rule_id in pruned:
            ltsm.heuristics.pop(rule_id, None)
        return sorted(promoted), sorted(pruned)

    def _summarize_communities(self, ltsm: LongTermSemanticMemory) -> Dict[str, str]:
        communities: Dict[str, List[HeuristicRule]] = defaultdict(list)
        for rule in ltsm.heuristics.values():
            for tag in rule.tags:
                prefix = tag.split(":", 1)[0] if ":" in tag else "misc"
                communities[prefix].append(rule)

        summaries: Dict[str, str] = {}
        for community, rules in communities.items():
            top = sorted(rules, key=lambda r: r.score, reverse=True)[:2]
            if not top:
                continue
            actions = ", ".join(sorted({r.action for r in top}))
            summaries[community] = f"{community}: strongest procedures currently favor [{actions}]."
        ltsm.community_summaries = summaries
        return summaries

