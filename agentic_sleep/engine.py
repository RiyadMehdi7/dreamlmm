from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .consolidation import ConsolidationEngine
from .dream import DreamSimulator
from .entropy import entropy_spike, normalize, shannon_entropy
from .memory import MemoryHierarchy
from .models import AgentMetrics, EpisodeStep, GameSummary, TurnResult, TurnTask, estimate_tokens
from .policy import BasePolicy, PolicyContext, PolicyDecision


@dataclass
class AgentConfig:
    mode: str  # "sleep" or "control"
    sleep_interval: int = 10
    entropy_threshold: float = 0.18
    top_universal_rules: int = 5
    min_sleep_gap: int = 4


@dataclass
class AgentState:
    messages: List[str] = field(default_factory=list)
    heuristics: List[str] = field(default_factory=list)
    knowledge_graph: Dict[str, object] = field(default_factory=dict)
    metrics: AgentMetrics = field(default_factory=AgentMetrics)


class AgenticSleepRunner:
    def __init__(
        self,
        config: AgentConfig,
        seed: int = 0,
        policy: BasePolicy | None = None,
        record_events: bool = False,
    ) -> None:
        self.config = config
        self.rng = random.Random(seed)
        self.memory = MemoryHierarchy()
        self.state = AgentState()
        self.consolidator = ConsolidationEngine()
        self.dreamer = DreamSimulator(self.rng)
        self.policy = policy
        self.raw_history: List[EpisodeStep] = []
        self.last_consolidation_turn = 0
        self.record_events = record_events
        self.events: List[Dict[str, Any]] = []

    def play_game(self, tasks: List[TurnTask]) -> GameSummary:
        successes = 0
        for task in tasks:
            result, step = self._run_turn(task)
            if result.success:
                successes += 1
            self._post_turn(step)
        win_threshold = max(1, int(len(tasks) * 0.6))
        return GameSummary(
            domain=tasks[0].domain if tasks else "unknown",
            game_id=tasks[0].game_id if tasks else -1,
            turns=len(tasks),
            successes=successes,
            win=successes >= win_threshold,
            deductive_accuracy=successes / max(1, len(tasks)),
        )

    def _run_turn(self, task: TurnTask) -> tuple[TurnResult, EpisodeStep]:
        action, thought, probs, policy_usage_tokens, policy_meta = self._choose_action_runtime(task)
        success = action == task.best_action
        reward = 1.0 if success else 0.0
        entropy = shannon_entropy(probs)
        observation = self._format_observation(task, action, success)
        step = EpisodeStep(
            turn=task.turn,
            domain=task.domain,
            thought=thought,
            action=action,
            observation=observation,
            success=success,
            entropy=entropy,
            tags=set(task.tags),
        )
        self.state.metrics.update_entropy(entropy)
        self.state.metrics.turns += 1
        if success:
            self.state.metrics.successes += 1
        else:
            self.state.metrics.failures += 1
        turn_text = f"{task.domain} t={task.turn} tags={sorted(task.tags)} action={action} obs={observation}"
        self.state.messages.append(turn_text)
        if policy_usage_tokens > 0:
            self.state.metrics.token_budget += policy_usage_tokens
        else:
            # Count both prompt context and generated trace to approximate inference token budget.
            self.state.metrics.token_budget += self._effective_context_tokens()
            self.state.metrics.token_budget += estimate_tokens(turn_text + thought)
        if self.record_events:
            self.events.append(
                {
                    "type": "turn",
                    "mode": self.config.mode,
                    "domain": task.domain,
                    "game_id": task.game_id,
                    "turn": task.turn,
                    "tags": sorted(task.tags),
                    "action": action,
                    "best_action": task.best_action,
                    "success": success,
                    "entropy": entropy,
                    "probs": {a: p for a, p in zip(task.candidates, probs)},
                    "policy": policy_meta,
                    "heuristics_count": len(self.memory.ltsm.get_top_rules(self.config.top_universal_rules)),
                }
            )
        return TurnResult(success=success, reward=reward, observation=observation), step

    def _choose_action_runtime(
        self, task: TurnTask
    ) -> tuple[str, str, List[float], int, Dict[str, Any]]:
        if self.policy is None:
            action, thought, probs = self._choose_action(task)
            return action, thought, probs, 0, {"kind": "sim"}

        context = self._build_policy_context(task)
        decision = self.policy.select_action(context)
        action = decision.action if decision.action in task.candidates else self._fallback_action_from_decision(task, decision)
        probs = [float(decision.action_probs.get(c, 0.0)) for c in task.candidates]
        if sum(probs) <= 0:
            probs = [1.0 if c == action else 0.0 for c in task.candidates]
        thought = (decision.rationale_summary or "LLM-selected action.")[:400]
        usage_tokens = max(0, int(decision.prompt_tokens) + int(decision.completion_tokens))
        meta = {
            "kind": getattr(self.policy, "kind", "unknown"),
            "prompt_tokens": int(decision.prompt_tokens),
            "completion_tokens": int(decision.completion_tokens),
            "raw_text": decision.raw_text[:1200],
            **decision.metadata,
        }
        return action, thought, probs, usage_tokens, meta

    def _build_policy_context(self, task: TurnTask) -> PolicyContext:
        recent_steps = self.raw_history[-80:] if self.config.mode == "control" else self.memory.stim.snapshot()[-80:]
        top_rules = [r.description() for r in self.memory.ltsm.get_top_rules(self.config.top_universal_rules)]
        return PolicyContext(
            mode=self.config.mode,
            task=task,
            recent_steps=list(recent_steps),
            top_rule_descriptions=top_rules,
            community_summaries=dict(self.memory.ltsm.community_summaries),
            effective_context_tokens=self._effective_context_tokens(),
        )

    def _fallback_action_from_decision(self, task: TurnTask, decision: PolicyDecision) -> str:
        if decision.action_probs:
            ranked = sorted(
                ((c, float(decision.action_probs.get(c, 0.0))) for c in task.candidates),
                key=lambda x: x[1],
                reverse=True,
            )
            if ranked and ranked[0][1] > 0:
                return ranked[0][0]
        return task.candidates[0]

    def _choose_action(self, task: TurnTask) -> tuple[str, str, List[float]]:
        scores: Dict[str, float] = {a: 0.0 for a in task.candidates}

        # Mid/long memory signal for sleep mode.
        if self.config.mode == "sleep":
            for rule in self.memory.ltsm.heuristics.values():
                if rule.action not in scores:
                    continue
                if rule.tags & task.tags:
                    overlap = len(rule.tags & task.tags)
                    scores[rule.action] += rule.score * (1.4 + 0.25 * overlap)
                elif rule.posterior_success < 0.42:
                    # Weak negative prior for low-performing rules outside matching tags.
                    scores[rule.action] -= 0.03

        # Raw episodic retrieval (used by both, dominates control behavior).
        history = self.raw_history if self.config.mode == "control" else self.memory.stim.snapshot()
        for past in history[-80:]:
            overlap = len(task.tags & past.tags)
            if overlap == 0 or past.action not in scores:
                continue
            delta = (1.0 if past.success else -0.7) * overlap
            scores[past.action] += delta

        # Latent model quality signal that degrades with contextual noise.
        context_tokens = self._effective_context_tokens()
        if self.config.mode == "control":
            noise_penalty = min(4.2, (context_tokens / 320.0) + (len(self.state.messages) / 65.0))
        else:
            noise_penalty = min(1.9, (context_tokens / 950.0) + (len(self.state.messages) / 220.0))

        # Base preference from hidden difficulty proxy (simulates imperfect reasoning).
        for idx, action in enumerate(task.candidates):
            scores[action] += self.rng.uniform(-0.2, 0.2)
            if action == task.best_action:
                scores[action] += (1.5 - task.difficulty)
            # Larger contexts flatten the distribution -> higher entropy and more mistakes.
            scores[action] -= noise_penalty * (0.18 if self.config.mode == "control" else 0.08)
            scores[action] -= abs(idx - (task.turn % len(task.candidates))) * 0.02

        if self.config.mode == "control" and len(self.raw_history) > 150:
            # Saturation effect: stale traces add random interference in long windows.
            for action in task.candidates:
                scores[action] += self.rng.uniform(-0.25, 0.25)

        temperature = 0.85 + noise_penalty * (0.55 if self.config.mode == "control" else 0.22)
        probs = normalize([scores[a] for a in task.candidates], temperature=temperature)
        action = self.rng.choices(task.candidates, weights=probs, k=1)[0]
        top_rules = self.memory.ltsm.get_top_rules(limit=self.config.top_universal_rules)
        heuristics_preview = ",".join(sorted({r.action for r in top_rules})) if top_rules else "none"
        thought = (
            f"Selected {action}. noise={noise_penalty:.2f}. "
            f"rule_count={len(top_rules)} top_actions={heuristics_preview}"
        )
        return action, thought, probs

    def _effective_context_tokens(self) -> int:
        if self.config.mode == "control":
            return sum(s.token_estimate for s in self.raw_history)
        return self.memory.stim.token_estimate + sum(
            estimate_tokens(rule.description()) for rule in self.memory.ltsm.get_top_rules(5)
        )

    def _format_observation(self, task: TurnTask, action: str, success: bool) -> str:
        outcome = "success" if success else "failure"
        return f"{outcome}; best={task.best_action}; chosen={action}; tags={','.join(sorted(task.tags))}"

    def _post_turn(self, step: EpisodeStep) -> None:
        self.raw_history.append(step)
        if self.config.mode == "sleep":
            self.memory.record_step(step)
            if self._should_sleep(step.turn):
                self._sleep_cycle(step.turn)
        else:
            # Control retains all context and never consolidates.
            pass

    def _should_sleep(self, turn: int) -> bool:
        turns_since_last = turn - self.last_consolidation_turn
        interval_trigger = (turn - self.last_consolidation_turn) >= self.config.sleep_interval
        entropy_trigger = entropy_spike(
            self.state.metrics.prev_entropy,
            self.state.metrics.shannon_entropy,
            self.config.entropy_threshold,
        )
        return interval_trigger or (entropy_trigger and turns_since_last >= self.config.min_sleep_gap)

    def _sleep_cycle(self, turn: int) -> None:
        episode = self.memory.flush_stim_to_episode()
        if not episode:
            return
        self.state.metrics.sleep_cycles += 1
        self.last_consolidation_turn = turn

        report = self.consolidator.consolidate(episode, self.memory.ltsm)
        self.state.metrics.consolidations += 1
        self.state.metrics.hcr_values.append(report.hcr)

        # Keep only high-density rules in active state (the "5 Universal Rules" constraint).
        self.state.heuristics = [r.description() for r in self.memory.ltsm.get_top_rules(self.config.top_universal_rules)]
        self.state.knowledge_graph = {
            "active_edges": len(self.memory.ltsm.active_edges()),
            "communities": dict(self.memory.ltsm.community_summaries),
        }

        # Failure-driven dream phase uses recent failures from the episode.
        failures = [s for s in episode.steps if not s.success or s.entropy > 1.6]
        dream_report = self.dreamer.harden(self.memory.ltsm, failures)

        # Structural reset: clear narrative context, keep abstractions only.
        self.state.messages.clear()
        self.state.metrics.token_budget += report.consolidated_tokens
        if self.record_events:
            self.events.append(
                {
                    "type": "sleep_cycle",
                    "mode": self.config.mode,
                    "turn": turn,
                    "episode_id": report.episode_id,
                    "hcr": report.hcr,
                    "raw_tokens": report.raw_tokens,
                    "consolidated_tokens": report.consolidated_tokens,
                    "promoted_rules": report.promoted_rule_ids,
                    "pruned_rules": report.pruned_rule_ids,
                    "dream": {
                        "seeds_tested": dream_report.seeds_tested,
                        "rules_tested": dream_report.rules_tested,
                        "rules_refined": dream_report.rules_refined,
                        "guardian_rejections": dream_report.guardian_rejections,
                        "mean_confidence": dream_report.mean_confidence,
                    },
                }
            )


# Optional adapter to mirror the prompt's LangGraph routing logic if langgraph is installed.
def calculate_entropy_spike(agent_state: AgentState, threshold: float) -> str:
    current_entropy = agent_state.metrics.shannon_entropy
    previous_entropy = agent_state.metrics.prev_entropy
    if (current_entropy - previous_entropy) > threshold:
        return "trigger_sleep"
    return "continue_waking"
