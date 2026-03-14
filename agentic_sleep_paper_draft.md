# Agentic Sleep: Off-Cycle Memory Consolidation for Long-Horizon Autonomous Agents

## Abstract

Long-horizon autonomous agents degrade as interaction histories grow, even when large context windows are available. The primary failure mode is not only token budget exhaustion but also contextual noise: irrelevant observations, transient errors, and repeated reasoning traces that flatten action distributions and increase uncertainty. We present **Agentic Sleep**, an off-cycle memory consolidation framework that periodically converts raw interaction traces into compact heuristic rules and a lightweight semantic graph, then clears the active narrative buffer. The framework is implemented as a prototype with (i) entropy-guided sleep triggering, (ii) a three-tier memory hierarchy (short-term interaction, episodic, and long-term semantic memory), (iii) Bayesian signal/noise filtering for heuristic promotion and pruning, and (iv) a failure-seeded synthetic “Dream” phase for heuristic hardening.

We evaluate the current implementation in a controlled simulation of long-horizon logic-game-like tasks (Codenames-like and murder-mystery-like domains) and observe improved token efficiency and higher task performance relative to a raw-context control. These results are preliminary and use a simulated policy/environment, not a production LLM backend. We provide a paper-ready experimental protocol and result tables with placeholders for future LLM-based validation.

## 1. Introduction

Autonomous LLM agents are often evaluated on short-horizon tasks, yet many practical workloads (software engineering, multi-stage research, incident response, and strategic games) require coherent reasoning across tens to hundreds of turns. A common engineering response is to increase context length and maintain more raw history. In practice, this introduces a different bottleneck: the model must repeatedly process stale, redundant, and low-value traces alongside task-critical evidence. The result is often semantic drift, delayed correction of invalid assumptions, and unstable action selection.

This paper studies a complementary approach: **structural abstraction over raw retention**. Instead of preserving the full trajectory in the active context, the agent periodically enters an off-cycle **Sleep Phase** that consolidates recent interaction logs into high-density, reusable heuristics and a semantic memory graph. The agent then resumes operation using these abstractions plus a short recency window, rather than the full narrative history.

The central claim is that, for long-horizon tasks, a memory system that compresses *what works* into reusable procedural rules can outperform a raw-context baseline in both token efficiency and task success.

## 2. Problem Statement

Let an autonomous agent operate over a sequence of turns \(t = 1, \dots, T\). At each turn, it receives observations, produces reasoning traces, chooses an action, and obtains a task outcome. In raw-context architectures, the agent carries an increasingly long transcript into subsequent decisions. As the transcript grows, task-irrelevant tokens accumulate and the effective signal-to-noise ratio decreases.

We define the operational problem as:

- Maintain or improve decision quality over long horizons.
- Reduce inference token cost relative to full-history control.
- Prevent repeated integration of transient errors into long-term strategy.
- Preserve historical lineage when knowledge changes (e.g., contradictory facts).

## 3. Agentic Sleep Framework

### 3.1 Overview

Agentic Sleep introduces a cyclical control loop:

1. **Waking Phase**: The agent performs the task using recency memory plus top-ranked heuristics.
2. **Trigger Check**: Sleep is triggered by elapsed turns and/or an entropy spike.
3. **Consolidation Phase**: Recent traces are converted into heuristic rules and semantic graph edges.
4. **Dream Phase**: Newly consolidated rules are stress-tested in synthetic counterfactual scenarios.
5. **Structural Reset**: The narrative buffer is cleared while compact rules and semantic summaries are retained.

### 3.2 Memory Hierarchy

The prototype implements a three-tier memory hierarchy:

- **Tier 1: Short-Term Interaction Memory (STIM)**  
  Linear buffer of recent `EpisodeStep` objects used for recency-sensitive decisions.

- **Tier 2: Mid-Term Episodic Memory (MTEM)**  
  Structured episodes formed by flushing STIM during sleep cycles.

- **Tier 3: Long-Term Semantic Memory (LTSM)**  
  A heuristic rule store plus a lightweight bi-temporal knowledge graph.

This design allows the agent to preserve recency, store audit-ready episodes, and maintain abstract strategic knowledge separately.

### 3.3 Entropy-Guided Sleep Trigger

The framework tracks Shannon entropy over the action probability distribution at each turn. Rising entropy is treated as a proxy for uncertainty or drift. Sleep can be triggered by:

- a fixed interval (e.g., every 10 turns), or
- an entropy spike above threshold \(\Delta H > \tau\), with a minimum gap to prevent thrashing.

This creates an adaptive reset mechanism: the agent consolidates when its decision distribution becomes unstable, not only on a rigid schedule.

## 4. Consolidation Mechanism

### 4.1 Input Representation

Each interaction step is recorded as an `EpisodeStep` containing:

- turn index
- domain label
- compact thought trace (logging artifact)
- selected action
- observation/outcome string
- success flag
- entropy
- semantic tags

STIM is flushed into an `Episode` during sleep. Episodes are retained in MTEM for traceability and future analysis.

### 4.2 Heuristic Extraction

The consolidation engine transforms episodic traces into procedural heuristics of the form:

> When observing tag set \(X\), prefer action \(A\).

In the current prototype, heuristics are stored as `(action, tag)` rules with Bayesian success statistics:

- `alpha`, `beta` initialize a Beta posterior
- `posterior_success = alpha / (alpha + beta)`
- `support` tracks empirical evidence count
- `repetitiveness` estimates low-information repetition in retained examples

This supports signal selection and pruning without requiring gradient updates.

### 4.3 Signal vs Noise Selection

After procedure extraction, rules are categorized:

- **Promote (signal)** if posterior success exceeds a threshold and repetitiveness remains below a threshold.
- **Prune (noise)** if support is sufficient but posterior success is consistently low.

This operationalizes the “Noise-Signal Dilemma” in a lightweight Bayesian filter.

### 4.4 Bi-Temporal Semantic Graph

The long-term semantic memory also stores fact-like edges with timestamps:

- system-time creation/expiration
- validity/invalidation timestamps
- provenance episode ID

When a new edge contradicts an active edge with the same `(subject, predicate)`, the previous edge is invalidated rather than deleted. This preserves lineage while maintaining an active current view of semantic memory.

## 5. Dream Phase: Failure-Seeded Synthetic Hardening

After consolidation, the Dream phase performs internal synthetic tests on top heuristics.

### 5.1 Failure Seeding

Dream uses recent failed steps (and optionally high-entropy steps) as seeds. Seed tags define the neighborhoods where the agent’s current strategy appears brittle.

### 5.2 Counterfactual Simulation

For each supported top-ranked rule, the simulator samples counterfactual tags (preferably from failure seeds) and evaluates whether the rule still appears successful under nearby conditions. This produces synthetic support and synthetic success counts.

### 5.3 Confidence Update

Each rule’s confidence is updated by mixing:

- empirical posterior success (from waking episodes), and
- synthetic success rate (from Dream self-play)

This changes future rule influence without rewriting the raw episodic record.

## 6. Prototype Implementation

The current implementation is a runnable research prototype in Python (stdlib-only), designed to later accept a real LLM backend. Key modules:

- `agentic_sleep/engine.py`: agent loop, scoring, entropy, triggers, sleep cycle
- `agentic_sleep/memory.py`: STIM/MTEM/LTSM hierarchy
- `agentic_sleep/consolidation.py`: heuristic extraction, graph updates, HCR computation
- `agentic_sleep/dream.py`: synthetic hardening
- `agentic_sleep/experiment.py`: control vs sleep benchmark harness
- `agentic_sleep/langgraph_adapter.py`: optional LangGraph-compatible graph skeleton

### 6.1 Decision Policy (Current Prototype)

The current agent policy is simulated (not an LLM) and combines:

- heuristic contributions from LTSM (sleep mode only),
- recent episodic evidence from raw or STIM history,
- a context-noise penalty that increases with effective context size,
- a stochastic sampling step over normalized action scores.

This setup is intentionally simple: it isolates memory architecture effects before integrating external model variability.

## 7. Experimental Protocol (Current Prototype)

### 7.1 Task Environments

We evaluate on synthetic long-horizon task families designed to mimic logic-heavy coordination and deduction:

- **Codenames-like domain**: clueing/guessing strategy abstractions and ambiguity management
- **Murder-mystery-like (MMG) domain**: contradiction checking, timeline reconstruction, scene reinspection

Each turn is represented by a tag set and a hidden best action. Difficulty and distractor tags create ambiguity.

### 7.2 Comparison Conditions

- **Control (Raw Context)**:
  - retains full raw trajectory for scoring
  - no consolidation
  - no sleep/dream phase

- **Sleep (Agentic Sleep)**:
  - uses STIM + top heuristics + LTSM summaries
  - triggers sleep by interval and entropy spike
  - consolidates and runs Dream hardening
  - clears active narrative buffer post-sleep

### 7.3 Metrics

Primary metrics:

- **Win Rate**
- **Deductive Accuracy**
- **Token Budget** (approximate inference token accounting)
- **Heuristic Compression Ratio (HCR)**  
  \[
  HCR = \frac{\text{tokens(raw traces)}}{\text{tokens(consolidated heuristics + summaries)}}
  \]
- **Consolidation Efficiency**  
  \[
  E_c = \frac{S_{sleep} - S_{control}}{B_{sleep} / B_{control}}
  \]
  where \(S\) is success (here: win rate) and \(B\) is token budget.

## 8. Preliminary Results (Simulation Only; Non-LLM)

The following run was produced by the current prototype:

```bash
python3 -m agentic_sleep.cli --games 20 --turns 50
```

### 8.1 Aggregate Results

| Metric | Control | Sleep |
|---|---:|---:|
| Games | 40 | 40 |
| Win Rate | 0.325 | 0.475 |
| Deductive Accuracy | 0.551 | 0.601 |
| Avg Entropy | 0.673 | 0.246 |
| Token Budget | 136,429,271 | 28,372,395 |
| Sleep Cycles | 0 | 9 |
| Consolidations | 0 | 9 |
| Avg HCR | N/A | 7.49 |

Derived metric:

- **Consolidation Efficiency \(E_c\)**: **0.721**

### 8.2 Domain Breakdown

| Domain | Control Win Rate | Sleep Win Rate | Control Accuracy | Sleep Accuracy |
|---|---:|---:|---:|---:|
| Codenames-like | 0.25 | 0.55 | 0.539 | 0.627 |
| MMG-like | 0.40 | 0.40 | 0.563 | 0.575 |

### 8.3 Interpretation

These preliminary results support the design hypothesis in the synthetic setting:

- Sleep improves overall performance while reducing token cost.
- Entropy is lower in the sleep condition, consistent with reduced decision diffusion.
- Gains are strongest in the Codenames-like domain, suggesting heuristic abstraction is especially effective when recurring semantic patterns map well to strategic actions.

However, these results **do not yet validate performance on real LLM agents**. They only indicate that the memory architecture is promising under a controlled simulated policy.

## 9. Planned LLM-Based Evaluation (To Be Filled)

This section is intentionally structured for later insertion of real-model results.

### 9.1 LLM Integration Plan

Replace the simulated policy with an LLM-backed action policy while preserving the same memory pipeline:

- Keep STIM/MTEM/LTSM unchanged
- Replace the score-based action selector with:
  - prompt construction from STIM + top rules + task state
  - model inference
  - action parsing + confidence estimation
- Use logprobs (if available) to compute entropy directly
- Keep the same sleep trigger, consolidation, and Dream phases

### 9.2 Proposed LLM Experiment Matrix (Placeholder)

| Model | Provider | Control Win Rate | Sleep Win Rate | Control Accuracy | Sleep Accuracy | Control Tokens | Sleep Tokens | \(E_c\) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `[[MODEL_A]]` | `[[PROVIDER]]` | `[[TBD]]` | `[[TBD]]` | `[[TBD]]` | `[[TBD]]` | `[[TBD]]` | `[[TBD]]` | `[[TBD]]` |
| `[[MODEL_B]]` | `[[PROVIDER]]` | `[[TBD]]` | `[[TBD]]` | `[[TBD]]` | `[[TBD]]` | `[[TBD]]` | `[[TBD]]` | `[[TBD]]` |

### 9.3 Statistical Validation Plan (Placeholder)

- `[[N_SEEDS]]` random seeds per model/condition
- Report mean ± std and confidence intervals
- Paired comparison over identical task instances
- Ablations:
  - no Dream
  - no entropy trigger (interval only)
  - no pruning
  - no graph summaries (rules only)
  - raw summaries vs heuristic rules

## 10. Discussion

The prototype demonstrates a practical pattern: memory performance can improve by **changing representation**, not only by increasing storage. The Sleep cycle acts as an adaptive compression and reset mechanism. In the simulated setting, it reduced token budget and uncertainty while improving task outcomes.

The strongest conceptual contribution is the move from:

- **Narrative retention** (“what happened”) to
- **Procedural retention** (“what tends to work under conditions \(X\)”).

This distinction matters for long-horizon agents where repeated task motifs appear across turns or sessions.

## 11. Limitations

This work is currently limited by the prototype setup:

- The decision policy is simulated and has access to a hidden “best action” via the environment.
- The tasks are synthetic proxies, not real interactive game environments.
- Entropy is computed over simulated action distributions, not true next-token distributions.
- The semantic graph is lightweight and domain-specific; richer relation extraction is not yet implemented.
- Dream self-play is a heuristic simulator, not a learned world model.

Therefore, current results should be treated as **architecture validation**, not definitive evidence of LLM capability gains.

## 12. Conclusion

Agentic Sleep provides a concrete framework for long-horizon agent stability through off-cycle consolidation, heuristic abstraction, and synthetic hardening. The current prototype demonstrates that such a design can improve performance-per-token in a controlled simulation. The next stage is direct LLM integration with the same memory loop and evaluation protocol, enabling a rigorous test of whether structural abstraction outperforms raw-context retention in real autonomous agent workloads.

## Appendix A: Reproduction (Current Prototype)

Run the prototype benchmark:

```bash
python3 -m agentic_sleep.cli --games 20 --turns 50
```

Key files:

- `agentic_sleep/engine.py`
- `agentic_sleep/consolidation.py`
- `agentic_sleep/dream.py`
- `agentic_sleep/memory.py`
- `agentic_sleep/experiment.py`

## Appendix B: Suggested Next Fill-Ins for the LLM Version

- Replace simulated `thought` string with model-generated rationale summary (optional logging only)
- Add logprob-based entropy extraction
- Add real tool traces (e.g., coding agent actions) into `EpisodeStep.tags`
- Persist MTEM/LTSM across sessions
- Evaluate on code repair / planning / debugging benchmarks

