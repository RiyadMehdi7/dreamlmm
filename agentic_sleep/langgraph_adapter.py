from __future__ import annotations

"""
Optional LangGraph adapter for the prototype.

This module is not required for the stdlib simulation to run. If `langgraph` is
installed, `build_langgraph_stategraph()` provides a minimal StateGraph wiring
that mirrors the prompt's waking -> sleep_consolidation -> dream_simulation cycle.
"""

from typing import Any, TypedDict

from .engine import calculate_entropy_spike


class LangGraphAgentState(TypedDict, total=False):
    messages: list[str]
    heuristics: list[str]
    knowledge_graph: dict[str, Any]
    metrics: dict[str, float]


def build_langgraph_stategraph():
    try:
        from langgraph.graph import END, StateGraph
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "langgraph is not installed. Install it and re-run to build the StateGraph adapter."
        ) from exc

    def waking_agent(state: LangGraphAgentState) -> LangGraphAgentState:
        # Placeholder node: in production, call the live agent/LLM step here.
        return state

    def sleep_consolidation(state: LangGraphAgentState) -> LangGraphAgentState:
        # Placeholder node: run packer/consolidation and clear message buffer.
        state["messages"] = []
        return state

    def dream_simulation(state: LangGraphAgentState) -> LangGraphAgentState:
        # Placeholder node: run synthetic self-play / heuristic hardening.
        return state

    def route_after_waking(state: LangGraphAgentState) -> str:
        metrics = state.get("metrics", {})
        current = float(metrics.get("shannon_entropy", 0.0))
        prev = float(metrics.get("prev_entropy", 0.0))

        class _StubMetrics:
            shannon_entropy = current
            prev_entropy = prev

        class _StubState:
            metrics = _StubMetrics()

        return calculate_entropy_spike(_StubState(), threshold=0.16)

    graph = StateGraph(LangGraphAgentState)
    graph.add_node("waking_agent", waking_agent)
    graph.add_node("sleep_consolidation", sleep_consolidation)
    graph.add_node("dream_simulation", dream_simulation)

    graph.set_entry_point("waking_agent")
    graph.add_conditional_edges(
        "waking_agent",
        route_after_waking,
        {
            "trigger_sleep": "sleep_consolidation",
            "continue_waking": END,
        },
    )
    graph.add_edge("sleep_consolidation", "dream_simulation")
    graph.add_edge("dream_simulation", END)
    return graph.compile()

