"""Conditional edge logic for the agent graph."""

from __future__ import annotations

from diffusion_agent.agents.state import AgentState


def route_phase(state: AgentState) -> str:
    """Route to the appropriate node based on the current phase."""
    phase = state.get("phase", "init")
    if phase == "init":
        return "init_agent"
    elif phase == "coding":
        return "coding_agent"
    else:
        return "done"


def should_continue(state: AgentState) -> str:
    """After coding_agent runs, decide whether to continue or finish."""
    if state.get("should_stop", False):
        return "done"
    if state.get("phase") == "done":
        return "done"
    return "coding_agent"
