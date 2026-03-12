"""LangGraph StateGraph wiring for the two-stage agent architecture."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from diffusion_agent.agents.coding_agent import coding_node
from diffusion_agent.agents.init_agent import init_node
from diffusion_agent.agents.router import route_phase, should_continue
from diffusion_agent.agents.state import AgentState


def build_graph() -> StateGraph:
    """Build and compile the two-stage agent graph.

    Flow:
        START -> router -> init_agent -> coding_agent -> (loop or done)
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("init_agent", init_node)
    graph.add_node("coding_agent", coding_node)

    # Entry: route based on phase
    graph.set_conditional_entry_point(route_phase, {
        "init_agent": "init_agent",
        "coding_agent": "coding_agent",
        "done": END,
    })

    # After init, always go to coding
    graph.add_edge("init_agent", "coding_agent")

    # After coding, check if more work remains
    graph.add_conditional_edges("coding_agent", should_continue, {
        "coding_agent": "coding_agent",
        "done": END,
    })

    return graph.compile()
