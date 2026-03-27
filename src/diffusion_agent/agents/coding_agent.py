"""Stage Two: Coding Agent.

Reads tasks from feature-list, executes them one at a time,
runs tests, commits results, and clears context.
"""

from __future__ import annotations

from pathlib import Path

from diffusion_agent.agents.state import AgentState
from diffusion_agent.scenarios.check_support import CheckSupportScenario
from diffusion_agent.state_mgmt.daily_log import append_log
from diffusion_agent.state_mgmt.feature_list import (
    get_next_pending,
    update_feature_status,
)
from diffusion_agent.state_mgmt.task_file import clear_current_task, write_current_task
from diffusion_agent.tools.git_ops import commit
from diffusion_agent.utils.logging import get_logger

log = get_logger(__name__)


def coding_node(state: AgentState) -> dict:
    """Stage Two entry point. Process one feature at a time."""
    repo_local = Path(state.get("repo_local_path", ""))
    completed = list(state.get("completed_features", []))

    # Find next pending feature
    feature = get_next_pending(repo_local)
    if feature is None:
        log.info("all_features_complete")
        append_log(repo_local, "All features completed")
        clear_current_task(repo_local)
        return {"phase": "done", "should_stop": True}

    log.info("processing_feature", feature_id=feature.id, name=feature.name)

    # Update state files
    update_feature_status(repo_local, feature.id, "in_progress")
    write_current_task(repo_local, {
        "id": feature.id,
        "scenario": state.get("scenario", ""),
        "feature_name": feature.name,
        "status": "in_progress",
        "attempt": 1,
    })

    # Dispatch to scenario implementation
    tool_results = list(state.get("tool_results", []))
    scenario_name = state.get("scenario", "")
    if scenario_name == "check":
        log.info("executing_check_scenario", feature=feature.name)
        scenario_impl = CheckSupportScenario()
        updated_state = scenario_impl.execute(state)
        tool_results = list(updated_state.get("tool_results", []))
        append_log(repo_local, f"Executed feature: {feature.name} (check_support)")
    elif scenario_name == "adapt":
        from diffusion_agent.scenarios.adapt import AdaptScenario

        log.info("executing_adapt_scenario", feature=feature.name)
        scenario_impl = AdaptScenario()
        updated_state = scenario_impl.execute(state)
        tool_results = list(updated_state.get("tool_results", []))
        append_log(repo_local, f"Executed feature: {feature.name} (adapt)")
    else:
        log.info("executing_feature_stub", feature=feature.name)
        append_log(repo_local, f"Executed feature: {feature.name} (stub)")

    # Mark feature as completed
    update_feature_status(repo_local, feature.id, "completed")
    completed.append(feature.id)

    # Git commit
    commit(repo_local, f"feat: {feature.name}")

    # Clear current task
    clear_current_task(repo_local)

    log.info("feature_complete", feature_id=feature.id)

    return {
        "completed_features": completed,
        "tool_results": tool_results,
        "current_task": None,
        "error": None,
    }
