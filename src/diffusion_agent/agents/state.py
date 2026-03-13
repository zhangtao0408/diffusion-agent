"""Agent state schema shared across all graph nodes."""

from __future__ import annotations

from enum import Enum
from typing import Any, TypedDict


class Scenario(str, Enum):
    CHECK = "check"
    ADAPT = "adapt"
    ANALYZE = "analyze"
    OPTIMIZE = "optimize"
    VERIFY = "verify"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class CurrentTask(TypedDict, total=False):
    id: str
    scenario: str
    feature_name: str
    status: str
    attempt: int
    error: str | None


class AgentState(TypedDict, total=False):
    # Input
    repo_url: str
    model_name: str
    scenario: str

    # Runtime
    phase: str  # "init", "coding", "done"
    repo_local_path: str
    current_task: CurrentTask | None
    completed_features: list[str]
    tool_results: list[dict[str, Any]]

    # NPU
    torch_npu_version: str | None

    # Control
    git_branch: str
    error: str | None
    should_stop: bool
