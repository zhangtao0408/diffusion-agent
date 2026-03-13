"""Shared data types for the Phase 2 adaptation loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Failure taxonomy
# ---------------------------------------------------------------------------

class FailureCategory(str, Enum):
    """Fixed taxonomy for classifying adaptation failures."""

    ENVIRONMENT_SETUP = "environment_setup"
    IMPORT_MODULE = "import_module"
    DEVICE_SELECTION = "device_selection"
    CUDA_ONLY_API = "cuda_only_api"
    DISTRIBUTED_BACKEND = "distributed_backend"
    UNSUPPORTED_OP = "unsupported_op"
    CUSTOM_EXTENSION = "custom_extension"
    DTYPE_AUTOCAST = "dtype_autocast"
    RUNTIME_REGRESSION = "runtime_regression"
    UNKNOWN_BLOCKER = "unknown_blocker"


# ---------------------------------------------------------------------------
# Hypothesis — single unit of work per iteration
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    """One candidate fix to try in a single iteration."""

    id: str
    category: FailureCategory
    description: str
    target_files: list[str]
    proposed_action: str
    source: str = "planner"  # "planner", "rule", "llm"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category.value,
            "description": self.description,
            "target_files": self.target_files,
            "proposed_action": self.proposed_action,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Validation run result
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Captured output from a validation run."""

    exit_code: int
    stdout: str
    stderr: str
    error_signature: str  # normalized key for comparing failures
    duration_s: float
    command: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "exit_code": self.exit_code,
            "stdout_tail": self.stdout[-500:] if self.stdout else "",
            "stderr_tail": self.stderr[-500:] if self.stderr else "",
            "error_signature": self.error_signature,
            "duration_s": self.duration_s,
            "command": self.command,
        }


# ---------------------------------------------------------------------------
# Progress verdict — judge's decision
# ---------------------------------------------------------------------------

class Verdict(str, Enum):
    """Judge's assessment of an iteration's outcome."""

    FIXED = "fixed"
    IMPROVED = "improved"
    UNCHANGED = "unchanged"
    REGRESSED = "regressed"
    DIFFERENT_FAILURE = "different_failure"
    BLOCKED = "blocked"


# ---------------------------------------------------------------------------
# Iteration record — one complete cycle
# ---------------------------------------------------------------------------

@dataclass
class IterationRecord:
    """Full record of one adapt-run-judge cycle."""

    iteration: int
    hypothesis: Hypothesis
    patch_description: str
    files_changed: list[str]
    run_before: RunResult | None
    run_after: RunResult
    verdict: Verdict
    accepted: bool
    commit_sha: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "hypothesis": self.hypothesis.to_dict(),
            "patch_description": self.patch_description,
            "files_changed": self.files_changed,
            "run_before": self.run_before.to_dict() if self.run_before else None,
            "run_after": self.run_after.to_dict(),
            "verdict": self.verdict.value,
            "accepted": self.accepted,
            "commit_sha": self.commit_sha,
        }


# ---------------------------------------------------------------------------
# Stop conditions
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Execution configuration
# ---------------------------------------------------------------------------

@dataclass
class ExecutionConfig:
    """Configuration for how the runner executes validation commands.

    Controls whether execution is local or remote (SSH), and provides
    all parameters needed for the selected mode.
    """

    mode: str = "local"  # "local" or "ssh"

    # SSH settings (ignored when mode="local")
    ssh_host: str | None = None
    ssh_user: str = "root"
    ssh_port: int = 22
    remote_workdir: str | None = None

    # Environment settings (used in both modes)
    conda_env: str | None = None
    pre_commands: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)

    # Execution settings
    timeout: int = 120

    # Validation command (custom smoke command for model adaptation)
    validation_command: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "ssh_host": self.ssh_host,
            "ssh_user": self.ssh_user,
            "ssh_port": self.ssh_port,
            "remote_workdir": self.remote_workdir,
            "conda_env": self.conda_env,
            "pre_commands": self.pre_commands,
            "timeout": self.timeout,
            "validation_command": self.validation_command,
        }


class StopReason(str, Enum):
    """Why the adaptation loop terminated."""

    INFERENCE_SUCCESS = "inference_success"
    MAX_ITERATIONS = "max_iterations"
    REPEATED_NO_PROGRESS = "repeated_no_progress"
    BLOCKER_DETECTED = "blocker_detected"
    TASK_BLOCKED = "task_blocked"
    ALL_RULES_APPLIED = "all_rules_applied"


# ---------------------------------------------------------------------------
# Adaptation task — one unit in the feature decomposition
# ---------------------------------------------------------------------------

@dataclass
class AdaptationTask:
    """A granular migration task produced by feature decomposition."""

    id: str
    name: str
    description: str
    category: FailureCategory
    target_files: list[str] = field(default_factory=list)
    status: str = "pending"  # pending / in_progress / completed / blocked
    blocker_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "target_files": self.target_files,
            "status": self.status,
            "blocker_reason": self.blocker_reason,
        }


# ---------------------------------------------------------------------------
# Adaptation session state
# ---------------------------------------------------------------------------

@dataclass
class AdaptationState:
    """Mutable state for one full adaptation session."""

    repo_path: Path
    model_name: str
    max_iterations: int = 20
    no_progress_limit: int = 3

    # Progress tracking
    iteration: int = 0
    iterations: list[IterationRecord] = field(default_factory=list)
    tasks: list[AdaptationTask] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    stop_reason: StopReason | None = None

    # Accumulated results
    files_modified: set[str] = field(default_factory=set)
    total_rules_applied: int = 0
    consecutive_no_progress: int = 0
    batch_migration_results: list[Any] = field(default_factory=list)  # MigrationResult objects from batch pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_path": str(self.repo_path),
            "model_name": self.model_name,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
            "tasks": [t.to_dict() for t in self.tasks],
            "blockers": self.blockers,
            "iterations": [i.to_dict() for i in self.iterations],
            "files_modified": sorted(self.files_modified),
            "total_rules_applied": self.total_rules_applied,
        }
