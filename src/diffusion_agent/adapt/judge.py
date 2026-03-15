"""Progress judge — compares before/after run results and decides verdict.

The judge does NOT decide what to do next. It only classifies the outcome
of a single iteration so the supervisor can make informed decisions.
"""

from __future__ import annotations

import re

from diffusion_agent.adapt.types import (
    AdaptationTask,
    FailureCategory,
    RunResult,
    TaskStopReason,
    Verdict,
)
from diffusion_agent.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Structured patterns — checked first, in order (most specific wins)
# ---------------------------------------------------------------------------

_BLOCKER_PATTERNS: list[tuple[str, FailureCategory]] = [
    # NPU dtype limitations — fixable via view_as_real / safe-cast (MUST precede generic "not implemented")
    (r"not implemented for DT_(COMPLEX128|COMPLEX64|DOUBLE)", FailureCategory.DTYPE_AUTOCAST),
    # Unsupported ops (generic fallback)
    (r"not implemented for", FailureCategory.UNSUPPORTED_OP),
    (r"Could not run .*on", FailureCategory.UNSUPPORTED_OP),
    (r"No kernel registered for", FailureCategory.UNSUPPORTED_OP),
    # Custom extensions
    (r"No module named '(flash_attn|xformers|triton)'", FailureCategory.CUSTOM_EXTENSION),
    # OOM (must precede generic device/cuda keywords)
    (r"out of memory", FailureCategory.OOM),
    (r"OutOfMemoryError", FailureCategory.OOM),
    # Syntax errors (must precede generic heuristics)
    (r"SyntaxError:", FailureCategory.SYNTAX_ERROR),
    (r"IndentationError:", FailureCategory.SYNTAX_ERROR),
    (r"TabError:", FailureCategory.SYNTAX_ERROR),
    # Logic / shape bugs
    (r"shape mismatch", FailureCategory.LOGIC_BUG),
    (r"size of tensor \w+ \(\d+\) must match", FailureCategory.LOGIC_BUG),
    (r"shapes cannot be multiplied", FailureCategory.LOGIC_BUG),
    (r"AssertionError:", FailureCategory.LOGIC_BUG),
    # Device availability
    (r"CUDA.*not available", FailureCategory.DEVICE_SELECTION),
    (r"nccl.*not available", FailureCategory.DISTRIBUTED_BACKEND),
]


def classify_failure(stderr: str) -> FailureCategory:
    """Classify a failure's stderr into a FailureCategory."""
    if not stderr:
        return FailureCategory.UNKNOWN_BLOCKER

    lower = stderr.lower()

    # Check structured patterns first
    for pattern, category in _BLOCKER_PATTERNS:
        if re.search(pattern, stderr, re.IGNORECASE):
            return category

    # Heuristic keyword matching (order matters — more specific first)
    if "ModuleNotFoundError" in stderr or "ImportError" in stderr:
        return FailureCategory.IMPORT_MODULE
    if "no module named" in lower:
        return FailureCategory.IMPORT_MODULE
    if "torch.cuda" in stderr:
        return FailureCategory.CUDA_ONLY_API
    if any(k in lower for k in ["cuda", "npu", "device", ".to("]):
        return FailureCategory.DEVICE_SELECTION
    if any(k in lower for k in ["nccl", "hccl", "distributed", "gloo"]):
        return FailureCategory.DISTRIBUTED_BACKEND
    if any(k in lower for k in ["float64", "double", "autocast", "dtype"]):
        return FailureCategory.DTYPE_AUTOCAST
    if any(k in lower for k in ["pip", "install", "setup.py", "requirements"]):
        return FailureCategory.ENVIRONMENT_SETUP

    return FailureCategory.UNKNOWN_BLOCKER


def is_blocker(category: FailureCategory) -> bool:
    """Whether this failure category is a hard blocker that cannot be patched."""
    return category in {
        FailureCategory.UNSUPPORTED_OP,
        FailureCategory.CUSTOM_EXTENSION,
    }


# ---------------------------------------------------------------------------
# Error context extraction — trims noisy NPU stderr to the relevant traceback
# ---------------------------------------------------------------------------

def extract_error_context(stderr: str, max_lines: int = 30) -> str:
    """Extract the most relevant traceback block from stderr.

    Strategy:
      1. Find the LAST ``Traceback (most recent call last):`` block.
      2. Return from that line through the exception line.
      3. Cap at *max_lines* to prevent token explosion.
      4. If no traceback found, return the tail lines.
    """
    if not stderr:
        return ""

    lines = stderr.splitlines()

    # Find the start of the last traceback block
    last_tb_start = -1
    for i, line in enumerate(lines):
        if "Traceback (most recent call last):" in line:
            last_tb_start = i

    if last_tb_start >= 0:
        # Extract from the last traceback to end (the exception line follows)
        block = lines[last_tb_start:]
        if len(block) > max_lines:
            block = block[:max_lines]
        return "\n".join(block)

    # No traceback — return tail lines
    tail = lines[-max_lines:] if len(lines) > max_lines else lines
    return "\n".join(tail)


# ---------------------------------------------------------------------------
# Task-level progress evaluation — consecutive blocker escalation
# ---------------------------------------------------------------------------

# Categories that trigger auto-block when seen N consecutive times.
# Only truly un-patchable categories belong here.
_ESCALATION_CATEGORIES: set[FailureCategory] = {
    FailureCategory.UNSUPPORTED_OP,
}

_ESCALATION_THRESHOLD = 3


def evaluate_task_progress(
    task: AdaptationTask,
    attempt_stderrs: list[str],
) -> TaskStopReason | None:
    """Check whether a task should be force-stopped based on attempt history.

    Detects: N consecutive attempts whose stderr classifies as the same
    escalation-eligible category (e.g. UNSUPPORTED_OP).  When the threshold
    is reached, returns ``TaskStopReason.BLOCKED``.

    Args:
        task: The task with recorded attempts.
        attempt_stderrs: stderr strings for each attempt, in order.

    Returns:
        ``TaskStopReason.BLOCKED`` if escalation fires, else ``None``.
    """
    if len(attempt_stderrs) < _ESCALATION_THRESHOLD:
        return None

    # Classify the last N stderrs
    tail = attempt_stderrs[-_ESCALATION_THRESHOLD:]
    categories = [classify_failure(s) for s in tail]

    # Check: all the same AND in the escalation set
    if len(set(categories)) == 1 and categories[0] in _ESCALATION_CATEGORIES:
        log.info(
            "judge_escalation_blocked",
            task=task.id,
            category=categories[0].value,
            consecutive=_ESCALATION_THRESHOLD,
        )
        return TaskStopReason.BLOCKED

    return None


# ---------------------------------------------------------------------------
# AdaptJudge — single-iteration verdict
# ---------------------------------------------------------------------------


class AdaptJudge:
    """Compares before/after run results to decide progress verdict."""

    def judge(self, before: RunResult | None, after: RunResult) -> Verdict:
        """Compare two run results and return a verdict.

        Args:
            before: Result from before the patch (None on first iteration).
            after: Result from after the patch.

        Returns:
            A Verdict indicating what happened.
        """
        # Success case
        if after.exit_code == 0:
            if before is None or before.exit_code != 0:
                return Verdict.FIXED
            return Verdict.FIXED  # was already passing, still passing

        # First iteration with no baseline — just classify
        if before is None:
            # Failure on first run, check if blocked
            category = classify_failure(after.stderr)
            if is_blocker(category):
                return Verdict.BLOCKED
            return Verdict.UNCHANGED  # no baseline to compare against

        # Both failed — compare signatures
        if before.exit_code != 0 and after.exit_code != 0:
            before_sig = before.error_signature
            after_sig = after.error_signature

            # Same error = unchanged
            if before_sig == after_sig:
                return Verdict.UNCHANGED

            # Check if blocked (before checking improved — blockers take priority)
            category = classify_failure(after.stderr)
            if is_blocker(category):
                return Verdict.BLOCKED

            # Different error — could be progress or regression
            # Heuristic: if we moved past the old error, that's progress
            if _is_later_error(before_sig, after_sig, after.stderr):
                return Verdict.IMPROVED

            return Verdict.DIFFERENT_FAILURE

        # Was passing, now failing = regression
        if before.exit_code == 0 and after.exit_code != 0:
            return Verdict.REGRESSED

        return Verdict.UNCHANGED

    def should_accept(self, verdict: Verdict) -> bool:
        """Whether a verdict means the patch should be committed."""
        return verdict in {Verdict.FIXED, Verdict.IMPROVED}

    def should_stop(self, verdict: Verdict) -> bool:
        """Whether a verdict means the loop should terminate."""
        return verdict in {Verdict.FIXED, Verdict.BLOCKED}


def _is_later_error(before_sig: str, after_sig: str, stderr: str) -> bool:
    """Heuristic: did we move past the previous error to a new one?

    This is imperfect but useful. If the old error signature no longer
    appears anywhere in the new stderr, we likely fixed it and hit something new.
    """
    if not before_sig:
        return False

    # Extract the core error type from before (e.g., "ModuleNotFoundError")
    before_type = before_sig.split(":")[0].strip()
    if before_type and before_type not in stderr:
        return True

    return False
