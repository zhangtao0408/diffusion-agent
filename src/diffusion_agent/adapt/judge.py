"""Progress judge — compares before/after run results and decides verdict.

The judge does NOT decide what to do next. It only classifies the outcome
of a single iteration so the supervisor can make informed decisions.
"""

from __future__ import annotations

import re

from diffusion_agent.adapt.types import FailureCategory, RunResult, Verdict
from diffusion_agent.utils.logging import get_logger

log = get_logger(__name__)

# Patterns that indicate hard blockers (unsupported ops, missing extensions)
_BLOCKER_PATTERNS: list[tuple[str, FailureCategory]] = [
    (r"not implemented for", FailureCategory.UNSUPPORTED_OP),
    (r"Could not run .*on", FailureCategory.UNSUPPORTED_OP),
    (r"No kernel registered for", FailureCategory.UNSUPPORTED_OP),
    (r"No module named '(flash_attn|xformers|triton)'", FailureCategory.CUSTOM_EXTENSION),
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
