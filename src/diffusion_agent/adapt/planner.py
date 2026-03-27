"""Adaptation planner — classifies failures and generates hypotheses.

The planner takes scan results, runtime logs, and previous attempt history
to produce a single hypothesis for the next iteration.  It does NOT apply
patches or execute commands.
"""

from __future__ import annotations

import re
from pathlib import Path

from diffusion_agent.adapt.types import (
    AdaptationTask,
    FailureCategory,
    Hypothesis,
    IterationRecord,
    RunResult,
)
from diffusion_agent.tools.code_migrator import MigrationPlan, RuleRegistry
from diffusion_agent.tools.code_scanner import Finding, PatternType
from diffusion_agent.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pattern → FailureCategory mapping
# ---------------------------------------------------------------------------

_PATTERN_CATEGORY: dict[PatternType, FailureCategory] = {
    PatternType.CUDA_CALL: FailureCategory.DEVICE_SELECTION,
    PatternType.CUDA_TO: FailureCategory.DEVICE_SELECTION,
    PatternType.CUDA_API: FailureCategory.CUDA_ONLY_API,
    PatternType.CUDA_DEVICE_STR: FailureCategory.DEVICE_SELECTION,
    PatternType.FLOAT64: FailureCategory.DTYPE_AUTOCAST,
    PatternType.NCCL: FailureCategory.DISTRIBUTED_BACKEND,
    PatternType.FLASH_ATTN: FailureCategory.CUSTOM_EXTENSION,
    PatternType.XFORMERS: FailureCategory.CUSTOM_EXTENSION,
    PatternType.BFLOAT16: FailureCategory.DTYPE_AUTOCAST,
    PatternType.DISTRIBUTED: FailureCategory.DISTRIBUTED_BACKEND,
    PatternType.SDPA: FailureCategory.CUDA_ONLY_API,
    PatternType.CUDA_AMP: FailureCategory.CUDA_ONLY_API,
    PatternType.TORCH_IMPORT: FailureCategory.IMPORT_MODULE,
    PatternType.AUTOCAST_DTYPE: FailureCategory.DTYPE_AUTOCAST,
    PatternType.DTYPE_ASSERT: FailureCategory.LOGIC_BUG,
}

# Priority order for deterministic task decomposition
_CATEGORY_PRIORITY: list[FailureCategory] = [
    FailureCategory.CUSTOM_EXTENSION,   # flash_attn, xformers first (blocks imports)
    FailureCategory.IMPORT_MODULE,
    FailureCategory.DEVICE_SELECTION,    # .cuda() → .npu()
    FailureCategory.CUDA_ONLY_API,       # torch.cuda.* → torch.npu.*
    FailureCategory.DISTRIBUTED_BACKEND, # nccl → hccl
    FailureCategory.DTYPE_AUTOCAST,      # float64, bfloat16
    FailureCategory.UNSUPPORTED_OP,
    FailureCategory.OOM,
    FailureCategory.SYNTAX_ERROR,
    FailureCategory.LOGIC_BUG,
    FailureCategory.ENVIRONMENT_SETUP,
    FailureCategory.RUNTIME_REGRESSION,
    FailureCategory.UNKNOWN_BLOCKER,
]


# ---------------------------------------------------------------------------
# Error log trimming — prevents token explosion in LLM context
# ---------------------------------------------------------------------------

def trim_error_log(stderr: str, max_lines: int = 30) -> str:
    """Extract the most relevant traceback from stderr, capped at *max_lines*.

    Strategy:
      1. Find the LAST ``Traceback (most recent call last):`` block.
      2. Collect from that line through the final exception line.
      3. If the block exceeds *max_lines*, keep the traceback header,
         the last frames, and always the exception line.
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
        block = lines[last_tb_start:]
        if len(block) <= max_lines:
            return "\n".join(block)

        # Block too long — keep header + tail with exception line
        # Always include: first line (Traceback header) + last (max_lines-1) lines
        header = [block[0]]
        tail = block[-(max_lines - 1):]
        return "\n".join(header + tail)

    # No traceback — return tail lines
    tail = lines[-max_lines:] if len(lines) > max_lines else lines
    return "\n".join(tail)


# ---------------------------------------------------------------------------
# Reflection context builder — "error notebook" for the LLM
# ---------------------------------------------------------------------------

def _build_reflection_context(task: AdaptationTask) -> str:
    """Build a reflection prompt from the task's previous attempt history.

    Returns an empty string if there are no prior attempts.
    """
    if not task.attempts:
        return ""

    last = task.attempts[-1]
    parts: list[str] = []

    parts.append(
        f"[REFLECTION] Previous attempt #{last.attempt} "
        f"tried: {last.hypothesis.proposed_action}"
    )
    if last.error_signature:
        parts.append(f"  Result: FAILED with error -> {last.error_signature}")
    else:
        parts.append(f"  Result: verdict={last.verdict.value}, no error signature captured")
    parts.append(
        "  Analyze why the previous fix failed and propose a DIFFERENT approach."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Traceback file extraction — maps runtime errors to source files
# ---------------------------------------------------------------------------

_TB_FILE_RE = re.compile(r'File "([^"]+\.py)", line (\d+)')

# Paths containing any of these substrings are considered stdlib/venv noise
_SKIP_SUBSTRINGS = (
    "site-packages", "/lib/python", "/usr/lib/",
    "<frozen", "<string>", ".venv/", "/.local/",
)


def parse_traceback_files(
    stderr: str,
    repo_path: Path | None = None,
    remote_workdir: str | None = None,
) -> list[str]:
    """Extract file paths from a Python traceback and resolve to local repo paths.

    Handles both local paths (already in repo) and remote paths (from SSH execution).
    Filters out stdlib / site-packages.  Returns **local absolute paths** that exist
    on disk, ordered by last-seen (the most relevant frame is last in a traceback).
    """
    if not stderr or repo_path is None:
        return []

    raw_paths: list[str] = []
    for match in _TB_FILE_RE.finditer(stderr):
        raw_paths.append(match.group(1))

    if not raw_paths:
        return []

    resolved: list[str] = []
    seen: set[str] = set()

    # Process in reverse so the deepest (most relevant) frame comes first
    for raw in reversed(raw_paths):
        if any(skip in raw for skip in _SKIP_SUBSTRINGS):
            continue

        p = Path(raw)

        # 1. Direct resolution: path is already inside repo_path
        if p.is_absolute():
            try:
                rel = p.relative_to(repo_path)
                local = str(repo_path / rel)
                if local not in seen and Path(local).exists():
                    resolved.append(local)
                    seen.add(local)
                    continue
            except ValueError:
                pass

        # 2. Strip remote_workdir prefix to get a relative path
        if remote_workdir:
            remote_prefix = remote_workdir.rstrip("/") + "/"
            if raw.startswith(remote_prefix):
                rel = raw[len(remote_prefix):]
                local = str(repo_path / rel)
                if local not in seen and Path(local).exists():
                    resolved.append(local)
                    seen.add(local)
                    continue

        # 3. Fallback: match by filename within repo
        name = p.name
        candidates = list(repo_path.rglob(name))
        if len(candidates) == 1 and str(candidates[0]) not in seen:
            resolved.append(str(candidates[0]))
            seen.add(str(candidates[0]))

    return resolved


class AdaptPlanner:
    """Plans adaptation by decomposing findings into tasks and generating hypotheses."""

    def __init__(self, registry: RuleRegistry) -> None:
        self.registry = registry

    # ----- Feature decomposition -----

    def decompose_tasks(
        self,
        findings: list[Finding],
        plan: MigrationPlan,
    ) -> list[AdaptationTask]:
        """Decompose scan results into granular adaptation tasks.

        Groups findings by category and creates one task per category+file group.
        Tasks are ordered by priority so blocking issues are handled first.
        """
        # Group findings by (category, file)
        groups: dict[tuple[FailureCategory, str], list[Finding]] = {}
        for finding in findings:
            cat = _PATTERN_CATEGORY.get(finding.pattern_type, FailureCategory.UNKNOWN_BLOCKER)
            key = (cat, finding.file_path)
            groups.setdefault(key, []).append(finding)

        tasks: list[AdaptationTask] = []
        task_id = 0

        # Sort by category priority, then file
        sorted_keys = sorted(
            groups.keys(),
            key=lambda k: (_CATEGORY_PRIORITY.index(k[0]) if k[0] in _CATEGORY_PRIORITY else 99, k[1]),
        )

        for category, file_path in sorted_keys:
            file_findings = groups[(category, file_path)]
            task_id += 1
            pattern_types = {f.pattern_type.value for f in file_findings}
            tasks.append(AdaptationTask(
                id=f"task-{task_id:03d}",
                name=f"{category.value}:{Path(file_path).name}",
                description=(
                    f"Migrate {len(file_findings)} {category.value} pattern(s) "
                    f"in {file_path}: {', '.join(sorted(pattern_types))}"
                ),
                category=category,
                target_files=[file_path],
            ))

        log.info("planner_decomposed", tasks=len(tasks), categories=len({k[0] for k in groups}))
        return tasks

    # ----- Hypothesis generation -----

    def generate_hypothesis(
        self,
        task: AdaptationTask,
        findings: list[Finding],
        plan: MigrationPlan,
        previous_iterations: list[IterationRecord] | None = None,
    ) -> Hypothesis | None:
        """Generate a single hypothesis for the given task.

        Uses the rule registry to find matching rules. If rules exist,
        proposes a deterministic rule-based fix. Otherwise, proposes
        an LLM-assisted fix.

        Deduplicates against the task's attempt history so the same
        hypothesis ID is never proposed twice for the same task.
        """
        seen_ids = task.seen_hypothesis_ids
        seen_errors = task.seen_error_signatures

        # Filter findings for this task's files
        task_findings = [f for f in findings if f.file_path in task.target_files]
        if not task_findings:
            return None

        # Check if deterministic rules can handle these findings
        matched_rules: list[str] = []
        for finding in task_findings:
            rule = self.registry.match(finding)
            if rule is not None:
                matched_rules.append(rule.name)

        if matched_rules:
            hyp_id = f"hyp-{task.id}-rules"
            if hyp_id not in seen_ids:
                return Hypothesis(
                    id=hyp_id,
                    category=task.category,
                    description=f"Apply migration rules to {Path(task.target_files[0]).name}",
                    target_files=task.target_files,
                    proposed_action=f"Apply rules: {', '.join(set(matched_rules))}",
                    source="rule",
                )
            log.info("planner_skip_seen_hypothesis", hyp_id=hyp_id, task=task.id)

        # No deterministic rules (or already tried) — propose LLM-assisted fix
        pattern_desc = ", ".join({f.pattern_type.value for f in task_findings})
        attempt_num = task.attempt_count + 1
        hyp_id = f"hyp-{task.id}-llm-{attempt_num}"

        # If the last error signature was already seen, signal repeated failure
        if task.last_error_signature and task.last_error_signature in seen_errors:
            repeated_count = sum(
                1 for a in task.attempts if a.error_signature == task.last_error_signature
            )
            if repeated_count >= 2:
                log.info(
                    "planner_repeated_error",
                    task=task.id,
                    sig=task.last_error_signature[:80],
                    seen=repeated_count,
                )
                return None  # signal: no new hypothesis available

        # Build reflection context from prior attempts (the "error notebook")
        reflection = _build_reflection_context(task)

        if reflection:
            proposed_action = (
                f"{reflection}\n"
                f"Use LLM to propose fix for: {pattern_desc}"
            )
        else:
            proposed_action = f"Use LLM to propose fix for: {pattern_desc}"

        return Hypothesis(
            id=hyp_id,
            category=task.category,
            description=f"LLM-assisted fix for {pattern_desc} in {Path(task.target_files[0]).name}",
            target_files=task.target_files,
            proposed_action=proposed_action,
            source="llm",
        )

    def generate_runtime_hypothesis(
        self,
        run_result: RunResult,
        previous_iterations: list[IterationRecord],
        repo_path: Path | None = None,
        remote_workdir: str | None = None,
    ) -> Hypothesis | None:
        """Generate a hypothesis from a runtime failure (post-rule-application).

        This is used in the iterative loop when deterministic rules have been
        applied but the code still fails at runtime.  Uses trimmed error
        context to keep the LLM prompt within token budget.

        When *repo_path* is provided, the traceback is parsed to extract
        the target file(s) so the PatchWorker knows which files to patch.
        """
        from diffusion_agent.adapt.judge import classify_failure

        if run_result.exit_code == 0:
            return None

        category = classify_failure(run_result.stderr)

        # Check if we've already tried this exact error
        seen_sigs = {it.run_after.error_signature for it in previous_iterations}
        if run_result.error_signature in seen_sigs:
            log.info("planner_repeated_error", sig=run_result.error_signature[:80])
            return None  # signal repeated failure

        # Use trimmed error context instead of raw error_signature
        trimmed = trim_error_log(run_result.stderr)
        description = f"Fix runtime failure:\n{trimmed}" if trimmed else (
            f"Fix runtime failure: {run_result.error_signature[:80]}"
        )

        # Parse traceback to identify target files
        target_files = parse_traceback_files(
            run_result.stderr, repo_path, remote_workdir,
        )
        if target_files:
            log.info(
                "planner_runtime_target_files",
                files=target_files[:5],
                error_sig=run_result.error_signature[:60],
            )

        return Hypothesis(
            id=f"hyp-runtime-{len(previous_iterations) + 1}",
            category=category,
            description=description,
            target_files=target_files,
            proposed_action=f"Investigate and fix: {category.value}",
            source="llm",
            deepest_file=target_files[0] if target_files else None,
        )

    # ----- Previous attempt analysis -----

    def count_no_progress(self, iterations: list[IterationRecord]) -> int:
        """Count consecutive iterations with no progress from the tail."""
        count = 0
        for it in reversed(iterations):
            if it.verdict.value in ("unchanged", "regressed", "different_failure"):
                count += 1
            else:
                break
        return count
