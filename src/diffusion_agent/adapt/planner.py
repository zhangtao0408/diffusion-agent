"""Adaptation planner — classifies failures and generates hypotheses.

The planner takes scan results, runtime logs, and previous attempt history
to produce a single hypothesis for the next iteration.  It does NOT apply
patches or execute commands.
"""

from __future__ import annotations

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
    FailureCategory.ENVIRONMENT_SETUP,
    FailureCategory.RUNTIME_REGRESSION,
    FailureCategory.UNKNOWN_BLOCKER,
]


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

        return Hypothesis(
            id=hyp_id,
            category=task.category,
            description=f"LLM-assisted fix for {pattern_desc} in {Path(task.target_files[0]).name}",
            target_files=task.target_files,
            proposed_action=f"Use LLM to propose fix for: {pattern_desc}",
            source="llm",
        )

    def generate_runtime_hypothesis(
        self,
        run_result: RunResult,
        previous_iterations: list[IterationRecord],
    ) -> Hypothesis | None:
        """Generate a hypothesis from a runtime failure (post-rule-application).

        This is used in the iterative loop when deterministic rules have been
        applied but the code still fails at runtime.
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

        return Hypothesis(
            id=f"hyp-runtime-{len(previous_iterations) + 1}",
            category=category,
            description=f"Fix runtime failure: {run_result.error_signature[:80]}",
            target_files=[],  # runner determines affected files
            proposed_action=f"Investigate and fix: {category.value}",
            source="llm",
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
