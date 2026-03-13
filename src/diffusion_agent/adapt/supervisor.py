"""Adaptation supervisor — orchestrates the full adapt-run-judge loop.

The supervisor is the top-level controller. It does NOT directly edit files.
Instead it:
  1. Uses the **planner** to decompose tasks and generate hypotheses
  2. Uses the **patch worker** to generate and apply patches
  3. Uses the **runner** to execute validation
  4. Uses the **judge** to decide progress
  5. Uses **git memory** to commit accepted changes or rollback failures

This separation is the core architectural boundary of Phase 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from diffusion_agent.adapt.git_memory import GitMemory
from diffusion_agent.adapt.judge import AdaptJudge
from diffusion_agent.adapt.patch_worker import PatchWorker
from diffusion_agent.adapt.planner import AdaptPlanner
from diffusion_agent.adapt.runner import AdaptRunner
from diffusion_agent.adapt.types import (
    AdaptationState,
    AdaptationTask,
    ExecutionConfig,
    FailureCategory,
    Hypothesis,
    IterationRecord,
    StopReason,
    Verdict,
)
from diffusion_agent.tools.code_migrator import RuleRegistry, create_default_registry
from diffusion_agent.tools.code_scanner import Finding, scan_directory
from diffusion_agent.utils.logging import get_logger

log = get_logger(__name__)


class AdaptSupervisor:
    """Orchestrates the supervised adaptation loop.

    Flow:
      1. Scan repo for CUDA patterns
      2. Decompose into granular tasks (planner)
      3. For each task:
         a. Generate hypothesis (planner)
         b. Snapshot git state (git_memory)
         c. Apply patch (patch_worker)
         d. Run validation (runner)
         e. Judge progress (judge)
         f. Accept+commit or rollback (git_memory)
      4. Optionally enter iterative loop for runtime failures
      5. Produce structured result
    """

    def __init__(
        self,
        repo_path: Path,
        model_name: str = "unknown",
        max_iterations: int = 20,
        no_progress_limit: int = 3,
        llm: Any | None = None,
        ssh_host: str | None = None,
        conda_env: str | None = None,
        use_git: bool = True,
        execution_config: ExecutionConfig | None = None,
    ) -> None:
        self.registry: RuleRegistry = create_default_registry()
        self.planner = AdaptPlanner(self.registry)
        self.worker = PatchWorker(self.registry, llm=llm)
        self.runner = AdaptRunner(
            repo_path,
            ssh_host=ssh_host,
            conda_env=conda_env,
            execution_config=execution_config,
        )
        self.judge = AdaptJudge()
        self.git = GitMemory(repo_path) if use_git else None

        self.state = AdaptationState(
            repo_path=repo_path,
            model_name=model_name,
            max_iterations=max_iterations,
            no_progress_limit=no_progress_limit,
        )

    def run(self) -> AdaptationState:
        """Execute the full adaptation loop. Returns final state."""
        log.info("supervisor_start", repo=str(self.state.repo_path), model=self.state.model_name)

        # 0. Create adaptation branch
        if self.git:
            try:
                self.git.ensure_branch(self.state.model_name)
            except Exception:
                log.warning("supervisor_branch_failed")

        # 1. Scan repo
        findings = scan_directory(self.state.repo_path)
        log.info("supervisor_scan", findings=len(findings))

        if not findings:
            self.state.stop_reason = StopReason.ALL_RULES_APPLIED
            log.info("supervisor_no_findings")
            return self.state

        # 2. Build migration plan
        plan = self.registry.match_all(findings)

        # 3. Decompose into tasks
        self.state.tasks = self.planner.decompose_tasks(findings, plan)
        log.info("supervisor_tasks", count=len(self.state.tasks))

        # 4. Phase A: Apply deterministic rules in batch
        self._apply_batch_rules(findings)

        # 5. Phase B: Iterative loop for remaining issues
        self._iterative_loop(findings)

        # 6. Determine final stop reason if not set
        if self.state.stop_reason is None:
            if self.state.blockers:
                self.state.stop_reason = StopReason.BLOCKER_DETECTED
            else:
                self.state.stop_reason = StopReason.ALL_RULES_APPLIED

        log.info(
            "supervisor_done",
            iterations=self.state.iteration,
            files_modified=len(self.state.files_modified),
            stop_reason=self.state.stop_reason.value,
            blockers=len(self.state.blockers),
        )
        return self.state

    # ----- Phase A: Batch rule application -----

    def _apply_batch_rules(self, findings: list[Finding]) -> None:
        """Apply all deterministic migration rules in one batch."""
        patch_result = self.worker.apply_batch_rules(findings)

        # Store migration results for report generation
        if patch_result.migration_results:
            self.state.batch_migration_results = patch_result.migration_results

        if patch_result.files_changed:
            self.state.files_modified.update(patch_result.files_changed)
            self.state.total_rules_applied += len(patch_result.rules_applied)

            # Mark tasks as completed if their files were handled
            handled_files = set(patch_result.files_changed)
            for task in self.state.tasks:
                if all(f in handled_files for f in task.target_files):
                    task.status = "completed"

            # Commit batch rules
            if self.git:
                batch_hyp = Hypothesis(
                    id="batch-rules",
                    category=FailureCategory.DEVICE_SELECTION,
                    description="Batch apply deterministic migration rules",
                    target_files=patch_result.files_changed,
                    proposed_action=f"Applied {len(patch_result.rules_applied)} rules",
                    source="rule",
                )
                self.git.commit_iteration(0, batch_hyp, Verdict.IMPROVED)

            log.info(
                "supervisor_batch_done",
                files=len(patch_result.files_changed),
                rules=len(patch_result.rules_applied),
            )

    # ----- Phase B: Iterative loop -----

    def _iterative_loop(self, original_findings: list[Finding]) -> None:
        """Run the iterative adapt-run-judge loop for remaining tasks."""
        # Find pending tasks after batch phase
        pending = [t for t in self.state.tasks if t.status == "pending"]
        if not pending:
            return

        # Re-scan to get updated findings (after batch rules modified files)
        findings = scan_directory(self.state.repo_path)

        for task in pending:
            if self.state.stop_reason is not None:
                break

            if self.state.iteration >= self.state.max_iterations:
                self.state.stop_reason = StopReason.MAX_ITERATIONS
                break

            self._process_task(task, findings)

    def _process_task(self, task: AdaptationTask, findings: list[Finding]) -> None:
        """Process a single adaptation task through the hypothesis cycle."""
        task.status = "in_progress"
        self.state.iteration += 1

        log.info("supervisor_task_start", task=task.name, iteration=self.state.iteration)

        # 1. Generate hypothesis (planner)
        plan = self.registry.match_all(findings)
        hypothesis = self.planner.generate_hypothesis(task, findings, plan)

        if hypothesis is None:
            task.status = "completed"
            log.info("supervisor_task_skip", task=task.name, reason="no hypothesis")
            return

        # 2. Snapshot git state
        snapshot = self.git.snapshot() if self.git else ""

        # 3. Run validation BEFORE patch (baseline)
        run_before = self.runner.run_syntax_check(
            [f for f in task.target_files if Path(f).exists()]
        )

        # 4. Apply patch (patch worker)
        patch_result = self.worker.apply_patch(hypothesis, findings)

        # 5. Run validation AFTER patch
        run_after = self.runner.run_syntax_check(
            [f for f in task.target_files if Path(f).exists()]
        )

        # 6. Judge progress
        verdict = self.judge.judge(run_before, run_after)

        # 7. Accept or rollback
        accepted = self.judge.should_accept(verdict) or (
            verdict == Verdict.UNCHANGED and patch_result.files_changed
        )

        commit_sha = None
        if accepted:
            self.state.files_modified.update(patch_result.files_changed)
            self.state.total_rules_applied += len(patch_result.rules_applied)
            task.status = "completed"

            if self.git and self.git.has_changes():
                commit_sha = self.git.commit_iteration(
                    self.state.iteration, hypothesis, verdict,
                )
        else:
            # Rollback
            if self.git and snapshot:
                self.git.rollback_to(snapshot)
            task.status = "pending"  # can retry with different hypothesis

        # 8. Record iteration
        record = IterationRecord(
            iteration=self.state.iteration,
            hypothesis=hypothesis,
            patch_description=patch_result.description,
            files_changed=patch_result.files_changed,
            run_before=run_before,
            run_after=run_after,
            verdict=verdict,
            accepted=accepted,
            commit_sha=commit_sha,
        )
        self.state.iterations.append(record)

        # 9. Update no-progress counter
        if verdict in {Verdict.UNCHANGED, Verdict.REGRESSED, Verdict.DIFFERENT_FAILURE}:
            self.state.consecutive_no_progress += 1
        else:
            self.state.consecutive_no_progress = 0

        # 10. Check stop conditions
        if self.judge.should_stop(verdict):
            if verdict == Verdict.BLOCKED:
                self.state.blockers.append(
                    f"{task.name}: {hypothesis.description}"
                )
                task.status = "blocked"
                task.blocker_reason = run_after.error_signature

        if self.state.consecutive_no_progress >= self.state.no_progress_limit:
            self.state.stop_reason = StopReason.REPEATED_NO_PROGRESS

        log.info(
            "supervisor_iteration_done",
            iteration=self.state.iteration,
            task=task.name,
            verdict=verdict.value,
            accepted=accepted,
        )

    # ----- Blocker report -----

    def get_blocker_report(self) -> dict[str, Any]:
        """Produce a structured blocker report if adaptation is incomplete."""
        blocked_tasks = [t for t in self.state.tasks if t.status == "blocked"]
        pending_tasks = [t for t in self.state.tasks if t.status == "pending"]

        return {
            "model_name": self.state.model_name,
            "stop_reason": self.state.stop_reason.value if self.state.stop_reason else "unknown",
            "total_iterations": self.state.iteration,
            "files_modified": sorted(self.state.files_modified),
            "total_rules_applied": self.state.total_rules_applied,
            "blockers": [
                {
                    "task": t.name,
                    "category": t.category.value,
                    "reason": t.blocker_reason,
                    "files": t.target_files,
                }
                for t in blocked_tasks
            ],
            "pending_tasks": [t.to_dict() for t in pending_tasks],
            "iteration_summary": [
                {
                    "iteration": it.iteration,
                    "hypothesis": it.hypothesis.description,
                    "verdict": it.verdict.value,
                    "accepted": it.accepted,
                }
                for it in self.state.iterations
            ],
        }
