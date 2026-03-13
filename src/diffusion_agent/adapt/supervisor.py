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
    TaskStopReason,
    Verdict,
)
from diffusion_agent.adapt.workspace_sync import (
    NoOpSync,
    SyncResult,
    WorkspaceSync,
    create_workspace_sync,
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
        workspace_sync: WorkspaceSync | None = None,
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

        # Workspace sync: explicit injection > config-derived > noop
        if workspace_sync is not None:
            self.sync: WorkspaceSync = workspace_sync
        elif execution_config and execution_config.mode == "ssh" and execution_config.sync_enabled:
            self.sync = create_workspace_sync(
                mode=execution_config.mode,
                ssh_host=execution_config.ssh_host,
                ssh_user=execution_config.ssh_user,
                ssh_port=execution_config.ssh_port,
                remote_workdir=execution_config.remote_workdir,
                exclude_patterns=execution_config.sync_exclude,
                sync_timeout=execution_config.sync_timeout,
                delete=execution_config.sync_delete,
                prefer_scp=execution_config.sync_prefer_scp,
            )
        else:
            self.sync = NoOpSync()

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

    # ----- Workspace sync -----

    def _sync_to_remote(self, changed_files: list[str]) -> SyncResult:
        """Sync changed files to the remote workspace before validation.

        Returns a SyncResult.  On failure, logs a warning but does not
        halt the loop — the caller decides how to handle it.
        """
        result = self.sync.sync(changed_files, self.state.repo_path)
        if result.success:
            log.info(
                "supervisor_sync_ok",
                mode=result.mode,
                files=result.files_requested,
            )
        else:
            log.warning(
                "supervisor_sync_failed",
                mode=result.mode,
                error=result.error,
            )
        return result

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

            # Sync batch-patched files to remote before iterative validation
            self._sync_to_remote(patch_result.files_changed)

            log.info(
                "supervisor_batch_done",
                files=len(patch_result.files_changed),
                rules=len(patch_result.rules_applied),
            )

    # ----- Phase B: Iterative loop -----

    def _iterative_loop(self, original_findings: list[Finding]) -> None:
        """Run the iterative adapt-run-judge loop for remaining tasks.

        Each task gets up to ``task.max_attempts`` hypothesis cycles.
        The outer loop iterates over tasks; the inner loop iterates
        attempts within a single task.
        """
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

            # Reset per-task no-progress counter so one hard task
            # doesn't poison the remaining tasks.
            self.state.consecutive_no_progress = 0
            self._process_task(task, findings)

    def _process_task(self, task: AdaptationTask, findings: list[Finding]) -> None:
        """Process a single adaptation task through multiple hypothesis attempts.

        Runs up to ``task.max_attempts`` iterations.  Each attempt follows:
        hypothesis → snapshot → baseline → patch → sync → validate → judge →
        accept/rollback → record.  The task transitions to a terminal status
        (completed / blocked / exhausted) when a stop condition fires.
        """
        task.status = "in_progress"
        log.info("supervisor_task_start", task=task.name, max_attempts=task.max_attempts)

        while task.attempt_count < task.max_attempts:
            # Global stop conditions
            if self.state.stop_reason is not None:
                break
            if self.state.iteration >= self.state.max_iterations:
                self.state.stop_reason = StopReason.MAX_ITERATIONS
                break

            self.state.iteration += 1

            # 1. Generate hypothesis (planner checks task.seen_hypothesis_ids)
            plan = self.registry.match_all(findings)
            hypothesis = self.planner.generate_hypothesis(task, findings, plan)

            if hypothesis is None:
                task.status = "exhausted" if task.attempt_count > 0 else "completed"
                task.stop_reason = (
                    TaskStopReason.NO_HYPOTHESIS
                    if task.attempt_count > 0
                    else TaskStopReason.FIXED
                )
                log.info(
                    "supervisor_task_no_hypothesis",
                    task=task.name,
                    attempts=task.attempt_count,
                )
                return

            # 2. Snapshot git state
            snapshot = self.git.snapshot() if self.git else ""

            # 3. Baseline validation
            run_before = self.runner.run_syntax_check(
                [f for f in task.target_files if Path(f).exists()]
            )

            # 4. Apply patch
            patch_result = self.worker.apply_patch(hypothesis, findings)

            # 4b. Sync patched files to remote
            if patch_result.files_changed:
                self._sync_to_remote(patch_result.files_changed)

            # 5. Post-patch validation
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
                if self.git and self.git.has_changes():
                    commit_sha = self.git.commit_iteration(
                        self.state.iteration, hypothesis, verdict,
                    )
            else:
                if self.git and snapshot:
                    self.git.rollback_to(snapshot)
                if patch_result.files_changed:
                    self._sync_to_remote(patch_result.files_changed)

            # 8. Record attempt on the task
            task.record_attempt(
                hypothesis=hypothesis,
                verdict=verdict,
                accepted=accepted,
                error_signature=run_after.error_signature,
                files_changed=patch_result.files_changed,
            )

            # 9. Record iteration on the session
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

            # 10. Update global no-progress counter
            if verdict in {Verdict.UNCHANGED, Verdict.REGRESSED, Verdict.DIFFERENT_FAILURE}:
                self.state.consecutive_no_progress += 1
            else:
                self.state.consecutive_no_progress = 0

            if self.state.consecutive_no_progress >= self.state.no_progress_limit:
                self.state.stop_reason = StopReason.REPEATED_NO_PROGRESS

            log.info(
                "supervisor_attempt_done",
                iteration=self.state.iteration,
                task=task.name,
                attempt=task.attempt_count,
                verdict=verdict.value,
                accepted=accepted,
            )

            # 11. Per-task stop decisions
            if verdict == Verdict.BLOCKED:
                task.status = "blocked"
                task.stop_reason = TaskStopReason.BLOCKED
                task.blocker_reason = run_after.error_signature
                self.state.blockers.append(
                    f"{task.name}: {hypothesis.description}"
                )
                return

            if accepted and verdict in {Verdict.FIXED, Verdict.IMPROVED}:
                task.status = "completed"
                task.stop_reason = TaskStopReason.FIXED
                return

            # Repeated error across attempts → stop early
            if (
                run_after.error_signature
                and run_after.error_signature == task.last_error_signature
                and task.attempt_count >= 2
            ):
                task.status = "exhausted"
                task.stop_reason = TaskStopReason.REPEATED_ERROR
                log.info(
                    "supervisor_task_repeated_error",
                    task=task.name,
                    sig=run_after.error_signature[:80],
                )
                return

        # If we exit the while loop without returning, attempts exhausted
        if task.status == "in_progress":
            task.status = "exhausted"
            task.stop_reason = TaskStopReason.EXHAUSTED
            log.info("supervisor_task_exhausted", task=task.name, attempts=task.attempt_count)

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
