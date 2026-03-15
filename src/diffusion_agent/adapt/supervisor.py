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
from diffusion_agent.adapt.judge import AdaptJudge, evaluate_task_progress
from diffusion_agent.adapt.patch_worker import PatchWorker
from diffusion_agent.adapt.planner import AdaptPlanner
from diffusion_agent.adapt.runner import AdaptRunner, validate_syntax_local
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
        max_phase_c_iterations: int = 10,
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
        self._exec_config = execution_config

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
            max_phase_c_iterations=max_phase_c_iterations,
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

        # 6. Phase C: Runtime validation loop
        #    Run the actual validation command and fix runtime errors.
        #    Phase C has its own iteration budget (max_phase_c_iterations),
        #    so clear any MAX_ITERATIONS stop from Phase B.
        if self.state.stop_reason == StopReason.MAX_ITERATIONS:
            self.state.stop_reason = None
        self._runtime_validation_loop()

        # 7. Phase A ↔ Phase C rescan loop
        #    After Phase C completes without full success, rescan for new
        #    rule matches (e.g. FlashAttnUsageRule matching assert guards
        #    discovered during Phase C).  Max 2 rescan cycles.
        #    Only runs when Phase C was actually attempted (validation command exists).
        has_validation = self._exec_config and self._exec_config.validation_command
        if has_validation and self.state.stop_reason != StopReason.INFERENCE_SUCCESS:
            rescan_cycles = 0
            MAX_RESCAN = 2
            while rescan_cycles < MAX_RESCAN:
                new_rules = self._phase_a_rescan()
                if new_rules == 0:
                    break
                rescan_cycles += 1
                # Sync newly patched files and re-enter Phase C
                self._runtime_validation_loop()
                if self.state.stop_reason == StopReason.INFERENCE_SUCCESS:
                    break

        # 8. Determine final stop reason if not set
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

    def _apply_batch_rules(self, findings: list[Finding]) -> int:
        """Apply all deterministic migration rules in one batch.

        Returns the number of rules applied.
        """
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
            return len(patch_result.rules_applied)
        return 0

    # ----- Phase A rescan -----

    def _phase_a_rescan(self) -> int:
        """Re-scan and apply rules that match newly-discovered patterns.

        Returns the number of new rules applied.
        """
        findings = scan_directory(self.state.repo_path)
        if not findings:
            return 0
        new_applied = self._apply_batch_rules(findings)
        if new_applied > 0:
            log.info("supervisor_rescan_applied", new_rules=new_applied)
        return new_applied

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
        accept/rollback → record → escalation check.  The task transitions to
        a terminal status (completed / blocked / exhausted) when a stop
        condition fires.
        """
        task.status = "in_progress"
        log.info("supervisor_task_start", task=task.name, max_attempts=task.max_attempts)

        # Collect stderrs for task-level escalation analysis
        attempt_stderrs: list[str] = []

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

            # 8. Save previous error sig BEFORE recording (fixes comparison bug)
            prev_error_sig = task.last_error_signature

            # 9. Record attempt on the task
            task.record_attempt(
                hypothesis=hypothesis,
                verdict=verdict,
                accepted=accepted,
                error_signature=run_after.error_signature,
                files_changed=patch_result.files_changed,
            )
            attempt_stderrs.append(run_after.stderr)

            # 10. Record iteration on the session
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

            # 11. Update global no-progress counter
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

            # 12. Per-task stop decisions
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

            # 13. Repeated error across attempts → stop early
            #     Compare with prev_error_sig (from BEFORE record_attempt)
            if (
                run_after.error_signature
                and run_after.error_signature == prev_error_sig
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

            # 14. Task-level escalation: consecutive same-category blocker
            escalation = evaluate_task_progress(task, attempt_stderrs)
            if escalation == TaskStopReason.BLOCKED:
                task.status = "blocked"
                task.stop_reason = TaskStopReason.BLOCKED
                task.blocker_reason = (
                    "Escalation: consecutive attempts classified as "
                    "same blocker category"
                )
                self.state.blockers.append(
                    f"{task.name}: escalation — repeated blocker category"
                )
                log.info(
                    "supervisor_task_escalation_blocked",
                    task=task.name,
                    attempts=task.attempt_count,
                )
                return

        # If we exit the while loop without returning, attempts exhausted
        if task.status == "in_progress":
            task.status = "exhausted"
            task.stop_reason = TaskStopReason.EXHAUSTED
            log.info("supervisor_task_exhausted", task=task.name, attempts=task.attempt_count)

    # ----- Phase C: Runtime validation loop -----

    def _runtime_validation_loop(self) -> None:
        """Run the actual validation command and iterate on runtime failures.

        Phase A/B handle static patterns (syntax, imports).  Phase C runs the
        real workload (e.g. ``bash run_npu_test.sh``) and feeds runtime errors
        back through the planner → patch_worker → runner → judge cycle.

        Each iteration:
          1. Run validation command via Runner
          2. If success → done (INFERENCE_SUCCESS)
          3. If failure → generate runtime hypothesis via Planner
          4. Apply LLM patch via PatchWorker
          5. Sync, validate, judge, accept/rollback
        """
        validation_result = self.runner.run_validation()
        if validation_result is None:
            log.info("supervisor_phase_c_skip", reason="no validation command configured")
            return

        if validation_result.exit_code == 0:
            self.state.stop_reason = StopReason.INFERENCE_SUCCESS
            log.info("supervisor_phase_c_success", msg="validation passed on first run")
            return

        log.info(
            "supervisor_phase_c_start",
            exit_code=validation_result.exit_code,
            error_sig=validation_result.error_signature[:120] if validation_result.error_signature else "",
        )

        # Phase C has its own iteration budget, independent of Phase B.
        # This prevents Phase A/B consuming all iterations and starving Phase C.
        max_runtime_attempts = self.state.max_phase_c_iterations

        # Track which error signatures have been used as INPUT to a
        # hypothesis.  This avoids the false-positive dedup problem where
        # a progressive chain of errors (A→B→C→success) gets blocked
        # because B appears in both run_after(iter1) and as input(iter2).
        attempted_error_sigs: set[str] = set()

        for attempt in range(max_runtime_attempts):
            if self.state.stop_reason is not None:
                break

            self.state.iteration += 1

            # 1. Generate hypothesis from the runtime error.
            #    Only reject if we already TRIED to fix this exact error.
            if validation_result.error_signature in attempted_error_sigs:
                log.info(
                    "supervisor_phase_c_repeated_input",
                    attempt=attempt + 1,
                    error_sig=validation_result.error_signature[:80],
                )
                break

            attempted_error_sigs.add(validation_result.error_signature)

            remote_workdir = (
                self._exec_config.remote_workdir
                if self._exec_config else None
            )
            hypothesis = self.planner.generate_runtime_hypothesis(
                validation_result,
                [],  # empty: we handle dedup above
                repo_path=self.state.repo_path,
                remote_workdir=remote_workdir,
            )

            if hypothesis is None:
                log.info(
                    "supervisor_phase_c_no_hypothesis",
                    attempt=attempt + 1,
                    error_sig=validation_result.error_signature[:80],
                )
                break

            # 2. Snapshot git state
            snapshot = self.git.snapshot() if self.git else ""

            # 3. Apply LLM patch
            #    For runtime errors, we pass an empty findings list since these
            #    aren't from static scanning — the hypothesis itself contains
            #    the error context.
            patch_result = self.worker.apply_patch(hypothesis, [])

            if not patch_result.files_changed:
                log.info("supervisor_phase_c_no_patch", hypothesis=hypothesis.id)
                # Record as unchanged attempt
                record = IterationRecord(
                    iteration=self.state.iteration,
                    hypothesis=hypothesis,
                    patch_description="no files changed",
                    files_changed=[],
                    run_before=validation_result,
                    run_after=validation_result,
                    verdict=Verdict.UNCHANGED,
                    accepted=False,
                )
                self.state.iterations.append(record)
                continue

            # 3b. Fast-fail: local syntax check before expensive remote
            #     validation.  Catches broken LLM patches (e.g. unterminated
            #     strings) in milliseconds instead of after a ~20s SSH cycle.
            syntax_result = validate_syntax_local(patch_result.files_changed)
            if syntax_result.exit_code != 0:
                log.warning(
                    "supervisor_phase_c_syntax_fail",
                    files=patch_result.files_changed,
                    error=syntax_result.error_signature[:120],
                )
                # Rollback the broken patch
                if self.git and snapshot:
                    self.git.rollback_to(snapshot)

                # Un-mark this error_sig so the LLM gets another chance.
                # A SyntaxError means the *patch* was bad, not that we've
                # exhausted ideas for this error.  After rollback the code
                # is back to the pre-patch state, so the error_sig will be
                # identical on the next loop — but that's a valid retry.
                attempted_error_sigs.discard(
                    validation_result.error_signature,
                )

                # Record as rejected iteration — no remote validation needed
                record = IterationRecord(
                    iteration=self.state.iteration,
                    hypothesis=hypothesis,
                    patch_description=f"SYNTAX_ERROR: {syntax_result.error_signature[:100]}",
                    files_changed=patch_result.files_changed,
                    run_before=validation_result,
                    run_after=syntax_result,
                    verdict=Verdict.REGRESSED,
                    accepted=False,
                )
                self.state.iterations.append(record)
                continue

            # 4. Sync patched files
            self._sync_to_remote(patch_result.files_changed)

            # 5. Re-run validation
            new_result = self.runner.run_validation()
            if new_result is None:
                break

            # 6. Judge
            verdict = self.judge.judge(validation_result, new_result)

            # 7. Accept or rollback
            #    In Phase C, DIFFERENT_FAILURE means the error CHANGED — the fix
            #    resolved the previous error but exposed a new one.  This is
            #    forward progress in a progressive error chain (e.g.
            #    decord→librosa→soundfile→success) and should be accepted.
            accepted = self.judge.should_accept(verdict) or (
                verdict == Verdict.DIFFERENT_FAILURE
            )
            commit_sha = None

            if accepted:
                self.state.files_modified.update(patch_result.files_changed)
                if self.git and self.git.has_changes():
                    commit_sha = self.git.commit_iteration(
                        self.state.iteration, hypothesis, verdict,
                    )
                # Update baseline for next iteration
                validation_result = new_result
            else:
                if self.git and snapshot:
                    self.git.rollback_to(snapshot)
                if patch_result.files_changed:
                    self._sync_to_remote(patch_result.files_changed)

            # 8. Record iteration
            record = IterationRecord(
                iteration=self.state.iteration,
                hypothesis=hypothesis,
                patch_description=patch_result.description,
                files_changed=patch_result.files_changed,
                run_before=validation_result if not accepted else None,
                run_after=new_result,
                verdict=verdict,
                accepted=accepted,
                commit_sha=commit_sha,
            )
            self.state.iterations.append(record)

            log.info(
                "supervisor_phase_c_attempt",
                iteration=self.state.iteration,
                attempt=attempt + 1,
                verdict=verdict.value,
                accepted=accepted,
                error_sig=new_result.error_signature[:80] if new_result.error_signature else "none",
            )

            # 9. Check for success
            if new_result.exit_code == 0:
                self.state.stop_reason = StopReason.INFERENCE_SUCCESS
                log.info("supervisor_phase_c_success", iterations=attempt + 1)
                return

            # 10. Check for blocked verdict
            if verdict == Verdict.BLOCKED:
                self.state.blockers.append(
                    f"runtime: {hypothesis.description[:100]}"
                )
                break

        # If we exit without success, the final stop_reason will be set
        # by the caller based on remaining state.

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
