"""Tests for adapt/supervisor.py — the orchestration loop."""

from __future__ import annotations

from pathlib import Path

from unittest.mock import MagicMock

from diffusion_agent.adapt.supervisor import AdaptSupervisor
from diffusion_agent.adapt.types import (
    AdaptationTask,
    ExecutionConfig,
    FailureCategory,
    Hypothesis,
    RunResult,
    StopReason,
    TaskStopReason,
    Verdict,
)
from diffusion_agent.adapt.patch_worker import PatchResult
from diffusion_agent.adapt.workspace_sync import NoOpSync, SyncResult


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


class TestSupervisorBasic:
    def test_empty_repo(self, tmp_path: Path) -> None:
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        state = sup.run()
        assert state.stop_reason == StopReason.ALL_RULES_APPLIED
        assert len(state.files_modified) == 0
        assert state.iteration == 0

    def test_single_cuda_call(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        state = sup.run()

        assert len(state.files_modified) >= 1
        assert state.total_rules_applied >= 1

        content = (tmp_path / "model.py").read_text()
        assert ".npu()" in content
        assert "import torch_npu" in content

    def test_multiple_patterns(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.py", 'import torch\nx = tensor.cuda()\ny = model.to("cuda")\n')
        _write(tmp_path, "train.py", 'import torch\ntorch.cuda.set_device(0)\n')

        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        state = sup.run()

        assert len(state.files_modified) >= 2
        assert state.total_rules_applied >= 3  # cuda_call + cuda_to + cuda_api

    def test_flash_attn_handled(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.py", "import torch\nfrom flash_attn import flash_attn_func\n")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        state = sup.run()

        assert len(state.files_modified) >= 1
        content = (tmp_path / "model.py").read_text()
        assert "# [NPU]" in content


class TestSupervisorTaskDecomposition:
    def test_creates_tasks(self, tmp_path: Path) -> None:
        _write(tmp_path, "a.py", "import torch\nx = tensor.cuda()\n")
        _write(tmp_path, "b.py", 'import torch\ny = "nccl"\n')

        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        state = sup.run()

        assert len(state.tasks) >= 2

    def test_tasks_completed_after_batch(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        state = sup.run()

        completed = [t for t in state.tasks if t.status == "completed"]
        assert len(completed) >= 1


class TestSupervisorStopConditions:
    def test_max_iterations(self, tmp_path: Path) -> None:
        # With only deterministic rules, we should complete without hitting max
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False, max_iterations=1)
        state = sup.run()
        assert state.stop_reason is not None

    def test_no_findings_stops_immediately(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.py", "import os\nprint('hello')\n")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        state = sup.run()
        assert state.stop_reason == StopReason.ALL_RULES_APPLIED
        assert state.iteration == 0


class TestSupervisorBlockerReport:
    def test_empty_blocker_report(self, tmp_path: Path) -> None:
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        sup.run()
        report = sup.get_blocker_report()
        assert "blockers" in report
        assert report["blockers"] == []
        assert report["model_name"] == "test"

    def test_blocker_report_structure(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        sup.run()
        report = sup.get_blocker_report()
        assert "total_iterations" in report
        assert "files_modified" in report
        assert "total_rules_applied" in report
        assert "iteration_summary" in report


class TestSupervisorWithGit:
    def test_with_git_repo(self, tmp_path: Path) -> None:
        from git import Repo
        repo = Repo.init(str(tmp_path))
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")
        repo.index.add(["model.py"])
        repo.index.commit("initial")

        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=True)
        state = sup.run()

        assert len(state.files_modified) >= 1

        # Should have created commits
        commits = list(repo.iter_commits())
        assert len(commits) >= 2  # initial + batch

    def test_creates_adaptation_branch(self, tmp_path: Path) -> None:
        from git import Repo
        repo = Repo.init(str(tmp_path))
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")
        repo.index.add(["model.py"])
        repo.index.commit("initial")

        sup = AdaptSupervisor(tmp_path, model_name="my-model", use_git=True)
        sup.run()

        assert repo.active_branch.name == "adapt/my-model"


class TestSupervisorSync:
    def test_noop_sync_by_default(self, tmp_path: Path) -> None:
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        assert isinstance(sup.sync, NoOpSync)

    def test_injected_sync(self, tmp_path: Path) -> None:
        mock_sync = MagicMock()
        mock_sync.sync.return_value = SyncResult(success=True, mode="mock")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False, workspace_sync=mock_sync)
        assert sup.sync is mock_sync

    def test_sync_called_on_batch_changes(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")
        mock_sync = MagicMock()
        mock_sync.sync.return_value = SyncResult(success=True, mode="mock")

        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False, workspace_sync=mock_sync)
        sup.run()

        # sync should have been called at least once (for batch changes)
        assert mock_sync.sync.called
        # First call should include the changed file
        call_args = mock_sync.sync.call_args_list[0]
        changed_files = call_args[0][0]  # first positional arg
        assert any("model.py" in f for f in changed_files)

    def test_sync_not_called_when_no_changes(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.py", "import os\nprint('hello')\n")
        mock_sync = MagicMock()
        mock_sync.sync.return_value = SyncResult(success=True, mode="mock")

        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False, workspace_sync=mock_sync)
        sup.run()

        # No CUDA patterns → no patches → no sync
        mock_sync.sync.assert_not_called()

    def test_sync_from_ssh_execution_config(self, tmp_path: Path) -> None:
        cfg = ExecutionConfig(
            mode="ssh",
            ssh_host="h.example.com",
            remote_workdir="/data/repo",
            sync_enabled=True,
        )
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False, execution_config=cfg)
        # Should have created an RsyncSync, not NoOpSync
        from diffusion_agent.adapt.workspace_sync import RsyncSync
        assert isinstance(sup.sync, RsyncSync)
        assert sup.sync.config.host == "h.example.com"
        assert sup.sync.config.remote_workdir == "/data/repo"

    def test_sync_disabled_in_config(self, tmp_path: Path) -> None:
        cfg = ExecutionConfig(
            mode="ssh",
            ssh_host="h.example.com",
            remote_workdir="/data/repo",
            sync_enabled=False,
        )
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False, execution_config=cfg)
        assert isinstance(sup.sync, NoOpSync)


class TestSupervisorMultiAttempt:
    """Tests for the multi-attempt per-task loop."""

    def test_task_records_attempts(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        state = sup.run()

        # Batch phase handles cuda_call → tasks completed by batch
        for task in state.tasks:
            if task.status == "completed":
                # Tasks completed by batch have no attempts (batch bypass)
                # Tasks completed by iterative loop have attempts
                pass

    def test_task_exhausted_status(self, tmp_path: Path) -> None:
        """A task that runs out of attempts gets status 'exhausted'."""
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        state = sup.run()

        # With deterministic rules, tasks should be completed by batch, not exhausted
        exhausted = [t for t in state.tasks if t.status == "exhausted"]
        # No tasks should be exhausted for simple CUDA patterns
        assert len(exhausted) == 0

    def test_task_stop_reason_set(self, tmp_path: Path) -> None:
        """Completed tasks from the iterative loop have a stop_reason."""
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        state = sup.run()

        # Batch-completed tasks don't go through _process_task
        # so they won't have a stop_reason. This is correct behavior.
        for task in state.tasks:
            if task.status == "completed" and task.attempt_count > 0:
                assert task.stop_reason is not None

    def test_no_hypothesis_marks_task_done(self, tmp_path: Path) -> None:
        """If planner returns None on first attempt, task is marked completed."""
        # File with no CUDA patterns but scanner still finds something
        # In practice, batch handles everything → iterative gets empty tasks
        _write(tmp_path, "model.py", "import os\nprint('hello')\n")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        state = sup.run()
        # No findings → no tasks → stop immediately
        assert state.stop_reason == StopReason.ALL_RULES_APPLIED

    def test_task_to_dict_includes_stop_reason(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        state = sup.run()

        for task in state.tasks:
            d = task.to_dict()
            assert "stop_reason" in d
            assert "attempt_count" in d
            assert "attempts" in d


# =========================================================================
# New tests: Supervisor integration with evaluate_task_progress escalation
# =========================================================================


class TestSupervisorEscalationIntegration:
    """Tests that _process_task calls evaluate_task_progress and escalates
    to BLOCKED when 3 consecutive UNSUPPORTED_OP attempts are detected.

    Key design note: the per-iteration judge already returns BLOCKED for
    obvious UNSUPPORTED_OP stderr.  The escalation catches the case where
    the judge returns a non-BLOCKED verdict (e.g., UNCHANGED because
    before/after sigs match) but the *accumulated* stderrs consistently
    classify as UNSUPPORTED_OP.  To test this, we mock judge.judge to
    return UNCHANGED, simulating imperfect per-iteration detection.
    """

    def test_escalation_blocks_task_after_three_unsupported_op(self, tmp_path: Path) -> None:
        """3 consecutive UNSUPPORTED_OP stderrs → evaluate_task_progress
        should force-stop the task as BLOCKED, even when the per-iteration
        judge returns UNCHANGED."""
        _write(tmp_path, "model.py", "import torch\nx = torch.ops.custom()\n")

        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        sup.state.no_progress_limit = 10  # prevent global REPEATED_NO_PROGRESS

        task = AdaptationTask(
            id="t-escalate",
            name="unsupported_op:model.py",
            description="Test escalation",
            category=FailureCategory.UNSUPPORTED_OP,
            target_files=[str(tmp_path / "model.py")],
            max_attempts=5,  # allow more than 3 to verify escalation kicks in
        )

        # Different UNSUPPORTED_OP errors each attempt (different ops)
        unsupported_stderrs = [
            "RuntimeError: not implemented for 'PrivateUse1' on op aten::scatter_add",
            "RuntimeError: not implemented for 'PrivateUse1' on op aten::index_put_",
            "RuntimeError: not implemented for 'PrivateUse1' on op aten::unique",
            "RuntimeError: not implemented for 'PrivateUse1' on op aten::histc",
            "RuntimeError: not implemented for 'PrivateUse1' on op aten::bincount",
        ]
        hyp_count = [0]
        run_count = [0]

        def fake_generate_hypothesis(t, findings, plan, **kw):
            hyp_count[0] += 1
            return Hypothesis(
                id=f"hyp-escalate-{hyp_count[0]}",
                category=FailureCategory.UNSUPPORTED_OP,
                description=f"LLM fix attempt {hyp_count[0]}",
                target_files=t.target_files,
                proposed_action=f"Try fix {hyp_count[0]}",
                source="llm",
            )

        def fake_apply_patch(hyp, findings):
            return PatchResult(
                hypothesis_id=hyp.id,
                files_changed=[str(tmp_path / "model.py")],
                rules_applied=[],
                description="fake patch",
                success=True,
            )

        def fake_run(*a, **kw):
            idx = run_count[0] % len(unsupported_stderrs)
            run_count[0] += 1
            stderr = unsupported_stderrs[idx]
            return RunResult(
                exit_code=1, stdout="", stderr=stderr,
                error_signature=stderr, duration_s=0.5,
            )

        # Mock judge to return UNCHANGED (simulates noisy stderr where
        # per-iteration blocker detection doesn't fire)
        sup.judge.judge = lambda before, after: Verdict.UNCHANGED
        sup.planner.generate_hypothesis = fake_generate_hypothesis
        sup.worker.apply_patch = fake_apply_patch
        sup.runner.run_syntax_check = fake_run

        sup._process_task(task, [])

        # Should be blocked by escalation, not exhausted at max_attempts=5
        assert task.status == "blocked"
        assert task.stop_reason == TaskStopReason.BLOCKED
        # Should have stopped at 3, not continued to 5
        assert task.attempt_count == 3

    def test_no_escalation_when_errors_change_category(self, tmp_path: Path) -> None:
        """If the error categories alternate, escalation should NOT fire."""
        _write(tmp_path, "model.py", "import torch\nx = torch.ops.custom()\n")

        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        sup.state.no_progress_limit = 10

        task = AdaptationTask(
            id="t-no-esc",
            name="mixed:model.py",
            description="Test no escalation",
            category=FailureCategory.UNSUPPORTED_OP,
            target_files=[str(tmp_path / "model.py")],
            max_attempts=4,
        )

        # Alternating categories: UNSUPPORTED_OP, IMPORT, UNSUPPORTED_OP, IMPORT
        # Each stderr is UNIQUE to avoid REPEATED_ERROR
        mixed_stderrs = [
            "RuntimeError: not implemented for 'PrivateUse1' on op aten::scatter",
            "ModuleNotFoundError: No module named 'some_lib'",
            "RuntimeError: not implemented for 'PrivateUse1' on op aten::unique",
            "ModuleNotFoundError: No module named 'other_lib'",
        ]
        hyp_count = [0]
        run_count = [0]

        def fake_generate_hypothesis(t, findings, plan, **kw):
            hyp_count[0] += 1
            if hyp_count[0] > len(mixed_stderrs):
                return None
            return Hypothesis(
                id=f"hyp-mixed-{hyp_count[0]}",
                category=FailureCategory.UNSUPPORTED_OP,
                description=f"attempt {hyp_count[0]}",
                target_files=t.target_files,
                proposed_action=f"Try fix {hyp_count[0]}",
                source="llm",
            )

        def fake_apply_patch(hyp, findings):
            return PatchResult(
                hypothesis_id=hyp.id,
                files_changed=[str(tmp_path / "model.py")],
                rules_applied=[],
                description="fake patch",
                success=True,
            )

        def fake_run(*a, **kw):
            idx = run_count[0] % len(mixed_stderrs)
            run_count[0] += 1
            stderr = mixed_stderrs[idx]
            return RunResult(
                exit_code=1, stdout="", stderr=stderr,
                error_signature=stderr, duration_s=0.5,
            )

        # Mock judge to return UNCHANGED to prevent per-iteration BLOCKED
        sup.judge.judge = lambda before, after: Verdict.UNCHANGED
        sup.planner.generate_hypothesis = fake_generate_hypothesis
        sup.worker.apply_patch = fake_apply_patch
        sup.runner.run_syntax_check = fake_run

        sup._process_task(task, [])

        # Mixed categories → no escalation → should exhaust at max_attempts
        assert task.status == "exhausted"
        assert task.stop_reason == TaskStopReason.EXHAUSTED
        assert task.attempt_count == 4


class TestSupervisorPhaseC:
    """Tests for Phase C: Runtime Validation Loop."""

    def test_phase_c_skips_when_no_validation_command(self, tmp_path: Path) -> None:
        """Without a validation_command, Phase C is a no-op."""
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        state = sup.run()

        # No validation command → Phase C skipped → normal completion
        assert state.stop_reason == StopReason.ALL_RULES_APPLIED

    def test_phase_c_success_on_first_run(self, tmp_path: Path) -> None:
        """If the validation command passes on first try, stop with INFERENCE_SUCCESS."""
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")

        cfg = ExecutionConfig(validation_command="echo ok")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False, execution_config=cfg)

        # Mock the runner to return success for validation
        sup.runner.run_validation = lambda: RunResult(
            exit_code=0, stdout="ok", stderr="", error_signature="", duration_s=0.1,
        )

        state = sup.run()
        assert state.stop_reason == StopReason.INFERENCE_SUCCESS

    def test_phase_c_iterates_on_runtime_failure(self, tmp_path: Path) -> None:
        """Phase C should iterate when validation fails and planner generates hypotheses."""
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")

        cfg = ExecutionConfig(validation_command="python model.py")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False, execution_config=cfg)

        # Track validation call count — each call returns a DIFFERENT error
        # so the planner doesn't deduplicate them.  The planner checks
        # all previous iteration error_signatures, so each must be unique.
        val_count = [0]
        errors = [
            "ModuleNotFoundError: No module named 'decord'",
            "ModuleNotFoundError: No module named 'soundfile'",
            "ModuleNotFoundError: No module named 'torchaudio'",
        ]

        def fake_validation():
            val_count[0] += 1
            if val_count[0] > len(errors):
                # Succeed after all distinct errors have been seen
                return RunResult(
                    exit_code=0, stdout="ok", stderr="", error_signature="", duration_s=0.1,
                )
            err = errors[val_count[0] - 1]
            return RunResult(
                exit_code=1, stdout="", stderr=err,
                error_signature=err, duration_s=0.5,
            )

        sup.runner.run_validation = fake_validation

        # Mock the patch worker to always "change" a file
        sup.worker.apply_patch = lambda hyp, findings: PatchResult(
            hypothesis_id=hyp.id,
            files_changed=[str(tmp_path / "model.py")],
            rules_applied=[],
            description="lazy import fix",
            success=True,
        )

        # Mock judge to return IMPROVED when errors differ (simulating progress)
        def smart_judge(before, after):
            if after.exit_code == 0:
                return Verdict.FIXED
            if before.error_signature != after.error_signature:
                return Verdict.IMPROVED
            return Verdict.UNCHANGED

        sup.judge.judge = smart_judge

        state = sup.run()
        assert state.stop_reason == StopReason.INFERENCE_SUCCESS
        # Should have Phase C iterations
        phase_c_iters = [
            it for it in state.iterations
            if it.hypothesis.source == "llm" and "runtime" in it.hypothesis.id
        ]
        assert len(phase_c_iters) >= 1

    def test_phase_c_stops_when_no_hypothesis(self, tmp_path: Path) -> None:
        """Phase C stops if planner can't generate a hypothesis."""
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")

        cfg = ExecutionConfig(validation_command="python model.py")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False, execution_config=cfg)

        # Always fail
        sup.runner.run_validation = lambda: RunResult(
            exit_code=1, stdout="",
            stderr="SomeError: unknown",
            error_signature="SomeError: unknown",
            duration_s=0.5,
        )

        # Planner returns None after 1 attempt (same error)
        call_count = [0]
        original_gen = sup.planner.generate_runtime_hypothesis

        def limited_gen(run_result, prev_iters, **kwargs):
            call_count[0] += 1
            if call_count[0] > 1:
                return None
            return original_gen(run_result, prev_iters, **kwargs)

        sup.planner.generate_runtime_hypothesis = limited_gen

        sup.worker.apply_patch = lambda hyp, findings: PatchResult(
            hypothesis_id=hyp.id, files_changed=[], rules_applied=[],
            description="no change", success=True,
        )

        state = sup.run()
        # Should have stopped cleanly (not infinite loop)
        assert state.stop_reason is not None


class TestSupervisorPhaseCTargetFiles:
    """Test Phase C with traceback-based target file resolution."""

    def test_phase_c_passes_repo_path_to_planner(self, tmp_path: Path) -> None:
        """Phase C should pass repo_path and remote_workdir to generate_runtime_hypothesis."""
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")

        cfg = ExecutionConfig(
            validation_command="python model.py",
            mode="ssh",
            ssh_host="test-host",
            remote_workdir="/data/remote",
            sync_enabled=False,
        )
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False, execution_config=cfg)

        # Track what args the planner receives
        captured_kwargs = {}

        def spy_gen(run_result, prev_iters, **kwargs):
            captured_kwargs.update(kwargs)
            return None  # stop immediately

        sup.planner.generate_runtime_hypothesis = spy_gen

        # Mock validation to fail
        sup.runner.run_validation = lambda: RunResult(
            exit_code=1, stdout="",
            stderr="ModuleNotFoundError: No module named 'decord'",
            error_signature="ModuleNotFoundError: No module named 'decord'",
            duration_s=0.5,
        )

        sup.run()

        # Planner should have received repo_path and remote_workdir
        assert captured_kwargs.get("repo_path") == tmp_path
        assert captured_kwargs.get("remote_workdir") == "/data/remote"

    def test_phase_c_runtime_patch_dispatches_correctly(self, tmp_path: Path) -> None:
        """Phase C hypothesis with target_files should trigger apply_runtime_llm_patch."""
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")
        _write(tmp_path, "broken.py", "import decord\n")

        cfg = ExecutionConfig(validation_command="python broken.py")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False, execution_config=cfg)

        val_count = [0]
        target = str(tmp_path / "broken.py")

        def fake_validation():
            val_count[0] += 1
            if val_count[0] > 1:
                return RunResult(exit_code=0, stdout="ok", stderr="", error_signature="", duration_s=0.1)
            return RunResult(
                exit_code=1, stdout="",
                stderr=(
                    "Traceback (most recent call last):\n"
                    f'  File "{target}", line 1, in <module>\n'
                    "    import decord\n"
                    "ModuleNotFoundError: No module named 'decord'\n"
                ),
                error_signature="ModuleNotFoundError: No module named 'decord'",
                duration_s=0.5,
            )

        sup.runner.run_validation = fake_validation

        # Track which patch method is called
        runtime_patch_called = [False]

        def spy_runtime_patch(hypothesis):
            runtime_patch_called[0] = True
            return PatchResult(
                hypothesis_id=hypothesis.id,
                files_changed=[target],
                rules_applied=["runtime_fix"],
                description="lazy import fix",
                success=True,
            )

        sup.worker.apply_runtime_llm_patch = spy_runtime_patch
        sup.judge.judge = lambda before, after: (
            Verdict.FIXED if after.exit_code == 0 else Verdict.IMPROVED
        )

        state = sup.run()

        assert runtime_patch_called[0], "apply_runtime_llm_patch should have been called"
        assert state.stop_reason == StopReason.INFERENCE_SUCCESS


class TestSupervisorRepeatedErrorBugfix:
    """Regression test: the REPEATED_ERROR check must compare with the
    PREVIOUS last_error_signature, not the one just set by record_attempt."""

    def test_different_errors_not_flagged_as_repeated(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.py", "import torch\nx = 1\n")

        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        sup.state.no_progress_limit = 10

        task = AdaptationTask(
            id="t-bug",
            name="test:model.py",
            description="Test repeated error bugfix",
            category=FailureCategory.DEVICE_SELECTION,
            target_files=[str(tmp_path / "model.py")],
            max_attempts=3,
        )

        # 3 DIFFERENT error sigs — should NOT trigger REPEATED_ERROR
        different_stderrs = [
            "RuntimeError: error alpha",
            "RuntimeError: error beta",
            "RuntimeError: error gamma",
        ]
        hyp_count = [0]
        run_count = [0]

        def fake_generate_hypothesis(t, findings, plan, **kw):
            hyp_count[0] += 1
            if hyp_count[0] > len(different_stderrs):
                return None
            return Hypothesis(
                id=f"hyp-bug-{hyp_count[0]}",
                category=FailureCategory.DEVICE_SELECTION,
                description=f"attempt {hyp_count[0]}",
                target_files=t.target_files,
                proposed_action=f"fix {hyp_count[0]}",
                source="llm",
            )

        def fake_apply_patch(hyp, findings):
            return PatchResult(
                hypothesis_id=hyp.id,
                files_changed=[str(tmp_path / "model.py")],
                rules_applied=[],
                description="fake",
                success=True,
            )

        def fake_run(*a, **kw):
            idx = run_count[0] % len(different_stderrs)
            run_count[0] += 1
            stderr = different_stderrs[idx]
            return RunResult(
                exit_code=1, stdout="", stderr=stderr,
                error_signature=stderr, duration_s=0.5,
            )

        sup.judge.judge = lambda before, after: Verdict.UNCHANGED
        sup.planner.generate_hypothesis = fake_generate_hypothesis
        sup.worker.apply_patch = fake_apply_patch
        sup.runner.run_syntax_check = fake_run

        sup._process_task(task, [])

        # All 3 different errors → must reach max_attempts, not early-stop
        assert task.attempt_count == 3
        assert task.status == "exhausted"
        assert task.stop_reason == TaskStopReason.EXHAUSTED


class TestPhaseCDifferentFailureAcceptance:
    """Phase C must accept DIFFERENT_FAILURE as forward progress.

    In a progressive import chain (decord→librosa→soundfile→success),
    each fix resolves one error and reveals the next.  DIFFERENT_FAILURE
    means the error changed — i.e. the fix worked, and a new error appeared.
    Phase C should accept this and continue, not rollback.
    """

    def test_different_failure_accepted_in_phase_c(self, tmp_path: Path) -> None:
        """When Phase C fix produces DIFFERENT_FAILURE, it should be accepted
        and the loop should continue with the new error."""
        _write(tmp_path, "model.py", "import torch\nx = 1\n")

        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        sup.sync = NoOpSync()

        # Progressive error chain: decord → librosa → success
        # Both errors share "ModuleNotFoundError" type prefix, which makes
        # _is_later_error() return False → judge returns DIFFERENT_FAILURE
        validation_calls = [0]
        validation_results = [
            RunResult(exit_code=1, stdout="",
                      stderr="Traceback:\n  File foo.py\nModuleNotFoundError: No module named 'decord'",
                      error_signature="ModuleNotFoundError: No module named 'decord'", duration_s=1),
            RunResult(exit_code=1, stdout="",
                      stderr="Traceback:\n  File foo.py\nModuleNotFoundError: No module named 'librosa'",
                      error_signature="ModuleNotFoundError: No module named 'librosa'", duration_s=1),
            RunResult(exit_code=0, stdout="Success!", stderr="", error_signature="", duration_s=1),
        ]

        def fake_run_validation():
            idx = validation_calls[0]
            validation_calls[0] += 1
            if idx >= len(validation_results):
                return validation_results[-1]
            return validation_results[idx]

        sup.runner.run_validation = fake_run_validation

        hyp_count = [0]

        def fake_runtime_hyp(run_result, prev_iters, **kwargs):
            hyp_count[0] += 1
            return Hypothesis(
                id=f"runtime-{hyp_count[0]}",
                category=FailureCategory.IMPORT_MODULE,
                description=f"Fix runtime failure:\n{run_result.stderr}",
                target_files=[str(tmp_path / "model.py")],
                proposed_action="lazy import fix",
                source="llm",
            )

        sup.planner.generate_runtime_hypothesis = fake_runtime_hyp

        def fake_apply_patch(hyp, findings):
            return PatchResult(
                hypothesis_id=hyp.id,
                files_changed=[str(tmp_path / "model.py")],
                rules_applied=["llm_fix_runtime"],
                description="Runtime LLM fix",
                success=True,
            )

        sup.worker.apply_patch = fake_apply_patch

        sup._runtime_validation_loop()

        # Should reach INFERENCE_SUCCESS through the chain
        assert sup.state.stop_reason == StopReason.INFERENCE_SUCCESS
        # Should have 2 accepted iterations (decord fix, librosa fix)
        accepted_iters = [it for it in sup.state.iterations if it.accepted]
        assert len(accepted_iters) == 2

    def test_different_failure_not_rolled_back(self, tmp_path: Path) -> None:
        """DIFFERENT_FAILURE in Phase C should NOT trigger rollback."""
        _write(tmp_path, "model.py", "import torch\nx = 1\n")

        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        sup.sync = NoOpSync()

        # Phase C: error changes from A to B (same type = DIFFERENT_FAILURE)
        validation_calls = [0]
        results = [
            RunResult(exit_code=1, stdout="",
                      stderr="Traceback:\nModuleNotFoundError: No module named 'decord'",
                      error_signature="ModuleNotFoundError: No module named 'decord'", duration_s=1),
            RunResult(exit_code=1, stdout="",
                      stderr="Traceback:\nModuleNotFoundError: No module named 'librosa'",
                      error_signature="ModuleNotFoundError: No module named 'librosa'", duration_s=1),
            # No more hypothesis generated for librosa
        ]

        def fake_run_validation():
            idx = validation_calls[0]
            validation_calls[0] += 1
            if idx >= len(results):
                return results[-1]
            return results[idx]

        sup.runner.run_validation = fake_run_validation

        hyp_calls = [0]

        def fake_runtime_hyp(run_result, prev_iters, **kwargs):
            hyp_calls[0] += 1
            if hyp_calls[0] > 1:
                return None  # No hypothesis for ErrorB
            return Hypothesis(
                id="runtime-1",
                category=FailureCategory.IMPORT_MODULE,
                description="Fix decord import",
                target_files=[str(tmp_path / "model.py")],
                proposed_action="fix",
                source="llm",
            )

        sup.planner.generate_runtime_hypothesis = fake_runtime_hyp

        def fake_apply_patch(hyp, findings):
            return PatchResult(
                hypothesis_id=hyp.id,
                files_changed=[str(tmp_path / "model.py")],
                rules_applied=["fix"],
                description="fix",
                success=True,
            )

        sup.worker.apply_patch = fake_apply_patch

        sup._runtime_validation_loop()

        # The DIFFERENT_FAILURE should have been accepted
        assert len(sup.state.iterations) >= 1
        assert sup.state.iterations[0].accepted is True
        assert sup.state.iterations[0].verdict == Verdict.DIFFERENT_FAILURE


class TestPhaseCFastFailSyntaxCheck:
    """Phase C should skip expensive remote validation when LLM patch has SyntaxError."""

    def test_syntax_error_skips_remote_validation(self, tmp_path: Path) -> None:
        """When LLM patch introduces a SyntaxError, remote run_validation must NOT
        be called — the local syntax check catches it first."""
        good_file = _write(tmp_path, "model.py", "import torch\nx = 1\n")

        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        sup.sync = NoOpSync()

        # Initial validation fails → enters Phase C
        validation_calls = [0]
        initial_result = RunResult(
            exit_code=1, stdout="",
            stderr="Traceback:\nModuleNotFoundError: No module named 'decord'",
            error_signature="ModuleNotFoundError: No module named 'decord'",
            duration_s=1,
        )

        def fake_run_validation():
            validation_calls[0] += 1
            return initial_result

        sup.runner.run_validation = fake_run_validation

        # Planner returns one hypothesis
        hyp_count = [0]

        def fake_runtime_hyp(run_result, prev_iters, **kwargs):
            hyp_count[0] += 1
            if hyp_count[0] > 1:
                return None
            return Hypothesis(
                id="runtime-1",
                category=FailureCategory.IMPORT_MODULE,
                description=f"Fix runtime failure:\n{run_result.stderr}",
                target_files=[str(good_file)],
                proposed_action="lazy import fix",
                source="llm",
            )

        sup.planner.generate_runtime_hypothesis = fake_runtime_hyp

        # Patch worker introduces a SyntaxError into the file
        def fake_apply_patch(hyp, findings):
            good_file.write_text('raise RuntimeError("unterminated\n')
            return PatchResult(
                hypothesis_id=hyp.id,
                files_changed=[str(good_file)],
                rules_applied=["llm_fix_runtime"],
                description="Runtime LLM fix",
                success=True,
            )

        sup.worker.apply_patch = fake_apply_patch

        sup._runtime_validation_loop()

        # run_validation called once for initial check, but NOT again
        # after the broken patch (syntax check caught it first)
        assert validation_calls[0] == 1, (
            f"Expected 1 validation call (initial only), got {validation_calls[0]}. "
            "Syntax check should have prevented re-validation."
        )

        # The iteration should be recorded as rejected (syntax error)
        assert len(sup.state.iterations) >= 1
        last_iter = sup.state.iterations[-1]
        assert last_iter.accepted is False

    def test_syntax_ok_proceeds_to_validation(self, tmp_path: Path) -> None:
        """When LLM patch is syntactically valid, normal remote validation proceeds."""
        good_file = _write(tmp_path, "model.py", "import torch\nx = 1\n")

        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        sup.sync = NoOpSync()

        validation_calls = [0]
        validation_results = [
            RunResult(exit_code=1, stdout="",
                      stderr="ModuleNotFoundError: No module named 'decord'",
                      error_signature="ModuleNotFoundError: No module named 'decord'",
                      duration_s=1),
            RunResult(exit_code=0, stdout="Success!", stderr="",
                      error_signature="", duration_s=1),
        ]

        def fake_run_validation():
            idx = validation_calls[0]
            validation_calls[0] += 1
            if idx >= len(validation_results):
                return validation_results[-1]
            return validation_results[idx]

        sup.runner.run_validation = fake_run_validation

        hyp_count = [0]

        def fake_runtime_hyp(run_result, prev_iters, **kwargs):
            hyp_count[0] += 1
            if hyp_count[0] > 1:
                return None
            return Hypothesis(
                id="runtime-1",
                category=FailureCategory.IMPORT_MODULE,
                description="Fix decord import",
                target_files=[str(good_file)],
                proposed_action="lazy import fix",
                source="llm",
            )

        sup.planner.generate_runtime_hypothesis = fake_runtime_hyp

        def fake_apply_patch(hyp, findings):
            # Valid Python — should pass syntax check
            good_file.write_text("import torch\nx = 2  # patched\n")
            return PatchResult(
                hypothesis_id=hyp.id,
                files_changed=[str(good_file)],
                rules_applied=["llm_fix_runtime"],
                description="Runtime LLM fix",
                success=True,
            )

        sup.worker.apply_patch = fake_apply_patch

        sup._runtime_validation_loop()

        # Should have 2 validation calls: initial + after valid patch
        assert validation_calls[0] == 2
        assert sup.state.stop_reason == StopReason.INFERENCE_SUCCESS


class TestSupervisorPhaseARescan:
    """Tests for Phase A ↔ Phase C rescan loop."""

    def test_phase_a_rescan_finds_new_rules_after_phase_c(self, tmp_path: Path) -> None:
        """After Phase C, rescan should find new patterns and apply rules."""
        # File with flash_attn import AND assert guard
        code = (
            "import torch\n"
            "import flash_attn\n"
            "assert FLASH_ATTN_2_AVAILABLE\n"
            "x = 1\n"
        )
        _write(tmp_path, "model.py", code)

        cfg = ExecutionConfig(validation_command="echo ok")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False, execution_config=cfg)

        # Mock validation to succeed (so Phase C doesn't loop)
        sup.runner.run_validation = lambda: RunResult(
            exit_code=0, stdout="ok", stderr="", error_signature="", duration_s=0.1,
        )

        sup.run()

        # The flash_attn import should be commented out by FlashAttnRule
        content = (tmp_path / "model.py").read_text()
        assert "# [NPU]" in content
        # The assert guard should be handled by FlashAttnUsageRule
        assert "assert FLASH_ATTN_2_AVAILABLE" not in content

    def test_phase_a_rescan_no_new_rules_skips_loop(self, tmp_path: Path) -> None:
        """If rescan finds no new rules, the loop exits immediately."""
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")

        cfg = ExecutionConfig(validation_command="echo ok")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False, execution_config=cfg)

        sup.runner.run_validation = lambda: RunResult(
            exit_code=0, stdout="ok", stderr="", error_signature="", duration_s=0.1,
        )

        state = sup.run()
        assert state.stop_reason == StopReason.INFERENCE_SUCCESS

    def test_rescan_max_cycles_respected(self, tmp_path: Path) -> None:
        """Rescan loop must not exceed MAX_RESCAN (2) cycles."""
        _write(tmp_path, "model.py", "import torch\nx = 1\n")

        cfg = ExecutionConfig(validation_command="python model.py")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False, execution_config=cfg)

        # Always fail validation
        sup.runner.run_validation = lambda: RunResult(
            exit_code=1, stdout="",
            stderr="SomeError",
            error_signature="SomeError",
            duration_s=0.1,
        )

        # Make _phase_a_rescan always claim it applied new rules
        rescan_calls = [0]

        def counting_rescan():
            rescan_calls[0] += 1
            # Return 1 to simulate "new rules applied" every time
            return 1

        sup._phase_a_rescan = counting_rescan

        sup.run()

        # Should have called rescan at most 2 times (MAX_RESCAN)
        assert rescan_calls[0] <= 2

    def test_rescan_triggers_sync_before_phase_c(self, tmp_path: Path) -> None:
        """When rescan applies new rules, sync should be called before Phase C re-enters."""
        code = (
            "import torch\n"
            "import flash_attn\n"
            "assert FLASH_ATTN_2_AVAILABLE\n"
        )
        _write(tmp_path, "model.py", code)

        mock_sync = MagicMock()
        mock_sync.sync.return_value = SyncResult(success=True, mode="mock")

        cfg = ExecutionConfig(validation_command="echo ok")
        sup = AdaptSupervisor(
            tmp_path, model_name="test", use_git=False,
            execution_config=cfg, workspace_sync=mock_sync,
        )

        sup.runner.run_validation = lambda: RunResult(
            exit_code=0, stdout="ok", stderr="", error_signature="", duration_s=0.1,
        )

        sup.run()

        # Sync should have been called (at least for Phase A batch)
        assert mock_sync.sync.called


class TestPhaseCDecoupledBudget:
    """Phase C must use its own iteration budget, NOT share with Phase B."""

    def test_phase_c_runs_even_when_phase_b_exhausted_iterations(self, tmp_path: Path) -> None:
        """Phase B consuming all max_iterations must NOT prevent Phase C from running."""
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")

        cfg = ExecutionConfig(validation_command="python model.py")
        sup = AdaptSupervisor(
            tmp_path, model_name="test", max_iterations=5, use_git=False,
            execution_config=cfg,
        )

        # Simulate Phase B consuming all iterations
        sup.state.iteration = 5

        # Phase C should still run — track validation calls
        val_calls = [0]

        def fake_validation():
            val_calls[0] += 1
            if val_calls[0] > 1:
                return RunResult(exit_code=0, stdout="ok", stderr="", error_signature="", duration_s=0.1)
            return RunResult(
                exit_code=1, stdout="",
                stderr="AttributeError: 'NoneType' object has no attribute 'flash_attn_varlen_func'",
                error_signature="AttributeError: 'NoneType' object has no attribute 'flash_attn_varlen_func'",
                duration_s=1.0,
            )

        sup.runner.run_validation = fake_validation

        # Provide a runtime hypothesis for the error
        def fake_gen(run_result, prev_iters, **kwargs):
            if val_calls[0] <= 1:
                return Hypothesis(
                    id="runtime-fix-1", category=FailureCategory.CUSTOM_EXTENSION,
                    description="Fix flash_attn call", target_files=[str(tmp_path / "model.py")],
                    proposed_action="Replace with SDPA", source="llm",
                )
            return None

        sup.planner.generate_runtime_hypothesis = fake_gen
        sup.worker.apply_patch = lambda h, f: PatchResult(
            hypothesis_id="runtime-fix-1", description="fix", success=True,
            files_changed=[str(tmp_path / "model.py")],
            rules_applied=[], migration_results=[],
        )

        sup._runtime_validation_loop()

        # Phase C must have run (validation called at least twice)
        assert val_calls[0] >= 2

    def test_phase_c_budget_independent_of_phase_b(self, tmp_path: Path) -> None:
        """max_phase_c_iterations should be configurable and independent."""
        _write(tmp_path, "model.py", "import torch\nx = 1\n")

        cfg = ExecutionConfig(validation_command="echo ok")
        sup = AdaptSupervisor(
            tmp_path, model_name="test",
            max_iterations=3,  # Phase B budget (small)
            max_phase_c_iterations=10,  # Phase C budget (large)
            use_git=False,
            execution_config=cfg,
        )

        assert sup.state.max_phase_c_iterations == 10

    def test_phase_b_iteration_counter_not_shared_with_phase_c(self, tmp_path: Path) -> None:
        """Phase B and Phase C iteration counts are tracked separately."""
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")

        cfg = ExecutionConfig(validation_command="echo ok")
        sup = AdaptSupervisor(
            tmp_path, model_name="test", max_iterations=2, use_git=False,
            execution_config=cfg,
        )

        # Mock validation to succeed immediately
        sup.runner.run_validation = lambda: RunResult(
            exit_code=0, stdout="ok", stderr="", error_signature="", duration_s=0.1,
        )

        state = sup.run()

        # Phase B may have consumed iterations, but Phase C should still succeed
        assert state.stop_reason == StopReason.INFERENCE_SUCCESS


class TestPhaseCRetriesAfterSyntaxError:
    """Phase C should retry LLM fixes when previous attempt had a SyntaxError.

    A SyntaxError means the LLM patch was malformed — we rolled back and
    never validated it remotely.  The error_signature hasn't changed (because
    the code is back to the pre-patch state), but we should NOT treat this as
    'repeated input' — the LLM can generate a different fix on retry.
    """

    def test_phase_c_retries_after_syntax_error(self, tmp_path: Path) -> None:
        """After a SyntaxError rollback, Phase C should retry with the same
        error_signature instead of breaking with 'repeated_input'."""
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")

        cfg = ExecutionConfig(validation_command="bash test.sh")
        sup = AdaptSupervisor(
            tmp_path, model_name="test",
            max_iterations=5,
            max_phase_c_iterations=5,
            use_git=False,
            execution_config=cfg,
        )

        # Validation always fails with same error (because rollback resets code)
        runtime_error = RunResult(
            exit_code=1,
            stdout="",
            stderr="RuntimeError: shape mismatch",
            error_signature="RuntimeError: shape mismatch",
            duration_s=1.0,
        )
        val_call_count = [0]

        def mock_validation():
            val_call_count[0] += 1
            if val_call_count[0] >= 4:
                # Third real validation succeeds
                return RunResult(
                    exit_code=0, stdout="ok", stderr="",
                    error_signature="", duration_s=0.1,
                )
            return runtime_error

        sup.runner.run_validation = mock_validation

        # Planner generates hypotheses with target_files
        hyp_count = [0]
        def mock_gen_hyp(result, seen, **kwargs):
            hyp_count[0] += 1
            return Hypothesis(
                id=f"hyp-runtime-{hyp_count[0]}",
                category=FailureCategory.LOGIC_BUG,
                description="Fix shape mismatch",
                target_files=[str(tmp_path / "model.py")],
                proposed_action="Replace attention",
                source="llm",
                deepest_file=str(tmp_path / "model.py"),
            )

        sup.planner.generate_runtime_hypothesis = mock_gen_hyp

        # Worker: first attempt → SyntaxError file, second → valid patch
        patch_call = [0]

        def mock_apply_patch(hyp, findings):
            patch_call[0] += 1
            target = str(tmp_path / "model.py")
            if patch_call[0] == 1:
                # Write syntactically invalid code
                (tmp_path / "model.py").write_text("def broken(\n")
                return PatchResult(
                    hypothesis_id=hyp.id, files_changed=[target],
                    rules_applied=["llm_fix"], description="bad patch",
                    success=True,
                )
            else:
                # Write valid code
                (tmp_path / "model.py").write_text("import torch\nx = 1\n")
                return PatchResult(
                    hypothesis_id=hyp.id, files_changed=[target],
                    rules_applied=["llm_fix"], description="good patch",
                    success=True,
                )

        sup.worker.apply_patch = mock_apply_patch

        sup.run()

        # The key assertion: we should have gotten past the SyntaxError
        # and tried again (at least 2 hypotheses generated)
        assert hyp_count[0] >= 2, (
            f"Expected >=2 hypothesis attempts after SyntaxError retry, got {hyp_count[0]}"
        )

    def test_phase_c_syntax_error_does_not_consume_error_sig(self, tmp_path: Path) -> None:
        """A SyntaxError should not mark the error_signature as 'attempted'."""
        _write(tmp_path, "model.py", "import torch\nx = 1\n")

        cfg = ExecutionConfig(validation_command="bash test.sh")
        sup = AdaptSupervisor(
            tmp_path, model_name="test",
            max_iterations=5,
            max_phase_c_iterations=3,
            use_git=False,
            execution_config=cfg,
        )

        # Always fail with same error
        runtime_error = RunResult(
            exit_code=1, stdout="", stderr="Error: bad op",
            error_signature="Error: bad op", duration_s=1.0,
        )
        sup.runner.run_validation = lambda: runtime_error

        hyp_count = [0]

        def mock_gen_hyp(result, seen, **kwargs):
            hyp_count[0] += 1
            return Hypothesis(
                id=f"hyp-rt-{hyp_count[0]}",
                category=FailureCategory.LOGIC_BUG,
                description="Fix op",
                target_files=[str(tmp_path / "model.py")],
                proposed_action="Replace op",
                source="llm",
                deepest_file=str(tmp_path / "model.py"),
            )

        sup.planner.generate_runtime_hypothesis = mock_gen_hyp

        # All patches produce SyntaxError
        def mock_apply(hyp, findings):
            target = str(tmp_path / "model.py")
            (tmp_path / "model.py").write_text("def broken(\n")
            return PatchResult(
                hypothesis_id=hyp.id, files_changed=[target],
                rules_applied=["llm"], description="broken patch",
                success=True,
            )

        sup.worker.apply_patch = mock_apply

        sup.run()

        # With 3 max_phase_c_iterations, we should get 3 hypothesis attempts
        # (all SyntaxError), NOT 1 attempt + immediate break on 'repeated_input'
        assert hyp_count[0] == 3, (
            f"Expected 3 hypothesis attempts (one per Phase C iteration), got {hyp_count[0]}"
        )
