"""Tests for adapt/types.py — shared data types for Phase 2."""

from __future__ import annotations

from pathlib import Path

from diffusion_agent.adapt.types import (
    AdaptationState,
    AdaptationTask,
    FailureCategory,
    Hypothesis,
    IterationRecord,
    RunResult,
    StopReason,
    Verdict,
)


class TestFailureCategory:
    def test_all_categories_are_strings(self) -> None:
        for cat in FailureCategory:
            assert isinstance(cat.value, str)

    def test_has_required_categories(self) -> None:
        names = {c.value for c in FailureCategory}
        assert "device_selection" in names
        assert "cuda_only_api" in names
        assert "unsupported_op" in names
        assert "unknown_blocker" in names


class TestHypothesis:
    def test_to_dict(self) -> None:
        h = Hypothesis(
            id="h-1",
            category=FailureCategory.DEVICE_SELECTION,
            description="Fix .cuda() calls",
            target_files=["model.py"],
            proposed_action="Apply cuda_call rule",
        )
        d = h.to_dict()
        assert d["id"] == "h-1"
        assert d["category"] == "device_selection"
        assert d["target_files"] == ["model.py"]

    def test_default_source(self) -> None:
        h = Hypothesis(
            id="h-1", category=FailureCategory.DEVICE_SELECTION,
            description="test", target_files=[], proposed_action="test",
        )
        assert h.source == "planner"


class TestRunResult:
    def test_to_dict_truncates_output(self) -> None:
        r = RunResult(
            exit_code=1,
            stdout="x" * 1000,
            stderr="y" * 1000,
            error_signature="RuntimeError: test",
            duration_s=1.5,
            command="python test.py",
        )
        d = r.to_dict()
        assert len(d["stdout_tail"]) == 500
        assert len(d["stderr_tail"]) == 500
        assert d["exit_code"] == 1

    def test_to_dict_empty(self) -> None:
        r = RunResult(exit_code=0, stdout="", stderr="", error_signature="", duration_s=0.0)
        d = r.to_dict()
        assert d["stdout_tail"] == ""
        assert d["error_signature"] == ""


class TestVerdict:
    def test_all_verdicts(self) -> None:
        assert Verdict.FIXED.value == "fixed"
        assert Verdict.IMPROVED.value == "improved"
        assert Verdict.UNCHANGED.value == "unchanged"
        assert Verdict.REGRESSED.value == "regressed"
        assert Verdict.BLOCKED.value == "blocked"


class TestIterationRecord:
    def test_to_dict(self) -> None:
        h = Hypothesis(
            id="h-1", category=FailureCategory.DEVICE_SELECTION,
            description="test", target_files=["a.py"], proposed_action="fix",
        )
        r = RunResult(exit_code=0, stdout="ok", stderr="", error_signature="", duration_s=0.1)
        it = IterationRecord(
            iteration=1, hypothesis=h, patch_description="applied cuda_call",
            files_changed=["a.py"], run_before=None, run_after=r,
            verdict=Verdict.FIXED, accepted=True, commit_sha="abc123",
        )
        d = it.to_dict()
        assert d["iteration"] == 1
        assert d["verdict"] == "fixed"
        assert d["accepted"] is True
        assert d["run_before"] is None
        assert d["commit_sha"] == "abc123"


class TestAdaptationTask:
    def test_to_dict(self) -> None:
        t = AdaptationTask(
            id="task-001", name="device_selection:model.py",
            description="Fix 3 cuda patterns", category=FailureCategory.DEVICE_SELECTION,
            target_files=["model.py"],
        )
        d = t.to_dict()
        assert d["id"] == "task-001"
        assert d["status"] == "pending"
        assert d["category"] == "device_selection"

    def test_default_status(self) -> None:
        t = AdaptationTask(
            id="t-1", name="test", description="test",
            category=FailureCategory.UNKNOWN_BLOCKER,
        )
        assert t.status == "pending"
        assert t.blocker_reason is None


class TestAdaptationState:
    def test_to_dict(self) -> None:
        s = AdaptationState(repo_path=Path("/tmp/test"), model_name="test-model")
        d = s.to_dict()
        assert d["model_name"] == "test-model"
        assert d["iteration"] == 0
        assert d["stop_reason"] is None
        assert isinstance(d["files_modified"], list)

    def test_default_values(self) -> None:
        s = AdaptationState(repo_path=Path("/tmp/test"), model_name="m")
        assert s.max_iterations == 20
        assert s.no_progress_limit == 3
        assert s.consecutive_no_progress == 0
        assert s.stop_reason is None


class TestStopReason:
    def test_all_reasons(self) -> None:
        assert StopReason.INFERENCE_SUCCESS.value == "inference_success"
        assert StopReason.MAX_ITERATIONS.value == "max_iterations"
        assert StopReason.REPEATED_NO_PROGRESS.value == "repeated_no_progress"
        assert StopReason.ALL_RULES_APPLIED.value == "all_rules_applied"
