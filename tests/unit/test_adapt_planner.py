"""Tests for adapt/planner.py — failure classification and hypothesis generation."""

from __future__ import annotations


from diffusion_agent.adapt.planner import AdaptPlanner
from diffusion_agent.adapt.types import (
    AdaptationTask,
    FailureCategory,
    Hypothesis,
    IterationRecord,
    RunResult,
    Verdict,
)
from diffusion_agent.tools.code_migrator import create_default_registry
from diffusion_agent.tools.code_scanner import Finding, PatternType


def _finding(file_path: str, pattern: PatternType, line: int = 1) -> Finding:
    return Finding(
        file_path=file_path,
        line_number=line,
        pattern_type=pattern,
        code_snippet="test",
        context="context",
    )


class TestDecomposeTasks:
    def setup_method(self) -> None:
        self.planner = AdaptPlanner(create_default_registry())

    def test_groups_by_category_and_file(self) -> None:
        findings = [
            _finding("a.py", PatternType.CUDA_CALL, 1),
            _finding("a.py", PatternType.CUDA_TO, 2),
            _finding("b.py", PatternType.NCCL, 1),
        ]
        plan = create_default_registry().match_all(findings)
        tasks = self.planner.decompose_tasks(findings, plan)
        assert len(tasks) >= 2  # at least a.py group and b.py group
        assert all(isinstance(t, AdaptationTask) for t in tasks)

    def test_empty_findings(self) -> None:
        plan = create_default_registry().match_all([])
        tasks = self.planner.decompose_tasks([], plan)
        assert tasks == []

    def test_task_has_required_fields(self) -> None:
        findings = [_finding("model.py", PatternType.CUDA_CALL)]
        plan = create_default_registry().match_all(findings)
        tasks = self.planner.decompose_tasks(findings, plan)
        assert len(tasks) == 1
        t = tasks[0]
        assert t.id.startswith("task-")
        assert t.category == FailureCategory.DEVICE_SELECTION
        assert "model.py" in t.target_files
        assert t.status == "pending"

    def test_priority_order(self) -> None:
        findings = [
            _finding("a.py", PatternType.CUDA_CALL),
            _finding("b.py", PatternType.FLASH_ATTN),  # should come first (custom_extension)
        ]
        plan = create_default_registry().match_all(findings)
        tasks = self.planner.decompose_tasks(findings, plan)
        assert len(tasks) == 2
        # flash_attn → CUSTOM_EXTENSION should be first
        assert tasks[0].category == FailureCategory.CUSTOM_EXTENSION
        assert tasks[1].category == FailureCategory.DEVICE_SELECTION


class TestGenerateHypothesis:
    def setup_method(self) -> None:
        self.registry = create_default_registry()
        self.planner = AdaptPlanner(self.registry)

    def test_rule_based_hypothesis(self) -> None:
        findings = [_finding("model.py", PatternType.CUDA_CALL)]
        plan = self.registry.match_all(findings)
        task = AdaptationTask(
            id="t-1", name="test", description="test",
            category=FailureCategory.DEVICE_SELECTION,
            target_files=["model.py"],
        )
        h = self.planner.generate_hypothesis(task, findings, plan)
        assert h is not None
        assert h.source == "rule"
        assert "cuda_call" in h.proposed_action

    def test_llm_hypothesis_for_unmatched(self) -> None:
        findings = [_finding("model.py", PatternType.DISTRIBUTED)]
        plan = self.registry.match_all(findings)
        task = AdaptationTask(
            id="t-1", name="test", description="test",
            category=FailureCategory.DISTRIBUTED_BACKEND,
            target_files=["model.py"],
        )
        h = self.planner.generate_hypothesis(task, findings, plan)
        assert h is not None
        assert h.source == "llm"

    def test_no_hypothesis_for_empty_files(self) -> None:
        findings = [_finding("other.py", PatternType.CUDA_CALL)]
        plan = self.registry.match_all(findings)
        task = AdaptationTask(
            id="t-1", name="test", description="test",
            category=FailureCategory.DEVICE_SELECTION,
            target_files=["model.py"],  # doesn't match other.py
        )
        h = self.planner.generate_hypothesis(task, findings, plan)
        assert h is None


class TestCountNoProgress:
    def setup_method(self) -> None:
        self.planner = AdaptPlanner(create_default_registry())

    def _make_iteration(self, verdict: Verdict) -> IterationRecord:
        h = Hypothesis(
            id="h-1", category=FailureCategory.DEVICE_SELECTION,
            description="test", target_files=[], proposed_action="test",
        )
        r = RunResult(exit_code=0, stdout="", stderr="", error_signature="", duration_s=0.1)
        return IterationRecord(
            iteration=1, hypothesis=h, patch_description="test",
            files_changed=[], run_before=None, run_after=r,
            verdict=verdict, accepted=False,
        )

    def test_no_iterations(self) -> None:
        assert self.planner.count_no_progress([]) == 0

    def test_all_progress(self) -> None:
        its = [self._make_iteration(Verdict.FIXED)]
        assert self.planner.count_no_progress(its) == 0

    def test_trailing_no_progress(self) -> None:
        its = [
            self._make_iteration(Verdict.FIXED),
            self._make_iteration(Verdict.UNCHANGED),
            self._make_iteration(Verdict.UNCHANGED),
        ]
        assert self.planner.count_no_progress(its) == 2

    def test_all_no_progress(self) -> None:
        its = [
            self._make_iteration(Verdict.REGRESSED),
            self._make_iteration(Verdict.DIFFERENT_FAILURE),
            self._make_iteration(Verdict.UNCHANGED),
        ]
        assert self.planner.count_no_progress(its) == 3
