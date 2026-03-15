"""Tests for adapt/planner.py — failure classification and hypothesis generation."""

from __future__ import annotations


from pathlib import Path

from diffusion_agent.adapt.planner import AdaptPlanner, parse_traceback_files, trim_error_log
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


class TestHypothesisDedup:
    """Tests that the planner skips already-tried hypotheses."""

    def setup_method(self) -> None:
        self.registry = create_default_registry()
        self.planner = AdaptPlanner(self.registry)

    def test_skips_already_tried_rule_hypothesis(self) -> None:
        findings = [_finding("model.py", PatternType.CUDA_CALL)]
        plan = self.registry.match_all(findings)
        task = AdaptationTask(
            id="t-1", name="test", description="test",
            category=FailureCategory.DEVICE_SELECTION,
            target_files=["model.py"],
        )
        # First call: should return rule hypothesis
        h1 = self.planner.generate_hypothesis(task, findings, plan)
        assert h1 is not None
        assert h1.source == "rule"

        # Record attempt with that hypothesis
        task.record_attempt(h1, Verdict.UNCHANGED, False, "err-1")

        # Second call: rule hyp already seen → falls through to LLM
        h2 = self.planner.generate_hypothesis(task, findings, plan)
        assert h2 is not None
        assert h2.source == "llm"
        assert h2.id != h1.id

    def test_returns_none_on_repeated_error(self) -> None:
        findings = [_finding("model.py", PatternType.DISTRIBUTED)]
        plan = self.registry.match_all(findings)
        task = AdaptationTask(
            id="t-1", name="test", description="test",
            category=FailureCategory.DISTRIBUTED_BACKEND,
            target_files=["model.py"],
        )
        # Record two attempts with the same error signature
        h_dummy = Hypothesis(
            id="hyp-t-1-llm-1", category=FailureCategory.DISTRIBUTED_BACKEND,
            description="test", target_files=["model.py"],
            proposed_action="fix", source="llm",
        )
        task.record_attempt(h_dummy, Verdict.UNCHANGED, False, "same-err")
        h_dummy2 = Hypothesis(
            id="hyp-t-1-llm-2", category=FailureCategory.DISTRIBUTED_BACKEND,
            description="test2", target_files=["model.py"],
            proposed_action="fix2", source="llm",
        )
        task.record_attempt(h_dummy2, Verdict.UNCHANGED, False, "same-err")

        # Now planner should return None (repeated error seen >= 2 times)
        h = self.planner.generate_hypothesis(task, findings, plan)
        assert h is None

    def test_llm_hypothesis_ids_are_unique(self) -> None:
        findings = [_finding("model.py", PatternType.DISTRIBUTED)]
        plan = self.registry.match_all(findings)
        task = AdaptationTask(
            id="t-1", name="test", description="test",
            category=FailureCategory.DISTRIBUTED_BACKEND,
            target_files=["model.py"],
        )
        h1 = self.planner.generate_hypothesis(task, findings, plan)
        assert h1 is not None
        task.record_attempt(h1, Verdict.UNCHANGED, False, "err-1")

        h2 = self.planner.generate_hypothesis(task, findings, plan)
        assert h2 is not None
        assert h2.id != h1.id


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


# =========================================================================
# New tests: trim_error_log
# =========================================================================


class TestTrimErrorLog:
    """trim_error_log should extract the core traceback and cap line count."""

    def test_extracts_traceback_block(self) -> None:
        stderr = (
            "INFO: loading model weights\n"
            "INFO: model loaded in 2.3s\n"
            "WARNING: some deprecation notice\n"
            "Traceback (most recent call last):\n"
            '  File "train.py", line 100, in <module>\n'
            "    model.forward(x)\n"
            '  File "model.py", line 42, in forward\n'
            "    return self.attn(x)\n"
            "RuntimeError: not implemented for 'PrivateUse1'\n"
        )
        trimmed = trim_error_log(stderr)
        assert "Traceback (most recent call last):" in trimmed
        assert "RuntimeError: not implemented" in trimmed
        # Noise lines should be gone
        assert "INFO: loading model weights" not in trimmed

    def test_respects_max_lines(self) -> None:
        lines = [f"noise line {i}" for i in range(200)]
        lines.append("Traceback (most recent call last):")
        for i in range(40):
            lines.append(f'  File "f{i}.py", line {i}')
        lines.append("RuntimeError: boom")
        stderr = "\n".join(lines)
        trimmed = trim_error_log(stderr, max_lines=15)
        assert len(trimmed.strip().splitlines()) <= 15

    def test_empty_stderr(self) -> None:
        assert trim_error_log("") == ""

    def test_no_traceback_returns_tail(self) -> None:
        stderr = "error: bad config\ndetails: missing key 'model_path'\n"
        trimmed = trim_error_log(stderr)
        assert "bad config" in trimmed or "missing key" in trimmed

    def test_preserves_exception_line(self) -> None:
        """The final exception line should always be present even if max_lines is tight."""
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "a.py", line 1\n'
            '  File "b.py", line 2\n'
            '  File "c.py", line 3\n'
            '  File "d.py", line 4\n'
            '  File "e.py", line 5\n'
            "RuntimeError: critical error here\n"
        )
        trimmed = trim_error_log(stderr, max_lines=4)
        # Even with tight limit, the exception line must survive
        assert "RuntimeError: critical error here" in trimmed


# =========================================================================
# New tests: Reflection context in LLM hypothesis
# =========================================================================


class TestHypothesisReflection:
    """When a task has prior failed attempts, the LLM hypothesis must include
    a reflection prompt analyzing why the previous attempt failed."""

    def setup_method(self) -> None:
        self.registry = create_default_registry()
        self.planner = AdaptPlanner(self.registry)

    def test_first_attempt_has_no_reflection(self) -> None:
        """First attempt should NOT have reflection context."""
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
        # No reflection keywords in first-attempt hypothesis
        assert "previous attempt" not in h.proposed_action.lower()
        assert "failed because" not in h.proposed_action.lower()

    def test_second_attempt_includes_reflection(self) -> None:
        """After a failed attempt, the next hypothesis must contain reflection."""
        findings = [_finding("model.py", PatternType.DISTRIBUTED)]
        plan = self.registry.match_all(findings)
        task = AdaptationTask(
            id="t-1", name="test", description="test",
            category=FailureCategory.DISTRIBUTED_BACKEND,
            target_files=["model.py"],
        )
        # First attempt fails
        h1 = self.planner.generate_hypothesis(task, findings, plan)
        assert h1 is not None
        task.record_attempt(
            h1, Verdict.UNCHANGED, False,
            error_signature="RuntimeError: nccl not available",
        )

        # Second attempt should include reflection
        h2 = self.planner.generate_hypothesis(task, findings, plan)
        assert h2 is not None
        assert h2.source == "llm"
        # Must reference the previous error
        assert "nccl not available" in h2.proposed_action
        # Must contain reflection framing
        assert "previous" in h2.proposed_action.lower() or "last" in h2.proposed_action.lower()

    def test_reflection_includes_previous_action(self) -> None:
        """Reflection should mention what was tried before."""
        findings = [_finding("model.py", PatternType.DISTRIBUTED)]
        plan = self.registry.match_all(findings)
        task = AdaptationTask(
            id="t-1", name="test", description="test",
            category=FailureCategory.DISTRIBUTED_BACKEND,
            target_files=["model.py"],
        )
        # Simulate a prior rule attempt that didn't fully fix the problem
        prior_hyp = Hypothesis(
            id="hyp-t-1-rules", category=FailureCategory.DISTRIBUTED_BACKEND,
            description="Apply migration rules to model.py",
            target_files=["model.py"],
            proposed_action="Apply rules: nccl_to_hccl",
            source="rule",
        )
        task.record_attempt(
            prior_hyp, Verdict.UNCHANGED, False,
            error_signature="RuntimeError: distributed backend not configured",
        )

        h2 = self.planner.generate_hypothesis(task, findings, plan)
        assert h2 is not None
        # Should reference what was previously tried
        assert "nccl_to_hccl" in h2.proposed_action or "migration rules" in h2.proposed_action.lower()

    def test_runtime_hypothesis_uses_trimmed_context(self) -> None:
        """generate_runtime_hypothesis should use trimmed error context,
        not the raw full stderr."""
        long_noise = "\n".join([f"INFO log line {i}" for i in range(100)])
        stderr = (
            long_noise + "\n"
            "Traceback (most recent call last):\n"
            '  File "main.py", line 10\n'
            "RuntimeError: shape mismatch\n"
        )
        run_result = RunResult(
            exit_code=1, stdout="", stderr=stderr,
            error_signature="RuntimeError: shape mismatch",
            duration_s=1.0,
        )
        h = self.planner.generate_runtime_hypothesis(run_result, [])
        assert h is not None
        # The description should contain the trimmed context, not 100 noise lines
        assert "shape mismatch" in h.description
        assert len(h.description) < len(stderr)  # must be shorter than raw stderr


# =========================================================================
# New tests: parse_traceback_files
# =========================================================================


class TestParseTracebackFiles:
    """Test traceback file extraction for Phase C runtime errors."""

    def test_extracts_local_repo_file(self, tmp_path: Path) -> None:
        """When the traceback path is inside repo_path, resolve it directly."""
        (tmp_path / "model.py").write_text("import torch\n")
        stderr = (
            "Traceback (most recent call last):\n"
            f'  File "{tmp_path / "model.py"}", line 1, in <module>\n'
            "    import torch\n"
            "ImportError: No module named 'torch'\n"
        )
        files = parse_traceback_files(stderr, repo_path=tmp_path)
        assert len(files) == 1
        assert files[0] == str(tmp_path / "model.py")

    def test_extracts_remote_path_via_workdir(self, tmp_path: Path) -> None:
        """Remote paths should be resolved by stripping the remote_workdir prefix."""
        (tmp_path / "wan").mkdir()
        (tmp_path / "wan" / "speech2video.py").write_text("import decord\n")
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "/home/user/work/wan/speech2video.py", line 1, in <module>\n'
            "    import decord\n"
            "ModuleNotFoundError: No module named 'decord'\n"
        )
        files = parse_traceback_files(
            stderr, repo_path=tmp_path, remote_workdir="/home/user/work",
        )
        assert len(files) == 1
        assert files[0] == str(tmp_path / "wan" / "speech2video.py")

    def test_skips_site_packages(self, tmp_path: Path) -> None:
        """stdlib and site-packages paths should be filtered out."""
        (tmp_path / "model.py").write_text("x = 1\n")
        stderr = (
            "Traceback (most recent call last):\n"
            f'  File "{tmp_path / "model.py"}", line 1, in <module>\n'
            '  File "/usr/lib/python3.10/importlib/__init__.py", line 126\n'
            '  File "/home/user/.local/lib/python3.10/site-packages/torch/nn.py", line 42\n'
            "RuntimeError: error\n"
        )
        files = parse_traceback_files(stderr, repo_path=tmp_path)
        assert len(files) == 1
        assert "model.py" in files[0]

    def test_returns_empty_for_no_traceback(self, tmp_path: Path) -> None:
        assert parse_traceback_files("just some log output", repo_path=tmp_path) == []

    def test_returns_empty_when_no_repo_path(self) -> None:
        stderr = 'Traceback (most recent call last):\n  File "x.py", line 1\nError\n'
        assert parse_traceback_files(stderr, repo_path=None) == []

    def test_deduplicates_files(self, tmp_path: Path) -> None:
        """Same file appearing multiple times in traceback should be deduped."""
        (tmp_path / "model.py").write_text("x = 1\n")
        stderr = (
            "Traceback (most recent call last):\n"
            f'  File "{tmp_path / "model.py"}", line 1\n'
            f'  File "{tmp_path / "model.py"}", line 5\n'
            "RuntimeError: error\n"
        )
        files = parse_traceback_files(stderr, repo_path=tmp_path)
        assert len(files) == 1

    def test_fallback_by_filename(self, tmp_path: Path) -> None:
        """When path doesn't match repo or remote, fall back to filename search."""
        (tmp_path / "unique_file.py").write_text("x = 1\n")
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "/some/unknown/path/unique_file.py", line 1\n'
            "RuntimeError: error\n"
        )
        files = parse_traceback_files(stderr, repo_path=tmp_path)
        assert len(files) == 1
        assert "unique_file.py" in files[0]

    def test_deepest_frame_first(self, tmp_path: Path) -> None:
        """The deepest frame (last in traceback) should be first in results."""
        (tmp_path / "entry.py").write_text("x = 1\n")
        (tmp_path / "inner.py").write_text("y = 2\n")
        stderr = (
            "Traceback (most recent call last):\n"
            f'  File "{tmp_path / "entry.py"}", line 1\n'
            f'  File "{tmp_path / "inner.py"}", line 2\n'
            "RuntimeError: error\n"
        )
        files = parse_traceback_files(stderr, repo_path=tmp_path)
        assert len(files) == 2
        # inner.py is the deepest frame → should be first
        assert "inner.py" in files[0]
        assert "entry.py" in files[1]


class TestRuntimeHypothesisTargetFiles:
    """Test that generate_runtime_hypothesis populates target_files from traceback."""

    def setup_method(self) -> None:
        self.planner = AdaptPlanner(create_default_registry())

    def test_populates_target_files_from_traceback(self, tmp_path: Path) -> None:
        (tmp_path / "model.py").write_text("import decord\n")
        stderr = (
            "Traceback (most recent call last):\n"
            f'  File "{tmp_path / "model.py"}", line 1, in <module>\n'
            "    import decord\n"
            "ModuleNotFoundError: No module named 'decord'\n"
        )
        run_result = RunResult(
            exit_code=1, stdout="", stderr=stderr,
            error_signature="ModuleNotFoundError: No module named 'decord'",
            duration_s=1.0,
        )
        h = self.planner.generate_runtime_hypothesis(
            run_result, [], repo_path=tmp_path,
        )
        assert h is not None
        assert len(h.target_files) >= 1
        assert any("model.py" in f for f in h.target_files)

    def test_remote_workdir_resolves_files(self, tmp_path: Path) -> None:
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "utils.py").write_text("x = 1\n")
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "/data/remote/lib/utils.py", line 1\n'
            "RuntimeError: error\n"
        )
        run_result = RunResult(
            exit_code=1, stdout="", stderr=stderr,
            error_signature="RuntimeError: error", duration_s=0.5,
        )
        h = self.planner.generate_runtime_hypothesis(
            run_result, [], repo_path=tmp_path, remote_workdir="/data/remote",
        )
        assert h is not None
        assert any("utils.py" in f for f in h.target_files)

    def test_empty_target_files_when_no_repo_path(self) -> None:
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "model.py", line 1\n'
            "RuntimeError: error\n"
        )
        run_result = RunResult(
            exit_code=1, stdout="", stderr=stderr,
            error_signature="RuntimeError: error", duration_s=0.5,
        )
        # No repo_path → can't resolve files
        h = self.planner.generate_runtime_hypothesis(run_result, [])
        assert h is not None
        assert h.target_files == []


# =========================================================================
# New tests: deepest_file in runtime hypothesis
# =========================================================================


class TestRuntimeHypothesisDeepestFile:
    """Runtime hypotheses should identify the deepest user-code frame."""

    def setup_method(self) -> None:
        self.planner = AdaptPlanner(create_default_registry())

    def test_deepest_file_set_from_traceback(self, tmp_path: Path) -> None:
        """deepest_file should be the deepest user-code frame (first in target_files)."""
        (tmp_path / "entry.py").write_text("import inner\n")
        (tmp_path / "inner.py").write_text("import decord\n")
        stderr = (
            "Traceback (most recent call last):\n"
            f'  File "{tmp_path / "entry.py"}", line 1, in <module>\n'
            "    import inner\n"
            f'  File "{tmp_path / "inner.py"}", line 1, in <module>\n'
            "    import decord\n"
            "ModuleNotFoundError: No module named 'decord'\n"
        )
        run_result = RunResult(
            exit_code=1, stdout="", stderr=stderr,
            error_signature="ModuleNotFoundError: No module named 'decord'",
            duration_s=1.0,
        )
        h = self.planner.generate_runtime_hypothesis(
            run_result, [], repo_path=tmp_path,
        )
        assert h is not None
        assert h.deepest_file is not None
        assert "inner.py" in h.deepest_file  # deepest frame, not entry.py

    def test_deepest_file_none_when_no_repo_path(self) -> None:
        """Without repo_path, no target files and no deepest_file."""
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "model.py", line 1\n'
            "RuntimeError: error\n"
        )
        run_result = RunResult(
            exit_code=1, stdout="", stderr=stderr,
            error_signature="RuntimeError: error", duration_s=0.5,
        )
        h = self.planner.generate_runtime_hypothesis(run_result, [])
        assert h is not None
        assert h.deepest_file is None

    def test_deepest_file_single_frame(self, tmp_path: Path) -> None:
        """When there's only one frame, deepest_file == the only file."""
        (tmp_path / "model.py").write_text("x = 1\n")
        stderr = (
            "Traceback (most recent call last):\n"
            f'  File "{tmp_path / "model.py"}", line 1\n'
            "RuntimeError: error\n"
        )
        run_result = RunResult(
            exit_code=1, stdout="", stderr=stderr,
            error_signature="RuntimeError: error", duration_s=0.5,
        )
        h = self.planner.generate_runtime_hypothesis(
            run_result, [], repo_path=tmp_path,
        )
        assert h is not None
        assert h.deepest_file is not None
        assert "model.py" in h.deepest_file
