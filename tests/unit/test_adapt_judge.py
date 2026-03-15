"""Tests for adapt/judge.py — progress judgment."""

from __future__ import annotations

from diffusion_agent.adapt.judge import (
    AdaptJudge,
    classify_failure,
    extract_error_context,
    evaluate_task_progress,
    is_blocker,
)
from diffusion_agent.adapt.types import (
    AdaptationTask,
    FailureCategory,
    Hypothesis,
    RunResult,
    TaskStopReason,
    Verdict,
)


class TestClassifyFailure:
    def test_empty_stderr(self) -> None:
        assert classify_failure("") == FailureCategory.UNKNOWN_BLOCKER

    def test_module_not_found(self) -> None:
        # flash_attn is a custom extension, so it matches CUSTOM_EXTENSION before IMPORT_MODULE
        assert classify_failure("ModuleNotFoundError: No module named 'flash_attn'") == FailureCategory.CUSTOM_EXTENSION

    def test_module_not_found_generic(self) -> None:
        assert classify_failure("ModuleNotFoundError: No module named 'my_module'") == FailureCategory.IMPORT_MODULE

    def test_import_error(self) -> None:
        assert classify_failure("ImportError: cannot import name 'foo'") == FailureCategory.IMPORT_MODULE

    def test_cuda_device(self) -> None:
        assert classify_failure("RuntimeError: CUDA error: device not found") == FailureCategory.DEVICE_SELECTION

    def test_torch_cuda_api(self) -> None:
        assert classify_failure("torch.cuda.is_available() returned False") == FailureCategory.CUDA_ONLY_API

    def test_nccl(self) -> None:
        assert classify_failure("nccl not available for distributed backend") == FailureCategory.DISTRIBUTED_BACKEND

    def test_dtype(self) -> None:
        assert classify_failure("float64 is not supported, use float32") == FailureCategory.DTYPE_AUTOCAST

    def test_unsupported_op(self) -> None:
        cat = classify_failure("not implemented for 'PrivateUse1'")
        assert cat == FailureCategory.UNSUPPORTED_OP

    def test_no_kernel(self) -> None:
        cat = classify_failure("No kernel registered for operator 'aten::foo' on NPU")
        assert cat == FailureCategory.UNSUPPORTED_OP

    def test_environment(self) -> None:
        assert classify_failure("pip install failed for requirements.txt") == FailureCategory.ENVIRONMENT_SETUP

    def test_custom_extension(self) -> None:
        cat = classify_failure("No module named 'flash_attn'")
        assert cat == FailureCategory.CUSTOM_EXTENSION


class TestIsBlocker:
    def test_unsupported_op_is_blocker(self) -> None:
        assert is_blocker(FailureCategory.UNSUPPORTED_OP) is True

    def test_custom_extension_is_blocker(self) -> None:
        assert is_blocker(FailureCategory.CUSTOM_EXTENSION) is True

    def test_device_selection_not_blocker(self) -> None:
        assert is_blocker(FailureCategory.DEVICE_SELECTION) is False

    def test_import_module_not_blocker(self) -> None:
        assert is_blocker(FailureCategory.IMPORT_MODULE) is False


def _make_run(exit_code: int, error_sig: str = "", stderr: str = "") -> RunResult:
    return RunResult(
        exit_code=exit_code, stdout="", stderr=stderr,
        error_signature=error_sig, duration_s=0.1,
    )


class TestAdaptJudge:
    def setup_method(self) -> None:
        self.judge = AdaptJudge()

    def test_fixed_from_failure(self) -> None:
        before = _make_run(1, "RuntimeError: cuda")
        after = _make_run(0)
        assert self.judge.judge(before, after) == Verdict.FIXED

    def test_fixed_no_baseline(self) -> None:
        after = _make_run(0)
        assert self.judge.judge(None, after) == Verdict.FIXED

    def test_regressed(self) -> None:
        before = _make_run(0)
        after = _make_run(1, "RuntimeError: new", stderr="RuntimeError: new error")
        assert self.judge.judge(before, after) == Verdict.REGRESSED

    def test_unchanged_same_error(self) -> None:
        before = _make_run(1, "RuntimeError: same")
        after = _make_run(1, "RuntimeError: same")
        assert self.judge.judge(before, after) == Verdict.UNCHANGED

    def test_blocked(self) -> None:
        before = _make_run(1, "old error")
        after = _make_run(1, "not impl", stderr="not implemented for 'PrivateUse1'")
        assert self.judge.judge(before, after) == Verdict.BLOCKED

    def test_different_failure(self) -> None:
        before = _make_run(1, "RuntimeError: A", stderr="RuntimeError: A")
        after = _make_run(1, "ValueError: B", stderr="ValueError: B something")
        # Should detect different failure since A no longer in stderr
        v = self.judge.judge(before, after)
        assert v in {Verdict.IMPROVED, Verdict.DIFFERENT_FAILURE}

    def test_first_run_failure_not_blocker(self) -> None:
        after = _make_run(1, "RuntimeError: test", stderr="RuntimeError: test")
        assert self.judge.judge(None, after) == Verdict.UNCHANGED

    def test_first_run_failure_blocker(self) -> None:
        after = _make_run(1, "not impl", stderr="not implemented for 'PrivateUse1'")
        assert self.judge.judge(None, after) == Verdict.BLOCKED


class TestShouldAccept:
    def setup_method(self) -> None:
        self.judge = AdaptJudge()

    def test_accept_fixed(self) -> None:
        assert self.judge.should_accept(Verdict.FIXED) is True

    def test_accept_improved(self) -> None:
        assert self.judge.should_accept(Verdict.IMPROVED) is True

    def test_reject_unchanged(self) -> None:
        assert self.judge.should_accept(Verdict.UNCHANGED) is False

    def test_reject_regressed(self) -> None:
        assert self.judge.should_accept(Verdict.REGRESSED) is False


class TestShouldStop:
    def setup_method(self) -> None:
        self.judge = AdaptJudge()

    def test_stop_fixed(self) -> None:
        assert self.judge.should_stop(Verdict.FIXED) is True

    def test_stop_blocked(self) -> None:
        assert self.judge.should_stop(Verdict.BLOCKED) is True

    def test_continue_unchanged(self) -> None:
        assert self.judge.should_stop(Verdict.UNCHANGED) is False

    def test_continue_improved(self) -> None:
        assert self.judge.should_stop(Verdict.IMPROVED) is False


# =========================================================================
# New tests: OOM / SYNTAX_ERROR / LOGIC_BUG classification
# =========================================================================


class TestClassifyFailureNewCategories:
    """classify_failure should recognize OOM, SyntaxError, and logic bugs."""

    # --- OOM ---
    def test_oom_cuda_out_of_memory(self) -> None:
        stderr = "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
        assert classify_failure(stderr) == FailureCategory.OOM

    def test_oom_npu_out_of_memory(self) -> None:
        stderr = "RuntimeError: NPU out of memory. NPU0 has 32GB total capacity"
        assert classify_failure(stderr) == FailureCategory.OOM

    def test_oom_torch_allocator(self) -> None:
        stderr = "torch.npu.OutOfMemoryError: NPU0 ran out of memory"
        assert classify_failure(stderr) == FailureCategory.OOM

    def test_oom_generic_allocate_failed(self) -> None:
        stderr = "RuntimeError: Can't call numpy() on Tensor. Failed to allocate memory"
        # "allocate" alone is not enough — needs OOM-specific context
        # This should NOT match OOM
        cat = classify_failure(stderr)
        assert cat != FailureCategory.OOM

    # --- SyntaxError ---
    def test_syntax_error_basic(self) -> None:
        stderr = (
            "  File \"model.py\", line 42\n"
            "    def forward(self\n"
            "                    ^\n"
            "SyntaxError: invalid syntax"
        )
        assert classify_failure(stderr) == FailureCategory.SYNTAX_ERROR

    def test_syntax_error_indentation(self) -> None:
        stderr = "IndentationError: unexpected indent"
        assert classify_failure(stderr) == FailureCategory.SYNTAX_ERROR

    def test_syntax_error_tab(self) -> None:
        stderr = "TabError: inconsistent use of tabs and spaces in indentation"
        assert classify_failure(stderr) == FailureCategory.SYNTAX_ERROR

    # --- LOGIC_BUG ---
    def test_logic_bug_assertion(self) -> None:
        stderr = "AssertionError: expected shape (3, 224, 224) but got (224, 224, 3)"
        assert classify_failure(stderr) == FailureCategory.LOGIC_BUG

    def test_logic_bug_shape_mismatch(self) -> None:
        stderr = "RuntimeError: shape mismatch: input tensor has 512 elements but got 256"
        assert classify_failure(stderr) == FailureCategory.LOGIC_BUG

    def test_logic_bug_size_mismatch(self) -> None:
        stderr = "RuntimeError: The size of tensor a (4) must match the size of tensor b (8)"
        assert classify_failure(stderr) == FailureCategory.LOGIC_BUG

    def test_logic_bug_dimension_error(self) -> None:
        stderr = "RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x4 and 5x6)"
        assert classify_failure(stderr) == FailureCategory.LOGIC_BUG


# =========================================================================
# NPU dtype errors should NOT be classified as UNSUPPORTED_OP blocker
# =========================================================================


class TestClassifyFailureNpuDtype:
    """NPU dtype limitation errors (DT_COMPLEX128 etc.) are fixable via
    view_as_real/view_as_complex and must NOT be classified as UNSUPPORTED_OP."""

    def test_aclnn_complex128_not_implemented(self) -> None:
        """'not implemented for DT_COMPLEX128' is a dtype issue, not an op issue."""
        stderr = (
            "AclNN_Parameter_Error(EZ1001): tensor 0 not implemented for DT_COMPLEX128, "
            "should be in dtype support list [DT_FLOAT,DT_INT32,...]."
        )
        cat = classify_failure(stderr)
        assert cat == FailureCategory.DTYPE_AUTOCAST
        assert not is_blocker(cat)

    def test_aclnn_complex64_not_implemented(self) -> None:
        stderr = "not implemented for DT_COMPLEX64, should be in dtype support list"
        cat = classify_failure(stderr)
        assert cat == FailureCategory.DTYPE_AUTOCAST
        assert not is_blocker(cat)

    def test_aclnn_double_not_implemented(self) -> None:
        """DT_DOUBLE (float64) may appear in some aclnn errors."""
        stderr = "not implemented for DT_DOUBLE, should be in dtype support list"
        cat = classify_failure(stderr)
        assert cat == FailureCategory.DTYPE_AUTOCAST
        assert not is_blocker(cat)

    def test_generic_not_implemented_still_unsupported_op(self) -> None:
        """'not implemented for PrivateUse1' (device issue) stays UNSUPPORTED_OP."""
        stderr = "not implemented for 'PrivateUse1'"
        cat = classify_failure(stderr)
        assert cat == FailureCategory.UNSUPPORTED_OP

    def test_aclnn_cat_full_stderr(self) -> None:
        """Full V9c aclnnCat error should be DTYPE_AUTOCAST, not blocker."""
        stderr = (
            "RuntimeError: cat:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:148 "
            "NPU function error: call aclnnCat failed, error code is 161002\n"
            "[PID: 1589280] AclNN_Parameter_Error(EZ1001): tensor 0 not implemented "
            "for DT_COMPLEX128, should be in dtype support list "
            "[DT_FLOAT,DT_INT32,DT_INT64,DT_FLOAT16,DT_INT16,DT_INT8,DT_UINT8,"
            "DT_DOUBLE,DT_COMPLEX64,DT_BFLOAT16,DT_BOOL,]."
        )
        cat = classify_failure(stderr)
        assert cat == FailureCategory.DTYPE_AUTOCAST
        assert not is_blocker(cat)


# =========================================================================
# New tests: extract_error_context
# =========================================================================


class TestExtractErrorContext:
    """extract_error_context trims noisy NPU stderr to the relevant traceback."""

    def test_extracts_last_traceback_block(self) -> None:
        stderr = (
            "Some info log line\n"
            "Another info line\n"
            "Traceback (most recent call last):\n"
            '  File "train.py", line 10, in <module>\n'
            "    model(x)\n"
            '  File "model.py", line 42, in forward\n'
            "    return self.attn(x)\n"
            "RuntimeError: not implemented for 'PrivateUse1'\n"
        )
        ctx = extract_error_context(stderr)
        assert "Traceback (most recent call last):" in ctx
        assert "RuntimeError: not implemented" in ctx
        # Info lines should be stripped
        assert "Some info log line" not in ctx

    def test_multiple_tracebacks_takes_last(self) -> None:
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "setup.py", line 1\n'
            "ImportError: no module\n"
            "\n"
            "During handling of the above exception:\n"
            "\n"
            "Traceback (most recent call last):\n"
            '  File "main.py", line 5\n'
            "RuntimeError: OOM\n"
        )
        ctx = extract_error_context(stderr)
        # Should contain the last traceback
        assert "main.py" in ctx
        assert "RuntimeError: OOM" in ctx

    def test_respects_max_lines(self) -> None:
        lines = ["line %d" % i for i in range(100)]
        lines.append("Traceback (most recent call last):")
        lines.append('  File "x.py", line 1')
        lines.append("RuntimeError: boom")
        stderr = "\n".join(lines)
        ctx = extract_error_context(stderr, max_lines=10)
        ctx_lines = ctx.strip().splitlines()
        assert len(ctx_lines) <= 10

    def test_empty_stderr(self) -> None:
        assert extract_error_context("") == ""

    def test_no_traceback_returns_tail(self) -> None:
        stderr = "error: something went wrong\ndetails: bad config\n"
        ctx = extract_error_context(stderr)
        # Should return the tail lines when no traceback is found
        assert "something went wrong" in ctx or "bad config" in ctx


# =========================================================================
# New tests: evaluate_task_progress (consecutive blocker escalation)
# =========================================================================


def _make_task_with_attempts(
    category: FailureCategory,
    attempt_categories: list[FailureCategory],
    attempt_stderrs: list[str],
) -> AdaptationTask:
    """Helper: build a task with pre-recorded attempts using specific stderrs."""
    task = AdaptationTask(
        id="t-1",
        name="test-task",
        description="test",
        category=category,
        target_files=["model.py"],
    )
    for i, (cat, stderr) in enumerate(zip(attempt_categories, attempt_stderrs)):
        hyp = Hypothesis(
            id=f"hyp-{i}",
            category=cat,
            description="test",
            target_files=["model.py"],
            proposed_action="test",
            source="llm",
        )
        task.record_attempt(
            hypothesis=hyp,
            verdict=Verdict.UNCHANGED,
            accepted=False,
            error_signature=stderr[:80],
        )
    return task


class TestEvaluateTaskProgress:
    """evaluate_task_progress detects consecutive same-category blocker patterns."""

    def test_three_consecutive_unsupported_op_returns_blocked(self) -> None:
        stderrs = [
            "not implemented for 'PrivateUse1'",
            "not implemented for 'PrivateUse1' on op aten::foo",
            "not implemented for 'PrivateUse1' on op aten::bar",
        ]
        task = _make_task_with_attempts(
            FailureCategory.UNSUPPORTED_OP,
            [FailureCategory.UNSUPPORTED_OP] * 3,
            stderrs,
        )
        result = evaluate_task_progress(task, stderrs)
        assert result == TaskStopReason.BLOCKED

    def test_two_unsupported_op_not_enough(self) -> None:
        stderrs = [
            "not implemented for 'PrivateUse1'",
            "not implemented for 'PrivateUse1' on op aten::foo",
        ]
        task = _make_task_with_attempts(
            FailureCategory.UNSUPPORTED_OP,
            [FailureCategory.UNSUPPORTED_OP] * 2,
            stderrs,
        )
        result = evaluate_task_progress(task, stderrs)
        assert result is None

    def test_mixed_categories_not_blocked(self) -> None:
        stderrs = [
            "not implemented for 'PrivateUse1'",
            "ModuleNotFoundError: No module named 'my_pkg'",
            "not implemented for 'PrivateUse1'",
        ]
        cats = [
            FailureCategory.UNSUPPORTED_OP,
            FailureCategory.IMPORT_MODULE,
            FailureCategory.UNSUPPORTED_OP,
        ]
        task = _make_task_with_attempts(FailureCategory.UNSUPPORTED_OP, cats, stderrs)
        result = evaluate_task_progress(task, stderrs)
        # Not 3 *consecutive* UNSUPPORTED_OP
        assert result is None

    def test_empty_attempts_returns_none(self) -> None:
        task = AdaptationTask(
            id="t-1", name="test", description="test",
            category=FailureCategory.DEVICE_SELECTION,
            target_files=["model.py"],
        )
        result = evaluate_task_progress(task, [])
        assert result is None

    def test_three_consecutive_oom_not_blocked(self) -> None:
        """OOM is retriable (can be fixed by reducing batch size), not auto-blocked."""
        stderrs = [
            "RuntimeError: CUDA out of memory",
            "RuntimeError: NPU out of memory",
            "RuntimeError: out of memory again",
        ]
        task = _make_task_with_attempts(
            FailureCategory.OOM,
            [FailureCategory.OOM] * 3,
            stderrs,
        )
        result = evaluate_task_progress(task, stderrs)
        # OOM should NOT trigger auto-block (it's fixable via code changes)
        assert result is None
