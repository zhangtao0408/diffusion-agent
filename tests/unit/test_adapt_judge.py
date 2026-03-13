"""Tests for adapt/judge.py — progress judgment."""

from __future__ import annotations

from diffusion_agent.adapt.judge import AdaptJudge, classify_failure, is_blocker
from diffusion_agent.adapt.types import FailureCategory, RunResult, Verdict


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
