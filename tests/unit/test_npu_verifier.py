"""Tests for the NPU runtime verifier (SSH-based op verification)."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

from diffusion_agent.tools.npu_verifier import (
    VerifyResult,
    run_basic_checks,
    verify_import,
    verify_op,
)


# ---------------------------------------------------------------------------
# verify_op
# ---------------------------------------------------------------------------

class TestVerifyOp:
    @patch("diffusion_agent.tools.npu_verifier.subprocess.run")
    def test_verify_op_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="OK\n", stderr=""
        )
        result = verify_op("root@host", "myenv", "torch.matmul(a, b)", op_name="matmul")
        assert result.passed is True
        assert result.op_name == "matmul"
        assert result.error is None

    @patch("diffusion_agent.tools.npu_verifier.subprocess.run")
    def test_verify_op_failure(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="RuntimeError: unsupported op"
        )
        result = verify_op("root@host", "myenv", "torch.bad_op()", op_name="bad_op")
        assert result.passed is False
        assert result.error is not None
        assert "unsupported" in result.error

    @patch("diffusion_agent.tools.npu_verifier.subprocess.run")
    def test_verify_op_timeout(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=30)
        result = verify_op("root@host", "myenv", "slow_op()", op_name="slow")
        assert result.passed is False
        assert result.error is not None
        assert "timeout" in result.error.lower()

    @patch("diffusion_agent.tools.npu_verifier.subprocess.run")
    def test_verify_op_without_conda_env(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="OK\n", stderr=""
        )
        result = verify_op("root@host", None, "torch.ones(2)", op_name="ones")
        assert result.passed is True


# ---------------------------------------------------------------------------
# verify_import
# ---------------------------------------------------------------------------

class TestVerifyImport:
    @patch("diffusion_agent.tools.npu_verifier.subprocess.run")
    def test_import_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="OK\n", stderr=""
        )
        assert verify_import("root@host", "myenv", "torch_npu") is True

    @patch("diffusion_agent.tools.npu_verifier.subprocess.run")
    def test_import_failure(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="ModuleNotFoundError"
        )
        assert verify_import("root@host", "myenv", "nonexistent") is False


# ---------------------------------------------------------------------------
# run_basic_checks
# ---------------------------------------------------------------------------

class TestRunBasicChecks:
    @patch("diffusion_agent.tools.npu_verifier.subprocess.run")
    def test_basic_checks_all_pass(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="OK\n", stderr=""
        )
        results = run_basic_checks("root@host", "myenv")
        assert len(results) > 0
        assert all(r.passed for r in results)
        assert all(isinstance(r, VerifyResult) for r in results)

    @patch("diffusion_agent.tools.npu_verifier.subprocess.run")
    def test_basic_checks_some_fail(self, mock_run: MagicMock) -> None:
        def side_effect(*args, **kwargs):
            cmd = args[0][-1] if args[0] else kwargs.get("args", [""])[-1]
            if "matmul" in str(cmd):
                return subprocess.CompletedProcess(args=[], returncode=0, stdout="OK", stderr="")
            return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="Error")
        mock_run.side_effect = side_effect
        results = run_basic_checks("root@host", "myenv")
        assert len(results) > 0
        # At least some should pass (matmul) and some fail
        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]
        assert len(passed) >= 1
        assert len(failed) >= 1
