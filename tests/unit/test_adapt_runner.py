"""Tests for adapt/runner.py — validation execution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from diffusion_agent.adapt.executors import ExecutionResult, LocalExecutor, SSHExecutor
from diffusion_agent.adapt.runner import (
    AdaptRunner,
    create_executor,
    normalize_error,
    validate_syntax_local,
)
from diffusion_agent.adapt.types import ExecutionConfig


# ---------------------------------------------------------------------------
# normalize_error
# ---------------------------------------------------------------------------

class TestNormalizeError:
    def test_empty_stderr(self) -> None:
        assert normalize_error("") == ""

    def test_module_not_found(self) -> None:
        stderr = "Traceback (most recent call last):\n  File ...\nModuleNotFoundError: No module named 'flash_attn'"
        sig = normalize_error(stderr)
        assert "ModuleNotFoundError" in sig

    def test_runtime_error(self) -> None:
        stderr = "RuntimeError: CUDA error: device-side assert triggered"
        sig = normalize_error(stderr)
        assert "RuntimeError" in sig

    def test_segfault(self) -> None:
        sig = normalize_error("Segmentation fault (core dumped)")
        assert "Segmentation fault" in sig

    def test_not_implemented(self) -> None:
        sig = normalize_error("not implemented for 'PrivateUse1'")
        assert "not implemented" in sig

    def test_strips_file_paths(self) -> None:
        stderr = 'File "/home/user/test.py", line 42\nRuntimeError: bad'
        sig = normalize_error(stderr)
        assert "/home/user/" not in sig

    def test_fallback_last_line(self) -> None:
        sig = normalize_error("some unknown error message")
        assert sig == "some unknown error message"

    def test_truncation(self) -> None:
        stderr = "RuntimeError: " + "x" * 300
        sig = normalize_error(stderr)
        assert len(sig) <= 200


# ---------------------------------------------------------------------------
# AdaptRunner — local execution (backward compat)
# ---------------------------------------------------------------------------

class TestAdaptRunner:
    def test_run_local_echo(self, tmp_path: Path) -> None:
        runner = AdaptRunner(tmp_path)
        result = runner.run_local(["echo", "hello"])
        assert result.exit_code == 0
        assert "hello" in result.stdout
        assert result.error_signature == ""
        assert result.duration_s >= 0

    def test_run_local_failure(self, tmp_path: Path) -> None:
        runner = AdaptRunner(tmp_path)
        result = runner.run_local(["python", "-c", "raise RuntimeError('test fail')"])
        assert result.exit_code != 0
        assert "RuntimeError" in result.error_signature

    def test_run_local_timeout(self, tmp_path: Path) -> None:
        runner = AdaptRunner(tmp_path, timeout=1)
        result = runner.run_local(["sleep", "10"])
        assert result.exit_code == -1
        assert "Timeout" in result.stderr or "timed out" in result.stderr.lower()

    def test_run_smoke_import(self, tmp_path: Path) -> None:
        runner = AdaptRunner(tmp_path)
        result = runner.run_smoke_import("os")
        assert result.exit_code == 0
        assert "import ok" in result.stdout

    def test_run_smoke_import_failure(self, tmp_path: Path) -> None:
        runner = AdaptRunner(tmp_path)
        result = runner.run_smoke_import("nonexistent_module_xyz")
        assert result.exit_code != 0

    def test_run_syntax_check_valid(self, tmp_path: Path) -> None:
        (tmp_path / "ok.py").write_text("x = 1\n")
        runner = AdaptRunner(tmp_path)
        result = runner.run_syntax_check([str(tmp_path / "ok.py")])
        assert result.exit_code == 0

    def test_run_syntax_check_no_files(self, tmp_path: Path) -> None:
        runner = AdaptRunner(tmp_path)
        result = runner.run_syntax_check([])
        assert result.exit_code == 0

    def test_run_command_string(self, tmp_path: Path) -> None:
        runner = AdaptRunner(tmp_path)
        result = runner.run_command_string("echo hello world")
        assert result.exit_code == 0
        assert "hello world" in result.stdout

    def test_run_command(self, tmp_path: Path) -> None:
        runner = AdaptRunner(tmp_path)
        result = runner.run_command("echo via_executor")
        assert result.exit_code == 0
        assert "via_executor" in result.stdout


# ---------------------------------------------------------------------------
# Executor integration
# ---------------------------------------------------------------------------

class TestRunnerWithExecutor:
    def test_custom_executor(self, tmp_path: Path) -> None:
        """Runner uses injected executor."""
        mock_exec = MagicMock()
        mock_exec.execute.return_value = ExecutionResult(
            exit_code=0, stdout="mocked\n", stderr="", duration_s=0.01, command="cmd"
        )
        runner = AdaptRunner(tmp_path, executor=mock_exec)
        result = runner.run_command("anything")
        assert result.exit_code == 0
        assert "mocked" in result.stdout
        mock_exec.execute.assert_called_once()

    def test_executor_from_local_config(self, tmp_path: Path) -> None:
        cfg = ExecutionConfig(mode="local", timeout=30)
        runner = AdaptRunner(tmp_path, execution_config=cfg)
        assert isinstance(runner.executor, LocalExecutor)
        assert runner.timeout == 30

    def test_executor_from_ssh_config(self, tmp_path: Path) -> None:
        cfg = ExecutionConfig(
            mode="ssh",
            ssh_host="example.com",
            ssh_user="user",
            conda_env="myenv",
            timeout=60,
        )
        runner = AdaptRunner(tmp_path, execution_config=cfg)
        assert isinstance(runner.executor, SSHExecutor)
        assert runner.timeout == 60

    def test_legacy_ssh_host_param(self, tmp_path: Path) -> None:
        """Legacy ssh_host param still creates an SSHExecutor."""
        runner = AdaptRunner(tmp_path, ssh_host="old.server.com", conda_env="env1")
        assert isinstance(runner.executor, SSHExecutor)

    def test_run_validation_configured(self, tmp_path: Path) -> None:
        cfg = ExecutionConfig(validation_command="echo validated")
        runner = AdaptRunner(tmp_path, execution_config=cfg)
        result = runner.run_validation()
        assert result is not None
        assert result.exit_code == 0
        assert "validated" in result.stdout

    def test_run_validation_not_configured(self, tmp_path: Path) -> None:
        runner = AdaptRunner(tmp_path)
        result = runner.run_validation()
        assert result is None


# ---------------------------------------------------------------------------
# create_executor factory
# ---------------------------------------------------------------------------

class TestCreateExecutor:
    def test_local_mode(self, tmp_path: Path) -> None:
        cfg = ExecutionConfig(mode="local")
        ex = create_executor(cfg, tmp_path)
        assert isinstance(ex, LocalExecutor)

    def test_ssh_mode(self) -> None:
        cfg = ExecutionConfig(mode="ssh", ssh_host="host.com", ssh_user="u", ssh_port=2222)
        ex = create_executor(cfg)
        assert isinstance(ex, SSHExecutor)
        assert ex.config.host == "host.com"
        assert ex.config.port == 2222

    def test_ssh_mode_no_host_falls_back_local(self) -> None:
        cfg = ExecutionConfig(mode="ssh")  # no host → local
        ex = create_executor(cfg)
        assert isinstance(ex, LocalExecutor)


# ---------------------------------------------------------------------------
# validate_syntax_local — fast local syntax check using ast.parse
# ---------------------------------------------------------------------------


class TestValidateSyntaxLocal:
    """Fast-fail local syntax check before expensive remote validation."""

    def test_valid_python_passes(self, tmp_path: Path) -> None:
        """Valid Python files should pass with exit_code=0."""
        (tmp_path / "ok.py").write_text("x = 1\nprint(x)\n")
        result = validate_syntax_local([str(tmp_path / "ok.py")])
        assert result.exit_code == 0
        assert result.error_signature == ""

    def test_syntax_error_detected(self, tmp_path: Path) -> None:
        """A file with a SyntaxError should fail with exit_code=1."""
        (tmp_path / "bad.py").write_text('def foo(\n  x = "unterminated\n')
        result = validate_syntax_local([str(tmp_path / "bad.py")])
        assert result.exit_code == 1
        assert "SyntaxError" in result.error_signature

    def test_unterminated_string_literal(self, tmp_path: Path) -> None:
        """Detects unterminated string literal — the exact error from Wan2.2 E2E."""
        (tmp_path / "init.py").write_text(
            'raise RuntimeError("AudioEncoder unavailable: librosa not installed.\n'
        )
        result = validate_syntax_local([str(tmp_path / "init.py")])
        assert result.exit_code == 1
        assert "SyntaxError" in result.error_signature

    def test_multiple_files_one_bad(self, tmp_path: Path) -> None:
        """If any file has a SyntaxError, the check fails."""
        (tmp_path / "good.py").write_text("x = 1\n")
        (tmp_path / "bad.py").write_text("def (:\n")
        result = validate_syntax_local([
            str(tmp_path / "good.py"),
            str(tmp_path / "bad.py"),
        ])
        assert result.exit_code == 1
        assert "bad.py" in result.stderr

    def test_empty_file_list_passes(self) -> None:
        """No files to check → success."""
        result = validate_syntax_local([])
        assert result.exit_code == 0

    def test_nonexistent_file_skipped(self, tmp_path: Path) -> None:
        """Nonexistent files should not crash the check."""
        result = validate_syntax_local([str(tmp_path / "nope.py")])
        assert result.exit_code == 0  # no parseable files → pass

    def test_stderr_contains_file_and_line(self, tmp_path: Path) -> None:
        """The error output should identify the broken file."""
        (tmp_path / "broken.py").write_text("class (\n")
        result = validate_syntax_local([str(tmp_path / "broken.py")])
        assert result.exit_code == 1
        assert "broken.py" in result.stderr
