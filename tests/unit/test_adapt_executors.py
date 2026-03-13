"""Tests for adapt/executors.py — command execution backends."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from diffusion_agent.adapt.executors import (
    ExecutionResult,
    LocalExecutor,
    SSHConfig,
    SSHExecutor,
)


# ---------------------------------------------------------------------------
# LocalExecutor
# ---------------------------------------------------------------------------

class TestLocalExecutor:
    def test_echo(self, tmp_path: Path) -> None:
        ex = LocalExecutor(cwd=str(tmp_path))
        result = ex.execute("echo hello")
        assert result.exit_code == 0
        assert "hello" in result.stdout
        assert result.duration_s >= 0

    def test_failure(self, tmp_path: Path) -> None:
        ex = LocalExecutor(cwd=str(tmp_path))
        result = ex.execute("python -c 'raise RuntimeError(\"boom\")'")
        assert result.exit_code != 0
        assert "RuntimeError" in result.stderr

    def test_timeout(self, tmp_path: Path) -> None:
        ex = LocalExecutor(cwd=str(tmp_path))
        result = ex.execute("sleep 10", timeout=1)
        assert result.exit_code == -1
        assert "timed out" in result.stderr.lower()

    def test_cwd_is_used(self, tmp_path: Path) -> None:
        ex = LocalExecutor(cwd=str(tmp_path))
        result = ex.execute("pwd")
        assert result.exit_code == 0
        assert str(tmp_path) in result.stdout

    def test_no_cwd(self) -> None:
        ex = LocalExecutor()
        result = ex.execute("echo ok")
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# SSHConfig
# ---------------------------------------------------------------------------

class TestSSHConfig:
    def test_defaults(self) -> None:
        cfg = SSHConfig(host="example.com")
        assert cfg.user == "root"
        assert cfg.port == 22
        assert cfg.conda_env is None
        assert cfg.remote_workdir is None

    def test_to_dict(self) -> None:
        cfg = SSHConfig(host="h", user="u", conda_env="env1")
        d = cfg.to_dict()
        assert d["host"] == "h"
        assert d["conda_env"] == "env1"


# ---------------------------------------------------------------------------
# SSHExecutor — command construction (no real SSH)
# ---------------------------------------------------------------------------

class TestSSHExecutorBuild:
    def test_basic_command(self) -> None:
        cfg = SSHConfig(host="server.example.com")
        ex = SSHExecutor(cfg)
        remote = ex._build_remote_command("echo hello")
        assert "echo hello" in remote

    def test_workdir(self) -> None:
        cfg = SSHConfig(host="h", remote_workdir="/data/models/flux")
        ex = SSHExecutor(cfg)
        remote = ex._build_remote_command("python run.py")
        assert "cd " in remote
        assert "/data/models/flux" in remote
        assert "python run.py" in remote

    def test_conda_activation(self) -> None:
        cfg = SSHConfig(host="h", conda_env="torch280_py310_diffusion")
        ex = SSHExecutor(cfg)
        remote = ex._build_remote_command("python -c 'import torch'")
        assert "conda activate" in remote
        assert "torch280_py310_diffusion" in remote

    def test_pre_commands(self) -> None:
        cfg = SSHConfig(host="h", pre_commands=["source /opt/env.sh"])
        ex = SSHExecutor(cfg)
        remote = ex._build_remote_command("echo ok")
        assert "source /opt/env.sh" in remote

    def test_full_composition(self) -> None:
        cfg = SSHConfig(
            host="175.100.2.7",
            user="root",
            remote_workdir="/data/models/LTX-2",
            conda_env="torch280",
            pre_commands=["export ASCEND_HOME=/usr/local/Ascend"],
        )
        ex = SSHExecutor(cfg)
        remote = ex._build_remote_command("python infer.py")
        parts = remote.split(" && ")
        assert len(parts) == 4  # cd, export, conda activate, python infer.py

    def test_ssh_args_default_port(self) -> None:
        cfg = SSHConfig(host="server.com")
        ex = SSHExecutor(cfg)
        args = ex._build_ssh_args("echo test")
        assert "ssh" in args
        assert "root@server.com" in args
        assert "-p" not in args  # default port, not added

    def test_ssh_args_custom_port(self) -> None:
        cfg = SSHConfig(host="server.com", port=2222)
        ex = SSHExecutor(cfg)
        args = ex._build_ssh_args("echo test")
        idx = args.index("-p")
        assert args[idx + 1] == "2222"

    def test_ssh_args_batch_mode(self) -> None:
        cfg = SSHConfig(host="h")
        ex = SSHExecutor(cfg)
        args = ex._build_ssh_args("cmd")
        assert "BatchMode=yes" in args

    def test_ssh_args_uses_bash_lc(self) -> None:
        cfg = SSHConfig(host="h")
        ex = SSHExecutor(cfg)
        args = ex._build_ssh_args("echo test")
        assert "bash" in args
        assert "-lc" in args


# ---------------------------------------------------------------------------
# SSHExecutor — execute (mocked subprocess)
# ---------------------------------------------------------------------------

class TestSSHExecutorExecute:
    def test_success(self) -> None:
        cfg = SSHConfig(host="h")
        ex = SSHExecutor(cfg)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "ok\n"
        mock_result.stderr = ""
        with patch("diffusion_agent.adapt.executors.subprocess.run", return_value=mock_result):
            result = ex.execute("echo ok")
        assert result.exit_code == 0
        assert "ok" in result.stdout

    def test_failure(self) -> None:
        cfg = SSHConfig(host="h")
        ex = SSHExecutor(cfg)
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "error\n"
        with patch("diffusion_agent.adapt.executors.subprocess.run", return_value=mock_result):
            result = ex.execute("bad_cmd")
        assert result.exit_code == 1
        assert "error" in result.stderr

    def test_timeout(self) -> None:
        import subprocess as sp
        cfg = SSHConfig(host="h")
        ex = SSHExecutor(cfg)
        with patch("diffusion_agent.adapt.executors.subprocess.run", side_effect=sp.TimeoutExpired("ssh", 5)):
            result = ex.execute("sleep 999", timeout=5)
        assert result.exit_code == -1
        assert "timed out" in result.stderr.lower()

    def test_os_error(self) -> None:
        cfg = SSHConfig(host="h")
        ex = SSHExecutor(cfg)
        with patch("diffusion_agent.adapt.executors.subprocess.run", side_effect=OSError("no ssh")):
            result = ex.execute("echo ok")
        assert result.exit_code == -1
        assert "SSH execution error" in result.stderr


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------

class TestExecutionResult:
    def test_fields(self) -> None:
        r = ExecutionResult(exit_code=0, stdout="out", stderr="err", duration_s=1.5, command="cmd")
        assert r.exit_code == 0
        assert r.command == "cmd"
