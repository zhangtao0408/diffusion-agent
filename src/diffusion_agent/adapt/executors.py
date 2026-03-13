"""Command executors — local and SSH backends for running validation commands.

The executor abstraction allows the adaptation runner to execute commands
either locally or on a remote Ascend NPU server without changing the
runner's logic.

Design:
  - CommandExecutor: protocol defining the execute() contract
  - LocalExecutor: runs commands via subprocess
  - SSHExecutor: runs commands on a remote host via ssh CLI
  - ExecutionResult: raw output from a single command execution
"""

from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass, field
from typing import Protocol

from diffusion_agent.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Execution result (raw, pre-normalization)
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Raw output from a single command execution."""

    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    command: str = ""


# ---------------------------------------------------------------------------
# Executor protocol
# ---------------------------------------------------------------------------

class CommandExecutor(Protocol):
    """Protocol for command execution backends."""

    def execute(self, command: str, timeout: int = 120) -> ExecutionResult:
        """Execute a shell command and return raw results.

        Args:
            command: Shell command string to execute.
            timeout: Maximum seconds before killing the command.

        Returns:
            ExecutionResult with captured output.
        """
        ...


# ---------------------------------------------------------------------------
# Local executor
# ---------------------------------------------------------------------------

class LocalExecutor:
    """Executes commands locally via subprocess."""

    def __init__(self, cwd: str | None = None, env: dict[str, str] | None = None) -> None:
        self.cwd = cwd
        self.env = env

    def execute(self, command: str, timeout: int = 120) -> ExecutionResult:
        log.info("exec_local", cmd=command[:120], cwd=self.cwd)
        start = time.monotonic()
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=self.env,
            )
            elapsed = time.monotonic() - start
            return ExecutionResult(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_s=round(elapsed, 2),
                command=command,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=f"Command timed out after {timeout}s",
                duration_s=round(elapsed, 2),
                command=command,
            )
        except OSError as exc:
            elapsed = time.monotonic() - start
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=str(exc),
                duration_s=round(elapsed, 2),
                command=command,
            )


# ---------------------------------------------------------------------------
# SSH executor
# ---------------------------------------------------------------------------

@dataclass
class SSHConfig:
    """Configuration for SSH-based remote execution."""

    host: str
    user: str = "root"
    port: int = 22
    remote_workdir: str | None = None
    conda_env: str | None = None
    pre_commands: list[str] = field(default_factory=list)
    ssh_options: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "user": self.user,
            "port": self.port,
            "remote_workdir": self.remote_workdir,
            "conda_env": self.conda_env,
            "pre_commands": self.pre_commands,
        }


class SSHExecutor:
    """Executes commands on a remote host via ssh CLI.

    Assumes SSH key-based authentication is configured.
    Uses the system ``ssh`` binary for maximum compatibility.
    """

    def __init__(self, config: SSHConfig) -> None:
        self.config = config

    def _build_remote_command(self, command: str) -> str:
        """Compose the full remote shell command with env/workdir setup.

        Uses ``conda run -n <env> --no-capture-output`` instead of
        ``conda activate`` because activate requires conda init in the
        shell profile, which is unreliable in non-interactive ``bash -lc``.

        The workdir ``cd`` is placed inside the conda run invocation
        (via ``bash -c 'cd ... && ...'``) because conda run starts a
        fresh subshell that doesn't inherit the outer cwd.
        """
        parts: list[str] = []

        # Run any pre-commands (e.g. source scripts)
        for pre in self.config.pre_commands:
            parts.append(pre)

        # Build the inner command (cd + actual command)
        inner_parts: list[str] = []
        if self.config.remote_workdir:
            inner_parts.append(f"cd {shlex.quote(self.config.remote_workdir)}")
        inner_parts.append(command)
        inner_cmd = " && ".join(inner_parts)

        # Wrap with conda run if needed
        if self.config.conda_env:
            env = shlex.quote(self.config.conda_env)
            # Use bash -c to preserve the cd + command chain inside conda run
            escaped_inner = inner_cmd.replace("'", "'\\''")
            parts.append(f"conda run -n {env} --no-capture-output bash -c '{escaped_inner}'")
        else:
            parts.append(inner_cmd)

        return " && ".join(parts)

    def _build_ssh_args(self, remote_command: str) -> list[str]:
        """Build the full ssh invocation as a list of args."""
        args = ["ssh"]

        # Port
        if self.config.port != 22:
            args.extend(["-p", str(self.config.port)])

        # Standard options for non-interactive use
        args.extend([
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=10",
        ])

        # Extra user-provided SSH options
        for opt in self.config.ssh_options:
            args.extend(["-o", opt])

        # Target
        args.append(f"{self.config.user}@{self.config.host}")

        # Remote command — must be a SINGLE argument so SSH passes it intact.
        # Using bash -lc ensures login profile is sourced (needed for conda init).
        # IMPORTANT: we combine "bash -lc <quoted_cmd>" into one string because
        # SSH concatenates multiple command args with spaces, which breaks
        # bash -c's argument parsing (it would only execute the first word).
        args.append(f"bash -lc {shlex.quote(remote_command)}")

        return args

    def execute(self, command: str, timeout: int = 120) -> ExecutionResult:
        remote_cmd = self._build_remote_command(command)
        ssh_args = self._build_ssh_args(remote_cmd)

        cmd_preview = f"ssh {self.config.user}@{self.config.host}: {command[:80]}"
        log.info("exec_ssh", cmd=cmd_preview)

        start = time.monotonic()
        try:
            result = subprocess.run(
                ssh_args,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            elapsed = time.monotonic() - start
            return ExecutionResult(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_s=round(elapsed, 2),
                command=cmd_preview,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=f"SSH command timed out after {timeout}s",
                duration_s=round(elapsed, 2),
                command=cmd_preview,
            )
        except OSError as exc:
            elapsed = time.monotonic() - start
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=f"SSH execution error: {exc}",
                duration_s=round(elapsed, 2),
                command=cmd_preview,
            )
