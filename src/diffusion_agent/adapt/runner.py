"""Validation runner — executes commands and captures structured results.

The runner is a deterministic execution component. It does not decide what to run
or whether the result is good. It only runs commands and captures output.

Execution is delegated to a CommandExecutor (local or SSH), keeping the runner
independent of where commands actually run.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from diffusion_agent.adapt.executors import (
    CommandExecutor,
    ExecutionResult,
    LocalExecutor,
    SSHConfig,
    SSHExecutor,
)
from diffusion_agent.adapt.types import ExecutionConfig, RunResult
from diffusion_agent.utils.logging import get_logger

log = get_logger(__name__)

# Patterns for normalizing error signatures
_ERROR_PATTERNS: list[tuple[str, str]] = [
    # Python tracebacks: extract the last exception line
    (r"(?:^|\n)((?:ModuleNotFoundError|ImportError|RuntimeError|AttributeError|"
     r"TypeError|ValueError|NotImplementedError|FileNotFoundError|OSError|"
     r"KeyError|IndexError|NameError|AssertionError)[^\n]*)", "exception"),
    # CUDA / NPU errors
    (r"(?:CUDA|cuda|npu|NPU)\s+(?:error|Error)[^\n]*", "device_error"),
    # Segfault / signal
    (r"(?:Segmentation fault|Signal \d+|Killed|core dumped)", "crash"),
    # torch op not implemented
    (r"(?:not implemented for|Could not run)[^\n]*", "op_missing"),
]


def normalize_error(stderr: str) -> str:
    """Extract a normalized error signature from stderr.

    The signature is a short, comparable string that identifies the *kind*
    of failure without path-specific or instance-specific noise.
    """
    if not stderr:
        return ""

    # Try each pattern, return the first match
    for pattern, _tag in _ERROR_PATTERNS:
        m = re.search(pattern, stderr, re.MULTILINE)
        if m:
            sig = m.group(0).strip()
            # Strip file paths and line numbers for stable comparison
            sig = re.sub(r'File "[^"]+", line \d+', "File ...", sig)
            # Strip hex addresses
            sig = re.sub(r"0x[0-9a-fA-F]+", "0x...", sig)
            return sig[:200]

    # Fallback: last non-empty line of stderr
    lines = [ln.strip() for ln in stderr.strip().splitlines() if ln.strip()]
    if lines:
        return lines[-1][:200]
    return ""


def validate_syntax_local(file_paths: list[str]) -> RunResult:
    """Fast local syntax check using ``ast.parse``.

    Parses each file without executing it.  Returns a :class:`RunResult` with
    ``exit_code=0`` when all files are syntactically valid, or ``exit_code=1``
    with the first ``SyntaxError`` details in *stderr* / *error_signature*.

    This is meant to be called **before** expensive remote validation so that
    obviously broken LLM patches are caught locally in milliseconds rather
    than after a ~20-second SSH round-trip + execution cycle.
    """
    if not file_paths:
        return RunResult(
            exit_code=0, stdout="syntax_check_local: no files",
            stderr="", error_signature="", duration_s=0.0,
            command="validate_syntax_local",
        )

    errors: list[str] = []
    for fp in file_paths:
        try:
            source = Path(fp).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # skip unreadable files
        try:
            ast.parse(source, filename=fp)
        except SyntaxError as exc:
            line_info = f"line {exc.lineno}" if exc.lineno else ""
            msg = f"  File \"{fp}\", {line_info}\nSyntaxError: {exc.msg}"
            errors.append(msg)

    if errors:
        stderr = "\n".join(errors)
        return RunResult(
            exit_code=1,
            stdout="",
            stderr=stderr,
            error_signature=normalize_error(stderr),
            duration_s=0.0,
            command="validate_syntax_local",
        )

    return RunResult(
        exit_code=0, stdout=f"syntax_check_local: {len(file_paths)} files OK",
        stderr="", error_signature="", duration_s=0.0,
        command="validate_syntax_local",
    )


def _result_from_exec(er: ExecutionResult) -> RunResult:
    """Convert an ExecutionResult into a RunResult with error normalization."""
    return RunResult(
        exit_code=er.exit_code,
        stdout=er.stdout,
        stderr=er.stderr,
        error_signature=normalize_error(er.stderr) if er.exit_code != 0 else "",
        duration_s=er.duration_s,
        command=er.command,
    )


def create_executor(config: ExecutionConfig, repo_path: Path | None = None) -> CommandExecutor:
    """Factory: create the right executor from an ExecutionConfig."""
    if config.mode == "ssh" and config.ssh_host:
        ssh_cfg = SSHConfig(
            host=config.ssh_host,
            user=config.ssh_user,
            port=config.ssh_port,
            remote_workdir=config.remote_workdir,
            conda_env=config.conda_env,
            pre_commands=list(config.pre_commands),
        )
        return SSHExecutor(ssh_cfg)

    # Default: local
    return LocalExecutor(
        cwd=str(repo_path) if repo_path else None,
        env=dict(config.env_vars) if config.env_vars else None,
    )


class AdaptRunner:
    """Runs validation commands against the target repo and captures results.

    Delegates actual command execution to a CommandExecutor (local or SSH).
    """

    def __init__(
        self,
        repo_path: Path,
        timeout: int = 120,
        ssh_host: str | None = None,
        conda_env: str | None = None,
        executor: CommandExecutor | None = None,
        execution_config: ExecutionConfig | None = None,
    ) -> None:
        self.repo_path = repo_path
        self.timeout = timeout

        # Build executor from the most specific source available
        if executor is not None:
            self._executor = executor
        elif execution_config is not None:
            self._executor = create_executor(execution_config, repo_path)
            self.timeout = execution_config.timeout
        elif ssh_host:
            # Legacy path: construct SSH executor from individual args
            ssh_cfg = SSHConfig(
                host=ssh_host,
                conda_env=conda_env,
            )
            self._executor = SSHExecutor(ssh_cfg)
        else:
            self._executor = LocalExecutor(cwd=str(repo_path))

        # Store validation command from config
        self._validation_command = (
            execution_config.validation_command if execution_config else None
        )

    @property
    def executor(self) -> CommandExecutor:
        """The underlying command executor."""
        return self._executor

    def run_command(self, command: str) -> RunResult:
        """Run a shell command through the executor and return a RunResult."""
        er = self._executor.execute(command, timeout=self.timeout)
        return _result_from_exec(er)

    # ----- Convenience methods -----

    def run_local(self, cmd: list[str]) -> RunResult:
        """Run a command locally in the repo directory.

        For backward compatibility. Always uses local subprocess regardless
        of the configured executor.
        """
        cmd_str = " ".join(cmd)
        local = LocalExecutor(cwd=str(self.repo_path))
        er = local.execute(cmd_str, timeout=self.timeout)
        return _result_from_exec(er)

    def run_smoke_import(self, module_name: str = "") -> RunResult:
        """Quick smoke test: try importing the main package."""
        if module_name:
            script = f"import {module_name}; print('import ok')"
        else:
            script = "import torch; import torch_npu; print('torch_npu ok')"
        return self.run_command(f'python -c "{script}"')

    def run_syntax_check(self, file_paths: list[str] | None = None) -> RunResult:
        """Check Python files for syntax errors.

        When running via SSH, local absolute paths are converted to paths
        relative to the repo root so they resolve inside ``remote_workdir``.
        """
        if file_paths:
            targets = file_paths
        else:
            targets = [
                str(p) for p in self.repo_path.rglob("*.py")
                if ".bak" not in p.name and "__pycache__" not in str(p)
            ]

        if not targets:
            return RunResult(
                exit_code=0, stdout="no files", stderr="",
                error_signature="", duration_s=0.0, command="syntax_check",
            )

        # Convert local paths to repo-relative for remote execution
        repo_str = str(self.repo_path)
        relative_targets: list[str] = []
        for t in targets[:50]:
            if t.startswith(repo_str):
                relative_targets.append(t[len(repo_str):].lstrip("/"))
            else:
                relative_targets.append(t)

        cmd = "python -m py_compile " + " ".join(relative_targets)
        return self.run_command(cmd)

    def run_command_string(self, cmd_string: str) -> RunResult:
        """Run an arbitrary command string through the executor."""
        return self.run_command(cmd_string)

    def run_validation(self) -> RunResult | None:
        """Run the configured validation command, if any.

        Returns None if no validation command is configured.
        """
        if not self._validation_command:
            return None
        return self.run_command(self._validation_command)
