"""SSH-based runtime verification of ops on Ascend NPU hardware."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass

from diffusion_agent.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class VerifyResult:
    """Result of verifying a single op on real NPU hardware."""

    op_name: str
    passed: bool
    error: str | None = None
    execution_time_ms: float | None = None


# Basic op verification snippets — each must print "OK" on success
_BASIC_OPS: dict[str, str] = {
    "matmul": (
        "import torch, torch_npu; "
        "a = torch.randn(4, 4).npu(); b = torch.randn(4, 4).npu(); "
        "c = torch.matmul(a, b); print('OK')"
    ),
    "conv2d": (
        "import torch, torch_npu; "
        "m = torch.nn.Conv2d(3, 16, 3).npu(); "
        "x = torch.randn(1, 3, 8, 8).npu(); "
        "y = m(x); print('OK')"
    ),
    "linear": (
        "import torch, torch_npu; "
        "m = torch.nn.Linear(32, 16).npu(); "
        "x = torch.randn(2, 32).npu(); "
        "y = m(x); print('OK')"
    ),
    "layernorm": (
        "import torch, torch_npu; "
        "m = torch.nn.LayerNorm(32).npu(); "
        "x = torch.randn(2, 32).npu(); "
        "y = m(x); print('OK')"
    ),
    "attention": (
        "import torch, torch_npu; "
        "q = torch.randn(1, 4, 8, 16, dtype=torch.float16).npu(); "
        "k = torch.randn(1, 4, 8, 16, dtype=torch.float16).npu(); "
        "v = torch.randn(1, 4, 8, 16, dtype=torch.float16).npu(); "
        "o = torch.nn.functional.scaled_dot_product_attention(q, k, v); print('OK')"
    ),
    "bfloat16": (
        "import torch, torch_npu; "
        "x = torch.randn(4, 4, dtype=torch.bfloat16).npu(); "
        "y = x + x; print('OK')"
    ),
}


def _build_ssh_command(ssh_host: str, conda_env: str | None, python_code: str) -> list[str]:
    """Build an SSH command that runs Python code on the remote host."""
    cmd = f'python -c "{python_code}"'
    if conda_env:
        cmd = f"source /root/.bashrc && conda activate {conda_env} && {cmd}"
    return [
        "ssh",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=no",
        ssh_host,
        cmd,
    ]


def verify_op(
    ssh_host: str,
    conda_env: str | None,
    op_code: str,
    *,
    op_name: str = "custom",
    timeout: int = 60,
) -> VerifyResult:
    """Execute a Python snippet on the NPU server and return pass/fail."""
    ssh_cmd = _build_ssh_command(ssh_host, conda_env, op_code)
    start = time.monotonic()

    try:
        result = subprocess.run(
            ssh_cmd, capture_output=True, text=True, timeout=timeout,
        )
        elapsed_ms = (time.monotonic() - start) * 1000

        if result.returncode == 0:
            return VerifyResult(
                op_name=op_name, passed=True, execution_time_ms=elapsed_ms,
            )
        return VerifyResult(
            op_name=op_name,
            passed=False,
            error=result.stderr.strip() or result.stdout.strip(),
            execution_time_ms=elapsed_ms,
        )
    except subprocess.TimeoutExpired:
        return VerifyResult(
            op_name=op_name, passed=False, error=f"Timeout after {timeout}s",
        )
    except FileNotFoundError:
        return VerifyResult(
            op_name=op_name, passed=False, error="SSH client not found",
        )


def verify_import(
    ssh_host: str, conda_env: str | None, module: str,
) -> bool:
    """Check if a Python module is importable on the remote NPU server."""
    code = f"import {module}; print('OK')"
    result = verify_op(ssh_host, conda_env, code, op_name=f"import_{module}")
    return result.passed


def run_basic_checks(
    ssh_host: str, conda_env: str | None,
) -> list[VerifyResult]:
    """Run a suite of basic op checks on the NPU server."""
    results: list[VerifyResult] = []
    for op_name, code in _BASIC_OPS.items():
        log.info("verifying_op", op=op_name, host=ssh_host)
        result = verify_op(ssh_host, conda_env, code, op_name=op_name)
        results.append(result)
        log.info("verify_result", op=op_name, passed=result.passed, error=result.error)
    return results
