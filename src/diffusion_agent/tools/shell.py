"""Safe subprocess runner with timeout and error handling."""

from __future__ import annotations

import subprocess
from pathlib import Path

from diffusion_agent.utils.logging import get_logger

log = get_logger(__name__)


def run_command(
    cmd: list[str],
    cwd: Path | None = None,
    timeout: int = 120,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    log.info("running_command", cmd=" ".join(cmd), cwd=str(cwd))
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )
    if result.returncode != 0:
        log.warning(
            "command_failed",
            cmd=" ".join(cmd),
            returncode=result.returncode,
            stderr=result.stderr[:500],
        )
    return result
