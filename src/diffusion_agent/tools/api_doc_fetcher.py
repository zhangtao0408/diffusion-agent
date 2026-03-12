"""Fetch and cache torch_npu API docs from the Ascend/pytorch GitHub repo."""

from __future__ import annotations

import subprocess
import urllib.request
from pathlib import Path
from urllib.error import URLError  # noqa: F401 – re-exported for callers

GITHUB_RAW_URL = (
    "https://raw.githubusercontent.com/Ascend/pytorch/{branch}/docs/api/torch_npu_apis.md"
)
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "diffusion_agent" / "api_docs"


def resolve_branch(torch_npu_version: str) -> str:
    """Map torch_npu version to Ascend/pytorch branch name.

    Examples:
        "2.8.0"       -> "v2.8.0"
        "2.8.0.post3" -> "v2.8.0"
        "2.8.0+cpu"   -> "v2.8.0"
    """
    base = torch_npu_version.split(".post")[0].split("+")[0]
    return f"v{base}"


def fetch_api_doc(version: str, cache_dir: Path | None = None) -> str:
    """Fetch torch_npu API docs for a given version, with local caching."""
    branch = resolve_branch(version)
    cache = (cache_dir or DEFAULT_CACHE_DIR) / f"{branch}.md"

    if cache.exists():
        return cache.read_text(encoding="utf-8")

    url = GITHUB_RAW_URL.format(branch=branch)
    with urllib.request.urlopen(url, timeout=30) as resp:
        content = resp.read().decode("utf-8")

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(content, encoding="utf-8")
    return content


def detect_torch_npu_version(
    ssh_host: str | None = None,
    conda_env: str | None = None,
) -> str | None:
    """Auto-detect torch_npu version from a remote NPU server via SSH."""
    if not ssh_host:
        return None

    cmd = 'python -c "import torch_npu; print(torch_npu.__version__)"'
    if conda_env:
        cmd = f"source /root/.bashrc && conda activate {conda_env} && {cmd}"

    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no", ssh_host, cmd],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None
