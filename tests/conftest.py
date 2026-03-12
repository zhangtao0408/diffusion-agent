"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Create a temporary directory simulating a repo with .diffusion_agent/ state dir."""
    state_dir = tmp_path / ".diffusion_agent"
    state_dir.mkdir()
    return tmp_path
