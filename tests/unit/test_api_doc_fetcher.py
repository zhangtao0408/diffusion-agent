"""Tests for api_doc_fetcher — version resolution, fetching, caching, and detection."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest

from diffusion_agent.tools.api_doc_fetcher import (
    detect_torch_npu_version,
    fetch_api_doc,
    resolve_branch,
)


# ── resolve_branch ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("2.8.0", "v2.8.0"),
        ("2.9.0", "v2.9.0"),
        ("2.8.0.post3", "v2.8.0"),
        ("2.8.0+cpu", "v2.8.0"),
    ],
)
def test_resolve_branch(version: str, expected: str) -> None:
    assert resolve_branch(version) == expected


# ── fetch_api_doc ───────────────────────────────────────────────────────────

SAMPLE_CONTENT = "# torch_npu API docs\nSample content for testing."


@patch("diffusion_agent.tools.api_doc_fetcher.urllib.request.urlopen")
def test_fetch_api_doc_downloads(mock_urlopen: MagicMock, tmp_path: Path) -> None:
    """First call downloads from GitHub and returns content."""
    resp = MagicMock()
    resp.read.return_value = SAMPLE_CONTENT.encode("utf-8")
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = resp

    result = fetch_api_doc("2.8.0", cache_dir=tmp_path)

    assert result == SAMPLE_CONTENT
    mock_urlopen.assert_called_once()
    url_arg = mock_urlopen.call_args[0][0]
    assert "v2.8.0" in url_arg


@patch("diffusion_agent.tools.api_doc_fetcher.urllib.request.urlopen")
def test_fetch_api_doc_caches(mock_urlopen: MagicMock, tmp_path: Path) -> None:
    """First call downloads, second call reads from cache (no network)."""
    resp = MagicMock()
    resp.read.return_value = SAMPLE_CONTENT.encode("utf-8")
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = resp

    # First call — downloads
    fetch_api_doc("2.8.0", cache_dir=tmp_path)
    assert (tmp_path / "v2.8.0.md").exists()

    # Second call — cache hit
    mock_urlopen.reset_mock()
    result = fetch_api_doc("2.8.0", cache_dir=tmp_path)
    assert result == SAMPLE_CONTENT
    mock_urlopen.assert_not_called()


def test_fetch_api_doc_cache_hit(tmp_path: Path) -> None:
    """Pre-populated cache is used without any network call."""
    cache_file = tmp_path / "v2.8.0.md"
    cache_file.write_text(SAMPLE_CONTENT, encoding="utf-8")

    with patch("diffusion_agent.tools.api_doc_fetcher.urllib.request.urlopen") as mock_urlopen:
        result = fetch_api_doc("2.8.0", cache_dir=tmp_path)
        assert result == SAMPLE_CONTENT
        mock_urlopen.assert_not_called()


@patch("diffusion_agent.tools.api_doc_fetcher.urllib.request.urlopen", side_effect=URLError("network down"))
def test_fetch_api_doc_network_error(mock_urlopen: MagicMock, tmp_path: Path) -> None:
    """Network errors propagate as URLError."""
    with pytest.raises(URLError):
        fetch_api_doc("2.8.0", cache_dir=tmp_path)


# ── detect_torch_npu_version ────────────────────────────────────────────────


@patch("diffusion_agent.tools.api_doc_fetcher.subprocess.run")
def test_detect_version_with_host(mock_run: MagicMock) -> None:
    """Successful SSH returns parsed version string."""
    mock_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="2.8.0\n", stderr=""
    )
    result = detect_torch_npu_version(ssh_host="root@server")
    assert result == "2.8.0"
    # Verify SSH command constructed correctly
    call_args = mock_run.call_args[0][0]
    assert "ssh" in call_args
    assert "root@server" in call_args


def test_detect_version_no_host() -> None:
    """No ssh_host returns None immediately."""
    assert detect_torch_npu_version(ssh_host=None) is None


@patch("diffusion_agent.tools.api_doc_fetcher.subprocess.run")
def test_detect_version_ssh_failure(mock_run: MagicMock) -> None:
    """SSH failure returns None."""
    mock_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="Connection refused"
    )
    assert detect_torch_npu_version(ssh_host="root@server") is None


@patch(
    "diffusion_agent.tools.api_doc_fetcher.subprocess.run",
    side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=30),
)
def test_detect_version_timeout(mock_run: MagicMock) -> None:
    """SSH timeout returns None."""
    assert detect_torch_npu_version(ssh_host="root@server") is None
