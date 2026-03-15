"""Tests for adapt/workspace_sync.py — workspace synchronization."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from diffusion_agent.adapt.workspace_sync import (
    NoOpSync,
    RsyncConfig,
    RsyncSync,
    SyncResult,
    _classify_rsync_error,
    _to_relative_paths,
    create_workspace_sync,
)


# ---------------------------------------------------------------------------
# SyncResult
# ---------------------------------------------------------------------------

class TestSyncResult:
    def test_success_result(self) -> None:
        r = SyncResult(success=True, mode="rsync", files_synced=["a.py"])
        assert r.success
        assert r.error is None

    def test_failure_result(self) -> None:
        r = SyncResult(success=False, mode="rsync", error="connection refused")
        assert not r.success
        assert "connection" in r.error

    def test_to_dict(self) -> None:
        r = SyncResult(
            success=True, mode="noop", files_synced=["x.py"],
            files_requested=1, local_root="/a", remote_target="host:/b",
            duration_s=0.5,
        )
        d = r.to_dict()
        assert d["mode"] == "noop"
        assert d["files_requested"] == 1
        assert d["success"] is True


# ---------------------------------------------------------------------------
# NoOpSync
# ---------------------------------------------------------------------------

class TestNoOpSync:
    def test_always_succeeds(self, tmp_path: Path) -> None:
        sync = NoOpSync()
        result = sync.sync(["a.py", "b.py"], tmp_path)
        assert result.success
        assert result.mode == "noop"
        assert result.files_synced == ["a.py", "b.py"]
        assert result.files_requested == 2

    def test_empty_files(self, tmp_path: Path) -> None:
        sync = NoOpSync()
        result = sync.sync([], tmp_path)
        assert result.success
        assert result.files_requested == 0


# ---------------------------------------------------------------------------
# RsyncConfig
# ---------------------------------------------------------------------------

class TestRsyncConfig:
    def test_defaults(self) -> None:
        cfg = RsyncConfig(host="server.com", remote_workdir="/data")
        assert cfg.user == "root"
        assert cfg.port == 22
        assert "__pycache__" in cfg.exclude_patterns
        assert cfg.delete is False

    def test_custom(self) -> None:
        cfg = RsyncConfig(
            host="h", user="u", port=2222, remote_workdir="/w",
            exclude_patterns=["*.log"], timeout=30, delete=True,
        )
        assert cfg.port == 2222
        assert cfg.delete is True
        assert cfg.exclude_patterns == ["*.log"]


# ---------------------------------------------------------------------------
# RsyncSync — command construction
# ---------------------------------------------------------------------------

class TestRsyncSyncBuild:
    def test_ssh_transport_default_port(self) -> None:
        cfg = RsyncConfig(host="h", remote_workdir="/data")
        sync = RsyncSync(cfg)
        transport = sync._build_ssh_transport()
        assert "ssh" in transport
        assert "-p" not in transport
        assert "BatchMode=yes" in transport

    def test_ssh_transport_custom_port(self) -> None:
        cfg = RsyncConfig(host="h", port=2222, remote_workdir="/data")
        sync = RsyncSync(cfg)
        transport = sync._build_ssh_transport()
        assert "-p 2222" in transport

    def test_rsync_args_full_tree(self) -> None:
        cfg = RsyncConfig(host="h", user="u", remote_workdir="/data/repo")
        sync = RsyncSync(cfg)
        args = sync._build_rsync_args(Path("/local/repo"))
        assert "rsync" in args
        assert "-az" in args
        assert any("/local/repo/" in a for a in args)
        assert any("u@h:/data/repo/" in a for a in args)
        assert "--files-from" not in args

    def test_rsync_args_files_from(self, tmp_path: Path) -> None:
        cfg = RsyncConfig(host="h", remote_workdir="/data")
        sync = RsyncSync(cfg)
        filelist = str(tmp_path / "files.txt")
        (tmp_path / "files.txt").write_text("a.py\n")
        args = sync._build_rsync_args(Path("/local"), files_from_path=filelist)
        assert "--files-from" in args
        idx = args.index("--files-from")
        assert args[idx + 1] == filelist

    def test_rsync_args_excludes(self) -> None:
        cfg = RsyncConfig(host="h", remote_workdir="/r", exclude_patterns=["*.pyc", ".git"])
        sync = RsyncSync(cfg)
        args = sync._build_rsync_args(Path("/l"))
        assert args.count("--exclude") == 2

    def test_rsync_args_delete(self) -> None:
        cfg = RsyncConfig(host="h", remote_workdir="/r", delete=True)
        sync = RsyncSync(cfg)
        args = sync._build_rsync_args(Path("/l"))
        assert "--delete" in args

    def test_rsync_args_no_delete(self) -> None:
        cfg = RsyncConfig(host="h", remote_workdir="/r", delete=False)
        sync = RsyncSync(cfg)
        args = sync._build_rsync_args(Path("/l"))
        assert "--delete" not in args


# ---------------------------------------------------------------------------
# RsyncSync — execute (mocked subprocess)
# ---------------------------------------------------------------------------

class TestRsyncSyncExecute:
    def test_success_changed_files(self, tmp_path: Path) -> None:
        (tmp_path / "model.py").write_text("x = 1\n")
        cfg = RsyncConfig(host="h", remote_workdir="/data")
        sync = RsyncSync(cfg)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "sent 100 bytes\n"
        mock_result.stderr = ""
        with patch("diffusion_agent.adapt.workspace_sync.subprocess.run", return_value=mock_result):
            result = sync.sync(["model.py"], tmp_path)

        assert result.success
        assert result.mode == "rsync"
        assert result.files_synced == ["model.py"]
        assert result.files_requested == 1

    def test_success_full_tree(self, tmp_path: Path) -> None:
        cfg = RsyncConfig(host="h", remote_workdir="/data")
        sync = RsyncSync(cfg)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        with patch("diffusion_agent.adapt.workspace_sync.subprocess.run", return_value=mock_result):
            result = sync.sync([], tmp_path)

        assert result.success
        assert result.files_synced == []
        assert result.files_requested == 0

    def test_rsync_failure(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x = 1\n")
        cfg = RsyncConfig(host="h", remote_workdir="/data")
        sync = RsyncSync(cfg)

        mock_result = MagicMock()
        mock_result.returncode = 255
        mock_result.stdout = ""
        mock_result.stderr = "ssh: connect to host h port 22: Connection refused\n"
        with patch("diffusion_agent.adapt.workspace_sync.subprocess.run", return_value=mock_result):
            result = sync.sync(["a.py"], tmp_path)

        assert not result.success
        assert "255" in result.error
        assert "SSH" in result.error

    def test_timeout(self, tmp_path: Path) -> None:
        cfg = RsyncConfig(host="h", remote_workdir="/data", timeout=5)
        sync = RsyncSync(cfg)

        with patch(
            "diffusion_agent.adapt.workspace_sync.subprocess.run",
            side_effect=subprocess.TimeoutExpired("rsync", 35),
        ):
            result = sync.sync(["a.py"], tmp_path)

        assert not result.success
        assert "timed out" in result.error.lower()

    def test_rsync_not_found(self, tmp_path: Path) -> None:
        cfg = RsyncConfig(host="h", remote_workdir="/data")
        sync = RsyncSync(cfg)

        with patch(
            "diffusion_agent.adapt.workspace_sync.subprocess.run",
            side_effect=FileNotFoundError("rsync"),
        ):
            result = sync.sync(["a.py"], tmp_path)

        assert not result.success
        assert "rsync not found" in result.error

    def test_no_remote_workdir(self, tmp_path: Path) -> None:
        cfg = RsyncConfig(host="h", remote_workdir="")
        sync = RsyncSync(cfg)
        result = sync.sync(["a.py"], tmp_path)
        assert not result.success
        assert "No remote_workdir" in result.error

    def test_absolute_paths_converted(self, tmp_path: Path) -> None:
        """Absolute file paths are converted to relative before sync."""
        (tmp_path / "sub" / "model.py").parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "sub" / "model.py").write_text("x = 1\n")
        cfg = RsyncConfig(host="h", remote_workdir="/data")
        sync = RsyncSync(cfg)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        with patch("diffusion_agent.adapt.workspace_sync.subprocess.run", return_value=mock_result):
            result = sync.sync([str(tmp_path / "sub" / "model.py")], tmp_path)

        assert result.success
        assert result.files_synced == ["sub/model.py"]

    def test_os_error(self, tmp_path: Path) -> None:
        cfg = RsyncConfig(host="h", remote_workdir="/data")
        sync = RsyncSync(cfg)

        with patch(
            "diffusion_agent.adapt.workspace_sync.subprocess.run",
            side_effect=OSError("disk full"),
        ):
            result = sync.sync(["a.py"], tmp_path)

        assert not result.success
        assert "OS error" in result.error


# ---------------------------------------------------------------------------
# create_workspace_sync factory
# ---------------------------------------------------------------------------

class TestCreateWorkspaceSync:
    def test_local_mode(self) -> None:
        sync = create_workspace_sync(mode="local")
        assert isinstance(sync, NoOpSync)

    def test_ssh_mode_with_workdir(self) -> None:
        sync = create_workspace_sync(
            mode="ssh", ssh_host="h", remote_workdir="/data",
            ssh_user="u", ssh_port=2222,
        )
        assert isinstance(sync, RsyncSync)
        assert sync.config.host == "h"
        assert sync.config.port == 2222

    def test_ssh_mode_no_host(self) -> None:
        sync = create_workspace_sync(mode="ssh")
        assert isinstance(sync, NoOpSync)

    def test_ssh_mode_no_workdir(self) -> None:
        sync = create_workspace_sync(mode="ssh", ssh_host="h")
        assert isinstance(sync, NoOpSync)

    def test_custom_excludes(self) -> None:
        sync = create_workspace_sync(
            mode="ssh", ssh_host="h", remote_workdir="/data",
            exclude_patterns=["*.log"],
        )
        assert isinstance(sync, RsyncSync)
        assert sync.config.exclude_patterns == ["*.log"]

    def test_delete_flag(self) -> None:
        sync = create_workspace_sync(
            mode="ssh", ssh_host="h", remote_workdir="/d", delete=True,
        )
        assert isinstance(sync, RsyncSync)
        assert sync.config.delete is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestToRelativePaths:
    def test_absolute_to_relative(self) -> None:
        result = _to_relative_paths(["/repo/a.py", "/repo/sub/b.py"], Path("/repo"))
        assert result == ["a.py", "sub/b.py"]

    def test_already_relative(self) -> None:
        result = _to_relative_paths(["a.py", "sub/b.py"], Path("/repo"))
        assert result == ["a.py", "sub/b.py"]

    def test_mixed(self) -> None:
        result = _to_relative_paths(["/repo/a.py", "b.py"], Path("/repo"))
        assert result == ["a.py", "b.py"]


class TestClassifyRsyncError:
    def test_known_code(self) -> None:
        err = _classify_rsync_error(255, "some ssh error")
        assert "255" in err
        assert "SSH connection failed" in err

    def test_unknown_code(self) -> None:
        err = _classify_rsync_error(99, "weird\nerror detail")
        assert "99" in err
        assert "error detail" in err

    def test_partial_transfer(self) -> None:
        err = _classify_rsync_error(23, "")
        assert "Partial transfer" in err
