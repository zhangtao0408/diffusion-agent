"""Workspace sync — transfers patched local files to a remote workspace.

Ensures the remote validation target reflects the exact local patched state
before every validation run.  This is the key correctness property: the
Judge's verdict must correspond to the *current* local code, not stale
remote code.

Design:
  - WorkspaceSync: protocol for all sync backends
  - NoOpSync: used in local mode (nothing to sync)
  - RsyncSync: uses ``rsync`` over SSH for changed-file sync
  - SyncResult: structured metadata about what was synced

The Supervisor owns the sync lifecycle.  The Runner stays execution-focused.
"""

from __future__ import annotations

import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from diffusion_agent.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Sync result
# ---------------------------------------------------------------------------

@dataclass
class SyncResult:
    """Structured outcome of a workspace sync operation."""

    success: bool
    mode: str  # "noop", "rsync", "scp"
    files_synced: list[str] = field(default_factory=list)
    files_requested: int = 0
    local_root: str = ""
    remote_target: str = ""
    command: str = ""
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "mode": self.mode,
            "files_synced": self.files_synced,
            "files_requested": self.files_requested,
            "local_root": self.local_root,
            "remote_target": self.remote_target,
            "duration_s": self.duration_s,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Sync protocol
# ---------------------------------------------------------------------------

class WorkspaceSync(Protocol):
    """Protocol for workspace synchronization backends."""

    def sync(
        self,
        changed_files: list[str],
        local_root: Path,
    ) -> SyncResult:
        """Sync changed files from local to remote workspace.

        Args:
            changed_files: Relative paths (to local_root) of files to sync.
                           If empty, syncs the entire repo subtree.
            local_root: The local repo root directory.

        Returns:
            SyncResult with metadata about what was synced.
        """
        ...


# ---------------------------------------------------------------------------
# No-op sync (local mode)
# ---------------------------------------------------------------------------

class NoOpSync:
    """No synchronization needed — used when execution is local."""

    def sync(self, changed_files: list[str], local_root: Path) -> SyncResult:
        return SyncResult(
            success=True,
            mode="noop",
            files_synced=list(changed_files),
            files_requested=len(changed_files),
            local_root=str(local_root),
        )


# ---------------------------------------------------------------------------
# Rsync-over-SSH sync
# ---------------------------------------------------------------------------

@dataclass
class RsyncConfig:
    """Configuration for rsync-based remote sync."""

    host: str
    user: str = "root"
    port: int = 22
    remote_workdir: str = ""
    exclude_patterns: list[str] = field(default_factory=lambda: [
        "__pycache__",
        "*.pyc",
        ".git",
        "*.bak",
        ".diffusion_agent",
    ])
    timeout: int = 60
    delete: bool = False  # whether to --delete files on remote not present locally
    ssh_options: list[str] = field(default_factory=list)


class RsyncSync:
    """Syncs local files to a remote workspace via rsync over SSH.

    Two modes:
      1. **Changed-files mode**: when specific files are provided, uses
         ``--files-from`` to sync only those files.  This is fast and
         precise for per-iteration sync.
      2. **Full-tree mode**: when no specific files are given, syncs the
         entire local repo tree to the remote workdir.  Used for initial
         sync or batch-phase sync.
    """

    def __init__(self, config: RsyncConfig) -> None:
        self.config = config

    def _build_ssh_transport(self) -> str:
        """Build the -e 'ssh ...' string for rsync."""
        parts = ["ssh"]
        if self.config.port != 22:
            parts.append(f"-p {self.config.port}")
        parts.extend([
            "-o BatchMode=yes",
            "-o StrictHostKeyChecking=accept-new",
            "-o ConnectTimeout=10",
        ])
        for opt in self.config.ssh_options:
            parts.append(f"-o {opt}")
        return " ".join(parts)

    def _build_rsync_args(
        self,
        local_root: Path,
        files_from_path: str | None = None,
    ) -> list[str]:
        """Build rsync command args."""
        remote_target = (
            f"{self.config.user}@{self.config.host}:"
            f"{self.config.remote_workdir}"
        )

        args = [
            "rsync",
            "-az",                    # archive + compress
            "--partial",              # resume partial transfers
            "--timeout", str(self.config.timeout),
            "-e", self._build_ssh_transport(),
        ]

        # Exclude patterns
        for pat in self.config.exclude_patterns:
            args.extend(["--exclude", pat])

        # Delete mode (optional)
        if self.config.delete:
            args.append("--delete")

        if files_from_path:
            # Changed-files mode: sync only listed files, paths relative to local_root
            args.extend(["--files-from", files_from_path])
            # Source is the root dir (files-from are relative to this)
            args.append(str(local_root) + "/")
        else:
            # Full-tree mode
            args.append(str(local_root) + "/")

        args.append(remote_target + "/")
        return args

    def sync(
        self,
        changed_files: list[str],
        local_root: Path,
    ) -> SyncResult:
        """Sync files to remote workspace via rsync."""
        if not self.config.remote_workdir:
            return SyncResult(
                success=False,
                mode="rsync",
                files_requested=len(changed_files),
                local_root=str(local_root),
                error="No remote_workdir configured",
            )

        # Resolve relative paths: strip local_root prefix if present
        relative_files = _to_relative_paths(changed_files, local_root)

        # Filter to files that actually exist locally
        existing = [f for f in relative_files if (local_root / f).exists()]

        files_from_path: str | None = None
        tmp_file = None
        try:
            if existing:
                # Write file list for --files-from
                tmp_file = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False, prefix="sync_files_"
                )
                tmp_file.write("\n".join(existing) + "\n")
                tmp_file.flush()
                tmp_file.close()
                files_from_path = tmp_file.name

            args = self._build_rsync_args(local_root, files_from_path)
            cmd_str = " ".join(shlex.quote(a) for a in args)

            log.info(
                "sync_rsync",
                files=len(existing) if existing else "full-tree",
                remote=f"{self.config.user}@{self.config.host}:{self.config.remote_workdir}",
            )

            start = time.monotonic()
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=self.config.timeout + 30,  # extra buffer beyond rsync's own timeout
                check=False,
            )
            elapsed = time.monotonic() - start

            if result.returncode == 0:
                return SyncResult(
                    success=True,
                    mode="rsync",
                    files_synced=existing if existing else [],
                    files_requested=len(changed_files),
                    local_root=str(local_root),
                    remote_target=f"{self.config.user}@{self.config.host}:{self.config.remote_workdir}",
                    command=cmd_str,
                    stdout=result.stdout[-1000:],
                    stderr=result.stderr[-1000:],
                    duration_s=round(elapsed, 2),
                )
            else:
                error = _classify_rsync_error(result.returncode, result.stderr)
                return SyncResult(
                    success=False,
                    mode="rsync",
                    files_requested=len(changed_files),
                    local_root=str(local_root),
                    remote_target=f"{self.config.user}@{self.config.host}:{self.config.remote_workdir}",
                    command=cmd_str,
                    stdout=result.stdout[-1000:],
                    stderr=result.stderr[-1000:],
                    duration_s=round(elapsed, 2),
                    error=error,
                )
        except subprocess.TimeoutExpired:
            return SyncResult(
                success=False,
                mode="rsync",
                files_requested=len(changed_files),
                local_root=str(local_root),
                error=f"Rsync timed out after {self.config.timeout + 30}s",
            )
        except FileNotFoundError:
            return SyncResult(
                success=False,
                mode="rsync",
                files_requested=len(changed_files),
                local_root=str(local_root),
                error="rsync not found — install rsync or use a different sync mode",
            )
        except OSError as exc:
            return SyncResult(
                success=False,
                mode="rsync",
                files_requested=len(changed_files),
                local_root=str(local_root),
                error=f"Sync OS error: {exc}",
            )
        finally:
            if tmp_file is not None:
                try:
                    Path(tmp_file.name).unlink(missing_ok=True)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# SCP-based sync (fallback when rsync is unavailable)
# ---------------------------------------------------------------------------

class ScpSync:
    """Syncs files to remote via scp — fallback when rsync is not available.

    Less efficient than rsync (no delta transfer) but universally available.
    Syncs only the specific changed files, one scp invocation per batch.
    """

    def __init__(self, config: RsyncConfig) -> None:
        self.config = config

    def sync(
        self,
        changed_files: list[str],
        local_root: Path,
    ) -> SyncResult:
        if not self.config.remote_workdir:
            return SyncResult(
                success=False, mode="scp",
                error="No remote_workdir configured",
            )

        relative_files = _to_relative_paths(changed_files, local_root)
        existing = [f for f in relative_files if (local_root / f).exists()]

        if not existing:
            return SyncResult(
                success=True, mode="scp",
                files_synced=[], files_requested=0,
                local_root=str(local_root),
            )

        log.info("sync_scp", files=len(existing),
                 remote=f"{self.config.user}@{self.config.host}:{self.config.remote_workdir}")

        start = time.monotonic()
        errors: list[str] = []
        synced: list[str] = []

        # Ensure remote directories exist first
        remote_dirs = {str(Path(f).parent) for f in existing if "/" in f}
        if remote_dirs:
            mkdir_cmd = " && ".join(
                f"mkdir -p {shlex.quote(self.config.remote_workdir + '/' + d)}"
                for d in sorted(remote_dirs)
            )
            self._run_ssh_cmd(mkdir_cmd)

        # Copy files in batches via scp
        for f in existing:
            local_path = str(local_root / f)
            remote_path = (
                f"{self.config.user}@{self.config.host}:"
                f"{self.config.remote_workdir}/{f}"
            )
            try:
                args = self._build_scp_args(local_path, remote_path)
                result = subprocess.run(
                    args, capture_output=True, text=True,
                    timeout=self.config.timeout, check=False,
                )
                if result.returncode == 0:
                    synced.append(f)
                else:
                    errors.append(f"scp {f}: {result.stderr.strip()[:100]}")
            except subprocess.TimeoutExpired:
                errors.append(f"scp {f}: timed out")
            except OSError as exc:
                errors.append(f"scp {f}: {exc}")

        elapsed = time.monotonic() - start

        if errors:
            return SyncResult(
                success=False, mode="scp",
                files_synced=synced, files_requested=len(changed_files),
                local_root=str(local_root),
                remote_target=f"{self.config.user}@{self.config.host}:{self.config.remote_workdir}",
                duration_s=round(elapsed, 2),
                error="; ".join(errors[:5]),
            )

        return SyncResult(
            success=True, mode="scp",
            files_synced=synced, files_requested=len(changed_files),
            local_root=str(local_root),
            remote_target=f"{self.config.user}@{self.config.host}:{self.config.remote_workdir}",
            duration_s=round(elapsed, 2),
        )

    def _build_scp_args(self, local_path: str, remote_path: str) -> list[str]:
        args = ["scp", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
        if self.config.port != 22:
            args.extend(["-P", str(self.config.port)])
        args.extend([local_path, remote_path])
        return args

    def _run_ssh_cmd(self, cmd: str) -> None:
        """Run a command on the remote host via ssh (for mkdir etc.)."""
        args = ["ssh", "-o", "BatchMode=yes"]
        if self.config.port != 22:
            args.extend(["-p", str(self.config.port)])
        args.append(f"{self.config.user}@{self.config.host}")
        args.extend(["bash", "-c", cmd])
        try:
            subprocess.run(args, capture_output=True, timeout=30, check=False)
        except (subprocess.TimeoutExpired, OSError):
            pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_workspace_sync(
    mode: str,
    ssh_host: str | None = None,
    ssh_user: str = "root",
    ssh_port: int = 22,
    remote_workdir: str | None = None,
    exclude_patterns: list[str] | None = None,
    sync_timeout: int = 60,
    delete: bool = False,
    ssh_options: list[str] | None = None,
    prefer_scp: bool = False,
) -> WorkspaceSync:
    """Create the appropriate sync backend from parameters.

    When ``prefer_scp`` is True, uses ScpSync instead of RsyncSync
    (useful when the remote host doesn't have rsync installed).
    """
    if mode == "ssh" and ssh_host and remote_workdir:
        cfg = RsyncConfig(
            host=ssh_host,
            user=ssh_user,
            port=ssh_port,
            remote_workdir=remote_workdir,
            timeout=sync_timeout,
            delete=delete,
        )
        if exclude_patterns is not None:
            cfg.exclude_patterns = exclude_patterns
        if ssh_options is not None:
            cfg.ssh_options = ssh_options

        if prefer_scp:
            return ScpSync(cfg)
        return RsyncSync(cfg)

    return NoOpSync()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_relative_paths(paths: list[str], root: Path) -> list[str]:
    """Convert absolute paths to paths relative to root.  Already-relative
    paths are returned unchanged."""
    root_str = str(root)
    result: list[str] = []
    for p in paths:
        if p.startswith(root_str):
            rel = p[len(root_str):].lstrip("/")
            result.append(rel)
        else:
            result.append(p)
    return result


# Known rsync exit codes
_RSYNC_ERRORS: dict[int, str] = {
    1: "Syntax or usage error",
    2: "Protocol incompatibility",
    3: "Errors selecting input/output files",
    5: "Error starting client-server protocol",
    10: "Error in socket I/O",
    11: "Error in file I/O",
    12: "Error in rsync protocol data stream",
    23: "Partial transfer due to error",
    24: "Partial transfer due to vanished source files",
    25: "The --max-delete limit stopped deletions",
    30: "Timeout in data send/receive",
    35: "Timeout waiting for daemon connection",
    255: "SSH connection failed (check auth / host / port)",
}


def _classify_rsync_error(returncode: int, stderr: str) -> str:
    """Produce a human-readable error from rsync exit code + stderr."""
    known = _RSYNC_ERRORS.get(returncode)
    if known:
        return f"rsync error (code {returncode}): {known}"

    # Fall back to last line of stderr
    lines = [ln.strip() for ln in stderr.strip().splitlines() if ln.strip()]
    detail = lines[-1] if lines else "unknown error"
    return f"rsync failed (code {returncode}): {detail}"
