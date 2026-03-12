"""Append-only daily log for tracking agent activity."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

STATE_DIR = ".diffusion_agent"
LOG_FILE = "daily-log.md"


def _log_path(base: Path) -> Path:
    return base / STATE_DIR / LOG_FILE


def append_log(base: Path, entry: str) -> None:
    path = _log_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"- [{timestamp}] {entry}\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def read_log(base: Path) -> str | None:
    path = _log_path(base)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")
