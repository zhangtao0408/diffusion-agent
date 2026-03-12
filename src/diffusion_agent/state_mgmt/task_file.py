"""Read/write current-task.json for tracking the active task."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STATE_DIR = ".diffusion_agent"
TASK_FILE = "current-task.json"


def _task_path(base: Path) -> Path:
    return base / STATE_DIR / TASK_FILE


def read_current_task(base: Path) -> dict[str, Any] | None:
    path = _task_path(base)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_current_task(base: Path, task: dict[str, Any]) -> None:
    path = _task_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")


def clear_current_task(base: Path) -> None:
    path = _task_path(base)
    if path.exists():
        path.unlink()
