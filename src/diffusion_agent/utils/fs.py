"""File system utility helpers."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_write(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def safe_read(path: Path) -> str | None:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None
