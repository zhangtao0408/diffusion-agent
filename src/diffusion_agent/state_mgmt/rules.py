"""Read standing rules that constrain agent behavior."""

from __future__ import annotations

from pathlib import Path

STATE_DIR = ".diffusion_agent"
RULES_FILE = "standing-rules.md"


def _rules_path(base: Path) -> Path:
    return base / STATE_DIR / RULES_FILE


def read_rules(base: Path) -> str | None:
    path = _rules_path(base)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def write_rules(base: Path, content: str) -> None:
    path = _rules_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
