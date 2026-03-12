"""CRUD operations on feature-list.yaml for task tracking."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

STATE_DIR = ".diffusion_agent"
FEATURE_FILE = "feature-list.yaml"


@dataclass
class Feature:
    id: str
    name: str
    description: str
    status: str = "pending"  # pending | in_progress | completed | failed
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _feature_path(base: Path) -> Path:
    return base / STATE_DIR / FEATURE_FILE


def read_features(base: Path) -> list[Feature]:
    path = _feature_path(base)
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not data or "features" not in data:
        return []
    return [Feature(**f) for f in data["features"]]


def write_features(base: Path, features: list[Feature]) -> None:
    path = _feature_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"features": [asdict(f) for f in features]}
    path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True), encoding="utf-8")


def update_feature_status(
    base: Path, feature_id: str, status: str, error: str | None = None
) -> None:
    features = read_features(base)
    for f in features:
        if f.id == feature_id:
            f.status = status
            f.error = error
            break
    write_features(base, features)


def get_next_pending(base: Path) -> Feature | None:
    features = read_features(base)
    for f in features:
        if f.status == "pending":
            return f
    return None
