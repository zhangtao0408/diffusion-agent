"""Torch NPU op compatibility checker — looks up ops against bundled support matrix."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class OpStatus(Enum):
    """Compatibility status of a PyTorch op on Ascend NPU."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


@dataclass
class CheckResult:
    """Result of checking a single op or pattern against the support matrix."""

    op_name: str
    status: OpStatus
    note: str


_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_MATRIX_CACHE: dict | None = None


def load_op_matrix() -> dict:
    """Load the bundled op support matrix JSON, caching after first read."""
    global _MATRIX_CACHE  # noqa: PLW0603
    if _MATRIX_CACHE is None:
        path = _DATA_DIR / "op_support.json"
        _MATRIX_CACHE = json.loads(path.read_text(encoding="utf-8"))
    return _MATRIX_CACHE


def _status_from_str(s: str) -> OpStatus:
    try:
        return OpStatus(s)
    except ValueError:
        return OpStatus.UNKNOWN


def check_op(name: str) -> CheckResult:
    """Look up a single op name in the support matrix."""
    matrix = load_op_matrix()
    entry = matrix.get("ops", {}).get(name)
    if entry is None:
        return CheckResult(op_name=name, status=OpStatus.UNKNOWN, note="Op not found in support matrix")
    return CheckResult(
        op_name=name,
        status=_status_from_str(entry["status"]),
        note=entry.get("note", ""),
    )


def check_ops(names: list[str]) -> list[CheckResult]:
    """Batch lookup for multiple op names."""
    return [check_op(n) for n in names]


def check_pattern(pattern_type: str) -> CheckResult:
    """Check a code pattern type (e.g. 'cuda_call', 'float64') against the matrix."""
    matrix = load_op_matrix()
    entry = matrix.get("patterns", {}).get(pattern_type)
    if entry is None:
        return CheckResult(op_name=pattern_type, status=OpStatus.UNKNOWN, note="Pattern not found in support matrix")
    return CheckResult(
        op_name=pattern_type,
        status=_status_from_str(entry["status"]),
        note=entry.get("note", ""),
    )


def get_compatibility_summary(results: list[CheckResult]) -> dict:
    """Summarize a list of CheckResults into counts by status."""
    summary = {
        "total": len(results),
        "supported": 0,
        "unsupported": 0,
        "partial": 0,
        "unknown": 0,
    }
    for r in results:
        key = r.status.value
        if key in summary:
            summary[key] += 1
    return summary
