"""Torch NPU op compatibility checker — looks up ops against support matrix.

Supports two modes:
- **Static** (default): Uses bundled ``op_support.json`` (42 hand-curated entries).
- **Dynamic**: Fetches the real API docs from Ascend/pytorch for a given
  ``torch_npu`` version, yielding 3 000+ ops.  Pass ``version`` to any
  public function to activate dynamic mode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from diffusion_agent.utils.logging import get_logger

log = get_logger(__name__)


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
_MATRIX_CACHE: dict[str | None, dict] = {}


def _load_static_matrix() -> dict:
    """Load the bundled op_support.json."""
    path = _DATA_DIR / "op_support.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_dynamic_matrix(version: str) -> dict:
    """Fetch + parse the real API doc for *version*, merge static patterns."""
    from diffusion_agent.tools.api_doc_fetcher import fetch_api_doc
    from diffusion_agent.tools.api_doc_parser import build_op_matrix, parse_api_doc

    log.info("fetching_api_doc", version=version)
    content = fetch_api_doc(version)
    entries = parse_api_doc(content)
    matrix = build_op_matrix(entries)
    log.info("parsed_api_doc", op_count=len(matrix.get("ops", {})))

    # Merge patterns from static file (API doc doesn't cover code-level patterns)
    static = _load_static_matrix()
    matrix["patterns"] = static.get("patterns", {})

    # Also merge static ops that have "partial"/"unsupported" status into dynamic
    # matrix — these are curated migration hints (e.g., torch.cuda.amp -> torch_npu)
    for op_name, op_info in static.get("ops", {}).items():
        if op_info.get("status") in ("partial", "unsupported"):
            if op_name not in matrix["ops"]:
                matrix["ops"][op_name] = op_info

    return matrix


def load_op_matrix(version: str | None = None) -> dict:
    """Load the op support matrix, optionally for a specific torch_npu version.

    When *version* is provided, fetches the real API docs from the
    Ascend/pytorch GitHub repo and parses 3 000+ ops.  Falls back to
    the bundled ``op_support.json`` on network error or when no version
    is given.
    """
    if version in _MATRIX_CACHE:
        return _MATRIX_CACHE[version]

    if version is not None:
        try:
            matrix = _load_dynamic_matrix(version)
            _MATRIX_CACHE[version] = matrix
            return matrix
        except Exception:
            log.warning("dynamic_matrix_failed", version=version, fallback="static")
            # fall through to static

    if None not in _MATRIX_CACHE:
        _MATRIX_CACHE[None] = _load_static_matrix()
    return _MATRIX_CACHE[None]


def _status_from_str(s: str) -> OpStatus:
    try:
        return OpStatus(s)
    except ValueError:
        return OpStatus.UNKNOWN


def check_op(name: str, *, version: str | None = None) -> CheckResult:
    """Look up a single op name in the support matrix."""
    matrix = load_op_matrix(version)
    entry = matrix.get("ops", {}).get(name)
    if entry is None:
        return CheckResult(
            op_name=name, status=OpStatus.UNKNOWN, note="Op not found in support matrix"
        )
    return CheckResult(
        op_name=name,
        status=_status_from_str(entry["status"]),
        note=entry.get("note", ""),
    )


def check_ops(names: list[str], *, version: str | None = None) -> list[CheckResult]:
    """Batch lookup for multiple op names."""
    return [check_op(n, version=version) for n in names]


def check_pattern(
    pattern_type: str, *, version: str | None = None
) -> CheckResult:
    """Check a code pattern type (e.g. 'cuda_call', 'float64') against the matrix."""
    matrix = load_op_matrix(version)
    entry = matrix.get("patterns", {}).get(pattern_type)
    if entry is None:
        return CheckResult(
            op_name=pattern_type,
            status=OpStatus.UNKNOWN,
            note="Pattern not found in support matrix",
        )
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
