"""Tests for torch_npu_checker — op compatibility checking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from diffusion_agent.tools.torch_npu_checker import (
    CheckResult,
    OpStatus,
    check_op,
    check_ops,
    check_pattern,
    get_compatibility_summary,
    load_op_matrix,
)


# ---------------------------------------------------------------------------
# OpStatus enum
# ---------------------------------------------------------------------------

class TestOpStatus:
    def test_has_supported(self):
        assert OpStatus.SUPPORTED.value == "supported"

    def test_has_unsupported(self):
        assert OpStatus.UNSUPPORTED.value == "unsupported"

    def test_has_partial(self):
        assert OpStatus.PARTIAL.value == "partial"

    def test_has_unknown(self):
        assert OpStatus.UNKNOWN.value == "unknown"


# ---------------------------------------------------------------------------
# CheckResult dataclass
# ---------------------------------------------------------------------------

class TestCheckResult:
    def test_fields(self):
        r = CheckResult(op_name="torch.nn.Conv2d", status=OpStatus.SUPPORTED, note="Full support")
        assert r.op_name == "torch.nn.Conv2d"
        assert r.status is OpStatus.SUPPORTED
        assert r.note == "Full support"

    def test_equality(self):
        a = CheckResult("op1", OpStatus.SUPPORTED, "note")
        b = CheckResult("op1", OpStatus.SUPPORTED, "note")
        assert a == b


# ---------------------------------------------------------------------------
# load_op_matrix
# ---------------------------------------------------------------------------

class TestLoadOpMatrix:
    def test_returns_dict(self):
        matrix = load_op_matrix()
        assert isinstance(matrix, dict)

    def test_has_ops_key(self):
        matrix = load_op_matrix()
        assert "ops" in matrix

    def test_has_patterns_key(self):
        matrix = load_op_matrix()
        assert "patterns" in matrix

    def test_ops_not_empty(self):
        matrix = load_op_matrix()
        assert len(matrix["ops"]) > 0

    def test_each_op_has_status_and_note(self):
        matrix = load_op_matrix()
        for name, info in matrix["ops"].items():
            assert "status" in info, f"Op {name} missing 'status'"
            assert "note" in info, f"Op {name} missing 'note'"

    def test_bundled_json_exists(self):
        json_path = Path(__file__).resolve().parents[2] / "src" / "diffusion_agent" / "data" / "op_support.json"
        assert json_path.exists(), f"Bundled JSON not found at {json_path}"


# ---------------------------------------------------------------------------
# check_op — single lookup
# ---------------------------------------------------------------------------

class TestCheckOp:
    def test_supported_op(self):
        result = check_op("torch.nn.Conv2d")
        assert result.status is OpStatus.SUPPORTED
        assert result.op_name == "torch.nn.Conv2d"

    def test_unsupported_op(self):
        result = check_op("torch.distributed.nccl")
        assert result.status is OpStatus.UNSUPPORTED

    def test_partial_op(self):
        result = check_op("torch.cuda.amp.autocast")
        assert result.status is OpStatus.PARTIAL

    def test_unknown_op(self):
        result = check_op("torch.nonexistent.FakeOp")
        assert result.status is OpStatus.UNKNOWN
        assert result.op_name == "torch.nonexistent.FakeOp"

    def test_unknown_op_note(self):
        result = check_op("torch.nonexistent.FakeOp")
        assert "unknown" in result.note.lower() or "not found" in result.note.lower()


# ---------------------------------------------------------------------------
# check_ops — batch lookup
# ---------------------------------------------------------------------------

class TestCheckOps:
    def test_returns_list(self):
        results = check_ops(["torch.nn.Conv2d", "torch.nn.Linear"])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_preserves_order(self):
        names = ["torch.nn.Linear", "torch.nn.Conv2d", "torch.distributed.nccl"]
        results = check_ops(names)
        assert [r.op_name for r in results] == names

    def test_empty_list(self):
        results = check_ops([])
        assert results == []

    def test_mixed_statuses(self):
        names = ["torch.nn.Conv2d", "torch.distributed.nccl", "torch.nonexistent.Foo"]
        results = check_ops(names)
        statuses = {r.op_name: r.status for r in results}
        assert statuses["torch.nn.Conv2d"] is OpStatus.SUPPORTED
        assert statuses["torch.distributed.nccl"] is OpStatus.UNSUPPORTED
        assert statuses["torch.nonexistent.Foo"] is OpStatus.UNKNOWN


# ---------------------------------------------------------------------------
# check_pattern
# ---------------------------------------------------------------------------

class TestCheckPattern:
    def test_known_pattern(self):
        result = check_pattern("cuda_call")
        assert result.status is not OpStatus.UNKNOWN

    def test_float64_pattern(self):
        result = check_pattern("float64")
        assert result.status is OpStatus.UNSUPPORTED

    def test_unknown_pattern(self):
        result = check_pattern("totally_unknown_pattern")
        assert result.status is OpStatus.UNKNOWN

    def test_returns_check_result(self):
        result = check_pattern("cuda_call")
        assert isinstance(result, CheckResult)


# ---------------------------------------------------------------------------
# get_compatibility_summary
# ---------------------------------------------------------------------------

class TestGetCompatibilitySummary:
    def test_empty_results(self):
        summary = get_compatibility_summary([])
        assert summary["total"] == 0
        assert summary["supported"] == 0

    def test_counts_each_status(self):
        results = [
            CheckResult("a", OpStatus.SUPPORTED, ""),
            CheckResult("b", OpStatus.SUPPORTED, ""),
            CheckResult("c", OpStatus.UNSUPPORTED, ""),
            CheckResult("d", OpStatus.PARTIAL, ""),
            CheckResult("e", OpStatus.UNKNOWN, ""),
        ]
        summary = get_compatibility_summary(results)
        assert summary["total"] == 5
        assert summary["supported"] == 2
        assert summary["unsupported"] == 1
        assert summary["partial"] == 1
        assert summary["unknown"] == 1

    def test_all_supported(self):
        results = [CheckResult("x", OpStatus.SUPPORTED, "") for _ in range(3)]
        summary = get_compatibility_summary(results)
        assert summary["supported"] == 3
        assert summary["unsupported"] == 0

    def test_has_all_keys(self):
        summary = get_compatibility_summary([])
        for key in ("total", "supported", "unsupported", "partial", "unknown"):
            assert key in summary
