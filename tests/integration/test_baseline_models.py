"""Integration tests for CI baseline models (LTX-2, Wan2.2).

These tests clone real repos and are slow. Run with:
    pytest tests/integration/test_baseline_models.py -v --run-slow
"""

from __future__ import annotations

import pytest

from diffusion_agent.tools.baseline_runner import (
    compare_with_baselines,
    run_baseline_check,
)


slow = pytest.mark.skipif(
    "not config.getoption('--run-slow', default=False)",
    reason="Slow test: clones real repos. Use --run-slow to run.",
)


def pytest_addoption(parser):
    """Add --run-slow CLI option for pytest."""
    try:
        parser.addoption("--run-slow", action="store_true", default=False)
    except ValueError:
        pass  # already added


@slow
class TestLtx2Baseline:
    def test_ltx2_check(self, tmp_path):
        result = run_baseline_check("ltx2", version="2.8.0", cache_dir=tmp_path)
        assert result["baseline_name"] == "ltx2"
        assert result["verdict"] in ("compatible", "partially_compatible", "incompatible")
        assert result["total_findings"] > 0
        # LTX-2 should have CUDA patterns
        ftypes = result["findings_by_type"]
        assert any(t in ftypes for t in ("cuda_call", "cuda_api", "cuda_to"))


@slow
class TestWan22Baseline:
    def test_wan22_check(self, tmp_path):
        result = run_baseline_check("wan22", version="2.8.0", cache_dir=tmp_path)
        assert result["baseline_name"] == "wan22"
        assert result["verdict"] in ("compatible", "partially_compatible", "incompatible")
        assert result["total_findings"] > 0
        ftypes = result["findings_by_type"]
        # Wan2.2 should have distributed/NCCL patterns
        assert any(t in ftypes for t in ("nccl", "distributed", "cuda_call"))


@slow
class TestBaselineComparison:
    def test_compare_two_baselines(self, tmp_path):
        ltx2 = run_baseline_check("ltx2", version="2.8.0", cache_dir=tmp_path)
        wan22 = run_baseline_check("wan22", version="2.8.0", cache_dir=tmp_path)
        comparison = compare_with_baselines(ltx2, {"wan22": wan22})
        assert "wan22" in comparison
        assert "shared_patterns" in comparison["wan22"]
