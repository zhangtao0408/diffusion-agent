"""Tests for the CI baseline runner (LTX-2, Wan2.2)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from diffusion_agent.tools.baseline_runner import (
    CI_BASELINES,
    compare_with_baselines,
    load_or_run_baseline,
    run_baseline_check,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_ci_baselines_has_ltx2(self) -> None:
        assert "ltx2" in CI_BASELINES

    def test_ci_baselines_has_wan22(self) -> None:
        assert "wan22" in CI_BASELINES

    def test_baselines_have_urls(self) -> None:
        for name, info in CI_BASELINES.items():
            assert "url" in info
            assert info["url"].startswith("https://")


# ---------------------------------------------------------------------------
# run_baseline_check
# ---------------------------------------------------------------------------

class TestRunBaselineCheck:
    @patch("diffusion_agent.tools.baseline_runner._clone_repo")
    @patch("diffusion_agent.tools.baseline_runner.scan_directory")
    @patch("diffusion_agent.tools.baseline_runner.check_pattern")
    def test_run_baseline_check_returns_dict(
        self, mock_check: MagicMock, mock_scan: MagicMock, mock_clone: MagicMock, tmp_path: Path
    ) -> None:
        mock_clone.return_value = tmp_path
        mock_scan.return_value = []
        mock_check.return_value = MagicMock(
            op_name="test", status=MagicMock(value="supported"), note=""
        )
        result = run_baseline_check("ltx2", version="2.8.0", cache_dir=tmp_path)
        assert isinstance(result, dict)
        assert "verdict" in result
        assert "baseline_name" in result

    @patch("diffusion_agent.tools.baseline_runner._clone_repo")
    @patch("diffusion_agent.tools.baseline_runner.scan_directory")
    def test_run_baseline_empty_repo(
        self, mock_scan: MagicMock, mock_clone: MagicMock, tmp_path: Path
    ) -> None:
        mock_clone.return_value = tmp_path
        mock_scan.return_value = []
        result = run_baseline_check("ltx2", version="2.8.0", cache_dir=tmp_path)
        assert result["verdict"] == "compatible"


# ---------------------------------------------------------------------------
# compare_with_baselines
# ---------------------------------------------------------------------------

class TestCompareWithBaselines:
    def test_compare_produces_summary(self) -> None:
        target = {
            "findings_by_type": {"cuda_call": 5, "flash_attn": 2},
            "verdict": "partially_compatible",
        }
        baselines = {
            "ltx2": {
                "findings_by_type": {"cuda_call": 10, "flash_attn": 5, "xformers": 3},
                "verdict": "partially_compatible",
            },
        }
        comparison = compare_with_baselines(target, baselines)
        assert "ltx2" in comparison
        assert "shared_patterns" in comparison["ltx2"]
        assert "target_only" in comparison["ltx2"]
        assert "baseline_only" in comparison["ltx2"]

    def test_compare_empty_baselines(self) -> None:
        target = {"findings_by_type": {"cuda_call": 5}, "verdict": "compatible"}
        comparison = compare_with_baselines(target, {})
        assert comparison == {}


# ---------------------------------------------------------------------------
# load_or_run_baseline
# ---------------------------------------------------------------------------

class TestLoadOrRunBaseline:
    def test_loads_from_cache(self, tmp_path: Path) -> None:
        import json
        cache_file = tmp_path / "ltx2_v2.8.0.json"
        cached_data = {"baseline_name": "ltx2", "verdict": "partially_compatible", "findings_by_type": {}}
        cache_file.write_text(json.dumps(cached_data))
        result = load_or_run_baseline("ltx2", "2.8.0", cache_dir=tmp_path)
        assert result["verdict"] == "partially_compatible"

    @patch("diffusion_agent.tools.baseline_runner.run_baseline_check")
    def test_runs_when_no_cache(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.return_value = {"baseline_name": "wan22", "verdict": "compatible", "findings_by_type": {}}
        result = load_or_run_baseline("wan22", "2.8.0", cache_dir=tmp_path)
        assert result["verdict"] == "compatible"
        mock_run.assert_called_once()
        # Check it was cached
        import json
        cached = json.loads((tmp_path / "wan22_v2.8.0.json").read_text())
        assert cached["verdict"] == "compatible"
