"""Tests for the check_support scenario — orchestrates scanner + checker → report."""

from __future__ import annotations

import json
from pathlib import Path

from diffusion_agent.scenarios.check_support import CheckReport, CheckSupportScenario


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def _make_state(tmp_path: Path) -> dict:
    """Minimal AgentState dict for tests."""
    return {
        "repo_local_path": str(tmp_path),
        "scenario": "check",
        "tool_results": [],
    }


# ---------------------------------------------------------------------------
# plan()
# ---------------------------------------------------------------------------

class TestPlan:
    def test_plan_returns_features(self) -> None:
        scenario = CheckSupportScenario()
        state = _make_state(Path("/tmp/fake"))
        features = scenario.plan(state)
        assert isinstance(features, list)
        assert len(features) >= 1


# ---------------------------------------------------------------------------
# execute() — basic flows
# ---------------------------------------------------------------------------

class TestExecuteBasic:
    def test_execute_with_cuda_code(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.py", "x = tensor.cuda()\n")
        state = _make_state(tmp_path)
        scenario = CheckSupportScenario()
        result_state = scenario.execute(state)
        assert "tool_results" in result_state
        assert len(result_state["tool_results"]) > 0
        report_data = result_state["tool_results"][-1]
        assert "verdict" in report_data

    def test_execute_empty_repo(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        scenario = CheckSupportScenario()
        result_state = scenario.execute(state)
        report_data = result_state["tool_results"][-1]
        assert report_data["verdict"] == "compatible"


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

class TestVerdictLogic:
    def test_verdict_compatible(self, tmp_path: Path) -> None:
        """No findings → compatible."""
        state = _make_state(tmp_path)
        scenario = CheckSupportScenario()
        result_state = scenario.execute(state)
        report_data = result_state["tool_results"][-1]
        assert report_data["verdict"] == "compatible"

    def test_verdict_partially_compatible(self, tmp_path: Path) -> None:
        """Mixed patterns → partially_compatible."""
        # cuda_call is "partial", float64 is "unsupported" — mix
        code = (
            "x = tensor.cuda()\n"
            "y = x.to(torch.float64)\n"
        )
        _write(tmp_path, "mixed.py", code)
        state = _make_state(tmp_path)
        scenario = CheckSupportScenario()
        result_state = scenario.execute(state)
        report_data = result_state["tool_results"][-1]
        assert report_data["verdict"] in ("partially_compatible", "incompatible")

    def test_verdict_incompatible(self, tmp_path: Path) -> None:
        """Only unsupported patterns → incompatible."""
        # float64 is unsupported in the support matrix
        code = "y = x.to(torch.float64)\n"
        _write(tmp_path, "bad.py", code)
        state = _make_state(tmp_path)
        scenario = CheckSupportScenario()
        result_state = scenario.execute(state)
        report_data = result_state["tool_results"][-1]
        assert report_data["verdict"] == "incompatible"


# ---------------------------------------------------------------------------
# Report file outputs
# ---------------------------------------------------------------------------

class TestReportFiles:
    def test_report_json_written(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.py", "x = tensor.cuda()\n")
        da_dir = tmp_path / ".diffusion_agent" / "reports"
        state = _make_state(tmp_path)
        scenario = CheckSupportScenario()
        scenario.execute(state)
        assert (da_dir / "check-report.json").exists()

    def test_report_md_written(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.py", "x = tensor.cuda()\n")
        da_dir = tmp_path / ".diffusion_agent" / "reports"
        state = _make_state(tmp_path)
        scenario = CheckSupportScenario()
        scenario.execute(state)
        assert (da_dir / "check-report.md").exists()

    def test_report_json_valid(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.py", "x = tensor.cuda()\n")
        state = _make_state(tmp_path)
        scenario = CheckSupportScenario()
        scenario.execute(state)
        json_path = tmp_path / ".diffusion_agent" / "reports" / "check-report.json"
        data = json.loads(json_path.read_text())
        assert "verdict" in data
        assert "findings" in data
        assert "compatibility_results" in data


# ---------------------------------------------------------------------------
# Report content
# ---------------------------------------------------------------------------

class TestReportContent:
    def test_report_has_recommendations(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.py", "x = tensor.cuda()\n")
        state = _make_state(tmp_path)
        scenario = CheckSupportScenario()
        result_state = scenario.execute(state)
        report_data = result_state["tool_results"][-1]
        assert "recommendations" in report_data
        assert isinstance(report_data["recommendations"], list)

    def test_report_summary_stats(self, tmp_path: Path) -> None:
        code = (
            "x = tensor.cuda()\n"
            "y = x.to(torch.float64)\n"
        )
        _write(tmp_path, "model.py", code)
        state = _make_state(tmp_path)
        scenario = CheckSupportScenario()
        result_state = scenario.execute(state)
        report_data = result_state["tool_results"][-1]
        stats = report_data["summary_stats"]
        assert "total_findings" in stats
        assert "by_pattern_type" in stats
        assert "by_compatibility" in stats
        assert stats["total_findings"] >= 2

    def test_empty_repo_has_zero_stats(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        scenario = CheckSupportScenario()
        result_state = scenario.execute(state)
        report_data = result_state["tool_results"][-1]
        assert report_data["summary_stats"]["total_findings"] == 0


# ---------------------------------------------------------------------------
# CheckReport dataclass
# ---------------------------------------------------------------------------

class TestCheckReport:
    def test_to_dict(self) -> None:
        report = CheckReport(
            verdict="compatible",
            findings=[],
            compatibility_results=[],
            summary_stats={"total_findings": 0, "by_pattern_type": {}, "by_compatibility": {}},
            recommendations=[],
        )
        d = report.to_dict()
        assert d["verdict"] == "compatible"
        assert isinstance(d["findings"], list)
        assert isinstance(d["summary_stats"], dict)
