"""Tests for the adapt scenario — CUDA→NPU migration orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from diffusion_agent.scenarios.adapt import AdaptReport, AdaptScenario, _render_npu_readme


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def _make_state(tmp_path: Path) -> dict:
    return {
        "repo_local_path": str(tmp_path),
        "scenario": "adapt",
        "tool_results": [],
    }


# ---------------------------------------------------------------------------
# plan()
# ---------------------------------------------------------------------------

class TestPlan:
    def test_plan_returns_features(self) -> None:
        scenario = AdaptScenario()
        state = _make_state(Path("/tmp/fake"))
        features = scenario.plan(state)
        assert isinstance(features, list)
        assert len(features) >= 1
        assert features[0]["name"] == "adapt_for_npu"


# ---------------------------------------------------------------------------
# execute()
# ---------------------------------------------------------------------------

class TestExecute:
    def test_execute_produces_report(self, tmp_path: Path) -> None:
        code = 'import torch\nx = tensor.cuda()\nif torch.cuda.is_available():\n    pass\n'
        _write(tmp_path, "model.py", code)
        state = _make_state(tmp_path)
        scenario = AdaptScenario()
        result_state = scenario.execute(state)

        assert "tool_results" in result_state
        assert len(result_state["tool_results"]) > 0
        report = result_state["tool_results"][-1]
        assert "verdict" in report
        assert "total_files_modified" in report
        assert "migration_results" in report

        # Check report files written
        reports_dir = tmp_path / ".diffusion_agent" / "reports"
        assert (reports_dir / "adapt-report.json").exists()
        assert (reports_dir / "adapt-report.md").exists()

    def test_execute_replaces_cuda(self, tmp_path: Path) -> None:
        code = 'import torch\nx = tensor.cuda()\nmodel.to("cuda")\n'
        _write(tmp_path, "model.py", code)
        state = _make_state(tmp_path)
        AdaptScenario().execute(state)

        content = (tmp_path / "model.py").read_text()
        assert ".cuda()" not in content
        assert ".npu()" in content
        assert '"npu"' in content

    def test_execute_adds_torch_npu(self, tmp_path: Path) -> None:
        code = "import torch\nx = tensor.cuda()\n"
        _write(tmp_path, "model.py", code)
        state = _make_state(tmp_path)
        AdaptScenario().execute(state)

        content = (tmp_path / "model.py").read_text()
        assert "import torch_npu" in content

    def test_execute_empty_repo(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        result_state = AdaptScenario().execute(state)
        report = result_state["tool_results"][-1]
        assert report["verdict"] == "success"
        assert report["total_files_modified"] == 0

    def test_report_includes_llm_stats(self, tmp_path: Path) -> None:
        _write(tmp_path, "a.py", "import torch\nx = tensor.cuda()\n")
        state = _make_state(tmp_path)
        result_state = AdaptScenario().execute(state)
        report = result_state["tool_results"][-1]
        assert "llm_fixes_applied" in report
        assert "llm_fixes_skipped" in report
        assert isinstance(report["llm_fixes_applied"], int)
        assert isinstance(report["llm_fixes_skipped"], int)


# ---------------------------------------------------------------------------
# AdaptReport dataclass
# ---------------------------------------------------------------------------

class TestAdaptReport:
    def test_to_dict(self) -> None:
        report = AdaptReport(
            model_name="test",
            repo_local_path="/tmp/test",
            verdict="success",
            total_files_modified=1,
            total_migrations_applied=2,
            migration_results=[],
            llm_fixes_applied=0,
            llm_fixes_skipped=0,
            skipped_patterns=[],
            recommendations=["Run check after adapt"],
        )
        d = report.to_dict()
        assert d["verdict"] == "success"
        assert d["model_name"] == "test"
        assert d["llm_fixes_applied"] == 0
        assert isinstance(d["recommendations"], list)


# ---------------------------------------------------------------------------
# Report JSON validity
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# NPU Adaptation README
# ---------------------------------------------------------------------------

class TestNpuReadme:
    def test_readme_generated(self, tmp_path: Path) -> None:
        code = "import torch\nx = tensor.cuda()\n"
        _write(tmp_path, "model.py", code)
        state = _make_state(tmp_path)
        AdaptScenario().execute(state)
        assert (tmp_path / "NPU_ADAPTATION_README.md").exists()

    def test_readme_contains_changed_files(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")
        state = _make_state(tmp_path)
        AdaptScenario().execute(state)
        readme = (tmp_path / "NPU_ADAPTATION_README.md").read_text()
        assert "model.py" in readme

    def test_readme_contains_run_instructions(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")
        state = _make_state(tmp_path)
        AdaptScenario().execute(state)
        readme = (tmp_path / "NPU_ADAPTATION_README.md").read_text()
        assert "## How to Run" in readme

    def test_render_npu_readme_empty_report(self) -> None:
        report = AdaptReport(
            model_name="empty",
            repo_local_path="/tmp/empty",
            verdict="success",
            total_files_modified=0,
            total_migrations_applied=0,
            migration_results=[],
            llm_fixes_applied=0,
            llm_fixes_skipped=0,
            skipped_patterns=[],
            recommendations=[],
        )
        result = _render_npu_readme(report)
        assert "## What Changed" in result
        assert "## How to Run" in result
        assert "## Next Steps" in result


# ---------------------------------------------------------------------------
# Report JSON validity
# ---------------------------------------------------------------------------

class TestReportFiles:
    def test_report_json_valid(self, tmp_path: Path) -> None:
        _write(tmp_path, "a.py", "import torch\nx = tensor.cuda()\n")
        state = _make_state(tmp_path)
        AdaptScenario().execute(state)
        json_path = tmp_path / ".diffusion_agent" / "reports" / "adapt-report.json"
        data = json.loads(json_path.read_text())
        assert "verdict" in data
        assert "migration_results" in data
