"""End-to-end integration tests for the check_support scenario through the full LangGraph pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from git import Repo

from diffusion_agent.agents.graph import build_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODEL_PY = """\
import torch

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        x = x.cuda()
        return self.linear(x)

if __name__ == "__main__":
    if torch.cuda.is_available():
        model = SimpleModel().cuda()
        x = torch.randn(1, 10).to("cuda")
        print(model(x))
"""

TRAIN_PY = """\
import torch
import torch.distributed as dist

def train():
    dist.init_process_group(backend="nccl")
    model = Model()
    model = model.to(torch.float64)
    model.cuda()
"""

UTILS_PY = """\
import os
def get_config():
    return {"lr": 0.001}
"""


def _init_git_repo(path: Path) -> Repo:
    """Initialize a git repo with an initial commit so the graph can commit on top."""
    repo = Repo.init(str(path))
    repo.config_writer().set_value("user", "name", "Test").release()
    repo.config_writer().set_value("user", "email", "test@test.com").release()
    # Need at least one commit for the branch to exist
    marker = path / ".gitkeep"
    marker.write_text("")
    repo.index.add([".gitkeep"])
    repo.index.commit("initial commit")
    return repo


def _write_file(path: Path, name: str, content: str) -> Path:
    p = path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def _run_graph(repo_path: Path, scenario: str = "check") -> dict:
    """Run the full LangGraph pipeline and return final state."""
    graph = build_graph()
    initial_state = {
        "repo_local_path": str(repo_path),
        "scenario": scenario,
        "model_name": "test-model",
        "phase": "init",
        "completed_features": [],
        "tool_results": [],
    }
    result = graph.invoke(initial_state)
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestE2ECheckScenario:
    """Full pipeline: init → coding (check_support) → done."""

    def test_e2e_check_scenario(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Run full graph on a repo with CUDA patterns — expect findings and report."""
        # Point work_dir to tmp_path so init_agent doesn't write outside it
        monkeypatch.setenv("DA_WORK_DIR", str(tmp_path))

        repo_path = tmp_path / "test-repo"
        repo_path.mkdir()
        _init_git_repo(repo_path)
        _write_file(repo_path, "model.py", MODEL_PY)
        _write_file(repo_path, "train.py", TRAIN_PY)
        _write_file(repo_path, "utils.py", UTILS_PY)

        result = _run_graph(repo_path)

        # Init phase created .diffusion_agent/ dir with state files
        state_dir = repo_path / ".diffusion_agent"
        assert state_dir.exists()
        assert (state_dir / "feature-list.yaml").exists()
        assert (state_dir / "daily-log.md").exists()
        assert (state_dir / "standing-rules.md").exists()

        # Coding phase produced check report
        reports_dir = state_dir / "reports"
        assert reports_dir.exists()
        assert (reports_dir / "check-report.json").exists()
        assert (reports_dir / "check-report.md").exists()

        # Report finds CUDA patterns from model.py and train.py
        report = json.loads((reports_dir / "check-report.json").read_text())
        assert report["summary_stats"]["total_findings"] > 0
        finding_files = {f["file_path"] for f in report["findings"]}
        # At least model.py and train.py should have findings
        model_found = any("model.py" in f for f in finding_files)
        train_found = any("train.py" in f for f in finding_files)
        assert model_found, f"Expected model.py findings, got: {finding_files}"
        assert train_found, f"Expected train.py findings, got: {finding_files}"

        # Verdict should be partially_compatible or incompatible (due to nccl, float64)
        assert report["verdict"] in ("partially_compatible", "incompatible")

        # Git commits were made (init commit + state file commit + feature commit)
        repo = Repo(str(repo_path))
        commits = list(repo.iter_commits())
        assert len(commits) >= 2  # at least initial + one from the graph

        # Phase should be done
        assert result.get("phase") == "done"

    def test_e2e_check_clean_repo(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Run on a repo with only clean Python (no CUDA) → compatible verdict."""
        monkeypatch.setenv("DA_WORK_DIR", str(tmp_path))

        repo_path = tmp_path / "clean-repo"
        repo_path.mkdir()
        _init_git_repo(repo_path)
        _write_file(repo_path, "utils.py", UTILS_PY)

        result = _run_graph(repo_path)

        reports_dir = repo_path / ".diffusion_agent" / "reports"
        assert (reports_dir / "check-report.json").exists()

        report = json.loads((reports_dir / "check-report.json").read_text())
        assert report["verdict"] == "compatible"
        assert report["summary_stats"]["total_findings"] == 0

    def test_e2e_state_files_created(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify feature-list.yaml, daily-log.md, standing-rules.md exist after run."""
        monkeypatch.setenv("DA_WORK_DIR", str(tmp_path))

        repo_path = tmp_path / "state-repo"
        repo_path.mkdir()
        _init_git_repo(repo_path)
        _write_file(repo_path, "utils.py", UTILS_PY)

        _run_graph(repo_path)

        state_dir = repo_path / ".diffusion_agent"
        assert (state_dir / "feature-list.yaml").exists()
        assert (state_dir / "daily-log.md").exists()
        assert (state_dir / "standing-rules.md").exists()
