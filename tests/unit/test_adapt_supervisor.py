"""Tests for adapt/supervisor.py — the orchestration loop."""

from __future__ import annotations

from pathlib import Path

from unittest.mock import MagicMock

from diffusion_agent.adapt.supervisor import AdaptSupervisor
from diffusion_agent.adapt.types import ExecutionConfig, StopReason
from diffusion_agent.adapt.workspace_sync import NoOpSync, SyncResult


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


class TestSupervisorBasic:
    def test_empty_repo(self, tmp_path: Path) -> None:
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        state = sup.run()
        assert state.stop_reason == StopReason.ALL_RULES_APPLIED
        assert len(state.files_modified) == 0
        assert state.iteration == 0

    def test_single_cuda_call(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        state = sup.run()

        assert len(state.files_modified) >= 1
        assert state.total_rules_applied >= 1

        content = (tmp_path / "model.py").read_text()
        assert ".npu()" in content
        assert "import torch_npu" in content

    def test_multiple_patterns(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.py", 'import torch\nx = tensor.cuda()\ny = model.to("cuda")\n')
        _write(tmp_path, "train.py", 'import torch\ntorch.cuda.set_device(0)\n')

        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        state = sup.run()

        assert len(state.files_modified) >= 2
        assert state.total_rules_applied >= 3  # cuda_call + cuda_to + cuda_api

    def test_flash_attn_handled(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.py", "import torch\nfrom flash_attn import flash_attn_func\n")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        state = sup.run()

        assert len(state.files_modified) >= 1
        content = (tmp_path / "model.py").read_text()
        assert "# [NPU]" in content


class TestSupervisorTaskDecomposition:
    def test_creates_tasks(self, tmp_path: Path) -> None:
        _write(tmp_path, "a.py", "import torch\nx = tensor.cuda()\n")
        _write(tmp_path, "b.py", 'import torch\ny = "nccl"\n')

        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        state = sup.run()

        assert len(state.tasks) >= 2

    def test_tasks_completed_after_batch(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        state = sup.run()

        completed = [t for t in state.tasks if t.status == "completed"]
        assert len(completed) >= 1


class TestSupervisorStopConditions:
    def test_max_iterations(self, tmp_path: Path) -> None:
        # With only deterministic rules, we should complete without hitting max
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False, max_iterations=1)
        state = sup.run()
        assert state.stop_reason is not None

    def test_no_findings_stops_immediately(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.py", "import os\nprint('hello')\n")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        state = sup.run()
        assert state.stop_reason == StopReason.ALL_RULES_APPLIED
        assert state.iteration == 0


class TestSupervisorBlockerReport:
    def test_empty_blocker_report(self, tmp_path: Path) -> None:
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        sup.run()
        report = sup.get_blocker_report()
        assert "blockers" in report
        assert report["blockers"] == []
        assert report["model_name"] == "test"

    def test_blocker_report_structure(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        sup.run()
        report = sup.get_blocker_report()
        assert "total_iterations" in report
        assert "files_modified" in report
        assert "total_rules_applied" in report
        assert "iteration_summary" in report


class TestSupervisorWithGit:
    def test_with_git_repo(self, tmp_path: Path) -> None:
        from git import Repo
        repo = Repo.init(str(tmp_path))
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")
        repo.index.add(["model.py"])
        repo.index.commit("initial")

        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=True)
        state = sup.run()

        assert len(state.files_modified) >= 1

        # Should have created commits
        commits = list(repo.iter_commits())
        assert len(commits) >= 2  # initial + batch

    def test_creates_adaptation_branch(self, tmp_path: Path) -> None:
        from git import Repo
        repo = Repo.init(str(tmp_path))
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")
        repo.index.add(["model.py"])
        repo.index.commit("initial")

        sup = AdaptSupervisor(tmp_path, model_name="my-model", use_git=True)
        sup.run()

        assert repo.active_branch.name == "adapt/my-model"


class TestSupervisorSync:
    def test_noop_sync_by_default(self, tmp_path: Path) -> None:
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False)
        assert isinstance(sup.sync, NoOpSync)

    def test_injected_sync(self, tmp_path: Path) -> None:
        mock_sync = MagicMock()
        mock_sync.sync.return_value = SyncResult(success=True, mode="mock")
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False, workspace_sync=mock_sync)
        assert sup.sync is mock_sync

    def test_sync_called_on_batch_changes(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.py", "import torch\nx = tensor.cuda()\n")
        mock_sync = MagicMock()
        mock_sync.sync.return_value = SyncResult(success=True, mode="mock")

        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False, workspace_sync=mock_sync)
        sup.run()

        # sync should have been called at least once (for batch changes)
        assert mock_sync.sync.called
        # First call should include the changed file
        call_args = mock_sync.sync.call_args_list[0]
        changed_files = call_args[0][0]  # first positional arg
        assert any("model.py" in f for f in changed_files)

    def test_sync_not_called_when_no_changes(self, tmp_path: Path) -> None:
        _write(tmp_path, "model.py", "import os\nprint('hello')\n")
        mock_sync = MagicMock()
        mock_sync.sync.return_value = SyncResult(success=True, mode="mock")

        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False, workspace_sync=mock_sync)
        sup.run()

        # No CUDA patterns → no patches → no sync
        mock_sync.sync.assert_not_called()

    def test_sync_from_ssh_execution_config(self, tmp_path: Path) -> None:
        cfg = ExecutionConfig(
            mode="ssh",
            ssh_host="h.example.com",
            remote_workdir="/data/repo",
            sync_enabled=True,
        )
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False, execution_config=cfg)
        # Should have created an RsyncSync, not NoOpSync
        from diffusion_agent.adapt.workspace_sync import RsyncSync
        assert isinstance(sup.sync, RsyncSync)
        assert sup.sync.config.host == "h.example.com"
        assert sup.sync.config.remote_workdir == "/data/repo"

    def test_sync_disabled_in_config(self, tmp_path: Path) -> None:
        cfg = ExecutionConfig(
            mode="ssh",
            ssh_host="h.example.com",
            remote_workdir="/data/repo",
            sync_enabled=False,
        )
        sup = AdaptSupervisor(tmp_path, model_name="test", use_git=False, execution_config=cfg)
        assert isinstance(sup.sync, NoOpSync)
