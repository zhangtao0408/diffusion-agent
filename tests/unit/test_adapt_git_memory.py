"""Tests for adapt/git_memory.py — git branch/commit/rollback."""

from __future__ import annotations

from pathlib import Path

from git import Repo

from diffusion_agent.adapt.git_memory import GitMemory
from diffusion_agent.adapt.types import FailureCategory, Hypothesis, Verdict


def _init_repo(tmp_path: Path) -> Repo:
    repo = Repo.init(str(tmp_path))
    (tmp_path / "init.txt").write_text("init\n")
    repo.index.add(["init.txt"])
    repo.index.commit("initial")
    return repo


def _make_hypothesis() -> Hypothesis:
    return Hypothesis(
        id="h-1", category=FailureCategory.DEVICE_SELECTION,
        description="Fix cuda calls", target_files=["model.py"],
        proposed_action="Apply cuda_call rule",
    )


class TestGitMemory:
    def test_ensure_branch(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        gm = GitMemory(tmp_path)
        branch = gm.ensure_branch("test-model")
        assert branch == "adapt/test-model"
        assert gm.repo.active_branch.name == "adapt/test-model"

    def test_ensure_branch_already_on(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        gm = GitMemory(tmp_path)
        gm.ensure_branch("test")
        branch = gm.ensure_branch("test")  # should be idempotent
        assert branch == "adapt/test"

    def test_snapshot_and_rollback(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        gm = GitMemory(tmp_path)
        sha = gm.snapshot()

        # Make a change
        (tmp_path / "new.txt").write_text("new\n")
        gm.repo.index.add(["new.txt"])
        gm.repo.index.commit("added new")

        assert (tmp_path / "new.txt").exists()
        gm.rollback_to(sha)
        assert not (tmp_path / "new.txt").exists()

    def test_commit_iteration(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        gm = GitMemory(tmp_path)

        (tmp_path / "model.py").write_text("modified\n")
        sha = gm.commit_iteration(1, _make_hypothesis(), Verdict.IMPROVED)
        assert sha != ""
        assert len(sha) == 40  # full sha

        # Check commit message
        msg = gm.repo.head.commit.message
        assert "adapt(iter-1)" in msg
        assert "Fix cuda calls" in msg

    def test_commit_nothing_to_commit(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        gm = GitMemory(tmp_path)
        sha = gm.commit_iteration(1, _make_hypothesis(), Verdict.UNCHANGED)
        assert sha == ""

    def test_has_changes(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        gm = GitMemory(tmp_path)
        assert gm.has_changes() is False

        (tmp_path / "new.txt").write_text("new\n")
        assert gm.has_changes() is True

    def test_get_changed_files(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        gm = GitMemory(tmp_path)
        sha = gm.snapshot()

        (tmp_path / "model.py").write_text("new\n")
        repo.index.add(["model.py"])
        repo.index.commit("add model")

        changed = gm.get_changed_files(sha)
        assert "model.py" in changed

    def test_rollback_last_commit(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        gm = GitMemory(tmp_path)

        (tmp_path / "test.py").write_text("x = 1\n")
        gm.repo.index.add(["test.py"])
        gm.repo.index.commit("added test")
        assert (tmp_path / "test.py").exists()

        gm.rollback_last_commit()
        assert not (tmp_path / "test.py").exists()
