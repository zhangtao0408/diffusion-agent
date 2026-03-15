"""Git memory — branch, commit, and rollback for the adaptation loop.

Each accepted iteration gets a meaningful commit. Failed iterations are
rolled back so the working tree stays clean for the next attempt.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from git import GitCommandError, Repo

from diffusion_agent.adapt.types import Hypothesis, Verdict
from diffusion_agent.utils.logging import get_logger

log = get_logger(__name__)


def reset_to_clean_main(repo_path: Path) -> None:
    """Thoroughly reset a target repo to pristine main-branch state.

    This removes all adapt branches, resets the working tree to main,
    and cleans up ``.diffusion_agent/`` state directory.  Used before
    E2E runs to prevent pollution from previous runs.
    """
    repo = Repo(str(repo_path))

    # 1. Determine main branch name (main or master)
    main_branch = "main"
    branch_names = [b.name for b in repo.branches]
    if "main" not in branch_names and "master" in branch_names:
        main_branch = "master"

    # 2. Switch to main branch
    try:
        repo.git.checkout(main_branch)
    except GitCommandError:
        log.warning("reset_checkout_failed", branch=main_branch)

    # 3. Hard reset to match the branch HEAD
    repo.git.reset("--hard", main_branch)

    # 4. Clean untracked files and directories
    repo.git.clean("-fd")

    # 5. Delete all adapt/* branches
    for branch in list(repo.branches):
        if branch.name.startswith("adapt/"):
            try:
                repo.git.branch("-D", branch.name)
                log.info("reset_branch_deleted", branch=branch.name)
            except GitCommandError:
                log.warning("reset_branch_delete_failed", branch=branch.name)

    # 6. Remove .diffusion_agent/ state directory
    state_dir = repo_path / ".diffusion_agent"
    if state_dir.exists():
        shutil.rmtree(state_dir)
        log.info("reset_state_dir_removed", path=str(state_dir))

    log.info("reset_to_clean_main_done", branch=main_branch)


class GitMemory:
    """Manages git operations for the adaptation loop."""

    def __init__(self, repo_path: Path, branch_prefix: str = "adapt") -> None:
        self.repo_path = repo_path
        self.branch_prefix = branch_prefix
        self._repo: Repo | None = None

    @property
    def repo(self) -> Repo:
        if self._repo is None:
            self._repo = Repo(str(self.repo_path))
        return self._repo

    def ensure_branch(self, model_name: str = "") -> str:
        """Create and switch to adaptation branch if not already on one."""
        suffix = model_name.replace("/", "-").replace(" ", "-") if model_name else "npu"
        branch_name = f"{self.branch_prefix}/{suffix}"

        try:
            current = self.repo.active_branch.name
        except TypeError:
            current = ""

        if current == branch_name:
            return branch_name

        try:
            self.repo.git.checkout("-b", branch_name)
            log.info("git_branch_created", branch=branch_name)
        except GitCommandError:
            # Branch already exists, just switch to it
            try:
                self.repo.git.checkout(branch_name)
                log.info("git_branch_switched", branch=branch_name)
            except GitCommandError:
                log.warning("git_branch_failed", branch=branch_name)
                return current or "HEAD"

        return branch_name

    def snapshot(self) -> str:
        """Take a snapshot of current HEAD for rollback."""
        return self.repo.head.commit.hexsha

    def commit_iteration(
        self,
        iteration: int,
        hypothesis: Hypothesis,
        verdict: Verdict,
    ) -> str:
        """Commit the current changes with a descriptive message."""
        self.repo.git.add(A=True)

        if not self.repo.is_dirty(untracked_files=True):
            log.info("git_nothing_to_commit", iteration=iteration)
            return ""

        message = (
            f"adapt(iter-{iteration}): {hypothesis.description}\n\n"
            f"Category: {hypothesis.category.value}\n"
            f"Verdict: {verdict.value}\n"
            f"Action: {hypothesis.proposed_action}\n"
            f"Files: {', '.join(hypothesis.target_files)}"
        )

        commit = self.repo.index.commit(message)
        sha = commit.hexsha
        log.info("git_committed", sha=sha[:8], iteration=iteration)
        return sha

    def rollback_to(self, sha: str) -> None:
        """Hard-reset to a previous snapshot, discarding all changes since."""
        self.repo.git.reset("--hard", sha)
        # Clean untracked files too
        self.repo.git.clean("-fd")
        log.info("git_rollback", target=sha[:8])

    def rollback_last_commit(self) -> None:
        """Undo the most recent commit, keeping changes unstaged."""
        self.repo.git.reset("HEAD~1")
        # Then discard the changes
        self.repo.git.checkout("--", ".")
        self.repo.git.clean("-fd")
        log.info("git_rollback_last")

    def get_changed_files(self, since_sha: str | None = None) -> list[str]:
        """List files changed since a given commit (or in working tree)."""
        if since_sha:
            diff = self.repo.git.diff("--name-only", since_sha)
        else:
            diff = self.repo.git.diff("--name-only")

        if not diff:
            return []
        return [f.strip() for f in diff.splitlines() if f.strip()]

    def has_changes(self) -> bool:
        """Whether the working tree has uncommitted changes."""
        return self.repo.is_dirty(untracked_files=True)
