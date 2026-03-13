"""Git memory — branch, commit, and rollback for the adaptation loop.

Each accepted iteration gets a meaningful commit. Failed iterations are
rolled back so the working tree stays clean for the next attempt.
"""

from __future__ import annotations

from pathlib import Path

from git import GitCommandError, Repo

from diffusion_agent.adapt.types import Hypothesis, Verdict
from diffusion_agent.utils.logging import get_logger

log = get_logger(__name__)


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
