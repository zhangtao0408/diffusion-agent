"""Git operations for repository management."""

from __future__ import annotations

from pathlib import Path

from git import Repo

from diffusion_agent.utils.logging import get_logger

log = get_logger(__name__)


def clone_repo(url: str, dest: Path) -> Repo:
    log.info("cloning_repo", url=url, dest=str(dest))
    return Repo.clone_from(url, str(dest))


def init_repo(path: Path) -> Repo:
    log.info("initializing_repo", path=str(path))
    return Repo.init(str(path))


def open_repo(path: Path) -> Repo:
    return Repo(str(path))


def commit(path: Path, message: str) -> str:
    repo = open_repo(path)
    repo.git.add(A=True)
    if repo.is_dirty(untracked_files=True):
        c = repo.index.commit(message)
        log.info("committed", sha=c.hexsha[:8], message=message)
        return c.hexsha
    log.info("nothing_to_commit")
    return ""


def create_branch(path: Path, name: str) -> None:
    repo = open_repo(path)
    repo.git.checkout("-b", name)
    log.info("branch_created", name=name)
