"""Stage One: Initialization Agent.

Handles scaffolding, environment setup, Git initialization,
and creation of state tracking files. No business code is written here.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from diffusion_agent.agents.state import AgentState
from diffusion_agent.config import Settings
from diffusion_agent.state_mgmt.daily_log import append_log
from diffusion_agent.state_mgmt.feature_list import Feature, write_features
from diffusion_agent.state_mgmt.rules import write_rules
from diffusion_agent.state_mgmt.task_file import write_current_task
from diffusion_agent.tools.git_ops import clone_repo
from diffusion_agent.utils.fs import ensure_dir, safe_read
from diffusion_agent.utils.logging import get_logger

log = get_logger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "templates"


def _load_template(name: str) -> str:
    path = TEMPLATES_DIR / name
    content = safe_read(path)
    if content is None:
        raise FileNotFoundError(f"Template not found: {path}")
    return content


def init_node(state: AgentState) -> dict:
    """Stage One entry point. Clone repo, create state files, transition to coding."""
    settings = Settings()
    work_dir = settings.get_work_dir()
    repo_url = state.get("repo_url", "")
    model_name = state.get("model_name", "unknown")
    scenario = state.get("scenario", "check")

    # Determine repo local path
    if repo_url:
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        repo_local = work_dir / repo_name
        if repo_local.exists():
            shutil.rmtree(repo_local)
        clone_repo(repo_url, repo_local)
    else:
        repo_local = Path(state.get("repo_local_path", str(work_dir / "default")))
        ensure_dir(repo_local)

    # Create .diffusion_agent/ state directory
    state_dir = repo_local / ".diffusion_agent"
    ensure_dir(state_dir)

    # Write standing rules from template
    rules_content = _load_template("standing-rules.md")
    write_rules(repo_local, rules_content)

    # Write initial feature list (placeholder — scenarios will populate this properly)
    initial_features = [
        Feature(
            id=f"{scenario}-001",
            name=f"Initialize {scenario} scenario",
            description=f"Set up {scenario} scenario for model {model_name}",
            status="pending",
        )
    ]
    write_features(repo_local, initial_features)

    # Write initial current task
    write_current_task(repo_local, {
        "id": initial_features[0].id,
        "scenario": scenario,
        "feature_name": initial_features[0].name,
        "status": "pending",
        "attempt": 0,
    })

    # Log initialization
    append_log(repo_local, f"Initialized project for scenario={scenario}, model={model_name}")

    log.info(
        "init_complete",
        repo_local=str(repo_local),
        scenario=scenario,
        features=len(initial_features),
    )

    return {
        "phase": "coding",
        "repo_local_path": str(repo_local),
        "completed_features": [],
        "git_branch": "main",
        "error": None,
    }
