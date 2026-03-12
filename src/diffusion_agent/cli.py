"""CLI entry point using Typer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from diffusion_agent.agents.graph import build_graph
from diffusion_agent.agents.state import AgentState
from diffusion_agent.config import load_settings
from diffusion_agent.scenarios import Scenario
from diffusion_agent.utils.logging import setup_logging

app = typer.Typer(
    name="diffusion-agent",
    help="Agent-integrated system for enabling PyTorch model support on Huawei Ascend NPU.",
)


@app.command()
def run(
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Git repository URL to clone"),
    local_path: Optional[str] = typer.Option(
        None, "--local-path", "-l", help="Local path to an existing repository"
    ),
    scenario: str = typer.Option(
        "check",
        "--scenario",
        "-s",
        help="Scenario to execute: check, adapt, analyze, optimize, verify",
    ),
    model_name: str = typer.Option(
        "unknown", "--model", "-m", help="Model name for identification"
    ),
) -> None:
    """Run the diffusion agent pipeline."""
    settings = load_settings()
    setup_logging(settings.log_level)

    # Validate scenario
    try:
        Scenario(scenario)
    except ValueError:
        typer.echo(f"Invalid scenario: {scenario}. Must be one of: {[s.value for s in Scenario]}")
        raise typer.Exit(1)

    # Build initial state
    initial_state: AgentState = {
        "scenario": scenario,
        "model_name": model_name,
        "phase": "init",
        "completed_features": [],
        "tool_results": [],
        "should_stop": False,
    }

    if repo:
        initial_state["repo_url"] = repo
    elif local_path:
        initial_state["repo_local_path"] = str(Path(local_path).resolve())
    else:
        typer.echo("Either --repo or --local-path is required.")
        raise typer.Exit(1)

    # Build and run the graph
    graph = build_graph()
    typer.echo(f"Starting diffusion-agent: scenario={scenario}, model={model_name}")

    result = graph.invoke(initial_state)

    typer.echo(f"\nCompleted. Phase: {result.get('phase', 'unknown')}")
    completed = result.get("completed_features", [])
    if completed:
        typer.echo(f"Features completed: {len(completed)}")
    if result.get("error"):
        typer.echo(f"Error: {result['error']}")


if __name__ == "__main__":
    app()
