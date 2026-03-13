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
    npu_version: Optional[str] = typer.Option(
        None,
        "--npu-version",
        help="torch_npu version to use for op support lookup (e.g. 2.8.0). "
        "Auto-detected from NPU server if DA_NPU_SSH_HOST is set.",
    ),
    run_baselines: bool = typer.Option(
        False,
        "--run-baselines",
        help="Also run CI baseline checks (LTX-2, Wan2.2) for comparison.",
    ),
    runtime_verify: bool = typer.Option(
        False,
        "--runtime-verify",
        help="Run runtime verification of critical ops on the NPU server.",
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

    # Resolve torch_npu version (explicit > auto-detect > None)
    resolved_version = npu_version
    if resolved_version is None and settings.npu_ssh_host:
        from diffusion_agent.tools.api_doc_fetcher import detect_torch_npu_version

        detected = detect_torch_npu_version(settings.npu_ssh_host, settings.npu_conda_env)
        if detected:
            typer.echo(f"Auto-detected torch_npu version: {detected}")
            resolved_version = detected

    # Build initial state
    initial_state: AgentState = {
        "scenario": scenario,
        "model_name": model_name,
        "phase": "init",
        "completed_features": [],
        "tool_results": [],
        "should_stop": False,
    }

    if resolved_version:
        initial_state["torch_npu_version"] = resolved_version

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

    # Post-pipeline: runtime verification
    if runtime_verify and scenario == "check" and settings.npu_ssh_host:
        from diffusion_agent.tools.npu_verifier import run_basic_checks

        typer.echo("\nRunning NPU runtime verification...")
        runtime_results = run_basic_checks(settings.npu_ssh_host, settings.npu_conda_env)
        passed = sum(1 for r in runtime_results if r.passed)
        typer.echo(f"Runtime verification: {passed}/{len(runtime_results)} ops passed")

    # Post-pipeline: baseline comparison
    if run_baselines and scenario == "check" and resolved_version:
        from diffusion_agent.tools.baseline_runner import CI_BASELINES, load_or_run_baseline

        typer.echo("\nRunning CI baseline checks...")
        for bname in CI_BASELINES:
            typer.echo(f"  Checking baseline: {bname}...")
            baseline_result = load_or_run_baseline(bname, resolved_version)
            typer.echo(f"  {bname}: {baseline_result.get('verdict', 'unknown')} "
                       f"({baseline_result.get('total_findings', 0)} findings)")

    typer.echo(f"\nCompleted. Phase: {result.get('phase', 'unknown')}")
    completed = result.get("completed_features", [])
    if completed:
        typer.echo(f"Features completed: {len(completed)}")
    if result.get("error"):
        typer.echo(f"Error: {result['error']}")

    # Exit code based on verdict (for check scenario)
    if scenario == "check":
        tool_results = result.get("tool_results", [])
        if tool_results:
            verdict = tool_results[-1].get("verdict", "compatible")
            if verdict == "incompatible":
                raise typer.Exit(2)
            elif verdict == "partially_compatible":
                raise typer.Exit(1)


if __name__ == "__main__":
    app()
