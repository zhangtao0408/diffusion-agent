"""Adapt scenario — supervised CUDA→NPU migration with an internal agent loop.

This scenario delegates all adaptation work to the AdaptSupervisor, which
orchestrates: planner → patch_worker → runner → judge → git_memory.

The scenario itself is responsible for:
  1. Feature decomposition (plan)
  2. Launching the supervisor (execute)
  3. Building the final report from supervisor state
  4. Writing report files and NPU README
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from diffusion_agent.adapt.supervisor import AdaptSupervisor
from diffusion_agent.adapt.types import AdaptationState, ExecutionConfig
from diffusion_agent.scenarios import ScenarioBase
from diffusion_agent.tools.code_migrator import MigrationResult
from diffusion_agent.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Report dataclass (preserved for backward compatibility)
# ---------------------------------------------------------------------------

@dataclass
class AdaptReport:
    model_name: str
    repo_local_path: str
    verdict: str  # "success" | "partial" | "failed"
    total_files_modified: int
    total_migrations_applied: int
    migration_results: list[MigrationResult]
    llm_fixes_applied: int
    llm_fixes_skipped: int
    skipped_patterns: list[str]
    recommendations: list[str]
    check_report_before: dict | None = None
    # New fields from supervisor
    iterations_count: int = 0
    blockers: list[str] | None = None
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "repo_local_path": self.repo_local_path,
            "verdict": self.verdict,
            "total_files_modified": self.total_files_modified,
            "total_migrations_applied": self.total_migrations_applied,
            "migration_results": [asdict(r) for r in self.migration_results],
            "llm_fixes_applied": self.llm_fixes_applied,
            "llm_fixes_skipped": self.llm_fixes_skipped,
            "skipped_patterns": self.skipped_patterns,
            "recommendations": self.recommendations,
            "check_report_before": self.check_report_before,
            "iterations_count": self.iterations_count,
            "blockers": self.blockers or [],
            "stop_reason": self.stop_reason,
        }


# ---------------------------------------------------------------------------
# Markdown renderers
# ---------------------------------------------------------------------------

def _render_adapt_markdown(report: AdaptReport) -> str:
    """Render AdaptReport as human-readable Markdown."""
    lines = [
        "# Ascend NPU Adaptation Report",
        "",
        f"**Model**: {report.model_name}",
        f"**Local Path**: {report.repo_local_path}",
        f"**Verdict**: {report.verdict}",
        "",
        "## Summary",
        "",
        f"- Files modified: {report.total_files_modified}",
        f"- Migrations applied: {report.total_migrations_applied}",
        f"- LLM fixes applied: {report.llm_fixes_applied}",
        f"- LLM fixes skipped (low confidence): {report.llm_fixes_skipped}",
        f"- Adaptation iterations: {report.iterations_count}",
    ]

    if report.stop_reason:
        lines.append(f"- Stop reason: {report.stop_reason}")

    if report.blockers:
        lines.extend(["", "## Blockers", ""])
        for b in report.blockers:
            lines.append(f"- {b}")

    if report.skipped_patterns:
        lines.extend(["", "## Unresolved Patterns", ""])
        for p in report.skipped_patterns:
            lines.append(f"- {p}")

    if report.migration_results:
        lines.extend(["", "## Migration Details", ""])
        for r in report.migration_results:
            status = "OK" if r.success else "FAIL"
            lines.append(f"- [{status}] `{r.file_path}`: {', '.join(r.applied_rules)}")
            if r.error:
                lines.append(f"  - Error: {r.error}")

    if report.recommendations:
        lines.extend(["", "## Recommendations", ""])
        for rec in report.recommendations:
            lines.append(f"- {rec}")

    lines.append("")
    return "\n".join(lines)


def _render_npu_readme(report: AdaptReport) -> str:
    """Render a user-facing NPU adaptation README for the repo root."""
    lines = [
        "# NPU Adaptation Guide",
        "",
        "This repository has been adapted to run on **Huawei Ascend NPU** "
        "using `diffusion-agent`.",
        "",
    ]

    # --- What Changed ---
    lines.append("## What Changed")
    lines.append("")
    lines.append(
        f"- **{report.total_files_modified}** file(s) modified, "
        f"**{report.total_migrations_applied}** migration(s) applied"
    )
    if report.llm_fixes_applied:
        lines.append(f"- **{report.llm_fixes_applied}** additional LLM-assisted fix(es)")
    lines.append("")

    modified = [r for r in report.migration_results if r.success and r.applied_rules]
    if modified:
        lines.append("| File | Rules Applied |")
        lines.append("|------|--------------|")
        for r in modified:
            lines.append(f"| `{r.file_path}` | {', '.join(r.applied_rules)} |")
        lines.append("")

    # --- Environment Requirements ---
    lines.append("## Environment Requirements")
    lines.append("")
    lines.append("- Huawei Ascend NPU (Atlas 300/800/900 series)")
    lines.append("- CANN toolkit (matching your driver version)")
    lines.append("- PyTorch 2.x + torch_npu (matching CANN version)")
    lines.append("- Python 3.10+")
    lines.append("")

    # --- How to Run ---
    lines.append("## How to Run")
    lines.append("")
    lines.append("```bash")
    lines.append("# 1. Install torch_npu (if not already)")
    lines.append("pip install torch_npu")
    lines.append("")
    lines.append("# 2. Verify NPU is visible")
    lines.append("python -c \"import torch, torch_npu; print(torch.npu.is_available())\"")
    lines.append("")
    lines.append("# 3. Run your model as usual — CUDA calls have been replaced with NPU equivalents")
    lines.append("```")
    lines.append("")

    # --- Backup & Rollback ---
    lines.append("## Backup & Rollback")
    lines.append("")
    lines.append(
        "Original files are saved with a `.bak` extension next to each modified file. "
        "To revert a file:"
    )
    lines.append("")
    lines.append("```bash")
    lines.append("mv path/to/file.py.bak path/to/file.py")
    lines.append("```")
    lines.append("")

    # --- Unresolved Patterns ---
    if report.skipped_patterns:
        lines.append("## Unresolved Patterns")
        lines.append("")
        lines.append(
            "The following patterns could not be automatically migrated "
            "and may need manual attention:"
        )
        lines.append("")
        for p in report.skipped_patterns:
            lines.append(f"- {p}")
        lines.append("")

    # --- Next Steps ---
    lines.append("## Next Steps")
    lines.append("")
    lines.append(
        "Run the check scenario to verify the adaptation:"
    )
    lines.append("")
    lines.append("```bash")
    lines.append(f'python -m diffusion_agent --local-path "{report.repo_local_path}" --scenario check')
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verdict / recommendations helpers
# ---------------------------------------------------------------------------

def _determine_verdict(state: AdaptationState) -> str:
    """Determine overall verdict from supervisor state."""
    if not state.files_modified and not state.blockers:
        return "success"  # nothing to do
    if state.blockers and not state.files_modified:
        return "failed"
    if state.blockers:
        return "partial"
    blocked_tasks = [t for t in state.tasks if t.status == "blocked"]
    pending_tasks = [t for t in state.tasks if t.status == "pending"]
    exhausted_tasks = [t for t in state.tasks if t.status == "exhausted"]
    if blocked_tasks:
        return "partial"
    if pending_tasks or exhausted_tasks:
        return "partial"
    return "success"


def _build_recommendations(state: AdaptationState) -> list[str]:
    recs: list[str] = []
    blocked = [t for t in state.tasks if t.status == "blocked"]
    pending = [t for t in state.tasks if t.status == "pending"]
    exhausted = [t for t in state.tasks if t.status == "exhausted"]
    if blocked:
        recs.append(f"{len(blocked)} task(s) blocked — may require manual intervention or unsupported-op workaround.")
    if exhausted:
        recs.append(f"{len(exhausted)} task(s) exhausted all attempts — consider increasing max_attempts or trying a different approach.")
    if pending:
        recs.append(f"{len(pending)} task(s) still pending — consider re-running with LLM enabled.")
    recs.append("Run `--scenario check` after adaptation to verify remaining issues.")
    return recs


# ---------------------------------------------------------------------------
# Collect migration results from supervisor iterations
# ---------------------------------------------------------------------------

def _collect_migration_results(state: AdaptationState) -> list[MigrationResult]:
    """Extract MigrationResult objects from supervisor state.

    Includes both batch-phase results and iterative-loop results.
    """
    results: list[MigrationResult] = []
    seen_files: set[str] = set()

    # First: batch migration results (stored directly as MigrationResult)
    for r in state.batch_migration_results:
        if r.file_path not in seen_files:
            seen_files.add(r.file_path)
            results.append(r)

    # Second: iterative-loop results
    for it in state.iterations:
        if it.accepted and it.files_changed:
            for fp in it.files_changed:
                if fp not in seen_files:
                    seen_files.add(fp)
                    rules = [r for r in (it.hypothesis.proposed_action or "").split(": ")[-1].split(", ") if r]
                    results.append(MigrationResult(
                        file_path=fp,
                        applied_rules=rules if rules != [""] else [it.hypothesis.description],
                        original_hash="",
                        success=True,
                    ))
    return results


# ---------------------------------------------------------------------------
# Scenario implementation
# ---------------------------------------------------------------------------

class AdaptScenario(ScenarioBase):
    """Scenario 2: Migrate CUDA code to run on Ascend NPU.

    Delegates all adaptation work to AdaptSupervisor which manages the
    internal agent loop: planner → patch_worker → runner → judge → git_memory.
    """

    def plan(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "name": "adapt_for_npu",
                "description": "Migrate CUDA code to NPU using supervised adaptation loop",
            }
        ]

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        repo_path = Path(state["repo_local_path"])
        model_name: str = state.get("model_name", "unknown")
        log.info("adapt_start", repo=str(repo_path))

        # Resolve optional LLM
        llm = None
        try:
            from diffusion_agent.config import load_settings
            settings = load_settings()
            if settings.llm_api_key:
                from diffusion_agent.llm.provider import create_llm
                llm = create_llm(settings)
        except Exception:
            log.info("adapt_llm_not_configured")

        # Build execution config from settings/state
        exec_config = None
        try:
            from diffusion_agent.config import load_settings
            s = load_settings()
            if s.npu_ssh_host:
                exec_config = ExecutionConfig(
                    mode="ssh",
                    ssh_host=s.npu_ssh_host,
                    conda_env=s.npu_conda_env,
                    remote_workdir=state.get("remote_workdir"),
                    validation_command=state.get("validation_command"),
                )
        except Exception:
            pass

        # Allow state-level overrides
        if state.get("execution_config"):
            raw = state["execution_config"]
            if isinstance(raw, ExecutionConfig):
                exec_config = raw
            elif isinstance(raw, dict):
                exec_config = ExecutionConfig(**raw)

        # Launch the supervisor — all adaptation work happens inside
        supervisor = AdaptSupervisor(
            repo_path=repo_path,
            model_name=model_name,
            llm=llm,
            use_git=False,  # coding_agent handles git externally
            execution_config=exec_config,
        )
        adapt_state = supervisor.run()

        # Build report from supervisor state
        migration_results = _collect_migration_results(adapt_state)
        skipped_patterns = [
            f"{t.category.value}: {t.description}"
            for t in adapt_state.tasks
            if t.status in ("pending", "blocked", "exhausted")
        ]
        llm_fixes_applied = sum(
            1 for it in adapt_state.iterations
            if it.accepted and it.hypothesis.source == "llm"
        )

        report = AdaptReport(
            model_name=model_name,
            repo_local_path=str(repo_path),
            verdict=_determine_verdict(adapt_state),
            total_files_modified=len(adapt_state.files_modified),
            total_migrations_applied=adapt_state.total_rules_applied,
            migration_results=migration_results,
            llm_fixes_applied=llm_fixes_applied,
            llm_fixes_skipped=0,
            skipped_patterns=skipped_patterns,
            recommendations=_build_recommendations(adapt_state),
            iterations_count=adapt_state.iteration,
            blockers=adapt_state.blockers,
            stop_reason=adapt_state.stop_reason.value if adapt_state.stop_reason else None,
        )

        # Write report files
        reports_dir = repo_path / ".diffusion_agent" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_dict = report.to_dict()
        (reports_dir / "adapt-report.json").write_text(
            json.dumps(report_dict, indent=2, default=str), encoding="utf-8"
        )
        (reports_dir / "adapt-report.md").write_text(
            _render_adapt_markdown(report), encoding="utf-8"
        )

        # Generate NPU adaptation README in repo root
        readme_path = repo_path / "NPU_ADAPTATION_README.md"
        readme_path.write_text(_render_npu_readme(report), encoding="utf-8")
        log.info("adapt_readme_generated", path=str(readme_path))

        log.info("adapt_done", verdict=report.verdict, files=report.total_files_modified)

        # Return updated state
        tool_results = list(state.get("tool_results", []))
        tool_results.append(report_dict)
        return {**state, "tool_results": tool_results}
