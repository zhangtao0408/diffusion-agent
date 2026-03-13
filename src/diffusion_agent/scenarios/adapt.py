"""Adapt scenario — automated CUDA→NPU migration with extensible rules + LLM assist."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from diffusion_agent.scenarios import ScenarioBase
from diffusion_agent.tools.code_migrator import (
    MigrationResult,
    add_torch_npu_import,
    apply_all_migrations,
    create_default_registry,
)
from diffusion_agent.tools.code_scanner import scan_directory
from diffusion_agent.utils.logging import get_logger

log = get_logger(__name__)


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
        }


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
    ]

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


def _determine_verdict(results: list[MigrationResult], unresolved: int) -> str:
    """Determine overall verdict."""
    if not results and unresolved == 0:
        return "success"  # nothing to do
    failed = sum(1 for r in results if not r.success)
    if failed == 0 and unresolved == 0:
        return "success"
    if failed > len(results) / 2:
        return "failed"
    return "partial"


def _build_recommendations(
    results: list[MigrationResult],
    skipped: list[str],
) -> list[str]:
    recs: list[str] = []
    if skipped:
        recs.append(
            f"{len(skipped)} pattern(s) could not be auto-migrated. "
            "Review manually or configure LLM for assistance."
        )
    failed = [r for r in results if not r.success]
    if failed:
        recs.append(f"{len(failed)} file(s) had migration errors. Check .bak files for originals.")
    recs.append("Run `--scenario check` after adaptation to verify remaining issues.")
    return recs


class AdaptScenario(ScenarioBase):
    """Scenario 2: Migrate CUDA code to run on Ascend NPU."""

    def plan(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "name": "adapt_for_npu",
                "description": "Migrate CUDA code to NPU using extensible rules + LLM",
            }
        ]

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        repo_path = Path(state["repo_local_path"])
        model_name: str = state.get("model_name", "unknown")
        log.info("adapt_start", repo=str(repo_path))

        # 1. Scan repo (reuse code_scanner)
        findings = scan_directory(repo_path)
        log.info("adapt_scan_complete", count=len(findings))

        # 2. Match findings to rules via registry
        registry = create_default_registry()
        plan = registry.match_all(findings)
        log.info(
            "adapt_plan_ready",
            matched_files=plan.total_files,
            matched_migrations=plan.total_migrations,
            unmatched=len(plan.unmatched),
        )

        # 3. Apply deterministic rules
        results = apply_all_migrations(plan)

        # 4. LLM pass for unmatched findings (if LLM configured)
        llm_fixes_applied = 0
        llm_fixes_skipped = 0
        remaining_unmatched = plan.unmatched

        try:
            from diffusion_agent.config import load_settings
            settings = load_settings()
            if plan.unmatched and settings.llm_api_key:
                from diffusion_agent.llm.provider import create_llm
                from diffusion_agent.tools.llm_migrator import (
                    apply_llm_fixes,
                    review_unmatched_findings,
                )

                llm = create_llm(settings)
                file_contents = {}
                for f in plan.unmatched:
                    if f.file_path not in file_contents:
                        try:
                            file_contents[f.file_path] = Path(f.file_path).read_text("utf-8")
                        except (OSError, UnicodeDecodeError):
                            pass

                llm_fixes = review_unmatched_findings(llm, plan.unmatched, file_contents)
                llm_results = apply_llm_fixes(llm_fixes)
                results.extend(llm_results)

                llm_fixes_applied = sum(
                    len(r.applied_rules) for r in llm_results if r.success
                )
                llm_fixes_skipped = len(plan.unmatched) - llm_fixes_applied
                # Remaining unmatched = those still not fixed
                remaining_unmatched = [
                    f for f in plan.unmatched
                    if not any(
                        f.file_path == r.file_path and r.success and r.applied_rules
                        for r in llm_results
                    )
                ]
        except Exception:
            log.info("adapt_llm_skipped", reason="no LLM configured or import error")

        # 5. Add torch_npu imports to all modified files
        all_modified_files = {r.file_path for r in results if r.success}
        for fp in all_modified_files:
            add_torch_npu_import(fp)

        # 6. Build report
        skipped_patterns = [
            f"{f.pattern_type.value} at {f.file_path}:{f.line_number}"
            for f in remaining_unmatched
        ]

        successful_results = [r for r in results if r.success]
        total_migrations = sum(len(r.applied_rules) for r in successful_results)

        report = AdaptReport(
            model_name=model_name,
            repo_local_path=str(repo_path),
            verdict=_determine_verdict(results, len(remaining_unmatched)),
            total_files_modified=len(all_modified_files),
            total_migrations_applied=total_migrations,
            migration_results=results,
            llm_fixes_applied=llm_fixes_applied,
            llm_fixes_skipped=llm_fixes_skipped,
            skipped_patterns=skipped_patterns,
            recommendations=_build_recommendations(results, skipped_patterns),
        )

        # 7. Write report files
        reports_dir = repo_path / ".diffusion_agent" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_dict = report.to_dict()
        (reports_dir / "adapt-report.json").write_text(
            json.dumps(report_dict, indent=2, default=str), encoding="utf-8"
        )
        (reports_dir / "adapt-report.md").write_text(
            _render_adapt_markdown(report), encoding="utf-8"
        )

        log.info("adapt_done", verdict=report.verdict, files=report.total_files_modified)

        # 8. Return updated state
        tool_results = list(state.get("tool_results", []))
        tool_results.append(report_dict)
        return {**state, "tool_results": tool_results}
