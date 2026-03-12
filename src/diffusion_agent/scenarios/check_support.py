"""Check Support scenario — scans repo for CUDA patterns and checks NPU compatibility."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from diffusion_agent.scenarios import ScenarioBase
from diffusion_agent.tools.code_scanner import Finding, PatternType, scan_directory
from diffusion_agent.tools.torch_npu_checker import (
    CheckResult,
    OpStatus,
    check_pattern,
    get_compatibility_summary,
)
from diffusion_agent.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class CheckReport:
    """Structured report produced by the check_support scenario."""

    verdict: str  # "compatible" | "partially_compatible" | "incompatible"
    findings: list[Finding]
    compatibility_results: list[CheckResult]
    summary_stats: dict[str, Any]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "findings": [asdict(f) for f in self.findings],
            "compatibility_results": [asdict(r) for r in self.compatibility_results],
            "summary_stats": self.summary_stats,
            "recommendations": self.recommendations,
        }


def _determine_verdict(results: list[CheckResult]) -> str:
    """Decide overall compatibility verdict from checker results."""
    if not results:
        return "compatible"

    statuses = [r.status for r in results]
    unsupported_count = sum(1 for s in statuses if s is OpStatus.UNSUPPORTED)
    supported_count = sum(1 for s in statuses if s is OpStatus.SUPPORTED)
    total = len(statuses)

    if supported_count == total:
        return "compatible"
    if unsupported_count > total / 2:
        return "incompatible"
    return "partially_compatible"


def _build_recommendations(results: list[CheckResult]) -> list[str]:
    """Generate actionable recommendations from checker results."""
    recs: list[str] = []
    for r in results:
        if r.status in (OpStatus.UNSUPPORTED, OpStatus.PARTIAL) and r.note:
            recs.append(f"{r.op_name}: {r.note}")
    return recs


def _build_summary_stats(
    findings: list[Finding], results: list[CheckResult]
) -> dict[str, Any]:
    """Build summary statistics dict."""
    by_pattern = dict(Counter(f.pattern_type.value for f in findings))
    compat_summary = get_compatibility_summary(results)
    return {
        "total_findings": len(findings),
        "by_pattern_type": by_pattern,
        "by_compatibility": compat_summary,
    }


def _render_markdown(report: CheckReport) -> str:
    """Render report as human-readable Markdown."""
    lines = [
        "# Ascend NPU Compatibility Check Report",
        "",
        f"**Verdict**: {report.verdict}",
        "",
        "## Summary Statistics",
        "",
        f"- Total findings: {report.summary_stats['total_findings']}",
    ]
    for ptype, count in report.summary_stats.get("by_pattern_type", {}).items():
        lines.append(f"  - {ptype}: {count}")

    compat = report.summary_stats.get("by_compatibility", {})
    lines.extend([
        "",
        "## Compatibility Breakdown",
        "",
        f"- Supported: {compat.get('supported', 0)}",
        f"- Unsupported: {compat.get('unsupported', 0)}",
        f"- Partial: {compat.get('partial', 0)}",
        f"- Unknown: {compat.get('unknown', 0)}",
    ])

    if report.findings:
        lines.extend(["", "## Findings", ""])
        for f in report.findings:
            lines.append(f"- **{f.pattern_type.value}** at `{f.file_path}:{f.line_number}`: `{f.code_snippet}`")

    if report.recommendations:
        lines.extend(["", "## Recommendations", ""])
        for rec in report.recommendations:
            lines.append(f"- {rec}")

    lines.append("")
    return "\n".join(lines)


class CheckSupportScenario(ScenarioBase):
    """Scenario 1: Check if a model repo is already adapted for Ascend NPU."""

    def plan(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "name": "check_ascend_support",
                "description": "Scan repository for CUDA patterns and check NPU compatibility",
            }
        ]

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        repo_path = Path(state["repo_local_path"])
        log.info("check_support_start", repo=str(repo_path))

        # 1. Scan repo
        findings = scan_directory(repo_path)
        log.info("scan_complete", count=len(findings))

        # 2. Extract unique pattern types and check each
        unique_patterns = list({f.pattern_type.value for f in findings})
        results = [check_pattern(p) for p in unique_patterns]

        # 3. Build report
        verdict = _determine_verdict(results)
        recommendations = _build_recommendations(results)
        summary_stats = _build_summary_stats(findings, results)

        report = CheckReport(
            verdict=verdict,
            findings=findings,
            compatibility_results=results,
            summary_stats=summary_stats,
            recommendations=recommendations,
        )

        # 4. Write report files
        reports_dir = repo_path / ".diffusion_agent" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_dict = report.to_dict()
        (reports_dir / "check-report.json").write_text(
            json.dumps(report_dict, indent=2, default=str), encoding="utf-8"
        )
        (reports_dir / "check-report.md").write_text(
            _render_markdown(report), encoding="utf-8"
        )

        log.info("check_support_done", verdict=verdict)

        # 5. Return updated state
        tool_results = list(state.get("tool_results", []))
        tool_results.append(report_dict)
        return {**state, "tool_results": tool_results}
