"""Check Support scenario — scans repo for CUDA patterns and checks NPU compatibility."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from diffusion_agent.scenarios import ScenarioBase
from diffusion_agent.tools.code_scanner import Finding, scan_directory
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

    # Results
    verdict: str  # "compatible" | "partially_compatible" | "incompatible"
    findings: list[Finding]
    compatibility_results: list[CheckResult]
    summary_stats: dict[str, Any]
    recommendations: list[str]

    # Identification
    model_name: str = "unknown"
    repo_url: str | None = None
    repo_local_path: str = ""

    # Source tracking
    torch_npu_version: str | None = None
    api_reference_branch: str | None = None
    op_matrix_source: str = "static"  # "static" | "dynamic"

    # Evidence
    blocking_issues: list[str] = field(default_factory=list)
    suspected_root_causes: list[str] = field(default_factory=list)

    # Runtime verification (if available)
    runtime_results: list[dict[str, Any]] | None = None

    # CI baseline comparison
    baseline_comparison: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "findings": [asdict(f) for f in self.findings],
            "compatibility_results": [asdict(r) for r in self.compatibility_results],
            "summary_stats": self.summary_stats,
            "recommendations": self.recommendations,
            "model_name": self.model_name,
            "repo_url": self.repo_url,
            "repo_local_path": self.repo_local_path,
            "torch_npu_version": self.torch_npu_version,
            "api_reference_branch": self.api_reference_branch,
            "op_matrix_source": self.op_matrix_source,
            "blocking_issues": self.blocking_issues,
            "suspected_root_causes": self.suspected_root_causes,
            "runtime_results": self.runtime_results,
            "baseline_comparison": self.baseline_comparison,
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


def _build_blocking_issues(results: list[CheckResult]) -> list[str]:
    """Extract issues that prevent running on NPU (unsupported ops)."""
    return [
        f"{r.op_name}: {r.note}"
        for r in results
        if r.status is OpStatus.UNSUPPORTED and r.note
    ]


def _build_suspected_root_causes(
    findings: list[Finding], results: list[CheckResult],
) -> list[str]:
    """Identify likely root causes of incompatibility."""
    causes: list[str] = []
    finding_types = {f.pattern_type.value for f in findings}

    if "flash_attn" in finding_types:
        causes.append("Uses flash_attn (CUDA-only). Replace with SDPA or torch_npu attention.")
    if "xformers" in finding_types:
        causes.append("Uses xformers (CUDA-only). Replace with SDPA or torch_npu attention.")
    if "nccl" in finding_types:
        causes.append("Uses NCCL backend. Replace with HCCL for Ascend NPU.")
    if "float64" in finding_types:
        causes.append("Uses float64/double. NPU supports float32/float16/bfloat16 only.")
    return causes


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
        f"**Model**: {report.model_name}",
    ]
    if report.repo_url:
        lines.append(f"**Repository**: {report.repo_url}")
    lines.append(f"**Local Path**: {report.repo_local_path}")
    lines.extend([
        f"**Verdict**: {report.verdict}",
        f"**Op Matrix Source**: {report.op_matrix_source}",
    ])
    if report.torch_npu_version:
        lines.append(f"**torch_npu Version**: {report.torch_npu_version}")
    if report.api_reference_branch:
        lines.append(f"**API Reference Branch**: {report.api_reference_branch}")

    lines.extend([
        "",
        "## Summary Statistics",
        "",
        f"- Total findings: {report.summary_stats['total_findings']}",
    ])
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

    if report.blocking_issues:
        lines.extend(["", "## Blocking Issues", ""])
        for issue in report.blocking_issues:
            lines.append(f"- {issue}")

    if report.suspected_root_causes:
        lines.extend(["", "## Suspected Root Causes", ""])
        for cause in report.suspected_root_causes:
            lines.append(f"- {cause}")

    if report.findings:
        lines.extend(["", "## Findings", ""])
        for f in report.findings:
            lines.append(f"- **{f.pattern_type.value}** at `{f.file_path}:{f.line_number}`: `{f.code_snippet}`")

    if report.recommendations:
        lines.extend(["", "## Recommendations", ""])
        for rec in report.recommendations:
            lines.append(f"- {rec}")

    if report.runtime_results:
        lines.extend(["", "## Runtime Verification", ""])
        for rv in report.runtime_results:
            status = "PASS" if rv.get("passed") else "FAIL"
            lines.append(f"- [{status}] {rv.get('op_name', 'unknown')}")
            if rv.get("error"):
                lines.append(f"  - Error: {rv['error']}")

    if report.baseline_comparison:
        lines.extend(["", "## Baseline Comparison", ""])
        for bname, comp in report.baseline_comparison.items():
            lines.append(f"### vs {bname}")
            lines.append(f"- Shared patterns: {', '.join(comp.get('shared_patterns', [])) or 'none'}")
            lines.append(f"- Target-only patterns: {', '.join(comp.get('target_only', [])) or 'none'}")
            lines.append(f"- Baseline-only patterns: {', '.join(comp.get('baseline_only', [])) or 'none'}")

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
        npu_version: str | None = state.get("torch_npu_version")
        model_name: str = state.get("model_name", "unknown")
        repo_url: str | None = state.get("repo_url")
        log.info("check_support_start", repo=str(repo_path), npu_version=npu_version)

        # Resolve API reference branch
        api_ref_branch: str | None = None
        if npu_version:
            from diffusion_agent.tools.api_doc_fetcher import resolve_branch
            api_ref_branch = resolve_branch(npu_version)

        # 1. Scan repo
        findings = scan_directory(repo_path)
        log.info("scan_complete", count=len(findings))

        # 2. Extract unique pattern types and check each
        unique_patterns = list({f.pattern_type.value for f in findings})
        results = [check_pattern(p, version=npu_version) for p in unique_patterns]

        # 3. Build report
        verdict = _determine_verdict(results)
        recommendations = _build_recommendations(results)
        summary_stats = _build_summary_stats(findings, results)
        blocking_issues = _build_blocking_issues(results)
        root_causes = _build_suspected_root_causes(findings, results)

        report = CheckReport(
            verdict=verdict,
            findings=findings,
            compatibility_results=results,
            summary_stats=summary_stats,
            recommendations=recommendations,
            model_name=model_name,
            repo_url=repo_url,
            repo_local_path=str(repo_path),
            torch_npu_version=npu_version,
            api_reference_branch=api_ref_branch,
            op_matrix_source="dynamic" if npu_version else "static",
            blocking_issues=blocking_issues,
            suspected_root_causes=root_causes,
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
