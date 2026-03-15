"""Patch worker — generates and applies code changes from a hypothesis.

The patch worker is the *inner* adaptation component. It receives a narrow
hypothesis from the planner and produces a small, focused patch.  It does
NOT decide whether the patch is good — that is the judge's job.

Two modes:
  1. Rule-based: uses code_migrator's RuleRegistry (deterministic)
  2. LLM-assisted: uses llm_migrator for unmatched patterns (stochastic)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from diffusion_agent.adapt.types import Hypothesis
from diffusion_agent.tools.code_migrator import (
    MigrationResult,
    RuleRegistry,
    add_torch_npu_import,
    apply_all_migrations,
)
from diffusion_agent.tools.code_scanner import Finding
from diffusion_agent.utils.logging import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

log = get_logger(__name__)


@dataclass
class PatchResult:
    """Outcome of a single patch application."""

    hypothesis_id: str
    files_changed: list[str]
    rules_applied: list[str]
    description: str
    success: bool
    error: str | None = None
    migration_results: list[MigrationResult] | None = None


class PatchWorker:
    """Generates and applies patches based on hypotheses.

    The worker uses the existing code_migrator rule engine for deterministic
    patches and optionally the llm_migrator for LLM-assisted fixes.
    """

    def __init__(
        self,
        registry: RuleRegistry,
        llm: BaseChatModel | None = None,
    ) -> None:
        self.registry = registry
        self.llm = llm

    def apply_rule_patch(
        self,
        hypothesis: Hypothesis,
        findings: list[Finding],
    ) -> PatchResult:
        """Apply deterministic rule-based migrations for a hypothesis.

        Filters findings to the hypothesis's target files, matches them
        against the registry, and applies in bottom-up order.
        """
        # Filter findings for target files
        target_findings = [
            f for f in findings
            if f.file_path in hypothesis.target_files
        ]

        if not target_findings:
            return PatchResult(
                hypothesis_id=hypothesis.id,
                files_changed=[],
                rules_applied=[],
                description="No matching findings for target files",
                success=True,
            )

        # Build a mini migration plan for just these files
        plan = self.registry.match_all(target_findings)

        if plan.total_migrations == 0:
            return PatchResult(
                hypothesis_id=hypothesis.id,
                files_changed=[],
                rules_applied=[],
                description="No rules matched the findings",
                success=True,
            )

        # Apply migrations
        results = apply_all_migrations(plan)

        # Add torch_npu imports to modified files
        modified_files = [r.file_path for r in results if r.success and r.applied_rules]
        for fp in modified_files:
            add_torch_npu_import(fp)

        all_rules = []
        for r in results:
            all_rules.extend(r.applied_rules)

        return PatchResult(
            hypothesis_id=hypothesis.id,
            files_changed=modified_files,
            rules_applied=all_rules,
            description=f"Applied {len(all_rules)} rule(s) to {len(modified_files)} file(s)",
            success=all(r.success for r in results),
            error="; ".join(r.error for r in results if r.error) or None,
            migration_results=results,
        )

    def apply_llm_patch(
        self,
        hypothesis: Hypothesis,
        findings: list[Finding],
    ) -> PatchResult:
        """Apply LLM-assisted patches for unmatched findings."""
        if self.llm is None:
            return PatchResult(
                hypothesis_id=hypothesis.id,
                files_changed=[],
                rules_applied=[],
                description="LLM not configured — skipping LLM patch",
                success=True,
            )

        # Filter to target files
        target_findings = [
            f for f in findings
            if f.file_path in hypothesis.target_files
        ]

        if not target_findings:
            return PatchResult(
                hypothesis_id=hypothesis.id,
                files_changed=[],
                rules_applied=[],
                description="No findings for LLM to fix",
                success=True,
            )

        try:
            from diffusion_agent.tools.llm_migrator import (
                apply_llm_fixes,
                review_unmatched_findings,
            )

            # Read file contents for context
            file_contents: dict[str, str] = {}
            for f in target_findings:
                if f.file_path not in file_contents:
                    try:
                        file_contents[f.file_path] = Path(f.file_path).read_text("utf-8")
                    except (OSError, UnicodeDecodeError):
                        pass

            llm_fixes = review_unmatched_findings(self.llm, target_findings, file_contents)
            results = apply_llm_fixes(llm_fixes)

            modified = [r.file_path for r in results if r.success and r.applied_rules]
            for fp in modified:
                add_torch_npu_import(fp)

            all_rules = []
            for r in results:
                all_rules.extend(r.applied_rules)

            return PatchResult(
                hypothesis_id=hypothesis.id,
                files_changed=modified,
                rules_applied=all_rules,
                description=f"LLM applied {len(all_rules)} fix(es) to {len(modified)} file(s)",
                success=True,
                migration_results=results,
            )
        except Exception as exc:
            log.warning("llm_patch_failed", error=str(exc))
            return PatchResult(
                hypothesis_id=hypothesis.id,
                files_changed=[],
                rules_applied=[],
                description=f"LLM patch failed: {exc}",
                success=False,
                error=str(exc),
            )

    def apply_batch_rules(
        self,
        findings: list[Finding],
    ) -> PatchResult:
        """Apply all deterministic rules to all findings at once.

        Used for the initial bulk migration pass before entering
        the iterative loop.
        """
        plan = self.registry.match_all(findings)

        if plan.total_migrations == 0:
            return PatchResult(
                hypothesis_id="batch-rules",
                files_changed=[],
                rules_applied=[],
                description="No rules matched any findings",
                success=True,
            )

        results = apply_all_migrations(plan)

        modified_files = list({r.file_path for r in results if r.success and r.applied_rules})
        for fp in modified_files:
            add_torch_npu_import(fp)

        all_rules: list[str] = []
        for r in results:
            all_rules.extend(r.applied_rules)

        return PatchResult(
            hypothesis_id="batch-rules",
            files_changed=modified_files,
            rules_applied=all_rules,
            description=f"Batch: {len(all_rules)} rule(s) across {len(modified_files)} file(s)",
            success=all(r.success for r in results),
            error="; ".join(r.error for r in results if r.error) or None,
            migration_results=results,
        )

    def apply_runtime_llm_patch(
        self,
        hypothesis: Hypothesis,
    ) -> PatchResult:
        """Apply LLM patch for a runtime error (no static findings needed).

        Uses the hypothesis description (which contains the trimmed traceback)
        and target_files to ask the LLM for a fix.  This is the Phase C code
        path where errors come from actual execution, not static scanning.
        """
        if self.llm is None:
            return PatchResult(
                hypothesis_id=hypothesis.id,
                files_changed=[],
                rules_applied=[],
                description="LLM not configured — skipping runtime patch",
                success=True,
            )

        if not hypothesis.target_files:
            return PatchResult(
                hypothesis_id=hypothesis.id,
                files_changed=[],
                rules_applied=[],
                description="No target files identified from traceback",
                success=True,
            )

        try:
            from diffusion_agent.tools.llm_migrator import (
                apply_llm_fixes,
                fix_runtime_error,
            )

            # Read target file contents
            file_contents: dict[str, str] = {}
            for fp in hypothesis.target_files:
                try:
                    file_contents[fp] = Path(fp).read_text("utf-8")
                except (OSError, UnicodeDecodeError):
                    pass

            if not file_contents:
                return PatchResult(
                    hypothesis_id=hypothesis.id,
                    files_changed=[],
                    rules_applied=[],
                    description="Could not read target files",
                    success=True,
                )

            # Extract error context from hypothesis description
            error_context = hypothesis.description
            prefix = "Fix runtime failure:\n"
            if error_context.startswith(prefix):
                error_context = error_context[len(prefix):]

            llm_fixes = fix_runtime_error(
                self.llm, error_context, file_contents,
                deepest_file=hypothesis.deepest_file,
            )
            results = apply_llm_fixes(llm_fixes)

            modified = [r.file_path for r in results if r.success and r.applied_rules]
            for fp in modified:
                add_torch_npu_import(fp)

            all_rules: list[str] = []
            for r in results:
                all_rules.extend(r.applied_rules)

            return PatchResult(
                hypothesis_id=hypothesis.id,
                files_changed=modified,
                rules_applied=all_rules,
                description=f"Runtime LLM fix: {len(all_rules)} change(s) in {len(modified)} file(s)",
                success=True,
                migration_results=results,
            )
        except Exception as exc:
            log.warning("runtime_llm_patch_failed", error=str(exc))
            return PatchResult(
                hypothesis_id=hypothesis.id,
                files_changed=[],
                rules_applied=[],
                description=f"Runtime LLM patch failed: {exc}",
                success=False,
                error=str(exc),
            )

    def apply_patch(
        self,
        hypothesis: Hypothesis,
        findings: list[Finding],
    ) -> PatchResult:
        """Dispatch to the right patch method based on hypothesis source.

        For LLM hypotheses with no static findings but identified target files
        (i.e., runtime errors from Phase C), routes to ``apply_runtime_llm_patch``.
        """
        if hypothesis.source == "rule":
            return self.apply_rule_patch(hypothesis, findings)
        elif hypothesis.source == "llm":
            # Runtime error path: no static findings, but target files from traceback
            if not findings and hypothesis.target_files:
                return self.apply_runtime_llm_patch(hypothesis)
            return self.apply_llm_patch(hypothesis, findings)
        else:
            return self.apply_rule_patch(hypothesis, findings)
