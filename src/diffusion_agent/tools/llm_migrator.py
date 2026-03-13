"""LLM-powered migration assistant for unmatched CUDA→NPU findings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from diffusion_agent.tools.code_migrator import MigrationResult, _file_hash
from diffusion_agent.tools.code_scanner import Finding
from diffusion_agent.utils.logging import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

log = get_logger(__name__)


@dataclass
class LLMFix:
    """A fix proposed by the LLM for an unmatched finding."""

    finding: Finding
    original_code: str
    proposed_code: str
    explanation: str
    confidence: float  # 0.0-1.0


def _get_context_lines(file_path: str, line_number: int, window: int = 5) -> str:
    """Extract surrounding context lines from a file."""
    try:
        lines = Path(file_path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return ""

    start = max(0, line_number - 1 - window)
    end = min(len(lines), line_number + window)
    numbered = [f"{i + 1}: {lines[i]}" for i in range(start, end)]
    return "\n".join(numbered)


_REVIEW_PROMPT = """\
You are a CUDA→Ascend NPU migration assistant.

Given the following CUDA/GPU-specific code pattern found in a Python file, propose a \
minimal replacement that makes this code work on Huawei Ascend NPU (using torch_npu).

## Finding
- File: {file_path}
- Line: {line_number}
- Pattern type: {pattern_type}
- Code snippet: {code_snippet}

## Surrounding context
```python
{context}
```

## Instructions
1. Propose the minimal code replacement (change as few lines as possible)
2. Explain why this change is needed
3. Rate your confidence from 0.0 (guess) to 1.0 (certain)

Respond in JSON format:
{{
  "original_code": "<the exact line(s) to replace>",
  "proposed_code": "<the replacement line(s)>",
  "explanation": "<why this change is needed>",
  "confidence": <float 0.0-1.0>
}}
"""

_RULE_SUGGESTION_PROMPT = """\
Given this one-off CUDA→NPU fix, generalize it into a reusable migration rule description.

## Fix applied
- Original: {original_code}
- Replacement: {proposed_code}
- Explanation: {explanation}

Describe a general rule that could handle similar patterns. Include:
1. A regex or AST pattern to match
2. The transformation to apply
3. Any caveats or edge cases

Respond as a plain text description (not code).
"""


def review_unmatched_findings(
    llm: BaseChatModel,
    unmatched: list[Finding],
    file_contents: dict[str, str],
) -> list[LLMFix]:
    """Ask LLM to propose fixes for findings that no built-in rule handles."""
    fixes: list[LLMFix] = []

    for finding in unmatched:
        context = _get_context_lines(finding.file_path, finding.line_number)
        prompt = _REVIEW_PROMPT.format(
            file_path=finding.file_path,
            line_number=finding.line_number,
            pattern_type=finding.pattern_type.value,
            code_snippet=finding.code_snippet,
            context=context,
        )

        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            # Parse JSON from response
            # Handle markdown code blocks
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            data = json.loads(content.strip())
            fixes.append(LLMFix(
                finding=finding,
                original_code=data.get("original_code", finding.code_snippet),
                proposed_code=data.get("proposed_code", ""),
                explanation=data.get("explanation", ""),
                confidence=float(data.get("confidence", 0.5)),
            ))
        except Exception:
            log.warning("llm_review_failed", file=finding.file_path, line=finding.line_number)

    return fixes


def generate_rule_suggestion(llm: BaseChatModel, fix: LLMFix) -> str:
    """Ask LLM to generalize a one-off fix into a reusable rule description."""
    prompt = _RULE_SUGGESTION_PROMPT.format(
        original_code=fix.original_code,
        proposed_code=fix.proposed_code,
        explanation=fix.explanation,
    )
    try:
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception:
        log.warning("llm_rule_suggestion_failed")
        return ""


def apply_llm_fixes(
    fixes: list[LLMFix],
    min_confidence: float = 0.7,
) -> list[MigrationResult]:
    """Apply LLM-proposed fixes above confidence threshold."""
    results: list[MigrationResult] = []

    # Group fixes by file
    by_file: dict[str, list[LLMFix]] = {}
    for fix in fixes:
        if fix.confidence >= min_confidence and fix.proposed_code:
            by_file.setdefault(fix.finding.file_path, []).append(fix)

    for file_path, file_fixes in by_file.items():
        path = Path(file_path)
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            results.append(MigrationResult(
                file_path=file_path, applied_rules=[], original_hash="",
                success=False, error=str(exc),
            ))
            continue

        original_hash = _file_hash(source)
        modified = source
        applied: list[str] = []

        # Sort bottom-up
        sorted_fixes = sorted(file_fixes, key=lambda f: f.finding.line_number, reverse=True)

        for fix in sorted_fixes:
            if fix.original_code in modified:
                modified = modified.replace(fix.original_code, fix.proposed_code, 1)
                applied.append(f"llm_fix_L{fix.finding.line_number}")

        if applied:
            path.write_text(modified, encoding="utf-8")

        results.append(MigrationResult(
            file_path=file_path,
            applied_rules=applied,
            original_hash=original_hash,
            success=True,
        ))

    return results
