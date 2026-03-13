"""Unit tests for the LLM migrator — mocked LLM, no real API calls."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from diffusion_agent.tools.code_scanner import Finding, PatternType
from diffusion_agent.tools.llm_migrator import (
    LLMFix,
    apply_llm_fixes,
    generate_rule_suggestion,
    review_unmatched_findings,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def _finding(file_path: str, line: int, ptype: PatternType, snippet: str = "") -> Finding:
    return Finding(file_path=file_path, line_number=line, pattern_type=ptype, code_snippet=snippet)


def _mock_llm_response(data: dict) -> MagicMock:
    """Create a mock LLM that returns a JSON response."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content=json.dumps(data))
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReviewUnmatched:
    def test_review_unmatched_returns_fixes(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", "x = some_cuda_op()\nprint(x)\n")
        finding = _finding(str(p), 1, PatternType.BFLOAT16, "x = some_cuda_op()")

        llm = _mock_llm_response({
            "original_code": "x = some_cuda_op()",
            "proposed_code": "x = some_npu_op()",
            "explanation": "Replace CUDA op with NPU equivalent",
            "confidence": 0.9,
        })

        fixes = review_unmatched_findings(llm, [finding], {str(p): p.read_text()})
        assert len(fixes) == 1
        assert fixes[0].confidence == 0.9
        assert fixes[0].proposed_code == "x = some_npu_op()"


class TestApplyLLMFixes:
    def test_apply_above_threshold(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", "x = some_cuda_op()\n")
        fix = LLMFix(
            finding=_finding(str(p), 1, PatternType.BFLOAT16, "x = some_cuda_op()"),
            original_code="x = some_cuda_op()",
            proposed_code="x = some_npu_op()",
            explanation="test",
            confidence=0.9,
        )
        results = apply_llm_fixes([fix], min_confidence=0.7)
        assert len(results) == 1
        assert results[0].success is True
        assert "some_npu_op" in p.read_text()

    def test_apply_below_threshold_skipped(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", "x = some_cuda_op()\n")
        fix = LLMFix(
            finding=_finding(str(p), 1, PatternType.BFLOAT16, "x = some_cuda_op()"),
            original_code="x = some_cuda_op()",
            proposed_code="x = some_npu_op()",
            explanation="test",
            confidence=0.3,
        )
        results = apply_llm_fixes([fix], min_confidence=0.7)
        assert len(results) == 0  # nothing applied
        assert "some_cuda_op" in p.read_text()  # unchanged


class TestGenerateRuleSuggestion:
    def test_returns_description(self) -> None:
        fix = LLMFix(
            finding=_finding("/tmp/a.py", 1, PatternType.BFLOAT16),
            original_code="old",
            proposed_code="new",
            explanation="test fix",
            confidence=0.9,
        )
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="A rule that replaces old with new.")
        result = generate_rule_suggestion(llm, fix)
        assert "rule" in result.lower()


class TestNoLLMConfigured:
    def test_no_llm_skips_gracefully(self, tmp_path: Path) -> None:
        """When LLM is not available, review returns empty list on error."""
        p = _write(tmp_path, "a.py", "x = 1\n")
        finding = _finding(str(p), 1, PatternType.BFLOAT16)

        llm = MagicMock()
        llm.invoke.side_effect = Exception("No API key")

        fixes = review_unmatched_findings(llm, [finding], {})
        assert fixes == []
