"""Unit tests for the LLM migrator — mocked LLM, no real API calls."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from diffusion_agent.tools.code_scanner import Finding, PatternType
from diffusion_agent.tools.llm_migrator import (
    LLMFix,
    _RUNTIME_FIX_PROMPT,
    _normalize_original_code,
    apply_llm_fixes,
    fix_runtime_error,
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


# ---------------------------------------------------------------------------
# _normalize_original_code tests
# ---------------------------------------------------------------------------

class TestNormalizeOriginalCode:
    """Test whitespace-normalized matching for LLM original_code."""

    def test_verbatim_match(self) -> None:
        content = "from decord import VideoReader\n"
        result = _normalize_original_code("from decord import VideoReader", content)
        assert result == "from decord import VideoReader"

    def test_trailing_whitespace_stripped(self) -> None:
        content = "from decord import VideoReader\n"
        result = _normalize_original_code("from decord import VideoReader  ", content)
        assert result == "from decord import VideoReader"

    def test_leading_whitespace_stripped(self) -> None:
        content = "    from decord import VideoReader\n"
        result = _normalize_original_code("from decord import VideoReader", content)
        # Stripped version is a valid substring → used for replacement
        assert result == "from decord import VideoReader"
        assert result in content

    def test_multiline_normalized(self) -> None:
        content = "import foo\nfrom decord import VideoReader\nimport bar\n"
        result = _normalize_original_code(
            "from decord import VideoReader \n", content
        )
        assert result == "from decord import VideoReader"

    def test_empty_returns_empty(self) -> None:
        content = "import foo\n"
        assert _normalize_original_code("", content) == ""

    def test_no_match_returns_empty(self) -> None:
        content = "import foo\n"
        assert _normalize_original_code("import bar", content) == ""

    def test_whitespace_collapsed_match(self) -> None:
        """Multiple spaces collapsed to one should still match."""
        content = "x = foo(a, b, c)\n"
        result = _normalize_original_code("x = foo(a,  b,   c)", content)
        assert result == "x = foo(a, b, c)"

    def test_indentation_mismatch_single_line(self) -> None:
        """LLM returns no indentation but content is indented → still valid for replacement."""
        content = "        x = flash_attn.func(q, k, v)\n"
        result = _normalize_original_code("x = flash_attn.func(q, k, v)", content)
        # The unindented text IS a verbatim substring, so it works for str.replace()
        assert result != ""
        assert result in content

    def test_multiline_fuzzy_match_with_indent_diff(self) -> None:
        """Multi-line code with different indentation should still match."""
        content = "    if True:\n        x = 1\n        y = 2\n    z = 3\n"
        original = "x = 1\ny = 2"  # no indentation
        result = _normalize_original_code(original, content)
        assert "x = 1" in result
        assert "y = 2" in result

    def test_multiline_fuzzy_skips_blank_lines(self) -> None:
        """Multi-line match should tolerate blank lines in the content."""
        content = "    x = 1\n\n    y = 2\n    z = 3\n"
        original = "    x = 1\n    y = 2"
        result = _normalize_original_code(original, content)
        # Should find the block spanning the blank line
        assert "x = 1" in result
        assert "y = 2" in result


# ---------------------------------------------------------------------------
# fix_runtime_error tests — normalized matching and import fallback
# ---------------------------------------------------------------------------

class TestFixRuntimeErrorNormalizedMatch:
    """fix_runtime_error should find a match even with whitespace diffs."""

    def test_whitespace_normalized_match(self, tmp_path: Path) -> None:
        """LLM returns original_code with extra whitespace → still matches."""
        p = _write(tmp_path, "s2v.py", "from decord import VideoReader\nclass Foo: pass\n")

        llm = _mock_llm_response({
            "original_code": "from decord import VideoReader  ",  # trailing ws
            "proposed_code": "try:\n    from decord import VideoReader\nexcept ImportError:\n    VideoReader = None",
            "explanation": "lazy import",
            "confidence": 0.9,
        })

        fixes = fix_runtime_error(llm, "ModuleNotFoundError: No module named 'decord'", {str(p): p.read_text()})
        assert len(fixes) == 1
        assert fixes[0].original_code == "from decord import VideoReader"

    def test_empty_original_code_import_fallback(self, tmp_path: Path) -> None:
        """LLM returns empty original_code for ModuleNotFoundError → fallback finds import line."""
        p = _write(tmp_path, "s2v.py", "import os\nfrom decord import VideoReader\nclass Foo: pass\n")

        llm = _mock_llm_response({
            "original_code": "",  # empty!
            "proposed_code": "try:\n    from decord import VideoReader\nexcept ImportError:\n    VideoReader = None",
            "explanation": "lazy import for unavailable module",
            "confidence": 0.8,
        })

        fixes = fix_runtime_error(
            llm,
            "ModuleNotFoundError: No module named 'decord'",
            {str(p): p.read_text()},
        )
        assert len(fixes) == 1
        assert "decord" in fixes[0].original_code
        assert fixes[0].proposed_code.startswith("try:")

    def test_no_match_import_fallback_plain_import(self, tmp_path: Path) -> None:
        """Fallback also handles `import X` (not just `from X import`)."""
        p = _write(tmp_path, "foo.py", "import os\nimport decord\nclass Foo: pass\n")

        llm = _mock_llm_response({
            "original_code": "wrong line that doesn't exist",
            "proposed_code": "try:\n    import decord\nexcept ImportError:\n    decord = None",
            "explanation": "lazy import",
            "confidence": 0.85,
        })

        fixes = fix_runtime_error(
            llm,
            "ModuleNotFoundError: No module named 'decord'",
            {str(p): p.read_text()},
        )
        assert len(fixes) == 1
        assert fixes[0].original_code == "import decord"

    def test_no_fallback_for_non_import_error(self, tmp_path: Path) -> None:
        """Import fallback only triggers for ModuleNotFoundError."""
        p = _write(tmp_path, "foo.py", "import os\nfrom decord import VideoReader\n")

        llm = _mock_llm_response({
            "original_code": "",  # empty
            "proposed_code": "pass",
            "explanation": "fix",
            "confidence": 0.8,
        })

        fixes = fix_runtime_error(
            llm,
            "RuntimeError: something else",
            {str(p): p.read_text()},
        )
        assert len(fixes) == 0  # no fallback for non-import errors


# ---------------------------------------------------------------------------
# _RUNTIME_FIX_PROMPT namespace preservation constraint
# ---------------------------------------------------------------------------


class TestRuntimeFixPromptNamespaceConstraint:
    """The runtime fix prompt MUST instruct LLM to preserve module exports."""

    def test_prompt_mentions_namespace_preservation(self) -> None:
        """Prompt must contain a constraint about preserving exports."""
        assert "__init__.py" in _RUNTIME_FIX_PROMPT
        assert "export" in _RUNTIME_FIX_PROMPT.lower() or "namespace" in _RUNTIME_FIX_PROMPT.lower()

    def test_prompt_warns_about_lazy_import_scoping(self) -> None:
        """Prompt must warn about hiding variables inside local scopes."""
        lower = _RUNTIME_FIX_PROMPT.lower()
        assert "module level" in lower or "module-level" in lower

    def test_prompt_sent_to_llm_for_init_file(self, tmp_path: Path) -> None:
        """When fixing an __init__.py file, the prompt must include the constraint."""
        init_file = _write(tmp_path, "__init__.py", "from .speech2video import WanS2V\n")

        llm = _mock_llm_response({
            "original_code": "from .speech2video import WanS2V",
            "proposed_code": "try:\n    from .speech2video import WanS2V\nexcept ImportError:\n    WanS2V = None",
            "explanation": "lazy import",
            "confidence": 0.9,
        })

        fix_runtime_error(
            llm,
            "ModuleNotFoundError: No module named 'decord'",
            {str(init_file): init_file.read_text()},
        )

        # Verify the prompt sent to the LLM contains the constraint
        call_args = llm.invoke.call_args
        prompt_text = call_args[0][0] if call_args[0] else str(call_args)
        assert "module level" in prompt_text.lower() or "module-level" in prompt_text.lower()


# ---------------------------------------------------------------------------
# Deep-frame-first: fix_runtime_error deepest_file guidance
# ---------------------------------------------------------------------------


class TestFixRuntimeErrorDeepestFile:
    """fix_runtime_error should tell the LLM which file is the root cause."""

    def test_only_deepest_file_sent_to_llm(self, tmp_path: Path) -> None:
        """When deepest_file is provided, ONLY that file's content goes to the LLM.

        Caller files (generate.py, etc.) must be physically excluded from the
        LLM payload — the LLM should never see them at all.
        """
        entry = _write(tmp_path, "generate.py", "import wan\n")
        inner = _write(tmp_path, "wan/animate.py", "from decord import VideoReader\n")

        llm = _mock_llm_response({
            "original_code": "from decord import VideoReader",
            "proposed_code": "try:\n    from decord import VideoReader\nexcept ImportError:\n    VideoReader = None",
            "explanation": "lazy import",
            "confidence": 0.9,
        })

        fix_runtime_error(
            llm,
            "ModuleNotFoundError: No module named 'decord'",
            {str(entry): entry.read_text(), str(inner): inner.read_text()},
            deepest_file=str(inner),
        )

        # LLM should be called ONLY for the deepest file, not the caller
        assert llm.invoke.call_count == 1

        prompt_text = llm.invoke.call_args[0][0]
        assert "animate.py" in prompt_text
        assert "generate.py" not in prompt_text

    def test_no_deepest_file_all_files_sent(self, tmp_path: Path) -> None:
        """When deepest_file is None, ALL files are sent to the LLM (backward compat)."""
        file_a = _write(tmp_path, "a.py", "import decord\n")
        file_b = _write(tmp_path, "b.py", "from decord import VideoReader\n")

        llm = _mock_llm_response({
            "original_code": "import decord",
            "proposed_code": "try:\n    import decord\nexcept ImportError:\n    decord = None",
            "explanation": "lazy import",
            "confidence": 0.9,
        })

        fix_runtime_error(
            llm,
            "ModuleNotFoundError: No module named 'decord'",
            {str(file_a): file_a.read_text(), str(file_b): file_b.read_text()},
            deepest_file=None,
        )

        # Both files should be processed when no deepest_file
        assert llm.invoke.call_count == 2

    def test_deepest_file_not_in_contents_falls_back_to_all(self, tmp_path: Path) -> None:
        """If deepest_file isn't in file_contents, fall back to processing all files."""
        file_a = _write(tmp_path, "a.py", "import decord\n")

        llm = _mock_llm_response({
            "original_code": "import decord",
            "proposed_code": "try:\n    import decord\nexcept ImportError:\n    decord = None",
            "explanation": "lazy import",
            "confidence": 0.9,
        })

        fix_runtime_error(
            llm,
            "ModuleNotFoundError: No module named 'decord'",
            {str(file_a): file_a.read_text()},
            deepest_file="/nonexistent/deep.py",
        )

        # Falls back to all files since deepest_file is not in file_contents
        assert llm.invoke.call_count == 1


# ---------------------------------------------------------------------------
# Task 2: Ban sys.modules hacks in prompt
# ---------------------------------------------------------------------------


class TestRuntimeFixPromptBanSysModulesHacks:
    """The runtime fix prompt must forbid ALL sys.modules writes."""

    def test_prompt_bans_sys_modules_class_assignment(self) -> None:
        """Prompt must ban sys.modules[__name__].__class__ = ... pattern."""
        lower = _RUNTIME_FIX_PROMPT.lower()
        assert "sys.modules" in lower

    def test_prompt_bans_dynamic_module_override(self) -> None:
        """Prompt must use 'forbidden' or 'must not' language for module hacks."""
        lower = _RUNTIME_FIX_PROMPT.lower()
        assert "forbidden" in lower or "must not" in lower or "never" in lower

    def test_prompt_recommends_safe_lazy_import(self) -> None:
        """Prompt must recommend importing inside functions as the safe alternative."""
        lower = _RUNTIME_FIX_PROMPT.lower()
        assert "inside function" in lower or "inside a function" in lower or "try/except" in lower

    def test_prompt_bans_all_sys_modules_writes(self) -> None:
        """Prompt must ban ALL sys.modules writes, not just __class__ reassignment.

        V3 showed LLM using `sys.modules['wan'] = DummyWanModule()` to bypass
        the __class__-only ban. The rule must cover any sys.modules[key] = ...
        """
        lower = _RUNTIME_FIX_PROMPT.lower()
        # Must explicitly ban assignment/write to sys.modules dict
        assert "sys.modules[" in lower or "sys.modules [" in lower
        # Must cover the general case, not just __class__
        assert ("any form" in lower or "any write" in lower
                or "any assignment" in lower or "any modification" in lower)

    def test_prompt_bans_sys_modules_key_assignment(self) -> None:
        """Prompt must explicitly mention sys.modules[key] = ... pattern."""
        # The prompt must cover `sys.modules['module_name'] = obj` not just __class__
        assert "sys.modules[" in _RUNTIME_FIX_PROMPT


# ---------------------------------------------------------------------------
# Task 3: NPU dtype bypass strategy in prompt
# ---------------------------------------------------------------------------


class TestRuntimeFixPromptNpuDtypeStrategy:
    """The runtime fix prompt MUST teach the LLM to bypass NPU dtype limitations."""

    def test_prompt_mentions_view_as_real(self) -> None:
        """Prompt must mention torch.view_as_real as the complex→real conversion."""
        assert "view_as_real" in _RUNTIME_FIX_PROMPT

    def test_prompt_mentions_view_as_complex(self) -> None:
        """Prompt must mention torch.view_as_complex as the real→complex recovery."""
        assert "view_as_complex" in _RUNTIME_FIX_PROMPT

    def test_prompt_mentions_complex128_dtype(self) -> None:
        """Prompt must explicitly name complex128 as a problematic NPU dtype."""
        assert "complex128" in _RUNTIME_FIX_PROMPT or "complex64" in _RUNTIME_FIX_PROMPT

    def test_prompt_mentions_float64_safecast(self) -> None:
        """Prompt must instruct safe-cast of float64 → float32."""
        lower = _RUNTIME_FIX_PROMPT.lower()
        assert "float64" in lower
        assert "float32" in lower

    def test_prompt_mentions_aclnn_error_pattern(self) -> None:
        """Prompt must reference the aclnn error pattern as the trigger signal."""
        lower = _RUNTIME_FIX_PROMPT.lower()
        assert "aclnn" in lower

    def test_prompt_has_complex_number_workflow(self) -> None:
        """Prompt must describe the full workflow: view_as_real → operate → view_as_complex."""
        # All three steps must appear in order
        text = _RUNTIME_FIX_PROMPT
        real_pos = text.find("view_as_real")
        complex_pos = text.find("view_as_complex")
        assert real_pos >= 0 and complex_pos >= 0
        assert real_pos < complex_pos, "view_as_real must appear before view_as_complex"

    def test_prompt_has_npu_fusion_attention(self) -> None:
        """Prompt must describe torch_npu.npu_fusion_attention as attention replacement."""
        assert "npu_fusion_attention" in _RUNTIME_FIX_PROMPT
        assert "TND" in _RUNTIME_FIX_PROMPT
        assert "input_layout" in _RUNTIME_FIX_PROMPT

    def test_prompt_has_npu_attention_sparse_mode(self) -> None:
        """Prompt must document sparse_mode parameter for causal/non-causal attention."""
        assert "sparse_mode" in _RUNTIME_FIX_PROMPT
        assert "actual_seq_qlen" in _RUNTIME_FIX_PROMPT
        assert "actual_seq_kvlen" in _RUNTIME_FIX_PROMPT
