"""LLM-powered migration assistant for unmatched CUDA→NPU findings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from diffusion_agent.tools.code_migrator import MigrationResult, _file_hash
from diffusion_agent.tools.code_scanner import Finding, PatternType
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


_RUNTIME_FIX_PROMPT = """\
You are a CUDA→Ascend NPU migration assistant fixing a RUNTIME error.

The code was statically migrated from CUDA to Huawei Ascend NPU but fails at runtime.
Analyze the error and propose a minimal code fix.

## Runtime Error
```
{error_context}
```
{root_cause_hint}
## File: {file_path}
```python
{file_content}
```

## Instructions
1. Analyze the runtime error and identify the root cause in THIS file
2. Propose the MINIMAL code change to fix the error
3. Common runtime fixes:
   - Lazy imports for packages unavailable on aarch64 (decord, flash_attn, etc.)
   - API replacements (torchaudio → soundfile, imageio.v3, etc.)
   - torch_npu-specific workarounds
   - dtype adjustments for NPU compatibility
4. Only change code in THIS file — do not assume other files will be changed
5. CRITICAL RULE for `__init__.py` and lazy imports:
   When modifying `__init__.py` files or wrapping imports to make them lazy/conditional,
   you MUST preserve the module's external API. Any global constants, configuration
   dictionaries, and `__all__` exports MUST remain accessible at the module level.
   DO NOT hide previously exported variables inside local function scopes or
   try/except blocks. If an import fails, assign a safe fallback (e.g. `None`)
   to every name that was previously exported so downstream code can still reference it.
6. ABSOLUTE BAN ON `sys.modules`: You are STRICTLY FORBIDDEN from performing
   any form of write, assignment, or modification to the `sys.modules` dictionary.
   This includes but is not limited to: `sys.modules[key] = ...`,
   `sys.modules[__name__].__class__ = ...`, `sys.modules.pop(...)`,
   `sys.modules.update(...)`, or creating dummy module objects to inject.
   Such hacks break entry point scripts and are NEVER necessary.
   Instead, use ONLY standard Python patterns for lazy loading: wrapping the
   import inside a try/except at module level, or importing inside a function
   or method that uses it. No exceptions.
7. CRITICAL NPU DTYPE STRATEGY: Ascend NPUs (CANN) often lack kernel support
   for certain data types (e.g., `complex128`, `complex64`, `float64`).
   If you encounter an `aclnn...` or `PTA call acl api failed` error mentioning
   an unsupported dtype (DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, etc.),
   **YOU MUST NOT FAIL**. Instead, use PyTorch structural equivalents:
   - **For Complex Numbers** (`complex128`/`complex64`): convert to real view via
     `torch.view_as_real(tensor)`, perform the operation on real-valued tensor,
     then convert back via `torch.view_as_complex(result)`.
     Example: `torch.cat([a, b])` where a,b are complex128 →
     `torch.view_as_complex(torch.cat([torch.view_as_real(a), torch.view_as_real(b)]))`
   - **For float64**: safe-cast to float32 before the failing op, then cast back:
     `result = op(tensor.float()).to(original_dtype)`
8. NPU ATTENTION REPLACEMENT: When `flash_attn` or `scaled_dot_product_attention`
   fails on NPU (shape mismatch, missing kernel, etc.), replace with
   `torch_npu.npu_fusion_attention` which is the native Ascend attention API.
   Key usage for variable-length (varlen) attention with TND layout:
   ```python
   import torch_npu
   # q, k, v are [total_tokens, num_heads, head_dim] (TND layout)
   # cu_seqlens_q, cu_seqlens_k are cumulative sequence lengths (int32)
   num_heads = q.shape[-2]
   actual_seq_qlen = cu_seqlens_q[1:].to(torch.int32)  # per-sequence lengths
   actual_seq_kvlen = cu_seqlens_k[1:].to(torch.int32)
   # For causal attention: sparse_mode=3, atten_mask=upper triangular bool
   # For non-causal: sparse_mode=0, atten_mask=None
   if causal:
       atten_mask = torch.triu(torch.ones(max_seqlen_q, max_seqlen_k,
           dtype=torch.bool, device=q.device), diagonal=1)
       sparse_mode = 3
   else:
       atten_mask = None
       sparse_mode = 0
   scale = softmax_scale if softmax_scale is not None else (q.shape[-1] ** -0.5)
   output = torch_npu.npu_fusion_attention(
       q, k, v, num_heads, input_layout="TND",
       atten_mask=atten_mask, scale=scale, sparse_mode=sparse_mode,
       actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen,
   )[0]  # returns tuple, take first element
   ```
   This replaces BOTH `flash_attn.flash_attn_varlen_func()` AND
   `torch.nn.functional.scaled_dot_product_attention()` when tensors are in
   TND (flattened batch) layout. The `.unflatten(0, (b, lq))` reshape must be
   preserved after the call to restore the [B, L, N, C] shape

Respond in JSON format:
{{
  "original_code": "<the exact line(s) to replace — must match the file verbatim>",
  "proposed_code": "<the replacement line(s)>",
  "explanation": "<why this change is needed>",
  "confidence": <float 0.0-1.0>
}}
"""


def _normalize_original_code(original_code: str, content: str) -> str:
    """Try to find *original_code* in *content* with relaxed whitespace matching.

    Returns the verbatim substring from *content* that matches, or ``""`` if
    no match is found.  Uses progressively fuzzier strategies:

    1. Verbatim match
    2. Strip leading/trailing whitespace
    3. Single-line: match by stripped line content
    4. Multi-line: join stripped lines
    5. Single-line: whitespace-collapsed comparison (multiple spaces → one)
    6. Multi-line: per-line fuzzy sequential matching
    """
    if not original_code:
        return ""
    # 1. Verbatim match — fast path
    if original_code in content:
        return original_code

    stripped = original_code.strip()
    if not stripped:
        return ""

    # 2. Try stripping trailing whitespace
    if stripped in content:
        return stripped

    # 3. For single-line original_code, find matching content line (preserving indentation)
    if "\n" not in stripped:
        for line in content.splitlines():
            if line.strip() == stripped:
                return line

    # 4. Multi-line: try joining stripped lines against content
    stripped_lines = [ln.strip() for ln in original_code.splitlines() if ln.strip()]
    joined = "\n".join(stripped_lines)
    if joined in content:
        return joined

    # 5. Single-line: whitespace-collapsed comparison
    if "\n" not in stripped:
        import re as _re
        normalized = _re.sub(r"\s+", " ", stripped)
        for line in content.splitlines():
            if _re.sub(r"\s+", " ", line.strip()) == normalized:
                return line

    # 6. Multi-line: per-line fuzzy sequential matching
    if len(stripped_lines) >= 2:
        content_lines = content.splitlines()
        for start_i in range(len(content_lines)):
            if content_lines[start_i].strip() == stripped_lines[0]:
                j = 1
                end_i = start_i
                matched = True
                for ci in range(start_i + 1, len(content_lines)):
                    if j >= len(stripped_lines):
                        break
                    cl = content_lines[ci].strip()
                    if cl == stripped_lines[j]:
                        j += 1
                        end_i = ci
                    elif cl:  # non-empty line doesn't match
                        matched = False
                        break
                if matched and j == len(stripped_lines):
                    block = "\n".join(content_lines[start_i:end_i + 1])
                    if block in content:
                        return block

    return ""


def _extract_module_name(error_context: str) -> str | None:
    """Extract module name from a ModuleNotFoundError message."""
    import re

    m = re.search(r"ModuleNotFoundError: No module named '([^']+)'", error_context)
    return m.group(1) if m else None


def _find_import_line(module_name: str, content: str) -> str | None:
    """Find the import line for *module_name* in *content*.

    Matches both ``from <module> import ...`` and ``import <module>``.
    Returns the full line (with original indentation) or None.
    """
    for line in content.splitlines():
        stripped = line.strip()
        # "from decord import VideoReader" or "from decord.xyz import ..."
        if stripped.startswith(f"from {module_name} import") or stripped.startswith(f"from {module_name}."):
            return line
        # "import decord" or "import decord as ..."
        if stripped == f"import {module_name}" or stripped.startswith(f"import {module_name} "):
            return line
    return None


def fix_runtime_error(
    llm: "BaseChatModel",
    error_context: str,
    file_contents: dict[str, str],
    deepest_file: str | None = None,
) -> list[LLMFix]:
    """Ask LLM to fix a runtime error given the traceback and file contents.

    Unlike ``review_unmatched_findings`` which works from static scan results,
    this function works from a runtime traceback.  It sends each target file
    along with the error context to the LLM and collects proposed fixes.

    Includes two fallback mechanisms when the LLM's ``original_code`` doesn't
    match the file verbatim:

    1. **Normalized matching** — strips leading/trailing whitespace and retries
    2. **Import-line fallback** — for ``ModuleNotFoundError``, finds the
       offending ``import`` / ``from X import`` line directly in the file
    """
    fixes: list[LLMFix] = []

    # --- Physical isolation: only send deepest_file to LLM ---
    # When deepest_file is known AND present in file_contents, discard all
    # caller files entirely.  The LLM cannot modify what it cannot see.
    if deepest_file and deepest_file in file_contents:
        effective_contents = {deepest_file: file_contents[deepest_file]}
        log.info(
            "llm_runtime_fix_isolated",
            deepest=Path(deepest_file).name,
            dropped=[Path(f).name for f in file_contents if f != deepest_file],
        )
    else:
        effective_contents = file_contents

    for file_path, content in effective_contents.items():
        root_cause_hint = "\n"

        prompt = _RUNTIME_FIX_PROMPT.format(
            error_context=error_context,
            root_cause_hint=root_cause_hint,
            file_path=file_path,
            file_content=content,
        )

        try:
            response = llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)

            # Parse JSON — handle markdown code blocks
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text.strip())

            raw_original = data.get("original_code", "")

            # Try verbatim first, then normalized matching
            original_code = _normalize_original_code(raw_original, content)

            # Fallback: for ModuleNotFoundError, find the import line directly
            if not original_code:
                module_name = _extract_module_name(error_context)
                if module_name:
                    import_line = _find_import_line(module_name, content)
                    if import_line:
                        original_code = import_line
                        log.info(
                            "llm_runtime_fix_import_fallback",
                            file=file_path,
                            module=module_name,
                            line=import_line.strip()[:80],
                        )

            if not original_code or original_code not in content:
                log.warning(
                    "llm_runtime_fix_no_match",
                    file=file_path,
                    original_code=raw_original[:80] if raw_original else "<empty>",
                )
                continue

            # Create a synthetic Finding for the LLMFix
            # Find approximate line number
            line_num = 1
            for i, line in enumerate(content.splitlines(), 1):
                if original_code.splitlines()[0].strip() in line:
                    line_num = i
                    break

            finding = Finding(
                file_path=file_path,
                line_number=line_num,
                pattern_type=PatternType.CUDA_CALL,  # placeholder
                code_snippet=original_code,
                context="runtime error fix",
            )

            fixes.append(LLMFix(
                finding=finding,
                original_code=original_code,
                proposed_code=data.get("proposed_code", ""),
                explanation=data.get("explanation", ""),
                confidence=float(data.get("confidence", 0.7)),
            ))
            log.info(
                "llm_runtime_fix_proposed",
                file=file_path,
                confidence=data.get("confidence", 0.7),
                explanation=data.get("explanation", "")[:80],
            )
        except Exception:
            log.warning("llm_runtime_fix_failed", file=file_path)

    return fixes


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
