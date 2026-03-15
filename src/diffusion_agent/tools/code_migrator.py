"""Extensible CUDA→NPU migration engine with a rule registry."""

from __future__ import annotations

import hashlib
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from diffusion_agent.tools.code_scanner import Finding, PatternType
from diffusion_agent.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# NPU varlen attention wrapper (injected by FlashAttnUsageRule)
# ---------------------------------------------------------------------------

_NPU_WRAPPER_MARKER = "# __NEEDS_NPU_VARLEN_WRAPPER__"

_NPU_VARLEN_WRAPPER = '''\


def _npu_varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                          dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    """[NPU] Variable-length attention via torch_npu.npu_fusion_attention (TND layout)."""
    import torch_npu
    num_heads = q.shape[-2]  # TND: [total_tokens, num_heads, head_dim]
    actual_seq_qlen = cu_seqlens_q[1:].to(torch.int32)
    actual_seq_kvlen = cu_seqlens_k[1:].to(torch.int32)
    if causal:
        atten_mask = torch.triu(
            torch.ones(max_seqlen_q, max_seqlen_k, dtype=torch.bool, device=q.device),
            diagonal=1)
        sparse_mode = 3
    else:
        atten_mask = None
        sparse_mode = 0
    scale = softmax_scale if softmax_scale is not None else (q.shape[-1] ** -0.5)
    return torch_npu.npu_fusion_attention(
        q, k, v, num_heads,
        input_layout="TND",
        atten_mask=atten_mask,
        scale=scale,
        sparse_mode=sparse_mode,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )[0]
'''


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class MigrationResult:
    file_path: str
    applied_rules: list[str]
    original_hash: str
    success: bool
    error: str | None = None


@dataclass
class MigrationPlan:
    matched: dict[str, list[tuple[Finding, "MigrationRule"]]]  # by file
    unmatched: list[Finding]
    total_files: int
    total_migrations: int


# ---------------------------------------------------------------------------
# Rule plugin framework
# ---------------------------------------------------------------------------

class MigrationRule(ABC):
    """Base class for all migration rules (plugin interface)."""

    name: str
    description: str
    pattern_type: PatternType

    @abstractmethod
    def apply(self, source: str, finding: Finding) -> str:
        """Transform source code. Return modified source."""

    @abstractmethod
    def is_already_applied(self, source: str, finding: Finding) -> bool:
        """Return True if the rule's transformation is already present."""

    def matches(self, finding: Finding) -> bool:
        """Whether this rule can handle the given finding."""
        return finding.pattern_type == self.pattern_type


class RuleRegistry:
    """Central registry for migration rules. Extensible at runtime."""

    def __init__(self) -> None:
        self._rules: list[MigrationRule] = []

    def register(self, rule: MigrationRule) -> None:
        self._rules.append(rule)

    def unregister(self, rule_name: str) -> None:
        self._rules = [r for r in self._rules if r.name != rule_name]

    def get_rules(self) -> list[MigrationRule]:
        return list(self._rules)

    def match(self, finding: Finding) -> MigrationRule | None:
        for rule in self._rules:
            if rule.matches(finding):
                return rule
        return None

    def match_all(self, findings: list[Finding]) -> MigrationPlan:
        matched: dict[str, list[tuple[Finding, MigrationRule]]] = {}
        unmatched: list[Finding] = []

        for finding in findings:
            rule = self.match(finding)
            if rule is not None:
                matched.setdefault(finding.file_path, []).append((finding, rule))
            else:
                unmatched.append(finding)

        return MigrationPlan(
            matched=matched,
            unmatched=unmatched,
            total_files=len(matched),
            total_migrations=sum(len(v) for v in matched.values()),
        )


# ---------------------------------------------------------------------------
# Built-in rules
# ---------------------------------------------------------------------------

class CudaCallRule(MigrationRule):
    name = "cuda_call"
    description = ".cuda() → .npu()"
    pattern_type = PatternType.CUDA_CALL

    def is_already_applied(self, source: str, finding: Finding) -> bool:
        lines = source.splitlines()
        idx = finding.line_number - 1
        if 0 <= idx < len(lines):
            return ".cuda(" not in lines[idx]
        return False

    def apply(self, source: str, finding: Finding) -> str:
        lines = source.splitlines(keepends=True)
        idx = finding.line_number - 1
        if 0 <= idx < len(lines):
            lines[idx] = lines[idx].replace(".cuda(", ".npu(")
        return "".join(lines)


class CudaToRule(MigrationRule):
    name = "cuda_to"
    description = '.to("cuda"/"cuda:N") → .to("npu"/"npu:N")'
    pattern_type = PatternType.CUDA_TO

    def is_already_applied(self, source: str, finding: Finding) -> bool:
        lines = source.splitlines()
        idx = finding.line_number - 1
        if 0 <= idx < len(lines):
            return "cuda" not in lines[idx]
        return False

    def apply(self, source: str, finding: Finding) -> str:
        lines = source.splitlines(keepends=True)
        idx = finding.line_number - 1
        if 0 <= idx < len(lines):
            lines[idx] = re.sub(
                r'"cuda(:\d+)?"',
                lambda m: f'"npu{m.group(1) or ""}"',
                lines[idx],
            )
            lines[idx] = re.sub(
                r"'cuda(:\d+)?'",
                lambda m: f"'npu{m.group(1) or ''}'",
                lines[idx],
            )
        return "".join(lines)


class CudaApiRule(MigrationRule):
    name = "cuda_api"
    description = "torch.cuda.* → torch.npu.*"
    pattern_type = PatternType.CUDA_API

    def is_already_applied(self, source: str, finding: Finding) -> bool:
        lines = source.splitlines()
        idx = finding.line_number - 1
        if 0 <= idx < len(lines):
            return "torch.cuda." not in lines[idx]
        return False

    def apply(self, source: str, finding: Finding) -> str:
        lines = source.splitlines(keepends=True)
        idx = finding.line_number - 1
        if 0 <= idx < len(lines):
            lines[idx] = lines[idx].replace("torch.cuda.", "torch.npu.")
        return "".join(lines)


class NcclToHcclRule(MigrationRule):
    name = "nccl_to_hccl"
    description = '"nccl" → "hccl"'
    pattern_type = PatternType.NCCL

    def is_already_applied(self, source: str, finding: Finding) -> bool:
        lines = source.splitlines()
        idx = finding.line_number - 1
        if 0 <= idx < len(lines):
            line = lines[idx]
            return '"nccl"' not in line and "'nccl'" not in line
        return False

    def apply(self, source: str, finding: Finding) -> str:
        lines = source.splitlines(keepends=True)
        idx = finding.line_number - 1
        if 0 <= idx < len(lines):
            lines[idx] = lines[idx].replace('"nccl"', '"hccl"')
            lines[idx] = lines[idx].replace("'nccl'", "'hccl'")
        return "".join(lines)


class FlashAttnRule(MigrationRule):
    name = "flash_attn"
    description = "Comment out flash_attn import + add SDPA note"
    pattern_type = PatternType.FLASH_ATTN

    def is_already_applied(self, source: str, finding: Finding) -> bool:
        lines = source.splitlines()
        idx = finding.line_number - 1
        if 0 <= idx < len(lines):
            # Already applied if NPU marker present
            if "# [NPU]" in lines[idx] or (
                idx + 1 < len(lines) and "# [NPU]" in lines[idx + 1]
            ):
                return True
            # Already guarded if import is inside try/except with proper fallback
            if self._is_inside_try_except_guard(lines, idx):
                return True
        return False

    @staticmethod
    def _is_inside_try_except_guard(lines: list[str], idx: int) -> bool:
        """Check if import at idx is inside try/except with fallback (= False / = None)."""
        for i in range(idx - 1, max(idx - 5, -1), -1):
            if lines[i].strip() == "try:":
                for j in range(idx + 1, min(idx + 10, len(lines))):
                    if lines[j].strip().startswith("except"):
                        for k in range(j + 1, min(j + 5, len(lines))):
                            fline = lines[k].strip()
                            if "= False" in fline or "= None" in fline:
                                return True
                        return False
                break
        return False

    def apply(self, source: str, finding: Finding) -> str:
        lines = source.splitlines(keepends=True)
        idx = finding.line_number - 1
        if 0 <= idx < len(lines):
            indent = len(lines[idx]) - len(lines[idx].lstrip())
            original = lines[idx].rstrip("\n")
            comment = f"{' ' * indent}# [NPU] {original.strip()}  # TODO: use torch SDPA instead\n"
            # If inside a try/except block, add pass to keep block valid
            if idx > 0 and lines[idx - 1].rstrip().endswith(("try:", "except:", "except ImportError:")):
                lines[idx] = f"{' ' * indent}pass\n{comment}"
            else:
                lines[idx] = comment
        return "".join(lines)


class XformersRule(MigrationRule):
    name = "xformers"
    description = "Comment out xformers import + add SDPA note"
    pattern_type = PatternType.XFORMERS

    def is_already_applied(self, source: str, finding: Finding) -> bool:
        lines = source.splitlines()
        idx = finding.line_number - 1
        if 0 <= idx < len(lines):
            if "# [NPU]" in lines[idx] or (
                idx + 1 < len(lines) and "# [NPU]" in lines[idx + 1]
            ):
                return True
            # Already guarded if inside try/except with proper fallback
            if FlashAttnRule._is_inside_try_except_guard(lines, idx):
                return True
        return False

    def apply(self, source: str, finding: Finding) -> str:
        lines = source.splitlines(keepends=True)
        idx = finding.line_number - 1
        if 0 <= idx < len(lines):
            indent = len(lines[idx]) - len(lines[idx].lstrip())
            original = lines[idx].rstrip("\n")
            comment = f"{' ' * indent}# [NPU] {original.strip()}  # TODO: use torch SDPA instead\n"
            # If inside a try/except block, add pass to keep block valid
            if idx > 0 and lines[idx - 1].rstrip().endswith(("try:", "except:", "except ImportError:")):
                lines[idx] = f"{' ' * indent}pass\n{comment}"
            else:
                lines[idx] = comment
        return "".join(lines)


class CudaAmpRule(MigrationRule):
    name = "cuda_amp"
    description = "torch.cuda.amp → torch.amp (vendor-neutral since PyTorch 2.x)"
    pattern_type = PatternType.CUDA_AMP

    def is_already_applied(self, source: str, finding: Finding) -> bool:
        lines = source.splitlines()
        idx = finding.line_number - 1
        if 0 <= idx < len(lines):
            return "torch.cuda.amp" not in lines[idx]
        return False

    def apply(self, source: str, finding: Finding) -> str:
        lines = source.splitlines(keepends=True)
        idx = finding.line_number - 1
        if 0 <= idx < len(lines):
            lines[idx] = lines[idx].replace("torch.cuda.amp", "torch.amp")
        return "".join(lines)


class CudaDeviceStrRule(MigrationRule):
    name = "cuda_device_str"
    description = '"cuda" string literals → "npu" (torch.device, defaults, f-strings, etc.)'
    pattern_type = PatternType.CUDA_DEVICE_STR

    # Matches "cuda", "cuda:0", "cuda: 0", "cuda :0", "cuda : 0" in double/single quotes
    _CUDA_STR_DQ = re.compile(r'"cuda(\s*:\s*[^"]*)?(?=")')
    _CUDA_STR_SQ = re.compile(r"'cuda(\s*:\s*[^']*)?(?=')")

    def is_already_applied(self, source: str, finding: Finding) -> bool:
        lines = source.splitlines()
        idx = finding.line_number - 1
        if 0 <= idx < len(lines):
            return "cuda" not in lines[idx]
        return False

    def apply(self, source: str, finding: Finding) -> str:
        lines = source.splitlines(keepends=True)
        idx = finding.line_number - 1
        if 0 <= idx < len(lines):
            line = lines[idx]

            # Handle .startswith("cuda") → .startswith("npu") first (before general replace)
            line = line.replace('.startswith("cuda")', '.startswith("npu")')
            line = line.replace(".startswith('cuda')", ".startswith('npu')")

            # Replace "cuda..." with "npu..." — handles spaces around colons
            line = self._CUDA_STR_DQ.sub(
                lambda m: f'"npu{m.group(1) or ""}' if m.group(1) is None
                else f'"npu{m.group(1).replace("cuda", "npu")}',
                line,
            )
            # Simplify: just replace "cuda with "npu in all remaining double-quoted contexts
            line = re.sub(
                r'"cuda(\s*:\s*[^"]*)"',
                lambda m: f'"npu{m.group(1)}"',
                line,
            )
            line = re.sub(
                r'"cuda"',
                '"npu"',
                line,
            )
            line = re.sub(
                r"'cuda(\s*:\s*[^']*)'",
                lambda m: f"'npu{m.group(1)}'",
                line,
            )
            line = re.sub(
                r"'cuda'",
                "'npu'",
                line,
            )

            # Handle f-string patterns: f"cuda:{expr}" → f"npu:{expr}"
            # In source code, f-strings appear literally, so we match the raw text
            line = re.sub(r'f"cuda:', 'f"npu:', line)
            line = re.sub(r"f'cuda:", "f'npu:", line)
            line = re.sub(r'f"cuda"', 'f"npu"', line)
            line = re.sub(r"f'cuda'", "f'npu'", line)

            lines[idx] = line
        return "".join(lines)


class NpuInitInjectorRule(MigrationRule):
    """Inject ``import torch_npu`` + ``from torch_npu.contrib import transfer_to_npu``
    after ``import torch`` statements.  ``transfer_to_npu`` monkey-patches CUDA calls
    to NPU transparently, solving ``Torch not compiled with CUDA enabled`` errors.
    """
    name = "npu_init_injector"
    description = "Inject torch_npu + transfer_to_npu after import torch"
    pattern_type = PatternType.TORCH_IMPORT

    def is_already_applied(self, source: str, finding: Finding) -> bool:
        return "import torch_npu" in source and "from torch_npu.contrib import transfer_to_npu" in source

    def apply(self, source: str, finding: Finding) -> str:
        has_torch_npu = "import torch_npu" in source
        has_transfer = "from torch_npu.contrib import transfer_to_npu" in source

        if has_torch_npu and has_transfer:
            return source  # already fully present

        lines = source.splitlines(keepends=True)

        if not has_torch_npu:
            # Insert both lines after the import torch line
            idx = finding.line_number - 1
            if 0 <= idx < len(lines):
                lines.insert(idx + 1, "import torch_npu\n")
                lines.insert(idx + 2, "from torch_npu.contrib import transfer_to_npu\n")
        elif not has_transfer:
            # torch_npu exists but transfer missing — insert after torch_npu
            for i, line in enumerate(lines):
                if line.strip() == "import torch_npu":
                    lines.insert(i + 1, "from torch_npu.contrib import transfer_to_npu\n")
                    break

        return "".join(lines)


class AutocastDtypeRule(MigrationRule):
    """Replace ``dtype=torch.float32`` with ``dtype=torch.bfloat16`` in autocast calls.

    NPU autocast may not preserve float32 semantics the same way CUDA does.
    bfloat16 is the recommended mixed-precision dtype on Ascend NPU.
    """
    name = "autocast_dtype"
    description = "autocast dtype=torch.float32 → dtype=torch.bfloat16"
    pattern_type = PatternType.AUTOCAST_DTYPE

    def is_already_applied(self, source: str, finding: Finding) -> bool:
        lines = source.splitlines()
        idx = finding.line_number - 1
        if 0 <= idx < len(lines):
            return "dtype=torch.float32" not in lines[idx]
        return False

    def apply(self, source: str, finding: Finding) -> str:
        # Content-based: replace first unreplaced occurrence matching the snippet
        snippet = finding.code_snippet.strip()
        lines = source.splitlines(keepends=True)

        # Find by content (robust against line shifts from prior rules)
        target_idx = None
        for i, line in enumerate(lines):
            if "dtype=torch.float32" in line and line.strip() == snippet:
                target_idx = i
                break
        if target_idx is None:
            # Fallback: find any line with dtype=torch.float32 containing autocast
            idx = finding.line_number - 1
            if 0 <= idx < len(lines) and "dtype=torch.float32" in lines[idx]:
                target_idx = idx
            else:
                # Last resort: find first unreplaced autocast line
                for i, line in enumerate(lines):
                    if "dtype=torch.float32" in line and "autocast" in line:
                        target_idx = i
                        break
        if target_idx is not None:
            lines[target_idx] = lines[target_idx].replace("dtype=torch.float32", "dtype=torch.bfloat16")
        return "".join(lines)


class DtypeAssertRule(MigrationRule):
    """Downgrade ``assert X.dtype == torch.float32`` to a rank-0 warning.

    On NPU, autocast output dtypes may differ from CUDA (e.g. bfloat16 instead
    of float32).  Instead of crashing, emit a logging warning on rank 0 only.
    Injects ``import os`` and ``import logging`` at file top if missing.
    """
    name = "dtype_assert"
    description = "assert .dtype == torch.float32 → rank-0 logging.warning"
    pattern_type = PatternType.DTYPE_ASSERT

    _ASSERT_RE = re.compile(r"^(\s*)assert\s+(.+)$")

    def is_already_applied(self, source: str, finding: Finding) -> bool:
        snippet = finding.code_snippet.strip()
        # If the original assert line is no longer in the source, the rule was applied
        for line in source.splitlines():
            if line.strip() == snippet:
                return False
        return True

    def apply(self, source: str, finding: Finding) -> str:
        # Use the code snippet to locate the assert (robust against line shifts)
        snippet = finding.code_snippet.strip()
        lines = source.splitlines(keepends=True)

        # Find the line by content match (handles line-number drift from prior rules)
        target_idx = None
        for i, line in enumerate(lines):
            if line.strip() == snippet:
                target_idx = i
                break
        if target_idx is None:
            # Fallback: try original line number
            idx = finding.line_number - 1
            if 0 <= idx < len(lines) and self._ASSERT_RE.match(lines[idx].rstrip("\n")):
                target_idx = idx
            else:
                return source

        m = self._ASSERT_RE.match(lines[target_idx].rstrip("\n"))
        if not m:
            return source

        indent = m.group(1)
        condition = m.group(2)

        # Build rank-0 warning replacement
        replacement = (
            f'{indent}# [NPU] assert downgraded: NPU autocast may produce different dtypes\n'
            f'{indent}if not ({condition}):\n'
            f'{indent}    if os.environ.get("RANK", "0") == "0":\n'
            f'{indent}        logging.warning("[NPU] dtype assertion would have failed: {condition.replace(chr(34), chr(39))}")\n'
        )

        lines[target_idx] = replacement

        # NOTE: Do NOT inject imports here — it shifts all line numbers and
        # breaks subsequent rule applications in the same file.  Instead, mark
        # the file as needing imports; they'll be injected in a post-pass.
        # For safety, we inject a marker comment that the post-pass can find.
        source_out = "".join(lines)
        if "import os" not in source_out:
            source_out = "# __NEEDS_IMPORT_OS__\n" + source_out
        if "import logging" not in source_out:
            source_out = "# __NEEDS_IMPORT_LOGGING__\n" + source_out
        return source_out

    @staticmethod
    def _ensure_import(source: str, import_line: str, module_name: str) -> str:
        """Insert *import_line* after ``from __future__`` block if *module_name* not imported."""
        if f"import {module_name}" in source:
            return source
        lines = source.splitlines(keepends=True)
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("from __future__"):
                insert_idx = i + 1
            elif line.strip().startswith("import ") or line.strip().startswith("from "):
                if insert_idx == 0:
                    insert_idx = i
                break
            elif line.strip() and insert_idx == 0:
                insert_idx = i
                break
        lines.insert(insert_idx, import_line)
        return "".join(lines)


class FlashAttnUsageRule(MigrationRule):
    """Handle flash_attn usage sites: assert guards and function calls.

    - Assert guards (e.g. ``assert FLASH_ATTN_2_AVAILABLE``) → replaced with ``pass``
    - Varlen function calls (``flash_attn.flash_attn_varlen_func(...)``) →
      replaced with ``_npu_varlen_attention(...)`` using ``torch_npu.npu_fusion_attention``
      with TND input layout (handles variable-length sequences natively on NPU)
    - Other flash_attn calls → replaced with ``F.scaled_dot_product_attention(q, k, v)``
    """
    name = "flash_attn_usage"
    description = "Replace flash_attn assert guards and function calls"
    pattern_type = PatternType.FLASH_ATTN_USAGE

    _ASSERT_RE = re.compile(r"^(\s*)assert\s+\w*(?:flash_attn|FLASH_ATTN)\w*")
    _FUNC_CALL_RE = re.compile(r"flash_attn\.\w+\(")
    _VARLEN_RE = re.compile(r"flash_attn\.flash_attn_varlen_func\(")

    def is_already_applied(self, source: str, finding: Finding) -> bool:
        lines = source.splitlines()
        idx = finding.line_number - 1
        if 0 <= idx < len(lines):
            line = lines[idx]
            # Already applied if line is a comment (from previous application)
            if line.strip().startswith("# [NPU]"):
                return True
            # Already applied if no flash_attn assert or call remains
            has_assert = self._ASSERT_RE.search(line)
            has_call = self._FUNC_CALL_RE.search(line)
            return not has_assert and not has_call
        return False

    @staticmethod
    def _find_call_extent(lines: list[str], start_idx: int) -> tuple[int, str]:
        """Find the closing paren of a multi-line call starting at start_idx.

        Returns (end_idx, chained_suffix) where end_idx is the last line index
        of the call and chained_suffix is any text after the closing paren
        (e.g. ``.unflatten(0, (b, lq))``).
        """
        depth = 0
        for i in range(start_idx, len(lines)):
            for ch in lines[i]:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        # Find everything after the closing paren on this line
                        close_pos = lines[i].index(")")
                        suffix = lines[i][close_pos + 1:].rstrip("\n").rstrip()
                        return i, suffix
        # Fallback: couldn't find closing paren, return start line
        return start_idx, ""

    def _apply_varlen_npu(self, lines: list[str], idx: int) -> str:
        """Replace flash_attn_varlen_func with _npu_varlen_attention in-place.

        Does an in-place function-name substitution (preserving all original
        arguments), then marks the file for wrapper injection in the post-pass.
        """
        line = lines[idx]
        new_line = self._VARLEN_RE.sub("_npu_varlen_attention(", line.rstrip("\n"))
        if "# [NPU]" not in new_line:
            new_line += "  # [NPU] varlen -> npu_fusion_attention"
        lines[idx] = new_line + "\n"
        source = "".join(lines)
        if "def _npu_varlen_attention" not in source:
            source = _NPU_WRAPPER_MARKER + "\n" + source
        return source

    def apply(self, source: str, finding: Finding) -> str:
        lines = source.splitlines(keepends=True)
        idx = finding.line_number - 1
        if 0 <= idx < len(lines):
            line = lines[idx]
            indent = len(line) - len(line.lstrip())
            indent_str = " " * indent

            # Handle assert guards
            if self._ASSERT_RE.match(line):
                lines[idx] = f"{indent_str}pass  # [NPU] flash_attn assert guard removed\n"
                return "".join(lines)

            # Handle function calls
            if self._FUNC_CALL_RE.search(line):
                # Varlen calls → NPU fusion attention (TND layout)
                if self._VARLEN_RE.search(line):
                    return self._apply_varlen_npu(lines, idx)

                # Non-varlen flash_attn calls → SDPA replacement
                # Check if the call is multi-line
                stripped = line.rstrip()
                open_count = stripped.count("(") - stripped.count(")")
                if open_count > 0:
                    # Multi-line call — find extent and replace entire block
                    end_idx, chained = self._find_call_extent(lines, idx)

                    # Extract variable assignment from the first line
                    assignment_match = re.match(r"^(\s*\w[\w.]*\s*=\s*)", line)

                    # Comment out all lines of the original call
                    commented = []
                    for i in range(idx, end_idx + 1):
                        commented.append(f"{indent_str}# [NPU] {lines[i].strip()}\n")

                    # Build SDPA replacement
                    if assignment_match:
                        prefix = assignment_match.group(1).rstrip()
                        sdpa_line = f"{prefix} torch.nn.functional.scaled_dot_product_attention(q, k, v)"
                    else:
                        sdpa_line = f"{indent_str}torch.nn.functional.scaled_dot_product_attention(q, k, v)"

                    # Append chained methods (e.g. .unflatten(...))
                    if chained:
                        sdpa_line += chained

                    sdpa_line += "  # [NPU] SDPA replacement\n"

                    # Replace the multi-line block
                    lines[idx:end_idx + 1] = commented + [sdpa_line]
                    return "".join(lines)

                # Single-line call
                original = line.rstrip("\n")
                comment = f"{indent_str}# [NPU] {original.strip()}  # replaced with PyTorch native SDPA\n"
                assignment_match = re.match(r"^(\s*\w[\w.]*\s*=\s*)", line)
                if assignment_match:
                    prefix = assignment_match.group(1).rstrip()
                    replacement = f"{comment}{prefix} F.scaled_dot_product_attention(q, k, v)  # [NPU] SDPA\n"
                else:
                    replacement = f"{comment}{indent_str}F.scaled_dot_product_attention(q, k, v)  # [NPU] SDPA\n"
                lines[idx] = replacement
        return "".join(lines)


class AutocastDeviceRule(MigrationRule):
    """Inject ``'npu'`` as the first positional arg in ``autocast()`` calls.

    After ``CudaAmpRule`` migrates ``torch.cuda.amp`` → ``torch.amp``, the
    resulting ``amp.autocast(dtype=...)`` calls require an explicit
    ``device_type`` as the first positional argument.  Without it, PyTorch
    raises ``TypeError: missing required positional argument 'device_type'``.
    """
    name = "autocast_device"
    description = "autocast(dtype=...) → autocast('npu', dtype=...)"
    pattern_type = PatternType.AUTOCAST_NO_DEVICE

    _AUTOCAST_RE = re.compile(r"(\.autocast\()")

    def is_already_applied(self, source: str, finding: Finding) -> bool:
        lines = source.splitlines()
        idx = finding.line_number - 1
        if 0 <= idx < len(lines):
            line = lines[idx]
            # Check if autocast already has a string device_type arg
            if re.search(r"\.autocast\(\s*['\"]", line):
                return True
        return False

    def apply(self, source: str, finding: Finding) -> str:
        lines = source.splitlines(keepends=True)
        idx = finding.line_number - 1
        if 0 <= idx < len(lines):
            lines[idx] = self._AUTOCAST_RE.sub(r".autocast('npu', ", lines[idx])
        return "".join(lines)


# ---------------------------------------------------------------------------
# Dependency migration (requirements.txt)
# ---------------------------------------------------------------------------

# Package name normalization: pip treats - and _ as equivalent
_PKG_NAME_RE = re.compile(r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)")


def _normalize_pkg(name: str) -> str:
    """Normalize package name for comparison (lowercase, replace - with _)."""
    return name.lower().replace("-", "_")


def _parse_pkg_name(line: str) -> str | None:
    """Extract package name from a requirements.txt line. Returns None for comments/blanks."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("-"):
        return None
    m = _PKG_NAME_RE.match(stripped)
    return m.group(1) if m else None


def _parse_version_spec(line: str, pkg_name: str) -> str | None:
    """Extract version specifier from a requirements line (e.g. '>=2.1.0' from 'torch>=2.1.0')."""
    stripped = line.strip()
    after_name = stripped[len(pkg_name):]
    # Match ==, >=, <=, ~=, !=, >, <
    m = re.match(r"[=<>!~]+(.+)", after_name.strip())
    return m.group(1).strip() if m else None


def _extract_base_version(version_str: str) -> str | None:
    """Extract major.minor.patch from a version string like '2.1.0.post2'."""
    m = re.match(r"(\d+\.\d+\.\d+)", version_str)
    return m.group(1) if m else None


@runtime_checkable
class VersionResolver(Protocol):
    """Resolves available versions for a package from a package index."""

    def get_available_versions(self, package: str) -> list[str]:
        """Return available versions sorted newest-first."""
        ...


class PipVersionResolver:
    """Resolve versions via ``pip index versions <pkg>``."""

    def get_available_versions(self, package: str) -> list[str]:
        try:
            result = subprocess.run(
                ["pip", "index", "versions", package],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                return []
            # Output: "torch-npu (2.5.0)\n  Available versions: 2.5.0, 2.4.0, 2.3.0, ..."
            for output_line in result.stdout.splitlines():
                if "Available versions:" in output_line:
                    versions_str = output_line.split("Available versions:")[-1]
                    return [v.strip() for v in versions_str.split(",") if v.strip()]
            return []
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return []


class StaticVersionResolver:
    """Injectable version resolver for tests."""

    def __init__(self, versions: dict[str, list[str]]) -> None:
        self._versions = versions

    def get_available_versions(self, package: str) -> list[str]:
        return self._versions.get(package, [])


class DependencyMigrationRule(MigrationRule):
    """Migrate requirements.txt: prune NVIDIA packages, add NPU essentials, align torch versions.

    Operates on dependency manifest files (requirements.txt) rather than Python source.
    """

    name = "dependency_migration"
    description = "Prune NVIDIA deps, add NPU essentials, align torch/torch-npu versions"
    pattern_type = PatternType.DEPENDENCY_FILE

    # Packages to unconditionally remove
    BLACKLIST: frozenset[str] = frozenset({
        "flash_attn", "flash-attn",
        "xformers",
        "triton",
        "accelerate",
    })
    BLACKLIST_NORMALIZED: frozenset[str] = frozenset(
        _normalize_pkg(p) for p in BLACKLIST
    )

    # Packages to ensure present (value = version pin or None for latest)
    REQUIRED: dict[str, str | None] = {
        "imageio-ffmpeg": None,
    }

    def __init__(self, version_resolver: VersionResolver | None = None) -> None:
        self._resolver = version_resolver or PipVersionResolver()

    def is_already_applied(self, source: str, finding: Finding) -> bool:
        lines = source.splitlines()

        # Check 1: No blacklisted packages remain
        for line in lines:
            pkg = _parse_pkg_name(line)
            if pkg and _normalize_pkg(pkg) in self.BLACKLIST_NORMALIZED:
                return False

        # Check 2: All required packages present
        present = {_normalize_pkg(p) for line in lines if (p := _parse_pkg_name(line))}
        for req_pkg in self.REQUIRED:
            if _normalize_pkg(req_pkg) not in present:
                return False

        # Check 3: torch-npu present if torch is present
        if "torch" in present and "torch_npu" not in present:
            return False

        return True

    def apply(self, source: str, finding: Finding) -> str:
        lines = source.splitlines()
        result_lines: list[str] = []
        removed_pkgs: list[str] = []
        torch_version: str | None = None
        has_torch_npu = False
        present_pkgs: set[str] = set()

        # First pass: collect metadata
        for line in lines:
            pkg = _parse_pkg_name(line)
            if pkg:
                norm = _normalize_pkg(pkg)
                present_pkgs.add(norm)
                if norm == "torch":
                    torch_version = _parse_version_spec(line, pkg)
                if norm == "torch_npu":
                    has_torch_npu = True

        # Second pass: filter and transform
        for line in lines:
            pkg = _parse_pkg_name(line)
            if pkg and _normalize_pkg(pkg) in self.BLACKLIST_NORMALIZED:
                removed_pkgs.append(pkg)
                result_lines.append(f"# [NPU] removed: {line.strip()}")
                continue
            result_lines.append(line)

        # Version alignment: torch + torch-npu
        if "torch" in present_pkgs and torch_version:
            base_ver = _extract_base_version(torch_version)
            if base_ver:
                best_npu_ver = self._find_best_npu_version(base_ver)
                if best_npu_ver:
                    best_npu_base = _extract_base_version(best_npu_ver)
                    # Pin torch to aligned version if needed
                    if best_npu_base and best_npu_base != base_ver:
                        result_lines = self._replace_torch_version(result_lines, best_npu_base)
                    # Add or update torch-npu
                    npu_line = f"torch-npu=={best_npu_ver}"
                    if has_torch_npu:
                        result_lines = self._replace_torch_npu_line(result_lines, npu_line)
                    else:
                        result_lines.append(npu_line)
                elif not has_torch_npu:
                    # No versions available from resolver, add torch-npu with matching base version
                    result_lines.append(f"torch-npu=={base_ver}.*")
        elif "torch" in present_pkgs and not has_torch_npu:
            # torch present without version spec — just add torch-npu
            result_lines.append("torch-npu")

        # Add missing required packages
        for req_pkg, req_ver in self.REQUIRED.items():
            if _normalize_pkg(req_pkg) not in present_pkgs:
                if req_ver:
                    result_lines.append(f"{req_pkg}=={req_ver}")
                else:
                    result_lines.append(req_pkg)

        return "\n".join(result_lines) + "\n" if result_lines else ""

    def _find_best_npu_version(self, torch_base_ver: str) -> str | None:
        """Find the best torch-npu version matching the given torch base version."""
        available = self._resolver.get_available_versions("torch-npu")
        if not available:
            return None

        # Exact prefix match first
        exact_matches = [v for v in available if v.startswith(torch_base_ver)]
        if exact_matches:
            return exact_matches[0]  # newest first

        # No exact match — find closest higher version
        from packaging.version import Version, InvalidVersion

        try:
            target = Version(torch_base_ver)
        except InvalidVersion:
            return available[0] if available else None

        higher: list[tuple[Version, str]] = []
        for v_str in available:
            try:
                v = Version(v_str)
                if v >= target:
                    higher.append((v, v_str))
            except InvalidVersion:
                continue

        if higher:
            higher.sort(key=lambda x: x[0])
            return higher[0][1]  # closest higher

        # All available are lower — return newest
        return available[0]

    @staticmethod
    def _replace_torch_version(lines: list[str], new_base: str) -> list[str]:
        """Replace the torch version pin in the lines list."""
        result = []
        for line in lines:
            pkg = _parse_pkg_name(line)
            if pkg and _normalize_pkg(pkg) == "torch":
                result.append(f"torch=={new_base}")
            else:
                result.append(line)
        return result

    @staticmethod
    def _replace_torch_npu_line(lines: list[str], new_line: str) -> list[str]:
        """Replace the torch-npu line in the lines list."""
        result = []
        for line in lines:
            pkg = _parse_pkg_name(line)
            if pkg and _normalize_pkg(pkg) == "torch_npu":
                result.append(new_line)
            else:
                result.append(line)
        return result


class Float64Rule(MigrationRule):
    name = "float64"
    description = "torch.float64/.double() → torch.float32 + warning"
    pattern_type = PatternType.FLOAT64

    def is_already_applied(self, source: str, finding: Finding) -> bool:
        lines = source.splitlines()
        idx = finding.line_number - 1
        if 0 <= idx < len(lines):
            line = lines[idx]
            return "torch.float64" not in line and ".double()" not in line
        return False

    def apply(self, source: str, finding: Finding) -> str:
        lines = source.splitlines(keepends=True)
        idx = finding.line_number - 1
        if 0 <= idx < len(lines):
            line = lines[idx]
            line = line.replace("torch.float64", "torch.float32")
            line = line.replace(".double()", ".float()")
            if "# [NPU]" not in line:
                line = line.rstrip("\n") + "  # [NPU] float64→float32\n"
            lines[idx] = line
        return "".join(lines)


# ---------------------------------------------------------------------------
# Factory: default registry with all built-in rules
# ---------------------------------------------------------------------------

_BUILTIN_RULES: list[type[MigrationRule]] = [
    DependencyMigrationRule,
    CudaCallRule,
    CudaToRule,
    CudaApiRule,
    CudaAmpRule,
    AutocastDeviceRule,
    CudaDeviceStrRule,
    NcclToHcclRule,
    FlashAttnRule,
    FlashAttnUsageRule,
    XformersRule,
    Float64Rule,
    NpuInitInjectorRule,
    AutocastDtypeRule,
    DtypeAssertRule,
]


def create_default_registry() -> RuleRegistry:
    """Create a registry pre-loaded with all built-in rules."""
    registry = RuleRegistry()
    for cls in _BUILTIN_RULES:
        registry.register(cls())
    return registry


# ---------------------------------------------------------------------------
# Migration application
# ---------------------------------------------------------------------------

def _resolve_import_markers(source: str) -> str:
    """Replace deferred import markers with actual imports at the right location."""
    needs_os = "# __NEEDS_IMPORT_OS__" in source
    needs_logging = "# __NEEDS_IMPORT_LOGGING__" in source

    if not needs_os and not needs_logging:
        return source

    # Remove markers
    source = source.replace("# __NEEDS_IMPORT_OS__\n", "")
    source = source.replace("# __NEEDS_IMPORT_LOGGING__\n", "")

    # Insert actual imports after __future__ or at file top
    imports_to_add: list[str] = []
    if needs_os and "import os" not in source:
        imports_to_add.append("import os\n")
    if needs_logging and "import logging" not in source:
        imports_to_add.append("import logging\n")

    if not imports_to_add:
        return source

    lines = source.splitlines(keepends=True)
    insert_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("from __future__"):
            insert_idx = i + 1
        elif stripped.startswith("import ") or stripped.startswith("from "):
            if insert_idx == 0:
                insert_idx = i
            break
        elif stripped and insert_idx == 0:
            insert_idx = i
            break

    for j, imp in enumerate(imports_to_add):
        lines.insert(insert_idx + j, imp)
    return "".join(lines)


def _resolve_npu_wrapper_marker(source: str) -> str:
    """Replace deferred NPU varlen attention wrapper marker with the actual function."""
    if _NPU_WRAPPER_MARKER not in source:
        return source

    # Remove marker
    source = source.replace(_NPU_WRAPPER_MARKER + "\n", "")

    # Skip if wrapper already present
    if "def _npu_varlen_attention" in source:
        return source

    # Find insertion point: before first top-level def/class
    lines = source.splitlines(keepends=True)
    insert_idx = len(lines)
    for i, line in enumerate(lines):
        if line.startswith("def ") or line.startswith("class "):
            insert_idx = i
            break

    # Insert wrapper
    wrapper_lines = _NPU_VARLEN_WRAPPER.splitlines(keepends=True)
    for j, wl in enumerate(wrapper_lines):
        lines.insert(insert_idx + j, wl if wl.endswith("\n") else wl + "\n")

    return "".join(lines)


def _file_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def apply_migration(file_path: str, migrations: list[tuple[Finding, MigrationRule]]) -> MigrationResult:
    """Apply rules to one file, bottom-up by line number. Creates .bak backup."""
    path = Path(file_path)
    try:
        original = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return MigrationResult(
            file_path=file_path, applied_rules=[], original_hash="",
            success=False, error=str(exc),
        )

    original_hash = _file_hash(original)

    # Create backup
    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(original, encoding="utf-8")

    # Sort bottom-up so line numbers stay valid
    sorted_migrations = sorted(migrations, key=lambda m: m[0].line_number, reverse=True)

    source = original
    applied: list[str] = []
    for finding, rule in sorted_migrations:
        if rule.is_already_applied(source, finding):
            log.debug("rule_already_applied", rule=rule.name, file=file_path)
            continue
        source = rule.apply(source, finding)
        applied.append(rule.name)

    # Post-pass: resolve deferred markers
    source = _resolve_import_markers(source)
    source = _resolve_npu_wrapper_marker(source)

    path.write_text(source, encoding="utf-8")
    log.info("migration_applied", file=file_path, rules=applied)

    return MigrationResult(
        file_path=file_path,
        applied_rules=applied,
        original_hash=original_hash,
        success=True,
    )


def apply_all_migrations(plan: MigrationPlan) -> list[MigrationResult]:
    """Apply all migrations from a plan."""
    results: list[MigrationResult] = []
    for file_path, migrations in plan.matched.items():
        result = apply_migration(file_path, migrations)
        results.append(result)
    return results


def add_torch_npu_import(file_path: str) -> bool:
    """Add ``import torch_npu`` and ``from torch_npu.contrib import transfer_to_npu``
    after ``import torch`` if not already present.

    ``transfer_to_npu`` is a monkey-patching utility that transparently redirects
    CUDA tensor/module operations to NPU, solving ``Torch not compiled with CUDA
    enabled`` errors at runtime.

    Returns True if the file was modified, False if both imports already exist.
    """
    path = Path(file_path)
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False

    has_torch_npu = "import torch_npu" in source
    has_transfer = "from torch_npu.contrib import transfer_to_npu" in source

    if has_torch_npu and has_transfer:
        return False  # both already present

    lines = source.splitlines(keepends=True)
    modified = False

    if not has_torch_npu:
        # Find `import torch` and insert both lines after it
        insert_idx: int | None = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "import torch" or stripped.startswith("import torch "):
                insert_idx = i + 1
                break
            if stripped.startswith("from torch ") or stripped == "from torch":
                insert_idx = i + 1
                break

        if insert_idx is None:
            return False

        lines.insert(insert_idx, "import torch_npu\n")
        lines.insert(insert_idx + 1, "from torch_npu.contrib import transfer_to_npu\n")
        modified = True

    elif not has_transfer:
        # torch_npu exists but transfer_to_npu missing — insert after torch_npu
        for i, line in enumerate(lines):
            if line.strip() == "import torch_npu":
                lines.insert(i + 1, "from torch_npu.contrib import transfer_to_npu\n")
                modified = True
                break

    if modified:
        path.write_text("".join(lines), encoding="utf-8")

    return modified
