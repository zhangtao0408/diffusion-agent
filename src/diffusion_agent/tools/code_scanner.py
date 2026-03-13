"""AST-based scanner to detect CUDA and other Ascend-incompatible patterns in Python files."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from diffusion_agent.utils.logging import get_logger

log = get_logger(__name__)


class PatternType(str, Enum):
    CUDA_CALL = "cuda_call"       # .cuda()
    CUDA_TO = "cuda_to"           # .to("cuda") / .to(device)
    CUDA_API = "cuda_api"         # torch.cuda.*
    CUDA_DEVICE_STR = "cuda_device_str"  # "cuda" string literals (torch.device, defaults, etc.)
    FLOAT64 = "float64"           # float64 / double
    NCCL = "nccl"                 # nccl backend
    FLASH_ATTN = "flash_attn"     # flash_attn imports
    XFORMERS = "xformers"         # xformers imports
    BFLOAT16 = "bfloat16"         # bfloat16 usage
    DISTRIBUTED = "distributed"   # torch.distributed usage
    SDPA = "sdpa"                 # scaled_dot_product_attention


@dataclass
class Finding:
    file_path: str
    line_number: int
    pattern_type: PatternType
    code_snippet: str
    context: str | None = field(default=None)


class _PatternVisitor(ast.NodeVisitor):
    """Walk an AST and collect findings."""

    def __init__(self, file_path: str, lines: list[str]) -> None:
        self.file_path = file_path
        self.lines = lines
        self.findings: list[Finding] = []
        self._cuda_to_lines: set[int] = set()  # lines already reported as CUDA_TO

    def _add(self, node: ast.AST, ptype: PatternType) -> None:
        lineno = getattr(node, "lineno", 0)
        snippet = self.lines[lineno - 1].rstrip() if 0 < lineno <= len(self.lines) else ""
        self.findings.append(
            Finding(
                file_path=self.file_path,
                line_number=lineno,
                pattern_type=ptype,
                code_snippet=snippet,
            )
        )

    # --- .cuda() -----------------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr

            # .cuda(...)
            if attr == "cuda":
                self._add(node, PatternType.CUDA_CALL)

            # .to("cuda") / .to("cuda:N")
            elif attr == "to":
                self._check_to_call(node)

            # .double()
            elif attr == "double":
                self._add(node, PatternType.FLOAT64)

            # scaled_dot_product_attention
            elif attr == "scaled_dot_product_attention":
                self._add(node, PatternType.SDPA)

            # init_process_group (distributed)
            elif attr == "init_process_group":
                self._add(node, PatternType.DISTRIBUTED)

        # torch.cuda.* calls  (e.g. torch.cuda.is_available())
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Attribute):
            val = node.func.value
            if (
                isinstance(val.value, ast.Name)
                and val.value.id == "torch"
                and val.attr == "cuda"
            ):
                self._add(node, PatternType.CUDA_API)

        self.generic_visit(node)

    def _check_to_call(self, node: ast.Call) -> None:
        """Detect .to("cuda"), .to("cuda:0"), .to(device), .to(torch.float64), .to(torch.bfloat16)."""
        for arg in node.args:
            # String literal containing "cuda"
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if "cuda" in arg.value:
                    self._add(node, PatternType.CUDA_TO)
                    self._cuda_to_lines.add(getattr(node, "lineno", 0))
                    return
            # Variable named 'device'
            if isinstance(arg, ast.Name) and arg.id == "device":
                self._add(node, PatternType.CUDA_TO)
                return
            # torch.float64 / torch.bfloat16
            if isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name):
                if arg.value.id == "torch" and arg.attr == "float64":
                    self._add(node, PatternType.FLOAT64)
                    return
                if arg.value.id == "torch" and arg.attr == "bfloat16":
                    self._add(node, PatternType.BFLOAT16)
                    return

    # --- torch.cuda.* attribute access (non-call) --------------------------
    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name):
            if node.value.value.id == "torch" and node.value.attr == "cuda":
                # Only flag if this node is NOT already the func of a Call
                # (Call case is handled in visit_Call)
                pass  # visit_Call handles all torch.cuda.* calls
        self.generic_visit(node)

    # --- string literals: "nccl", "bfloat16", "cuda" ------------------------
    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: N802
        if isinstance(node.value, str):
            if node.value == "nccl":
                self._add(node, PatternType.NCCL)
            elif node.value == "bfloat16":
                self._add(node, PatternType.BFLOAT16)
            elif "cuda" in node.value:
                # Avoid double-counting lines already caught as CUDA_TO
                lineno = getattr(node, "lineno", 0)
                if lineno not in self._cuda_to_lines:
                    self._add(node, PatternType.CUDA_DEVICE_STR)
        self.generic_visit(node)

    # --- imports: flash_attn, xformers, torch.distributed -----------------
    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            name = alias.name
            if name == "flash_attn" or name.startswith("flash_attn."):
                self._add(node, PatternType.FLASH_ATTN)
            elif name == "xformers" or name.startswith("xformers."):
                self._add(node, PatternType.XFORMERS)
            elif name == "torch.distributed":
                self._add(node, PatternType.DISTRIBUTED)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        module = node.module or ""
        if module == "flash_attn" or module.startswith("flash_attn."):
            self._add(node, PatternType.FLASH_ATTN)
        elif module == "xformers" or module.startswith("xformers."):
            self._add(node, PatternType.XFORMERS)
        self.generic_visit(node)


def scan_file(path: Path) -> list[Finding]:
    """Scan a single Python file for CUDA/Ascend-incompatible patterns."""
    try:
        source = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, ValueError):
        log.warning("skipping_non_text_file", path=str(path))
        return []

    if not source.strip():
        return []

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        log.warning("skipping_syntax_error", path=str(path))
        return []

    lines = source.splitlines()
    visitor = _PatternVisitor(file_path=str(path), lines=lines)
    visitor.visit(tree)
    return visitor.findings


def scan_directory(path: Path, pattern: str = "**/*.py") -> list[Finding]:
    """Recursively scan Python files under *path* for CUDA patterns."""
    findings: list[Finding] = []
    for py_file in sorted(path.glob(pattern)):
        if py_file.is_file():
            findings.extend(scan_file(py_file))
    return findings
