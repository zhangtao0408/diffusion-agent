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
    FLOAT64 = "float64"           # float64 / double
    NCCL = "nccl"                 # nccl backend


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
        """Detect .to("cuda"), .to("cuda:0"), .to(device), .to(torch.float64)."""
        for arg in node.args:
            # String literal containing "cuda"
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if "cuda" in arg.value:
                    self._add(node, PatternType.CUDA_TO)
                    return
            # Variable named 'device'
            if isinstance(arg, ast.Name) and arg.id == "device":
                self._add(node, PatternType.CUDA_TO)
                return
            # torch.float64
            if isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name):
                if arg.value.id == "torch" and arg.attr == "float64":
                    self._add(node, PatternType.FLOAT64)
                    return

    # --- torch.cuda.* attribute access (non-call) --------------------------
    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name):
            if node.value.value.id == "torch" and node.value.attr == "cuda":
                # Only flag if this node is NOT already the func of a Call
                # (Call case is handled in visit_Call)
                pass  # visit_Call handles all torch.cuda.* calls
        self.generic_visit(node)

    # --- string literals: "nccl", "float64" --------------------------------
    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: N802
        if isinstance(node.value, str):
            if node.value == "nccl":
                self._add(node, PatternType.NCCL)
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
