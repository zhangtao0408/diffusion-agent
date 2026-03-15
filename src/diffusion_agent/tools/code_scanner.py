"""AST-based scanner to detect CUDA and other Ascend-incompatible patterns in Python files."""

from __future__ import annotations

import ast
import re
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
    CUDA_AMP = "cuda_amp"         # torch.cuda.amp imports/usage
    TORCH_IMPORT = "torch_import" # import torch (for NPU init injection)
    AUTOCAST_DTYPE = "autocast_dtype"  # autocast with dtype=torch.float32
    DTYPE_ASSERT = "dtype_assert"      # assert ... .dtype == torch.float32
    FLASH_ATTN_USAGE = "flash_attn_usage"  # flash_attn assert guards and function calls
    AUTOCAST_NO_DEVICE = "autocast_no_device"  # autocast() missing device_type arg
    DEPENDENCY_FILE = "dependency_file"  # requirements.txt / dependency manifests


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

        # autocast(..., dtype=torch.float32)
        self._check_autocast_dtype(node)

        # autocast() missing device_type (after CudaAmpRule migrates torch.cuda.amp → torch.amp)
        self._check_autocast_no_device(node)

        self.generic_visit(node)

    def _check_autocast_dtype(self, node: ast.Call) -> None:
        """Detect autocast calls with dtype=torch.float32."""
        # Match torch.amp.autocast(..., dtype=torch.float32)
        func = node.func
        is_autocast = False
        if isinstance(func, ast.Attribute) and func.attr == "autocast":
            is_autocast = True
        if not is_autocast:
            return
        for kw in node.keywords:
            if kw.arg == "dtype" and isinstance(kw.value, ast.Attribute):
                if (
                    isinstance(kw.value.value, ast.Name)
                    and kw.value.value.id == "torch"
                    and kw.value.attr == "float32"
                ):
                    self._add(node, PatternType.AUTOCAST_DTYPE)
                    return

    def _check_autocast_no_device(self, node: ast.Call) -> None:
        """Detect autocast() calls missing device_type as first positional arg.

        After CudaAmpRule migrates ``torch.cuda.amp`` → ``torch.amp``, the
        resulting ``amp.autocast(dtype=...)`` calls need an explicit
        ``device_type='npu'`` as the first positional argument.  This method
        detects such calls while excluding ``torch.cuda.amp.autocast`` (which
        is handled separately by CudaAmpRule).
        """
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "autocast"):
            return

        # Exclude torch.cuda.amp.autocast — handled by CudaAmpRule
        if isinstance(func.value, ast.Attribute):
            val = func.value
            # torch.cuda.amp.autocast  (val = torch.cuda.amp, val.attr = amp)
            if isinstance(val.value, ast.Attribute):
                inner = val.value
                if (
                    isinstance(inner.value, ast.Name)
                    and inner.value.id == "torch"
                    and inner.attr == "cuda"
                ):
                    return

        # Check if the first positional arg is a string constant (device_type)
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            return  # Already has device_type like 'npu' or 'cuda'

        # If we get here, autocast() is called without a device_type string
        # Only flag if it has at least one keyword arg (otherwise it's likely not an autocast call we care about)
        if node.keywords or node.args:
            self._add(node, PatternType.AUTOCAST_NO_DEVICE)

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

    # --- assert ... .dtype == torch.float32 ---------------------------------
    def visit_Assert(self, node: ast.Assert) -> None:  # noqa: N802
        """Detect ``assert ... .dtype == torch.float32`` patterns."""
        if node.test and self._has_dtype_float32_compare(node.test):
            self._add(node, PatternType.DTYPE_ASSERT)
        self.generic_visit(node)

    def _has_dtype_float32_compare(self, node: ast.AST) -> bool:
        """Check if a node contains a ``.dtype == torch.float32`` comparison."""
        if isinstance(node, ast.Compare):
            return self._is_dtype_float32_compare(node)
        if isinstance(node, ast.BoolOp):
            return any(self._has_dtype_float32_compare(v) for v in node.values)
        return False

    def _is_dtype_float32_compare(self, node: ast.Compare) -> bool:
        """Check ``X.dtype == torch.float32``."""
        # Left side: *.dtype
        left = node.left
        if isinstance(left, ast.Attribute) and left.attr == "dtype":
            for op, comparator in zip(node.ops, node.comparators):
                if isinstance(op, ast.Eq) and isinstance(comparator, ast.Attribute):
                    if (
                        isinstance(comparator.value, ast.Name)
                        and comparator.value.id == "torch"
                        and comparator.attr == "float32"
                    ):
                        return True
        return False

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
            elif name == "torch.cuda.amp" or name.startswith("torch.cuda.amp."):
                self._add(node, PatternType.CUDA_AMP)
            elif name == "torch":
                self._add(node, PatternType.TORCH_IMPORT)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        module = node.module or ""
        if module == "flash_attn" or module.startswith("flash_attn."):
            self._add(node, PatternType.FLASH_ATTN)
        elif module == "xformers" or module.startswith("xformers."):
            self._add(node, PatternType.XFORMERS)
        elif module == "torch.cuda.amp" or module.startswith("torch.cuda.amp."):
            self._add(node, PatternType.CUDA_AMP)
        elif module == "torch":
            self._add(node, PatternType.TORCH_IMPORT)
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

    # Regex post-pass: detect flash_attn usage patterns (assert guards, function calls)
    _FLASH_ATTN_USAGE_RE = re.compile(
        r"assert\s+\w*(?:flash_attn|FLASH_ATTN)\w*|flash_attn\.\w+\("
    )
    flash_attn_import_lines = {
        f.line_number for f in visitor.findings
        if f.pattern_type == PatternType.FLASH_ATTN
    }
    for i, line in enumerate(lines, start=1):
        if i in flash_attn_import_lines:
            continue
        # Skip comment lines — commented-out code shouldn't generate findings
        if line.lstrip().startswith("#"):
            continue
        if _FLASH_ATTN_USAGE_RE.search(line):
            visitor.findings.append(
                Finding(
                    file_path=str(path),
                    line_number=i,
                    pattern_type=PatternType.FLASH_ATTN_USAGE,
                    code_snippet=line.rstrip(),
                )
            )

    return visitor.findings


def scan_dependency_files(path: Path) -> list[Finding]:
    """Scan for dependency manifest files (requirements.txt, etc.) under *path*.

    Returns one Finding per discovered file with ``PatternType.DEPENDENCY_FILE``.
    """
    findings: list[Finding] = []
    # Patterns to search for dependency files
    globs = [
        "requirements.txt",
        "requirements/*.txt",
        "requirements-*.txt",
        "requirements_*.txt",
    ]
    seen: set[Path] = set()
    for g in globs:
        for dep_file in sorted(path.glob(g)):
            resolved = dep_file.resolve()
            if not dep_file.is_file() or resolved in seen:
                continue
            seen.add(resolved)
            try:
                content = dep_file.read_text(encoding="utf-8").strip()
            except (UnicodeDecodeError, ValueError):
                continue
            if not content:
                continue
            first_line = content.splitlines()[0]
            findings.append(
                Finding(
                    file_path=str(dep_file),
                    line_number=1,
                    pattern_type=PatternType.DEPENDENCY_FILE,
                    code_snippet=first_line,
                )
            )
    return findings


def scan_directory(
    path: Path, pattern: str = "**/*.py", *, include_deps: bool = True,
) -> list[Finding]:
    """Recursively scan Python files under *path* for CUDA patterns.

    When *include_deps* is True (default), also scans for dependency manifest
    files (``requirements.txt``, etc.) via :func:`scan_dependency_files`.
    """
    findings: list[Finding] = []
    for py_file in sorted(path.glob(pattern)):
        if py_file.is_file():
            findings.extend(scan_file(py_file))
    if include_deps:
        findings.extend(scan_dependency_files(path))
    return findings
