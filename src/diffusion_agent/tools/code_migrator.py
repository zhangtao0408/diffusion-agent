"""Extensible CUDA→NPU migration engine with a rule registry."""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from diffusion_agent.tools.code_scanner import Finding, PatternType
from diffusion_agent.utils.logging import get_logger

log = get_logger(__name__)


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


class CudaDeviceStrRule(MigrationRule):
    name = "cuda_device_str"
    description = '"cuda" string literals → "npu" (torch.device, defaults, etc.)'
    pattern_type = PatternType.CUDA_DEVICE_STR

    def apply(self, source: str, finding: Finding) -> str:
        lines = source.splitlines(keepends=True)
        idx = finding.line_number - 1
        if 0 <= idx < len(lines):
            # Replace "cuda:N" first, then "cuda" (order matters)
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
            # Handle .startswith("cuda") → .startswith("npu")
            lines[idx] = lines[idx].replace('.startswith("cuda")', '.startswith("npu")')
            lines[idx] = lines[idx].replace(".startswith('cuda')", ".startswith('npu')")
        return "".join(lines)


class Float64Rule(MigrationRule):
    name = "float64"
    description = "torch.float64/.double() → torch.float32 + warning"
    pattern_type = PatternType.FLOAT64

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
    CudaCallRule,
    CudaToRule,
    CudaApiRule,
    CudaDeviceStrRule,
    NcclToHcclRule,
    FlashAttnRule,
    XformersRule,
    Float64Rule,
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
        source = rule.apply(source, finding)
        applied.append(rule.name)

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
    """Add `import torch_npu` after `import torch` if not already present."""
    path = Path(file_path)
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False

    if "import torch_npu" in source:
        return False  # already present

    lines = source.splitlines(keepends=True)
    insert_idx: int | None = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Match `import torch` but not `import torch_*` or `import torchvision`
        if stripped == "import torch" or stripped.startswith("import torch "):
            insert_idx = i + 1
            break
        if stripped.startswith("from torch ") or stripped == "from torch":
            insert_idx = i + 1
            break

    if insert_idx is None:
        return False

    lines.insert(insert_idx, "import torch_npu\n")
    path.write_text("".join(lines), encoding="utf-8")
    return True
