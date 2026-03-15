"""Unit tests for the code_migrator — rule registry + built-in rules."""

from __future__ import annotations

from pathlib import Path

from diffusion_agent.tools.code_migrator import (
    AutocastDeviceRule,
    AutocastDtypeRule,
    CudaAmpRule,
    CudaApiRule,
    CudaCallRule,
    CudaDeviceStrRule,
    CudaToRule,
    DependencyMigrationRule,
    DtypeAssertRule,
    Float64Rule,
    FlashAttnRule,
    FlashAttnUsageRule,
    MigrationRule,
    NcclToHcclRule,
    NpuInitInjectorRule,
    RuleRegistry,
    StaticVersionResolver,
    XformersRule,
    add_torch_npu_import,
    apply_all_migrations,
    apply_migration,
    create_default_registry,
)
from diffusion_agent.tools.code_scanner import Finding, PatternType


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


# ---------------------------------------------------------------------------
# Rule registry tests
# ---------------------------------------------------------------------------

class TestRuleRegistry:
    def test_register_rule(self) -> None:
        registry = RuleRegistry()
        rule = CudaCallRule()
        registry.register(rule)
        assert rule in registry.get_rules()

    def test_unregister_rule(self) -> None:
        registry = RuleRegistry()
        registry.register(CudaCallRule())
        registry.unregister("cuda_call")
        assert len(registry.get_rules()) == 0

    def test_match_finding_to_rule(self) -> None:
        registry = create_default_registry()
        for ptype, expected_name in [
            (PatternType.CUDA_CALL, "cuda_call"),
            (PatternType.CUDA_TO, "cuda_to"),
            (PatternType.CUDA_API, "cuda_api"),
            (PatternType.CUDA_AMP, "cuda_amp"),
            (PatternType.CUDA_DEVICE_STR, "cuda_device_str"),
            (PatternType.NCCL, "nccl_to_hccl"),
            (PatternType.FLASH_ATTN, "flash_attn"),
            (PatternType.FLASH_ATTN_USAGE, "flash_attn_usage"),
            (PatternType.XFORMERS, "xformers"),
            (PatternType.FLOAT64, "float64"),
            (PatternType.TORCH_IMPORT, "npu_init_injector"),
            (PatternType.AUTOCAST_NO_DEVICE, "autocast_device"),
            (PatternType.AUTOCAST_DTYPE, "autocast_dtype"),
            (PatternType.DTYPE_ASSERT, "dtype_assert"),
        ]:
            finding = _finding("/tmp/x.py", 1, ptype)
            rule = registry.match(finding)
            assert rule is not None, f"No rule matched for {ptype}"
            assert rule.name == expected_name

    def test_unmatched_findings(self) -> None:
        registry = create_default_registry()
        # BFLOAT16 has no built-in rule
        findings = [_finding("/tmp/x.py", 1, PatternType.BFLOAT16)]
        plan = registry.match_all(findings)
        assert len(plan.unmatched) == 1
        assert plan.total_migrations == 0

    def test_default_registry_has_all_builtins(self) -> None:
        registry = create_default_registry()
        assert len(registry.get_rules()) == 15


# ---------------------------------------------------------------------------
# Individual rule tests
# ---------------------------------------------------------------------------

class TestCudaCallRule:
    def test_cuda_to_npu(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", "x = tensor.cuda()\n")
        finding = _finding(str(p), 1, PatternType.CUDA_CALL)
        rule = CudaCallRule()
        result = rule.apply(p.read_text(), finding)
        assert ".npu()" in result
        assert ".cuda()" not in result


class TestCudaToRule:
    def test_to_cuda_string(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", 'model.to("cuda")\n')
        finding = _finding(str(p), 1, PatternType.CUDA_TO)
        result = CudaToRule().apply(p.read_text(), finding)
        assert '"npu"' in result
        assert '"cuda"' not in result

    def test_to_cuda_device(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", 'model.to("cuda:0")\n')
        finding = _finding(str(p), 1, PatternType.CUDA_TO)
        result = CudaToRule().apply(p.read_text(), finding)
        assert '"npu:0"' in result
        assert '"cuda:0"' not in result


class TestCudaApiRule:
    def test_torch_cuda_to_npu(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", "if torch.cuda.is_available():\n    pass\n")
        finding = _finding(str(p), 1, PatternType.CUDA_API)
        result = CudaApiRule().apply(p.read_text(), finding)
        assert "torch.npu.is_available()" in result
        assert "torch.cuda" not in result


class TestCudaDeviceStrRule:
    def test_torch_device_cuda(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", 'device = torch.device("cuda")\n')
        finding = _finding(str(p), 1, PatternType.CUDA_DEVICE_STR)
        result = CudaDeviceStrRule().apply(p.read_text(), finding)
        assert '"npu"' in result
        assert '"cuda"' not in result

    def test_cuda_default_param(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", 'def foo(device="cuda"):\n    pass\n')
        finding = _finding(str(p), 1, PatternType.CUDA_DEVICE_STR)
        result = CudaDeviceStrRule().apply(p.read_text(), finding)
        assert '"npu"' in result
        assert '"cuda"' not in result

    def test_startswith_cuda(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", 'if str(device).startswith("cuda"):\n    pass\n')
        finding = _finding(str(p), 1, PatternType.CUDA_DEVICE_STR)
        result = CudaDeviceStrRule().apply(p.read_text(), finding)
        assert '.startswith("npu")' in result

    # --- Task 2: Enhanced device string handling ---

    def test_cuda_with_space_before_colon(self, tmp_path: Path) -> None:
        """Handle 'cuda :0' (space before colon)."""
        p = _write(tmp_path, "a.py", 'device = torch.device("cuda :0")\n')
        finding = _finding(str(p), 1, PatternType.CUDA_DEVICE_STR)
        result = CudaDeviceStrRule().apply(p.read_text(), finding)
        assert "cuda" not in result
        assert "npu" in result

    def test_cuda_with_space_after_colon(self, tmp_path: Path) -> None:
        """Handle 'cuda: 0' (space after colon)."""
        p = _write(tmp_path, "a.py", 'device = torch.device("cuda: 0")\n')
        finding = _finding(str(p), 1, PatternType.CUDA_DEVICE_STR)
        result = CudaDeviceStrRule().apply(p.read_text(), finding)
        assert "cuda" not in result
        assert "npu" in result

    def test_cuda_with_spaces_around_colon(self, tmp_path: Path) -> None:
        """Handle 'cuda : 0' (spaces both sides)."""
        p = _write(tmp_path, "a.py", 'device = torch.device("cuda : 0")\n')
        finding = _finding(str(p), 1, PatternType.CUDA_DEVICE_STR)
        result = CudaDeviceStrRule().apply(p.read_text(), finding)
        assert "cuda" not in result
        assert "npu" in result

    def test_fstring_cuda_device_id(self, tmp_path: Path) -> None:
        """Handle f'cuda:{device_id}' → f'npu:{device_id}'."""
        p = _write(tmp_path, "a.py", 'self.device = torch.device(f"cuda:{device_id}")\n')
        finding = _finding(str(p), 1, PatternType.CUDA_DEVICE_STR)
        result = CudaDeviceStrRule().apply(p.read_text(), finding)
        assert "cuda" not in result
        assert "npu" in result

    def test_fstring_cuda_bare(self, tmp_path: Path) -> None:
        """Handle f'cuda:{rank}' where cuda is a prefix."""
        p = _write(tmp_path, "a.py", 'device = f"cuda:{rank}"\n')
        finding = _finding(str(p), 1, PatternType.CUDA_DEVICE_STR)
        result = CudaDeviceStrRule().apply(p.read_text(), finding)
        assert "cuda" not in result
        assert "npu" in result

    def test_torch_device_two_args(self, tmp_path: Path) -> None:
        """Handle torch.device('cuda', device_id) → torch.device('npu', device_id)."""
        p = _write(tmp_path, "a.py", 'dev = torch.device("cuda", device_id)\n')
        finding = _finding(str(p), 1, PatternType.CUDA_DEVICE_STR)
        result = CudaDeviceStrRule().apply(p.read_text(), finding)
        assert '"npu"' in result
        assert '"cuda"' not in result


class TestAutocastDtypeRule:
    """Replace dtype=torch.float32 with dtype=torch.bfloat16 in autocast calls."""

    def test_autocast_float32_to_bfloat16(self, tmp_path: Path) -> None:
        code = "with torch.amp.autocast('npu', dtype=torch.float32):\n    pass\n"
        p = _write(tmp_path, "a.py", code)
        finding = _finding(str(p), 1, PatternType.AUTOCAST_DTYPE)
        result = AutocastDtypeRule().apply(p.read_text(), finding)
        assert "dtype=torch.bfloat16" in result
        assert "dtype=torch.float32" not in result

    def test_preserves_other_args(self, tmp_path: Path) -> None:
        code = "with torch.amp.autocast('npu', dtype=torch.float32, enabled=True):\n    pass\n"
        p = _write(tmp_path, "a.py", code)
        finding = _finding(str(p), 1, PatternType.AUTOCAST_DTYPE)
        result = AutocastDtypeRule().apply(p.read_text(), finding)
        assert "dtype=torch.bfloat16" in result
        assert "enabled=True" in result

    def test_decorator_autocast(self, tmp_path: Path) -> None:
        code = "@torch.amp.autocast('npu', dtype=torch.float32)\ndef foo(): pass\n"
        p = _write(tmp_path, "a.py", code)
        finding = _finding(str(p), 1, PatternType.AUTOCAST_DTYPE)
        result = AutocastDtypeRule().apply(p.read_text(), finding)
        assert "dtype=torch.bfloat16" in result
        assert "dtype=torch.float32" not in result


class TestDtypeAssertRule:
    """Downgrade assert ... .dtype == torch.float32 to rank-0 warning."""

    def test_single_assert_downgraded(self, tmp_path: Path) -> None:
        code = "    assert e.dtype == torch.float32\n"
        p = _write(tmp_path, "a.py", code)
        finding = _finding(str(p), 1, PatternType.DTYPE_ASSERT)
        result = DtypeAssertRule().apply(p.read_text(), finding)
        assert "assert" not in result or "# [NPU]" in result
        assert "logging.warning" in result
        assert "RANK" in result

    def test_compound_assert_downgraded(self, tmp_path: Path) -> None:
        code = "            assert e.dtype == torch.float32 and e0.dtype == torch.float32\n"
        p = _write(tmp_path, "a.py", code)
        finding = _finding(str(p), 1, PatternType.DTYPE_ASSERT)
        result = DtypeAssertRule().apply(p.read_text(), finding)
        assert "assert e.dtype" not in result or "# [NPU]" in result
        assert "logging.warning" in result

    def test_preserves_indentation(self, tmp_path: Path) -> None:
        code = "        assert e.dtype == torch.float32\n"
        p = _write(tmp_path, "a.py", code)
        finding = _finding(str(p), 1, PatternType.DTYPE_ASSERT)
        result = DtypeAssertRule().apply(p.read_text(), finding)
        # Warning line must have at least the same indentation
        for line in result.splitlines():
            if "logging.warning" in line:
                assert line.startswith("        ")

    def test_result_is_valid_python(self, tmp_path: Path) -> None:
        code = "import os, logging\nassert e.dtype == torch.float32\n"
        p = _write(tmp_path, "a.py", code)
        finding = _finding(str(p), 2, PatternType.DTYPE_ASSERT)
        result = DtypeAssertRule().apply(p.read_text(), finding)
        compile(result, "<test>", "exec")


class TestNpuInitInjectorRule:
    """Task 1: NpuInitInjectorRule — injects torch_npu + transfer_to_npu via rule registry."""

    def test_inject_both_lines(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", "import torch\nx = 1\n")
        finding = _finding(str(p), 1, PatternType.TORCH_IMPORT)
        rule = NpuInitInjectorRule()
        result = rule.apply(p.read_text(), finding)
        assert "import torch_npu" in result
        assert "from torch_npu.contrib import transfer_to_npu" in result

    def test_injection_order(self, tmp_path: Path) -> None:
        """torch_npu must come right after import torch, transfer_to_npu right after that."""
        p = _write(tmp_path, "a.py", "import torch\nimport os\n")
        finding = _finding(str(p), 1, PatternType.TORCH_IMPORT)
        result = NpuInitInjectorRule().apply(p.read_text(), finding)
        lines = result.splitlines()
        idx_torch = lines.index("import torch")
        idx_npu = lines.index("import torch_npu")
        idx_transfer = lines.index("from torch_npu.contrib import transfer_to_npu")
        assert idx_torch + 1 == idx_npu
        assert idx_npu + 1 == idx_transfer

    def test_idempotent_both_present(self, tmp_path: Path) -> None:
        code = "import torch\nimport torch_npu\nfrom torch_npu.contrib import transfer_to_npu\n"
        p = _write(tmp_path, "a.py", code)
        finding = _finding(str(p), 1, PatternType.TORCH_IMPORT)
        result = NpuInitInjectorRule().apply(p.read_text(), finding)
        assert result == code  # unchanged

    def test_partial_idempotent_missing_transfer(self, tmp_path: Path) -> None:
        code = "import torch\nimport torch_npu\nx = 1\n"
        p = _write(tmp_path, "a.py", code)
        finding = _finding(str(p), 1, PatternType.TORCH_IMPORT)
        result = NpuInitInjectorRule().apply(p.read_text(), finding)
        assert "from torch_npu.contrib import transfer_to_npu" in result
        assert result.count("import torch_npu") == 1  # no duplication

    def test_from_torch_import(self, tmp_path: Path) -> None:
        """Should also trigger on `from torch import ...`."""
        code = "from torch import nn\nx = 1\n"
        p = _write(tmp_path, "a.py", code)
        finding = _finding(str(p), 1, PatternType.TORCH_IMPORT)
        result = NpuInitInjectorRule().apply(p.read_text(), finding)
        assert "import torch_npu" in result
        assert "from torch_npu.contrib import transfer_to_npu" in result


class TestNcclRule:
    def test_nccl_to_hccl(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", 'dist.init_process_group(backend="nccl")\n')
        finding = _finding(str(p), 1, PatternType.NCCL)
        result = NcclToHcclRule().apply(p.read_text(), finding)
        assert '"hccl"' in result
        assert '"nccl"' not in result


class TestFlashAttnRule:
    def test_flash_attn_commented(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", "import flash_attn\n")
        finding = _finding(str(p), 1, PatternType.FLASH_ATTN)
        result = FlashAttnRule().apply(p.read_text(), finding)
        assert "# [NPU]" in result
        assert "SDPA" in result

    def test_flash_attn_in_try_block(self, tmp_path: Path) -> None:
        code = "try:\n    import flash_attn\nexcept ImportError:\n    pass\n"
        p = _write(tmp_path, "a.py", code)
        finding = _finding(str(p), 2, PatternType.FLASH_ATTN)
        result = FlashAttnRule().apply(p.read_text(), finding)
        assert "pass" in result
        assert "# [NPU]" in result
        # Verify it's syntactically valid
        compile(result, "<test>", "exec")


class TestXformersRule:
    def test_xformers_commented(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", "import xformers\n")
        finding = _finding(str(p), 1, PatternType.XFORMERS)
        result = XformersRule().apply(p.read_text(), finding)
        assert "# [NPU]" in result
        assert "SDPA" in result

    def test_xformers_in_try_block(self, tmp_path: Path) -> None:
        code = "try:\n    from xformers.ops import memory_efficient_attention\nexcept ImportError:\n    pass\n"
        p = _write(tmp_path, "a.py", code)
        finding = _finding(str(p), 2, PatternType.XFORMERS)
        result = XformersRule().apply(p.read_text(), finding)
        assert "pass" in result
        assert "# [NPU]" in result
        compile(result, "<test>", "exec")


class TestFloat64Rule:
    def test_float64_to_float32(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", "x = x.to(torch.float64)\n")
        finding = _finding(str(p), 1, PatternType.FLOAT64)
        result = Float64Rule().apply(p.read_text(), finding)
        assert "torch.float32" in result
        assert "torch.float64" not in result

    def test_double_to_float(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", "x = x.double()\n")
        finding = _finding(str(p), 1, PatternType.FLOAT64)
        result = Float64Rule().apply(p.read_text(), finding)
        assert ".float()" in result
        assert ".double()" not in result


# ---------------------------------------------------------------------------
# Integration: apply migrations
# ---------------------------------------------------------------------------

class TestCudaAmpRule:
    def test_import_as_alias(self, tmp_path: Path) -> None:
        """import torch.cuda.amp as amp → import torch.amp as amp"""
        p = _write(tmp_path, "a.py", "import torch.cuda.amp as amp\n")
        finding = _finding(str(p), 1, PatternType.CUDA_AMP)
        result = CudaAmpRule().apply(p.read_text(), finding)
        assert "import torch.amp as amp" in result
        assert "torch.cuda.amp" not in result

    def test_from_import(self, tmp_path: Path) -> None:
        """from torch.cuda.amp import autocast → from torch.amp import autocast"""
        p = _write(tmp_path, "a.py", "from torch.cuda.amp import autocast, GradScaler\n")
        finding = _finding(str(p), 1, PatternType.CUDA_AMP)
        result = CudaAmpRule().apply(p.read_text(), finding)
        assert "from torch.amp import autocast, GradScaler" in result
        assert "torch.cuda.amp" not in result

    def test_inline_usage(self, tmp_path: Path) -> None:
        """torch.cuda.amp.autocast() → torch.amp.autocast('npu')"""
        p = _write(tmp_path, "a.py", "with torch.cuda.amp.autocast():\n    pass\n")
        finding = _finding(str(p), 1, PatternType.CUDA_AMP)
        result = CudaAmpRule().apply(p.read_text(), finding)
        assert "torch.cuda.amp" not in result


class TestAutocastDeviceRule:
    def test_inject_npu_device_type_context_manager(self, tmp_path: Path) -> None:
        """amp.autocast(dtype=self.dtype) → amp.autocast('npu', dtype=self.dtype)"""
        code = "import torch.amp as amp\nwith amp.autocast(dtype=self.dtype):\n    pass\n"
        p = _write(tmp_path, "a.py", code)
        finding = _finding(str(p), 2, PatternType.AUTOCAST_NO_DEVICE)
        result = AutocastDeviceRule().apply(code, finding)
        assert "amp.autocast('npu', dtype=self.dtype)" in result

    def test_inject_npu_device_type_decorator(self, tmp_path: Path) -> None:
        """@amp.autocast(enabled=False) → @amp.autocast('npu', enabled=False)"""
        code = "import torch.amp as amp\n@amp.autocast(enabled=False)\ndef foo(): pass\n"
        p = _write(tmp_path, "a.py", code)
        finding = _finding(str(p), 2, PatternType.AUTOCAST_NO_DEVICE)
        result = AutocastDeviceRule().apply(code, finding)
        assert "amp.autocast('npu', enabled=False)" in result

    def test_inject_npu_torch_amp_autocast(self, tmp_path: Path) -> None:
        """torch.amp.autocast(dtype=...) → torch.amp.autocast('npu', dtype=...)"""
        code = "with torch.amp.autocast(dtype=torch.bfloat16):\n    pass\n"
        p = _write(tmp_path, "a.py", code)
        finding = _finding(str(p), 1, PatternType.AUTOCAST_NO_DEVICE)
        result = AutocastDeviceRule().apply(code, finding)
        assert "torch.amp.autocast('npu', dtype=torch.bfloat16)" in result

    def test_idempotent_already_has_device(self, tmp_path: Path) -> None:
        """Already has 'npu' → is_already_applied returns True."""
        code = "with amp.autocast('npu', dtype=self.dtype):\n    pass\n"
        finding = _finding("a.py", 1, PatternType.AUTOCAST_NO_DEVICE)
        assert AutocastDeviceRule().is_already_applied(code, finding) is True

    def test_not_idempotent_missing_device(self, tmp_path: Path) -> None:
        """Missing device_type → is_already_applied returns False."""
        code = "with amp.autocast(dtype=self.dtype):\n    pass\n"
        finding = _finding("a.py", 1, PatternType.AUTOCAST_NO_DEVICE)
        assert AutocastDeviceRule().is_already_applied(code, finding) is False


class TestApplyMigration:
    def test_add_torch_npu_import(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", "import torch\nx = 1\n")
        added = add_torch_npu_import(str(p))
        assert added is True
        content = p.read_text()
        assert "import torch_npu" in content
        assert "from torch_npu.contrib import transfer_to_npu" in content

    def test_add_torch_npu_import_idempotent(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", "import torch\nimport torch_npu\nfrom torch_npu.contrib import transfer_to_npu\nx = 1\n")
        added = add_torch_npu_import(str(p))
        assert added is False

    def test_add_torch_npu_import_injection_order(self, tmp_path: Path) -> None:
        """import torch_npu must come before transfer_to_npu, both after import torch."""
        p = _write(tmp_path, "a.py", "import torch\nimport os\n")
        add_torch_npu_import(str(p))
        lines = p.read_text().splitlines()
        idx_torch = lines.index("import torch")
        idx_npu = lines.index("import torch_npu")
        idx_transfer = lines.index("from torch_npu.contrib import transfer_to_npu")
        assert idx_torch < idx_npu < idx_transfer

    def test_add_torch_npu_import_partial_idempotent(self, tmp_path: Path) -> None:
        """If torch_npu exists but transfer_to_npu is missing, inject only the missing line."""
        p = _write(tmp_path, "a.py", "import torch\nimport torch_npu\nx = 1\n")
        added = add_torch_npu_import(str(p))
        assert added is True
        content = p.read_text()
        assert "from torch_npu.contrib import transfer_to_npu" in content
        # torch_npu should NOT be duplicated
        assert content.count("import torch_npu") == 1

    def test_add_torch_npu_import_full_idempotent(self, tmp_path: Path) -> None:
        """If both lines already present, return False and don't modify."""
        code = "import torch\nimport torch_npu\nfrom torch_npu.contrib import transfer_to_npu\nx = 1\n"
        p = _write(tmp_path, "a.py", code)
        added = add_torch_npu_import(str(p))
        assert added is False
        assert p.read_text() == code  # unchanged

    def test_plan_migrations_grouping(self) -> None:
        registry = create_default_registry()
        findings = [
            _finding("/tmp/a.py", 1, PatternType.CUDA_CALL),
            _finding("/tmp/a.py", 2, PatternType.CUDA_API),
            _finding("/tmp/b.py", 1, PatternType.NCCL),
        ]
        plan = registry.match_all(findings)
        assert len(plan.matched) == 2  # two files
        assert plan.total_migrations == 3

    def test_apply_all_end_to_end(self, tmp_path: Path) -> None:
        code = 'import torch\nx = tensor.cuda()\nif torch.cuda.is_available():\n    pass\n'
        p = _write(tmp_path, "model.py", code)

        findings = [
            _finding(str(p), 2, PatternType.CUDA_CALL, "x = tensor.cuda()"),
            _finding(str(p), 3, PatternType.CUDA_API, "if torch.cuda.is_available():"),
        ]
        registry = create_default_registry()
        plan = registry.match_all(findings)
        results = apply_all_migrations(plan)

        assert len(results) == 1
        assert results[0].success is True

        content = p.read_text()
        assert ".npu()" in content
        assert "torch.npu.is_available()" in content
        assert ".cuda()" not in content
        assert "torch.cuda" not in content

    def test_backup_created(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", "x = tensor.cuda()\n")
        finding = _finding(str(p), 1, PatternType.CUDA_CALL)
        registry = create_default_registry()
        plan = registry.match_all([finding])
        apply_all_migrations(plan)
        assert (tmp_path / "a.py.bak").exists()

    def test_cuda_amp_rule_in_default_registry(self) -> None:
        """CUDA_AMP pattern must have a matching rule in the default registry."""
        registry = create_default_registry()
        finding = _finding("/tmp/x.py", 1, PatternType.CUDA_AMP)
        rule = registry.match(finding)
        assert rule is not None, "No rule matched for CUDA_AMP"
        assert rule.name == "cuda_amp"

    def test_custom_rule_plugin(self, tmp_path: Path) -> None:
        """Register a user-defined rule and verify it's applied."""

        class CustomRule(MigrationRule):
            name = "custom_bfloat16"
            description = "bfloat16 → float16"
            pattern_type = PatternType.BFLOAT16

            def is_already_applied(self, source: str, finding: Finding) -> bool:
                lines = source.splitlines()
                idx = finding.line_number - 1
                if 0 <= idx < len(lines):
                    return "bfloat16" not in lines[idx]
                return False

            def apply(self, source: str, finding: Finding) -> str:
                lines = source.splitlines(keepends=True)
                idx = finding.line_number - 1
                if 0 <= idx < len(lines):
                    lines[idx] = lines[idx].replace("bfloat16", "float16")
                return "".join(lines)

        p = _write(tmp_path, "a.py", 'x = x.to("bfloat16")\n')
        finding = _finding(str(p), 1, PatternType.BFLOAT16, 'x = x.to("bfloat16")')

        registry = create_default_registry()
        # Before: BFLOAT16 unmatched
        plan_before = registry.match_all([finding])
        assert len(plan_before.unmatched) == 1

        # After: register custom rule
        registry.register(CustomRule())
        plan_after = registry.match_all([finding])
        assert len(plan_after.unmatched) == 0
        assert plan_after.total_migrations == 1

        results = apply_all_migrations(plan_after)
        assert results[0].success is True
        assert "float16" in p.read_text()
        assert "bfloat16" not in p.read_text()


# ---------------------------------------------------------------------------
# Step 1: Idempotency tests
# ---------------------------------------------------------------------------

class TestIdempotencyGuards:
    """Each rule, when applied twice, should produce identical output."""

    def test_is_already_applied_base_class_is_abstract(self) -> None:
        """MigrationRule.is_already_applied() is abstract."""
        assert "is_already_applied" in MigrationRule.__abstractmethods__

    def test_cuda_call_idempotent(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", "x = tensor.cuda()\n")
        finding = _finding(str(p), 1, PatternType.CUDA_CALL)
        rule = CudaCallRule()
        first = rule.apply(p.read_text(), finding)
        assert rule.is_already_applied(first, finding)
        second = rule.apply(first, finding)
        assert first == second

    def test_cuda_to_idempotent(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", 'model.to("cuda")\n')
        finding = _finding(str(p), 1, PatternType.CUDA_TO)
        rule = CudaToRule()
        first = rule.apply(p.read_text(), finding)
        assert rule.is_already_applied(first, finding)

    def test_cuda_api_idempotent(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", "if torch.cuda.is_available():\n    pass\n")
        finding = _finding(str(p), 1, PatternType.CUDA_API)
        rule = CudaApiRule()
        first = rule.apply(p.read_text(), finding)
        assert rule.is_already_applied(first, finding)

    def test_nccl_idempotent(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", 'dist.init_process_group(backend="nccl")\n')
        finding = _finding(str(p), 1, PatternType.NCCL)
        rule = NcclToHcclRule()
        first = rule.apply(p.read_text(), finding)
        assert rule.is_already_applied(first, finding)

    def test_flash_attn_idempotent(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", "import flash_attn\n")
        finding = _finding(str(p), 1, PatternType.FLASH_ATTN)
        rule = FlashAttnRule()
        first = rule.apply(p.read_text(), finding)
        assert rule.is_already_applied(first, finding)

    def test_flash_attn_skips_try_except_guard(self, tmp_path: Path) -> None:
        """FlashAttnRule must NOT fire when import is inside try/except with fallback."""
        code = (
            "try:\n"
            "    import flash_attn\n"
            "    FLASH_ATTN_2_AVAILABLE = True\n"
            "except (ModuleNotFoundError, ImportError):\n"
            "    FLASH_ATTN_2_AVAILABLE = False\n"
            "    flash_attn = None\n"
        )
        p = _write(tmp_path, "a.py", code)
        finding = _finding(str(p), 2, PatternType.FLASH_ATTN, "import flash_attn")
        rule = FlashAttnRule()
        assert rule.is_already_applied(code, finding)

    def test_xformers_idempotent(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", "import xformers\n")
        finding = _finding(str(p), 1, PatternType.XFORMERS)
        rule = XformersRule()
        first = rule.apply(p.read_text(), finding)
        assert rule.is_already_applied(first, finding)

    def test_cuda_amp_idempotent(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", "from torch.cuda.amp import autocast\n")
        finding = _finding(str(p), 1, PatternType.CUDA_AMP)
        rule = CudaAmpRule()
        first = rule.apply(p.read_text(), finding)
        assert rule.is_already_applied(first, finding)

    def test_cuda_device_str_idempotent(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", 'device = torch.device("cuda")\n')
        finding = _finding(str(p), 1, PatternType.CUDA_DEVICE_STR)
        rule = CudaDeviceStrRule()
        first = rule.apply(p.read_text(), finding)
        assert rule.is_already_applied(first, finding)

    def test_npu_init_injector_idempotent(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", "import torch\nx = 1\n")
        finding = _finding(str(p), 1, PatternType.TORCH_IMPORT)
        rule = NpuInitInjectorRule()
        first = rule.apply(p.read_text(), finding)
        assert rule.is_already_applied(first, finding)
        second = rule.apply(first, finding)
        assert first == second

    def test_float64_idempotent(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", "x = x.to(torch.float64)\n")
        finding = _finding(str(p), 1, PatternType.FLOAT64)
        rule = Float64Rule()
        first = rule.apply(p.read_text(), finding)
        assert rule.is_already_applied(first, finding)

    def test_autocast_dtype_idempotent(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", "with torch.amp.autocast('npu', dtype=torch.float32):\n    pass\n")
        finding = _finding(str(p), 1, PatternType.AUTOCAST_DTYPE)
        rule = AutocastDtypeRule()
        first = rule.apply(p.read_text(), finding)
        assert rule.is_already_applied(first, finding)

    def test_dtype_assert_idempotent(self, tmp_path: Path) -> None:
        code = "assert e.dtype == torch.float32\n"
        p = _write(tmp_path, "a.py", code)
        finding = _finding(str(p), 1, PatternType.DTYPE_ASSERT, snippet=code.strip())
        rule = DtypeAssertRule()
        first = rule.apply(p.read_text(), finding)
        assert rule.is_already_applied(first, finding)

    def test_apply_migration_skips_already_applied(self, tmp_path: Path) -> None:
        """apply_migration() should skip rules that are already applied."""
        # Pre-migrated file (already has .npu())
        p = _write(tmp_path, "a.py", "x = tensor.npu()\n")
        finding = _finding(str(p), 1, PatternType.CUDA_CALL)
        rule = CudaCallRule()
        result = apply_migration(str(p), [(finding, rule)])
        assert result.success is True
        # Rule was skipped — no rules actually applied
        assert "cuda_call" not in result.applied_rules
        # Content unchanged
        assert p.read_text() == "x = tensor.npu()\n"


# ---------------------------------------------------------------------------
# Step 3: FlashAttnUsageRule tests
# ---------------------------------------------------------------------------

class TestFlashAttnUsageRule:
    def test_flash_attn_usage_rule_removes_assert(self, tmp_path: Path) -> None:
        code = "assert FLASH_ATTN_2_AVAILABLE\n"
        p = _write(tmp_path, "a.py", code)
        finding = _finding(str(p), 1, PatternType.FLASH_ATTN_USAGE, code.strip())
        rule = FlashAttnUsageRule()
        result = rule.apply(p.read_text(), finding)
        assert not result.startswith("assert")
        assert "pass" in result
        assert "[NPU]" in result

    def test_flash_attn_varlen_replaced_with_npu_fusion(self, tmp_path: Path) -> None:
        """Varlen calls should be replaced with _npu_varlen_attention, NOT SDPA."""
        code = "out = flash_attn.flash_attn_varlen_func(q, k, v)\n"
        p = _write(tmp_path, "a.py", code)
        finding = _finding(str(p), 1, PatternType.FLASH_ATTN_USAGE, code.strip())
        rule = FlashAttnUsageRule()
        result = rule.apply(p.read_text(), finding)
        assert "_npu_varlen_attention(" in result
        assert "# [NPU]" in result
        assert "flash_attn.flash_attn_varlen_func" not in result.replace("# [NPU]", "")

    def test_flash_attn_usage_rule_idempotent(self, tmp_path: Path) -> None:
        code = "assert FLASH_ATTN_2_AVAILABLE\n"
        p = _write(tmp_path, "a.py", code)
        finding = _finding(str(p), 1, PatternType.FLASH_ATTN_USAGE, code.strip())
        rule = FlashAttnUsageRule()
        first = rule.apply(p.read_text(), finding)
        assert rule.is_already_applied(first, finding)

    def test_flash_attn_usage_rule_registered(self) -> None:
        registry = create_default_registry()
        finding = _finding("/tmp/x.py", 1, PatternType.FLASH_ATTN_USAGE)
        rule = registry.match(finding)
        assert rule is not None
        assert rule.name == "flash_attn_usage"

    def test_flash_attn_usage_multi_line_varlen_replaced_with_npu(self, tmp_path: Path) -> None:
        """Multi-line varlen call should use _npu_varlen_attention (in-place substitution)."""
        code = (
            "    x = flash_attn.flash_attn_varlen_func(\n"
            "        q=q,\n"
            "        k=k,\n"
            "        v=v,\n"
            "        cu_seqlens_q=cu_q,\n"
            "        cu_seqlens_k=cu_k,\n"
            "        max_seqlen_q=lq,\n"
            "        max_seqlen_k=lk,\n"
            "        dropout_p=dropout_p,\n"
            "        softmax_scale=softmax_scale,\n"
            "        causal=causal,\n"
            "        deterministic=deterministic).unflatten(0, (b, lq))\n"
        )
        p = _write(tmp_path, "attn.py", code)
        finding = _finding(str(p), 1, PatternType.FLASH_ATTN_USAGE, "flash_attn.flash_attn_varlen_func(")
        rule = FlashAttnUsageRule()
        result = rule.apply(code, finding)
        # In-place substitution: function name replaced, args preserved
        assert "_npu_varlen_attention(" in result
        assert "# [NPU]" in result
        # All original arguments should be preserved (they're on subsequent lines)
        assert "q=q," in result
        assert "cu_seqlens_q=cu_q," in result
        assert "deterministic=deterministic)" in result

    def test_flash_attn_usage_multi_line_varlen_idempotent(self, tmp_path: Path) -> None:
        """Applying multi-line varlen rule twice should not double-modify."""
        code = (
            "    x = flash_attn.flash_attn_varlen_func(\n"
            "        q=q,\n"
            "        k=k,\n"
            "        v=v,\n"
            "    ).unflatten(0, (b, lq))\n"
        )
        p = _write(tmp_path, "attn.py", code)
        finding = _finding(str(p), 1, PatternType.FLASH_ATTN_USAGE, "flash_attn.flash_attn_varlen_func(")
        rule = FlashAttnUsageRule()
        first = rule.apply(code, finding)
        assert rule.is_already_applied(first, finding)

    def test_flash_attn_usage_multi_line_preserves_chained_method(self, tmp_path: Path) -> None:
        """Chained .unflatten() after the multi-line call should be preserved (in-place sub)."""
        code = (
            "    x = flash_attn.flash_attn_varlen_func(\n"
            "        q=q,\n"
            "        k=k,\n"
            "        v=v,\n"
            "    ).unflatten(0, (b, lq))\n"
        )
        p = _write(tmp_path, "attn.py", code)
        finding = _finding(str(p), 1, PatternType.FLASH_ATTN_USAGE, "flash_attn.flash_attn_varlen_func(")
        rule = FlashAttnUsageRule()
        result = rule.apply(code, finding)
        # In-place substitution preserves chained method (it's on a later line)
        assert "unflatten" in result

    def test_flash_attn_non_varlen_still_uses_sdpa(self, tmp_path: Path) -> None:
        """Non-varlen flash_attn calls should still use SDPA replacement."""
        code = "out = flash_attn.flash_attn_func(q, k, v)\n"
        p = _write(tmp_path, "a.py", code)
        finding = _finding(str(p), 1, PatternType.FLASH_ATTN_USAGE, code.strip())
        rule = FlashAttnUsageRule()
        result = rule.apply(p.read_text(), finding)
        assert "scaled_dot_product_attention" in result
        assert "# [NPU]" in result
        # Should NOT use NPU fusion for non-varlen
        assert "_npu_varlen_attention" not in result

    def test_flash_attn_varlen_wrapper_injected(self, tmp_path: Path) -> None:
        """Varlen replacement should inject wrapper function (via marker + post-pass)."""
        from diffusion_agent.tools.code_migrator import _resolve_npu_wrapper_marker
        code = "import torch\ndef foo():\n    out = flash_attn.flash_attn_varlen_func(q, k, v)\n"
        p = _write(tmp_path, "a.py", code)
        finding = _finding(str(p), 3, PatternType.FLASH_ATTN_USAGE, "flash_attn.flash_attn_varlen_func(")
        rule = FlashAttnUsageRule()
        result = rule.apply(code, finding)
        # Marker should be present
        assert "__NEEDS_NPU_VARLEN_WRAPPER__" in result
        # Post-pass resolves the marker
        resolved = _resolve_npu_wrapper_marker(result)
        assert "def _npu_varlen_attention" in resolved
        assert "__NEEDS_NPU_VARLEN_WRAPPER__" not in resolved

    def test_flash_attn_varlen_wrapper_has_tnd_layout(self) -> None:
        """Wrapper function must use input_layout='TND'."""
        from diffusion_agent.tools.code_migrator import _NPU_VARLEN_WRAPPER
        assert 'input_layout="TND"' in _NPU_VARLEN_WRAPPER

    def test_flash_attn_varlen_wrapper_has_int32_cast(self) -> None:
        """Wrapper function must cast cu_seqlens to int32."""
        from diffusion_agent.tools.code_migrator import _NPU_VARLEN_WRAPPER
        assert ".to(torch.int32)" in _NPU_VARLEN_WRAPPER

    def test_flash_attn_varlen_wrapper_has_sparse_mode(self) -> None:
        """Wrapper function must set sparse_mode=3 for causal, 0 for non-causal."""
        from diffusion_agent.tools.code_migrator import _NPU_VARLEN_WRAPPER
        assert "sparse_mode = 3" in _NPU_VARLEN_WRAPPER
        assert "sparse_mode = 0" in _NPU_VARLEN_WRAPPER

    def test_flash_attn_varlen_wrapper_not_duplicated(self, tmp_path: Path) -> None:
        """If wrapper already exists in file, marker resolution should not duplicate it."""
        from diffusion_agent.tools.code_migrator import (
            _NPU_WRAPPER_MARKER,
            _resolve_npu_wrapper_marker,
        )
        code = (
            "import torch\n"
            + "def _npu_varlen_attention(*args, **kwargs): pass\n"
            + _NPU_WRAPPER_MARKER + "\n"
            + "def foo(): pass\n"
        )
        result = _resolve_npu_wrapper_marker(code)
        assert result.count("def _npu_varlen_attention") == 1


# ---------------------------------------------------------------------------
# DependencyMigrationRule tests
# ---------------------------------------------------------------------------

class TestDependencyMigrationRule:
    """Tests for the DependencyMigrationRule (requirements.txt migration)."""

    @staticmethod
    def _dep_finding(file_path: str) -> Finding:
        return Finding(
            file_path=file_path, line_number=1,
            pattern_type=PatternType.DEPENDENCY_FILE, code_snippet="",
        )

    def test_removes_blacklisted_packages(self, tmp_path: Path) -> None:
        content = "torch>=2.1.0\nflash-attn\nxformers\ntriton\naccelerate\nnumpy\n"
        rule = DependencyMigrationRule(version_resolver=StaticVersionResolver({}))
        result = rule.apply(content, self._dep_finding("requirements.txt"))
        assert "flash-attn" not in result or "# [NPU] removed:" in result
        assert "xformers" not in result or "# [NPU] removed:" in result
        assert "triton" not in result or "# [NPU] removed:" in result
        assert "accelerate" not in result or "# [NPU] removed:" in result
        assert "numpy" in result
        # Verify removal comments
        assert result.count("# [NPU] removed:") == 4

    def test_removes_flash_attn_underscore(self, tmp_path: Path) -> None:
        content = "flash_attn>=2.0\nnumpy\n"
        rule = DependencyMigrationRule(version_resolver=StaticVersionResolver({}))
        result = rule.apply(content, self._dep_finding("r.txt"))
        assert "# [NPU] removed: flash_attn>=2.0" in result
        assert "numpy" in result

    def test_adds_required_packages(self, tmp_path: Path) -> None:
        content = "numpy\nscipy\n"
        rule = DependencyMigrationRule(version_resolver=StaticVersionResolver({}))
        result = rule.apply(content, self._dep_finding("r.txt"))
        assert "imageio-ffmpeg" in result

    def test_preserves_comments_and_blanks(self, tmp_path: Path) -> None:
        content = "# Core dependencies\ntorch>=2.1.0\n\n# Utils\nnumpy\n"
        rule = DependencyMigrationRule(version_resolver=StaticVersionResolver({}))
        result = rule.apply(content, self._dep_finding("r.txt"))
        assert "# Core dependencies" in result
        assert "# Utils" in result

    def test_version_alignment_exact_match(self, tmp_path: Path) -> None:
        """When torch==2.1.0 and torch-npu 2.1.0 is available, pin both."""
        content = "torch==2.1.0\nnumpy\n"
        resolver = StaticVersionResolver({
            "torch-npu": ["2.2.0", "2.1.0", "2.0.0"],
        })
        rule = DependencyMigrationRule(version_resolver=resolver)
        result = rule.apply(content, self._dep_finding("r.txt"))
        assert "torch-npu==2.1.0" in result
        assert "torch==2.1.0" in result

    def test_version_alignment_no_exact_match_picks_higher(self, tmp_path: Path) -> None:
        """When torch==2.1.0 but only torch-npu 2.2.0 available, align to 2.2.0."""
        content = "torch==2.1.0\nnumpy\n"
        resolver = StaticVersionResolver({
            "torch-npu": ["2.3.0", "2.2.0"],
        })
        rule = DependencyMigrationRule(version_resolver=resolver)
        result = rule.apply(content, self._dep_finding("r.txt"))
        assert "torch-npu==2.2.0" in result
        assert "torch==2.2.0" in result  # torch realigned to match

    def test_version_alignment_with_existing_torch_npu(self, tmp_path: Path) -> None:
        """When torch-npu already in requirements, update it."""
        content = "torch==2.1.0\ntorch-npu==2.0.0\nnumpy\n"
        resolver = StaticVersionResolver({
            "torch-npu": ["2.1.0", "2.0.0"],
        })
        rule = DependencyMigrationRule(version_resolver=resolver)
        result = rule.apply(content, self._dep_finding("r.txt"))
        assert "torch-npu==2.1.0" in result
        assert "torch-npu==2.0.0" not in result

    def test_version_alignment_resolver_failure(self, tmp_path: Path) -> None:
        """When resolver returns empty list, add torch-npu with wildcard."""
        content = "torch==2.1.0\nnumpy\n"
        resolver = StaticVersionResolver({})
        rule = DependencyMigrationRule(version_resolver=resolver)
        result = rule.apply(content, self._dep_finding("r.txt"))
        assert "torch-npu==2.1.0.*" in result

    def test_no_torch_line_skips_version_alignment(self, tmp_path: Path) -> None:
        """When no torch in requirements, don't add torch-npu."""
        content = "numpy\nscipy\n"
        rule = DependencyMigrationRule(version_resolver=StaticVersionResolver({}))
        result = rule.apply(content, self._dep_finding("r.txt"))
        assert "torch-npu" not in result
        assert "imageio-ffmpeg" in result

    def test_torch_without_version_adds_bare_torch_npu(self, tmp_path: Path) -> None:
        """When torch present without version spec, add bare torch-npu."""
        content = "torch\nnumpy\n"
        rule = DependencyMigrationRule(version_resolver=StaticVersionResolver({}))
        result = rule.apply(content, self._dep_finding("r.txt"))
        assert "torch-npu" in result

    def test_idempotent_is_already_applied(self, tmp_path: Path) -> None:
        """After applying, is_already_applied returns True."""
        content = "torch==2.1.0\nflash-attn\nnumpy\n"
        resolver = StaticVersionResolver({"torch-npu": ["2.1.0"]})
        rule = DependencyMigrationRule(version_resolver=resolver)
        finding = self._dep_finding("r.txt")
        result = rule.apply(content, finding)
        assert rule.is_already_applied(result, finding)

    def test_idempotent_second_apply_stable(self, tmp_path: Path) -> None:
        """Applying twice produces the same result."""
        content = "torch==2.1.0\nflash-attn\nxformers\nnumpy\n"
        resolver = StaticVersionResolver({"torch-npu": ["2.1.0"]})
        rule = DependencyMigrationRule(version_resolver=resolver)
        finding = self._dep_finding("r.txt")
        first = rule.apply(content, finding)
        second = rule.apply(first, finding)
        assert first == second

    def test_removal_comment_format(self, tmp_path: Path) -> None:
        """Verify exact format of removal comment."""
        content = "accelerate>=0.20.0\nnumpy\n"
        rule = DependencyMigrationRule(version_resolver=StaticVersionResolver({}))
        result = rule.apply(content, self._dep_finding("r.txt"))
        assert "# [NPU] removed: accelerate>=0.20.0" in result

    def test_default_registry_includes_dependency_rule(self) -> None:
        """DependencyMigrationRule is in the default registry."""
        registry = create_default_registry()
        finding = self._dep_finding("requirements.txt")
        rule = registry.match(finding)
        assert rule is not None
        assert rule.name == "dependency_migration"

    def test_version_alignment_gte_spec(self, tmp_path: Path) -> None:
        """torch>=2.1.0 should also trigger version alignment."""
        content = "torch>=2.1.0\nnumpy\n"
        resolver = StaticVersionResolver({"torch-npu": ["2.1.0.post2", "2.1.0"]})
        rule = DependencyMigrationRule(version_resolver=resolver)
        result = rule.apply(content, self._dep_finding("r.txt"))
        assert "torch-npu==2.1.0.post2" in result
