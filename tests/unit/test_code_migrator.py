"""Unit tests for the code_migrator — rule registry + built-in rules."""

from __future__ import annotations

from pathlib import Path

from diffusion_agent.tools.code_migrator import (
    CudaApiRule,
    CudaCallRule,
    CudaDeviceStrRule,
    CudaToRule,
    Float64Rule,
    FlashAttnRule,
    MigrationRule,
    NcclToHcclRule,
    RuleRegistry,
    XformersRule,
    add_torch_npu_import,
    apply_all_migrations,
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
            (PatternType.CUDA_DEVICE_STR, "cuda_device_str"),
            (PatternType.NCCL, "nccl_to_hccl"),
            (PatternType.FLASH_ATTN, "flash_attn"),
            (PatternType.XFORMERS, "xformers"),
            (PatternType.FLOAT64, "float64"),
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
        assert len(registry.get_rules()) == 8


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

class TestApplyMigration:
    def test_add_torch_npu_import(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", "import torch\nx = 1\n")
        added = add_torch_npu_import(str(p))
        assert added is True
        content = p.read_text()
        assert "import torch_npu" in content

    def test_add_torch_npu_import_idempotent(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "a.py", "import torch\nimport torch_npu\nx = 1\n")
        added = add_torch_npu_import(str(p))
        assert added is False

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

    def test_custom_rule_plugin(self, tmp_path: Path) -> None:
        """Register a user-defined rule and verify it's applied."""

        class CustomRule(MigrationRule):
            name = "custom_bfloat16"
            description = "bfloat16 → float16"
            pattern_type = PatternType.BFLOAT16

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
