"""Tests for extended code scanner patterns: flash_attn, distributed, bfloat16, xformers, sdpa."""

from __future__ import annotations

from pathlib import Path

from diffusion_agent.tools.code_scanner import (
    PatternType,
    scan_file,
)


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# PatternType: FLASH_ATTN
# ---------------------------------------------------------------------------

class TestFlashAttn:
    def test_import_flash_attn(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", "import flash_attn\n")
        findings = scan_file(src)
        assert any(f.pattern_type == PatternType.FLASH_ATTN for f in findings)

    def test_from_flash_attn_import(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", "from flash_attn import flash_attn_func\n")
        findings = scan_file(src)
        assert any(f.pattern_type == PatternType.FLASH_ATTN for f in findings)

    def test_from_flash_attn_submodule(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", "from flash_attn.flash_attn_interface import flash_attn_varlen_func\n")
        findings = scan_file(src)
        assert any(f.pattern_type == PatternType.FLASH_ATTN for f in findings)


# ---------------------------------------------------------------------------
# PatternType: XFORMERS
# ---------------------------------------------------------------------------

class TestXformers:
    def test_import_xformers(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", "import xformers\n")
        findings = scan_file(src)
        assert any(f.pattern_type == PatternType.XFORMERS for f in findings)

    def test_from_xformers_import(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", "from xformers.ops import memory_efficient_attention\n")
        findings = scan_file(src)
        assert any(f.pattern_type == PatternType.XFORMERS for f in findings)


# ---------------------------------------------------------------------------
# PatternType: BFLOAT16
# ---------------------------------------------------------------------------

class TestBfloat16:
    def test_torch_bfloat16_attribute(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", "x = x.to(torch.bfloat16)\n")
        findings = scan_file(src)
        assert any(f.pattern_type == PatternType.BFLOAT16 for f in findings)

    def test_bfloat16_string_literal(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", 'dtype = "bfloat16"\n')
        findings = scan_file(src)
        assert any(f.pattern_type == PatternType.BFLOAT16 for f in findings)


# ---------------------------------------------------------------------------
# PatternType: DISTRIBUTED
# ---------------------------------------------------------------------------

class TestDistributed:
    def test_init_process_group_call(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", 'torch.distributed.init_process_group(backend="nccl")\n')
        findings = scan_file(src)
        assert any(f.pattern_type == PatternType.DISTRIBUTED for f in findings)

    def test_dist_init_process_group(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", 'dist.init_process_group(backend="nccl")\n')
        findings = scan_file(src)
        assert any(f.pattern_type == PatternType.DISTRIBUTED for f in findings)

    def test_import_torch_distributed(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", "import torch.distributed as dist\n")
        findings = scan_file(src)
        assert any(f.pattern_type == PatternType.DISTRIBUTED for f in findings)


# ---------------------------------------------------------------------------
# PatternType: SDPA
# ---------------------------------------------------------------------------

class TestSdpa:
    def test_scaled_dot_product_attention_call(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", "out = torch.nn.functional.scaled_dot_product_attention(q, k, v)\n")
        findings = scan_file(src)
        assert any(f.pattern_type == PatternType.SDPA for f in findings)

    def test_f_scaled_dot_product_attention(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", "out = F.scaled_dot_product_attention(q, k, v)\n")
        findings = scan_file(src)
        assert any(f.pattern_type == PatternType.SDPA for f in findings)
