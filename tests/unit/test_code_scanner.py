"""Unit tests for the code scanner tool."""

from __future__ import annotations

from pathlib import Path

from diffusion_agent.tools.code_scanner import (
    PatternType,
    scan_directory,
    scan_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# PatternType: .cuda()
# ---------------------------------------------------------------------------

class TestCudaCall:
    def test_detect_dot_cuda(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", "x = tensor.cuda()\n")
        findings = scan_file(src)
        assert len(findings) == 1
        assert findings[0].pattern_type == PatternType.CUDA_CALL
        assert findings[0].line_number == 1

    def test_detect_dot_cuda_with_device(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", "x = tensor.cuda(0)\n")
        findings = scan_file(src)
        assert len(findings) == 1
        assert findings[0].pattern_type == PatternType.CUDA_CALL


# ---------------------------------------------------------------------------
# PatternType: .to("cuda") / .to(device)
# ---------------------------------------------------------------------------

class TestCudaTo:
    def test_to_cuda_string(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", 'model.to("cuda")\n')
        findings = scan_file(src)
        assert any(f.pattern_type == PatternType.CUDA_TO for f in findings)

    def test_to_cuda_colon_device(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", 'model.to("cuda:0")\n')
        findings = scan_file(src)
        assert any(f.pattern_type == PatternType.CUDA_TO for f in findings)

    def test_to_device_variable(self, tmp_path: Path) -> None:
        code = 'device = "cuda"\nmodel.to(device)\n'
        src = _write(tmp_path, "a.py", code)
        findings = scan_file(src)
        assert any(f.pattern_type == PatternType.CUDA_TO for f in findings)


# ---------------------------------------------------------------------------
# PatternType: torch.cuda.*
# ---------------------------------------------------------------------------

class TestCudaAPI:
    def test_torch_cuda_is_available(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", "if torch.cuda.is_available():\n    pass\n")
        findings = scan_file(src)
        assert any(f.pattern_type == PatternType.CUDA_API for f in findings)

    def test_torch_cuda_device_count(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", "n = torch.cuda.device_count()\n")
        findings = scan_file(src)
        assert any(f.pattern_type == PatternType.CUDA_API for f in findings)

    def test_torch_cuda_set_device(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", "torch.cuda.set_device(0)\n")
        findings = scan_file(src)
        assert any(f.pattern_type == PatternType.CUDA_API for f in findings)


# ---------------------------------------------------------------------------
# PatternType: float64 / double
# ---------------------------------------------------------------------------

class TestFloat64:
    def test_torch_float64(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", "x = x.to(torch.float64)\n")
        findings = scan_file(src)
        assert any(f.pattern_type == PatternType.FLOAT64 for f in findings)

    def test_torch_double(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", "x = x.double()\n")
        findings = scan_file(src)
        assert any(f.pattern_type == PatternType.FLOAT64 for f in findings)


# ---------------------------------------------------------------------------
# PatternType: nccl
# ---------------------------------------------------------------------------

class TestCudaDeviceStr:
    def test_torch_device_cuda(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", 'device = torch.device("cuda")\n')
        findings = scan_file(src)
        assert any(f.pattern_type == PatternType.CUDA_DEVICE_STR for f in findings)

    def test_cuda_default_param(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", 'def foo(device="cuda"):\n    pass\n')
        findings = scan_file(src)
        assert any(f.pattern_type == PatternType.CUDA_DEVICE_STR for f in findings)

    def test_not_double_counted_with_cuda_to(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", 'model.to("cuda")\n')
        findings = scan_file(src)
        # Should have CUDA_TO but NOT CUDA_DEVICE_STR for the same line
        assert any(f.pattern_type == PatternType.CUDA_TO for f in findings)
        assert not any(f.pattern_type == PatternType.CUDA_DEVICE_STR for f in findings)


class TestNccl:
    def test_nccl_backend(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", 'dist.init_process_group(backend="nccl")\n')
        findings = scan_file(src)
        assert any(f.pattern_type == PatternType.NCCL for f in findings)


# ---------------------------------------------------------------------------
# scan_file edge cases
# ---------------------------------------------------------------------------

class TestScanFileEdgeCases:
    def test_empty_file(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "empty.py", "")
        assert scan_file(src) == []

    def test_no_patterns(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "clean.py", "import os\nprint('hello')\n")
        assert scan_file(src) == []

    def test_syntax_error_file(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "bad.py", "def foo(\n")
        findings = scan_file(src)
        assert findings == []  # gracefully skip

    def test_binary_file(self, tmp_path: Path) -> None:
        p = tmp_path / "data.bin"
        p.write_bytes(b"\x00\x01\x02\xff")
        assert scan_file(p) == []

    def test_finding_has_code_snippet(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", "x = tensor.cuda()\n")
        findings = scan_file(src)
        assert findings[0].code_snippet  # non-empty

    def test_multiple_patterns_in_one_file(self, tmp_path: Path) -> None:
        code = (
            "x = tensor.cuda()\n"
            "if torch.cuda.is_available():\n"
            "    pass\n"
            "y = x.double()\n"
        )
        src = _write(tmp_path, "multi.py", code)
        findings = scan_file(src)
        types = {f.pattern_type for f in findings}
        assert PatternType.CUDA_CALL in types
        assert PatternType.CUDA_API in types
        assert PatternType.FLOAT64 in types


# ---------------------------------------------------------------------------
# scan_directory
# ---------------------------------------------------------------------------

class TestScanDirectory:
    def test_scans_python_files_recursively(self, tmp_path: Path) -> None:
        sub = tmp_path / "pkg"
        sub.mkdir()
        _write(sub, "m.py", "x = tensor.cuda()\n")
        _write(tmp_path, "top.py", "torch.cuda.is_available()\n")
        findings = scan_directory(tmp_path)
        assert len(findings) >= 2
        files = {f.file_path for f in findings}
        assert str(sub / "m.py") in files
        assert str(tmp_path / "top.py") in files

    def test_ignores_non_python_files(self, tmp_path: Path) -> None:
        _write(tmp_path, "readme.md", "tensor.cuda()")
        assert scan_directory(tmp_path) == []

    def test_custom_glob_pattern(self, tmp_path: Path) -> None:
        _write(tmp_path, "script.py", "tensor.cuda()\n")
        _write(tmp_path, "other.py", "print('hi')\n")
        findings = scan_directory(tmp_path, pattern="script.py")
        assert len(findings) == 1

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert scan_directory(tmp_path) == []


# ---------------------------------------------------------------------------
# PatternType: torch.cuda.amp
# ---------------------------------------------------------------------------

class TestTorchImport:
    """Task 1: Scanner must detect `import torch` as TORCH_IMPORT."""

    def test_detect_import_torch(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", "import torch\nx = 1\n")
        findings = scan_file(src)
        ti = [f for f in findings if f.pattern_type == PatternType.TORCH_IMPORT]
        assert len(ti) == 1
        assert ti[0].line_number == 1

    def test_detect_from_torch_import(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "a.py", "from torch import nn\n")
        findings = scan_file(src)
        ti = [f for f in findings if f.pattern_type == PatternType.TORCH_IMPORT]
        assert len(ti) == 1

    def test_not_detect_import_torchvision(self, tmp_path: Path) -> None:
        """import torchvision should NOT trigger TORCH_IMPORT."""
        src = _write(tmp_path, "a.py", "import torchvision\n")
        findings = scan_file(src)
        ti = [f for f in findings if f.pattern_type == PatternType.TORCH_IMPORT]
        assert len(ti) == 0

    def test_not_detect_import_torch_npu(self, tmp_path: Path) -> None:
        """import torch_npu should NOT trigger TORCH_IMPORT."""
        src = _write(tmp_path, "a.py", "import torch_npu\n")
        findings = scan_file(src)
        ti = [f for f in findings if f.pattern_type == PatternType.TORCH_IMPORT]
        assert len(ti) == 0

    def test_only_first_import_torch_detected(self, tmp_path: Path) -> None:
        """Multiple `import torch` should each be detected (scanner reports all)."""
        code = "import torch\nimport os\nimport torch\n"
        src = _write(tmp_path, "a.py", code)
        findings = scan_file(src)
        ti = [f for f in findings if f.pattern_type == PatternType.TORCH_IMPORT]
        assert len(ti) == 2


class TestCudaDeviceStrEnhanced:
    """Task 2: Scanner must detect cuda device strings in f-strings, spaces, two-arg."""

    def test_detect_fstring_cuda(self, tmp_path: Path) -> None:
        """f'cuda:{device_id}' should be detected as CUDA_DEVICE_STR."""
        src = _write(tmp_path, "a.py", 'self.device = torch.device(f"cuda:{device_id}")\n')
        findings = scan_file(src)
        ds = [f for f in findings if f.pattern_type == PatternType.CUDA_DEVICE_STR]
        assert len(ds) >= 1

    def test_detect_cuda_with_spaces(self, tmp_path: Path) -> None:
        """'cuda: 0' (space after colon) should be detected."""
        src = _write(tmp_path, "a.py", 'device = torch.device("cuda: 0")\n')
        findings = scan_file(src)
        ds = [f for f in findings if f.pattern_type == PatternType.CUDA_DEVICE_STR]
        assert len(ds) >= 1

    def test_detect_cuda_two_arg_device(self, tmp_path: Path) -> None:
        """torch.device('cuda', device_id) should be detected."""
        src = _write(tmp_path, "a.py", 'dev = torch.device("cuda", device_id)\n')
        findings = scan_file(src)
        ds = [f for f in findings if f.pattern_type == PatternType.CUDA_DEVICE_STR]
        assert len(ds) >= 1


class TestAutocastDtype:
    """Scanner must detect autocast calls with dtype=torch.float32."""

    def test_detect_autocast_float32(self, tmp_path: Path) -> None:
        code = "with torch.amp.autocast('npu', dtype=torch.float32):\n    pass\n"
        src = _write(tmp_path, "a.py", code)
        findings = scan_file(src)
        ac = [f for f in findings if f.pattern_type == PatternType.AUTOCAST_DTYPE]
        assert len(ac) == 1

    def test_no_detect_autocast_without_dtype(self, tmp_path: Path) -> None:
        code = "with torch.amp.autocast('npu'):\n    pass\n"
        src = _write(tmp_path, "a.py", code)
        findings = scan_file(src)
        ac = [f for f in findings if f.pattern_type == PatternType.AUTOCAST_DTYPE]
        assert len(ac) == 0

    def test_detect_decorator_autocast(self, tmp_path: Path) -> None:
        code = "@torch.amp.autocast('npu', dtype=torch.float32)\ndef foo(): pass\n"
        src = _write(tmp_path, "a.py", code)
        findings = scan_file(src)
        ac = [f for f in findings if f.pattern_type == PatternType.AUTOCAST_DTYPE]
        assert len(ac) == 1


class TestDtypeAssert:
    """Scanner must detect assert ... .dtype == torch.float32."""

    def test_detect_single_assert(self, tmp_path: Path) -> None:
        code = "assert e.dtype == torch.float32\n"
        src = _write(tmp_path, "a.py", code)
        findings = scan_file(src)
        da = [f for f in findings if f.pattern_type == PatternType.DTYPE_ASSERT]
        assert len(da) == 1

    def test_detect_compound_assert(self, tmp_path: Path) -> None:
        code = "assert e.dtype == torch.float32 and e0.dtype == torch.float32\n"
        src = _write(tmp_path, "a.py", code)
        findings = scan_file(src)
        da = [f for f in findings if f.pattern_type == PatternType.DTYPE_ASSERT]
        assert len(da) == 1

    def test_detect_subscript_assert(self, tmp_path: Path) -> None:
        code = "assert e[0].dtype == torch.float32\n"
        src = _write(tmp_path, "a.py", code)
        findings = scan_file(src)
        da = [f for f in findings if f.pattern_type == PatternType.DTYPE_ASSERT]
        assert len(da) == 1

    def test_no_detect_other_dtype(self, tmp_path: Path) -> None:
        code = "assert e.dtype == torch.float16\n"
        src = _write(tmp_path, "a.py", code)
        findings = scan_file(src)
        da = [f for f in findings if f.pattern_type == PatternType.DTYPE_ASSERT]
        assert len(da) == 0


class TestCudaAmp:
    def test_detect_import_torch_cuda_amp(self, tmp_path: Path) -> None:
        """Detect `import torch.cuda.amp as amp`."""
        src = _write(tmp_path, "a.py", "import torch.cuda.amp as amp\n")
        findings = scan_file(src)
        assert len(findings) >= 1
        amp_findings = [f for f in findings if f.pattern_type == PatternType.CUDA_AMP]
        assert len(amp_findings) == 1
        assert amp_findings[0].line_number == 1

    def test_detect_from_torch_cuda_amp_import(self, tmp_path: Path) -> None:
        """Detect `from torch.cuda.amp import autocast, GradScaler`."""
        src = _write(tmp_path, "a.py", "from torch.cuda.amp import autocast, GradScaler\n")
        findings = scan_file(src)
        amp_findings = [f for f in findings if f.pattern_type == PatternType.CUDA_AMP]
        assert len(amp_findings) == 1

    def test_detect_import_torch_cuda_amp_plain(self, tmp_path: Path) -> None:
        """Detect `import torch.cuda.amp` (no alias)."""
        src = _write(tmp_path, "a.py", "import torch.cuda.amp\n")
        findings = scan_file(src)
        amp_findings = [f for f in findings if f.pattern_type == PatternType.CUDA_AMP]
        assert len(amp_findings) == 1


# ---------------------------------------------------------------------------
# PatternType: flash_attn_usage
# ---------------------------------------------------------------------------

class TestFlashAttnUsage:
    def test_flash_attn_usage_assert_detected(self, tmp_path: Path) -> None:
        """assert FLASH_ATTN_2_AVAILABLE should be detected as FLASH_ATTN_USAGE."""
        src = _write(tmp_path, "a.py", "FLASH_ATTN_2_AVAILABLE = False\nassert FLASH_ATTN_2_AVAILABLE\n")
        findings = scan_file(src)
        usage = [f for f in findings if f.pattern_type == PatternType.FLASH_ATTN_USAGE]
        assert len(usage) == 1
        assert usage[0].line_number == 2

    def test_flash_attn_usage_function_call_detected(self, tmp_path: Path) -> None:
        """flash_attn.flash_attn_varlen_func() should be detected."""
        code = "import flash_attn\nout = flash_attn.flash_attn_varlen_func(q, k, v)\n"
        src = _write(tmp_path, "a.py", code)
        findings = scan_file(src)
        usage = [f for f in findings if f.pattern_type == PatternType.FLASH_ATTN_USAGE]
        assert len(usage) == 1
        assert usage[0].line_number == 2

    def test_flash_attn_usage_not_false_positive_on_import(self, tmp_path: Path) -> None:
        """import flash_attn should NOT be detected as FLASH_ATTN_USAGE (it's FLASH_ATTN)."""
        src = _write(tmp_path, "a.py", "import flash_attn\n")
        findings = scan_file(src)
        usage = [f for f in findings if f.pattern_type == PatternType.FLASH_ATTN_USAGE]
        assert len(usage) == 0

    def test_flash_attn_usage_pattern_type_exists(self) -> None:
        """PatternType.FLASH_ATTN_USAGE must exist."""
        assert hasattr(PatternType, "FLASH_ATTN_USAGE")
        assert PatternType.FLASH_ATTN_USAGE.value == "flash_attn_usage"

    def test_flash_attn_usage_skips_comment_lines(self, tmp_path: Path) -> None:
        """Commented-out flash_attn usage (# [NPU] ...) should NOT generate findings."""
        code = (
            "import torch\n"
            "# [NPU] x = flash_attn.flash_attn_varlen_func(q, k, v)\n"
            "x = _npu_varlen_attention(q, k, v)\n"
        )
        src = _write(tmp_path, "a.py", code)
        findings = scan_file(src)
        usage = [f for f in findings if f.pattern_type == PatternType.FLASH_ATTN_USAGE]
        assert len(usage) == 0


# ---------------------------------------------------------------------------
# PatternType: autocast_no_device
# ---------------------------------------------------------------------------

class TestAutocastNoDevice:
    def test_detect_amp_autocast_missing_device(self, tmp_path: Path) -> None:
        """amp.autocast(dtype=...) without device_type should be detected."""
        code = "import torch.amp as amp\nwith amp.autocast(dtype=torch.float):\n    pass\n"
        src = _write(tmp_path, "a.py", code)
        findings = scan_file(src)
        hits = [f for f in findings if f.pattern_type == PatternType.AUTOCAST_NO_DEVICE]
        assert len(hits) == 1
        assert hits[0].line_number == 2

    def test_detect_decorator_autocast_missing_device(self, tmp_path: Path) -> None:
        """@amp.autocast(enabled=False) without device_type should be detected."""
        code = "import torch.amp as amp\n@amp.autocast(enabled=False)\ndef foo(): pass\n"
        src = _write(tmp_path, "a.py", code)
        findings = scan_file(src)
        hits = [f for f in findings if f.pattern_type == PatternType.AUTOCAST_NO_DEVICE]
        assert len(hits) == 1

    def test_skip_autocast_with_device(self, tmp_path: Path) -> None:
        """amp.autocast('npu', dtype=...) should NOT be detected (already has device_type)."""
        code = "import torch.amp as amp\nwith amp.autocast('npu', dtype=torch.bfloat16):\n    pass\n"
        src = _write(tmp_path, "a.py", code)
        findings = scan_file(src)
        hits = [f for f in findings if f.pattern_type == PatternType.AUTOCAST_NO_DEVICE]
        assert len(hits) == 0

    def test_skip_torch_cuda_amp_autocast(self, tmp_path: Path) -> None:
        """torch.cuda.amp.autocast(dtype=...) should NOT match (handled by CudaAmpRule)."""
        code = "import torch\nwith torch.cuda.amp.autocast(dtype=torch.float):\n    pass\n"
        src = _write(tmp_path, "a.py", code)
        findings = scan_file(src)
        hits = [f for f in findings if f.pattern_type == PatternType.AUTOCAST_NO_DEVICE]
        assert len(hits) == 0
