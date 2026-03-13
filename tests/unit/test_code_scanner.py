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
