"""Tests for adapt/patch_worker.py — patch generation and application."""

from __future__ import annotations

from pathlib import Path

from diffusion_agent.adapt.patch_worker import PatchWorker
from diffusion_agent.adapt.types import FailureCategory, Hypothesis
from diffusion_agent.tools.code_migrator import create_default_registry
from diffusion_agent.tools.code_scanner import Finding, PatternType


def _finding(file_path: str, pattern: PatternType, line: int = 1, snippet: str = "test") -> Finding:
    return Finding(
        file_path=file_path, line_number=line,
        pattern_type=pattern, code_snippet=snippet, context="",
    )


def _hypothesis(target_files: list[str], source: str = "rule") -> Hypothesis:
    return Hypothesis(
        id="h-1", category=FailureCategory.DEVICE_SELECTION,
        description="test", target_files=target_files,
        proposed_action="Apply rules", source=source,
    )


class TestApplyRulePatch:
    def test_applies_cuda_call_rule(self, tmp_path: Path) -> None:
        fp = tmp_path / "model.py"
        fp.write_text("import torch\nx = tensor.cuda()\n")

        registry = create_default_registry()
        worker = PatchWorker(registry)
        findings = [_finding(str(fp), PatternType.CUDA_CALL, line=2)]
        hyp = _hypothesis([str(fp)])

        result = worker.apply_rule_patch(hyp, findings)
        assert result.success
        assert len(result.files_changed) == 1
        assert "cuda_call" in result.rules_applied

        content = fp.read_text()
        assert ".npu()" in content
        assert "import torch_npu" in content

    def test_no_findings_for_target(self, tmp_path: Path) -> None:
        registry = create_default_registry()
        worker = PatchWorker(registry)
        findings = [_finding("other.py", PatternType.CUDA_CALL)]
        hyp = _hypothesis(["model.py"])

        result = worker.apply_rule_patch(hyp, findings)
        assert result.success
        assert result.files_changed == []

    def test_multiple_rules_same_file(self, tmp_path: Path) -> None:
        fp = tmp_path / "model.py"
        fp.write_text('import torch\nx = tensor.cuda()\ny = model.to("cuda")\n')

        registry = create_default_registry()
        worker = PatchWorker(registry)
        findings = [
            _finding(str(fp), PatternType.CUDA_CALL, line=2),
            _finding(str(fp), PatternType.CUDA_TO, line=3),
        ]
        hyp = _hypothesis([str(fp)])

        result = worker.apply_rule_patch(hyp, findings)
        assert result.success
        assert len(result.rules_applied) == 2

        content = fp.read_text()
        assert ".npu()" in content
        assert '"npu"' in content


class TestApplyLlmPatch:
    def test_no_llm_configured(self, tmp_path: Path) -> None:
        registry = create_default_registry()
        worker = PatchWorker(registry, llm=None)
        findings = [_finding("model.py", PatternType.DISTRIBUTED)]
        hyp = _hypothesis(["model.py"], source="llm")

        result = worker.apply_llm_patch(hyp, findings)
        assert result.success  # graceful skip
        assert result.files_changed == []
        assert "not configured" in result.description


class TestApplyBatchRules:
    def test_batch_applies_all(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("import torch\nx = tensor.cuda()\n")
        (tmp_path / "b.py").write_text('import torch\ny = model.to("cuda")\n')

        registry = create_default_registry()
        worker = PatchWorker(registry)
        findings = [
            _finding(str(tmp_path / "a.py"), PatternType.CUDA_CALL, line=2),
            _finding(str(tmp_path / "b.py"), PatternType.CUDA_TO, line=2),
        ]

        result = worker.apply_batch_rules(findings)
        assert result.success
        assert len(result.files_changed) == 2
        assert "cuda_call" in result.rules_applied
        assert "cuda_to" in result.rules_applied

        # Verify torch_npu was added
        assert "import torch_npu" in (tmp_path / "a.py").read_text()
        assert "import torch_npu" in (tmp_path / "b.py").read_text()

    def test_batch_empty(self) -> None:
        registry = create_default_registry()
        worker = PatchWorker(registry)
        result = worker.apply_batch_rules([])
        assert result.success
        assert result.files_changed == []


class TestApplyPatch:
    def test_dispatches_to_rule(self, tmp_path: Path) -> None:
        fp = tmp_path / "model.py"
        fp.write_text("import torch\nx = tensor.cuda()\n")

        registry = create_default_registry()
        worker = PatchWorker(registry)
        findings = [_finding(str(fp), PatternType.CUDA_CALL, line=2)]
        hyp = _hypothesis([str(fp)], source="rule")

        result = worker.apply_patch(hyp, findings)
        assert result.success
        assert ".npu()" in fp.read_text()

    def test_dispatches_to_llm(self) -> None:
        registry = create_default_registry()
        worker = PatchWorker(registry, llm=None)
        findings = [_finding("model.py", PatternType.DISTRIBUTED)]
        hyp = _hypothesis(["model.py"], source="llm")

        result = worker.apply_patch(hyp, findings)
        assert result.success  # graceful skip when no LLM
