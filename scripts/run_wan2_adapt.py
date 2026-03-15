#!/usr/bin/env python3
"""Launch Wan2.2 TI2V-5B NPU adaptation via AdaptSupervisor.

This script directly invokes the Phase 2 supervised adaptation loop against
the Wan2.2 target repo.  It bypasses the LangGraph agent pipeline and
drives the supervisor directly for maximum observability.

Usage:
    # Set LLM env vars first, then:
    source .venv/bin/activate
    python scripts/run_wan2_adapt.py

Environment variables:
    DA_LLM_API_KEY       — LLM API key (DeepSeek)
    DA_LLM_BASE_URL      — LLM base URL (https://api.deepseek.com/v1)
    DA_LLM_MODEL         — LLM model name (deepseek-reasoner)
    DA_LLM_PROVIDER      — LLM provider (openai — DeepSeek is OpenAI-compatible)
    DA_NPU_SSH_HOST      — NPU server host (175.100.2.7)
    DA_NPU_CONDA_ENV     — Conda environment on NPU server
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from diffusion_agent.adapt.git_memory import reset_to_clean_main
from diffusion_agent.adapt.supervisor import AdaptSupervisor
from diffusion_agent.adapt.types import ExecutionConfig
from diffusion_agent.utils.logging import setup_logging


def main() -> None:
    setup_logging("DEBUG")

    # ---- Paths ----
    target_repo = ROOT / "target_wan2"
    if not target_repo.exists():
        print(f"ERROR: Target repo not found at {target_repo}")
        sys.exit(1)

    # ---- Clean reset DISABLED for V11 incremental E2E ----
    # We preserve V10 progress to test idempotency + rescan loop
    # print("[+] Resetting target repo to clean main branch...")
    # reset_to_clean_main(target_repo)
    # print("[+] Target repo reset complete.")
    print("[+] Incremental mode: preserving V15 progress (reset disabled for V16)")

    # ---- LLM setup ----
    llm = None
    api_key = os.environ.get("DA_LLM_API_KEY")
    if api_key:
        from diffusion_agent.config import load_settings
        from diffusion_agent.llm.provider import create_llm
        settings = load_settings()
        llm = create_llm(settings)
        print(f"[+] LLM configured: provider={settings.llm_provider}, model={settings.llm_model}")
    else:
        print("[!] No DA_LLM_API_KEY set — Phase B (LLM hypotheses) will be skipped")
        print("    Phase A (deterministic rules) will still run.")

    # ---- Remote execution config ----
    ssh_host = os.environ.get("DA_NPU_SSH_HOST", "175.100.2.7")
    conda_env = os.environ.get("DA_NPU_CONDA_ENV", "torch280_py310_diffusion")
    remote_workdir = os.environ.get("DA_REMOTE_WORKDIR", "/home/z00879328/07_WAN2_TEST")

    exec_config = ExecutionConfig(
        mode="ssh",
        ssh_host=ssh_host,
        ssh_user="root",
        remote_workdir=remote_workdir,
        conda_env=conda_env,
        timeout=600,  # Wan2.2 inference can be slow
        validation_command="bash run_npu_test.sh",
        sync_enabled=True,
        sync_delete=False,
        sync_prefer_scp=True,  # Remote host doesn't have rsync
    )

    print(f"[+] Target repo: {target_repo}")
    print(f"[+] SSH host: {ssh_host}")
    print(f"[+] Remote workdir: {remote_workdir}")
    print(f"[+] Conda env: {conda_env}")
    print(f"[+] Validation: bash run_npu_test.sh")
    print()

    # ---- Launch supervisor ----
    supervisor = AdaptSupervisor(
        repo_path=target_repo,
        model_name="Wan2.2-TI2V-5B",
        max_iterations=20,
        no_progress_limit=5,
        max_phase_c_iterations=10,
        llm=llm,
        use_git=True,
        execution_config=exec_config,
    )

    t0 = time.time()
    state = supervisor.run()
    elapsed = time.time() - t0

    # ---- Results ----
    print("\n" + "=" * 72)
    print("  WAN2.2 NPU ADAPTATION — RESULTS")
    print("=" * 72)
    print(f"  Stop reason     : {state.stop_reason.value if state.stop_reason else 'N/A'}")
    print(f"  Iterations      : {state.iteration}")
    print(f"  Files modified  : {len(state.files_modified)}")
    print(f"  Rules applied   : {state.total_rules_applied}")
    print(f"  Blockers        : {len(state.blockers)}")
    print(f"  Elapsed         : {elapsed:.1f}s")
    print()

    # Task summary
    for t in state.tasks:
        print(f"  [{t.status:>10}] {t.name} ({t.category.value})"
              f" — {t.attempt_count}/{t.max_attempts} attempts"
              + (f" [{t.stop_reason.value}]" if t.stop_reason else ""))

    # Blocker details
    if state.blockers:
        print("\n  BLOCKERS:")
        for b in state.blockers:
            print(f"    - {b}")

    # Iteration details
    if state.iterations:
        print("\n  ITERATION LOG:")
        for it in state.iterations:
            mark = "✓" if it.accepted else "✗"
            print(f"    [{mark}] #{it.iteration} {it.hypothesis.description[:60]}"
                  f" → {it.verdict.value}")

    # Save full report
    report_dir = target_repo / ".diffusion_agent" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "wan2-adapt-state.json"
    report_path.write_text(
        json.dumps(state.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n  Full report: {report_path}")

    # Blocker report
    blocker_report = supervisor.get_blocker_report()
    blocker_path = report_dir / "wan2-blocker-report.json"
    blocker_path.write_text(
        json.dumps(blocker_report, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"  Blocker report: {blocker_path}")
    print()


if __name__ == "__main__":
    main()
