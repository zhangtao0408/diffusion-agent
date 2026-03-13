"""CI baseline runner — check LTX-2 and Wan2.2 for regression comparison."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

from diffusion_agent.tools.code_scanner import scan_directory
from diffusion_agent.tools.torch_npu_checker import (
    CheckResult,
    OpStatus,
    check_pattern,
    get_compatibility_summary,
)
from diffusion_agent.utils.logging import get_logger

log = get_logger(__name__)

CI_BASELINES: dict[str, dict[str, str]] = {
    "ltx2": {
        "url": "https://github.com/Lightricks/LTX-2",
        "description": "19B audio-video DiT, uses xformers/flash3, bfloat16",
    },
    "wan22": {
        "url": "https://github.com/Wan-Video/Wan2.2",
        "description": "27B MoE DiT, uses flash_attn, NCCL, FSDP+Ulysses",
    },
}

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "diffusion_agent" / "baselines"


def _clone_repo(url: str, dest: Path) -> Path:
    """Shallow-clone a git repo to dest. Returns the repo path."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(dest)],
        check=True,
        capture_output=True,
        timeout=300,
    )
    return dest


def run_baseline_check(
    baseline_name: str,
    version: str | None = None,
    cache_dir: Path | None = None,
) -> dict:
    """Clone a baseline model repo, scan it, and return a summary report dict."""
    info = CI_BASELINES.get(baseline_name)
    if info is None:
        raise ValueError(f"Unknown baseline: {baseline_name}. Must be one of: {list(CI_BASELINES)}")

    base = cache_dir or DEFAULT_CACHE_DIR
    repo_dir = base / "repos" / baseline_name
    repo_path = _clone_repo(info["url"], repo_dir)

    # Scan
    findings = scan_directory(repo_path)
    log.info("baseline_scan_complete", baseline=baseline_name, count=len(findings))

    # Check each unique pattern type
    unique_patterns = list({f.pattern_type.value for f in findings})
    results: list[CheckResult] = [check_pattern(p, version=version) for p in unique_patterns]

    # Build summary
    findings_by_type = dict(Counter(f.pattern_type.value for f in findings))
    compat_summary = get_compatibility_summary(results)

    # Determine verdict
    if not results:
        verdict = "compatible"
    else:
        statuses = [r.status for r in results]
        unsupported = sum(1 for s in statuses if s is OpStatus.UNSUPPORTED)
        supported = sum(1 for s in statuses if s is OpStatus.SUPPORTED)
        total = len(statuses)
        if supported == total:
            verdict = "compatible"
        elif unsupported > total / 2:
            verdict = "incompatible"
        else:
            verdict = "partially_compatible"

    return {
        "baseline_name": baseline_name,
        "repo_url": info["url"],
        "verdict": verdict,
        "findings_by_type": findings_by_type,
        "compatibility": compat_summary,
        "total_findings": len(findings),
        "pattern_count": len(unique_patterns),
    }


def compare_with_baselines(
    target: dict, baselines: dict[str, dict],
) -> dict:
    """Compare target model's findings with baseline reports.

    Returns a dict keyed by baseline name with shared/unique pattern analysis.
    """
    comparison: dict[str, dict] = {}
    target_types = set(target.get("findings_by_type", {}).keys())

    for name, baseline in baselines.items():
        baseline_types = set(baseline.get("findings_by_type", {}).keys())
        comparison[name] = {
            "shared_patterns": sorted(target_types & baseline_types),
            "target_only": sorted(target_types - baseline_types),
            "baseline_only": sorted(baseline_types - target_types),
            "baseline_verdict": baseline.get("verdict", "unknown"),
        }

    return comparison


def load_or_run_baseline(
    name: str,
    version: str,
    cache_dir: Path | None = None,
) -> dict:
    """Load baseline results from cache, or run the check and cache results."""
    from diffusion_agent.tools.api_doc_fetcher import resolve_branch

    base = cache_dir or DEFAULT_CACHE_DIR
    branch = resolve_branch(version)
    cache_file = base / f"{name}_{branch}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    result = run_baseline_check(name, version=version, cache_dir=base)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
