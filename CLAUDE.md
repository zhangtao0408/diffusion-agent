# Agent Rules for diffusion-agent

## Project Context
This is an Agent-integrated system for enabling PyTorch model support on Huawei Ascend NPU, built on a Two-Stage Architecture (Init Agent → Coding Agent) with LangGraph.

## Development Workflow
- **TDD is mandatory**: Write failing tests first, then implement code to make them pass.
- **One feature per commit**: The coding agent processes one feature from feature-list.yaml per iteration.
- **Commit convention**: `feat(phase-N): <description>` for features, `fix(phase-N): <description>` for fixes.
- **All tests must pass** before any commit: `source .venv/bin/activate && pytest -v`
- **Lint must pass** before any commit: `source .venv/bin/activate && ruff check src/ tests/`

## NPU Verification Server
- **Host**: `root@175.100.2.7`
- **Conda environment**: `conda activate torch280_py310_diffusion`
- **Hardware**: 8x Ascend 910B3
- **Software**: PyTorch 2.8.0, torch_npu 2.8.0
- **Usage**: All NPU-related verification (op compatibility, profiling, accuracy checks) should be validated on this server.
- **SSH command**: `ssh root@175.100.2.7` then `conda activate torch280_py310_diffusion`

## Key Conventions
- Python 3.10+, src layout (`src/diffusion_agent/`)
- Virtual environment at `.venv/`
- Config via Pydantic Settings with `DA_` env prefix
- Structured logging via structlog
- State files live in `.diffusion_agent/` inside the target repo
- Scenarios implement `ScenarioBase` ABC from `diffusion_agent.scenarios`

## Source of Truth
- Op compatibility data comes from `gitcode.com/Ascend/pytorch/blob/{branch}/docs/api/torch_npu_apis.md`
- GitHub mirror: `raw.githubusercontent.com/Ascend/pytorch/{branch}/docs/api/torch_npu_apis.md`
- Branch is derived from the torch_npu version (e.g., `2.8.0` → branch `v2.8.0`)
- Strict source correctness: all op status claims must trace back to official API docs

## CI Baseline Models
- **LTX-2**: `https://github.com/Lightricks/LTX-2` — 19B audio-video DiT, uses xformers/flash3, bfloat16
- **Wan2.2**: `https://github.com/Wan-Video/Wan2.2` — 27B MoE DiT, uses flash_attn, NCCL, FSDP+Ulysses
- These two baselines are fixed across all phases for regression testing

## Execution Environment
- **Host**: `root@175.100.2.7`
- **Conda env**: `torch280_py310_diffusion`
- **Hardware**: 8x Ascend 910B3
- **Software**: PyTorch 2.8.0, torch_npu 2.8.0

## Workflow
1. Identify CUDA/GPU patterns in target repository
2. Check each pattern against Ascend API docs (source of truth)
3. Compare findings with CI baseline models (LTX-2, Wan2.2)
4. Runtime-test critical ops on NPU server
5. Summarize with structured, evidence-based output

## Output Requirements
- Reports must be structured with: model info, API reference branch, evidence, blocking issues, recommendations
- Every compatibility claim must cite its source (API doc branch or runtime test)
- Separate blocking issues (prevent running) from warnings (suboptimal but functional)

## Testing
- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`
- Run all: `make test`
- Integration tests should NOT require network access or LLM API keys (mock them)
