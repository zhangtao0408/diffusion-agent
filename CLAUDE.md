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

## Testing
- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`
- Run all: `make test`
- Integration tests should NOT require network access or LLM API keys (mock them)
