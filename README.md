# Diffusion Agent

Agent-integrated system for enabling end-to-end PyTorch model support on Huawei Ascend NPU. Built on a Two-Stage Architecture (Init Agent → Coding Agent) with LangGraph.

## What It Does

Diffusion Agent automates five common workflows for porting PyTorch models to Ascend NPU:

| Scenario | Command | Status |
|----------|---------|--------|
| **Check Support** — Is this model adapted for Ascend? | `--scenario check` | Available |
| **Adapt Model** — Migrate CUDA code to NPU | `--scenario adapt` | Planned |
| **Performance Analysis** — Profile and find bottlenecks | `--scenario analyze` | Planned |
| **Performance Optimization** — Apply optimizations | `--scenario optimize` | Planned |
| **Accuracy Verification** — Compare CPU/CUDA vs NPU | `--scenario verify` | Planned |

## Quick Start

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Run

```bash
# Check if a local repo is adapted for Ascend
python -m diffusion_agent --local-path /path/to/model/repo --scenario check --model my-model

# Check a remote repo
python -m diffusion_agent --repo https://github.com/user/model-repo --scenario check --model my-model
```

### Output

The check scenario produces:
- `.diffusion_agent/reports/check-report.json` — structured compatibility data
- `.diffusion_agent/reports/check-report.md` — human-readable report with verdict, findings, and recommendations

### Configuration

Environment variables (prefix `DA_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DA_LLM_PROVIDER` | `openai` | LLM provider: `openai`, `anthropic`, `local` |
| `DA_LLM_MODEL` | `gpt-4o` | Model identifier |
| `DA_LLM_API_KEY` | — | API key for the LLM provider |
| `DA_LLM_BASE_URL` | — | Custom API base URL (for local/vLLM) |
| `DA_LOG_LEVEL` | `INFO` | Logging level |
| `DA_WORK_DIR` | `./workspace` | Working directory for cloned repos |

Copy `.env.example` to `.env` and fill in your values.

## Architecture

```
User (CLI)
    │
    ▼
┌──────────────────────────────────────────┐
│           LangGraph StateGraph           │
│                                          │
│  ┌────────────┐     ┌────────────────┐   │
│  │ Init Agent │────▶│ Coding Agent   │   │
│  │ (Stage 1)  │     │ (Stage 2)      │   │
│  │            │     │                │   │
│  │ - Clone    │     │ - Read task    │   │
│  │ - Scaffold │     │ - Run scenario │   │
│  │ - Plan     │     │ - Commit       │   │
│  │ - Git init │     │ - Next task    │   │
│  └────────────┘     └────────────────┘   │
└──────────────────────────────────────────┘
    │                          │
    ▼                          ▼
 State Files               Git History
 (.diffusion_agent/)       (audit trail)
```

### Two-Stage Architecture

- **Stage One (Init Agent)**: Scaffolding only — clones repo, creates state files, plans features. No business code.
- **Stage Two (Coding Agent)**: Reads tasks from `feature-list.yaml`, executes one at a time via the appropriate scenario module, commits results, and advances to the next task.

### State Management (Three-File Pattern)

All state lives in `.diffusion_agent/` inside the target repo:

| File | Purpose |
|------|---------|
| `feature-list.yaml` | Ordered task list with status tracking |
| `daily-log.md` | Append-only activity log with timestamps |
| `standing-rules.md` | Constraints the agent must follow |

## Development

```bash
make install   # Install in dev mode
make test      # Run pytest
make lint      # Run ruff
make format    # Auto-format code
```

### Project Structure

```
src/diffusion_agent/
├── agents/          # LangGraph orchestration (graph, init_agent, coding_agent, router)
├── scenarios/       # Business logic per scenario (check_support, ...)
├── tools/           # Deterministic tools (code_scanner, torch_npu_checker, git_ops)
├── state_mgmt/      # Three-File Pattern read/write
├── llm/             # LLM provider factory
├── data/            # Bundled data (op support matrix)
├── config.py        # Pydantic Settings
└── cli.py           # Typer CLI
```

## Roadmap

### Phase 0 — Project Skeleton
- [x] LangGraph two-stage pipeline
- [x] Three-File Pattern state management
- [x] Configurable LLM provider (OpenAI/Anthropic/local)
- [x] Typer CLI, structlog logging
- [x] Git operations and shell tools

### Phase 1 — Check Ascend Support
- [x] AST-based code scanner (CUDA pattern detection)
- [x] torch_npu op compatibility checker (42-op matrix)
- [x] Check scenario with JSON/Markdown report generation
- [x] End-to-end integration tests (79 tests total)

### Phase 2 — Adapt Model to Ascend
- [ ] Automated `.cuda()` → `.npu()` migrator
- [ ] LLM-guided migration for complex cases (custom kernels, unsupported ops)
- [ ] Per-file feature decomposition and commits

### Phase 3 — Performance Analysis
- [ ] `torch_npu.npu.profile` and `msprof` integration
- [ ] Trace parser with op-level timing extraction
- [ ] Bottleneck identification and recommendations report

### Phase 4 — Performance Optimization
- [ ] Optimization pattern library (fused ops, AMP, memory format)
- [ ] LLM-guided optimization selection
- [ ] Before/after profiling comparison

### Phase 5 — Accuracy Verification
- [ ] CPU/CUDA baseline runner
- [ ] Tensor comparison (allclose, cosine similarity, per-layer diff)
- [ ] Pass/fail report with configurable tolerances

### Phase 6 — Polish
- [ ] API mode (FastAPI)
- [ ] Resume from checkpoint
- [ ] Interactive mode (human-in-the-loop)
- [ ] Multi-scenario chaining (check → adapt → verify)

## License

MIT
