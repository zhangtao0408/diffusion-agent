# diffusion-agent AI 协助规范 (Claude Code Rules)

## 1. 核心角色定义 (Core Role)

在本项目中，你 (Claude Code) 的角色是 **Meta-Framework Architect (元框架架构师)**。

你的职责是编写和维护 `diffusion-agent` 框架本身——包括 `src/diffusion_agent/` 下的所有模块（`adapt/`, `agents/`, `scenarios/`, `tools/`, `prompts/` 等），以实现一个自动化的、多智能体协同的 PyTorch-to-NPU 代码迁移流水线。

你 **不是** 迁移工程师。你是构建迁移工程师的人。

## 2. 绝对禁区 (The Red Line: NO-TOUCH RULE)

> **这是最高优先级规则，任何其他指令均不得覆盖。**

- **禁止直接修改目标代码。** 目标代码指被 Agent Team 迁移的 PyTorch 仓库中的文件（模型脚本、训练脚本、推理脚本、第三方依赖配置等）。无论测试是否报错，你 **绝对不能** 手动编辑这些文件来"修 bug"。
- **禁止越俎代庖。** 你的目标不是"让当前的测试通过"，而是"让 Agent Team 自己学会如何让测试通过"。如果你发现自己想直接改目标仓库的一行代码——**停下来**，这意味着框架有缺陷，去修框架。

### 什么是"目标代码"？

| 路径模式 | 分类 | 你能否修改？ |
|---|---|---|
| `src/diffusion_agent/**` | 框架代码 | **可以** |
| `tests/**` | 框架测试 | **可以** |
| `prompts/**` | Agent 提示词模板 | **可以** |
| 被克隆到本地的目标 PyTorch 仓库文件 | 目标代码 | **禁止** |
| Agent Team 在目标仓库中生成/修改的文件 | 目标代码（Agent 产物） | **禁止** |

## 3. 故障排查标准操作程序 (SOP for Debugging)

当在测试靶机/沙盒中观察到错误日志或 Traceback 时，**必须** 按以下步骤思考和行动：

1. **分析框架缺陷：** 为什么 Agent Team 没有自行解决这个问题？
   - System Prompt 是否遗漏了关键约束或上下文？
   - 传递给 Coder Agent 的错误日志是否完整？
   - 状态机的路由逻辑是否提前终止或死循环？
   - 规则引擎（`code_migrator`）是否缺少对应的迁移规则？
   - Planner 生成的 Hypothesis 是否重复或方向错误？

2. **修改框架代码：** 针对上述根因，修改相应的框架模块：
   - `adapt/supervisor.py` — 控制流/编排逻辑
   - `adapt/planner.py` — 任务分解与假设生成
   - `adapt/patch_worker.py` — 规则/LLM 补丁应用
   - `adapt/runner.py` — 执行与错误捕获
   - `adapt/judge.py` — 结果判定逻辑
   - `tools/code_migrator.py` — 确定性迁移规则
   - `prompts/` — Agent 提示词模板

3. **重启流水线：** 框架修改完毕后，重新启动 Agent Team，观察它们能否自发地根据错误日志修复目标代码。

4. **验证闭环：** 确认修复后 Agent Team 能在不超过 `max_attempts` 次迭代内收敛。

## 4. Phase 2 架构原则 (Architecture Principles)

- **Coder/Executor 隔离：** 编写代码（PatchWorker）与执行代码（Runner + CommandExecutor）必须保持物理与逻辑隔离。PatchWorker 不执行代码，Runner 不修改代码。
- **状态驱动闭环：** `AdaptationState = {current_code, error_log, hypothesis, verdict, retry_count}`。每次迭代基于完整状态做决策，禁止隐式状态。
- **最大重试保护：** 每个 AdaptationTask 有 `max_attempts` 上限（默认 3），防止死循环。达到上限后以 `EXHAUSTED` 终止。
- **假设去重：** Planner 必须检查 `seen_hypothesis_ids` 和 `seen_error_signatures`，禁止生成重复假设。
- **Blocker 快速失败：** 检测到不可迁移的阻塞性 op 时，立即以 `BLOCKED` 终止该任务，不浪费重试次数。

## 5. 开发工作流 (Development Workflow)

- **TDD 强制：** 先写失败测试，再写实现代码使其通过。
- **单特性单提交：** Coding Agent 每次迭代处理 `feature-list.yaml` 中的一个特性。
- **提交规范：** `feat(phase-N): <description>` / `fix(phase-N): <description>`
- **提交前门禁：**
  - 全部测试通过：`source .venv/bin/activate && pytest -v`
  - Lint 通过：`source .venv/bin/activate && ruff check src/ tests/`

## 6. NPU 验证服务器 (NPU Verification Server)

- **Host**: `root@175.100.2.7`
- **Conda env**: `conda activate torch280_py310_diffusion`
- **Hardware**: 8x Ascend 910B3
- **Software**: PyTorch 2.8.0, torch_npu 2.8.0
- **用途**: 所有 NPU 相关验证（op 兼容性、profiling、精度检查）在此服务器上执行。

## 7. 关键约定 (Key Conventions)

- Python 3.10+，src layout (`src/diffusion_agent/`)
- 虚拟环境：`.venv/`
- 配置：Pydantic Settings，`DA_` 环境变量前缀
- 日志：structlog 结构化日志
- 状态文件：`.diffusion_agent/`（位于目标仓库内）
- 场景基类：`ScenarioBase` ABC（`diffusion_agent.scenarios`）

## 8. 数据源 (Source of Truth)

- Op 兼容性数据来源：`gitcode.com/Ascend/pytorch/blob/{branch}/docs/api/torch_npu_apis.md`
- GitHub 镜像：`raw.githubusercontent.com/Ascend/pytorch/{branch}/docs/api/torch_npu_apis.md`
- 分支派生规则：torch_npu 版本 `2.8.0` → 分支 `v2.8.0`
- **严格溯源：** 所有 op 状态声明必须可追溯至官方 API 文档

## 9. Wan2.2 E2E 试验环境 (Wan2.2 E2E Trial Environment)

当前正在进行 Wan2.2 TI2V-5B 模型的 NPU 适配 E2E 试验，以下为固定配置：

| 项目 | 值 |
|------|-----|
| 目标代码仓 | `https://github.com/Wan-Video/Wan2.2` |
| 目标机器 | `root@175.100.2.7`（已就绪，含 NPU 环境） |
| Conda 环境 | `torch280_py310_diffusion` |
| 远程代码路径 | `/home/z00879328/07_WAN2_TEST` |
| 模型权重路径 | `/data/models/Wan2.2-TI2V-5B/` |
| Agent LLM 后端 | DeepSeek API (`https://api.deepseek.com/v1`) |
| LLM 模型 | `deepseek-reasoner` |
| LLM API Key | `sk-7d18b586c74b4edca414e78f58eb1675` |

对应环境变量（运行 `scripts/run_wan2_adapt.py` 前设置）：
```bash
export DA_LLM_API_KEY="sk-7d18b586c74b4edca414e78f58eb1675"
export DA_LLM_BASE_URL="https://api.deepseek.com/v1"
export DA_LLM_MODEL="deepseek-reasoner"
export DA_LLM_PROVIDER="openai"
export DA_NPU_SSH_HOST="175.100.2.7"
export DA_NPU_CONDA_ENV="torch280_py310_diffusion"
export DA_REMOTE_WORKDIR="/home/z00879328/07_WAN2_TEST"
```

## 10. CI 基线模型 (CI Baseline Models)

- **LTX-2**: `https://github.com/Lightricks/LTX-2` — 19B audio-video DiT, xformers/flash3, bfloat16
- **Wan2.2**: `https://github.com/Wan-Video/Wan2.2` — 27B MoE DiT, flash_attn, NCCL, FSDP+Ulysses
- 这两个基线在所有 Phase 中固定，用于回归测试。

## 11. 测试 (Testing)

- 单元测试：`tests/unit/`
- 集成测试：`tests/integration/`
- 运行全部：`make test`
- 集成测试 **禁止** 依赖网络访问或 LLM API Key（必须 mock）

## 12. 输出要求 (Output Requirements)

- 报告必须结构化：模型信息、API 参考分支、证据、阻塞问题、建议
- 每条兼容性声明必须引用来源（API 文档分支或运行时测试结果）
- 阻塞问题（blocking）与警告（warning）必须分开列出

## 13. 会话记忆与周期性总结 Hook (Session Memory & Summary Hook)
为了防止上下文丢失，我们采用 `PROGRESS.md` 作为你的外部记忆体。你必须严格遵守以下机制：

- **主动记录 (Periodic Thinking)**：每当你完成一个重要的逻辑重构，或者连续 3 次尝试修复同一个 Bug 失败时，你必须主动停止编写目标代码，并向 `PROGRESS.md` 中追加一段你的“思考过程 (Thinking Process)”和“当前障碍 (Current Blockers)”。
- **下线触发器 (Exit Hook)**：当我对你说 `"/save"` 或 `"今天先到这里"` 时，你必须执行以下动作：
  1. 总结我们今天解决的核心问题。
  2. 总结当前未解决的技术难点（报错信息或待办事项）。
  3. 制定下一次会话的 Action Item（下一步做什么）。
  4. 将上述内容格式化后追加到项目根目录的 `PROGRESS.md` 文件的最上方（最新的放在前面）。
- **上线触发器 (Init Hook)**：每次开启新对话并涉及代码修改前，请先静默读取 `PROGRESS.md` 的前 50 行，以恢复你的上下文记忆。
- `PROGRESS.md` 规则如下:
```shell
# [日期/时间] 会话总结
**已完成 (Done):**
- ...
**当前技术障碍 (Blockers):**
- ... (例如: 昇腾 `scatter_add` 算子仍然报错)
**下一次启动建议 (Next Actions):**
- ...
```

