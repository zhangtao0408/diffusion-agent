# Standing Rules

These rules constrain the agent's behavior throughout all phases.

## Code Changes
- One feature per commit
- Run tests before committing
- Never commit broken code intentionally
- Preserve existing functionality when making changes

## Ascend NPU Migration
- Replace `.cuda()` with `.npu()` consistently
- Replace NCCL with HCCL for distributed training
- Avoid float64 (double) — use float32 instead
- Set appropriate ACL environment variables

## Quality
- Log every significant action to daily-log.md
- Update current-task.json before and after each task
- Report errors clearly — never silently skip failures
