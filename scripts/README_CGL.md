# CGL-PSO scripts

| 脚本 | 说明 |
|---|---|
| `cgl_run_all.ps1` | CGL-PSO 全流程：doctor → 测试 → pilot → 预注册主筛选 → （可选）CLIP 合成 smoke → 报告 |
| `stage2_report.py` / `stage3_report.py` | 第二、三阶段文档再生成（由各自 CLI 的 `report` 调用） |
| `run_all.ps1` / `run_all.sh` | GLDI-PSO（前三阶段）全流程 |

```powershell
# CGL 快速验证（不跑主筛选）
powershell -ExecutionPolicy Bypass -File scripts/cgl_run_all.ps1 -Quick

# CGL 完整流程 + CLIP 合成 smoke
powershell -ExecutionPolicy Bypass -File scripts/cgl_run_all.ps1 -WithClipSmoke
```

预算提示：主筛选固定为 4 任务 × 10 种子 × 10,000 FEs × ≤8 算法 = 3.2M FEs，
加上 pilot 360k 已接近 `docs/CGL_PROTOCOL.md` 规定的 4M 上限；**不要重复运行主筛选**。
实际执行记录见 `docs/CGL_RESULTS_zh.md` 的预算账本（含一次如实记录的超支）。
