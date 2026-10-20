# CGL-PSO scripts

| 脚本 | 说明 |
|---|---|
| `cgl_run_all.ps1` | CGL-PSO 全流程：doctor → 测试 → pilot → 预注册主筛选 → （可选）CLIP 合成 smoke → 报告 |
| `verify_export.py` | 导出侧单格复现比对：reuse-only 锚点 + 修复后代码 vs 已提交产物 |
| `cgl_sw_verify_repro.py` | 零 FEs 复核：ring/gbest 逐位一致、smallworld 判定复算、预算账本 |
| `cgl_sw_resummarise.py` | 从 `runs.jsonl` 重导出 summary/verdict（零 FEs） |
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

## 2026-10-20 修复轮（审查后）

```powershell
# 全量测试（56 项）
python -m pytest tests -q

# 单格复现比对（输出应包含 ANCHOR MATCHES: True）
python scripts/verify_export.py

# 零 FEs 的一致性 / 判定 / 预算复核
python scripts/cgl_sw_verify_repro.py
```

预算提示补充：Ring-SmallWorld 机制筛选一轮共 3.6M FEs（预试验 960k + 诊断 240k + 主筛选 2.4M），上限 3.8M；**不要重复运行** `cgl smallworld --stage pre|main`，结果已在 `results/cgl/sw_*` 中。
