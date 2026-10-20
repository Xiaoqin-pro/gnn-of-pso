# CGL-DECISION — 机制与路线判定

## 结论：**未通过开发门槛（精简或停止）**

- `cgl_full` 相对 `cgl_nocontrib`：**3/4** 任务改善（判据要求 ≥3/4）；
- 相对经典基线：明显更差的任务数 3（判据要求 0）；
- 运行时间中位倍数：6.5209231372638（判据要求 ≤3）；
- 判定为**开发门槛**，不是统计显著性；10 种子不构成论文结论。

### 后续动作

### 最终判定：**停止**该机制路线（保留负结果）

1. **机制自身门槛**：主筛选中 `cgl_full` 相对 `cgl_nocontrib` 仅在 3/4 任务上取得微弱改善（配对中位 Δlog10 −0.096 ~ +0.015）；
2. **修复复测证伪**：去掉有害的 far/random 候选源后绝对性能确实改善（4/4），但 `cgl_full` 相对无贡献版的优势**随之消失**（仅 1/4）；
3. **家族级门槛失败**：`cgl_full` 在 3/4 任务上明显劣于 ring/gbest，运行时间中位倍数为 6.5×（阈值 3×）；
4. **预算已用尽**（并已超支，见结果文档），不再执行新的数值实验。

**结论**：贡献记忆（M1）没有稳健净收益；方向评分与停滞刷新也未显示独立价值。CGL-PSO 作为算法路线**停止**；若后续仍要做 PSO 改进，应回到更简单、开销更低的机制（例如仅在 ring 拓扑上做停滞驱动的邻域半径调整），并用更小的预算先验证。

### 保留的可复用成果

- 可运行的 CGL-PSO 实现与 5 个可消融变体、独立 CLI 与测试；
- 在线贡献图的**诊断方法**（贡献项离散度、决策翻转率、共性分量剔除）——这些诊断揭示了「决策被大幅改变但结果无改善」这一关键失败模式；
- 合成 logits 的 CLIP 校准 smoke 全流程（见 `docs/CGL_CLIP_APPLICATION.md`）。

---

## 审查后补充（2026-10-20）：实现修复 + Ring-SmallWorld 机制判定

### A. 四项实现问题的修复不改变 CGL 判定

审查指出的三处实现问题（`global_gain` 误用 `max`、停滞惩罚缺 `stalled` 掩码、
最远邻居 `argmax` 把 self 置 `+inf`）已修复并补回归测试，self 候选也改成了显式开关。
逐条 before/after 与测试见 `docs/CGL_REVIEW_RESPONSE.md`。

用修复后的代码在**相同的 4 任务 × 10 配对种子 × 30D × 40 粒子 × 10,000 FEs**
上重跑（`results/cgl/sw_main/`，标签 `cgl_full_fixed`）：

| 任务 | ring（重跑，逐位等于 dev10） | gbest（重跑，逐位等于 dev10） | cgl_full（修复前） | cgl_full_fixed（修复后） |
|---|---|---|---|---|
| alpine1 | 0.842 | 0.523 | 1.195 | 1.211 |
| griewank | 0.070 | -0.717 | 1.187 | 1.145 |
| rosenbrock | 1.990 | 1.673 | 2.145 | 2.173 |
| zakharov | 1.893 | 1.418 | 1.723 | 1.744 |
| 均值 | 1.198 | 0.724 | 1.5627 | 1.5683 |

**结论：修复后 CGL 表现基本不变（均值 +0.006），"停止 CGL 机制路线"的判定维持。**

### B. Ring-SmallWorld-PSO：只有"停滞触发"版本通过低开销门槛

预注册判据：相对 ring ≥1 个任务改善、无任务劣化 > +0.5、运行时间 ≤1.5×。

| 候选 | 改善任务数 | 崩塌 | 运行时间倍数 | 判定 |
|---|---|---|---|---|
| `sw_fixed_p15` | 4/4 | 0 | **1.72×** | ✗ 超时（但方向最一致） |
| `sw_stall_t10_p05` | 2/4 | 0 | **1.44×** | ✅ 唯一通过，推荐 |
| `sw_bandit_p05` | 1/4 | 0 | 1.78× | ✗ 在线学习无净收益 |
| `cgl_full_fixed` | 1/4 | 1 | 6.56× | ✗ |

* 收益几乎全部来自 zakharov（各 small-world 变体 10W/0L），
  griewank/rosenbrock 上最好也只是打平；**不得**把本轮说成"小世界拓扑显著优于 ring"；
* `p=0.15` 的固定连接把末端多样性从 0.085 压到 0.034（近腰斩），这是它的隐性代价；
* 在线 bandit 模块按预注册规则**删除**（固定版 1.035 < 在线版 1.278）。

### C. 建议的下一步（尚未执行）

1. 若继续 small-world 方向：先做 `τ ∈ {5,10,20}` 的敏感性预试验，
   再考虑在 CEC2022 子集上验证（需另行批准预算）；
2. 无论哪条路线，**都应先通过"低开销 + 不损伤"门槛**，
   再讨论是否值得引入 GNN/PPO 这类高开销决策器；
3. 完整结果、预算账本与局限见 `docs/CGL_SMALLWORLD_RESULTS_zh.md`。

## 复现命令

```bash
python -m cgl_pso doctor
python -m cgl_pso smoke -c configs/cgl/smoke.yaml
python -m cgl_pso dev10 -c configs/cgl/dev10.yaml
python -m cgl_pso report
python -m cgl_pso smallworld -c configs/cgl/smallworld.yaml --stage pre --diagnostics
python -m cgl_pso smallworld -c configs/cgl/smallworld.yaml --stage main
```
