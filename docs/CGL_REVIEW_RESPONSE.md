# CGL 审查响应 — 四项实现问题的修复记录（2026-10-20）

> 审查对象：导出仓库 [gnn-of-pso](https://github.com/Xiaoqin-pro/gnn-of-pso) `main` 分支提交 `082ce2f`
> （静态审查，未在本机重跑实验）。
> 本文件是在 `feat/cgl-pso` 分支上对四项问题的**修复前 / 修复后**记录，配套回归测试与重测证据。
>
> **旧结果不可覆盖**：`results/cgl/{pilot,dev10,repair1,clip}` 与 `results/cgl/dev10/verdict.json`
> 全部保持原样；本轮修复的产物写在 `results/cgl/sw_pretrial/`、`results/cgl/sw_main/`，
> 二者互不覆盖。

---

## 0. 一句话结论

| 项 | 审查等级 | 性质 | 状态 |
|---|---|---|---|
| ① `global_gain` 用 `max(pbest)` | P0 | **确凿实现错误** | 已修复 + 回归测试 |
| ② 停滞惩罚未加 `stalled` 掩码 | P0 | **确凿实现错误** | 已修复 + 回归测试 |
| ③ 最远邻居把 self 设为 `+inf` | P1 | **候选构造错误** | 已修复 + 回归测试 |
| ④ CGL 无条件禁止 self 候选 | P1 | 需消融验证的**设计选择** | 已加开关 + 选择组成对照 |

修复后按同一预算重测，**CGL 的逐函数结果几乎没有变化**（4 任务平均
log10(error+1e-8)：修复前 `1.5627` → 修复后 `1.5683`），仍然明显劣于 `ring`/`gbest`
且慢 6.56×。也就是说：**这四项修复纠正了实现语义，但没有把 CGL 变成一个有效机制**；
审查中"当前 CGL 结论受实现问题污染"的担忧可以部分消除——污染确实存在，但修完之后
结论方向不变。逐函数数据、预算账本与 Ring-SmallWorld 判定见
`docs/CGL_SMALLWORLD_RESULTS_zh.md`。

---

## 1. ①  P0 — 最小化问题的"全局最优改善"误用 `max(pbest)`

**位置**：`src/cgl_pso/policy.py::CGLPolicy._apply_feedback()`

```python
# 修复前（错误）
global_gain = float(np.max(self._pending.pbest_val) - np.max(state.pbest_val))
# 修复后（正确）
global_gain = float(np.min(self._pending.pbest_val) - np.min(state.pbest_val))
```

**为什么是错的**：PSO 在这里是最小化问题，`min(pbest)` 才是全局最优。
`max(pbest)` 衡量的是"历史最差粒子"的变化，与全局最优改善无关，并且符号还经常相反
（最优粒子改进时 `max` 常常不变或下降），因此该项作为贡献图反馈时提供的是**噪声甚至反向信号**。

**审查同时指出测试验证了错误公式**：`tests/test_cgl.py` 旧断言按 `max` 编写，
所以三项测试全绿却掩盖了 bug。该断言已按真实数学语义改写，并新增
`tests/test_cgl.py::test_global_best_gain_uses_min_not_max`：

* 构造一个只有**最差粒子**变化的场景，使 `min` 与 `max` 两种公式给出不同数值；
* 先断言两式确实不等（防止测试退化），再断言写入贡献图的分量等于 `0.3 * gain_min`。

**对既有结论的影响**：`global_gain` 参与"共用分量剔除"（`w_g` 项），
因此 `results/cgl/dev10` 里贡献图的学习信号**部分是错的**。修复后必须重测才可引用，
这一点已在结果文档中标注。

---

## 2. ②  P0 — 停滞惩罚未受 `stalled` 掩码约束

**位置**：`src/cgl_pso/policy.py::CGLPolicy._score_candidates()`

```python
# 修复前：任何粒子的"近期低收益边"都会被罚
bad_edge = unproductive[...] & recent[...]
penalty = np.where(bad_edge, self.refresh_penalty, 0.0)
# 修复后：M3 是"停滞触发"机制，只有停滞粒子才允许被罚
bad_edge = bad_edge & stalled[:, None]
```

**为什么是错的**：文档把 M3 描述为"停滞触发关系刷新"，但代码里探索概率上调受停滞门控、
**惩罚却不受门控**。结果是正在持续改善的粒子也会被压低它历史上用过的边，
这与机制描述不一致，也会污染 `cgl_norefresh` 这一消融的解释
（该消融本来应该只关掉"停滞触发"这一件事）。

**回归测试**：`tests/test_cgl.py::test_stagnation_penalty_only_touches_stalled_particles`
——在同一个 swarm 里人为构造一部分停滞、一部分未停滞的粒子，断言未停滞行的打分
与"无惩罚参照实现"逐元素一致，且停滞行确实被罚过（防止测试因条件不成立而空跑）。

---

## 3. ③  P1 — 最远邻居采样：self 被设为 `+inf` 后赢得 `argmax`

**位置**：`src/cgl_pso/candidates.py::build_candidates()`

```python
# 修复前：抽到自身时距离置 +inf → 自身反而成为"最远点"→ 又被自环过滤掉，
#         这一行白白丢掉了长距离候选槽
dist[rows, sample == rows] = np.inf
far = sample[np.arange(n), np.argmax(dist, axis=1)]
# 修复后：自身置 -inf，并提供"全是自身"时的回退
valid = sample != rows
dist = np.where(valid, dist, -np.inf)
fallback = (np.arange(n) + 1) % n
far = np.where(valid.any(axis=1), sample[np.arange(n), np.argmax(dist, axis=1)], fallback)
```

**回归测试**：`tests/test_cgl.py::test_farthest_candidate_is_truly_farthest_and_never_self`
——对随机小规模 swarm 逐粒子暴力枚举真实最远非自身候选，断言构建结果与之相等
（取样 40 次 / 6 粒子下必然命中真实最远点），并断言长距离槽永不指向自身。

---

## 4. ④  P1 — self-pbest 候选：从"隐式禁止"变成显式开关

审查指出这是一个**重要的机制差别而非 bug**：`ring` 的邻域包含自身
（邻域最优可能就是自己），而 CGL 在候选构建阶段把自身完全移除，
优秀粒子被迫向别人学习。现已实现为显式开关：

* `src/cgl_pso/candidates.py`：`build_candidates(..., include_self=False)`；新增候选源标签 `"self"`，
  并把它插到候选块**首位**，这样去重时标签不会被 ring/elite 块抢占；
* `src/cgl_pso/policy.py`：`CGLPolicy(..., allow_self_candidate=False)`，透传给候选构建器，并写入 `summary()`。

**选择组成对照**（`results/cgl/sw_pretrial/selection_diagnostics.csv`，
seed 0 × 4 任务 × 10,000 FEs，24 条记录；`cgl_full_self` 即开启该开关）：

| 设定 | 选到全局精英 | 选到环精英 | 选到自身 | 平均 pbest 排名（0=最好） | 末端多样性 |
|---|---|---|---|---|---|
| `cgl_full`（禁止 self） | 0.161 | 0.065 | 0.000 | 0.232 | 0.086 |
| `cgl_full_self`（允许 self） | 0.152 | 0.092 | 0.076 | 0.244 | 0.088 |
| `ring`（参照） | – | – | – | – | 0.085 |

**读法**：开启 self 后只有 7.6% 的选择落在自身，环精英份额从 6.5% 升到 9.2%，
多样性几乎不变（0.086 → 0.088）。所以**"禁止自环"并不是本轮性能差的主因**，
但差异现在是可消融、可复现的实验变量，而不是被写死的实现细节。

---

## 5. 修复后的重测：CGL 结论是否改变？

`results/cgl/sw_main/` 用**修复后的代码**重跑了 `cgl_full`（标签 `cgl_full_fixed`，
候选源保持默认全集），与 `ring`/`gbest` 同 10 个配对种子、30D、40 粒子、10,000 FEs：

| 任务 | ring（重跑） | gbest（重跑） | cgl_full（修复前, dev10） | cgl_full_fixed（修复后） |
|---|---|---|---|---|
| alpine1 | 0.842 | **0.523** | 1.195 | 1.211 |
| griewank | 0.070 | **-0.717** | 1.187 | 1.145 |
| rosenbrock | 1.990 | **1.673** | 2.145 | 2.173 |
| zakharov | 1.893 | **1.418** | 1.723 | 1.744 |
| **均值** | 1.198 | **0.724** | 1.5627 | 1.5683 |

数值为 mean log10(error + 1e-8)，越低越好。

**可复现性证据**：重跑得到的 `ring`/`gbest` 与 `dev10` 提交值在 4 任务 × 10 种子上
**逐位相同（maxdiff = 0.0）**，说明两次筛选的实例、初始化与 FEs 完全一致，
`cgl_full_fixed` 与 `cgl_full` 的差值可以干净地归因于代码修复本身。

**结论**：修完三项实现问题后，CGL 的相对与绝对表现都**没有实质变化**
（均值 +0.006，即略差），仍不优于经典基线，且运行时间中位 6.56×（阈值 3×）。
因此"停止 CGL 机制路线、保留负结果"的判定**维持不变**，
但它现在建立在一份公式正确的实现之上。

---

## 6. 回归测试与复现

```powershell
# 环境（本机实测：Python 3.13.7 / torch 2.7.1+cu118，CPU 路径）
.\.venv\Scripts\python.exe -m pytest tests -q            # 全量测试

# 本轮新增/改写的针对性测试
.\.venv\Scripts\python.exe -m pytest tests/test_cgl.py tests/test_smallworld.py -q

# 复现修复后的 CGL 重测（不覆盖旧结果）
.\.venv\Scripts\python.exe -m cgl_pso smallworld -c configs/cgl/smallworld.yaml --stage main

# 零 FEs 复核：逐位一致性 + 判定复算 + 预算账本
.\.venv\Scripts\python.exe scripts/cgl_sw_verify_repro.py
```

新增回归测试清单：

* `test_global_best_gain_uses_min_not_max`
* `test_stagnation_penalty_only_touches_stalled_particles`
* `test_farthest_candidate_is_truly_farthest_and_never_self`
* `test_self_candidate_is_optional`

---

## 7. 仍未解决 / 不声称的部分（诚实清单）

1. 这四项修复**不构成**"图学习选择机制有效"的证据；CGL 的负结果保留。
2. 贡献图的其余语义（`novelty`、`recency`、温度 `T=0.5`、`epsilon` 探索）
   都未逐项做敏感性验证，仅作为实现细节记录。
3. 本轮所有筛选均为 30D / 40 粒子 / 10,000 FEs 的**开发规模**，
   10 个配对种子仍不足以支撑论文级显著性声明。
4. CEC2017/CEC2022 与真实 CLIP/OpenCLIP 实验**未在本轮执行**
   （未获批准 + 核心筛选未过），`docs/CGL_CLIP_APPLICATION.md` 中的 0.6688
   仍然是**合成 logits smoke**，不是视觉语言任务的测量结果。
5. `cgl_full` 修复后仍被 far/random 候选源拖慢（选择开销约 0.50 ms/次，ring 的
   选择开销可忽略），这一开销结构性问题只被记录，未在本轮继续优化。
