# Ring-SmallWorld-PSO — 机制筛选结果（中文）

> 本文记录审查（2026-10-20）要求的"轻量 Ring-SmallWorld-PSO 原型"这一轮**唯一一次**
> 机制筛选的真实结果：预试验（3 配对种子）→ 选择机制诊断 → 主筛选（10 配对种子）。
> 所有数字均来自实际执行，原始记录在 `results/cgl/{sw_pretrial,sw_main}/runs.jsonl`。
> **没有预设结论**：如果机制无效，就保留负结果。

---

## 1. 方法与协议（先声明，再看结果）

**基础拓扑**：经过验证的 `ring` 环状邻域（半径 1，邻域含自身，与 `cgl_pso._infra.optim.policies`
的 ring-lbest 完全一致，有等价性单测）。

**唯一新增自由度**：允许粒子以概率 `p` 把自己的社会吸引子从"环内最优"
替换为"**非局部精英**"（全局 pbest 前 `q=5` 名中、位于自身环邻域**之外**的粒子）。
速度/位置更新、边界策略、vmax、初始化、FEs 计数**全部复用同一个引擎**，不改任何搜索算子。

**四种模式**（`src/cgl_pso/smallworld.py`）：

| 模式 | 语义 |
|---|---|
| `sw_ring` | 恒等于 ring（控制组，用于等价性验证） |
| `sw_fixed` | 常数 `p` 的跨邻域连接 |
| `sw_stall` | 只有 `stall >= τ` 的粒子才允许连接（停滞触发） |
| `sw_bandit` | 用上一代"非局部用户 vs 本地用户"的改善差在线微调 `p`（随机逼近，无模型、无 PPO、无 GNN） |

**因果性与零额外 FEs**：上一代的非局部尝试用**下一代已评价的 pbest** 打分
（与 CGL 相同的因果模式），因此反馈不消耗任何目标函数评价。

**预算账本（预注册上限 3.8M FEs，本轮只跑一轮）**：

| 阶段 | 设定数 | 种子 | 任务 | FEs |
|---|---|---|---|---|
| 预试验 `sw_pretrial` | 8 | 3 | 4 | 960,000 |
| 选择机制诊断 `selection_diagnostics` | 6 | 1 | 4 | 240,000 |
| 主筛选 `sw_main` | 6 | 10 | 4 | 2,400,000 |
| **合计** | | | | **3,600,000 ≤ 3,800,000** |

统一设置：30 维、40 粒子、10,000 FEs/run（249 次迭代）、`ω=0.7298`、`c1=c2=1.49445`、
`vmax_frac=0.2`、边界 `clip`、均匀初始化；同一 `(task, seed)` 的实例、初始种群与预算在
所有算法间**完全一致**（`ring`/`gbest` 与先前 `dev10` 提交值逐位相同，见 §6）。

---

## 2. 预试验：8 个设定的粗筛（3 配对种子 × 4 任务）

mean log10(error + 1e-8)，越低越好：

| 任务 | ring | fixed p=.05 | fixed p=.15 | stall τ10 p.05 | stall τ10 p.15 | stall τ20 p.05 | stall τ20 p.15 | bandit p.05 |
|---|---|---|---|---|---|---|---|---|
| alpine1 | 0.919 | 0.684 | **0.173** | 0.958 | 1.015 | 0.963 | 0.949 | 1.016 |
| griewank | 0.057 | 0.022 | 0.050 | 0.086 | 0.049 | 0.053 | 0.061 | 0.515 |
| rosenbrock | 1.971 | 1.942 | 1.895 | 1.976 | 2.017 | 1.993 | 1.990 | 2.162 |
| zakharov | 1.849 | 1.659 | **1.492** | 1.656 | 1.697 | 1.784 | 1.719 | 1.723 |
| **均值** | 1.199 | 1.077 | **0.902** | 1.169 | 1.195 | 1.198 | 1.180 | 1.354 |

中位运行时间（s/run）：`ring` 0.031 < `stall τ10 p.05` 0.050 < `stall τ20 p.05` 0.045
< `stall τ20 p.15` 0.056 < `stall τ10 p.15` 0.062 < `fixed p.05` 0.064
< `fixed p.15` 0.067 < `bandit` 0.072。

**预试验判定**（`results/cgl/sw_pretrial/smallworld_verdict.json`，
判据：≥1 个任务配对中位改善、无任务劣化 > +0.5、运行时间 ≤1.5× ring）：

| 候选 | 改善任务数 | 崩塌任务数 | 运行时间倍数 | 通过 |
|---|---|---|---|---|
| `sw_fixed_p05` | 4/4 | 0 | 2.10× | ✗（太慢） |
| `sw_fixed_p15` | 3/4 | 0 | 2.22× | ✗（太慢） |
| `sw_stall_t10_p05` | 3/4 | 0 | 1.52× | ✗（差 0.02 倍）|
| `sw_stall_t10_p15` | 2/4 | 0 | 1.94× | ✗ |
| `sw_stall_t20_p05` | 0/4 | 0 | 1.40× | ✗ |
| `sw_stall_t20_p15` | 2/4 | 0 | 1.77× | ✗ |
| `sw_bandit_p05` | 1/4 | 0 | 2.32× | ✗ |

→ 预试验无人通过；按预注册规则选出"最好的固定版"（`sw_fixed_p15`）、
"最好的停滞版"（`sw_stall_t10_p05`）与固定设定的 bandit 一起进入主筛选。
**没有**因为预试验结果而新增/调整任何参数（防止用 3 种子的噪声挑参数）。

---

## 3. 选择机制诊断：机制到底改变了什么？

`results/cgl/sw_pretrial/selection_diagnostics.csv`（seed 0 × 4 任务 × 10,000 FEs = 240,000 FEs）：

| 设定 | 选到全局精英 | 选到环精英 | 选到自身 | 平均排名(0最好) | 非局部份额 | 停滞份额 | 选择耗时 ms | 末端多样性 |
|---|---|---|---|---|---|---|---|---|
| `ring` | – | – | – | – | 0 | – | ~0 | 0.085 |
| `sw_fixed_p15` | – | – | – | – | 0.149 | 0.487 | 0.085 | **0.034** |
| `sw_stall_t10_p05` | – | – | – | – | 0.016 | 0.315 | 0.054 | 0.070 |
| `sw_bandit_p05` | – | – | – | – | 0.271 | 0.580 | 0.083 | 0.058 |
| `cgl_full` | 0.161 | 0.065 | 0.000 | 0.232 | – | 0.709 | 0.507¹ | 0.086 |
| `cgl_full_self` | 0.152 | 0.092 | 0.076 | 0.244 | – | 0.697 | 0.507¹ | 0.088 |

¹ CGL 的耗时取自 `results/cgl/sw_main/per_function.csv` 的 `mean_select_ms`（同实现）；
本 CSV 中该列对 CGL 为空，属记录字段差异，不影响结论。

**两个关键读数**：

1. **固定高概率连接会牺牲多样性**：`p=0.15` 让 15% 的更新指向全局前 5，
   末端多样性从 0.085 掉到 0.034（近腰斩），这也是它在 alpine1 上很亮、
   但在 rosenbrock/griewank 上没有优势的原因；而停滞触发版只在真正停滞时才连接，
   非局部份额 0.016，多样性保持 0.070。
2. **CGL 与 ring 的差别不是"选不选自己"**：开启 self 候选后只有 7.6% 的选择落在自身，
   选择组成与多样性几乎不变（§见 `docs/CGL_REVIEW_RESPONSE.md` 第 4 节），
   它更多是**把 15%–27% 的更新指向远离邻域的随机/远距离目标**（该差异在 CGL 侧更极端）。

---

## 4. 主筛选：10 配对种子（4 任务 × 6 设定 = 2,400,000 FEs）

### 4.1 mean log10(error + 1e-8)（越低越好）

| 任务 | ring | gbest | `sw_fixed_p15` | `sw_stall_t10_p05` | `sw_bandit_p05` | `cgl_full_fixed` |
|---|---|---|---|---|---|---|
| alpine1 | 0.842 | **0.523** | 0.642 | 0.749 | 0.893 | 1.211 |
| griewank | 0.070 | **-0.717** | 0.048 | 0.075 | 0.543 | 1.145 |
| rosenbrock | 1.990 | **1.673** | 1.878 | 2.005 | 2.004 | 2.173 |
| zakharov | 1.893 | **1.418** | 1.571 | 1.753 | 1.671 | 1.744 |
| **均值** | 1.199 | **0.724** | 1.035 | 1.146 | 1.278 | 1.568 |

### 4.2 中位 log10（稳健口径）

| 任务 | ring | gbest | `sw_fixed_p15` | `sw_stall_t10_p05` | `sw_bandit_p05` | `cgl_full_fixed` |
|---|---|---|---|---|---|---|
| alpine1 | 0.875 | **0.509** | 0.813 | 0.796 | 0.874 | 1.218 |
| griewank | 0.066 | **-0.734** | 0.049 | 0.073 | 0.511 | 1.153 |
| rosenbrock | 1.985 | **1.833** | 1.927 | 1.984 | 1.989 | 2.183 |
| zakharov | 1.895 | **1.408** | 1.589 | 1.759 | 1.667 | 1.756 |

### 4.3 收敛曲线 AUC（`results/cgl/sw_main/auc.csv`，20 个曲线点，越低越好）

| 任务 | ring | gbest | `sw_fixed_p15` | `sw_stall_t10_p05` | `sw_bandit_p05` | `cgl_full_fixed` |
|---|---|---|---|---|---|---|
| alpine1 | 1.099 | **0.861** | 1.044 | 1.052 | 1.115 | 1.258 |
| griewank | 0.808 | **0.282** | 0.666 | 0.813 | 0.991 | 1.336 |
| rosenbrock | 2.344 | **1.979** | 2.234 | 2.349 | 2.349 | 2.381 |
| zakharov | 2.122 | **1.873** | **1.871** | 2.051 | 1.968 | 1.912 |

### 4.4 成功率

在 10,000 FEs / 30D 下，**所有算法在所有任务上的成功率（err ≤ 1e-4）均为 0**，
更严格的 1e-8 同样为 0。该预算下成功率没有区分度，本文档不把它当作证据，
只如实记录（`results/cgl/sw_main/runs.csv` 的 `success_loose/success_tight`）。

### 4.5 运行时间与选择开销（s/run，同一进程、CPU 路径）

| 任务 | ring | gbest | `sw_fixed_p15` | `sw_stall_t10_p05` | `sw_bandit_p05` | `cgl_full_fixed` |
|---|---|---|---|---|---|---|
| alpine1 | 0.028 | 0.025 | 0.048 | 0.042 | 0.052 | 0.185 |
| griewank | 0.029 | 0.028 | 0.048 | 0.037 | 0.051 | 0.190 |
| rosenbrock | 0.028 | 0.027 | 0.047 | 0.038 | 0.051 | 0.184 |
| zakharov | 0.029 | 0.027 | 0.049 | 0.047 | 0.048 | 0.187 |
| **中位** | 0.0277 | 0.0260 | 0.0476 | 0.0406 | 0.0495 | 0.1820 |
| **相对 ring** | 1.00× | 0.94× | **1.72×** | **1.44×** | 1.78× | **6.56×** |

单次选择耗时（`mean_select_ms`）：`sw_*` 0.040–0.094 ms，
`cgl_full_fixed` 0.50 ms，`ring` 可忽略。small-world 的选择开销**不是**瓶颈，
CGL 的 0.5 ms/次才把它推到了 6.56×。

### 4.6 非局部连接与在线概率的实际走向

| 任务 | `sw_fixed_p15` 实测非局部份额 | `sw_stall_t10_p05` 实测份额 | `sw_bandit_p05` 实测份额 | bandit 末端 `p` |
|---|---|---|---|---|
| alpine1 | 0.149 | 0.020 | 0.291 | 0.323 |
| griewank | 0.150 | 0.005 | 0.407 | 0.453 |
| rosenbrock | 0.148 | 0.009 | 0.387 | 0.444 |
| zakharov | 0.149 | 0.030 | 0.067 | 0.092 |

固定版的经验份额与名义 `p` 一致（0.15 → 0.148–0.150），
说明实现没有偷偷多/少用连接；bandit 在 griewank/rosenbrock 上把 `p` 推到 0.45
（上限 0.5），恰恰在它最差的任务上（griewank 0.543 vs ring 0.070），
说明它的相对信号在这些任务上被误导。

---

## 5. 判定（`results/cgl/sw_main/smallworld_verdict.json`）

**判据（预注册）**：相对 ring 至少 1 个任务配对中位 Δlog10 < 0；无任务劣化 > +0.5；
运行时间中位 ≤ 1.5×；若固定概率版已经同样好，则不保留在线模块。

| 候选 | 改善任务数 | 崩塌任务数 | 运行时间倍数 | 通过 |
|---|---|---|---|---|
| `sw_fixed_p15` | 4/4 | 0 | 1.72× | ✗（超时阈值） |
| `sw_stall_t10_p05` | 2/4 | 0 | **1.44×** | ✅ **唯一通过** |
| `sw_bandit_p05` | 1/4 | 0 | 1.78× | ✗ |
| `cgl_full_fixed` | 1/4 | 1 | 6.56× | ✗ |

* `online_vs_fixed.keep_online = false`（固定版 1.035 < 在线版 1.278）→ **在线 bandit 模块应删除**；
* `recommended = sw_stall_t10_p05`；
* `any_candidate_passed = true`，但**注意这只是一个 1.5× 运行时间门槛下的"低开销可行"判定**，
  不是"显著优于 ring"。

### 逐任务配对证据（中位 Δlog10，负 = 更好）

| 任务 | `sw_fixed_p15` | `sw_stall_t10_p05` | `sw_bandit_p05` | `cgl_full_fixed` |
|---|---|---|---|---|
| alpine1 | −0.036 (6W/4L) | −0.086 (8W/2L) | +0.015 (5W/5L) | +0.316 (0W/10L) |
| griewank | −0.026 (8W/2L) | +0.005 (4W/6L) | +0.451 (0W/10L) | +1.088 (0W/10L) |
| rosenbrock | −0.065 (6W/4L) | +0.018 (4W/6L) | +0.015 (5W/5L) | +0.246 (3W/7L) |
| zakharov | −0.292 (10W/0L) | −0.125 (10W/0L) | −0.249 (10W/0L) | −0.153 (10W/0L) |

**读法**：所有 small-world 变体的收益几乎全部来自 zakharov（唯一 10/0 全胜的任务），
griewank/rosenbrock 上最好也只是打平；固定连接版的收益比停滞版更普遍，
但代价是多样性（0.034）和 1.72× 运行时间。因此本轮结论是
**"低开销地接入少量长程连接不损伤性能"**，而**不是**"小世界拓扑显著优于 ring"。

---

## 6. 可复现性与资源

* **逐位复现**：主筛选重跑的 `ring`/`gbest` 与 `results/cgl/dev10` 提交值在
  4 任务 × 10 种子上 `maxdiff = 0.0`（脚本 `scripts/cgl_sw_verify_repro.py`，零额外 FEs），
  证明两次筛选使用同一实例、同一初始化与同一 FEs 计数。
* **资源日志**：`results/logs/resources_cgl_sw_main.{csv,jsonl}`；
  主筛选 30.4 s 墙钟，采样 RAM 均值 76.9%（峰值 77.9%，硬上限 95%）、
  进程 RSS 0.40 GB、CPU 均值 3.6%、GPU 温度 47–48 °C、`guard_events = 0`。
  本轮为 CPU 路径（40 粒子的微图在 CPU 上更快），GPU 仅做温度监控。
* **无未授权实验**：未扩展 30 种子、未跑 CEC2017/CEC2022 全套、未下载/运行真实 CLIP。

```powershell
# 预试验（3 配对种子，960k FEs）
.\.venv\Scripts\python.exe -m cgl_pso smallworld -c configs/cgl/smallworld.yaml --stage pre --diagnostics
# 主筛选（10 配对种子，2.4M FEs；会自动沿用预试验选出的最佳 fixed/stall 设定）
.\.venv\Scripts\python.exe -m cgl_pso smallworld -c configs/cgl/smallworld.yaml --stage main
```

产物：`results/cgl/sw_pretrial/{runs.jsonl,runs.csv,per_function.csv,auc.csv,summary.json,smallworld_verdict.json,selection_diagnostics.csv,figures/}`、
`results/cgl/sw_main/{同上, smallworld_verdict.json}`。

零 FEs 的复核脚本（复现逐位一致性、重算判定、核对预算账本）：

```powershell
.\.venv\Scripts\python.exe scripts/cgl_sw_verify_repro.py
```

期望输出末行为 `PASS  bitwise reproducibility`、`PASS  verdict reproduction`、`PASS  budget ledger`。

从 `runs.jsonl` 重新导出 `summary.json` / `smallworld_verdict.json`（同样零 FEs，用于在判定辅助函数修复后刷新旧产物）：

```powershell
.\.venv\Scripts\python.exe scripts/cgl_sw_resummarise.py
```

> 注意：`results/cgl/sw_{pretrial,main}/verdict.json` 是**旧 CGL 门槛**的判定文件；这两个目录里不含 `cgl_nocontrib`，因此该文件已显式标注 `"applicable": false`，本轮的判定应以 `smallworld_verdict.json` 为准。

---

## 7. 局限（必须随结果一起引用）

1. 10 个配对种子是**开发规模**，不是论文级显著性；逐任务 W/L 计数已列出，
   但未做多重比较校正的显著性检验。
2. 任务集只有 4 个合成函数（alpine1/griewank/rosenbrock/zakharov，旋转变换 + 平移 0.25），
   **不是** CEC 基准；结论不能外推为"CEC 上有效"。
3. `stall τ=10, p=0.05` 只是粗筛胜出者，没有做敏感性扫描；
   预试验中 `stall τ20 p.05` 在 0/4 任务改善，说明该机制**对 τ 敏感**。
4. 运行时间倍数依赖当前实现与 CPU；`sw_fixed_p15` 的 1.72× 主要来自向量化的
   top-q 抽取与额外索引，不是算法复杂度问题，未进一步优化。
5. 本机制**不含图学习、不含 PPO**：它是一个"轻量拓扑调度"基线，
   用来判断"跨邻域长程连接到底有没有价值"，而不是对原 GNN 研究问题的回答。
