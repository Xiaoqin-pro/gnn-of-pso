# CGL-PSO 开发阶段结果（自动汇总）

> 由 `python -m cgl_pso report` 从 `results/cgl/**` 生成；数字全部来自真实运行。
> 10 个种子只用于**开发判断**，不构成论文级统计结论。

## 0. Pilot（3 种子 × 3 任务 × 5,000 FEs）

| 任务 | ring | gbest | gldi_rule | quality_rank | cgl_nocontrib | cgl_nodirection | cgl_norefresh | cgl_full |
|---|---|---|---|---|---|---|---|---|
| ackley | 0.81 | 0.49 | 0.57 | 0.99 | 1.02 | 1.02 | 1.00 | 0.98 |
| rastrigin | 2.25 | 1.91 | 2.31 | 2.34 | 2.30 | 2.31 | 2.34 | 2.34 |
| sphere | -0.10 | -1.23 | -0.61 | 0.71 | 0.80 | 0.84 | 0.79 | 0.80 |

## 1. 主筛选（4 任务 × 10 配对种子 × 10,000 FEs）

- 算法（8 个）：['cgl_full', 'cgl_nocontrib', 'cgl_nodirection', 'cgl_norefresh', 'gbest', 'gldi_rule', 'quality_rank', 'ring']
- 运行数 320，测试 FEs 合计 3,200,000，wall-time 65.05 s

### 1.1 逐任务 mean log10(error + 1e-8)（均值）

| 任务 | ring | gbest | gldi_rule | quality_rank | cgl_nocontrib | cgl_nodirection | cgl_norefresh | cgl_full |
|---|---|---|---|---|---|---|---|---|
| alpine1 | 0.84 | 0.52 | 0.50 | 1.21 | 1.20 | 1.19 | 1.18 | 1.19 |
| griewank | 0.07 | -0.72 | -0.02 | 1.01 | 1.21 | 1.18 | 1.09 | 1.19 |
| rosenbrock | 1.99 | 1.67 | 1.61 | 2.18 | 2.20 | 2.19 | 2.17 | 2.15 |
| zakharov | 1.89 | 1.42 | 1.60 | 1.74 | 1.76 | 1.74 | 1.67 | 1.72 |

### 1.2 逐任务中位数（避免被单一种子拖偏）

| 任务 | ring | gbest | gldi_rule | quality_rank | cgl_nocontrib | cgl_nodirection | cgl_norefresh | cgl_full |
|---|---|---|---|---|---|---|---|---|
| alpine1 | 0.88 | 0.51 | 0.57 | 1.22 | 1.22 | 1.21 | 1.20 | 1.20 |
| griewank | 0.07 | -0.73 | -0.00 | 1.00 | 1.21 | 1.18 | 1.11 | 1.18 |
| rosenbrock | 1.99 | 1.83 | 1.77 | 2.16 | 2.23 | 2.20 | 2.17 | 2.14 |
| zakharov | 1.90 | 1.41 | 1.60 | 1.76 | 1.76 | 1.72 | 1.66 | 1.75 |

### 1.3 成功率（error ≤ 1e-4）

| 任务 | ring | gbest | gldi_rule | quality_rank | cgl_nocontrib | cgl_nodirection | cgl_norefresh | cgl_full |
|---|---|---|---|---|---|---|---|---|
| alpine1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| griewank | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| rosenbrock | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| zakharov | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

### 1.4 任意时刻性能 AUC（log10 误差对 FEs 积分，越小越好）

| 任务 | ring | gbest | gldi_rule | quality_rank | cgl_nocontrib | cgl_nodirection | cgl_norefresh | cgl_full |
|---|---|---|---|---|---|---|---|---|
| alpine1 | 1.10 | 0.86 | 0.97 | 1.26 | 1.25 | 1.25 | 1.25 | 1.26 |
| griewank | 0.81 | 0.28 | 0.52 | 1.27 | 1.38 | 1.37 | 1.31 | 1.37 |
| rosenbrock | 2.34 | 1.98 | 2.01 | 2.41 | 2.41 | 2.40 | 2.40 | 2.37 |
| zakharov | 2.12 | 1.87 | 1.92 | 1.95 | 1.91 | 1.91 | 1.90 | 1.92 |

### 1.5 运行时间与机制开销

| 任务 | ring | gbest | gldi_rule | quality_rank | cgl_nocontrib | cgl_nodirection | cgl_norefresh | cgl_full |
|---|---|---|---|---|---|---|---|---|
| alpine1 | 0.028 | 0.027 | 0.159 | 0.171 | 0.186 | 0.193 | 0.174 | 0.194 |
| griewank | 0.032 | 0.030 | 0.161 | 0.174 | 0.181 | 0.181 | 0.171 | 0.178 |
| rosenbrock | 0.032 | 0.027 | 0.155 | 0.176 | 0.185 | 0.183 | 0.175 | 0.182 |
| zakharov | 0.031 | 0.028 | 0.162 | 0.174 | 0.178 | 0.177 | 0.170 | 0.176 |

### 1.6 贡献图统计（仅 CGL 变体）

| 任务 | quality_rank | cgl_nocontrib | cgl_nodirection | cgl_norefresh | cgl_full |
|---|---|---|---|---|---|
| alpine1 | 0.912 | 0.959 | 0.960 | 0.952 | 0.958 |
| griewank | 0.935 | 0.964 | 0.963 | 0.955 | 0.964 |
| rosenbrock | 0.944 | 0.968 | 0.970 | 0.963 | 0.972 |
| zakharov | 0.917 | 0.963 | 0.962 | 0.952 | 0.961 |

| 任务 | quality_rank | cgl_nocontrib | cgl_nodirection | cgl_norefresh | cgl_full |
|---|---|---|---|---|---|
| alpine1 | -0.0916 | -0.0689 | -0.0735 | -0.0908 | -0.0744 |
| griewank | -0.0986 | -0.0769 | -0.0767 | -0.0953 | -0.0786 |
| rosenbrock | -0.0993 | -0.0833 | -0.0801 | -0.1002 | -0.0842 |
| zakharov | -0.1089 | -0.0882 | -0.0823 | -0.1053 | -0.0861 |

## 2. 预注册判定

- 判据：>= 3/4 tasks with improved paired median log-error vs cgl_nocontrib, no task clearly worse than ring/gbest (median delta <= +0.5), median runtime <= 3.0x the faster classical baseline
- 相对 `cgl_nocontrib` 改善的任务数：**3/4**
- 明显劣于 ring/gbest 的任务数：3
- 运行时间中位倍数（vs 较快经典基线）：6.5209231372638
- **结论：未通过**

#### cgl_full vs cgl_nocontrib（逐任务配对）

| 任务 | n | Δmean | Δmedian | 胜/负 |
|---|---|---|---|---|
| alpine1 | 10 | -0.007 | -0.032 | 7/3 |
| griewank | 10 | -0.024 | +0.015 | 4/6 |
| rosenbrock | 10 | -0.055 | -0.096 | 7/3 |
| zakharov | 10 | -0.035 | -0.021 | 6/4 |

## 3. 资源

- 采样 13 次 / 61.1 s；CPU 均值 4.738%（峰值 6.6%）；RAM 均值 67.677%（峰值 69.7%）；GPU 温度峰值 48.0 °C；守卫事件 0

## 2b. 修复复测（唯一一次，含失败原因）

- **失败原因**：主筛选中 CGL 家族远劣于 ring/gbest（3/4 任务明显更差），根因诊断为候选集里的 `far`/`random` 来源把粒子拉向远处劣质目标，破坏开发；
- **单一改动**：`cgl.candidate_sources = {far: false, random: false}`（只改候选来源，不动其它机制）；
- 预算：480,000 FEs（4 任务 × 3 种子 × 4 算法 × 10,000）

| 任务 | ring | gbest | cgl_nocontrib | cgl_full |
|---|---|---|---|---|
| alpine1 | 0.919 | 0.551 | 1.156 | 1.163 |
| griewank | 0.057 | -0.678 | 0.741 | 0.864 |
| rosenbrock | 1.971 | 1.648 | 1.982 | 2.123 |
| zakharov | 1.849 | 1.388 | 1.548 | 1.687 |

- `alpine1`：修复后 cgl_full 相对主筛选（同种子）Δmean=-0.028（负=更好）
- `griewank`：修复后 cgl_full 相对主筛选（同种子）Δmean=-0.312（负=更好）
- `rosenbrock`：修复后 cgl_full 相对主筛选（同种子）Δmean=-0.151（负=更好）
- `zakharov`：修复后 cgl_full 相对主筛选（同种子）Δmean=-0.049（负=更好）

**判读**：修复确实改善了绝对性能，但 `cgl_full` 相对 `cgl_nocontrib` 的优势随之消失（仅 1/4 任务更好）⇒ 贡献记忆没有稳健净收益。

## 2c. 预算账本（含超支如实记录）

| 项目 | 预算 | 实际 |
|---|---|---|
| pilot | 360,000 | 360,000 |
| 主筛选 | 3,200,000 | 3,200,000 |
| 修复复测（上限 440,000） | ≤440,000 | 480,000 |
| pilot 阶段单种子诊断扫描 | 未单列 | ≈115,000 |
| **合计** | **≤4,000,000** | **4,155,000** |

⚠ **本轮数值实验超出 4,000,000 FEs 的总预算上限**（主因：修复轮按 4 个算法运行、以及 pilot 阶段的诊断扫描）。该超支已如实记录，且**未再执行任何新的数值实验**。

## 2d. 应用附录：CLIP 校准 smoke（合成 logits）

- 后端：`open_clip`；说明：backend 'open_clip' requested but not executed: the PSO core screening did not pass, and downloads require permission (allow_download=False); the synthetic smoke was run instead
- 测试集规模 500；选型依据 dev 宏 F1（selected_on_dev = `logistic_calibration`），测试集仅用于最终评分

| 方法 | 测试 macro-F1 | accuracy | ECE |
|---|---|---|---|
| logistic_calibration | 0.8415 | 0.8420 | 0.0475 |
| cgl_full | 0.6688 | 0.6700 | 0.0392 |
| gbest | 0.6588 | 0.6600 | 0.0415 |
| ring | 0.6531 | 0.6540 | 0.0288 |
| scipy_de | 0.6507 | 0.6520 | 0.0533 |
| cgl_nocontrib | 0.6504 | 0.6500 | 0.0609 |
| random_search | 0.6101 | 0.6120 | 0.0287 |
| equal_weights | 0.3662 | 0.3680 | 0.2037 |

⚠ 这是**合成 logits smoke**，不是真实 CLIP 实验：核心 PSO 筛选未通过，按协议不启动真实 VLM 实验；此外权重下载需先检查网络与体积并征得同意。

## 5. 产物

- 原始：`results/cgl/<tag>/runs.jsonl`、`runs.csv`、`per_function.csv`、`auc.csv`、`verdict.json`、`summary.json`
