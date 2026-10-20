# CGL-PSO 方法（公式、伪代码、创新边界）

> 版本：与 `configs/cgl/dev10.yaml` 中冻结的参数一致。所有符号与代码一一对应
> （`src/cgl_pso/{candidates,contribution,policy}.py`）。

## 1. 一句话定位

**把"粒子 i 向粒子 j 的 pbest 学习"建模为一条有向边，并用该边实际产生的搜索收益
在线修正这条边的期望价值**；质量、历史收益与方向相容度共同决定下一个社会学习对象，
停滞时优先刷新无效关系而不是重启粒子。

## 2. 标准 PSO 部分（未修改）

对粒子 $i$ 的第 $d$ 维：

$$v_{i,d}\leftarrow w\,v_{i,d}+c_1 r_1 (p_{i,d}-x_{i,d})+c_2 r_2 (p_{j(i),d}-x_{i,d})$$

$$x_{i,d}\leftarrow \mathrm{clip}\big(x_{i,d}+v_{i,d},\ [L_d,U_d]\big)$$

其中 $w=0.7298,\ c_1=c_2=1.49445$，$r_1,r_2\sim U(0,1)$ 逐维独立，
$j(i)$ 是**本方法选择的社会学习对象**（只有一个），惯性项/认知项/速度上限/边界处理
全部沿用经过验证的引擎（`gldi_pso.optim.engine.PSOEngine`），CGL 只替换 $j(i)$ 的来源。

## 3. M1 在线贡献图

有向边集合 $E=\{(i,j): i \text{ 曾向 } j \text{ 学习}\}$，每条边维护：

| 符号 | 含义 | 初值 |
|---|---|---|
| $n_{ij}$ | 实际被选用的次数 | 0 |
| $Q_{ij}$ | 折扣历史收益 | $0$（中性） |
| $\ell_{ij}$ | 最近一次被使用的迭代号 | $-1$ |
| $w_{ij}$ | 产生正收益的次数 | 0 |

**收益（只在被选中的边上写入）**：设第 $t$ 次迭代粒子 $i$ 使用了 $j$，则在 $t+1$
（此时 $x_i^{(t+1)}$ 的适应度已被真实评价）计算

$$\tilde{\Delta}f_i=\Delta f_i-\overline{\Delta f},\qquad \tilde{\Delta}p_i=\Delta p_i-\overline{\Delta p}$$

$$g_i=\mathrm{clip}\Big(\frac{0.6\,\tilde{\Delta}f_i+0.3\,\tilde{\Delta}p_i+0.1\,\Delta f^{\text{best}}}{s},-1,1\Big)$$

* $\overline{\Delta f}$ 是该迭代**全群的平均改善**：pilot 阶段诊断发现所有粒子的改善共享一个很大的共性分量，
  原始收益几乎等于"常数+噪声"，导致边间比较被噪声主导（实测：贡献项在候选集内 std≈0.59、
  决策翻转率 70–84%，但性能与无贡献版本无差异）。剔除共性分量后保留的是**相对可归因部分**；
  该修正记为 `gain_common_mode_removal: true`（可关闭，用于消融）。

* $s$ = 上一次迭代群体适应度的稳健尺度（IQR，退化为标准差，再退化为 1），
* $\Delta f^{\text{best}}$ = 群体最优的改善，
* **只使用已经评价过的量**：没有反事实分支、没有额外目标函数调用（有单测）。

更新规则（指数折扣）：

$$Q_{ij}\leftarrow (1-\rho)Q_{ij}+\rho\,g_i,\qquad n_{ij}\leftarrow n_{ij}+1,\qquad \ell_{ij}\leftarrow t$$

$\rho=0.7$。**诚实声明**：$Q_{ij}$ 是"向 $j$ 学习"与"随后取得收益"之间的**观测关联**，
不是严格因果贡献（可能存在混杂：同一时刻其他粒子也在改善）。论文中必须如此表述。

### 3.1 贡献项标准化（pilot 修正）

原始 $Q_{ij}$ 的量级约 $10^{-2}$，而质量项在 $[0,1]$——直接相加会把 M1 淹没。
因此分数里使用**粒子自身候选集内**的标准化：

$$\hat Q_{ij}=\tanh\!\Big(\frac{Q_{ij}-\overline{Q_{i\cdot}}}{\max(\sigma_{Q_{i\cdot}},\varepsilon)}\Big)$$

它表达的是"**在我的候选里，哪条边历史上更划算**"，这正是选择问题本身；
同时保持尺度无关，且不引入新的超参。

## 4. M2 质量与方向联合选择

**候选集** $C_i$（大小 $K$，去重、去自环）：环邻域（半径 1，左右各 1）+ pbest 前 $q=4$ +
近似最远点（对 8 个随机样本取最远，$O(mD)$ 而非 $O(ND)$ 全图）+ $n_r=2$ 个随机粒子。

**分数**（有界，逐项归一化）：

$$S_{ij}=\alpha\,\hat q_j+\beta\,\hat Q_{ij}+\gamma\,d_{ij}+\eta\,\nu_{ij}$$

* $\hat q_j=1-\mathrm{rank}(p_j)/(N-1)\in[0,1]$：pbest 质量排名（1 最好）；
* $\hat Q_{ij}=\tanh(Q_{ij})\in[-1,1]$：贡献记忆；
* $d_{ij}=\cos\!\big(p_j-x_i,\ p_i-x_i\big)\in[-1,1]$，**方向零向量或退化时取 0**；
* $\nu_{ij}=1/(1+n_{ij})$：未试边奖金（$\nu=1$），防止零试验边被永久饿死。

**动作分布**（Top-k 候选内 softmax + 探索下界）：

$$\pi_{ij}=(1-\epsilon_i)\frac{\exp(S_{ij}/T)}{\sum_{k\in C_i}\exp(S_{ik}/T)}+\epsilon_i\frac{1}{|C_i|}$$

$T=0.5$，$\epsilon_i=0.15$（常态）/ $0.4$（停滞态，见 M3）。

## 5. M3 停滞触发关系刷新

当 $stall_i\ge\tau$（连续 $\tau=12$ 次未改进 pbest）：

1. 对**近期使用过（$t-\ell_{ij}\le\tau$）且 $Q_{ij}\le 0$** 的边施加分数惩罚 $-1.0$；
2. 该粒子的探索下界提高到 $\epsilon_i=0.4$（提高未试/多样性候选的采样概率）；
3. **不重启粒子、不加混沌/Levy/对立学习**，个人历史最优 $p_i$ 完整保留；
4. 事件计数与停滞粒子比例写入日志，机制可整体关闭（消融）。

## 6. 伪代码

```
输入: N, D, w, c1, c2, 参数 (α,β,γ,η,T,ε,ε_stall,τ,ρ)
初始化 x, v, p=x, f=f(x);  Q←0, n←0, ℓ←-1
for t = 1 .. T_max:
    # --- M1 反馈写入（t-1 的实际收益） ---
    if 有 pending 选择:
        g = clip((0.6·Δf + 0.3·Δpbest + 0.1·Δf_best)/s, -1, 1)
        Q[pending] ← (1-ρ)·Q[pending] + ρ·g;  n[pending] += 1;  ℓ[pending] = t-1
    # --- M2 候选与打分 ---
    C_i ← build_candidates(x, p, ring=1, top_q=4, random=2, far_samples=8)   # 去自环/去重
    S_ij ← α·q̂_j + β·tanh(Q_ij) + γ·cos(p_j-x_i, p_i-x_i) + η/(1+n_ij)
    ε_i  ← ε_stall if stall_i ≥ τ else ε                    # --- M3 ---
    S_ij ← S_ij - 1.0·[上述 M3 条件]                        # --- M3 ---
    j(i) ~ (1-ε_i)·softmax(S_i/T) + ε_i·uniform(C_i)
    # --- 标准 PSO 更新（引擎负责） ---
    v_i ← clip(w·v_i + c1·r1⊙(p_i-x_i) + c2·r2⊙(p_j(i)-x_i), ±v_max)
    x_i ← boundary(x_i + v_i);  f(x_i) 由引擎评价（每迭代 N 次，唯一 FEs 来源）
    p_i ← argmin(f(x_i), f(p_i));  stall_i ← 0 if 改进 else stall_i+1
    note_selection(i→j(i), t)                              # 供下一轮反馈
```

## 7. 复杂度

| 步骤 | 复杂度 | 说明 |
|---|---|---|
| 候选构造 | $O(N\cdot K + N\cdot m\cdot D)$ | $m=8$ 个随机样本算"近似最远"，**避免** $O(N^2D)$ 全距离矩阵 |
| 打分 | $O(N\cdot K\cdot D)$ | 方向相容度是主项；$K\le 9$ |
| 图反馈 | $O(N)$ | 仅写被选中的 $N$ 条边 |
| PSO 更新 | $O(N\cdot D)$ | 与标准 PSO 相同 |
| 额外内存 | $O(N^2)$ 稀疏统计（$N=40\Rightarrow$ 1600 个 float/int） | 可用稀疏字典，但 $N\le 200$ 时稠密更简单 |

实测选择开销（$N=20,D=10$）：**0.39–0.59 ms/次迭代**（见 `results/cgl/smoke/summary.json`）。

## 8. 消融矩阵（每个变体只改一个因素）

| 变体 | M1 贡献 | M2 方向 | M3 刷新 | 用途 |
|---|---|---|---|---|
| `quality_rank` | 计算但不使用（β=0） | 关 | 关 | 同候选集上的纯质量排名基线 |
| `cgl_nocontrib` | 关（β=0） | 开 | 开 | **主消融**：贡献记忆是否有净收益 |
| `cgl_nodirection` | 开 | 关（γ=0） | 开 | 方向项是否有净收益 |
| `cgl_norefresh` | 开 | 开 | 关 | 停滞刷新是否有净收益 |
| `cgl_full` | 开 | 开 | 开 | 完整算法 |

对照基线：`ring`、`gbest`（经典）与 `gldi_rule`（历史参考，**仅作参照，不移植其逻辑**）。

## 9. 创新边界（写作时必须遵守）

* 可主张：机制**组合**的具体形式、逐机制消融、与经典/无贡献版本在统一预算下的公平比较、
  以及（若成立）**关系刷新带来的净收益**。
* 不可主张：首次图学习 PSO、首次在线贡献评估、首次奖励驱动自适应选择
  （多臂赌博机自适应算子选择早于本工作，见 `docs/CGL_LITERATURE.md` L6）；
  不可把 $Q_{ij}$ 说成严格因果贡献；不可把"粒子自己变好"当成"群体进展"
  （前三阶段已证明两者可负相关）。
