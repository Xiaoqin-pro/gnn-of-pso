# CGL-PSO 文献矩阵与重合风险（CGL_LITERATURE）

> 核实日期 2026-10-20；核实渠道：Crossref REST API、OpenAlex API、PMLR 页面、GitHub API。
> **本文件不写"首次图学习 PSO""首次在线贡献评估"等未经支持的断言。**
> 每条文献给出：准确出处、核心机制、与 CGL-PSO 的重合风险、区别、证据级别。

## 1. 直接约束选题的 2025–2026 工作

| # | 出处（已核实） | 核心机制 | 与 CGL-PSO 的重合风险 | CGL 的区别 | 证据级别 |
|---|---|---|---|---|---|
| L1 | Dikshit Chauhan, Shivani, P. N. Suganthan. **Learning strategies for particle swarm optimizer: A critical review and performance analysis.** *Swarm and Evolutionary Computation*, 2025. DOI 10.1016/j.swevo.2025.102048（Crossref 核实，被引 46） | 系统梳理 PSO 的自适应/综合学习/多策略/混合学习 | **高**：说明"多学习机制组合"已有大量先例，必须解释组合为何有效，不能把组合本身当创新 | CGL 的组合是**同一机制链**（反馈→边权→刷新），不是并列堆叠多个算子；并提供逐机制消融 | 综述（高） |
| L2 | Weitao Zhao, Zati Hakim Azizul, Chaw Seng Woo, Weijie Kuang, Yafeng Li. **Potential-driven multi-learning particle swarm optimisation (PDML-PSO).** *Swarm Evol. Comput.*, 2025. DOI 10.1016/j.swevo.2025.101993（Crossref，被引 29） | 潜力粒子识别 + 分层学习 + 动态扰动 | **中高**：分层选择学习对象与 CGL 的"候选打分选源"关系接近 | CGL 不做多层分群，也不加扰动算子；区别在于**边的历史收益记忆**与**停滞驱动的边级刷新** | 期刊（高） |
| L3 | Shuying Zhang, Xiaohong Hu, Yue Gao, Minghan Gao, Yufei Zhang. **Exemplar Learning and Memory Retrieval-Based Particle Swarm Optimization Algorithm with Engineering Applications (EMPSO).** *Biomimetics*, 2025. DOI 10.3390/biomimetics10100708（Crossref，被引 6） | 精英示例 + 成功记忆 + 自适应位置更新 | **高**：成功记忆 + 精英引导与"贡献记忆 + 质量引导"高度重合 | CGL 的记忆落在**有向边**（i→j 的实际收益）而非粒子级成功池；刷新作用于**关系**而非位置 | 期刊（高） |
| L4 | Zhi-Tao Lai, Zi-Jia Wang, Shuai Liu, Zong-Gan Chen, Zhi-Hui Zhan, Sam Kwong. **Adaptive pattern learning particle swarm optimization for large-scale optimization (APLPSO).** *Swarm Evol. Comput.*, 2026. DOI 10.1016/j.swevo.2025.102268（Crossref，被引 3） | 粒子动态选择不同数量的学习示例 | **中**：示例数量自适应 ≠ 关系选择自适应 | CGL 固定 Top-k 候选内采样，不引入示例数量这一额外维度；创新点不在"多示例" | 期刊（高） |
| L5 | Baotong Wu, Zhixiang Chen, Emrah Demir. **Reinforcement learning-guided multi-objective particle swarm optimization with adaptive topology (RL-MOPSO).** *Computers & Operations Research*, 2026. DOI 10.1016/j.cor.2026.107624（Crossref，被引 1） | Q-learning 选择粒子通信拓扑 | **高**：动态拓扑选择已有直接先例 | CGL 不做 RL/Q 学习，也不改拓扑集合；只用**轻量折扣反馈**更新已有有向边的期望收益，且不声称因果 | 期刊（高） |
| L6 | Luis DaCosta, Álvaro Fialho, Marc Schoenauer, Michèle Sébag. **Adaptive operator selection with dynamic multi-armed bandits.** *GECCO 2008*. DOI 10.1145/1389095.1389272（OpenAlex 核实：GECCO 2008，被引 165） | 多臂赌博机驱动的自适应算子选择（信用分配 + 探索） | **概念性高**：奖励驱动的自适应选择不是新思想 | CGL 的"臂"是**粒子间有向边**（社会学习对象），不是整个算子；且仅做轻量折扣记忆，无 UCB/概率匹配理论主张 | 会议（经典，高） |

**结论（写作约束）**：可以主张的是**具体机制组合 + 逐机制消融 + 与经典基线的公平比较**；
不可主张"首次用反馈/图/自适应选择"。CGL 的差异点必须写成：
1) 反馈落在**有向社会学习边**上（i→j 的实测收益），而非粒子级成功记忆或全局算子；
2) 质量、历史收益、方向相容度与探索下界在**同一有界分数**内联合；
3) 停滞只用来**刷新关系**（降权无效旧边 + 提高未试边概率），不重启粒子、不加扰动算子。

## 2. 应用侧文献（视觉语言模型）

| # | 出处 | 核心机制 | 对本工作的约束 |
|---|---|---|---|
| A1 | James Urquhart Allingham, Jie Ren, Michael W. Dusenberry, Xiuye Gu, Yin Cui, Dustin Tran, Jeremiah Zhe Liu, Balaji Lakshminarayanan. **A Simple Zero-shot Prompt Weighting Technique to Improve Prompt Ensembling in Text-Image Models.** *ICML 2023*, PMLR 202 (https://proceedings.mlr.press/v202/allingham23a.html，页面核实) | 零样本提示词加权集成（Prompt Weighting） | **不能声称首次优化 CLIP 提示词权重**；本轮把它作为**可实现的对照方法之一**，若无法忠实复现则明确标注"近似/不可复现"，不当作结论依据 |
| A2 | mlfoundations/open_clip（GitHub API 核实：14,211 stars，最新 release v3.3.0，2026-02-27；仓库 license 显示 NOASSERTION） | 开源 CLIP 训练/推理实现与预训练权重 | 权重许可证需单独核对（代码与权重许可不同）；**下载前必须先检查网络与体积并向用户确认** |

**应用定位（写作约束）**：本工作是"**冻结 VLM + 验证集监督的后处理校准**"，
**不得**称为零样本方法；模板权重优化本身已有先例（A1）。

## 3. 与前三阶段（GLDI-PSO）的关系

* 前三阶段（GAT/PPO/MLP 选择邻居）的实验结论**仅作为失败经验与基础设施来源**：
  复用经过验证的 FEs 计数、随机性隔离、资源监控、结果记录与基准函数；
  **不延续**其神经网络与强化学习架构，也不移植任何外部 MATLAB 实现。
* 前三阶段已观察的 CEC2022 结果**不构成本轮盲测**（见 `docs/CGL_PROTOCOL.md`）。

## 4. 待补与不确定项

- [ ] L5（RL-MOPSO）的完整算法细节需阅读原文后才能做逐点对比（本轮只依据其摘要页信息作重合判断）。
- [ ] 多目标 PSO 与单目标 CGL 的评价协议不同，不能直接比较数值。
- [ ] A1 的官方实现是否可直接用于本机 CLIP 后端需在应用阶段验证；若不可用，退化为等权集成与随机搜索对照。
