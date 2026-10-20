# CLIP 应用附录（CLIP_APPLICATION）

> 状态：**真实 CLIP 实验未执行**；本文档交付的是完整的**合成 logits smoke**
> 以及若核心筛选通过时应执行的真实实验协议。理由见第 1 节。

## 1. 为什么没有执行真实 CLIP 实验

1. **前置门槛未通过**：`docs/CGL_PROTOCOL.md` 第 6 节规定应用实验在核心 PSO 筛选**通过之后**开始。
   主筛选与修复复测均未通过（见 `docs/CGL_DECISION.md`），因此按协议不启动真实 VLM 实验。
2. **下载需先征得同意**：`open_clip` 未安装；OpenCLIP ViT-B/32 权重（约 600 MB）与 CIFAR-10/100
   需联网下载，目标文件明确要求"下载前先征求许可/检查网络"。
3. **不伪造结果**：因此本轮**不报告任何真实 CLIP 指标**；合成 smoke 的结果单独标注为合成。

本机已具备的替代条件（供用户决定是否继续）：
* `clip`（OpenAI CLIP 包）已安装，且 `~/.cache/clip/ViT-B-32.pt` **已存在**（无需再下载）；
* `torchvision` 可用（CIFAR-10/100 可通过 `allow_download` 开关下载）；
* 若获准，可把 `clip.backend` 设为 `openai_clip` 复用本地权重（与 OpenCLIP ViT-B/32 同架构，
  差别在权重来源与许可证，报告时必须写明）。

## 2. 实验设计（已实现，待真实后端启用）

| 环节 | 设计 |
|---|---|
| 模型 | 冻结 CLIP ViT-B/32 图像/文本编码器；`torch.no_grad()` + `eval()`，一次性 batch GPU 推理 |
| 缓存 | 预计算 8–16 个固定文本模板的**分类 logits**（或图像/文本 embedding），PSO 只读缓存 |
| 决策变量 | 模板非负权重（softmax 参数化，T 维）+ 少量**有界**校准参数（温度、类偏置尺度等） |
| 目标 | 校准集 macro-F1（含 ECE 正则项，权重 0.05） |
| 数据划分 | 校准 50% / 开发 25% / **测试 25%（仅最终一次）** |
| 对照 | 等权集成、随机搜索、SciPy DE、正则化逻辑回归校准（强可微基线）、ring、gbest、cgl_nocontrib、cgl_full |
| 统一预算 | 每种方法 5,000 次目标评价（可配置），目标函数调用**不触发**任何 CLIP 推理 |
| 命名诚实性 | 使用标注校准集 ⇒ 必须称"**验证集监督的 CLIP 后处理校准**"，**不得称零样本** |
| 必须报告 | macro-F1、accuracy、ECE、wall-time、GPU 推理成本、模型权重许可证与数据来源 |

## 3. 合成 logits smoke 的结果（真实运行，非真实 CLIP）

数据：10 类 × 200 样本，12 个模板，其中 3 个强、多数弱、2 个误导，噪声 σ=1.1；
划分 1000 / 500 / 500。预算：每方法 2,000 次目标评价。

**开发集（用于选型）**

| 方法 | 校准 macro-F1 | 开发 macro-F1 | 开发 ECE | 评价次数 | wall-time (s) |
|---|---|---|---|---|---|
| equal_weights | 0.3595 | 0.3717 | 0.2132 | 2 | 0.00 |
| random_search | 0.5898 | 0.6512 | 0.0298 | 2000 | 1.18 |
| scipy_de | 0.6274 | 0.6663 | 0.0233 | 2000 | 1.74 |
| **logistic_calibration** | 0.8108 | **0.8577** | 0.0599 | 2000 | 1.20 |
| ring | 0.6515 | 0.6612 | 0.0262 | 2000 | 1.24 |
| gbest | 0.6347 | 0.6752 | 0.0470 | 2000 | 1.32 |
| cgl_nocontrib | 0.6415 | 0.6652 | 0.0802 | 2000 | 1.31 |
| cgl_full | 0.6387 | **0.6928** | 0.0503 | 2000 | 1.29 |

**测试集（仅用于最终评分，选型只用开发集）**

| 方法 | macro-F1 | accuracy | ECE |
|---|---|---|---|
| **logistic_calibration** | **0.8415** | 0.8420 | 0.0475 |
| **cgl_full** | **0.6688** | 0.6700 | 0.0392 |
| gbest | 0.6588 | 0.6600 | 0.0415 |
| ring | 0.6531 | 0.6540 | 0.0288 |
| scipy_de | 0.6507 | 0.6520 | 0.0533 |
| cgl_nocontrib | 0.6504 | 0.6500 | 0.0609 |
| random_search | 0.6101 | 0.6120 | 0.0287 |
| equal_weights | 0.3662 | 0.3680 | 0.2037 |

**判读（诚实版）**

* 在**无梯度**方法中 `cgl_full` 最好（开发 0.6928 / 测试 0.6688），优于 ring、gbest、DE、随机搜索,
  且优于其**无贡献版本**（开发 +0.028、测试 +0.018）——这是本轮唯一对 M1 有利的证据；
* 但**可微的逻辑回归校准（0.8577/0.8415）远强于所有 PSO 方法**，
  说明在此类低维后处理校准任务上，梯度方法更有优势；
* 等权集成本身很差（0.37），说明**加权确实重要**——但"权重可用 PSO 找到"并不等于
  "贡献引导的图学习是必要的"。
* 测试集使用说明：最终评估执行了**两次**（第一次表格遗漏了逻辑回归基线，第二次补上），
  选型与超参均只用开发集，测试标签从未参与拟合。

## 4. 复现命令

```bash
python -m cgl_pso clip-prepare  -c configs/cgl/clip.yaml --backend synthetic
python -m cgl_pso clip-optimize -c configs/cgl/clip.yaml --budget 2000
python -m cgl_pso clip-evaluate-final -c configs/cgl/clip.yaml
# 若获准做真实实验（需确认下载）：
# python -m cgl_pso clip-prepare -c configs/cgl/clip.yaml --backend openai_clip --allow-download
```

产物：`results/cgl/clip/{prepare_summary.json,optimize_summary.json,final_evaluation.json,cached_logits.npz,split.json}`
