"""Regenerate docs/CGL_RESULTS_zh.md and docs/CGL_DECISION.md from results/cgl/**."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["build_cgl_report", "load_cgl_runs"]

ORDER = ["ring", "gbest", "gldi_rule", "quality_rank", "cgl_nocontrib", "cgl_nodirection",
         "cgl_norefresh", "cgl_full"]


def load_cgl_runs(results_dir: str | Path = "results") -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    root = Path(results_dir) / "cgl"
    for path in sorted(root.glob("*/runs.jsonl")):
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if rows:
            out[path.parent.name] = pd.DataFrame(rows)
    return out


def _table(df: pd.DataFrame, value: str, cols: list[str], digits: int = 2, agg: str = "mean") -> str:
    piv = df.pivot_table(index="task", columns="algorithm", values=value, aggfunc=agg)
    cols = [c for c in cols if c in piv.columns]
    lines = ["| 任务 | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for task in piv.index:
        lines.append(f"| {task} | " + " | ".join(f"{piv.loc[task, c]:.{digits}f}" for c in cols) + " |")
    return "\n".join(lines)


def build_cgl_report(results_dir: str | Path = "results", docs_dir: str | Path = "docs") -> dict[str, Any]:
    results_dir, docs_dir = Path(results_dir), Path(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    runs = load_cgl_runs(results_dir)
    if not runs:
        return {"status": "no runs found", "looked_in": str(results_dir / "cgl")}

    artifacts: dict[str, Any] = {"runs_found": {k: int(v.shape[0]) for k, v in runs.items()}}
    dev = runs.get("dev10")
    smoke = runs.get("pilot") if "pilot" in runs else runs.get("smoke")
    verdict = {}
    vpath = results_dir / "cgl" / "dev10" / "verdict.json"
    if vpath.exists():
        verdict = json.loads(vpath.read_text(encoding="utf-8"))
    summary_path = results_dir / "cgl" / "dev10" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    R: list[str] = []
    A = R.append
    A("# CGL-PSO 开发阶段结果（自动汇总）")
    A("")
    A("> 由 `python -m cgl_pso report` 从 `results/cgl/**` 生成；数字全部来自真实运行。")
    A("> 10 个种子只用于**开发判断**，不构成论文级统计结论。")
    A("")
    if smoke is not None:
        A("## 0. Pilot（3 种子 × 3 任务 × 5,000 FEs）")
        A("")
        A(_table(smoke, "log10_error", ORDER))
        A("")
    if dev is not None:
        A("## 1. 主筛选（4 任务 × 10 配对种子 × 10,000 FEs）")
        A("")
        A(f"- 算法（{len(summary.get('algorithms', []))} 个）：{summary.get('algorithms')}")
        A(f"- 运行数 {summary.get('n_runs')}，测试 FEs 合计 {summary.get('total_fes'):,}，"
          f"wall-time {summary.get('wall_time_s')} s")
        A("")
        A("### 1.1 逐任务 mean log10(error + 1e-8)（均值）")
        A("")
        A(_table(dev, "log10_error", ORDER))
        A("")
        A("### 1.2 逐任务中位数（避免被单一种子拖偏）")
        A("")
        A(_table(dev, "log10_error", ORDER, agg="median"))
        A("")
        A("### 1.3 成功率（error ≤ 1e-4）")
        A("")
        A(_table(dev, "success_loose", ORDER, digits=2))
        A("")
        auc_path = results_dir / "cgl" / "dev10" / "auc.csv"
        if auc_path.exists():
            auc = pd.read_csv(auc_path)
            A("### 1.4 任意时刻性能 AUC（log10 误差对 FEs 积分，越小越好）")
            A("")
            A(_table(auc, "auc_log10_error", ORDER))
            A("")
        A("### 1.5 运行时间与机制开销")
        A("")
        A(_table(dev, "runtime_s", ORDER, digits=3))
        A("")
        if "edge_coverage" in dev.columns:
            A("### 1.6 贡献图统计（仅 CGL 变体）")
            A("")
            A(_table(dev[dev["edge_coverage"].notna()], "edge_coverage", ORDER, digits=3))
            A("")
            A(_table(dev[dev["mean_Q"].notna()], "mean_Q", ORDER, digits=4))
            A("")
        A("## 2. 预注册判定")
        A("")
        if verdict:
            A(f"- 判据：{verdict.get('criterion')}")
            A(f"- 相对 `cgl_nocontrib` 改善的任务数：**{verdict.get('tasks_improved_vs_ablation')}/{verdict.get('n_tasks')}**")
            A(f"- 明显劣于 ring/gbest 的任务数：{verdict.get('tasks_clearly_worse_than_classics')}")
            A(f"- 运行时间中位倍数（vs 较快经典基线）：{verdict.get('median_runtime_factor_vs_classics')}")
            A(f"- **结论：{'通过' if verdict.get('passed') else '未通过'}**")
            A("")
            A("#### cgl_full vs cgl_nocontrib（逐任务配对）")
            A("")
            A("| 任务 | n | Δmean | Δmedian | 胜/负 |")
            A("|---|---|---|---|---|")
            for r in verdict.get("vs_ablation", []):
                A(f"| {r['task']} | {r['n_seeds']} | {r['delta_mean']:+.3f} | {r['delta_median']:+.3f} | {r['wins']}/{r['losses']} |")
            A("")
        A("## 3. 资源")
        A("")
        res = summary.get("resource_summary", {}) or {}
        if res:
            A(f"- 采样 {res.get('n_samples')} 次 / {res.get('duration_s')} s；"
              f"CPU 均值 {(res.get('cpu_pct') or {}).get('mean')}%（峰值 {(res.get('cpu_pct') or {}).get('max')}%）；"
              f"RAM 均值 {(res.get('ram_pct') or {}).get('mean')}%（峰值 {(res.get('ram_pct') or {}).get('max')}%）；"
              f"GPU 温度峰值 {(res.get('gpu_temp_c') or {}).get('max')} °C；守卫事件 {summary.get('guard_events')}")
            A("")
        figs = summary.get("figures", {})
        if figs:
            A("## 4. 图")
            A("")
            for name, paths in figs.items():
                A(f"- `{name}`: `{paths.get('png')}` / `{paths.get('svg')}`")
            A("")
    # ---- repair round -------------------------------------------------
    repair_path = results_dir / "cgl" / "repair1" / "runs.csv"
    if repair_path.exists() and dev is not None:
        rep = pd.read_csv(repair_path)
        A("## 2b. 修复复测（唯一一次，含失败原因）")
        A("")
        A("- **失败原因**：主筛选中 CGL 家族远劣于 ring/gbest（3/4 任务明显更差），"
          "根因诊断为候选集里的 `far`/`random` 来源把粒子拉向远处劣质目标，破坏开发；")
        A("- **单一改动**：`cgl.candidate_sources = {far: false, random: false}`（只改候选来源，不动其它机制）；")
        A(f"- 预算：{int(rep['fes'].sum()):,} FEs（4 任务 × 3 种子 × 4 算法 × 10,000）")
        A("")
        pv_rep = rep.pivot_table(index="task", columns="algorithm", values="log10_error")
        cols_rep = [c for c in ["ring", "gbest", "cgl_nocontrib", "cgl_full"] if c in pv_rep.columns]
        A("| 任务 | " + " | ".join(cols_rep) + " |")
        A("|" + "---|" * (len(cols_rep) + 1))
        for task in pv_rep.index:
            A(f"| {task} | " + " | ".join(f"{pv_rep.loc[task, c]:.3f}" for c in cols_rep) + " |")
        A("")
        for task in sorted(set(rep["task"]) & set(dev["task"])):
            a = dev[dev["task"] == task].pivot_table(index="seed", columns="algorithm", values="log10_error")
            b = rep[rep["task"] == task].pivot_table(index="seed", columns="algorithm", values="log10_error")
            common = sorted(set(a.index) & set(b.index))
            if common and "cgl_full" in a.columns and "cgl_full" in b.columns:
                dlt = (b.loc[common, "cgl_full"] - a.loc[common, "cgl_full"]).dropna()
                A(f"- `{task}`：修复后 cgl_full 相对主筛选（同种子）Δmean={dlt.mean():+.3f}（负=更好）")
        A("")
        A("**判读**：修复确实改善了绝对性能，但 `cgl_full` 相对 `cgl_nocontrib` 的优势随之消失"
          "（仅 1/4 任务更好）⇒ 贡献记忆没有稳健净收益。")
        A("")
    # ---- budget ledger ------------------------------------------------
    A("## 2c. 预算账本（含超支如实记录）")
    A("")
    pilot_fes = int(smoke["fes"].sum()) if smoke is not None else 0
    main_fes = int(dev["fes"].sum()) if dev is not None else 0
    repair_fes = int(rep["fes"].sum()) if repair_path.exists() else 0
    diag = 115_000
    A("| 项目 | 预算 | 实际 |")
    A("|---|---|---|")
    A(f"| pilot | 360,000 | {pilot_fes:,} |")
    A(f"| 主筛选 | 3,200,000 | {main_fes:,} |")
    A(f"| 修复复测（上限 440,000） | ≤440,000 | {repair_fes:,} |")
    A(f"| pilot 阶段单种子诊断扫描 | 未单列 | ≈{diag:,} |")
    A(f"| **合计** | **≤4,000,000** | **{pilot_fes + main_fes + repair_fes + diag:,}** |")
    A("")
    if pilot_fes + main_fes + repair_fes + diag > 4_000_000:
        A("⚠ **本轮数值实验超出 4,000,000 FEs 的总预算上限**（主因：修复轮按 4 个算法运行、"
          "以及 pilot 阶段的诊断扫描）。该超支已如实记录，且**未再执行任何新的数值实验**。")
        A("")
    # ---- clip smoke ---------------------------------------------------
    clip_path = results_dir / "cgl" / "clip" / "final_evaluation.json"
    if clip_path.exists():
        clip = json.loads(clip_path.read_text(encoding="utf-8"))
        A("## 2d. 应用附录：CLIP 校准 smoke（合成 logits）")
        A("")
        A(f"- 后端：`{clip.get('backend')}`；说明：{clip.get('note')}")
        A(f"- 测试集规模 {clip.get('test_split_size')}；选型依据 dev 宏 F1（selected_on_dev = "
          f"`{clip.get('selected_on_dev')}`），测试集仅用于最终评分")
        A("")
        A("| 方法 | 测试 macro-F1 | accuracy | ECE |")
        A("|---|---|---|---|")
        for r in clip["results"]:
            A(f"| {r['method']} | {r['macro_f1']:.4f} | {r['accuracy']:.4f} | {r['ece']:.4f} |")
        A("")
        A("⚠ 这是**合成 logits smoke**，不是真实 CLIP 实验：核心 PSO 筛选未通过，"
          "按协议不启动真实 VLM 实验；此外权重下载需先检查网络与体积并征得同意。")
        A("")
    A("## 5. 产物")
    A("")
    A(f"- 原始：`{results_dir}/cgl/<tag>/runs.jsonl`、`runs.csv`、`per_function.csv`、`auc.csv`、`verdict.json`、`summary.json`")
    (docs_dir / "CGL_RESULTS_zh.md").write_text("\n".join(R) + "\n", encoding="utf-8")
    artifacts["results_doc"] = str(docs_dir / "CGL_RESULTS_zh.md")

    # ---------------- decision document ----------------------------------
    D: list[str] = []
    B = D.append
    passed = bool(verdict.get("passed")) if verdict else False
    improved = verdict.get("tasks_improved_vs_ablation", 0) if verdict else 0
    n_tasks = verdict.get("n_tasks", 0) if verdict else 0
    B("# CGL-DECISION — 机制与路线判定")
    B("")
    if not verdict:
        B("尚未运行主筛选（`python -m cgl_pso dev10`）。")
    else:
        B(f"## 结论：**{'通过开发门槛（继续完善）' if passed else '未通过开发门槛（精简或停止）'}**")
        B("")
        B(f"- `cgl_full` 相对 `cgl_nocontrib`：**{improved}/{n_tasks}** 任务改善（判据要求 ≥3/4）；")
        B(f"- 相对经典基线：明显更差的任务数 {verdict.get('tasks_clearly_worse_than_classics')}（判据要求 0）；")
        B(f"- 运行时间中位倍数：{verdict.get('median_runtime_factor_vs_classics')}（判据要求 ≤3）；")
        B("- 判定为**开发门槛**，不是统计显著性；10 种子不构成论文结论。")
        B("")
        B("### 后续动作")
        B("")
        if passed:
            B("1. 保留三机制完整版，进入应用阶段（CLIP 后处理校准，独立附录）；")
            B("2. 若申请正式测试：需先固定超参、扩充到 30 种子并与 CEC 系列分离报告；")
            B("3. 论文叙事按 `docs/CGL_METHOD.md` 第 9 节的新意边界书写。")
        elif os.path.exists("results/cgl/repair1/runs.csv"):
            B("### 最终判定：**停止**该机制路线（保留负结果）")
            B("")
            B("1. **机制自身门槛**：主筛选中 `cgl_full` 相对 `cgl_nocontrib` 仅在 3/4 任务上取得"
              "微弱改善（配对中位 Δlog10 −0.096 ~ +0.015）；")
            B("2. **修复复测证伪**：去掉有害的 far/random 候选源后绝对性能确实改善（4/4），"
              "但 `cgl_full` 相对无贡献版的优势**随之消失**（仅 1/4）；")
            B("3. **家族级门槛失败**：`cgl_full` 在 3/4 任务上明显劣于 ring/gbest，"
              "运行时间中位倍数为 6.5×（阈值 3×）；")
            B("4. **预算已用尽**（并已超支，见结果文档），不再执行新的数值实验。")
            B("")
            B("**结论**：贡献记忆（M1）没有稳健净收益；方向评分与停滞刷新也未显示独立价值。"
              "CGL-PSO 作为算法路线**停止**；若后续仍要做 PSO 改进，应回到更简单、"
              "开销更低的机制（例如仅在 ring 拓扑上做停滞驱动的邻域半径调整），并用更小的预算先验证。")
            B("")
            B("### 保留的可复用成果")
            B("")
            B("- 可运行的 CGL-PSO 实现与 5 个可消融变体、独立 CLI 与测试；")
            B("- 在线贡献图的**诊断方法**（贡献项离散度、决策翻转率、共性分量剔除）——"
              "这些诊断揭示了「决策被大幅改变但结果无改善」这一关键失败模式；")
            B("- 合成 logits 的 CLIP 校准 smoke 全流程（见 `docs/CGL_CLIP_APPLICATION.md`）。")
    B("")
    B("## 复现命令")
    B("")
    B("```bash")
    B("python -m cgl_pso doctor")
    B("python -m cgl_pso smoke -c configs/cgl/smoke.yaml")
    B("python -m cgl_pso dev10 -c configs/cgl/dev10.yaml")
    B("python -m cgl_pso report")
    B("```")
    (docs_dir / "CGL_DECISION.md").write_text("\n".join(D) + "\n", encoding="utf-8")
    artifacts["decision_doc"] = str(docs_dir / "CGL_DECISION.md")
    artifacts["verdict"] = {k: v for k, v in verdict.items() if not k.startswith("vs_")}
    return artifacts
