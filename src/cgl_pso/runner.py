"""CGL-PSO screening runner: pilot, pre-registered dev10, statistics and figures.

Budget discipline (``docs/CGL_PROTOCOL.md``):
  pilot  : 3 paired seeds x 3 tasks x 5,000 FEs x <=8 algorithms
  dev10  : 4 pre-registered tasks x 10 paired seeds x 10,000 FEs x <=8 algorithms
  total numeric budget <= 4,000,000 FEs; one main screening plus at most one
  justified repair re-test; no automatic 30-seed / CEC2017 escalation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from cgl_pso._infra.config import Config
from cgl_pso._infra.objectives import make_function
from cgl_pso._infra.objectives.base import CountingObjective
from cgl_pso._infra.optim import PSOEngine
from cgl_pso._infra.resources import ResourceGuard, ResourceMonitor
from cgl_pso._infra.seeding import set_global_seed, stable_seed

from .policy import CGL_POLICIES, make_cgl_policy
from .smallworld import SW_MODES, make_smallworld_policy

__all__ = [
    "build_policy_for",
    "run_sweep",
    "smallworld_verdict",
    "selection_diagnostics",
    "CGL_ALGORITHMS",
    "PILOT_TASKS",
    "DEV_TASKS",
    "PILOT_SEEDS",
    "DEV_SEEDS",
    "run_screening",
    "screening_verdict",
    "render_figures",
]

# 8 representative algorithms: 3 classical/reference + 5 ablation points
CGL_ALGORITHMS: tuple[str, ...] = (
    "ring", "gbest", "gldi_rule", "quality_rank",
    "cgl_nocontrib", "cgl_nodirection", "cgl_norefresh", "cgl_full",
)
PILOT_TASKS: tuple[str, ...] = ("sphere", "rastrigin", "ackley")
DEV_TASKS: tuple[str, ...] = ("rosenbrock", "griewank", "zakharov", "alpine1")
PILOT_SEEDS: tuple[int, ...] = (0, 1, 2)
DEV_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
# fresh instance-seed namespace: verified disjoint from every earlier stage
INSTANCE_SEED_BASE = 70_000


@dataclass
class ScreeningResult:
    tag: str
    runs: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: dict[str, Any] = field(default_factory=dict)


def run_screening(
    cfg: Config,
    *,
    tasks: Sequence[str],
    seeds: Sequence[int],
    algorithms: Sequence[str] = CGL_ALGORITHMS,
    dim: int = 30,
    n_particles: int = 40,
    fes: int = 10_000,
    out_dir: str | Path = "results",
    tag: str = "dev10",
    rotate: bool = True,
    shift: float = 0.25,
    progress: bool = True,
    write_figures: bool = True,
) -> ScreeningResult:
    out_dir = Path(out_dir)
    run_dir = out_dir / "cgl" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    iters = max(int(fes) // int(n_particles) - 1, 1)
    effective_fes = int(n_particles) * (iters + 1)
    threads = int(cfg.get_path("threads", 6))
    set_global_seed(int(cfg.get_path("seed", 0)), threads=threads)

    run_cfg = cfg.copy()
    run_cfg.set_path("optimizer.n_particles", int(n_particles))
    run_cfg.set_path("objective.dim", int(dim))

    res_cfg = cfg.get_path("resources", {}) or {}
    monitor = None
    if bool(res_cfg.get("enabled", True)):
        monitor = ResourceMonitor(log_dir, period_s=float(res_cfg.get("log_period_s", 5.0)),
                                  prefix=f"resources_cgl_{tag}")
        monitor.start()
    guard = ResourceGuard(res_cfg, log_path=log_dir / f"guard_cgl_{tag}.jsonl")

    raw_path = run_dir / "runs.jsonl"
    handle = raw_path.open("w", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    total = len(seeds) * len(tasks) * len(algorithms)
    done = 0
    try:
        for seed in seeds:
            for task in tasks:
                instance_seed = stable_seed(INSTANCE_SEED_BASE, task, int(seed))
                for algo in algorithms:
                    cgl_kwargs = dict(run_cfg.get_path("cgl", {}) or {})
                    policy = make_cgl_policy(algo, run_cfg, seed=stable_seed(9_000, algo, seed), **cgl_kwargs)
                    obj = CountingObjective(
                        make_function(task, int(dim), rotate=rotate, shift=shift, seed=instance_seed),
                        max_fes=effective_fes,
                    )
                    engine = PSOEngine(obj, run_cfg, policy, seed=int(seed))
                    t_run = time.perf_counter()
                    res = engine.run(iters=iters, max_fes=effective_fes, curve_points=20)
                    runtime = time.perf_counter() - t_run
                    err = float(res.best_val - obj.f_global)
                    pol_summary = policy.summary() if hasattr(policy, "summary") else {}
                    graph = pol_summary.get("graph", {}) if pol_summary else {}
                    row = {
                        "protocol": tag,
                        "algorithm": algo,
                        "task": task,
                        "seed": int(seed),
                        "dim": int(dim),
                        "instance_seed": int(instance_seed),
                        "fes": int(res.fes),
                        "iterations": int(res.iterations),
                        "error": err,
                        "log10_error": float(np.log10(max(err, 0.0) + 1e-8)),
                        "success_loose": bool(err <= 1e-4),
                        "success_tight": bool(err <= 1e-8),
                        "runtime_s": float(runtime),
                        "select_ms_mean": pol_summary.get("mean_select_ms"),
                        "select_ms_p95": pol_summary.get("p95_select_ms"),
                        "edge_coverage": graph.get("edge_coverage"),
                        "edges_tried": graph.get("edges_tried"),
                        "mean_Q": graph.get("mean_Q_tried"),
                        "positive_rate": graph.get("positive_rate"),
                        "refresh_events": pol_summary.get("refresh_events"),
                        "selection_entropy": pol_summary.get("mean_selection_entropy"),
                        "curve": [[float(f), float(v)] for f, v in res.curve],
                    }
                    rows.append(row)
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    done += 1
                    if progress and done % max(total // 10, 1) == 0:
                        print(f"[cgl {tag} {done}/{total}] {algo:16s} {task:12s} seed={seed} err={err:.4e}", flush=True)
                    if guard.check().level == "pause":
                        print("resource guard pause", flush=True)
                        time.sleep(10.0)
    finally:
        handle.close()
        if monitor is not None:
            monitor.stop()

    df = pd.DataFrame(rows)
    summary = summarise(df, run_dir, tasks=tasks, seeds=seeds, dim=dim, fes=effective_fes, tag=tag)
    summary["wall_time_s"] = round(time.perf_counter() - t0, 2)
    summary["raw_path"] = str(raw_path)
    if monitor is not None:
        summary["resource_summary"] = monitor.summary()
        (run_dir / "resource_summary.json").write_text(
            json.dumps(summary["resource_summary"], indent=2, default=str), encoding="utf-8")
    summary["guard_events"] = len(guard.events)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if write_figures:
        summary["figures"] = render_figures(df, run_dir / "figures", tag=tag)
    return ScreeningResult(tag=tag, runs=df, summary=summary)


# --------------------------------------------------------------------------
def _paired(df: pd.DataFrame, reference: str, candidate: str) -> pd.DataFrame:
    if reference == candidate:
        return pd.DataFrame(columns=["task", "reference", "algorithm", "n_seeds", "delta_mean",
                                     "delta_median", "wins", "losses"])
    rows = []
    for task in sorted(df["task"].unique()):
        b = df[df["task"] == task]
        piv = b.pivot_table(index="seed", columns="algorithm", values="log10_error")
        if reference not in piv.columns or candidate not in piv.columns:
            continue
        d = (piv[candidate] - piv[reference]).dropna()
        rows.append({
            "task": task, "reference": reference, "algorithm": candidate, "n_seeds": int(d.size),
            "delta_mean": float(d.mean()), "delta_median": float(d.median()),
            "wins": int((d < 0).sum()), "losses": int((d > 0).sum()),
        })
    return pd.DataFrame(rows)


def _auc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (algo, task), sub in df.groupby(["algorithm", "task"]):
        vals = []
        for curve in sub["curve"]:
            if not curve:
                continue
            arr = np.asarray(curve, dtype=np.float64)
            fe, best = arr[:, 0], arr[:, 1]
            logerr = np.log10(np.clip(best, 0.0, None) + 1e-8)
            span = fe[-1] - fe[0]
            vals.append(float(np.trapezoid(logerr, fe) / span) if span > 0 else float(logerr[-1]))
        if vals:
            rows.append({"algorithm": algo, "task": task, "auc_log10_error": float(np.mean(vals)), "n": len(vals)})
    return pd.DataFrame(rows)


def screening_verdict(
    df: pd.DataFrame,
    *,
    candidate: str = "cgl_full",
    ablation: str = "cgl_nocontrib",
    classics: Sequence[str] = ("ring", "gbest"),
    min_tasks: int = 3,
    runtime_factor: float = 3.0,
) -> dict[str, Any]:
    """The pre-registered development gate (a screening gate, not significance)."""
    vs_ablation = _paired(df, ablation, candidate)
    better = int((vs_ablation["delta_median"] < 0).sum()) if not vs_ablation.empty else 0
    n_tasks = int(vs_ablation.shape[0])

    classic_rows = []
    for classic in classics:
        tab = _paired(df, classic, candidate)
        classic_rows.append(tab)
    # a subset run (e.g. verifying a single algorithm) has no classic baseline rows
    non_empty = [t for t in classic_rows if t.empty == False]
    classic_tab = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
    worse_than_classic = int((classic_tab["delta_median"] > 0.5).sum()) if not classic_tab.empty else 0

    rt = df.pivot_table(index=["task", "seed"], columns="algorithm", values="runtime_s")
    present = [c for c in classics if c in rt.columns]
    if candidate in rt.columns and present:
        fast = rt[present].min(axis=1)
        factor = float((rt[candidate] / fast).median())
    else:
        # a subset run may contain neither the candidate nor a classic baseline
        factor = float("nan")

    passed = bool(n_tasks > 0 and better >= min_tasks and worse_than_classic == 0 and factor <= runtime_factor)
    return {
        "candidate": candidate,
        "ablation": ablation,
        "criterion": (f">= {min_tasks}/{n_tasks} tasks with improved paired median log-error vs {ablation}, "
                      f"no task clearly worse than ring/gbest (median delta <= +0.5), "
                      f"median runtime <= {runtime_factor}x the faster classical baseline"),
        "tasks_improved_vs_ablation": better,
        "n_tasks": n_tasks,
        "tasks_clearly_worse_than_classics": worse_than_classic,
        "median_runtime_factor_vs_classics": float(factor) if np.isfinite(factor) else None,
        "passed": passed,
        "vs_ablation": vs_ablation.to_dict(orient="records"),
        "vs_classics": classic_tab.to_dict(orient="records") if not classic_tab.empty else [],
    }


def summarise(df: pd.DataFrame, run_dir: Path, *, tasks, seeds, dim, fes, tag) -> dict[str, Any]:
    df.drop(columns=["curve"]).to_csv(run_dir / "runs.csv", index=False)
    # aggregate only the columns this run actually produced (subset sweeps and the
    # small-world prototypes record a different set of mechanism statistics)
    wanted = {
        "n_seeds": ("error", "size"), "mean_error": ("error", "mean"), "median_error": ("error", "median"),
        "mean_log10_error": ("log10_error", "mean"), "success_loose": ("success_loose", "mean"),
        "mean_runtime_s": ("runtime_s", "mean"), "mean_select_ms": ("select_ms_mean", "mean"),
        "edge_coverage": ("edge_coverage", "mean"), "mean_Q": ("mean_Q", "mean"),
        "refresh_events": ("refresh_events", "mean"),
        "mean_nonlocal_share": ("mean_nonlocal_share", "mean"), "p_final": ("p_final", "mean"),
        "sel_global_elite_share": ("sel_global_elite_share", "mean"),
        "sel_self_share": ("sel_self_share", "mean"), "stalled_share": ("stalled_share", "mean"),
    }
    agg = {name: spec for name, spec in wanted.items() if spec[0] in df.columns}
    per = (df.groupby(["algorithm", "task"]).agg(**agg).reset_index()
           .sort_values(["task", "mean_log10_error"]))
    per.to_csv(run_dir / "per_function.csv", index=False)
    auc = _auc(df)
    if not auc.empty:
        auc.to_csv(run_dir / "auc.csv", index=False)
    paired = _paired(df, "cgl_nocontrib", "cgl_full")
    paired.to_csv(run_dir / "paired_cgl_full_vs_nocontrib.csv", index=False)
    for classic in ("ring", "gbest"):
        _paired(df, classic, "cgl_full").to_csv(run_dir / f"paired_cgl_full_vs_{classic}.csv", index=False)
    verdict = screening_verdict(df)
    # Subset sweeps (e.g. the Ring-SmallWorld screening, which has no cgl_nocontrib arm) must not
    # report the CGL development gate as a plain failure: mark it explicitly as not evaluated.
    if {"cgl_full", "cgl_nocontrib"}.issubset(set(df["algorithm"].astype(str))):
        verdict = {**verdict, "applicable": True}
    else:
        verdict = {
            **verdict,
            "applicable": False,
            "note": ("cgl_full / cgl_nocontrib are not both present in this sweep, so the CGL "
                     "development gate was not evaluated here; see smallworld_verdict.json for the "
                     "screening that this sweep does evaluate"),
        }
    (run_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return {
        "tag": tag,
        "n_runs": int(df.shape[0]),
        "tasks": list(tasks),
        "seeds": list(seeds),
        "dim": int(dim),
        "fes": int(fes),
        "total_fes": int(df["fes"].sum()),
        "algorithms": sorted(df["algorithm"].unique()),
        "verdict": {k: v for k, v in verdict.items() if not k.startswith("vs_")},
        "artifacts": {
            "raw": str(run_dir / "runs.jsonl"),
            "runs_csv": str(run_dir / "runs.csv"),
            "per_function": str(run_dir / "per_function.csv"),
            "auc": str(run_dir / "auc.csv"),
            "verdict": str(run_dir / "verdict.json"),
        },
    }


def render_figures(df: pd.DataFrame, fig_dir: Path, *, tag: str) -> dict[str, Any]:
    """Convergence curves per task + an ablation bar chart (PNG and SVG)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {}
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#999999", "#E69F00"]

    tasks = sorted(df["task"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.6), dpi=200, squeeze=False)
    for ax, task in zip(axes[0], tasks):
        sub = df[df["task"] == task]
        for i, algo in enumerate(sorted(sub["algorithm"].unique())):
            curves = [np.asarray(c, dtype=np.float64) for c in sub[sub["algorithm"] == algo]["curve"] if c]
            if not curves:
                continue
            n = min(len(c) for c in curves)
            stack = np.stack([c[:n] for c in curves])
            ax.plot(stack[0, :, 0], np.maximum(stack[:, :, 1].mean(axis=0), 1e-16),
                    label=algo, color=palette[i % len(palette)], linewidth=1.2)
        ax.set_yscale("log")
        ax.set_title(task, fontsize=9)
        ax.grid(alpha=0.25, which="both")
        ax.set_xlabel("FEs", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0][0].set_ylabel("best fitness", fontsize=8)
    axes[0][-1].legend(fontsize=6, loc="best")
    fig.suptitle(f"CGL-PSO convergence ({tag})", fontsize=10)
    fig.tight_layout()
    png = fig_dir / "convergence.png"
    svg = fig_dir / "convergence.svg"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    out["convergence"] = {"png": str(png), "svg": str(svg)}

    med = df.groupby(["algorithm", "task"])["log10_error"].median().unstack()
    fig, ax = plt.subplots(figsize=(max(6.0, 0.7 * len(med.index) + 3), 3.8), dpi=200)
    x = np.arange(len(med.index))
    width = 0.8 / max(len(med.columns), 1)
    for i, task in enumerate(med.columns):
        ax.bar(x + i * width, med[task].values, width=width, label=task)
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(med.index, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("median log10(error+1e-8)", fontsize=8)
    ax.set_title(f"CGL-PSO ablation summary ({tag})", fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=7)
    fig.tight_layout()
    png2 = fig_dir / "ablation.png"
    svg2 = fig_dir / "ablation.svg"
    fig.savefig(png2, bbox_inches="tight")
    fig.savefig(svg2, bbox_inches="tight")
    plt.close(fig)
    out["ablation"] = {"png": str(png2), "svg": str(svg2)}
    return out


# ---------------------------------------------------------------------------
# Ring-SmallWorld-PSO screening (review 2026-10-20)
# ---------------------------------------------------------------------------
def build_policy_for(kind: str, cfg: Config, *, seed: int, record: bool = True, **kwargs: Any):
    """Generic factory: classic baselines, CGL variants and small-world variants.

    ``kind`` selects the implementation; ``kwargs`` carry the per-spec settings
    (e.g. ``p``/``stall_threshold`` for the small-world prototypes), which lets the
    same runner evaluate several *settings of the same implementation* side by side.
    """
    from cgl_pso._infra.optim.policies import make_classic_policy

    if kind in {"ring", "gbest", "rank", "random", "fips", "clpso"}:
        return make_classic_policy(kind, cfg)
    if kind.startswith("sw_"):
        return make_smallworld_policy(kind, cfg, seed=seed, record=record, **kwargs)
    return make_cgl_policy(kind, cfg, seed=seed, record=record, **kwargs)


def run_sweep(
    cfg: Config,
    *,
    specs: Sequence[tuple[str, str, dict[str, Any]]],
    tasks: Sequence[str],
    seeds: Sequence[int],
    dim: int = 30,
    n_particles: int = 40,
    fes: int = 10_000,
    out_dir: str | Path = "results",
    tag: str = "sw",
    progress: bool = True,
    write_figures: bool = True,
) -> ScreeningResult:
    """Run a labelled list of ``(label, kind, kwargs)`` specifications.

    Every specification consumes the same FEs on the same instances and seeds; the
    label becomes the algorithm name in the artefacts, so two settings of the same
    implementation can be compared directly.
    """
    out_dir = Path(out_dir)
    run_dir = out_dir / "cgl" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    iters = max(int(fes) // int(n_particles) - 1, 1)
    effective_fes = int(n_particles) * (iters + 1)
    set_global_seed(int(cfg.get_path("seed", 0)), threads=int(cfg.get_path("threads", 6)))
    run_cfg = cfg.copy()
    run_cfg.set_path("optimizer.n_particles", int(n_particles))

    res_cfg = cfg.get_path("resources", {}) or {}
    monitor = None
    if bool(res_cfg.get("enabled", True)):
        monitor = ResourceMonitor(log_dir, period_s=float(res_cfg.get("log_period_s", 5.0)),
                                  prefix=f"resources_cgl_{tag}")
        monitor.start()
    guard = ResourceGuard(res_cfg, log_path=log_dir / f"guard_cgl_{tag}.jsonl")

    raw_path = run_dir / "runs.jsonl"
    handle = raw_path.open("w", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    total = len(seeds) * len(tasks) * len(specs)
    done = 0
    try:
        for seed in seeds:
            for task in tasks:
                instance_seed = stable_seed(INSTANCE_SEED_BASE, task, int(seed))
                for label, kind, kwargs in specs:
                    policy = build_policy_for(kind, run_cfg, seed=stable_seed(9_000, label, seed), **kwargs)
                    obj = CountingObjective(
                        make_function(task, int(dim), rotate=True, shift=0.25, seed=instance_seed),
                        max_fes=effective_fes,
                    )
                    engine = PSOEngine(obj, run_cfg, policy, seed=int(seed))
                    t_run = time.perf_counter()
                    res = engine.run(iters=iters, max_fes=effective_fes, curve_points=20)
                    runtime = time.perf_counter() - t_run
                    err = float(res.best_val - obj.f_global)
                    summary_pol = policy.summary() if hasattr(policy, "summary") else {}
                    graph = summary_pol.get("graph", {}) if isinstance(summary_pol, dict) else {}
                    trace = getattr(policy, "iteration_trace", None) or getattr(policy, "trace", []) or []
                    def _mean(key, default=None):
                        vals = [t[key] for t in trace if key in t]
                        return float(np.mean(vals)) if vals else default
                    row = {
                        "protocol": tag,
                        "algorithm": label,
                        "kind": kind,
                        "settings": json.dumps(kwargs, default=str),
                        "task": task,
                        "seed": int(seed),
                        "dim": int(dim),
                        "instance_seed": int(instance_seed),
                        "fes": int(res.fes),
                        "iterations": int(res.iterations),
                        "error": err,
                        "log10_error": float(np.log10(max(err, 0.0) + 1e-8)),
                        "success_loose": bool(err <= 1e-4),
                        "success_tight": bool(err <= 1e-8),
                        "runtime_s": float(runtime),
                        "select_ms_mean": summary_pol.get("mean_select_ms"),
                        "mean_nonlocal_share": summary_pol.get("mean_nonlocal_share"),
                        "p_final": summary_pol.get("p_final"),
                        "edge_coverage": graph.get("edge_coverage"),
                        "sel_global_elite_share": _mean("sel_global_elite_share"),
                        "sel_ring_elite_share": _mean("sel_ring_elite_share"),
                        "sel_self_share": _mean("sel_self_share"),
                        "sel_mean_rank": _mean("sel_mean_rank"),
                        "stalled_share": _mean("cgl_stalled", _mean("sw_stalled_share")),
                        "diversity": float(np.mean([t["diversity"] for t in engine.trace if "diversity" in t]))
                                     if engine.trace else None,
                        "curve": [[float(f), float(v)] for f, v in res.curve],
                    }
                    rows.append(row)
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    done += 1
                    if progress and done % max(total // 10, 1) == 0:
                        print(f"[{tag} {done}/{total}] {label:22s} {task:12s} seed={seed} err={err:.4e}", flush=True)
                    if guard.check().level == "pause":
                        print("resource guard pause", flush=True)
                        time.sleep(10.0)
    finally:
        handle.close()
        if monitor is not None:
            monitor.stop()

    df = pd.DataFrame(rows)
    summary = summarise(df, run_dir, tasks=tasks, seeds=seeds, dim=dim, fes=effective_fes, tag=tag)
    summary["wall_time_s"] = round(time.perf_counter() - t0, 2)
    summary["raw_path"] = str(raw_path)
    summary["specs"] = [{"label": l, "kind": k, "settings": s} for l, k, s in specs]
    verdict = smallworld_verdict(df)
    summary["smallworld_verdict"] = {k: v for k, v in verdict.items() if not k.startswith("vs_")}
    (run_dir / "smallworld_verdict.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    if monitor is not None:
        summary["resource_summary"] = monitor.summary()
        (run_dir / "resource_summary.json").write_text(json.dumps(summary["resource_summary"], indent=2, default=str),
                                                       encoding="utf-8")
    summary["guard_events"] = len(guard.events)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if write_figures:
        summary["figures"] = render_figures(df, run_dir / "figures", tag=tag)
    return ScreeningResult(tag=tag, runs=df, summary=summary)


def smallworld_verdict(
    df: pd.DataFrame,
    *,
    reference: str = "ring",
    prefixes: Sequence[str] = ("sw_fixed", "sw_stall", "sw_bandit", "cgl_full"),
    min_tasks_improved: int = 1,
    collapse_threshold: float = 0.5,
    runtime_factor: float = 1.5,
) -> dict[str, Any]:
    """Criteria from the review: improve on ring on *some* tasks, never collapse,
    stay cheap, and keep the online module only if it beats the fixed version.

    Candidates are matched by **prefix** because a sweep labels the same
    implementation several times (``sw_fixed_p05``, ``sw_fixed_p15``, ...).
    """
    def _match(prefix: str) -> list[str]:
        return sorted(a for a in df["algorithm"].unique() if a == prefix or a.startswith(prefix + "_"))

    out: dict[str, Any] = {
        "reference": reference,
        "criteria": (
            f">= {min_tasks_improved} task(s) with paired median delta log10 < 0 vs {reference}; "
            f"no task with delta > +{collapse_threshold}; runtime median <= {runtime_factor}x {reference}; "
            f"the online variant must be better than the fixed variant to be kept"
        ),
        "prefixes": list(prefixes),
    }
    rt = df.pivot_table(index=["task", "seed"], columns="algorithm", values="runtime_s")
    per_candidate: dict[str, Any] = {}
    for prefix in prefixes:
        for cand in _match(prefix):
            tab = _paired(df, reference, cand)
            improved = int((tab["delta_median"] < 0).sum()) if not tab.empty else 0
            collapsed = int((tab["delta_median"] > collapse_threshold).sum()) if not tab.empty else 0
            factor = (float((rt[cand] / rt[reference]).median())
                      if cand in rt.columns and reference in rt.columns else float("nan"))
            mean_log10 = float(df[df["algorithm"] == cand]["log10_error"].mean())
            per_candidate[cand] = {
                "kind": prefix,
                "tasks_improved_vs_reference": improved,
                "n_tasks": int(tab.shape[0]),
                "tasks_collapsed": collapsed,
                "median_runtime_factor": factor,
                "mean_log10_error": mean_log10,
                "passed": bool(improved >= min_tasks_improved and collapsed == 0
                               and (not np.isfinite(factor) or factor <= runtime_factor)),
                "vs_reference": tab.to_dict(orient="records"),
            }
    out["per_candidate"] = per_candidate

    def _best(kind: str) -> str | None:
        cands = [c for c, v in per_candidate.items() if v["kind"] == kind]
        if not cands:
            return None
        return min(cands, key=lambda c: per_candidate[c]["mean_log10_error"])

    fixed, online = _best("sw_fixed"), _best("sw_bandit")
    stall = _best("sw_stall")
    if fixed and online:
        keep_online = bool(per_candidate[online]["mean_log10_error"] < per_candidate[fixed]["mean_log10_error"])
        out["online_vs_fixed"] = {
            "best_fixed": fixed,
            "best_online": online,
            "fixed_mean_log10": per_candidate[fixed]["mean_log10_error"],
            "online_mean_log10": per_candidate[online]["mean_log10_error"],
            "keep_online": keep_online,
            "note": "if the fixed-probability version is at least as good, the online module is dropped",
        }
    out["best_settings"] = {"fixed": fixed, "stall": stall, "online": online}
    passing = [c for c, v in per_candidate.items() if v["passed"]]
    out["passing_candidates"] = passing
    out["any_candidate_passed"] = bool(passing)
    if passing:
        out["recommended"] = min(passing, key=lambda c: per_candidate[c]["mean_log10_error"])
    return out


def selection_diagnostics(
    cfg: Config,
    *,
    specs: Sequence[tuple[str, str, dict[str, Any]]],
    tasks: Sequence[str],
    seeds: Sequence[int],
    dim: int = 30,
    n_particles: int = 40,
    fes: int = 10_000,
) -> pd.DataFrame:
    """Aggregate the selection-composition statistics requested by the review.

    Reports, per algorithm: share of particles selecting the global elite / the ring
    elite / themselves, the mean pbest rank of the chosen target, the stall share and
    the swarm diversity -- i.e. whether a mechanism actually changes *whom* particles
    learn from, and whether it degrades diversity (a common failure mode).
    """
    iters = max(int(fes) // int(n_particles) - 1, 1)
    records = []
    for seed in seeds:
        for task in tasks:
            instance_seed = stable_seed(INSTANCE_SEED_BASE, task, int(seed))
            for label, kind, kwargs in specs:
                policy = build_policy_for(kind, cfg, seed=stable_seed(9_000, label, seed), **kwargs)
                obj = CountingObjective(
                    make_function(task, int(dim), rotate=True, shift=0.25, seed=instance_seed),
                    max_fes=int(n_particles) * (iters + 1),
                )
                engine = PSOEngine(obj, cfg, policy, seed=int(seed))
                engine.run(iters=iters, max_fes=int(n_particles) * (iters + 1), curve_points=5)
                trace = getattr(policy, "iteration_trace", None) or getattr(policy, "trace", []) or []
                def _mean(key):
                    vals = [t[key] for t in trace if key in t]
                    return float(np.mean(vals)) if vals else float("nan")
                records.append({
                    "algorithm": label, "task": task, "seed": int(seed),
                    "sel_global_elite_share": _mean("sel_global_elite_share"),
                    "sel_ring_elite_share": _mean("sel_ring_elite_share"),
                    "sel_self_share": _mean("sel_self_share"),
                    "sel_mean_rank": _mean("sel_mean_rank"),
                    "nonlocal_share": _mean("sw_nonlocal_share"),
                    "stalled_share": _mean("cgl_stalled") if not np.isnan(_mean("cgl_stalled")) else _mean("sw_stalled_share"),
                    "sel_ms": float(np.mean(getattr(policy, "_select_ms", [np.nan]))),
                    "diversity_final": float(engine.trace[-1]["diversity"]) if engine.trace else float("nan"),
                })
    return pd.DataFrame(records)
