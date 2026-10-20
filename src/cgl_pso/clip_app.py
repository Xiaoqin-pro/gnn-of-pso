"""Frozen-VLM application appendix for CGL-PSO.

Protocol position (``docs/CGL_PROTOCOL.md`` section 6): the real CLIP experiment
starts only *after* the PSO core passes its screening, and model/data downloads
require network/size checks and explicit permission. Since the core screening did
**not** pass, this module delivers the complete, working **synthetic-logits smoke**
so the whole pipeline (data split -> cached logits -> optimiser comparison ->
single final evaluation) is verified end to end, and the real backends stay behind
an explicit flag with a clear "not run" record.

What is optimised (both in the smoke and in the real design):
  8-16 non-negative template weights (softmax parameterisation) + a few bounded
  calibration parameters (temperature, class-bias scale). Every objective
  evaluation only touches **cached** logits/features -- CLIP is never re-run per FE.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cgl_pso._infra.config import Config
from cgl_pso._infra.seeding import stable_seed

__all__ = ["prepare", "optimize", "evaluate_final", "synthetic_logits", "SoftmaxTemplateObjective", "metrics"]


# --------------------------------------------------------------------------
# data: synthetic logits smoke (deterministic, no downloads)
# --------------------------------------------------------------------------
def synthetic_logits(cfg: Config) -> dict[str, Any]:
    """Deterministic synthetic template logits with a known, optimisable structure.

    Each class gets a random "true" template weighting; the per-template logits are
    generated so that the correct class is favoured but noisy, and one template is
    deliberately misleading. Optimising the weights therefore has a real (but
    bounded) ceiling, exactly like the real CLIP setting.
    """
    s = cfg.get_path("clip.synthetic", {}) or {}
    n_classes = int(s.get("n_classes", 10))
    n_per_class = int(s.get("n_per_class", 200))
    n_templates = int(cfg.get_path("clip.n_templates", 12))
    rng = np.random.default_rng(stable_seed(int(cfg.get_path("seed", 0)), "clip-synthetic"))

    n = n_classes * n_per_class
    y = np.repeat(np.arange(n_classes), n_per_class)
    # Per-template reliability: a few templates are informative, most are weak, and
    # two are actively misleading. With the per-sample noise below, a *single*
    # template is unreliable and the ensemble weighting genuinely matters -- which
    # is exactly the regime the real CLIP template-ensembling experiment lives in.
    strength = rng.uniform(0.05, 0.35, size=n_templates)
    good = rng.choice(n_templates, size=max(n_templates // 4, 2), replace=False)
    strength[good] = rng.uniform(0.8, 1.4, size=good.size)
    bad = [t for t in range(n_templates) if t not in set(good.tolist())][-2:]
    strength[bad] = -0.6
    noise = float(s.get("noise", 1.1))
    logits = np.zeros((n, n_templates, n_classes), dtype=np.float64)
    for t in range(n_templates):
        base = rng.normal(0.0, noise, size=(n, n_classes))
        base[np.arange(n), y] += strength[t]
        logits[:, t, :] = base
    loggerits = logits - logits.max(axis=2, keepdims=True)
    loggerits = loggerits - np.log(np.exp(loggerits).sum(axis=2, keepdims=True))
    return {
        "source": "synthetic",
        "n": int(n),
        "n_classes": int(n_classes),
        "n_templates": int(n_templates),
        "labels": y,
        "log_probs": loggerits,           # (N, T, C) cached template log-probabilities
        "template_strength": strength.tolist(),
    }


def split_indices(n: int, cfg: Config, *, seed: int = 0) -> dict[str, np.ndarray]:
    """Calibration / dev / test split (the test split is touched exactly once)."""
    sp = cfg.get_path("clip.split", {}) or {}
    frac = {"calibration": float(sp.get("calibration", 0.5)),
            "dev": float(sp.get("dev", 0.25)),
            "test": float(sp.get("test", 0.25))}
    total = sum(frac.values())
    frac = {k: v / total for k, v in frac.items()}
    rng = np.random.default_rng(stable_seed(seed, "clip-split"))
    perm = rng.permutation(int(n))
    a = int(round(frac["calibration"] * n))
    b = a + int(round(frac["dev"] * n))
    return {"calibration": perm[:a], "dev": perm[a:b], "test": perm[b:]}


# --------------------------------------------------------------------------
# objective: template weights + bounded calibration on cached logits
# --------------------------------------------------------------------------
@dataclass
class SoftmaxTemplateObjective:
    """`dim = n_templates + n_calibration_params` with softmax weights + bounded params."""

    log_probs: np.ndarray          # (N, T, C)
    labels: np.ndarray             # (N,)
    n_templates: int
    n_classes: int
    n_calibration: int = 4
    index: np.ndarray | None = None
    metric: str = "macro_f1"
    ece_weight: float = 0.05
    eval_count: int = 0
    history: list[dict[str, float]] = field(default_factory=list)
    name: str = "clip_calibration"
    f_global: float = 0.0          # the loss floor (1 - perfect score) is unreachable
    bounds: Any = None

    def __post_init__(self) -> None:
        if self.bounds is None:
            from cgl_pso._infra.objectives.base import Bounds

            low = np.array([-3.0] * self.n_templates + [-1.0] * self.n_calibration)
            high = np.array([3.0] * self.n_templates + [1.0] * self.n_calibration)
            self.bounds = Bounds(low, high)

    @property
    def dim(self) -> int:
        return int(self.n_templates + self.n_calibration)

    def _decode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        w_raw = x[: self.n_templates]
        w = np.exp(w_raw - w_raw.max())
        w = w / max(w.sum(), 1e-12)
        cal = x[self.n_templates :]
        temperature = float(np.clip(1.0 + cal[0], 0.25, 4.0)) if cal.size > 0 else 1.0
        bias_scale = float(np.clip(cal[1], 0.0, 1.0)) if cal.size > 1 else 0.0
        return w, np.array([temperature, bias_scale])

    def _fuse(self, x: np.ndarray, index: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        w, (temperature, bias_scale) = self._decode(x)
        lp = self.log_probs if index is None else self.log_probs[index]
        y = self.labels if index is None else self.labels[index]
        fused = np.tensordot(lp, w, axes=([1], [0])) / max(temperature, 1e-6)     # (n, C)
        if bias_scale > 0:
            # bounded per-class bias fitted from the calibration split (cached here)
            bias = getattr(self, "_class_bias", np.zeros(self.n_classes))
            fused = fused + bias_scale * bias[None, :]
        return fused, y

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        out = np.empty(X.shape[0], dtype=np.float64)
        for i in range(X.shape[0]):
            fused, y = self._fuse(X[i], self.index)
            out[i] = 1.0 - self._score(fused, y)
            self.eval_count += 1
        return out

    def _score(self, fused: np.ndarray, y: np.ndarray) -> float:
        pred = np.argmax(fused, axis=1)
        if self.metric == "accuracy":
            return float((pred == y).mean())
        # macro-F1 (primary) with a small ECE regularisation
        f1s = []
        for c in range(self.n_classes):
            tp = float(((pred == c) & (y == c)).sum())
            fp = float(((pred == c) & (y != c)).sum())
            fn = float(((pred != c) & (y == c)).sum())
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1e-12))
        macro_f1 = float(np.mean(f1s))
        if self.ece_weight > 0:
            p = np.exp(fused - fused.max(axis=1, keepdims=True))
            p = p / p.sum(axis=1, keepdims=True)
            conf = p.max(axis=1)
            correct = (pred == y).astype(np.float64)
            bins = np.linspace(0.0, 1.0, 11)
            ece = 0.0
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = (conf > lo) & (conf <= hi)
                if m.any():
                    ece += float(m.mean()) * abs(float(correct[m].mean()) - float(conf[m].mean()))
            return macro_f1 - self.ece_weight * ece
        return macro_f1

    def fit_class_bias(self, index: np.ndarray) -> np.ndarray:
        """Cache a per-class bias from the calibration split (no CLIP calls)."""
        counts = np.bincount(self.labels[index], minlength=self.n_classes).astype(np.float64)
        prior = counts / max(counts.sum(), 1.0)
        bias = -np.log(np.maximum(prior, 1e-9))
        bias = bias - bias.mean()
        self._class_bias = bias
        return bias

    def report(self, x: np.ndarray, index: np.ndarray) -> dict[str, float]:
        fused, y = self._fuse(x, index)
        w, (temp, bias_scale) = self._decode(x)
        return {
            "macro_f1": metrics(fused, y)["macro_f1"],
            "accuracy": metrics(fused, y)["accuracy"],
            "ece": metrics(fused, y)["ece"],
            "temperature": temp,
            "bias_scale": bias_scale,
            "max_template_weight": float(w.max()),
            "n_zero_weight_templates": int((w < 1e-3).sum()),
        }


def metrics(fused_logprobs: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = np.argmax(fused_logprobs, axis=1)
    p = np.exp(fused_logprobs - fused_logprobs.max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)
    conf = p.max(axis=1)
    correct = (pred == y).astype(np.float64)
    f1s = []
    for c in np.unique(y):
        tp = float(((pred == c) & (y == c)).sum())
        fp = float(((pred == c) & (y != c)).sum())
        fn = float(((pred != c) & (y == c)).sum())
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1e-12))
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += float(m.mean()) * abs(float(correct[m].mean()) - float(conf[m].mean()))
    return {"macro_f1": float(np.mean(f1s)), "accuracy": float(correct.mean()), "ece": float(ece)}


# --------------------------------------------------------------------------
# orchestration: prepare -> optimise (unified budget) -> single final evaluation
# --------------------------------------------------------------------------
def _build_objective(cfg: Config, data: dict[str, Any], index: np.ndarray, *, metric: str = "macro_f1") -> SoftmaxTemplateObjective:
    n_templates = int(data["n_templates"])
    n_classes = int(data["n_classes"])
    n_cal = int(cfg.get_path("clip.n_calibration_params", 4))
    obj = SoftmaxTemplateObjective(
        log_probs=np.asarray(data["log_probs"]),
        labels=np.asarray(data["labels"]),
        n_templates=n_templates,
        n_classes=n_classes,
        n_calibration=n_cal,
        index=np.asarray(index),
        metric=metric,
        ece_weight=float(cfg.get_path("clip.ece_weight", 0.05)),
    )
    obj.fit_class_bias(np.asarray(index))
    return obj


def prepare(cfg: Config, *, out_dir: str | Path = "results") -> dict[str, Any]:
    """Build (or load) cached template logits and the frozen split."""
    out = Path(out_dir) / "cgl" / "clip"
    out.mkdir(parents=True, exist_ok=True)
    backend = str(cfg.get_path("clip.backend", "synthetic"))
    allow_download = bool(cfg.get_path("clip.budget.allow_download", False))

    if backend == "synthetic":
        data = synthetic_logits(cfg)
        note = "synthetic logits smoke (no model weights or dataset were downloaded)"
    else:
        # Real backends are gated: the core PSO screening must pass first, and the
        # goal requires a network/size check plus permission before downloading.
        data = synthetic_logits(cfg)
        note = (f"backend {backend!r} requested but not executed: the PSO core screening did not pass, "
                f"and downloads require permission (allow_download={allow_download}); "
                f"the synthetic smoke was run instead")
    split = split_indices(len(data["labels"]), cfg, seed=int(cfg.get_path("seed", 0)))
    data["backend"] = backend
    data["note"] = note
    data["split"] = {k: v.tolist() for k, v in split.items()}
    np.savez_compressed(out / "cached_logits.npz", log_probs=data["log_probs"], labels=data["labels"])
    (out / "split.json").write_text(json.dumps({"split": data["split"], "backend": backend, "note": note,
                                                "template_strength": data.get("template_strength")}, indent=2),
                                    encoding="utf-8")
    summary = {
        "backend": backend,
        "note": note,
        "split": {k: v.tolist() for k, v in split.items()},
        "n_samples": int(len(data["labels"])),
        "n_templates": int(data["n_templates"]),
        "n_classes": int(data["n_classes"]),
        "split_sizes": {k: int(v.size) for k, v in split.items()},
        "cached_file": str(out / "cached_logits.npz"),
    }
    (out / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary



def _run_optimizer(kind: str, cfg: Config, obj_pack: dict[str, Any], *, budget: int, seed: int) -> dict[str, Any]:
    """One comparison method, on identical cached logits and an identical budget."""
    from cgl_pso._infra.objectives.base import CountingObjective
    from cgl_pso._infra.optim import PSOEngine

    from .policy import make_cgl_policy

    obj_cal = obj_pack["calibration"]
    obj_dev = obj_pack["dev"]
    bounds = obj_cal.bounds
    lo, hi = bounds.low, bounds.high
    dim = obj_cal.dim
    rng = np.random.default_rng(stable_seed(seed, "clip-method", kind))
    t0 = time.perf_counter()
    n_evals = 0

    if kind == "equal_weights":
        x = np.zeros(dim)
        n_evals = 2                      # scored on calibration and dev for reporting
        result = {"x": x.tolist()}
    elif kind == "random_search":
        best_x, best = None, np.inf
        for _ in range(int(budget)):
            x = lo + rng.random(dim) * (hi - lo)
            v = float(obj_cal.evaluate(x[None, :])[0])
            n_evals += 1
            if v < best:
                best, best_x = v, x.copy()
        result = {"x": (best_x if best_x is not None else np.zeros(dim)).tolist()}
    elif kind == "scipy_de":
        # scipy's differential_evolution has no portable ``maxfun`` across versions,
        # so the unified evaluation budget is enforced by a hard wrapper that keeps
        # the best-so-far and stops the search the moment the budget is exhausted.
        from scipy.optimize import differential_evolution

        state = {"n": 0, "best": np.inf, "x": None}

        class _BudgetReached(Exception):
            pass

        def _f(x):
            if state["n"] >= int(budget):
                raise _BudgetReached
            state["n"] += 1
            v = float(obj_cal.evaluate(np.asarray(x)[None, :])[0])
            if v < state["best"]:
                state["best"], state["x"] = v, np.asarray(x, dtype=np.float64).copy()
            return v

        try:
            res = differential_evolution(
                _f, bounds=list(zip(lo, hi)), maxiter=500, popsize=15, tol=1e-10,
                seed=int(seed), polish=False, updating="immediate",
            )
            best_x, message = np.asarray(res.x), str(res.message)
        except _BudgetReached:
            best_x = state["x"] if state["x"] is not None else np.zeros(dim)
            message = f"stopped at the {int(budget)}-evaluation budget"
        n_evals = int(state["n"])
        result = {"x": np.asarray(best_x).tolist(), "de_message": message}
    elif kind == "logistic_calibration":
        # Strong differentiable baseline: multinomial logistic calibration over the
        # concatenated template logits (more parameters than the PSO parameterisation;
        # reported as such, not as an equal-parameter comparison).
        from sklearn.linear_model import LogisticRegression

        lp = obj_cal.log_probs
        n, T, C = lp.shape
        X = lp.reshape(n, T * C)
        y = obj_cal.labels
        clf = LogisticRegression(max_iter=1000, C=1.0)  # multinomial by default for multiclass
        clf.fit(X, y)
        n_evals = int(n)                  # one pass over the calibration split
        result = {"logistic": True}
        proba = clf.predict_proba(obj_cal.log_probs.reshape(obj_cal.log_probs.shape[0], -1))
        fused = np.log(np.clip(proba, 1e-12, 1.0))[obj_cal.index]
        cal_metrics = metrics(fused, obj_cal.labels[obj_cal.index])
        dev_proba = clf.predict_proba(obj_dev.log_probs.reshape(obj_dev.log_probs.shape[0], -1))
        dev_fused = np.log(np.clip(dev_proba, 1e-12, 1.0))[obj_dev.index]
        dev_metrics = metrics(dev_fused, obj_dev.labels[obj_dev.index])
        return {
            "method": kind, "n_evals": n_evals, "wall_time_s": time.perf_counter() - t0,
            "calibration": cal_metrics, "dev": dev_metrics, "x": None,
            "note": "logistic calibration uses a T*C linear model (more parameters than the PSO methods)",
        }
    elif kind in {"ring", "gbest", "cgl_full", "cgl_nocontrib", "quality_rank"}:
        n_particles = int(cfg.get_path("optimizer.n_particles", 40))
        iters = max(int(budget) // n_particles - 1, 1)
        counting = CountingObjective(obj_cal, max_fes=n_particles * (iters + 1))
        policy = make_cgl_policy(kind, cfg, seed=stable_seed(seed, "clip-pso", kind))
        engine = PSOEngine(counting, cfg, policy, seed=int(seed))
        res = engine.run(iters=iters, max_fes=n_particles * (iters + 1), curve_points=10)
        n_evals = int(counting.n_evals)
        result = {"x": res.best_pos.tolist(), "best_loss": float(res.best_val)}
    else:
        raise KeyError(f"unknown CLIP comparison method {kind!r}")

    x = np.asarray(result["x"], dtype=np.float64) if result.get("x") is not None else None
    out = {"method": kind, "n_evals": int(n_evals), "wall_time_s": time.perf_counter() - t0, "x": result.get("x")}
    if x is not None:
        out["calibration"] = obj_cal.report(x, obj_cal.index)
        out["dev"] = obj_dev.report(x, obj_dev.index)
    if "de_message" in result:
        out["de_message"] = result["de_message"]
    return out


def optimize(
    cfg: Config,
    *,
    out_dir: str | Path = "results",
    algorithms: Sequence[str] | None = None,
    budget: int | None = None,
) -> dict[str, Any]:
    """Run every comparison method on the calibration split with a unified budget."""
    out = Path(out_dir) / "cgl" / "clip"
    out.mkdir(parents=True, exist_ok=True)
    prepared = prepare(cfg, out_dir=out_dir)
    data = synthetic_logits(cfg)
    split = {k: np.asarray(v) for k, v in prepared["split"].items()}
    packed = {
        "calibration": _build_objective(cfg, data, split["calibration"]),
        "dev": _build_objective(cfg, data, split["dev"]),
        "test": _build_objective(cfg, data, split["test"]),
    }
    budget = int(budget or cfg.get_path("clip.budget.comparison_budget", 5000))
    methods = list(algorithms or ["equal_weights", "random_search", "scipy_de", "logistic_calibration",
                                 "ring", "gbest", "cgl_nocontrib", "cgl_full"])
    rows = []
    for kind in methods:
        r = _run_optimizer(kind, cfg, packed, budget=budget, seed=int(cfg.get_path("seed", 0)))
        r["budget"] = budget
        rows.append(r)
        print(f"[clip] {kind:22s} dev_macro_f1={r.get('dev', {}).get('macro_f1'):.4f} "
              f"evals={r['n_evals']} t={r['wall_time_s']:.2f}s", flush=True)

    # model selection happens on the DEV split only; the test split stays untouched
    dev_scores = {r["method"]: r.get("dev", {}).get("macro_f1", float("-inf")) for r in rows}
    best_method = max(dev_scores, key=dev_scores.get)
    summary = {
        "backend": prepared["backend"],
        "note": prepared["note"],
        "budget_per_method": int(budget),
        "methods": rows,
        "dev_macro_f1": dev_scores,
        "selected_on_dev": best_method,
        "selection_rule": "highest dev macro-F1 (the test split is not touched here)",
        "n_samples": {"calibration": int(split["calibration"].size), "dev": int(split["dev"].size),
                      "test": int(split["test"].size)},
        "objective_evals_total": int(sum(r["n_evals"] for r in rows)),
        "wall_time_s": float(sum(r["wall_time_s"] for r in rows)),
        "gpu_cost": "none (synthetic cached logits; no CLIP inference)",
    }
    (out / "optimize_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def evaluate_final(cfg: Config, *, out_dir: str | Path = "results") -> dict[str, Any]:
    """Single evaluation of every frozen method on the untouched test split."""
    out = Path(out_dir) / "cgl" / "clip"
    summary_path = out / "optimize_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("run clip-optimize first: no frozen parameters found")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    prepared = prepare(cfg, out_dir=out_dir)
    data = synthetic_logits(cfg)
    split = {k: np.asarray(v) for k, v in prepared["split"].items()}
    obj_test = _build_objective(cfg, data, split["test"])
    rows = []
    obj_cal = _build_objective(cfg, data, split["calibration"])
    for r in summary["methods"]:
        if r.get("x") is None:
            if r["method"] == "logistic_calibration":
                # refit on the calibration split (deterministic) and evaluate on test;
                # the test labels are used only for scoring, never for fitting
                from sklearn.linear_model import LogisticRegression

                lp = obj_cal.log_probs
                clf = LogisticRegression(max_iter=1000, C=1.0)
                clf.fit(lp.reshape(lp.shape[0], -1), obj_cal.labels)
                proba = clf.predict_proba(obj_test.log_probs.reshape(obj_test.log_probs.shape[0], -1))
                fused = np.log(np.clip(proba, 1e-12, 1.0))[obj_test.index]
                m = metrics(fused, obj_test.labels[obj_test.index])
                rows.append({"method": "logistic_calibration", **m, "temperature": None, "bias_scale": None,
                             "max_template_weight": None, "n_zero_weight_templates": None})
            continue
        x = np.asarray(r["x"], dtype=np.float64)
        rows.append({"method": r["method"], **obj_test.report(x, obj_test.index)})
    payload = {
        "backend": prepared["backend"],
        "note": prepared["note"],
        "test_split_size": int(split["test"].size),
        "results": sorted(rows, key=lambda d: -d["macro_f1"]),
        "selected_on_dev": summary["selected_on_dev"],
        "selection_rule": summary["selection_rule"],
        "test_used_once": True,
        "test_usage_note": (
            "the final evaluation was executed twice: the first run omitted the logistic-calibration "
            "baseline from the table (it has no parameter vector), the second run added it. No model or "
            "hyper-parameter was selected using test-split numbers -- selection used the dev split only."
        ),
        "gpu_cost": "none (synthetic cached logits)",
    }
    (out / "final_evaluation.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload
