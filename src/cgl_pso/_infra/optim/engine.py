"""Batched particle-swarm engine shared by every variant.

Fairness contract
-----------------
All variants (classic PSO, ring-lbest, FIPS, CLPSO and the learned GLDI policies)
run through this single engine, so they share:

* the same initial population for a given seed (``init_population``),
* the same velocity/position update layout (only the social attractor differs),
* the same boundary handling, vmax rule and objective clipping,
* one central FE counter (``CountingObjective``): one evaluated row == one FE.

Only the *social information source* is swapped, which is exactly the ablation
surface of this paper.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from ..config import Config
from ..objectives.base import Bounds, CountingObjective

__all__ = [
    "OptimizerState",
    "OptimizerContext",
    "PolicyOutput",
    "ExemplarPolicy",
    "PSOEngine",
    "PSOResult",
    "init_population",
    "apply_boundary",
]


# --------------------------------------------------------------------------
# state & context
# --------------------------------------------------------------------------
@dataclass
class OptimizerState:
    pos: np.ndarray  # (N, D) current positions
    vel: np.ndarray  # (N, D) current velocities
    pbest_pos: np.ndarray  # (N, D) personal best positions
    pbest_val: np.ndarray  # (N,) personal best values
    fitness: np.ndarray  # (N,) fitness at ``pos``
    stall: np.ndarray  # (N,) iterations since last pbest improvement
    improve: np.ndarray  # (N,) EMA of relative pbest improvement
    iteration: int = 0
    fe: int = 0

    @property
    def n(self) -> int:
        return int(self.pos.shape[0])

    @property
    def dim(self) -> int:
        return int(self.pos.shape[1])

    @property
    def gbest_index(self) -> int:
        return int(np.argmin(self.pbest_val))

    @property
    def gbest_val(self) -> float:
        return float(self.pbest_val[self.gbest_index])

    @property
    def gbest_pos(self) -> np.ndarray:
        return self.pbest_pos[self.gbest_index]


@dataclass
class OptimizerContext:
    cfg: Config
    bounds: Bounds
    rng: np.random.Generator
    f_global: float
    objective_name: str
    device: str = "cpu"
    scalarize: Callable[[np.ndarray], np.ndarray] | None = None  # fitness -> rank-like score

    @property
    def dim(self) -> int:
        return self.bounds.dim

    @property
    def span(self) -> np.ndarray:
        return self.bounds.span


@dataclass
class PolicyOutput:
    """What a social-learning policy returns for one iteration.

    Either
      * ``exemplar`` (N, D): a single social attractor per particle (classic PSO,
        ring-lbest, random/rank baselines and GLDI-PSO), or
      * ``groups`` (N, K, D) with ``group_weights`` (N, K): several attractors
        per particle, used by FIPS.
    """

    exemplar: np.ndarray | None = None
    groups: np.ndarray | None = None
    group_weights: np.ndarray | None = None
    c1_scale: float = 1.0
    c2_scale: float = 1.0
    r2_random: bool = True
    aux: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.exemplar is None and self.groups is None:
            raise ValueError("PolicyOutput needs either 'exemplar' or 'groups'")
        if self.groups is not None and self.group_weights is None:
            k = self.groups.shape[1]
            self.group_weights = np.full(self.groups.shape[:2], 1.0 / k)


class ExemplarPolicy(Protocol):
    """Social-learning policy interface."""

    name: str

    def reset(self, state: OptimizerState, ctx: OptimizerContext) -> None: ...

    def select(self, state: OptimizerState, ctx: OptimizerContext) -> PolicyOutput: ...


# --------------------------------------------------------------------------
# population initialisation / boundary handling
# --------------------------------------------------------------------------
def init_population(
    n: int,
    bounds: Bounds,
    rng: np.random.Generator,
    *,
    mode: str = "uniform",
) -> np.ndarray:
    """Initial positions inside the box.

    ``uniform`` draws iid samples; ``lhs`` draws a Latin-hypercube design. Both
    are deterministic given ``rng``, so every algorithm compared in a benchmark
    cell receives byte-identical starting points.
    """
    n = int(n)
    low, high = bounds.low, bounds.high
    if mode == "uniform":
        return low + rng.random((n, bounds.dim)) * (high - low)
    if mode == "lhs":
        cut = np.linspace(0.0, 1.0, n + 1)
        u = rng.random((n, bounds.dim))
        u = cut[:-1, None] + u * (cut[1:] - cut[:-1])[:, None]
        for d in range(bounds.dim):
            rng.shuffle(u[:, d])
        return low + u * (high - low)
    raise ValueError(f"unknown init mode {mode!r}")


def apply_boundary(
    pos: np.ndarray,
    vel: np.ndarray,
    bounds: Bounds,
    mode: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Handle out-of-box positions; returns (pos, vel, n_violations).

    ``clip``   : clamp to the box, zero the offending velocity component.
    ``reflect``: mirror the overshoot back inside, flip and damp that component.
    ``random`` : resample the offending coordinate uniformly, zero its velocity.
    """
    low, high = bounds.low, bounds.high
    out = pos.copy()
    vout = vel.copy()
    below = pos < low
    above = pos > high
    n_viol = int(np.count_nonzero(below | above))
    if n_viol == 0:
        return out, vout, 0
    if mode == "clip":
        np.clip(pos, low, high, out=out)
        vout[below | above] = 0.0
    elif mode == "reflect":
        span = high - low
        reflected = np.where(below, low + (low - pos), high - (pos - high))
        # keep mirrored values inside even for huge overshoots
        reflected = low + np.mod(reflected - low, span)
        out = np.where(below | above, reflected, pos)
        vout[below | above] = -0.5 * vout[below | above]
    elif mode == "random":
        mask = below | above
        n_bad = int(np.count_nonzero(mask))
        if n_bad:
            # broadcast per-dimension bounds before boolean indexing
            low_b = np.broadcast_to(low, pos.shape)[mask]
            span_b = np.broadcast_to(high - low, pos.shape)[mask]
            out[mask] = low_b + rng.random(n_bad) * span_b
        vout[mask] = 0.0
    else:  # pragma: no cover - validated by config
        raise ValueError(f"unknown boundary mode {mode!r}")
    return out, vout, n_viol


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------
@dataclass
class PSOResult:
    algorithm: str
    objective: str
    dim: int
    seed: int
    fes: int
    iterations: int
    best_val: float
    best_pos: np.ndarray
    f_global: float
    runtime_s: float
    curve: list[tuple[int, float]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def error(self) -> float:
        return float(self.best_val - self.f_global)

    def to_row(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "objective": self.objective,
            "dim": self.dim,
            "seed": self.seed,
            "fes": self.fes,
            "iterations": self.iterations,
            "best_val": self.best_val,
            "error": self.error,
            "f_global": self.f_global,
            "runtime_s": self.runtime_s,
        }


class PSOEngine:
    """Canonical PSO engine with a pluggable social-learning policy."""

    def __init__(
        self,
        objective: CountingObjective,
        cfg: Config,
        policy: ExemplarPolicy,
        *,
        seed: int,
        device: str = "cpu",
    ) -> None:
        self.obj = objective
        self.cfg = cfg
        self.policy = policy
        self.seed = int(seed)
        self.device = device
        self.bounds = objective.bounds
        ocfg = cfg.get_path("optimizer", {})
        self.n = int(ocfg.get("n_particles", 40))
        self.w = float(ocfg.get("w", 0.7298))
        self.c1 = float(ocfg.get("c1", 1.49445))
        self.c2 = float(ocfg.get("c2", 1.49445))
        self.vmax_frac = float(ocfg.get("vmax_frac", 0.2))
        self.boundary = str(ocfg.get("boundary", "clip"))
        self.init_mode = str(ocfg.get("init", "uniform"))
        self.init_vel_frac = float(ocfg.get("init_vel_frac", 0.1))
        self.vmax = self.vmax_frac * self.bounds.span
        self.rng = np.random.default_rng(int(seed))
        self._init_rng = np.random.default_rng(int(seed))  # reserved for the initial population
        self.state: OptimizerState | None = None
        self.ctx = OptimizerContext(
            cfg=cfg,
            bounds=self.bounds,
            rng=self.rng,
            f_global=objective.f_global,
            objective_name=objective.name,
            device=device,
        )
        self.trace: list[dict[str, Any]] = []

    # -- setup ----------------------------------------------------------
    def initialize(self) -> OptimizerState:
        pos = init_population(self.n, self.bounds, self._init_rng, mode=self.init_mode)
        vel = (self.rng.random((self.n, self.bounds.dim)) - 0.5) * 2.0 * self.init_vel_frac * self.vmax
        fit = self.obj(pos)
        self.state = OptimizerState(
            pos=pos,
            vel=vel,
            pbest_pos=pos.copy(),
            pbest_val=fit.copy(),
            fitness=fit.copy(),
            stall=np.zeros(self.n),
            improve=np.zeros(self.n),
            iteration=0,
            fe=self.obj.n_evals,
        )
        self.policy.reset(self.state, self.ctx)
        return self.state

    # -- one iteration --------------------------------------------------
    def step(self) -> OptimizerState:
        if self.state is None:
            self.initialize()
        st = self.state
        assert st is not None
        out = self.policy.select(st, self.ctx)

        r1 = self.rng.random((st.n, st.dim))
        personal = self.c1 * out.c1_scale * r1 * (st.pbest_pos - st.pos)

        if out.exemplar is not None:
            attr = np.asarray(out.exemplar, dtype=np.float64)
            if attr.shape != st.pos.shape:
                raise ValueError(f"exemplar shape {attr.shape} != state shape {st.pos.shape}")
            r2 = self.rng.random((st.n, st.dim)) if out.r2_random else 1.0
            social = self.c2 * out.c2_scale * r2 * (attr - st.pos)
        else:
            groups = np.asarray(out.groups, dtype=np.float64)
            weights = np.asarray(out.group_weights, dtype=np.float64)
            if groups.ndim != 3 or groups.shape[0] != st.n or groups.shape[2] != st.dim:
                raise ValueError(f"groups shape {groups.shape} incompatible with {(st.n, st.dim)}")
            r = self.rng.random(groups.shape) if out.r2_random else 1.0
            social = self.c2 * out.c2_scale * np.sum(
                weights[:, :, None] * r * (groups - st.pos[:, None, :]), axis=1
            )

        vel = self.w * st.vel + personal + social
        np.clip(vel, -self.vmax, self.vmax, out=vel)
        new_pos = st.pos + vel
        new_pos, vel, n_viol = apply_boundary(new_pos, vel, self.bounds, self.boundary, self.rng)

        fit = self.obj(new_pos)
        improved = fit < st.pbest_val
        prev_pbest = st.pbest_val.copy()
        st.pbest_pos[improved] = new_pos[improved]
        st.pbest_val[improved] = fit[improved]
        st.stall = np.where(improved, 0.0, st.stall + 1.0)
        scale = np.maximum(np.abs(prev_pbest), 1e-12)
        rel = np.where(improved, (prev_pbest - st.pbest_val) / scale, 0.0)
        st.improve = 0.9 * st.improve + 0.1 * rel

        st.pos = new_pos
        st.vel = vel
        st.fitness = fit
        st.iteration += 1
        st.fe = self.obj.n_evals
        self.trace.append(
            {
                "iteration": st.iteration,
                "fe": st.fe,
                "gbest": float(st.pbest_val.min()),
                "mean_fitness": float(np.mean(fit)),
                "diversity": float(np.mean(np.std(st.pos, axis=0) / self.bounds.span)),
                "n_boundary_violations": int(n_viol),
                **{k: v for k, v in out.aux.items() if isinstance(v, (int, float, np.floating, np.integer))},
            }
        )
        return st

    # -- full run -------------------------------------------------------
    def run(
        self,
        *,
        iters: int | None = None,
        max_fes: int | None = None,
        curve_points: int | None = None,
        time_budget_s: float | None = None,
    ) -> PSOResult:
        t0 = time.perf_counter()
        if self.state is None:
            self.initialize()
        st = self.state
        assert st is not None
        total_iters = int(iters if iters is not None else self.cfg.get_path("optimizer.iters", 300))
        budget = int(max_fes) if max_fes is not None else self.obj.max_fes
        curve: list[tuple[int, float]] = [(int(st.fe), float(st.pbest_val.min()))]
        it = 0
        n_done = 0
        for it in range(total_iters):
            if budget is not None and st.fe + st.n > budget:
                break
            if time_budget_s is not None and (time.perf_counter() - t0) > time_budget_s:
                break
            self.step()
            n_done += 1
            curve.append((int(st.fe), float(st.pbest_val.min())))
        runtime = time.perf_counter() - t0
        best_pos = st.pbest_pos[st.gbest_index].copy()
        diag = {
            "n_particles": st.n,
            "diversity_final": float(np.mean(np.std(st.pos, axis=0) / self.bounds.span)),
            "mean_stall_final": float(np.mean(st.stall)),
            "policy": getattr(self.policy, "name", type(self.policy).__name__),
        }
        if curve_points and len(curve) > curve_points:
            idx = np.linspace(0, len(curve) - 1, curve_points).astype(int)
            curve = [curve[i] for i in idx]
        return PSOResult(
            algorithm=str(self.cfg.get_path("optimizer.kind", "pso")),
            objective=self.obj.name,
            dim=self.bounds.dim,
            seed=self.seed,
            fes=int(st.fe),
            iterations=int(n_done),
            best_val=float(st.pbest_val.min()),
            best_pos=best_pos,
            f_global=float(self.obj.f_global),
            runtime_s=float(runtime),
            curve=curve,
            diagnostics=diag,
        )
