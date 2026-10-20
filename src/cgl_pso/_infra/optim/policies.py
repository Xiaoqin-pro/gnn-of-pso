"""Classical social-learning policies (the baselines) plus a policy registry.

Each policy only decides *which personal-best attractor* each particle should
follow this iteration; the velocity/position update stays in ``PSOEngine``.
That is what makes the comparison fair and the ablation well-scoped.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import Config
from .engine import ExemplarPolicy, OptimizerContext, OptimizerState, PolicyOutput

__all__ = [
    "GbestPolicy",
    "RingLbestPolicy",
    "FIPSPolicy",
    "CLPSOPolicy",
    "FitnessRankPolicy",
    "RandomNeighborPolicy",
    "make_classic_policy",
    "CLASSIC_POLICIES",
]


class _BasePolicy:
    name = "base"

    def reset(self, state: OptimizerState, ctx: OptimizerContext) -> None:  # pragma: no cover - trivial
        return None

    def select(self, state: OptimizerState, ctx: OptimizerContext) -> PolicyOutput:  # pragma: no cover - abstract
        raise NotImplementedError


class GbestPolicy(_BasePolicy):
    """Canonical gbest PSO: everybody follows the global best's pbest."""

    name = "gbest"

    def select(self, state: OptimizerState, ctx: OptimizerContext) -> PolicyOutput:
        gbest = state.pbest_pos[state.gbest_index]
        exemplar = np.broadcast_to(gbest, state.pos.shape).copy()
        return PolicyOutput(exemplar=exemplar, aux={"target_index": state.gbest_index})


class RingLbestPolicy(_BasePolicy):
    """Ring topology lbest PSO (radius ``ring_radius``, best in neighbourhood)."""

    name = "ring"

    def __init__(self, radius: int = 1):
        self.radius = int(radius)

    def select(self, state: OptimizerState, ctx: OptimizerContext) -> PolicyOutput:
        n = state.n
        r = self.radius
        idx = np.arange(n)
        # candidate window (n, 2r+1) on a ring
        offsets = np.arange(-r, r + 1)
        neigh = (idx[:, None] + offsets[None, :]) % n
        vals = state.pbest_val[neigh]
        best_local = neigh[np.arange(n), np.argmin(vals, axis=1)]
        return PolicyOutput(
            exemplar=state.pbest_pos[best_local],
            aux={"target_index": int(best_local.mean())},
        )


class FIPSPolicy(_BasePolicy):
    """Fully informed PSO (Mendes et al., 2004) with normalised weights.

    Each particle is informed by its ring neighbours (radius ``fips_neighbors``);
    neighbour attractors are averaged with weights that sum to 1, and everyone
    keeps its own cognitive term.

    ``fips_weights='uniform'`` (default) is the equal-weight formulation,
    ``'rank'`` replaces it by fitness-rank weights. The published normalisation
    of Mendes et al. (2004) is re-derived here as documented; the deviation is
    flagged as an approximation in docs/literature.md and LIMITATIONS.md.
    """

    name = "fips"

    def __init__(self, radius: int = 2, weights: str = "uniform"):
        self.radius = int(radius)
        self.weights = weights

    def select(self, state: OptimizerState, ctx: OptimizerContext) -> PolicyOutput:
        n = state.n
        r = self.radius
        idx = np.arange(n)
        offsets = np.arange(-r, r + 1)
        neigh = (idx[:, None] + offsets[None, :]) % n  # (n, k)
        groups = state.pbest_pos[neigh]  # (n, k, D)
        k = neigh.shape[1]
        if self.weights == "uniform":
            w = np.full((n, k), 1.0 / k)
        else:
            vals = state.pbest_val[neigh]
            ranks = np.argsort(np.argsort(vals, axis=1), axis=1).astype(np.float64)
            raw = (k - ranks)  # best neighbour gets weight k
            w = raw / raw.sum(axis=1, keepdims=True)
        return PolicyOutput(groups=groups, group_weights=w, aux={"policy_radius": r})


class CLPSOPolicy(_BasePolicy):
    """CLPSO (Liang et al., 2006): per-dimension exemplars + refresh gap.

    Implementation notes
    --------------------
    * Each dimension of each particle has its own exemplar particle, drawn from
      the better of two random particles (canonical ``f_i(d)`` rule).
    * A particle refreshes its exemplar dimensions once its personal best has
      not improved for ``clpso_refresh_gap`` iterations, dimension by dimension
      with probability ``clpso_pc``.
    * The cognitive term is disabled (``c1_scale=0``) and the social
      acceleration is set to the published 1.49405 rather than the engine's
      default, using a coefficient scale.
    """

    name = "clpso"

    def __init__(self, pc: float = 0.5, refresh_gap: int = 7):
        self.pc = float(pc)
        self.refresh_gap = int(refresh_gap)
        self._exemplar_idx: np.ndarray | None = None

    def reset(self, state: OptimizerState, ctx: OptimizerContext) -> None:
        n, d = state.n, state.dim
        rng = ctx.rng
        a = rng.integers(0, n, size=(n, d))
        b = rng.integers(0, n, size=(n, d))
        better = np.where(state.pbest_val[a] < state.pbest_val[b], a, b)
        same = better == np.arange(n)[:, None]
        better[same] = np.where(same, b, better)[same] if same.any() else better
        self._exemplar_idx = better

    def select(self, state: OptimizerState, ctx: OptimizerContext) -> PolicyOutput:
        n, d = state.n, state.dim
        if self._exemplar_idx is None:
            self.reset(state, ctx)
        assert self._exemplar_idx is not None
        idx = self._exemplar_idx
        # refresh dimensions for stalled particles
        stalled = state.stall >= self.refresh_gap
        if np.any(stalled):
            draw = ctx.rng.random((n, d)) < self.pc
            a = ctx.rng.integers(0, n, size=(n, d))
            b = ctx.rng.integers(0, n, size=(n, d))
            better = np.where(state.pbest_val[a] < state.pbest_val[b], a, b)
            mask = stalled[:, None] & draw
            idx = np.where(mask, better, idx)
            self._exemplar_idx = idx
        bad = idx == np.arange(n)[:, None]
        if np.any(bad):  # never let a particle inform itself
            idx = np.where(bad, (idx + 1) % n, idx)
            self._exemplar_idx = idx
        exemplar = np.take_along_axis(state.pbest_pos, idx, axis=0)
        engine_c2 = float(ctx.cfg.get_path("optimizer.c2", 1.49445))
        return PolicyOutput(
            exemplar=exemplar,
            c1_scale=0.0,
            c2_scale=(1.49405 / engine_c2) if engine_c2 > 0 else 1.0,
            aux={"clpso_refresh_gap": self.refresh_gap},
        )


class FitnessRankPolicy(_BasePolicy):
    """Rank-roulette baseline: probability of being chosen grows with pbest rank."""

    name = "rank"

    def select(self, state: OptimizerState, ctx: OptimizerContext) -> PolicyOutput:
        n = state.n
        ranks = np.argsort(np.argsort(state.pbest_val))  # 0 == best
        weights = (n - ranks).astype(np.float64)
        weights[ranks == 0] += n  # keep global best competitive but not exclusive
        probs = weights / weights.sum()
        picks = ctx.rng.choice(n, size=n, p=probs)
        same = picks == np.arange(n)
        if np.any(same):
            picks = np.where(same, (picks + 1) % n, picks)
        return PolicyOutput(
            exemplar=state.pbest_pos[picks],
            aux={"rank_pick_mean_rank": float(ranks[picks].mean())},
        )


class RandomNeighborPolicy(_BasePolicy):
    """Uniform random social neighbour (no learning, no fitness information)."""

    name = "random"

    def select(self, state: OptimizerState, ctx: OptimizerContext) -> PolicyOutput:
        n = state.n
        picks = ctx.rng.integers(0, n, size=n)
        same = picks == np.arange(n)
        if np.any(same):
            picks = np.where(same, (picks + 1) % n, picks)
        return PolicyOutput(exemplar=state.pbest_pos[picks])


CLASSIC_POLICIES = {
    "gbest": GbestPolicy,
    "ring": RingLbestPolicy,
    "fips": FIPSPolicy,
    "clpso": CLPSOPolicy,
    "rank": FitnessRankPolicy,
    "random": RandomNeighborPolicy,
}


def make_classic_policy(kind: str, cfg: Config) -> ExemplarPolicy:
    key = kind.lower()
    if key not in CLASSIC_POLICIES:
        raise KeyError(f"unknown classic policy {kind!r}; known: {sorted(CLASSIC_POLICIES)}")
    ocfg: dict[str, Any] = cfg.get_path("optimizer", {})
    if key == "gbest":
        return GbestPolicy()
    if key == "ring":
        return RingLbestPolicy(radius=int(ocfg.get("ring_radius", 1)))
    if key == "fips":
        return FIPSPolicy(
            radius=int(ocfg.get("fips_neighbors", 2)),
            weights=str(ocfg.get("fips_weights", "half")),
        )
    if key == "clpso":
        return CLPSOPolicy(
            pc=float(ocfg.get("clpso_pc", 0.5)),
            refresh_gap=int(ocfg.get("clpso_refresh_gap", 7)),
        )
    return CLASSIC_POLICIES[key]()
