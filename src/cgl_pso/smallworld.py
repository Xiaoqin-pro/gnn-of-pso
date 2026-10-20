"""Lightweight Ring-SmallWorld-PSO: ring topology + sparse non-local elite links.

Design (deliberately minimal -- no GNN, no PPO, no reward matrix, no sub-swarms):

* the **base** social attractor is the classic ring neighbourhood best (which, as in
  the classical ring-lbest policy, may be the particle itself);
* a particle may replace that attractor with a **non-local elite** (a particle from
  the global top-``q`` that is *outside* its ring neighbourhood) with a small
  probability ``p``;
* three ways to decide ``p``, all ablatable and all cheap:
    - ``fixed``      : constant ``p`` for every particle/iteration,
    - ``stall``      : the link is only allowed for particles stalled >= ``tau``,
    - ``bandit``     : a single probability is nudged online from the realised success
                       of the previous non-local links (stochastic approximation, no model);
* the standard PSO velocity update is untouched, and only one target is chosen per
  particle (identical FEs to every other algorithm).

Feedback reuse: the previous iteration's non-local attempts are scored with the
already-evaluated fitness/pbest at the *next* selection (the same causal pattern used
by CGL), so no extra objective calls are ever made.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cgl_pso._infra.config import Config
from cgl_pso._infra.optim.engine import OptimizerContext, OptimizerState, PolicyOutput
from cgl_pso._infra.seeding import stable_seed

__all__ = ["SmallWorldPolicy", "SW_MODES", "make_smallworld_policy"]

SW_MODES = ("ring", "fixed", "stall", "bandit")


@dataclass
class _PendingLink:
    used: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    fitness: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pbest: np.ndarray = field(default_factory=lambda: np.zeros(0))
    valid: bool = False


class SmallWorldPolicy:
    """Ring topology with sparse, optionally stagnation-aware non-local links."""

    def __init__(
        self,
        cfg: Config,
        *,
        mode: str = "stall",
        p: float = 0.05,
        stall_threshold: int = 10,
        ring_radius: int = 1,
        top_q: int = 5,
        bandit_eta: float = 0.05,
        p_min: float = 0.01,
        p_max: float = 0.5,
        success_margin: float = 0.0,
        seed: int = 0,
        record: bool = True,
    ) -> None:
        if mode not in SW_MODES:
            raise KeyError(f"unknown SmallWorld mode {mode!r}; known {SW_MODES}")
        self.cfg = cfg
        self.mode = mode
        self.name = "sw_ring" if mode == "ring" else f"sw_{mode}"
        self.p = float(p)
        self.p0 = float(p)
        self.stall_threshold = int(stall_threshold)
        self.ring_radius = int(ring_radius)
        self.top_q = int(top_q)
        self.bandit_eta = float(bandit_eta)
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.success_margin = float(success_margin)
        self.seed = int(seed)
        self.record = bool(record)
        self.rng = np.random.default_rng(stable_seed(seed, "smallworld"))
        self._pending = _PendingLink()
        self.trace: list[dict[str, float]] = []
        self._select_ms: list[float] = []
        self.p_history: list[float] = []

    # ------------------------------------------------------------------
    def reset(self, state: OptimizerState, ctx: OptimizerContext) -> None:
        self.rng = np.random.default_rng(stable_seed(self.seed, "smallworld"))
        self.p = self.p0
        self._pending = _PendingLink()
        self.trace = []
        self._select_ms = []
        self.p_history = []

    # ------------------------------------------------------------------
    def _ring_exemplar(self, state: OptimizerState) -> np.ndarray:
        idx = np.arange(state.n)
        offsets = np.arange(-self.ring_radius, self.ring_radius + 1)
        neigh = (idx[:, None] + offsets[None, :]) % state.n
        best = neigh[idx, np.argmin(state.pbest_val[neigh], axis=1)]
        return best

    def _nonlocal_elite(self, state: OptimizerState) -> np.ndarray:
        """Vectorised draw from the global top-q, excluding the particle's own ring.

        The per-particle Python loop was the dominant cost of the prototype
        (0.067 s vs ring's 0.031 s per run); this version draws for the whole swarm
        at once and repairs in-ring hits with a bounded number of redraws plus a
        deterministic fallback.
        """
        order = np.argsort(state.pbest_val, kind="stable")
        pool = order[: max(int(self.top_q), 1)]
        idx = np.arange(state.n)
        target = pool[self.rng.integers(0, pool.size, size=state.n)]
        r = int(self.ring_radius)
        if r > 0 and pool.size > 1:
            for _ in range(3):
                distance = np.abs(target - idx)
                in_ring = (distance <= r) | (distance >= state.n - r)
                if not in_ring.any():
                    break
                target[in_ring] = pool[self.rng.integers(0, pool.size, size=int(in_ring.sum()))]
            distance = np.abs(target - idx)
            still = (distance <= r) | (distance >= state.n - r)
            if still.any():
                # deterministic non-local fallback: exactly r+1 hops away on the ring
                target[still] = (idx[still] + r + 1) % state.n
        bad_self = target == idx
        if bad_self.any():
            target[bad_self] = (idx[bad_self] + max(r, 0) + 1) % state.n
        return target

    def _apply_feedback(self, state: OptimizerState) -> dict[str, float]:
        """Score last iteration's non-local links with already-evaluated values."""
        if not self._pending.valid or not self._pending.used.any():
            return {}
        used = self._pending.used
        improved = (self._pending.pbest - state.pbest_val) > self.success_margin
        success = float(np.mean(improved[used])) if used.any() else 0.0
        # A *relative* signal: comparing against the fixed 0.5 baseline makes p
        # collapse to zero (most particles do not improve their pbest every step).
        # Instead compare the non-local users with the particles that stayed local
        # in the same iteration, which is the actual question the bandit must answer.
        local = ~used
        success_local = float(np.mean(improved[local])) if local.any() else 0.0
        if self.mode == "bandit" and used.any() and local.any():
            self.p = float(np.clip(self.p + self.bandit_eta * (success - success_local), self.p_min, self.p_max))
        self._pending.valid = False
        return {
            "sw_nonlocal_success": success,
            "sw_local_success": success_local,
            "sw_nonlocal_trials": float(used.sum()),
        }

    # ------------------------------------------------------------------
    def select(self, state: OptimizerState, ctx: OptimizerContext) -> PolicyOutput:
        t0 = time.perf_counter()
        n = state.n
        feedback = self._apply_feedback(state)

        ring_target = self._ring_exemplar(state)
        target = ring_target.copy()
        used = np.zeros(n, dtype=bool)
        if self.mode != "ring":
            if self.mode == "fixed":
                allowed = np.ones(n, dtype=bool)
            elif self.mode == "stall":
                allowed = state.stall >= self.stall_threshold
            else:  # bandit: always eligible, probability is learned
                allowed = np.ones(n, dtype=bool)
            want = allowed & (self.rng.random(n) < self.p)
            if want.any():
                elite = self._nonlocal_elite(state)
                target = np.where(want, elite, target)
                used = want

        self._pending = _PendingLink(used=used.copy(), fitness=state.fitness.copy(),
                                     pbest=state.pbest_val.copy(), valid=True)
        aux = {
            "sw_nonlocal_share": float(used.mean()),
            "sw_p": float(self.p),
            "sw_stalled_share": float(np.mean(state.stall >= self.stall_threshold)),
            "sw_selected_global_best": float(np.mean(
                target == int(np.argmin(state.pbest_val))
            )),
            **feedback,
        }
        self.trace.append(aux)
        self.p_history.append(self.p)
        self._select_ms.append((time.perf_counter() - t0) * 1000.0)
        return PolicyOutput(exemplar=state.pbest_pos[target], aux=aux)

    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "p_final": float(self.p),
            "p_initial": float(self.p0),
            "stall_threshold": int(self.stall_threshold),
            "top_q": int(self.top_q),
            "mean_nonlocal_share": float(np.mean([t["sw_nonlocal_share"] for t in self.trace])) if self.trace else 0.0,
            "mean_select_ms": float(np.mean(self._select_ms)) if self._select_ms else 0.0,
        }


def make_smallworld_policy(kind: str, cfg: Config, *, seed: int = 0, record: bool = True, **kwargs: Any) -> SmallWorldPolicy:
    """Factory: ``sw_ring`` | ``sw_fixed`` | ``sw_stall`` | ``sw_bandit``."""
    mapping = {"sw_ring": "ring", "sw_fixed": "fixed", "sw_stall": "stall", "sw_bandit": "bandit"}
    if kind not in mapping:
        raise KeyError(f"unknown small-world policy {kind!r}; known {sorted(mapping)}")
    return SmallWorldPolicy(cfg, mode=mapping[kind], seed=seed, record=record, **kwargs)
