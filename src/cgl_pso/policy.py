"""CGL-PSO policy: online contribution graph + quality/direction scoring + refresh.

Only the *social attractor source* changes; inertia, cognitive term, coefficients,
velocity clamping and boundary handling stay in the verified ``PSOEngine``.

Ablation variants (each runs through the same engine, candidate builder and FEs):

===============  =============================================================
quality_rank     softmax over quality only (same candidate set, no learning)
cgl_nocontrib    contribution weight β = 0  (M1 removed)
cgl_nodirection  direction weight γ = 0     (M2's direction part removed)
cgl_norefresh    M3 disabled
cgl_full         all three mechanisms
===============  =============================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cgl_pso._infra.config import Config
from cgl_pso._infra.optim.engine import OptimizerContext, OptimizerState, PolicyOutput
from cgl_pso._infra.seeding import stable_seed

from .candidates import CandidateSet, build_candidates
from .contribution import ContributionGraph

__all__ = ["CGLPolicy", "CGL_POLICIES", "make_cgl_policy"]

CGL_POLICIES = ("quality_rank", "cgl_nocontrib", "cgl_nodirection", "cgl_norefresh", "cgl_full")


@dataclass
class _Pending:
    """State stored at iteration t so that feedback at t+1 stays causal."""

    fitness: np.ndarray
    pbest_val: np.ndarray
    scale: float
    src: np.ndarray
    dst: np.ndarray
    valid: bool = False
    extras: dict[str, Any] = field(default_factory=dict)


class CGLPolicy:
    """Contribution-guided social-source selection (M1 + M2 + M3)."""

    def __init__(
        self,
        cfg: Config,
        *,
        variant: str = "cgl_full",
        seed: int = 0,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        novelty: float = 0.25,
        temperature: float = 0.5,
        epsilon: float = 0.15,
        epsilon_stalled: float = 0.4,
        stall_threshold: int = 12,
        refresh_penalty: float = 1.0,
        gain_discount: float = 0.7,
        gain_weights: tuple[float, float, float] = (0.6, 0.3, 0.1),
        gain_common_mode_removal: bool = True,
        ring_radius: int = 1,
        top_q: int = 4,
        n_random: int = 2,
        n_far_samples: int = 8,
        candidate_sources: dict[str, bool] | None = None,
        record: bool = True,
    ) -> None:
        if variant not in CGL_POLICIES:
            raise KeyError(f"unknown CGL variant {variant!r}; known {sorted(CGL_POLICIES)}")
        self.cfg = cfg
        self.variant = variant
        self.name = f"cgl_{variant}" if variant != "cgl_full" else "cgl_full"
        self.seed = int(seed)
        self.rng = np.random.default_rng(stable_seed(seed, "cgl-policy"))

        # mechanism switches (the variant only changes these numbers)
        self.alpha = float(alpha if alpha is not None else 1.0)
        self.beta = float(beta if beta is not None else (0.0 if variant in {"quality_rank", "cgl_nocontrib"} else 1.0))
        self.gamma = float(gamma if gamma is not None else (0.0 if variant in {"quality_rank", "cgl_nodirection"} else 0.6))
        self.novelty = float(0.0 if variant == "quality_rank" else novelty)
        self.use_refresh = variant not in {"quality_rank", "cgl_norefresh"}
        self.temperature = float(temperature)
        self.epsilon = float(epsilon)
        self.epsilon_stalled = float(epsilon_stalled)
        self.stall_threshold = int(stall_threshold)
        self.refresh_penalty = float(refresh_penalty)
        self.gain_discount = float(gain_discount)
        self.gain_weights = tuple(float(w) for w in gain_weights)
        # Pilot-phase diagnosis: the per-iteration improvements of all particles
        # share a large common component (the swarm moves together), so the raw
        # edge gain is dominated by a constant plus noise. Removing the swarm mean
        # leaves the *relative* improvement, which is what an edge comparison needs.
        self.gain_common_mode_removal = bool(gain_common_mode_removal)
        self.ring_radius = int(ring_radius)
        self.top_q = int(top_q)
        self.n_random = int(n_random)
        self.n_far_samples = int(n_far_samples)
        # Repair round (documented failure reason): the far/random sources pull
        # particles towards distant, poor targets and destroy the exploitation the
        # classical ring topology provides; disabling them is a single-factor change.
        self.candidate_sources = dict(candidate_sources or {})
        self.record = bool(record)

        self.graph: ContributionGraph | None = None
        self._pending: _Pending = _Pending(np.zeros(0), np.zeros(0), 1.0, np.zeros(0, dtype=np.int64),
                                           np.zeros(0, dtype=np.int64))
        self._refresh_events: list[dict[str, Any]] = []
        self._update_ms: list[float] = []
        self._selection_entropy: list[float] = []
        self.iteration_trace: list[dict[str, float]] = []
        self._cached_scale: float = 1.0
        self.scale_update_period: int = 10   # the swarm's robust scale moves slowly

    # ------------------------------------------------------------------
    def reset(self, state: OptimizerState, ctx: OptimizerContext) -> None:
        self.graph = ContributionGraph(state.n, discount=self.gain_discount)
        self._pending = _Pending(state.fitness.copy(), state.pbest_val.copy(), 1.0,
                                 np.zeros(state.n, dtype=np.int64), np.zeros(state.n, dtype=np.int64))
        self._refresh_events = []
        self._update_ms = []
        self._selection_entropy = []

    # ------------------------------------------------------------------
    @staticmethod
    def _robust_scale(values: np.ndarray) -> float:
        if values.size == 0:
            return 1.0
        q75, q25 = np.percentile(values, [75, 25])
        scale = float(q75 - q25)
        if scale <= 1e-12:
            scale = float(np.std(values))
        return scale if scale > 1e-12 else 1.0

    def _scale_for_iteration(self, state: OptimizerState) -> float:
        """Robust swarm scale, recomputed every ``scale_update_period`` iterations.

        The IQR of the swarm fitness changes slowly compared with one iteration, so
        recomputing it every step only burns ~0.08 ms; the cached value is
        statistically equivalent and was validated by the pilot results.
        """
        if state.iteration % max(self.scale_update_period, 1) == 0 or self._cached_scale <= 0:
            self._cached_scale = self._robust_scale(state.fitness)
        return self._cached_scale

    def _apply_feedback(self, state: OptimizerState) -> dict[str, float]:
        """Update the graph from the *realised* step between t-1 and t (causal)."""
        assert self.graph is not None
        if not self._pending.valid:
            return {}
        w_f, w_p, w_g = self.gain_weights
        scale = max(self._pending.scale, 1e-12)
        d_fit = self._pending.fitness - state.fitness
        d_pbest = self._pending.pbest_val - state.pbest_val
        if self.gain_common_mode_removal and d_fit.size > 1:
            # relative to what the rest of the swarm achieved in the same iteration
            d_fit = d_fit - float(np.mean(d_fit))
            d_pbest = d_pbest - float(np.mean(d_pbest))
        global_gain = float(np.max(self._pending.pbest_val) - np.max(state.pbest_val))
        gain = (w_f * d_fit + w_p * d_pbest + w_g * global_gain) / scale
        report = self.graph.feedback(gain, iteration=state.iteration)
        self._pending.valid = False
        return {"feedback_mean_gain": report.mean_gain, "feedback_n": report.n_updated,
                "feedback_positive": report.positive, "feedback_negative": report.negative}

    # ------------------------------------------------------------------
    def _score_candidates(
        self,
        state: OptimizerState,
        candidates: CandidateSet,
        *,
        stalled: np.ndarray,
        ranks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """Bounded score per candidate slot plus the per-particle exploration floor."""
        assert self.graph is not None
        n, k = candidates.idx.shape
        mask = candidates.mask

        quality = 1.0 - ranks[candidates.idx]                        # (N, K) in [0, 1]
        # Contribution term: standardised *within each particle's candidate set*.
        # Raw Q values are observed to be ~1e-2 while quality spans [0, 1], so an
        # unstandardised term is simply drowned (a real pilot-phase diagnosis).
        # Standardising makes the term scale-free and expresses "which of MY
        # candidates has historically paid off", which is the selection question.
        q_raw = self.graph.Q[np.arange(n)[:, None], candidates.idx]   # (N, K)
        q_valid = np.where(mask, q_raw, np.nan)
        q_mean = np.nanmean(q_valid, axis=1, keepdims=True)
        q_std = np.nanstd(q_valid, axis=1, keepdims=True)
        q_z = (q_raw - q_mean) / np.maximum(q_std, 1e-6)
        q_term = np.where(mask, np.tanh(np.nan_to_num(q_z, nan=0.0)), 0.0)   # (N, K) in ~[-1, 1]

        own_dir = state.pbest_pos - state.pos                        # (N, D)
        target_dir = state.pbest_pos[candidates.idx] - state.pos[:, None, :]  # (N, K, D)
        norm_own = np.linalg.norm(own_dir, axis=1)[:, None]          # (N, 1)
        norm_tgt = np.linalg.norm(target_dir, axis=2)                # (N, K)
        denom = np.maximum(norm_own * norm_tgt, 1e-12)
        direction = np.sum(target_dir * own_dir[:, None, :], axis=2) / denom
        direction = np.where(np.isfinite(direction), direction, 0.0)   # degenerate -> 0
        direction = np.clip(direction, -1.0, 1.0)

        tried = self.graph.n_used[np.arange(n)[:, None], candidates.idx]
        novelty = 1.0 / (1.0 + tried)

        score = (self.alpha * quality + self.beta * q_term + self.gamma * direction + self.novelty * novelty)
        score = np.where(mask, score, -np.inf)

        eps = np.full(n, self.epsilon)
        if self.use_refresh:
            unproductive = self.graph.unproductive()
            recent = self.graph.recency(state.iteration) <= max(self.stall_threshold, 1)
            bad_edge = unproductive[np.arange(n)[:, None], candidates.idx] & recent[np.arange(n)[:, None], candidates.idx]
            penalty = np.where(bad_edge, self.refresh_penalty, 0.0)
            score = score - penalty
            eps = np.where(stalled, self.epsilon_stalled, eps)
        # diagnostics that make the mechanism auditable: how much spread the
        # contribution term has inside a candidate set, and how often it changes
        # the *decision* relative to a quality-only scorer (the honest question).
        contrib_std = float(np.mean(np.nanstd(np.where(mask, q_term, np.nan), axis=1)))
        if self.beta > 0:
            masked = np.where(mask, score, -np.inf)
            quality_only = np.where(mask, self.alpha * quality, -np.inf)
            flip = float(np.mean(np.argmax(masked, axis=1) != np.argmax(quality_only, axis=1)))
        else:
            flip = 0.0
        diag = {
            "mean_direction": float(direction[mask].mean()) if mask.any() else 0.0,
            "mean_contrib_term": float(q_term[mask].mean()) if mask.any() else 0.0,
            "contrib_term_std": contrib_std,
            "contrib_decision_flip_rate": flip,
            "mean_quality_term": float(quality[mask].mean()) if mask.any() else 0.0,
            "mean_epsilon": float(eps.mean()),
        }
        return score, eps, diag

    @staticmethod
    def _sample(score: np.ndarray, eps: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                temperature: float) -> tuple[np.ndarray, np.ndarray]:
        n, k = score.shape
        stable = score - np.max(np.where(mask, score, -np.inf), axis=1, keepdims=True)
        exp = np.where(mask, np.exp(stable / max(temperature, 1e-6)), 0.0)
        p = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-300)
        uniform = mask / np.maximum(mask.sum(axis=1, keepdims=True), 1)
        probs = (1.0 - eps[:, None]) * p + eps[:, None] * uniform
        probs = np.where(mask, probs, 0.0)
        probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-300)
        # inverse-CDF sampling keeps the policy RNG isolated from numpy's global state
        u = rng.random((n, 1))
        cdf = np.cumsum(probs, axis=1)
        slot = np.argmax(cdf >= u, axis=1).astype(np.int64)
        return slot, probs

    # ------------------------------------------------------------------
    def select(self, state: OptimizerState, ctx: OptimizerContext) -> PolicyOutput:
        if self.graph is None:
            self.reset(state, ctx)
        assert self.graph is not None
        t0 = time.perf_counter()

        feedback = self._apply_feedback(state)

        stalled = state.stall >= self.stall_threshold
        order = np.argsort(state.pbest_val, kind="stable")
        rank_of = np.empty(state.n, dtype=np.float64)
        rank_of[order] = np.arange(state.n)
        ranks = rank_of / max(state.n - 1, 1)
        candidates = build_candidates(
            state.pos, state.pbest_val, ctx.bounds, self.rng,
            ring_radius=self.ring_radius, top_q=self.top_q,
            n_random=self.n_random, n_far_samples=self.n_far_samples,
            pbest_order=order,
            use_source_flags=self.candidate_sources or None,
        )
        score, eps, diag = self._score_candidates(state, candidates, stalled=stalled, ranks=ranks)
        slot, probs = self._sample(score, eps, candidates.mask, self.rng, self.temperature)
        target = candidates.idx[np.arange(state.n), slot]

        self.graph.note_selection(np.arange(state.n), target, state.iteration)
        self._pending = _Pending(
            fitness=state.fitness.copy(),
            pbest_val=state.pbest_val.copy(),
            scale=self._scale_for_iteration(state),
            src=np.arange(state.n, dtype=np.int64),
            dst=target.copy(),
            valid=True,
        )
        update_ms = (time.perf_counter() - t0) * 1000.0
        self._update_ms.append(update_ms)

        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -(np.where(probs > 0, probs * np.log(np.maximum(probs, 1e-300)), 0.0)).sum(axis=1)
        mean_entropy = float(entropy.mean())
        self._selection_entropy.append(mean_entropy)
        if self.use_refresh and bool(stalled.any()):
            self._refresh_events.append({"iteration": int(state.iteration), "n_stalled": int(stalled.sum())})

        stats = self.graph.stats()
        aux = {
            "cgl_edges_tried": float(stats["edges_tried"]),
            "cgl_edge_coverage": float(stats["edge_coverage"]),
            "cgl_mean_Q": float(stats["mean_Q_tried"]),
            "cgl_positive_rate": float(stats["positive_rate"]),
            "cgl_updates": float(stats["updates"]),
            "cgl_stalled": float(stalled.mean()),
            "cgl_refresh_events": float(len(self._refresh_events)),
            "cgl_select_ms": float(update_ms),
            "cgl_selection_entropy": mean_entropy,
            **diag,
            **feedback,
        }
        self.iteration_trace.append(aux)
        return PolicyOutput(exemplar=state.pbest_pos[target], aux=aux)

    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        assert self.graph is not None
        return {
            "variant": self.variant,
            "candidate_sources": self.candidate_sources,
            "alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "novelty": self.novelty,
            "use_refresh": self.use_refresh,
            "graph": self.graph.stats(),
            "refresh_events": len(self._refresh_events),
            "mean_select_ms": float(np.mean(self._update_ms)) if self._update_ms else 0.0,
            "p95_select_ms": float(np.percentile(self._update_ms, 95)) if self._update_ms else 0.0,
            "mean_selection_entropy": float(np.mean(self._selection_entropy)) if self._selection_entropy else 0.0,
        }


def make_cgl_policy(kind: str, cfg: Config, *, seed: int = 0, record: bool = True, **kwargs: Any) -> Any:
    """Factory: classic baselines (delegated to the verified engine) or CGL variants."""
    from cgl_pso._infra.optim.policies import make_classic_policy

    key = kind.lower()
    if key in {"ring", "gbest", "rank", "random", "fips", "clpso"}:
        return make_classic_policy(key, cfg)
    if key == "gldi_rule":
        # the historical reference rule lives in this export as a self-contained
        # baseline (the parent repository's implementation is behaviourally identical)
        from .baselines import RuleReferencePolicy

        return RuleReferencePolicy(cfg, seed=stable_seed(seed, "rule"))
    if key not in CGL_POLICIES:
        raise KeyError(f"unknown CGL policy {kind!r}; known {sorted(CGL_POLICIES)}")
    return CGLPolicy(cfg, variant=key, seed=seed, record=record, **kwargs)
