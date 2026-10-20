"""Online contribution graph: directed edges i -> j with realised-return weights.

Design (M1)
-----------
* ``n_ij``      : how many times particle ``i`` actually used ``j`` as its social
                  learning source,
* ``Q_ij``      : exponentially discounted return built **only** from the gain
                  that was actually realised on that iteration,
* ``last_used`` : iteration index of the most recent use (for recency/refresh),
* ``wins_ij``   : how many uses produced a positive gain (for reporting).

Honesty notes that are enforced by tests:
* only *selected* edges receive feedback -- there are no counterfactual branches
  and no extra fitness evaluations,
* feedback for iteration ``t`` may only use fitness values that already existed at
  ``t-1`` and ``t`` (temporal causality),
* ``Q_ij`` is an *observed association* between "learning from j" and the realised
  gain; it is not a strict causal contribution and is documented as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["ContributionGraph", "ContributionUpdate"]


@dataclass
class ContributionUpdate:
    """What one feedback round did (for logging/ablation analysis)."""

    n_updated: int = 0
    mean_gain: float = 0.0
    positive: int = 0
    negative: int = 0
    available: bool = True


class ContributionGraph:
    """Sparse directed edge statistics with neutral initialisation."""

    def __init__(
        self,
        n_particles: int,
        *,
        discount: float = 0.7,
        gain_clip: float = 1.0,
        init_value: float = 0.0,
    ) -> None:
        n = int(n_particles)
        self.n = n
        self.discount = float(discount)
        self.gain_clip = float(gain_clip)
        self.Q = np.full((n, n), float(init_value), dtype=np.float64)  # neutral start
        self.n_used = np.zeros((n, n), dtype=np.int64)
        self.wins = np.zeros((n, n), dtype=np.int64)
        self.last_used = np.full((n, n), -1, dtype=np.int64)
        self.updates = 0
        self.history: list[dict[str, Any]] = []
        # rows that were actually used at the previous iteration
        self._pending_src = np.zeros(0, dtype=np.int64)
        self._pending_dst = np.zeros(0, dtype=np.int64)
        self._pending_gain = np.zeros(0, dtype=np.float64)

    # ------------------------------------------------------------------
    def note_selection(self, src: np.ndarray, dst: np.ndarray, iteration: int) -> None:
        """Record which edges were used this iteration (no reward yet)."""
        self._pending_src = np.asarray(src, dtype=np.int64)
        self._pending_dst = np.asarray(dst, dtype=np.int64)
        self._pending_last_iter = int(iteration)

    def feedback(self, gain: np.ndarray, iteration: int) -> ContributionUpdate:
        """Apply the realised normalised gain of the *pending* selections.

        ``gain`` must have one entry per pending edge and must already be
        normalised/clipped by the caller. Returns a small report.
        """
        g = np.asarray(gain, dtype=np.float64)
        if self._pending_src.size == 0 or g.size != self._pending_src.size:
            return ContributionUpdate(n_updated=0, available=False)
        g = np.clip(np.nan_to_num(g, nan=0.0, posinf=self.gain_clip, neginf=-self.gain_clip),
                    -self.gain_clip, self.gain_clip)
        src, dst = self._pending_src, self._pending_dst
        self.Q[src, dst] = self.discount * self.Q[src, dst] + (1.0 - self.discount) * g
        self.n_used[src, dst] += 1
        self.wins[src, dst] += (g > 0).astype(np.int64)
        self.last_used[src, dst] = int(iteration)
        self.updates += 1
        self.history.append(
            {
                "iteration": int(iteration),
                "n_updated": int(src.size),
                "mean_gain": float(g.mean()),
                "positive": int((g > 0).sum()),
                "negative": int((g < 0).sum()),
            }
        )
        self._pending_src = np.zeros(0, dtype=np.int64)
        self._pending_dst = np.zeros(0, dtype=np.int64)
        return ContributionUpdate(
            n_updated=int(src.size), mean_gain=float(g.mean()),
            positive=int((g > 0).sum()), negative=int((g < 0).sum()),
        )

    # ------------------------------------------------------------------
    def score_matrix(self, scale: float = 1.0) -> np.ndarray:
        """Bounded contribution score in [-1, 1] (tanh of the discounted return)."""
        return np.tanh(self.Q / max(float(scale), 1e-12))

    def tried(self) -> np.ndarray:
        return self.n_used > 0

    def recency(self, iteration: int) -> np.ndarray:
        """Iterations since the last use (``inf`` for never-used edges)."""
        last = self.last_used.astype(np.float64)
        age = float(iteration) - last
        age[self.last_used < 0] = np.inf
        return age

    def unproductive(self, *, min_uses: int = 1, threshold: float = 0.0) -> np.ndarray:
        """Edges that were used but never produced a positive gain."""
        return (self.n_used >= int(min_uses)) & (self.Q <= float(threshold))

    # ------------------------------------------------------------------
    def stats(self) -> dict[str, float]:
        tried = self.tried()
        n_tried = int(tried.sum())
        q = self.Q[tried]
        n_offdiag = self.n * (self.n - 1)
        return {
            "edges_tried": n_tried,
            "edge_coverage": float(n_tried / max(n_offdiag, 1)),
            "mean_Q_tried": float(q.mean()) if n_tried else 0.0,
            "std_Q_tried": float(q.std()) if n_tried else 0.0,
            "positive_rate": float(self.wins[tried].sum() / max(self.n_used[tried].sum(), 1)) if n_tried else 0.0,
            "updates": int(self.updates),
            "mean_uses": float(self.n_used[tried].mean()) if n_tried else 0.0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"n_particles": self.n, "discount": self.discount, **self.stats()}

    def snapshot(self) -> dict[str, np.ndarray]:
        return {"Q": self.Q.copy(), "n_used": self.n_used.copy(), "wins": self.wins.copy(),
                "last_used": self.last_used.copy()}
