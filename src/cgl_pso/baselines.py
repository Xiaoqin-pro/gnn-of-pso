"""Reference baselines that are not CGL mechanisms.

``RuleReferencePolicy`` reproduces the fixed-score rule baseline that the CGL
development screening used as a historical reference point (``gldi_rule``):
score = w1 * (-pbest rank) + w2 * cos(own pbest direction, candidate direction)
        + w3 * (1 - relative distance), argmax per particle over its candidate graph.

It consumes the *same* candidate-graph builder as CGL and therefore isolates
"learned/feedback-driven selection" from "candidate set membership". The rule uses
no feedback and no CGL mechanism.
"""

from __future__ import annotations

import numpy as np

from ._infra.config import Config
from ._infra.graph.candidates import GraphBuilder
from ._infra.optim.engine import OptimizerContext, OptimizerState, PolicyOutput
from ._infra.seeding import stable_seed

__all__ = ["RuleReferencePolicy"]


class RuleReferencePolicy:
    """Fixed-weight quality + direction + proximity rule over the candidate graph."""

    name = "gldi_rule"

    def __init__(self, cfg: Config, *, seed: int = 0, weights: tuple[float, float, float] = (1.0, 0.35, 0.15), **_: object) -> None:
        self.cfg = cfg
        self.kind = "gldi_rule"
        self.seed = int(seed)
        self.weights = tuple(float(w) for w in weights)
        self.builder: GraphBuilder | None = None
        self.graph_stats: list[dict[str, float]] = []

    def reset(self, state: OptimizerState, ctx: OptimizerContext) -> None:
        self.builder = GraphBuilder(self.cfg, stable_seed(self.seed, "graph"))
        self.graph_stats = []

    def select(self, state: OptimizerState, ctx: OptimizerContext) -> PolicyOutput:
        if self.builder is None:
            self.reset(state, ctx)
        assert self.builder is not None
        graph = self.builder.get(state.pos, state.pbest_val, ctx.bounds.span, state.iteration)
        src, dst = graph.edge_index[0], graph.edge_index[1]
        n = state.n
        rank = np.argsort(np.argsort(state.pbest_val)) / max(n - 1, 1)
        own_dir = state.pbest_pos[src] - state.pos[src]
        cand_dir = state.pbest_pos[dst] - state.pos[src]
        cos = np.sum(cand_dir * own_dir, axis=1) / np.maximum(
            np.linalg.norm(cand_dir, axis=1) * np.linalg.norm(own_dir, axis=1), 1e-12
        )
        span = ctx.bounds.span
        dist = np.linalg.norm((state.pos[dst] - state.pos[src]) / span[None, :], axis=1)
        deg = np.bincount(src, minlength=n).astype(np.float64)
        mean_dist = np.bincount(src, weights=dist, minlength=n) / np.maximum(deg, 1.0)
        rel = dist / np.maximum(mean_dist[src], 1e-12) - 1.0
        w1, w2, w3 = self.weights
        score = w1 * (-rank[dst]) + w2 * cos + w3 * (1.0 - rel)

        target = np.empty(n, dtype=np.int64)
        for i in range(n):
            rows = np.flatnonzero(src == i)
            target[i] = dst[rows[int(np.argmax(score[rows]))]]
        self.graph_stats.append({"n_edges": float(graph.n_edges), "graph_builds": float(self.builder.n_builds)})
        return PolicyOutput(exemplar=state.pbest_pos[target], aux={"n_edges": float(graph.n_edges)})
