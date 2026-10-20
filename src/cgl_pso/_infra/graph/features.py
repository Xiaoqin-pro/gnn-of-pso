"""Scale-free node and edge features for GLDI-PSO.

Every feature is (a) dimensionless, (b) invariant to an affine rescaling of the
objective, and (c) permutation equivariant, so the learned policy transfers
across dimensions, swarm sizes and objective scales. This is what the unit tests
assert; it is also the mechanism behind generalisation to unseen functions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import Config
from ..optim.engine import OptimizerContext, OptimizerState

__all__ = [
    "NODE_FEATURE_NAMES",
    "EDGE_FEATURE_NAMES",
    "node_features",
    "edge_features",
    "all_features",
    "HISTORY_NODE_INDICES",
    "N_NODE_FEATURES",
    "N_EDGE_FEATURES",
]


NODE_FEATURE_NAMES = (
    "fitness_rank",
    "pbest_rank",
    "improvement_ema",
    "stagnation",
    "normalised_speed",
    "diversity_ratio",
)

EDGE_FEATURE_NAMES = (
    "normalised_distance",
    "relative_distance",
    "fitness_gap",
    "pbest_gap",
    "cos_to_own_pbest",
    "cos_to_velocity",
)

# node feature columns that encode the particle's own history
HISTORY_NODE_INDICES = (2, 3)

N_NODE_FEATURES = len(NODE_FEATURE_NAMES)
N_EDGE_FEATURES = len(EDGE_FEATURE_NAMES)

_EPS = 1e-12


def _rank01(values: np.ndarray) -> np.ndarray:
    """Normalised rank in [0, 1]; 0 == best (smallest value)."""
    n = values.size
    order = np.argsort(np.argsort(values, kind="stable"))
    return order.astype(np.float64) / max(n - 1, 1)


def _robust_scale(values: np.ndarray) -> float:
    """Affine-invariant scale of a fitness vector."""
    if values.size < 2:
        return 1.0
    q75, q25 = np.percentile(values, [75, 25])
    iqr = float(q75 - q25)
    if iqr > _EPS:
        return iqr
    std = float(np.std(values))
    return std if std > _EPS else 1.0


def node_features(state: OptimizerState, ctx: OptimizerContext, cfg: Config | None = None) -> np.ndarray:
    """(N, N_NODE_FEATURES) node feature matrix."""
    cfg = cfg or ctx.cfg
    n, d = state.n, state.dim
    span = ctx.bounds.span
    vmax = np.maximum(float(cfg.get_path("optimizer.vmax_frac", 0.2)) * span, _EPS)

    fit_rank = _rank01(state.fitness)
    pbest_rank = _rank01(state.pbest_val)

    improve = np.tanh(state.improve)

    stall_window = float(cfg.get_path("optimizer.stall_window", 10.0)) or 10.0
    stagnation = np.clip(state.stall / stall_window, 0.0, 1.0)

    # normalised speed: mean |v| / vmax per dimension -> O(1) for any D and box
    speed = np.mean(np.abs(state.vel) / vmax[None, :], axis=1)
    speed = np.clip(speed, 0.0, 2.0) / 2.0

    # diversity: own mean distance to the swarm vs the swarm-average distance
    delta = (state.pos[:, None, :] - state.pos[None, :, :]) / span[None, None, :]
    dist = np.sqrt(np.sum(delta**2, axis=-1))
    np.fill_diagonal(dist, np.nan)
    mean_dist = np.nanmean(dist, axis=1)
    overall = float(np.nanmean(mean_dist))
    diversity = np.tanh(mean_dist / max(overall, _EPS) - 1.0)

    feats = np.stack([fit_rank, pbest_rank, improve, stagnation, speed, diversity], axis=1)
    if not np.all(np.isfinite(feats)):
        feats = np.nan_to_num(feats, nan=0.0, posinf=2.0, neginf=-2.0)
    return feats.astype(np.float64)


def edge_features(
    state: OptimizerState,
    edge_index: np.ndarray,
    ctx: OptimizerContext,
    cfg: Config | None = None,
) -> np.ndarray:
    """(E, N_EDGE_FEATURES) edge feature matrix for ``edge_index`` (2, E)."""
    span = ctx.bounds.span
    src, dst = edge_index[0], edge_index[1]

    unit = state.pos / span[None, :]
    delta = unit[dst] - unit[src]
    dist = np.sqrt(np.sum(delta**2, axis=1))

    # per-source relative distance
    deg = np.bincount(src, minlength=state.n).astype(np.float64)
    sum_dist = np.bincount(src, weights=dist, minlength=state.n)
    mean_dist = sum_dist / np.maximum(deg, 1.0)
    rel_dist = dist / np.maximum(mean_dist[src], _EPS) - 1.0

    scale = _robust_scale(state.pbest_val)
    fit_gap = np.tanh((state.fitness[dst] - state.fitness[src]) / scale)
    pbest_gap = np.tanh((state.pbest_val[dst] - state.pbest_val[src]) / scale)

    own_direction = state.pbest_pos[src] - state.pos[src]
    cand_direction = state.pbest_pos[dst] - state.pos[src]
    cos_pbest = _cosine(cand_direction, own_direction)
    cos_vel = _cosine(cand_direction, state.vel[src])

    # dimension-independent distance: ||delta_unit|| / sqrt(D)
    norm_dist = dist / np.sqrt(state.dim)

    feats = np.stack([norm_dist, rel_dist, fit_gap, pbest_gap, cos_pbest, cos_vel], axis=1)
    if not np.all(np.isfinite(feats)):
        feats = np.nan_to_num(feats, nan=0.0, posinf=2.0, neginf=-2.0)
    return feats.astype(np.float64)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = np.maximum(na * nb, 1e-12)
    out = np.sum(a * b, axis=1) / denom
    return np.clip(out, -1.0, 1.0)


def all_features(
    state: OptimizerState,
    edge_index: np.ndarray,
    ctx: OptimizerContext,
    cfg: Config | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    return node_features(state, ctx, cfg), edge_features(state, edge_index, ctx, cfg)
