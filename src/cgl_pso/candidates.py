"""Sparse candidate construction for CGL-PSO (M2).

Candidate sources per particle ``i``:

* ``ring``      : the ``ring_radius`` neighbours on each side (the proven topology),
* ``elite``     : the ``top_q`` particles by personal-best value,
* ``far``       : the most distant particle among a small random sample
                  (an O(N * m * D) approximation of the global farthest point,
                  instead of the O(N^2 D) full distance matrix),
* ``random``    : a few uniformly random particles for exploration.

Self-loops are always removed and rows are de-duplicated, so a particle can never
learn from itself and never receives the same target twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cgl_pso._infra.objectives.base import Bounds

__all__ = ["CandidateSet", "build_candidates", "CANDIDATE_SOURCES"]

CANDIDATE_SOURCES = ("ring", "elite", "far", "random", "self")
_SOURCE_ID = {name: i for i, name in enumerate(CANDIDATE_SOURCES)}


@dataclass
class CandidateSet:
    idx: np.ndarray  # (N, K) candidate particle ids (0 where padded)
    mask: np.ndarray  # (N, K) bool, True = real candidate
    source: np.ndarray  # (N, K) int, see CANDIDATE_SOURCES
    n_candidates: np.ndarray  # (N,) per-particle count

    def to_dict(self) -> dict[str, Any]:
        counts = {n: int((self.source[self.mask] == i).sum()) for n, i in _SOURCE_ID.items()}
        return {"mean_candidates": float(self.n_candidates.mean()),
                "min_candidates": int(self.n_candidates.min()),
                "source_counts": counts}


def build_candidates(
    position: np.ndarray,
    pbest_val: np.ndarray,
    bounds: Bounds,
    rng: np.random.Generator,
    *,
    ring_radius: int = 1,
    top_q: int = 4,
    n_random: int = 2,
    n_far_samples: int = 8,
    use_source_flags: dict[str, bool] | None = None,
    pbest_order: np.ndarray | None = None,
    include_self: bool = False,
) -> CandidateSet:
    """Build the padded candidate set.

    Fully vectorised (no per-particle Python loop): ring neighbours are index
    arithmetic, elites come from one argsort, the long-range slot is the farthest
    of a small random sample, and duplicates/self-loops are removed with a
    slot-vs-earlier-slot comparison. Removing entries leaves padding holes, which
    the downstream scorer masks out, so no compaction is needed.
    """
    n, d = position.shape
    flags = {"ring": True, "elite": True, "far": True, "random": True}
    if use_source_flags:
        flags.update({k: bool(v) for k, v in use_source_flags.items() if k in flags})

    blocks: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    rows = np.arange(n)[:, None]

    if flags["ring"] and int(ring_radius) > 0:
        offs = np.array([o for o in range(-int(ring_radius), int(ring_radius) + 1) if o != 0])
        ring = (rows + offs[None, :]) % n
        blocks.append(ring)
        sources.append(np.full(ring.shape, _SOURCE_ID["ring"], dtype=np.int64))
    if flags["elite"] and int(top_q) > 0:
        order_full = pbest_order if pbest_order is not None else np.argsort(pbest_val, kind="stable")
        order = order_full[: int(top_q)]
        elite = np.broadcast_to(order[None, :], (n, order.size)).copy()
        blocks.append(elite)
        sources.append(np.full(elite.shape, _SOURCE_ID["elite"], dtype=np.int64))
    if flags["far"] and int(n_far_samples) > 0:
        m = int(n_far_samples)
        sample = rng.integers(0, n, size=(n, m))
        span = bounds.span
        unit = position / span[None, :]
        dist = np.linalg.norm(unit[:, None, :] - unit[sample], axis=2)
        # sampled self must be *excluded* from the argmax: a self-draw previously
        # scored +inf and therefore won "farthest", after which the self-loop filter
        # silently dropped the row's long-range slot. -inf fixes the semantics.
        valid = sample != rows
        dist = np.where(valid, dist, -np.inf)
        pick = np.argmax(dist, axis=1)
        has_valid = valid.any(axis=1)
        fallback = (np.arange(n) + 1) % n          # only if every draw was self
        far = np.where(has_valid, sample[np.arange(n), pick], fallback)[:, None]
        blocks.append(far)
        sources.append(np.full(far.shape, _SOURCE_ID["far"], dtype=np.int64))
    if flags["random"] and int(n_random) > 0:
        rand = rng.integers(0, n, size=(n, int(n_random)))
        blocks.append(rand)
        sources.append(np.full(rand.shape, _SOURCE_ID["random"], dtype=np.int64))

    if include_self:
        # inserted FIRST so the de-duplication keeps the explicit `self` label even
        # when the same particle also appears in the ring/elite/far/random blocks
        self_col = np.arange(n, dtype=np.int64)[:, None]
        blocks.insert(0, self_col)
        sources.insert(0, np.full(self_col.shape, _SOURCE_ID["self"], dtype=np.int64))

    if not blocks:  # degenerate configuration: fall back to the global best
        base = pbest_order if pbest_order is not None else np.argsort(pbest_val, kind="stable")
        order = base[: max(1, min(n, 2))]
        blk = np.broadcast_to(order[None, :], (n, order.size)).copy()
        blocks.append(blk)
        sources.append(np.full(blk.shape, _SOURCE_ID["elite"], dtype=np.int64))

    idx = np.concatenate(blocks, axis=1)
    source = np.concatenate(sources, axis=1)
    k = idx.shape[1]

    # drop self-loops and later duplicates of an earlier slot (vectorised)
    earlier = np.arange(k)[None, None, :] < np.arange(k)[None, :, None]
    dup = np.any((idx[:, :, None] == idx[:, None, :]) & earlier, axis=2)
    self_ok = np.ones(idx.shape, dtype=bool) if include_self else (idx != rows)
    mask = self_ok & (~dup)
    idx = np.where(mask, idx, 0)
    source = np.where(mask, source, -1)
    counts = mask.sum(axis=1)

    # guaranteed coverage: any row left empty takes its best non-self particle
    empty = np.flatnonzero(counts == 0)
    if empty.size:
        order = pbest_order if pbest_order is not None else np.argsort(pbest_val, kind="stable")
        for i in empty:  # only for degenerate rows; normally empty
            cand = next(int(j) for j in order if int(j) != int(i))
            idx[i, 0] = cand
            mask[i, 0] = True
            source[i, 0] = _SOURCE_ID["elite"]
            counts[i] = 1
    return CandidateSet(idx=idx, mask=mask, source=source, n_candidates=counts)
