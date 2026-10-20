"""Objective (black-box function) interfaces with explicit FE accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

__all__ = ["Bounds", "Objective", "FunctionObjective", "CountingObjective", "NAN_PENALTY"]


NAN_PENALTY = 1e12


@dataclass(frozen=True)
class Bounds:
    low: np.ndarray
    high: np.ndarray

    def __post_init__(self) -> None:
        low = np.atleast_1d(np.asarray(self.low, dtype=np.float64))
        high = np.atleast_1d(np.asarray(self.high, dtype=np.float64))
        if low.shape != high.shape:
            raise ValueError(f"bounds shape mismatch: {low.shape} vs {high.shape}")
        if np.any(high <= low):
            raise ValueError("every upper bound must be strictly greater than its lower bound")
        object.__setattr__(self, "low", np.asarray(low, dtype=np.float64))
        object.__setattr__(self, "high", np.asarray(high, dtype=np.float64))

    @property
    def dim(self) -> int:
        return int(self.low.size)

    @property
    def span(self) -> np.ndarray:
        return self.high - self.low

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.high + self.low)

    def to_unit(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return (x - self.low) / self.span

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        return self.low + u * self.span

    def as_dict(self) -> dict:
        return {"low": self.low.tolist(), "high": self.high.tolist()}


class Objective:
    """Base class for batched black-box objectives.

    Subclasses implement :meth:`evaluate`; the contract is one fitness value per
    row of ``X`` (shape ``(n, dim)``).  Vectorised numpy implementations are
    strongly preferred but not required.
    """

    name: str = "objective"
    dim: int = 2
    f_global: float = 0.0
    modality: str = "unknown"
    source: str = "builtin"
    verified: bool = True  # whether the reported global optimum was checked

    def __init__(self, dim: int | None = None, bounds: Bounds | Sequence[Sequence[float]] | None = None):
        if dim is not None:
            self.dim = int(dim)
        if bounds is None:
            self.bounds = Bounds(np.full(self.dim, -5.12), np.full(self.dim, 5.12))
        elif isinstance(bounds, Bounds):
            self.bounds = bounds
        else:
            arr = np.asarray(bounds, dtype=np.float64)
            if arr.shape == (2,):
                self.bounds = Bounds(np.full(self.dim, arr[0]), np.full(self.dim, arr[1]))
            elif arr.shape == (self.dim, 2):
                self.bounds = Bounds(arr[:, 0], arr[:, 1])
            else:
                raise ValueError(f"bounds must be (2,) or (dim, 2); got {arr.shape}")

    # -- core API -------------------------------------------------------
    def evaluate(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.evaluate(X)

    def evaluate_rows(self, rows: Iterable[np.ndarray]) -> np.ndarray:
        X = np.atleast_2d(np.asarray(list(rows), dtype=np.float64))
        return self.evaluate(X)

    # -- convenience ----------------------------------------------------
    def clip(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X, self.bounds.low, self.bounds.high)

    @property
    def tag(self) -> str:
        return f"{self.name}-d{self.dim}"


class FunctionObjective(Objective):
    """Wrap a callable ``f(X) -> (n,)`` into an Objective."""

    def __init__(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        *,
        name: str,
        dim: int,
        bounds: Bounds | Sequence[Sequence[float]],
        f_global: float = 0.0,
        modality: str = "unknown",
        source: str = "builtin",
        verified: bool = True,
    ):
        super().__init__(dim=dim, bounds=bounds)
        self._fn = fn
        self.name = name
        self.f_global = float(f_global)
        self.modality = modality
        self.source = source
        self.verified = verified

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        raw = np.asarray(self._fn(X), dtype=np.float64).reshape(-1)
        if raw.size != X.shape[0]:
            raise ValueError(f"{self.name}: expected {X.shape[0]} values, got {raw.size}")
        return _sanitize(raw)


def _sanitize(values: np.ndarray, penalty: float = NAN_PENALTY) -> np.ndarray:
    """Replace NaN/inf with a large finite penalty so the search cannot exploit it."""
    bad = ~np.isfinite(values)
    if np.any(bad):
        values = np.array(values, dtype=np.float64, copy=True)
        values[bad] = penalty
    return values


class CountingObjective:
    """Fitness-evaluation counter shared by every optimizer in the benchmark.

    Fairness across algorithms depends on this being the *only* place FEs are
    counted: one row evaluated == one FE, no matter which optimizer asks.
    """

    def __init__(self, objective: Objective, *, max_fes: int | None = None, batch_limit: int | None = None):
        self.objective = objective
        self.max_fes = int(max_fes) if max_fes is not None else None
        self.batch_limit = batch_limit
        self.n_evals = 0
        self.history: list[tuple[int, float]] = []  # (fes, best-so-far)
        self._best = np.inf

    # -- passthrough properties ----------------------------------------
    @property
    def dim(self) -> int:
        return self.objective.dim

    @property
    def name(self) -> str:
        return self.objective.name

    @property
    def bounds(self) -> Bounds:
        return self.objective.bounds

    @property
    def f_global(self) -> float:
        return self.objective.f_global

    @property
    def best(self) -> float:
        return self._best

    @property
    def exhausted(self) -> bool:
        return self.max_fes is not None and self.n_evals >= self.max_fes

    # -- evaluation -----------------------------------------------------
    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        n = X.shape[0]
        if self.max_fes is not None and self.n_evals + n > self.max_fes:
            keep = self.max_fes - self.n_evals
            if keep <= 0:
                return np.full(n, np.inf)
            vals = np.full(n, np.inf)
            vals[:keep] = self._eval_chunked(X[:keep])
            self.n_evals += keep
        else:
            vals = self._eval_chunked(X)
            self.n_evals += n
        finite = vals[np.isfinite(vals)]
        if finite.size:
            self._best = min(self._best, float(finite.min()))
            self.history.append((self.n_evals, self._best))
        return vals

    def _eval_chunked(self, X: np.ndarray) -> np.ndarray:
        limit = self.batch_limit or X.shape[0]
        if X.shape[0] <= limit:
            return self.objective.evaluate(X)
        chunks = [self.objective.evaluate(X[i : i + limit]) for i in range(0, X.shape[0], limit)]
        return np.concatenate(chunks)

    def evaluate_rows(self, rows: Iterable[np.ndarray]) -> np.ndarray:
        X = np.atleast_2d(np.asarray(list(rows), dtype=np.float64))
        return self(X)

    def reset(self) -> None:
        self.n_evals = 0
        self.history.clear()
        self._best = np.inf
