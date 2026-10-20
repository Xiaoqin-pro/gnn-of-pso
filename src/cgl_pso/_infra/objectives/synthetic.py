"""Parameterised synthetic black-box functions.

Every instance can be shifted and rotated with a known global optimum, which is
what makes RL training instances *independent samples* rather than one fixed
landscape. ``x_opt`` is stored so tests can assert ``f(x_opt) == f_global``, and
the search box is always centred so that ``x_opt`` is reachable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .base import Bounds, FunctionObjective, Objective

__all__ = [
    "SyntheticObjective",
    "make_function",
    "synthetic_suite",
    "FUNCTION_REGISTRY",
    "random_orthogonal",
    "SCHWEFEL_OPT",
]


SCHWEFEL_OPT = 420.96874636


# --------------------------------------------------------------------------
# analytic kernels, defined on a plain z-space (n, d) -> (n,)
# --------------------------------------------------------------------------
def _sphere(z: np.ndarray) -> np.ndarray:
    return np.sum(z**2, axis=1)


def _ellipsoid(z: np.ndarray) -> np.ndarray:
    d = z.shape[1]
    coeff = 10.0 ** (6.0 * np.arange(d) / max(d - 1, 1))
    return np.sum(coeff * z**2, axis=1)


def _rosenbrock(z: np.ndarray) -> np.ndarray:
    return np.sum(100.0 * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + (z[:, :-1] - 1.0) ** 2, axis=1)


def _rastrigin(z: np.ndarray) -> np.ndarray:
    return 10.0 * z.shape[1] + np.sum(z**2 - 10.0 * np.cos(2.0 * np.pi * z), axis=1)


def _ackley(z: np.ndarray) -> np.ndarray:
    mean_sq = np.mean(z**2, axis=1)
    mean_cos = np.mean(np.cos(2.0 * np.pi * z), axis=1)
    return -20.0 * np.exp(-0.2 * np.sqrt(mean_sq)) - np.exp(mean_cos) + 20.0 + np.e


def _griewank(z: np.ndarray) -> np.ndarray:
    idx = np.sqrt(np.arange(1, z.shape[1] + 1))
    return np.sum(z**2, axis=1) / 4000.0 - np.prod(np.cos(z / idx), axis=1) + 1.0


def _schwefel(z: np.ndarray) -> np.ndarray:
    return 418.9828872724339 * z.shape[1] - np.sum(z * np.sin(np.sqrt(np.abs(z))), axis=1)


def _levy(z: np.ndarray) -> np.ndarray:
    w = 1.0 + (z - 1.0) / 4.0
    term1 = np.sin(np.pi * w[:, 0]) ** 2
    term3 = (w[:, -1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[:, -1]) ** 2)
    mid = np.sum((w[:, :-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:, :-1] + 1.0) ** 2), axis=1)
    return term1 + mid + term3


def _styblinski_tang(z: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(z**4 - 16.0 * z**2 + 5.0 * z, axis=1)


def _michalewicz(z: np.ndarray) -> np.ndarray:
    idx = np.arange(1, z.shape[1] + 1)
    return -np.sum(np.sin(z) * np.sin(idx * z**2 / np.pi) ** 20, axis=1)


def _zakharov(z: np.ndarray) -> np.ndarray:
    s1 = np.sum(z**2, axis=1)
    s2 = np.sum(0.5 * np.arange(1, z.shape[1] + 1) * z, axis=1)
    return s1 + s2**2 + s2**4


def _alpine1(z: np.ndarray) -> np.ndarray:
    """Alpine N.1: f(z) = sum |z sin z + 0.1 z|, minimum 0 at z = 0."""
    return np.sum(np.abs(z * np.sin(z) + 0.1 * z), axis=1)


def _weierstrass(z: np.ndarray) -> np.ndarray:
    """Weierstrass: continuous but nowhere differentiable; minimum 0 at z = 0."""
    a, b, kmax = 0.5, 3.0, 20
    k = np.arange(kmax + 1)
    prefactor = np.sum(a**k * np.cos(np.pi * b**k))
    return np.sum(
        np.sum(a**k[None, :] * np.cos(2.0 * np.pi * b**k[None, :] * (z[:, :, None] + 0.5)), axis=-1),
        axis=1,
    ) - z.shape[1] * prefactor


def _hybrid_rz(z: np.ndarray) -> np.ndarray:
    """Composite with a known optimum at z = 0: Rastrigin (+) Zakharov.

    First half: multimodal Rastrigin. Second half: ill-conditioned Zakharov.
    Both components are zero at the origin, so the composite minimum is exactly 0
    (the previous Rana-based version had no analytic optimum at the origin and was
    therefore rejected by the ``f(x*) == f_global`` test).
    """
    d = z.shape[1]
    half = max(d // 2, 1)
    a, b = z[:, :half], z[:, half:]
    rast = 10.0 * half + np.sum(a**2 - 10.0 * np.cos(2.0 * np.pi * a), axis=1)
    if b.shape[1] == 0:
        return rast
    s1 = np.sum(b**2, axis=1)
    s2 = np.sum(0.5 * np.arange(1, b.shape[1] + 1) * b, axis=1)
    return rast + s1 + s2**2 + s2**4


@dataclass(frozen=True)
class _Kernel:
    fn: Callable[[np.ndarray], np.ndarray]
    z_opt: float | str  # float, "zeros", or "ones"
    native_bounds: tuple[float, float]
    f_global_unbiased: float
    modality: str


FUNCTION_REGISTRY: dict[str, _Kernel] = {
    "sphere": _Kernel(_sphere, "zeros", (-5.12, 5.12), 0.0, "unimodal"),
    "ellipsoid": _Kernel(_ellipsoid, "zeros", (-5.12, 5.12), 0.0, "unimodal"),
    "rosenbrock": _Kernel(_rosenbrock, "ones", (-2.048, 2.048), 0.0, "unimodal-valley"),
    "rastrigin": _Kernel(_rastrigin, "zeros", (-5.12, 5.12), 0.0, "multimodal"),
    "ackley": _Kernel(_ackley, "zeros", (-32.768, 32.768), 0.0, "multimodal"),
    "griewank": _Kernel(_griewank, "zeros", (-600.0, 600.0), 0.0, "multimodal"),
    "schwefel": _Kernel(_schwefel, SCHWEFEL_OPT, (-500.0, 500.0), 0.0, "multimodal-deceptive"),
    "levy": _Kernel(_levy, "ones", (-10.0, 10.0), 0.0, "multimodal"),
    # Stage-3 additions: fresh families for the pre-registered dev tasks and
    # harder (less saturated) training instances.
    # Styblinski-Tang optimum is at z_i = -2.903534 with F = -39.16599 * d.
    # NOTE: the commonly quoted per-dimension optimum -39.16599 belongs to a
    # slightly different parameterisation. The value below is the exact minimum
    # of THIS polynomial, computed from its derivative root
    # (x* = -2.9035340277711783), so f(x*) == f_global holds to machine precision.
    "styblinski_tang": _Kernel(
        _styblinski_tang, -2.9035340277711783, (-5.0, 5.0), -39.16616570377141, "multimodal"
    ),
    "michalewicz": _Kernel(_michalewicz, "zeros", (0.0, np.pi), -0.0, "multimodal-ridge"),
    "zakharov": _Kernel(_zakharov, "zeros", (-5.0, 10.0), 0.0, "unimodal-ill-conditioned"),
    "alpine1": _Kernel(_alpine1, "zeros", (-10.0, 10.0), 0.0, "multimodal-rugged"),
    "weierstrass": _Kernel(_weierstrass, "zeros", (-0.5, 0.5), 0.0, "multimodal-fractal"),
    "hybrid_rz": _Kernel(_hybrid_rz, "zeros", (-5.12, 5.12), 0.0, "composite"),
}


def random_orthogonal(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform orthogonal matrix via sign-corrected QR of a Gaussian matrix."""
    a = rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(a)
    d = np.diagonal(r)
    q = q * np.where(d >= 0, 1.0, -1.0)
    return q


def _kernel_z_opt(kernel: _Kernel, dim: int) -> np.ndarray:
    if kernel.z_opt == "zeros":
        return np.zeros(dim)
    if kernel.z_opt == "ones":
        return np.ones(dim)
    return np.full(dim, float(kernel.z_opt))


class SyntheticObjective(FunctionObjective):
    """A shifted/rotated instance of a synthetic kernel."""

    def __init__(
        self,
        name: str,
        dim: int,
        *,
        shift: np.ndarray | None = None,
        rotation: np.ndarray | None = None,
        bias: float = 0.0,
        bounds: Bounds | None = None,
        instance_seed: int | None = None,
    ) -> None:
        if name not in FUNCTION_REGISTRY:
            raise KeyError(f"unknown synthetic function {name!r}; known: {sorted(FUNCTION_REGISTRY)}")
        kernel = FUNCTION_REGISTRY[name]
        d = int(dim)
        self.kernel_name = name
        self.instance_seed = instance_seed
        self.bias = float(bias)
        self.shift = np.zeros(d) if shift is None else np.asarray(shift, dtype=np.float64).reshape(-1)
        if self.shift.size != d:
            raise ValueError(f"shift must have {d} entries, got {self.shift.size}")
        if rotation is None:
            self.rotation = np.eye(d)
            self.rotated = False
        else:
            rot = np.asarray(rotation, dtype=np.float64)
            if rot.shape != (d, d):
                raise ValueError(f"rotation must be ({d}, {d}), got {rot.shape}")
            self.rotation = rot
            self.rotated = True

        self.z_opt = _kernel_z_opt(kernel, d)
        # x_opt solves z = R (x - shift) = z_opt  ->  x = shift + R^T z_opt
        self.x_opt = self.shift + self.rotation.T @ self.z_opt

        # Styblinski-Tang's optimum scales with the dimension (-39.16599 * d)
        f_unbiased = kernel.f_global_unbiased * (d if name == "styblinski_tang" else 1.0)

        half = 0.5 * (kernel.native_bounds[1] - kernel.native_bounds[0])
        if bounds is None:
            # centre the box on x_opt so the optimum is always reachable
            bounds = Bounds(self.x_opt - half, self.x_opt + half)
        super().__init__(
            self._evaluate_impl,
            name=f"{name}{'r' if self.rotated else ''}",
            dim=d,
            bounds=bounds,
            f_global=self.bias + f_unbiased,
            modality=kernel.modality,
            source="synthetic",
            verified=True,
        )

    def _evaluate_impl(self, X: np.ndarray) -> np.ndarray:
        z = (X - self.shift) @ self.rotation.T
        return FUNCTION_REGISTRY[self.kernel_name].fn(z) + self.bias

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "kernel": self.kernel_name,
            "dim": self.dim,
            "rotated": self.rotated,
            "shift_norm": float(np.linalg.norm(self.shift)),
            "instance_seed": self.instance_seed,
            "f_global": self.f_global,
            "bounds": self.bounds.as_dict(),
        }


def make_function(
    name: str,
    dim: int,
    *,
    rotate: bool = False,
    shift: float | Sequence[float] = 0.0,
    seed: int | None = None,
    bias: float = 0.0,
    bounds: Bounds | Sequence[Sequence[float]] | None = None,
) -> SyntheticObjective:
    """Build one instance.

    ``shift`` may be a scalar fraction of the native half-range (drawn
    per-coordinate) or an explicit vector.
    """
    kernel = FUNCTION_REGISTRY[name]
    d = int(dim)
    rng = np.random.default_rng(seed)
    if isinstance(shift, (int, float)):
        frac = float(shift)
        half = 0.5 * (kernel.native_bounds[1] - kernel.native_bounds[0])
        base_shift = np.zeros(d) if frac == 0.0 else frac * half * rng.uniform(-1.0, 1.0, size=d)
    else:
        base_shift = np.asarray(shift, dtype=np.float64).reshape(-1)
        if base_shift.size == 1:
            base_shift = np.full(d, float(base_shift[0]))
        if base_shift.size != d:
            raise ValueError(f"shift vector must have {d} entries, got {base_shift.size}")
    rotation = random_orthogonal(d, rng) if rotate else None
    if isinstance(bounds, Bounds) or bounds is None:
        b: Bounds | None = bounds if isinstance(bounds, Bounds) else None
    else:
        arr = np.asarray(bounds, dtype=np.float64)
        if arr.shape == (dim, 2):
            b = Bounds(arr[:, 0], arr[:, 1])
        elif arr.shape == (2,):
            b = Bounds(np.full(dim, arr[0]), np.full(dim, arr[1]))
        else:
            raise ValueError(f"bounds must be (dim, 2) or (2,), got {arr.shape}")
    return SyntheticObjective(
        name,
        d,
        shift=base_shift,
        rotation=rotation,
        bias=bias,
        bounds=b,
        instance_seed=seed,
    )


def synthetic_suite(
    names: Sequence[str],
    dim: int,
    *,
    seed: int = 0,
    rotate: bool = False,
    shift: float = 0.0,
) -> list[SyntheticObjective]:
    """Independent instances per function name, seeded deterministically."""
    out = []
    for i, name in enumerate(names):
        out.append(make_function(name, dim, rotate=rotate, shift=shift, seed=seed + 1000 * i))
    return out
