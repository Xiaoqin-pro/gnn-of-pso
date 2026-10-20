"""Objective registry helpers shared by the CLI and benchmarks."""

from __future__ import annotations

from typing import Sequence

from .base import Bounds, CountingObjective, Objective
from .synthetic import SyntheticObjective, make_function, synthetic_suite

__all__ = [
    "Bounds",
    "CountingObjective",
    "Objective",
    "SyntheticObjective",
    "make_function",
    "synthetic_suite",
    "build_objective",
    "build_suite",
]


def build_objective(name: str, dim: int, **kwargs) -> Objective:
    """Create an objective by name from any registered suite."""
    lowered = name.lower()
    if lowered.startswith("cec2017") or lowered.startswith("cec2022"):
        raise NotImplementedError(
            "CEC suites are not bundled in this CGL-PSO export (the CGL development "
            "protocol never used them); install the full GLDI-PSO repository for CEC support"
        )
    return make_function(lowered, dim, **kwargs)


def build_suite(
    names: Sequence[str],
    dim: int,
    *,
    suite: str = "synthetic",
    seed: int = 0,
    rotate: bool = False,
    shift: float = 0.0,
) -> list[Objective]:
    """Build a list of objectives.

    ``suite='synthetic'`` -> parameterised synthetic instances.
    ``suite='cec2017'/'cec2022'`` -> opfunu-backed CEC functions (independent test set).
    """
    suite = suite.lower()
    if suite == "synthetic":
        return list(synthetic_suite(names, dim, seed=seed, rotate=rotate, shift=shift))
    if suite in {"cec2017", "cec2022"}:
        raise NotImplementedError(
            "CEC suites are not bundled in this CGL-PSO export (the CGL development "
            "protocol never used them); install the full GLDI-PSO repository for CEC support"
        )
    raise ValueError(f"unknown suite {suite!r}")
