"""Random-seed control for reproducible runs (CPU + CUDA)."""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch

__all__ = ["set_global_seed", "SeedBank", "stable_seed"]


def stable_seed(*parts: object, modulus: int = 2**31 - 1) -> int:
    """Deterministic 31-bit seed from arbitrary parts.

    Uses BLAKE2b rather than ``hash()`` because Python's string hash is salted
    per process (PYTHONHASHSEED), which would silently break reproducibility.
    """
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulus


def set_global_seed(seed: int, *, deterministic: bool = False, threads: int | None = None) -> dict:
    """Seed python/numpy/torch so that optimizer runs are comparable.

    Returns a small dict describing what was set; it is stored in run metadata.
    """
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if threads is not None:
        torch.set_num_threads(int(threads))
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return {
        "seed": seed,
        "deterministic_algorithms": bool(deterministic),
        "torch_threads": int(torch.get_num_threads()),
        "cuda_available": bool(torch.cuda.is_available()),
    }


@dataclass
class SeedBank:
    """Deterministic per-component RNG streams derived from a root seed.

    Independent streams avoid the accidental coupling you get when an optimizer
    and a benchmark share a single global numpy state.
    """

    root: int
    _streams: dict = field(default_factory=dict)

    def stream(self, name: str) -> np.random.Generator:
        if name not in self._streams:
            entropy = np.random.SeedSequence([int(self.root), stable_seed(name, modulus=2**31 - 1)])
            self._streams[name] = np.random.default_rng(entropy)
        return self._streams[name]

    def seed_for(self, name: str) -> int:
        return stable_seed(int(self.root), name)

    def torch_generator(self, name: str, device: str = "cpu") -> torch.Generator:
        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed_for(name))
        return gen
