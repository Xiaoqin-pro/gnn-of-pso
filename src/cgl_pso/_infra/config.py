"""Configuration loading: YAML files + dotted-path overrides.

The config object is intentionally a thin, validated dict wrapper so that CLI
overrides (``--set optimizer.n_particles=60``) stay trivial while every field
still has a documented default and range check.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping

import yaml

__all__ = ["Config", "DEFAULTS", "load_config", "validate_config", "ConfigError"]


class ConfigError(ValueError):
    """Raised when a configuration value is invalid."""


DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "device": "auto",
    "threads": 8,
    "run_name": "default",
    "results_dir": "results",
    "objective": {
        "name": "sphere",
        "dim": 10,
        "bounds": None,  # None -> objective default
        "shift": 0.0,
        "rotate": False,
        "instance_seed": 12345,
    },
    "optimizer": {
        "kind": "gldi",  # gbest|ring|fips|clpso|rank|random|gldi
        "n_particles": 40,
        "iters": 300,
        "w": 0.7298,
        "c1": 1.49445,
        "c2": 1.49445,
        "vmax_frac": 0.2,
        "boundary": "clip",  # clip|reflect|random
        "init": "uniform",  # uniform|lhs
        "init_vel_frac": 0.1,
        "ring_radius": 1,
        "fips_neighbors": 2,
        "clpso_pc": 0.5,
        "clpso_refresh_gap": 7,
        "fips_weights": "uniform",  # uniform|rank
        "graph": {
            "k_quality": 3,
            "k_dist": 1,
            "k_random_far": 2,
            "fixed": False,
            "rebuild_period": 1,
            "distance_quantile": 0.5,
        },
        "model": {
            "hidden": 32,
            "heads": 4,
            "layers": 2,
            "dropout": 0.0,
            "residual": True,
        },
    },
    "rl": {
        "enabled": False,
        "algo": "ppo",
        "lr": 3e-4,
        "gamma": 0.99,
        "lam": 0.95,
        "clip": 0.2,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "epochs": 4,
        "minibatch_particles": 2048,
        "episodes_per_update": 8,
        "updates": 200,
        "max_grad_norm": 0.5,
        "target_kl": 0.03,
        "reward_scale": "gap",  # gap|improvement|hybrid (all dimensionless)
        "train_functions": ["sphere", "rastrigin", "ackley", "rosenbrock"],
        "train_dims": [10, 20],
        "train_particles": [20, 40],
        "eval_every": 10,
        "value_clip": 10.0,
        "policy": "gnn",  # gnn|mlp|uniform|static|shuffled|no_history
    },
    "resources": {
        "enabled": True,
        "log_period_s": 5.0,
        "cpu_soft_pct": 90.0,
        "cpu_hard_pct": 95.0,
        "ram_soft_pct": 90.0,
        "ram_hard_pct": 95.0,
        "gpu_soft_c": 82.0,
        "gpu_hard_c": 85.0,
        "vram_budget_gb": 6.0,
        "gpu_reserve_gb": 0.5,
        "check_every_steps": 1,
        "max_batch_scale": 4.0,
        "min_batch_scale": 0.25,
    },
    "benchmark": {
        "algorithms": ["gbest", "ring", "fips", "clpso", "rank", "random", "gldi_rule", "gldi_ppo"],
        "functions": ["sphere", "rastrigin", "ackley", "rosenbrock"],
        "dim": 10,
        "repetitions": 30,
        "fes_budget": 60000,
        "success_tolerance": 1e-8,
        "tier": "preliminary",
        "suite": "synthetic",  # synthetic|cec2017|cec2022
        "checkpoints": 20,
    },
    "robot": {
        "enabled": False,
        "urdf": "kuka_iiwa/model.urdf",
        "obstacles": True,
        "n_waypoints": 12,
        "n_control_points": 5,
        "collision_samples": 9,
        "iterations": 60,
        "pop_size": 32,
        "length_weight": 1.0,
        "smoothness_weight": 0.3,
        "collision_weight": 100.0,
    },
}


class Config(MutableMapping[str, Any]):
    """Nested configuration with dotted-path access."""

    def __init__(self, data: Mapping[str, Any] | None = None):
        self._data: dict[str, Any] = copy.deepcopy(DEFAULTS)
        if data:
            self._merge(data)

    # -- construction ---------------------------------------------------
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Config":
        return cls(data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"config file not found: {p}")
        loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, Mapping):
            raise ConfigError(f"config file must contain a mapping, got {type(loaded).__name__}")
        return cls(loaded)

    def copy(self) -> "Config":
        return Config(self._data)

    # -- nested access --------------------------------------------------
    def _merge(self, data: Mapping[str, Any]) -> None:
        def rec(target: dict, src: Mapping[str, Any]) -> None:
            for key, value in src.items():
                if isinstance(value, Mapping) and isinstance(target.get(key), dict):
                    rec(target[key], value)
                else:
                    target[key] = copy.deepcopy(value)

        rec(self._data, data)

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> "Config":
        parts = dotted.split(".")
        node = self._data
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value
        return self

    def apply_overrides(self, overrides: Mapping[str, Any]) -> "Config":
        for key, value in overrides.items():
            if key in self._data and isinstance(self._data[key], Mapping) and isinstance(value, Mapping):
                self._merge({key: value})
            else:
                self.set_path(key, value)
        return self

    # -- mapping protocol ----------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    # -- persistence ----------------------------------------------------
    def to_yaml(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(self._data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return p

    def to_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self._data, indent=2, sort_keys=False, default=str), encoding="utf-8")
        return p


def parse_override(text: str) -> tuple[str, Any]:
    """Parse ``key=value`` CLI overrides with YAML scalar typing."""
    if "=" not in text:
        raise ConfigError(f"override must look like key=value, got {text!r}")
    key, raw = text.split("=", 1)
    key = key.strip()
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"cannot parse override value {raw!r}: {exc}") from exc
    return key, value


def load_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | list[str] | None = None,
) -> Config:
    cfg = Config.from_yaml(path) if path is not None else Config()
    if overrides:
        if isinstance(overrides, Mapping):
            cfg.apply_overrides(overrides)
        else:
            parsed = dict(parse_override(item) for item in overrides)
            cfg.apply_overrides(parsed)
    errors = validate_config(cfg)
    if errors:
        raise ConfigError("; ".join(errors))
    return cfg


def validate_config(cfg: Config) -> list[str]:
    """Return a list of human-readable validation problems (empty == valid)."""
    errs: list[str] = []

    def need(cond: bool, msg: str) -> None:
        if not cond:
            errs.append(msg)

    need(int(cfg.get_path("objective.dim", 0)) >= 1, "objective.dim must be >= 1")
    need(int(cfg.get_path("optimizer.n_particles", 0)) >= 2, "optimizer.n_particles must be >= 2")
    need(int(cfg.get_path("optimizer.iters", 0)) >= 1, "optimizer.iters must be >= 1")
    need(float(cfg.get_path("optimizer.w", 1.0)) >= 0.0, "optimizer.w must be >= 0")
    need(float(cfg.get_path("optimizer.c1", 1.0)) >= 0.0, "optimizer.c1 must be >= 0")
    need(float(cfg.get_path("optimizer.c2", 1.0)) >= 0.0, "optimizer.c2 must be >= 0")
    need(
        cfg.get_path("optimizer.boundary") in {"clip", "reflect", "random"},
        "optimizer.boundary must be one of clip/reflect/random",
    )
    need(cfg.get_path("optimizer.init") in {"uniform", "lhs"}, "optimizer.init must be uniform or lhs")
    need(cfg.get_path("optimizer.kind") in {"gbest", "ring", "fips", "clpso", "rank", "random", "gldi"},
         "optimizer.kind must be gbest/ring/fips/clpso/rank/random/gldi")
    need(
        cfg.get_path("rl.policy") in {"gnn", "mlp", "uniform", "static", "shuffled", "no_history"},
        "rl.policy must be gnn/mlp/uniform/static/shuffled/no_history",
    )
    need(int(cfg.get_path("rl.episodes_per_update", 1)) >= 1, "rl.episodes_per_update must be >= 1")
    need(
        cfg.get_path("rl.reward_scale") in {"gap", "improvement", "hybrid"},
        "rl.reward_scale must be gap/improvement/hybrid",
    )
    need(float(cfg.get_path("rl.lr", 1e-4)) > 0, "rl.lr must be > 0")
    need(0.0 < float(cfg.get_path("rl.clip", 0.2)) < 1.0, "rl.clip must be in (0, 1)")
    sm = cfg.get_path("resources.ram_soft_pct", 90.0)
    hm = cfg.get_path("resources.ram_hard_pct", 95.0)
    need(float(sm) < float(hm), "resources.ram_soft_pct must be < ram_hard_pct")
    need(float(hm) <= 95.0, "resources.ram_hard_pct must be <= 95 (user-authorized ceiling)")
    sc = cfg.get_path("resources.gpu_soft_c", 82.0)
    hc = cfg.get_path("resources.gpu_hard_c", 85.0)
    need(float(sc) < float(hc), "resources.gpu_soft_c must be < gpu_hard_c")
    need(float(cfg.get_path("benchmark.repetitions", 1)) >= 1, "benchmark.repetitions must be >= 1")
    return errs
