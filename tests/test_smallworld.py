"""Tests for the Ring-SmallWorld-PSO prototype (review 2026-10-20)."""

from __future__ import annotations

import numpy as np
import pytest

from cgl_pso.runner import build_policy_for, smallworld_verdict
from cgl_pso.smallworld import SW_MODES, SmallWorldPolicy, make_smallworld_policy
from cgl_pso._infra.config import Config, load_config
from cgl_pso._infra.objectives import make_function
from cgl_pso._infra.objectives.base import CountingObjective
from cgl_pso._infra.optim import PSOEngine, make_classic_policy
from cgl_pso._infra.seeding import stable_seed


def _cfg(n: int = 30, dim: int = 8) -> Config:
    cfg = Config()
    cfg.set_path("objective.dim", dim)
    cfg.set_path("optimizer.n_particles", n)
    cfg.set_path("threads", 2)
    return cfg


def _run(kind: str, cfg: Config, *, iters: int = 20, seed: int = 0, task: str = "sphere", **kwargs):
    pol = make_smallworld_policy(kind, cfg, seed=seed, **kwargs)
    obj = CountingObjective(make_function(task, int(cfg.get_path("objective.dim")), rotate=True,
                                          shift=0.25, seed=70000), max_fes=None)
    eng = PSOEngine(obj, cfg, pol, seed=seed)
    state = eng.initialize()
    for _ in range(iters):
        state = eng.step()
    return pol, eng, state, obj


def test_modes_are_registered() -> None:
    assert set(SW_MODES) == {"ring", "fixed", "stall", "bandit"}
    for kind in ("sw_ring", "sw_fixed", "sw_stall", "sw_bandit"):
        pol = make_smallworld_policy(kind, _cfg(), seed=0)
        assert pol.mode == kind.replace("sw_", "")


def test_ring_mode_never_uses_a_non_local_link() -> None:
    cfg = _cfg()
    pol, eng, state, obj = _run("sw_ring", cfg, iters=20, p=0.5)
    assert all(t["sw_nonlocal_share"] == 0.0 for t in pol.trace)
    # and its attractors match the classical ring policy step by step
    cfg2 = cfg.copy()
    obj_a = CountingObjective(make_function("sphere", 8, rotate=True, shift=0.25, seed=70000), max_fes=None)
    obj_b = CountingObjective(make_function("sphere", 8, rotate=True, shift=0.25, seed=70000), max_fes=None)
    pol_sw = make_smallworld_policy("sw_ring", cfg2, seed=3)
    pol_ring = make_classic_policy("ring", cfg2)
    eng_a = PSOEngine(obj_a, cfg2, pol_sw, seed=5)
    eng_b = PSOEngine(obj_b, cfg2, pol_ring, seed=5)
    sa = eng_a.initialize()
    sb = eng_b.initialize()
    assert np.array_equal(sa.pos, sb.pos)
    for _ in range(10):
        sa = eng_a.step()
        sb = eng_b.step()
    assert np.allclose(sa.pbest_val, sb.pbest_val), "sw_ring must behave exactly like ring-lbest"


@pytest.mark.parametrize("p,lo,hi", [(0.0, 0.0, 1e-12), (1.0, 0.999, 1.001), (0.25, 0.15, 0.35)])
def test_fixed_mode_empirical_share_matches_p(p: float, lo: float, hi: float) -> None:
    cfg = _cfg(n=60, dim=6)
    pol, _, _, _ = _run("sw_fixed", cfg, iters=30, p=p)
    share = float(np.mean([t["sw_nonlocal_share"] for t in pol.trace]))
    assert lo <= share <= hi, (p, share)


def test_nonlocal_target_is_outside_the_ring_and_never_self() -> None:
    cfg = _cfg(n=40, dim=6)
    for trial in range(10):
        pol = make_smallworld_policy("sw_fixed", cfg, seed=trial, p=1.0, top_q=3, ring_radius=1)
        obj = CountingObjective(make_function("sphere", 6, seed=1), max_fes=None)
        eng = PSOEngine(obj, cfg, pol, seed=trial)
        state = eng.initialize()
        for _ in range(3):
            state = eng.step()
        target = pol._nonlocal_elite(state)
        idx = np.arange(state.n)
        d = np.abs(target - idx)
        d = np.minimum(d, state.n - d)
        assert not bool((target == idx).any()), "a particle must never be its own non-local elite"
        assert int(d.min()) >= 2, f"non-local target fell inside the ring (distance {int(d.min())})"


def test_stall_mode_links_only_stalled_particles() -> None:
    cfg = _cfg(n=40, dim=6)
    pol = make_smallworld_policy("sw_stall", cfg, seed=0, p=1.0, stall_threshold=4)
    obj = CountingObjective(make_function("sphere", 6, seed=2), max_fes=None)
    eng = PSOEngine(obj, cfg, pol, seed=0)
    state = eng.initialize()
    violations = 0
    for _ in range(25):
        stall_seen_by_policy = state.stall.copy()      # the policy decides *before* the update
        state = eng.step()
        used = pol._pending.used
        allowed = stall_seen_by_policy >= pol.stall_threshold
        if used.any() and not np.all(allowed[used]):
            violations += 1
    assert violations == 0, "the stagnation gate must control every non-local link"


def test_bandit_moves_p_in_the_relative_direction() -> None:
    cfg = _cfg(n=20, dim=4)
    pol = make_smallworld_policy("sw_bandit", cfg, seed=0, p=0.2, bandit_eta=0.2, p_min=0.01, p_max=0.5)
    used = np.array([True] * 10 + [False] * 10)

    def pending():
        return type(pol._pending)(used=used.copy(), fitness=np.zeros(20),
                                  pbest=np.ones(20), valid=True)

    # non-local users improve, local users do not -> p must increase
    pol._pending = pending()
    pol._apply_feedback(type("S", (), {"pbest_val": np.array([0.5] * 10 + [1.0] * 10)})())
    assert pol.p > 0.2
    # and the opposite case must decrease it
    pol.p = 0.2
    pol._pending = pending()
    pol._apply_feedback(type("S", (), {"pbest_val": np.array([1.0] * 10 + [0.5] * 10)})())
    assert pol.p < 0.2
    # the probability can never be switched off silently
    for _ in range(20):
        pol._pending = pending()
        pol._apply_feedback(type("S", (), {"pbest_val": np.array([1.0] * 10 + [0.5] * 10)})())
    assert pol.p >= pol.p_min


@pytest.mark.parametrize("kind", ["sw_ring", "sw_fixed", "sw_stall", "sw_bandit"])
def test_variants_keep_identical_fes_and_are_deterministic(kind: str) -> None:
    cfg = _cfg(n=12, dim=5)
    _, _, _, obj_a = _run(kind, cfg, iters=12, seed=1, p=0.2, stall_threshold=3)
    assert obj_a.n_evals == 12 * 13
    a = _run(kind, cfg, iters=12, seed=1, p=0.2, stall_threshold=3)[2]
    b = _run(kind, cfg, iters=12, seed=1, p=0.2, stall_threshold=3)[2]
    assert np.array_equal(a.pbest_pos, b.pbest_pos)


def test_selection_cost_stays_small() -> None:
    cfg = _cfg(n=40, dim=30)
    pol, _, _, _ = _run("sw_fixed", cfg, iters=40, p=0.15)
    assert float(np.mean(pol._select_ms[-30:])) < 0.5, f"SmallWorld selection too slow: {np.mean(pol._select_ms):.3f} ms"


def test_policy_factory_handles_all_kinds() -> None:
    cfg = _cfg()
    for kind in ("ring", "gbest", "sw_fixed", "sw_stall", "sw_bandit", "cgl_full"):
        pol = build_policy_for(kind, cfg, seed=0)
        assert pol is not None


def test_smallworld_config_is_pre_registered() -> None:
    cfg = load_config("configs/cgl/smallworld.yaml")
    pre = cfg.get_path("smallworld.pre_trial")
    main = cfg.get_path("smallworld.main")
    assert len(pre["settings"]) == 8
    assert set(pre["seeds"]) == {0, 1, 2}
    assert len(main["seeds"]) == 10
    assert set(cfg.get_path("smallworld.tasks")) == {"alpine1", "griewank", "rosenbrock", "zakharov"}
    assert int(cfg.get_path("smallworld.fes")) == 10_000
    assert int(cfg.get_path("smallworld.dim")) == 30
    budget = cfg.get_path("budget")
    assert budget["pre_trial_max_fes"] + budget["diagnostics_max_fes"] + budget["main_max_fes"] <= budget["total_max_fes"]
    assert budget["mechanism_rounds"] == 1
    labels = {s["label"] for s in pre["settings"]}
    assert {"ring", "sw_fixed_p05", "sw_stall_t10_p05", "sw_bandit_p05"} <= labels


def test_verdict_requires_no_collapse_and_cheap_runtime() -> None:
    import pandas as pd

    rows = []
    for seed in range(4):
        for task in ("t1", "t2", "t3", "t4"):
            base = {"seed": seed, "task": task, "fes": 100, "success_loose": False, "error": 1.0}
            rows.append({**base, "algorithm": "ring", "log10_error": 1.0, "runtime_s": 0.03})
            rows.append({**base, "algorithm": "sw_fixed", "log10_error": 0.8, "runtime_s": 0.04})
            rows.append({**base, "algorithm": "sw_bandit", "log10_error": 1.1, "runtime_s": 0.07})
            rows.append({**base, "algorithm": "cgl_full", "log10_error": 1.2, "runtime_s": 0.2})
    verdict = smallworld_verdict(pd.DataFrame(rows))
    assert verdict["per_candidate"]["sw_fixed"]["tasks_improved_vs_reference"] == 4
    assert verdict["per_candidate"]["sw_fixed"]["passed"] is True
    assert verdict["per_candidate"]["cgl_full"]["passed"] is False
    assert verdict["online_vs_fixed"]["keep_online"] is False
