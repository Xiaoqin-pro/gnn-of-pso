"""CGL-PSO tests: graph updates, temporal causality, candidates, mechanisms, budget."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cgl_pso.candidates import CANDIDATE_SOURCES, build_candidates
from cgl_pso.contribution import ContributionGraph
from cgl_pso.policy import CGL_POLICIES, CGLPolicy, make_cgl_policy
from cgl_pso.runner import (
    CGL_ALGORITHMS,
    DEV_SEEDS,
    DEV_TASKS,
    INSTANCE_SEED_BASE,
    PILOT_TASKS,
    screening_verdict,
)
from cgl_pso._infra.config import Config, load_config
from cgl_pso._infra.objectives import make_function
from cgl_pso._infra.objectives.base import CountingObjective
from cgl_pso._infra.optim import PSOEngine, make_classic_policy
from cgl_pso._infra.seeding import set_global_seed, stable_seed


def _cfg(n: int = 12, dim: int = 6, iters: int = 10) -> Config:
    cfg = Config()
    cfg.set_path("objective.dim", dim)
    cfg.set_path("optimizer.n_particles", n)
    cfg.set_path("optimizer.iters", iters)
    cfg.set_path("threads", 2)
    return cfg


def _run(variant: str, cfg: Config, *, iters: int = 10, task: str = "sphere", seed: int = 0, **kwargs):
    pol = make_cgl_policy(variant, cfg, seed=seed, **kwargs)
    obj = CountingObjective(make_function(task, int(cfg.get_path("objective.dim")), rotate=True,
                                          shift=0.25, seed=70000), max_fes=None)
    eng = PSOEngine(obj, cfg, pol, seed=seed)
    state = eng.initialize()
    for _ in range(iters):
        state = eng.step()
    return pol, eng, state, obj


# ------------------------------------------------------------- candidates
def test_candidates_have_no_self_loops_and_no_duplicates() -> None:
    cfg = _cfg(n=25, dim=8)
    rng = np.random.default_rng(0)
    cs = build_candidates(rng.random((25, 8)), rng.random(25), make_function("sphere", 8, seed=0).bounds, rng)
    for i in range(25):
        targets = cs.idx[i][cs.mask[i]].tolist()
        assert i not in targets, "a particle must never learn from itself"
        assert len(targets) == len(set(targets)), "duplicate candidates inside one row"
        assert len(targets) >= 1
    assert set(np.unique(cs.source[cs.mask]).tolist()) <= set(range(len(CANDIDATE_SOURCES)))


def test_candidate_sources_cover_ring_elite_far_random() -> None:
    cfg = _cfg(n=30, dim=6)
    rng = np.random.default_rng(3)
    cs = build_candidates(rng.random((30, 6)), rng.random(30), make_function("sphere", 6, seed=0).bounds, rng)
    seen = set(np.unique(cs.source[cs.mask]).tolist())
    expected = {CANDIDATE_SOURCES.index(name) for name in ("ring", "elite", "far", "random")}
    assert seen == expected, cs.to_dict()
    # `self` is an explicit opt-in ablation, not part of the default candidate set
    assert CANDIDATE_SOURCES.index("self") not in seen


def test_candidate_builder_respects_source_flags() -> None:
    cfg = _cfg(n=20, dim=5)
    rng = np.random.default_rng(1)
    cs = build_candidates(rng.random((20, 5)), rng.random(20), make_function("sphere", 5, seed=0).bounds, rng,
                          use_source_flags={"far": False, "random": False})
    used = set(np.unique(cs.source[cs.mask]).tolist())
    assert CANDIDATE_SOURCES.index("far") not in used
    assert CANDIDATE_SOURCES.index("random") not in used


def test_degenerate_two_particle_swarm_still_works() -> None:
    cfg = _cfg(n=2, dim=3)
    pol, eng, state, obj = _run("cgl_full", cfg, iters=5)
    assert np.isfinite(state.pbest_val.min())
    assert obj.n_evals == 2 * 6


# ----------------------------------------------------------- contribution
def test_contribution_graph_discounted_update_is_exact() -> None:
    g = ContributionGraph(3, discount=0.5)
    g.note_selection(np.array([0, 1]), np.array([1, 2]), iteration=0)
    g.feedback(np.array([1.0, -1.0]), iteration=1)
    assert g.Q[0, 1] == pytest.approx(0.5 * 0.0 + 0.5 * 1.0)
    assert g.Q[1, 2] == pytest.approx(0.5 * 0.0 + 0.5 * -1.0)
    assert g.n_used[0, 1] == 1 and g.last_used[0, 1] == 1
    g.note_selection(np.array([0]), np.array([1]), iteration=1)
    g.feedback(np.array([1.0]), iteration=2)
    assert g.Q[0, 1] == pytest.approx(0.5 * 0.5 + 0.5 * 1.0)
    assert g.wins[0, 1] == 2


def test_contribution_graph_only_updates_selected_edges() -> None:
    """No counterfactual edges may be updated (that would need extra FEs)."""
    iters = 6
    cfg = _cfg(n=10, dim=5, iters=iters)
    pol, eng, state, obj = _run("cgl_full", cfg, iters=iters)
    g = pol.graph
    assert g is not None
    # the selection of the final iteration has no feedback yet (no future fitness)
    assert g.updates == iters - 1
    assert int(g.n_used.sum()) == (iters - 1) * 10
    assert int((g.n_used > 0).sum()) <= (iters - 1) * 10


def test_feedback_is_temporally_causal() -> None:
    """The gain written at iteration t+1 uses (state at t, state at t+1 select-time).

    Concretely: the decision made in iteration t is credited with the improvement
    that is visible when iteration t+1 starts -- no future fitness is ever read.
    """
    cfg = _cfg(n=6, dim=4, iters=4)
    pol = make_cgl_policy("cgl_full", cfg, seed=0)
    obj = CountingObjective(make_function("sphere", 4, seed=1), max_fes=None)
    eng = PSOEngine(obj, cfg, pol, seed=0)
    state = eng.initialize()
    state = eng.step()                      # iteration 1: selection recorded, no feedback yet
    assert pol._pending.valid
    prev_fit = pol._pending.fitness.copy()
    prev_pbest = pol._pending.pbest_val.copy()
    scale = pol._pending.scale
    src = pol.graph._pending_src.copy()
    dst = pol.graph._pending_dst.copy()
    assert src.size == 6 and dst.size == 6

    # state exactly as the policy will observe it at the start of iteration 2
    select_time_fit = state.fitness.copy()
    select_time_pbest = state.pbest_val.copy()
    training_landed = eng.step()            # iteration 2: feedback for iteration 1 is written
    assert pol.graph.updates == 1

    d_fit = prev_fit - select_time_fit
    d_fit = d_fit - np.mean(d_fit)                       # common-mode removal
    d_pbest = prev_pbest - select_time_pbest
    d_pbest = d_pbest - np.mean(d_pbest)
    # minimisation: the global-best improvement is min(pbest) before minus after
    global_gain = float(np.min(prev_pbest) - np.min(select_time_pbest))
    expected_gain = np.clip(
        (0.6 * d_fit + 0.3 * d_pbest + 0.1 * global_gain) / max(scale, 1e-12), -1.0, 1.0
    )
    q_expected = 0.3 * expected_gain   # discount 0.7 applied to a neutral (0) prior
    for s, d, ge in zip(src, dst, q_expected):
        assert pol.graph.Q[s, d] == pytest.approx(ge, abs=1e-9)
    # the *next* evaluation may not retroactively change what was written
    n_before = pol.graph.n_used[src, dst].copy()
    eng.step()
    assert np.all(pol.graph.n_used[src, dst] >= n_before)


def test_reward_leakage_absent_in_runner_contract() -> None:
    """Every iteration consumes exactly N fitness evaluations (no hidden trials)."""
    cfg = _cfg(n=9, dim=5, iters=7)
    for variant in ("cgl_full", "cgl_nocontrib", "quality_rank"):
        pol, eng, state, obj = _run(variant, cfg, iters=7)
        assert obj.n_evals == 9 * 8, variant


# ---------------------------------------------------------------- policy
@pytest.mark.parametrize("variant", CGL_POLICIES)
def test_every_variant_runs_with_identical_fes(variant: str) -> None:
    cfg = _cfg(n=12, dim=6, iters=8)
    pol, eng, state, obj = _run(variant, cfg, iters=8)
    assert obj.n_evals == 12 * 9
    assert np.isfinite(state.pbest_val.min())
    assert "cgl_edges_tried" in eng.trace[0]


def test_variant_switches_are_what_they_claim() -> None:
    cfg = _cfg()
    full = make_cgl_policy("cgl_full", cfg, seed=0)
    noc = make_cgl_policy("cgl_nocontrib", cfg, seed=0)
    nod = make_cgl_policy("cgl_nodirection", cfg, seed=0)
    nor = make_cgl_policy("cgl_norefresh", cfg, seed=0)
    qr = make_cgl_policy("quality_rank", cfg, seed=0)
    assert full.beta > 0 and full.gamma > 0 and full.use_refresh
    assert noc.beta == 0 and noc.gamma > 0 and noc.use_refresh
    assert nod.beta > 0 and nod.gamma == 0 and nod.use_refresh
    assert nor.beta > 0 and nor.gamma > 0 and not nor.use_refresh
    assert qr.beta == 0 and qr.gamma == 0 and qr.novelty == 0.0 and not qr.use_refresh


def test_degenerate_direction_is_zero() -> None:
    """Zero-length pbest direction must contribute exactly 0, not NaN."""
    cfg = _cfg(n=6, dim=3, iters=3)
    pol = make_cgl_policy("cgl_full", cfg, seed=0)
    obj = CountingObjective(make_function("sphere", 3, seed=0), max_fes=None)
    eng = PSOEngine(obj, cfg, pol, seed=0)
    state = eng.initialize()
    state.pos[:] = state.pbest_pos  # own direction is exactly zero for every particle
    pol.select(state, eng.ctx)
    diag = pol.iteration_trace[-1]
    assert diag["mean_direction"] == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(diag["mean_direction"])


def test_stagnation_triggers_refresh_and_is_logged() -> None:
    cfg = _cfg(n=10, dim=4, iters=12)
    pol, eng, state, obj = _run("cgl_full", cfg, iters=12, stall_threshold=3)
    assert pol._refresh_events, "stagnation must produce refresh events"
    assert any(t.get("cgl_refresh_events", 0) > 0 for t in eng.trace)
    assert all(0.0 <= t["cgl_stalled"] <= 1.0 for t in eng.trace)
    # the refresh mechanism must not restart particles
    assert np.all(state.iteration > 0)


def test_refresh_changes_behaviour_only_when_enabled() -> None:
    cfg = _cfg(n=10, dim=4, iters=10)
    a = _run("cgl_full", cfg, iters=10, stall_threshold=3)[0]
    b = _run("cgl_norefresh", cfg, iters=10, stall_threshold=3)[0]
    assert a._refresh_events and not b._refresh_events
    assert a.use_refresh and not b.use_refresh


def test_contribution_term_standardisation_has_spread_and_influence() -> None:
    cfg = _cfg(n=20, dim=10, iters=40)
    pol, eng, state, obj = _run("cgl_full", cfg, iters=40)
    tr = pol.iteration_trace[10:]   # after the graph has seen some feedback
    stds = [t["contrib_term_std"] for t in tr]
    flips = [t["contrib_decision_flip_rate"] for t in tr]
    assert np.mean(stds) > 0.05, f"contribution term has no spread: {np.mean(stds)}"
    assert 0.0 <= np.mean(flips) <= 1.0


def test_policy_rng_is_isolated_from_engine_and_global_state() -> None:
    cfg = _cfg(n=10, dim=5, iters=6)
    a = _run("cgl_full", cfg, iters=6, seed=1)[2]
    b = _run("cgl_full", cfg, iters=6, seed=1)[2]
    assert np.array_equal(a.pbest_pos, b.pbest_pos), "same seed must reproduce exactly"
    c = _run("cgl_full", cfg, iters=6, seed=2)[2]
    assert not np.array_equal(a.pbest_pos, c.pbest_pos)


def test_cgl_matches_classic_initial_population() -> None:
    cfg = _cfg(n=12, dim=5)
    pol = make_cgl_policy("cgl_full", cfg, seed=0)
    obj = CountingObjective(make_function("sphere", 5, rotate=True, shift=0.25, seed=70000), max_fes=None)
    s1 = PSOEngine(obj, cfg, pol, seed=0).initialize()
    obj2 = CountingObjective(make_function("sphere", 5, rotate=True, shift=0.25, seed=70000), max_fes=None)
    s2 = PSOEngine(obj2, cfg, make_classic_policy("ring", cfg), seed=0).initialize()
    assert np.array_equal(s1.pos, s2.pos)
    assert np.array_equal(s1.pbest_val, s2.pbest_val)


def test_selection_cost_is_bounded() -> None:
    """The vectorised selector must stay well under a millisecond at dev scale."""
    cfg = _cfg(n=40, dim=30, iters=40)
    pol, eng, state, obj = _run("cgl_full", cfg, iters=40)
    mean_ms = float(np.mean(pol._update_ms[-30:]))
    assert mean_ms < 1.5, f"CGL selection is too slow: {mean_ms:.3f} ms"


# ------------------------------------------------------- protocol/budget
def test_registered_task_and_instance_separation() -> None:
    assert not (set(PILOT_TASKS) & set(DEV_TASKS)), "pilot and dev tasks must be disjoint"
    # the 70000 namespace must not collide with any earlier stage's instance seeds
    earlier = set()
    for base in (11000, 12000, 20240, 41000, 51000, 61000, 90000):
        for task in ("sphere", "rastrigin", "ackley", "rosenbrock", "griewank", "levy",
                     "ellipsoid", "schwefel", "zakharov", "alpine1", "hybrid_rz", "styblinski_tang"):
            for seed in range(6):
                earlier.add(stable_seed(base, task, seed))
    cgl = {stable_seed(INSTANCE_SEED_BASE, task, seed) for task in DEV_TASKS for seed in DEV_SEEDS}
    assert not (cgl & earlier), "CGL instance namespace overlaps an earlier stage"


def test_budget_caps_in_configs() -> None:
    dev = load_config("configs/cgl/dev10.yaml")
    assert int(dev.get_path("budget.max_total_fes")) <= 4_000_000
    assert int(dev.get_path("budget.max_algorithms_dev10")) <= 8
    assert len(dev.get_path("screening.algorithms")) <= 8
    assert len(dev.get_path("screening.dev10.tasks")) == 4
    assert len(dev.get_path("screening.dev10.seeds")) == 10
    assert int(dev.get_path("screening.dev10.fes")) == 10_000


def test_cli_rejects_more_than_eight_algorithms() -> None:
    from typer.testing import CliRunner

    from cgl_pso.cli import app

    res = CliRunner().invoke(app, ["dev10", "-c", "configs/cgl/dev10.yaml", "--seeds", "1", "--fes", "400",
                                   *sum([["--algorithm", a] for a in CGL_ALGORITHMS + ("ring",)], [])])
    assert res.exit_code != 0
    assert "8 algorithms" in str(res.output) or "at most 8" in str(res.output)


def test_screening_verdict_semantics() -> None:
    import pandas as pd

    rows = []
    for seed in range(4):
        for task in ("t1", "t2", "t3", "t4"):
            base = {"seed": seed, "task": task, "fes": 100, "runtime_s": 0.1, "success_loose": False,
                    "error": 1.0, "log10_error": 1.0}
            rows.append({**base, "algorithm": "cgl_nocontrib", "log10_error": 1.0})
            rows.append({**base, "algorithm": "cgl_full", "log10_error": 0.5, "runtime_s": 0.25})
            rows.append({**base, "algorithm": "ring", "log10_error": 1.2, "runtime_s": 0.1})
            rows.append({**base, "algorithm": "gbest", "log10_error": 1.1, "runtime_s": 0.1})
    df = pd.DataFrame(rows)
    v = screening_verdict(df)
    assert v["tasks_improved_vs_ablation"] == 4
    assert v["median_runtime_factor_vs_classics"] == pytest.approx(2.5, abs=0.01)
    assert v["passed"] is True

    df2 = df.copy()
    df2.loc[df2["algorithm"] == "cgl_full", "log10_error"] = 1.5   # now worse than the ablation
    assert screening_verdict(df2)["passed"] is False

    df3 = df.copy()
    df3.loc[df3["algorithm"] == "cgl_full", "runtime_s"] = 0.9      # 9x the classics
    assert screening_verdict(df3)["passed"] is False

    df4 = df.copy()
    df4.loc[df4["algorithm"] == "cgl_full", "log10_error"] = 2.0    # clearly worse than ring/gbest
    assert screening_verdict(df4)["passed"] is False


# ---------------------------------------------------------------------------
# Regression tests for the four review findings (2026-10-20)
# ---------------------------------------------------------------------------
def test_global_best_gain_uses_min_not_max() -> None:
    """Regression (review P0): the global-best term must use min for minimisation.

    The previous implementation used ``max(pbest)`` and therefore credited edges
    with the *worst* particle's movement. The test below constructs a step where the
    two formulas differ and asserts the stored contribution matches the min-based
    (correct) definition.
    """
    cfg = _cfg(n=6, dim=4, iters=3)
    pol = make_cgl_policy("cgl_full", cfg, seed=0)
    obj = CountingObjective(make_function("sphere", 4, seed=1), max_fes=None)
    eng = PSOEngine(obj, cfg, pol, seed=0)
    state = eng.initialize()
    state = eng.step()
    prev_pbest = pol._pending.pbest_val.copy()
    prev_fit = pol._pending.fitness.copy()
    scale = pol._pending.scale
    src = pol.graph._pending_src.copy()
    dst = pol.graph._pending_dst.copy()
    select_time_pbest = state.pbest_val.copy()
    select_time_fit = state.fitness.copy()
    eng.step()

    d_fit = prev_fit - select_time_fit
    d_fit = d_fit - np.mean(d_fit)
    d_pbest = prev_pbest - select_time_pbest
    d_pbest = d_pbest - np.mean(d_pbest)
    gain_min = np.clip((0.6 * d_fit + 0.3 * d_pbest + 0.1 * (np.min(prev_pbest) - np.min(select_time_pbest))) / scale, -1, 1)
    gain_max = np.clip((0.6 * d_fit + 0.3 * d_pbest + 0.1 * (np.max(prev_pbest) - np.max(select_time_pbest))) / scale, -1, 1)
    assert not np.allclose(gain_min, gain_max), "the test step must distinguish min from max"
    for s_i, d_i, g in zip(src, dst, gain_min):
        assert pol.graph.Q[s_i, d_i] == pytest.approx(0.3 * g, abs=1e-9)


def test_stagnation_penalty_only_touches_stalled_particles() -> None:
    """Regression (review P0): M3 is stagnation-triggered, so non-stalled particles
    must receive exactly the same scores whether the refresh is enabled or not."""
    cfg = _cfg(n=8, dim=4, iters=6)
    for variant, use_refresh in (("cgl_full", True), ("cgl_norefresh", False)):
        pol = make_cgl_policy(variant, cfg, seed=0, stall_threshold=2)
        obj = CountingObjective(make_function("sphere", 4, seed=1), max_fes=None)
        eng = PSOEngine(obj, cfg, pol, seed=0)
        state = eng.initialize()
        for _ in range(3):
            state = eng.step()
        # force a mixed population and make the historical edges unproductive
        state.stall[:] = 0.0
        state.stall[:4] = 5.0                     # first half stalled, second half improving
        pol.graph.Q[:] = -0.5
        pol.graph.n_used[:] = 1
        pol.graph.last_used[:] = state.iteration
        order = np.argsort(state.pbest_val, kind="stable")
        rank_of = np.empty(state.n, dtype=np.float64)
        rank_of[order] = np.arange(state.n)
        cands = build_candidates(state.pos, state.pbest_val, eng.ctx.bounds, np.random.default_rng(0),
                                 pbest_order=order)
        score, eps, diag = pol._score_candidates(state, cands, stalled=state.stall >= pol.stall_threshold,
                                                 ranks=rank_of / max(state.n - 1, 1))
        if use_refresh:
            stalled_mask = state.stall >= pol.stall_threshold
            # non-stalled rows keep their unpenalised scores -> compare against the
            # same computation with refresh disabled
            pol2 = make_cgl_policy("cgl_norefresh", cfg, seed=0)
            pol2.graph = pol.graph
            score_ref, _, _ = pol2._score_candidates(state, cands, stalled=stalled_mask,
                                                     ranks=rank_of / max(state.n - 1, 1))
            for i in range(state.n):
                if not stalled_mask[i]:
                    valid = cands.mask[i]
                    assert np.allclose(score[i][valid], score_ref[i][valid]), f"non-stalled row {i} was penalised"
            penalised = [
                i for i in range(state.n)
                if stalled_mask[i] and not np.allclose(score[i][cands.mask[i]], score_ref[i][cands.mask[i]])
            ]
            assert penalised, "stalled particles must actually be penalised"


def test_farthest_candidate_is_truly_farthest_and_never_self() -> None:
    """Regression (review P1): a self-draw used to win the argmax of the long-range
    slot (distance was set to +inf) and was then silently filtered out."""
    cfg = _cfg(n=6, dim=3)
    rng = np.random.default_rng(0)
    pos = rng.random((6, 3)) * 10.0
    pbest = rng.random(6)
    bounds = make_function("sphere", 3, seed=0).bounds
    for seed in range(5):
        cs = build_candidates(pos, pbest, bounds, np.random.default_rng(seed),
                              n_far_samples=40, top_q=0, ring_radius=0, n_random=0)
        far_source = CANDIDATE_SOURCES.index("far")
        assert cs.n_candidates.min() == 1
        for i in range(6):
            slot = int(np.flatnonzero(cs.source[i] == far_source)[0])
            chosen = int(cs.idx[i, slot])
            assert chosen != i, "the long-range slot must never be the particle itself"
            d = np.linalg.norm(pos[chosen] - pos[i])
            d_best_non_self = np.max([np.linalg.norm(pos[j] - pos[i]) for j in range(6) if j != i])
            assert d == pytest.approx(d_best_non_self), "with 40 samples of 6 particles the farthest must be found"


def test_self_candidate_is_optional() -> None:
    """Regression (review P1): ring allows the neighbourhood best to be the particle
    itself; CGL removed self unconditionally. It is now an explicit ablation."""
    cfg = _cfg(n=6, dim=3)
    rng = np.random.default_rng(1)
    pos = rng.random((6, 3))
    pbest = rng.random(6)
    bounds = make_function("sphere", 3, seed=0).bounds
    no_self = build_candidates(pos, pbest, bounds, np.random.default_rng(2), include_self=False)
    with_self = build_candidates(pos, pbest, bounds, np.random.default_rng(2), include_self=True)
    self_source = CANDIDATE_SOURCES.index("self")
    for i in range(6):
        assert i not in no_self.idx[i][no_self.mask[i]].tolist()
        targets = with_self.idx[i][with_self.mask[i]].tolist()
        assert targets.count(i) == 1, "exactly one self slot per particle"
        slot = int(np.flatnonzero(with_self.source[i] == self_source)[0])
        assert int(with_self.idx[i, slot]) == i and bool(with_self.mask[i, slot])

    # policy level: with the switch on, a particle may choose its own pbest
    pol = make_cgl_policy("cgl_full", cfg, seed=0, allow_self_candidate=True)
    assert pol.allow_self_candidate is True
    obj = CountingObjective(make_function("sphere", 3, seed=1), max_fes=None)
    eng = PSOEngine(obj, cfg, pol, seed=0)
    state = eng.initialize()
    for _ in range(30):
        state = eng.step()
    assert pol.summary()["allow_self_candidate"] is True
