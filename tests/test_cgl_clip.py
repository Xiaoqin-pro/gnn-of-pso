"""CLIP application-appendix tests (synthetic smoke, splits, objective, leakage)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cgl_pso.clip_app import (
    SoftmaxTemplateObjective,
    _build_objective,
    evaluate_final,
    metrics,
    optimize,
    prepare,
    split_indices,
    synthetic_logits,
)
from cgl_pso._infra.config import Config, load_config


def _cfg() -> Config:
    cfg = load_config("configs/cgl/clip.yaml")
    cfg.set_path("clip.synthetic.n_per_class", 60)
    cfg.set_path("optimizer.n_particles", 10)
    return cfg


# ------------------------------------------------------------------ data
def test_synthetic_logits_are_deterministic_and_structured() -> None:
    cfg = _cfg()
    a = synthetic_logits(cfg)
    b = synthetic_logits(cfg)
    assert np.array_equal(a["log_probs"], b["log_probs"])
    assert np.array_equal(a["labels"], b["labels"])
    strength = np.asarray(a["template_strength"])
    assert (strength < 0).sum() >= 1, "the smoke must contain a misleading template"
    assert strength.max() > 0.5, "the smoke must contain informative templates"
    # log-probabilities must be normalised per (sample, template)
    p = np.exp(a["log_probs"])
    assert np.allclose(p.sum(axis=2), 1.0, atol=1e-9)


def test_split_is_disjoint_and_complete() -> None:
    cfg = _cfg()
    data = synthetic_logits(cfg)
    split = split_indices(len(data["labels"]), cfg, seed=0)
    sets = {k: set(v.tolist()) for k, v in split.items()}
    assert not (sets["calibration"] & sets["dev"])
    assert not (sets["calibration"] & sets["test"])
    assert not (sets["dev"] & sets["test"])
    union = sets["calibration"] | sets["dev"] | sets["test"]
    assert union == set(range(len(data["labels"])))


def test_prepare_does_not_write_test_labels(tmp_path) -> None:
    cfg = _cfg()
    cfg.set_path("clip.backend", "synthetic")
    summary = prepare(cfg, out_dir=tmp_path)
    text = (tmp_path / "cgl" / "clip" / "split.json").read_text(encoding="utf-8")
    payload = json.loads(text)
    assert "labels" not in payload
    assert set(payload["split"]) == {"calibration", "dev", "test"}
    assert summary["backend"] == "synthetic"
    assert "no model weights" in summary["note"] or "synthetic" in summary["note"]


# ------------------------------------------------------------- objective
def test_objective_dimension_and_bounds() -> None:
    cfg = _cfg()
    data = synthetic_logits(cfg)
    split = split_indices(len(data["labels"]), cfg, seed=0)
    obj = _build_objective(cfg, data, split["calibration"])
    assert obj.dim == data["n_templates"] + int(cfg.get_path("clip.n_calibration_params", 4))
    assert obj.bounds.low.shape == (obj.dim,)
    assert obj.bounds.high.shape == (obj.dim,)
    assert np.all(obj.bounds.low < obj.bounds.high)


def test_softmax_parameterisation_is_valid() -> None:
    cfg = _cfg()
    data = synthetic_logits(cfg)
    split = split_indices(len(data["labels"]), cfg, seed=0)
    obj = _build_objective(cfg, data, split["dev"])
    w, (temp, bias_scale) = obj._decode(np.array([5.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert np.all(w >= 0.0) and w.sum() == pytest.approx(1.0)
    assert 0.25 <= temp <= 4.0
    assert 0.0 <= bias_scale <= 1.0


def test_equal_weights_are_worse_than_an_optimised_point() -> None:
    """The synthetic objective must actually have optimisation signal."""
    from cgl_pso._infra.objectives.base import CountingObjective
    from cgl_pso._infra.optim import PSOEngine
    from cgl_pso.policy import make_cgl_policy

    cfg = _cfg()
    data = synthetic_logits(cfg)
    split = split_indices(len(data["labels"]), cfg, seed=0)
    obj = _build_objective(cfg, data, split["calibration"])
    dev = _build_objective(cfg, data, split["dev"])
    baseline = dev.report(np.zeros(obj.dim), dev.index)["macro_f1"]
    counting = CountingObjective(obj, max_fes=500)
    engine = PSOEngine(counting, cfg, make_cgl_policy("cgl_full", cfg, seed=0), seed=0)
    res = engine.run(iters=40, max_fes=500)
    optimised = dev.report(res.best_pos, dev.index)["macro_f1"]
    assert optimised > baseline, (baseline, optimised)


def test_metrics_are_bounded() -> None:
    rng = np.random.default_rng(0)
    lp = rng.normal(size=(50, 5))
    y = rng.integers(0, 5, size=50)
    m = metrics(lp, y)
    assert 0.0 <= m["macro_f1"] <= 1.0
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["ece"] <= 1.0


# -------------------------------------------------------------- pipeline
def test_clip_pipeline_end_to_end(tmp_path) -> None:
    cfg = _cfg()
    cfg.set_path("clip.synthetic.n_per_class", 40)
    cfg.set_path("optimizer.n_particles", 8)
    summary = optimize(cfg, out_dir=tmp_path, algorithms=["equal_weights", "ring", "cgl_full"], budget=200)
    assert summary["budget_per_method"] == 200
    assert "test" not in summary, "optimisation must not touch the test split"
    assert summary["selected_on_dev"] in {r["method"] for r in summary["methods"]}
    final = evaluate_final(cfg, out_dir=tmp_path)
    assert final["test_used_once"] is True
    assert len(final["results"]) >= 3
    for r in final["results"]:
        assert 0.0 <= r["macro_f1"] <= 1.0


def test_evaluate_final_requires_optimize_first(tmp_path) -> None:
    cfg = _cfg()
    with pytest.raises(FileNotFoundError):
        evaluate_final(cfg, out_dir=tmp_path)


def test_optimize_respects_the_unified_budget(tmp_path) -> None:
    cfg = _cfg()
    cfg.set_path("clip.synthetic.n_per_class", 30)
    cfg.set_path("optimizer.n_particles", 10)
    summary = optimize(cfg, out_dir=tmp_path, algorithms=["random_search", "ring"], budget=150)
    for r in summary["methods"]:
        assert r["n_evals"] <= 150 + 10, (r["method"], r["n_evals"])


def test_real_backend_is_not_silently_faked(tmp_path) -> None:
    """Requesting a real backend without permission must run the smoke and say so."""
    cfg = _cfg()
    cfg.set_path("clip.backend", "open_clip")
    cfg.set_path("clip.budget.allow_download", False)
    summary = prepare(cfg, out_dir=tmp_path)
    assert summary["backend"] == "open_clip"
    assert "not executed" in summary["note"] and "synthetic" in summary["note"]
