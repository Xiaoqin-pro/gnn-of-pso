# CGL-PSO — Contribution-Guided Graph Learning Particle Swarm Optimization

Development-stage research code for **CGL-PSO**: the social learning relation between PSO
particles is an **online, directed, sparse graph** whose edge weights are updated from the
search feedback that was *actually realised*. Three small, ablatable mechanisms on top of the
standard PSO velocity update (no GNN, no PPO, no offline training):

| Mechanism | What it does |
|---|---|
| **M1 online contribution graph** | directed edge `i -> j` keeps usage count `n_ij`, discounted return `Q_ij` (from realised gains only) and last-use time; neutral initialisation; the observed association is **not** claimed to be a causal contribution |
| **M2 quality + direction scoring** | candidates from ring neighbours, top-pbest elites, an approximate-farthest slot and a few random slots (no self-loops, no duplicates); bounded score `a*q_j + b*Q_ij + c*cos(dir) + d*novelty` inside a softmax with an exploration floor |
| **M3 stagnation-triggered refresh** | a stalled particle temporarily down-weights its recently used unproductive edges and raises its exploration floor; particles are **never restarted**, no chaos/Levy/opposition terms |

## Headline result (honest, negative)

The pre-registered development gate was **not passed**, so this route is **stopped**:

* main screening (4 training-external tasks x 10 paired seeds x 30D x 40 particles x 10,000 FEs
  x 8 algorithms = 3.2M FEs): `cgl_full` improves on the no-contribution ablation in **3/4** tasks
  (paired median Δlog10 −0.096 … +0.015) but is **clearly worse than `ring`/`gbest` in 3/4** tasks
  and **6.5x slower** than the faster classical baseline (gate: ≤ 3x);
* the single allowed repair round (removing the harmful far/random candidate sources, 480k FEs)
  improved absolute performance on 4/4 tasks yet **removed M1's advantage** (1/4) →
  contribution memory has **no robust net benefit**;
* budget ledger is reported in full, **including an overspend** (4,155,000 FEs vs the 4M cap);
  no further numeric experiment was run after that.

Per-task `mean log10(error + 1e-8)` (10 seeds, 30-D, 10,000 FEs):

| task | ring | gbest | gldi_rule | quality_rank | cgl_nocontrib | cgl_full |
|---|---|---|---|---|---|---|
| alpine1 | 0.842 | **0.523** | 0.504 | 1.208 | 1.202 | 1.195 |
| griewank | 0.070 | **-0.717** | -0.015 | 1.015 | 1.211 | 1.187 |
| rosenbrock | 1.990 | 1.673 | **1.613** | 2.181 | 2.200 | 2.145 |
| zakharov | 1.893 | **1.418** | 1.602 | 1.745 | 1.758 | 1.723 |

## Review round (2026-10-20): implementation fixes + Ring-SmallWorld-PSO

A static review of this repository at commit `082ce2f` found three implementation bugs and one
design choice worth ablating. All four are fixed here with regression tests; the before/after
record is in `docs/CGL_REVIEW_RESPONSE.md`.

| item | before | after |
|---|---|---|
| global-best gain used `max(pbest)` on a minimisation problem | `max(pbest) - max(pbest)` | `min(pbest) - min(pbest)` |
| stagnation penalty ignored the `stalled` mask | penalised every particle's recent unproductive edges | `bad_edge &= stalled[:, None]` |
| the "farthest" candidate scored a self-draw as `+inf` | self won the `argmax`, then the self-loop filter dropped the slot | self scored `-inf`, deterministic fallback when every draw is self |
| CGL forbade self-pbest candidates while `ring` allows them | hard-coded | `allow_self_candidate` switch (default off) |

Re-running 4 tasks x 10 paired seeds x 30-D x 40 particles x 10,000 FEs shows the fixes do **not**
rescue CGL: mean log10(error+1e-8) moves from 1.5627 (pre-fix) to 1.5683 (fixed), still clearly
worse than `ring` (1.199) and `gbest` (0.724) and still 6.56x slower. `ring`/`gbest` reproduce the
frozen dev10 values **bit-for-bit**, so that comparison is clean (`scripts/cgl_sw_verify_repro.py`).

The same review asked for a lightweight **Ring-SmallWorld-PSO** prototype - no GNN, no PPO, no
reward matrix, no sub-swarms: the verified ring topology plus a sparse non-local elite link used
with probability `p`, either always (`fixed`), only for stalled particles (`stall`), or adjusted
online from realised feedback (`bandit`).

| task | ring | gbest | `sw_fixed_p15` | `sw_stall_t10_p05` | `sw_bandit_p05` | `cgl_full_fixed` |
|---|---|---|---|---|---|---|
| alpine1 | 0.842 | **0.523** | 0.642 | 0.749 | 0.893 | 1.211 |
| griewank | 0.070 | **-0.717** | 0.048 | 0.075 | 0.543 | 1.145 |
| rosenbrock | 1.990 | **1.673** | 1.878 | 2.005 | 2.004 | 2.173 |
| zakharov | 1.893 | **1.418** | 1.571 | 1.753 | 1.671 | 1.744 |
| **mean** | 1.199 | **0.724** | 1.035 | 1.146 | 1.278 | 1.568 |

* pre-registered gate: >= 1 task improved vs `ring`, no task collapsed (> +0.5), runtime <= 1.5x,
  and the online variant must beat the fixed one to be kept;
* `sw_stall_t10_p05` (tau=10, p=0.05) is the **only** setting that passes (2/4 tasks improved,
  1.44x runtime); `sw_fixed_p15` improves 4/4 but costs 1.72x and halves swarm diversity
  (0.085 -> 0.034); `sw_bandit_p05` helps 1/4, so the **online module is dropped**;
* honest framing: the gain is concentrated on `zakharov` (10W/0L for every small-world variant)
  while griewank/rosenbrock at best tie - this is "cheap long-range links do not hurt", **not**
  "small-world beats ring";
* budget ledger: 960k (pre-trial) + 240k (selection diagnostics) + 2.4M (main) = **3.6M of the
  3.8M cap**. No CEC suite, no 30-seed expansion, no real CLIP run; limits are listed in
  `docs/CGL_SMALLWORLD_RESULTS_zh.md`.

**Application appendix (CLIP)**: the real frozen-VLM experiment was **not started** (the core gate
failed and weight downloads require permission). A complete **synthetic-logits calibration smoke**
is included instead — 8 methods, unified 2,000 objective evaluations, selection on the dev split
only, single test use: logistic calibration 0.8415 > **cgl_full 0.6688** > gbest 0.6588 >
ring 0.6531 > scipy_de 0.6507 > cgl_nocontrib 0.6504 > random 0.6101 > equal weights 0.3662.

## Repository layout

```
src/cgl_pso/
  contribution.py    M1: online contribution graph (exhaustive unit tests)
  candidates.py      sparse candidate construction (vectorised)
  policy.py          CGL variants + policy factory
  smallworld.py      Ring-SmallWorld-PSO prototype (fixed / stall / bandit links)
  baselines.py       fixed-score rule reference (no feedback)
  runner.py          pilot / pre-registered dev10 screening, statistics, figures
  clip_app.py        frozen-VLM application appendix (synthetic smoke + gated real backends)
  reporting.py       regenerates the result/decision documents from the artefacts
  cli.py             cgl doctor|smoke|dev10|report|clip-prepare|clip-optimize|clip-evaluate-final
  _infra/            trimmed, mechanically renamed subset of the reused, verified infrastructure
configs/cgl/         smoke.yaml, dev10.yaml, clip.yaml, smallworld.yaml
results/cgl/         raw JSONL/CSV, paired tables, figures (PNG+SVG), resource logs
docs/CGL_*.md        literature matrix, method, protocol, results, decision, CLIP appendix
tests/               candidates/graph updates/temporal causality/FE accounting/budget/CLIP splits
```

`_infra/` contains **only** what CGL-PSO needs (benchmark functions with verified optima, FE
counting, the batched PSO engine, seeding, resource monitor/guard, config). It is a byte-level
copy of the parent project's modules with the import prefix rewritten; **no GNN, PPO, robot or
earlier-stage code is included**. Because the reused infrastructure logs CUDA memory through
torch, `torch` stays a dependency even though the algorithm itself is pure NumPy.

## Install and run

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"      # Linux/macOS: .venv/bin/pip
python -m pytest tests -q                  # 56 tests

cgl doctor
cgl dev10 -c configs/cgl/dev10.yaml --pilot        # pilot scale (3 seeds x 3 tasks x 5,000 FEs)
cgl report                                          # regenerates docs/CGL_RESULTS_zh.md + CGL_DECISION.md
cgl clip-prepare -c configs/cgl/clip.yaml --backend synthetic
cgl clip-optimize -c configs/cgl/clip.yaml --budget 2000
cgl clip-evaluate-final -c configs/cgl/clip.yaml
```

**Do not re-run the main screening**: it is a pre-registered, budgeted experiment
(`docs/CGL_PROTOCOL.md`) whose numbers are already recorded in `results/cgl/dev10/`.

## Documentation

| file | content |
|---|---|
| `docs/CGL_LITERATURE.md` | verified references with DOIs (Chauhan 2025, PDML-PSO, EMPSO, APLPSO, RL-MOPSO, GECCO 2008 bandit operator selection, ICML 2023 prompt weighting), overlap risk, novelty boundary — **no first-claim assertions** |
| `docs/CGL_METHOD.md` | formulas (gain, discounted edge update, standardised contribution term, score, exploration floor, refresh), pseudocode, complexity, ablation matrix |
| `docs/CGL_PROTOCOL.md` | pre-registered tasks/seeds/instances, fairness contract, budget ledger, gates, resource policy |
| `docs/CGL_RESULTS_zh.md` | auto-generated: pilot, main screening, repair round, mechanism statistics, AUC, runtime, budget overspend, CLIP smoke |
| `docs/CGL_DECISION.md` | the explicit STOP verdict with its four evidence points and the reusable outcomes |
| `docs/CGL_REVIEW_RESPONSE.md` | the 2026-10-20 review: four findings, before/after code, regression tests, effect on the verdict |
| `docs/CGL_SMALLWORLD_RESULTS_zh.md` | Ring-SmallWorld-PSO: pre-trial, selection diagnostics, main screening, budget ledger, limits |
| `docs/CGL_CLIP_APPLICATION.md` | application design, why the real CLIP run was not started, synthetic smoke results |

## Reused infrastructure vs. this work

Provenance is stated explicitly so nothing is mis-attributed:

* **New in this repository**: all three CGL mechanisms, the five ablatable variants, the
  candidate builder, the screening runner and statistics, the CLIP calibration appendix,
  the CLI, the documentation and the tests.
* **Reused (trimmed copy, prefix renamed)**: benchmark functions and optimum checks, FE
  counting, the batched PSO engine and the ring/gbest policies, seeding, resource
  monitor/guard and the config loader. They are unchanged apart from the import prefix.
* Prior GNN/PPO experiments are **not** part of this repository; they are referenced only as
  failure experience in `docs/CGL_DECISION.md`.

## Export verification (why the numbers here are trustworthy)

`scripts/verify_export.py` re-runs four single cells of the committed screenings
(`alpine1`, seed 0: `gldi_rule`, `cgl_full_fixed`, `cgl_full`, `cgl_nocontrib`) inside this
repository, on the same instance/initialisation/budget, and compares them with the artefacts:

| check | committed | re-run here | outcome |
|---|---|---|---|
| reuse-only anchor `gldi_rule` vs `results/cgl/dev10` | 0.838597 | 0.838597 | **exact** |
| fixed code `cgl_full_fixed` vs `results/cgl/sw_main` | 1.235970 | 1.235970 | **exact** |
| pre-fix `cgl_full` (dev10, frozen) | 1.115213 | 1.224636 | differs by design |
| pre-fix `cgl_nocontrib` (dev10, frozen) | 1.160373 | 1.281446 | differs by design |

The first two rows are the verification: the reuse-only path is bit-stable, and the fixed
implementation reproduces the recorded review-round cell. The last two rows are shown on purpose
- they are the pre-fix cells the review flagged, kept as historical evidence and deliberately not
reproducible by the fixed code.

`scripts/cgl_sw_verify_repro.py` adds a **zero-FE** check on the committed artefacts: `ring`/`gbest`
of the review round match the frozen dev10 values bit-for-bit, the small-world verdict recomputes
to the same recommendation, and the budget ledger stays inside its cap.
`scripts/cgl_sw_resummarise.py` regenerates `summary.json` / `smallworld_verdict.json` from the raw
`runs.jsonl` without re-running anything.

Two robustness bugs found by this verification were fixed in the screening's verdict helper
(subset runs without classic baselines no longer crash on `pd.concat([])` / missing columns, and
such runs now mark the CGL gate explicitly as `"applicable": false` instead of reporting a plain
failure).

## License

MIT (as declared in the parent project's `pyproject.toml`). Model weights or datasets used by
any future CLIP experiment keep their own licenses and are not redistributed here.
