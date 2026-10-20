"""Export verification: re-run four single cells here and compare with the committed artefacts.

Same instance, initialisation and budget as the committed runs (alpine1, seed 0, 30-D,
40 particles, 10,000 FEs). Three checks:

1. **anchor** - ``gldi_rule`` must still equal the frozen pre-fix value in
   ``results/cgl/dev10/runs.csv``: the reuse-only path is untouched by the review fixes.
2. **fixed code** - ``cgl_full_fixed`` must equal ``results/cgl/sw_main/runs.csv``: this export
   carries the fixed implementation and reproduces the recorded screening cell exactly.
3. **expected divergence** - ``cgl_full`` / ``cgl_nocontrib`` re-run here use the *same* label
   and seed derivation as the frozen dev10 cells, but the fixes changed their trajectory, so a
   mismatch is the correct outcome; the frozen numbers are kept for the record.

Usage (repository root)::

    .venv\\Scripts\\python.exe scripts/verify_export.py
"""

from __future__ import annotations

import sys

import pandas as pd

from cgl_pso._infra.config import load_config
from cgl_pso.runner import run_sweep

TASK, SEED, DIM, PARTICLES, FES = "alpine1", 0, 30, 40, 10_000
SPECS = [
    ("gldi_rule", "gldi_rule", {}),
    ("cgl_full_fixed", "cgl_full", {}),
    ("cgl_full", "cgl_full", {}),
    ("cgl_nocontrib", "cgl_nocontrib", {}),
]

cfg = load_config("configs/cgl/smallworld.yaml")
res = run_sweep(cfg, specs=SPECS, tasks=(TASK,), seeds=(SEED,), dim=DIM, n_particles=PARTICLES,
                fes=FES, out_dir="results", tag="verify", progress=False, write_figures=False)
new = res.runs

frozen = pd.read_csv("results/cgl/dev10/runs.csv")      # pre-fix screening (frozen evidence)
fixed = pd.read_csv("results/cgl/sw_main/runs.csv")     # review-round screening (fixed code)


def cell(df: pd.DataFrame, algo: str) -> float | None:
    row = df[(df.task == TASK) & (df.seed == SEED) & (df.algorithm == algo)]
    return float(row.iloc[0]["log10_error"]) if len(row) else None


rerun = {label: cell(new, label) for label, _, _ in SPECS}
print(f"re-ran {len(new)} cells here ({TASK}, seed {SEED}, {DIM}-D, {PARTICLES} particles, {FES:,} FEs)\n")
ok = True

print("check 1 - anchor / reuse-only path (must match)")
ref, got = cell(frozen, "gldi_rule"), rerun["gldi_rule"]
match = ref is not None and got is not None and abs(ref - got) < 1e-12
ok &= match
print(f"   gldi_rule       dev10={ref:.6f}  re-run={got:.6f}  delta={got - ref:+.2e}  match={match}\n")

print("check 2 - fixed implementation (must match the review-round artefact)")
ref, got = cell(fixed, "cgl_full_fixed"), rerun["cgl_full_fixed"]
match = ref is not None and got is not None and abs(ref - got) < 1e-12
ok &= match
print(f"   cgl_full_fixed  sw_main={ref:.6f}  re-run={got:.6f}  delta={got - ref:+.2e}  match={match}\n")

print("check 3 - expected divergence for the pre-fix cells (mismatch is correct here)")
for algo in ("cgl_full", "cgl_nocontrib"):
    ref, got = cell(frozen, algo), rerun[algo]
    print(f"   {algo:14s} dev10(pre-fix)={ref:.6f}  re-run(fixed code)={got:.6f}  delta={got - ref:+.2e}")
print("   these frozen pre-fix numbers are kept as historical evidence and are documented in"
      " docs/CGL_REVIEW_RESPONSE.md\n")

print("ANCHOR MATCHES:", bool(ok))
print("EXPORT IS CONSISTENT WITH THE COMMITTED ARTEFACTS:", bool(ok))
sys.exit(0 if ok else 1)
