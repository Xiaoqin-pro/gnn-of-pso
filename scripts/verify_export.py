import sys, io, json, pathlib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pandas as pd
from cgl_pso._infra.config import load_config
from cgl_pso.runner import run_screening

committed = pd.read_csv("results/cgl/dev10/runs.csv")
cells = [("alpine1", 0, "gldi_rule"), ("alpine1", 0, "cgl_full"), ("alpine1", 0, "cgl_nocontrib")]
algos = tuple(sorted({c[2] for c in cells}))
cfg = load_config("configs/cgl/dev10.yaml")
res = run_screening(cfg, tasks=("alpine1",), seeds=(0,), algorithms=algos, dim=30, n_particles=40,
                    fes=10_000, out_dir="results", tag="verify", progress=False, write_figures=False)
new = res.runs
print(f"{'algo':16s} {'committed log10':>16s} {'export log10':>14s} {'Δ':>10s} {'FEs':>7s} {'match':>6s}")
ok = True
for task, seed, algo in cells:
    a = committed[(committed.task == task) & (committed.seed == seed) & (committed.algorithm == algo)]
    b = new[(new.task == task) & (new.seed == seed) & (new.algorithm == algo)]
    if a.empty or b.empty:
        print(f"{algo:16s} MISSING (committed={len(a)}, export={len(b)})"); ok = False; continue
    la, lb = float(a.iloc[0]["log10_error"]), float(b.iloc[0]["log10_error"])
    match = abs(la - lb) < 1e-12
    ok &= match
    print(f"{algo:16s} {la:16.6f} {lb:14.6f} {lb - la:+10.2e} {int(b.iloc[0]['fes']):7d} {str(match):>6s}")
print()
print("EXPORT REPRODUCES THE COMMITTED RESULTS:", ok)
