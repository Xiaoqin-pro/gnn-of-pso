"""Verify the Ring-SmallWorld round without spending a single new objective evaluation.

Checks performed (all read-only on committed artefacts):

1. **Bitwise reproducibility** -- ``ring``/``gbest`` errors in ``results/cgl/sw_main/runs.csv``
   must equal the ones already committed in ``results/cgl/dev10/runs.csv`` for the same
   ``(task, seed)``; anything else means the instances/initialisation/FEs changed.
2. **Verdict reproduction** -- re-derive ``smallworld_verdict`` from the recorded
   ``runs.jsonl`` and compare the ``passed`` / ``recommended`` fields with the stored JSON.
3. **Budget ledger** -- sum the recorded FEs per stage and compare with
   ``configs/cgl/smallworld.yaml`` caps.

Usage (from the repository root)::

    .venv\\Scripts\\python.exe scripts/cgl_sw_verify_repro.py

Exit code 0 means every check passed; 1 means at least one mismatch (details on stdout).
"""

from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DEV10 = ROOT / "results/cgl/dev10/runs.csv"
SW_MAIN = ROOT / "results/cgl/sw_main"
SW_PRE = ROOT / "results/cgl/sw_pretrial"
CFG = ROOT / "configs/cgl/smallworld.yaml"

TASKS = ("alpine1", "griewank", "rosenbrock", "zakharov")


def check_reproducibility() -> bool:
    a = pd.read_csv(DEV10)
    b = pd.read_csv(SW_MAIN / "runs.csv")
    ok = True
    for algo in ("ring", "gbest"):
        for task in TASKS:
            ea = a[(a.algorithm == algo) & (a.task == task)].sort_values("seed").error.to_numpy()
            eb = b[(b.algorithm == algo) & (b.task == task)].sort_values("seed").error.to_numpy()
            diff = float(abs(ea - eb).max()) if len(ea) == len(eb) and len(ea) else float("inf")
            same = len(ea) == len(eb) and diff == 0.0
            ok &= same
            print(f"[repro] {algo:5s} {task:11s} n={len(ea):2d}/{len(eb):2d} maxdiff={diff:.3e} {'OK' if same else 'MISMATCH'}")
    return ok


def check_verdict() -> bool:
    from cgl_pso.runner import smallworld_verdict  # noqa: PLC0415

    df = pd.read_json(SW_MAIN / "runs.jsonl", lines=True)
    fresh = smallworld_verdict(df)
    stored = json.loads((SW_MAIN / "smallworld_verdict.json").read_text(encoding="utf-8"))
    same = bool(fresh.get("recommended") == stored.get("recommended")) and bool(
        fresh.get("any_candidate_passed") == stored.get("any_candidate_passed")
    )
    for name, cand in sorted(fresh["per_candidate"].items()):
        ref = stored["per_candidate"].get(name, {})
        agree = bool(cand["passed"]) == bool(ref.get("passed"))
        same &= agree
        print(f"[verdict] {name:20s} passed={cand['passed']!s:5s} runtime_x={cand['median_runtime_factor']:.2f} "
              f"{'OK' if agree else 'MISMATCH'}")
    print(f"[verdict] recommended={fresh.get('recommended')} stored={stored.get('recommended')}")
    keep = fresh.get("online_vs_fixed", {}).get("keep_online")
    print(f"[verdict] keep_online={keep} (must be False: the online bandit is dropped)")
    return same and keep is False


def check_budget() -> bool:
    from cgl_pso._infra.config import load_config  # noqa: PLC0415

    cfg = load_config(CFG)
    ledger: dict[str, int] = {}
    for stage, path in (("pre", SW_PRE / "runs.jsonl"), ("main", SW_MAIN / "runs.jsonl")):
        df = pd.read_json(path, lines=True)
        ledger[stage] = int(df.fes.sum())
    diag = pd.read_csv(SW_PRE / "selection_diagnostics.csv")
    ledger["diagnostics"] = int(len(diag) * int(cfg.get_path("smallworld.fes", 10_000)))
    total = sum(ledger.values())
    cap = int(cfg.get_path("budget.total_max_fes", 3_800_000))
    ok = total <= cap
    for name, fes in ledger.items():
        print(f"[budget] {name:12s} {fes:>10,d} FEs")
    print(f"[budget] total={total:,d} / cap={cap:,d} {'OK' if ok else 'OVER BUDGET'}")
    return ok


def main() -> int:
    results = {
        "bitwise reproducibility": check_reproducibility(),
        "verdict reproduction": check_verdict(),
        "budget ledger": check_budget(),
    }
    print()
    for name, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
