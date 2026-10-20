"""Re-derive the Ring-SmallWorld summaries and verdicts from the recorded raw runs.

Costs **zero** objective evaluations: everything is recomputed from the committed
``runs.jsonl`` files. Used to refresh ``summary.json`` / ``smallworld_verdict.json``
after the verdict helper was fixed, without re-running the screening.

Usage (repository root)::

    .venv\\Scripts\\python.exe scripts/cgl_sw_resummarise.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cgl_pso.runner import smallworld_verdict, summarise  # noqa: E402
from cgl_pso._infra.config import load_config  # noqa: E402

STAGES = ("sw_pretrial", "sw_main")


def main() -> int:
    cfg = load_config(ROOT / "configs/cgl/smallworld.yaml")
    tasks = tuple(cfg.get_path("smallworld.tasks"))
    dim = int(cfg.get_path("smallworld.dim", 30))
    fes = int(cfg.get_path("smallworld.fes", 10_000))
    for tag in STAGES:
        run_dir = ROOT / "results/cgl" / tag
        raw = run_dir / "runs.jsonl"
        records = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
        df = pd.DataFrame(records)
        seeds = tuple(sorted(int(s) for s in df["seed"].unique()))
        before = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        summary = summarise(df, run_dir, tasks=tasks, seeds=seeds, dim=dim, fes=fes, tag=tag)
        # preserve the runtime/resource blocks that only the live run can produce
        for key in ("wall_time_s", "resource_summary", "guard_events", "raw_path"):
            if key in before:
                summary[key] = before[key]
        verdict = smallworld_verdict(df)
        summary["smallworld_verdict"] = {k: v for k, v in verdict.items() if not k.startswith("vs_")}
        if tag == "sw_pretrial":
            summary["specs"] = cfg.get_path("smallworld.pre_trial.settings")
        summary["note"] = "re-summarised from the recorded runs.jsonl; no new FEs consumed"
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        (run_dir / "smallworld_verdict.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
        gate = json.loads((run_dir / "verdict.json").read_text(encoding="utf-8"))
        print(f"[{tag}] runs={len(df)} FEs={int(df['fes'].sum()):,d} "
              f"cgl_gate_applicable={gate.get('applicable')} recommended={summary['smallworld_verdict'].get('recommended')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
