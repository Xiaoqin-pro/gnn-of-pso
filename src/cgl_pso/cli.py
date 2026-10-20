"""CGL-PSO command line interface (independent from the GLDI-PSO CLI).

    python -m cgl_pso doctor
    python -m cgl_pso smoke   -c configs/cgl/smoke.yaml
    python -m cgl_pso dev10   -c configs/cgl/dev10.yaml
    python -m cgl_pso report
    python -m cgl_pso clip-prepare / clip-optimize / clip-evaluate-final
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from cgl_pso._infra.config import ConfigError, load_config
from cgl_pso._infra.seeding import set_global_seed
from cgl_pso._infra.version import __version__ as gldi_version

from . import __version__

app = typer.Typer(add_completion=False, help="CGL-PSO: contribution-guided graph learning PSO")

DEFAULT_CONFIGS = {
    "smoke": "configs/cgl/smoke.yaml",
    "dev10": "configs/cgl/dev10.yaml",
    "clip": "configs/cgl/clip.yaml",
}


def _load(config: Optional[Path], script: str, sets: Optional[list[str]]):
    path = Path(config) if config else Path(DEFAULT_CONFIGS[script])
    return load_config(path if path.exists() else None, overrides=list(sets) if sets else None)


def _print(obj) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str, ensure_ascii=False))


@app.command()
def version() -> None:
    """Print versions of the CGL-PSO package and the reused GLDI-PSO infrastructure."""
    typer.echo(json.dumps({"cgl_pso": __version__, "cgl_pso._infra": gldi_version}, indent=2))


@app.command()
def doctor(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    output: Path = typer.Option(Path("results/cgl/doctor.json"), "--output"),
    check_clip: bool = typer.Option(True, "--check-clip/--no-check-clip"),
) -> None:
    """Environment, hardware, dependency and literature-artefact check."""
    import importlib
    import platform
    import sys

    from cgl_pso._infra.resources import detect_hardware

    report: dict = {"cgl_pso": __version__, "cgl_pso._infra": gldi_version,
                    "python": sys.version.split()[0], "platform": platform.platform(),
                    "checkpoint_dir": "results/cgl", "hardware": detect_hardware(), "checks": []}

    def check(name: str, fn) -> None:
        try:
            report["checks"].append({"name": name, "ok": True, "detail": fn()})
        except Exception as exc:  # noqa: BLE001
            report["checks"].append({"name": name, "ok": False, "detail": f"{type(exc).__name__}: {exc}"})

    for mod in ("numpy", "pandas", "scipy", "matplotlib", "psutil", "sklearn"):
        check(f"dep:{mod}", lambda m=mod: getattr(importlib.import_module(m), "__version__", "?"))
    if check_clip:
        for mod in ("torch", "torchvision", "clip", "open_clip"):
            check(f"dep:{mod}", lambda m=mod: getattr(importlib.import_module(m), "__version__", "installed"))

    for cfg_name in ("smoke", "dev10"):
        check(f"config:{cfg_name}", lambda n=cfg_name: load_config(DEFAULT_CONFIGS[n]).get_path("run_name"))
    for doc in ("docs/CGL_LITERATURE.md", "docs/CGL_METHOD.md", "docs/CGL_PROTOCOL.md"):
        check(f"doc:{Path(doc).name}", lambda d=doc: str(Path(d).stat().st_size) + " bytes")

    try:
        cfg = _load(config, "smoke", None)
        report["config_ok"] = True
        report["config"] = cfg.to_dict()
    except ConfigError as exc:
        report["config_ok"] = False
        report["checks"].append({"name": "config", "ok": False, "detail": str(exc)})

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _print({"python": report["python"], "gpu": report["hardware"]["gpu"].get("name"),
            "cuda": report["hardware"].get("torch", {}).get("cuda_available"),
            "checks_failed": [c["name"] for c in report["checks"] if not c["ok"]],
            "report": str(output)})


@app.command()
def smoke(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    out_dir: Path = typer.Option(Path("results"), "--out-dir"),
    sets: Optional[list[str]] = typer.Option(None, "--set", "-s"),
) -> None:
    """Pilot-scale screening: 3 paired seeds x 3 tasks x 5,000 FEs (small config)."""
    from .runner import run_screening

    cfg = _load(config, "smoke", sets)
    pilot = cfg.get_path("screening.pilot", {})
    summary = run_screening(
        cfg,
        tasks=tuple(pilot.get("tasks", ["sphere", "rastrigin", "ackley"])),
        seeds=tuple(pilot.get("seeds", [0, 1, 2])),
        algorithms=tuple(cfg.get_path("screening.algorithms", ["ring", "gbest", "quality_rank",
                                                               "cgl_nocontrib", "cgl_full"])),
        dim=int(pilot.get("dim", 10)),
        n_particles=int(cfg.get_path("optimizer.n_particles", 20)),
        fes=int(pilot.get("fes", 5000)),
        out_dir=out_dir,
        tag="smoke",
    )
    _print({k: v for k, v in summary.items() if k != "artifacts"})


@app.command()
def dev10(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    out_dir: Path = typer.Option(Path("results"), "--out-dir"),
    tag: str = typer.Option("dev10", "--tag", help="output tag (a repair round uses e.g. 'dev10r1')"),
    seeds: Optional[int] = typer.Option(None, "--seeds", help="override the number of paired seeds"),
    fes: Optional[int] = typer.Option(None, "--fes"),
    algorithms: Optional[list[str]] = typer.Option(None, "--algorithm", "-a"),
    pilot: bool = typer.Option(False, "--pilot", help="run the pre-registered pilot instead of the main screening"),
    sets: Optional[list[str]] = typer.Option(None, "--set", "-s"),
) -> None:
    """Pre-registered development screening: 4 tasks x 10 paired seeds x 10,000 FEs x <=8 algorithms."""
    from .runner import CGL_ALGORITHMS, DEV_SEEDS, DEV_TASKS, PILOT_SEEDS, PILOT_TASKS, run_screening

    cfg = _load(config, "dev10", sets)
    if pilot:
        pilot_cfg = cfg.get_path("screening.pilot", {}) or {}
        algos = tuple(algorithms) if algorithms else tuple(cfg.get_path("screening.algorithms", CGL_ALGORITHMS))
        result = run_screening(
            cfg,
            tasks=tuple(pilot_cfg.get("tasks", PILOT_TASKS)),
            seeds=tuple(pilot_cfg.get("seeds", PILOT_SEEDS)),
            algorithms=algos,
            dim=int(pilot_cfg.get("dim", 30)),
            n_particles=int(cfg.get_path("optimizer.n_particles", 40)),
            fes=int(fes or pilot_cfg.get("fes", 5_000)),
            out_dir=out_dir,
            tag="pilot",
        )
        _print({k: v for k, v in result.summary.items() if k != "artifacts"})
        return
    dev = cfg.get_path("screening.dev10", {}) or {}
    algos = tuple(algorithms) if algorithms else tuple(cfg.get_path("screening.algorithms", CGL_ALGORITHMS))
    if len(algos) > int(cfg.get_path("budget.max_algorithms_dev10", 8)):
        raise typer.BadParameter("the pre-registered screening allows at most 8 algorithms")
    use_seeds = tuple(DEV_SEEDS[: int(seeds)]) if seeds else tuple(dev.get("seeds", DEV_SEEDS))
    result = run_screening(
        cfg,
        tasks=tuple(dev.get("tasks", DEV_TASKS)),
        seeds=use_seeds,
        algorithms=algos,
        dim=int(dev.get("dim", 30)),
        n_particles=int(cfg.get_path("optimizer.n_particles", 40)),
        fes=int(fes or dev.get("fes", 10_000)),
        out_dir=out_dir,
        tag=str(tag),
    )
    summary = result.summary
    total = int(summary["total_fes"])
    cap = int(cfg.get_path("budget.max_total_fes", 4_000_000))
    if total > cap:
        typer.echo(f"WARNING: this screening used {total:,} FEs, above the configured cap of {cap:,}", err=True)
    _print({k: v for k, v in summary.items() if k not in {"artifacts"}})


@app.command()
def report(
    results_dir: Path = typer.Option(Path("results"), "--results-dir"),
    docs_dir: Path = typer.Option(Path("docs"), "--docs-dir"),
) -> None:
    """Regenerate docs/CGL_RESULTS_zh.md and docs/CGL_DECISION.md from the artefacts."""
    from .reporting import build_cgl_report

    _print(build_cgl_report(results_dir=results_dir, docs_dir=docs_dir))


@app.command("clip-prepare")
def clip_prepare(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    out_dir: Path = typer.Option(Path("results"), "--out-dir"),
    backend: Optional[str] = typer.Option(None, "--backend", help="open_clip | openai_clip | synthetic"),
    allow_download: bool = typer.Option(False, "--allow-download", help="permit weight/dataset download"),
    sets: Optional[list[str]] = typer.Option(None, "--set", "-s"),
) -> None:
    """Cache image/text features (or synthetic logits) for the CLIP calibration experiment."""
    from .clip_app import prepare

    cfg = _load(config, "clip", sets)
    if backend:
        cfg.set_path("clip.backend", backend)
    cfg.set_path("clip.budget.allow_download", bool(allow_download))
    _print(prepare(cfg, out_dir=out_dir))


@app.command("clip-optimize")
def clip_optimize(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    out_dir: Path = typer.Option(Path("results"), "--out-dir"),
    algorithms: Optional[list[str]] = typer.Option(None, "--algorithm", "-a"),
    budget: Optional[int] = typer.Option(None, "--budget", help="unified objective evaluations per method"),
    sets: Optional[list[str]] = typer.Option(None, "--set", "-s"),
) -> None:
    """Optimise template weights + calibration params on the calibration split."""
    from .clip_app import optimize

    cfg = _load(config, "clip", sets)
    _print(optimize(cfg, out_dir=out_dir, algorithms=tuple(algorithms) if algorithms else None,
                    budget=int(budget) if budget else None))


@app.command("clip-evaluate-final")
def clip_evaluate_final(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    out_dir: Path = typer.Option(Path("results"), "--out-dir"),
    sets: Optional[list[str]] = typer.Option(None, "--set", "-s"),
) -> None:
    """Single, last evaluation of the frozen choices on the untouched test split."""
    from .clip_app import evaluate_final

    cfg = _load(config, "clip", sets)
    _print(evaluate_final(cfg, out_dir=out_dir))


if __name__ == "__main__":  # pragma: no cover
    app()
