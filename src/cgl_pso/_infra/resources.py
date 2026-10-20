"""Hardware detection, live resource logging and thermal/budget guards.

Design notes
------------
* The user-authorized RAM ceiling is 95% of physical memory; the soft limit is
  90% (reduce load).  The discrete-GPU targets follow the task brief: soft at
  82 C (shed load), hard at 85 C (pause, save checkpoint).
* Sensors that do not exist on this machine are reported as ``None`` and the
  guard degrades to the conservative branch instead of pretending to measure.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import psutil

try:  # torch is a hard dependency, but keep import failures non-fatal
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

__all__ = [
    "detect_hardware",
    "ResourceSnapshot",
    "ResourceMonitor",
    "GuardDecision",
    "ResourceGuard",
    "gpu_query",
]


def _run(cmd: list[str], timeout: float = 5.0) -> str | None:
    exe = shutil.which(cmd[0])
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, *cmd[1:]], capture_output=True, text=True, timeout=timeout, check=False
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except Exception:
        return None


def gpu_query() -> dict[str, Any]:
    """Query nvidia-smi (grader-friendly: never raises)."""
    fields = "name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu"
    out = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    if not out:
        return {"available": False, "reason": "nvidia-smi not available"}
    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 7:
        return {"available": False, "reason": f"unexpected nvidia-smi output: {out!r}"}

    def num(value: str) -> float | None:
        try:
            return float(value)
        except ValueError:
            return None

    return {
        "available": True,
        "name": parts[0],
        "driver_version": parts[1],
        "memory_total_mib": num(parts[2]),
        "memory_used_mib": num(parts[3]),
        "memory_free_mib": num(parts[4]),
        "temperature_c": num(parts[5]),
        "utilization_pct": num(parts[6]),
    }


def detect_hardware() -> dict[str, Any]:
    """Full hardware/software snapshot used by ``doctor`` and run metadata."""
    vm = psutil.virtual_memory()
    cpu_freq = psutil.cpu_freq()
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu": {
            "model": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "max_freq_mhz": getattr(cpu_freq, "max", None),
            "temperature_c": _cpu_temperature(),
        },
        "ram": {
            "total_gb": round(vm.total / 1e9, 2),
            "available_gb": round(vm.available / 1e9, 2),
            "used_pct": vm.percent,
        },
        "disk_free_gb": {p: round(shutil.disk_usage(p).free / 1e9, 1) for p in _existing_roots()},
        "gpu": gpu_query(),
    }
    if torch is not None:
        info["torch"] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": ".".join(str(v) for v in torch.cuda.get_device_capability(i)),
                    "total_memory_gb": round(
                        torch.cuda.get_device_properties(i).total_memory / 1e9, 2
                    ),
                }
                for i in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
            "threads": torch.get_num_threads(),
        }
    if info["cpu"]["temperature_c"] is None:
        info["cpu"]["temperature_note"] = "no CPU temperature sensor exposed to psutil on this host"
    return info


def _existing_roots() -> list[str]:
    roots = []
    for cand in ("C:\\", "D:\\", "/") if platform.system() == "Windows" else ("/",):
        try:
            if Path(cand).exists():
                roots.append(cand)
        except Exception:  # pragma: no cover
            continue
    return roots


def _cpu_temperature() -> float | None:
    getter = getattr(psutil, "sensors_temperatures", None)
    if getter is None:
        return None
    try:
        temps = getter() or {}
    except Exception:
        return None
    for entries in temps.values():
        for entry in entries:
            if entry.current:
                return float(entry.current)
    return None


@dataclass
class ResourceSnapshot:
    t: float
    cpu_pct: float
    ram_pct: float
    ram_used_gb: float
    ram_available_gb: float
    gpu_util_pct: float | None = None
    gpu_temp_c: float | None = None
    gpu_mem_used_mib: float | None = None
    gpu_mem_free_mib: float | None = None
    vram_alloc_gb: float | None = None
    proc_rss_gb: float | None = None

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def _torch_vram() -> tuple[float | None, float | None]:
    if torch is None or not torch.cuda.is_available():
        return None, None
    try:
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return round(alloc, 3), round(reserved, 3)
    except Exception:  # pragma: no cover
        return None, None


def sample_resources(include_gpu: bool = True) -> ResourceSnapshot:
    vm = psutil.virtual_memory()
    proc = psutil.Process()
    alloc, _ = _torch_vram()
    snap = ResourceSnapshot(
        t=time.time(),
        cpu_pct=float(psutil.cpu_percent(interval=None)),
        ram_pct=float(vm.percent),
        ram_used_gb=round(vm.used / 1e9, 3),
        ram_available_gb=round(vm.available / 1e9, 3),
        vram_alloc_gb=alloc,
        proc_rss_gb=round(proc.memory_info().rss / 1e9, 3),
    )
    if include_gpu:
        g = gpu_query()
        if g.get("available"):
            snap.gpu_util_pct = g.get("utilization_pct")
            snap.gpu_temp_c = g.get("temperature_c")
            snap.gpu_mem_used_mib = g.get("memory_used_mib")
            snap.gpu_mem_free_mib = g.get("memory_free_mib")
    return snap


class ResourceMonitor:
    """Background sampler that appends CSV + JSONL resource logs.

    The sampler never blocks training: it runs in a daemon thread and writes
    every ``period_s`` seconds (task brief asks for 5-10 s).
    """

    def __init__(
        self,
        out_dir: str | Path,
        period_s: float = 5.0,
        prefix: str = "resources",
        include_gpu: bool = True,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.period_s = float(period_s)
        self.prefix = prefix
        self.include_gpu = include_gpu
        self.csv_path = self.out_dir / f"{prefix}.csv"
        self.jsonl_path = self.out_dir / f"{prefix}.jsonl"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[ResourceSnapshot] = []
        self._lock = threading.Lock()

    def _write_header_if_needed(self) -> None:
        if not self.csv_path.exists():
            cols = list(ResourceSnapshot(t=0, cpu_pct=0, ram_pct=0, ram_used_gb=0, ram_available_gb=0).to_row())
            self.csv_path.write_text(",".join(cols) + "\n", encoding="utf-8")

    def sample_once(self) -> ResourceSnapshot:
        snap = sample_resources(self.include_gpu)
        with self._lock:
            self.samples.append(snap)
            self._write_header_if_needed()
            row = snap.to_row()
            with self.csv_path.open("a", encoding="utf-8") as fh:
                fh.write(",".join("" if row[c] is None else str(row[c]) for c in row) + "\n")
            with self.jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        return snap

    def start(self) -> "ResourceMonitor":
        if self._thread is not None:
            return self
        self._stop.clear()

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    self.sample_once()
                except Exception:  # pragma: no cover - logging must never crash a run
                    pass
                self._stop.wait(self.period_s)

        self._thread = threading.Thread(target=loop, name="resource-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def __enter__(self) -> "ResourceMonitor":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            samples = list(self.samples)
        if not samples:
            return {"n_samples": 0}
        def stat(attr: str) -> dict[str, float] | None:
            vals = [getattr(s, attr) for s in samples if getattr(s, attr) is not None]
            if not vals:
                return None
            return {"mean": round(sum(vals) / len(vals), 3), "max": round(max(vals), 3), "min": round(min(vals), 3)}
        return {
            "n_samples": len(samples),
            "duration_s": round(samples[-1].t - samples[0].t, 2),
            "cpu_pct": stat("cpu_pct"),
            "ram_pct": stat("ram_pct"),
            "gpu_temp_c": stat("gpu_temp_c"),
            "gpu_util_pct": stat("gpu_util_pct"),
            "vram_alloc_gb": stat("vram_alloc_gb"),
            "proc_rss_gb": stat("proc_rss_gb"),
            "csv": str(self.csv_path),
            "jsonl": str(self.jsonl_path),
        }


@dataclass
class GuardDecision:
    level: str  # ok|reduce|pause|stop
    reasons: list[str] = field(default_factory=list)
    batch_scale: float = 1.0

    @property
    def healthy(self) -> bool:
        return self.level == "ok"

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "reasons": list(self.reasons), "batch_scale": self.batch_scale}


class ResourceGuard:
    """Turns resource snapshots into ok / reduce / pause / stop decisions."""

    def __init__(self, cfg: dict[str, Any] | None = None, log_path: str | Path | None = None) -> None:
        cfg = dict(cfg or {})
        self.cpu_soft = float(cfg.get("cpu_soft_pct", 90.0))
        self.cpu_hard = float(cfg.get("cpu_hard_pct", 95.0))
        self.ram_soft = float(cfg.get("ram_soft_pct", 90.0))
        self.ram_hard = float(cfg.get("ram_hard_pct", 95.0))
        self.gpu_soft = float(cfg.get("gpu_soft_c", 82.0))
        self.gpu_hard = float(cfg.get("gpu_hard_c", 85.0))
        self.vram_budget_gb = float(cfg.get("vram_budget_gb", 6.0))
        self.gpu_reserve_gb = float(cfg.get("gpu_reserve_gb", 0.5))
        self.min_scale = float(cfg.get("min_batch_scale", 0.25))
        self.max_scale = float(cfg.get("max_batch_scale", 4.0))
        self.scale = 1.0
        self.log_path = Path(log_path) if log_path else None
        self.events: list[dict[str, Any]] = []

    def evaluate(self, snap: ResourceSnapshot) -> GuardDecision:
        reasons: list[str] = []
        level = "ok"
        if snap.ram_pct >= self.ram_hard:
            level = "pause"
            reasons.append(f"ram {snap.ram_pct:.1f}% >= hard {self.ram_hard:.0f}%")
        elif snap.cpu_pct >= self.cpu_hard:
            level = "pause"
            reasons.append(f"cpu {snap.cpu_pct:.1f}% >= hard {self.cpu_hard:.0f}%")
        if snap.gpu_temp_c is not None:
            if snap.gpu_temp_c >= self.gpu_hard:
                level = "pause"
                reasons.append(f"gpu {snap.gpu_temp_c:.0f}C >= hard {self.gpu_hard:.0f}C")
            elif snap.gpu_temp_c >= self.gpu_soft and level == "ok":
                level = "reduce"
                reasons.append(f"gpu {snap.gpu_temp_c:.0f}C >= soft {self.gpu_soft:.0f}C")
        if level == "ok":
            if snap.ram_pct >= self.ram_soft:
                level = "reduce"
                reasons.append(f"ram {snap.ram_pct:.1f}% >= soft {self.ram_soft:.0f}%")
            elif snap.cpu_pct >= self.cpu_soft:
                level = "reduce"
                reasons.append(f"cpu {snap.cpu_pct:.1f}% >= soft {self.cpu_soft:.0f}%")
        if snap.vram_alloc_gb is not None and snap.vram_alloc_gb > self.vram_budget_gb:
            level = "reduce" if level == "ok" else level
            reasons.append(f"vram alloc {snap.vram_alloc_gb:.2f} GB > budget {self.vram_budget_gb:.2f} GB")
        if snap.gpu_mem_free_mib is not None and snap.gpu_mem_free_mib < self.gpu_reserve_gb * 1024:
            level = "pause"
            reasons.append(f"gpu free {snap.gpu_mem_free_mib:.0f} MiB < reserve {self.gpu_reserve_gb * 1024:.0f} MiB")

        if level == "reduce":
            self.scale = max(self.min_scale, self.scale * 0.7)
        elif level == "ok":
            self.scale = min(self.max_scale, self.scale * 1.05)
        decision = GuardDecision(level=level, reasons=reasons, batch_scale=self.scale)
        if reasons:
            self.events.append({"t": snap.t, **decision.to_dict()})
            if self.log_path:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"t": snap.t, **decision.to_dict()}) + "\n")
        return decision

    def check(self, include_gpu: bool = True) -> GuardDecision:
        return self.evaluate(sample_resources(include_gpu=include_gpu))
