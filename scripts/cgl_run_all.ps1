# =====================================================================
# CGL-PSO reproduction script (Windows / PowerShell)
#   powershell -ExecutionPolicy Bypass -File scripts/cgl_run_all.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/cgl_run_all.ps1 -Quick
# Budget discipline: pilot + one main screening + at most one repair round,
# total numeric budget <= 4,000,000 FEs (see docs/CGL_PROTOCOL.md).
# =====================================================================
param(
  [switch]$Quick,                 # skip the main screening (pilot + smoke only)
  [int]$PilotFes = 5000,
  [int]$DevFes = 10000,
  [switch]$WithClipSmoke
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = if (Test-Path ".venv/Scripts/python.exe") { ".venv/Scripts/python.exe" } else { "python" }
Write-Host "== interpreter: $Py" -ForegroundColor Cyan

Write-Host "`n[1/6] doctor" -ForegroundColor Cyan
& $Py -m cgl_pso doctor

Write-Host "`n[2/6] unit tests (CGL + existing)" -ForegroundColor Cyan
& $Py -m pytest tests -q

Write-Host "`n[3/6] pilot (3 seeds x 3 tasks x 5,000 FEs)" -ForegroundColor Cyan
& $Py -m cgl_pso dev10 -c configs/cgl/dev10.yaml --pilot --fes $PilotFes

if (-not $Quick) {
  Write-Host "`n[4/6] pre-registered main screening (4 tasks x 10 seeds x 10,000 FEs x 8 algorithms)" -ForegroundColor Cyan
  & $Py -m cgl_pso dev10 -c configs/cgl/dev10.yaml --fes $DevFes
} else {
  Write-Host "`n[4/6] main screening skipped (-Quick)" -ForegroundColor Yellow
}

if ($WithClipSmoke) {
  Write-Host "`n[5/6] CLIP calibration smoke (synthetic logits)" -ForegroundColor Cyan
  & $Py -m cgl_pso clip-prepare  -c configs/cgl/clip.yaml --backend synthetic
  & $Py -m cgl_pso clip-optimize -c configs/cgl/clip.yaml --budget 2000
  & $Py -m cgl_pso clip-evaluate-final -c configs/cgl/clip.yaml
} else {
  Write-Host "`n[5/6] CLIP smoke skipped (use -WithClipSmoke)" -ForegroundColor Yellow
}

Write-Host "`n[6/6] report" -ForegroundColor Cyan
& $Py -m cgl_pso report
Write-Host "`nDone. See docs/CGL_RESULTS_zh.md and docs/CGL_DECISION.md" -ForegroundColor Green
