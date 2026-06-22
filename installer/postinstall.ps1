# Robust post-install setup for Boat Vision (GPU edition).
# - bundles its own Python (no dependency on anything pre-installed)
# - uses absolute paths (no reliance on PATH / the py launcher)
# - retries downloads, and falls back to CPU PyTorch if the CUDA build fails
# - logs everything to %LOCALAPPDATA%\BoatVision\install-log.txt
$ErrorActionPreference = "Continue"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$logDir = Join-Path $env:LOCALAPPDATA "BoatVision"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir "install-log.txt"
function Log($m) { $t = (Get-Date).ToString('HH:mm:ss'); ("$t  $m") | Tee-Object -FilePath $log -Append }

Log "=== Boat Vision setup starting in $root ==="

# 1) Locate Python 3.12 (common per-user / all-users locations), else install bundled.
$cands = @(
  "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
  "$env:ProgramFiles\Python312\python.exe",
  "C:\Python312\python.exe"
)
$py = $null
foreach ($c in $cands) { if (Test-Path $c) { $py = $c; break } }
if (-not $py) {
  try { $e = & py -3.12 -c "import sys;print(sys.executable)" 2>$null; if ($LASTEXITCODE -eq 0 -and $e) { $py = $e.Trim() } } catch {}
}
if (-not $py -or -not (Test-Path $py)) {
  $inst = Join-Path $root "installer\python-3.12.7-amd64.exe"
  if (Test-Path $inst) {
    Log "Installing bundled Python 3.12 (per-user)..."
    Start-Process $inst -ArgumentList "/quiet","InstallAllUsers=0","PrependPath=1","Include_launcher=1","Include_pip=1" -Wait
    foreach ($c in $cands) { if (Test-Path $c) { $py = $c; break } }
  } else { Log "Bundled Python installer not found at $inst" }
}
if (-not $py -or -not (Test-Path $py)) { Log "FATAL: could not find or install Python 3.12."; exit 1 }
Log "Using Python: $py"

# 2) Create the virtual environment.
Log "Creating environment (.venv)..."
& $py -m venv .venv 2>&1 | Tee-Object -FilePath $log -Append
$venvpy = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvpy)) { Log "FATAL: virtual environment was not created."; exit 1 }

& $venvpy -m pip install --upgrade pip 2>&1 | Tee-Object -FilePath $log -Append

function PipInstall([string[]]$pkgs) {
  for ($i = 1; $i -le 3; $i++) {
    Log ("pip install " + ($pkgs -join ' ') + "  (attempt $i)")
    & $venvpy -m pip install @pkgs 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -eq 0) { return $true }
    Start-Sleep -Seconds 5
  }
  return $false
}

# 3) PyTorch: try the GPU (CUDA) build, fall back to CPU so install always succeeds.
Log "Installing PyTorch GPU build (downloads ~2.5 GB)..."
if (-not (PipInstall @("torch","torchvision","--index-url","https://download.pytorch.org/whl/cu121"))) {
  Log "CUDA build failed - falling back to CPU PyTorch."
  PipInstall @("torch","torchvision") | Out-Null
}
PipInstall @("-r","requirements.txt") | Out-Null
PipInstall @("pywebview") | Out-Null

foreach ($d in @("outputs\events","outputs\annotated","data\datasets\maritime\raw_frames","models\maritime")) {
  New-Item -ItemType Directory -Force $d | Out-Null
}
Log "=== DONE ==="
