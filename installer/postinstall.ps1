# Runs during installation: ensures Python, creates the environment, installs
# the AI components (PyTorch GPU build + the app dependencies).
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

function Have-Py312 {
  try { & py -3.12 --version 2>$null; return ($LASTEXITCODE -eq 0) } catch { return $false }
}

if (-not (Have-Py312)) {
  Write-Host "Installing Python 3.12 (per-user)..."
  $url = "https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe"
  $tmp = Join-Path $env:TEMP "python-3.12.7-amd64.exe"
  Invoke-WebRequest -Uri $url -OutFile $tmp
  Start-Process $tmp -ArgumentList "/quiet","InstallAllUsers=0","PrependPath=1","Include_launcher=1","Include_pip=1" -Wait
}

Write-Host "Creating environment..."
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

Write-Host "Installing PyTorch (GPU/CUDA build) - this downloads about 2.5 GB, please wait..."
.\.venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Write-Host "Installing application dependencies..."
.\.venv\Scripts\pip.exe install -r requirements.txt

Write-Host "Installing native-window support..."
.\.venv\Scripts\pip.exe install pywebview

# Make sure runtime folders exist (writable, since we install per-user).
foreach ($d in @("outputs\events","outputs\annotated","data\datasets\maritime\raw_frames","models\maritime")) {
  New-Item -ItemType Directory -Force -Path $d | Out-Null
}

Write-Host "Done."
