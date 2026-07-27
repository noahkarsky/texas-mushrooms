<#
.SYNOPSIS
    Launches the Texas Mushrooms web UI: the Python image proxy + the Vite dev server.

.DESCRIPTION
    Starts two things:
      1. The image proxy (python -m texas_mushrooms.web_proxy) in a background window,
         which serves local images and proxies hotlink-protected remote images.
      2. The Vite dev server (npm run dev) in this window.

    Set the UI's "Local Images Base URL" to http://127.0.0.1:<ProxyPort> to make photos load.

.PARAMETER Export
    Re-run scripts/export_web_assets.py first to regenerate web/public/data/*.

.PARAMETER SkipProxy
    Don't start the image proxy (e.g. if you don't need images to load).

.PARAMETER ProxyPort
    Port for the image proxy. Default 8001.

.EXAMPLE
    ./scripts/run_web.ps1
.EXAMPLE
    ./scripts/run_web.ps1 -Export
#>
[CmdletBinding()]
param(
    [switch]$Export,
    [switch]$SkipProxy,
    [int]$ProxyPort = 8001
)

$ErrorActionPreference = "Stop"

# Repo root = parent of the scripts/ folder this file lives in.
$RepoRoot = Split-Path -Parent $PSScriptRoot
$WebDir   = Join-Path $RepoRoot "web"

# --- Pick a Python interpreter -------------------------------------------------
# Prefer the project's Poetry venv if present, else fall back to `python` on PATH.
$PoetryPython = "C:/Users/noahk/AppData/Local/pypoetry/Cache/virtualenvs/texas-mushrooms-FHq9QisE-py3.12/Scripts/python.exe"
if (Test-Path $PoetryPython) {
    $Python = $PoetryPython
} else {
    $Python = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $Python) {
        throw "No Python found. Install Python or activate the project venv."
    }
}
Write-Host "Using Python: $Python" -ForegroundColor Cyan

# --- Check Node/npm ------------------------------------------------------------
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "npm not found on PATH. Install Node.js (LTS) first."
}

# --- Optional: regenerate web assets ------------------------------------------
if ($Export) {
    Write-Host "Exporting web assets..." -ForegroundColor Yellow
    & $Python (Join-Path $RepoRoot "scripts/export_web_assets.py")
    if ($LASTEXITCODE -ne 0) { throw "export_web_assets.py failed (exit $LASTEXITCODE)." }
}

# --- Install web deps if needed ------------------------------------------------
if (-not (Test-Path (Join-Path $WebDir "node_modules"))) {
    Write-Host "Installing web dependencies (npm install)..." -ForegroundColor Yellow
    Push-Location $WebDir
    try { npm install } finally { Pop-Location }
}

# --- Start the image proxy in a background window ------------------------------
$proxyProc = $null
if (-not $SkipProxy) {
    $imagesRoot = Join-Path $RepoRoot "data/raw/images"
    if (Test-Path $imagesRoot) {
        Write-Host "Starting image proxy on http://127.0.0.1:$ProxyPort ..." -ForegroundColor Green
        $proxyProc = Start-Process -FilePath $Python `
            -ArgumentList @("-m", "texas_mushrooms.web_proxy", "--port", "$ProxyPort") `
            -WorkingDirectory $RepoRoot -PassThru
        Write-Host "  -> Set 'Local Images Base URL' in the UI to http://127.0.0.1:$ProxyPort" -ForegroundColor Green
    } else {
        Write-Warning "No images at $imagesRoot; skipping proxy. Run a crawl with --download-images to populate it."
    }
}

# --- Run the Vite dev server (foreground) -------------------------------------
Write-Host "Starting Vite dev server (npm run dev)... Ctrl+C to stop." -ForegroundColor Green
Push-Location $WebDir
try {
    npm run dev
} finally {
    Pop-Location
    # Clean up the proxy when the dev server exits.
    if ($proxyProc -and -not $proxyProc.HasExited) {
        Write-Host "Stopping image proxy (PID $($proxyProc.Id))..." -ForegroundColor Yellow
        Stop-Process -Id $proxyProc.Id -Force -ErrorAction SilentlyContinue
    }
}
