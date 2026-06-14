# Arrête le scraper lancé par start_bg.ps1.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$pidFile = "$root\scraper.pid"

if (-not (Test-Path $pidFile)) {
    Write-Host "No PID file. Nothing to stop." -ForegroundColor Yellow
    exit 0
}

$pidValue = (Get-Content $pidFile).Trim()
$proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue

if (-not $proc) {
    Write-Host "PID $pidValue not running. Cleaning pid file." -ForegroundColor Yellow
    Remove-Item $pidFile
    exit 0
}

Stop-Process -Id $pidValue -Force
Remove-Item $pidFile
Write-Host "Stopped PID=$pidValue" -ForegroundColor Green
