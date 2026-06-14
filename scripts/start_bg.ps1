# Lance run_forever.py comme processus Windows détaché.
# Le processus survit à la fermeture du shell qui l'a lancé.
# Logs Python → logs/scraper.log (rotation automatique).
# stdout/stderr du process → logs/stdout.log / logs/stderr.log.
# PID enregistré dans scraper.pid pour stop_bg.ps1.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if (Test-Path "$root\scraper.pid") {
    $existing = Get-Content "$root\scraper.pid"
    $proc = Get-Process -Id $existing -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "Already running (PID=$existing). Use stop_bg.ps1 first." -ForegroundColor Yellow
        exit 1
    }
    Remove-Item "$root\scraper.pid"
}

New-Item -ItemType Directory -Force "$root\logs" | Out-Null

$proc = Start-Process `
    -FilePath "$root\.venv\Scripts\python.exe" `
    -ArgumentList "$root\scripts\run_forever.py" `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput "$root\logs\stdout.log" `
    -RedirectStandardError "$root\logs\stderr.log" `
    -PassThru

$proc.Id | Out-File -FilePath "$root\scraper.pid" -Encoding ascii -NoNewline
Write-Host "Started PID=$($proc.Id)" -ForegroundColor Green
Write-Host "Logs : $root\logs\scraper.log"
Write-Host "Stop : .\scripts\stop_bg.ps1"
