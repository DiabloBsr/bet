# Lance le collecteur Aviator en continu (fenêtre navigateur visible, session connectée).
# Édite BET_USER / BET_PASS ci-dessous OU passe-les en variables d'environnement.
# À mettre dans le dossier Démarrage de Windows pour une collecte 24/7.
param(
    [string]$User = $env:BET_USER,
    [string]$Pass = $env:BET_PASS
)
$R = Split-Path (Split-Path $PSScriptRoot)   # racine projet
$env:BET_USER = $User
$env:BET_PASS = $Pass
$env:PYTHONUTF8 = "1"
Write-Host "Collecteur Aviator continu — Ctrl+C pour arrêter."
& "$R\.venv\Scripts\python.exe" "$R\scripts\aviator\collector_service.py" 0
