#!/usr/bin/env bash
# Orchestrateur cloud : seed DB -> scraper (fond) -> tracker (fond) -> app trio (avant-plan)
set -u

DB=/data/virtual_sports.db

# ---- 1. seed de la base au premier démarrage ----
# Ordre : (a) seed embarqué dans l'image (repo/HF Space), (b) release GitHub,
# (c) sinon base vide (le scraper accumulera).
#
# DiabloBsr/bet est PUBLIC : l'asset de la release se télécharge sans jeton.
# Cette branche était gatée sur `[ -n "$GH_TOKEN" ]` (le repo était privé à
# l'époque) : un Space sans ce secret démarrait donc sur une base VIDE alors que
# le fichier est librement accessible — historique et face-à-face vides le temps
# que le scraper ré-accumule des semaines de données. On tente toujours, et le
# jeton ne sert plus que de repli si le repo redevenait privé.
_gh_curl() {                    # curl API GitHub, en-tête d'auth seulement si jeton
  if [ -n "${GH_TOKEN:-}" ]; then
    curl -s -H "Authorization: token ${GH_TOKEN}" "$@"
  else
    curl -s "$@"
  fi
}

if [ ! -s "$DB" ]; then
  if [ -s /app/seed/virtual_sports_seed.db.gz ]; then
    echo "[start] seed embarqué -> installation…"
    gunzip -c /app/seed/virtual_sports_seed.db.gz > "$DB"
  else
    echo "[start] téléchargement du seed (release GitHub, jeton $([ -n "${GH_TOKEN:-}" ] && echo présent || echo absent))…"
    ASSET_ID=$(_gh_curl \
      "https://api.github.com/repos/DiabloBsr/bet/releases/tags/seed-db" \
      | python -c "import json,sys; print(json.load(sys.stdin)['assets'][0]['id'])" || echo "")
    if [ -n "$ASSET_ID" ]; then
      _gh_curl -L -H "Accept: application/octet-stream" \
        "https://api.github.com/repos/DiabloBsr/bet/releases/assets/${ASSET_ID}" -o /tmp/seed.gz \
        && gunzip -c /tmp/seed.gz > "$DB" && rm -f /tmp/seed.gz
    else
      echo "[start] release seed-db introuvable (repo redevenu privé sans GH_TOKEN ?)"
    fi
  fi
  if [ -s "$DB" ]; then
    echo "[start] seed installé : $(du -h "$DB" | cut -f1)"
  else
    echo "[start] AVERTISSEMENT : pas de seed -> base vide, le scraper accumule."
  fi
fi

# ---- 2. calibration embarquée -> emplacement attendu par predict_trio ----
mkdir -p data/vfoot_ml logs
cp -f config/score_calibration.json data/vfoot_ml/score_calibration.json 2>/dev/null || true

# ---- 3. scraper LÉGER (httpx, pas de pandas/modèle) — throttlé, pour les matchs À VENIR ----
#   180s au lieu de 45s : 4x moins de charge CPU sur cpu-basic. Le scraper seul est léger ;
#   c'était le rendu Fiabilité (446 Mo) + la grosse DB qui pégaient le CPU (crash-loop), pas lui.
( while true; do
    python scripts/_scrape_loop.py --interval 180 --n 100000 >> /data/scrape.log 2>&1 || true
    sleep 15
  done ) &

# ---- 4-5. jobs LOURDS (tracker, monitors, calibration) — OPT-IN (chargent pandas+modèle) ----
#   Le vrai stack d'analyse tourne en LOCAL. Pour les activer en ligne : CLOUD_FULL_STACK=1.
if [ "${CLOUD_FULL_STACK:-0}" = "1" ]; then
  echo "[start] CLOUD_FULL_STACK=1 -> tracker + monitors + calibration activés"
  ( sleep 120
    while true; do
      python scripts/trio_tracker.py >> /data/tracker.log 2>&1 || true
      sleep 1800
    done ) &
  ( sleep 600
    while true; do
      python scripts/vfoot_ml/line_edge_monitor.py >> /data/line_monitor.log 2>&1 || true
      python scripts/vfoot_ml/line_paper_trader.py >> /data/line_paper.log 2>&1 || true
      sleep 86400
    done ) &
  ( sleep 900
    while true; do
      python scripts/refresh_calibration.py >> /data/calib.log 2>&1 || true
      sleep 604800
    done ) &
else
  echo "[start] mode LÉGER (défaut) : scraper + app (jobs lourds désactivés = stable sur cpu-basic)"
fi

# ---- 6. app trio (avant-plan = process principal du conteneur) ----
exec streamlit run scripts/dashboard_trio.py \
  --server.headless true --server.port "${PORT:-8080}" --server.address 0.0.0.0 \
  --browser.gatherUsageStats false
