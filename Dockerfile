# Déploiement cloud COMPLET : app trio + scraper 9 ligues + tracker + SQLite (/data)
FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl gzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir playwright  # module requis par scraper.collector (API-only, pas de navigateur)

COPY . .
RUN mkdir -p logs data/vfoot_ml && chmod +x deploy/start_cloud.sh

# le volume persistant est monté sur /data par la plateforme (Railway/Render/Fly)
ENV DB_URL="sqlite:////data/virtual_sports.db" \
    LEAGUE_IDS="8035,8065,8056,8060,8036,8037,8042,8043,8044" \
    PYTHONUNBUFFERED=1 PYTHONUTF8=1

CMD ["bash", "deploy/start_cloud.sh"]
