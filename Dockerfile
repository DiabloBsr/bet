# Déploiement cloud COMPLET : app trio + scraper 9 ligues + tracker + SQLite
# Compatible Hugging Face Spaces (user 1000, port 7860) et Docker générique.
FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl gzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir playwright  # module requis par scraper.collector (API-only, pas de navigateur)

COPY . .
RUN mkdir -p logs data/vfoot_ml /data \
    && useradd -m -u 1000 appuser && chown -R appuser:appuser /app /data

USER appuser
ENV HOME=/home/appuser \
    DB_URL="sqlite:////data/virtual_sports.db" \
    LEAGUE_IDS="8035,8065,8056,8060,8036,8037,8042,8043,8044" \
    PYTHONUNBUFFERED=1 PYTHONUTF8=1 PORT=7860

CMD ["bash", "deploy/start_cloud.sh"]
