#!/usr/bin/env bash
# Bootstrap script — Ubuntu 22.04 / 24.04 fresh VPS.
# Idempotent : safe to re-run.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/YOUR_USERNAME/virtual-sports-scraper.git}"
APP_DIR="/opt/virtual-sports-scraper"
SERVICE_USER="scraper"

echo "==> System update"
apt-get update -y
apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git curl ca-certificates \
    sqlite3 cron rsync

echo "==> Service user"
id -u "$SERVICE_USER" &>/dev/null || useradd --system --create-home --shell /bin/bash "$SERVICE_USER"

echo "==> Clone / pull"
if [[ ! -d "$APP_DIR/.git" ]]; then
    git clone "$REPO_URL" "$APP_DIR"
else
    git -C "$APP_DIR" pull
fi
chown -R "$SERVICE_USER:$SERVICE_USER" "$APP_DIR"

echo "==> Python venv + deps"
sudo -u "$SERVICE_USER" bash -c "
    cd $APP_DIR
    python3.11 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m playwright install --with-deps chromium
"

echo "==> Folders"
sudo -u "$SERVICE_USER" mkdir -p "$APP_DIR/data" "$APP_DIR/logs" "$APP_DIR/exports"

if [[ ! -f "$APP_DIR/.env" ]]; then
    echo "==> Creating .env from template — EDIT IT before starting"
    sudo -u "$SERVICE_USER" cp "$APP_DIR/.env.example" "$APP_DIR/.env"
fi

echo "==> systemd unit"
cp "$APP_DIR/deploy/scraper.service" /etc/systemd/system/scraper.service
systemctl daemon-reload

echo ""
echo "DONE."
echo "Edit:    sudoedit $APP_DIR/.env"
echo "Start:   systemctl enable --now scraper"
echo "Status:  systemctl status scraper"
echo "Logs:    journalctl -u scraper -f"
