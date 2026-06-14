#!/usr/bin/env bash
# Snapshot atomique de la base SQLite vers un dossier de backup.
# A coller dans /etc/cron.daily/scraper-backup ou via crontab du user scraper :
#   15 3 * * *  /opt/virtual-sports-scraper/deploy/backup_db.sh

set -euo pipefail

APP_DIR="/opt/virtual-sports-scraper"
BACKUP_DIR="$APP_DIR/backups"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$BACKUP_DIR"

# .backup est atomique : SQLite gere les writes concurrents
sqlite3 "$APP_DIR/data/virtual_sports.db" ".backup '$BACKUP_DIR/virtual_sports_$TS.db'"

# garde les 14 derniers, supprime le reste
ls -1t "$BACKUP_DIR"/virtual_sports_*.db | tail -n +15 | xargs -r rm -f

echo "[$TS] backup ok -> $BACKUP_DIR/virtual_sports_$TS.db"
