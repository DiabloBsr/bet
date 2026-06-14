# Deploiement VPS 24/7

Tout ce qui suit suppose un **VPS Linux Ubuntu 22.04 ou 24.04** avec acces root SSH.
Aucune dependance Claude/IA en runtime — le scraper tourne en solo via systemd.

## 0. Choix du VPS

| Provider | Plan | Prix | RAM | OK pour ce projet |
|---|---|---|---|---|
| Hetzner | CX22 | 4.51 €/mois HT | 4 Go | **Recommande** |
| Scaleway | PLAY2-PICO | 4.99 €/mois HT | 2 Go | OK, latence Paris |
| Oracle Cloud | ARM Ampere (free tier) | 0 € | 24 Go | Gratuit a vie si compte valide |
| Contabo | VPS S | 6.50 €/mois | 8 Go | Surdimensionne mais cheap |

Recommande : **Hetzner CX22 Falkenstein** ou Helsinki. Tu paies par l'heure, tu peux tester pour 5 cents et detruire si ca te plait pas.

## 1. Creer le VPS

1. Cree un compte chez le provider, ajoute ta carte
2. Cree un Cloud Server : Ubuntu 24.04, region Europe
3. Ajoute ta cle SSH publique (genere une si t'en a pas : `ssh-keygen -t ed25519`)
4. Note l'IP publique du serveur

## 2. Connexion + installation

```bash
ssh root@TON_IP

# Recupere les scripts (option A : git clone direct)
git clone https://github.com/TON_USER/virtual-sports-scraper.git /tmp/scraper
cd /tmp/scraper/deploy

# Edite install.sh ligne 7 pour pointer vers TON repo
nano install.sh

# Lance l'install (5-10 min, telecharge Chromium ~150 Mo)
bash install.sh

# Edite la conf
sudoedit /opt/virtual-sports-scraper/.env
# TARGET_URL=https://bet261.mg/virtual/category/instant-league/8035/matches
# EXTRA_URLS=...
# HEADLESS=true
# SCRAPE_INTERVAL_SECONDS=90

# Demarre le service
systemctl enable --now scraper
systemctl status scraper
```

### Option B sans git : `rsync` depuis ton PC Windows

Depuis Windows PowerShell (ou WSL) :
```powershell
# copie tout sauf .venv, data, logs
$exclude = '.venv', 'data', 'logs', 'exports', 'backups', '*.pid'
scp -r d:\AGENTOVA\SAMY\virtual-sports-scraper root@TON_IP:/tmp/
ssh root@TON_IP "mv /tmp/virtual-sports-scraper /opt/ && bash /opt/virtual-sports-scraper/deploy/install.sh"
```

## 3. Verifier que ca tourne

```bash
# logs en direct
journalctl -u scraper -f

# 5 dernieres iterations
journalctl -u scraper --since "10 min ago" | grep "iteration ok"

# inspection DB
cd /opt/virtual-sports-scraper
sudo -u scraper .venv/bin/python scripts/_progress.py 2>/dev/null \
  || sudo -u scraper .venv/bin/python scripts/analyze_signals.py
```

## 4. Recuperer les donnees pour analyse en local

### Option A — rsync periodique vers ton PC
Depuis ton PC Windows (idealement via WSL bash) :
```bash
rsync -avz --progress \
  root@TON_IP:/opt/virtual-sports-scraper/data/virtual_sports.db \
  /d/AGENTOVA/SAMY/virtual-sports-scraper/data/
```
Tu lances ca quand tu veux analyser localement. Tres simple.

### Option B — analyse a distance via SSH
```bash
ssh root@TON_IP "cd /opt/virtual-sports-scraper && \
  sudo -u scraper .venv/bin/python scripts/analyze_deep.py"
```

### Option C — pousser sur S3 / Backblaze (optionnel)
Si tu veux un backup cloud automatique, edite `backup_db.sh` pour ajouter un `aws s3 cp` ou `rclone copy`.

## 5. Sauvegarde quotidienne automatique

```bash
# en root
cp /opt/virtual-sports-scraper/deploy/backup_db.sh /etc/cron.daily/scraper-backup
chmod +x /etc/cron.daily/scraper-backup
```
Garde 14 snapshots horodates dans `/opt/virtual-sports-scraper/backups/`.

## 6. Operations courantes

```bash
# stop / start / restart
systemctl stop scraper
systemctl start scraper
systemctl restart scraper

# changement de .env
sudoedit /opt/virtual-sports-scraper/.env
systemctl restart scraper

# mettre a jour le code
cd /opt/virtual-sports-scraper
sudo -u scraper git pull
sudo -u scraper .venv/bin/pip install -r requirements.txt
systemctl restart scraper
```

## 7. Securite minimale

Une fois SSH avec cle marche, durcis :
```bash
# desactive le password login
sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl reload ssh

# firewall basique
apt-get install -y ufw
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw --force enable
```

## 8. Cout total

| Item | Prix |
|---|---|
| Hetzner CX22 | 4.51 €/mois HT (~5.4 € TTC) |
| Trafic sortant | inclus |
| Stockage DB | inclus (DB < 100 Mo apres 1 mois de collecte) |
| **TOTAL** | **~5 €/mois** |

Aucun token Claude / OpenAI consomme. Aucune API tierce payante. Le scraper tourne en autonomie complete.
