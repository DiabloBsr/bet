# Virtual Sports Collector

Outil d'observation **passive** d'une page de sports virtuels, pour analyse statistique personnelle.

> **Avertissement.** Cet outil ne contourne aucune protection (captcha, Cloudflare, anti-bot, authentification) et ne place aucun pari. Si la cible bloque l'accès, le scraper s'arrête proprement et il faut basculer sur une source autorisée (API officielle d'odds, export manuel, simulation locale).

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m playwright install chromium
copy .env.example .env
# éditer .env : TARGET_URL, SCRAPE_INTERVAL_SECONDS, etc.
```

## Utilisation

```powershell
# une seule collecte (debug / test des sélecteurs)
python scripts/run_once.py

# boucle continue (Ctrl+C pour stop propre)
python scripts/run_forever.py

# export vers CSV pour pandas
python scripts/export_csv.py --out ./exports

# analyse rapide
python scripts/analyze.py
```

## Configurer pour un nouveau site

Tout se passe dans `scraper/parser.py` :

1. Renseigner `XHR_URL_PATTERNS` avec le fragment d'URL des endpoints JSON (vu dans DevTools → Network → Fetch/XHR).
2. Sinon, ajouter une clé dans `EMBEDDED_JSON_KEYS` (ex. `__NEXT_DATA__`).
3. Sinon, ajuster `DOM_SELECTORS` à partir de l'inspecteur d'éléments.

Le collector essaie XHR → JSON embarqué → DOM dans cet ordre.

## Schéma de base

- `events` : un événement unique par `(external_id, source_url)`
- `odds_snapshots` : N snapshots horodatés par event, dédupliqués par hash de contenu
- `results` : un résultat final par event (unique)
- `scrape_runs` : journal d'exécution (succès/erreur, compteurs)

## Limites connues

- Si la cible utilise un challenge anti-bot persistant, Playwright sera bloqué dès `goto()`. Le run sera marqué `error` dans `scrape_runs` — ne pas chercher à contourner.
- Les sélecteurs DOM sont volatils par nature. Préférer XHR dès que possible.
