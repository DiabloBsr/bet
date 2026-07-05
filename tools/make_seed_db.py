"""Fabrique la DB SEED pour le déploiement cloud.

Copie events + results + odds_snapshots en NULLant les colonnes lourdes
(extra_markets, goals) des données HISTORIQUES : les fits V5/V2 n'ont besoin
que des cotes 1X2 + scores ; les extra_markets ne servent qu'aux matchs À
VENIR, que le scraper cloud capturera lui-même.
Garde les extra_markets des 7 derniers jours (continuité tracker/app).

-> data/seed/virtual_sports_seed.db (puis gzip pour la release GitHub)
"""
from __future__ import annotations
import sqlite3, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "virtual_sports.db"
DST = ROOT / "data" / "seed" / "virtual_sports_seed.db"
DST.parent.mkdir(parents=True, exist_ok=True)
if DST.exists():
    DST.unlink()

src = sqlite3.connect(f"file:{SRC.as_posix()}?mode=ro", uri=True, timeout=60)
dst = sqlite3.connect(DST.as_posix())
dst.execute("PRAGMA journal_mode=OFF"); dst.execute("PRAGMA synchronous=OFF")

TABLES = ["events", "results", "odds_snapshots", "team_rankings", "scrape_runs"]
HEAVY = {"extra_markets", "goals", "raw_payload", "raw"}

for t in TABLES:
    row = src.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (t,)).fetchone()
    if not row:
        print(f"  (table {t} absente — sautée)"); continue
    dst.execute(row[0])
    cols = [r[1] for r in src.execute(f"PRAGMA table_info({t})")]
    if t == "odds_snapshots":
        sel = ", ".join(
            f"CASE WHEN captured_at >= datetime('now','-7 days') THEN {c} ELSE NULL END AS {c}"
            if c in HEAVY else c for c in cols)
    elif t == "results":
        sel = ", ".join("NULL AS " + c if c in HEAVY else c for c in cols)
    elif t == "scrape_runs":
        sel = None  # schéma seul (historique de runs inutile au cloud)
    else:
        sel = ", ".join(cols)
    if sel:
        t0 = time.time()
        cur = src.execute(f"SELECT {sel} FROM {t}")
        ph = ", ".join("?" * len(cols))
        ins = f"INSERT INTO {t} ({', '.join(cols)}) VALUES ({ph})"
        n = 0
        while True:
            rows = cur.fetchmany(50000)
            if not rows:
                break
            dst.executemany(ins, rows)
            n += len(rows)
        dst.commit()
        print(f"  {t}: {n} lignes ({time.time()-t0:.0f}s)", flush=True)
    else:
        print(f"  {t}: schéma seul", flush=True)

# index utiles (accélèrent les requêtes de l'app)
for ix in src.execute("SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"):
    try:
        dst.execute(ix[0])
    except Exception:
        pass
dst.commit()
dst.execute("VACUUM")
dst.close(); src.close()
print(f"seed : {DST} — {DST.stat().st_size/1e6:.0f} Mo")
