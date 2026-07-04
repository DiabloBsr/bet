"""PAPER-TRADER forward du signal MOUVEMENT DE LIGNE — à planifier (30 min).

Pour chaque match DÉMARRÉ depuis le déploiement (2026-07-02) ayant >=2 snapshots
pré-coup d'envoi : loggue un pari VIRTUEL domicile à la cote de clôture avec le
mouvement mesuré (ouverture -> clôture). Règle au résultat. Zéro look-ahead :
la décision n'utilise que des données antérieures au coup d'envoi.
Table : line_paper_bets (une ligne par match éligible, INSERT OR IGNORE).
Rapport : ROI forward par seuil de mouvement — la preuve définitive du signal.
"""
from __future__ import annotations
import os, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
LOGS = ROOT / "data" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
if sys.stdout is None:               # pythonw / Task Scheduler
    _lg = open(LOGS / "line_paper.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = _lg
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

LG = "InstantLeague-%"               # POOL des 9 ligues (même fournisseur/marge)
DEPLOY = "2026-07-02"                # début du forward — ne jamais reculer
THRESHOLDS = (0.005, 0.01, 0.02)

DDL = """
CREATE TABLE IF NOT EXISTS line_paper_bets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_key TEXT UNIQUE, comp TEXT, ts TEXT, team_a TEXT, team_b TEXT,
  oh_open REAL, oh_close REAL, od_close REAL, oa_close REAL,
  od_open REAL, oa_open REAL, move_h REAL, move_d REAL, move_a REAL,
  res TEXT, placed_at TEXT,
  home_win INTEGER, pnl_home REAL
)
"""
MIGRATIONS = ["od_open REAL", "oa_open REAL", "move_d REAL", "move_a REAL", "res TEXT"]

SQL_NEW = f"""
WITH snaps AS (
  SELECT o.event_id, o.captured_at, o.odds_home, o.odds_draw, o.odds_away,
    ROW_NUMBER() OVER (PARTITION BY o.event_id ORDER BY o.captured_at ASC) ro,
    ROW_NUMBER() OVER (PARTITION BY o.event_id ORDER BY o.captured_at DESC) rc
  FROM odds_snapshots o JOIN events e ON e.id=o.event_id
  WHERE e.competition LIKE '{LG}' AND o.captured_at < e.expected_start
    AND o.odds_home>1 AND o.odds_draw>1 AND o.odds_away>1)
SELECT e.competition comp, e.expected_start ts, e.team_a, e.team_b,
  op.odds_home oh_o, op.odds_draw od_o, op.odds_away oa_o,
  cl.odds_home oh_c, cl.odds_draw od_c, cl.odds_away oa_c
FROM events e
JOIN snaps op ON op.event_id=e.id AND op.ro=1
JOIN snaps cl ON cl.event_id=e.id AND cl.rc=1
WHERE e.competition LIKE '{LG}' AND op.captured_at < cl.captured_at
  AND e.expected_start >= '{DEPLOY}'
  AND e.expected_start < :now
"""


def imp_h(h, d, a):
    inv = 1/h + 1/d + 1/a
    return (1/h)/inv


def main():
    eng = create_engine(load_settings().db_url, connect_args={"timeout": 30})
    with eng.begin() as cx:
        cx.execute(text(DDL))
        for col in MIGRATIONS:           # table existante -> ajoute les colonnes manquantes
            try:
                cx.execute(text(f"ALTER TABLE line_paper_bets ADD COLUMN {col}"))
            except Exception:
                pass
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # 1) ingestion des matchs démarrés (décision figée à la clôture, avant le résultat)
    new = pd.read_sql(text(SQL_NEW), eng, params={"now": now})
    n_ins = 0
    with eng.begin() as cx:
        for r in new.itertuples():
            key = f"{r.comp}|{r.team_a}|{r.team_b}|{r.ts}"
            inv_c = 1/r.oh_c + 1/r.od_c + 1/r.oa_c
            inv_o = 1/r.oh_o + 1/r.od_o + 1/r.oa_o
            mv_h = (1/r.oh_c)/inv_c - (1/r.oh_o)/inv_o
            mv_d = (1/r.od_c)/inv_c - (1/r.od_o)/inv_o
            mv_a = (1/r.oa_c)/inv_c - (1/r.oa_o)/inv_o
            res = cx.execute(text("""INSERT OR IGNORE INTO line_paper_bets
                (event_key, comp, ts, team_a, team_b, oh_open, oh_close, od_close, oa_close,
                 od_open, oa_open, move_h, move_d, move_a, placed_at)
                VALUES (:k,:c,:ts,:ta,:tb,:oo,:oc,:dc,:ac,:do,:ao,:mh,:md,:ma,:pa)"""),
                {"k": key, "c": r.comp, "ts": str(r.ts), "ta": r.team_a, "tb": r.team_b,
                 "oo": float(r.oh_o), "oc": float(r.oh_c), "dc": float(r.od_c),
                 "ac": float(r.oa_c), "do": float(r.od_o), "ao": float(r.oa_o),
                 "mh": round(float(mv_h), 5), "md": round(float(mv_d), 5),
                 "ma": round(float(mv_a), 5), "pa": now})
            n_ins += res.rowcount or 0

    # 2) règlement (résultat 1/X/2)
    with eng.begin() as cx:
        cx.execute(text("""
            UPDATE line_paper_bets SET
              res = (SELECT CASE WHEN r.score_a > r.score_b THEN '1'
                                 WHEN r.score_a = r.score_b THEN 'X' ELSE '2' END
                     FROM events e JOIN results r ON r.event_id=e.id
                     WHERE e.competition=line_paper_bets.comp
                       AND e.team_a=line_paper_bets.team_a
                       AND e.team_b=line_paper_bets.team_b
                       AND e.expected_start=line_paper_bets.ts AND r.score_a IS NOT NULL
                     LIMIT 1)
            WHERE res IS NULL"""))
        cx.execute(text("""UPDATE line_paper_bets SET home_win = (res='1'),
            pnl_home = (res='1')*oh_close - 1 WHERE res IS NOT NULL AND pnl_home IS NULL"""))

    # 3) rapport forward : 3 signaux, à la cote de CLÔTURE + CLV (cote d'OUVERTURE)
    d = pd.read_sql(text("""SELECT move_h, move_d, move_a, oh_open, oh_close, od_open,
        od_close, oa_open, oa_close, res FROM line_paper_bets WHERE res IS NOT NULL"""), eng)
    tot = pd.read_sql(text("SELECT COUNT(*) c FROM line_paper_bets"), eng).c.iloc[0]
    print(f"[{now}] ingest +{n_ins} | ledger {int(tot)} matchs, {len(d)} regles (forward depuis {DEPLOY})")
    SIGNALS = [("DOM", "move_h", "1", "oh_close", "oh_open"),
               ("EXT", "move_a", "2", "oa_close", "oa_open"),
               ("NUL", "move_d", "X", "od_close", "od_open")]
    for sig, mv, win, oc, oo in SIGNALS:
        for thr in THRESHOLDS:
            b = d[d[mv].notna() & (d[mv] > thr)]
            if not len(b):
                continue
            hit = (b.res == win).astype(int)
            roi_c = (hit * b[oc] - 1).mean()
            bo = b[b[oo].notna()]
            clv = float(((bo.res == win).astype(int) * bo[oo] - 1).mean()) if len(bo) else float("nan")
            cl = f"{100*clv:+6.2f}%" if clv == clv else "   n/a"
            print(f"    {sig} move>{thr}: n={len(b):>5}  ROI cloture {100*roi_c:+6.2f}%"
                  f"  | ROI ouverture (CLV) {cl}  (hit {100*hit.mean():.0f}%)")


if __name__ == "__main__":
    main()
