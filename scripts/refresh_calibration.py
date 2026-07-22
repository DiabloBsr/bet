"""Rafraîchit la table de calibration des scores (score_calibration.json) sur les
données RÉCENTES (fenêtre 25 000 matchs PAR LIGUE). À planifier chaque semaine.

correction[h,a] = freq_empirique / freq_modele (apply_sim_deviations), bornée.
Suit le RNG s'il dérive. Sortie ASCII (compatible Task Scheduler).

UNE TABLE PAR LIGUE. Les constantes du simulateur (MU_BOOST_H/A, RHO_SIM,
SIM_CELL_BOOST) ont été ajustées sur la seule ligue anglaise ; appliquer sa table
ailleurs dé-calibre (mesuré sur CAN, lambda 1.49 vs 2.83 : ecart max 3.5pp -> 8.0pp).
Une correction propre à chaque ligue absorbe d'un coup tout ce biais anglais.
Le champ "correction" reste en tête pour la compatibilité ascendante (= anglaise).
"""
from __future__ import annotations
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)                       # db_url relative -> CWD projet obligatoire
sys.path.insert(0, str(ROOT))
if sys.stdout is None:               # pythonw / Task Scheduler
    (ROOT / "data" / "logs").mkdir(parents=True, exist_ok=True)
    _lg = open(ROOT / "data" / "logs" / "calib.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = _lg
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings
from scraper.market_inversion import exact_invert_1x2, apply_sim_deviations

LG = "InstantLeague-8035"          # ligue de reference (compat ascendante)
N = 7
CLIP = (0.4, 2.5)
WINDOW = 25000                     # matchs les + recents retenus par ligue
MIN_N = 3000                       # en dessous, l'estimation empirique est trop bruitee
SAMPLE = 4000                      # inversions de marche par ligue (cout scipy)
OUT = ROOT / "data" / "vfoot_ml" / "score_calibration.json"

# Une seule passe, sans sous-requete correlee (celles-ci verrouillent la base
# pendant que le scraper ecrit -> "database is locked").
SQL = """
    SELECT e.competition lg, o.odds_home oh, o.odds_draw od, o.odds_away oa,
           r.score_a sa, r.score_b sb
    FROM events e
    JOIN (SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) f
      ON f.event_id = e.id
    JOIN odds_snapshots o ON o.id = f.mid
    JOIN results r ON r.event_id = e.id
    WHERE r.score_a IS NOT NULL AND o.odds_home > 1
    ORDER BY e.expected_start
"""


def _calibrate(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """(correction, empirique, modele, n_inversions) pour un sous-ensemble de matchs."""
    sa = df.sa.clip(0, N - 1).astype(int)
    sb = df.sb.clip(0, N - 1).astype(int)
    emp = np.zeros((N, N))
    for a, b in zip(sa, sb):
        emp[a, b] += 1
    emp /= emp.sum()

    s = df.sample(min(SAMPLE, len(df)), random_state=0)
    mod = np.zeros((N, N)); n = 0
    for r in s.itertuples():
        try:
            lh, la = exact_invert_1x2(r.oh, r.od, r.oa)
            g = np.asarray(apply_sim_deviations(lh, la, "cells"))[:N, :N]
            mod += g / g.sum(); n += 1
        except Exception:
            pass
    mod /= max(n, 1)
    return np.clip(emp / np.clip(mod, 1e-5, None), *CLIP), emp, mod, n


def main():
    eng = create_engine(load_settings().db_url)
    allrows = pd.read_sql(text(SQL), eng)
    print(f"{len(allrows)} matchs joues, {allrows.lg.nunique()} ligues")

    per_league, ref = {}, None
    for lg, grp in allrows.groupby("lg", sort=False):
        df = grp.tail(WINDOW)
        if len(df) < MIN_N:
            print(f"  {lg:<22} n={len(df):<7} IGNOREE (< {MIN_N})")
            continue
        corr, emp, mod, n = _calibrate(df)
        per_league[lg] = corr.tolist()
        # L1 residuel : distance entre le modele corrige et la realite (0 = parfait)
        l1_av = float(np.abs(mod - emp).sum())
        cm = mod * corr; cm /= cm.sum()
        l1_ap = float(np.abs(cm - emp).sum())
        print(f"  {lg:<22} n={len(df):<7} inv={n:<5} L1 {l1_av:.4f} -> {l1_ap:.4f}"
              f"  ({100*(1-l1_ap/max(l1_av,1e-9)):+.0f}%)")
        if lg == LG:
            ref = corr

    if not per_league:
        raise SystemExit("aucune ligue calibrable")
    if ref is None:                          # ligue de reference absente : on prend la + fournie
        ref = np.asarray(per_league[max(per_league, key=lambda k: 1)], float)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "N": N,
        "correction": np.asarray(ref, float).tolist(),   # compat ascendante = LG
        "reference_league": LG,
        "per_league": per_league,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "n_matches": int(len(allrows)),
    }, indent=1), encoding="utf-8")
    print(f"-> {OUT}  ({len(per_league)} tables)")


if __name__ == "__main__":
    main()
