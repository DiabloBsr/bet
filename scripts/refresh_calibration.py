"""Rafraîchit la table de calibration des scores (score_calibration.json) sur les
données RÉCENTES (fenêtre 25 000 matchs). À planifier chaque semaine (schtasks).

correction[h,a] = freq_empirique / freq_modele (apply_sim_deviations), bornée.
Suit le RNG s'il dérive. Sortie ASCII (compatible Task Scheduler).
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

LG = "InstantLeague-8035"; N = 7
CLIP = (0.4, 2.5)
WINDOW = 25000
OUT = ROOT / "data" / "vfoot_ml" / "score_calibration.json"


def main():
    eng = create_engine(load_settings().db_url)
    df = pd.read_sql(text(f"""
        SELECT o.odds_home oh,o.odds_draw od,o.odds_away oa,r.score_a sa,r.score_b sb
        FROM events e JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
        JOIN results r ON r.event_id=e.id
        WHERE r.score_a IS NOT NULL AND e.competition='{LG}' AND o.odds_home>1
        ORDER BY e.expected_start"""), eng).tail(WINDOW)
    df["sa"] = df.sa.clip(0, N - 1); df["sb"] = df.sb.clip(0, N - 1)

    emp = np.zeros((N, N))
    for a, b in zip(df.sa, df.sb):
        emp[int(a), int(b)] += 1
    emp /= emp.sum()

    s = df.sample(min(6000, len(df)), random_state=0)
    mod = np.zeros((N, N)); n = 0
    for r in s.itertuples():
        try:
            lh, la = exact_invert_1x2(r.oh, r.od, r.oa)
            g = np.asarray(apply_sim_deviations(lh, la, "cells"))[:N, :N]
            mod += g / g.sum(); n += 1
        except Exception:
            pass
    mod /= max(n, 1)

    corr = np.clip(emp / np.clip(mod, 1e-5, None), *CLIP)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"N": N, "correction": corr.tolist(),
                               "updated_utc": datetime.now(timezone.utc).isoformat(),
                               "n_matches": int(len(df))}, indent=1), encoding="utf-8")
    l1 = float(np.abs((mod * corr / (mod * corr).sum()) - emp).sum())
    print(f"calibration rafraichie sur {len(df)} matchs (echantillon modele n={n})")
    print(f"L1 residuel apres correction : {l1:.4f} | 3-1: emp={100*emp[3,1]:.2f}% "
          f"mod={100*mod[3,1]:.2f}% corr={corr[3][1]:.2f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
