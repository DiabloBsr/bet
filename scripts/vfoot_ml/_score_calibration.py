"""Calibration des scores : corrige les biais par cellule du modèle réalisé
(ex. 3-1 sous-estimé) en alignant sa marginale sur la fréquence empirique.

correction[h,a] = freq_empirique[h,a] / freq_modele[h,a]   (bornée)
appliquée à chaque grille de match puis renormalisée.

Fit sur TRAIN, validé en OOS sur TEST. Sauve data/vfoot_ml/score_calibration.json.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings
from scraper.market_inversion import exact_invert_1x2, apply_sim_deviations

LG = "InstantLeague-8035"; N = 7
CLIP = (0.4, 2.5)


def load():
    e = create_engine(load_settings().db_url)
    df = pd.read_sql(text(f"""
        SELECT e.expected_start ts, o.odds_home oh,o.odds_draw od,o.odds_away oa, r.score_a sa,r.score_b sb
        FROM events e JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
        JOIN results r ON r.event_id=e.id
        WHERE r.score_a IS NOT NULL AND e.competition='{LG}' AND o.odds_home>1
        ORDER BY e.expected_start"""), e)
    df["sa"] = df.sa.clip(0, N - 1); df["sb"] = df.sb.clip(0, N - 1)
    return df


def emp_grid(df):
    g = np.zeros((N, N))
    for sa, sb in zip(df.sa, df.sb):
        g[int(sa), int(sb)] += 1
    return g / g.sum()


def model_grid(df, sample=5000):
    s = df.sample(min(sample, len(df)), random_state=0)
    acc = np.zeros((N, N)); n = 0
    for r in s.itertuples():
        try:
            lh, la = exact_invert_1x2(r.oh, r.od, r.oa)
            g = np.asarray(apply_sim_deviations(lh, la, "cells"))[:N, :N]
            acc += g / g.sum(); n += 1
        except Exception:
            pass
    return acc / n


def main():
    df = load()
    cut = df.ts.iloc[len(df) * 7 // 10]
    tr, te = df[df.ts < cut], df[df.ts >= cut]
    print(f"train={len(tr)} test={len(te)}")

    emp_tr = emp_grid(tr); mod_tr = model_grid(tr)
    corr = np.clip(emp_tr / np.clip(mod_tr, 1e-5, None), *CLIP)

    # validation OOS : marginale corrigée vs empirique TEST
    emp_te = emp_grid(te); mod_te = model_grid(te)
    cal_te = mod_te * corr; cal_te /= cal_te.sum()

    def err(g):  # erreur L1 vs empirique test
        return float(np.abs(g - emp_te).sum())
    print(f"\nErreur L1 de marginale vs réel(TEST) : modèle brut={err(mod_te):.4f}  "
          f"-> calibré={err(cal_te):.4f}")
    print(f"{'score':<7}{'réel(test)':<12}{'modèle':<10}{'calibré':<10}")
    for sc in ["3-1", "2-2", "1-3", "3-2", "4-1", "1-1", "2-1"]:
        h, a = map(int, sc.split("-"))
        print(f"{sc:<7}{100*emp_te[h,a]:<12.2f}{100*mod_te[h,a]:<10.2f}{100*cal_te[h,a]:<10.2f}")

    Path("data/vfoot_ml").mkdir(parents=True, exist_ok=True)
    Path("data/vfoot_ml/score_calibration.json").write_text(
        json.dumps({"N": N, "correction": corr.tolist()}, indent=1), encoding="utf-8")
    print("\n-> data/vfoot_ml/score_calibration.json écrit.")


if __name__ == "__main__":
    main()
