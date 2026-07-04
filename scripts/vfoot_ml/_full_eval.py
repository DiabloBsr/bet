"""Audit complet : score réel (base) vs prédiction calibrée, en OOS.

Mesure tout : accuracy score exact (top-1/3), 1X2, Over/Under, BTTS ; calibration
des probabilités ; accuracy par force de favori ; match de distribution ; effet de
la calibration ; comparaison au plafond bayésien. -> où peut-on encore améliorer ?
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
calib = np.asarray(json.load(open("data/vfoot_ml/score_calibration.json"))["correction"], float)


def x12(h, a):
    return 0 if h > a else (1 if h == a else 2)


def main():
    e = create_engine(load_settings().db_url)
    df = pd.read_sql(text(f"""
        SELECT e.expected_start ts, o.odds_home oh,o.odds_draw od,o.odds_away oa, r.score_a sa,r.score_b sb
        FROM events e JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
        JOIN results r ON r.event_id=e.id
        WHERE r.score_a IS NOT NULL AND e.competition='{LG}' AND o.odds_home>1
        ORDER BY e.expected_start"""), e)
    df["sa"] = df.sa.clip(0, N - 1); df["sb"] = df.sb.clip(0, N - 1)
    cut = int(len(df) * 0.7)
    te = df.iloc[cut:].reset_index(drop=True)        # OOS (jamais vu par la calibration)
    print(f"base totale={len(df)} | éval OOS sur {len(te)} matchs\n")

    res = {"raw": {}, "cal": {}}
    for tag in ("raw", "cal"):
        res[tag] = {"t1": 0, "t3": 0, "x12": 0, "ou": 0, "btts": 0}
    cal_bins = {}                                    # calibration : prob_top1 -> [hits, n]
    fav_acc = {}                                     # accuracy par bande de favori
    pred_grid = np.zeros((N, N)); real_grid = np.zeros((N, N))

    for r in te.itertuples():
        try:
            lh, la = exact_invert_1x2(r.oh, r.od, r.oa)
            g0 = np.asarray(apply_sim_deviations(lh, la, "cells"))[:N, :N]; g0 = g0 / g0.sum()
        except Exception:
            continue
        gc = g0 * calib; gc = gc / gc.sum()
        sa, sb = int(r.sa), int(r.sb); real = sa * N + sb
        real_grid[sa, sb] += 1
        rx = x12(sa, sb); rover = (sa + sb) > 2.5; rbtts = (sa > 0 and sb > 0)
        for tag, g in (("raw", g0), ("cal", gc)):
            order = np.argsort(-g.ravel())
            t1 = order[0]; top3 = set(order[:3].tolist())
            res[tag]["t1"] += (t1 == real); res[tag]["t3"] += (real in top3)
            ph = np.tril(g, -1).sum(); px = np.trace(g); pa = np.triu(g, 1).sum()
            res[tag]["x12"] += (int(np.argmax([ph, px, pa])) == rx)
            pover = sum(g[h, a] for h in range(N) for a in range(N) if h + a > 2.5)
            res[tag]["ou"] += ((pover > 0.5) == rover)
            pb = sum(g[h, a] for h in range(1, N) for a in range(1, N))
            res[tag]["btts"] += ((pb > 0.5) == rbtts)
        # calibration sur le modèle calibré
        order = np.argsort(-gc.ravel()); t1 = order[0]; p1 = gc.ravel()[t1]
        b = round(p1 * 100)
        cal_bins.setdefault(b, [0, 0]); cal_bins[b][0] += (t1 == real); cal_bins[b][1] += 1
        pred_grid[t1 // N, t1 % N] += 1
        # favori
        fav = max(1 / r.oh, 1 / r.oa) / (1 / r.oh + 1 / r.od + 1 / r.oa)
        band = "équilibré(<0.45)" if fav < 0.45 else ("moyen(0.45-0.6)" if fav < 0.6 else "fort(>0.6)")
        fav_acc.setdefault(band, [0, 0, 0]); fav_acc[band][0] += (t1 == real)
        fav_acc[band][1] += (real in set(np.argsort(-gc.ravel())[:3].tolist())); fav_acc[band][2] += 1

    n = len(te)
    print("=== ACCURACY OOS : modèle brut vs CALIBRÉ vs plafond ===")
    print(f"  {'métrique':<22}{'brut':>8}{'calibré':>10}{'plafond':>10}")
    ceil = {"t1": 12.0, "t3": 31.0, "x12": 55.5, "ou": 62.0, "btts": 57.0}
    names = {"t1": "score exact Top-1", "t3": "score exact Top-3", "x12": "1X2",
             "ou": "Over/Under 2.5", "btts": "BTTS"}
    for k in ("t1", "t3", "x12", "ou", "btts"):
        print(f"  {names[k]:<22}{100*res['raw'][k]/n:>7.1f}%{100*res['cal'][k]/n:>9.1f}%{ceil[k]:>9.1f}%")

    print("\n=== CALIBRATION des probas (prob annoncée top-1 -> réalisé) ===")
    for b in sorted(cal_bins):
        hit, cnt = cal_bins[b]
        if cnt >= 80:
            print(f"  annonce ~{b}% -> réalisé {100*hit/cnt:5.1f}% (n={cnt})")

    print("\n=== accuracy par force de favori (calibré) ===")
    for band, (h1, h3, c) in sorted(fav_acc.items()):
        print(f"  {band:<18} Top-1 {100*h1/c:4.1f}%  Top-3 {100*h3/c:4.1f}%  (n={c})")

    pred_grid /= pred_grid.sum(); real_grid /= real_grid.sum()
    print(f"\n=== match de distribution (prédit-argmax vs réel) L1={np.abs(pred_grid-real_grid).sum():.3f} ===")
    print("  (note : l'argmax concentre sur peu de scores -> normal d'avoir un écart ici)")


if __name__ == "__main__":
    main()
