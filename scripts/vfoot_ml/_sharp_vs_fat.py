"""Prédire le SCORE : marché gras (Score-exact ~18% marge) VS grille reconstruite
des marchés NETS (1X2 ~5% marge -> Poisson + déviations sim + calibration).

Question : la reconstruction depuis les cotes NETTES bat-elle le marché Score-exact
gras sur le Top-1/Top-3 réel ? Split chrono, calibration sur TRAIN seul (zéro fuite).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings
from scraper.predictor_v2 import market_score_grid, grid_top_k_scores
from scraper.market_inversion import exact_invert_1x2, apply_sim_deviations

LG = "InstantLeague-8035"
eng = create_engine(load_settings().db_url)
df = pd.read_sql(text(f"""
    SELECT o.odds_home oh,o.odds_draw od,o.odds_away oa,o.extra_markets xm,r.score_a sa,r.score_b sb
    FROM events e JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id
    WHERE r.score_a IS NOT NULL AND e.competition='{LG}' AND o.odds_home>1 AND o.odds_draw>1 AND o.odds_away>1
    ORDER BY e.expected_start"""), eng)
n = len(df); cut = int(n * 0.7)
print(f"{n} matchs | train {cut} / test {n-cut}")

# grilles sim (depuis 1X2 net) + score réel
sa6 = df.sa.clip(0, 6).astype(int).values; sb6 = df.sb.clip(0, 6).astype(int).values
G = np.zeros((n, 7, 7)); ok = np.zeros(n, bool)
for i, r in enumerate(df.itertuples()):
    try:
        lh, la = exact_invert_1x2(r.oh, r.od, r.oa)
        g = np.asarray(apply_sim_deviations(lh, la, "cells"), float)[:7, :7]
        G[i] = g / g.sum(); ok[i] = True
    except Exception:
        pass
# calibration 7x7 sur TRAIN seul
emp = np.zeros((7, 7))
for i in range(cut):
    if ok[i]:
        emp[sa6[i], sb6[i]] += 1
emp /= emp.sum()
CAL = np.clip(emp / np.clip(G[:cut][ok[:cut]].mean(0), 1e-5, None), 0.4, 2.5)
Gc = G * CAL[None]; Gc /= Gc.sum((1, 2), keepdims=True) + 1e-12


def top3_from_grid(gr):
    flat = gr.ravel(); order = np.argsort(-flat)[:3]
    return [(o // 7, o % 7) for o in order]


hits = {k: [0, 0] for k in ("marché gras (Score-exact)", "grille NETTE (1X2+sim)",
                            "grille NETTE + calibration", "blend gras×nette")}
cnt = 0
for i in range(cut, n):
    if not ok[i]:
        continue
    actual = (sa6[i], sb6[i]); cnt += 1
    # a) marché gras
    try:
        gm = market_score_grid(json.loads(df.xm.iloc[i]).get("Score exact")
                               if isinstance(df.xm.iloc[i], str) else None)
    except Exception:
        gm = None
    for name, cells in (
        ("marché gras (Score-exact)",
         [tuple(map(int, s.split("-"))) for s, _ in grid_top_k_scores(gm, 3)] if gm is not None else None),
        ("grille NETTE (1X2+sim)", top3_from_grid(G[i])),
        ("grille NETTE + calibration", top3_from_grid(Gc[i])),
    ):
        if cells is None:
            continue
        hits[name][0] += int(actual == cells[0]); hits[name][1] += int(actual in cells)
    # d) blend gras × nette calibrée
    if gm is not None:
        gg = np.zeros((7, 7))
        for s, p in grid_top_k_scores(gm, 25):
            h, a = map(int, s.split("-"))
            if h < 7 and a < 7:
                gg[h, a] = p
        gg = gg / (gg.sum() or 1)
        blend = 0.5 * gg + 0.5 * Gc[i]
        c3 = top3_from_grid(blend)
        hits["blend gras×nette"][0] += int(actual == c3[0]); hits["blend gras×nette"][1] += int(actual in c3)

print(f"\n{cnt} matchs test\n{'source':<32}{'Top-1':>9}{'Top-3':>9}")
for name, (h1, h3) in hits.items():
    if h3:
        print(f"{name:<32}{100*h1/cnt:>8.2f}%{100*h3/cnt:>8.2f}%")
print("\n-> Si 'grille NETTE + calibration' >= 'marché gras', prédire le score depuis les")
print("   marchés à FAIBLE marge est meilleur -> on route le score vers la grille nette.")
