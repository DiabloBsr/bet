"""Grille de score AJUSTÉE À TOUS LES MARCHÉS (IPF) — le prédicteur de score à
information maximale : la grille 7x7 forcée d'être cohérente avec 1X2 + O/U 3.5 +
Total de buts + BTTS simultanément (chaque marché = contrainte marginale).

Combine les observations bruitées de plusieurs marchés -> estimation de la vraie
grille RNG moins bruitée qu'une source unique. Bat-elle le meilleur single-source
(31.56% Top-3) ? Split chrono, calibration + IPF calés sur les données visibles.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings
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
print(f"{n} matchs | train {cut} / test {n-cut}", flush=True)

sa6 = df.sa.clip(0, 6).astype(int).values; sb6 = df.sb.clip(0, 6).astype(int).values
I, J = np.meshgrid(np.arange(7), np.arange(7), indexing="ij")
TOT = I + J
R_H, R_X, R_A = TOT[I > J], None, None
M_H, M_X, M_A = (I > J), (I == J), (I < J)
M_O35 = (TOT > 3.5); M_GG = (I > 0) & (J > 0)
M_TOTK = [(TOT == k) if k < 6 else (TOT >= 6) for k in range(7)]

G = np.zeros((n, 7, 7)); ok = np.zeros(n, bool)
for i, r in enumerate(df.itertuples()):
    try:
        lh, la = exact_invert_1x2(r.oh, r.od, r.oa)
        g = np.asarray(apply_sim_deviations(lh, la, "cells"), float)[:7, :7]
        G[i] = g / g.sum(); ok[i] = True
    except Exception:
        pass
emp = np.zeros((7, 7))
for i in range(cut):
    if ok[i]:
        emp[sa6[i], sb6[i]] += 1
emp /= emp.sum()
CAL = np.clip(emp / np.clip(G[:cut][ok[:cut]].mean(0), 1e-5, None), 0.4, 2.5)
Gc = G * CAL[None]; Gc /= Gc.sum((1, 2), keepdims=True) + 1e-12


def devig(sels, keys):
    v = {}
    for k in keys:
        o = (sels or {}).get(k)
        if isinstance(o, (int, float)) and 1 < o < 99.99:
            v[k] = 1/o
    s = sum(v.values())
    return {k: v[k]/s for k in v} if s > 0 and len(v) == len(keys) else None


def gm(xm, pref):
    for k, val in (xm or {}).items():
        if k.replace("\x82", "é").replace("\xe9", "é").startswith(pref):
            return val
    return None


def ipf(grid, oh, od, oa, xm, iters=12):
    g = grid.copy()
    inv = 1/oh + 1/od + 1/oa
    t1x2 = [(1/oh)/inv, (1/od)/inv, (1/oa)/inv]
    tot = devig(gm(xm, "Total de buts"), [str(k) for k in range(7)])
    ou = devig(gm(xm, "+/-"), ["> 3.5", "< 3.5"])
    gg = devig(gm(xm, "G/NG"), ["Oui", "Non"])
    for _ in range(iters):
        # 1X2
        for mask, tgt in ((M_H, t1x2[0]), (M_X, t1x2[1]), (M_A, t1x2[2])):
            cur = g[mask].sum()
            if cur > 1e-9:
                g[mask] *= tgt / cur
        # total de buts (0..6+)
        if tot:
            for k, mask in enumerate(M_TOTK):
                cur = g[mask].sum()
                if cur > 1e-9 and str(k) in tot:
                    g[mask] *= tot[str(k)] / cur
        # over/under 3.5
        if ou:
            for mask, tgt in ((M_O35, ou["> 3.5"]), (~M_O35, ou["< 3.5"])):
                cur = g[mask].sum()
                if cur > 1e-9:
                    g[mask] *= tgt / cur
        # BTTS
        if gg:
            for mask, tgt in ((M_GG, gg["Oui"]), (~M_GG, gg["Non"])):
                cur = g[mask].sum()
                if cur > 1e-9:
                    g[mask] *= tgt / cur
        g /= g.sum() or 1
    return g


def top3(gr):
    o = np.argsort(-gr.ravel())[:3]
    return [(x // 7, x % 7) for x in o]


hits = {"single (nette+calib)": [0, 0], "IPF tous marchés": [0, 0]}
cnt = 0
for i in range(cut, n):
    if not ok[i]:
        continue
    try:
        xm = json.loads(df.xm.iloc[i]) if isinstance(df.xm.iloc[i], str) else (df.xm.iloc[i] or {})
    except Exception:
        xm = {}
    cnt += 1; actual = (sa6[i], sb6[i])
    c1 = top3(Gc[i])
    gj = ipf(Gc[i], df.oh.iloc[i], df.od.iloc[i], df.oa.iloc[i], xm)
    c2 = top3(gj)
    hits["single (nette+calib)"][0] += int(actual == c1[0]); hits["single (nette+calib)"][1] += int(actual in c1)
    hits["IPF tous marchés"][0] += int(actual == c2[0]); hits["IPF tous marchés"][1] += int(actual in c2)
    if cnt % 3000 == 0:
        print(f"  {cnt}…", flush=True)

print(f"\n{cnt} matchs test\n{'source':<26}{'Top-1':>9}{'Top-3':>9}")
for name, (h1, h3) in hits.items():
    print(f"{name:<26}{100*h1/cnt:>8.2f}%{100*h3/cnt:>8.2f}%")
d1 = 100*(hits['IPF tous marchés'][1]-hits['single (nette+calib)'][1])/cnt
print(f"\ngain IPF sur Top-3 : {d1:+.2f}pp  -> "
      f"{'AMÉLIORATION réelle' if d1 > 0.5 else 'nul (bruit) — marchés redondants, un seul grid'}")
