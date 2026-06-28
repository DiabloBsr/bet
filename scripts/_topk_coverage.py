"""Quelle DISTRIBUTION couvre le mieux le vrai score, et avec COMBIEN de scores
atteint-on 70 / 80 / 90% de fiabilité ? On essaie 6 distributions et on mesure
la couverture Top-K (OOS) pour K=1..14. But : maximiser la fiabilité par niveau.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.market_inversion import exact_invert_1x2, apply_sim_deviations, _fast_grid
from scraper.score_final import FAV_EDGES, TOT_EDGES, _band

e = create_engine(load_settings().db_url)
d = pd.read_sql("""SELECT o.odds_home oh,o.odds_draw od,o.odds_away oa,e.expected_start,
  r.score_a sa,r.score_b sb FROM events e
  JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
  JOIN results r ON r.event_id=e.id WHERE r.score_a IS NOT NULL AND e.competition='InstantLeague-8035'""", e)
d = d[(d.oh > 1) & (d.od > 1) & (d.oa > 1)].copy()
d["es"] = pd.to_datetime(d.expected_start, utc=True, errors="coerce")
d = d.dropna(subset=["es"]).sort_values("es").reset_index(drop=True)
print("inversion...", flush=True)
cl = {}
def lam(oh, od, oa):
    k = (round(oh, 2), round(od, 2), round(oa, 2))
    if k not in cl: cl[k] = exact_invert_1x2(oh, od, oa)
    return cl[k]
L = d.apply(lambda r: lam(r.oh, r.od, r.oa), axis=1)
d["lh"] = [x[0] for x in L]; d["la"] = [x[1] for x in L]; d["lt"] = d["lh"] + d["la"]
d["fav_home"] = d.oh < d.oa; d["favc"] = d[["oh", "oa"]].min(axis=1)
fg = np.where(d.fav_home, d.sa, d.sb); dg = np.where(d.fav_home, d.sb, d.sa)
d["fs"] = pd.Series(fg.astype(int).astype(str), index=d.index) + "-" + pd.Series(dg.astype(int).astype(str), index=d.index)
d["fb"] = d.favc.map(lambda v: _band(v, FAV_EDGES)); d["tb"] = d["lt"].map(lambda v: _band(v, TOT_EDGES))
n = len(d); cut = int(n * 0.7); tr, te = d.iloc[:cut].copy(), d.iloc[cut:].copy()
print(f"n={n} | test={len(te)}\n")

M = 8
# distribution empirique par bucket (fav-oriented), fit TRAIN
emp = defaultdict(Counter)
for r in tr.itertuples():
    emp[(r.fb, r.tb)][r.fs] += 1
emp_norm = {k: {s: c / sum(v.values()) for s, c in v.items()} for k, v in emp.items()}
glob_emp = Counter(tr.fs); glob_norm = {s: c / len(tr) for s, c in glob_emp.items()}

def fav_grid(r, mode):
    g = apply_sim_deviations(r.lh, r.la, mode)
    return g if r.fav_home else g.T
def pois_grid(r):
    g = _fast_grid(r.lh, r.la, 0.0)
    return g if r.fav_home else g.T
def grid_to_dict(g):
    return {f"{i}-{j}": g[i, j] for i in range(M) for j in range(M)}

def dist_cells(r): return grid_to_dict(fav_grid(r, "cells"))
def dist_dc(r): return grid_to_dict(fav_grid(r, "dc"))
def dist_pois(r): return grid_to_dict(pois_grid(r))
def dist_emp(r): return emp_norm.get((r.fb, r.tb), glob_norm)
def dist_blend(r):  # grille(cells) × empirique
    g = grid_to_dict(fav_grid(r, "cells")); ed = emp_norm.get((r.fb, r.tb), glob_norm)
    return {s: g.get(s, 0) * ed.get(s, 1e-6) for s in set(g) | set(ed)}
def dist_ens(r):  # moyenne grille+empirique
    g = grid_to_dict(fav_grid(r, "cells")); ed = emp_norm.get((r.fb, r.tb), glob_norm)
    keys = set(g) | set(ed); return {s: 0.5 * g.get(s, 0) + 0.5 * ed.get(s, 0) for s in keys}

DISTS = {"grille cells": dist_cells, "grille dc": dist_dc, "Poisson pur": dist_pois,
         "empirique bucket": dist_emp, "blend grille×emp": dist_blend, "ensemble 50/50": dist_ens}
KS = [1, 2, 3, 5, 7, 9, 11, 13]
cov = {nm: {k: 0 for k in KS} for nm in DISTS}
for r in te.itertuples():
    for nm, fn in DISTS.items():
        ranked = [s for s, _ in sorted(fn(r).items(), key=lambda x: -x[1])]
        pos = ranked.index(r.fs) + 1 if r.fs in ranked else 999
        for k in KS:
            if pos <= k: cov[nm][k] += 1
N = len(te)
print("="*78)
print("COUVERTURE Top-K (% des matchs où le VRAI score est dans les K meilleurs) — OOS")
print("="*78)
hdr = "distribution".ljust(20) + "".join(f"K={k}".rjust(7) for k in KS)
print(hdr); print("-" * len(hdr))
for nm in DISTS:
    print(nm.ljust(20) + "".join(f"{cov[nm][k]/N*100:6.1f}" for k in KS))
# meilleure distribution par K + combien de scores pour 70/80/90%
best = {k: max(DISTS, key=lambda nm: cov[nm][k]) for k in KS}
print("\n  meilleure distribution par K :")
for k in KS:
    print(f"    K={k:<2} : {best[k]:<18} -> {cov[best[k]][k]/N*100:.1f}%")
bestfn = DISTS[max(DISTS, key=lambda nm: cov[nm][13])]
# courbe fine pour trouver K(70/80/90)
covK = []
for r in te.itertuples():
    ranked = [s for s, _ in sorted(bestfn(r).items(), key=lambda x: -x[1])]
    covK.append(ranked.index(r.fs) + 1 if r.fs in ranked else 999)
covK = np.array(covK)
print("\n  Avec la MEILLEURE distribution, combien de scores pour atteindre :")
for target in [50, 60, 70, 80, 90]:
    kneed = next((k for k in range(1, 25) if (covK <= k).mean() * 100 >= target), None)
    print(f"    {target}% de fiabilité -> jouer les {kneed} meilleurs scores")
print("\n  => Top-3 plafonne ~31%. Pour 7-9/10 fiable, il faut couvrir ~Top-8 à Top-13.")
