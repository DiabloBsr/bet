"""CHAÎNAGE + VOTE : on fusionne 6 distributions (grille cells/dc/Poisson, empirique
bucket, blend, ensemble) par 3 méthodes (moyenne de probas, vote Borda, vote pondéré
par accuracy) et on mesure si le Top-3 VOTÉ bat le meilleur seul (~31.3%). OOS.
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
emp = defaultdict(Counter)
for r in tr.itertuples():
    emp[(r.fb, r.tb)][r.fs] += 1
emp_norm = {k: {s: c / sum(v.values()) for s, c in v.items()} for k, v in emp.items()}
glob_norm = {s: c / len(tr) for s, c in Counter(tr.fs).items()}

def fav_grid(r, mode):
    g = apply_sim_deviations(r.lh, r.la, mode); g = g if r.fav_home else g.T
    return {f"{i}-{j}": float(g[i, j]) for i in range(M) for j in range(M)}
def pois(r):
    g = _fast_grid(r.lh, r.la, 0.0); g = g if r.fav_home else g.T
    return {f"{i}-{j}": float(g[i, j]) for i in range(M) for j in range(M)}
def norm(dd):
    s = sum(dd.values()); return {k: v / s for k, v in dd.items()} if s > 0 else dd
def D_cells(r): return fav_grid(r, "cells")
def D_dc(r): return fav_grid(r, "dc")
def D_pois(r): return pois(r)
def D_emp(r): return emp_norm.get((r.fb, r.tb), glob_norm)
def D_blend(r):
    g = fav_grid(r, "cells"); ed = emp_norm.get((r.fb, r.tb), glob_norm)
    return norm({s: g.get(s, 0) * ed.get(s, 1e-6) for s in set(g) | set(ed)})
def D_ens(r):
    g = fav_grid(r, "cells"); ed = emp_norm.get((r.fb, r.tb), glob_norm)
    return {s: 0.5 * g.get(s, 0) + 0.5 * ed.get(s, 0) for s in set(g) | set(ed)}
DISTS = [D_cells, D_dc, D_pois, D_emp, D_blend, D_ens]

# poids = Top-3 accuracy TRAIN de chaque distribution
def top3_acc_train(fn):
    hit = 0
    for r in tr.itertuples():
        top3 = [s for s, _ in sorted(fn(r).items(), key=lambda x: -x[1])[:3]]
        hit += r.fs in top3
    return hit / len(tr)
W = [top3_acc_train(fn) for fn in DISTS]

# ---- fusions ----
def fuse_mean(r):
    agg = defaultdict(float)
    for fn in DISTS:
        for s, p in norm(fn(r)).items(): agg[s] += p / len(DISTS)
    return agg
def fuse_wvote(r):
    agg = defaultdict(float)
    for w, fn in zip(W, DISTS):
        for s, p in norm(fn(r)).items(): agg[s] += w * p
    return agg
def fuse_borda(r):
    agg = defaultdict(float)
    for fn in DISTS:
        ranked = [s for s, _ in sorted(fn(r).items(), key=lambda x: -x[1])]
        for rank, s in enumerate(ranked[:15]): agg[s] += (15 - rank)
    return agg

CAND = {"meilleur seul (ensemble)": D_ens, "VOTE moyenne": fuse_mean,
        "VOTE pondéré": fuse_wvote, "VOTE Borda": fuse_borda}
res = {nm: {1: 0, 2: 0, 3: 0} for nm in CAND}
for r in te.itertuples():
    for nm, fn in CAND.items():
        ranked = [s for s, _ in sorted(fn(r).items(), key=lambda x: -x[1])]
        pos = ranked.index(r.fs) + 1 if r.fs in ranked else 999
        for k in (1, 2, 3):
            if pos <= k: res[nm][k] += 1
N = len(te)
print("="*60); print("FUSION DES ALGOS — couverture (OOS)"); print("="*60)
print(f"{'méthode':<26}{'Top-1':>8}{'Top-2':>8}{'Top-3':>8}")
for nm in CAND:
    print(f"{nm:<26}{res[nm][1]/N*100:>7.1f}%{res[nm][2]/N*100:>7.1f}%{res[nm][3]/N*100:>7.1f}%")
best = max(CAND, key=lambda nm: res[nm][3])
print(f"\n  meilleur Top-3 : {best} = {res[best][3]/N*100:.2f}%")
print(f"  (plafond connu ~31% — voir si la fusion gratte qqch ou plafonne)")
