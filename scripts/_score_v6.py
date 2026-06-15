"""MOTEUR SCORE v6 (NOUVEAU — ne touche pas V5/V2). Hypothèse issue de toute la
recherche : la meilleure prédiction de score = l'échelle de cotes score OFFERTE
devigée (le book price la vraie distribution déviée du RNG), affinée par la
grille sim (RNG-aware) + boost book-modal + chaînage directionnel.
Backtest honnête : Top1/Top3 OOS de chaque composant + ensembles, sur split chrono.
Usage: ./.venv/Scripts/python.exe scripts/_score_v6.py"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scraper.market_inversion import invert_markets, apply_sim_deviations, grid_modal_score, grid_top_k_scores

CSV = Path(__file__).resolve().parents[1]/"exports"/"combokeys_features.csv"
SCORES = ['1-1','2-1','1-2','1-0','0-1','2-0','0-2','0-0','2-2','3-0','0-3','3-1','1-3','3-2','2-3']
df = pd.read_csv(CSV).sort_values("expected_start").reset_index(drop=True)
cut = int(len(df)*0.70); te = df.iloc[cut:].reset_index(drop=True)
print(f"backtest sur TEST (OOS) n={len(te)} (train={cut})\n")

def book_dist(row):
    """distribution devigée de l'échelle de cotes score offerte (off_s_*)."""
    d = {}
    for s in SCORES:
        c = row.get(f"off_s_{s}")
        if c and c > 1: d[s] = 1.0/c
    tot = sum(d.values())
    return {k: v/tot for k, v in d.items()} if tot > 0 else {}

def sim_dist(row):
    inv = invert_markets(float(row["oh"]), float(row["od"]), float(row["oa"]), row.get("extra_markets"))
    g = apply_sim_deviations(inv.lam_h, inv.lam_a, "cells")
    d = {}
    for h in range(g.shape[0]):
        for a in range(g.shape[1]):
            if g[h,a] > 0: d[f"{h}-{a}"] = float(g[h,a])
    return d

def topk(d, k): return [s for s,_ in sorted(d.items(), key=lambda x:-x[1])[:k]]
def acc(rows, distfn, k):
    h=0; n=0
    for r in rows.itertuples():
        d = distfn(r._asdict() if hasattr(r,'_asdict') else r)
        if not d: continue
        n+=1
        if r.exact_score in topk(d,k): h+=1
    return h, n

# pré-calcul des distributions par match test (sim coûteux -> une fois)
rows = te.to_dict("records")
sims=[]; books=[]; reals=[]
for row in rows:
    if not (row["oh"] and row["od"] and row["oa"]) or row["oh"]<=1: continue
    b = book_dist(row)
    try: s = sim_dist(row)
    except Exception:
        s = {}
    if not b and not s: continue
    sims.append(s); books.append(b); reals.append(row["exact_score"])

def ens(b,s,wb,ws):
    keys=set(b)|set(s); return {k: wb*b.get(k,0)+ws*s.get(k,0) for k in keys}
def evaldist(dists, reals, k):
    h=sum(1 for d,r in zip(dists,reals) if d and r in [x for x,_ in sorted(d.items(),key=lambda y:-y[1])[:k]])
    n=sum(1 for d in dists if d); return h/n*100 if n else 0

print(f"{'modèle':<28}{'Top1':>7}{'Top3':>7}{'n':>7}")
print("-"*49)
# composants seuls
print(f"{'SIM (grille RNG-aware)':<28}{evaldist(sims,reals,1):>6.1f}%{evaldist(sims,reals,3):>6.1f}%{len(sims):>7}")
print(f"{'BOOK (cotes offertes devig)':<28}{evaldist(books,reals,1):>6.1f}%{evaldist(books,reals,3):>6.1f}%{sum(1 for b in books if b):>7}")
# ensembles book+sim
best=None
for wb in [0.3,0.5,0.6,0.7,0.8,1.0]:
    ws=1-wb
    e=[ens(b,s,wb,ws) for b,s in zip(books,sims)]
    t1=evaldist(e,reals,1); t3=evaldist(e,reals,3)
    lbl='V6 ens book%.1f+sim%.1f'%(wb,ws)
    print(f"{lbl:<28}{t1:>6.1f}%{t3:>6.1f}%{len(e):>7}")
    if best is None or t3>best[2]: best=(wb,t1,t3)
print(f"\n-> meilleur ensemble : book={best[0]:.1f} / sim={1-best[0]:.1f}  (Top1 {best[1]:.1f}% / Top3 {best[2]:.1f}%)")
print('Plafond empirique connu : Top1 ~12-15% / Top3 ~30-36%.')
