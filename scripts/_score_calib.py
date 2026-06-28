"""Pourquoi FINAL sort trop de 2-1 ? Et peut-on corriger SANS perdre l'accuracy ?
Mesure : distribution des scores ÉMIS par FINAL vs distribution RÉELLE.
Test d'une version 'FINAL-fin' : dans chaque cellule, on garde les top-3 scores
empiriques, puis on tranche entre eux avec le λ continu (grille sim) -> plus de
variété, même calibration. + une version 'calibrée' qui matche la distribution réelle.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.market_inversion import exact_invert_1x2, apply_sim_deviations, grid_modal_score

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
d["real"] = d.sa.astype(int).astype(str) + "-" + d.sb.astype(int).astype(str)
fg = np.where(d.fav_home, d.sa, d.sb); dg = np.where(d.fav_home, d.sb, d.sa)
d["fav_score"] = pd.Series(fg.astype(int).astype(str), index=d.index) + "-" + pd.Series(dg.astype(int).astype(str), index=d.index)
n = len(d); cut = int(n * 0.7); tr, te = d.iloc[:cut].copy(), d.iloc[cut:].copy()
print(f"n={n} | test={len(te)}\n")

FB = [1, 1.3, 1.5, 1.8, 2.2, 1e9]; TB = [0, 2.1, 2.5, 2.9, 3.3, 1e9]
def band(v, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]: return i
    return len(edges) - 2
for f in (tr, te):
    f["fb"] = f.favc.map(lambda v: band(v, FB)); f["tb"] = f["lt"].map(lambda v: band(v, TB))
cell_dist = {k: Counter(v) for k, v in tr.groupby(["fb", "tb"]).fav_score.agg(list).to_dict().items()}
glob = tr.fav_score.value_counts().idxmax()
def orient(r, fs): a, b = fs.split("-"); return f"{a}-{b}" if r.fav_home else f"{b}-{a}"

def pred_modal(r):
    c = cell_dist.get((r.fb, r.tb)); return orient(r, c.most_common(1)[0][0] if c else glob)

def sim_prob(r, fs):  # proba grille-sim du score fav-orienté fs
    g = apply_sim_deviations(r["lh"], r["la"], "cells")
    a, b = map(int, fs.split("-")); h, aw = (a, b) if r.fav_home else (b, a)
    return g[h, aw] if h < g.shape[0] and aw < g.shape[0] else 0.0

def pred_fin(r):  # FINAL-fin : top-4 candidats de la cellule, tranchés par λ continu
    c = cell_dist.get((r.fb, r.tb))
    if not c: return orient(r, glob)
    cands = [s for s, _ in c.most_common(4)]
    best = max(cands, key=lambda s: sim_prob(r, s))
    return orient(r, best)

def pred_sim(r): return grid_modal_score(apply_sim_deviations(r["lh"], r["la"], "cells"))

# éval
def acc(fn): return (te.apply(fn, axis=1).values == te.real.values).mean()
def emis(fn):
    p = te.apply(fn, axis=1); return Counter(p.values)
real_dist = Counter(te.real.values)
N = len(te)
print("="*70)
print("DISTRIBUTION ÉMISE vs RÉELLE (top scores, % sur le test)")
print("="*70)
fin_emit = emis(pred_modal); finfin_emit = emis(pred_fin); sim_emit = emis(pred_sim)
keys = [k for k, _ in real_dist.most_common(10)]
print(f"{'score':<7}{'RÉEL':>8}{'FINAL':>8}{'FINAL-fin':>11}{'sim':>8}")
for k in keys:
    print(f"{k:<7}{real_dist[k]/N*100:>7.1f}%{fin_emit.get(k,0)/N*100:>7.1f}%{finfin_emit.get(k,0)/N*100:>10.1f}%{sim_emit.get(k,0)/N*100:>7.1f}%")
print(f"\n  scores DISTINCTS émis : FINAL={len(fin_emit)}  FINAL-fin={len(finfin_emit)}  sim={len(sim_emit)}")
print("\n" + "="*70)
print("ACCURACY (le prix de la diversité)")
print("="*70)
print(f"  FINAL (modal)   : {acc(pred_modal)*100:.2f}%   2-1 émis {fin_emit.get('2-1',0)/N*100:.0f}%")
print(f"  FINAL-fin       : {acc(pred_fin)*100:.2f}%   2-1 émis {finfin_emit.get('2-1',0)/N*100:.0f}%")
print(f"  moteur sim      : {acc(pred_sim)*100:.2f}%   2-1 émis {sim_emit.get('2-1',0)/N*100:.0f}%")
print(f"\n  réel 2-1 = {real_dist.get('2-1',0)/N*100:.1f}%")
print("  -> on cherche : 2-1 émis proche du réel, sans chute d'accuracy.")
