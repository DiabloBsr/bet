"""ZOO v3 — on POUSSE : table 3D (favori × λtot × BTTS), table 3D (favori × λtot ×
dominance), et MÉGA-VOTE pondéré. On compare au FINAL 2D (11.92%) pour voir si on
perce le plafond ou si tout converge vers ~12%. Min-n par cellule + fallback 2D/global
(anti-surajustement). Split chrono 70/30.
"""
from __future__ import annotations
import sys, math
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
d["fav_home"] = d.oh < d.oa
d["favc"] = d[["oh", "oa"]].min(axis=1)
d["pbtts"] = (1 - np.exp(-d["lh"])) * (1 - np.exp(-d["la"]))
d["ldiff"] = (d["lh"] - d["la"]).abs()
d["real"] = d.sa.astype(int).astype(str) + "-" + d.sb.astype(int).astype(str)
fg = np.where(d.fav_home, d.sa, d.sb); dg = np.where(d.fav_home, d.sb, d.sa)
d["fav_score"] = pd.Series(fg.astype(int).astype(str), index=d.index) + "-" + pd.Series(dg.astype(int).astype(str), index=d.index)
n = len(d); cut = int(n * 0.7); tr, te = d.iloc[:cut].copy(), d.iloc[cut:].copy()
print(f"n={n} | test={len(te)}\n")

FB = [1, 1.3, 1.5, 1.8, 2.2, 1e9]; TB = [0, 2.1, 2.5, 2.9, 3.3, 1e9]
BB = [0, 0.45, 0.58, 1.01]; DB = [0, 0.4, 0.9, 1e9]
def band(v, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]: return i
    return len(edges) - 2
for f in (tr, te):
    f["fb"] = f.favc.map(lambda v: band(v, FB)); f["tb"] = f["lt"].map(lambda v: band(v, TB))
    f["bb"] = f.pbtts.map(lambda v: band(v, BB)); f["db"] = f.ldiff.map(lambda v: band(v, DB))

def modal_of(s): return s.value_counts().idxmax()
glob = modal_of(tr.fav_score)
tab2d = tr.groupby(["fb", "tb"]).fav_score.agg(modal_of).to_dict()

def make_3d(extra):
    g = tr.groupby(["fb", "tb", extra]).fav_score
    counts = tr.groupby(["fb", "tb", extra]).size().to_dict()
    modal = g.agg(modal_of).to_dict()
    return modal, counts
tab_btts, cnt_btts = make_3d("bb")
tab_diff, cnt_diff = make_3d("db")

MIN3D = 80
def orient(r, fs): fg, dg = fs.split("-"); return f"{fg}-{dg}" if r.fav_home else f"{dg}-{fg}"
def pred_2d(r): return orient(r, tab2d.get((r.fb, r.tb), glob))
def pred_3d(r, modal, cnt, ex):
    key = (r.fb, r.tb, r[ex])
    if cnt.get(key, 0) >= MIN3D and key in modal:
        return orient(r, modal[key])
    return orient(r, tab2d.get((r.fb, r.tb), glob))   # fallback 2D
def pred_engine(r): return grid_modal_score(apply_sim_deviations(r["lh"], r["la"], "cells"))
def pred_lam(r): return f"{int(round(r['lh']))}-{int(round(r['la']))}"
def pred_fav21(r): return orient(r, "2-1")
def pred_11(r): return "1-1"

# ---- MÉGA-VOTE pondéré : poids = accuracy TRAIN de chaque base ----
bases = {"2D": pred_2d, "engine": pred_engine, "lam": pred_lam, "fav21": pred_fav21, "11": pred_11,
         "btts3d": lambda r: pred_3d(r, tab_btts, cnt_btts, "bb"),
         "diff3d": lambda r: pred_3d(r, tab_diff, cnt_diff, "db")}
wts = {}
for nm, fn in bases.items():
    wts[nm] = (tr.apply(fn, axis=1).values == tr.real.values).mean()
def megavote(r):
    sc = defaultdict(float)
    for nm, fn in bases.items():
        sc[fn(r)] += wts[nm]
    return max(sc.items(), key=lambda kv: kv[1])[0]

# ---- éval OOS ----
preds = {
    "FINAL 2D (fav×λtot)": pred_2d,
    "3D + BTTS": lambda r: pred_3d(r, tab_btts, cnt_btts, "bb"),
    "3D + dominance": lambda r: pred_3d(r, tab_diff, cnt_diff, "db"),
    "méga-vote pondéré": megavote,
    "moteur sim (réf)": pred_engine,
}
def winner(s):
    a, b = map(int, s.split("-")); return "H" if a > b else ("A" if a < b else "D")
rw = te.real.apply(winner).values
print("="*60); print("PUSH — comparaison OOS (score exact)"); print("="*60)
print(f"{'modèle':<26}{'score':>9}{'1X2':>7}{'Δ vs 2D':>10}")
print("-"*52)
base2d = (te.apply(pred_2d, axis=1).values == te.real.values).mean()
for nm, fn in preds.items():
    p = te.apply(fn, axis=1)
    acc = (p.values == te.real.values).mean()
    accw = (p.apply(winner).values == rw).mean()
    print(f"{nm:<26}{acc*100:>8.2f}%{accw*100:>6.0f}%{(acc-base2d)*100:>+9.2f}")
se = math.sqrt(base2d*(1-base2d)/len(te))*100
print(f"\nmarge d'erreur (±1σ) sur {len(te)} matchs : ±{se:.2f} pt")
print("poids méga-vote (acc train) :", {k: round(v*100, 1) for k, v in wts.items()})

# ---- top-3 (couverture) du 3D-BTTS : si on jouait 3 scores ----
def top3_3d(r):
    key = (r.fb, r.tb, r.bb)
    sub = tr[(tr.fb == r.fb) & (tr.tb == r.tb) & (tr.bb == r.bb)]
    if len(sub) < MIN3D: sub = tr[(tr.fb == r.fb) & (tr.tb == r.tb)]
    top = [orient(r, s) for s, _ in Counter(sub.fav_score).most_common(3)]
    return top
hit3 = te.apply(lambda r: r.real in top3_3d(r), axis=1).mean()
print(f"\nCouverture Top-3 (3D-BTTS) : {hit3*100:.1f}%  (plafond Top-3 connu ~31%)")
