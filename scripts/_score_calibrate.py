"""Couche de CALIBRATION : corrige la grille pour que la distribution prédite colle
à la réalité par profil (favband × totband). correction[score] = freq_réel/freq_modèle
(lissée, bornée). On applique grid *= correction -> renorm -> argmax.
Validé OOS : l'accuracy ne doit pas baisser, la distribution doit se rapprocher.
Si OK -> on sauvegarde les corrections dans la table FINAL.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.market_inversion import exact_invert_1x2, apply_sim_deviations
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
d["real"] = d.sa.astype(int).astype(str) + "-" + d.sb.astype(int).astype(str)
fg = np.where(d.fav_home, d.sa, d.sb); dg = np.where(d.fav_home, d.sb, d.sa)
d["fs"] = pd.Series(fg.astype(int).astype(str), index=d.index) + "-" + pd.Series(dg.astype(int).astype(str), index=d.index)
d["fb"] = d.favc.map(lambda v: _band(v, FAV_EDGES)); d["tb"] = d["lt"].map(lambda v: _band(v, TOT_EDGES))
n = len(d); cut = int(n * 0.7); tr, te = d.iloc[:cut].copy(), d.iloc[cut:].copy()
print(f"n={n} | test={len(te)}\n")

MAXG = 8
def fav_grid(r):
    g = apply_sim_deviations(r.lh, r.la, "cells")
    return g if r.fav_home else g.T

# ---- fit corrections sur TRAIN : par bucket, freq réel vs freq modèle ----
emp = defaultdict(Counter); mod = defaultdict(lambda: np.zeros((MAXG, MAXG))); cnt = Counter()
for r in tr.itertuples():
    key = (r.fb, r.tb)
    emp[key][r.fs] += 1; mod[key] += fav_grid(r); cnt[key] += 1
SMOOTH = 0.01; LO, HI = 0.55, 1.8; MINN = 200
corr = {}   # corr[key][ "i-j" ] = facteur
for key, c in cnt.items():
    if c < MINN: continue
    mg = mod[key] / c
    cc = {}
    for i in range(MAXG):
        for j in range(MAXG):
            er = emp[key].get(f"{i}-{j}", 0) / c
            mr = mg[i, j]
            if mr > 0.005 or er > 0.005:   # on ne corrige que les scores non-négligeables
                f = (er + SMOOTH) / (mr + SMOOTH)
                cc[f"{i}-{j}"] = float(min(max(f, LO), HI))
    corr[key] = cc

# ---- prédicteurs ----
def pred_raw(r):
    g = fav_grid(r); ij = np.unravel_index(g.argmax(), g.shape)
    fs = f"{ij[0]}-{ij[1]}"; a, b = ij
    return f"{a}-{b}" if r.fav_home else f"{b}-{a}"
def pred_cal(r):
    g = fav_grid(r).copy()
    cc = corr.get((r.fb, r.tb))
    if cc:
        for s, f in cc.items():
            i, j = map(int, s.split("-"))
            g[i, j] *= f
    ij = np.unravel_index(g.argmax(), g.shape); a, b = ij
    return f"{a}-{b}" if r.fav_home else f"{b}-{a}"

acc_raw = (te.apply(pred_raw, axis=1).values == te.real.values).mean()
acc_cal = (te.apply(pred_cal, axis=1).values == te.real.values).mean()
print("="*56); print("ACCURACY OOS"); print("="*56)
print(f"  grille brute (FINAL actuel)   : {acc_raw*100:.2f}%")
print(f"  grille CALIBRÉE               : {acc_cal*100:.2f}%   ({(acc_cal-acc_raw)*100:+.2f} pt)")

# ---- distribution émise : avant/après vs réel ----
emit_raw = Counter(te.apply(pred_raw, axis=1).values)
emit_cal = Counter(te.apply(pred_cal, axis=1).values)
realc = Counter(te.real.values); N = len(te)
print("\n" + "="*56); print("DISTRIBUTION ÉMISE vs RÉELLE (top scores)"); print("="*56)
print(f"{'score':<7}{'RÉEL':>8}{'brut':>8}{'calibré':>9}")
for s, _ in realc.most_common(10):
    print(f"{s:<7}{realc[s]/N*100:>7.1f}%{emit_raw.get(s,0)/N*100:>7.1f}%{emit_cal.get(s,0)/N*100:>8.1f}%")
print(f"\n  scores distincts émis : brut={len(emit_raw)}  calibré={len(emit_cal)}")
# écart-type émis-vs-réel (qualité de répartition)
def sigma(emit):
    return np.std([emit.get(s, 0)/N*100 - realc[s]/N*100 for s in realc])
print(f"  σ(émis - réel) : brut={sigma(emit_raw):.2f}  calibré={sigma(emit_cal):.2f}  (plus bas = mieux réparti)")

# ---- sauvegarde si gagnant ----
if acc_cal >= acc_raw - 0.001:
    out = Path(__file__).resolve().parents[1] / "exports" / "score_corrections.json"
    out.write_text(json.dumps({f"{k[0]}-{k[1]}": v for k, v in corr.items()}, indent=1), encoding="utf-8")
    print(f"\n✅ corrections sauvegardées : {out.name} ({len(corr)} buckets)")
else:
    print("\n⚠️ la calibration baisse l'accuracy -> NON sauvegardée")
