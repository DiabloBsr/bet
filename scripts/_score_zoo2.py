"""ZOO v2 — on pousse, on garde les meilleurs, on analyse leurs FORCES,
et on injecte tout dans un PRÉDICTEUR FINAL (lookup 2D orienté-favori, data-driven).
Split chrono 70/30. But : se rapprocher au max du plafond ~13% avec un modèle
simple et interprétable qui combine : scores bas + orientation favori + échelle λtot.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from collections import Counter
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.market_inversion import exact_invert_1x2, apply_sim_deviations, _fast_grid, grid_modal_score

e = create_engine(load_settings().db_url)
d = pd.read_sql("""SELECT e.team_a,e.team_b,e.expected_start,o.odds_home oh,o.odds_draw od,o.odds_away oa,
  r.score_a sa,r.score_b sb FROM events e
  JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
  JOIN results r ON r.event_id=e.id WHERE r.score_a IS NOT NULL AND e.competition='InstantLeague-8035'""", e)
d = d[(d.oh > 1) & (d.od > 1) & (d.oa > 1)].copy()
d["es"] = pd.to_datetime(d.expected_start, utc=True, errors="coerce")
d = d.dropna(subset=["es"]).sort_values("es").reset_index(drop=True)
d["real"] = d.sa.astype(int).astype(str) + "-" + d.sb.astype(int).astype(str)
d["fav_home"] = d.oh < d.oa
d["favc"] = d[["oh", "oa"]].min(axis=1)

# lambda + totaux (cache)
cl = {}
def lam(oh, od, oa):
    k = (round(oh, 2), round(od, 2), round(oa, 2))
    if k not in cl: cl[k] = exact_invert_1x2(oh, od, oa)
    return cl[k]
print("inversion des cotes (cache)...", flush=True)
LAM = d.apply(lambda r: lam(r.oh, r.od, r.oa), axis=1)
d["lh"] = [x[0] for x in LAM]; d["la"] = [x[1] for x in LAM]; d["lt"] = d.lh + d.la
n = len(d); cut = int(n * 0.7); tr, te = d.iloc[:cut].copy(), d.iloc[cut:].copy()
print(f"n={n} | test={len(te)}\n")

def fav(r, hs, as_): return f"{hs}-{as_}" if r.fav_home else f"{as_}-{hs}"

# ---- ZOO élargi ----
ZOO = {
    "💀 1-1": lambda r: "1-1",
    "🃏 favori 2-1": lambda r: fav(r, 2, 1),
    "🃏 favori 1-0": lambda r: fav(r, 1, 0),
    "📐 λ arrondi": lambda r: f"{int(round(r.lh))}-{int(round(r.la))}",
    "🧠 moteur sim": lambda r: grid_modal_score(apply_sim_deviations(r.lh, r.la, "cells")),
    # conditionnels λtot (semi-malin orienté favori)
    "🔀 fav: tot<2.4→1-0, <3.2→2-1, +→2-1": lambda r: fav(r, 1, 0) if r["lt"] < 2.4 else fav(r, 2, 1),
    "🔀 fav: <2.0→1-0,<2.8→2-1,<3.6→2-1,+→3-1": lambda r: (fav(r, 1, 0) if r["lt"] < 2.0 else fav(r, 2, 1) if r["lt"] < 3.6 else fav(r, 3, 1)),
    "🔀 1-0/1-1/2-1 par écart": lambda r: (fav(r, 1, 0) if abs(r.lh - r.la) > 0.7 else "1-1"),
    "🔀 nul si serré sinon fav2-1": lambda r: ("1-1" if abs(r.lh - r.la) < 0.4 else fav(r, 2, 1)),
    "📐 λ arrondi orienté fav": lambda r: fav(r, int(round(max(r.lh, r.la))), int(round(min(r.lh, r.la)))),
}

# ---- prédicteur DATA-DRIVEN 1D (bande favori) déjà testé ----
def fit_modal(frame, keycol):
    return frame.groupby(keycol).fav_score.agg(lambda s: s.value_counts().idxmax()).to_dict()

# fav-oriented realized score (fg-dg)
for f in (tr, te):
    fg = np.where(f.fav_home, f.sa, f.sb); dg = np.where(f.fav_home, f.sb, f.sa)
    f["fav_score"] = pd.Series(fg.astype(int).astype(str), index=f.index) + "-" + pd.Series(dg.astype(int).astype(str), index=f.index)

# ---- PRÉDICTEUR FINAL : lookup 2D (bande favori × bande λtot), orienté favori ----
FB = [1, 1.3, 1.5, 1.8, 2.2, 99]; TB = [0, 2.1, 2.5, 2.9, 3.3, 99]
tr["fb"] = pd.cut(tr.favc, FB).astype(str); tr["tb"] = pd.cut(tr["lt"], TB).astype(str)
te["fb"] = pd.cut(te.favc, FB).astype(str); te["tb"] = pd.cut(te["lt"], TB).astype(str)
cell_modal = tr.groupby(["fb", "tb"]).fav_score.agg(lambda s: s.value_counts().idxmax()).to_dict()
glob_modal = tr.fav_score.value_counts().idxmax()
def FINAL(r):
    fs = cell_modal.get((r.fb, r.tb), glob_modal)
    fg, dg = fs.split("-")
    return f"{fg}-{dg}" if r.fav_home else f"{dg}-{fg}"
ZOO["🏆 FINAL (lookup 2D fav×λtot)"] = FINAL

# ---- éval ----
def winner(s):
    a, b = map(int, s.split("-")); return "H" if a > b else ("A" if a < b else "D")
real_w = te.real.apply(winner).values
rows = []
for name, fn in ZOO.items():
    p = te.apply(fn, axis=1)
    acc = (p.values == te.real.values).mean()
    accw = (p.apply(winner).values == real_w).mean()
    rows.append((name, acc, accw))
res = pd.DataFrame(rows, columns=["pred", "s", "w"]).sort_values("s", ascending=False)
print("="*66)
print("CLASSEMENT élargi (score exact OOS)")
print("="*66)
print(f"{'predicteur':<40}{'score':>9}{'1X2':>7}")
print("-"*56)
for r in res.itertuples():
    star = " 👑" if "FINAL" in r.pred else ""
    print(f"{r.pred:<40}{r.s*100:>8.2f}%{r.w*100:>6.0f}%{star}")

print("\n" + "="*66)
print("CE QUI REND LES MEILLEURS FORTS — la table FINALE (orientée favori)")
print("="*66)
print("  bande favori × bande λtot  ->  score modal (fav-dog) [n train]")
piv = tr.groupby(["fb", "tb"]).fav_score.agg([("modal", lambda s: s.value_counts().idxmax()), ("n", "size")])
for (fb, tb), row in piv.iterrows():
    if row["n"] >= 50:
        print(f"    fav{fb:<12} tot{tb:<12} -> {row['modal']:<5} (n={int(row['n'])})")
print("\n  Lecture : λtot bas -> 0-0/1-0 ; λtot haut -> 2-1/3-1 ; toujours orienté favori.")
print("  => les 3 forces (scores bas + favori + échelle λtot) sont injectées dans FINAL.")
