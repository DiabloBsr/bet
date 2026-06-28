"""LE ZOO DES PRÉDICTEURS DE SCORE — du plus débile au plus sérieux.
On lance ~20 prédicteurs (constants, absurdes, numérologiques, ET principiels)
sur tous les résultats 8035, split chrono 70/30, et on classe par accuracy Top-1
du score exact (+ direction 1X2). Objectif : voir quelles 'bêtises' tiennent
vraiment la route face au plafond ~13%.
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
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
n = len(d); cut = int(n * 0.7); tr, te = d.iloc[:cut], d.iloc[cut:]
print(f"n={n} | test={len(te)}\n")

# cache inversion par triplet de cotes arrondi (vitesse)
cache_modal_sim, cache_modal_poi, cache_lam = {}, {}, {}
def lam(oh, od, oa):
    k = (round(oh, 2), round(od, 2), round(oa, 2))
    if k not in cache_lam:
        cache_lam[k] = exact_invert_1x2(oh, od, oa)
    return cache_lam[k]
def modal_sim(oh, od, oa):
    k = (round(oh, 2), round(od, 2), round(oa, 2))
    if k not in cache_modal_sim:
        lh, la = lam(oh, od, oa); cache_modal_sim[k] = grid_modal_score(apply_sim_deviations(lh, la, "cells"))
    return cache_modal_sim[k]
def modal_poi(oh, od, oa):
    k = (round(oh, 2), round(od, 2), round(oa, 2))
    if k not in cache_modal_poi:
        lh, la = lam(oh, od, oa); cache_modal_poi[k] = grid_modal_score(_fast_grid(lh, la, 0.0))
    return cache_modal_poi[k]

# --- LE ZOO : chaque fonction prend une ligne r et renvoie 'h-a' ---
def fav(r, hs, as_):  # score orienté favori (home si fav_home)
    return f"{hs}-{as_}" if r.fav_home else f"{as_}-{hs}"

ZOO = {
    # --- DÉBILE / CONSTANT ---
    "💀 toujours 1-1": lambda r: "1-1",
    "💀 toujours 0-0": lambda r: "0-0",
    "💀 toujours 1-0": lambda r: "1-0",
    "💀 toujours 2-1": lambda r: "2-1",
    "💀 toujours 2-2 (rare)": lambda r: "2-2",
    # --- ORIENTÉ FAVORI (semi-malin) ---
    "🃏 favori 1-0": lambda r: fav(r, 1, 0),
    "🃏 favori 2-1": lambda r: fav(r, 2, 1),
    "🃏 favori 2-0": lambda r: fav(r, 2, 0),
    "🃏 favori 3-0": lambda r: fav(r, 3, 0),
    # --- ABSURDE / NUMÉROLOGIE ---
    "🎲 anti-favori 0-1": lambda r: fav(r, 0, 1),  # contrarian
    "🎲 chiffres des cotes": lambda r: f"{int(r.oh)%5}-{int(r.oa)%5}",
    "🎲 décimale des cotes": lambda r: f"{int(round(r.oh*10))%4}-{int(round(r.oa*10))%4}",
    "🎲 longueur des noms %4": lambda r: f"{len(str(r.team_a))%4}-{len(str(r.team_b))%4}",
    "🎲 somme cotes mod": lambda r: f"{int(r.oh+r.od)%4}-{int(r.oa+r.od)%4}",
    # --- λ ARRONDI (mi-malin) ---
    "📐 λ arrondi": lambda r: (lambda lh, la: f"{int(round(lh))}-{int(round(la))}")(*lam(r.oh, r.od, r.oa)),
    "📐 λ plancher": lambda r: (lambda lh, la: f"{int(lh)}-{int(la)}")(*lam(r.oh, r.od, r.oa)),
    # --- λtot -> nul échelonné ---
    "📐 nul échelonné": lambda r: (lambda s: "0-0" if s < 2.0 else ("1-1" if s < 3.2 else "2-2"))(sum(lam(r.oh, r.od, r.oa))),
    # --- PRINCIPIEL (sérieux) ---
    "🧠 Poisson pur modal": lambda r: modal_poi(r.oh, r.od, r.oa),
    "🧠 moteur (sim modal)": lambda r: modal_sim(r.oh, r.od, r.oa),
}

# --- prédicteur data-driven : score modal par bande de cote favori (fit TRAIN) ---
tr2 = tr.copy(); tr2["favc"] = tr2[["oh", "oa"]].min(axis=1)
tr2["band"] = pd.cut(tr2.favc, [1, 1.3, 1.5, 1.8, 2.2, 99]).astype(str)
band_modal = tr2.groupby("band").real.agg(lambda s: s.value_counts().idxmax()).to_dict()
def databand(r):
    favc = min(r.oh, r.oa)
    b = str(pd.cut([favc], [1, 1.3, 1.5, 1.8, 2.2, 99])[0])
    sc = band_modal.get(b, "1-1")
    # réorienter selon le favori (le modal TRAIN est home-fav par construction des cotes ? non) -> garder tel quel
    return sc
ZOO["🧠 modal par bande (data)"] = databand

# --- ÉVAL sur TEST ---
rows = []
for name, fn in ZOO.items():
    preds = te.apply(fn, axis=1)
    acc = (preds.values == te.real.values).mean()
    # direction 1X2
    def winner(s):
        a, b = map(int, s.split("-")); return "H" if a > b else ("A" if a < b else "D")
    pred_w = preds.apply(winner).values
    real_w = te.real.apply(winner).values
    accw = (pred_w == real_w).mean()
    rows.append((name, acc, accw))
res = pd.DataFrame(rows, columns=["predicteur", "score%", "1x2%"]).sort_values("score%", ascending=False)

print("="*64)
print("CLASSEMENT — accuracy Top-1 score exact (test OOS)")
print("="*64)
print(f"{'predicteur':<28}{'score exact':>12}{'1X2 dir.':>10}")
print("-"*50)
for r in res.itertuples():
    print(f"{r.predicteur:<28}{r._2*100:>11.2f}%{r._3*100:>9.0f}%")
print(f"\nplafond connu ~13% Top-1 | base 'toujours 1-1' = repère du débile-qui-marche")

# ensemble : vote majoritaire des 'sérieux' + meilleurs débiles
serious = ["🧠 moteur (sim modal)", "🧠 Poisson pur modal", "🃏 favori 2-1", "💀 toujours 1-1"]
def vote(r):
    from collections import Counter
    cs = Counter(ZOO[s](r) for s in serious)
    return cs.most_common(1)[0][0]
acc_vote = (te.apply(vote, axis=1).values == te.real.values).mean()
print(f"\n🗳️  ENSEMBLE vote(moteur+Poisson+fav2-1+1-1) : {acc_vote*100:.2f}%")
print("\n-> si le meilleur 'débile' (ex: favori 1-0 ou 1-1) talonne le moteur,")
print("   c'est que le plafond RNG aplatit tout : la bêtise simple ≈ l'intelligence.")
