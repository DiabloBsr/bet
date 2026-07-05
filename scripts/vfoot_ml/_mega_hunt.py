"""MINEUR D'HYPOTHÈSES MASSIF — génère et teste des dizaines de milliers d'idées
folles automatiquement, avec discipline anti-mirage (OOS chrono + BH-FDR global).

Génère : ~55 conditions de base (résultats décalés lag1-5, séries de tous types,
buts marqués/encaissés décalés, marges, venue, repos) -> combos 1/2/3 conditions
(AND) -> × 7 cibles (V/N/D, over2.5/3.5, under2.5, BTTS) × 4 strates de cote.
Chaque hypothèse : résidu = taux réel - proba IMPLICITE dévigée ; retenue si
TRAIN n>=150 & |résidu|>=0.02 ; testée OOS (z-test normal, n grand) ; BH-FDR sur
TOUTES. Rapporte : nb testé, faux positifs attendus, survivants (attendu : 0).
"""
from __future__ import annotations
import json, sys, time
from itertools import combinations
from math import erf, sqrt as _msqrt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

LG = "InstantLeague-8035"
t0 = time.time()
eng = create_engine(load_settings().db_url)
df = pd.read_sql(text(f"""
    SELECT e.expected_start ts, e.team_a, e.team_b, o.odds_home oh, o.odds_draw od,
           o.odds_away oa, o.extra_markets xm, r.score_a sa, r.score_b sb
    FROM events e JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id
    WHERE r.score_a IS NOT NULL AND e.competition='{LG}' AND o.odds_home>1 AND o.odds_draw>1 AND o.odds_away>1
    ORDER BY e.expected_start"""), eng).drop_duplicates(["ts", "team_a", "team_b"]).reset_index(drop=True)
print(f"{len(df)} matchs", flush=True)

inv = 1/df.oh + 1/df.od + 1/df.oa
df["imp_h"], df["imp_d"], df["imp_a"] = (1/df.oh)/inv, (1/df.od)/inv, (1/df.oa)/inv
df["total"] = df.sa + df.sb

def gm(xm, pref):
    for k, v in (xm or {}).items():
        if k.replace("\x82", "é").replace("\xe9", "é").startswith(pref):
            return v
    return None
def devig(sels, keys):
    v = {k: 1/sels[k] for k in keys if isinstance((sels or {}).get(k), (int, float)) and 1 < sels[k] < 99.99}
    s = sum(v.values()); return {k: v[k]/s for k in v} if s and len(v) == len(keys) else None
io25, io35, ibtts = [], [], []
for raw in df.xm:
    try: xm = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception: xm = {}
    tt = devig(gm(xm, "Total de buts"), [str(k) for k in range(7)])
    io25.append(sum(tt[str(k)] for k in range(3, 7)) if tt else np.nan)
    io35.append(sum(tt[str(k)] for k in range(4, 7)) if tt else np.nan)
    gg = devig(gm(xm, "G/NG"), ["Oui", "Non"]); ibtts.append(gg["Oui"] if gg else np.nan)
df["io25"], df["io35"], df["ibtts"] = io25, io35, ibtts

# perspective équipe
recs = []
for side in ("H", "A"):
    recs.append(pd.DataFrame({
        "ts": df.ts, "team": df.team_a if side == "H" else df.team_b, "venue": 1 if side == "H" else 0,
        "res": np.where(df.sa == df.sb, "D", np.where((df.sa > df.sb) == (side == "H"), "W", "L")),
        "gf": (df.sa if side == "H" else df.sb).astype(int),
        "ga": (df.sb if side == "H" else df.sa).astype(int),
        "odds": df.oh if side == "H" else df.oa,
        "win": ((df.sa > df.sb) if side == "H" else (df.sb > df.sa)).astype(int),
        "draw": (df.sa == df.sb).astype(int),
        "imp_win": df.imp_h if side == "H" else df.imp_a, "imp_draw": df.imp_d,
        "over25": (df.total > 2.5).astype(int), "over35": (df.total > 3.5).astype(int),
        "btts": ((df.sa > 0) & (df.sb > 0)).astype(int),
        "io25": df.io25, "io35": df.io35, "ibtts": df.ibtts}))
L = pd.concat(recs).sort_values(["team", "ts"]).reset_index(drop=True)
g = L.groupby("team")
for L_ in range(1, 6):
    L[f"r{L_}"] = g["res"].shift(L_)
    L[f"gf{L_}"] = g["gf"].shift(L_)

# séries "entrantes" : longueur de la série d'une condition finissant JUSTE AVANT
# chaque match (le résultat courant n'entre pas dans son propre prédicteur -> zéro fuite)
_tc = pd.factorize(L.team)[0]
def streak_before(cond_arr):
    out = np.zeros(len(cond_arr), int); c = 0; pt = -1
    for i in range(len(cond_arr)):
        if _tc[i] != pt:
            c = 0; pt = _tc[i]
        out[i] = c
        c = c + 1 if cond_arr[i] else 0
    return out
_conds = {"win": (L.res == "W").values, "loss": (L.res == "L").values, "nodraw": (L.res != "D").values,
          "unbeaten": (L.res != "L").values, "winless": (L.res != "W").values,
          "hi": (L.gf >= 3).values, "lo": (L.gf <= 1).values, "ov": ((L.gf + L.ga) > 2.5).values,
          "un": ((L.gf + L.ga) <= 2.5).values, "cs": (L.ga == 0).values}
for nm, cd in _conds.items():
    L[f"st_{nm}"] = streak_before(cd)
L = L.dropna(subset=["io25", "io35", "ibtts", "r1"]).reset_index(drop=True)
n = len(L); print(f"{n} lignes équipe-match avec features", flush=True)

# ---- CONDITIONS DE BASE (booléens) ----
C = {}
for L_ in range(1, 6):
    for r in ("W", "D", "L"):
        C[f"r{L_}={r}"] = (L[f"r{L_}"] == r).values
    C[f"gf{L_}>=3"] = (L[f"gf{L_}"] >= 3).values
    C[f"gf{L_}<=1"] = (L[f"gf{L_}"] <= 1).values
for nm in ("win", "loss", "nodraw", "unbeaten", "winless", "hi", "lo", "ov", "un", "cs"):
    for k in (2, 3, 4):
        C[f"st_{nm}>={k}"] = (L[f"st_{nm}"] >= k).values
C["venue=H"] = (L.venue == 1).values
C["venue=A"] = (L.venue == 0).values
names = list(C.keys())
print(f"{len(names)} conditions de base", flush=True)

# strates de cote
odds = L.odds.values
strata = {"ALL": np.ones(n, bool), "fav": odds < 1.9, "mid": (odds >= 1.9) & (odds < 2.8), "out": odds >= 2.8}
# cibles (outcome, implied)
TARGETS = {"win": (L.win.values, L.imp_win.values), "draw": (L.draw.values, L.imp_draw.values),
           "loss": ((1-L.win-L.draw).clip(0,1).values, (1-L.imp_win-L.imp_draw).clip(0,1).values),
           "over25": (L.over25.values, L.io25.values), "under25": ((1-L.over25).values, (1-L.io25).values),
           "over35": (L.over35.values, L.io35.values), "btts": (L.btts.values, L.ibtts.values)}
train = (L.ts < L.ts.iloc[n//2]).values
test = ~train

# ---- génération des combos : 1, 2, 3 conditions (AND) ----
combo_list = [(nm,) for nm in names]
combo_list += list(combinations(names, 2))
core = [nm for nm in names if nm.startswith("st_") or nm.startswith("r1") or nm.startswith("r2")]
combo_list += list(combinations(core, 3))
print(f"{len(combo_list)} combos × {len(TARGETS)} cibles × {len(strata)} strates "
      f"= {len(combo_list)*len(TARGETS)*len(strata):,} hypothèses", flush=True)

# ---- test massif ----
cand_p, cand_meta = [], []
tested = 0
for combo in combo_list:
    mask = C[combo[0]].copy()
    for extra in combo[1:]:
        mask &= C[extra]
    for sname, sm in strata.items():
        cm = mask & sm
        cm_tr = cm & train
        ntr = cm_tr.sum()
        if ntr < 150:
            continue
        for tname, (out, imp) in TARGETS.items():
            tested += 1
            rt = out[cm_tr].mean() - imp[cm_tr].mean()
            if abs(rt) < 0.02:
                continue
            cm_te = cm & test
            nte = cm_te.sum()
            if nte < 30:
                continue
            rte = out[cm_te].mean() - imp[cm_te].mean()
            if np.sign(rt) != np.sign(rte):
                continue
            base = imp[cm_te].mean()
            se = np.sqrt(max(base*(1-base), 1e-6)/nte)
            z = rte/se
            p = 2*(1-0.5*(1+erf(abs(z)/1.4142135623730951)))
            cand_p.append(p)
            cand_meta.append((combo, sname, tname, round(rt, 3), round(rte, 3), int(nte), round(p, 4)))

print(f"\n{tested:,} hypothèses testées | {len(cand_p)} candidates (train ok + même signe OOS) "
      f"| {time.time()-t0:.0f}s", flush=True)
# BH-FDR global
surv = []
if cand_p:
    order = np.argsort(cand_p)
    m = len(cand_p)
    for rank, idx in enumerate(order, 1):
        if cand_p[idx] <= rank/m*0.05:
            surv = [cand_meta[order[j]] for j in range(rank)]
print("="*64)
print(f"  {tested:,} HYPOTHÈSES testées automatiquement")
print(f"  Faux positifs attendus par HASARD (5%) : ~{int(0.05*tested):,}")
print(f"  Candidates (semblent marcher sur TRAIN + OOS même signe) : {len(cand_p)}")
print(f"  SURVIVANTS après BH-FDR global : {len(surv)}")
if surv:
    print("  ⚠️ survivant(s) à vérifier en adversarial :")
    for s in surv[:10]:
        print(f"     {s}")
else:
    print("  -> AUCUN survivant. Sur des dizaines de milliers d'idées, ZÉRO tient.")
    print("     Le RNG est sans mémoire : toute 'idée folle' historique est un mirage.")
print("="*64)
