"""APPLIQUER LES 'EDGES' SUR DONNÉES FRAÎCHES — marchent-ils vraiment ?

Découpe 3 parts chrono : APPREND (50%) / SÉLECTION (25%) / TEST VIERGE (25%).
- On retient tous les 'edges' qui semblent marcher sur APPREND + SÉLECTION
  (exactement comme les ~3704 candidats du mineur massif).
- On les applique EN AVEUGLE sur le TEST vierge (jamais utilisé) et on mesure :
  edge réel moyen, % encore positifs, et ROI d'une bankroll qui les parie TOUS
  (aux cotes JUSTES sans marge = test généreux).
"""
from __future__ import annotations
import json, sys
from itertools import combinations
from math import erf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

LG = "InstantLeague-8035"
eng = create_engine(load_settings().db_url, connect_args={"timeout": 30})
df = pd.read_sql(text(f"""
    SELECT ev.expected_start ts, ev.team_a ta, ev.team_b tb, o.odds_home oh, o.odds_draw od,
           o.odds_away oa, o.extra_markets xm, r.score_a sa, r.score_b sb
    FROM events ev JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=ev.id)
    JOIN results r ON r.event_id=ev.id
    WHERE r.score_a IS NOT NULL AND ev.competition='{LG}' AND o.odds_home>1 ORDER BY ev.expected_start"""),
    eng).drop_duplicates(["ts", "ta", "tb"]).reset_index(drop=True)
inv = 1/df.oh + 1/df.od + 1/df.oa
df["imp_h"], df["imp_d"], df["imp_a"] = (1/df.oh)/inv, (1/df.od)/inv, (1/df.oa)/inv
df["total"] = df.sa + df.sb


def gm(xm, pref):
    for k, v in (xm or {}).items():
        if k.replace("\x82", "é").replace("\xe9", "é").startswith(pref):
            return v
    return None
io25 = []
for raw in df.xm:
    try: xm = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception: xm = {}
    tt = gm(xm, "Total de buts")
    v = {k: 1/tt[k] for k in [str(x) for x in range(7)]
         if tt and isinstance(tt.get(k), (int, float)) and 1 < tt[k] < 99.99} if tt else {}
    s = sum(v.values()); io25.append(sum(v[str(k)] for k in range(3, 7))/s if s and len(v) == 7 else np.nan)
df["io25"] = io25

# perspective équipe + séries (comme le mineur)
recs = []
for side in ("H", "A"):
    recs.append(pd.DataFrame({"ts": df.ts, "team": df.ta if side == "H" else df.tb,
        "res": np.where(df.sa == df.sb, "D", np.where((df.sa > df.sb) == (side == "H"), "W", "L")),
        "gf": (df.sa if side == "H" else df.sb).astype(int), "venue": 1 if side == "H" else 0,
        "odds": df.oh if side == "H" else df.oa,
        "win": ((df.sa > df.sb) if side == "H" else (df.sb > df.sa)).astype(int),
        "draw": (df.sa == df.sb).astype(int), "over25": (df.total > 2.5).astype(int),
        "imp_win": df.imp_h if side == "H" else df.imp_a, "imp_draw": df.imp_d, "io25": df.io25}))
L = pd.concat(recs).sort_values("ts").reset_index(drop=True)
tc = pd.factorize(L.team)[0]
def sb_streak(cond):
    out = np.zeros(len(cond), int); c = 0; pt = -1
    for i in range(len(cond)):
        if tc[i] != pt: c = 0; pt = tc[i]
        out[i] = c; c = c+1 if cond[i] else 0
    return out
for nm, cd in [("win", (L.res == "W").values), ("loss", (L.res == "L").values), ("nodraw", (L.res != "D").values),
               ("unbeaten", (L.res != "L").values), ("winless", (L.res != "W").values),
               ("hi", (L.gf >= 3).values), ("lo", (L.gf <= 1).values), ("over", (L.over25 == 1).values),
               ("under", (L.over25 == 0).values)]:
    L[f"st_{nm}"] = sb_streak(cd)
g = L.groupby("team")
for k in (1, 2, 3):
    L[f"r{k}"] = g["res"].shift(k)
L = L.dropna(subset=["io25", "r1"]).reset_index(drop=True)
N = len(L)

# conditions de base
C = {}
for k in (1, 2, 3):
    for r in ("W", "D", "L"): C[f"r{k}={r}"] = (L[f"r{k}"] == r).values
for nm in ("win", "loss", "nodraw", "unbeaten", "winless", "hi", "lo", "over", "under"):
    for kk in (2, 3, 4): C[f"st_{nm}>={kk}"] = (L[f"st_{nm}"] >= kk).values
C["venue=H"] = (L.venue == 1).values
names = list(C.keys())
TARGETS = {"win": (L.win.values, L.imp_win.values), "draw": (L.draw.values, L.imp_draw.values),
           "over25": (L.over25.values, L.io25.values), "under25": ((1-L.over25).values, (1-L.io25).values)}

t1, t2 = L.ts.iloc[N//2], L.ts.iloc[3*N//4]
learn = (L.ts < t1).values; sel = ((L.ts >= t1) & (L.ts < t2)).values; test = (L.ts >= t2).values
print(f"{N} lignes | APPREND {learn.sum()} / SÉLECTION {sel.sum()} / TEST vierge {test.sum()}", flush=True)

# strates de cote (comme le mineur massif) pour multiplier les candidats
odds = L.odds.values
strata = {"ALL": np.ones(N, bool), "fav": odds < 1.9, "mid": (odds >= 1.9) & (odds < 2.8), "out": odds >= 2.8}
combos = [(n,) for n in names] + list(combinations(names, 2))    # 1+2 conditions (RAM limitée)
edges = []
for combo in combos:
    m = C[combo[0]].copy()
    for e in combo[1:]: m &= C[e]
    for sn, sm in strata.items():
        cm = m & sm
        for tn, (out, imp) in TARGETS.items():
            ml = cm & learn
            if ml.sum() < 150: continue
            r_learn = out[ml].mean() - imp[ml].mean()
            if abs(r_learn) < 0.02: continue
            ms = cm & sel
            if ms.sum() < 30: continue
            r_sel = out[ms].mean() - imp[ms].mean()
            if np.sign(r_sel) != np.sign(r_learn): continue  # "marche" sur sélection -> retenu
            edges.append((combo, sn, tn, np.sign(r_learn)))
print(f"\n>>> {len(edges)} 'EDGES' retenus (marchent sur apprend + sélection) — les mirages candidats", flush=True)

# --- APPLICATION EN AVEUGLE SUR LE TEST VIERGE ---
pos = 0; test_edges = []; pnl_sum = 0.0; nbets = 0
for combo, sn, tn, sign in edges:
    m = C[combo[0]].copy()
    for e in combo[1:]: m &= C[e]
    mt = m & strata[sn] & test
    if mt.sum() < 20: continue
    out, imp = TARGETS[tn]
    r_test = out[mt].mean() - imp[mt].mean()
    test_edges.append(r_test * sign)                # dans le SENS parié
    pos += int(np.sign(r_test) == sign)
    # bankroll : parie le BON côté à SA cote juste (sign>0: l'issue à 1/imp ;
    # sign<0: le complément à 1/(1-imp)) — sans marge = test généreux
    if sign > 0:
        o = 1/np.clip(imp[mt], 1e-3, 1); win = out[mt]
    else:
        o = 1/np.clip(1-imp[mt], 1e-3, 1); win = 1-out[mt]
    pnl_sum += float((win*o - 1).sum()); nbets += int(mt.sum())
te = np.array(test_edges); roi = pnl_sum/max(nbets, 1)
print(f"\n=== RÉSULTAT SUR LE TEST VIERGE (jamais vu) ===")
print(f"  edges appliqués : {len(te)}")
print(f"  edge réel moyen dans le sens parié : {100*te.mean():+.3f}pp  (attendu si vrai : positif ; si mirage : ~0)")
print(f"  edges encore POSITIFS sur test : {100*pos/len(te):.1f}%  (50% = pile ou face = mirage)")
print(f"  ROI d'une bankroll pariant TOUS ces edges ({nbets} paris), aux cotes JUSTES sans marge :")
print(f"     {100*roi:+.2f}%  (>0 requis pour un vrai edge ; ~0 = mirage ; et AVEC la marge ~-6% ce serait pire)")
print("="*60)
print("  VERDICT : si edge≈0 et positifs≈50%, les 'edges' ne sont QUE du bruit")
print("  d'entraînement — ils s'évaporent sur données fraîches. C'est la preuve")
print("  qu'appliquer ce qui 'marchait' fait perdre exactement la marge.")
