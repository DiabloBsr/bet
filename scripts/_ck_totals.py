"""Track B+ : analyse EXHAUSTIVE des marches TOTAL DE BUTS par cles combinees.

Pour chaque marche (Over/Under 1.5/2.5/3.5/4.5, total exact 0..6) et chaque cle
combinee 1/2/3 signaux : taux EMPIRIQUE (realise) vs taux PRICE (grille Poisson),
le GAP = edge (le simulateur devie : +0.12 but, drama mode +74%), lift, et EV vs
cote offerte (totals exacts via off_t). Split forward chrono 70/30.

Insight : le total price est exactement Poisson(lam_tot) (somme de Poissons indep),
donc priced_Over2.5 = 1 - cdf(2, lam_tot). Le gap empirique-vs-price est le vrai signal.

Sortie : exports/totals_edges_report.md
Usage: ./.venv/Scripts/python.exe scripts/_ck_totals.py
"""
import sys
sys.path.insert(0, ".")
import json
import itertools
import numpy as np
import pandas as pd
from scipy.stats import poisson, norm

try:
    df = pd.read_parquet("exports/combokeys_features.parquet")
except Exception:
    df = pd.read_csv("exports/combokeys_features.csv")
df["expected_start"] = pd.to_datetime(df.expected_start, utc=True, errors="coerce")
df = df.dropna(subset=["expected_start", "lam_tot"]).sort_values("expected_start").reset_index(drop=True)

# buts par equipe (depuis exact_score)
sp = df.exact_score.str.split("-", expand=True)
df["hg"] = pd.to_numeric(sp[0], errors="coerce")
df["ag"] = pd.to_numeric(sp[1], errors="coerce")
df = df.dropna(subset=["hg", "ag"])
print(f"n={len(df)}")

BINSPEC = json.load(open("exports/combokeys_binspec.json", encoding="utf-8"))
SIGNALS = ["fav", "dog", "odds_ratio", "od", "lam_tot", "lam_diff", "p_btts"]
for s in SIGNALS:
    if s in BINSPEC:
        df[f"b_{s}"] = pd.cut(df[s], BINSPEC[s])

# split forward 70/30 par ligue
df["is_test"] = False
for comp, g in df.groupby("competition"):
    df.loc[g.index[int(len(g) * 0.70):], "is_test"] = True
tr, te = df[~df.is_test], df[df.is_test]
print(f"train={len(tr)} test={len(te)}")

# ---- marches totals : (nom, fonction outcome, priced rate fn(lam_tot)) ----
MARKETS = []
for thr in (1, 2, 3, 4):  # Over X.5
    MARKETS.append((f"Over{thr}.5",
                    lambda d, t=thr: (d.total_goals > t).astype(float),
                    lambda lt, t=thr: 1 - poisson.cdf(t, lt), None))
    MARKETS.append((f"Under{thr}.5",
                    lambda d, t=thr: (d.total_goals <= t).astype(float),
                    lambda lt, t=thr: poisson.cdf(t, lt), None))
for k in range(0, 7):  # total exact = k (avec EV via off_t{k})
    MARKETS.append((f"Total={k}",
                    lambda d, kk=k: (d.total_goals == kk).astype(float),
                    lambda lt, kk=k: poisson.pmf(kk, lt), f"off_t{k}"))

# ---- global : biais systematique empirique vs price ----
print("\n=== BIAIS GLOBAL empirique vs price (tout le test) ===")
for name, outc, pr, _ in MARKETS:
    if not name.startswith(("Over", "Under")):
        continue
    emp = outc(te).mean()
    prc = pr(te.lam_tot.values).mean()
    print(f"  {name:<9} empirique {100*emp:5.1f}%  price {100*prc:5.1f}%  gap {100*(emp-prc):+5.1f}%")

# stratifie par equilibre (lam_diff) : le drama mode frappe les matchs serres
print("\n=== Over2.5 : gap par |lam_diff| (drama mode sur les matchs serres) ===")
te = te.copy()
te["absdiff"] = te.lam_diff.abs()
for lo, hi in [(0, 0.3), (0.3, 0.7), (0.7, 1.2), (1.2, 9)]:
    g = te[(te.absdiff >= lo) & (te.absdiff < hi)]
    if len(g) < 50: continue
    emp = (g.total_goals > 2).mean(); prc = (1 - poisson.cdf(2, g.lam_tot.values)).mean()
    print(f"  |lam_diff|∈[{lo},{hi}) n={len(g):>4}  Over2.5 emp {100*emp:5.1f}%  price {100*prc:5.1f}%  gap {100*(emp-prc):+5.1f}%")

# ---- balayage cles combinees : top edges par marche ----
N_MIN_TR, N_MIN_TE = 150, 150
bcols_all = [f"b_{s}" for s in SIGNALS if f"b_{s}" in df.columns]


def ztest(emp, prc, n):
    se = np.sqrt(max(prc * (1 - prc), 1e-9) / n)
    return (emp - prc) / se if se > 0 else 0.0


rows = []
M = 0
for k in (1, 2, 3):
    for combo in itertools.combinations(bcols_all, k):
        gte = {key: g for key, g in te.groupby(list(combo), observed=True)}
        for key, ctr in tr.groupby(list(combo), observed=True):
            cte = gte.get(key)
            if cte is None or len(ctr) < N_MIN_TR or len(cte) < N_MIN_TE:
                continue
            kv = key if isinstance(key, tuple) else (key,)
            keystr = " & ".join(f"{s[2:]}∈{v}" for s, v in zip(combo, kv))
            for name, outc, pr, evcol in MARKETS:
                emp = float(outc(cte).mean())
                prc = float(pr(cte.lam_tot.values).mean())
                n = len(cte)
                z = ztest(emp, prc, n)
                M += 1
                ev = np.nan
                if evcol and evcol in cte:
                    cote = cte[evcol]
                    win = outc(cte).values
                    profit = np.where(np.isnan(cote.values), np.nan, cote.values * win - 1.0)
                    ev = 100 * np.nanmean(profit) if np.isfinite(profit).any() else np.nan
                rows.append(dict(market=name, key=keystr, ncombo=k, n=n,
                                 emp=round(100 * emp, 1), price=round(100 * prc, 1),
                                 gap=round(100 * (emp - prc), 1), z=round(z, 2),
                                 ev=round(ev, 1) if ev == ev else np.nan))

res = pd.DataFrame(rows)
zstar = norm.ppf(1 - 0.05 / (2 * max(M, 1)))
print(f"\nM={M} tests  Bonferroni z*={zstar:.2f}")

# EV estimee des Over (cote offerte ~ 1/(price*1.06), marge '+/-' documentee 6%)
# + EV REELLE Over/Under 3.5 via la cote '+/-' captee (off_ou_over35/under35)
MARGIN_OU = 1.06
res["ev_est"] = np.nan
mask_over = res.market.str.startswith(("Over", "Under"))
res.loc[mask_over, "ev_est"] = ((res.loc[mask_over, "emp"] / 100) /
                                 ((res.loc[mask_over, "price"] / 100) * MARGIN_OU) - 1) * 100
res["ev_est"] = res.ev_est.round(1)

print("\n=== EV REELLE Over/Under 3.5 (cote '+/-' captee) — global test ===")
if "off_ou_over35" in te.columns:
    for nm, win, col in [("Over3.5", (te.total_goals > 3), "off_ou_over35"),
                         ("Under3.5", (te.total_goals <= 3), "off_ou_under35")]:
        cc = te[col]
        prof = np.where(cc.notna().values, cc.values * win.astype(float).values - 1.0, np.nan)
        if np.isfinite(prof).any():
            print(f"  {nm}: EV {100*np.nanmean(prof):+.1f}%  (n={int(np.isfinite(prof).sum())}, cote med {cc.median():.2f})")
else:
    print("  (cote +/- non presente dans l'extract — relancer _ck_extract.py)")

# top edges POSITIFS (empirique > price) significatifs
pos = res[(res.gap > 0) & (res.z.abs() >= zstar)].sort_values("gap", ascending=False)
print("\n=== TOP edges Over (empirique >> price, Bonferroni) ===")
over_pos = pos[pos.market.str.startswith("Over")].head(20)
print(over_pos[["market", "key", "n", "emp", "price", "gap", "z", "ev_est"]].to_string(index=False) if len(over_pos) else "aucun")

# top EV sur totals exacts
ev_pos = res[(res.ev > 5) & (res.market.str.startswith("Total="))].sort_values("ev", ascending=False)
print("\n=== TOP EV totals exacts (>5%) ===")
print(ev_pos[["market", "key", "n", "emp", "price", "gap", "ev"]].head(20).to_string(index=False) if len(ev_pos) else "aucun (marge mange l'edge)")

# ---- rapport ----
lines = ["# Totals Edges Report (exhaustif, cles combinees)", "",
         f"- n={len(df)} train={len(tr)} test={len(te)} | M={M} | Bonferroni z*={zstar:.2f}", "",
         "## Biais global empirique vs price (le simulateur sur-produit des buts)", ""]
for name, outc, pr, _ in MARKETS:
    if name.startswith(("Over", "Under")):
        emp = outc(te).mean(); prc = pr(te.lam_tot.values).mean()
        lines.append(f"- {name}: empirique {100*emp:.1f}% / price {100*prc:.1f}% / gap {100*(emp-prc):+.1f}%")
lines += ["", "## Top edges Over significatifs (Bonferroni) — ev_est = EV si Over offert a marge 6%", ""]
lines.append(over_pos[["market", "key", "n", "emp", "price", "gap", "z", "ev_est"]].to_string(index=False) if len(over_pos) else "aucun")
lines += ["", "## Top EV totals exacts", ""]
lines.append(ev_pos[["market", "key", "n", "emp", "price", "gap", "ev"]].head(20).to_string(index=False) if len(ev_pos) else "aucun")
with open("exports/totals_edges_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\necrit exports/totals_edges_report.md")
res.to_csv("exports/totals_edges_all.csv", index=False)
print("ecrit exports/totals_edges_all.csv")

# ---- calibration frozen : taux Over EMPIRIQUE par bande (lam_tot x lam_diff) ----
# Pour le predicteur live : proba Over calibree (mesuree), a comparer a la cote affichee.
TOT_EDGES = [0, 2.2, 2.6, 3.0, 3.4, 3.8, 99]
DIFF_EDGES = [-9, -0.5, 0, 0.5, 9]
cal = {"_edges": {"lam_tot": TOT_EDGES, "lam_diff": DIFF_EDGES},
       "_global": {}, "_lamtot": {}, "cells": {}}
allg = df  # tout (train+test) pour la table de reference figee
allg = allg.copy()
allg["bt"] = pd.cut(allg.lam_tot, TOT_EDGES)
allg["bd"] = pd.cut(allg.lam_diff, DIFF_EDGES)
for thr in (1, 2, 3):
    cal["_global"][f"over{thr}.5"] = round(float((allg.total_goals > thr).mean()), 4)
# repli marginal par bande de lam_tot SEUL (bien plus juste que la moyenne globale
# quand la cellule 2D (lam_tot x lam_diff) est vide)
for bt, g in allg.groupby("bt", observed=True):
    if len(g) < 80:
        continue
    key = f"{float(bt.left)}|{float(bt.right)}"
    cal["_lamtot"][key] = {f"over{thr}.5": round(float((g.total_goals > thr).mean()), 4) for thr in (1, 2, 3)}
    cal["_lamtot"][key]["n"] = len(g)
for (bt, bd), g in allg.groupby(["bt", "bd"], observed=True):
    if len(g) < 80:
        continue
    key = f"{bt.left}|{bt.right}|{bd.left}|{bd.right}"
    cal["cells"][key] = {f"over{thr}.5": round(float((g.total_goals > thr).mean()), 4) for thr in (1, 2, 3)}
    cal["cells"][key]["n"] = len(g)
with open("exports/totals_calibration.json", "w", encoding="utf-8") as f:
    json.dump(cal, f, indent=1)
print(f"ecrit exports/totals_calibration.json ({len(cal['cells'])} cellules)")
