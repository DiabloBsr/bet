"""Track B++ : les findings 8035 generalisent-ils aux 9 ligues ?

Compare par ligue : (1) biais Over (empirique vs price Poisson(lam_tot)),
(2) gap Over2.5 sur la bande de pic |lam_diff|~0.7-1.2, (3) chaînage BTTS
(home-dominant total~3 : BTTS-non -> 3-0 ? BTTS-oui -> 2-1 ?), (4) plafond
score exact empirique.

Lit exports/combokeys_features_all.csv. Sortie : exports/leagues_compare.json
Usage: ./.venv/Scripts/python.exe scripts/_ck_leagues.py
"""
import sys
sys.path.insert(0, ".")
import json
import numpy as np
import pandas as pd
from scipy.stats import poisson

try:
    df = pd.read_parquet("exports/combokeys_features_all.parquet")
except Exception:
    df = pd.read_csv("exports/combokeys_features_all.csv")
df = df.dropna(subset=["lam_tot", "lam_diff", "total_goals", "exact_score"])
print(f"n total = {len(df)}  | ligues = {sorted(df.competition.unique())}\n")

LEAGUE_NAME = {
    "InstantLeague-8035": "Anglais", "InstantLeague-8036": "Italien",
    "InstantLeague-8037": "Espagnol", "InstantLeague-8042": "Francais",
    "InstantLeague-8043": "Allemand", "InstantLeague-8044": "Portugais",
    "InstantLeague-8056": "Champions(coupe)", "InstantLeague-8060": "CAN(coupe)",
    "InstantLeague-8065": "CdM(coupe)",
}


def league_metrics(g):
    out = {"n": len(g)}
    # biais Over (empirique vs price Poisson(lam_tot))
    for thr in (1, 2, 3):
        emp = float((g.total_goals > thr).mean())
        prc = float((1 - poisson.cdf(thr, g.lam_tot.values)).mean())
        out[f"over{thr}.5_emp"] = round(100 * emp, 1)
        out[f"over{thr}.5_price"] = round(100 * prc, 1)
        out[f"over{thr}.5_gap"] = round(100 * (emp - prc), 1)
    # gap Over2.5 sur la bande de pic |lam_diff| in [0.7,1.2)
    pk = g[g.lam_diff.abs().between(0.7, 1.2, inclusive="left")]
    if len(pk) > 50:
        emp = (pk.total_goals > 2).mean(); prc = (1 - poisson.cdf(2, pk.lam_diff.abs() * 0 + pk.lam_tot.values)).mean()
        out["over2.5_gap_peak"] = round(100 * (emp - prc), 1)
        out["over2.5_gap_peak_n"] = len(pk)
    # plafond score exact empirique
    vc = g.exact_score.value_counts(normalize=True)
    out["score_top1"] = round(100 * vc.iloc[0], 1)
    out["score_top3"] = round(100 * vc.iloc[:3].sum(), 1)
    out["score_top1_label"] = vc.index[0]
    # chaînage : home-dominant total~3 -> BTTS bascule 3-0 <-> 2-1 ?
    hd = g[(g.lam_diff > 0.5) & g.lam_tot.between(2.8, 3.3)]
    out["chain_n"] = len(hd)
    if len(hd) > 60 and hd.p_btts.notna().any():
        bnon = hd[hd.p_btts < 0.55]; boui = hd[hd.p_btts >= 0.55]
        out["chain_btts_non_top"] = bnon.exact_score.value_counts().index[0] if len(bnon) > 20 else None
        out["chain_btts_oui_top"] = boui.exact_score.value_counts().index[0] if len(boui) > 20 else None
        out["chain_btts_non_n"] = len(bnon); out["chain_btts_oui_n"] = len(boui)
    return out


results = {}
print(f"{'ligue':<18}{'n':>6}{'O2.5emp':>9}{'O2.5prc':>9}{'gap':>7}{'gapPeak':>9}"
      f"{'scoreT1':>9}{'scoreT3':>9}{'chain non→/oui→':>22}")
for comp, g in df.groupby("competition"):
    m = league_metrics(g)
    results[comp] = m
    name = LEAGUE_NAME.get(comp, comp)
    chain = f"{m.get('chain_btts_non_top','?')}/{m.get('chain_btts_oui_top','?')}"
    print(f"{name:<18}{m['n']:>6}{m['over2.5_emp']:>8.1f}%{m['over2.5_price']:>8.1f}%"
          f"{m['over2.5_gap']:>+7.1f}{m.get('over2.5_gap_peak', float('nan')):>+9.1f}"
          f"{m['score_top1']:>8.1f}%{m['score_top3']:>8.1f}%{chain:>22}")

# synthese : le gap Over2.5 est-il universel ?
gaps = {c: results[c]["over2.5_gap"] for c in results}
print(f"\n=== SYNTHESE gap Over2.5 ===")
print(f"  min {min(gaps.values()):+.1f}%  max {max(gaps.values()):+.1f}%  "
      f"median {np.median(list(gaps.values())):+.1f}%")
universal = all(v > 3 for v in gaps.values())
print(f"  gap Over2.5 > +3% dans TOUTES les ligues : {universal}")
print(f"  ligues avec gap < +5% : {[LEAGUE_NAME.get(c,c) for c,v in gaps.items() if v < 5]}")

with open("exports/leagues_compare.json", "w", encoding="utf-8") as f:
    json.dump({LEAGUE_NAME.get(c, c): results[c] for c in results}, f, indent=1, default=str)
print("\necrit exports/leagues_compare.json")
