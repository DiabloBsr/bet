"""Track B+ : moteur de CHAINAGE par contraintes -> resserrer vers UN score exact.

Logique (intention utilisateur) :
  total=3  -> {3-0,0-3,2-1,1-2}
  + dominance domicile (lam_diff>0) -> {3-0,2-1}
  + BTTS (les deux marquent ?) -> 2-1 si oui, 3-0 si non  => UN score.

On construit la table de decision sur 3 axes derives de l'inversion :
  - bande de total attendu (lam_tot)
  - dominance (lam_diff : qui mene)
  - BTTS implicite (p_btts du marche G/NG)
Pour chaque cellule : score modal realise + son taux + cote offerte + EV.
Validation forward-walk 70/30. Montre explicitement le cas total=3.

Sortie : exports/narrow_decision_table.md
Usage: ./.venv/Scripts/python.exe scripts/_ck_narrow.py
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd

try:
    df = pd.read_parquet("exports/combokeys_features.parquet")
except Exception:
    df = pd.read_csv("exports/combokeys_features.csv")
df["expected_start"] = pd.to_datetime(df.expected_start, utc=True, errors="coerce")
df = df.dropna(subset=["expected_start", "lam_tot", "lam_diff"]).sort_values("expected_start").reset_index(drop=True)
print(f"n={len(df)}")

# split forward 70/30 par ligue
df["is_test"] = False
for comp, g in df.groupby("competition"):
    df.loc[g.index[int(len(g) * 0.70):], "is_test"] = True
tr, te = df[~df.is_test].copy(), df[df.is_test].copy()

# axes de chaînage
TOT_BANDS = [(1.8, 2.3, "~2"), (2.3, 2.8, "~2-3"), (2.8, 3.3, "~3"), (3.3, 3.8, "~3-4"), (3.8, 9, "4+")]
DIFF_BANDS = [(-9, -0.5, "away++"), (-0.5, -0.15, "away+"), (-0.15, 0.15, "égal"),
              (0.15, 0.5, "home+"), (0.5, 9, "home++")]
BTTS_BANDS = [(0, 0.55, "BTTS-non"), (0.55, 1.01, "BTTS-oui")]


def band(v, bands):
    for lo, hi, lbl in bands:
        if lo <= v < hi:
            return lbl
    return None


for d in (tr, te):
    d["b_tot"] = d.lam_tot.map(lambda v: band(v, TOT_BANDS))
    d["b_diff"] = d.lam_diff.map(lambda v: band(v, DIFF_BANDS))
    d["b_btts"] = d.p_btts.map(lambda v: band(v, BTTS_BANDS) if pd.notna(v) else None)

SCORES_COTE = ["1-1", "2-1", "1-2", "1-0", "0-1", "2-0", "0-2", "0-0", "2-2",
               "3-0", "0-3", "3-1", "1-3", "3-2", "2-3"]

print("\n=== TABLE DE DECISION (total × dominance × BTTS) — test OOS ===")
print(f"{'total':<7}{'dominance':<10}{'btts':<10}{'n':>5} {'score':>6}{'taux':>7}{'2e':>14}{'cote':>7}{'EV':>7}")
rows = []
for tb in [b[2] for b in TOT_BANDS]:
    for db in [b[2] for b in DIFF_BANDS]:
        for bb in [b[2] for b in BTTS_BANDS]:
            ctr = tr[(tr.b_tot == tb) & (tr.b_diff == db) & (tr.b_btts == bb)]
            cte = te[(te.b_tot == tb) & (te.b_diff == db) & (te.b_btts == bb)]
            if len(ctr) < 120 or len(cte) < 80:
                continue
            vc_tr = ctr.exact_score.value_counts(normalize=True)
            S = vc_tr.index[0]
            rate = (cte.exact_score == S).mean()
            vc_te = cte.exact_score.value_counts(normalize=True)
            second = f"{vc_te.index[1]}({100*vc_te.iloc[1]:.0f}%)" if len(vc_te) > 1 else ""
            # EV via cote offerte du score S
            col = f"off_s_{S}"
            ev = np.nan; cote = np.nan
            if col in cte:
                cc = cte[col]
                if cc.notna().any():
                    cote = float(cc.median())
                    win = (cte.exact_score == S).astype(float).values
                    profit = np.where(np.isnan(cc.values), np.nan, cc.values * win - 1.0)
                    ev = 100 * np.nanmean(profit)
            print(f"{tb:<7}{db:<10}{bb:<10}{len(cte):>5} {S:>6}{100*rate:>6.0f}%{second:>14}"
                  f"{cote:>7.1f}" if cote == cote else
                  f"{tb:<7}{db:<10}{bb:<10}{len(cte):>5} {S:>6}{100*rate:>6.0f}%{second:>14}{'n/d':>7}", end="")
            print(f"{ev:>+6.0f}%" if ev == ev else f"{'':>7}")
            rows.append(dict(total=tb, dominance=db, btts=bb, n=len(cte), score=S,
                             rate=round(100 * rate, 0), cote=round(cote, 1) if cote == cote else None,
                             ev=round(ev, 0) if ev == ev else None))

# focus explicite : total ~3
print("\n=== FOCUS total~3 (ta démonstration : 3-0 vs 2-1 selon BTTS) ===")
g3 = te[te.b_tot == "~3"]
for db in [b[2] for b in DIFF_BANDS]:
    for bb in [b[2] for b in BTTS_BANDS]:
        c = g3[(g3.b_diff == db) & (g3.b_btts == bb)]
        if len(c) < 50: continue
        vc = c.exact_score.value_counts(normalize=True).head(3)
        print(f"  dominance={db:<8} {bb:<9} n={len(c):>4} -> " +
              " · ".join(f"{s}({100*p:.0f}%)" for s, p in vc.items()))

out = pd.DataFrame(rows)

# table frozen pour le predicteur live (lookup par bandes)
import json
narrow_json = {
    "_bands": {"tot": TOT_BANDS, "diff": DIFF_BANDS, "btts": BTTS_BANDS},
    "cells": {f"{r['total']}|{r['dominance']}|{r['btts']}":
              {k: r[k] for k in ("score", "rate", "cote", "ev", "n")} for r in rows},
}
with open("exports/narrow_table.json", "w", encoding="utf-8") as f:
    json.dump(narrow_json, f, indent=1, default=str)
print(f"ecrit exports/narrow_table.json ({len(rows)} cellules)")

lines = ["# Narrow Decision Table — chaînage total × dominance × BTTS -> 1 score", "",
         f"- n={len(df)} test={len(te)}", "",
         "Lecture : pour un match dont l'inversion donne (total attendu, qui mène, BTTS),",
         "le score modal historique + son taux OOS + la cote offerte + l'EV.", "",
         "```", out.to_string(index=False) if len(out) else "aucune cellule au seuil", "```"]
with open("exports/narrow_decision_table.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\necrit exports/narrow_decision_table.md")
