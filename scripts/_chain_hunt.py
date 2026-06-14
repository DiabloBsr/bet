"""Mining de CHAÎNAGES : total × dominance × BTTS → score/total le + probable.
Découverte TRAIN, validation TEST (réplication obligatoire). Produit une table
de décision lisible + exports/chain_table.json pour le prédicteur live."""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd

CSV = Path(__file__).resolve().parents[1] / "exports" / "combokeys_features.csv"
OUT = Path(__file__).resolve().parents[1] / "exports" / "chain_table.json"
MIN_TR, MIN_TE = 120, 40

# bandes (mêmes bornes que les edges confirmés) --------------------------------
TOT_BANDS  = [(-1, 2.45, "tot_bas"), (2.45, 3.13, "tot_moy"), (3.13, 99, "tot_haut")]
DIFF_BANDS = [(-99, -1.0, "ext_ext"), (-1.0, -0.3, "ext"), (-0.3, 0.3, "equilibre"),
              (0.3, 1.0, "dom"), (1.0, 99, "dom_dom")]
BTTS_BANDS = [(-1, 0.45, "btts_bas"), (0.45, 0.60, "btts_moy"), (0.60, 99, "btts_haut")]

def band(v, bands):
    for lo, hi, lbl in bands:
        if lo <= v < hi: return lbl
    return None

df = pd.read_csv(CSV).sort_values("expected_start").reset_index(drop=True)
df["bt"] = df.lam_tot.apply(lambda v: band(v, TOT_BANDS))
df["bd"] = df.lam_diff.apply(lambda v: band(v, DIFF_BANDS))
df["bb"] = df.p_btts.apply(lambda v: band(v, BTTS_BANDS) if pd.notna(v) else None)
cut = int(len(df) * 0.70)
tr, te = df.iloc[:cut], df.iloc[cut:]

def cell_stats(sub):
    vc = sub.exact_score.value_counts(normalize=True)
    top3 = list(vc.head(3).index)
    vt = sub.total_goals.value_counts(normalize=True)
    o25 = (sub.total_goals >= 3).mean()
    return dict(score=vc.index[0], rate=float(vc.iloc[0]), top3=top3,
                top3_cum=float(sub.exact_score.isin(top3).mean()),
                total=int(vt.index[0]), total_rate=float(vt.iloc[0]),
                ou=("Over2.5" if o25 >= 0.5 else "Under2.5"),
                ou_rate=float(max(o25, 1 - o25)), n=len(sub))

print("="*128)
print(f"{'CHAÎNAGE (total | dominance | btts)':<42} {'n tr/te':>10} {'SCORE':>6} {'tr→te':>10} "
      f"{'TOP-3':>14} {'top3 tr→te':>11} {'O/U':>9} {'tr→te':>10}")
print("="*128)

rows = []
prod_cells = {}
for bt in [b[2] for b in TOT_BANDS]:
    for bd in [b[2] for b in DIFF_BANDS]:
        for bb in [b[2] for b in BTTS_BANDS]:
            str_tr = tr[(tr.bt==bt)&(tr.bd==bd)&(tr.bb==bb)]
            str_te = te[(te.bt==bt)&(te.bd==bd)&(te.bb==bb)]
            if len(str_tr) < MIN_TR or len(str_te) < MIN_TE: continue
            s_tr = cell_stats(str_tr)
            # rates OOS pour les MÊMES sélections issues du train
            r_te_score = (str_te.exact_score == s_tr["score"]).mean()
            r_te_top3  = str_te.exact_score.isin(s_tr["top3"]).mean()
            ou_te = (str_te.total_goals >= 3).mean() if s_tr["ou"]=="Over2.5" else (str_te.total_goals <= 2).mean()
            robust = (r_te_top3 >= s_tr["top3_cum"] - 0.05) and (ou_te >= s_tr["ou_rate"] - 0.05)
            key = f"{bt}|{bd}|{bb}"
            flag = "✅" if robust else "  "
            print(f"{flag} {key:<40} {len(str_tr):>4}/{len(str_te):<4} {s_tr['score']:>6} "
                  f"{s_tr['rate']*100:>3.0f}→{r_te_score*100:>3.0f}% {'+'.join(s_tr['top3']):>14} "
                  f"{s_tr['top3_cum']*100:>3.0f}→{r_te_top3*100:>3.0f}% {s_tr['ou']:>9} "
                  f"{s_tr['ou_rate']*100:>3.0f}→{ou_te*100:>3.0f}%")
            rows.append((key, robust, r_te_top3, s_tr))
            if robust:
                # table de prod : recalcul sur TOUT le data pour la meilleure estimation
                full = df[(df.bt==bt)&(df.bd==bd)&(df.bb==bb)]
                fs = cell_stats(full)
                prod_cells[key] = dict(score=fs["score"], rate=round(fs["rate"],3),
                                       top3=fs["top3"], top3_cum=round(fs["top3_cum"],3),
                                       ou=fs["ou"], ou_rate=round(fs["ou_rate"],3),
                                       total=fs["total"], n=fs["n"])

print("="*128)
nrob = sum(1 for _,r,_,_ in rows if r)
print(f"\nCellules testées : {len(rows)} | robustes (répliquent OOS) : {nrob}")
print("\n🏆 TOP chaînages robustes par couverture Top-3 OOS :")
for key, robust, r_te_top3, s in sorted([x for x in rows if x[1]], key=lambda x: -x[2])[:10]:
    print(f"   {key:<40} → {s['score']} | Top3 {'+'.join(s['top3'])} = {r_te_top3*100:.0f}% OOS | {s['ou']} {s['ou_rate']*100:.0f}%")

OUT.write_text(json.dumps({
    "_bands": {"tot": TOT_BANDS, "diff": DIFF_BANDS, "btts": BTTS_BANDS},
    "cells": prod_cells,
}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n→ écrit {OUT.name} ({len(prod_cells)} cellules robustes pour le prédicteur)")
