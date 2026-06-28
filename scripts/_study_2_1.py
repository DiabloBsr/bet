"""Étude approfondie 2-1 / 1-2 — le score le plus accuraté.
(A) quelle COTE-home vs COTE-away maximise 2-1 et 1-2 (+ OOS)
(B) quelle ÉQUIPE à domicile produit le + de 2-1 ; quelle équipe ext. le + de 1-2
(C) quelle PAIRE exacte
+ verdict : identité d'équipe ou juste les cotes ?"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
CSV=Path(__file__).resolve().parents[1]/"exports"/"combokeys_features.csv"
df=pd.read_csv(CSV).sort_values("expected_start").reset_index(drop=True)
df["is21"]=(df.exact_score=="2-1").astype(int); df["is12"]=(df.exact_score=="1-2").astype(int)
cut=int(len(df)*0.7); tr,te=df.iloc[:cut],df.iloc[cut:]
base21=df.is21.mean(); base12=df.is12.mean()
print(f"base 2-1 = {base21*100:.1f}% | base 1-2 = {base12*100:.1f}% (n={len(df)})\n")

# ===== (A) COTE home × COTE away =====
HB=[1.0,1.30,1.55,1.80,2.10,2.50,3.2,99]
df["ohb"]=pd.cut(df.oh,HB); df["oab"]=pd.cut(df.oa,HB)
print("="*78); print("(A) P(2-1) par bande [cote HOME × cote AWAY] — où le 2-1 pique"); print("="*78)
piv21=df.pivot_table(index="ohb",columns="oab",values="is21",aggfunc="mean",observed=True)*100
cnt=df.pivot_table(index="ohb",columns="oab",values="is21",aggfunc="size",observed=True)
# afficher cellules n>=40 triées
cells=[]
for i in piv21.index:
    for j in piv21.columns:
        v=piv21.loc[i,j]; n=cnt.loc[i,j] if (i in cnt.index and j in cnt.columns) else 0
        if pd.notna(v) and n>=40: cells.append((f"home{i} × away{j}",v,int(n)))
print("  TOP cellules 2-1 (n>=40):")
for lbl,v,n in sorted(cells,key=lambda x:-x[1])[:8]: print(f"    {lbl:<34} {v:.1f}%  (n={n})")
cells12=[]
piv12=df.pivot_table(index="ohb",columns="oab",values="is12",aggfunc="mean",observed=True)*100
for i in piv12.index:
    for j in piv12.columns:
        v=piv12.loc[i,j]; n=cnt.loc[i,j] if (i in cnt.index and j in cnt.columns) else 0
        if pd.notna(v) and n>=40: cells12.append((f"home{i} × away{j}",v,int(n)))
print("  TOP cellules 1-2 (n>=40):")
for lbl,v,n in sorted(cells12,key=lambda x:-x[1])[:8]: print(f"    {lbl:<34} {v:.1f}%  (n={n})")

# ===== (B) ÉQUIPE =====
print("\n"+"="*78); print("(B) ÉQUIPE à domicile -> 2-1  |  équipe extérieure -> 1-2"); print("="*78)
h=df.groupby("team_a").agg(n=("is21","size"),r21=("is21","mean"),fav=("oh","mean")).query("n>=40").sort_values("r21",ascending=False)
print("  HOME -> 2-1 (top, n>=40) :")
for t,row in h.head(8).iterrows(): print(f"    {t:<18} {row.r21*100:.1f}%  (n={int(row.n)}, cote home moy {row.fav:.2f})")
a=df.groupby("team_b").agg(n=("is12","size"),r12=("is12","mean"),fav=("oa","mean")).query("n>=40").sort_values("r12",ascending=False)
print("  AWAY -> 1-2 (top, n>=40) :")
for t,row in a.head(8).iterrows(): print(f"    {t:<18} {row.r12*100:.1f}%  (n={int(row.n)}, cote away moy {row.fav:.2f})")

# ===== (C) PAIRE =====
print("\n"+"="*78); print("(C) PAIRE exacte (n>=12)"); print("="*78)
p=df.groupby(["team_a","team_b"]).agg(n=("is21","size"),r21=("is21","mean"),r12=("is12","mean")).query("n>=12")
print("  TOP paires -> 2-1 :")
for (a_,b_),row in p.sort_values("r21",ascending=False).head(6).iterrows(): print(f"    {a_} v {b_:<16} 2-1={row.r21*100:.0f}%  (n={int(row.n)})")
print("  TOP paires -> 1-2 :")
for (a_,b_),row in p.sort_values("r12",ascending=False).head(6).iterrows(): print(f"    {a_} v {b_:<16} 1-2={row.r12*100:.0f}%  (n={int(row.n)})")

# ===== VERDICT : identité vs cotes (OOS) =====
print("\n"+"="*78); print("VERDICT — l'identité d'équipe ajoute-t-elle qqch aux cotes ? (OOS)"); print("="*78)
# meilleure cellule cote sur train -> tient sur test ?
trc=tr.groupby([pd.cut(tr.oh,HB),pd.cut(tr.oa,HB)],observed=True).is21.agg(["mean","size"]).query("size>=40")
best=trc.sort_values("mean",ascending=False).head(1)
bi=best.index[0]; ptr=best["mean"].iloc[0]
tec=te[(pd.cut(te.oh,HB)==bi[0])&(pd.cut(te.oa,HB)==bi[1])]
pte=tec.is21.mean() if len(tec)>20 else float("nan")
print(f"  Meilleure cellule-cote 2-1 (train) {bi}: TR {ptr*100:.0f}% -> TE {pte*100:.0f}% (n_te={len(tec)})")
# meilleure équipe-home sur train -> tient sur test ?
trh=tr.groupby("team_a").is21.agg(["mean","size"]).query("size>=30").sort_values("mean",ascending=False).head(1)
tname=trh.index[0]; phtr=trh["mean"].iloc[0]; phte=te[te.team_a==tname].is21.mean()
print(f"  Meilleure équipe-home 2-1 (train) {tname}: TR {phtr*100:.0f}% -> TE {phte*100:.0f}% (n_te={ (te.team_a==tname).sum() })")
