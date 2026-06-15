"""BACKTEST HONNÊTE par total (60/20/20) — recettes FIGÉES (interprétables, pas
sélectionnées sur le test), hit-rate sur fold1/2/3 + EV au prix offert sur fold3
(jamais touché). Répond sans biais : peut-on TAPER chaque total en live + est-ce rentable ?"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd

df = pd.read_csv(Path(__file__).resolve().parents[1]/"exports"/"combokeys_features.csv").sort_values("expected_start").reset_index(drop=True)
df["abs_diff"]=df.lam_diff.abs()
n=len(df); f1,f2=int(n*0.6),int(n*0.8)
fold=np.where(df.index<f1,"F1",np.where(df.index<f2,"F2","F3"))
df["fold"]=fold

# recettes FIGÉES par total (bande lam_tot où il pique + raffinement) — choisies a priori, PAS sur le test
RECIPES = {
 0: ("lam_tot<2.05 & p_btts<0.45", (df.lam_tot<2.05)&(df.p_btts<0.45)),
 1: ("lam_tot<2.25",                (df.lam_tot<2.25)),
 2: ("2.25<=lam_tot<2.6",           (df.lam_tot>=2.25)&(df.lam_tot<2.6)),
 3: ("2.6<=lam_tot<3.1",            (df.lam_tot>=2.6)&(df.lam_tot<3.1)),
 4: ("3.1<=lam_tot<3.55",           (df.lam_tot>=3.1)&(df.lam_tot<3.55)),
 5: ("3.55<=lam_tot<4.3",           (df.lam_tot>=3.55)&(df.lam_tot<4.3)),
 6: ("lam_tot>=4.0",                (df.lam_tot>=4.0)),
}
glob = {t:(df.total_goals==t).mean() for t in range(7)}
print("="*104)
print("BACKTEST HONNÊTE PAR TOTAL — recette figée, hit-rate fold1/2/3 + EV fold3 (jamais touché pour sélection)")
print("="*104)
print(f"{'tot':>4}{'recette (a priori)':<26}{'base':>6}{'  hit F1':>8}{'hit F2':>8}{'hit F3':>8}{'  EV F3':>8}{'cote':>7}  verdict")
for t,(lbl,mask) in RECIPES.items():
    tl="6+" if t==6 else str(t)
    sub=df[mask]
    def hit(fold):
        s=sub[sub.fold==fold]
        return (s.total_goals==t).mean() if len(s)>=30 else float('nan'), len(s)
    h1,n1=hit("F1"); h2,n2=hit("F2"); h3,n3=hit("F3")
    s3=sub[(sub.fold=="F3")].dropna(subset=[f"off_t{t}"])
    ev3=((s3.total_goals==t)*s3[f"off_t{t}"]-1).mean() if len(s3)>=30 else float('nan')
    cote=s3[f"off_t{t}"].mean() if len(s3)>0 else float('nan')
    base=glob[t]
    # verdict accuracy : hit F3 > 1.25x base ET hit replique (F1~F2~F3)
    repl = (h3==h3 and h2==h2 and h1==h1 and min(h1,h2,h3)>base)
    tap = "🎯 lean réel" if (h3==h3 and h3>base*1.25 and repl) else ("~ lean faible" if (h3==h3 and h3>base*1.1) else "≈ base")
    rent = " +RENTABLE?" if (ev3==ev3 and ev3>0.03) else (" −EV" if ev3==ev3 else "")
    print(f"{tl:>4} {lbl:<25}{base*100:>5.0f}%{h1*100:>7.0f}%{h2*100:>7.0f}%{h3*100:>7.0f}%{ev3*100:>+7.0f}%{cote:>7.1f}  {tap}{rent}")
print("\n'lean réel' = le total se penche vraiment (hit F3 > 1.25× base ET stable sur les 3 folds).")
print("EV F3 = rendement réel au prix offert sur la tranche JAMAIS utilisée pour choisir la recette (honnête).")
print("Si EV F3 ~0 ou négatif partout malgré le lean -> on PRÉDIT le total mais le book le price = pas rentable.")
