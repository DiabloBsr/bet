"""Étude UPSET — quand le FAVORI (petite cote) se fait battre ?
Objectif pratique : trouver les conditions où le favori perd PLUS que sa cote
ne l'implique (zone à éviter) — ou prouver que l'upset est purement aléatoire.
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
df=pd.read_csv(Path(__file__).resolve().parents[1]/"exports"/"combokeys_features.csv").sort_values("expected_start").reset_index(drop=True)
df=df[(df.oh>1)&(df.oa>1)&(df.od>1)].copy()
df["sa"]=df.exact_score.str.split("-").str[0].astype(int); df["sb"]=df.exact_score.str.split("-").str[1].astype(int)
df["fav_home"]=df.oh<df.oa
df["fav_cote"]=df[["oh","oa"]].min(axis=1)
df["fav_won"]=np.where(df.fav_home, df.sa>df.sb, df.sb>df.sa)
inv=1/df.oh+1/df.od+1/df.oa
df["imp_fav"]=np.where(df.fav_home,(1/df.oh)/inv,(1/df.oa)/inv)
df["spread"]=(df.oh-df.oa).abs()
n=len(df); cut=int(n*0.7); tr,te=df.iloc[:cut],df.iloc[cut:]
def z(p,p0,n): return (p-p0)/math.sqrt(p0*(1-p0)/n) if (n>0 and 0<p0<1) else 0.0

print(f"n={n} | le favori gagne globalement {df.fav_won.mean()*100:.1f}% (implicite moy {df.imp_fav.mean()*100:.1f}%)\n")
print("="*86)
print("(1) CALIBRATION du favori par bande de cote — gagne-t-il ce que sa cote dit ?")
print("="*86)
print(f"{'cote favori':<14}{'n':>6}{'WIN réel':>10}{'implicite':>11}{'upset réel':>12}{'écart':>8}")
for lo,hi in [(1.0,1.2),(1.2,1.35),(1.35,1.5),(1.5,1.7),(1.7,1.9),(1.9,2.1),(2.1,2.5)]:
    s=df[(df.fav_cote>=lo)&(df.fav_cote<hi)]
    if len(s)<60: continue
    w=s.fav_won.mean(); imp=s.imp_fav.mean()
    print(f"{str(lo)+'-'+str(hi):<14}{len(s):>6}{w*100:>9.1f}%{imp*100:>10.1f}%{(1-w)*100:>11.1f}%{(w-imp)*100:>+7.1f}")
print("  -> si WIN réel ≈ implicite partout : aucune zone où le favori 'sur-perd'. L'upset = (1-cote), aléatoire.")

print("\n"+"="*86)
print("(2) Conditions qui FERAIENT sur-perdre le favori (upset > implicite) ? — sur fold3 propre")
print("="*86)
def test(name, mask):
    s=te[mask]
    if len(s)<40: print(f"  {name}: n={len(s)} trop peu"); return
    w=s.fav_won.mean(); imp=s.imp_fav.mean()
    tag="⚠️ favori SUR-PERD" if (w-imp)<-0.04 and z(w,imp,len(s))<-2 else ("✅ favori sur-perf" if (w-imp)>0.04 else "≈ conforme")
    print(f"  {name:<42} n={len(s):>4} WIN {w*100:.0f}% vs impl {imp*100:.0f}% (écart {(w-imp)*100:+.0f}pt) {tag}")
te_idx=te
test("favori À DOMICILE", te.fav_home)
test("favori À L'EXTÉRIEUR", ~te.fav_home)
test("match SERRÉ (spread<0.5)", te.spread<0.5)
test("nul cher (od>4.0)", te.od>4.0)
test("nul pas cher (od<3.3)", te.od<3.3)
test("favori léger (cote 1.9-2.3)", (te.fav_cote>=1.9)&(te.fav_cote<2.3))
test("gros total attendu (lam_tot>=3.3)", te.lam_tot>=3.3)
test("faible total (lam_tot<2.4)", te.lam_tot<2.4)
test("BTTS implicite haut (p_btts>0.6)", te.p_btts>0.6)

print("\n"+"="*86)
print("(3) Le PLUS sûr vs le PLUS risqué (par cote favori) — pour choisir où miser")
print("="*86)
print("  Favori le + sûr  : cote 1.0-1.2 -> gagne ~", f"{df[df.fav_cote<1.2].fav_won.mean()*100:.0f}%")
print("  Favori risqué    : cote 1.9-2.3 -> gagne ~", f"{df[(df.fav_cote>=1.9)&(df.fav_cote<2.3)].fav_won.mean()*100:.0f}% (upset ~", f"{(1-df[(df.fav_cote>=1.9)&(df.fav_cote<2.3)].fav_won.mean())*100:.0f}%)")
print("  -> conclusion : la seule chose qui prédit l'upset = la COTE elle-même (plus elle est haute, plus l'upset est probable).")
