"""Études profondes : (A) discriminant 2-1↔1-2, (B) recette 0-0 + EV + combiné,
(C) corrélation entre events d'un même round + rôle de la fourchette de cotes."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT/"exports"/"combokeys_features.csv").sort_values("expected_start").reset_index(drop=True)
df["abs_diff"]=df.lam_diff.abs(); df["spread"]=(df.oh-df.oa).abs()
cut=int(len(df)*0.70); tr,te=df.iloc[:cut],df.iloc[cut:]

print("█"*92); print("(A) DISCRIMINANT 2-1 ↔ 1-2  — quand un 'match à but unique d'écart' penche home vs away")
print("█"*92)
# matchs où le modèle dirait 2-1 OU 1-2 : total moyen-élevé, dominance modérée
zone = df[(df.lam_tot>=2.45)&(df.abs_diff.between(0.3,1.5))].copy()
zone["res"]=zone.exact_score
for lbl, sub in [("lam_diff>0 (HOME domine)", zone[zone.lam_diff>0]),
                 ("lam_diff<0 (AWAY domine)", zone[zone.lam_diff<0])]:
    n=len(sub); p21=(sub.res=="2-1").mean(); p12=(sub.res=="1-2").mean()
    print(f"  {lbl}: n={n} | 2-1={p21*100:.0f}% 1-2={p12*100:.0f}%  -> ratio 2-1/1-2 = {p21/max(p12,1e-9):.1f}x")
# dans la zone HOME-domine (prédit 2-1), quels signaux annoncent le FLIP vers 1-2 (upset) ?
home_zone = zone[zone.lam_diff>0].copy(); home_zone["flip"]=(home_zone.res=="1-2")
base_flip=home_zone.flip.mean()
print(f"\n  Dans la zone HOME-domine (prédit 2-1), P(résultat=1-2 upset) base = {base_flip*100:.1f}%")
print("  Signaux qui AUGMENTENT le risque de flip vers 1-2 :")
for col in ["dc_X2","p_btts","lam_a","od","spread"]:
    try:
        hi=home_zone[home_zone[col]>=home_zone[col].quantile(0.75)]
        lo=home_zone[home_zone[col]<=home_zone[col].quantile(0.25)]
        print(f"    {col:8} haut→flip {hi.flip.mean()*100:.0f}%  vs bas→flip {lo.flip.mean()*100:.0f}%")
    except Exception: pass

print("\n"+"█"*92); print("(B) RECETTE 0-0  — profil + EV réel à la cote offerte + faisabilité combiné")
print("█"*92)
def recipe(d, name, mask):
    sub=d[mask]; n=len(sub)
    if n<30: print(f"  {name}: n={n} trop peu"); return
    p00=(sub.exact_score=="0-0").mean()
    cote=sub["off_s_0-0"].dropna()
    # EV : profit moyen en misant 0-0 (hit*cote - 1), cote par match
    s2=sub.dropna(subset=["off_s_0-0"]).copy(); s2["hit"]=(s2.exact_score=="0-0").astype(int)
    ev=(s2.hit*s2["off_s_0-0"]-1).mean(); cote_moy=cote.mean()
    print(f"  {name}: n={n} | P(0-0)={p00*100:.1f}% | cote 0-0 moy={cote_moy:.1f} | EV/mise={ev*100:+.0f}% (n_cote={len(s2)})")
print("  Base + recettes (testé sur TOUT le data) :")
recipe(df,"GLOBAL", df.index>=0)
recipe(df,"faible total (lam_tot<2.2)", df.lam_tot<2.2)
recipe(df,"faible total + faible BTTS (<0.50)", (df.lam_tot<2.2)&(df.p_btts<0.50))
recipe(df,"tres faible tot + faible BTTS + serre", (df.lam_tot<2.2)&(df.p_btts<0.50)&(df.spread<1.3))
recipe(df,"ultra defensif (lam_tot<2.0 & BTTS<0.47)", (df.lam_tot<2.0)&(df.p_btts<0.47))
# faisabilité combiné : si on prend les K matchs/round les + 0-0-prone, prob qu'au moins un finisse 0-0
df["es"]=df.expected_start
cand=df[(df.lam_tot<2.2)&(df.p_btts<0.50)]
if len(cand):
    pr=(cand.exact_score=="0-0").mean()
    print(f"\n  Faisabilité combiné 0-0 : sur les matchs 'defensifs', P(0-0)={pr*100:.0f}%.")
    for k in [2,3,4]:
        print(f"    P(au moins un 0-0 sur {k} matchs defensifs) = {(1-(1-pr)**k)*100:.0f}%  "
              f"| cote combinee 'un 0-0 parmi {k}' approx -> miser chaque: {k} tickets")

print("\n"+"█"*92); print("(C) CORRÉLATION ENTRE EVENTS d'un même round + fourchette de cotes")
print("█"*92)
# regrouper par round (expected_start identique) ; la ligue est un RNG -> indépendance attendue
g=df.groupby("expected_start")
df["tot_match"]=df.total_goals
round_over=g.apply(lambda x:(x.total_goals>=3).mean(), include_groups=False)
round_n=g.size()
big=round_over[round_n>=8]
print(f"  Rounds (n>=8 matchs) : {len(big)} | %Over2.5 par round : moy={big.mean()*100:.0f}% écart-type={big.std()*100:.1f}%")
# overdispersion : si indépendant, var(%over par round) ~ p(1-p)/n . Compare réel vs théorique
p=df.total_goals.ge(3).mean(); n_avg=round_n[round_n>=8].mean()
theo_sd=np.sqrt(p*(1-p)/n_avg)
print(f"  Écart-type théorique (indépendance) = {theo_sd*100:.1f}%  -> {'SUR-dispersion (corrélation!)' if big.std()>theo_sd*1.3 else 'compatible avec indépendance (pas de corrélation exploitable)'}")
# corrélation total match i vs match i+1 dans le temps
s=df.total_goals.reset_index(drop=True)
print(f"  Corrélation total(match_t, match_t+1) = {s.autocorr(1):.3f}  (≈0 = indépendant)")
# fourchette : la spread |oh-oa| discrimine-t-elle le total ?
print("\n  Rôle de la fourchette |oh-oa| (spread) sur le total :")
for lbl,mask in [("serré spread<1",df.spread<1),("moyen 1-3",df.spread.between(1,3)),("large >3",df.spread>3)]:
    sub=df[mask]; print(f"    {lbl}: n={len(sub)} Over2.5={ (sub.total_goals>=3).mean()*100:.0f}% 0-0={(sub.exact_score=='0-0').mean()*100:.1f}%")
