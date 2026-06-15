"""DEEP TOTAL=3 — quel edge est le mieux placé pour faire 3 buts ? Analyse au
millimètre : (A) classement des conditions par P(total=3) sur fold3 propre ;
(B) grille fine de cotes/λ ; (C) le CLASSEMENT (rankings_snapshots) ajoute-t-il
quoi que ce soit au-delà des cotes ?"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings

ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/"exports"/"combokeys_features.csv").sort_values("expected_start").reset_index(drop=True)
df["abs_diff"]=df.lam_diff.abs(); df["spread"]=(df.oh-df.oa).abs(); df["t3"]=(df.total_goals==3).astype(int)
n=len(df); f2=int(n*0.8); tr,te=df.iloc[:f2],df.iloc[f2:]
base=tr.t3.mean(); base_te=te.t3.mean()
def z(p,p0,n): return (p-p0)/math.sqrt(p0*(1-p0)/n) if (n>0 and 0<p0<1) else 0.0

print("="*96); print(f"(A) CONDITIONS qui maximisent P(total=3) — base TR {base*100:.1f}% / TE {base_te*100:.1f}%"); print("="*96)
cand=[]
for col in ["lam_tot","lam_diff","abs_diff","fav","od","p_btts","odds_ratio","spread","lam_h","lam_a"]:
    try: ed=pd.qcut(tr[col].dropna(),6,duplicates="drop").cat.categories
    except Exception: continue
    bins=[ed[0].left]+[c.right for c in ed]
    ctr=pd.cut(tr[col],bins,include_lowest=True); cte=pd.cut(te[col],bins,include_lowest=True)
    for b in ctr.cat.categories:
        mtr=(ctr==b);
        if mtr.sum()<150: continue
        p_tr=tr.t3[mtr].mean(); p_te=te.t3[(cte==b)].mean(); nte=(cte==b).sum()
        cand.append((f"{col}∈{b}",int(mtr.sum()),int(nte),p_tr,p_te,z(p_tr,base,int(mtr.sum()))))
cand.sort(key=lambda x:-x[4])
print(f"{'condition':<34}{'n tr/te':>11}{'P3 TR':>7}{'P3 TE':>7}{'z':>6}")
for c in cand[:10]:
    print(f"{c[0][:34]:<34}{c[1]:>5}/{c[2]:<5}{c[3]*100:>6.0f}%{c[4]*100:>6.0f}%{c[5]:>6.1f}")
print(f"\n-> max P(total=3) atteignable ≈ {max(c[4] for c in cand)*100:.0f}% (vs base {base_te*100:.0f}%). Plafond = total central inélastique.")

print("\n"+"="*96); print("(B) GRILLE FINE λ_tot (0.2) — où P(total=3) pique"); print("="*96)
for lo in np.arange(2.0,4.0,0.2):
    s=df[(df.lam_tot>=lo)&(df.lam_tot<lo+0.2)]
    if len(s)<80: continue
    print(f"  λ_tot [{lo:.1f},{lo+0.2:.1f}): n={len(s):>4}  P(3)={s.t3.mean()*100:>4.0f}%  (Over2.5 {(s.total_goals>=3).mean()*100:.0f}%)")

print("\n"+"="*96); print("(C) LE CLASSEMENT ajoute-t-il qqch ? (rankings_snapshots)"); print("="*96)
e=create_engine(load_settings().db_url)
rk=pd.read_sql("SELECT team_name,position,points,captured_at FROM rankings_snapshots WHERE competition='InstantLeague-8035'",e)
rk["captured_at"]=pd.to_datetime(rk.captured_at,utc=True,errors="coerce"); rk=rk.dropna(subset=["captured_at"]).sort_values("captured_at")
ev=pd.read_sql("SELECT e.team_a,e.team_b,e.expected_start,r.score_a,r.score_b FROM events e JOIN results r ON r.event_id=e.id WHERE r.score_a IS NOT NULL AND e.competition='InstantLeague-8035'",e)
ev["expected_start"]=pd.to_datetime(ev.expected_start,utc=True); ev=ev.dropna(subset=["expected_start"]).sort_values("expected_start")
ev["t3"]=((ev.score_a+ev.score_b)==3).astype(int)
def pos_for(team,t):
    s=rk[(rk.team_name==team)&(rk.captured_at<=t)]
    return s.position.iloc[-1] if len(s) else np.nan
# échantillon pour vitesse
samp=ev.sample(min(4000,len(ev)),random_state=1).copy() if len(ev)>4000 else ev.copy()
samp["hp"]=samp.apply(lambda r:pos_for(r.team_a,r.expected_start),axis=1)
samp["ap"]=samp.apply(lambda r:pos_for(r.team_b,r.expected_start),axis=1)
samp=samp.dropna(subset=["hp","ap"]); samp["posgap"]=(samp.ap-samp.hp).abs()
print(f"  events avec classement: {len(samp)}")
print("  P(total=3) par écart de position |home-away| :")
for lo,hi in [(0,3),(3,6),(6,10),(10,20)]:
    s=samp[(samp.posgap>=lo)&(samp.posgap<hi)]
    if len(s)>50: print(f"    écart [{lo},{hi}): n={len(s):>4}  P(3)={s.t3.mean()*100:.0f}%")
corr=samp[["posgap","t3"]].corr().iloc[0,1]
print(f"  corrélation |écart position| vs total=3 : {corr:+.3f}  (≈0 = le classement n'apporte RIEN sur le total=3)")
# top de classement vs bas
print(f"  P(3) quand home est top-5 : {samp[samp.hp<=5].t3.mean()*100:.0f}% | home bottom-5 (>=16) : {samp[samp.hp>=16].t3.mean()*100:.0f}% | global {samp.t3.mean()*100:.0f}%")
