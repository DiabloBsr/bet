"""Comparaison d'EFFICIENCE par ligue — chercher une ligue moins calibrée que
l'anglaise (8035) = exploitable. Pour chaque compétition avec assez de données :
 - le favori gagne-t-il PLUS que sa cote (EV>0 = edge à miser) ?
 - réel vs implicite sur BTTS / Over2.5 (écart = mispricing).
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings

LEAGUE_NAMES={"8035":"Anglaise","8065":"Coupe du monde","8056":"Champions","8060":"CAN",
              "8036":"Italienne","8037":"Espagnole","8042":"Française","8043":"Allemande","8044":"Portugaise"}
e=create_engine(load_settings().db_url)
df=pd.read_sql("""SELECT e.competition, o.odds_home oh, o.odds_draw od, o.odds_away oa,
  r.score_a sa, r.score_b sb FROM events e
  JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
  JOIN results r ON r.event_id=e.id WHERE r.score_a IS NOT NULL AND e.competition LIKE 'InstantLeague-%'""",e)
df=df[(df.oh>1)&(df.oa>1)&(df.od>1)].copy()
df["lid"]=df.competition.str.replace("InstantLeague-","",regex=False)
df=df[df.lid.str.isdigit()]
df["fav_home"]=df.oh<df.oa; df["fav_cote"]=df[["oh","oa"]].min(axis=1)
df["fav_won"]=np.where(df.fav_home,df.sa>df.sb,df.sb>df.sa)
inv=1/df.oh+1/df.od+1/df.oa
df["imp_fav"]=np.where(df.fav_home,(1/df.oh)/inv,(1/df.oa)/inv)
df["tot"]=df.sa+df.sb; df["btts"]=((df.sa>=1)&(df.sb>=1)).astype(int); df["o25"]=(df.tot>=3).astype(int)
df["overround"]=inv

def z(p,p0,n): return (p-p0)/math.sqrt(p0*(1-p0)/n) if (n>0 and 0<p0<1) else 0.0
print(f"{'ligue':<16}{'n':>6}{'marge':>7}{'favWIN':>8}{'implic':>8}{'EVfav':>8}{'z':>6}  {'BTTSr/impl':>12}{'O2.5r/impl':>12}")
print("-"*95)
rows=[]
for lid,g in df.groupby("lid"):
    if len(g)<400: continue
    nm=LEAGUE_NAMES.get(lid,lid)
    w=g.fav_won.mean(); imp=g.imp_fav.mean()
    # EV de miser le favori à la cote brute offerte
    evfav=(g.fav_won*g.fav_cote-1).mean()
    zz=z(w,imp,len(g))
    bt_r=g.btts.mean(); o_r=g.o25.mean()
    rows.append((nm,len(g),g.overround.mean(),w,imp,evfav,zz,bt_r,o_r))
    print(f"{nm:<16}{len(g):>6}{(g.overround.mean()-1)*100:>6.0f}%{w*100:>7.0f}%{imp*100:>7.0f}%{evfav*100:>+7.0f}%{zz:>+6.1f}  {bt_r*100:>5.0f}%{'':>7}{o_r*100:>5.0f}%")
print("\n-> EVfav > +3% avec z>2 = LIGUE EXPLOITABLE (le favori y est sous-coté).")
print("   Sinon : marché efficient comme l'anglaise (le favori gagne pile sa cote).")
best=sorted(rows,key=lambda x:-x[5])[:1]
if best:
    nm,n,_,w,imp,ev,zz,_,_=best[0]
    print(f"\nMeilleure piste : {nm} (EVfav {ev*100:+.0f}%, z={zz:+.1f}, n={n})")
