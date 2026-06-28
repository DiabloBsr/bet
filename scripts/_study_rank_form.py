"""Étude multi-input : CLASSEMENT + FORME + COTES sur le résultat/buts/BTTS/score.
(A) les équipes haut classées gagnent à combien % ? le classement bat-il les cotes ?
(B) 2 équipes qui n'ont PAS gagné leur dernier match -> BTTS/total/score particulier ?
(C) combos multi-input (même improbables) -> une sortie concentrée ?
Join : pour chaque match, dernier snapshot de classement AVANT le coup d'envoi (merge_asof).
"""
from __future__ import annotations
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings

e=create_engine(load_settings().db_url)
ev=pd.read_sql("""SELECT e.team_a,e.team_b,e.expected_start,o.odds_home oh,o.odds_draw od,o.odds_away oa,
  r.score_a sa,r.score_b sb FROM events e
  JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
  JOIN results r ON r.event_id=e.id WHERE r.score_a IS NOT NULL AND e.competition='InstantLeague-8035'""",e)
ev["es"]=pd.to_datetime(ev.expected_start,utc=True,errors="coerce"); ev=ev.dropna(subset=["es"]).sort_values("es").reset_index(drop=True)
ev=ev[(ev.oh>1)&(ev.oa>1)&(ev.od>1)]
rk=pd.read_sql("SELECT team_name,position,history,captured_at FROM rankings_snapshots WHERE competition='InstantLeague-8035'",e)
rk["captured_at"]=pd.to_datetime(rk.captured_at,utc=True,errors="coerce"); rk=rk.dropna(subset=["captured_at"]).sort_values("captured_at")
def last_res(h):
    try: return json.loads(h)[0] if h else None
    except Exception: return None
rk["last"]=rk.history.apply(last_res)

def attach(side_team):
    m=pd.merge_asof(ev.sort_values("es"), rk.rename(columns={"team_name":side_team}).sort_values("captured_at"),
                    left_on="es", right_on="captured_at", left_by=side_team, right_by=side_team, direction="backward")
    return m["position"].values, m["last"].values
ev["hp"],ev["hlast"]=attach("team_a"); ev["ap"],ev["alast"]=attach("team_b")
ev=ev.dropna(subset=["hp","ap"]).copy()
ev["home_win"]=(ev.sa>ev.sb).astype(int); ev["tot"]=ev.sa+ev.sb
ev["btts"]=((ev.sa>=1)&(ev.sb>=1)).astype(int); ev["sc"]=ev.sa.astype(int).astype(str)+"-"+ev.sb.astype(int).astype(str)
ev["imp_h"]=(1/ev.oh)/((1/ev.oh)+(1/ev.od)+(1/ev.oa))
ev["posgap"]=ev.ap-ev.hp  # >0 = home mieux classé
print(f"matchs avec classement+forme : {len(ev)}\n")

print("="*82); print("(A) ÉQUIPES HAUT CLASSÉES — % de victoire + le classement bat-il les cotes ?"); print("="*82)
print("  Win-rate HOME par position du home :")
for lo,hi in [(1,4),(4,8),(8,12),(12,17),(17,21)]:
    s=ev[(ev.hp>=lo)&(ev.hp<hi)]
    if len(s)>50: print(f"    home classé [{lo}-{hi}) : n={len(s):>4}  win={s.home_win.mean()*100:.0f}%  cote home moy {s.oh.mean():.2f}")
# le classement ajoute-t-il au-delà des cotes ? résidu (réel - implicite) vs posgap
ev["resid"]=ev.home_win-ev.imp_h
c=ev[["posgap","resid"]].corr().iloc[0,1]
print(f"\n  Corrélation (écart de classement) vs (résidu réel-implicite) = {c:+.3f}")
print("    -> ≈0 = le classement n'ajoute RIEN aux cotes (déjà encodé). >0 = le classement bat la cote.")
# contrôle pur : dans une bande de cote serrée, le classement change-t-il le win ?
band=ev[(ev.oh>=1.7)&(ev.oh<2.0)]
print(f"  Contrôle (cote home 1.7-2.0, n={len(band)}) : home top-6 win {band[band.hp<=6].home_win.mean()*100:.0f}% vs home 13+ win {band[band.hp>=13].home_win.mean()*100:.0f}% (implicite ~{band.imp_h.mean()*100:.0f}%)")

print("\n"+"="*82); print("(B) LES 2 ÉQUIPES N'ONT PAS GAGNÉ LEUR DERNIER MATCH -> sortie particulière ?"); print("="*82)
noW=ev[(ev.hlast.isin(["Lost","Draw"]))&(ev.alast.isin(["Lost","Draw"]))]
print(f"  Sous-ensemble 'aucun n'a gagné son dernier' : n={len(noW)} ({len(noW)/len(ev)*100:.0f}% des matchs)")
print(f"    BTTS  : {noW.btts.mean()*100:.0f}%  (base {ev.btts.mean()*100:.0f}%)")
print(f"    total moyen : {noW.tot.mean():.2f}  (base {ev.tot.mean():.2f})")
print(f"    Over2.5 : {(noW.tot>=3).mean()*100:.0f}%  (base {(ev.tot>=3).mean()*100:.0f}%)")
vc=noW.sc.value_counts(normalize=True).head(4)
top_str=" ".join(f"{k}({v*100:.0f}%)" for k,v in vc.items()); base11=ev.sc.eq("1-1").mean()*100
print(f"    Top scores : {top_str}  (vs 1-1 base {base11:.0f}%)")
print(f"    total exact modal : {noW.tot.value_counts().idxmax()} buts ({(noW.tot==noW.tot.value_counts().idxmax()).mean()*100:.0f}%)")
# variantes de forme
for lbl,mask in [("les 2 ont PERDU (Lost+Lost)", (ev.hlast=='Lost')&(ev.alast=='Lost')),
                 ("les 2 ont GAGNÉ (Won+Won)", (ev.hlast=='Won')&(ev.alast=='Won')),
                 ("home a perdu / away a gagné", (ev.hlast=='Lost')&(ev.alast=='Won'))]:
    s=ev[mask]
    if len(s)>50: print(f"    [{lbl}] n={len(s):>4} BTTS {s.btts.mean()*100:.0f}% | total moy {s.tot.mean():.2f} | 1-1 {s.sc.eq('1-1').mean()*100:.0f}%")

print("\n"+"="*82); print("(C) MULTI-INPUT — combos (même improbables) cherchant une sortie concentrée"); print("="*82)
ev["fh"]=ev.hlast.map({"Won":"W","Lost":"L","Draw":"D"}); ev["fa"]=ev.alast.map({"Won":"W","Lost":"L","Draw":"D"})
ev["ohb"]=pd.cut(ev.oh,[0,1.6,2.2,3.2,99]).astype(str)
best=[]
for keys in [["fh","fa"],["fh","fa","ohb"],["posgap"],["fh","fa","posgap"]]:
    if "posgap" in keys: ev["_pg"]=pd.cut(ev.posgap,[-99,-5,0,5,99]).astype(str); g=ev.groupby([k if k!="posgap" else "_pg" for k in keys],observed=True)
    else: g=ev.groupby(keys,observed=True)
    for name,grp in g:
        if len(grp)<80: continue
        # meilleure concentration BTTS / total modal / score modal
        for metric,val in [("BTTS",grp.btts.mean()),("Under2.5",(grp.tot<=2).mean()),("Over2.5",(grp.tot>=3).mean())]:
            best.append((f"{keys}={name}",metric,val,len(grp)))
print("  Combos avec la sortie la + concentrée (toutes ~ proches de la base = pas d'edge) :")
for lbl,m,v,n in sorted(best,key=lambda x:-x[2])[:8]:
    print(f"    {lbl[:46]:<46} {m}={v*100:.0f}% (n={n})")
print("\n-> Si tous ces 'meilleurs' combos restent proches des bases (BTTS~58%, Over~63%), c'est que")
print("   classement+forme n'ajoutent rien : le RNG est sans mémoire et piloté par les cotes.")
