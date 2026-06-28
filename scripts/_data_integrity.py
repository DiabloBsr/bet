"""AUDIT INTÉGRITÉ DONNÉES (bloc 3) + sensibilité.
Ordre chrono, ex-aequo timestamps, ordre intra-manche, doublons, trous, contamination,
parsing cotes (overround), cohérence résultats, cohérence inter-marchés.
Sensibilité : le résidu marginal [0.40,0.45) survit-il aux données impures / perturbations ?
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.market_inversion import parse_extra_markets, total_buts_odds, devig_market

e = create_engine(load_settings().db_url)
a = pd.read_sql("""SELECT e.competition comp, e.expected_start, e.id ev,
  o.odds_home oh, o.odds_draw od, o.odds_away oa, o.extra_markets em,
  r.score_a sa, r.score_b sb FROM events e
  JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
  JOIN results r ON r.event_id=e.id
  WHERE e.competition LIKE 'InstantLeague-%'""", e)
print(f"lignes brutes (avec résultat) : {len(a)}")
a["es"] = pd.to_datetime(a.expected_start, utc=True, errors="coerce")

print("\n" + "="*80); print("INTÉGRITÉ"); print("="*80)
# 1 timestamps
print(f"  expected_start NULL          : {a.es.isna().sum()}")
print(f"  scores NULL                  : {a.sa.isna().sum()} / {a.sb.isna().sum()}")
a = a.dropna(subset=["es", "sa", "sb"])
# 2 doublons stricts (même event)
dup_ev = a.ev.duplicated().sum()
dup_match = a.duplicated(["comp", "es", "ev"]).sum()
print(f"  ev dupliqués                 : {dup_ev}")
print(f"  (comp,es,ev) dupliqués       : {dup_match}")
# 3 ordre intra-manche : ev unique & croissant dans chaque manche
g = a.groupby(["comp", "es"])
ev_dup_in_round = (g.ev.transform("nunique") != g.ev.transform("size")).sum()
print(f"  ev non-uniques dans une manche : {ev_dup_in_round}")
# 4 tailles de manche
sz = g.ev.transform("size")
vc = pd.Series(a.assign(sz=sz).groupby(["comp", "es"]).sz.first()).value_counts().sort_index()
print(f"  tailles de manche (distribution) : {dict((int(k), int(v)) for k, v in vc.items())}")
# 5 contamination : 2 manches de la même ligue à < 30s d'écart
print("  contamination (manches < 60s d'écart, même ligue) :")
for lg, gg in a.groupby("comp"):
    times = np.sort(gg.es.drop_duplicates().values.astype("datetime64[s]").astype(np.int64))
    if len(times) > 1:
        d2 = np.diff(times); near = int((d2 < 60).sum())
        print(f"    {lg.replace('InstantLeague-','')}: {near} paires de manches à <60s (sur {len(times)} manches)")
# 6 parsing cotes : overround 1X2
a["over_round"] = 1/a.oh + 1/a.od + 1/a.oa
val = a[(a.oh > 1) & (a.od > 1) & (a.oa > 1)]
print(f"  overround 1X2 : médiane={val.over_round.median():.3f} p1={val.over_round.quantile(.01):.3f} p99={val.over_round.quantile(.99):.3f}")
print(f"    cotes <=1 (placeholder)      : {(a.oh<=1).sum()+(a.od<=1).sum()+(a.oa<=1).sum()}")
print(f"    overround < 1.0 (arbitrage?) : {(val.over_round<1.0).sum()}")
# 7 cohérence résultats
print(f"  scores négatifs              : {((a.sa<0)|(a.sb<0)).sum()}")
print(f"  scores > 12                  : {((a.sa>12)|(a.sb>12)).sum()}")
# 8 cohérence inter-marchés : favori = cote la plus basse
v = val.copy()
v["fav_home_odds"] = v.oh < v.oa
inv = 1/v.oh + 1/v.od + 1/v.oa
v["fav_home_imp"] = (1/v.oh)/inv > (1/v.oa)/inv
print(f"  cohérence favori (cote min == proba max) : {(v.fav_home_odds==v.fav_home_imp).mean()*100:.1f}%")
# total ladder devig sum
ok_tot = 0; tot_n = 0; ovr = []
for r in v.head(8000).itertuples():
    em = parse_extra_markets(r.em); tb = total_buts_odds(em)
    if len(tb) >= 6:
        tot_n += 1; s = sum(1/x for x in tb.values()); ovr.append(s)
        if abs(s - 1) < 0.3: ok_tot += 1
if tot_n:
    print(f"  marché Total de buts : overround médian={np.median(ovr):.3f} (n={tot_n} parsés sur 8000)")

print("\n" + "="*80); print("SENSIBILITÉ — le résidu marginal [0.40,0.45) est-il robuste ?"); print("="*80)
v = v.sort_values(["es", "comp", "ev"]).reset_index(drop=True)
v["rsize"] = v.groupby(["comp", "es"]).ev.transform("size")
v["imp_home"] = (1/v.oh)/inv.loc[v.index] if False else (1/v.oh)/(1/v.oh+1/v.od+1/v.oa)
v["imp_away"] = (1/v.oa)/(1/v.oh+1/v.od+1/v.oa)
v["p_fav"] = v[["imp_home", "imp_away"]].max(axis=1)
v["fav_home"] = v.imp_home > v.imp_away
v["fav_won"] = np.where(v.fav_home, v.sa > v.sb, v.sb > v.sa).astype(float)
def band_resid(frame, lo=0.40, hi=0.45):
    s = frame[(frame.p_fav >= lo) & (frame.p_fav < hi)]
    if len(s) < 100: return None
    rr = (s.fav_won.mean() - s.p_fav.mean())*100
    se = (s.fav_won - s.p_fav).std()/math.sqrt(len(s))*100
    return len(s), rr, se
for lbl, fr in [("manches PROPRES (==10)", v[v.rsize == 10]),
                ("TOUTES manches", v),
                ("manches IMPURES (!=10)", v[v.rsize != 10])]:
    res = band_resid(fr)
    if res:
        nn, rr, se = res
        print(f"  {lbl:<26} n={nn:>5} résidu={rr:+.2f}pp z={rr/se:+.2f} IC95=[{rr-1.96*se:+.2f},{rr+1.96*se:+.2f}]")
# perturbation d'ordre : sans effet sur une calibration (test de cohérence)
print("\n  (la calibration ne dépend pas de l'ordre -> robuste par construction ;")
print("   les tests sériels, eux, ont déjà été montrés non-stationnaires.)")
