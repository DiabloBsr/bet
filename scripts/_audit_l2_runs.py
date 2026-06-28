"""(F) RUNS CONDITIONNELS — la séquence d'upsets est-elle aléatoire DANS chaque régime
de cote ? (clustering/persistence conditionnel). Wald-Wolfowitz par bande p_fav,
par ligue, par slot. BH-FDR sur l'ensemble. Manches propres (==10)."""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings

e = create_engine(load_settings().db_url)
raw = pd.read_sql("""SELECT e.competition comp, e.expected_start, e.id ev,
  o.odds_home oh, o.odds_draw od, o.odds_away oa, r.score_a sa, r.score_b sb FROM events e
  JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
  JOIN results r ON r.event_id=e.id
  WHERE r.score_a IS NOT NULL AND e.competition LIKE 'InstantLeague-%'""", e)
raw = raw[(raw.oh > 1) & (raw.od > 1) & (raw.oa > 1)].copy()
raw["es"] = pd.to_datetime(raw.expected_start, utc=True, errors="coerce"); raw = raw.dropna(subset=["es"])
raw["rsize"] = raw.groupby(["comp", "es"]).ev.transform("size")
d = raw[raw.rsize == 10].sort_values(["comp", "es", "ev"]).reset_index(drop=True)
inv = 1/d.oh + 1/d.od + 1/d.oa
d["imp_home"] = (1/d.oh)/inv; d["imp_away"] = (1/d.oa)/inv
d["fav_home"] = d.imp_home > d.imp_away; d["p_fav"] = d[["imp_home", "imp_away"]].max(axis=1)
d["fav_won"] = np.where(d.fav_home, d.sa > d.sb, d.sb > d.sa).astype(int)
d["upset"] = 1 - d.fav_won
d["band"] = pd.cut(d.p_fav, [.40, .50, .60, .70, .80, 1.01], right=False).astype(str)
d["league"] = d.comp.str.replace("InstantLeague-", "", regex=False)
d["slot"] = d.groupby(["comp", "es"]).cumcount()

def runs_z(x):
    x = np.asarray(x); n1 = int(x.sum()); n0 = len(x) - n1
    if n1 < 10 or n0 < 10: return None, len(x)
    runs = 1 + int(np.sum(x[1:] != x[:-1]))
    mu = 2*n1*n0/len(x) + 1
    var = 2*n1*n0*(2*n1*n0 - len(x))/(len(x)**2*(len(x)-1))
    if var <= 0: return None, len(x)
    return (runs - mu)/math.sqrt(var), len(x)

tests = []
# par bande (séquence temporelle des upsets dans chaque bande)
for key, grp in d.groupby("band"):
    if key == "nan": continue
    z, nn = runs_z(grp.sort_values("es").upset.values)
    if z is not None and nn >= 300: tests.append((f"band={key}", nn, z))
# par bande × ligue
for (lg, bd), grp in d.groupby(["league", "band"]):
    if bd == "nan": continue
    z, nn = runs_z(grp.sort_values("es").upset.values)
    if z is not None and nn >= 300: tests.append((f"lg={lg}|band={bd}", nn, z))
# par slot
for sl, grp in d.groupby("slot"):
    z, nn = runs_z(grp.sort_values("es").upset.values)
    if z is not None and nn >= 300: tests.append((f"slot={sl}", nn, z))

res = pd.DataFrame(tests, columns=["cell", "n", "z"])
res["p"] = res.z.apply(lambda z: 2*(1 - 0.5*(1+math.erf(abs(z)/math.sqrt(2)))))
res = res.sort_values("p").reset_index(drop=True)
m = len(res); res["bh_thr"] = (res.index+1)/m*0.10; res["bh_pass"] = res.p <= res.bh_thr
print(f"runs conditionnels testés : {m} (n>=300 chacun)\n")
print(f"{'cellule':<26}{'n':>6}{'z_runs':>9}{'p':>9}{'FDR':>6}")
print("-"*56)
for r in res.head(15).itertuples():
    print(f"{r.cell:<26}{r.n:>6}{r.z:>+9.2f}{r.p:>9.3f}{'pass' if r.bh_pass else '-':>6}")
npass = res.bh_pass.sum()
print(f"\ncellules avec clustering significatif (FDR) : {npass}")
print("z>0 = anti-clustering (alterné) ; z<0 = clustering (séries) ; |z| tous <~2 = aléatoire")
print("VERDICT:", "AUCUN clustering conditionnel — séquences aléatoires dans chaque régime" if npass == 0
      else "clustering détecté — à re-tester OOS")
