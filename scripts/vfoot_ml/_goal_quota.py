"""QUOTA DE BUTS ? — le RNG distribue-t-il un total FIXE de buts par round (ou saison) ?

Si quota : le total de buts d'un round (~10 matchs) serait ~constant, et les matchs
d'un même round seraient NÉGATIVEMENT corrélés (un match plein -> les autres pauvres).
Si indépendant : variance du total du round = 10 × variance d'un match, corrélation ~0.

Tests décisifs :
  1. variance observée du total/round  vs  variance attendue sous indépendance.
  2. corrélation (buts d'un match, buts des AUTRES matchs du même round).
  3. idem au niveau SAISON (somme sur ~380 matchs).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

LG = "InstantLeague-8035"
eng = create_engine(load_settings().db_url, connect_args={"timeout": 30})
df = pd.read_sql(text(f"""
    SELECT ev.expected_start ts, ev.team_a ta, ev.team_b tb, ev.round_info j,
           r.score_a sa, r.score_b sb
    FROM events ev JOIN results r ON r.event_id=ev.id
    WHERE r.score_a IS NOT NULL AND ev.competition='{LG}'"""), eng)
df = df.drop_duplicates(["ts", "ta", "tb"])       # dédup par IDENTITÉ de match (pas par score !)
df["tot"] = df.sa + df.sb
var_match = df.tot.var()
print(f"{len(df)} matchs | total moyen/match {df.tot.mean():.3f} | variance/match {var_match:.3f}", flush=True)

# ---- 1. total de buts par ROUND (groupé par timestamp = coup d'envoi simultané) ----
g = df.groupby("ts")["tot"].agg(["sum", "count"])
r10 = g[g["count"] == 10]                    # rounds complets de 10 matchs (comparaison propre)
print(f"\n=== ROUND (10 matchs) : {len(r10)} rounds complets ===")
obs_var = r10["sum"].var()
exp_var = 10 * var_match                       # variance attendue si 10 matchs INDÉPENDANTS
print(f"  total buts/round : moyenne {r10['sum'].mean():.2f} | min {int(r10['sum'].min())} "
      f"| max {int(r10['sum'].max())}")
print(f"  variance OBSERVÉE {obs_var:.2f}  vs  ATTENDUE sous indépendance {exp_var:.2f}")
print(f"  ratio {obs_var/exp_var:.3f}  -> {'QUOTA (variance écrasée !)' if obs_var/exp_var < 0.6 else 'INDÉPENDANT (pas de quota)'}")

# ---- 2. corrélation match <-> autres matchs du même round ----
d10 = df[df.ts.isin(r10.index)].copy()
d10 = d10.merge(r10["sum"].rename("round_sum"), left_on="ts", right_index=True)
d10["others"] = d10.round_sum - d10.tot
corr = np.corrcoef(d10.tot, d10.others)[0, 1]
print(f"  corrélation (buts du match, buts des 9 autres) : {corr:+.4f}  "
      f"-> {'NÉGATIF = quota ?' if corr < -0.05 else 'nul = indépendant'}")
# réplication train/test : un vrai effet doit tenir sur les 2 moitiés
ts_sorted = sorted(r10.index)
cut = ts_sorted[len(ts_sorted)//2]
for lbl, sub in (("1re moitié", d10[d10.ts < cut]), ("2e moitié", d10[d10.ts >= cut])):
    if len(sub) > 100:
        c = np.corrcoef(sub.tot, sub.others)[0, 1]
        print(f"     {lbl} : corr {c:+.4f}")
# permutation : rebâtir des rounds ALÉATOIRES (mêmes matchs mélangés) -> corr de référence
rng = np.random.default_rng(0)
perm = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
perm["fakeround"] = perm.index // 10
pg = perm.groupby("fakeround")["tot"].agg(["sum", "count"])
pg = pg[pg["count"] == 10]
pm = perm[perm.fakeround.isin(pg.index)].merge(pg["sum"].rename("rs"), left_on="fakeround", right_index=True)
pm["oth"] = pm.rs - pm.tot
print(f"  [contrôle] rounds ALÉATOIRES (indépendance forcée) : corr {np.corrcoef(pm.tot, pm.oth)[0,1]:+.4f} "
      f"(≈0 attendu ; sert de référence de bruit)")

# ---- 3. niveau SAISON (somme par journée->saison approx : somme par round_info sur toute la période) ----
print(f"\n=== distribution du total/round (histogramme) ===")
h = r10["sum"].value_counts(normalize=True).sort_index()
# regroupe en tranches pour lisibilité
bins = pd.cut(r10["sum"], bins=[0, 20, 24, 27, 30, 33, 36, 100])
print((r10.groupby(bins, observed=True)["sum"].count()/len(r10)*100).round(1).to_string())
print(f"\n  Si c'était un quota fixe, tout serait sur UNE valeur. Étalement large = pas de quota.")
