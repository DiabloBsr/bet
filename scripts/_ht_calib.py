"""Microscope LIVE -> calibration de la structure mi-temps du moteur (etude only,
prediction bettable pre-match).

Mesure depuis les resultats (8035) :
  1. Split H1/H2 : quelle fraction des buts tombe en 1ere mi-temps ? globale + par
     bande de lam_diff (drama mode : matchs serres -> + de buts H2 -> fraction H1 plus basse).
  2. Par cote : la fraction H1 par equipe (ht_h / lam_h) -> facteur f_h, f_a.
  3. HT 1X2 : empirique vs modele Poisson(lam*f) -> y a-t-il de l'ecart a exploiter ?
  4. Plafond score HT (Top1/Top3).
  5. Comebacks (etude) : si HT leader = home, taux de non-victoire FT.

Lit combokeys_features.csv (lam deja calcules) + results (ht_score). 8035.
Sortie console + exports/ht_calib.json
"""
import sys
sys.path.insert(0, ".")
import json
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.analysis_utils import load_corrupted_ids

feat = pd.read_csv("exports/combokeys_features.csv")[["id", "lam_h", "lam_a", "lam_tot", "lam_diff", "exact_score", "total_goals"]]
e = create_engine(load_settings().db_url)
corr = load_corrupted_ids()
ht = pd.read_sql("""
    SELECT e.id, r.ht_score_a, r.ht_score_b, r.score_a, r.score_b
    FROM events e JOIN results r ON r.event_id=e.id
    WHERE r.ht_score_a IS NOT NULL AND e.competition='InstantLeague-8035'
""", e)
df = feat.merge(ht, on="id", how="inner")
df = df[~df.id.isin(corr)].copy()
df["ht_h"] = df.ht_score_a.astype(int); df["ht_a"] = df.ht_score_b.astype(int)
df["ft_h"] = df.score_a.astype(int); df["ft_a"] = df.score_b.astype(int)
df = df[(df.ht_h <= df.ft_h) & (df.ht_a <= df.ft_a)]  # garde-fou HT<=FT
df["ht_tot"] = df.ht_h + df.ht_a; df["ft_tot"] = df.ft_h + df.ft_a
print(f"n={len(df)}")

out = {}
# 1. split global
h1_frac = df.ht_tot.sum() / df.ft_tot.sum()
out["h1_fraction_global"] = round(float(h1_frac), 4)
print(f"\n=== SPLIT H1/H2 ===")
print(f"fraction des buts en 1ere MT : {100*h1_frac:.1f}%  (H2 = {100*(1-h1_frac):.1f}%)")

# 2. par bande de |lam_diff| (drama : serres -> H1 plus basse)
print(f"\n=== fraction H1 par |lam_diff| (drama mode) ===")
df["absdiff"] = df.lam_diff.abs()
bands = [(0, 0.3, "serre"), (0.3, 0.7, "leger"), (0.7, 1.2, "marque"), (1.2, 9, "ecrase")]
out["h1_fraction_by_balance"] = {}
for lo, hi, lbl in bands:
    g = df[(df.absdiff >= lo) & (df.absdiff < hi)]
    if len(g) < 100: continue
    f = g.ht_tot.sum() / g.ft_tot.sum()
    out["h1_fraction_by_balance"][lbl] = round(float(f), 4)
    print(f"  {lbl:<8} |diff|∈[{lo},{hi}) n={len(g):>4}  H1={100*f:.1f}%  buts/match H1={g.ht_tot.mean():.2f} FT={g.ft_tot.mean():.2f}")

# 3. facteur par equipe : f_h = mean(ht_h)/mean(lam_h)
f_h = df.ht_h.mean() / df.lam_h.mean()
f_a = df.ht_a.mean() / df.lam_a.mean()
out["f_home"] = round(float(f_h), 4); out["f_away"] = round(float(f_a), 4)
print(f"\nfacteur HT par cote : f_home={f_h:.3f} (ht_h moy {df.ht_h.mean():.3f} vs lam_h {df.lam_h.mean():.3f})  f_away={f_a:.3f}")

# 4. HT 1X2 empirique vs modele Poisson(lam*f)
def ht_model_1x2(r, fh, fa, maxg=7):
    lh, la = r.lam_h * fh, r.lam_a * fa
    ph = sum(poisson.pmf(h, lh) * poisson.pmf(a, la) for h in range(maxg) for a in range(maxg) if h > a)
    pd_ = sum(poisson.pmf(h, lh) * poisson.pmf(a, la) for h in range(maxg) for a in range(maxg) if h == a)
    return ph, pd_, 1 - ph - pd_
emp_h = (df.ht_h > df.ht_a).mean(); emp_d = (df.ht_h == df.ht_a).mean(); emp_a = (df.ht_h < df.ht_a).mean()
# modele moyen sur un echantillon
samp = df.sample(min(2000, len(df)), random_state=1)
mod = np.array([ht_model_1x2(r, f_h, f_a) for r in samp.itertuples()])
print(f"\n=== HT 1X2 : empirique vs modele Poisson(lam*f) ===")
print(f"  empirique : 1={100*emp_h:.1f}% X={100*emp_d:.1f}% 2={100*emp_a:.1f}%")
print(f"  modele    : 1={100*mod[:,0].mean():.1f}% X={100*mod[:,1].mean():.1f}% 2={100*mod[:,2].mean():.1f}%")
out["ht_1x2_emp"] = [round(emp_h,4), round(emp_d,4), round(emp_a,4)]
out["ht_1x2_model"] = [round(float(mod[:,0].mean()),4), round(float(mod[:,1].mean()),4), round(float(mod[:,2].mean()),4)]

# 5. plafond score HT (Top1/Top3 par bucket fav)
df["ht_score"] = df.ht_h.astype(str) + "-" + df.ht_a.astype(str)
df["fav"] = df[["lam_h","lam_a"]].min(axis=1)  # proxy
vc = df.ht_score.value_counts(normalize=True)
out["ht_score_top1"] = round(float(vc.iloc[0]),4); out["ht_score_top3"] = round(float(vc.iloc[:3].sum()),4)
print(f"\n=== PLAFOND score HT ===")
print(f"  Top1 {100*vc.iloc[0]:.1f}% ({vc.index[0]}) · Top3 {100*vc.iloc[:3].sum():.1f}% · scores: {' '.join(vc.index[:5])}")

# 6. comebacks (etude) : HT leader home -> FT
home_lead = df[df.ht_h > df.ht_a]
if len(home_lead):
    ft_not_home = (home_lead.ft_h <= home_lead.ft_a).mean()
    print(f"\n=== COMEBACKS (etude) ===")
    print(f"  HT home mene (n={len(home_lead)}) -> FT pas victoire home : {100*ft_not_home:.1f}%")
    out["ht_home_lead_blown"] = round(float(ft_not_home),4)
away_lead = df[df.ht_a > df.ht_h]
if len(away_lead):
    ft_not_away = (away_lead.ft_a <= away_lead.ft_h).mean()
    print(f"  HT away mene (n={len(away_lead)}) -> FT pas victoire away : {100*ft_not_away:.1f}%")
    out["ht_away_lead_blown"] = round(float(ft_not_away),4)

with open("exports/ht_calib.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print("\necrit exports/ht_calib.json")
