"""LAG-10 : signal stationnaire & exploitable, ou artefact/non-stationnaire ?
Décisif :
 (1) l'autocorr du résidu à lag10 est-elle STABLE train(70%) vs test(30%) ?
     -> si forte en train et ~0 en test : non-stationnaire = inexploitable (cohérent avec OOS).
 (2) les manches sont-elles propres (exactement ~10) ? une manche 'fusionnée' (2 timestamps
     collés) gonflerait faussement la sur-dispersion ET créerait un faux lag-10.
 (3) sur-dispersion recalculée sur manches n==10 strictes uniquement.
 (4) le lag réel = la taille de manche ? scan autocorr résidu lag 8..12.
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings

e = create_engine(load_settings().db_url)
g = pd.read_sql("""SELECT e.expected_start, e.id ev, o.odds_home oh, o.odds_draw od, o.odds_away oa,
  r.score_a sa, r.score_b sb FROM events e
  JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
  JOIN results r ON r.event_id=e.id
  WHERE r.score_a IS NOT NULL AND e.competition='InstantLeague-8035'""", e)
g = g[(g.oh > 1) & (g.od > 1) & (g.oa > 1)].copy()
g["es"] = pd.to_datetime(g.expected_start, utc=True, errors="coerce")
g = g.dropna(subset=["es"]).sort_values(["es", "ev"]).reset_index(drop=True)
g["tot"] = g.sa + g.sb; g["hw"] = (g.sa > g.sb).astype(int)
g["btts"] = ((g.sa >= 1) & (g.sb >= 1)).astype(int)
inv = 1/g.oh + 1/g.od + 1/g.oa; g["imp_home"] = (1/g.oh)/inv

def ac(s, lag=10):
    s = np.asarray(s, float); s = s - s.mean()
    a, b = s[:-lag], s[lag:]
    if len(a) < 80: return 0.0, 0.0, 0
    c = np.sum(a*b)/math.sqrt(np.sum(a*a)*np.sum(b*b)); return c, c*math.sqrt(len(a)), len(a)

print("="*80)
print("(2) PROPRETÉ DES MANCHES (taille par expected_start)")
print("="*80)
sz = g.groupby("es").size()
print(f"  nb manches = {len(sz)} | taille: min={sz.min()} médiane={int(sz.median())} max={sz.max()}")
vc = sz.value_counts().sort_index()
print("  distribution des tailles :", {int(k): int(v) for k, v in vc.items()})
clean_es = sz[sz == 10].index
gc = g[g.es.isin(clean_es)].reset_index(drop=True)
print(f"  matchs dans des manches PROPRES (==10) : {len(gc)} / {len(g)}")

print("\n" + "="*80)
print("(1) STABILITÉ train/test de l'autocorr résidu lag10 (hw - imp_home)")
print("="*80)
g["res"] = g.hw - g.imp_home
cut = int(len(g)*0.7)
for lbl, sub in [("FULL", g), ("TRAIN(70%)", g.iloc[:cut]), ("TEST(30%)", g.iloc[cut:])]:
    c, z, n = ac(sub.res.values)
    print(f"  {lbl:<12} corr={c:+.4f}  z={z:+.1f}  (n={n})  {'stable' if abs(z)>3 else 'DISPARU'}")
print("  -> si TEST 'DISPARU' alors que FULL/TRAIN fort : non-stationnaire = inexploitable.")

print("\n" + "="*80)
print("(4) Le lag est-il EXACTEMENT 10 (= taille manche) ? scan résidu lag 8..12")
print("="*80)
for lag in range(8, 13):
    c, z, n = ac(g.res.values, lag)
    print(f"  lag {lag:>2}: corr={c:+.4f} z={z:+.1f}  {'<= pic' if lag==10 else ''}")

print("\n" + "="*80)
print("(3) SUR-DISPERSION home-wins recalculée — toutes manches vs manches PROPRES (==10)")
print("="*80)
for lbl, sub in [("toutes (n>=8)", g), ("propres (==10)", gc)]:
    rk = sub.groupby("es").agg(n=("hw", "size"), hw=("hw", "sum"),
                               pb=("imp_home", lambda s: float(np.sum(s*(1-s))))).query("n>=8")
    ratio = rk.hw.var()/rk.pb.mean()
    print(f"  {lbl:<16} var_obs={rk.hw.var():.3f}  var_PoissonBinom={rk.pb.mean():.3f}  ratio={ratio:.3f}")

# et l'autocorr lag10 sur manches propres seulement (réindexées)
gc["res"] = gc.hw - gc.imp_home
c, z, n = ac(gc.res.values)
print(f"\n  autocorr résidu lag10 sur manches PROPRES uniquement : corr={c:+.4f} z={z:+.1f}")
print("\nCONCLUSION :")
print("  - sur-dispersion qui s'effondre sur manches propres  => artefact de groupage.")
print("  - autocorr qui disparaît en TEST                      => non-stationnaire, inexploitable.")
print("  - si les deux tiennent ET lag pique à 10              => vrai micro-signal RNG (mais OOS nul).")
