"""PRÉDICTION SÉLECTIVE — filtrer par CONFIANCE lève-t-il le taux de réussite ?

Idée utilisateur : au lieu de prédire tous les matchs (Top-3 ~31%), n'agir que sur
ceux où la prédiction est la plus CONCENTRÉE. Question : le taux de réussite sur ce
sous-ensemble monte-t-il (et jusqu'où) ?

Confiance d'un match = masse de proba des Top-3 scores (grille Score-exact dévigée =
championne du tournoi). On bucket les matchs par confiance et on mesure le VRAI
taux de réussite Top-1 / Top-3 dans chaque bucket + combien de matchs qualifient.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings
from scraper.predictor_v2 import market_score_grid, grid_top_k_scores

LG = "InstantLeague-8035"
eng = create_engine(load_settings().db_url)
df = pd.read_sql(text(f"""
    SELECT o.extra_markets xm, r.score_a sa, r.score_b sb
    FROM events e JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id
    WHERE r.score_a IS NOT NULL AND e.competition='{LG}'"""), eng)
print(f"{len(df)} matchs")

conf, hit1, hit3, top1p = [], [], [], []
for r in df.itertuples():
    try:
        xm = json.loads(r.xm) if isinstance(r.xm, str) else (r.xm or {})
        sem = xm.get("Score exact")
        g = market_score_grid(sem)
        if g is None:
            continue
        top = grid_top_k_scores(g, 3)
    except Exception:
        continue
    if not top:
        continue
    actual = f"{min(int(r.sa),6)}-{min(int(r.sb),6)}"
    scores = [s for s, _ in top]
    mass3 = float(sum(p for _, p in top))
    conf.append(mass3)
    top1p.append(float(top[0][1]))
    hit1.append(int(scores[0] == actual))
    hit3.append(int(actual in scores))

conf = np.array(conf); hit1 = np.array(hit1); hit3 = np.array(hit3); top1p = np.array(top1p)
n = len(conf)
print(f"{n} matchs avec grille Score-exact\n")
print(f"BASE (tous les matchs) : Top-1 {100*hit1.mean():.1f}%  Top-3 {100*hit3.mean():.1f}%\n")

print("FILTRE PAR CONFIANCE (masse Top-3) — on ne garde que les X% plus concentrés :")
print(f"  {'garde top':<12}{'seuil masse':>12}{'n matchs':>10}{'Top-1':>9}{'Top-3':>9}")
for frac in (1.0, 0.5, 0.25, 0.10, 0.05, 0.02):
    thr = np.quantile(conf, 1 - frac)
    m = conf >= thr
    if m.sum() < 20:
        continue
    print(f"  {100*frac:>4.0f}%      {thr:>11.3f}{int(m.sum()):>10}"
          f"{100*hit1[m].mean():>8.1f}%{100*hit3[m].mean():>8.1f}%")

# calibration : le taux réel colle-t-il à la masse annoncée ? (honnêteté du chiffre)
print("\nCALIBRATION (le Top-3 réel = la masse Top-3 annoncée ?) :")
for lo, hi in ((0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.65), (0.65, 1.0)):
    m = (conf >= lo) & (conf < hi)
    if m.sum() >= 30:
        print(f"  masse {lo:.2f}-{hi:.2f} : annoncé ~{100*conf[m].mean():.0f}%  "
              f"réel {100*hit3[m].mean():.0f}%  (n={int(m.sum())})")

best = conf >= np.quantile(conf, 0.98)
print(f"\n>>> Sur le TOP 2% le plus concentré (n={int(best.sum())}) : "
      f"Top-3 = {100*hit3[best].mean():.0f}%  (vs 31% global)")
print("    -> filtrer LÈVE le taux, mais plafonne : même les matchs les plus 'sûrs' ne")
print("       dépassent pas la concentration permise par les cotes. 100% est impossible.")
print("    -> et l'EV reste négative : les cotes de ces scores sont basses en proportion.")
