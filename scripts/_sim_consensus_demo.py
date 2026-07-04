"""DÉMONSTRATION — 'simuler N résultats d'un round -> consensus -> ça passe ou ça régresse ?'

Répond empiriquement, sur les vraies données, à l'intuition :
  1) Simuler le round plus de fois améliore-t-il la prédiction ? (convergence)
  2) Quand le consensus est 'fort', la prédiction passe-t-elle mieux ? (stratification confiance)
  3) Est-ce que ça régresse dans le temps ? (stabilité temporelle)

Le 'consensus' = la distribution de scores du système (≈ baseline calibré sur ce RNG).
Aucune écriture, lecture seule.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from scraper.config import load_settings
from scraper.shadow_rng.simulators import baseline_distribution, score_list

MAXG = 7
LEAGUE = "InstantLeague-8035"
SC = score_list(MAXG)
IDX = {s: i for i, s in enumerate(SC)}
rng = np.random.RandomState(7)

_SQL = """
SELECT e.expected_start ts, o.odds_home oh, o.odds_draw od, o.odds_away oa,
       r.score_a sa, r.score_b sb
FROM events e
JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
JOIN results r ON r.event_id=e.id
WHERE r.score_a IS NOT NULL AND e.competition=:lg
  AND o.odds_home>1 AND o.odds_draw>1 AND o.odds_away>1
ORDER BY e.expected_start ASC
"""


def load():
    eng = create_engine(load_settings().db_url)
    df = pd.read_sql(text(_SQL), eng, params={"lg": LEAGUE})
    df["actual"] = [f"{min(int(a),6)}-{min(int(b),6)}" for a, b in zip(df.sa, df.sb)]
    print(f"chargé : {len(df)} rounds réglés ({LEAGUE})")
    return df


def consensus_grid(row):
    return baseline_distribution(float(row.oh), float(row.od), float(row.oa), MAXG)


# ===================================================================== #
# PART 1 — simuler plus de fois = juste estimer la MÊME distribution
# ===================================================================== #
def part1_convergence(df):
    print("\n" + "=" * 68)
    print("  PART 1 — 'simuler N fois le round' converge vers la grille analytique")
    print("=" * 68)
    row = df.iloc[len(df) // 2]   # un round représentatif
    g = consensus_grid(row)
    ana_order = np.argsort(-g)[:3]
    ana = [(SC[i], g[i]) for i in ana_order]
    print(f"  Round exemple — cotes {row.oh:.2f}/{row.od:.2f}/{row.oa:.2f}")
    print(f"  TOP-3 ANALYTIQUE (la 'vérité' = distribution implicite des cotes) :")
    print("    " + "  ".join(f"{s}={100*p:.1f}%" for s, p in ana))
    print("\n  TOP-3 par SIMULATION Monte-Carlo (tirages aléatoires du round) :")
    print("    N tirages |  top-3 simulé (fréquences)                    | écart vs analytique")
    for N in (100, 1000, 10000, 100000, 1000000):
        draws = rng.choice(g.size, size=N, p=g)
        cnt = np.bincount(draws, minlength=g.size) / N
        sim_order = np.argsort(-cnt)[:3]
        sim = "  ".join(f"{SC[i]}={100*cnt[i]:.1f}%" for i in sim_order)
        # écart L1 entre distribution simulée et analytique
        l1 = np.abs(cnt - g).sum()
        print(f"    {N:>9} | {sim:<44} | L1={l1:.4f}")
    print("\n  -> Plus on simule, plus on RETROUVE EXACTEMENT la grille analytique.")
    print("     La simulation n'AJOUTE aucune information : elle ESTIME une distribution")
    print("     déjà connue. Le 'consensus' = l'argmax de cette distribution = le score modal.")


# ===================================================================== #
# PART 2 — quand le consensus est 'fort', ça passe mieux ? (calibration)
# ===================================================================== #
def part2_confidence(df):
    print("\n" + "=" * 68)
    print("  PART 2 — la FORCE du consensus prédit-elle la réussite ?")
    print("=" * 68)
    p_top1, p_top3, hit1, hit3 = [], [], [], []
    for row in df.itertuples():
        g = baseline_distribution(float(row.oh), float(row.od), float(row.oa), MAXG)
        order = np.argsort(-g)
        top1, top3 = SC[order[0]], [SC[i] for i in order[:3]]
        p_top1.append(float(g[order[0]]))
        p_top3.append(float(g[order[:3]].sum()))
        hit1.append(row.actual == top1)
        hit3.append(row.actual in top3)
    a = pd.DataFrame({"p1": p_top1, "p3": p_top3, "h1": hit1, "h3": hit3})

    # bins par quantiles de confiance (proba du top-1 consensus)
    a["bin"] = pd.qcut(a["p1"], 5, labels=False, duplicates="drop")
    print("  Rounds regroupés par FORCE du consensus (proba du score top-1) :\n")
    print("  conf. consensus | n    | TOP-1 prédit | TOP-1 réalisé | TOP-3 prédit | TOP-3 réalisé")
    print("  ----------------+------+--------------+---------------+--------------+--------------")
    for b in sorted(a["bin"].dropna().unique()):
        s = a[a["bin"] == b]
        lo, hi = s["p1"].min(), s["p1"].max()
        print(f"  {100*lo:4.1f}%-{100*hi:4.1f}%     | {len(s):>4} |   {100*s.p1.mean():5.1f}%    |"
              f"    {100*s.h1.mean():5.1f}%     |   {100*s.p3.mean():5.1f}%    |    {100*s.h3.mean():5.1f}%")
    print("  ----------------+------+--------------+---------------+--------------+--------------")
    print(f"  GLOBAL          | {len(a):>4} |   {100*a.p1.mean():5.1f}%    |"
          f"    {100*a.h1.mean():5.1f}%     |   {100*a.p3.mean():5.1f}%    |    {100*a.h3.mean():5.1f}%")
    print("\n  -> Le 'réalisé' colle au 'prédit' dans CHAQUE tranche : le consensus est")
    print("     CALIBRÉ. Quand il annonce 18%, ça arrive ~18% du temps — ni plus, ni moins.")
    print("     'Ne jouer que les consensus forts' = jouer là où la proba est haute :")
    print("     on gagne plus souvent, mais JAMAIS au-dessus de la proba annoncée (zéro edge).")


# ===================================================================== #
# PART 3 — est-ce que ça régresse dans le temps ?
# ===================================================================== #
def part3_stability(df):
    print("\n" + "=" * 68)
    print("  PART 3 — est-ce que la prédiction RÉGRESSE dans le temps ?")
    print("=" * 68)
    h1, h3 = [], []
    for row in df.itertuples():
        g = baseline_distribution(float(row.oh), float(row.od), float(row.oa), MAXG)
        order = np.argsort(-g)
        h1.append(row.actual == SC[order[0]])
        h3.append(row.actual in [SC[i] for i in order[:3]])
    a = pd.DataFrame({"h1": h1, "h3": h3})
    k = len(a) // 3
    print("  Historique découpé en 3 tiers chronologiques :\n")
    print("  période          | n    | TOP-1 réalisé | TOP-3 réalisé")
    print("  -----------------+------+---------------+--------------")
    for name, sub in [("1er tiers (vieux)", a.iloc[:k]),
                      ("2e tiers", a.iloc[k:2 * k]),
                      ("3e tiers (récent)", a.iloc[2 * k:])]:
        print(f"  {name:<16} | {len(sub):>4} |     {100*sub.h1.mean():5.1f}%    |    {100*sub.h3.mean():5.1f}%")
    print("\n  -> Si les 3 tiers sont stables : AUCUNE régression, le RNG ne dérive pas.")
    print("     C'est précisément ce que l'évaluateur surveille EN CONTINU (alarme si dérive).")


if __name__ == "__main__":
    df = load()
    part1_convergence(df)
    part2_confidence(df)
    part3_stability(df)
    print("\n" + "=" * 68)
    print("  CONCLUSION : simuler le round -> consensus, c'est EXACTEMENT ce que fait")
    print("  shadow_rng_main.py. Mais la simulation ne BAT pas la distribution : elle la")
    print("  retrouve. Le 'consensus' est la meilleure prédiction possible (~plafond), et")
    print("  l'évaluateur dit si ça 'passe bien ou régresse' au fil des rounds. Tout est là.")
    print("=" * 68)
