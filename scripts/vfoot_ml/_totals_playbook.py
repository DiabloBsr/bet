"""PLAYBOOK TOTALS — tout le savoir mesurable sur les marchés de buts, + backtest
de LA stratégie utilisateur : combinés 2-3 jambes 100% totals, cote>=3, proba max.

Données : closing_team.csv (1 ligne/équipe-match ; venue=H -> 1 ligne/match,
o_over35/o_under35 offertes, total réel) + market_cells.csv (ROI par sélection).
"""
from __future__ import annotations
import sys
from itertools import combinations, product as iproduct
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
d = pd.read_csv(ROOT / "data/vfoot_ml/closing_team.csv")
d = d[d.venue == "H"].dropna(subset=["o_over35", "o_under35"]).copy()
d = d[(d.o_over35 > 1) & (d.o_over35 < 99) & (d.o_under35 > 1) & (d.o_under35 < 99)]

# ============ 1. LA DISTRIBUTION RÉELLE DES TOTAUX ============
print("=" * 74)
print("  1. DISTRIBUTION RÉELLE DES TOTAUX (32k matchs)")
print("=" * 74)
tot = d.total.clip(0, 7).astype(int)
dist = tot.value_counts(normalize=True).sort_index()
print("  buts   : " + "  ".join(f"{k}:{100*v:.1f}%" for k, v in dist.items()))
print(f"  moyenne {d.total.mean():.2f} buts | over2.5 {100*(d.total>2.5).mean():.1f}% | "
      f"over3.5 {100*(d.total>3.5).mean():.1f}% | under3.5 {100*(d.total<3.5).mean():.1f}%")

# par équilibre du match (drama mode = plus de buts sur les serrés ?)
d["fav"] = np.maximum(1/d.odds, 0)  # odds = cote victoire domicile ici (venue H)
inv = None
print("\n  Over 3.5 réel vs implicite, par cote domicile (proxy équilibre) :")
d["po_dev"] = (1/d.o_over35) / (1/d.o_over35 + 1/d.o_under35)
for lo, hi in ((1.0, 1.5), (1.5, 2.0), (2.0, 2.7), (2.7, 4.0), (4.0, 99)):
    m = d[(d.odds >= lo) & (d.odds < hi)]
    if len(m) < 300: continue
    print(f"    dom [{lo},{hi}) : n={len(m):>5}  réel {100*(m.total>3.5).mean():5.1f}%  "
          f"implicite {100*m.po_dev.mean():5.1f}%  gap {100*((m.total>3.5).mean()-m.po_dev.mean()):+5.2f}pp")

# ============ 2. ROI PAR SÉLECTION TOTALS (cellules campagne 17) ============
print()
print("=" * 74)
print("  2. CE QUE COÛTE CHAQUE PARI TOTALS (mesuré, phase test)")
print("=" * 74)
c = pd.read_csv(ROOT / "data/vfoot_ml/market_cells.csv")
c = c[(c.fav_band == "ALL") & (c.phase == "test") &
      (c.market.isin(["+/-", "Total de buts", "Multi-Buts"]))]
c["roi"] = 100 * c.pnl_sum / c.n
c["hit"] = 100 * c.k / c.n
c["cote"] = c.odds_sum / c.n
for r in c.sort_values("roi", ascending=False).itertuples():
    print(f"  {r.market:<15} {str(r.selection)[:34]:<36} hit {r.hit:5.1f}%  cote {r.cote:5.2f}  ROI {r.roi:+6.1f}%")

# ============ 3. BACKTEST : TA STRATÉGIE (combiné totals cote>=3, proba max) ============
print()
print("=" * 74)
print("  3. BACKTEST — combinés 100% TOTALS (O/U 3.5), proba max, sur ~3000 rounds")
print("=" * 74)
d["pu_dev"] = 1 - d.po_dev
rounds = [g for _, g in d.groupby("ts") if len(g) >= 4]
print(f"  rounds exploitables : {len(rounds)}")


def best_combo(g, target, max_legs=3):
    """Réplique exacte du construiseur de l'app, restreint O/U 3.5."""
    legs_by_match = []
    for r in g.itertuples():
        legs = []
        if r.pu_dev >= 0.45: legs.append(("U", r.pu_dev, r.o_under35, r.total < 3.5))
        if r.po_dev >= 0.45: legs.append(("O", r.po_dev, r.o_over35, r.total > 3.5))
        if legs: legs_by_match.append(sorted(legs, key=lambda l: -l[1])[:2])
    idxs = range(len(legs_by_match))
    best = None
    for k in range(2, max_legs + 1):
        for mix in combinations(idxs, k):
            for choice in iproduct(*[legs_by_match[i] for i in mix]):
                op = pp = 1.0
                for _, p, o, _w in choice: op *= o; pp *= p
                if op >= target and (best is None or pp > best[1]):
                    best = (choice, pp, op)
    return best


for target in (2.0, 3.0, 5.0):
    n = hits = 0; pnl = []; pest = []; odds_l = []
    for g in rounds:
        b = best_combo(g, target)
        if b is None: continue
        choice, pp, op = b
        won = all(w for _, _, _, w in choice)
        n += 1; hits += won; pnl.append(won * op - 1); pest.append(pp); odds_l.append(op)
    if n:
        print(f"  cible >={target}: n={n:>4} rounds | cote moy {np.mean(odds_l):4.2f} | "
              f"réussite RÉELLE {100*hits/n:5.1f}% vs annoncée {100*np.mean(pest):5.1f}% | "
              f"ROI {100*np.mean(pnl):+6.2f}%")

# baseline : 1 jambe unique (le pari totals le plus sûr du round)
n = hits = 0; pnl = []
for g in rounds:
    r = g.loc[g.pu_dev.idxmax()]
    won = r.total < 3.5
    n += 1; hits += won; pnl.append(won * r.o_under35 - 1)
print(f"  baseline 1 jambe (meilleur under du round) : réussite {100*hits/n:.1f}% | ROI {100*np.mean(pnl):+.2f}%")
